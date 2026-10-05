import time
import os
import types
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from omegaconf import open_dict, OmegaConf
from nemo.collections.asr.parts.utils.streaming_utils import ContextSize, StreamingBatchedAudioBuffer
from nemo.collections.asr.parts.utils.transcribe_utils import setup_model, get_inference_device, get_inference_dtype

from .hyp_utils import LCPHypothesisBuffer, LACPHypothesisBuffer, WaitKHypothesisBuffer, HoldNHypothesisBuffer

from nemo.collections.asr.parts.utils.rnnt_utils import batched_hyps_to_hypotheses
from .utils import _transcribe_output_processing2


import logging
logger = logging.getLogger(__name__)

from streaming_SFMs import LOG_LEVEL
logger.setLevel(LOG_LEVEL)

class StreamingBatchedAudioBufferWithOffset(StreamingBatchedAudioBuffer):
    def add_audio_batch_get_stride(
            self,
            audio_batch,
            audio_lengths,
            is_last_chunk,
            is_last_chunk_batch,
        ):
        """
        Add audio batch to buffer and get the number of extra frames on the left

        Args:
            Audio_batch: chunk with audio
            audio_lengths: length of audio
            is_last_chunk: if last chunk
            is_last_chunk_batch: if last chunk for each audio utterance
        """
        added_chunk_length = audio_batch.shape[1]

        if added_chunk_length > self.expected_context.chunk:
            logger.warning(f'added chunk length {added_chunk_length} is greater than expected context {self.expected_context.chunk}. Trimming')
            added_chunk_length = self.expected_context.chunk
            audio_batch = audio_batch[:, :added_chunk_length]

        self.samples = torch.cat((self.samples, audio_batch), dim=1)
        extra_samples_in_buffer = self.context_size.add_frames_get_removed_(
            added_chunk_length, is_last_chunk=is_last_chunk, expected_context=self.expected_context
        )
        self.context_size_batch.add_frames_(
            num_frames_batch=audio_lengths,
            is_last_chunk_batch=is_last_chunk_batch,
            expected_context=self.expected_context,
        )

        if extra_samples_in_buffer > 0:
            self.samples = self.samples[:, extra_samples_in_buffer:]
        return extra_samples_in_buffer

    def get_buffer_len(self):
        return (len(self.samples), len(self.samples[0]))

class BaseStreamingModel():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = get_inference_device(cuda=cfg.cuda, allow_mps=cfg.allow_mps)
        self.dtype = get_inference_dtype(cfg.compute_dtype, device=self.device)

        self.asr_model, self.model_name = setup_model(cfg, self.device)
        self.asr_model.eval()
        self.asr_model.to(self.dtype)

        self.sample_rate = self.asr_model._cfg.preprocessor['sample_rate']
        self.feature_stride_sec = self.asr_model._cfg.preprocessor['window_stride']
        self.subsampling_factor = self.asr_model.encoder.subsampling_factor
        print(self.subsampling_factor)
        self.features_per_sec = 1.0 / self.feature_stride_sec
        self.encoder_frame2audio_samples = int(self.sample_rate * self.feature_stride_sec) * self.subsampling_factor

        self.context_samples = self._compute_context(cfg)

    def _compute_context(self, cfg):
        """Returns the context size in samples for left, right and current chunk contexts"""
        encoder_frames = ContextSize(
            left=int(cfg.left_context_secs * self.features_per_sec / self.subsampling_factor),
            chunk=int(cfg.chunk_secs * self.features_per_sec / self.subsampling_factor),
            right=int(cfg.right_context_secs * self.features_per_sec / self.subsampling_factor),
        )
        print(f'CONTEXT SIZE = left: {encoder_frames.left*self.encoder_frame2audio_samples/self.sample_rate} chunk: {encoder_frames.chunk*self.encoder_frame2audio_samples/self.sample_rate} right: {encoder_frames.right*self.encoder_frame2audio_samples/self.sample_rate}')
        return ContextSize(
            left=encoder_frames.left * self.encoder_frame2audio_samples,
            chunk=encoder_frames.chunk * self.encoder_frame2audio_samples,
            right=encoder_frames.right * self.encoder_frame2audio_samples,
        )

    def _init_policy_buffer(self, debug=False):
        """Initializes the emission policy based on config."""
        if self.cfg.policy == 'WaitK':
            return WaitKHypothesisBuffer(K=self.cfg.K, features_per_second=self.features_per_sec,
                                        subsampling_factor=self.subsampling_factor, debug=debug)
        elif self.cfg.policy == 'HoldN':
            return HoldNHypothesisBuffer(N=self.cfg.N, debug=debug)
        elif self.cfg.policy == 'LACP':
            return LACPHypothesisBuffer(threshold=self.cfg.lacp_threshold, uncased=True, debug=debug)
        return LCPHypothesisBuffer(uncased=True, debug=debug)

    @abstractmethod
    def process_chunk(self, buffer, current_offset) -> List[Tuple[float, float, str]]:
        """Processes audio chunk. Must be implemented in the child class."""
        pass

    @torch.no_grad()
    def transcribe(self, audio_signal: torch.Tensor):
        """The main streaming loop shared by all models."""
        audio_signal = audio_signal.to(self.device)
        buffer = StreamingBatchedAudioBufferWithOffset(
            batch_size=1, context_samples=self.context_samples,
            dtype=audio_signal.dtype, device=self.device
        )
        hyp_buffer = self._init_policy_buffer(debug= True if logger.level == logging.DEBUG else False)

        committed_results = []
        current_offset = 0
        left_sample = 0
        right_sample = min(self.context_samples.chunk + self.context_samples.right, audio_signal.shape[1])

        while left_sample < audio_signal.shape[1]:
            is_last_chunk = right_sample >= audio_signal.shape[1]
            chunk_len = min(right_sample, audio_signal.shape[1]) - left_sample

            stride = buffer.add_audio_batch_get_stride(
                audio_signal[:, left_sample:right_sample],
                audio_lengths=torch.tensor([chunk_len], device=self.device),
                is_last_chunk=is_last_chunk,
                is_last_chunk_batch=torch.tensor([is_last_chunk], device=self.device)
            )
            current_offset += (stride // self.encoder_frame2audio_samples)

            formatted_hyp = self.process_chunk(buffer, current_offset)

            hyp_buffer.insert(formatted_hyp)

            if self.cfg.policy == 'WaitK':
                out = hyp_buffer.flush(last_instant=left_sample // self.encoder_frame2audio_samples)
            else:
                out = hyp_buffer.flush()

            committed_results.extend(out)
            left_sample = right_sample
            right_sample = min(right_sample + self.context_samples.chunk, audio_signal.shape[1])

        committed_results.extend(hyp_buffer.complete())
        return " ".join([t for _, _, t in committed_results])

class StreamingParakeet(BaseStreamingModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Timestamps require alignments + TDT durations to be preserved during decoding.
        # These flags must be applied via change_decoding_strategy on the *model decoding*
        # config — not the user/script cfg (chunk_secs, policy, ...).
        with open_dict(self.asr_model.cfg.decoding):
            self.asr_model.cfg.decoding.greedy.preserve_alignments = True
            self.asr_model.cfg.decoding.tdt_include_token_duration = True

        decoding_cfg = self.asr_model.cfg.decoding
        rnnt_decoding = OmegaConf.select(cfg, "rnnt_decoding", default=None)
        if rnnt_decoding is not None:
            decoding_cfg = OmegaConf.merge(decoding_cfg, rnnt_decoding)

        strategy = OmegaConf.select(decoding_cfg, "strategy", default="greedy_batch")
        if strategy != "greedy_batch":
            raise ValueError(
                f"Only decoding strategy 'greedy_batch' is supported, got '{strategy}'."
            )

        self.asr_model.change_decoding_strategy(OmegaConf.create(decoding_cfg))

    def process_chunk(self, buffer, current_offset):
        # Forward pass through encoder
        encoder_output, encoder_output_len = self.asr_model(
            input_signal=buffer.samples,
            input_signal_length=buffer.context_size_batch.total(),
        )
        encoder_output = encoder_output.transpose(1, 2)

        chunk_batched_hyps, _, _ = self.asr_model.decoding.decoding.decoding_computer(
            x=encoder_output, out_len=encoder_output_len, prev_batched_state=None
        )
        unbatched_hyp = batched_hyps_to_hypotheses(chunk_batched_hyps)[0]
        timestamped_hyp = self.asr_model.decoding.compute_rnnt_timestamps(unbatched_hyp)
        toks = self.asr_model.decoding.decode_ids_to_tokens(timestamped_hyp.y_sequence.tolist())

        return [(t['start_offset'] + current_offset, t['end_offset'] + current_offset, tok)
                for t, tok in zip(timestamped_hyp.timestamp['char'], toks)]

class StreamingCanary(BaseStreamingModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.asr_model._transcribe_output_processing = types.MethodType(
            _transcribe_output_processing2, self.asr_model
        )

    def process_chunk(self, buffer, current_offset):
        batched_timestamped_hyp = self.asr_model.transcribe(
            buffer.samples[0], timestamps=True, verbose=False
        )
        timestamped_hyp = batched_timestamped_hyp[0]

        return [(w['start_offset'] + current_offset, w['end_offset'] + current_offset, w['word'])
                for w in timestamped_hyp.timestamp['word']]
