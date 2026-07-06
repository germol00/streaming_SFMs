def _transcribe_output_processing2(self, outputs, trcfg):
    """
    Internal function to process the model's outputs to return the results to the user. This function is called by
    `transcribe()` and `transcribe_generator()` to process the model's outputs.
    If parallel chunking was used (enable_chunking=True), merges the hypotheses from each chunk
    into a single hypothesis, joining text, token sequences, and timestamps.

    Args:
        outputs: The model's outputs that are processed by `_transcribe_forward()`.
        trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.

    Returns:
        The output can be a list of
        objects, list of list of objects.
        Its type is defined in `TranscriptionReturnType`.

    """
    #print(self.prompt_format)

    log_probs = outputs.pop('log_probs')
    encoded_len = outputs.pop('encoded_lengths')
    enc_states = outputs.pop('encoder_states')
    enc_mask = outputs.pop('encoder_mask')
    decoder_input_ids = outputs.pop('decoder_input_ids')
    batch = outputs.pop('batch')

    del log_probs
    num_chunks = enc_states.shape[0]
    # Repear decoder_input_ids to match number of chunks
    if trcfg.enable_chunking and num_chunks > decoder_input_ids.shape[0]:
        decoder_input_ids = decoder_input_ids.repeat(num_chunks, 1)
    hypotheses = self.decoding.decode_predictions_tensor(
        encoder_hidden_states=enc_states,
        encoder_input_mask=enc_mask,
        decoder_input_ids=decoder_input_ids,
        return_hypotheses=trcfg.return_hypotheses,
    )
    merge_to_be_done = trcfg.enable_chunking and len(hypotheses) > 1

    del enc_states, enc_mask, decoder_input_ids

    if trcfg.timestamps and self.timestamps_asr_model is not None:
        hypotheses = get_forced_aligned_timestamps_with_external_model(
            audio=[audio.squeeze()[:audio_len] for audio, audio_len in zip(batch[0], batch[1])],
            batch_size=len(batch[0]),
            external_ctc_model=self.timestamps_asr_model,
            main_model_predictions=hypotheses,
            timestamp_type='char' if merge_to_be_done else ['word', 'segment'],
            viterbi_device=trcfg._internal.device,
        )
    elif trcfg.timestamps:
        hypotheses = process_aed_timestamp_outputs(
            hypotheses, self.encoder.subsampling_factor, self.cfg['preprocessor']['window_stride']
        )

    if merge_to_be_done and self.timestamps_asr_model is not None:
        merged_hypotheses = merge_parallel_chunks(
            hypotheses=hypotheses,
            encoded_len=encoded_len,
            model=self,
            timestamps=trcfg.timestamps,
            subsampling_factor=self.encoder.subsampling_factor,
            window_stride=self.cfg['preprocessor']['window_stride'],
            decoding=self.decoding,
        )
        # Inject the id of the cut to hypothese to later be used for separate batches
        setattr(merged_hypotheses, 'id', batch.cuts[0].id)
        return [merged_hypotheses]

    if trcfg.enable_chunking and len(hypotheses) == 1:
        setattr(hypotheses[0], 'id', batch.cuts[0].id)
    return hypotheses

