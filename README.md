# streaming_SFMs

**streaming_SFMs** is a small toolkit for streaming automatic speech recognition (ASR) with speech foundation models. It provides the reference implementation for the experiments in [*Improving streaming ASR with foundation models using emission policies*](https://www.isca-archive.org/interspeech_2026/masmolla26_interspeech.pdf).

The toolkit turns offline ASR models into a streaming pipeline by decoding audio over a sliding window, recovering token timestamps, and applying emission policies.

## Features

- **Models:** streaming wrappers for NVIDIA Parakeet and Canary speech foundation models
- **Emission policies:** LCP, LACP, WaitK, and HoldN

## Requirements

- **Python:** 3.10.12 or newer
- **NeMo:** `nemo-toolkit[asr]` ≥2.7.0, <3.0.0

## Installation

Clone the repository and install the package into your Python environment (conda, venv, or similar):

```bash
git clone https://github.com/germol00/streaming_SFMs.git
cd streaming_SFMs
pip install .
```

With [uv](https://github.com/astral-sh/uv):

```bash
uv pip install .
```

## Quickstart

Run the example script on a NeMo-style JSONL manifest (each line needs at least `audio_filepath`):

```bash
python prova.py --pretrained_name nvidia/parakeet-tdt-0.6b-v3 --manifest_path manifest.jsonl
```

Or use the library API directly:

```python
import torch
import librosa
from omegaconf import OmegaConf, open_dict
from streaming_SFMs.streaming_model import StreamingParakeet

cfg = OmegaConf.create({
    "pretrained_name": "nvidia/parakeet-tdt-0.6b-v3",
    "model_path": None,
    "chunk_secs": 1.0,
    "left_context_secs": 20.0,
    "right_context_secs": 0.0,
    "policy": "LACP",
    "lacp_threshold": 2,
    "compute_dtype": "float16",
})
with open_dict(cfg):
    cfg.cuda = 0
    cfg.allow_mps = False

streamer = StreamingParakeet(cfg)
audio, _ = librosa.load("path/to/audio.wav", sr=16000)
hyp = streamer.transcribe(torch.from_numpy(audio).unsqueeze(0))
print(hyp)
```

## License
The streaming_SFMs source code is licensed under the [Apache License 2.0](LICENSE).

This project is research software. Model names and trademarks belong to their respective owners.
