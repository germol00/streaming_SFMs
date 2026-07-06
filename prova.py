import argparse
import json
import torch
import librosa
from pathlib import Path
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

# Import your refactored classes
from streaming_sfm.streaming_model import StreamingParakeet, StreamingCanary

def parse_args():
    parser = argparse.ArgumentParser(description="Streaming ASR Evaluation Script")
    
    # Model and Data
    parser.add_argument("--model_path", type=str, default=None, help="Path to .nemo file")
    parser.add_argument("--pretrained_name", type=str, default=None, help="Name of a pretrained model")
    parser.add_argument("--manifest_path", type=str, default="vp.jsonl", help="Path to NeMo manifest")
    
    # Streaming / Windowing
    parser.add_argument("--chunk_secs", type=float, default=1, help="Duration of the sliding window chunk")
    parser.add_argument("--left_context_secs", type=float, default=20, help="Left context duration")
    parser.add_argument("--right_context_secs", type=float, default=0, help="Right context duration")
    
    # Emission Policies
    parser.add_argument("--policy", type=str, default="LACP", choices=["LCP", "LACP", "WaitK", "HoldN"])
    parser.add_argument("--lacp_threshold", type=float, default=2, help="Threshold for LACP policy")
    parser.add_argument("--K", type=int, default=2, help="K value for WaitK policy")
    parser.add_argument("--N", type=int, default=5, help="N value for HoldN policy")
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--compute_dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # 1. Convert Namespace to OmegaConf (which our Streamer classes expect)
    # We add 'cuda' and 'allow_mps' keys to match your existing setup logic
    cfg = OmegaConf.create(vars(args))
    with open_dict(cfg):
        cfg.cuda = 0 if args.device == "cuda" else 0
        cfg.allow_mps = True if args.device == "mps" else False

    # 2. Factory Logic to select the streamer
    # We check the model type without fully loading it first to save time/memory
    model_id = args.model_path or args.pretrained_name
    if not model_id:
        raise ValueError("Neither of --model_path or --pretrained_name were provided")

    if "canary" in model_id.lower():
        print(f"--- Initializing Streaming Canary ---")
        streamer = StreamingCanary(cfg)
    else:
        print(f"--- Initializing Streaming Parakeet ---")
        streamer = StreamingParakeet(cfg)

    print(streamer.model.encoder_subsampling_factor)

    # 3. Load and Process Manifest
    with open(args.manifest_path, 'r') as f:
        records = [json.loads(line) for line in f]

    print(f"Processing {len(records)} files from {args.manifest_path}...")
    
    for record in tqdm(records):
        audio_path = record['audio_filepath']
        ref_text = record.get('text', 'N/A')
        
        # Load audio (standardized to 16kHz)
        audio, _ = librosa.load(audio_path, sr=16000)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        
        # Transcribe using the class-based streaming logic
        hyp_text = streamer.transcribe(audio_tensor)
        
        print(f"\nFile: {Path(audio_path).name}")
        print(f"Ref: {ref_text}")
        print(f"Hyp: {hyp_text}")
        print("-" * 30)

if __name__ == "__main__":
    main()
