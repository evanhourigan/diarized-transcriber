#!/usr/bin/env python3

import os
import sys
import json
import whisperx
import torch
from dotenv import load_dotenv
from datetime import timedelta

def format_timestamp(seconds: float) -> str:
    """Format seconds into SRT timestamp (HH:MM:SS,mmm)."""
    return str(timedelta(seconds=seconds)).split(".")[0] + f",{int(seconds % 1 * 1000):03}"

def main():
    if len(sys.argv) < 2:
        print("Usage: transcribe_with_speakers.py <audio_file> [model]")
        sys.exit(1)

    audio_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "medium"

    if not os.path.exists(audio_path):
        print(f"❌ File not found: {audio_path}")
        sys.exit(1)

    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("❌ Missing Hugging Face token. Add HUGGINGFACE_TOKEN=... to a .env file.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Loading Whisper model '{model_name}' on {device}...")
    model = whisperx.load_model(model_name, device, compute_type="float32")

    print("🎧 Transcribing audio...")
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio)

    print("🧭 Aligning transcription...")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    aligned = whisperx.align(result["segments"], model_a, metadata, audio, device)

    print("🔉 Performing speaker diarization...")
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarize_segments = diarize_model(audio_path)

    print("🪢 Merging speaker labels...")
    final_segments = whisperx.merge_text_diarization(aligned["segments"], diarize_segments)

    base = os.path.splitext(audio_path)[0]
    txt_path = f"{base}_transcript.txt"
    srt_path = f"{base}_transcript.srt"
    json_path = f"{base}_transcript.json"

    print(f"💾 Writing plain-text transcript to {txt_path}")
    with open(txt_path, "w", encoding="utf-8") as f:
        for seg in final_segments:
            f.write(f"[{seg['speaker']}] {seg['text']}\n")

    print(f"💬 Writing SRT subtitles to {srt_path}")
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(final_segments, 1):
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            f.write(f"{i}\n{start} --> {end}\n[{seg['speaker']}] {seg['text']}\n\n")

    print(f"🧾 Writing JSON metadata to {json_path}")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_segments, f, indent=2)

    print("✅ Done!")

if __name__ == "__main__":
    main()