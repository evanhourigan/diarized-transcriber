#!/usr/bin/env python3

import os
import sys
import json
import whisperx
import torch
from dotenv import load_dotenv
from datetime import timedelta
from whisperx.diarize import assign_word_speakers
from whisper_podcast_transcriber.diarization import perform_diarization


def format_timestamp(seconds: float) -> str:
    """Format seconds into SRT timestamp (HH:MM:SS,mmm)."""
    return str(timedelta(seconds=seconds)).split(".")[0] + f",{int(seconds % 1 * 1000):03}"


def main():
    if len(sys.argv) < 2:
        print("Usage: transcribe <audio_file> [model]")
        sys.exit(1)

    # 📥 Resolve input
    audio_path = os.path.abspath(sys.argv[1])
    model_name = sys.argv[2] if len(sys.argv) > 2 else "medium"

    if not os.path.exists(audio_path):
        print(f"❌ File not found: {audio_path}")
        sys.exit(1)

    # 🔐 Load environment and Hugging Face token
    project_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(project_dir, "..", ".env")
    load_dotenv(dotenv_path)
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("❌ Missing Hugging Face token.")
        print("   Add HUGGINGFACE_TOKEN=your_token_here to a .env file or your shell environment.")
        print("   You can get a token from https://huggingface.co/settings/tokens\n")
        sys.exit(1)

    # 🧠 Load Whisper model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Loading Whisper model '{model_name}' on {device}...")
    model = whisperx.load_model(model_name, device, compute_type="float32")

    # 🎧 Transcription
    print("🎧 Transcribing audio...")
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio)

    # 🧭 Alignment
    print("🧭 Aligning transcription...")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    aligned = whisperx.align(result["segments"], model_a, metadata, audio, device)

    # 🔉 Diarization
    print("🔉 Performing speaker diarization...")
    diarize_segments = perform_diarization(audio_path, hf_token, device)

    # 🪢 Speaker assignment
    print("🪢 Merging speaker labels...")
    full_output = assign_word_speakers(diarize_segments, aligned)
    final_segments = full_output["segments"] if isinstance(full_output, dict) and "segments" in full_output else full_output

    # 💾 Output paths
    base = os.path.splitext(audio_path)[0]
    txt_path = f"{base}_transcript.txt"
    srt_path = f"{base}_transcript.srt"
    json_path = f"{base}_transcript.json"

    # ✍️ Write outputs
    print(f"💾 Writing plain-text transcript to {txt_path}")
    with open(txt_path, "w", encoding="utf-8") as f:
        for seg in final_segments:
            if isinstance(seg, dict) and "speaker" in seg and "text" in seg:
                f.write(f"[{seg['speaker']}] {seg['text']}\n")
            else:
                print("❌ Unexpected segment format:", seg)

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