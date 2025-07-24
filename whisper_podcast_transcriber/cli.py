#!/usr/bin/env python3

import os
import sys
import json
import whisperx
import torch
from dotenv import load_dotenv
from datetime import timedelta
from pathlib import Path
from whisperx.diarize import assign_word_speakers
from whisper_podcast_transcriber.diarization import perform_diarization


def format_timestamp(seconds: float) -> str:
    """Format seconds into SRT timestamp (HH:MM:SS,mmm)."""
    return str(timedelta(seconds=seconds)).split(".")[0] + f",{int(seconds % 1 * 1000):03}"

def main():
    args = parse_args()
    audio_path = Path(args.audio_path)
    model_name = args.model
    skip_diarization = args.skip_diarization

    if not audio_path.exists():
        print(f"❌ File not found: {audio_path}")
        sys.exit(1)

    project_dir = Path(__file__).resolve().parent
    dotenv_path = project_dir.parent / ".env"
    load_dotenv(dotenv_path)
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("❌ Missing Hugging Face token.")
        print("   Add HUGGINGFACE_TOKEN=your_token_here to a .env file or your shell environment.")
        print("   You can get a token from https://huggingface.co/settings/tokens\n")
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

    # 🔉 Diarization
    print("🔉 Performing speaker diarization...")
    diarize_segments = perform_diarization(audio_path, hf_token, device)

    # 🪢 Speaker assignment
    print("🪢 Merging speaker labels...")
    full_output = assign_word_speakers(diarize_segments, aligned)
    final_segments = full_output["segments"] if isinstance(full_output, dict) and "segments" in full_output else full_output

    base = audio_path.with_suffix("")
    txt_path = base.with_name(base.name + "_transcript.txt")
    srt_path = base.with_name(base.name + "_transcript.srt")
    json_path = base.with_name(base.name + "_transcript.json")

    print(f"💾 Writing plain-text transcript to {txt_path}")
    with open(txt_path, "w", encoding="utf-8") as f:
        for seg in final_segments:
            if isinstance(seg, dict):
                text = seg.get("text", "")
                speaker = seg.get("speaker", "SPEAKER")
            else:
                text = str(seg)
                speaker = "SPEAKER"

            f.write(f"[{speaker}] {text}\n")

    print(f"💬 Writing SRT subtitles to {srt_path}")
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(final_segments, 1):
            if isinstance(seg, dict):
                start = format_timestamp(seg.get("start", 0))
                end = format_timestamp(seg.get("end", 0))
                speaker = seg.get("speaker", "SPEAKER")
                text = seg.get("text", "")
            else:
                start = end = format_timestamp(0)
                speaker = "SPEAKER"
                text = str(seg)

            f.write(f"{i}\n{start} --> {end}\n[{speaker}] {text}\n\n")

    print(f"🧾 Writing JSON metadata to {json_path}")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_segments, f, indent=2)

    print(f"✅ All outputs written to:")
    for path in (txt_path, srt_path, json_path):
        try:
            print(f"  {path.relative_to(Path.cwd())}")
        except ValueError:
            print(f"  {path.resolve()}")

if __name__ == "__main__":
    main()
