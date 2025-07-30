#!/usr/bin/env python3

import argparse
import os
import warnings
import logging
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from whisper_podcast_transcriber.diarization import run_transcribe_with_diarization
from whisper_podcast_transcriber.srt_exporter import generate_speaker_aware_srt
from whisper_podcast_transcriber.txt_exporter import generate_txt
from whisper_podcast_transcriber.markdown_exporter import generate_markdown_transcript
from whisper_podcast_transcriber.html_exporter import generate_html_transcript
from whisper_podcast_transcriber.pdf_exporter import generate_pdf_transcript

def main():
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Transcribe and optionally diarize an audio file, exporting to various formats."
    )
    parser.add_argument("audio_path", help="Path to audio file (e.g., .wav)")
    parser.add_argument("--output-dir", dest="output_dir", default=".", help="Directory to save outputs (default: current directory)")
    parser.add_argument("--model", default="medium", help="Whisper model to use (default: medium)")
    parser.add_argument("--skip-diarization", dest="skip_diarization", action="store_true", help="Skip speaker diarization")
    parser.add_argument("--num-speakers", type=int, help="Exact number of speakers (improves diarization accuracy)")
    parser.add_argument("--timestamps", action="store_true", help="Include timestamps in output files")
    parser.add_argument("--formats", nargs="+", default=["md"], help="Output formats: txt, md, srt, json, html, pdf, all")
    parser.add_argument("--debug", action="store_true", help="Show detailed debug warnings and logs")

    args = parser.parse_args()

    if not args.debug:
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    # Generate base filename from input audio file
    audio_basename = os.path.splitext(os.path.basename(args.audio_path))[0]
    base_filename = f"{audio_basename}-transcript"

    # Start the transcription process
    print("🎙️ Starting podcast transcription...")
    print(f"📁 Audio file: {args.audio_path}")
    print(f"🤖 Model: {args.model}")
    print(f"📂 Output directory: {args.output_dir}")
    print(f"📊 Formats: {', '.join(args.formats)}")
    print(f"👥 Diarization: {'❌ Disabled' if args.skip_diarization else '✅ Enabled'}")
    
    # Warn if num_speakers is specified but diarization is skipped
    if args.num_speakers and args.skip_diarization:
        print("💡 Note: --num-speakers is ignored when --skip-diarization is used")
    
    print("─" * 50)

    start_time = time.time()
    
    # Run transcription and diarization
    result = run_transcribe_with_diarization(
        audio_path=args.audio_path,
        output_dir=args.output_dir,
        model_size=args.model,
        skip_diarization=args.skip_diarization,
        num_speakers=args.num_speakers
    )

    transcription_time = time.time() - start_time
    print(f"✅ Transcription completed in {transcription_time:.1f}s")
    print(f"📝 Found {len(result['segments'])} segments")
    
    # Create output directory if it doesn't exist
    if args.output_dir != ".":
        print("📁 Creating output directory...")
        os.makedirs(args.output_dir, exist_ok=True)

    # Export to different formats
    print("📤 Exporting transcripts...")
    export_start_time = time.time()
    
    formats_exported = 0
    
    if "all" in args.formats or "srt" in args.formats:
        print("💬 Generating SRT file...")
        srt_path = os.path.join(args.output_dir, f"{base_filename}.srt")
        generate_speaker_aware_srt(result["segments"], srt_path)
        print(f"✅ SRT saved to: {srt_path}")
        formats_exported += 1

    if "all" in args.formats or "txt" in args.formats:
        print("📝 Generating TXT file...")
        txt_path = os.path.join(args.output_dir, f"{base_filename}.txt")
        generate_txt(result["segments"], txt_path, include_timestamps=args.timestamps)
        print(f"✅ TXT saved to: {txt_path}")
        formats_exported += 1

    if "all" in args.formats or "md" in args.formats:
        print("🧾 Generating Markdown file...")
        md_path = os.path.join(args.output_dir, f"{base_filename}.md")
        generate_markdown_transcript(result["segments"], md_path, include_timestamps=args.timestamps)
        print(f"✅ Markdown saved to: {md_path}")
        formats_exported += 1

    if "all" in args.formats or "html" in args.formats:
        print("🌐 Generating HTML file...")
        html_path = os.path.join(args.output_dir, f"{base_filename}.html")
        generate_html_transcript(result["segments"], html_path)
        print(f"✅ HTML saved to: {html_path}")
        formats_exported += 1

    if "all" in args.formats or "pdf" in args.formats:
        print("📄 Generating PDF file...")
        pdf_path = os.path.join(args.output_dir, f"{base_filename}.pdf")
        generate_pdf_transcript(result["segments"], pdf_path)
        print(f"✅ PDF saved to: {pdf_path}")
        formats_exported += 1

    export_time = time.time() - export_start_time
    total_time = time.time() - start_time
    
    print("─" * 50)
    print(f"🎉 Transcription completed successfully!")
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"📊 Exported {formats_exported} format(s)")
    print(f"📁 All files saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
