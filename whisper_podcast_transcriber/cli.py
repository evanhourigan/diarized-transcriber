#!/usr/bin/env python3

import argparse
import os
import warnings
import logging
import time
import sys
import contextlib
import io
from dotenv import load_dotenv

# Set environment variables to suppress verbose output from underlying libraries
# This must be done BEFORE any other imports
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["PYTORCH_LIGHTNING_LOGGING_LEVEL"] = "ERROR"
os.environ["SPEECHBRAIN_VERBOSITY"] = "ERROR"
os.environ["PYTORCH_LIGHTNING_VERBOSITY"] = "ERROR"
os.environ["PYANNOTE_VERBOSITY"] = "ERROR"

# Redirect stderr to suppress DEBUG messages early
original_stderr = sys.stderr
sys.stderr = io.StringIO()

# Configure logging BEFORE any other imports - more aggressive suppression
logging.basicConfig(level=logging.ERROR)  # Only show ERROR and above by default
logging.getLogger("whisperx").setLevel(logging.ERROR)
logging.getLogger("pyannote").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("speechbrain").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.utilities.migration.utils").setLevel(logging.ERROR)

# Load environment variables from .env file
load_dotenv()

from whisper_podcast_transcriber.diarization import run_transcribe_with_diarization
from whisper_podcast_transcriber.srt_exporter import generate_speaker_aware_srt
from whisper_podcast_transcriber.txt_exporter import generate_txt
from whisper_podcast_transcriber.markdown_exporter import generate_markdown_transcript
from whisper_podcast_transcriber.html_exporter import generate_html_transcript
from whisper_podcast_transcriber.pdf_exporter import generate_pdf_transcript
from whisper_podcast_transcriber.rich_progress import PersistentProgress, print_success_panel

def format_duration(seconds: float) -> str:
    """Convert seconds to H:MM:SS format, showing hours only when needed."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"

def main():
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Transcribe and optionally diarize an audio file, exporting to various formats.",
        epilog="Example: transcribe audio.wav --formats all"
    )
    parser.add_argument("audio_path", help="Path to audio file (e.g., .wav)")
    parser.add_argument("--output-dir", dest="output_dir", default=".", help="Directory to save outputs (default: current directory)")
    parser.add_argument("--model", default="medium", help="Whisper model to use (default: medium)")
    parser.add_argument("--skip-diarization", dest="skip_diarization", action="store_true", help="Skip speaker diarization")
    parser.add_argument("--num-speakers", type=int, help="Exact number of speakers (improves diarization accuracy)")
    parser.add_argument("--no-timestamps", dest="no_timestamps", action="store_true", help="Exclude timestamps from output files")
    parser.add_argument("--formats", nargs="+", default=["md"], help="Output formats: txt, md, srt, json, html, pdf, all")
    parser.add_argument("--debug", action="store_true", help="Show detailed debug warnings and logs")
    parser.add_argument("--quiet", action="store_true", help="Suppress all output except progress bars")

    args = parser.parse_args()

    if not args.debug:
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
    else:
        # Enable debug logging only if --debug flag is used
        logging.basicConfig(level=logging.DEBUG)
        # Re-enable all loggers for debug mode
        logging.getLogger("whisperx").setLevel(logging.DEBUG)
        logging.getLogger("pyannote").setLevel(logging.DEBUG)
        logging.getLogger("torch").setLevel(logging.DEBUG)
        logging.getLogger("speechbrain").setLevel(logging.DEBUG)
        logging.getLogger("transformers").setLevel(logging.DEBUG)
        logging.getLogger("datasets").setLevel(logging.DEBUG)
        logging.getLogger("lightning").setLevel(logging.DEBUG)
        logging.getLogger("pytorch_lightning").setLevel(logging.DEBUG)
        logging.getLogger("huggingface_hub").setLevel(logging.DEBUG)
        logging.getLogger("urllib3").setLevel(logging.DEBUG)
        logging.getLogger("requests").setLevel(logging.DEBUG)

    # Generate base filename from input audio file
    audio_basename = os.path.splitext(os.path.basename(args.audio_path))[0]
    base_filename = f"{audio_basename}-transcript"

    # Start the transcription process
    if not args.quiet:
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
    print()  # Add blank line before progress bars

    start_time = time.time()
    
    # Suppress stderr during transcription if not in debug mode
    if not args.debug:
        with contextlib.redirect_stderr(io.StringIO()):
            result = run_transcribe_with_diarization(
                audio_path=args.audio_path,
                output_dir=args.output_dir,
                model_size=args.model,
                skip_diarization=args.skip_diarization,
                num_speakers=args.num_speakers,
                quiet=args.quiet
            )
    else:
        result = run_transcribe_with_diarization(
            audio_path=args.audio_path,
            output_dir=args.output_dir,
            model_size=args.model,
            skip_diarization=args.skip_diarization,
            num_speakers=args.num_speakers,
            quiet=args.quiet
        )

    transcription_time = time.time() - start_time
    if not args.quiet:
        print(f"✅ Transcription completed in {format_duration(transcription_time)}")
        print(f"📝 Found {len(result['segments'])} segments")
    
    # Create output directory if it doesn't exist
    if args.output_dir != ".":
        if not args.quiet:
            print("📁 Creating output directory...")
        os.makedirs(args.output_dir, exist_ok=True)

    # Export to different formats
    if not args.quiet:
        print("📤 Exporting transcripts...")
    print()  # Add blank line before progress bars
    export_start_time = time.time()
    
    # Determine which formats to export
    export_formats = []
    if "all" in args.formats:
        export_formats = ["srt", "txt", "md", "html", "pdf"]
    else:
        export_formats = args.formats
    
    formats_exported = 0
    total_formats = len(export_formats)
    
    # Export with Persistent progress
    with PersistentProgress() as progress:
        progress.start_task("Exporting formats")
        
        for i, format_type in enumerate(export_formats):
            if format_type == "srt":
                srt_path = os.path.join(args.output_dir, f"{base_filename}.srt")
                generate_speaker_aware_srt(result["segments"], srt_path)
                formats_exported += 1

            elif format_type == "txt":
                txt_path = os.path.join(args.output_dir, f"{base_filename}.txt")
                generate_txt(result["segments"], txt_path, include_timestamps=not args.no_timestamps)
                formats_exported += 1

            elif format_type == "md":
                md_path = os.path.join(args.output_dir, f"{base_filename}.md")
                generate_markdown_transcript(result["segments"], md_path, include_timestamps=not args.no_timestamps)
                formats_exported += 1

            elif format_type == "html":
                html_path = os.path.join(args.output_dir, f"{base_filename}.html")
                generate_html_transcript(result["segments"], html_path)
                formats_exported += 1

            elif format_type == "pdf":
                pdf_path = os.path.join(args.output_dir, f"{base_filename}.pdf")
                generate_pdf_transcript(result["segments"], pdf_path)
                formats_exported += 1
        
        progress.complete_task(f"Exported {formats_exported} format(s)")

    export_time = time.time() - export_start_time
    total_time = time.time() - start_time
    
    # Restore stderr
    sys.stderr = original_stderr
    
    if not args.quiet:
        print("─" * 50)
        print(f"🎉 Transcription completed successfully!")
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"📊 Exported {formats_exported} format(s)")
        print(f"📁 All files saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
