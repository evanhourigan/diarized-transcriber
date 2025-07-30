import os
import torch
import whisperx
from whisperx import diarize
from .rich_progress import PersistentProgress, print_success_panel
import threading
import time

def run_transcribe_with_diarization(audio_path, output_dir, model_size="large-v3", skip_diarization=False, num_speakers=None, quiet=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "float32"
    
    if not quiet:
        print(f"🔧 Using device: {device.upper()}")
        print(f"⚙️  Compute type: {compute_type}")

    # Use a single persistent progress instance for all steps
    with PersistentProgress() as progress:
        # Load Whisper model
        progress.start_task("Loading Whisper model")
        model = whisperx.load_model(model_size, device, compute_type=compute_type)
        progress.complete_task(f"Model '{model_size}' loaded successfully")
        
        # Transcribe
        progress.start_task("Transcribing audio")
        result = model.transcribe(audio_path)
        progress.complete_task(f"Transcription complete - {len(result['segments'])} segments found")

        # Load alignment model
        progress.start_task("Loading alignment model")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        progress.complete_task(f"Alignment model loaded for language: {result['language']}")
        
        # Align
        progress.start_task("Aligning ASR with audio")
        result = whisperx.align(result["segments"], model_a, metadata, audio_path, device)
        progress.complete_task("Audio alignment completed")

        if not skip_diarization:
            token = os.getenv("HUGGINGFACE_TOKEN")
            if not token:
                if not quiet:
                    print("⚠️  Warning: HUGGINGFACE_TOKEN not set — diarization may fail.")
                    print("💡 Set HUGGINGFACE_TOKEN environment variable for speaker diarization")
                    print("⏩ Continuing without speaker diarization...")
            else:
                # Diarization steps with same persistent progress
                # Load diarization model
                progress.start_task("Loading speaker diarization model")
                diarize_pipeline = diarize.DiarizationPipeline(use_auth_token=token, device=device)
                progress.complete_task("Diarization model loaded")
                
                # Run diarization
                progress.start_task("Running speaker diarization")
                if num_speakers:
                    if not quiet:
                        print(f"🎯 Specifying exact number of speakers: {num_speakers}")
                    diarize_segments = diarize_pipeline(audio_path, num_speakers=num_speakers)
                else:
                    diarize_segments = diarize_pipeline(audio_path)
                progress.complete_task("Speaker diarization completed")

                # Assign speakers
                progress.start_task("Assigning speakers to words")
                result = diarize.assign_word_speakers(diarize_segments, result)
                progress.complete_task("Speaker assignment completed")
        else:
            if not quiet:
                print("⏩ Skipping diarization as requested.")

    return result
