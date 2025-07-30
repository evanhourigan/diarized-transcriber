import os
import torch
import whisperx
from whisperx import diarize

def run_transcribe_with_diarization(audio_path, output_dir, model_size="large-v3", skip_diarization=False, num_speakers=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "float32"
    
    print(f"🔧 Using device: {device.upper()}")
    print(f"⚙️  Compute type: {compute_type}")

    print("🤖 Loading Whisper model...")
    model = whisperx.load_model(model_size, device, compute_type=compute_type)
    print(f"✅ Model '{model_size}' loaded successfully")
    
    print("🎤 Transcribing audio...")
    result = model.transcribe(audio_path)
    print(f"✅ Transcription complete - {len(result['segments'])} segments found")

    print("🔗 Loading alignment model...")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    print(f"✅ Alignment model loaded for language: {result['language']}")
    
    print("⚡ Aligning ASR with audio...")
    result = whisperx.align(result["segments"], model_a, metadata, audio_path, device)
    print("✅ Audio alignment completed")

    if not skip_diarization:
        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            print("⚠️  Warning: HUGGINGFACE_TOKEN not set — diarization may fail.")
            print("💡 Set HUGGINGFACE_TOKEN environment variable for speaker diarization")
            print("⏩ Continuing without speaker diarization...")
        else:
            print("🧠 Loading speaker diarization model...")
            diarize_pipeline = diarize.DiarizationPipeline(use_auth_token=token, device=device)
            print("✅ Diarization model loaded")
            
            print("👥 Running speaker diarization...")
            if num_speakers:
                print(f"🎯 Specifying exact number of speakers: {num_speakers}")
                diarize_segments = diarize_pipeline(audio_path, num_speakers=num_speakers)
            else:
                diarize_segments = diarize_pipeline(audio_path)
            print("✅ Speaker diarization completed")

            print("🎯 Assigning speakers to words...")
            result = diarize.assign_word_speakers(diarize_segments, result)
            print("✅ Speaker assignment completed")
    else:
        print("⏩ Skipping diarization as requested.")

    return result
