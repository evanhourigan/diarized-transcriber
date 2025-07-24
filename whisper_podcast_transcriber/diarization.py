from whisperx.diarize import DiarizationPipeline

def perform_diarization(audio_path, hf_token, device="cpu"):
    """
    Perform speaker diarization using the DiarizationPipeline.

    Args:
        audio_path: Path to the audio file.
        hf_token: Hugging Face token.
        device: "cuda" or "cpu"

    Returns:
        A pandas DataFrame with columns: segment, label, speaker, start, end
    """
    print("🔉 Performing speaker diarization...")
    pipeline = DiarizationPipeline(
        model_name="pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
        device=device,
    )
    diarize_df = pipeline(audio_path)
    print("📻 Running diarization inference...")
    return diarize_df