def format_timestamp(seconds):
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def generate_markdown_transcript(segments, output_path, include_timestamps=False):
    # Check if any segments have speaker information (indicating diarization was used)
    has_speakers = any("speaker" in seg and seg["speaker"] is not None for seg in segments)
    
    with open(output_path, "w") as f:
        if has_speakers:
            # Use speaker-aware format
            current_speaker = None
            for seg in segments:
                speaker = seg.get("speaker", "SPEAKER")
                text = seg["text"].strip()
                timestamp = f"[{format_timestamp(seg['start'])}]" if include_timestamps else ""
                
                if speaker != current_speaker:
                    if current_speaker is not None:
                        f.write("\n")
                    f.write(f"**{speaker}:**{timestamp}\n")
                    current_speaker = speaker
                f.write(f"{text} ")
        else:
            # Use simple format without speaker labels
            for seg in segments:
                text = seg["text"].strip()
                timestamp = f"[{format_timestamp(seg['start'])}] " if include_timestamps else ""
                f.write(f"{timestamp}{text}\n")
        f.write("\n")
