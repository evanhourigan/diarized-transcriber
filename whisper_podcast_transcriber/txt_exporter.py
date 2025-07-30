def format_timestamp(seconds):
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def generate_txt(segments, output_path, include_timestamps=False):
    # Check if any segments have speaker information (indicating diarization was used)
    has_speakers = any("speaker" in seg and seg["speaker"] is not None for seg in segments)
    
    with open(output_path, "w") as f:
        if has_speakers:
            # Use speaker-aware format
            current_speaker = None
            buffer = []
            
            for seg in segments:
                speaker = seg.get("speaker", "SPEAKER")
                text = seg["text"].strip()

                if speaker != current_speaker:
                    if buffer:
                        timestamp = f"[{format_timestamp(seg['start'])}] " if include_timestamps else ""
                        f.write(f"{current_speaker}: {timestamp}{' '.join(buffer)}\n\n")
                        buffer = []
                    current_speaker = speaker

                buffer.append(text)

            if buffer:
                timestamp = f"[{format_timestamp(segments[-1]['start'])}] " if include_timestamps else ""
                f.write(f"{current_speaker}: {timestamp}{' '.join(buffer)}\n")
        else:
            # Use simple format without speaker labels
            for seg in segments:
                text = seg["text"].strip()
                timestamp = f"[{format_timestamp(seg['start'])}] " if include_timestamps else ""
                f.write(f"{timestamp}{text}\n")
            f.write("\n")
