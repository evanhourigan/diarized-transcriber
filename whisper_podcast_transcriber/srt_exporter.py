def generate_speaker_aware_srt(segments, output_path):
    def format_timestamp(seconds):
        ms = int((seconds - int(seconds)) * 1000)
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    # Check if any segments have speaker information (indicating diarization was used)
    has_speakers = any("speaker" in seg and seg["speaker"] is not None for seg in segments)

    with open(output_path, "w") as f:
        for i, seg in enumerate(segments, 1):
            start = format_timestamp(seg['start'])
            end = format_timestamp(seg['end'])
            text = seg["text"].strip()
            
            if has_speakers:
                speaker = seg.get("speaker", "SPEAKER")
                f.write(f"{i}\n{start} --> {end}\n{speaker}: {text}\n\n")
            else:
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
