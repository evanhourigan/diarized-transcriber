def generate_html_transcript(segments, output_path):
    # Check if any segments have speaker information (indicating diarization was used)
    has_speakers = any("speaker" in seg and seg["speaker"] is not None for seg in segments)
    
    with open(output_path, "w") as f:
        f.write("<html><body>\n")
        
        if has_speakers:
            # Use speaker-aware format
            current_speaker = None
            for seg in segments:
                speaker = seg.get("speaker", "SPEAKER")
                text = seg["text"].strip()
                if speaker != current_speaker:
                    f.write(f"<h3>{speaker}</h3>\n")
                    current_speaker = speaker
                f.write(f"<p>{text}</p>\n")
        else:
            # Use simple format without speaker labels
            for seg in segments:
                text = seg["text"].strip()
                f.write(f"<p>{text}</p>\n")
                
        f.write("</body></html>")
