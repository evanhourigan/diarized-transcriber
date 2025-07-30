from fpdf import FPDF

def generate_pdf_transcript(segments, output_path):
    # Check if any segments have speaker information (indicating diarization was used)
    has_speakers = any("speaker" in seg and seg["speaker"] is not None for seg in segments)
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    if has_speakers:
        # Use speaker-aware format
        current_speaker = None
        for seg in segments:
            speaker = seg.get("speaker", "SPEAKER")
            text = seg["text"].strip()
            if speaker != current_speaker:
                pdf.set_font("Arial", style="B", size=12)
                pdf.cell(0, 10, f"{speaker}:", ln=True)
                pdf.set_font("Arial", size=12)
                current_speaker = speaker
            pdf.multi_cell(0, 10, text)
            pdf.ln(5)  # Add space between segments
    else:
        # Use simple format without speaker labels
        for seg in segments:
            text = seg["text"].strip()
            pdf.multi_cell(0, 10, text)
            pdf.ln(5)  # Add space between segments

    pdf.output(output_path)
