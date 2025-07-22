# Whisper Podcast Transcriber

A CLI tool for transcribing podcast episodes with speaker diarization using WhisperX.

Outputs:

- 📝 Plain-text transcript (`.txt`)
- 💬 Subtitle file (`.srt`)
- 🧾 Rich JSON metadata (`.json`)

## Features

- 🎙️ Transcription using OpenAI Whisper models
- 🧠 Automatic speaker diarization (via Hugging Face token)
- ⚙️ Supports model selection (e.g., `base`, `medium`, `large`)
- 🧵 Outputs include aligned timestamps and speaker tags

## Installation

Clone the repo and install in editable mode:

git clone https://github.com/yourusername/whisper-podcast-transcriber.git
cd whisper-podcast-transcriber
python -m pip install -e .

Then make sure you add your Hugging Face token:

# .env file in project root

HUGGINGFACE_TOKEN=your_token_here

## Usage

From anywhere on your machine:

transcribe <audio_file> [whisper_model]

Example:

transcribe my_podcast.mp3 medium

This will generate:

- my_podcast_transcript.txt
- my_podcast_transcript.srt
- my_podcast_transcript.json

## Requirements

- Python 3.8 – 3.12 (not yet compatible with 3.13)
- ffmpeg
- A valid [Hugging Face token](https://huggingface.co/settings/tokens) for diarization

## License

MIT
