# Whisper Podcast Transcriber

A CLI tool for transcribing podcast episodes with speaker diarization using WhisperX.

Outputs include:

- 📝 Speaker-paragraph TXT
- 💬 Subtitle file (.srt)
- 🧾 Markdown transcript (.md)
- 📄 PDF transcript (.pdf)
- 🌐 HTML transcript (.html)
- 🔢 Rich JSON metadata (.json)

## Features

- 🎙️ Transcription using OpenAI Whisper models (default: medium)
- 🧠 Automatic speaker diarization (via Hugging Face token)
- 🎯 Specify exact number of speakers for improved accuracy
- ⚙️ Supports model selection (e.g., `base`, `medium`, `large-v3`)
- 🔍 Optional debug mode for verbose output
- 🧵 Outputs include aligned timestamps and speaker tags
- ✅ Easily export all formats or specific ones
- 📁 Smart output naming based on input file
- 🎨 Emoji-fied progress logging
- 📊 Rich progress bars with time estimates

## Installation

First, [install Poetry](https://python-poetry.org/docs/#installation) if you haven't already.

```bash
git clone https://github.com/yourusername/whisper-podcast-transcriber.git
cd whisper-podcast-transcriber
poetry install
```

You can install the CLI globally via:

```bash
poetry run pip install -e .
```

Then make sure you add your Hugging Face token:

### `.env` file in project root

```env
HUGGINGFACE_TOKEN=your_token_here
```

## Usage

From anywhere on your machine:

```bash
transcribe <audio_file> [options]
```

### Basic Examples:

```bash
# Basic transcription with diarization (timestamps included by default)
transcribe my_podcast.mp3

# Specify exact number of speakers (improves accuracy)
transcribe my_podcast.mp3 --num-speakers 2

# Skip diarization for faster processing
transcribe my_podcast.mp3 --skip-diarization

# Use different Whisper model
transcribe my_podcast.mp3 --model large-v3

# Export specific formats
transcribe my_podcast.mp3 --formats txt md srt

# Export all formats
transcribe my_podcast.mp3 --formats all

# Exclude timestamps for clean output
transcribe my_podcast.mp3 --no-timestamps

# Combine multiple options
transcribe my_podcast.mp3 --formats all --num-speakers 2
```

### Advanced Examples:

```bash
# Specify output directory
transcribe my_podcast.mp3 --output-dir ./transcripts

# Use large model with 3 speakers
transcribe interview.wav --model large-v3 --num-speakers 3

# Skip diarization and use debug mode
transcribe call.wav --skip-diarization --debug

# Clean output without timestamps
transcribe call.wav --no-timestamps --formats txt md

# Quiet mode - only progress bars visible
transcribe call.wav --quiet
```

## Options

- `--model`: Whisper model to use (default: medium)
- `--num-speakers`: Exact number of speakers (improves diarization accuracy)
- `--skip-diarization`: Skip speaker diarization for faster processing
- `--no-timestamps`: Exclude timestamps from output files (timestamps included by default)
- `--output-dir`: Directory to save outputs (default: current directory)
- `--formats`: Output formats: txt, md, srt, json, html, pdf, all
- `--debug`: Show detailed debug warnings and logs
- `--quiet`: Suppress all output except progress bars

## Output Files

Files are automatically named based on the input file:

- `my_podcast.mp3` → `my_podcast-transcript.md`, `my_podcast-transcript.txt`, etc.

## Requirements

- Python 3.8 – 3.12 (not yet compatible with 3.13)
- `ffmpeg`
- A valid [Hugging Face token](https://huggingface.co/settings/tokens) for diarization

## License

MIT
