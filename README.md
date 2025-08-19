# Diarized Transcriber

A CLI tool for transcribing conversation audio files with speaker diarization using WhisperX.

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

### Quick Install (Recommended)

```bash
# Install with pipx (recommended for CLI tools)
pipx install git+https://github.com/evanhourigan/diarized-transcriber.git

# Install directly from GitHub
pip install git+https://github.com/evanhourigan/diarized-transcriber.git

# Or use uvx to run without installing
uvx run diarized-transcriber transcribe <audio_file>
```

### Development Install

First, [install Poetry](https://python-poetry.org/docs/#installation) if you haven't already.

```bash
git clone https://github.com/evanhourigan/diarized-transcriber.git
cd diarized-transcriber
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

## Testing

Run the test suite to ensure everything is working correctly:

```bash
# Run all tests
python run_tests.py

# Or run individual test modules
python test_markdown_exporter.py
python test_diarization.py
```

## Demo

Try it out with a sample conversation:

```bash
# Download a sample audio file (you'll need to provide your own)
# Then run:
transcribe sample-conversation.mp3 --formats all --num-speakers 2

# This will generate all output formats for you to explore
```

## Usage

From anywhere on your machine:

```bash
transcribe <audio_file> [options]
```

### Basic Examples:

```bash
# Basic transcription with diarization (timestamps included by default)
transcribe conversation.mp3

# Specify exact number of speakers (improves accuracy)
transcribe conversation.mp3 --num-speakers 2

# Skip diarization for faster processing
transcribe conversation.mp3 --skip-diarization

# Use different Whisper model
transcribe conversation.mp3 --model large-v3

# Export specific formats
transcribe conversation.mp3 --formats txt md srt

# Export all formats
transcribe conversation.mp3 --formats all

# Exclude timestamps for clean output
transcribe conversation.mp3 --no-timestamps

# Combine multiple options
transcribe conversation.mp3 --formats all --num-speakers 2
```

### Advanced Examples:

```bash
# Specify output directory
transcribe conversation.mp3 --output-dir ./transcripts

# Use large model with 3 speakers
transcribe conversation.wav --model large-v3 --num-speakers 3

# Skip diarization and use debug mode
transcribe conversation.wav --skip-diarization --debug

# Clean output without timestamps
transcribe conversation.wav --no-timestamps --formats txt md

# Quiet mode - only progress bars visible
transcribe conversation.wav --quiet
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

## Model Selection & Performance

### Whisper Model Tradeoffs

- **`base`**: Fastest (~1x speed), lower accuracy, good for quick drafts
- **`medium`**: Balanced (~2x speed), good accuracy, recommended default
- **`large-v3`**: Slowest (~4x speed), highest accuracy, best for final transcripts

### Speaker Diarization Accuracy

- **Accuracy varies** based on audio quality, speaker clarity, and background noise
- **Specifying `--num-speakers`** significantly improves accuracy when you know the exact count
- **Best results** with clear audio, minimal background noise, and distinct speaker voices
- **Processing time** increases with audio length and speaker count

## Output Files

Files are automatically named based on the input file:

- `conversation.mp3` → `conversation-transcript.md`, `conversation-transcript.txt`, etc.

## JSON Output Schema

When using `--formats json`, the output includes rich metadata:

```json
{
  "audio_file": "conversation.mp3",
  "model_used": "medium",
  "processing_time": 45.2,
  "segments": [
    {
      "start": 0.0,
      "end": 3.2,
      "speaker": "Speaker 1",
      "text": "Hello, how are you today?",
      "confidence": 0.95
    },
    {
      "start": 3.4,
      "end": 6.1,
      "speaker": "Speaker 2", 
      "text": "I'm doing well, thank you.",
      "confidence": 0.92
    }
  ],
  "speakers_detected": 2,
  "total_duration": 180.5
}
```

## Requirements

- Python 3.8 – 3.12 (not yet compatible with 3.13)
- `ffmpeg`
- A valid [Hugging Face token](https://huggingface.co/settings/tokens) for diarization

## License

MIT
