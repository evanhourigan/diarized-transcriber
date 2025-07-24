from setuptools import setup, find_packages

setup(
    name="whisper-podcast-transcriber",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "whisperx @ git+https://github.com/m-bain/whisperx.git@main",
        "pyannote.audio",
        "python-dotenv",
    ],
    entry_points={
        "console_scripts": [
            "transcribe = whisper_podcast_transcriber.cli:main"
        ]
    },
    python_requires=">=3.8,<3.13",
)