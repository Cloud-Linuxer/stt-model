# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository is for testing Speech-to-Text (STT) models with Large Language Models (LLMs). The project focuses on evaluating and comparing different STT model performance.

## Development Setup

Since this is a new project, the following structure is recommended:

### Python Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/Mac

# Install dependencies (once requirements.txt is created)
pip install -r requirements.txt
```

## Recommended Project Structure

```
stt-model/
├── models/           # STT model implementations
├── tests/            # Test files for model evaluation
├── data/             # Audio samples and test data
├── results/          # Test results and benchmarks
├── scripts/          # Utility scripts
├── config/           # Configuration files
└── requirements.txt  # Python dependencies
```

## Key Dependencies to Consider

For STT model testing, you'll likely need:
- **Audio Processing**: librosa, soundfile, pydub
- **STT Models**: whisper, speechrecognition, wav2vec2
- **LLM Integration**: transformers, openai, langchain
- **Evaluation**: jiwer (for WER calculation), nltk
- **Data Handling**: pandas, numpy

## Testing Approach

When implementing STT model tests:
1. Load audio samples from the data directory
2. Process audio through different STT models
3. Compare transcription accuracy using metrics like WER (Word Error Rate)
4. Store results in a structured format for analysis

## Configuration Management

Use configuration files (JSON or YAML) to manage:
- Model parameters
- Audio preprocessing settings
- Evaluation metrics
- API keys (store in environment variables, never commit)