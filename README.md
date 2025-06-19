# SceneIT
# 🎬 Multimodal Emotion Extraction and Meme Injection from Movie Content

This project extracts **multimodal emotions** (from audio, text, and transcript), and **augments videos** with **contextually and emotionally matching memes** from movie dialogues.

---

## 📁 Project Overview

### 1. **Audio-Based Multilingual Emotion Transcription (Whisper)**
This module:
- Extracts audio from an input MP4 video.
- Transcribes and **translates** the speech (if not in English).
- Uses [OpenAI's Whisper](https://github.com/openai/whisper) for speech-to-text.

> 📂 `whisper_transcription.py`

### 2. **Text-Based Emotion Extraction from Movie Transcript**
This module:
- Extracts dialogues from a **PDF transcript**.
- Uses Hugging Face’s transformer models to classify the **dominant emotion** of each line.
- Filters out high-confidence meme-worthy lines.

> 📂 `extract_emotions_from_transcript.py`

### 3. **Meme Insertion Based on Emotion & Semantic Similarity**
This module:
- Takes a scene (or line) and trigger word (like `money`, `love`, etc.).
- Uses **TF-IDF + Zero-shot Classification** to match with the best meme line from the transcript.
- Ensures:
  - Emotion Match ✅
  - Keyword Contextual Similarity ✅
  - Cosine Similarity ✅
- Inserts the meme right after the trigger word.

> 📂 `meme_injector.py`

---

## 🛠️ Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/your-username/movie-emotion-meme.git
cd movie-emotion-meme

