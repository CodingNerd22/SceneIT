# 🎥 Multimodal Emotion Detection & Meme Injection for Movie Videos

This project extracts and combines **multimodal emotions** (facial, audio, textual) from videos and **enriches them** with semantically and emotionally relevant **movie memes**.

---

## 📂 Project Structure

### 1. `emotion_detection_from_movie.py`

* Extracts **dialogues** from movie **transcript PDF**.
* Classifies emotion for each dialogue using a transformer model (`j-hartmann/emotion-english-distilroberta-base`).
* Returns meme-worthy lines with high-confidence emotional labels.

### 2. `emotion_recogniton_all_2.py`

* Takes a movie/video file.
* Extracts **visual emotion** using `DeepFace` from video frames.
* Extracts **audio emotion** using `wav2vec2`-based emotion recognition.
* Transcribes and detects **text emotion** using Whisper + Transformers.
* **Harmonizes and combines** these into a final emotion profile for the scene.

### 3. `add_meme_2.py`

* Takes a scene line (from user or system).
* Uses:

  * Emotion Matching
  * Semantic Similarity (TF-IDF)
  * Context Matching (Zero-shot Classification)
* Extracts the **best matching meme** line from movie dialogues (from transcript).
* **Injects** the meme line into the scene after a specified trigger word.

---

## 📚 Example Flow

1. Extract emotions from your input video (`any_video.mp4`) using `emotion_recogniton_all_2.py`.
2. Extract emotional movie dialogues from the transcript (`movie.pdf`) using `emotion_detection_from_movie.py`.
3. Inject the most relevant meme from movie into your generated scene using `add_meme_2.py`.

---

## 📆 Dependencies

Install all required libraries using:

```bash
pip install -r requirements.txt
```

---


## 🚀 Future Enhancements

* Add **meme overlay** directly on video clips.
* Use **facial expression dynamics** for deeper context.
* Support for **SRT subtitles** to align memes to timecodes.

---

## 🧡 Credits

* [OpenAI Whisper](https://github.com/openai/whisper)
* [Hugging Face Transformers](https://huggingface.co/models)
* [DeepFace](https://github.com/serengil/deepface)
* [wav2vec2 SER](https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition)

---

## 🔒 License

MIT License
