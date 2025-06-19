import cv2
from deepface import DeepFace
from collections import defaultdict
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, pipeline
import torch
import librosa
from pydub import AudioSegment
import whisper
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer

# ============ VIDEO SETTINGS ============
video_path = "/Users/user/VIIT/Backup/Part4/Media/test6-en.mp4 "  # <-- Fill this

# ============ VISUAL EMOTION ANALYSIS (DeepFace) ============
def analyze_video_emotions(video_path, skip_frames=5, resize_factor=0.5):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    emotion_sums = defaultdict(float)
    processed_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        # Resize frame for faster processing
        frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)

        try:
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'
            )
            emotions = result[0]['emotion']
            for emotion, score in emotions.items():
                emotion_sums[emotion] += score
            processed_frames += 1

        except Exception as e:
            print(f"Skipping frame {frame_count}: {e}")

    cap.release()

    if processed_frames == 0:
        print("No faces detected in the video.")
        return []

    normalized_emotions = []
    for emotion, total_score in emotion_sums.items():
        avg_score = total_score / processed_frames
        normalized_score = avg_score / 100
        normalized_emotions.append((emotion, normalized_score))

    normalized_emotions.sort(key=lambda x: x[1], reverse=True)
    return normalized_emotions

# ============ AUDIO EMOTION ANALYSIS ============
def extract_audio_from_mp4(mp4_path, output_wav_path="temp_audio.wav"):
    audio = AudioSegment.from_file(mp4_path, format="mp4")
    audio.export(output_wav_path, format="wav")
    return output_wav_path

def classify_audio_emotions(audio_path):
    emotion_model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    feature_extractor = AutoFeatureExtractor.from_pretrained(emotion_model_name)
    emotion_model = AutoModelForAudioClassification.from_pretrained(emotion_model_name)

    speech, sr = librosa.load(audio_path, sr=16000)
    inputs = feature_extractor(
        speech,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=16000 * 10,
    )

    with torch.no_grad():
        logits = emotion_model(**inputs).logits

    probabilities = torch.softmax(logits, dim=-1).numpy()[0]
    emotion_labels = emotion_model.config.id2label

    results = [{"label": emotion_labels[i], "score": float(prob)}
               for i, prob in enumerate(probabilities)]

    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    return sorted_results

# ============ SPEECH-TO-TEXT EMOTION ANALYSIS ============
def detect_language(text):
    try:
        return detect(text)
    except:
        return 'en'

def translate_to_english(text, src_lang):
    if src_lang == 'en':
        return text
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

def transcribe_and_classify_text_emotions(audio_path):
    whisper_model = whisper.load_model("small", device="cuda" if torch.cuda.is_available() else "cpu")
    emotion_classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True,
        device=0 if torch.cuda.is_available() else -1
    )

    result = whisper_model.transcribe(audio_path, fp16=torch.cuda.is_available())
    transcript = result["text"]

    lang = detect_language(transcript)
    if lang != 'en':
        transcript = translate_to_english(transcript, lang)

    emotions = emotion_classifier(transcript)
    sorted_emotions = sorted(emotions[0], key=lambda x: x['score'], reverse=True)
    return transcript, sorted_emotions

# ============ HARMONIZE + COMBINE ============
def harmonize_label(label):
    mapping = {
        "joy": "happy",
        "happiness": "happy",
        "sadness": "sad",
        "fearful": "fear",
        "fear": "fear",
        "anger": "angry",
        "angry": "angry",
        "calm": "neutral",
        "neutral": "neutral",
        "disgust": "disgust",
        "surprise": "surprise",
        "surprised": "surprise",
    }
    label = label.lower()
    return mapping.get(label, label)

def combine_emotions(video_emotions, audio_emotions, text_emotions):
    final_scores = defaultdict(float)

    # Weight settings
    visual_weight = 0.4
    audio_weight = 0.3
    text_weight = 0.3

    # --- Add Visual (Facial) Emotions ---
    for emotion, score in video_emotions:
        harmonized = harmonize_label(emotion)
        final_scores[harmonized] += score * visual_weight

    # --- Add Audio (Tonal) Emotions ---
    for result in audio_emotions:
        emotion = result['label']
        score = result['score']
        harmonized = harmonize_label(emotion)
        final_scores[harmonized] += score * audio_weight

    # --- Add Text (Transcript) Emotions ---
    for result in text_emotions:
        emotion = result['label']
        score = result['score']
        harmonized = harmonize_label(emotion)
        final_scores[harmonized] += score * text_weight

    # --- Sort final scores ---
    sorted_final = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_final

# ============ MAIN ============
def main(video_path):
    print("\n=== Extracting Emotions from Video Frames ===")
    video_emotions = analyze_video_emotions(video_path)

    print("\n=== Extracting Audio and Analyzing Audio Emotions ===")
    wav_audio_path = extract_audio_from_mp4(video_path)
    audio_emotions = classify_audio_emotions(wav_audio_path)

    print("\n=== Transcribing Audio and Analyzing Text Emotions ===")
    transcript, text_emotions = transcribe_and_classify_text_emotions(wav_audio_path)

    # ============ Print All Results ============
    print("\n\n=========== FINAL COMPARISON RESULTS ===========\n")

    print("\n--- Video Frame Emotions (Visual) ---")
    for emotion, score in video_emotions:
        print(f"{emotion}: {score:.4f}")

    print("\n--- Audio Emotions (Speech Sound) ---")
    for result in audio_emotions:
        print(f"{result['label']}: {result['score']:.4f}")

    print("\n--- Text Emotions (Speech Meaning) ---")
    print(f"\nTranscript:\n{transcript}\n")
    for result in text_emotions:
        print(f"{result['label']}: {result['score']:.4f}")

    # ====== Now Combine All ======
    print("\n\n--- Combined Final Emotions ---")
    combined_emotions = combine_emotions(video_emotions, audio_emotions, text_emotions)
    for emotion, score in combined_emotions:
        print(f"{emotion}: {score:.4f}")

# ============ RUN ============
if __name__ == "__main__":
    main(video_path)
