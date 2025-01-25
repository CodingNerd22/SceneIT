import moviepy.editor as mp
import speech_recognition as sr
from langdetect import detect

def extract_audio_from_video(video_path, audio_path):
    """Extracts audio from video."""
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)
    print(f"Audio extracted to {audio_path}")

def transcribe_audio(audio_path):
    """Transcribes audio to text."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        print("Processing audio...")
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        print("Transcription completed.")
        return text
    except sr.UnknownValueError:
        print("Speech Recognition could not understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"Error with Speech Recognition service: {e}")
        return None

def detect_language(text):
    """Detects the language of the given text."""
    try:
        language = detect(text)
        print(f"Detected language: {language}")
        return language
    except Exception as e:
        print(f"Language detection failed: {e}")
        return None

def main():
    # Paths to input and output files
    video_path = "example_video.mp4"  # Replace with your video file path
    audio_path = "extracted_audio.wav"
    
    # Step 1: Extract audio from video
    extract_audio_from_video(video_path, audio_path)
    
    # Step 2: Transcribe audio to text
    transcribed_text = transcribe_audio(audio_path)
    if transcribed_text:
        print(f"Transcribed Text: {transcribed_text}")
        
        # Step 3: Detect language
        language = detect_language(transcribed_text)
        if language:
            print(f"Language Detected: {language}")

if __name__ == "__main__":
    main()
