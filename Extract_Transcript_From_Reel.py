import os
import whisper
from moviepy import VideoFileClip

def extract_audio_from_mp4(mp4_path, output_audio_path):
    """Extract audio from MP4 file and save as temporary audio file"""
    video = VideoFileClip(mp4_path)
    audio = video.audio
    audio.write_audiofile(output_audio_path)
    audio.close()
    video.close()
    return output_audio_path

def transcribe_audio(file_path):
    """Transcribe audio and translate to English if needed"""
    model = whisper.load_model("small")
    result = model.transcribe(file_path)
    
    print("Detected Language:", result["language"])
    
    # If the detected language is not English, translate it
    if result["language"] != "en":
        translation = model.transcribe(file_path, task="translate")
        print("\nOriginal Text:\n", result["text"])
        print("\nTranslated to English:\n", translation["text"])
        return translation["text"]
    else:
        print("\nTranscribed Text:\n", result["text"])
        return result["text"]

# Example usage:
if __name__ == "__main__":
    mp4_file = "/Users/user/Downloads/test1.mp4"
    temp_audio = "temp_audio.wav"
    
    # Step 1: Extract audio from MP4
    audio_path = extract_audio_from_mp4(mp4_file, temp_audio)
    
    # Step 2: Transcribe (and translate if needed)
    transcribed_text = transcribe_audio(audio_path)
    
    # Step 3: Clean up temporary audio file
    os.remove(temp_audio)