import os
from moviepy.video.io.VideoFileClip import VideoFileClip
from utils import process_pdf, find_most_similar, find_full_context, parse_timestamps
import re

def extract_video_segment(input_video, output_video, start_time, end_time):
    """
    Extracts a segment from a video based on start and end timestamps.
    """
    video = VideoFileClip(input_video)
    segment = video.subclipped(start_time, end_time)
    segment.write_videofile(output_video, codec="libx264", audio=True, audio_codec="aac")
    video.close()

def search_pdfs_in_folder(folder_path, prompt):
    """
    Searches all PDFs in a folder for the most relevant match to the prompt.
    """
    best_match = None
    best_score = -1
    best_pdf_path = None

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            chunks, embeddings = process_pdf(pdf_path)
            results = find_most_similar(prompt, chunks, embeddings, top_k=1)
            if results and results[0][1] > best_score:
                best_match = results[0]
                best_score = results[0][1]
                best_pdf_path = pdf_path

    return best_match, best_pdf_path

def main():
    pdf_folder = input("Give the path to the folder containing PDFs: ")
    
    while True:
        user_prompt = input("prompt: ")
        if user_prompt.lower() == 'exit':
            break
        
        best_match, best_pdf_path = search_pdfs_in_folder(pdf_folder, user_prompt)
        
        if best_match:
            chunk, score = best_match
            context = find_full_context(chunk, process_pdf(best_pdf_path)[0])
            output_text = f"\nMatch (Page {chunk['page']})\nContext: {context}\n" + "-" * 80 + "\n"
            print(output_text)
            
            try:
                start_time, end_time = parse_timestamps(output_text)
                print(f"Extracted timestamps: Start = {start_time}s, End = {end_time}s")
                
                video_name = os.path.splitext(os.path.basename(best_pdf_path))[0] + ".mp4"
                input_video = os.path.join(pdf_folder, video_name)
                output_video = os.path.join(pdf_folder, "output_segment.mp4")
                
                if os.path.exists(input_video):
                    extract_video_segment(input_video, output_video, start_time, end_time)
                    print(f"Video segment saved to {output_video}")
                else:
                    print(f"Video file {input_video} not found.")
            except ValueError as e:
                print(f"Error: {e}")
        else:
            print("No relevant match found.")

if __name__ == "__main__":
    main()