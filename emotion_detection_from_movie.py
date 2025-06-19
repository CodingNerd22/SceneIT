import pdfplumber
from transformers import pipeline

# Load emotion classifier model
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Function to classify emotions in dialogues
def classify_emotions(text):
    dialogues = text.split("\n")  # Assuming dialogues are on separate lines
    meme_worthy_lines = []

    for dialogue in dialogues:
        if len(dialogue.strip()) > 0:  # Ignore empty lines
            result = emotion_classifier(dialogue)
            if result and result[0]['score'] > 0.9:  # Only keep high-confidence emotions
                meme_worthy_lines.append((dialogue, result[0]['label'], result[0]['score']))

    return meme_worthy_lines

# PDF file path
pdf_path = "path_to_file"

# Extract text
transcript = extract_text_from_pdf(pdf_path)

# Analyze emotions
meme_data = classify_emotions(transcript)

# Print the results
for line, emotion, score in meme_data:
    print(f"Dialogue: {line}\nEmotion: {emotion} (Score: {score:.2f})\n{'-'*50}")
