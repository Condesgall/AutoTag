from flask import Flask, request, jsonify
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
import torch
import os
#for extractinng audio from video files
from moviepy import editor as mp
#for working with audio files
from pydub import AudioSegment
import os
os.environ["FFMPEG_BINARY"] = "/path/to/custom/ffmpeg"
os.environ["FFPLAY_BINARY"] = "/path/to/custom/ffplay"

app = Flask(__name__)

# Initialize models on GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if device == "cuda" else -1)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if device == "cuda" else -1)

# Load Whisper model
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2").to(device)

def extract_audio(file_path):
    """Extracts audio from video file and saves as WAV"""
    audio_path = file_path.rsplit(".", 1)[0] + ".wav"
    video = mp.VideoFileClip(file_path)
    video.audio.write_audiofile(audio_path)
    return audio_path

def transcribe_audio(file_path):
    """Transcribes audio using Whisper"""
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export("converted.wav", format="wav")

    # Process with Whisper
    input_features = processor("converted.wav", return_tensors="pt").input_features.to(device)
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

@app.route("/generate-tags", methods=["POST"])
def generate_tags():
    """Processes audio and generates summary & tags"""
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file provided"}), 400

    file_path = f"temp_{file.filename}"
    file.save(file_path)

    if file_path.endswith((".mp4", ".mov", ".avi")):
        file_path = extract_audio(file_path)

    text = transcribe_audio(file_path)
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
    candidate_labels = ["technology", "health", "sports", "entertainment", "finance"]
    tags = classifier(summary, candidate_labels=candidate_labels)

    os.remove(file_path)

    return jsonify({"summary": summary, "tags": tags["labels"]})

@app.route("/process-video", methods=["POST"])
def process_video():
    """Handles direct video uploads, extracts speech, and summarizes"""
    file = request.files.get("video")
    if not file:
        return jsonify({"error": "No video file provided"}), 400

    file_path = f"temp_{file.filename}"
    file.save(file_path)

    audio_path = extract_audio(file_path)
    text = transcribe_audio(audio_path)
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
    candidate_labels = ["technology", "health", "sports", "entertainment", "finance"]
    tags = classifier(summary, candidate_labels=candidate_labels)

    os.remove(file_path)
    os.remove(audio_path)

    return jsonify({"transcription": text, "summary": summary, "tags": tags["labels"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)
