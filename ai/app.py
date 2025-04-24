import os
import ffmpeg
import whisper
import re
import pathlib
from collections import Counter
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
#from werkzeug.utils import secure_filename

# Initializes FastAPI
app = FastAPI()

# Enable CORS (we can adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    # We can restrict this to ["http://localhost:3000"] or our frontend URL
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Folder paths
UPLOAD_FOLDER = "./uploads"
PROCESSED_FOLDER = "./processed"
TRANSCRIPTS_FOLDER = "./transcripts"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPTS_FOLDER, exist_ok=True)

# Load Whisper model. We can change the size of the whisper model: tiny, base, small, medium, large.
whisper_model = whisper.load_model("base")


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    # Use pathlib to clean the filename
    filename = pathlib.Path(file.filename).name
    if not filename.lower().endswith(('.mp4', '.mov', '.wav', '.mp3', '.mp2', '.m4a', '.flac')):
        raise HTTPException(status_code=400, detail="Invalid file format")

    video_path = os.path.join(UPLOAD_FOLDER, filename)
    audio_filename = filename.rsplit('.', 1)[0] + '.mp3'
    audio_path = os.path.join(PROCESSED_FOLDER, audio_filename)

    # Save the uploaded file
    with open(video_path, "wb") as buffer:
        buffer.write(await file.read())

    # Convert to MP3 using ffmpeg
    try:
        ffmpeg.input(video_path).output(audio_path, acodec='libmp3lame', audio_bitrate='192k').run()
    except ffmpeg.Error as e:
        raise HTTPException(status_code=500, detail=f"FFmpeg conversion failed: {str(e)}")
    finally:
        os.remove(video_path)

    # Transcription and Tag generation
    try:
        result = whisper_model.transcribe(audio_path)
        transcript = result["text"]

        # Save transcript to file
        transcript_filename = audio_filename.rsplit('.', 1)[0] + '.txt'
        transcript_path = os.path.join(TRANSCRIPTS_FOLDER, transcript_filename)
        with open(transcript_path, 'w') as f:
            f.write(transcript)

        # Generate tags from transcript
        # Remove punctuation and convert to lowercase'
        cleaned_text = re.sub(r'[^\w\s]', '', transcript.lower())
        words = cleaned_text.split()
        # Filter out common stop words and keep only words with length > 3
        stop_words = {
            'the', 'and', 'for', 'that', 'this', 'with', 'you', 'have',
            'from', 'are', 'they', 'will', 'what', 'when', 'where', 'how',
            'why', 'who', 'which', 'there', 'here', 'their', 'your', 'our'
        }
        filtered_words = [word for word in words if len(word) > 3 and word not in stop_words]
        # Count frequency of words
        word_counts = Counter(filtered_words)
        # Get the 5 most common words as tags
        tags = [word for word, _ in word_counts.most_common(5)]

        # Create a simple summary (first few sentences as a preview)
        sentences = re.split(r'[.!?]+', transcript)
        # Take about 20% of the sentences or at least 3 sentences
        summary_length = max(3, int(len(sentences) * 0.2))
        short_summary = '. '.join(sentences[:summary_length]) + '.'
        detailed_summary_length = max(5, int(len(sentences) * 0.3))
        detailed_summary = '. '.join(sentences[:detailed_summary_length]) + '.'

        return {
            "tags": tags,
            "transcript": transcript,
            "shortSummary": short_summary,
            "detailedSummary": detailed_summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription or summarization failed: {str(e)}")


# Optional if you run with `python app.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
