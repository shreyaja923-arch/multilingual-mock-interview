# Whisper STT Service
# Run in Google Colab with T4 GPU

import whisper
import torch
import tempfile
import os
import time
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from deep_translator import GoogleTranslator

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("large-v3", device=device)

app = FastAPI(title="Whisper STT Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

LANG_CODES = {
    "Hindi": "hi", "Tamil": "ta", "Telugu": "te",
    "Bengali": "bn", "Marathi": "mr", "Gujarati": "gu",
    "Kannada": "kn", "Malayalam": "ml", "Punjabi": "pa",
    "English": "en"
}

@app.get("/")
def home():
    return {
        "status": "Whisper STT Running!",
        "model": "large-v3",
        "device": device
    }

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    start = time.time()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        result = model.transcribe(
            tmp_path,
            task="transcribe",
            best_of=5,
            beam_size=5,
            temperature=0.0,
            condition_on_previous_text=True
        )
        return {
            "success": True,
            "text": result["text"].strip(),
            "language_detected": result["language"],
            "time_taken": round(time.time() - start, 2)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        os.unlink(tmp_path)

@app.post("/transcribe-and-translate")
async def transcribe_and_translate(
    file: UploadFile = File(...),
    target_language: str = "English"
):
    result = await transcribe(file)
    if not result["success"]:
        return result
    if target_language != "English":
        translated = GoogleTranslator(
            source="auto",
            target=LANG_CODES.get(target_language, "en")
        ).translate(result["text"])
        result["translated_text"] = translated
        result["target_language"] = target_language
    return result
