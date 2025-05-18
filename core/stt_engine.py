import tempfile, time, torch
import numpy as np
import scipy.io.wavfile as wavfile
from faster_whisper import WhisperModel
from config import WHISPER_MODEL, SAMPLING_RATE
from utils.io_utils import log
from core.llm import queue_response

stt_model = WhisperModel(WHISPER_MODEL, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")

def transcriber(queue_transcription, queue_response):
    while True:
        audio = queue_transcription.get()
        start = time.time()
        temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wavfile.write(temp.name, SAMPLING_RATE, (audio * 32767).astype(np.int16))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            segments, _ = stt_model.transcribe(temp.name, language="fr")
            text = " ".join([segment.text for segment in segments])
            print("[📝]", text)
            log("STT", time.time() - start)
            queue_response.put(text)
        except Exception as e:
            print(f"[❌] Erreur STT : {e}")
