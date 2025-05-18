import re
import os
import sys
import time
import torch
import queue
import termios
import atexit
import threading
import sounddevice as sd
from langdetect import detect
import numpy as np
import tempfile
import subprocess
import requests
import json
import scipy.io.wavfile as wavfile
from faster_whisper import WhisperModel
from TTS.api import TTS
from pynput import keyboard
from datetime import datetime

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"
TTS_MODEL = "tts_models/multilingual/multi-dataset/your_tts"
WHISPER_MODEL = "turbo"
SAMPLING_RATE = 16000
AUDIO_FILE = "response.wav"
LOG_FILE = "performance_log.txt"
tts_process = None
interrupt_tts = False

# Initialisation des métriques
boot_time = time.time()

# Chargement des modèles
stt_start = time.time()
stt_model = WhisperModel(WHISPER_MODEL, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")
stt_load_time = time.time() - stt_start

tts_start = time.time()
tts_model = TTS(model_name=TTS_MODEL, gpu=torch.cuda.is_available())
available_speakers = [s.strip() for s in tts_model.speakers]
tts_load_time = time.time() - tts_start

# Variables
recording = False
audio_buffer = []
queue_transcription = queue.Queue()
queue_response = queue.Queue()
original_settings = termios.tcgetattr(sys.stdin.fileno())

# Clavier
def disable_echo():
    new_settings = termios.tcgetattr(sys.stdin.fileno())
    new_settings[3] = new_settings[3] & ~termios.ECHO
    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, new_settings)

def restore_echo():
    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, original_settings)
    print("[✔] Clavier restauré.")

# Logs
def log(label, duration):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {label}: {duration:.2f} seconds\n")

# Audio
def audio_callback(indata, frames, time_info, status):
    if recording:
        audio_buffer.append(indata.copy())

def start_audio_stream():
    return sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLING_RATE)

# Transcription
def transcriber():
    while True:
        audio = queue_transcription.get()
        start = time.time()
        temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wavfile.write(temp.name, SAMPLING_RATE, (audio * 32767).astype(np.int16))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        segments, _ = stt_model.transcribe(temp.name, language="fr")
        text = " ".join([segment.text for segment in segments])
        print("[📝]", text)
        log("STT", time.time() - start)
        queue_response.put(text)

# LLM + TTS (streaming optimisé)
def ollama_asker():
    sentence_end = re.compile(r"[.!?]\s")
    while True:
        prompt = queue_response.get()
        buffer = ""
        print("[🤖] Envoi au LLM...")
        start = time.time()
        try:
            response = requests.post(OLLAMA_URL, json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": True
            }, stream=True, timeout=60)

            if response.ok:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode("utf-8").replace("data: ", ""))
                            chunk = data["response"]
                            print(chunk, end="", flush=True)
                            buffer += chunk
                            sentences = sentence_end.split(buffer)
                            for s in sentences[:-1]:
                                speak(s.strip())
                            buffer = sentences[-1]
                        except Exception as e:
                            print("[⚠️] Erreur de chunk:", e)
                            continue
                # lecture finale si reste du texte
                if buffer.strip():
                    speak(buffer.strip())
                log("LLM+TTS", time.time() - start)
                print()
            else:
                print(f"[❌] Erreur LLM : {response.status_code}")
        except Exception as e:
            print(f"[❌] Erreur connexion LLM : {e}")
            log("LLM_ERROR", 0)

# TTS
def detect_lang(text):
    code = detect(text)
    if code == "fr":
        return "fr-fr"
    elif code == "en":
        return "en"
    elif code == "pt":
        return "pt-br"
    else:
        return "en"

def get_speaker(lang):
    print(f"[🔍] Speakers disponibles : {available_speakers}")
    mapping = {
        "fr-fr": "female-en-5",
        "en": "male-en-2",
        "pt-br": "male-pt-3"
    }
    speaker = mapping.get(lang, "male-en-2")
    if speaker not in available_speakers:
        print(f"[⚠] Speaker '{speaker}' indisponible. Speakers dispos : {available_speakers}")
        return available_speakers[0]
    return speaker

def clean_text_for_tts(text):
    text = re.sub(r"[-•:*]", "", text)
    text = re.sub(r"[!?:]", ".", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def speak(text):
    global tts_process, interrupt_tts
    if interrupt_tts:
        print("[🛑] TTS annulé avant génération.")
        return
    try:
        lang = detect_lang(text)
        speaker = get_speaker(lang)
        print(f"[🔊] Langue détectée : {lang} | Speaker : {speaker}")
        text = clean_text_for_tts(text)
        tts_model.tts_to_file(
            text=text,
            language=lang,
            speaker=speaker,
            file_path="response.wav"
        )
        if interrupt_tts:
            print("[🛑] TTS annulé après génération.")
            return
        tts_process = subprocess.Popen(["aplay", "response.wav"])
        while True:
            if interrupt_tts:
                print("[🛑] TTS interrompu pendant lecture.")
                if tts_process and tts_process.poll() is None:
                    tts_process.terminate()
                tts_process = None
                return
            if tts_process and tts_process.poll() is not None:
                break
            time.sleep(0.05)
        tts_process = None
    except Exception as e:
        print(f"[❌] Erreur TTS : {e}")

# Clavier
def on_press(key):
    global recording, audio_buffer, tts_process, interrupt_tts
    if key == keyboard.Key.f12:
        if tts_process and tts_process.poll() is None:
            print("[🛑] TTS interrompu par F12.")
            tts_process.terminate()
            interrupt_tts = True
            tts_process = None
        if not recording:
            print("[🎙️] Enregistrement démarré (F12)")
            recording = True
            audio_buffer = []

def on_release(key):
    global recording, interrupt_tts
    if key == keyboard.Key.f12 and recording:
        print("[🛑] Enregistrement arrêté")
        recording = False
        interrupt_tts = False
        if audio_buffer:
            audio = np.concatenate(audio_buffer, axis=0)
            queue_transcription.put(audio)

# Initialisation
disable_echo()
atexit.register(restore_echo)

# Démarrage métriques
boot_total = time.time() - boot_time
print(f"[⏱] Chargement STT : {stt_load_time:.2f}s | TTS : {tts_load_time:.2f}s | Total : {boot_total:.2f}s")
log("BOOT_TOTAL", boot_total)

# Threads
threading.Thread(target=transcriber, daemon=True).start()
threading.Thread(target=ollama_asker, daemon=True).start()

print("🟢 Maintiens F12 pour parler, relâche pour envoyer...")

stream = start_audio_stream()
stream.start()

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n[✋] Arrêt par Ctrl+C détecté.")
    stream.stop()
    stream.close()
    listener.stop()
    restore_echo()
