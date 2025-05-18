import re
from datetime import datetime
from langdetect import detect
from config import LOG_FILE

def detect_lang(text):
    code = detect(text)
    return {
        "fr": "fr-fr",
        "en": "en",
        "pt": "pt-br"
    }.get(code, "en")

def clean_text_for_tts(text):
    text = re.sub(r"[-•:*]", "", text)
    text = re.sub(r"[!?:]", ".", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def log(label, duration):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {label}: {duration:.2f} seconds\n")
