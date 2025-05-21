import time, subprocess, re
from TTS.api import TTS
from config import TTS_MODEL, AUDIO_FILE
from utils.io_utils import detect_lang, clean_text_for_tts
from core.shared import shared_state as state

tts_model = TTS(model_name=TTS_MODEL, gpu=True)
available_speakers = [s.strip() for s in tts_model.speakers]

def get_speaker(lang):
    print(f"[🔍] Speakers disponibles : {available_speakers}")
    mapping = {
        "fr-fr": "male-en-2",
        "en": "male-en-2",
        "pt-br": "male-pt-3"
    }
    speaker = mapping.get(lang, "male-en-2")
    if speaker not in available_speakers:
        print(f"[⚠] Speaker '{speaker}' indisponible. Speakers dispos : {available_speakers}")
        return available_speakers[0]
    return speaker

def speak(text):
    if state.interrupt_tts:
        print("[🛑] TTS annulé avant génération.")
        return
    try:
        lang = detect_lang(text)
        speaker = get_speaker(lang)
        print(f"[🔊] Langue détectée : {lang} | Speaker : {speaker}")
        text = clean_text_for_tts(text)
        tts_model.tts_to_file(text=text, language=lang, speaker=speaker, file_path=AUDIO_FILE)
        if state.interrupt_tts:
            print("[🛑] TTS annulé après génération.")
            return
        state.tts_process = subprocess.Popen(["aplay", AUDIO_FILE])
        while True:
            if state.interrupt_tts:
                print("[🛑] TTS interrompu pendant lecture.")
                if state.tts_process and state.tts_process.poll() is None:
                    state.tts_process.terminate()
                state.tts_process = None
                return
            if state.tts_process and state.tts_process.poll() is not None:
                break
            time.sleep(0.05)
        state.tts_process = None
    except Exception as e:
        print(f"[❌] Erreur TTS : {e}")
