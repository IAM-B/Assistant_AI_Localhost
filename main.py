import time, threading, termios, sys, atexit
import queue
from pynput import keyboard
from config import SAMPLING_RATE
from core import audio, keyboard as kb, stt_engine, llm
from utils.io_utils import log
from core.shared import shared_state as state

# État global du clavier (echo désactivé)
original_settings = termios.tcgetattr(sys.stdin.fileno())

# Désactivation temporaire de l’écho clavier (mode console propre)
def disable_echo():
    new_settings = termios.tcgetattr(sys.stdin.fileno())
    new_settings[3] = new_settings[3] & ~termios.ECHO
    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, new_settings)

def restore_echo():
    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, original_settings)
    print("[✔] Clavier restauré.")

# Initialisation
disable_echo()
atexit.register(restore_echo)

# File de communication entre les modules
queue_transcription = queue.Queue()
queue_response = queue.Queue()

print("🟢 Maintiens F12 pour parler, relâche pour envoyer...")

# Démarrage du stream audio
stream = audio.start_audio_stream(SAMPLING_RATE)
stream.start()

# Lancement des threads de transcription et de génération de réponse
threading.Thread(target=stt_engine.transcriber, args=(queue_transcription, queue_response), daemon=True).start()
threading.Thread(target=llm.ollama_asker, args=(queue_response,), daemon=True).start()

# Capture clavier
listener = keyboard.Listener(
    on_press=lambda k: kb.on_press(k, state),
    on_release=lambda k: kb.on_release(k, state, queue_transcription)
)
listener.start()

# Boucle principale
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n[✋] Arrêt par Ctrl+C détecté.")
    stream.stop()
    stream.close()
    listener.stop()
    restore_echo()
