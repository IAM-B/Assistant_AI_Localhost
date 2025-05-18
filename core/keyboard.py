from pynput import keyboard
import numpy as np
from core.shared import shared_state as state

def on_press(key, state):
    if key == keyboard.Key.f12:
        if state.tts_process and state.tts_process.poll() is None:
            print("[🛑] TTS interrompu par F12.")
            state.tts_process.terminate()
            state.interrupt_tts = True
            state.tts_process = None
        if not state.recording:
            print("[🎙️] Enregistrement démarré (F12)")
            state.recording = True
            state.audio_buffer = []

def on_release(key, state, queue_transcription):
    if key == keyboard.Key.f12 and state.recording:
        print("[🛑] Enregistrement arrêté")
        state.recording = False
        state.interrupt_tts = False
        if state.audio_buffer:
            audio = np.concatenate(state.audio_buffer, axis=0)
            queue_transcription.put(audio)
