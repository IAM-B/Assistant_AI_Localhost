import sounddevice as sd
import numpy as np
from config import SAMPLING_RATE
from core.shared import shared_state as state

def audio_callback(indata, frames, time_info, status):
    if state.recording:
        state.audio_buffer.append(indata.copy())

def start_audio_stream(samplerate):
    return sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLING_RATE)

