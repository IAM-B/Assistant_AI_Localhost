from subprocess import Popen


class SharedState:
    def __init__(self):
        self.tts_process: Popen | None = None
        self.interrupt_tts: bool = False
        self.recording: bool = False
        self.audio_buffer = []
        
shared_state = SharedState()