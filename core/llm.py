import queue
import re, time, json, requests
from config import OLLAMA_URL, MODEL_NAME
from utils.io_utils import log
from core.tts_engine import speak

queue_response = queue.Queue()

def ollama_asker(queue_response):
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
                if buffer.strip():
                    speak(buffer.strip())
                log("LLM+TTS", time.time() - start)
                print()
            else:
                print(f"[❌] Erreur LLM : {response.status_code}")
        except Exception as e:
            print(f"[❌] Erreur connexion LLM : {e}")
            log("LLM_ERROR", 0)
