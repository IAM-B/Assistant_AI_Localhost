# Assistant_AI_Localhost

A local personal voice assistant in Python, built with open source technologies:

-Voice transcription using faster-Whisper

-Conversational intelligence via Ollama (Mistral)

-Multilingual speech synthesis with Coqui TTS

-Direct interaction through microphone (push-to-talk)

Everything runs locally, with no cloud dependency, and is optimized for Linux PCs with NVIDIA GPUs starting from 6GB of VRAM.

---

## 📦 Fonctionnalités principales

- 🎤 Local speech recognition (Faster-Whisper)
- 🤖 Conversational AI (Ollama + Mistral)
- 🔊 Fluid voice synthesis and multi-lingual (Coqui TTS)
- 🧠 Automatic language detection (FR/EN/PT)
- 🖐️ Push-to-talk control via F12 key
- 📈 Performance logs: STT / LLM / TTS

---

## ⚙️ Installation

### 1. Clone this ripo
```bash
git clone https://github.com/IAM-B/Assistant_AI_Localhost
cd Assistant_AI_Localhost
```

### 2. Creating a Python Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Installing Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Launch

Make sure Ollama is running with a model like `neural-chat`:

```bash
ollama run neural-chat
```

Then launch the script:

```bash
python assistant.py
```

Hold down `F12` to speak, and release it to let the AI respond.

---

## 📂 Key Files

* `assistant.py`: main script
* `performance_log.txt`: logs for each processing step
* `.gitignore`: excludes the `venv/` folder and temporary files

---

## 🛠️ To Do / Suggestions

* Visual status indicator (listening / thinking / speaking)
* Graphical interface (PyQt, Gradio...)


