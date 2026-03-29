# Speech-to-Speech Translation (S2ST)

Translate speech from one language to another: record or upload audio, get translated speech. Uses a web UI or a command-line tool.

**You need:** Python 3.10 or newer, internet (first run downloads models; TTS uses the network).

---

## How to run (step by step)

### 1. Open the project folder

In PowerShell (or Command Prompt), go to where this project lives, for example:

```powershell
cd C:\Users\YourName\OneDrive\Desktop\project
```

Use your real path if it differs.

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the virtual environment

**PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Command Prompt (cmd):**

```cmd
.venv\Scripts\activate.bat
```

You should see `(.venv)` at the start of the line.

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

Wait until it finishes without errors.

### 5. Start the app (pick one)

#### Option A — Web UI (easiest)

```powershell
python ui_app.py
```

1. Open your browser at **http://127.0.0.1:7860**
2. Choose the **target language**
3. **Record** with the mic or **upload** a WAV/FLAC file
4. Click **Translate speech**
5. Wait for transcription, translation, and the audio player

**Other useful commands:**

```powershell
python ui_app.py --host 0.0.0.0 --port 7860
```

Use this if you need access from another machine on your network (binds to all interfaces).

```powershell
python ui_app.py --share
```

Creates a temporary public link (needs internet).

---

#### Option B — Command line (live microphone)

```powershell
python main.py --target-lang fr
```

Replace `fr` with your desired **target** language code (`en`, `es`, `de`, etc.). Speak into the mic; pause about half a second after each phrase. Press **Ctrl+C** to stop.

---

#### Option C — Command line (audio file)

```powershell
python main.py --wav path\to\your\audio.wav --target-lang en
```

Use your file path and target language.

**Common flags for `main.py`:**

| Flag | Example | Meaning |
|------|---------|--------|
| `--target-lang` | `en` | Language to translate **into** |
| `--source-lang` | `es` | Force source language (optional; default is auto-detect) |
| `--whisper-model` | `small` | Model size: `tiny`, `base`, `small`, `medium` |
| `--device` | `cpu` or `cuda` | Use GPU only if CUDA is installed |
| `--wav` | `clip.wav` | Process a file instead of the microphone |

---

## If something fails

- Run all commands **inside the activated** `.venv` (you should see `(.venv)`).
- First launch can take a long time while models download.
- The web UI and TTS need **network** access for `edge-tts`.
