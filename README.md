## VietnameseTTS Gwen-TTS GUI (Local / Portable)

This project provides a **local Vietnamese TTS** app for the **Gwen-TTS** voice-cloning model, with a **web GUI (Gradio)** as the recommended interface and an optional desktop GUI fallback.

Upstream references:
- Hugging Face model: [`g-group-ai-lab/gwen-tts-0.6B`](https://huggingface.co/g-group-ai-lab/gwen-tts-0.6B)
- Upstream repo / inference reference: [`ggroup-ai-lab/gwen-tts`](https://github.com/ggroup-ai-lab/gwen-tts)

### What this app does

- **Input**: type text in a textbox or load a `.txt` file
- **Voice**:
  - built-in speakers (downloaded locally from upstream `ref_info.json` + reference WAVs), or
  - custom reference WAV + transcript (voice cloning)
- **Output**: writes a `.wav` file
- **Device**: uses **CUDA** if available at runtime (`torch.cuda.is_available()`), otherwise CPU
- **Portable**: downloads **model snapshot + speaker assets into this folder** so you can copy the whole directory to another PC and run (no re-download)

### One-click install (Windows, recommended: uv)

1) Install `uv` (one-time)
- See: `https://astral.sh/uv`

2) In this folder, double-click:
- `install_uv_windows.bat`

3) Then double-click (recommended):
- `run_web.bat`

The browser opens at **http://127.0.0.1:7860**. Keep the terminal window open while using the app; close it to stop the server.

**Optional desktop fallback:** double-click `run_gui.bat` for the Tkinter desktop GUI.

### Notes about “don’t re-download”

#### Model files
On first bootstrap, the app downloads the model snapshot to:
- `models/g-group-ai-lab__gwen-tts-0.6B/`

To avoid slow downloads on the *same PC*, bootstrap will **reuse your existing Hugging Face cache** (if present) as a source, and then copies the snapshot into the local `models/` folder for portability.

#### Local cache folder
`run_web.bat` and `run_gui.bat` set these env vars so Hugging Face caches stay inside this project:
- `HF_HOME=.hf_cache`
- `HF_HUB_CACHE=.hf_cache\hub`
- `TRANSFORMERS_CACHE=.hf_cache\transformers`

This keeps runs reproducible and copy-paste friendly.

### Run without uv (pip fallback)

Create a venv and install deps:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Then run the web GUI (recommended):

```bash
.venv\Scripts\python -m app.webapp
```

Open **http://127.0.0.1:7860** in your browser.

Or run the desktop GUI:

```bash
.venv\Scripts\python -m app.gui
```

> For best performance, install a CUDA-enabled PyTorch build if your GPU/driver supports it.

### Author

phamtronglam2001@gmail.com
