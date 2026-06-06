@echo off
setlocal

REM Portable Hugging Face cache inside this project folder.
set "ROOT=%~dp0"
set "HF_HOME=%ROOT%.hf_cache"
set "HF_HUB_CACHE=%HF_HOME%\hub"

echo Starting Gwen-TTS Web GUI...
echo Browser will open at http://127.0.0.1:7860
echo Close this window to stop the server.

if exist "%ROOT%.venv\Scripts\python.exe" (
  "%ROOT%.venv\Scripts\python.exe" -m app.webapp
  exit /b %errorlevel%
)

python -m app.webapp
exit /b %errorlevel%
