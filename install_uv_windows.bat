@echo off
setlocal enabledelayedexpansion

set "ROOT=%~dp0"
cd /d "%ROOT%"

where uv >nul 2>nul
if errorlevel 1 (
  echo uv not found. Please install uv first:
  echo   https://astral.sh/uv
  exit /b 1
)

echo Creating venv and installing deps...
uv venv --python 3.11

REM Try CUDA builds first, then fall back to CPU wheels.
echo Installing torch (prefer CUDA)...
uv pip install --upgrade pip setuptools wheel
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
  echo CUDA torch install failed; falling back to CPU wheels...
  uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  if errorlevel 1 exit /b 1
)

echo Installing app dependencies...
uv pip install huggingface_hub transformers safetensors qwen-tts soundfile numpy requests gradio
if errorlevel 1 exit /b 1

echo Done.
echo Next: double-click run_web.bat  (recommended)
echo    or: double-click run_gui.bat (desktop GUI)

