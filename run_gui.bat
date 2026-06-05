@echo off
setlocal

REM Make the project portable: keep HF cache in this folder by default.
set "ROOT=%~dp0"
set "HF_HOME=%ROOT%.hf_cache"
set "TRANSFORMERS_CACHE=%HF_HOME%\transformers"
set "HF_HUB_CACHE=%HF_HOME%\hub"

REM If uv created a local venv, prefer it.
if exist "%ROOT%.venv\Scripts\python.exe" (
  "%ROOT%.venv\Scripts\python.exe" -m app.gui
  exit /b %errorlevel%
)

python -m app.gui
exit /b %errorlevel%

