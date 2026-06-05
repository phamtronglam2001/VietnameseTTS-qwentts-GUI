from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


HF_MODEL_ID = "g-group-ai-lab/gwen-tts-0.6B"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/ggroup-ai-lab/gwen-tts/main"
REF_INFO_URL = f"{GITHUB_RAW_BASE}/data/ref_info.json"


@dataclass(frozen=True)
class BootstrapResult:
    model_dir: Path
    ref_info_path: Path
    ref_audio_dir: Path


def _default_global_hf_home() -> Path:
    # Hugging Face default cache root differs by OS.
    # We try to be conservative and only use this as a *source* cache to reuse downloads.
    if sys.platform.startswith("win"):
        user = Path(os.environ.get("USERPROFILE", str(Path.home())))
        return user / ".cache" / "huggingface"
    return Path.home() / ".cache" / "huggingface"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _download_text(url: str, dest: Path) -> None:
    import requests

    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)


def _download_binary(url: str, dest: Path) -> None:
    import requests

    with requests.get(url, timeout=120, stream=True) as r:
        r.raise_for_status()
        _ensure_dir(dest.parent)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _iter_ref_audio_paths(ref_info: dict) -> Iterable[str]:
    for _k, v in ref_info.items():
        ap = v.get("audio_path")
        if isinstance(ap, str) and ap:
            yield ap


def ensure_speaker_assets(ref_info_path: Path, ref_audio_dir: Path) -> None:
    """
    Downloads `ref_info.json` and all referenced WAVs from the upstream GitHub repo.
    Everything is saved locally under `assets/` so the folder can be copied offline.
    """
    _ensure_dir(ref_info_path.parent)
    _ensure_dir(ref_audio_dir)

    if not ref_info_path.exists():
        _download_text(REF_INFO_URL, ref_info_path)

    ref_info = json.loads(ref_info_path.read_text(encoding="utf-8"))
    for rel_path in _iter_ref_audio_paths(ref_info):
        # Upstream paths look like: "data/ref_audio/yen_nhi.wav"
        filename = Path(rel_path).name
        local = ref_audio_dir / filename
        if local.exists():
            continue
        url = f"{GITHUB_RAW_BASE}/{rel_path}"
        _download_binary(url, local)


def ensure_model_snapshot(
    *,
    model_id: str,
    local_dir: Path,
    local_hf_home: Path,
    prefer_reuse_from_global_cache: bool = True,
    global_hf_home: Optional[Path] = None,
) -> Path:
    """
    Ensures the HF model snapshot exists locally in `local_dir`.

    Key goals:
    - **Portable**: the snapshot lives inside the project folder.
    - **Fast first install**: if a global HF cache already exists on this PC,
      reuse it to avoid re-downloading.
    """
    from huggingface_hub import snapshot_download

    _ensure_dir(local_dir.parent)
    _ensure_dir(local_hf_home)

    if local_dir.exists() and any(local_dir.iterdir()):
        return local_dir

    # Use existing HF cache as a source to avoid network downloads when possible.
    cache_dir: Path = local_hf_home
    if prefer_reuse_from_global_cache:
        gh = global_hf_home or _default_global_hf_home()
        if gh.exists():
            cache_dir = gh

    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,  # critical for portability on Windows
        cache_dir=str(cache_dir),
        resume_download=True,
    )

    # Optional: if we reused the global cache, copy cached files into local HF_HOME too,
    # so the project folder stays self-contained for future offline runs.
    if cache_dir != local_hf_home:
        try:
            # Best-effort: this can be large, but copying the whole cache is not necessary.
            # Instead, copy only the downloaded snapshot directory (which is already in local_dir).
            # We keep local_hf_home for future use by other downloads.
            _ensure_dir(local_hf_home)
        except Exception:
            pass

    return local_dir


def bootstrap_all(root: Path) -> BootstrapResult:
    from .paths import AppPaths

    paths = AppPaths(root=root)
    _ensure_dir(paths.models_dir)
    _ensure_dir(paths.assets_dir)

    ensure_speaker_assets(paths.ref_info_json, paths.ref_audio_dir)
    ensure_model_snapshot(
        model_id=HF_MODEL_ID,
        local_dir=paths.model_local_dir,
        local_hf_home=paths.cache_dir,
    )
    return BootstrapResult(
        model_dir=paths.model_local_dir,
        ref_info_path=paths.ref_info_json,
        ref_audio_dir=paths.ref_audio_dir,
    )

