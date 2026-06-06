"""
Post-warmup session state: voice-clone prompt tensors + tuned generation params.

The qwen_tts library does **not** persist voice-clone embeddings on the model object
between ``generate_voice_clone`` calls. The meaningful runtime artifact we can
serialize is the ``VoiceClonePromptItem`` list (ref_code + ref_spk_embedding)
computed after at least one generation, plus the generation parameters that
produced acceptable audio.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from .engine import GenerationParams
from .preset import RuntimeSnapshot
from .voice_prompt_cache import (
    VOICE_PROMPT_SIDECAR_SUFFIX,
    load_voice_prompt,
    resolve_voice_prompt_path,
    save_voice_prompt,
)

SESSION_SCHEMA_VERSION = 1
REF_AUDIO_SIDECAR_SUFFIX = ".ref.wav"
VoiceMode = Literal["speaker", "custom"]


def make_voice_cache_key(
    *,
    voice_mode: VoiceMode,
    speaker_key: Optional[str] = None,
    ref_audio: Optional[Path] = None,
    ref_text: Optional[str] = None,
    x_vector_only_mode: bool = False,
) -> str:
    """Stable id for the voice source + x_vector_only_mode combination."""
    xvec = "1" if x_vector_only_mode else "0"
    if voice_mode == "speaker":
        key = speaker_key or ""
        return f"speaker:{key}:xvec={xvec}"
    if ref_audio is None:
        raise ValueError("ref_audio is required for custom voice cache key.")
    digest = hashlib.sha256()
    digest.update(ref_audio.read_bytes())
    digest.update((ref_text or "").encode("utf-8"))
    return f"custom:{digest.hexdigest()[:16]}:xvec={xvec}"


@dataclass
class SessionRuntimeSnapshot:
    """
    Lightweight snapshot of post-warmup runtime state (no input text).

    Saved as ``<name>.json`` with an optional ``<name>.prompt.pt`` sidecar
    holding precomputed voice-clone tensors.
    """

    voice_mode: VoiceMode = "speaker"
    speaker_key: Optional[str] = "yen_nhi"
    ref_audio_file: Optional[str] = None
    ref_text: Optional[str] = None
    generation: GenerationParams = field(default_factory=GenerationParams)
    generation_count: int = 0
    voice_cache_key: Optional[str] = None
    voice_prompt_file: Optional[str] = None
    runtime: Optional[RuntimeSnapshot] = None
    name: Optional[str] = None
    created_at: Optional[str] = None
    schema_version: int = SESSION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["generation"] = self.generation.to_dict()
        if self.runtime is not None:
            data["runtime"] = asdict(self.runtime)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionRuntimeSnapshot:
        version = int(data.get("schema_version", 0))
        if version != SESSION_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported session schema version {version} "
                f"(expected {SESSION_SCHEMA_VERSION})."
            )
        runtime_raw = data.get("runtime")
        runtime = RuntimeSnapshot(**runtime_raw) if isinstance(runtime_raw, dict) else None
        gen_raw = data.get("generation") or {}
        return cls(
            voice_mode=data.get("voice_mode", "speaker"),
            speaker_key=data.get("speaker_key"),
            ref_audio_file=data.get("ref_audio_file"),
            ref_text=data.get("ref_text"),
            generation=GenerationParams.from_dict(gen_raw),
            generation_count=int(data.get("generation_count", 0)),
            voice_cache_key=data.get("voice_cache_key"),
            voice_prompt_file=data.get("voice_prompt_file"),
            runtime=runtime,
            name=data.get("name"),
            created_at=data.get("created_at"),
            schema_version=version,
        )


def _ref_audio_sidecar_name(session_path: Path) -> str:
    return session_path.with_suffix(REF_AUDIO_SIDECAR_SUFFIX).name


def _voice_prompt_sidecar_name(session_path: Path) -> str:
    return session_path.with_suffix(VOICE_PROMPT_SIDECAR_SUFFIX).name


def capture_session(
    *,
    voice_mode: VoiceMode,
    speaker_key: Optional[str],
    ref_audio: Optional[Path],
    ref_text: Optional[str],
    generation: GenerationParams,
    generation_count: int,
    voice_clone_prompt: Optional[list],
    device: Optional[str] = None,
    model_dir: Optional[Path] = None,
    name: Optional[str] = None,
) -> SessionRuntimeSnapshot:
    voice_cache_key = make_voice_cache_key(
        voice_mode=voice_mode,
        speaker_key=speaker_key,
        ref_audio=ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=generation.x_vector_only_mode,
    )
    runtime = RuntimeSnapshot.capture(device=device, model_dir=model_dir)
    return SessionRuntimeSnapshot(
        name=name,
        voice_mode=voice_mode,
        speaker_key=speaker_key if voice_mode == "speaker" else None,
        ref_text=ref_text if voice_mode == "custom" else None,
        generation=generation,
        generation_count=generation_count,
        voice_cache_key=voice_cache_key,
        runtime=runtime,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def save_session_snapshot(
    session: SessionRuntimeSnapshot,
    session_path: Path,
    *,
    ref_audio_src: Optional[Path] = None,
    voice_prompt_items: Optional[list] = None,
) -> Path:
    session_path = session_path.expanduser().resolve()
    session_path.parent.mkdir(parents=True, exist_ok=True)

    if session.voice_mode == "custom" and ref_audio_src is not None:
        sidecar = session_path.with_suffix(REF_AUDIO_SIDECAR_SUFFIX)
        shutil.copy2(ref_audio_src, sidecar)
        session.ref_audio_file = _ref_audio_sidecar_name(session_path)

    if voice_prompt_items:
        prompt_path = session_path.with_suffix(VOICE_PROMPT_SIDECAR_SUFFIX)
        save_voice_prompt(
            voice_prompt_items,
            prompt_path,
            voice_cache_key=session.voice_cache_key,
        )
        session.voice_prompt_file = _voice_prompt_sidecar_name(session_path)

    if not session.name:
        session.name = session_path.stem

    session_path.write_text(
        json.dumps(session.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return session_path


def load_session_snapshot(session_path: Path) -> tuple[SessionRuntimeSnapshot, Optional[list]]:
    session_path = session_path.expanduser().resolve()
    data = json.loads(session_path.read_text(encoding="utf-8"))
    session = SessionRuntimeSnapshot.from_dict(data)
    prompt_path = resolve_voice_prompt_path(session_path, session.voice_prompt_file)
    prompt_items = load_voice_prompt(prompt_path) if prompt_path is not None else None
    return session, prompt_items


def resolve_session_ref_audio(session: SessionRuntimeSnapshot, session_path: Path) -> Optional[Path]:
    if session.voice_mode != "custom" or not session.ref_audio_file:
        return None
    candidate = session_path.parent / session.ref_audio_file
    if candidate.exists():
        return candidate
    alt = session_path.with_suffix(REF_AUDIO_SIDECAR_SUFFIX)
    if alt.exists():
        return alt
    return candidate
