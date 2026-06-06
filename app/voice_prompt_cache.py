from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import torch

VOICE_PROMPT_SIDECAR_SUFFIX = ".prompt.pt"
VOICE_PROMPT_SCHEMA_VERSION = 1


def _import_prompt_item_class():
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

    return VoiceClonePromptItem


def sidecar_name(preset_path: Path) -> str:
    return preset_path.with_suffix(VOICE_PROMPT_SIDECAR_SUFFIX).name


def prompt_items_to_payload(items: List[Any], *, voice_cache_key: Optional[str] = None) -> dict:
    payload = {
        "schema_version": VOICE_PROMPT_SCHEMA_VERSION,
        "items": [
            {
                "ref_code": it.ref_code,
                "ref_spk_embedding": it.ref_spk_embedding,
                "x_vector_only_mode": bool(it.x_vector_only_mode),
                "icl_mode": bool(it.icl_mode),
                "ref_text": it.ref_text,
            }
            for it in items
        ],
    }
    if voice_cache_key:
        payload["voice_cache_key"] = voice_cache_key
    return payload


def payload_to_prompt_items(payload: dict) -> List[Any]:
    VoiceClonePromptItem = _import_prompt_item_class()
    version = int(payload.get("schema_version", 0))
    if version != VOICE_PROMPT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported voice prompt schema version {version} "
            f"(expected {VOICE_PROMPT_SCHEMA_VERSION})."
        )
    items = []
    for raw in payload.get("items") or []:
        items.append(
            VoiceClonePromptItem(
                ref_code=raw.get("ref_code"),
                ref_spk_embedding=raw["ref_spk_embedding"],
                x_vector_only_mode=bool(raw.get("x_vector_only_mode", False)),
                icl_mode=bool(raw.get("icl_mode", True)),
                ref_text=raw.get("ref_text"),
            )
        )
    return items


def save_voice_prompt(
    items: List[Any],
    path: Path,
    *,
    voice_cache_key: Optional[str] = None,
) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prompt_items_to_payload(items, voice_cache_key=voice_cache_key), path)
    return path


def load_voice_prompt(path: Path) -> tuple[List[Any], Optional[str]]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Voice prompt cache not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    key = payload.get("voice_cache_key") if isinstance(payload, dict) else None
    return payload_to_prompt_items(payload), key


def voice_cache_key_from_payload(path: Path) -> Optional[str]:
    path = path.expanduser().resolve()
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        return payload.get("voice_cache_key")
    return None


def resolve_voice_prompt_path(preset_path: Path, voice_prompt_file: Optional[str]) -> Optional[Path]:
    if voice_prompt_file:
        candidate = preset_path.parent / voice_prompt_file
        if candidate.exists():
            return candidate
    sidecar = preset_path.with_suffix(VOICE_PROMPT_SIDECAR_SUFFIX)
    if sidecar.exists():
        return sidecar
    return None
