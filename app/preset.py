from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from .bootstrap import HF_MODEL_ID
from .engine import GenerationParams, pick_attn_implementation, pick_device, pick_dtype


PRESET_SCHEMA_VERSION = 1
REF_AUDIO_SIDECAR_SUFFIX = ".ref.wav"
VOICE_PROMPT_SIDECAR_SUFFIX = ".prompt.pt"
VoiceMode = Literal["speaker", "custom"]


@dataclass
class RuntimeSnapshot:
    """
    Environment metadata recorded alongside a preset/session.

    This does **not** capture mutable model weights or CUDA warmup state.
    The serializable post-warmup artifact is the voice-clone prompt sidecar
    (``.prompt.pt``): precomputed ``ref_code`` and ``ref_spk_embedding`` tensors.
    """

    model_id: str = HF_MODEL_ID
    device: str = field(default_factory=pick_device)
    dtype: str = "torch.bfloat16"
    attn_implementation: str = field(default_factory=pick_attn_implementation)
    model_dir: Optional[str] = None

    @classmethod
    def capture(cls, *, device: Optional[str] = None, model_dir: Optional[Path] = None) -> RuntimeSnapshot:
        dev = device or pick_device()
        dtype = pick_dtype(dev)
        return cls(
            model_id=HF_MODEL_ID,
            device=dev,
            dtype=str(dtype).replace("torch.", ""),
            attn_implementation=pick_attn_implementation(),
            model_dir=str(model_dir) if model_dir else None,
        )


@dataclass
class GenerationPreset:
    """
    Complete session snapshot for reproducing a TTS generation run.

    Custom reference audio is stored as a sidecar WAV next to the JSON file
    (``<preset_stem>.ref.wav``) when saved via :func:`save_preset`.
    """

    text: str
    voice_mode: VoiceMode = "speaker"
    speaker_key: Optional[str] = "yen_nhi"
    ref_audio_file: Optional[str] = None
    ref_text: Optional[str] = None
    generation: GenerationParams = field(default_factory=GenerationParams)
    x_vector_only_mode: bool = False
    output_wav: Optional[str] = None
    voice_prompt_file: Optional[str] = None
    voice_cache_key: Optional[str] = None
    generation_count: int = 0
    name: Optional[str] = None
    created_at: Optional[str] = None
    runtime: Optional[RuntimeSnapshot] = None
    schema_version: int = PRESET_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["generation"] = self.generation.to_dict()
        if self.runtime is not None:
            data["runtime"] = asdict(self.runtime)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerationPreset:
        version = int(data.get("schema_version", 0))
        if version != PRESET_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported preset schema version {version} (expected {PRESET_SCHEMA_VERSION})."
            )

        gen_raw = data.get("generation") or {}
        runtime_raw = data.get("runtime")
        runtime = RuntimeSnapshot(**runtime_raw) if isinstance(runtime_raw, dict) else None

        return cls(
            text=str(data.get("text", "")),
            voice_mode=data.get("voice_mode", "speaker"),
            speaker_key=data.get("speaker_key"),
            ref_audio_file=data.get("ref_audio_file"),
            ref_text=data.get("ref_text"),
            generation=GenerationParams.from_dict(gen_raw),
            x_vector_only_mode=bool(data.get("x_vector_only_mode", False)),
            output_wav=data.get("output_wav"),
            voice_prompt_file=data.get("voice_prompt_file"),
            voice_cache_key=data.get("voice_cache_key"),
            generation_count=int(data.get("generation_count", 0)),
            name=data.get("name"),
            created_at=data.get("created_at"),
            runtime=runtime,
            schema_version=version,
        )

    def resolve_ref_audio(self, preset_path: Path) -> Optional[Path]:
        if self.voice_mode != "custom" or not self.ref_audio_file:
            return None
        candidate = preset_path.parent / self.ref_audio_file
        if candidate.exists():
            return candidate
        alt = preset_path.with_suffix(REF_AUDIO_SIDECAR_SUFFIX)
        if alt.exists():
            return alt
        return candidate


def _ref_audio_sidecar_name(preset_path: Path) -> str:
    return preset_path.with_suffix(REF_AUDIO_SIDECAR_SUFFIX).name


def _voice_prompt_sidecar_name(preset_path: Path) -> str:
    return preset_path.with_suffix(VOICE_PROMPT_SIDECAR_SUFFIX).name


def save_preset(
    preset: GenerationPreset,
    preset_path: Path,
    *,
    ref_audio_src: Optional[Path] = None,
    copy_ref_audio: bool = True,
    voice_prompt_items: Optional[list] = None,
    voice_cache_key: Optional[str] = None,
) -> Path:
    preset_path = preset_path.expanduser().resolve()
    preset_path.parent.mkdir(parents=True, exist_ok=True)

    if preset.voice_mode == "custom" and ref_audio_src is not None and copy_ref_audio:
        sidecar = preset_path.with_suffix(REF_AUDIO_SIDECAR_SUFFIX)
        shutil.copy2(ref_audio_src, sidecar)
        preset.ref_audio_file = _ref_audio_sidecar_name(preset_path)

    if voice_prompt_items:
        from .voice_prompt_cache import save_voice_prompt

        prompt_path = preset_path.with_suffix(VOICE_PROMPT_SIDECAR_SUFFIX)
        save_voice_prompt(voice_prompt_items, prompt_path, voice_cache_key=voice_cache_key)
        preset.voice_prompt_file = _voice_prompt_sidecar_name(preset_path)
        if voice_cache_key:
            preset.voice_cache_key = voice_cache_key

    if not preset.created_at:
        preset.created_at = datetime.now(timezone.utc).isoformat()
    if not preset.name:
        preset.name = preset_path.stem

    preset_path.write_text(
        json.dumps(preset.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return preset_path


def load_preset(preset_path: Path) -> GenerationPreset:
    preset_path = preset_path.expanduser().resolve()
    data = json.loads(preset_path.read_text(encoding="utf-8"))
    return GenerationPreset.from_dict(data)


def preset_to_generation_params(preset: GenerationPreset) -> GenerationParams:
    return preset.generation


def resolve_voice_inputs(
    preset: GenerationPreset,
    preset_path: Path,
) -> tuple[Optional[str], Optional[Path], Optional[str]]:
    if preset.voice_mode == "custom":
        ref_audio = preset.resolve_ref_audio(preset_path)
        if ref_audio is None or not ref_audio.exists():
            raise FileNotFoundError(
                f"Custom preset ref audio not found for {preset_path.name}. "
                f"Expected sidecar {_ref_audio_sidecar_name(preset_path)} or {preset.ref_audio_file!r}."
            )
        ref_text = (preset.ref_text or "").strip()
        if not ref_text:
            raise ValueError("Preset is missing ref_text for custom voice mode.")
        return None, ref_audio, ref_text

    if not preset.speaker_key:
        raise ValueError("Preset is missing speaker_key for built-in speaker mode.")
    return preset.speaker_key, None, None
