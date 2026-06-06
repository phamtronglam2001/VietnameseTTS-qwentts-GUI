from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from .platform_fixes import apply_platform_fixes, suppress_qwen_import_noise

apply_platform_fixes()

import soundfile as sf
import torch


@dataclass(frozen=True)
class GenerationParams:
    """Sampling and language options passed to ``generate_voice_clone``."""

    seed: Optional[int] = None
    deterministic: bool = False
    language: str = "Vietnamese"
    temperature: float = 0.3
    top_k: int = 20
    top_p: float = 0.9
    repetition_penalty: float = 2.0
    subtalker_temperature: float = 0.1
    subtalker_top_k: int = 20
    subtalker_top_p: float = 1.0
    max_new_tokens: int = 4096
    x_vector_only_mode: bool = False

    def to_dict(self) -> dict:
        return dict(
            seed=self.seed,
            deterministic=self.deterministic,
            language=self.language,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            subtalker_temperature=self.subtalker_temperature,
            subtalker_top_k=self.subtalker_top_k,
            subtalker_top_p=self.subtalker_top_p,
            max_new_tokens=self.max_new_tokens,
            x_vector_only_mode=self.x_vector_only_mode,
        )

    @classmethod
    def from_dict(cls, data: dict) -> GenerationParams:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def to_generate_kwargs(self) -> dict:
        # UI sliders may pass whole numbers as int; transformers requires real float
        # instances for repetition_penalty (RepetitionPenaltyLogitsProcessor).
        kwargs: dict = {
            "max_new_tokens": int(self.max_new_tokens),
            "repetition_penalty": float(self.repetition_penalty),
        }
        if self.deterministic:
            kwargs["do_sample"] = False
            kwargs["subtalker_dosample"] = False
        else:
            kwargs["subtalker_dosample"] = True
            kwargs.update(
                temperature=float(self.temperature),
                top_k=int(self.top_k),
                top_p=float(self.top_p),
                subtalker_temperature=float(self.subtalker_temperature),
                subtalker_top_k=int(self.subtalker_top_k),
                subtalker_top_p=float(self.subtalker_top_p),
            )
        return kwargs


DEFAULT_GENERATION_PARAMS = GenerationParams()

# Backward-compatible alias (library expects ``subtalker_dosample``, not ``subtalker_do_sample``).
GENERATION_CONFIG = DEFAULT_GENERATION_PARAMS.to_generate_kwargs()


def set_generation_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class Speaker:
    key: str
    name: str
    audio_path: Path
    ref_text: str


def pick_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def pick_dtype(device: str) -> torch.dtype:
    # BF16 is great on modern NVIDIA cards; CPU BF16 support is inconsistent.
    if device.startswith("cuda"):
        return torch.bfloat16
    return torch.float32


def pick_attn_implementation() -> str:
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:
        return "sdpa"


def load_ref_info(ref_info_path: Path) -> dict:
    return json.loads(ref_info_path.read_text(encoding="utf-8"))


def get_speaker(ref_info_path: Path, ref_audio_dir: Path, speaker_key: str) -> Speaker:
    info = load_ref_info(ref_info_path)
    if speaker_key not in info:
        raise KeyError(f"Unknown speaker: {speaker_key}")
    s = info[speaker_key]
    return Speaker(
        key=speaker_key,
        name=s.get("name", speaker_key),
        audio_path=ref_audio_dir / Path(s["audio_path"]).name,
        ref_text=s["text"],
    )


_SPEAKER_KEY_RE = re.compile(r"^[a-zA-Z0-9_]+$")


def add_builtin_speaker(
    root_dir: Path,
    key: str,
    display_name: str,
    ref_audio_src: Path,
    ref_text: str,
    *,
    overwrite: bool = False,
) -> Speaker:
    """Copy reference WAV into assets and register a new built-in speaker."""
    from .paths import AppPaths

    key = key.strip()
    display_name = display_name.strip()
    ref_text = ref_text.strip()

    if not key or not _SPEAKER_KEY_RE.match(key):
        raise ValueError("Key must be non-empty and contain only letters, digits, and underscores.")
    if not display_name:
        raise ValueError("Display name must not be empty.")
    if not ref_text:
        raise ValueError("Reference transcript must not be empty.")

    src = Path(ref_audio_src).expanduser()
    if not src.is_file():
        raise ValueError(f"Reference audio file not found: {src}")

    paths = AppPaths(root=root_dir)
    ref_info_path = paths.ref_info_json
    ref_audio_dir = paths.ref_audio_dir
    ref_audio_dir.mkdir(parents=True, exist_ok=True)
    ref_info_path.parent.mkdir(parents=True, exist_ok=True)

    ref_info = load_ref_info(ref_info_path) if ref_info_path.exists() else {}
    if key in ref_info and not overwrite:
        raise ValueError(f"Speaker key already exists: {key}")

    dest_wav = ref_audio_dir / f"{key}.wav"
    shutil.copy2(src, dest_wav)

    ref_info[key] = {
        "name": display_name,
        "audio_path": f"data/ref_audio/{key}.wav",
        "text": ref_text,
    }
    ref_info_path.write_text(
        json.dumps(ref_info, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )
    return get_speaker(ref_info_path, ref_audio_dir, key)


def load_model(model_dir: Path, device: str) -> "Qwen3TTSModel":
    with suppress_qwen_import_noise():
        from qwen_tts import Qwen3TTSModel

    attn_impl = pick_attn_implementation()

    model = Qwen3TTSModel.from_pretrained(
        str(model_dir),
        device_map=device,
        dtype=pick_dtype(device),
        attn_implementation=attn_impl,
    )
    return model


def current_voice_cache_key(
    *,
    voice_mode: str,
    speaker_key: Optional[str],
    ref_audio: Optional[Path],
    ref_text: Optional[str],
    x_vector_only_mode: bool,
) -> str:
    from .runtime_state import make_voice_cache_key

    return make_voice_cache_key(
        voice_mode="custom" if voice_mode == "custom" else "speaker",
        speaker_key=speaker_key,
        ref_audio=ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=x_vector_only_mode,
    )


def voice_prompt_matches(
    voice_clone_prompt: Optional[List[Any]],
    *,
    expected_key: Optional[str],
    stored_key: Optional[str] = None,
) -> bool:
    if voice_clone_prompt is None:
        return False
    if not expected_key:
        return True
    return stored_key == expected_key if stored_key else True


def build_voice_clone_prompt(
    model: "Qwen3TTSModel",
    *,
    ref_audio: Path,
    ref_text: str,
    x_vector_only_mode: bool = False,
) -> List[Any]:
    return model.create_voice_clone_prompt(
        ref_audio=str(ref_audio),
        ref_text=ref_text,
        x_vector_only_mode=x_vector_only_mode,
    )


def synthesize_to_wav(
    *,
    model_dir: Path,
    ref_info_path: Path,
    ref_audio_dir: Path,
    text: str,
    output_wav: Path,
    speaker_key: Optional[str] = "yen_nhi",
    ref_audio: Optional[Path] = None,
    ref_text: Optional[str] = None,
    device: Optional[str] = None,
    model: Optional["Qwen3TTSModel"] = None,
    generation_params: Optional[GenerationParams] = None,
    voice_clone_prompt: Optional[List[Any]] = None,
) -> tuple[Path, int, "Qwen3TTSModel", Optional[List[Any]]]:
    if not text.strip():
        raise ValueError("Text is empty.")

    params = generation_params or DEFAULT_GENERATION_PARAMS
    set_generation_seed(params.seed)

    device = device or pick_device()
    if model is None:
        model = load_model(model_dir, device=device)

    gen_kwargs = params.to_generate_kwargs()
    language = params.language
    prompt_items = voice_clone_prompt

    if prompt_items is None:
        if ref_audio is not None:
            if not ref_text or not ref_text.strip():
                raise ValueError("ref_text is required when using custom ref_audio.")
            prompt_items = build_voice_clone_prompt(
                model,
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=params.x_vector_only_mode,
            )
        else:
            if not speaker_key:
                raise ValueError("speaker_key is required when not using custom ref_audio.")
            speaker = get_speaker(ref_info_path, ref_audio_dir, speaker_key)
            prompt_items = build_voice_clone_prompt(
                model,
                ref_audio=speaker.audio_path,
                ref_text=speaker.ref_text,
                x_vector_only_mode=params.x_vector_only_mode,
            )

    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        voice_clone_prompt=prompt_items,
        **gen_kwargs,
    )

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_wav), wavs[0], sr)
    return output_wav, sr, model, prompt_items
