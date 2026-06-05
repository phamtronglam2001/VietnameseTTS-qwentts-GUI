from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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

    def to_generate_kwargs(self) -> dict:
        kwargs = dict(
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
            repetition_penalty=self.repetition_penalty,
            subtalker_temperature=self.subtalker_temperature,
            subtalker_top_k=self.subtalker_top_k,
            subtalker_top_p=self.subtalker_top_p,
        )
        if self.deterministic:
            kwargs["do_sample"] = False
            kwargs["subtalker_dosample"] = False
        else:
            kwargs["subtalker_dosample"] = True
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


def load_model(model_dir: Path, device: str) -> "Qwen3TTSModel":
    from qwen_tts import Qwen3TTSModel

    try:
        import flash_attn  # noqa: F401

        attn_impl = "flash_attention_2"
    except Exception:
        attn_impl = "sdpa"

    model = Qwen3TTSModel.from_pretrained(
        str(model_dir),
        device_map=device,
        dtype=pick_dtype(device),
        attn_implementation=attn_impl,
    )
    return model


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
) -> tuple[Path, int, "Qwen3TTSModel"]:
    if not text.strip():
        raise ValueError("Text is empty.")

    params = generation_params or DEFAULT_GENERATION_PARAMS
    set_generation_seed(params.seed)

    device = device or pick_device()
    if model is None:
        model = load_model(model_dir, device=device)

    gen_kwargs = params.to_generate_kwargs()
    language = params.language

    if ref_audio is not None:
        if not ref_text or not ref_text.strip():
            raise ValueError("ref_text is required when using custom ref_audio.")
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=str(ref_audio),
            ref_text=ref_text,
            **gen_kwargs,
        )
    else:
        if not speaker_key:
            raise ValueError("speaker_key is required when not using custom ref_audio.")
        speaker = get_speaker(ref_info_path, ref_audio_dir, speaker_key)
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=str(speaker.audio_path),
            ref_text=speaker.ref_text,
            **gen_kwargs,
        )

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_wav), wavs[0], sr)
    return output_wav, sr, model
