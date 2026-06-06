from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .platform_fixes import apply_platform_fixes, apply_windows_asyncio_fix, sox_status_line

apply_platform_fixes()

import gradio as gr

from .bootstrap import bootstrap_all
from .engine import (
    GenerationParams,
    add_builtin_speaker,
    current_voice_cache_key,
    load_ref_info,
    pick_device,
    synthesize_to_wav,
    voice_prompt_matches,
)
from .preset import GenerationPreset, RuntimeSnapshot, load_preset, resolve_voice_inputs, save_preset
from .runtime_state import capture_session, load_session_snapshot, save_session_snapshot
from .voice_prompt_cache import load_voice_prompt, resolve_voice_prompt_path


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "outputs"
PRESETS_DIR = ROOT_DIR / "presets"
SESSIONS_DIR = ROOT_DIR / "runtime_snapshots"


@dataclass
class AppState:
    bootstrap: Any = None
    ref_info: dict = field(default_factory=dict)
    model: Any = None
    device: str = field(default_factory=pick_device)
    voice_clone_prompt: Any = None
    voice_cache_key: Optional[str] = None
    generation_count: int = 0


STATE = AppState()


def _speaker_choices() -> list[tuple[str, str]]:
    return [(info.get("name", key), key) for key, info in STATE.ref_info.items()]


def on_bootstrap() -> tuple[str, gr.Dropdown, gr.Button]:
    try:
        STATE.bootstrap = bootstrap_all(ROOT_DIR)
        STATE.ref_info = load_ref_info(STATE.bootstrap.ref_info_path)
        STATE.model = None
        STATE.voice_clone_prompt = None
        STATE.voice_cache_key = None
        STATE.generation_count = 0
        speakers = _speaker_choices()
        default = speakers[0][1] if speakers else None
        msg = f"Bootstrap done on {STATE.device}. {len(speakers)} built-in speaker(s) ready."
        return msg, gr.Dropdown(choices=speakers, value=default), gr.Button(interactive=True)
    except Exception as exc:
        tb = traceback.format_exc()
        return f"Bootstrap failed: {exc}\n\n{tb}", gr.Dropdown(), gr.Button(interactive=False)


def on_load_txt(file: Optional[str]) -> str:
    if not file:
        return ""
    return Path(file).read_text(encoding="utf-8", errors="ignore")


def _can_save_builtin(voice_mode: str, ref_audio: Optional[str], ref_text: str) -> bool:
    return (
        voice_mode == "custom"
        and bool(ref_audio)
        and bool((ref_text or "").strip())
    )


def sync_save_builtin_btn(
    voice_mode: str,
    ref_audio: Optional[str],
    ref_text: str,
) -> gr.Button:
    return gr.Button(interactive=_can_save_builtin(voice_mode, ref_audio, ref_text))


def on_save_builtin_speaker(
    voice_mode: str,
    ref_audio: Optional[str],
    ref_text: str,
    builtin_key: str,
    builtin_name: str,
    overwrite: bool,
) -> tuple[gr.Dropdown, gr.Radio, gr.Textbox, gr.Audio, gr.Textbox, gr.Button, str]:
    if not STATE.bootstrap:
        raise gr.Error("Run Bootstrap first.")
    if voice_mode != "custom":
        raise gr.Error("Switch to Custom reference WAV mode first.")
    if not ref_audio:
        raise gr.Error("Upload reference audio before saving a built-in speaker.")
    if not (ref_text or "").strip():
        raise gr.Error("Enter the reference transcript before saving.")

    key = (builtin_key or "").strip()
    display_name = (builtin_name or "").strip()
    exists = key in STATE.ref_info
    if exists and not overwrite:
        raise gr.Error(
            f"Speaker key '{key}' already exists. Check 'Overwrite if key exists' to replace it."
        )

    try:
        speaker = add_builtin_speaker(
            ROOT_DIR,
            key,
            display_name,
            Path(ref_audio),
            ref_text,
            overwrite=exists,
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    STATE.ref_info = load_ref_info(STATE.bootstrap.ref_info_path)
    speakers = _speaker_choices()
    msg = f"Saved built-in speaker '{speaker.name}' ({speaker.key}). Switched to built-in speaker mode."
    return (
        gr.Dropdown(choices=speakers, value=speaker.key, interactive=True),
        gr.Radio(value="speaker"),
        gr.Textbox(interactive=False),
        gr.Audio(interactive=False),
        gr.Textbox(value=msg),
        gr.Button(interactive=False),
        msg,
    )


def _current_voice_key(
    voice_mode: str,
    speaker_key: str,
    ref_audio: Optional[str],
    ref_text: str,
    gen_params: GenerationParams,
) -> str:
    custom_audio = Path(ref_audio).expanduser() if ref_audio else None
    return current_voice_cache_key(
        voice_mode=voice_mode,
        speaker_key=speaker_key if voice_mode == "speaker" else None,
        ref_audio=custom_audio,
        ref_text=(ref_text or "").strip() or None,
        x_vector_only_mode=gen_params.x_vector_only_mode,
    )


def _resolve_cached_prompt(
    voice_mode: str,
    speaker_key: str,
    ref_audio: Optional[str],
    ref_text: str,
    gen_params: GenerationParams,
) -> Optional[Any]:
    expected = _current_voice_key(voice_mode, speaker_key, ref_audio, ref_text, gen_params)
    if voice_prompt_matches(
        STATE.voice_clone_prompt,
        expected_key=expected,
        stored_key=STATE.voice_cache_key,
    ):
        return STATE.voice_clone_prompt
    STATE.voice_clone_prompt = None
    STATE.voice_cache_key = None
    return None


def _build_generation_params(
    deterministic: bool,
    seed: Optional[float],
    language: str,
    temperature: float,
    top_k: float,
    top_p: float,
    repetition_penalty: float,
    subtalker_temperature: float,
    subtalker_top_k: float,
    subtalker_top_p: float,
    max_new_tokens: float,
) -> GenerationParams:
    seed_int: Optional[int] = None
    if seed is not None and seed >= 0:
        seed_int = int(seed)
    return GenerationParams(
        seed=seed_int,
        deterministic=deterministic,
        language=(language or "Vietnamese").strip() or "Vietnamese",
        temperature=temperature,
        top_k=int(top_k),
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        subtalker_temperature=subtalker_temperature,
        subtalker_top_k=int(subtalker_top_k),
        subtalker_top_p=subtalker_top_p,
        max_new_tokens=int(max_new_tokens),
    )


def _collect_preset(
    *,
    name: str,
    text: str,
    voice_mode: str,
    speaker_key: str,
    ref_audio: Optional[str],
    ref_text: str,
    output_wav: Optional[str],
    gen_params: GenerationParams,
) -> GenerationPreset:
    runtime = None
    if STATE.bootstrap:
        runtime = RuntimeSnapshot.capture(device=STATE.device, model_dir=STATE.bootstrap.model_dir)
    custom_audio = Path(ref_audio).expanduser() if ref_audio else None
    return GenerationPreset(
        name=(name or "").strip() or None,
        text=(text or "").strip(),
        voice_mode="custom" if voice_mode == "custom" else "speaker",
        speaker_key=speaker_key if voice_mode == "speaker" else None,
        ref_text=(ref_text or "").strip() or None if voice_mode == "custom" else None,
        ref_audio_file=None,
        generation=gen_params,
        output_wav=output_wav,
        runtime=runtime,
    )


def on_save_preset(
    preset_name: str,
    text: str,
    voice_mode: str,
    speaker_key: str,
    ref_audio: Optional[str],
    ref_text: str,
    deterministic: bool,
    seed: Optional[float],
    language: str,
    temperature: float,
    top_k: float,
    top_p: float,
    repetition_penalty: float,
    subtalker_temperature: float,
    subtalker_top_k: float,
    subtalker_top_p: float,
    max_new_tokens: float,
) -> tuple[Optional[str], str]:
    cleaned = (text or "").strip()
    if not cleaned:
        raise gr.Error("Input text is empty; nothing to save in preset.")

    if voice_mode == "custom" and not ref_audio:
        raise gr.Error("Upload reference audio before saving a custom-voice preset.")

    stem = (preset_name or "").strip() or datetime.now().strftime("preset_%Y%m%d_%H%M%S")
    safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    preset_path = PRESETS_DIR / f"{safe_stem}.json"

    gen_params = _build_generation_params(
        deterministic,
        seed,
        language,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        subtalker_temperature,
        subtalker_top_k,
        subtalker_top_p,
        max_new_tokens,
    )
    preset = _collect_preset(
        name=safe_stem,
        text=cleaned,
        voice_mode=voice_mode,
        speaker_key=speaker_key,
        ref_audio=ref_audio,
        ref_text=ref_text,
        output_wav=None,
        gen_params=gen_params,
    )
    if STATE.voice_clone_prompt is None:
        raise gr.Error(
            "No voice runtime cache yet. Generate audio at least once (after tuning) "
            "before saving, so the preset can store post-warmup voice embeddings."
        )

    ref_src = Path(ref_audio).expanduser() if voice_mode == "custom" and ref_audio else None
    cache_key = _current_voice_key(voice_mode, speaker_key, ref_audio, ref_text, gen_params)
    preset.generation_count = STATE.generation_count
    preset.voice_cache_key = cache_key
    save_preset(
        preset,
        preset_path,
        ref_audio_src=ref_src,
        voice_prompt_items=STATE.voice_clone_prompt,
        voice_cache_key=cache_key,
    )
    msg = f"Saved preset → {preset_path}"
    if ref_src is not None:
        msg += f" (+ {preset_path.with_suffix('.ref.wav').name})"
    if STATE.voice_clone_prompt is not None:
        msg += f" (+ {preset_path.with_suffix('.prompt.pt').name} voice cache)"
    return str(preset_path), msg


def on_load_preset(preset_file: Optional[str]) -> tuple:
    if not preset_file:
        raise gr.Error("Choose a preset JSON file to load.")

    preset_path = Path(preset_file).expanduser()
    preset = load_preset(preset_path)
    speaker_key, ref_audio_path, ref_text = resolve_voice_inputs(preset, preset_path)
    gen = preset.generation

    voice_mode = preset.voice_mode
    speaker_val = speaker_key if voice_mode == "speaker" else None
    ref_audio_val = str(ref_audio_path) if ref_audio_path else None
    ref_text_val = ref_text or preset.ref_text or ""

    speaker_update = gr.Dropdown(value=speaker_val, interactive=(voice_mode == "speaker"))
    ref_audio_update = gr.Audio(value=ref_audio_val, interactive=(voice_mode == "custom"))
    ref_text_update = gr.Textbox(value=ref_text_val, interactive=(voice_mode == "custom"))

    prompt_path = resolve_voice_prompt_path(preset_path, preset.voice_prompt_file)
    if prompt_path is not None:
        STATE.voice_clone_prompt, stored_key = load_voice_prompt(prompt_path)
        STATE.voice_cache_key = preset.voice_cache_key or stored_key
        prompt_note = f" (+ restored voice cache from {prompt_path.name})"
    else:
        STATE.voice_clone_prompt = None
        STATE.voice_cache_key = None
        prompt_note = " (no voice cache; will re-encode ref audio on next run)"
    STATE.generation_count = preset.generation_count

    return (
        preset.text,
        voice_mode,
        speaker_update,
        ref_audio_update,
        ref_text_update,
        gen.deterministic,
        gen.seed,
        gen.language,
        gen.temperature,
        gen.top_k,
        gen.top_p,
        gen.repetition_penalty,
        gen.subtalker_temperature,
        gen.subtalker_top_k,
        gen.subtalker_top_p,
        gen.max_new_tokens,
        f"Loaded preset {preset_path.name}"
        + (f" (saved {preset.created_at})" if preset.created_at else "")
        + prompt_note,
    )


def on_save_session(
    session_name: str,
    voice_mode: str,
    speaker_key: str,
    ref_audio: Optional[str],
    ref_text: str,
    deterministic: bool,
    seed: Optional[float],
    language: str,
    temperature: float,
    top_k: float,
    top_p: float,
    repetition_penalty: float,
    subtalker_temperature: float,
    subtalker_top_k: float,
    subtalker_top_p: float,
    max_new_tokens: float,
) -> tuple[Optional[str], str]:
    if STATE.voice_clone_prompt is None:
        raise gr.Error(
            "No runtime voice cache to save. Run Generate at least once after tuning "
            "deterministic/sampling until audio sounds correct."
        )

    gen_params = _build_generation_params(
        deterministic,
        seed,
        language,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        subtalker_temperature,
        subtalker_top_k,
        subtalker_top_p,
        max_new_tokens,
    )
    ref_src = Path(ref_audio).expanduser() if voice_mode == "custom" and ref_audio else None
    stem = (session_name or "").strip() or datetime.now().strftime("session_%Y%m%d_%H%M%S")
    safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    session_path = SESSIONS_DIR / f"{safe_stem}.json"

    session = capture_session(
        voice_mode="custom" if voice_mode == "custom" else "speaker",
        speaker_key=speaker_key if voice_mode == "speaker" else None,
        ref_audio=ref_src,
        ref_text=(ref_text or "").strip() or None,
        generation=gen_params,
        generation_count=STATE.generation_count,
        voice_clone_prompt=STATE.voice_clone_prompt,
        device=STATE.device,
        model_dir=STATE.bootstrap.model_dir if STATE.bootstrap else None,
        name=safe_stem,
    )
    save_session_snapshot(
        session,
        session_path,
        ref_audio_src=ref_src,
        voice_prompt_items=STATE.voice_clone_prompt,
    )
    msg = (
        f"Saved runtime snapshot → {session_path} "
        f"(+ {session_path.with_suffix('.prompt.pt').name}, "
        f"{STATE.generation_count} prior generation(s))"
    )
    return str(session_path), msg


def on_load_session(session_file: Optional[str]) -> tuple:
    if not session_file:
        raise gr.Error("Choose a runtime snapshot JSON file to load.")

    session_path = Path(session_file).expanduser()
    session, prompt_items = load_session_snapshot(session_path)

    voice_mode = session.voice_mode
    speaker_val = session.speaker_key if voice_mode == "speaker" else None
    ref_audio_val = None
    ref_text_val = session.ref_text or ""
    if voice_mode == "custom":
        from .runtime_state import resolve_session_ref_audio

        ref_path = resolve_session_ref_audio(session, session_path)
        if ref_path is not None and ref_path.exists():
            ref_audio_val = str(ref_path)
        else:
            raise gr.Error(f"Custom session ref audio sidecar missing next to {session_path.name}")

    speaker_update = gr.Dropdown(value=speaker_val, interactive=(voice_mode == "speaker"))
    ref_audio_update = gr.Audio(value=ref_audio_val, interactive=(voice_mode == "custom"))
    ref_text_update = gr.Textbox(value=ref_text_val, interactive=(voice_mode == "custom"))

    STATE.voice_clone_prompt = prompt_items
    STATE.voice_cache_key = session.voice_cache_key
    STATE.generation_count = session.generation_count
    gen = session.generation

    prompt_note = (
        f" (+ voice cache, {session.generation_count} warmup gen(s))"
        if prompt_items is not None
        else " (no voice cache sidecar)"
    )

    return (
        voice_mode,
        speaker_update,
        ref_audio_update,
        ref_text_update,
        gen.deterministic,
        gen.seed,
        gen.language,
        gen.temperature,
        gen.top_k,
        gen.top_p,
        gen.repetition_penalty,
        gen.subtalker_temperature,
        gen.subtalker_top_k,
        gen.subtalker_top_p,
        gen.max_new_tokens,
        f"Loaded runtime snapshot {session_path.name}{prompt_note}",
    )


def on_generate(
    text: str,
    voice_mode: str,
    speaker_key: str,
    ref_audio: Optional[str],
    ref_text: str,
    deterministic: bool,
    seed: Optional[float],
    language: str,
    temperature: float,
    top_k: float,
    top_p: float,
    repetition_penalty: float,
    subtalker_temperature: float,
    subtalker_top_k: float,
    subtalker_top_p: float,
    max_new_tokens: float,
    progress: gr.Progress = gr.Progress(),
) -> tuple[Optional[str], str]:
    if not STATE.bootstrap:
        raise gr.Error("Please click Bootstrap (Download/Cache) first.")

    cleaned = (text or "").strip()
    if not cleaned:
        raise gr.Error("Input text is empty.")

    speaker = speaker_key or None
    custom_audio = Path(ref_audio).expanduser() if ref_audio else None
    custom_text = (ref_text or "").strip() or None

    if voice_mode == "custom":
        if custom_audio is None:
            raise gr.Error("Upload a reference WAV file for custom voice cloning.")
        speaker = None
    else:
        if not speaker:
            raise gr.Error("Select a built-in speaker.")
        custom_audio = None
        custom_text = None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"output_{stamp}.wav"

    gen_params = _build_generation_params(
        deterministic,
        seed,
        language,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        subtalker_temperature,
        subtalker_top_k,
        subtalker_top_p,
        max_new_tokens,
    )

    cached_prompt = _resolve_cached_prompt(
        voice_mode, speaker_key or "", ref_audio, ref_text, gen_params
    )

    progress(0.1, desc="Loading model (cached after first run)…")
    try:
        out_path, sr, STATE.model, STATE.voice_clone_prompt = synthesize_to_wav(
            model_dir=STATE.bootstrap.model_dir,
            ref_info_path=STATE.bootstrap.ref_info_path,
            ref_audio_dir=STATE.bootstrap.ref_audio_dir,
            text=cleaned,
            output_wav=out_path,
            speaker_key=speaker,
            ref_audio=custom_audio,
            ref_text=custom_text,
            device=STATE.device,
            model=STATE.model,
            generation_params=gen_params,
            voice_clone_prompt=cached_prompt,
        )
    except Exception as exc:
        tb = traceback.format_exc()
        raise gr.Error(f"Generation failed: {exc}\n\n{tb}") from exc

    STATE.generation_count += 1
    STATE.voice_cache_key = _current_voice_key(
        voice_mode, speaker_key or "", ref_audio, ref_text, gen_params
    )
    progress(1.0, desc="Done")
    cache_note = " (reused voice cache)" if cached_prompt is not None else ""
    return str(out_path), f"Saved {out_path.name} ({sr} Hz) → {out_path}{cache_note}"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Gwen-TTS Vietnamese Web GUI") as demo:
        gr.Markdown(
            "# Gwen-TTS Vietnamese Web GUI\n"
            "Type Vietnamese in the browser (Unikey / Telex works here). "
            "Run **Bootstrap** once, then **Generate WAV**."
        )

        _sox_hint = sox_status_line()
        _ready_msg = f"Ready on {STATE.device}. Click Bootstrap first."
        if _sox_hint:
            _ready_msg = f"{_ready_msg}\n{_sox_hint}"
        status = gr.Textbox(
            label="Status",
            value=_ready_msg,
            interactive=False,
            lines=3 if _sox_hint else 2,
        )

        with gr.Row():
            bootstrap_btn = gr.Button("Bootstrap (Download/Cache)", variant="primary")
            generate_btn = gr.Button("Generate WAV", variant="primary", interactive=False)

        with gr.Row():
            with gr.Column(scale=3):
                text_in = gr.Textbox(
                    label="Input text",
                    placeholder="Nhập văn bản tiếng Việt tại đây…",
                    lines=14,
                )
                txt_file = gr.File(
                    label="Or load .txt file",
                    file_types=[".txt"],
                    type="filepath",
                )
            with gr.Column(scale=2):
                voice_mode = gr.Radio(
                    choices=[
                        ("Built-in speaker", "speaker"),
                        ("Custom reference WAV", "custom"),
                    ],
                    value="speaker",
                    label="Voice source",
                )
                speaker = gr.Dropdown(
                    label="Speaker",
                    choices=[],
                    value=None,
                    interactive=True,
                )
                ref_audio = gr.Audio(
                    label="Custom reference audio (WAV)",
                    type="filepath",
                    interactive=True,
                )
                ref_text = gr.Textbox(
                    label="Custom reference transcript",
                    placeholder="Transcript of the reference audio…",
                    lines=5,
                )

                with gr.Accordion("Save as built-in speaker", open=False):
                    gr.Markdown(
                        "After cloning with a custom WAV + transcript, register it as a "
                        "built-in speaker. It is copied to `assets/ref_audio/` and listed "
                        "automatically on the next Bootstrap."
                    )
                    builtin_key = gr.Textbox(
                        label="Built-in voice key",
                        placeholder="my_clone",
                        info="Letters, digits, and underscores only.",
                    )
                    builtin_name = gr.Textbox(
                        label="Display name",
                        placeholder="My Voice",
                    )
                    overwrite_builtin = gr.Checkbox(
                        label="Overwrite if key exists",
                        value=False,
                    )
                    save_builtin_btn = gr.Button(
                        "Save as built-in speaker",
                        interactive=False,
                    )
                    save_builtin_status = gr.Textbox(label="Built-in speaker", interactive=False, lines=2)

                gr.Markdown(
                    "**Reproducibility:** The model samples randomly by default, so the same text "
                    "can sound different each run. Enable **Deterministic / fixed output** and/or "
                    "set a **Seed** for repeatable results."
                )
                deterministic = gr.Checkbox(
                    label="Deterministic / fixed output",
                    value=False,
                    info="Disables sampling (do_sample=False, subtalker_dosample=False).",
                )
                seed = gr.Number(
                    label="Seed (optional)",
                    value=None,
                    precision=0,
                    info="Leave empty for random. Use the same seed + deterministic mode for identical WAVs.",
                )
                with gr.Accordion("Save / load preset", open=False):
                    gr.Markdown(
                        "After a **good** generation, save the preset here. It stores your tuned "
                        "sliders **and** the computed voice-clone embeddings (`.prompt.pt`), so "
                        "the next session can skip re-encoding and reuse the same runtime voice state."
                    )
                    preset_name = gr.Textbox(
                        label="Preset name",
                        placeholder="my_session",
                    )
                    with gr.Row():
                        save_preset_btn = gr.Button("Save preset")
                        preset_download = gr.File(label="Download preset JSON", interactive=False)
                    preset_load_file = gr.File(
                        label="Load preset JSON",
                        file_types=[".json"],
                        type="filepath",
                    )
                    preset_status = gr.Textbox(label="Preset", interactive=False, lines=2)

                with gr.Accordion("Runtime snapshot (post-warmup)", open=False):
                    gr.Markdown(
                        "After audio sounds **correct**, save the runtime snapshot here. "
                        "It stores tuned generation params and the precomputed voice-clone "
                        "embeddings (``.prompt.pt``), not model weights. "
                        "Load it in a new session to skip re-encoding reference audio."
                    )
                    session_name = gr.Textbox(
                        label="Snapshot name",
                        placeholder="warm_session",
                    )
                    with gr.Row():
                        save_session_btn = gr.Button("Save runtime snapshot")
                        session_download = gr.File(
                            label="Download snapshot JSON",
                            interactive=False,
                        )
                    session_load_file = gr.File(
                        label="Load runtime snapshot JSON",
                        file_types=[".json"],
                        type="filepath",
                    )
                    session_status = gr.Textbox(label="Runtime snapshot", interactive=False, lines=2)

                with gr.Accordion("Advanced generation", open=False):
                    language = gr.Textbox(
                        label="Language",
                        value="Vietnamese",
                        info="Passed to generate_voice_clone (e.g. Vietnamese, English).",
                    )
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.0,
                        maximum=2.0,
                        value=0.3,
                        step=0.05,
                    )
                    top_k = gr.Slider(
                        label="Top-k",
                        minimum=1,
                        maximum=100,
                        value=20,
                        step=1,
                        precision=0,
                    )
                    top_p = gr.Slider(
                        label="Top-p",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                    )
                    repetition_penalty = gr.Slider(
                        label="Repetition penalty",
                        minimum=1.0,
                        maximum=3.0,
                        value=2.0,
                        step=0.05,
                    )
                    subtalker_temperature = gr.Slider(
                        label="Subtalker temperature",
                        minimum=0.0,
                        maximum=2.0,
                        value=0.1,
                        step=0.05,
                    )
                    subtalker_top_k = gr.Slider(
                        label="Subtalker top-k",
                        minimum=1,
                        maximum=100,
                        value=20,
                        step=1,
                        precision=0,
                    )
                    subtalker_top_p = gr.Slider(
                        label="Subtalker top-p",
                        minimum=0.0,
                        maximum=1.0,
                        value=1.0,
                        step=0.05,
                    )
                    max_new_tokens = gr.Slider(
                        label="Max new tokens",
                        minimum=256,
                        maximum=8192,
                        value=4096,
                        step=256,
                        precision=0,
                    )

        audio_out = gr.Audio(label="Generated audio", type="filepath", interactive=False)
        out_status = gr.Textbox(label="Output", interactive=False)

        def sync_voice_mode(mode: str) -> tuple[gr.Dropdown, gr.Audio, gr.Textbox]:
            STATE.voice_clone_prompt = None
            STATE.voice_cache_key = None
            if mode == "speaker":
                return (
                    gr.Dropdown(interactive=True),
                    gr.Audio(interactive=False),
                    gr.Textbox(interactive=False),
                )
            return (
                gr.Dropdown(interactive=False),
                gr.Audio(interactive=True),
                gr.Textbox(interactive=True),
            )

        def invalidate_voice_cache(*_args) -> None:
            STATE.voice_clone_prompt = None
            STATE.voice_cache_key = None

        gen_inputs = [
            text_in,
            voice_mode,
            speaker,
            ref_audio,
            ref_text,
            deterministic,
            seed,
            language,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            subtalker_temperature,
            subtalker_top_k,
            subtalker_top_p,
            max_new_tokens,
        ]

        voice_mode.change(sync_voice_mode, inputs=voice_mode, outputs=[speaker, ref_audio, ref_text])
        speaker.change(invalidate_voice_cache, inputs=speaker, outputs=[])
        ref_audio.change(invalidate_voice_cache, inputs=ref_audio, outputs=[])
        ref_text.change(invalidate_voice_cache, inputs=ref_text, outputs=[])
        for _inp in (voice_mode, ref_audio, ref_text):
            _inp.change(
                sync_save_builtin_btn,
                inputs=[voice_mode, ref_audio, ref_text],
                outputs=save_builtin_btn,
            )
        save_builtin_btn.click(
            on_save_builtin_speaker,
            inputs=[voice_mode, ref_audio, ref_text, builtin_key, builtin_name, overwrite_builtin],
            outputs=[speaker, voice_mode, ref_text, ref_audio, save_builtin_status, save_builtin_btn, status],
        )
        txt_file.change(on_load_txt, inputs=txt_file, outputs=text_in)
        bootstrap_btn.click(on_bootstrap, outputs=[status, speaker, generate_btn])
        generate_btn.click(
            on_generate,
            inputs=gen_inputs,
            outputs=[audio_out, out_status],
        )
        save_preset_btn.click(
            on_save_preset,
            inputs=[preset_name, *gen_inputs],
            outputs=[preset_download, preset_status],
        )
        preset_load_file.change(
            on_load_preset,
            inputs=preset_load_file,
            outputs=[
                text_in,
                voice_mode,
                speaker,
                ref_audio,
                ref_text,
                deterministic,
                seed,
                language,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                subtalker_temperature,
                subtalker_top_k,
                subtalker_top_p,
                max_new_tokens,
                preset_status,
            ],
        )
        save_session_btn.click(
            on_save_session,
            inputs=[session_name, *gen_inputs[1:]],
            outputs=[session_download, session_status],
        )
        session_load_file.change(
            on_load_session,
            inputs=session_load_file,
            outputs=[
                voice_mode,
                speaker,
                ref_audio,
                ref_text,
                deterministic,
                seed,
                language,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                subtalker_temperature,
                subtalker_top_k,
                subtalker_top_p,
                max_new_tokens,
                session_status,
            ],
        )

    return demo


def main() -> None:
    apply_windows_asyncio_fix()
    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        inbrowser=True,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        quiet=False,
    )


if __name__ == "__main__":
    main()
