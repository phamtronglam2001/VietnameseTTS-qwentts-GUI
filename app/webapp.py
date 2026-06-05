from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import gradio as gr

from .bootstrap import bootstrap_all
from .engine import GenerationParams, load_ref_info, pick_device, synthesize_to_wav


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "outputs"


@dataclass
class AppState:
    bootstrap: Any = None
    ref_info: dict = field(default_factory=dict)
    model: Any = None
    device: str = field(default_factory=pick_device)


STATE = AppState()


def _speaker_choices() -> list[tuple[str, str]]:
    return [(info.get("name", key), key) for key, info in STATE.ref_info.items()]


def on_bootstrap() -> tuple[str, gr.Dropdown, gr.Button]:
    try:
        STATE.bootstrap = bootstrap_all(ROOT_DIR)
        STATE.ref_info = load_ref_info(STATE.bootstrap.ref_info_path)
        STATE.model = None
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


def _build_generation_params(
    deterministic: bool,
    seed: Optional[float],
    temperature: float,
    top_k: float,
    top_p: float,
    repetition_penalty: float,
) -> GenerationParams:
    seed_int: Optional[int] = None
    if seed is not None and seed >= 0:
        seed_int = int(seed)
    return GenerationParams(
        seed=seed_int,
        deterministic=deterministic,
        temperature=temperature,
        top_k=int(top_k),
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )


def on_generate(
    text: str,
    voice_mode: str,
    speaker_key: str,
    ref_audio: Optional[str],
    ref_text: str,
    deterministic: bool,
    seed: Optional[float],
    temperature: float,
    top_k: float,
    top_p: float,
    repetition_penalty: float,
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
        deterministic, seed, temperature, top_k, top_p, repetition_penalty
    )

    progress(0.1, desc="Loading model (cached after first run)…")
    try:
        out_path, sr, STATE.model = synthesize_to_wav(
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
        )
    except Exception as exc:
        tb = traceback.format_exc()
        raise gr.Error(f"Generation failed: {exc}\n\n{tb}") from exc

    progress(1.0, desc="Done")
    return str(out_path), f"Saved {out_path.name} ({sr} Hz) → {out_path}"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Gwen-TTS Vietnamese Web GUI") as demo:
        gr.Markdown(
            "# Gwen-TTS Vietnamese Web GUI\n"
            "Type Vietnamese in the browser (Unikey / Telex works here). "
            "Run **Bootstrap** once, then **Generate WAV**."
        )

        status = gr.Textbox(
            label="Status",
            value=f"Ready on {STATE.device}. Click Bootstrap first.",
            interactive=False,
            lines=2,
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
                with gr.Accordion("Advanced generation", open=False):
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

        audio_out = gr.Audio(label="Generated audio", type="filepath", interactive=False)
        out_status = gr.Textbox(label="Output", interactive=False)

        def sync_voice_mode(mode: str) -> tuple[gr.Dropdown, gr.Audio, gr.Textbox]:
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

        voice_mode.change(sync_voice_mode, inputs=voice_mode, outputs=[speaker, ref_audio, ref_text])
        txt_file.change(on_load_txt, inputs=txt_file, outputs=text_in)
        bootstrap_btn.click(on_bootstrap, outputs=[status, speaker, generate_btn])
        generate_btn.click(
            on_generate,
            inputs=[
                text_in,
                voice_mode,
                speaker,
                ref_audio,
                ref_text,
                deterministic,
                seed,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
            ],
            outputs=[audio_out, out_status],
        )

    return demo


def main() -> None:
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
