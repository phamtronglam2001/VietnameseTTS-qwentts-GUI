from __future__ import annotations

import threading
import traceback
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, TOP, X, Y, filedialog, messagebox
import tkinter as tk
from tkinter import ttk

from .platform_fixes import sox_status_line

from .bootstrap import bootstrap_all
from .engine import (
    GenerationParams,
    add_builtin_speaker,
    current_voice_cache_key,
    load_ref_info,
    synthesize_to_wav,
    voice_prompt_matches,
)
from .ime_support import configure_unicode_fonts, enable_windows_ime, setup_text_widget
from .preset import GenerationPreset, RuntimeSnapshot, load_preset, resolve_voice_inputs, save_preset
from .runtime_state import capture_session, load_session_snapshot, save_session_snapshot
from .voice_prompt_cache import load_voice_prompt, resolve_voice_prompt_path


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Gwen-TTS Vietnamese GUI (Local)")
        self.geometry("900x650")

        self.root_dir = Path(__file__).resolve().parents[1]
        self._bootstrap_done = False
        self._bootstrap = None
        self._ref_info = {}
        self._gen_extras = GenerationParams()
        self._voice_clone_prompt = None
        self._voice_cache_key: str | None = None
        self._generation_count = 0
        self._text_font = configure_unicode_fonts(self)
        enable_windows_ime(self)

        self._build_ui()

    def _build_ui(self) -> None:
        pad = 10

        top = ttk.Frame(self, padding=pad)
        top.pack(side=TOP, fill=X)

        _ready = "Ready. Click 'Bootstrap (Download/Cache)' first."
        _sox_hint = sox_status_line()
        if _sox_hint:
            _ready = f"{_ready}\n{_sox_hint}"
        self.status_var = tk.StringVar(value=_ready)
        ttk.Label(top, textvariable=self.status_var).pack(side=LEFT, fill=X, expand=True)

        self.bootstrap_btn = ttk.Button(top, text="Bootstrap (Download/Cache)", command=self.on_bootstrap)
        self.bootstrap_btn.pack(side=RIGHT)

        main = ttk.Frame(self, padding=pad)
        main.pack(fill=BOTH, expand=True)

        text_row = ttk.Frame(main)
        text_row.pack(fill=BOTH, expand=True)

        left = ttk.Frame(text_row)
        left.pack(side=LEFT, fill=BOTH, expand=True)

        ttk.Label(left, text="Input text").pack(anchor="w")
        self.text_box = tk.Text(left, height=15, wrap="word", font=self._text_font)
        setup_text_widget(self.text_box, self._text_font)
        self.text_box.pack(fill=BOTH, expand=True)

        file_btns = ttk.Frame(left)
        file_btns.pack(fill=X, pady=(6, 0))
        ttk.Button(file_btns, text="Load .txt…", command=self.on_load_txt).pack(side=LEFT)
        ttk.Button(file_btns, text="Clear", command=lambda: self.text_box.delete("1.0", END)).pack(side=LEFT, padx=(6, 0))

        right = ttk.Frame(text_row)
        right.pack(side=RIGHT, fill=Y, padx=(12, 0))

        ttk.Label(right, text="Voice source").pack(anchor="w")
        self.voice_mode = tk.StringVar(value="speaker")
        ttk.Radiobutton(right, text="Built-in speaker", variable=self.voice_mode, value="speaker", command=self._sync_voice_mode).pack(anchor="w")
        ttk.Radiobutton(right, text="Custom reference WAV", variable=self.voice_mode, value="custom", command=self._sync_voice_mode).pack(anchor="w")

        ttk.Separator(right, orient="horizontal").pack(fill=X, pady=8)

        ttk.Label(right, text="Speaker").pack(anchor="w")
        self.speaker_var = tk.StringVar(value="yen_nhi")
        self.speaker_combo = ttk.Combobox(right, textvariable=self.speaker_var, values=["yen_nhi"], state="readonly", width=28)
        self.speaker_combo.pack(fill=X)
        self.speaker_var.trace_add("write", lambda *_: self._invalidate_voice_cache())

        ttk.Label(right, text="Custom ref audio (WAV)").pack(anchor="w", pady=(10, 0))
        self.ref_audio_var = tk.StringVar(value="")
        ref_audio_row = ttk.Frame(right)
        ref_audio_row.pack(fill=X)
        self.ref_audio_entry = ttk.Entry(ref_audio_row, textvariable=self.ref_audio_var, width=28)
        self.ref_audio_entry.pack(side=LEFT, fill=X, expand=True)
        ttk.Button(ref_audio_row, text="…", width=3, command=self.on_pick_ref_audio).pack(side=RIGHT, padx=(6, 0))

        ttk.Label(right, text="Custom ref transcript").pack(anchor="w", pady=(10, 0))
        self.ref_text = tk.Text(right, height=6, wrap="word", font=self._text_font)
        setup_text_widget(self.ref_text, self._text_font)
        self.ref_text.pack(fill=X)
        self.ref_text.bind("<<Modified>>", self._on_ref_text_modified)

        save_builtin = ttk.LabelFrame(right, text="Save as built-in speaker", padding=6)
        save_builtin.pack(fill=X, pady=(10, 0))
        ttk.Label(save_builtin, text="Built-in voice key").pack(anchor="w")
        self.builtin_key_var = tk.StringVar(value="")
        ttk.Entry(save_builtin, textvariable=self.builtin_key_var, width=28).pack(fill=X)
        ttk.Label(save_builtin, text="Display name").pack(anchor="w", pady=(6, 0))
        self.builtin_name_var = tk.StringVar(value="")
        ttk.Entry(save_builtin, textvariable=self.builtin_name_var, width=28).pack(fill=X)
        self.save_builtin_btn = ttk.Button(
            save_builtin,
            text="Save as built-in speaker",
            command=self.on_save_builtin_speaker,
            state="disabled",
        )
        self.save_builtin_btn.pack(anchor="w", pady=(8, 0))
        self.builtin_key_var.trace_add("write", lambda *_: self._sync_save_builtin_state())
        self.ref_audio_var.trace_add("write", lambda *_: self._sync_save_builtin_state())

        gen_frame = ttk.LabelFrame(right, text="Generation", padding=6)
        gen_frame.pack(fill=X, pady=(12, 0))

        self.deterministic_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            gen_frame,
            text="Deterministic / fixed output",
            variable=self.deterministic_var,
        ).pack(anchor="w")

        seed_row = ttk.Frame(gen_frame)
        seed_row.pack(fill=X, pady=(6, 0))
        ttk.Label(seed_row, text="Seed (optional)").pack(side=LEFT)
        self.seed_var = tk.StringVar(value="")
        ttk.Entry(seed_row, textvariable=self.seed_var, width=12).pack(side=RIGHT)

        ttk.Label(
            gen_frame,
            text="Same text varies without seed/deterministic mode.",
            wraplength=260,
            font=("", 8),
        ).pack(anchor="w", pady=(6, 0))

        adv = ttk.LabelFrame(gen_frame, text="Advanced", padding=4)
        adv.pack(fill=X, pady=(8, 0))

        lang_row = ttk.Frame(adv)
        lang_row.pack(fill=X, pady=2)
        ttk.Label(lang_row, text="Language", width=16).pack(side=LEFT)
        self.language_var = tk.StringVar(value="Vietnamese")
        ttk.Entry(lang_row, textvariable=self.language_var, width=16).pack(side=RIGHT)

        self.temperature_var = tk.DoubleVar(value=0.3)
        self._add_spin(adv, "Temperature", self.temperature_var, 0.0, 2.0, 0.05)

        self.top_k_var = tk.IntVar(value=20)
        self._add_spin(adv, "Top-k", self.top_k_var, 1, 100, 1, is_int=True)

        self.top_p_var = tk.DoubleVar(value=0.9)
        self._add_spin(adv, "Top-p", self.top_p_var, 0.0, 1.0, 0.05)

        self.repetition_penalty_var = tk.DoubleVar(value=2.0)
        self._add_spin(adv, "Repetition penalty", self.repetition_penalty_var, 1.0, 3.0, 0.05)

        preset_row = ttk.Frame(main)
        preset_row.pack(fill=X, pady=(8, 0))
        ttk.Button(preset_row, text="Save preset…", command=self.on_save_preset).pack(side=LEFT)
        ttk.Button(preset_row, text="Load preset…", command=self.on_load_preset).pack(side=LEFT, padx=(8, 0))
        ttk.Button(preset_row, text="Save runtime…", command=self.on_save_session).pack(side=LEFT, padx=(8, 0))
        ttk.Button(preset_row, text="Load runtime…", command=self.on_load_session).pack(side=LEFT, padx=(8, 0))

        ttk.Separator(main, orient="horizontal").pack(fill=X, pady=12)

        out_row = ttk.Frame(main)
        out_row.pack(fill=X)

        ttk.Label(out_row, text="Output WAV").pack(side=LEFT)
        self.out_var = tk.StringVar(value=str(self.root_dir / "output.wav"))
        self.out_entry = ttk.Entry(out_row, textvariable=self.out_var)
        self.out_entry.pack(side=LEFT, fill=X, expand=True, padx=(10, 6))
        ttk.Button(out_row, text="Browse…", command=self.on_pick_output).pack(side=LEFT)

        action_row = ttk.Frame(main)
        action_row.pack(fill=X, pady=(10, 0))

        self.run_btn = ttk.Button(action_row, text="Generate WAV", command=self.on_generate, state="disabled")
        self.run_btn.pack(side=LEFT)

        ttk.Button(action_row, text="Open output folder", command=self.on_open_output_folder).pack(side=LEFT, padx=(10, 0))

        self._sync_voice_mode()

    def _add_spin(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.Variable,
        from_: float,
        to: float,
        increment: float,
        *,
        is_int: bool = False,
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=X, pady=2)
        ttk.Label(row, text=label, width=16).pack(side=LEFT)
        if is_int:
            ttk.Spinbox(
                row,
                from_=int(from_),
                to=int(to),
                increment=int(increment),
                textvariable=variable,
                width=8,
            ).pack(side=RIGHT)
        else:
            ttk.Spinbox(
                row,
                from_=from_,
                to=to,
                increment=increment,
                textvariable=variable,
                width=8,
            ).pack(side=RIGHT)

    def _build_generation_params(self) -> GenerationParams:
        seed_raw = self.seed_var.get().strip()
        seed: int | None = None
        if seed_raw:
            try:
                seed = int(seed_raw)
            except ValueError:
                raise ValueError("Seed must be a whole number or left empty.")
        return GenerationParams(
            seed=seed,
            deterministic=self.deterministic_var.get(),
            language=self.language_var.get().strip() or "Vietnamese",
            temperature=self.temperature_var.get(),
            top_k=self.top_k_var.get(),
            top_p=self.top_p_var.get(),
            repetition_penalty=self.repetition_penalty_var.get(),
            subtalker_temperature=self._gen_extras.subtalker_temperature,
            subtalker_top_k=self._gen_extras.subtalker_top_k,
            subtalker_top_p=self._gen_extras.subtalker_top_p,
            max_new_tokens=self._gen_extras.max_new_tokens,
            x_vector_only_mode=self._gen_extras.x_vector_only_mode,
        )

    def _invalidate_voice_cache(self) -> None:
        self._voice_clone_prompt = None
        self._voice_cache_key = None

    def _current_voice_key(self, mode: str, speaker_key: str | None, ref_audio: Path | None, ref_text: str | None, gen: GenerationParams) -> str:
        return current_voice_cache_key(
            voice_mode=mode,
            speaker_key=speaker_key,
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=gen.x_vector_only_mode,
        )

    def _resolve_cached_prompt(self, mode: str, speaker_key: str | None, ref_audio: Path | None, ref_text: str | None, gen: GenerationParams):
        expected = self._current_voice_key(mode, speaker_key, ref_audio, ref_text, gen)
        if voice_prompt_matches(
            self._voice_clone_prompt,
            expected_key=expected,
            stored_key=self._voice_cache_key,
        ):
            return self._voice_clone_prompt
        self._voice_clone_prompt = None
        self._voice_cache_key = None
        return None

    def _apply_generation_params(self, gen: GenerationParams) -> None:
        self._gen_extras = gen
        self.deterministic_var.set(gen.deterministic)
        self.seed_var.set("" if gen.seed is None else str(gen.seed))
        self.language_var.set(gen.language)
        self.temperature_var.set(gen.temperature)
        self.top_k_var.set(gen.top_k)
        self.top_p_var.set(gen.top_p)
        self.repetition_penalty_var.set(gen.repetition_penalty)

    def on_save_preset(self) -> None:
        text = self.text_box.get("1.0", END).strip()
        if not text:
            messagebox.showwarning("Nothing to save", "Input text is empty.")
            return

        mode = self.voice_mode.get()
        ref_audio = None
        ref_text = None
        speaker_key = None
        if mode == "speaker":
            speaker_key = self.speaker_var.get()
        else:
            ref_audio = Path(self.ref_audio_var.get()).expanduser() if self.ref_audio_var.get().strip() else None
            ref_text = self.ref_text.get("1.0", END).strip()
            if ref_audio is None:
                messagebox.showwarning("Missing reference audio", "Pick a reference WAV file.")
                return

        preset_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("TTS preset", "*.json"), ("All files", "*.*")],
            initialdir=str(self.root_dir / "presets"),
            title="Save generation preset",
        )
        if not preset_path:
            return

        try:
            gen_params = self._build_generation_params()
        except ValueError as e:
            messagebox.showwarning("Invalid generation settings", str(e))
            return

        runtime = None
        if self._bootstrap:
            runtime = RuntimeSnapshot.capture(
                model_dir=self._bootstrap.model_dir,
            )
        preset = GenerationPreset(
            name=Path(preset_path).stem,
            text=text,
            voice_mode=mode,
            speaker_key=speaker_key,
            ref_text=ref_text,
            generation=gen_params,
            output_wav=str(Path(self.out_var.get()).expanduser()),
            runtime=runtime,
        )
        if self._voice_clone_prompt is None:
            messagebox.showwarning(
                "No runtime cache",
                "Generate audio at least once after tuning before saving a preset.",
            )
            return

        cache_key = self._current_voice_key(mode, speaker_key, ref_audio, ref_text, gen_params)
        preset.generation_count = self._generation_count
        preset.voice_cache_key = cache_key
        save_preset(
            preset,
            Path(preset_path),
            ref_audio_src=ref_audio,
            voice_prompt_items=self._voice_clone_prompt,
            voice_cache_key=cache_key,
        )
        status = f"Saved preset: {preset_path}"
        if self._voice_clone_prompt is not None:
            status += f" (+ {Path(preset_path).with_suffix('.prompt.pt').name})"
        self.set_status(status)

    def on_load_preset(self) -> None:
        preset_path = filedialog.askopenfilename(
            filetypes=[("TTS preset", "*.json"), ("All files", "*.*")],
            initialdir=str(self.root_dir / "presets"),
            title="Load generation preset",
        )
        if not preset_path:
            return

        try:
            preset = load_preset(Path(preset_path))
            speaker_key, ref_audio, ref_text = resolve_voice_inputs(preset, Path(preset_path))
        except (ValueError, FileNotFoundError, OSError) as e:
            messagebox.showerror("Load preset failed", str(e))
            return

        self.text_box.delete("1.0", END)
        self.text_box.insert("1.0", preset.text)
        self.voice_mode.set(preset.voice_mode)
        self._sync_voice_mode()
        if preset.voice_mode == "speaker" and speaker_key:
            self.speaker_var.set(speaker_key)
        elif ref_audio is not None:
            self.ref_audio_var.set(str(ref_audio))
            self.ref_text.delete("1.0", END)
            self.ref_text.insert("1.0", ref_text or "")
        self._apply_generation_params(preset.generation)
        if preset.output_wav:
            self.out_var.set(preset.output_wav)
        prompt_path = resolve_voice_prompt_path(Path(preset_path), preset.voice_prompt_file)
        if prompt_path is not None:
            self._voice_clone_prompt, stored_key = load_voice_prompt(prompt_path)
            self._voice_cache_key = preset.voice_cache_key or stored_key
            prompt_note = f" (+ voice cache {prompt_path.name})"
        else:
            self._voice_clone_prompt = None
            self._voice_cache_key = None
            prompt_note = " (no voice cache)"
        self._generation_count = preset.generation_count
        self.set_status(f"Loaded preset: {Path(preset_path).name}{prompt_note}")

    def on_save_session(self) -> None:
        if self._voice_clone_prompt is None:
            messagebox.showwarning(
                "No runtime cache",
                "Generate audio at least once after tuning before saving a runtime snapshot.",
            )
            return

        mode = self.voice_mode.get()
        ref_audio = None
        ref_text = None
        speaker_key = None
        if mode == "speaker":
            speaker_key = self.speaker_var.get()
        else:
            ref_audio = Path(self.ref_audio_var.get()).expanduser() if self.ref_audio_var.get().strip() else None
            ref_text = self.ref_text.get("1.0", END).strip()

        try:
            gen_params = self._build_generation_params()
        except ValueError as e:
            messagebox.showwarning("Invalid generation settings", str(e))
            return

        session_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("Runtime snapshot", "*.json"), ("All files", "*.*")],
            initialdir=str(self.root_dir / "runtime_snapshots"),
            title="Save runtime snapshot",
        )
        if not session_path:
            return

        session = capture_session(
            voice_mode=mode,
            speaker_key=speaker_key,
            ref_audio=ref_audio,
            ref_text=ref_text,
            generation=gen_params,
            generation_count=self._generation_count,
            voice_clone_prompt=self._voice_clone_prompt,
            model_dir=self._bootstrap.model_dir if self._bootstrap else None,
            name=Path(session_path).stem,
        )
        save_session_snapshot(
            session,
            Path(session_path),
            ref_audio_src=ref_audio,
            voice_prompt_items=self._voice_clone_prompt,
        )
        self.set_status(f"Saved runtime snapshot: {session_path}")

    def on_load_session(self) -> None:
        session_path = filedialog.askopenfilename(
            filetypes=[("Runtime snapshot", "*.json"), ("All files", "*.*")],
            initialdir=str(self.root_dir / "runtime_snapshots"),
            title="Load runtime snapshot",
        )
        if not session_path:
            return

        try:
            session, prompt_items = load_session_snapshot(Path(session_path))
        except (ValueError, FileNotFoundError, OSError) as e:
            messagebox.showerror("Load runtime failed", str(e))
            return

        self.voice_mode.set(session.voice_mode)
        self._sync_voice_mode()
        if session.voice_mode == "speaker" and session.speaker_key:
            self.speaker_var.set(session.speaker_key)
        elif session.voice_mode == "custom":
            from .runtime_state import resolve_session_ref_audio

            ref_path = resolve_session_ref_audio(session, Path(session_path))
            if ref_path is not None and ref_path.exists():
                self.ref_audio_var.set(str(ref_path))
                self.ref_text.delete("1.0", END)
                self.ref_text.insert("1.0", session.ref_text or "")
        self._apply_generation_params(session.generation)
        self._voice_clone_prompt = prompt_items
        self._voice_cache_key = session.voice_cache_key
        self._generation_count = session.generation_count
        note = " (+ voice cache)" if prompt_items else " (no voice cache)"
        self.set_status(f"Loaded runtime snapshot: {Path(session_path).name}{note}")

    def _on_ref_text_modified(self, _event=None) -> None:
        if self.ref_text.edit_modified():
            self.ref_text.edit_modified(False)
            self._sync_save_builtin_state()

    def _sync_save_builtin_state(self) -> None:
        mode = self.voice_mode.get()
        has_audio = bool(self.ref_audio_var.get().strip())
        has_text = bool(self.ref_text.get("1.0", END).strip())
        state = "normal" if mode == "custom" and has_audio and has_text else "disabled"
        self.save_builtin_btn.configure(state=state)

    def _sync_voice_mode(self) -> None:
        self._invalidate_voice_cache()
        mode = self.voice_mode.get()
        if mode == "speaker":
            self.speaker_combo.configure(state="readonly")
            self.ref_audio_entry.configure(state="disabled")
            self.ref_text.configure(state="disabled")
        else:
            self.speaker_combo.configure(state="disabled")
            self.ref_audio_entry.configure(state="normal")
            self.ref_text.configure(state="normal")
        self._sync_save_builtin_state()

    def on_save_builtin_speaker(self) -> None:
        if not self._bootstrap_done or not self._bootstrap:
            messagebox.showwarning("Not ready", "Please run Bootstrap first.")
            return

        key = self.builtin_key_var.get().strip()
        display_name = self.builtin_name_var.get().strip()
        ref_audio = Path(self.ref_audio_var.get()).expanduser()
        ref_text = self.ref_text.get("1.0", END).strip()

        if self.voice_mode.get() != "custom":
            messagebox.showwarning("Wrong mode", "Switch to Custom reference WAV mode first.")
            return

        exists = key in self._ref_info
        if exists and not messagebox.askyesno(
            "Overwrite speaker?",
            f"Speaker key '{key}' already exists. Overwrite it?",
        ):
            return

        try:
            speaker = add_builtin_speaker(
                self.root_dir,
                key,
                display_name,
                ref_audio,
                ref_text,
                overwrite=exists,
            )
        except ValueError as exc:
            messagebox.showerror("Save failed", str(exc))
            return

        self._ref_info = load_ref_info(self._bootstrap.ref_info_path)
        speakers = list(self._ref_info.keys())
        self.speaker_combo.configure(values=speakers)
        self.speaker_var.set(speaker.key)
        self.voice_mode.set("speaker")
        self._sync_voice_mode()
        self.set_status(f"Saved built-in speaker '{speaker.name}' ({speaker.key}).")

    def on_load_txt(self) -> None:
        p = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not p:
            return
        txt = Path(p).read_text(encoding="utf-8", errors="ignore")
        self.text_box.delete("1.0", END)
        self.text_box.insert("1.0", txt)

    def on_pick_output(self) -> None:
        p = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV audio", "*.wav")])
        if p:
            self.out_var.set(p)

    def on_pick_ref_audio(self) -> None:
        p = filedialog.askopenfilename(filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")])
        if p:
            self.ref_audio_var.set(p)
            self._invalidate_voice_cache()

    def on_open_output_folder(self) -> None:
        out = Path(self.out_var.get()).expanduser()
        folder = out.parent if out.suffix.lower() == ".wav" else Path.cwd()
        try:
            import os

            os.startfile(str(folder))  # type: ignore[attr-defined]
        except Exception:
            messagebox.showerror("Error", f"Could not open folder:\n{folder}")

    def set_status(self, msg: str) -> None:
        self.status_var.set(msg)
        self.update_idletasks()

    def on_bootstrap(self) -> None:
        if self._bootstrap_done:
            self.set_status("Bootstrap already completed.")
            return

        self.bootstrap_btn.configure(state="disabled")
        self.set_status("Bootstrapping: downloading/caching model + speaker assets…")

        def work() -> None:
            try:
                self._bootstrap = bootstrap_all(self.root_dir)
                self._ref_info = load_ref_info(self._bootstrap.ref_info_path)
                speakers = list(self._ref_info.keys())
                self.after(0, lambda: self.speaker_combo.configure(values=speakers))
                self._bootstrap_done = True
                self.after(0, lambda: self.run_btn.configure(state="normal"))
                self.after(0, lambda: self.set_status("Bootstrap done. Ready to generate."))
            except Exception as e:
                tb = traceback.format_exc()
                self.after(0, lambda: messagebox.showerror("Bootstrap failed", f"{e}\n\n{tb}"))
                self.after(0, lambda: self.set_status("Bootstrap failed."))
            finally:
                self.after(0, lambda: self.bootstrap_btn.configure(state="normal"))

        threading.Thread(target=work, daemon=True).start()

    def on_generate(self) -> None:
        if not self._bootstrap_done or not self._bootstrap:
            messagebox.showwarning("Not ready", "Please run Bootstrap first.")
            return

        text = self.text_box.get("1.0", END).strip()
        out = Path(self.out_var.get()).expanduser()

        mode = self.voice_mode.get()
        speaker_key = None
        ref_audio = None
        ref_text = None
        if mode == "speaker":
            speaker_key = self.speaker_var.get()
        else:
            ref_audio = Path(self.ref_audio_var.get()).expanduser() if self.ref_audio_var.get().strip() else None
            ref_text = self.ref_text.get("1.0", END).strip()
            if ref_audio is None:
                messagebox.showwarning("Missing reference audio", "Pick a reference WAV file.")
                return

        try:
            gen_params = self._build_generation_params()
        except ValueError as e:
            messagebox.showwarning("Invalid generation settings", str(e))
            return

        self.run_btn.configure(state="disabled")
        self.set_status("Generating audio… (this can take a while)")

        cached_prompt = self._resolve_cached_prompt(mode, speaker_key, ref_audio, ref_text, gen_params)

        def work() -> None:
            try:
                wav_path, sr, _, prompt_items = synthesize_to_wav(
                    model_dir=self._bootstrap.model_dir,
                    ref_info_path=self._bootstrap.ref_info_path,
                    ref_audio_dir=self._bootstrap.ref_audio_dir,
                    text=text,
                    output_wav=out,
                    speaker_key=speaker_key,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    generation_params=gen_params,
                    voice_clone_prompt=cached_prompt,
                )
                self._voice_clone_prompt = prompt_items
                self._generation_count += 1
                self._voice_cache_key = self._current_voice_key(mode, speaker_key, ref_audio, ref_text, gen_params)
                cache_note = " (reused voice cache)" if cached_prompt is not None else ""
                self.after(0, lambda: self.set_status(f"Done: {wav_path.name} ({sr} Hz){cache_note}"))
            except Exception as e:
                tb = traceback.format_exc()
                self.after(0, lambda: messagebox.showerror("Generation failed", f"{e}\n\n{tb}"))
                self.after(0, lambda: self.set_status("Generation failed."))
            finally:
                self.after(0, lambda: self.run_btn.configure(state="normal"))

        threading.Thread(target=work, daemon=True).start()


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()

