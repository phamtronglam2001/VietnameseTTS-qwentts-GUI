from __future__ import annotations

import threading
import traceback
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, TOP, X, Y, filedialog, messagebox
import tkinter as tk
from tkinter import ttk

from .bootstrap import bootstrap_all
from .engine import GenerationParams, load_ref_info, synthesize_to_wav
from .ime_support import configure_unicode_fonts, enable_windows_ime, setup_text_widget


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Gwen-TTS Vietnamese GUI (Local)")
        self.geometry("900x650")

        self.root_dir = Path(__file__).resolve().parents[1]
        self._bootstrap_done = False
        self._bootstrap = None
        self._ref_info = {}
        self._text_font = configure_unicode_fonts(self)
        enable_windows_ime(self)

        self._build_ui()

    def _build_ui(self) -> None:
        pad = 10

        top = ttk.Frame(self, padding=pad)
        top.pack(side=TOP, fill=X)

        self.status_var = tk.StringVar(value="Ready. Click 'Bootstrap (Download/Cache)' first.")
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

        self.temperature_var = tk.DoubleVar(value=0.3)
        self._add_spin(adv, "Temperature", self.temperature_var, 0.0, 2.0, 0.05)

        self.top_k_var = tk.IntVar(value=20)
        self._add_spin(adv, "Top-k", self.top_k_var, 1, 100, 1, is_int=True)

        self.top_p_var = tk.DoubleVar(value=0.9)
        self._add_spin(adv, "Top-p", self.top_p_var, 0.0, 1.0, 0.05)

        self.repetition_penalty_var = tk.DoubleVar(value=2.0)
        self._add_spin(adv, "Repetition penalty", self.repetition_penalty_var, 1.0, 3.0, 0.05)

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
            temperature=self.temperature_var.get(),
            top_k=self.top_k_var.get(),
            top_p=self.top_p_var.get(),
            repetition_penalty=self.repetition_penalty_var.get(),
        )

    def _sync_voice_mode(self) -> None:
        mode = self.voice_mode.get()
        if mode == "speaker":
            self.speaker_combo.configure(state="readonly")
            self.ref_audio_entry.configure(state="disabled")
            self.ref_text.configure(state="disabled")
        else:
            self.speaker_combo.configure(state="disabled")
            self.ref_audio_entry.configure(state="normal")
            self.ref_text.configure(state="normal")

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

        def work() -> None:
            try:
                wav_path, sr, _ = synthesize_to_wav(
                    model_dir=self._bootstrap.model_dir,
                    ref_info_path=self._bootstrap.ref_info_path,
                    ref_audio_dir=self._bootstrap.ref_audio_dir,
                    text=text,
                    output_wav=out,
                    speaker_key=speaker_key,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    generation_params=gen_params,
                )
                self.after(0, lambda: self.set_status(f"Done: {wav_path.name} ({sr} Hz)"))
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

