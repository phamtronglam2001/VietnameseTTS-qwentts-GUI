from __future__ import annotations

import contextlib
import logging
import os
import shutil
import sys
import types
import warnings
from pathlib import Path
from typing import Optional

_SOX_SHIM_INSTALLED = False
_FIXES_APPLIED = False


def sox_binary_available() -> bool:
    return shutil.which("sox") is not None


def _normalize_peak_db(audio, db_level: float):
    import numpy as np

    samples = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(samples)))
    if peak < 1e-8:
        return samples
    target = 10 ** (db_level / 20.0)
    return (samples * (target / peak)).astype(np.float32)


def _install_sox_shim() -> None:
    """Provide a minimal ``sox`` module when the SoX CLI is not on PATH."""
    global _SOX_SHIM_INSTALLED
    if _SOX_SHIM_INSTALLED or "sox" in sys.modules:
        return

    class Transformer:
        def __init__(self) -> None:
            self._db_level: Optional[float] = None

        def norm(self, db_level: float = -6):
            self._db_level = db_level
            return self

        def build_array(self, input_array, sample_rate_in: int):
            del sample_rate_in
            return _normalize_peak_db(input_array, self._db_level if self._db_level is not None else -6)

    class SoxError(Exception):
        pass

    mod = types.ModuleType("sox")
    mod.NO_SOX = True
    mod.Transformer = Transformer
    mod.SoxError = SoxError
    mod.SoxiError = SoxError
    sys.modules["sox"] = mod
    _SOX_SHIM_INSTALLED = True


def apply_sox_compat() -> bool:
    """
    Ensure ``qwen_tts`` can import ``sox`` without a system SoX install.

    Returns True when the native SoX CLI is available; False when the shim is used.
    """
    if sox_binary_available():
        return True
    _install_sox_shim()
    logging.getLogger("sox").setLevel(logging.ERROR)
    return False


def apply_hf_env() -> None:
    """Use ``HF_HOME`` / ``HF_HUB_CACHE`` and drop deprecated ``TRANSFORMERS_CACHE``."""
    legacy_cache = os.environ.get("TRANSFORMERS_CACHE")
    hf_home = os.environ.get("HF_HOME")

    if not hf_home and legacy_cache:
        os.environ["HF_HOME"] = str(Path(legacy_cache).parent)

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        root = Path(hf_home)
        root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HUB_CACHE", str(root / "hub"))

    # transformers>=4.46 warns when TRANSFORMERS_CACHE is set; HF_HOME is canonical.
    if legacy_cache:
        os.environ.pop("TRANSFORMERS_CACHE", None)


def apply_warning_filters() -> None:
    warnings.filterwarnings(
        "ignore",
        message=".*TRANSFORMERS_CACHE.*",
        category=FutureWarning,
    )


def apply_logging_quiet() -> None:
    """Reduce noisy but harmless transformers / asyncio messages during TTS runs."""
    for name in (
        "transformers",
        "transformers.generation",
        "transformers.generation.utils",
        "transformers.generation.configuration_utils",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)


def _asyncio_exception_handler(loop, context) -> None:
    exc = context.get("exception")
    if isinstance(exc, ConnectionResetError):
        return
    loop.default_exception_handler(context)


def apply_windows_asyncio_fix() -> None:
    """Avoid noisy ProactorEventLoop disconnect tracebacks on Windows."""
    if not sys.platform.startswith("win"):
        return
    import asyncio

    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except AttributeError:
        pass

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.set_exception_handler(_asyncio_exception_handler)


def sox_status_line() -> str:
    if sox_binary_available():
        return ""
    return (
        "SoX CLI not on PATH — using built-in peak normalization for reference audio. "
        "Install SoX for upstream-identical voice encoding: "
        "https://sourceforge.net/projects/sox/files/sox/ "
        "(add install folder to PATH, then restart)."
    )


@contextlib.contextmanager
def suppress_qwen_import_noise():
    """
    Mute one-shot ``qwen_tts`` import prints when flash-attn is absent.

    The upstream package prints to stdout on import; sdpa remains the fallback.
    """
    try:
        import flash_attn  # noqa: F401

        yield
        return
    except Exception:
        pass

    devnull = open(os.devnull, "w", encoding="utf-8")
    saved = sys.stdout
    sys.stdout = devnull
    try:
        yield
    finally:
        sys.stdout = saved
        devnull.close()


def apply_platform_fixes() -> None:
    global _FIXES_APPLIED
    if _FIXES_APPLIED:
        return
    apply_hf_env()
    apply_warning_filters()
    apply_windows_asyncio_fix()
    apply_sox_compat()
    apply_logging_quiet()
    _FIXES_APPLIED = True
