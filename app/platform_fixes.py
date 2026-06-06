from __future__ import annotations

import logging
import shutil
import sys
import types
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


def apply_windows_asyncio_fix() -> None:
    """Avoid noisy ProactorEventLoop disconnect tracebacks on Windows."""
    if not sys.platform.startswith("win"):
        return
    import asyncio

    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except AttributeError:
        pass


def sox_status_line() -> str:
    if sox_binary_available():
        return ""
    return (
        "SoX CLI not on PATH — using built-in peak normalization for reference audio. "
        "Install SoX for upstream-identical voice encoding: "
        "https://sourceforge.net/projects/sox/files/sox/ "
        "(add install folder to PATH, then restart)."
    )


def apply_platform_fixes() -> None:
    global _FIXES_APPLIED
    if _FIXES_APPLIED:
        return
    apply_windows_asyncio_fix()
    apply_sox_compat()
    _FIXES_APPLIED = True
