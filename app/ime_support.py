"""Windows IME helpers so Vietnamese Telex/VNI input works in tkinter Text/Entry."""

from __future__ import annotations

import platform
import tkinter as tk
import tkinter.font as tkfont
from contextlib import suppress
VIET_FONT_FAMILY = "Segoe UI"
VIET_FONT_SIZE = 10
IME_KEYCODE = 229

if platform.system() == "Windows":
    import ctypes
    from ctypes import c_byte, c_int, c_long, c_wchar

    LOGPIXELSY = c_int(90)
    DEFAULT_CHARSET = c_byte(1)
    LF_FACESIZE = 32
    POINTS_PER_INCH = 72

    FW_MAPPING = {
        tkfont.NORMAL: c_long(400),
        tkfont.BOLD: c_long(700),
    }

    class LOGFONTW(ctypes.Structure):
        _fields_ = [
            ("lfHeight", c_long),
            ("lfWidth", c_long),
            ("lfEscapement", c_long),
            ("lfOrientation", c_long),
            ("lfWeight", c_long),
            ("lfItalic", c_byte),
            ("lfUnderline", c_byte),
            ("lfStrikeOut", c_byte),
            ("lfCharSet", c_byte),
            ("lfOutPrecision", c_byte),
            ("lfClipPrecision", c_byte),
            ("lfQuality", c_byte),
            ("lfPitchAndFamily", c_byte),
            ("lfFaceName", c_wchar * LF_FACESIZE),
        ]


def viet_text_font(root: tk.Misc) -> tkfont.Font:
    return tkfont.Font(root=root, family=VIET_FONT_FAMILY, size=VIET_FONT_SIZE)


def configure_unicode_fonts(root: tk.Misc) -> tkfont.Font:
    """Use a Vietnamese-capable font for built-in and text-entry widgets."""
    text_font = viet_text_font(root)
    for name in tkfont.names(root):
        with suppress(tk.TclError):
            tkfont.nametofont(name, root=root).configure(
                family=VIET_FONT_FAMILY,
                size=VIET_FONT_SIZE,
            )
    return text_font


def _configure_dpi() -> None:
    if platform.system() != "Windows":
        return
    import ctypes

    with suppress(AttributeError):
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    with suppress(AttributeError):
        ctypes.windll.user32.SetProcessDPIAware()


def enable_windows_ime(root: tk.Misc) -> None:
    """Sync IME composition font with tkinter text widgets on Windows."""
    if platform.system() != "Windows":
        return

    _configure_dpi()

    user32 = ctypes.WinDLL("user32")
    imm32 = ctypes.WinDLL("imm32")
    gdi32 = ctypes.WinDLL("gdi32")
    ime_active = {"value": False}

    def set_ime_font(event: tk.Event) -> None:
        widget = getattr(event, "widget", None)
        if widget is None:
            return

        if event.keycode == IME_KEYCODE and not ime_active["value"]:
            ime_active["value"] = True
            h_wnd = user32.GetForegroundWindow()
            h_dc = user32.GetDC(h_wnd)
            h_imc = imm32.ImmGetContext(h_wnd)
            if not h_imc:
                return

            font_spec = widget.cget("font")
            font = tkfont.nametofont(font_spec, root=widget)
            lplf = LOGFONTW()
            size = font.cget("size")
            if size > 0:
                lplf.lfHeight = c_long(
                    -round(size * gdi32.GetDeviceCaps(h_dc, LOGPIXELSY) / POINTS_PER_INCH)
                )
            else:
                lplf.lfHeight = c_long(size)
            lplf.lfWidth = c_long(0)
            lplf.lfEscapement = c_long(0)
            lplf.lfOrientation = c_long(0)
            lplf.lfWeight = FW_MAPPING.get(font.cget("weight"), c_long(400))
            lplf.lfItalic = c_byte(int(font.cget("slant") == "italic"))
            lplf.lfUnderline = c_byte(int(font.cget("underline")))
            lplf.lfStrikeOut = c_byte(int(font.cget("overstrike")))
            lplf.lfCharSet = DEFAULT_CHARSET
            lplf.lfOutPrecision = c_byte(0)
            lplf.lfClipPrecision = c_byte(0)
            lplf.lfQuality = c_byte(0)
            lplf.lfPitchAndFamily = c_byte(0)
            lplf.lfFaceName = font.cget("family")
            imm32.ImmSetCompositionFontW(h_imc, ctypes.byref(lplf))
            imm32.ImmReleaseContext(h_wnd, h_imc)
            user32.ReleaseDC(h_wnd, h_dc)
            return

        if event.char and ime_active["value"]:
            ime_active["value"] = False

    root.bind_all("<Key>", set_ime_font, add=True)


def setup_text_widget(widget: tk.Text, font: tkfont.Font) -> None:
    widget.configure(
        font=font,
        undo=True,
        maxundo=-1,
        autoseparators=True,
        insertofftime=0,
        insertontime=600,
        exportselection=True,
    )
