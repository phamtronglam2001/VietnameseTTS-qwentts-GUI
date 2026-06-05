from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppPaths:
    root: Path

    @property
    def cache_dir(self) -> Path:
        return self.root / ".hf_cache"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def model_local_dir(self) -> Path:
        # Fully local snapshot of the HF repo.
        return self.models_dir / "g-group-ai-lab__gwen-tts-0.6B"

    @property
    def assets_dir(self) -> Path:
        return self.root / "assets"

    @property
    def ref_info_json(self) -> Path:
        return self.assets_dir / "ref_info.json"

    @property
    def ref_audio_dir(self) -> Path:
        return self.assets_dir / "ref_audio"

