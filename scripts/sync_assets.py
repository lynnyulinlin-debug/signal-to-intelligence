#!/usr/bin/env python3
"""Mirror generated chart assets into the VitePress public directory."""

from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets"
DESTINATION = ROOT / "docs" / "public" / "assets"


def sync_assets() -> None:
    if not SOURCE.is_dir():
        raise SystemExit(f"Source asset directory does not exist: {SOURCE}")

    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)

    shutil.copytree(SOURCE, DESTINATION)
    print(f"Synced {SOURCE.relative_to(ROOT)}/ -> {DESTINATION.relative_to(ROOT)}/")


if __name__ == "__main__":
    sync_assets()
