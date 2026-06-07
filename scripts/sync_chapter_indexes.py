#!/usr/bin/env python3
"""Generate VitePress chapter index pages from chapter README files."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHAPTER_DIRS = [
    ROOT / "docs" / "03_deep_learning_fast",
    ROOT / "docs" / "04_transformer",
]
HEADER = "<!-- AUTO-GENERATED from README.md. Do not edit index.md directly. -->\n\n"


def sync_index(chapter_dir: Path) -> None:
    readme = chapter_dir / "README.md"
    index = chapter_dir / "index.md"

    if not readme.is_file():
        raise SystemExit(f"Missing README: {readme.relative_to(ROOT)}")

    content = readme.read_text(encoding="utf-8")
    index.write_text(HEADER + content, encoding="utf-8")
    print(f"Synced {readme.relative_to(ROOT)} -> {index.relative_to(ROOT)}")


def main() -> None:
    for chapter_dir in CHAPTER_DIRS:
        sync_index(chapter_dir)


if __name__ == "__main__":
    main()
