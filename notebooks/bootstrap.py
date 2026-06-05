"""Shared notebook bootstrap for interactive chapters.

Run this after switching to the repository root. It exposes the shared
``load_code_module`` helper without repeating import boilerplate in every
notebook.
"""

from notebooks.project import load_code_module

