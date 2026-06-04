"""Notebook helpers for reusing implementation code from the repository."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_code_module(relative_path):
    """Load a script under code/ without relying on package imports."""
    module_path = PROJECT_ROOT / relative_path
    module_name = relative_path.replace("/", "_").replace(".py", "")
    spec = spec_from_file_location(module_name, module_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
