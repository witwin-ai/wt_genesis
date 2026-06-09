"""Lazy access to the vendored MDM Tiny runtime."""
from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any


PLUGIN_DIR = Path(__file__).resolve().parents[1]
VENDOR_DIR = PLUGIN_DIR / "vendor" / "mdm_tiny"
DATA_DIR = VENDOR_DIR / "data"
DEFAULT_CHECKPOINT = DATA_DIR / "checkpoints" / "humanml_enc_512_50steps" / "model000750000.pt"
DEFAULT_STATS_DIR = DATA_DIR / "humanml"

_MDM_MODULES = {
    "mdm_text_to_smpl",
    "api",
    "runtime",
    "model",
    "diffusion",
    "humanml",
    "smplify",
}


def generate_smpl_motion(prompt: str, **kwargs: Any) -> dict[str, Any]:
    """Run vendored MDM Tiny inference for one prompt."""
    if not VENDOR_DIR.exists():
        raise FileNotFoundError(f"MDM Tiny vendor directory is missing: {VENDOR_DIR}")
    if not DEFAULT_CHECKPOINT.exists():
        raise FileNotFoundError(f"MDM Tiny checkpoint is missing: {DEFAULT_CHECKPOINT}")

    with _vendor_import_path():
        module = importlib.import_module("mdm_text_to_smpl")
        return module.generate_smpl_motion(
            prompt,
            model_path=DEFAULT_CHECKPOINT,
            stats_dir=DEFAULT_STATS_DIR,
            **kwargs,
        )


@contextmanager
def _vendor_import_path():
    vendor = str(VENDOR_DIR)
    original_path = list(sys.path)
    restored_modules = {
        name: sys.modules[name]
        for name in _MDM_MODULES
        if name in sys.modules and not _module_is_under(sys.modules[name], VENDOR_DIR)
    }
    for name in restored_modules:
        sys.modules.pop(name, None)
    if vendor in sys.path:
        sys.path.remove(vendor)
    sys.path.insert(0, vendor)
    try:
        yield
    finally:
        sys.path[:] = original_path
        for name, module in restored_modules.items():
            sys.modules[name] = module


def _module_is_under(module: object, root: Path) -> bool:
    file_name = getattr(module, "__file__", None)
    if not file_name:
        return False
    try:
        Path(file_name).resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False
