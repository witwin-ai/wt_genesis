"""Test harness for the wt_genesis plugin."""
from __future__ import annotations

import sys
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[1]
PLUGINS_DIR = Path(__file__).resolve().parents[2]
CODE_DIR = Path(__file__).resolve().parents[3]
STUDIO_SERVER_DIR = CODE_DIR / "witwin-studio" / "server"
PLATFORM_CORE_DIR = CODE_DIR / "witwin-platform" / "core"
PLATFORM_GENESIS_DIR = CODE_DIR / "witwin-platform" / "genesis"

for path in (PLUGINS_DIR, STUDIO_SERVER_DIR, PLATFORM_CORE_DIR, PLATFORM_GENESIS_DIR):
    if not path.exists():
        continue
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)