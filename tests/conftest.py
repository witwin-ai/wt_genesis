"""Test harness for the wt_genesis plugin."""
from __future__ import annotations

import sys
from pathlib import Path


PLUGINS_DIR = Path(__file__).resolve().parents[2]
REPO_DIR = Path(__file__).resolve().parents[3]
SERVER_DIR = REPO_DIR / "server"

for path in (PLUGINS_DIR, SERVER_DIR):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
