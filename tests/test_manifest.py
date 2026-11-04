"""Manifest contract for wt_genesis."""
from pathlib import Path

from witwin_server.api import testing


def test_manifest_declares_humanoid_plugin_dependency():
    manifest = testing.read_manifest(Path(__file__).resolve().parents[1] / "witwin.toml")

    assert "wt_humanoid" in manifest.plugin_dependencies


def test_manifest_declares_genesis_python_dependency():
    manifest = testing.read_manifest(Path(__file__).resolve().parents[1] / "witwin.toml")

    assert "witwin-genesis>=0.2,<0.3" in manifest.dependencies
