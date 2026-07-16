"""Manifest contract for wt_genesis."""
from pathlib import Path

from witwin_server.plugins.loader import PluginManifest


def test_manifest_declares_humanoid_plugin_dependency():
    manifest = PluginManifest.from_file(Path(__file__).resolve().parents[1] / "witwin.toml")

    assert "wt_humanoid" in manifest.plugin_dependencies


def test_manifest_declares_genesis_python_dependency():
    manifest = PluginManifest.from_file(Path(__file__).resolve().parents[1] / "witwin.toml")

    assert "witwin-genesis>=0.1,<0.2" in manifest.dependencies
