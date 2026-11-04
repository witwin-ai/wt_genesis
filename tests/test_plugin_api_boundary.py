"""This Plugin's Python code imports only the public Plugin API (D16).

Allowed, from dev_docs/everything-is-a-plugin-plan.md section 7 in witwin-studio:

- A: ``witwin.studio`` and anything that is not Studio or another Plugin;
- B: ``witwin_server.api`` (including ``from witwin_server import api``) and
  ``witwin_server.api.testing``;
- C: another first-party Plugin's ``<package>.api`` module, or its package
  itself for its import-time registrations, only when this Plugin lists that
  Plugin in ``pluginDependencies``.

Every other ``witwin_server`` module, and every other module of another
Plugin, fails this test: Studio's internals may move at any time. The test
reads the source with ``ast`` and never imports it, so it runs without Studio.

The canonical copy is ``server/scripts/plugin_api_boundary.py`` in
witwin-studio; each Plugin repo keeps an identical copy as
``tests/test_plugin_api_boundary.py``. Studio's
``tests/test_backend_architecture.py`` checks the copies are identical.
Run by hand: ``python tests/test_plugin_api_boundary.py [<plugin root> ...]``.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]

# First-party Plugin id -> Python package name.
PLUGIN_PACKAGES = {
    "acme.roomgen": "wt_room_generator",
    "witwin.channel": "wt_channel",
    "witwin.genesis": "wt_genesis",
    "witwin.home_simulation": "wt_home_simulation",
    "witwin.maxwell": "wt_maxwell",
    "witwin.motion_matching": "witwin_motion_matching",
    "witwin.node_editor": "wt_node_editor",
    "witwin.radar": "wt_radar",
    "witwin.sionna": "wt_sionna",
    "witwin.splatting": "wt_splatting",
    "witwin.streetmap": "wt_streetmap",
    "witwin.ui_profiler": "wt_ui_profiler",
    "witwin.volume": "wt_volume",
    "wt_humanoid": "wt_humanoid",
    "wt_optical": "wt_optical",
}
ALLOWED_STUDIO = ("witwin_server.api", "witwin_server.api.testing")
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", "build", "dist", ".pytest_cache",
             ".pytest-tmp", ".witwin", "site-packages", "web"}


def _manifest(root: Path) -> dict:
    return tomllib.loads((root / "witwin.toml").read_text(encoding="utf-8"))


def _own_package(manifest: dict, root: Path) -> str:
    return PLUGIN_PACKAGES.get(manifest.get("id", ""), root.name)


def _dependency_packages(manifest: dict) -> set[str]:
    out = set()
    for dep in manifest.get("pluginDependencies", []) or []:
        dep_id = dep if isinstance(dep, str) else (dep or {}).get("id", "")
        if dep_id in PLUGIN_PACKAGES:
            out.add(PLUGIN_PACKAGES[dep_id])
    return out


def _imports(tree: ast.AST):
    """``(line, module)`` for every absolute import, ``import_module("...")`` and ``importorskip("...")``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "witwin_server":
                for alias in node.names:
                    yield node.lineno, f"witwin_server.{alias.name}" if alias.name == "api" else "witwin_server"
            else:
                yield node.lineno, node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name in {"import_module", "importorskip", "__import__"} and isinstance(node.args[0].value, str):
                yield node.lineno, node.args[0].value


def violations(root: Path = ROOT) -> list[str]:
    """``file:line  module (reason)`` for every import outside the Plugin API under ``root``."""
    root = Path(root)
    manifest = _manifest(root)
    own = _own_package(manifest, root)
    deps = _dependency_packages(manifest)
    others = set(PLUGIN_PACKAGES.values()) - {own}
    found = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts[:-1]):
            continue
        tree = ast.parse(path.read_bytes(), filename=str(path))
        for line, module in _imports(tree):
            top = module.split(".")[0]
            if top == "witwin_server":
                if module not in ALLOWED_STUDIO:
                    found.append(f"{rel.as_posix()}:{line}  {module} (use witwin_server.api)")
            elif top in others:
                if top not in deps:
                    found.append(f"{rel.as_posix()}:{line}  {module} (Plugin not in pluginDependencies)")
                elif module not in (top, f"{top}.api"):
                    found.append(f"{rel.as_posix()}:{line}  {module} (use {top}.api)")
    return found


def test_imports_stay_inside_the_plugin_api() -> None:
    found = violations()
    assert not found, "imports outside the Plugin API:\n" + "\n".join(found)


if __name__ == "__main__":
    roots = [Path(arg) for arg in sys.argv[1:]] or [ROOT]
    lines = [f"{root.name}/{line}" if len(roots) > 1 else line for root in roots for line in violations(root)]
    print("\n".join(lines) or "ok")
    sys.exit(1 if lines else 0)
