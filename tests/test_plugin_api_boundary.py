"""This Plugin's Python code imports only the public Plugin API (D16).

Allowed, from dev_docs/everything-is-a-plugin-plan.md section 7 in witwin-studio:

- A: ``witwin.studio`` and anything that is not Studio or another Plugin;
- B: ``witwin_server.api`` (including ``from witwin_server import api``) and
  ``witwin_server.api.testing``;
- C: another first-party Plugin's ``<package>.api`` module, or its package
  itself for its import-time registrations, only when this Plugin lists that
  Plugin in ``pluginDependencies``. For a system Plugin ``studio.<name>``
  that module is ``witwin_server.plugins_builtin.<name>.api`` (for example
  ``studio.em``, 2026-11-03).

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

# A first-party Plugin's Python package is ``wt_<name>`` for the id
# ``<vendor>.<name>`` (pip distribution ``wt-<name>``), or the id itself when
# it has no dot (``wt_humanoid``). Only the Plugins whose package breaks that
# rule are listed here; a repo's own package comes from its pyproject.toml
# ``witwin.plugins`` entry point or its manifest ``main`` first.
PACKAGE_EXCEPTIONS = {
    "acme.roomgen": "wt_room_generator",
    "witwin.motion_matching": "witwin_motion_matching",
}
# Every top-level module with this prefix is a first-party Plugin package.
PLUGIN_PACKAGE_PREFIX = "wt_"
ALLOWED_STUDIO = ("witwin_server.api", "witwin_server.api.testing")
# A system Plugin ``studio.<name>`` publishes its layer-C API here; the folder
# under plugins_builtin is the id without "studio.".
SYSTEM_PLUGIN_API = "witwin_server.plugins_builtin.{}.api"
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", "build", "dist", ".pytest_cache",
             ".pytest-tmp", ".witwin", "site-packages", "web"}


def _manifest(root: Path) -> dict:
    return tomllib.loads((root / "witwin.toml").read_text(encoding="utf-8"))


def package_for(plugin_id: str) -> str:
    """The Python package of the first-party Plugin ``plugin_id``."""
    if plugin_id in PACKAGE_EXCEPTIONS:
        return PACKAGE_EXCEPTIONS[plugin_id]
    if "." not in plugin_id:
        return plugin_id
    return PLUGIN_PACKAGE_PREFIX + plugin_id.rpartition(".")[2]


def _own_package(manifest: dict, root: Path) -> str:
    plugin_id = manifest.get("id", "")
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        entry_points = tomllib.loads(pyproject.read_text(encoding="utf-8")).get("project", {}).get(
            "entry-points", {}).get("witwin.plugins", {})
        if entry_points.get(plugin_id):
            return str(entry_points[plugin_id]).split(":")[0].split(".")[0]
    if manifest.get("main"):
        return str(manifest["main"]).split(".")[0]
    return package_for(plugin_id) if plugin_id else root.name


def _is_plugin_package(top: str) -> bool:
    return top.startswith(PLUGIN_PACKAGE_PREFIX) or top in PACKAGE_EXCEPTIONS.values()


def _dependency_ids(manifest: dict) -> set[str]:
    out = set()
    for dep in manifest.get("pluginDependencies", []) or []:
        out.add(dep if isinstance(dep, str) else (dep or {}).get("id", ""))
    return out


def _dependency_packages(manifest: dict) -> set[str]:
    return {package_for(dep_id) for dep_id in _dependency_ids(manifest)
            if dep_id and not dep_id.startswith("studio.")}


def _system_apis(manifest: dict) -> set[str]:
    """``witwin_server.plugins_builtin.<name>.api`` of every ``studio.<name>`` dependency."""
    return {SYSTEM_PLUGIN_API.format(dep_id[len("studio."):])
            for dep_id in _dependency_ids(manifest) if dep_id.startswith("studio.")}


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
    allowed_studio = set(ALLOWED_STUDIO) | _system_apis(manifest)
    found = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts[:-1]):
            continue
        tree = ast.parse(path.read_bytes(), filename=str(path))
        for line, module in _imports(tree):
            top = module.split(".")[0]
            if top == "witwin_server":
                if module not in allowed_studio:
                    found.append(f"{rel.as_posix()}:{line}  {module} (use witwin_server.api)")
            elif top != own and _is_plugin_package(top):
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
