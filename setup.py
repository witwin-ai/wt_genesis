"""Build this Plugin repository as one import package.

The repository root is the package itself: ``witwin.toml`` and ``__init__.py``
sit next to this file. setuptools cannot discover that layout, so this file
maps the import name from the ``witwin.plugins`` entry point in
``pyproject.toml`` onto ``.``. It also:

- ships every runtime file (web assets, skills, examples, models) as package
  data, so a wheel install has the same Plugin root as a checkout;
- ships each ``[[contributes.templates]]`` payload exactly as its content
  manifest lists it;
- takes the version from ``witwin.toml``, so the two files cannot drift;
- fails the build when ``witwin.toml`` names a Python dependency that
  ``pyproject.toml`` does not declare, or when the entry point name is not the
  manifest ``id``. Studio reads the manifest; pip and uv read pyproject.

The same file is copied into every Plugin repo; keep the copies identical.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as _build_py

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent

# Never part of the Plugin root at runtime.
EXCLUDED_DIRS = {
    ".git", ".github", ".venv", "venv", "__pycache__", ".pytest_cache", ".pytest_tmp",
    "node_modules", "build", "dist", "tests", ".mypy_cache", ".ruff_cache",
}
EXCLUDED_FILES = {"setup.py", ".gitignore", ".gitattributes"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".egg-info"}


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    return _normalize(match.group(1)) if match else ""


def _read_toml(name: str) -> dict:
    with open(ROOT / name, "rb") as handle:
        return tomllib.load(handle)


def _plugin_entry_point(pyproject: dict) -> tuple[str, str]:
    entry_points = pyproject["project"].get("entry-points", {}).get("witwin.plugins", {})
    if len(entry_points) != 1:
        raise SystemExit("pyproject.toml must declare exactly one 'witwin.plugins' entry point")
    plugin_id, target = next(iter(entry_points.items()))
    return plugin_id, target.split(":", 1)[0].strip()


def _check_manifest(manifest: dict, pyproject: dict, plugin_id: str) -> None:
    if manifest.get("id") != plugin_id:
        raise SystemExit(
            f"witwin.plugins entry point '{plugin_id}' must equal the witwin.toml id '{manifest.get('id')}'"
        )
    project = pyproject["project"]
    declared = {_requirement_name(item) for item in project.get("dependencies", [])}
    for extra in project.get("optional-dependencies", {}).values():
        declared.update(_requirement_name(item) for item in extra)
    missing = [
        item for item in manifest.get("dependencies", []) or []
        if _requirement_name(item) not in declared
    ]
    if missing:
        raise SystemExit(
            "witwin.toml dependencies missing from pyproject.toml "
            f"(dependencies or optional-dependencies): {missing}"
        )


def _template_files(manifest: dict) -> tuple[set[Path], list[str]]:
    """Project template payloads, file for file as their content manifest lists them.

    Studio checks every byte of a template against that manifest, so a payload
    must not lose files to the exclusions below (a template's own ``tests/``)
    nor gain files the manifest does not list (``__pycache__`` from a local run).
    """
    roots: set[Path] = set()
    files: list[str] = []
    for item in manifest.get("contributes", {}).get("templates", []) or []:
        template_dir = ROOT / item["dir"]
        content = template_dir / item["manifest"]
        with open(content, "rb") as handle:
            listed = json.load(handle)["files"]
        roots.add(template_dir)
        files.append(content.relative_to(ROOT).as_posix())
        files.extend((template_dir / entry["path"]).relative_to(ROOT).as_posix() for entry in listed)
    return roots, files


def _package_data(package_dirs: set[Path], template_roots: set[Path]) -> list[str]:
    """Every runtime file that is not a module of a found package."""
    files: list[str] = []
    for current, dirs, names in os.walk(ROOT):
        here = Path(current)
        dirs[:] = [
            d for d in dirs
            if d not in EXCLUDED_DIRS and not d.endswith(".egg-info") and here / d not in template_roots
        ]
        for name in names:
            path = here / name
            if name in EXCLUDED_FILES and here == ROOT:
                continue
            if path.suffix in EXCLUDED_SUFFIXES:
                continue
            if path.suffix == ".py" and here in package_dirs:
                continue
            files.append(path.relative_to(ROOT).as_posix())
    return files


class build_py(_build_py):
    """Leave this build script out of the installed package."""

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        if package == PACKAGE:
            modules = [item for item in modules if item[1] != "setup"]
        return modules


PYPROJECT = _read_toml("pyproject.toml")
MANIFEST = _read_toml("witwin.toml")
PLUGIN_ID, PACKAGE = _plugin_entry_point(PYPROJECT)
_check_manifest(MANIFEST, PYPROJECT, PLUGIN_ID)

SUBPACKAGES = find_packages(
    where=str(ROOT),
    exclude=["tests", "tests.*", "*.tests", "*.tests.*", "build*", "dist*", "node_modules*"],
)
PACKAGES = [PACKAGE, *(f"{PACKAGE}.{name}" for name in SUBPACKAGES)]
PACKAGE_DIRS = {ROOT, *(ROOT.joinpath(*name.split(".")) for name in SUBPACKAGES)}
TEMPLATE_ROOTS, TEMPLATE_FILES = _template_files(MANIFEST)

setup(
    version=str(MANIFEST["version"]),
    packages=PACKAGES,
    package_dir={PACKAGE: "."},
    package_data={PACKAGE: _package_data(PACKAGE_DIRS, TEMPLATE_ROOTS) + TEMPLATE_FILES},
    include_package_data=False,
    cmdclass={"build_py": build_py},
)
