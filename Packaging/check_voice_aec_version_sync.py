#!/usr/bin/env python3
"""Fail closed when the app and native voice-AEC versions drift."""

from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path
import re
import sys
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPANION_NAME = "tldw-voice-aec"
PYBIND11_VERSION = "3.1.0"
PYBIND11_REQUIREMENT = f"pybind11=={PYBIND11_VERSION}"
PYBIND11_LICENSE_SHA256 = (
    "83965b843b98f670d3a85bd041ed4b372c8ec50d7b4a5995a83ac697ba675dcb"
)


def _normalized_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _project(path: Path) -> dict[str, object]:
    return tomllib.loads(path.read_text(encoding="utf-8"))["project"]


def _pyproject(path: Path) -> dict[str, object]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def check_version_sync(repo_root: Path = REPO_ROOT) -> list[str]:
    """Return version-lock violations without importing either package.

    Args:
        repo_root: Checkout containing the application and companion pyprojects.

    Returns:
        Human-readable violations. An empty list means the lock is exact.
    """
    errors: list[str] = []
    try:
        app = _project(repo_root / "pyproject.toml")
        companion_root = repo_root / "native" / "voice_aec"
        companion_metadata = _pyproject(companion_root / "pyproject.toml")
        companion = companion_metadata["project"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as error:
        return [f"cannot read project metadata: {error}"]

    app_version = app.get("version")
    companion_version = companion.get("version")
    companion_name = companion.get("name")
    if not isinstance(app_version, str) or not app_version:
        errors.append("application project.version must be a non-empty string")
    if _normalized_name(str(companion_name)) != COMPANION_NAME:
        errors.append(f"companion project.name must be exactly {COMPANION_NAME}")
    if companion_version != app_version:
        errors.append(
            "companion project.version must exactly equal application project.version: "
            f"{companion_version!r} != {app_version!r}"
        )

    try:
        module = ast.parse(
            (repo_root / "tldw_chatbook/__init__.py").read_text(encoding="utf-8")
        )
        versions = {
            target.id: ast.literal_eval(node.value)
            for node in module.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
            and target.id in {"__version__", "VERSION_TUPLE"}
        }
        if versions.get("__version__") != app_version:
            errors.append("public __version__ must equal application project.version")
        if versions.get("VERSION_TUPLE") != tuple(
            int(part) for part in str(app_version).split(".")
        ):
            errors.append("public VERSION_TUPLE must equal application project.version")
    except (OSError, SyntaxError, ValueError, TypeError) as error:
        errors.append(f"cannot read public application version metadata: {error}")

    build_system = companion_metadata.get("build-system")
    build_requirements = (
        build_system.get("requires") if isinstance(build_system, dict) else None
    )
    pybind11_requirements = (
        [
            requirement
            for requirement in build_requirements
            if isinstance(requirement, str)
            and _normalized_name(re.split(r"[<>=!~;\[\s]", requirement, 1)[0])
            == "pybind11"
        ]
        if isinstance(build_requirements, list)
        else []
    )
    if pybind11_requirements != [PYBIND11_REQUIREMENT]:
        errors.append(
            "companion build-system must contain exactly one unmarked exact "
            f"pybind11 pin {PYBIND11_REQUIREMENT!r}; found {pybind11_requirements!r}"
        )

    license_path = companion_root / "PYBIND11_LICENSE.txt"
    try:
        license_digest = hashlib.sha256(license_path.read_bytes()).hexdigest()
    except OSError as error:
        errors.append(f"cannot read reviewed pybind11 license: {error}")
    else:
        if license_digest != PYBIND11_LICENSE_SHA256:
            errors.append("reviewed pybind11 license does not match pybind11 v3.1.0")
    license_files = companion.get("license-files")
    if (
        not isinstance(license_files, list)
        or "PYBIND11_LICENSE.txt" not in license_files
    ):
        errors.append(
            "companion project.license-files must include PYBIND11_LICENSE.txt"
        )

    extras = app.get("optional-dependencies")
    speech = extras.get("speech_recording") if isinstance(extras, dict) else None
    if not isinstance(speech, list) or not all(
        isinstance(item, str) for item in speech
    ):
        errors.append("speech_recording must be a list of dependency strings")
        return errors

    expected = f"{COMPANION_NAME}=={app_version}"
    companion_requirements = [
        requirement
        for requirement in speech
        if _normalized_name(re.split(r"[<>=!~;\[\s]", requirement, 1)[0])
        == COMPANION_NAME
    ]
    if companion_requirements != [expected]:
        errors.append(
            "speech_recording must contain exactly one unmarked exact companion pin "
            f"{expected!r}; found {companion_requirements!r}"
        )
    return errors


def main(argv: list[str] | None = None) -> int:
    """Run the version-lock check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    errors = check_version_sync(args.repo_root.resolve())
    if errors:
        for error in errors:
            print(f"voice AEC version error: {error}", file=sys.stderr)
        return 1
    print("voice AEC app/companion version lock is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
