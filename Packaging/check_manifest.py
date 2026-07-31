#!/usr/bin/env python3
"""Validate the complete source-distribution and wheel content contract."""

from __future__ import annotations

import argparse
import configparser
from email.parser import Parser
import fnmatch
from pathlib import Path, PurePosixPath
import tarfile
import zipfile


TEMPLATE_NAMES = {
    "academic_paper",
    "code_documentation",
    "conversation",
    "ebook_chapters",
    "json",
    "legal_document",
    "paragraphs",
    "rolling_summarize",
    "semantic",
    "sentences",
    "tokens",
    "words",
    "xml",
}

REQUIRED_SDIST_PATHS = {
    "LICENSE",
    "README.md",
    "CLAUDE.md",
    "CHANGELOG.md",
    "MANIFEST.in",
    "pyproject.toml",
    "requirements.txt",
    "tldw_chatbook/__init__.py",
    "tldw_chatbook/app.py",
    "tldw_chatbook/css/tldw_cli_modular.tcss",
    "tldw_chatbook/css/components/stats_screen.css",
    "tldw_chatbook/Config_Files/rag_pipelines.toml",
    "tldw_chatbook/DB/migrations/chachanotes_v26_to_v27_citation_provenance.sql",
    "tldw_chatbook/DB/migrations/chachanotes_v27_to_v28_character_authority.sql",
    "tldw_chatbook/Evals/config/eval_config.yaml",
    "tldw_chatbook/Third_Party/aider/LICENSE.txt",
    "tldw_chatbook/Third_Party/textual_fspicker/LICENSE",
}

REQUIRED_WHEEL_PATHS = {
    "tldw_chatbook/__init__.py",
    "tldw_chatbook/app.py",
    "tldw_chatbook/css/tldw_cli_modular.tcss",
    "tldw_chatbook/Config_Files/rag_pipelines.toml",
    "tldw_chatbook/DB/migrations/chachanotes_v26_to_v27_citation_provenance.sql",
    "tldw_chatbook/DB/migrations/chachanotes_v27_to_v28_character_authority.sql",
    "tldw_chatbook/Evals/config/eval_config.yaml",
    "tldw_chatbook/Third_Party/aider/LICENSE.txt",
    "tldw_chatbook/Third_Party/textual_fspicker/LICENSE",
}

REQUIRED_SDIST_GLOBS = {
    "tldw_chatbook/css/*.tcss",
    "tldw_chatbook/css/Themes/*.tcss",
    "tldw_chatbook/css/core/*.tcss",
    "tldw_chatbook/css/features/*.tcss",
    "tldw_chatbook/css/layout/*.tcss",
    "tldw_chatbook/Config_Files/*.json",
    "tldw_chatbook/Config_Files/*.md",
    "tldw_chatbook/Chunking/templates/*.json",
    "tldw_chatbook/Evals/config/*.yaml",
}

REQUIRED_WHEEL_GLOBS = {
    "tldw_chatbook/css/*.tcss",
    "tldw_chatbook/css/Themes/*.tcss",
    "tldw_chatbook/css/core/*.tcss",
    "tldw_chatbook/css/features/*.tcss",
    "tldw_chatbook/css/layout/*.tcss",
    "tldw_chatbook/Config_Files/*.json",
    "tldw_chatbook/Config_Files/*.md",
    "tldw_chatbook/Chunking/templates/*.json",
    "tldw_chatbook/Evals/config/*.yaml",
}

FORBIDDEN_WHEEL_PATHS = {
    "tldw_chatbook/css/components/stats_screen.css",
    "tldw_chatbook/Config_Files/embedding_configs_examples.toml",
    "tldw_chatbook/Config_Files/pipeline_configs/custom_pipelines_example.toml",
    "tldw_chatbook/Chunking/templates/README.md",
    "tldw_chatbook/Chunking/templates/example_usage.py",
    "tldw_chatbook/Evals/DEVELOPER_GUIDE.md",
}

EXPECTED_CONSOLE_SCRIPTS = {
    "tldw-cli": "tldw_chatbook.cli:main_cli_runner",
    "tldw-serve": "tldw_chatbook.Web_Server.serve:main",
}


def _sdist_members(path: Path) -> tuple[set[str], list[str]]:
    errors: list[str] = []
    with tarfile.open(path, "r:gz") as archive:
        files = [member.name for member in archive.getmembers() if member.isfile()]
    roots = {name.split("/", 1)[0] for name in files}
    if len(roots) != 1:
        errors.append(
            "source distribution must have exactly one top-level directory; "
            f"found: {sorted(roots)}"
        )
    members = {name.split("/", 1)[1] for name in files if "/" in name}
    return members, errors


def _wheel_members(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        return {name for name in archive.namelist() if not name.endswith("/")}


def _common_forbidden_reason(name: str) -> str | None:
    parts = PurePosixPath(name).parts
    if parts and parts[0] in {"Tests", "tests", "STests"}:
        return "root test tree"
    if "__pycache__" in parts:
        return "Python cache directory"
    if name.endswith((".pyc", ".pyo")):
        return "compiled Python cache"
    if ".DS_Store" in parts:
        return "OS metadata"
    return None


def _validate_content(
    label: str,
    members: set[str],
    *,
    required_paths: set[str],
    required_globs: set[str],
    forbidden_paths: set[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    for path in sorted(required_paths - members):
        errors.append(f"{label}: missing required path: {path}")
    for pattern in sorted(required_globs):
        if not any(fnmatch.fnmatchcase(name, pattern) for name in members):
            errors.append(f"{label}: missing required pattern: {pattern}")

    for name in sorted(members):
        reason = _common_forbidden_reason(name)
        if reason is not None:
            errors.append(f"{label}: forbidden {reason}: {name}")

    for path in sorted((forbidden_paths or set()) & members):
        errors.append(f"{label}: forbidden path: {path}")

    if label == "wheel":
        for name in sorted(members):
            if name.endswith(".md") and not name.startswith(
                "tldw_chatbook/Config_Files/"
            ):
                errors.append(f"{label}: forbidden development Markdown: {name}")

    template_names = {
        PurePosixPath(name).stem
        for name in members
        if name.startswith("tldw_chatbook/Chunking/templates/")
        and name.endswith(".json")
    }
    if template_names != TEMPLATE_NAMES:
        missing = sorted(TEMPLATE_NAMES - template_names)
        unexpected = sorted(template_names - TEMPLATE_NAMES)
        errors.append(
            f"{label}: chunking templates differ; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return errors


def _validate_metadata(
    sdist: Path,
    sdist_members: set[str],
    wheel: Path,
    wheel_members: set[str],
) -> list[str]:
    errors: list[str] = []
    wheel_metadata_names = sorted(
        name for name in wheel_members if name.endswith(".dist-info/METADATA")
    )
    wheel_entry_point_names = sorted(
        name for name in wheel_members if name.endswith(".dist-info/entry_points.txt")
    )
    wheel_license_names = sorted(
        name for name in wheel_members if name.endswith(".dist-info/licenses/LICENSE")
    )
    sdist_metadata_names = sorted(name for name in sdist_members if name == "PKG-INFO")

    for label, names in (
        ("wheel METADATA", wheel_metadata_names),
        ("wheel entry_points.txt", wheel_entry_point_names),
        ("wheel project license", wheel_license_names),
        ("sdist PKG-INFO", sdist_metadata_names),
    ):
        if len(names) != 1:
            errors.append(f"{label}: expected exactly one, found {names}")
    if errors:
        return errors

    with zipfile.ZipFile(wheel) as archive:
        wheel_metadata = Parser().parsestr(
            archive.read(wheel_metadata_names[0]).decode("utf-8")
        )
        entry_points = configparser.ConfigParser()
        entry_points.read_string(
            archive.read(wheel_entry_point_names[0]).decode("utf-8")
        )

    with tarfile.open(sdist, "r:gz") as archive:
        member = next(
            item
            for item in archive.getmembers()
            if item.isfile() and item.name.endswith("/PKG-INFO")
        )
        stream = archive.extractfile(member)
        if stream is None:
            errors.append("sdist PKG-INFO: could not read metadata")
            return errors
        sdist_metadata = Parser().parsestr(stream.read().decode("utf-8"))

    for label, metadata in (
        ("wheel METADATA", wheel_metadata),
        ("sdist PKG-INFO", sdist_metadata),
    ):
        if metadata["Metadata-Version"] != "2.4":
            errors.append(
                f"{label}: expected Metadata-Version 2.4, "
                f"found {metadata['Metadata-Version']!r}"
            )
        if metadata["License-Expression"] != "AGPL-3.0-or-later":
            errors.append(
                f"{label}: expected License-Expression AGPL-3.0-or-later, "
                f"found {metadata['License-Expression']!r}"
            )
        if "LICENSE" not in (metadata.get_all("License-File") or []):
            errors.append(f"{label}: missing License-File: LICENSE")

    if not entry_points.has_section("console_scripts"):
        errors.append("wheel entry_points.txt: missing [console_scripts]")
    elif dict(entry_points["console_scripts"]) != EXPECTED_CONSOLE_SCRIPTS:
        errors.append(
            "wheel entry_points.txt: console scripts differ; "
            f"found={dict(entry_points['console_scripts'])}"
        )
    return errors


def check_distribution(dist_dir: Path = Path("dist")) -> bool:
    """Return whether exactly one sdist and wheel satisfy the release contract."""

    if not dist_dir.is_dir():
        print(f"distribution directory not found: {dist_dir}")
        return False

    sdists = sorted(dist_dir.glob("*.tar.gz"))
    wheels = sorted(dist_dir.glob("*.whl"))
    errors: list[str] = []
    if len(sdists) != 1:
        errors.append(
            "expected exactly one source distribution (*.tar.gz); "
            f"found: {[path.name for path in sdists]}"
        )
    if len(wheels) != 1:
        errors.append(
            "expected exactly one wheel (*.whl); "
            f"found: {[path.name for path in wheels]}"
        )
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return False

    sdist = sdists[0]
    wheel = wheels[0]
    sdist_members, sdist_errors = _sdist_members(sdist)
    wheel_members = _wheel_members(wheel)
    errors.extend(sdist_errors)
    errors.extend(
        _validate_content(
            "sdist",
            sdist_members,
            required_paths=REQUIRED_SDIST_PATHS,
            required_globs=REQUIRED_SDIST_GLOBS,
        )
    )
    errors.extend(
        _validate_content(
            "wheel",
            wheel_members,
            required_paths=REQUIRED_WHEEL_PATHS,
            required_globs=REQUIRED_WHEEL_GLOBS,
            forbidden_paths=FORBIDDEN_WHEEL_PATHS,
        )
    )
    errors.extend(
        _validate_metadata(
            sdist,
            sdist_members,
            wheel,
            wheel_members,
        )
    )

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return False

    print(f"Validated source distribution: {sdist}")
    print(f"Validated wheel: {wheel}")
    return True


def main() -> None:
    """Run the distribution checker as a command-line program."""

    parser = argparse.ArgumentParser()
    parser.add_argument("dist_dir", nargs="?", type=Path, default=Path("dist"))
    args = parser.parse_args()
    raise SystemExit(0 if check_distribution(args.dist_dir) else 1)


if __name__ == "__main__":
    main()
