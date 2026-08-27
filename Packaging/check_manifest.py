#!/usr/bin/env python3
"""Validate the complete source-distribution and wheel content contract."""

from __future__ import annotations

import argparse
import configparser
from email.parser import Parser
import fnmatch
from pathlib import Path, PurePosixPath
import re
import tarfile
import zipfile


# The file template store (tldw_chatbook/Chunking/templates/) is deleted
# (spec §8.1.2): no path under it may appear in either artifact.
CHUNKING_TEMPLATES_PREFIX = "tldw_chatbook/Chunking/templates/"

# Migration scripts are DERIVED, never listed (task-19860). Two independent
# derivations, because either one alone can be defeated:
#   * the source tree next to this file -- the full set the artifact owes;
#   * the ``.sql`` names the ARTIFACT'S OWN ``ChaChaNotes_DB.py`` opens --
#     which still holds when the checker is run somewhere the source tree is
#     not, and which is what actually decides whether the app starts.
MIGRATIONS_PREFIX = "tldw_chatbook/DB/migrations/"
CHACHANOTES_DB_MODULE_PATH = "tldw_chatbook/DB/ChaChaNotes_DB.py"
# Matches the ``Path(__file__).parent / "migrations" / "<name>.sql"`` form
# every file-backed migration step uses to locate its script.
RUNTIME_MIGRATION_READ = re.compile(r'"migrations"\s*/\s*"([^"\n]+\.sql)"')
REPO_ROOT = Path(__file__).resolve().parents[1]

SAMIRA_RESOURCE_ROOT = "tldw_chatbook/assets/characters/samira"
SAMIRA_REACTION_LABELS = (
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "neutral",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "thinking",
    "speaking",
    "error",
)
SAMIRA_RESOURCE_PATHS = {
    f"{SAMIRA_RESOURCE_ROOT}/{name}"
    for name in (
        "ASSET_LICENSE.md",
        "Samira.character.json",
        "Sammy.png",
        "visual_identity_pack.json",
    )
} | {
    f"{SAMIRA_RESOURCE_ROOT}/expressions/{label}.webp"
    for label in SAMIRA_REACTION_LABELS
}
TIKTOKEN_CACHE_PREFIX = "tldw_chatbook/assets/tiktoken_cache/"
TIKTOKEN_RESOURCE_PATHS = {
    f"{TIKTOKEN_CACHE_PREFIX}{name}"
    for name in (
        "0ea1e91bbb3a60f729a8dc8f777fd2fc07cd8df4",
        "6c7ea1a7e38e3a7f062df639a5b80947f075ffe6",
        "6d1cbeee0f20b3d9449abfede4726ed8212e3aee",
        "9b5ad71b2ce5302211f9c61530b329a4922fc6a4",
        "ec7223a39ce59f226a68acc30dc1af2788490e15",
        "fb374d419588a4632f3f557e76b4b70aebbca790",
        "LICENSE.txt",
        "NOTICE.txt",
        "manifest.json",
    )
}
TIKTOKEN_REQUIREMENT = "tiktoken==0.14.0"

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
    "tldw_chatbook/Evals/config/eval_config.yaml",
    "tldw_chatbook/Third_Party/aider/LICENSE.txt",
    "tldw_chatbook/Third_Party/textual_fspicker/LICENSE",
    # Apache-2.0 re-licensed subtrees whose modules ship (task-19860 review).
    "tldw_chatbook/LLM_Calls/LICENSE",
    "tldw_chatbook/tldw_api/LICENSE",
} | SAMIRA_RESOURCE_PATHS | TIKTOKEN_RESOURCE_PATHS

REQUIRED_WHEEL_PATHS = {
    "tldw_chatbook/__init__.py",
    "tldw_chatbook/app.py",
    "tldw_chatbook/css/tldw_cli_modular.tcss",
    "tldw_chatbook/Config_Files/rag_pipelines.toml",
    "tldw_chatbook/Evals/config/eval_config.yaml",
    "tldw_chatbook/Third_Party/aider/LICENSE.txt",
    "tldw_chatbook/Third_Party/textual_fspicker/LICENSE",
    # Apache-2.0 re-licensed subtrees whose modules ship (task-19860 review).
    "tldw_chatbook/LLM_Calls/LICENSE",
    "tldw_chatbook/tldw_api/LICENSE",
} | SAMIRA_RESOURCE_PATHS | TIKTOKEN_RESOURCE_PATHS

REQUIRED_SDIST_GLOBS = {
    "tldw_chatbook/css/*.tcss",
    "tldw_chatbook/css/Themes/*.tcss",
    "tldw_chatbook/css/core/*.tcss",
    "tldw_chatbook/css/features/*.tcss",
    "tldw_chatbook/css/layout/*.tcss",
    "tldw_chatbook/Config_Files/*.json",
    "tldw_chatbook/Config_Files/*.md",
    "tldw_chatbook/Evals/config/*.yaml",
    # Runtime eval datasets: matched, never enumerated (task-19860 review).
    "tldw_chatbook/Evals/eval_datasets/*.json",
}

REQUIRED_WHEEL_GLOBS = {
    "tldw_chatbook/css/*.tcss",
    "tldw_chatbook/css/Themes/*.tcss",
    "tldw_chatbook/css/core/*.tcss",
    "tldw_chatbook/css/features/*.tcss",
    "tldw_chatbook/css/layout/*.tcss",
    "tldw_chatbook/Config_Files/*.json",
    "tldw_chatbook/Config_Files/*.md",
    "tldw_chatbook/Evals/config/*.yaml",
    # Runtime eval datasets: matched, never enumerated (task-19860 review).
    "tldw_chatbook/Evals/eval_datasets/*.json",
}

FORBIDDEN_WHEEL_PATHS = {
    "tldw_chatbook/css/components/stats_screen.css",
    "tldw_chatbook/Config_Files/embedding_configs_examples.toml",
    "tldw_chatbook/Config_Files/pipeline_configs/custom_pipelines_example.toml",
    "tldw_chatbook/Evals/DEVELOPER_GUIDE.md",
}

EXPECTED_CONSOLE_SCRIPTS = {
    "tldw-cli": "tldw_chatbook.cli:main_cli_runner",
    "tldw-serve": "tldw_chatbook.Web_Server.serve:main",
}


def source_migration_paths(repo_root: Path = REPO_ROOT) -> set[str]:
    """Return every migration script the source tree owes the artifacts.

    Args:
        repo_root: Checkout root that contains ``tldw_chatbook/``.

    Returns:
        Archive-relative paths of every ``.sql`` under ``DB/migrations/``.

    Raises:
        FileNotFoundError: If the migrations directory is absent or empty --
            the check fails closed rather than requiring nothing.
    """
    directory = repo_root / "tldw_chatbook" / "DB" / "migrations"
    scripts = sorted(directory.glob("*.sql")) if directory.is_dir() else []
    if not scripts:
        raise FileNotFoundError(
            f"no migration scripts found under {directory}; "
            "run the checker from a checkout that contains tldw_chatbook/"
        )
    return {f"{MIGRATIONS_PREFIX}{path.name}" for path in scripts}


def runtime_migration_paths(module_source: str) -> set[str]:
    """Return the migrations the shipped schema runner opens at runtime.

    Parsed from the artifact's own ``ChaChaNotes_DB.py`` so the requirement
    holds even where no source checkout is present.

    Args:
        module_source: Text of the packaged ``ChaChaNotes_DB.py``.

    Returns:
        Archive-relative paths of the ``.sql`` files it reads.

    Raises:
        ValueError: If no read site is detected at all -- that means the
            detector has drifted from the code, not that the app stopped
            needing migrations.
    """
    names = set(RUNTIME_MIGRATION_READ.findall(module_source))
    if not names:
        raise ValueError(
            "no migration reads detected in the packaged ChaChaNotes_DB.py; "
            "RUNTIME_MIGRATION_READ no longer matches the schema runner"
        )
    return {f"{MIGRATIONS_PREFIX}{name}" for name in names}


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


def _sdist_member_text(path: Path, member: str) -> str | None:
    with tarfile.open(path, "r:gz") as archive:
        for item in archive.getmembers():
            if item.isfile() and item.name.split("/", 1)[-1] == member:
                stream = archive.extractfile(item)
                if stream is None:
                    return None
                return stream.read().decode("utf-8")
    return None


def _wheel_member_text(path: Path, member: str) -> str | None:
    with zipfile.ZipFile(path) as archive:
        if member not in archive.namelist():
            return None
        return archive.read(member).decode("utf-8")


def _archive_migration_requirements(
    label: str,
    module_source: str | None,
) -> tuple[set[str], list[str]]:
    """Derive the migrations an artifact must carry from its own schema runner.

    Args:
        label: ``"sdist"`` or ``"wheel"``, used in error text.
        module_source: The artifact's packaged ``ChaChaNotes_DB.py`` text, or
            ``None`` when the module itself is missing.

    Returns:
        The required migration paths, and any errors that block derivation.
    """
    if module_source is None:
        return set(), [f"{label}: missing required path: {CHACHANOTES_DB_MODULE_PATH}"]
    try:
        return runtime_migration_paths(module_source), []
    except ValueError as error:
        return set(), [f"{label}: {error}"]


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
            if (
                name.endswith(".md")
                and not name.startswith("tldw_chatbook/Config_Files/")
                and name != f"{SAMIRA_RESOURCE_ROOT}/ASSET_LICENSE.md"
            ):
                errors.append(f"{label}: forbidden development Markdown: {name}")

    samira_members = {
        name for name in members if name.startswith(f"{SAMIRA_RESOURCE_ROOT}/")
    }
    if samira_members != SAMIRA_RESOURCE_PATHS:
        errors.append(
            f"{label}: Samira resources differ; "
            f"missing={sorted(SAMIRA_RESOURCE_PATHS - samira_members)}, "
            f"unexpected={sorted(samira_members - SAMIRA_RESOURCE_PATHS)}"
        )

    tiktoken_members = {
        name for name in members if name.startswith(TIKTOKEN_CACHE_PREFIX)
    }
    if tiktoken_members != TIKTOKEN_RESOURCE_PATHS:
        errors.append(
            f"{label}: tiktoken cache resources differ; "
            f"missing={sorted(TIKTOKEN_RESOURCE_PATHS - tiktoken_members)}, "
            f"unexpected={sorted(tiktoken_members - TIKTOKEN_RESOURCE_PATHS)}"
        )

    template_store_paths = {
        name for name in members if name.startswith(CHUNKING_TEMPLATES_PREFIX)
    }
    if template_store_paths:
        errors.append(
            f"{label}: the file template store is deleted (spec §8.1.2) "
            f"but shipped: {sorted(template_store_paths)}"
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
        tiktoken_requirements = [
            requirement
            for requirement in metadata.get_all("Requires-Dist") or []
            if re.match(r"(?i)^tiktoken(?=$|\s|[<>=!~;\[])", requirement)
        ]
        if tiktoken_requirements != [TIKTOKEN_REQUIREMENT]:
            errors.append(
                f"{label}: expected exactly Requires-Dist: {TIKTOKEN_REQUIREMENT}; "
                f"found {tiktoken_requirements}"
            )

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

    # Migration requirements are derived, not listed (task-19860): the source
    # tree states what the artifacts owe, and each artifact's own schema
    # runner states what it cannot start without. Both derivations fail
    # closed, so a broken derivation is a red check rather than an empty one.
    try:
        source_migrations = source_migration_paths()
    except FileNotFoundError as error:
        source_migrations = set()
        errors.append(f"source tree: {error}")
    sdist_migrations, sdist_migration_errors = _archive_migration_requirements(
        "sdist", _sdist_member_text(sdist, CHACHANOTES_DB_MODULE_PATH)
    )
    wheel_migrations, wheel_migration_errors = _archive_migration_requirements(
        "wheel", _wheel_member_text(wheel, CHACHANOTES_DB_MODULE_PATH)
    )
    errors.extend(sdist_migration_errors)
    errors.extend(wheel_migration_errors)

    errors.extend(
        _validate_content(
            "sdist",
            sdist_members,
            required_paths=REQUIRED_SDIST_PATHS | source_migrations | sdist_migrations,
            required_globs=REQUIRED_SDIST_GLOBS,
        )
    )
    errors.extend(
        _validate_content(
            "wheel",
            wheel_members,
            required_paths=REQUIRED_WHEEL_PATHS | source_migrations | wheel_migrations,
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
