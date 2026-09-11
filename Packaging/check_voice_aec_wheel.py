#!/usr/bin/env python3
"""Validate native voice-AEC wheel and source-distribution contents."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
from email.policy import compat32
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import sys
import tarfile
import tomllib
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_ROOT = REPO_ROOT / "native" / "voice_aec"
PACKAGE_NAME = "tldw-voice-aec"
PACKAGE_DIR = "tldw_voice_aec"
PYBIND11_REQUIREMENT = "pybind11==3.1.0"
UPSTREAM_COMMIT = "109e23c9cec3a44e67c08774874a409741b1e58a"
UPSTREAM_TREE = "71115034a67c1e7f98a4c2a61d80278db5a9b2ae"
UPSTREAM_REPOSITORY = "https://webrtc.googlesource.com/src"
PROVENANCE_PREFIX = f"{PACKAGE_DIR}/provenance/"
REQUIRED_PROVENANCE = {
    "UPSTREAM.json": NATIVE_ROOT / "vendor" / "webrtc" / "UPSTREAM.json",
    "FILES.sha256": NATIVE_ROOT / "vendor" / "webrtc" / "FILES.sha256",
    "PRISTINE_FILES.sha256": (
        NATIVE_ROOT / "vendor" / "webrtc" / "PRISTINE_FILES.sha256"
    ),
    "PATCHES.md": NATIVE_ROOT / "vendor" / "webrtc" / "PATCHES.md",
    "THIRD_PARTY_NOTICES.md": NATIVE_ROOT / "THIRD_PARTY_NOTICES.md",
    "PYBIND11_LICENSE.txt": NATIVE_ROOT / "PYBIND11_LICENSE.txt",
    "LICENSE": NATIVE_ROOT / "vendor" / "webrtc" / "LICENSE",
    "PATENTS": NATIVE_ROOT / "vendor" / "webrtc" / "PATENTS",
    "OOURA_LICENSE": NATIVE_ROOT / "vendor" / "webrtc" / "OOURA_LICENSE",
}
DIST_INFO_LICENSES = {
    "THIRD_PARTY_NOTICES.md": NATIVE_ROOT / "THIRD_PARTY_NOTICES.md",
    "PYBIND11_LICENSE.txt": NATIVE_ROOT / "PYBIND11_LICENSE.txt",
    "vendor/webrtc/LICENSE": NATIVE_ROOT / "vendor" / "webrtc" / "LICENSE",
    "vendor/webrtc/OOURA_LICENSE": (
        NATIVE_ROOT / "vendor" / "webrtc" / "OOURA_LICENSE"
    ),
    "vendor/webrtc/PATENTS": NATIVE_ROOT / "vendor" / "webrtc" / "PATENTS",
}
SHARED_LIBRARY = re.compile(r"(?:\.so(?:\.[^/]+)*|\.dylib|\.dll|\.pyd)$", re.I)
SDIST_BLOCKED_DIRECTORIES = frozenset(
    {
        ".cache",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "_skbuild",
        "build",
        "cmakefiles",
        "dist",
        "wheelhouse",
    }
)
SDIST_BLOCKED_FILENAMES = frozenset(
    {
        ".ds_store",
        ".ninja_deps",
        ".ninja_log",
        "build.ninja",
        "cmake_install.cmake",
        "cmakecache.txt",
    }
)
SDIST_BLOCKED_SUFFIXES = (
    ".a",
    ".dll",
    ".dylib",
    ".lib",
    ".o",
    ".obj",
    ".pyc",
    ".pyd",
    ".pyo",
    ".so",
    ".tar.gz",
    ".whl",
    ".zip",
)


def _normalized_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _expected_version() -> str:
    metadata = tomllib.loads(
        (NATIVE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    return str(metadata["project"]["version"])


def _unsafe_member(member: zipfile.ZipInfo) -> bool:
    name = member.filename.removesuffix("/")
    path = PurePosixPath(name)
    return (
        not name
        or name == "."
        or "\\" in name
        or path.is_absolute()
        or bool(PureWindowsPath(name).drive)
        or ".." in path.parts
        or name != path.as_posix()
        or (
            member.filename.endswith("/")
            and (
                member.file_size != 0
                or stat.S_IFMT(member.external_attr >> 16) not in (0, stat.S_IFDIR)
            )
        )
    )


def _unsafe_tar_member(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        not name
        or "\\" in name
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
    )


def _is_generated_sdist_member(path: PurePosixPath) -> bool:
    lowered = tuple(part.lower() for part in path.parts)
    if any(
        part in SDIST_BLOCKED_DIRECTORIES
        or part.endswith(".egg-info")
        or part.startswith("cmake-build-")
        for part in lowered
    ):
        return True
    filename = lowered[-1]
    return filename in SDIST_BLOCKED_FILENAMES or filename.endswith(
        SDIST_BLOCKED_SUFFIXES
    )


def _expected_sdist_sources() -> dict[str, Path]:
    expected: dict[str, Path] = {}
    for source in NATIVE_ROOT.rglob("*"):
        if not source.is_file() or source.is_symlink():
            continue
        relative = PurePosixPath(source.relative_to(NATIVE_ROOT).as_posix())
        if relative.as_posix() == "PKG-INFO" or _is_generated_sdist_member(relative):
            continue
        expected[relative.as_posix()] = source
    return expected


def _is_native_extension(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        path.parent.as_posix() == PACKAGE_DIR
        and path.name.startswith("_native.")
        and (path.name.endswith(".so") or path.name.endswith(".pyd"))
    )


def _pybind11_build_requirements(pyproject_bytes: bytes) -> list[str]:
    try:
        pyproject = tomllib.loads(pyproject_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError):
        return []
    build_system = pyproject.get("build-system")
    requirements = (
        build_system.get("requires") if isinstance(build_system, dict) else None
    )
    if not isinstance(requirements, list):
        return []
    return [
        requirement
        for requirement in requirements
        if isinstance(requirement, str)
        and _normalized_name(re.split(r"[<>=!~;\[\s]", requirement, 1)[0]) == "pybind11"
    ]


def check_wheel(path: Path, *, expected_version: str | None = None) -> list[str]:
    """Return contract violations for one repaired companion wheel.

    Static linkage permits only the Python ``_native`` extension itself; repair tools
    must not introduce a private shared-library directory.

    Args:
        path: Wheel to inspect without extracting it.
        expected_version: Required companion version; defaults to the checkout version.

    Returns:
        Human-readable violations. An empty list means the wheel is qualified.
    """
    version = expected_version or _expected_version()
    errors: list[str] = []
    if not path.is_file() or path.is_symlink():
        return [f"wheel is not a regular file: {path}"]
    if path.suffix != ".whl":
        return [f"not a wheel filename: {path.name}"]
    if not path.name.startswith(f"tldw_voice_aec-{version}-"):
        errors.append(
            f"wheel filename must encode tldw_voice_aec version {version}: {path.name}"
        )

    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(names) != len({name.removesuffix("/") for name in names}):
                errors.append("wheel contains duplicate archive members")
            unsafe = sorted(
                member.filename for member in members if _unsafe_member(member)
            )
            if unsafe:
                errors.append(f"wheel contains unsafe archive members: {unsafe}")
            files = {name for name in names if not name.endswith("/")}

            expected_dist_info = f"{PACKAGE_DIR}-{version}.dist-info"
            dist_info_markers = {
                (name, index, part)
                for name in names
                for index, part in enumerate(PurePosixPath(name).parts)
                if ".dist-info" in part.lower()
            }
            invalid_dist_info_markers = sorted(
                marker
                for marker in dist_info_markers
                if marker[1] != 0 or marker[2] != expected_dist_info
            )
            dist_info_roots = {
                part
                for _, index, part in dist_info_markers
                if index == 0 and part.endswith(".dist-info")
            }
            required_dist_info_members = {
                f"{expected_dist_info}/METADATA",
                f"{expected_dist_info}/WHEEL",
                f"{expected_dist_info}/RECORD",
            }
            missing_dist_info_members = sorted(required_dist_info_members - files)
            if (
                dist_info_roots != {expected_dist_info}
                or invalid_dist_info_markers
                or missing_dist_info_members
            ):
                errors.append(
                    "wheel must contain the exact dist-info root "
                    f"{expected_dist_info}/ with METADATA, WHEEL, and RECORD; "
                    f"found roots {sorted(dist_info_roots)}, invalid markers "
                    f"{invalid_dist_info_markers}, missing {missing_dist_info_members}"
                )
            if expected_dist_info in dist_info_roots:
                dist_info = expected_dist_info
                metadata_path = f"{dist_info}/METADATA"
                wheel_path = f"{dist_info}/WHEEL"
                if metadata_path not in files:
                    errors.append("wheel is missing dist-info/METADATA")
                else:
                    metadata = BytesParser(policy=compat32).parsebytes(
                        archive.read(metadata_path)
                    )
                    if _normalized_name(str(metadata.get("Name", ""))) != PACKAGE_NAME:
                        errors.append(f"wheel Name must be {PACKAGE_NAME}")
                    if metadata.get("Version") != version:
                        errors.append(
                            f"wheel Version must be {version}; found {metadata.get('Version')!r}"
                        )
                    if metadata.get("Metadata-Version") != "2.4":
                        errors.append(
                            "wheel metadata must use PEP 639/Core Metadata 2.4"
                        )
                    if metadata.get("License-Expression") != "BSD-3-Clause":
                        errors.append("wheel License-Expression must be BSD-3-Clause")
                    license_headers = metadata.get_all("License-File", [])
                    if sorted(license_headers) != sorted(DIST_INFO_LICENSES):
                        errors.append(
                            "wheel License-File headers must exactly cover reviewed "
                            f"notices; found {license_headers}"
                        )
                    for relative, canonical_path in DIST_INFO_LICENSES.items():
                        member = f"{dist_info}/licenses/{relative}"
                        if member not in files:
                            errors.append(
                                f"wheel is missing dist-info license file {member}"
                            )
                        elif archive.read(member) != canonical_path.read_bytes():
                            errors.append(
                                f"wheel dist-info license file {member} differs from "
                                "the reviewed source notice"
                            )
                if wheel_path not in files:
                    errors.append("wheel is missing dist-info/WHEEL")
                elif "Root-Is-Purelib: false" not in archive.read(wheel_path).decode(
                    "utf-8", errors="replace"
                ):
                    errors.append(
                        "voice AEC wheel must be marked Root-Is-Purelib: false"
                    )

            for filename, canonical_path in REQUIRED_PROVENANCE.items():
                member = f"{PROVENANCE_PREFIX}{filename}"
                if member not in files:
                    errors.append(f"wheel is missing required {member}")
                    continue
                contents = archive.read(member)
                if not contents:
                    errors.append(f"wheel contains empty required {member}")
                if canonical_path.is_file() and contents != canonical_path.read_bytes():
                    errors.append(
                        f"wheel {member} differs from the reviewed source notice"
                    )

            upstream_member = f"{PROVENANCE_PREFIX}UPSTREAM.json"
            if upstream_member in files:
                try:
                    upstream = json.loads(archive.read(upstream_member))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    errors.append(f"invalid pinned WebRTC provenance JSON: {error}")
                else:
                    required = {
                        "repository": UPSTREAM_REPOSITORY,
                        "commit": UPSTREAM_COMMIT,
                        "commit_tree": UPSTREAM_TREE,
                    }
                    if any(
                        upstream.get(key) != value for key, value in required.items()
                    ):
                        errors.append(
                            "pinned WebRTC provenance does not match the reviewed commit/tree"
                        )

            shared = sorted(name for name in files if SHARED_LIBRARY.search(name))
            native = [name for name in shared if _is_native_extension(name)]
            unexpected = [name for name in shared if name not in native]
            if unexpected:
                errors.append(f"unexpected shared library in wheel: {unexpected}")
            if len(native) != 1:
                errors.append(
                    f"wheel must contain exactly one native extension; found {native}"
                )
    except (OSError, zipfile.BadZipFile) as error:
        errors.append(f"cannot read wheel {path}: {error}")
    return errors


def check_sdist(path: Path, *, expected_version: str | None = None) -> list[str]:
    """Return contract violations for one companion source distribution.

    The archive is checked without extraction. Only reviewed source-tree roots are
    allowed, required build/binding/provenance inputs must match the checkout, and
    the vendored WebRTC manifest must cover byte-identical sources. Generated build,
    cache, binary, and nested distribution artifacts are rejected at any depth.

    Args:
        path: Gzip-compressed source distribution to inspect.
        expected_version: Required companion version; defaults to the checkout version.

    Returns:
        Human-readable violations. An empty list means the sdist is qualified.
    """
    version = expected_version or _expected_version()
    errors: list[str] = []
    if not path.is_file() or path.is_symlink():
        return [f"sdist is not a regular file: {path}"]
    expected_name = f"tldw_voice_aec-{version}.tar.gz"
    if path.name != expected_name:
        errors.append(f"sdist filename must be {expected_name}; found {path.name}")

    root = f"tldw_voice_aec-{version}"
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)):
                errors.append("sdist contains duplicate archive members")

            unsafe = sorted(name for name in names if _unsafe_tar_member(name))
            if unsafe:
                errors.append(f"sdist contains unsafe archive members: {unsafe}")

            special = sorted(
                member.name
                for member in members
                if not member.isfile() and not member.isdir()
            )
            if special:
                errors.append(
                    "sdist must contain only regular files and directories; "
                    f"found {special}"
                )

            files: dict[str, tarfile.TarInfo] = {}
            unexpected_roots: set[str] = set()
            generated: list[str] = []
            for member in members:
                member_path = PurePosixPath(member.name)
                if not member_path.parts or member_path.parts[0] != root:
                    unexpected_roots.add(member.name)
                    continue
                relative = PurePosixPath(*member_path.parts[1:])
                if not relative.parts:
                    continue
                if _is_generated_sdist_member(relative):
                    generated.append(relative.as_posix())
                if member.isfile():
                    files[relative.as_posix()] = member

            if unexpected_roots:
                errors.append(
                    f"sdist members must share exact root {root}: "
                    f"{sorted(unexpected_roots)}"
                )
            if generated:
                errors.append(
                    f"sdist contains generated/build artifact: {sorted(generated)}"
                )

            expected_sources = _expected_sdist_sources()
            expected_names = set(expected_sources)
            actual_names = set(files)
            missing = sorted(expected_names - actual_names)
            unexpected = sorted(actual_names - expected_names - {"PKG-INFO"})
            if missing:
                errors.append(f"sdist is missing required source files: {missing}")
            if unexpected:
                errors.append(f"sdist contains unexpected source files: {unexpected}")

            for relative in sorted(expected_names & actual_names):
                canonical_path = expected_sources[relative]
                member = files[relative]
                extracted = archive.extractfile(member)
                if extracted is None:
                    errors.append(f"sdist cannot read required source {relative}")
                    continue
                if extracted.read() != canonical_path.read_bytes():
                    errors.append(
                        f"sdist required source {relative} differs from the checkout"
                    )

            pyproject_member = files.get("pyproject.toml")
            if pyproject_member is not None:
                extracted = archive.extractfile(pyproject_member)
                pyproject_bytes = extracted.read() if extracted is not None else b""
                pybind11_requirements = _pybind11_build_requirements(pyproject_bytes)
                if pybind11_requirements != [PYBIND11_REQUIREMENT]:
                    errors.append(
                        "sdist pybind11 build pin must be exactly "
                        f"{PYBIND11_REQUIREMENT!r}; found {pybind11_requirements!r}"
                    )

            pkg_info_member = files.get("PKG-INFO")
            if pkg_info_member is None:
                errors.append("sdist is missing generated PKG-INFO")
            else:
                extracted = archive.extractfile(pkg_info_member)
                pkg_info_bytes = extracted.read() if extracted is not None else b""
                pkg_info = BytesParser(policy=compat32).parsebytes(pkg_info_bytes)
                if pkg_info.get("Metadata-Version") != "2.4":
                    errors.append("sdist PKG-INFO must use PEP 639/Core Metadata 2.4")
                if _normalized_name(str(pkg_info.get("Name", ""))) != PACKAGE_NAME:
                    errors.append(f"sdist PKG-INFO Name must be {PACKAGE_NAME}")
                if pkg_info.get("Version") != version:
                    errors.append(
                        "sdist PKG-INFO Version must be "
                        f"{version}; found {pkg_info.get('Version')!r}"
                    )
                if pkg_info.get("License-Expression") != "BSD-3-Clause":
                    errors.append(
                        "sdist PKG-INFO License-Expression must be BSD-3-Clause"
                    )
                license_headers = pkg_info.get_all("License-File", [])
                if sorted(license_headers) != sorted(DIST_INFO_LICENSES):
                    errors.append(
                        "sdist PKG-INFO License-File headers must exactly cover "
                        f"reviewed notices; found {license_headers}"
                    )

            manifest_member = files.get("vendor/webrtc/FILES.sha256")
            if manifest_member is not None:
                extracted = archive.extractfile(manifest_member)
                manifest = extracted.read() if extracted is not None else b""
                try:
                    manifest_lines = manifest.decode("utf-8").splitlines()
                except UnicodeDecodeError as error:
                    errors.append(f"sdist has invalid WebRTC FILES.sha256: {error}")
                    manifest_lines = []
                for line_number, line in enumerate(manifest_lines, start=1):
                    digest, separator, relative = line.partition("  ")
                    relative_path = PurePosixPath(relative)
                    if (
                        not separator
                        or not re.fullmatch(r"[0-9a-f]{64}", digest)
                        or _unsafe_tar_member(relative)
                    ):
                        errors.append(
                            "sdist has invalid WebRTC FILES.sha256 entry at line "
                            f"{line_number}"
                        )
                        continue
                    source_name = f"vendor/webrtc/{relative_path.as_posix()}"
                    source_member = files.get(source_name)
                    if source_member is None:
                        errors.append(
                            f"sdist is missing WebRTC manifest source {source_name}"
                        )
                        continue
                    source = archive.extractfile(source_member)
                    contents = source.read() if source is not None else b""
                    if hashlib.sha256(contents).hexdigest() != digest:
                        errors.append(
                            f"sdist WebRTC manifest digest mismatch for {source_name}"
                        )
    except (OSError, tarfile.TarError) as error:
        errors.append(f"cannot read sdist {path}: {error}")
    return errors


def check_distribution(path: Path, *, expected_version: str | None = None) -> list[str]:
    """Dispatch one wheel or sdist to its fail-closed validator."""
    if path.name.endswith(".tar.gz"):
        return check_sdist(path, expected_version=expected_version)
    return check_wheel(path, expected_version=expected_version)


def main(argv: list[str] | None = None) -> int:
    """Validate one or more wheel or source-distribution paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("distributions", nargs="+", type=Path)
    parser.add_argument("--expected-version", default=None)
    args = parser.parse_args(argv)
    failed = False
    for distribution in args.distributions:
        errors = check_distribution(
            distribution, expected_version=args.expected_version
        )
        if errors:
            failed = True
            for error in errors:
                print(f"{distribution}: {error}", file=sys.stderr)
        else:
            print(f"qualified voice AEC distribution: {distribution}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
