#!/usr/bin/env python3
"""Vendor the exact WebRTC AEC3 compile closure used by ``tldw_voice_aec``.

The input must be an exact, prepared checkout including the pinned Abseil DEPS
subtree.  Only files reachable from the AEC3 GN target and residing under
``ALLOWED_ROOTS`` are copied.  Generated artifacts use the immutable upstream
commit timestamp so repeated imports are byte-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import deque
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator


REPOSITORY = "https://webrtc.googlesource.com/src"
COMMIT = "109e23c9cec3a44e67c08774874a409741b1e58a"
COMMIT_TREE = "71115034a67c1e7f98a4c2a61d80278db5a9b2ae"
COMMIT_TIMESTAMP = "2020-03-02T12:42:42+00:00"

ALLOWED_ROOTS = (
    "api/audio",
    "common_audio",
    "modules/audio_processing",
    "rtc_base",
    "system_wrappers",
    "third_party/abseil-cpp",
)
ALLOWED_FILES = (
    "api/array_view.h",
    "api/ref_counted_base.h",
    "api/rtp_headers.h",
    "api/rtp_packet_info.h",
    "api/rtp_packet_infos.h",
    "api/scoped_refptr.h",
    "api/units/time_delta.h",
    "api/units/timestamp.h",
    "api/video/color_space.h",
    "api/video/hdr_metadata.h",
    "api/video/video_content_type.h",
    "api/video/video_frame_marking.h",
    "api/video/video_rotation.h",
    "api/video/video_timing.h",
    "common_types.h",
)
ABSEIL_REPOSITORY = "https://chromium.googlesource.com/chromium/src/third_party"
ABSEIL_COMMIT = "ac875ae5393d0516243cfd5d078cd4b098388f6b"
ABSEIL_COMMIT_TIMESTAMP = "2020-02-28T15:49:05+00:00"
ABSEIL_SUBTREE_TREE = "7f84a7844f32a5a56a02bc7133e943b0932c50b2"
ABSEIL_LICENSE_SHA256 = (
    "c79a7fea0e3cac04cd43f20e7b648e5a0ff8fa5344e644b0ee09ca1162b62747"
)
OOURA_SOURCE_PATH = "modules/audio_processing/utility/ooura_fft.cc"
OOURA_SOURCE_URL = "http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html"
OOURA_SOURCE_SHA256 = "8a07c7cffe3471fbbd9b734cf658f7d0ed28904e4f828179dbebf650501040bd"
OOURA_LICENSE_SHA256 = (
    "efa6b38d923e14333047a48580043d366dc51e5d0ebd7d658fa341ad3aafb576"
)
COMPILER_DEFINES = {
    "common": ["WEBRTC_APM_DEBUG_DUMP=0", "WEBRTC_ENABLE_PROTOBUF=0"],
    "macos": ["WEBRTC_MAC", "WEBRTC_POSIX"],
    "linux": ["WEBRTC_LINUX", "WEBRTC_POSIX"],
    "windows": ["WEBRTC_WIN", "NOMINMAX", "_CRT_SECURE_NO_WARNINGS"],
}
# Translation-unit roots are an explicit checked-in allowlist.  GN umbrella
# targets such as rtc_base_approved deliberately are not traversed: doing so
# would pull unrelated WebRTC APIs outside this companion's approved roots.
COMPILE_TARGET_ALLOWLIST = (
    "//api:array_view",
    "//api/audio:aec3_config",
    "//api/audio:audio_frame_api",
    "//api/audio:echo_control",
    "//common_audio:common_audio",
    "//common_audio:common_audio_c",
    "//modules/audio_processing:apm_logging",
    "//modules/audio_processing:audio_buffer",
    "//modules/audio_processing:high_pass_filter",
    "//modules/audio_processing/aec3:aec3",
    "//modules/audio_processing/utility:cascaded_biquad_filter",
    "//modules/audio_processing/utility:ooura_fft",
    "//rtc_base:checks",
    "//rtc_base:criticalsection",
    "//rtc_base:logging",
    "//rtc_base:platform_thread_types",
    "//rtc_base:safe_compare",
    "//rtc_base:safe_minmax",
    "//rtc_base:stringutils",
    "//rtc_base:timeutils",
    "//rtc_base:type_traits",
    "//rtc_base/memory:aligned_malloc",
    "//rtc_base/system:arch",
    "//rtc_base/system:inline",
    "//rtc_base/system:rtc_export",
    "//system_wrappers:cpu_features_api",
    "//system_wrappers:field_trial",
    "//system_wrappers:metrics",
)
COMPILE_SOURCE_ALLOWLIST = (
    "rtc_base/race_checker.cc",
    # Owned by //system_wrappers:system_wrappers; cpu_features_api is header-only.
    "system_wrappers/source/cpu_features.cc",
)
METADATA_FILES = frozenset(
    {
        "COMPILER_DEFINES.cmake",
        "LICENSE",
        "OOURA_LICENSE",
        "PATENTS",
        "PATCHES.md",
        "PRISTINE_FILES.sha256",
        "UPSTREAM.json",
    }
)
MANIFEST_NAME = "FILES.sha256"
PRISTINE_MANIFEST_NAME = "PRISTINE_FILES.sha256"
PRISTINE_MANIFEST_SHA256 = (
    "596ddbb3291fc5fd432376ef4bdfee6fed67bd999709638d68436f2b1ef041a4"
)
PATCH_NAME = "0001-expose-delay-health-evidence.patch"
PATCH_SHA256 = "94f2a8dad384194c8b3ffb63695287ae7a046e1136b4357b9b3d048579ad3a1c"
PATCHES = (
    (PATCH_NAME, PATCH_SHA256),
    (
        "0002-include-stddef-for-clockdrift-detector.patch",
        "f3bbfe1b2f7d54030fba05fefefbcc181706975c09aa43c9aeb83ca96379ca6e",
    ),
    (
        "0003-include-memory-for-reverb-model-estimator.patch",
        "dfdad20c0732cf30364d6a4c5161e9e2612dc8c0123c511ea2537e2439c9e9f5",
    ),
)
PATCH_SERIES = "# WebRTC AEC3 patch series\n\n" + "".join(
    f"{number}. `{name}` - SHA-256 `{digest}`\n"
    for number, (name, digest) in enumerate(PATCHES, 1)
)
NOTICE_GENERATION_VERSION = 2
COMPILE_CLOSURE_FILE_COUNT = 316
WEBRTC_LICENSE_SHA256 = (
    "ab00a482b6a3902e40211b43c5d0441962ea99b6cc7c25c0f243fa270b78d482"
)
WEBRTC_PATENTS_SHA256 = (
    "01462e2068d1a04c2274f3389773014c14ed9bc3446b28303543bd3e3c064145"
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENDOR_ROOT = PACKAGE_ROOT / "vendor" / "webrtc"
DEFAULT_NOTICES_PATH = PACKAGE_ROOT / "THIRD_PARTY_NOTICES.md"

_TARGET_START_RE = re.compile(
    r'(?:rtc_library|rtc_source_set|source_set|static_library)\("(?P<name>[^"\n]+)"\)\s*\{'
)
_QUOTED_RE = re.compile(r'"([^"\n]+)"')
_INCLUDE_RE = re.compile(rb'^\s*#\s*include\s*"([^"\n]+)"', re.MULTILINE)


class VendorError(ValueError):
    """Raised when source provenance or vendored integrity is invalid."""


def _git(source: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_source(source: Path, *, verify_clean: bool) -> None:
    """Validate exact WebRTC provenance and the prepared dependency checkout."""
    if not (source / ".git").exists():
        raise VendorError(f"source is not a Git checkout: {source}")
    if _git(source, "rev-parse", "HEAD") != COMMIT:
        raise VendorError(f"source HEAD must be exact pinned commit {COMMIT}")
    if _git(source, "rev-parse", "HEAD^{tree}") != COMMIT_TREE:
        raise VendorError(f"source tree must be exact pinned object {COMMIT_TREE}")
    if _git(source, "show", "-s", "--format=%cI", "HEAD") != COMMIT_TIMESTAMP:
        raise VendorError("source commit timestamp does not match pinned provenance")
    if verify_clean and _git(source, "status", "--porcelain"):
        raise VendorError("source checkout is not clean")
    for required in ("LICENSE", "PATENTS", "modules/audio_processing/aec3/BUILD.gn"):
        if not (source / required).is_file():
            raise VendorError(
                f"prepared source is missing required WebRTC file: {required}"
            )
    abseil = source / "third_party" / "abseil-cpp"
    if not (abseil / "absl" / "types" / "optional.h").is_file():
        raise VendorError(
            "prepared source is missing required DEPS dependency "
            "third_party/abseil-cpp (AEC3 depends on absl/types:optional)"
        )
    if not (abseil / "LICENSE").is_file():
        raise VendorError("prepared Abseil dependency is missing LICENSE")
    deps = (source / "DEPS").read_text(encoding="utf-8")
    expected_pin = (
        f"https://chromium.googlesource.com/chromium/src/third_party@{ABSEIL_COMMIT}"
    )
    if expected_pin not in deps:
        raise VendorError(
            "WebRTC DEPS does not contain the pinned Chromium third_party revision"
        )
    actual_abseil_tree = git_tree_object_id(abseil)
    if actual_abseil_tree != ABSEIL_SUBTREE_TREE:
        raise VendorError(
            "prepared Abseil subtree provenance mismatch: "
            f"expected {ABSEIL_SUBTREE_TREE}, got {actual_abseil_tree}"
        )
    if _sha256(abseil / "LICENSE") != ABSEIL_LICENSE_SHA256:
        raise VendorError(
            "prepared Abseil LICENSE hash does not match pinned provenance"
        )
    ooura_source = source / OOURA_SOURCE_PATH
    if _sha256(ooura_source) != OOURA_SOURCE_SHA256:
        raise VendorError("pinned Ooura FFT source hash does not match provenance")
    ooura_notice = extract_ooura_notice(ooura_source)
    if hashlib.sha256(ooura_notice.encode("utf-8")).hexdigest() != OOURA_LICENSE_SHA256:
        raise VendorError("derived Ooura FFT notice hash does not match provenance")


def is_allowed_path(relative_path: str) -> bool:
    """Return whether a vendored path is within the exact approved allowlist."""
    try:
        path = normalized_relative_path(relative_path)
    except VendorError:
        return False
    return path in ALLOWED_FILES or any(
        path == root or path.startswith(f"{root}/") for root in ALLOWED_ROOTS
    )


def normalized_relative_path(relative_path: str) -> str:
    """Validate and return one normalized relative POSIX path."""
    if not isinstance(relative_path, str) or not relative_path:
        raise VendorError("path must be a normalized relative POSIX path")
    if "\\" in relative_path or "\0" in relative_path:
        raise VendorError("path must be a normalized relative POSIX path")
    if relative_path.startswith("/") or re.match(r"^[A-Za-z]:/", relative_path):
        raise VendorError("path must be a normalized relative POSIX path")
    components = relative_path.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise VendorError("path must be a normalized relative POSIX path")
    if PurePosixPath(relative_path).as_posix() != relative_path:
        raise VendorError("path must be a normalized relative POSIX path")
    return relative_path


def extract_ooura_notice(source_file: Path) -> str:
    """Extract the exact Ooura permission notice from the pinned source header."""
    text = source_file.read_text(encoding="utf-8")
    marker = " * Changes by the WebRTC authors:"
    header, separator, _ = text.partition(marker)
    if not separator or not header.startswith("/*\n"):
        raise VendorError("pinned Ooura FFT source header is malformed")

    notice_lines: list[str] = []
    for line in header.splitlines()[1:]:
        if line == " *":
            notice_lines.append("")
        elif line.startswith(" * "):
            notice_lines.append(line[3:])
        else:
            raise VendorError("pinned Ooura FFT notice has an unexpected format")
    notice = "\n".join(notice_lines).strip() + "\n"
    if OOURA_SOURCE_URL not in notice:
        raise VendorError("pinned Ooura FFT notice is missing its source URL")
    return notice


def git_tree_object_id(root: Path) -> str:
    """Compute Git's SHA-1 tree object ID for a filesystem subtree."""

    def blob_object_id(data: bytes) -> bytes:
        header = f"blob {len(data)}\0".encode("ascii")
        return hashlib.sha1(header + data).digest()

    entries: list[tuple[bytes, bytes]] = []
    children = sorted(
        root.iterdir(),
        key=lambda path: os.fsencode(path.name) + (b"/" if path.is_dir() else b""),
    )
    for child in children:
        name = os.fsencode(child.name)
        if child.is_symlink():
            mode = b"120000"
            object_id = blob_object_id(os.fsencode(os.readlink(child)))
        elif child.is_dir():
            mode = b"40000"
            object_id = bytes.fromhex(git_tree_object_id(child))
        else:
            mode = b"100755" if child.stat().st_mode & 0o111 else b"100644"
            object_id = blob_object_id(child.read_bytes())
        entries.append((name, mode + b" " + name + b"\0" + object_id))
    contents = b"".join(entry for _, entry in entries)
    header = f"tree {len(contents)}\0".encode("ascii")
    return hashlib.sha1(header + contents).hexdigest()


def _target_block(build_file: Path, target_name: str) -> str:
    text = build_file.read_text(encoding="utf-8")
    for match in _TARGET_START_RE.finditer(text):
        if match.group("name") != target_name:
            continue
        start = match.start()
        cursor = match.end()
        depth = 1
        in_string = False
        escaped = False
        while cursor < len(text) and depth:
            char = text[cursor]
            if escaped:
                escaped = False
            elif char == "\\" and in_string:
                escaped = True
            elif char == '"':
                in_string = not in_string
            elif not in_string and char == "{":
                depth += 1
            elif not in_string and char == "}":
                depth -= 1
            cursor += 1
        if depth:
            raise VendorError(f"unterminated GN target {target_name} in {build_file}")
        return text[start:cursor]
    raise VendorError(f"GN target {target_name!r} not found in {build_file}")


def _assignment_values(block: str, variable: str) -> Iterator[str]:
    assignment = re.compile(rf"\b{re.escape(variable)}\s*(?:\+?=)\s*\[")
    for match in assignment.finditer(block):
        cursor = match.end()
        depth = 1
        in_string = False
        escaped = False
        while cursor < len(block) and depth:
            char = block[cursor]
            if escaped:
                escaped = False
            elif char == "\\" and in_string:
                escaped = True
            elif char == '"':
                in_string = not in_string
            elif not in_string and char == "[":
                depth += 1
            elif not in_string and char == "]":
                depth -= 1
            cursor += 1
        if depth:
            raise VendorError(f"unterminated {variable} assignment")
        yield from _QUOTED_RE.findall(block[match.end() : cursor - 1])


def _canonical_label(label: str, current_dir: PurePosixPath) -> str:
    if label.startswith("$"):
        raise VendorError(
            f"dynamic GN dependency is not reproducibly resolvable: {label}"
        )
    if label.startswith("//"):
        raw = label[2:]
    elif label.startswith(":"):
        raw = f"{current_dir.as_posix()}{label}"
    else:
        raw_path, separator, target = label.partition(":")
        resolved = (current_dir / raw_path).as_posix()
        normalized = str(PurePosixPath(resolved))
        while normalized.startswith("../") or "/../" in normalized:
            parts: list[str] = []
            for part in normalized.split("/"):
                if part == ".." and parts:
                    parts.pop()
                elif part not in ("", "."):
                    parts.append(part)
            normalized = "/".join(parts)
        raw = f"{normalized}:{target}" if separator else normalized
    path_text, separator, target_name = raw.partition(":")
    path = PurePosixPath(path_text)
    if not separator:
        target_name = path.name
    return f"//{path.as_posix()}:{target_name}"


def _label_parts(label: str) -> tuple[PurePosixPath, str]:
    path_text, target = label[2:].split(":", 1)
    return PurePosixPath(path_text), target


def derive_compile_closure(source: Path) -> list[str]:
    """Derive source/header closure from AEC3's checked-in GN dependency graph."""
    pending = deque(COMPILE_TARGET_ALLOWLIST)
    visited: set[str] = set()
    files: set[str] = set()

    for relative in COMPILE_SOURCE_ALLOWLIST:
        if not is_allowed_path(relative):
            raise VendorError(f"explicit compile source outside allowlist: {relative}")
        if not (source / relative).is_file():
            raise VendorError(f"explicit compile source is missing: {relative}")
        files.add(relative)

    while pending:
        label = pending.popleft()
        if label in visited:
            continue
        visited.add(label)
        target_dir, target_name = _label_parts(label)
        build_file = source / target_dir.as_posix() / "BUILD.gn"
        if not build_file.is_file():
            raise VendorError(f"missing GN build metadata for {label}: {build_file}")
        block = _target_block(build_file, target_name)
        for item in _assignment_values(block, "sources"):
            if item.startswith("//") or "$" in item:
                raise VendorError(f"unsupported source expression {item!r} in {label}")
            if item.startswith("../../webrtc_overrides/"):
                # This is the mutually exclusive build_with_chromium branch;
                # the standalone companion uses WebRTC's in-tree implementation.
                continue
            relative = (target_dir / item).as_posix()
            relative = str(PurePosixPath(relative))
            if not is_allowed_path(relative):
                raise VendorError(
                    f"compile source outside allowlist: {relative} from {label}"
                )
            if not (source / relative).is_file():
                raise VendorError(
                    f"compile source listed by {label} is missing: {relative}"
                )
            files.add(relative)
    include_pending = deque(sorted(files))
    while include_pending:
        relative = include_pending.popleft()
        data = (source / relative).read_bytes()
        for raw_include in _INCLUDE_RE.findall(data):
            include = raw_include.decode("utf-8")
            resolved_include = (
                f"third_party/abseil-cpp/{include}"
                if include.startswith("absl/")
                else include
            )
            try:
                normalized_relative_path(resolved_include)
            except VendorError as exc:
                raise VendorError(
                    f"quoted include is not normalized: {resolved_include} from {relative}"
                ) from exc
            include_path = source / resolved_include
            if not include_path.is_file():
                continue
            if not is_allowed_path(resolved_include):
                raise VendorError(
                    f"quoted include outside allowlist: {resolved_include} from {relative}"
                )
            if resolved_include not in files:
                files.add(resolved_include)
                include_pending.append(resolved_include)
    return sorted(files)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_patch_path(name: str, expected_digest: str) -> Path:
    patch = PACKAGE_ROOT / "patches" / name
    if patch.is_symlink() or not patch.is_file():
        raise VendorError(f"declared patch is missing or not a regular file: {name}")
    actual_digest = _sha256(patch)
    if actual_digest != expected_digest:
        raise VendorError(
            f"declared patch SHA-256 mismatch for {name}: "
            f"expected {expected_digest}, got {actual_digest}"
        )
    return patch


def _git_apply(vendor_root: Path, patch: Path, *options: str) -> None:
    # Temporary pristine trees have no checkout attributes to protect their bytes.
    result = subprocess.run(
        ["git", "-c", "core.autocrlf=false", "apply", *options, str(patch)],
        cwd=vendor_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise VendorError(f"declared patch {patch.name} does not apply: {detail}")


def apply_patch_series(vendor_root: Path) -> None:
    """Apply the exact declared integration patch series to copied upstream files."""
    for name, expected_digest in PATCHES:
        patch = _verified_patch_path(name, expected_digest)
        _git_apply(vendor_root, patch, "--check")
        _git_apply(vendor_root, patch)


def _verify_patch_series_applied(vendor_root: Path) -> None:
    for name, expected_digest in reversed(PATCHES):
        patch = _verified_patch_path(name, expected_digest)
        try:
            _git_apply(vendor_root, patch, "--reverse", "--check")
        except VendorError as exc:
            raise VendorError(
                f"vendored tree does not contain declared patch {name}"
            ) from exc


def _write_manifest(vendor_root: Path) -> None:
    paths = sorted(
        path.relative_to(vendor_root).as_posix()
        for path in vendor_root.rglob("*")
        if path.is_file() and path.name != MANIFEST_NAME
    )
    for path in paths:
        normalized_relative_path(path)
    contents = "".join(f"{_sha256(vendor_root / path)}  {path}\n" for path in paths)
    (vendor_root / MANIFEST_NAME).write_text(contents, encoding="utf-8")


def _write_pristine_manifest(vendor_root: Path, relative_paths: Iterable[str]) -> None:
    paths = sorted(relative_paths)
    for path in paths:
        normalized_relative_path(path)
    contents = "".join(f"{_sha256(vendor_root / path)}  {path}\n" for path in paths)
    (vendor_root / PRISTINE_MANIFEST_NAME).write_text(contents, encoding="utf-8")


def render_compiler_defines_cmake() -> str:
    """Render the CMake interface generated from the provenance mapping."""
    lines = [
        "# Generated by tools/vendor_webrtc_aec.py; do not edit.\n",
    ]
    for platform, defines in COMPILER_DEFINES.items():
        variable = f"TLDW_WEBRTC_DEFINES_{platform.upper()}"
        lines.append(f"set({variable}\n")
        lines.extend(f'  "{define}"\n' for define in defines)
        lines.append(")\n")
    lines.extend(
        [
            "\nfunction(tldw_webrtc_defines_for_platform platform output_variable)\n",
            '  if(platform STREQUAL "macos")\n',
            "    set(platform_defines ${TLDW_WEBRTC_DEFINES_MACOS})\n",
            '  elseif(platform STREQUAL "linux")\n',
            "    set(platform_defines ${TLDW_WEBRTC_DEFINES_LINUX})\n",
            '  elseif(platform STREQUAL "windows")\n',
            "    set(platform_defines ${TLDW_WEBRTC_DEFINES_WINDOWS})\n",
            "  else()\n",
            '    message(FATAL_ERROR "Unsupported WebRTC platform: ${platform}")\n',
            "  endif()\n",
            "  set(\n",
            "    ${output_variable}\n",
            "    ${TLDW_WEBRTC_DEFINES_COMMON}\n",
            "    ${platform_defines}\n",
            "    PARENT_SCOPE\n",
            "  )\n",
            "endfunction()\n",
        ]
    )
    return "".join(lines)


def _metadata(vendor_root: Path, closure_file_count: int) -> dict[str, object]:
    return {
        "repository": REPOSITORY,
        "commit": COMMIT,
        "commit_tree": COMMIT_TREE,
        "import_timestamp": COMMIT_TIMESTAMP,
        "roots": list(ALLOWED_ROOTS),
        "file_allowlist": list(ALLOWED_FILES),
        "compile_closure_file_count": closure_file_count,
        "compiler_defines": COMPILER_DEFINES,
        "license_path": "LICENSE",
        "patent_notice_path": "PATENTS",
        "notice_generation_version": NOTICE_GENERATION_VERSION,
        "third_party_dependencies": [
            {
                "name": "Abseil",
                "repository": ABSEIL_REPOSITORY,
                "commit": ABSEIL_COMMIT,
                "commit_timestamp": ABSEIL_COMMIT_TIMESTAMP,
                "subtree_tree": ABSEIL_SUBTREE_TREE,
                "path": "third_party/abseil-cpp",
                "license_path": "third_party/abseil-cpp/LICENSE",
                "license_sha256": ABSEIL_LICENSE_SHA256,
                "vendored_tree": git_tree_object_id(
                    vendor_root / "third_party/abseil-cpp"
                ),
            },
            {
                "name": "Ooura FFT",
                "source_path": OOURA_SOURCE_PATH,
                "source_url": OOURA_SOURCE_URL,
                "source_sha256": OOURA_SOURCE_SHA256,
                "license_path": "OOURA_LICENSE",
                "license_sha256": OOURA_LICENSE_SHA256,
            },
        ],
    }


def _render_notices(vendor_root: Path) -> str:
    license_text = (vendor_root / "LICENSE").read_text(encoding="utf-8").strip()
    patents_text = (vendor_root / "PATENTS").read_text(encoding="utf-8").strip()
    notices = (
        "# Third-Party Notices\n\n"
        "## WebRTC\n\n"
        f"{license_text}\n\n"
        "### WebRTC PATENTS notice\n\n"
        f"{patents_text}\n"
    )
    metadata = json.loads((vendor_root / "UPSTREAM.json").read_text(encoding="utf-8"))
    for dependency in metadata["third_party_dependencies"]:
        dependency_license = (
            (vendor_root / dependency["license_path"])
            .read_text(encoding="utf-8")
            .strip()
        )
        notices += f"\n## {dependency['name']}\n\n{dependency_license}\n"
    return notices


def _write_notices(vendor_root: Path, notices_path: Path) -> None:
    notices_path.write_text(_render_notices(vendor_root), encoding="utf-8")


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _validate_existing_outputs(vendor_root: Path, notices_path: Path) -> None:
    try:
        verify_vendor_tree(vendor_root)
    except (OSError, VendorError, json.JSONDecodeError) as exc:
        raise VendorError(
            f"existing output is not an exact verified AEC vendor tree: {vendor_root}"
        ) from exc
    if notices_path.is_symlink() or not notices_path.is_file():
        raise VendorError(
            "existing verified AEC vendor tree requires its exact regular notices file"
        )
    if notices_path.read_text(encoding="utf-8") != _render_notices(vendor_root):
        raise VendorError("existing AEC notices do not match the verified vendor tree")


def validate_destination_paths(
    source: Path, vendor_root: Path, notices_path: Path
) -> tuple[Path, Path, Path]:
    """Resolve and validate non-destructive vendoring destinations."""
    if vendor_root.is_symlink():
        raise VendorError("output leaf must not be a symlink")
    if notices_path.is_symlink():
        raise VendorError("notices output leaf must not be a symlink")

    resolved_source = source.resolve()
    resolved_vendor = vendor_root.resolve()
    resolved_notices = notices_path.resolve()
    filesystem_root = Path(resolved_vendor.anchor)
    protected_roots = {PACKAGE_ROOT.resolve(), *PACKAGE_ROOT.resolve().parents}
    if (
        resolved_vendor == filesystem_root
        or resolved_vendor.parent == filesystem_root
        or resolved_vendor in protected_roots
    ):
        raise VendorError(f"refusing broad output target: {resolved_vendor}")
    if _paths_overlap(resolved_source, resolved_vendor):
        raise VendorError("source and output paths must not overlap")
    if _paths_overlap(resolved_source, resolved_notices):
        raise VendorError("source and notices output paths must not overlap")
    if _paths_overlap(resolved_vendor, resolved_notices):
        raise VendorError("vendor and notices output paths must not overlap")

    if resolved_vendor.exists():
        if not resolved_vendor.is_dir():
            raise VendorError(
                "existing output is not an exact verified AEC vendor tree"
            )
        _validate_existing_outputs(resolved_vendor, resolved_notices)
    elif resolved_notices.exists():
        raise VendorError(
            "refusing to overwrite notices without a verified AEC vendor tree"
        )
    return resolved_source, resolved_vendor, resolved_notices


def _vacant_directory_path(parent: Path, prefix: str) -> Path:
    path = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
    path.rmdir()
    return path


def _vacant_file_path(parent: Path, prefix: str) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=prefix, dir=parent)
    os.close(descriptor)
    path = Path(name)
    path.unlink()
    return path


def atomic_replace_outputs(
    staged_vendor: Path,
    staged_notices: Path,
    vendor_root: Path,
    notices_path: Path,
) -> None:
    """Atomically swap staged outputs, restoring both old outputs on failure."""
    if staged_vendor.parent.resolve() != vendor_root.parent.resolve():
        raise VendorError("staged vendor tree must share the output parent")
    if staged_notices.parent.resolve() != notices_path.parent.resolve():
        raise VendorError("staged notices must share the notices output parent")
    verify_vendor_tree(staged_vendor)
    if staged_notices.read_text(encoding="utf-8") != _render_notices(staged_vendor):
        raise VendorError("staged notices do not match the staged vendor tree")
    if vendor_root.exists():
        _validate_existing_outputs(vendor_root, notices_path)

    vendor_backup: Path | None = None
    notices_backup: Path | None = None
    vendor_installed = False
    notices_installed = False
    try:
        if vendor_root.exists():
            vendor_backup = _vacant_directory_path(
                vendor_root.parent, f".{vendor_root.name}.backup-"
            )
            os.replace(vendor_root, vendor_backup)
        if notices_path.exists():
            notices_backup = _vacant_file_path(
                notices_path.parent, f".{notices_path.name}.backup-"
            )
            os.replace(notices_path, notices_backup)
        os.replace(staged_vendor, vendor_root)
        vendor_installed = True
        os.replace(staged_notices, notices_path)
        notices_installed = True
    except OSError as exc:
        rollback_errors: list[OSError] = []

        def rollback(source_path: Path | None, destination_path: Path) -> None:
            if source_path is None or not source_path.exists():
                return
            try:
                os.replace(source_path, destination_path)
            except OSError as rollback_error:
                rollback_errors.append(rollback_error)

        if notices_installed:
            rollback(notices_path, staged_notices)
        rollback(notices_backup, notices_path)
        if vendor_installed:
            rollback(vendor_root, staged_vendor)
        rollback(vendor_backup, vendor_root)
        if rollback_errors:
            raise VendorError(
                "atomic output replacement and rollback failed; validated backups were preserved"
            ) from exc
        raise VendorError(
            "atomic output replacement failed; original outputs were restored"
        ) from exc
    else:
        if vendor_backup is not None:
            shutil.rmtree(vendor_backup)
        if notices_backup is not None:
            notices_backup.unlink()
    finally:
        if staged_vendor.exists():
            shutil.rmtree(staged_vendor)
        if staged_notices.exists():
            staged_notices.unlink()


def generate_vendor_tree(source: Path, vendor_root: Path, notices_path: Path) -> None:
    source, vendor_root, notices_path = validate_destination_paths(
        source, vendor_root, notices_path
    )
    closure = derive_compile_closure(source)
    vendor_root.parent.mkdir(parents=True, exist_ok=True)
    notices_path.parent.mkdir(parents=True, exist_ok=True)
    generated = Path(
        tempfile.mkdtemp(prefix=f".{vendor_root.name}.stage-", dir=vendor_root.parent)
    )
    descriptor, staged_notices_name = tempfile.mkstemp(
        prefix=f".{notices_path.name}.stage-", dir=notices_path.parent
    )
    os.close(descriptor)
    staged_notices = Path(staged_notices_name)
    try:
        for relative in closure:
            destination = generated / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, destination)
        abseil_license = generated / "third_party/abseil-cpp/LICENSE"
        abseil_license.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / "third_party/abseil-cpp/LICENSE", abseil_license)
        shutil.copyfile(source / "LICENSE", generated / "LICENSE")
        shutil.copyfile(source / "PATENTS", generated / "PATENTS")
        (generated / "OOURA_LICENSE").write_text(
            extract_ooura_notice(source / OOURA_SOURCE_PATH), encoding="utf-8"
        )
        (generated / "COMPILER_DEFINES.cmake").write_text(
            render_compiler_defines_cmake(), encoding="utf-8"
        )
        _write_pristine_manifest(generated, closure)
        apply_patch_series(generated)
        (generated / "PATCHES.md").write_text(PATCH_SERIES, encoding="utf-8")
        (generated / "UPSTREAM.json").write_text(
            json.dumps(_metadata(generated, len(closure)), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        _write_manifest(generated)
        verify_vendor_tree(generated)
        _write_notices(generated, staged_notices)
        atomic_replace_outputs(generated, staged_notices, vendor_root, notices_path)
    finally:
        if generated.exists():
            shutil.rmtree(generated)
        if staged_notices.exists():
            staged_notices.unlink()


def _manifest_entries(vendor_root: Path) -> dict[str, str]:
    manifest = vendor_root / MANIFEST_NAME
    if not manifest.is_file():
        raise VendorError("integrity manifest is missing")
    entries: dict[str, str] = {}
    lines = manifest.read_text(encoding="utf-8").splitlines()
    for line in lines:
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise VendorError(
                f"integrity manifest line is malformed: {line!r}"
            ) from exc
        normalized_relative_path(relative)
        if relative in entries or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise VendorError(f"integrity manifest entry is invalid: {line!r}")
        entries[relative] = digest
    if list(entries) != sorted(entries):
        raise VendorError("integrity manifest is not sorted")
    return entries


def _pristine_manifest_entries(vendor_root: Path) -> dict[str, str]:
    manifest = vendor_root / PRISTINE_MANIFEST_NAME
    if manifest.is_symlink() or not manifest.is_file():
        raise VendorError("pristine upstream manifest is missing")
    if _sha256(manifest) != PRISTINE_MANIFEST_SHA256:
        raise VendorError("pristine upstream manifest does not match its tool anchor")

    entries: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise VendorError(
                f"pristine upstream manifest line is malformed: {line!r}"
            ) from exc
        normalized_relative_path(relative)
        if relative in entries or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise VendorError(f"pristine upstream manifest entry is invalid: {line!r}")
        entries[relative] = digest
    if list(entries) != sorted(entries):
        raise VendorError("pristine upstream manifest is not sorted")
    if len(entries) != COMPILE_CLOSURE_FILE_COUNT:
        raise VendorError(
            "pristine upstream manifest does not contain the exact compile closure"
        )
    return entries


def _verify_pristine_upstream_closure(
    vendor_root: Path, patched_entries: dict[str, str]
) -> None:
    pristine_entries = _pristine_manifest_entries(vendor_root)
    source_paths = {
        relative
        for relative in patched_entries
        if relative not in METADATA_FILES
        and relative != "third_party/abseil-cpp/LICENSE"
    }
    if source_paths != set(pristine_entries):
        raise VendorError(
            "vendored source set does not match the pristine upstream closure"
        )

    with tempfile.TemporaryDirectory(prefix="tldw-webrtc-pristine-") as temporary:
        pristine_root = Path(temporary)
        for relative in pristine_entries:
            destination = pristine_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(vendor_root / relative, destination)
        for name, expected_digest in reversed(PATCHES):
            _git_apply(
                pristine_root,
                _verified_patch_path(name, expected_digest),
                "--reverse",
            )
        for relative, expected_digest in pristine_entries.items():
            if _sha256(pristine_root / relative) != expected_digest:
                raise VendorError(
                    "vendored source differs from pristine upstream outside "
                    f"the declared patch series: {relative}"
                )


def verify_vendor_tree(vendor_root: Path) -> None:
    """Raise ``VendorError`` for missing, extra, edited, or disallowed files."""
    if vendor_root.is_symlink() or not vendor_root.is_dir():
        raise VendorError("vendor root must be a regular directory")
    paths = list(vendor_root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise VendorError("integrity check found a forbidden symlink")
    entries = _manifest_entries(vendor_root)
    actual = {
        path.relative_to(vendor_root).as_posix()
        for path in paths
        if path.is_file() and path.name != MANIFEST_NAME
    }
    expected = set(entries)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise VendorError(f"integrity check found missing files: {missing}")
    if extra:
        raise VendorError(f"integrity check found extra files: {extra}")
    for relative, digest in entries.items():
        if relative not in METADATA_FILES and not is_allowed_path(relative):
            raise VendorError(f"allowlist violation in vendored path: {relative}")
        if _sha256(vendor_root / relative) != digest:
            raise VendorError(f"integrity hash mismatch for {relative}")
    if any(path.name == ".git" for path in vendor_root.rglob("*")):
        raise VendorError("integrity check found forbidden .git metadata")
    if (vendor_root / "PATCHES.md").read_text(encoding="utf-8") != PATCH_SERIES:
        raise VendorError("integrity check found an undeclared patch series")
    _verify_patch_series_applied(vendor_root)
    _verify_pristine_upstream_closure(vendor_root, entries)
    if _sha256(vendor_root / "LICENSE") != WEBRTC_LICENSE_SHA256:
        raise VendorError("integrity check found the wrong WebRTC license")
    if _sha256(vendor_root / "PATENTS") != WEBRTC_PATENTS_SHA256:
        raise VendorError("integrity check found the wrong WebRTC patent notice")
    if _sha256(vendor_root / "OOURA_LICENSE") != OOURA_LICENSE_SHA256:
        raise VendorError("integrity check found the wrong Ooura notice")
    metadata = json.loads((vendor_root / "UPSTREAM.json").read_text(encoding="utf-8"))
    expected_metadata = _metadata(vendor_root, COMPILE_CLOSURE_FILE_COUNT)
    if metadata != expected_metadata:
        raise VendorError("integrity check found incorrect pinned provenance metadata")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--source",
        type=Path,
        help="prepared WebRTC source checkout with its pinned Abseil subtree",
    )
    mode.add_argument(
        "--verify-vendor-tree",
        action="store_true",
        help="offline verification of the fixed checked-in vendor tree",
    )
    parser.add_argument(
        "--verify-clean",
        action="store_true",
        help="require the prepared source checkout to have no local changes",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.verify_vendor_tree and (args.verify_clean or args.output is not None):
        parser.error(
            "--verify-vendor-tree cannot be combined with generation arguments"
        )
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify_vendor_tree:
        verify_vendor_tree(DEFAULT_VENDOR_ROOT)
        return 0

    assert args.source is not None
    source = args.source.resolve()
    output = args.output or DEFAULT_VENDOR_ROOT
    if output.is_symlink():
        raise VendorError("output leaf must not be a symlink")
    vendor_root = output.resolve()
    notices_path = (
        DEFAULT_NOTICES_PATH.resolve()
        if vendor_root == DEFAULT_VENDOR_ROOT.resolve()
        else vendor_root.parent / "THIRD_PARTY_NOTICES.md"
    )
    validate_source(source, verify_clean=args.verify_clean)
    generate_vendor_tree(source, vendor_root, notices_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (subprocess.CalledProcessError, VendorError) as exc:
        raise SystemExit(f"error: {exc}") from exc
