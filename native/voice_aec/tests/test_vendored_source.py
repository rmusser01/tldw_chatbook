"""Integrity and provenance tests for the vendored WebRTC AEC3 closure."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = PACKAGE_ROOT / "vendor" / "webrtc"
UPSTREAM_PATH = VENDOR_ROOT / "UPSTREAM.json"
MANIFEST_PATH = VENDOR_ROOT / "FILES.sha256"
PRISTINE_MANIFEST_PATH = VENDOR_ROOT / "PRISTINE_FILES.sha256"
PATCHES_PATH = VENDOR_ROOT / "PATCHES.md"
PATCH_NAME = "0001-expose-delay-health-evidence.patch"
CLOCKDRIFT_PATCH_NAME = "0002-include-stddef-for-clockdrift-detector.patch"
CLOCKDRIFT_HEADER = "modules/audio_processing/aec3/clockdrift_detector.h"
REVERB_PATCH_NAME = "0003-include-memory-for-reverb-model-estimator.patch"
REVERB_HEADER = "modules/audio_processing/aec3/reverb_model_estimator.h"
COMPILER_DEFINES_PATH = VENDOR_ROOT / "COMPILER_DEFINES.cmake"
NOTICES_PATH = PACKAGE_ROOT / "THIRD_PARTY_NOTICES.md"
TOOL_PATH = PACKAGE_ROOT / "tools" / "vendor_webrtc_aec.py"

REPOSITORY = "https://webrtc.googlesource.com/src"
COMMIT = "109e23c9cec3a44e67c08774874a409741b1e58a"
TREE = "71115034a67c1e7f98a4c2a61d80278db5a9b2ae"
COMMIT_TIMESTAMP = "2020-03-02T12:42:42+00:00"
BROAD_ROOTS = [
    "api/audio",
    "common_audio",
    "modules/audio_processing",
    "rtc_base",
    "system_wrappers",
    "third_party/abseil-cpp",
]
FILE_ALLOWLIST = [
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
]
ABSEIL_REPOSITORY = "https://chromium.googlesource.com/chromium/src/third_party"
ABSEIL_COMMIT = "ac875ae5393d0516243cfd5d078cd4b098388f6b"
ABSEIL_TIMESTAMP = "2020-02-28T15:49:05+00:00"
ABSEIL_SUBTREE = "7f84a7844f32a5a56a02bc7133e943b0932c50b2"
ABSEIL_LICENSE_SHA256 = (
    "c79a7fea0e3cac04cd43f20e7b648e5a0ff8fa5344e644b0ee09ca1162b62747"
)
OOURA_SOURCE_PATH = "modules/audio_processing/utility/ooura_fft.cc"
OOURA_SOURCE_URL = "http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html"
OOURA_SOURCE_SHA256 = "8a07c7cffe3471fbbd9b734cf658f7d0ed28904e4f828179dbebf650501040bd"
OOURA_LICENSE_SHA256 = (
    "efa6b38d923e14333047a48580043d366dc51e5d0ebd7d658fa341ad3aafb576"
)
DEFINES = {
    "common": ["WEBRTC_APM_DEBUG_DUMP=0", "WEBRTC_ENABLE_PROTOBUF=0"],
    "macos": ["WEBRTC_MAC", "WEBRTC_POSIX"],
    "linux": ["WEBRTC_LINUX", "WEBRTC_POSIX"],
    "windows": ["WEBRTC_WIN", "NOMINMAX", "_CRT_SECURE_NO_WARNINGS"],
}


def _load_tool(path: Path = TOOL_PATH):
    spec = importlib.util.spec_from_file_location("vendor_webrtc_aec", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metadata() -> dict[str, object]:
    return json.loads(UPSTREAM_PATH.read_text(encoding="utf-8"))


def _manifest() -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        digest, relative_path = line.split("  ", 1)
        entries.append((digest, relative_path))
    return entries


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _configure_copied_package(
    package_root: Path, build_root: Path
) -> subprocess.CompletedProcess[str]:
    cmake = shutil.which("cmake")
    assert cmake is not None, "CMake is required for native build-integrity tests"
    return subprocess.run(
        [cmake, "-S", str(package_root), "-B", str(build_root)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_upstream_metadata_pins_exact_official_revision() -> None:
    metadata = _metadata()

    assert metadata["repository"] == REPOSITORY
    assert metadata["commit"] == COMMIT
    assert metadata["commit_tree"] == TREE
    assert metadata["import_timestamp"] == COMMIT_TIMESTAMP
    assert metadata["roots"] == BROAD_ROOTS
    assert metadata["file_allowlist"] == FILE_ALLOWLIST
    assert metadata["compile_closure_file_count"] == 316
    assert metadata["compiler_defines"] == DEFINES
    assert metadata["license_path"] == "LICENSE"
    assert metadata["patent_notice_path"] == "PATENTS"
    assert metadata["notice_generation_version"] == 2


def test_compiler_defines_are_generated_from_exact_provenance_mapping() -> None:
    tool = _load_tool()

    assert tool.COMPILER_DEFINES == DEFINES
    assert (
        COMPILER_DEFINES_PATH.read_text(encoding="utf-8")
        == tool.render_compiler_defines_cmake()
    )
    assert "COMPILER_DEFINES.cmake" in {path for _, path in _manifest()}


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        ("macos", DEFINES["common"] + DEFINES["macos"]),
        ("linux", DEFINES["common"] + DEFINES["linux"]),
        ("windows", DEFINES["common"] + DEFINES["windows"]),
    ],
)
def test_generated_cmake_selects_exact_platform_defines(
    tmp_path: Path, platform: str, expected: list[str]
) -> None:
    cmake = shutil.which("cmake")
    assert cmake is not None, "CMake is required for provenance selection tests"
    script = tmp_path / "select-defines.cmake"
    script.write_text(
        f'include("{COMPILER_DEFINES_PATH.as_posix()}")\n'
        f'tldw_webrtc_defines_for_platform("{platform}" selected)\n'
        'list(JOIN selected "|" joined)\n'
        'message(STATUS "SELECTED=${joined}")\n',
        encoding="utf-8",
    )

    result = subprocess.run(
        [cmake, "-P", str(script)], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"-- SELECTED={'|'.join(expected)}" in result.stdout + result.stderr


def test_metadata_uses_only_immutable_concrete_values() -> None:
    encoded = UPSTREAM_PATH.read_text(encoding="utf-8").lower()

    assert "main" not in encoded
    assert "master" not in encoded
    assert "placeholder" not in encoded
    assert len(COMMIT) == 40
    assert len(TREE) == 40


def test_manifest_is_sorted_complete_and_hash_correct() -> None:
    entries = _manifest()
    paths = [relative_path for _, relative_path in entries]
    files = sorted(
        path.relative_to(VENDOR_ROOT).as_posix()
        for path in VENDOR_ROOT.rglob("*")
        if path.is_file() and path != MANIFEST_PATH
    )

    assert paths == sorted(paths)
    assert len(paths) == len(set(paths))
    assert paths == files
    assert all(len(digest) == 64 and digest.isalnum() for digest, _ in entries)
    assert all(
        _sha256(VENDOR_ROOT / relative_path) == digest
        for digest, relative_path in entries
    )


def test_vendor_tree_contains_no_git_metadata() -> None:
    assert not any(path.name == ".git" for path in VENDOR_ROOT.rglob("*"))


def test_every_copied_upstream_path_is_allowlisted() -> None:
    metadata_files = {
        "COMPILER_DEFINES.cmake",
        "LICENSE",
        "OOURA_LICENSE",
        "PATENTS",
        "PATCHES.md",
        "PRISTINE_FILES.sha256",
        "UPSTREAM.json",
    }
    copied_paths = [
        path.relative_to(VENDOR_ROOT).as_posix()
        for path in VENDOR_ROOT.rglob("*")
        if path.is_file() and path.name != "FILES.sha256"
    ]

    for relative_path in copied_paths:
        if relative_path in metadata_files:
            continue
        assert relative_path in FILE_ALLOWLIST or any(
            relative_path == root or relative_path.startswith(f"{root}/")
            for root in BROAD_ROOTS
        ), relative_path


def test_non_exception_api_path_is_rejected() -> None:
    tool = _load_tool()

    assert tool.is_allowed_path("api/rtp_packet_infos.h")
    assert not tool.is_allowed_path("api/create_peerconnection_factory.cc")


@pytest.mark.parametrize(
    "relative_path",
    [
        "",
        ".",
        "..",
        "/api/audio/audio_frame.h",
        "C:/api/audio/audio_frame.h",
        "api\\audio\\audio_frame.h",
        "api//audio/audio_frame.h",
        "api/audio/./audio_frame.h",
        "api/audio/../outside.h",
        "api/audio/../../outside.cc",
        "api/audio/audio_frame.h/",
    ],
)
def test_allowlist_rejects_non_normalized_paths(relative_path: str) -> None:
    assert not _load_tool().is_allowed_path(relative_path)


def test_manifest_rejects_a_traversal_path(tmp_path: Path) -> None:
    copied = tmp_path / "webrtc"
    shutil.copytree(VENDOR_ROOT, copied)
    lines = (copied / "FILES.sha256").read_text(encoding="utf-8").splitlines()
    digest, _ = lines[0].split("  ", 1)
    lines[0] = f"{digest}  ../outside.cc"
    (copied / "FILES.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="normalized relative POSIX"):
        _load_tool().verify_vendor_tree(copied)


def test_cmake_configure_rejects_a_modified_vendored_file(tmp_path: Path) -> None:
    copied_package = tmp_path / "voice_aec"
    shutil.copytree(PACKAGE_ROOT, copied_package)
    modified = copied_package / "vendor" / "webrtc" / "api" / "array_view.h"
    modified.write_bytes(modified.read_bytes() + b"\n// tampered\n")

    result = _configure_copied_package(copied_package, tmp_path / "build")

    assert result.returncode != 0
    assert "WebRTC manifest SHA-256 mismatch for api/array_view.h" in (
        result.stdout + result.stderr
    )


def test_cmake_configure_rejects_a_modified_header_removed_from_manifest(
    tmp_path: Path,
) -> None:
    copied_package = tmp_path / "voice_aec"
    shutil.copytree(PACKAGE_ROOT, copied_package)
    modified = copied_package / "vendor" / "webrtc" / "api" / "array_view.h"
    modified.write_bytes(modified.read_bytes() + b"\n// tampered\n")
    manifest = copied_package / "vendor" / "webrtc" / "FILES.sha256"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    manifest.write_text(
        "\n".join(line for line in lines if not line.endswith("  api/array_view.h"))
        + "\n",
        encoding="utf-8",
    )

    result = _configure_copied_package(copied_package, tmp_path / "build")

    assert result.returncode != 0
    assert "WebRTC vendor tree has unmanifested files: api/array_view.h" in (
        result.stdout + result.stderr
    )


def test_cmake_configure_rejects_compiler_defines_removed_from_manifest(
    tmp_path: Path,
) -> None:
    copied_package = tmp_path / "voice_aec"
    shutil.copytree(PACKAGE_ROOT, copied_package)
    manifest = copied_package / "vendor" / "webrtc" / "FILES.sha256"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    manifest.write_text(
        "\n".join(
            line for line in lines if not line.endswith("  COMPILER_DEFINES.cmake")
        )
        + "\n",
        encoding="utf-8",
    )

    result = _configure_copied_package(copied_package, tmp_path / "build")

    assert result.returncode != 0
    assert "WebRTC vendor tree has unmanifested files: COMPILER_DEFINES.cmake" in (
        result.stdout + result.stderr
    )


def test_cmake_configure_rejects_an_unmanifested_extra_regular_file(
    tmp_path: Path,
) -> None:
    copied_package = tmp_path / "voice_aec"
    shutil.copytree(PACKAGE_ROOT, copied_package)
    extra = copied_package / "vendor" / "webrtc" / "rtc_base" / "unmanifested.cc"
    extra.write_text("// not in FILES.sha256\n", encoding="utf-8")

    result = _configure_copied_package(copied_package, tmp_path / "build")

    assert result.returncode != 0
    assert "WebRTC vendor tree has unmanifested files: rtc_base/unmanifested.cc" in (
        result.stdout + result.stderr
    )


def test_cmake_configure_rejects_an_unmanifested_symlink(tmp_path: Path) -> None:
    copied_package = tmp_path / "voice_aec"
    shutil.copytree(PACKAGE_ROOT, copied_package)
    extra = copied_package / "vendor" / "webrtc" / "api" / "unmanifested-link.h"
    extra.symlink_to("array_view.h")

    result = _configure_copied_package(copied_package, tmp_path / "build")

    assert result.returncode != 0
    assert "WebRTC vendor tree contains a symlink: api/unmanifested-link.h" in (
        result.stdout + result.stderr
    )


def test_cmake_configure_rejects_a_traversal_manifest_entry(tmp_path: Path) -> None:
    copied_package = tmp_path / "voice_aec"
    shutil.copytree(PACKAGE_ROOT, copied_package)
    manifest = copied_package / "vendor" / "webrtc" / "FILES.sha256"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    digest, _ = lines[0].split("  ", 1)
    lines[0] = f"{digest}  ../outside.cc"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = _configure_copied_package(copied_package, tmp_path / "build")

    assert result.returncode != 0
    assert "WebRTC manifest path is not normalized relative POSIX" in (
        result.stdout + result.stderr
    )


def test_destination_rejects_a_symlink_output_leaf(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    target = tmp_path / "real-target"
    target.mkdir()
    output = tmp_path / "webrtc"
    output.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        _load_tool().validate_destination_paths(
            source, output, tmp_path / "THIRD_PARTY_NOTICES.md"
        )


@pytest.mark.parametrize("relationship", ["same", "inside_source", "contains_source"])
def test_destination_rejects_source_overlap(tmp_path: Path, relationship: str) -> None:
    source = tmp_path / "source"
    source.mkdir()
    if relationship == "same":
        output = source
    elif relationship == "inside_source":
        output = source / "vendor" / "webrtc"
    else:
        output = tmp_path

    with pytest.raises(ValueError, match="overlap"):
        _load_tool().validate_destination_paths(
            source, output, tmp_path / "THIRD_PARTY_NOTICES.md"
        )


def test_destination_rejects_filesystem_root(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    filesystem_root = Path(source.anchor)

    with pytest.raises(ValueError, match="broad"):
        _load_tool().validate_destination_paths(
            source, filesystem_root, filesystem_root / "THIRD_PARTY_NOTICES.md"
        )


def test_destination_rejects_unverified_existing_target_without_deleting_it(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "webrtc"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("do not delete\n", encoding="utf-8")

    with pytest.raises(ValueError, match="verified AEC vendor tree"):
        _load_tool().validate_destination_paths(
            source, output, tmp_path / "THIRD_PARTY_NOTICES.md"
        )
    assert sentinel.read_text(encoding="utf-8") == "do not delete\n"


def test_destination_accepts_repeat_replacement_of_verified_tree(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "webrtc"
    notices = tmp_path / "THIRD_PARTY_NOTICES.md"
    shutil.copytree(VENDOR_ROOT, output)
    shutil.copyfile(NOTICES_PATH, notices)

    assert _load_tool().validate_destination_paths(source, output, notices) == (
        source.resolve(),
        output.resolve(),
        notices.resolve(),
    )


def test_atomic_replacement_rolls_back_tree_and_notice_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool()
    output = tmp_path / "webrtc"
    notices = tmp_path / "THIRD_PARTY_NOTICES.md"
    staged_output = tmp_path / ".webrtc.stage-test"
    staged_notices = tmp_path / ".THIRD_PARTY_NOTICES.stage-test"
    shutil.copytree(VENDOR_ROOT, output)
    shutil.copyfile(NOTICES_PATH, notices)
    shutil.copytree(VENDOR_ROOT, staged_output)
    shutil.copyfile(NOTICES_PATH, staged_notices)
    original_manifest = (output / "FILES.sha256").read_bytes()
    original_notices = notices.read_bytes()
    real_replace = tool.os.replace
    real_rmtree = tool.shutil.rmtree

    def fail_notice_install(source: object, destination: object) -> None:
        if Path(source) == staged_notices and Path(destination) == notices:
            raise OSError("injected notice install failure")
        real_replace(source, destination)

    def guard_user_target(path: object, *args: object, **kwargs: object) -> None:
        assert Path(path) != output
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(tool.os, "replace", fail_notice_install)
    monkeypatch.setattr(tool.shutil, "rmtree", guard_user_target)

    with pytest.raises(ValueError, match="replacement failed"):
        tool.atomic_replace_outputs(staged_output, staged_notices, output, notices)

    assert (output / "FILES.sha256").read_bytes() == original_manifest
    assert notices.read_bytes() == original_notices
    tool.verify_vendor_tree(output)


def test_cli_requires_one_explicit_mode() -> None:
    with pytest.raises(SystemExit):
        _load_tool().parse_args([])


def test_offline_verification_cli_checks_the_checked_in_vendor_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _load_tool()
    observed: list[Path] = []
    monkeypatch.setattr(tool, "verify_vendor_tree", observed.append)

    assert tool.main(["--verify-vendor-tree"]) == 0
    assert observed == [tool.DEFAULT_VENDOR_ROOT]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--verify-vendor-tree", "--source", "source"],
        ["--verify-vendor-tree", "--output", "elsewhere"],
        ["--verify-vendor-tree", "--verify-clean"],
    ],
)
def test_offline_verification_cli_rejects_ambiguous_generation_arguments(
    arguments: list[str],
) -> None:
    with pytest.raises(SystemExit):
        _load_tool().parse_args(arguments)


def test_custom_vendor_output_uses_an_adjacent_notices_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool()
    observed: dict[str, Path] = {}
    source = tmp_path / "source"
    output = tmp_path / "generated" / "webrtc"

    monkeypatch.setattr(tool, "validate_source", lambda *_args, **_kwargs: None)

    def record_generation(
        prepared_source: Path, vendor_root: Path, notices_path: Path
    ) -> None:
        observed["source"] = prepared_source
        observed["vendor_root"] = vendor_root
        observed["notices_path"] = notices_path

    monkeypatch.setattr(tool, "generate_vendor_tree", record_generation)

    assert tool.main(["--source", str(source), "--output", str(output)]) == 0
    assert observed == {
        "source": source.resolve(),
        "vendor_root": output.resolve(),
        "notices_path": (output.parent / "THIRD_PARTY_NOTICES.md").resolve(),
    }


def test_compile_closure_contains_exactly_316_files() -> None:
    metadata = _metadata()
    entries = {relative_path for _, relative_path in _manifest()}
    non_closure = {
        "COMPILER_DEFINES.cmake",
        "LICENSE",
        "OOURA_LICENSE",
        "PATENTS",
        "PATCHES.md",
        "PRISTINE_FILES.sha256",
        "UPSTREAM.json",
        "third_party/abseil-cpp/LICENSE",
    }

    assert len(entries - non_closure) == 316
    assert metadata["compile_closure_file_count"] == len(entries - non_closure)


@pytest.mark.parametrize(
    "relative_path",
    [
        "api/audio/audio_frame.cc",
        "rtc_base/race_checker.cc",
        "rtc_base/critical_section.cc",
        "rtc_base/platform_thread_types.cc",
        "rtc_base/time_utils.cc",
        "rtc_base/memory/aligned_malloc.cc",
    ],
)
def test_native_link_implementations_are_in_compile_closure(relative_path: str) -> None:
    assert (VENDOR_ROOT / relative_path).is_file(), relative_path


def test_compile_closure_excludes_unapproved_api_implementations() -> None:
    assert not (VENDOR_ROOT / "api/rtp_packet_info.cc").exists()
    assert not (VENDOR_ROOT / "api/task_queue").exists()


def test_cpu_feature_implementation_is_exact_pinned_compile_source() -> None:
    relative = "system_wrappers/source/cpu_features.cc"
    tool = _load_tool()
    assert relative in tool.COMPILE_SOURCE_ALLOWLIST
    source = (VENDOR_ROOT / relative).read_bytes()
    assert len(source) == 2044
    assert hashlib.sha256(source).hexdigest() == (
        "e4bac0600ca4a36436431db0e1377886c98a5362eb2a403a40c83dc53b85f643"
    )
    assert hashlib.sha1(b"blob 2044\0" + source).hexdigest() == (
        "ebcb48c15fb20ddeda6c5844e097d7b2835cbd81"
    )
    assert tool._pristine_manifest_entries(VENDOR_ROOT)[relative] == _sha256(
        VENDOR_ROOT / relative
    )
    old_ledger = b"".join(
        line
        for line in PRISTINE_MANIFEST_PATH.read_bytes().splitlines(keepends=True)
        if not line.endswith(f"  {relative}\n".encode())
    )
    assert hashlib.sha256(old_ledger).hexdigest() == (
        "e9a42c702eea0c1b0faa6aa0d11ba8b501c7c052f18ee837a77923f81e12314c"
    )


def test_every_quoted_include_resolves_inside_vendor_tree() -> None:
    include_pattern = re.compile(rb'^\s*#\s*include\s*"([^"\n]+)"', re.MULTILINE)
    quoted_system_headers = {"stddef.h"}
    missing: set[tuple[str, str]] = set()

    assert (VENDOR_ROOT / "third_party/abseil-cpp/absl/meta/type_traits.h").is_file(), (
        "missing quoted dependency absl/meta/type_traits.h"
    )

    for _, relative_path in _manifest():
        source_path = VENDOR_ROOT / relative_path
        for raw_include in include_pattern.findall(source_path.read_bytes()):
            include = raw_include.decode("utf-8")
            if include in quoted_system_headers:
                continue
            candidate = VENDOR_ROOT / include
            if include.startswith("absl/"):
                candidate = VENDOR_ROOT / "third_party" / "abseil-cpp" / include
            if not candidate.is_file():
                missing.add((relative_path, include))

    assert not missing, sorted(missing)


def test_patch_series_declares_three_hash_verified_patches() -> None:
    tool = _load_tool()
    assert tool.PATCHES == (
        (
            PATCH_NAME,
            "94f2a8dad384194c8b3ffb63695287ae7a046e1136b4357b9b3d048579ad3a1c",
        ),
        (
            CLOCKDRIFT_PATCH_NAME,
            "f3bbfe1b2f7d54030fba05fefefbcc181706975c09aa43c9aeb83ca96379ca6e",
        ),
        (
            REVERB_PATCH_NAME,
            "dfdad20c0732cf30364d6a4c5161e9e2612dc8c0123c511ea2537e2439c9e9f5",
        ),
    )
    expected = "# WebRTC AEC3 patch series\n\n"
    for number, (name, digest) in enumerate(tool.PATCHES, 1):
        assert _sha256(PACKAGE_ROOT / "patches" / name) == digest
        expected += f"{number}. `{name}` - SHA-256 `{digest}`\n"

    assert PATCHES_PATH.read_text(encoding="utf-8") == expected
    assert tool.PATCH_SERIES == expected


@pytest.mark.parametrize(
    "patch_name", [PATCH_NAME, CLOCKDRIFT_PATCH_NAME, REVERB_PATCH_NAME]
)
def test_declared_patch_has_no_git_whitespace_errors(
    tmp_path: Path, patch_name: str
) -> None:
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    result = subprocess.run(
        [
            "git",
            "diff",
            "--no-index",
            "--check",
            str(empty),
            str(PACKAGE_ROOT / "patches" / patch_name),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout == ""
    assert result.stderr == ""


def test_vendor_recipe_reapplies_declared_patch_to_exact_unpatched_tree(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "webrtc"
    shutil.copytree(VENDOR_ROOT, copied)
    tool = _load_tool()
    pristine = tool._pristine_manifest_entries(copied)
    for name, digest in reversed(tool.PATCHES):
        tool._git_apply(copied, tool._verified_patch_path(name, digest), "--reverse")
    assert all(_sha256(copied / path) == digest for path, digest in pristine.items())

    tool.apply_patch_series(copied)

    for relative_path in pristine:
        assert (copied / relative_path).read_bytes() == (
            VENDOR_ROOT / relative_path
        ).read_bytes()


@pytest.mark.parametrize(
    "patch_name", [PATCH_NAME, CLOCKDRIFT_PATCH_NAME, REVERB_PATCH_NAME]
)
def test_vendor_verifier_rejects_rehashed_tree_without_declared_patch(
    tmp_path: Path,
    patch_name: str,
) -> None:
    copied = tmp_path / "webrtc"
    shutil.copytree(VENDOR_ROOT, copied)
    subprocess.run(
        ["git", "apply", "--reverse", str(PACKAGE_ROOT / "patches" / patch_name)],
        cwd=copied,
        check=True,
        capture_output=True,
        text=True,
    )
    tool = _load_tool()
    tool._write_manifest(copied)

    with pytest.raises(ValueError, match="declared patch"):
        tool.verify_vendor_tree(copied)


def test_clockdrift_header_directly_includes_global_size_t_definition() -> None:
    assert b"#include <stddef.h>\n" in (VENDOR_ROOT / CLOCKDRIFT_HEADER).read_bytes()
    _load_tool().verify_vendor_tree(VENDOR_ROOT)


def test_reverb_header_directly_includes_unique_ptr_definition() -> None:
    assert b"#include <memory>\n" in (VENDOR_ROOT / REVERB_HEADER).read_bytes()
    _load_tool().verify_vendor_tree(VENDOR_ROOT)


@pytest.mark.parametrize(
    "patch_name", [PATCH_NAME, CLOCKDRIFT_PATCH_NAME, REVERB_PATCH_NAME]
)
def test_vendor_verifier_rejects_changed_patch_bytes(
    tmp_path: Path, patch_name: str
) -> None:
    shutil.copytree(PACKAGE_ROOT / "patches", tmp_path / "patches")
    (tmp_path / "tools").mkdir()
    copied_tool = tmp_path / "tools" / TOOL_PATH.name
    shutil.copyfile(TOOL_PATH, copied_tool)
    patch = tmp_path / "patches" / patch_name
    patch.write_bytes(patch.read_bytes().replace(b"\n", b"\r\n"))

    with pytest.raises(ValueError, match="declared patch SHA-256 mismatch"):
        _load_tool(copied_tool).verify_vendor_tree(VENDOR_ROOT)


def test_autocrlf_checkout_preserves_full_vendor_and_external_legal_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Git apply also reads global settings in the out-of-repository pristine stage.
    global_config = tmp_path / "gitconfig"
    global_config.write_text("[core]\n\tautocrlf = true\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    repository = tmp_path / "repository"
    package = repository / "native" / "voice_aec"
    package.mkdir(parents=True)
    for directory in ("vendor", "patches", "tools"):
        shutil.copytree(PACKAGE_ROOT / directory, package / directory)
    for name in (".gitattributes", "PYBIND11_LICENSE.txt", "THIRD_PARTY_NOTICES.md"):
        shutil.copyfile(PACKAGE_ROOT / name, package / name)
    (repository / "unprotected.txt").write_bytes(b"CRLF control\nsecond line\n")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    for args in (
        ("init", "--quiet"),
        ("-c", "core.autocrlf=false", "add", "."),
        (
            "-c",
            "core.autocrlf=true",
            "checkout-index",
            "--all",
            f"--prefix={checkout.as_posix()}/",
        ),
    ):
        subprocess.run(
            ["git", "-c", "gc.auto=0", *args],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )

    assert (
        checkout / "unprotected.txt"
    ).read_bytes() == b"CRLF control\r\nsecond line\r\n"
    checked_package = checkout / "native" / "voice_aec"
    tool = _load_tool(checked_package / "tools" / TOOL_PATH.name)
    assert tool.PACKAGE_ROOT == checked_package
    tool.verify_vendor_tree(checked_package / "vendor" / "webrtc")
    for name in ("PYBIND11_LICENSE.txt", "THIRD_PARTY_NOTICES.md"):
        assert (checked_package / name).read_bytes() == (
            PACKAGE_ROOT / name
        ).read_bytes()
    assert _sha256(checked_package / "PYBIND11_LICENSE.txt") == (
        "83965b843b98f670d3a85bd041ed4b372c8ec50d7b4a5995a83ac697ba675dcb"
    )


def test_pristine_manifest_is_independently_anchored_and_complete() -> None:
    tool = _load_tool()

    assert PRISTINE_MANIFEST_PATH.is_file()
    assert _sha256(PRISTINE_MANIFEST_PATH) == tool.PRISTINE_MANIFEST_SHA256
    assert len(tool._pristine_manifest_entries(VENDOR_ROOT)) == 316


@pytest.mark.parametrize(
    "relative_path", ["rtc_base/checks.h", "system_wrappers/source/cpu_features.cc"]
)
def test_vendor_verifier_rejects_rehashed_undeclared_source_edit(
    tmp_path: Path,
    relative_path: str,
) -> None:
    copied = tmp_path / "webrtc"
    shutil.copytree(VENDOR_ROOT, copied)
    source = copied / relative_path
    source.write_bytes(source.read_bytes() + b"\n// undeclared source edit\n")
    tool = _load_tool()
    tool._write_manifest(copied)

    with pytest.raises(ValueError, match="pristine upstream"):
        tool.verify_vendor_tree(copied)


def test_license_and_patents_are_concrete_pinned_copies() -> None:
    assert _sha256(VENDOR_ROOT / "LICENSE") == (
        "ab00a482b6a3902e40211b43c5d0441962ea99b6cc7c25c0f243fa270b78d482"
    )
    assert _sha256(VENDOR_ROOT / "PATENTS") == (
        "01462e2068d1a04c2274f3389773014c14ed9bc3446b28303543bd3e3c064145"
    )


def test_notices_cover_every_vendored_third_party_dependency() -> None:
    metadata = _metadata()
    notices = NOTICES_PATH.read_text(encoding="utf-8")
    dependencies = metadata["third_party_dependencies"]

    assert "## WebRTC" in notices
    assert (VENDOR_ROOT / "LICENSE").read_text(encoding="utf-8").strip() in notices
    assert (VENDOR_ROOT / "PATENTS").read_text(encoding="utf-8").strip() in notices
    assert dependencies
    for dependency in dependencies:
        assert f"## {dependency['name']}" in notices
        license_path = VENDOR_ROOT / dependency["license_path"]
        assert license_path.read_text(encoding="utf-8").strip() in notices


def test_abseil_provenance_pins_exact_deps_revision_and_subtree() -> None:
    dependency = _metadata()["third_party_dependencies"][0]

    assert dependency["name"] == "Abseil"
    assert dependency["repository"] == ABSEIL_REPOSITORY
    assert dependency["commit"] == ABSEIL_COMMIT
    assert dependency["commit_timestamp"] == ABSEIL_TIMESTAMP
    assert dependency["subtree_tree"] == ABSEIL_SUBTREE
    assert dependency["path"] == "third_party/abseil-cpp"
    assert dependency["license_path"] == "third_party/abseil-cpp/LICENSE"
    assert dependency["license_sha256"] == ABSEIL_LICENSE_SHA256
    assert len(dependency["vendored_tree"]) == 40
    assert _sha256(VENDOR_ROOT / dependency["license_path"]) == ABSEIL_LICENSE_SHA256
    assert (
        _load_tool().git_tree_object_id(VENDOR_ROOT / dependency["path"])
        == dependency["vendored_tree"]
    )


def test_ooura_notice_is_derived_from_the_pinned_source_header() -> None:
    dependencies = {
        dependency["name"]: dependency
        for dependency in _metadata()["third_party_dependencies"]
    }
    assert set(dependencies) == {"Abseil", "Ooura FFT"}

    dependency = dependencies["Ooura FFT"]
    assert dependency == {
        "name": "Ooura FFT",
        "source_path": OOURA_SOURCE_PATH,
        "source_url": OOURA_SOURCE_URL,
        "source_sha256": OOURA_SOURCE_SHA256,
        "license_path": "OOURA_LICENSE",
        "license_sha256": OOURA_LICENSE_SHA256,
    }
    license_path = VENDOR_ROOT / dependency["license_path"]
    assert _sha256(license_path) == OOURA_LICENSE_SHA256
    assert _load_tool().extract_ooura_notice(
        VENDOR_ROOT / OOURA_SOURCE_PATH
    ) == license_path.read_text(encoding="utf-8")


def test_verifier_detects_altered_ooura_notice(tmp_path: Path) -> None:
    copied = tmp_path / "webrtc"
    shutil.copytree(VENDOR_ROOT, copied)
    notice = copied / "OOURA_LICENSE"
    notice.write_text(
        notice.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="integrity hash mismatch"):
        _load_tool().verify_vendor_tree(copied)


@pytest.mark.parametrize("mutation", ["edited", "missing", "extra", "outside"])
def test_verifier_detects_tampering(tmp_path: Path, mutation: str) -> None:
    copied = tmp_path / "webrtc"
    shutil.copytree(VENDOR_ROOT, copied)
    manifest_entries = _manifest()
    target = copied / manifest_entries[-1][1]

    if mutation == "edited":
        target.write_bytes(target.read_bytes() + b"tampered")
    elif mutation == "missing":
        target.unlink()
    elif mutation == "extra":
        (copied / "rtc_base" / "unexpected.cc").write_text(
            "// extra\n", encoding="utf-8"
        )
    else:
        (copied / "api" / "create_peerconnection_factory.cc").parent.mkdir(
            parents=True, exist_ok=True
        )
        (copied / "api" / "create_peerconnection_factory.cc").write_text(
            "// outside\n", encoding="utf-8"
        )

    with pytest.raises(ValueError, match="integrity|missing|extra|allowlist"):
        _load_tool().verify_vendor_tree(copied)
