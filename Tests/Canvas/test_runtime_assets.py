"""Behavioral tests for the vendored Canvas JavaScript runtime assets."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from types import ModuleType
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "tldw_chatbook" / "Canvas" / "static"
MANIFEST = STATIC / "runtime-manifest.json"
VENDOR_SCRIPT = ROOT / "scripts" / "vendor_canvas_runtime.py"

EXPECTED_PACKAGES = {
    "quickjs-emscripten-core": {
        "version": "0.32.0",
        "source_url": "https://registry.npmjs.org/quickjs-emscripten-core/-/quickjs-emscripten-core-0.32.0.tgz",
        "integrity": "sha512-QFnPfjFey8EqknSrSxe1hZrf1/8z7/6s1QzGOmKo6++02r7QRRX7ZoyNaZh7JuVjWsVW87KnQrbZqnHkOAzUyg==",
        "license": "MIT",
    },
    "@jitl/quickjs-singlefile-browser-release-sync": {
        "version": "0.32.0",
        "source_url": "https://registry.npmjs.org/@jitl/quickjs-singlefile-browser-release-sync/-/quickjs-singlefile-browser-release-sync-0.32.0.tgz",
        "integrity": "sha512-Hfdl7rh8dzxNWFRiYAYNbhn0RMF1/tO6SMH2mUW0aTibqwaAtqPRbi4WkwaIDlhNz8Z4dksJi1Zjl1R54Jsc/Q==",
        "license": "MIT",
    },
    "@jitl/quickjs-ffi-types": {
        "version": "0.32.0",
        "source_url": "https://registry.npmjs.org/@jitl/quickjs-ffi-types/-/quickjs-ffi-types-0.32.0.tgz",
        "integrity": "sha512-v9T+GQpmk43VDJ7d72sf0Nexhk+ArvtUihW27dy7lqAl0zBObFKtSBBIm5RBjwIhE8VwsPPm9PNuvPvNqLWUEg==",
        "license": "MIT",
    },
}
EXPECTED_BUILD_TOOL = {
    "name": "esbuild-wasm",
    "version": "0.25.9",
    "source_url": "https://registry.npmjs.org/esbuild-wasm/-/esbuild-wasm-0.25.9.tgz",
    "integrity": "sha512-Jpv5tCSwQg18aCqCRD3oHIX/prBhXMDapIoG//A+6+dV0e7KQMGFg85ihJ5T1EeMjbZjON3TqFy0VrGAnIHLDA==",
    "license": "MIT",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_vendor_module() -> ModuleType:
    assert VENDOR_SCRIPT.is_file(), "the pinned Canvas vendoring command must exist"
    spec = importlib.util.spec_from_file_location(
        "vendor_canvas_runtime", VENDOR_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sri_sha512(data: bytes) -> str:
    return "sha512-" + base64.b64encode(hashlib.sha512(data).digest()).decode("ascii")


def _tar_bytes(members: list[tuple[str, bytes, str]]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(
        fileobj=stream, mode="w:gz", format=tarfile.PAX_FORMAT
    ) as archive:
        for name, contents, member_type in members:
            info = tarfile.TarInfo(name)
            info.mtime = 0
            if member_type == "file":
                info.size = len(contents)
                archive.addfile(info, io.BytesIO(contents))
            elif member_type == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = "package/package.json"
                archive.addfile(info)
            elif member_type == "hardlink":
                info.type = tarfile.LNKTYPE
                info.linkname = "package/package.json"
                archive.addfile(info)
            else:  # pragma: no cover - fixture misuse
                raise AssertionError(member_type)
    return stream.getvalue()


def _package_json(
    name: str = "fixture-package", version: str = "1.0.0", **extra: Any
) -> bytes:
    return json.dumps(
        {"name": name, "version": version, "license": "MIT", **extra},
        sort_keys=True,
    ).encode("utf-8")


def test_manifest_pins_the_exact_reviewed_dependency_closure() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert manifest["runtime_profile"] == "canvas-v1"
    packages = {package["name"]: package for package in manifest["packages"]}
    assert set(packages) == set(EXPECTED_PACKAGES)
    for name, expected in EXPECTED_PACKAGES.items():
        actual = packages[name]
        assert {key: actual[key] for key in expected} == expected
        assert actual["extracted_files"]
        assert all(
            len(digest) == 64 and digest == digest.lower()
            for digest in actual["extracted_files"].values()
        )

    assert {
        key: manifest["build_tool"][key] for key in EXPECTED_BUILD_TOOL
    } == EXPECTED_BUILD_TOOL
    assert manifest["build_tool"]["extracted_files"]
    assert all(
        len(digest) == 64 and digest == digest.lower()
        for digest in manifest["build_tool"]["extracted_files"].values()
    )
    assert manifest["reproducible_command"] == "python scripts/vendor_canvas_runtime.py"


def test_manifest_proves_a_single_file_embedded_wasm_bundle() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["quickjs"] == {
        "source_revision": "2025-09-13+f1139494",
        "module_format": "esm",
        "build_mode": "release-sync",
        "filesystem": False,
        "single_file": True,
    }
    assert manifest["runtime_layout"] == {
        "javascript": "quickjs-runtime.js",
        "wasm": "embedded",
        "wasm_fetch_required": False,
    }
    assert set(manifest["outputs"]) == {
        "quickjs-runtime.js",
        "THIRD_PARTY_LICENSES.txt",
    }

    for filename, metadata in manifest["outputs"].items():
        path = STATIC / filename
        assert path.is_file()
        assert metadata == {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def test_generated_bundle_passes_the_repository_whitespace_gate() -> None:
    completed = subprocess.run(
        [
            "git",
            "diff",
            "--no-index",
            "--check",
            "--",
            "/dev/null",
            str(STATIC / "quickjs-runtime.js"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    # ``--no-index`` returns 1 for a clean comparison with differences and 3
    # when ``--check`` finds whitespace errors.
    assert completed.returncode == 1, (completed.stdout + completed.stderr)[-2000:]
    assert completed.stdout == ""
    assert completed.stderr == ""


def test_runtime_loader_returns_only_verified_packaged_bytes() -> None:
    from tldw_chatbook.Canvas.runtime_assets import load_canvas_runtime_assets

    result = load_canvas_runtime_assets()

    assert result.enabled is True
    assert result.diagnostic is None
    assert result.javascript == (STATIC / "quickjs-runtime.js").read_bytes()
    assert result.manifest["runtime_profile"] == "canvas-v1"


@pytest.mark.parametrize("damage", ["asset", "manifest", "missing"])
def test_runtime_loader_fails_closed_with_a_bounded_content_free_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    import tldw_chatbook.Canvas.runtime_assets as runtime_assets

    package_root = tmp_path / "Canvas"
    shutil.copytree(STATIC, package_root / "static")
    marker = "DO-NOT-LEAK-UNTRUSTED-RUNTIME-CONTENT"
    if damage == "asset":
        (package_root / "static" / "quickjs-runtime.js").write_text(
            marker, encoding="utf-8"
        )
    elif damage == "manifest":
        (package_root / "static" / "runtime-manifest.json").write_text(
            marker, encoding="utf-8"
        )
    else:
        (package_root / "static" / "runtime-manifest.json").unlink()
    monkeypatch.setattr(runtime_assets, "files", lambda _package: package_root)

    result = runtime_assets.load_canvas_runtime_assets()

    assert result.enabled is False
    assert result.javascript is None
    assert result.manifest is None
    assert result.diagnostic == runtime_assets.RUNTIME_DISABLED_DIAGNOSTIC
    assert len(result.diagnostic.encode("utf-8")) <= 160
    assert marker not in result.diagnostic


@pytest.mark.parametrize(
    ("unsafe_name", "member_type"),
    [
        ("../escape", "file"),
        ("/absolute", "file"),
        ("package/link", "symlink"),
        ("package/link", "hardlink"),
        ("package/unexpected.js", "file"),
    ],
)
def test_verified_extraction_rejects_unsafe_or_unexpected_members(
    tmp_path: Path, unsafe_name: str, member_type: str
) -> None:
    vendor = _load_vendor_module()
    payload = _tar_bytes(
        [
            ("package/package.json", _package_json(), "file"),
            ("package/LICENSE", b"MIT fixture", "file"),
            (unsafe_name, b"malicious", member_type),
        ]
    )
    archive = tmp_path / "fixture.tgz"
    archive.write_bytes(payload)

    with pytest.raises(vendor.VendorError):
        vendor.extract_verified_package(
            archive_path=archive,
            expected_integrity=_sri_sha512(payload),
            expected_name="fixture-package",
            expected_version="1.0.0",
            expected_license="MIT",
            allowed_members=frozenset({"package/package.json", "package/LICENSE"}),
            destination=tmp_path / "output",
        )
    assert not (tmp_path / "escape").exists()


def test_verified_extraction_rejects_wrong_digest_version_license_and_dependency(
    tmp_path: Path,
) -> None:
    vendor = _load_vendor_module()

    cases = [
        (_package_json(), "sha512-" + "A" * 88),
        (_package_json(version="2.0.0"), None),
        (_package_json(license="Apache-2.0"), None),
        (_package_json(dependencies={"surprise": "latest"}), None),
    ]
    for index, (package_json, integrity_override) in enumerate(cases):
        payload = _tar_bytes(
            [
                ("package/package.json", package_json, "file"),
                ("package/LICENSE", b"MIT fixture", "file"),
            ]
        )
        archive = tmp_path / f"fixture-{index}.tgz"
        archive.write_bytes(payload)
        with pytest.raises(vendor.VendorError):
            vendor.extract_verified_package(
                archive_path=archive,
                expected_integrity=integrity_override or _sri_sha512(payload),
                expected_name="fixture-package",
                expected_version="1.0.0",
                expected_license="MIT",
                allowed_members=frozenset({"package/package.json", "package/LICENSE"}),
                destination=tmp_path / f"output-{index}",
            )


def test_reproducible_command_can_verify_committed_assets_without_network(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [sys.executable, str(VENDOR_SCRIPT), "--verify", "--output-dir", str(STATIC)],
        cwd=ROOT,
        env={"PATH": str(tmp_path), "PYTHONPATH": str(ROOT)},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "Canvas runtime assets verified"
