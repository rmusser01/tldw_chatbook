"""Behavioral tests for the vendored Canvas JavaScript runtime assets."""

from __future__ import annotations

import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from threading import Thread
from types import ModuleType
from typing import Any
from urllib.request import Request

import pytest


ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "tldw_chatbook" / "Canvas" / "static"
MANIFEST = STATIC / "runtime-manifest.json"
VENDOR_SCRIPT = ROOT / "scripts" / "vendor_canvas_runtime.py"
ARCHIVE_CACHE_ENV = "TLDW_CANVAS_RUNTIME_ARCHIVE_DIR"
SHELL_HTML = STATIC / "canvas_shell.html"

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
EXPECTED_RUNTIME_OUTPUTS = {
    "quickjs-runtime.js",
    "canvas_runtime_worker.js",
    "canvas_renderer.js",
    "THIRD_PARTY_LICENSES.txt",
}


def test_canvas_shell_keeps_approved_direction_as_first_body_child() -> None:
    html = SHELL_HTML.read_text(encoding="utf-8")
    body = html.split("<body>", 1)[1]
    first_child = body.lstrip()

    assert first_child.startswith("<!--")
    comment_end = first_child.index("-->")
    direction = first_child[:comment_end]
    assert all(
        f"{heading}:" in direction
        for heading in (
            "THESIS",
            "OWN-WORLD",
            "STORY",
            "FIRST VIEWPORT",
            "FORM",
            "FINISH",
        )
    )
    assert body.index("<!--") < body.index("<noscript>")


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
        "renderer": "canvas_renderer.js",
        "worker": "canvas_runtime_worker.js",
        "wasm": "embedded",
        "wasm_fetch_required": False,
    }
    assert set(manifest["outputs"]) == EXPECTED_RUNTIME_OUTPUTS

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
    assert (
        result.worker_javascript == (STATIC / "canvas_runtime_worker.js").read_bytes()
    )
    assert result.renderer_javascript == (STATIC / "canvas_renderer.js").read_bytes()
    assert result.manifest["runtime_profile"] == "canvas-v1"


@pytest.mark.parametrize(
    "damage", ["quickjs", "worker", "renderer", "manifest", "missing"]
)
def test_runtime_loader_fails_closed_with_a_bounded_content_free_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    import tldw_chatbook.Canvas.runtime_assets as runtime_assets

    package_root = tmp_path / "Canvas"
    shutil.copytree(STATIC, package_root / "static")
    marker = "DO-NOT-LEAK-UNTRUSTED-RUNTIME-CONTENT"
    if damage in {"quickjs", "worker", "renderer"}:
        damaged_name = {
            "quickjs": "quickjs-runtime.js",
            "worker": "canvas_runtime_worker.js",
            "renderer": "canvas_renderer.js",
        }[damage]
        (package_root / "static" / damaged_name).write_text(marker, encoding="utf-8")
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
    assert result.worker_javascript is None
    assert result.renderer_javascript is None
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


@pytest.mark.loopback_network
def test_download_rejects_redirect_before_target_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vendor = _load_vendor_module()
    requests = {"redirect": 0, "target": 0}

    class RedirectHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
            if self.path == "/redirect":
                requests["redirect"] += 1
                self.send_response(302)
                self.send_header(
                    "Location", f"http://127.0.0.1:{self.server.server_port}/target"
                )
                self.end_headers()
                return
            requests["target"] += 1
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"redirect target must remain unreachable")

        def log_message(self, _format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    redirect_url = f"http://127.0.0.1:{server.server_port}/redirect"
    real_request = Request
    monkeypatch.setattr(
        vendor,
        "Request",
        lambda _url, **kwargs: real_request(redirect_url, **kwargs),
    )
    pinned = vendor.RUNTIME_PACKAGES[0]
    try:
        with pytest.raises(vendor.VendorError):
            vendor._download(pinned, tmp_path)
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert requests == {"redirect": 1, "target": 0}


def test_oversized_archive_is_read_only_to_the_enforced_cap(
    tmp_path: Path,
) -> None:
    vendor = _load_vendor_module()
    payload = b"x" * (vendor.MAX_ARCHIVE_BYTES + 1)
    read_sizes: list[int] = []

    class ReadSpy(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            read_sizes.append(size)
            return super().read(size)

    class ObservedArchive:
        def read_bytes(self) -> bytes:
            read_sizes.append(-1)
            return payload

        def open(self, _mode: str) -> ReadSpy:
            return ReadSpy(payload)

    with pytest.raises(vendor.VendorError, match="byte limit"):
        vendor.extract_verified_package(
            archive_path=ObservedArchive(),
            expected_integrity=_sri_sha512(payload),
            expected_name="fixture-package",
            expected_version="1.0.0",
            expected_license="MIT",
            allowed_members=frozenset(),
            destination=tmp_path / "output",
        )

    assert read_sizes == [vendor.MAX_ARCHIVE_BYTES + 1]


def test_extraction_uses_the_exact_authenticated_archive_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vendor = _load_vendor_module()
    authentic_license = b"authenticated MIT license"
    authentic = _tar_bytes(
        [
            ("package/package.json", _package_json(), "file"),
            ("package/LICENSE", authentic_license, "file"),
        ]
    )
    replacement = _tar_bytes(
        [
            ("package/package.json", _package_json(), "file"),
            ("package/LICENSE", b"replaced after authentication", "file"),
        ]
    )
    archive = tmp_path / "fixture.tgz"
    archive.write_bytes(authentic)
    real_verify_sri = vendor._verify_sri

    def replace_path_after_verification(data: bytes, expected: str) -> None:
        real_verify_sri(data, expected)
        archive.write_bytes(replacement)

    monkeypatch.setattr(vendor, "_verify_sri", replace_path_after_verification)
    destination = tmp_path / "output"

    vendor.extract_verified_package(
        archive_path=archive,
        expected_integrity=_sri_sha512(authentic),
        expected_name="fixture-package",
        expected_version="1.0.0",
        expected_license="MIT",
        allowed_members=frozenset({"package/package.json", "package/LICENSE"}),
        destination=destination,
    )

    assert (destination / "LICENSE").read_bytes() == authentic_license


def test_verify_command_checks_committed_assets_without_network(
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


@pytest.mark.integration
def test_pinned_archive_regeneration_is_reproducible_and_instantiable(
    tmp_path: Path,
) -> None:
    archive_cache = os.environ.get(ARCHIVE_CACHE_ENV)
    if not archive_cache:
        pytest.skip(f"set {ARCHIVE_CACHE_ENV} to the verified pinned archive cache")
    archive_dir = Path(archive_cache).resolve()
    if not archive_dir.is_dir():
        pytest.fail(f"{ARCHIVE_CACHE_ENV} is not a directory: {archive_dir}")
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to exercise the generated browser module")

    output_a = tmp_path / "generated-a"
    output_b = tmp_path / "generated-b"
    for output in (output_a, output_b):
        completed = subprocess.run(
            [
                sys.executable,
                str(VENDOR_SCRIPT),
                "--archive-dir",
                str(archive_dir),
                "--output-dir",
                str(output),
                "--node",
                node,
            ],
            cwd=ROOT,
            env={"PATH": str(Path(node).parent), "PYTHONPATH": str(ROOT)},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr[-4000:]
        assert (
            completed.stdout.strip() == "Canvas runtime assets generated and verified"
        )

    generated_names = {
        "quickjs-runtime.js",
        "canvas_runtime_worker.js",
        "canvas_renderer.js",
        "THIRD_PARTY_LICENSES.txt",
        "runtime-manifest.json",
    }
    assert {path.name for path in output_a.iterdir()} == generated_names
    assert {path.name for path in output_b.iterdir()} == generated_names
    for name in generated_names:
        bytes_a = (output_a / name).read_bytes()
        assert (output_b / name).read_bytes() == bytes_a
        assert (STATIC / name).read_bytes() == bytes_a

    runtime_module = tmp_path / "quickjs-runtime.mjs"
    runtime_module.write_bytes((output_a / "quickjs-runtime.js").read_bytes())
    probe = tmp_path / "instantiate.mjs"
    probe.write_text(
        """\
import { newQuickJSWASMModule } from "./quickjs-runtime.mjs";
const QuickJS = await newQuickJSWASMModule();
const runtime = QuickJS.newRuntime();
for (const control of [
  "setMemoryLimit",
  "setMaxStackSize",
  "setInterruptHandler",
  "executePendingJobs",
  "hasPendingJob",
]) {
  if (typeof runtime[control] !== "function") throw new Error(`missing ${control}`);
}
runtime.setMemoryLimit(4 * 1024 * 1024);
runtime.setMaxStackSize(256 * 1024);
runtime.setInterruptHandler(() => false);
runtime.hasPendingJob();
runtime.executePendingJobs(1);
runtime.dispose();
console.log("generated runtime instantiated with required controls");
""",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [node, str(probe)],
        cwd=tmp_path,
        env={"PATH": str(Path(node).parent)},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr[-4000:]
    assert completed.stdout.strip() == (
        "generated runtime instantiated with required controls"
    )
