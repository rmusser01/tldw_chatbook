#!/usr/bin/env python3
"""Reproducibly vendor the integrity-pinned Canvas QuickJS runtime bundle."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import hashlib
import hmac
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Mapping
from urllib.request import build_opener, HTTPRedirectHandler, Request


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "tldw_chatbook" / "Canvas" / "static"
GENERATED_ASSETS = (
    "quickjs-runtime.js",
    "THIRD_PARTY_LICENSES.txt",
    "runtime-manifest.json",
)
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_EXTRACTED_BYTES = 32 * 1024 * 1024


class VendorError(RuntimeError):
    """Raised when a pinned input or generated output violates the vendor contract."""


class _RejectRedirects(HTTPRedirectHandler):
    """Prevent registry downloads from following any redirect response."""

    def redirect_request(self, *_args: object, **_kwargs: object) -> None:
        return None


@dataclass(frozen=True)
class PackageSpec:
    """Exact registry input and its closed published-file/dependency inventory."""

    name: str
    version: str
    source_url: str
    integrity: str
    license: str
    archive_filename: str
    allowed_members: frozenset[str]
    expected_dependencies: Mapping[str, str]
    license_member: str
    install_path: str


CORE_MEMBERS = frozenset(
    {
        "package/LICENSE",
        "package/README.md",
        "package/README.template.md",
        "package/dist/chunk-TAV5CUKK.mjs",
        "package/dist/chunk-TAV5CUKK.mjs.map",
        "package/dist/chunk-V2S4ZYJR.mjs",
        "package/dist/chunk-V2S4ZYJR.mjs.map",
        "package/dist/index.d.mts",
        "package/dist/index.d.ts",
        "package/dist/index.js",
        "package/dist/index.js.map",
        "package/dist/index.mjs",
        "package/dist/index.mjs.map",
        "package/dist/module-ES6BEMUI.mjs",
        "package/dist/module-ES6BEMUI.mjs.map",
        "package/dist/module-asyncify-2EFITU5U.mjs",
        "package/dist/module-asyncify-2EFITU5U.mjs.map",
        "package/package.json",
    }
)
VARIANT_MEMBERS = frozenset(
    {
        "package/LICENSE",
        "package/README.md",
        "package/dist/chunk-FGV2HSCH.mjs",
        "package/dist/chunk-FGV2HSCH.mjs.map",
        "package/dist/emscripten-module.browser-XIKQQPVU.mjs",
        "package/dist/emscripten-module.browser-XIKQQPVU.mjs.map",
        "package/dist/emscripten-module.browser.d.ts",
        "package/dist/emscripten-module.browser.mjs",
        "package/dist/ffi.d.mts",
        "package/dist/ffi.mjs",
        "package/dist/ffi.mjs.map",
        "package/dist/index.d.mts",
        "package/dist/index.mjs",
        "package/dist/index.mjs.map",
        "package/package.json",
    }
)
FFI_MEMBERS = frozenset(
    {
        "package/LICENSE",
        "package/README.md",
        "package/dist/index.d.mts",
        "package/dist/index.d.ts",
        "package/dist/index.js",
        "package/dist/index.js.map",
        "package/dist/index.mjs",
        "package/dist/index.mjs.map",
        "package/package.json",
    }
)
ESBUILD_MEMBERS = frozenset(
    {
        "package/LICENSE.md",
        "package/README.md",
        "package/bin/esbuild",
        "package/esbuild.wasm",
        "package/esm/browser.d.ts",
        "package/esm/browser.js",
        "package/esm/browser.min.js",
        "package/lib/browser.d.ts",
        "package/lib/browser.js",
        "package/lib/browser.min.js",
        "package/lib/main.d.ts",
        "package/lib/main.js",
        "package/package.json",
        "package/wasm_exec.js",
        "package/wasm_exec_node.js",
    }
)

RUNTIME_PACKAGES = (
    PackageSpec(
        name="quickjs-emscripten-core",
        version="0.32.0",
        source_url="https://registry.npmjs.org/quickjs-emscripten-core/-/quickjs-emscripten-core-0.32.0.tgz",
        integrity="sha512-QFnPfjFey8EqknSrSxe1hZrf1/8z7/6s1QzGOmKo6++02r7QRRX7ZoyNaZh7JuVjWsVW87KnQrbZqnHkOAzUyg==",
        license="MIT",
        archive_filename="quickjs-emscripten-core-0.32.0.tgz",
        allowed_members=CORE_MEMBERS,
        expected_dependencies={"@jitl/quickjs-ffi-types": "0.32.0"},
        license_member="package/LICENSE",
        install_path="node_modules/quickjs-emscripten-core",
    ),
    PackageSpec(
        name="@jitl/quickjs-singlefile-browser-release-sync",
        version="0.32.0",
        source_url="https://registry.npmjs.org/@jitl/quickjs-singlefile-browser-release-sync/-/quickjs-singlefile-browser-release-sync-0.32.0.tgz",
        integrity="sha512-Hfdl7rh8dzxNWFRiYAYNbhn0RMF1/tO6SMH2mUW0aTibqwaAtqPRbi4WkwaIDlhNz8Z4dksJi1Zjl1R54Jsc/Q==",
        license="MIT",
        archive_filename="quickjs-singlefile-browser-release-sync-0.32.0.tgz",
        allowed_members=VARIANT_MEMBERS,
        expected_dependencies={"@jitl/quickjs-ffi-types": "0.32.0"},
        license_member="package/LICENSE",
        install_path="node_modules/@jitl/quickjs-singlefile-browser-release-sync",
    ),
    PackageSpec(
        name="@jitl/quickjs-ffi-types",
        version="0.32.0",
        source_url="https://registry.npmjs.org/@jitl/quickjs-ffi-types/-/quickjs-ffi-types-0.32.0.tgz",
        integrity="sha512-v9T+GQpmk43VDJ7d72sf0Nexhk+ArvtUihW27dy7lqAl0zBObFKtSBBIm5RBjwIhE8VwsPPm9PNuvPvNqLWUEg==",
        license="MIT",
        archive_filename="quickjs-ffi-types-0.32.0.tgz",
        allowed_members=FFI_MEMBERS,
        expected_dependencies={},
        license_member="package/LICENSE",
        install_path="node_modules/@jitl/quickjs-ffi-types",
    ),
)
BUILD_TOOL = PackageSpec(
    name="esbuild-wasm",
    version="0.25.9",
    source_url="https://registry.npmjs.org/esbuild-wasm/-/esbuild-wasm-0.25.9.tgz",
    integrity="sha512-Jpv5tCSwQg18aCqCRD3oHIX/prBhXMDapIoG//A+6+dV0e7KQMGFg85ihJ5T1EeMjbZjON3TqFy0VrGAnIHLDA==",
    license="MIT",
    archive_filename="esbuild-wasm-0.25.9.tgz",
    allowed_members=ESBUILD_MEMBERS,
    expected_dependencies={},
    license_member="package/LICENSE.md",
    install_path="build-tools/esbuild-wasm",
)
ALL_INPUTS = (*RUNTIME_PACKAGES, BUILD_TOOL)

ENTRY_SOURCE = """\
import releaseSyncVariant from "@jitl/quickjs-singlefile-browser-release-sync";
import { newQuickJSWASMModuleFromVariant } from "quickjs-emscripten-core";
export * from "quickjs-emscripten-core";
export { releaseSyncVariant };
export function newQuickJSWASMModule() {
  return newQuickJSWASMModuleFromVariant(releaseSyncVariant);
}
"""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sri(data: bytes, expected_integrity: str) -> None:
    prefix = "sha512-"
    if not expected_integrity.startswith(prefix):
        raise VendorError("only SHA-512 Subresource Integrity values are accepted")
    actual = base64.b64encode(hashlib.sha512(data).digest()).decode("ascii")
    if not hmac.compare_digest(actual, expected_integrity[len(prefix) :]):
        raise VendorError("registry tarball integrity mismatch")


def _safe_member_name(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        bool(name)
        and not path.is_absolute()
        and "\\" not in name
        and "\x00" not in name
        and str(path) == name
        and all(part not in {"", ".", ".."} for part in path.parts)
        and path.parts[0] == "package"
    )


def extract_verified_package(
    *,
    archive_path: Path,
    expected_integrity: str,
    expected_name: str,
    expected_version: str,
    expected_license: str,
    allowed_members: frozenset[str],
    destination: Path,
    expected_dependencies: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Verify and safely extract one exact package archive.

    Returns a map from published member path to independently computed SHA-256.
    """

    try:
        with archive_path.open("rb") as archive_file:
            payload = archive_file.read(MAX_ARCHIVE_BYTES + 1)
    except OSError as exc:
        raise VendorError("registry tarball could not be read") from exc
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise VendorError("registry tarball exceeds the vendoring byte limit")
    _verify_sri(payload, expected_integrity)

    extracted: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)):
                raise VendorError("registry tarball contains duplicate members")
            if any(not _safe_member_name(name) for name in names):
                raise VendorError("registry tarball contains an unsafe member path")
            if any(member.issym() or member.islnk() for member in members):
                raise VendorError("registry tarball links are forbidden")
            if any(not member.isfile() for member in members):
                raise VendorError("registry tarball contains a non-file member")
            if set(names) != set(allowed_members):
                raise VendorError("registry tarball member inventory mismatch")
            if sum(member.size for member in members) > MAX_EXTRACTED_BYTES:
                raise VendorError("registry tarball extraction exceeds the byte limit")
            for member in members:
                source = archive.extractfile(member)
                if source is None:
                    raise VendorError("registry tarball member could not be read")
                contents = source.read(MAX_EXTRACTED_BYTES + 1)
                if len(contents) != member.size:
                    raise VendorError("registry tarball member size mismatch")
                extracted[member.name] = contents
    except (OSError, tarfile.TarError) as exc:
        raise VendorError("registry tarball is invalid") from exc

    try:
        package = json.loads(extracted["package/package.json"].decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VendorError("registry package metadata is invalid") from exc
    if not isinstance(package, dict):
        raise VendorError("registry package metadata is invalid")
    if (
        package.get("name") != expected_name
        or package.get("version") != expected_version
    ):
        raise VendorError("registry package identity mismatch")
    if package.get("license") != expected_license:
        raise VendorError("registry package license mismatch")
    runtime_dependencies = package.get("dependencies", {})
    if runtime_dependencies != dict(expected_dependencies or {}):
        raise VendorError("registry package dependency closure mismatch")
    for field in ("optionalDependencies", "peerDependencies", "bundledDependencies"):
        if package.get(field):
            raise VendorError(
                "registry package declares an unsupported dependency source"
            )

    destination.mkdir(parents=True, exist_ok=False)
    for member_name in sorted(extracted):
        relative = PurePosixPath(member_name).relative_to("package")
        target = destination.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(extracted[member_name])
    return {name: _sha256_bytes(extracted[name]) for name in sorted(extracted)}


def _download(spec: PackageSpec, destination: Path) -> Path:
    if not spec.source_url.startswith("https://"):
        raise VendorError("registry source URL must use HTTPS")
    request = Request(
        spec.source_url, headers={"User-Agent": "tldw-chatbook-canvas-vendor/1"}
    )
    try:
        with build_opener(_RejectRedirects()).open(request, timeout=30) as response:
            if response.geturl() != spec.source_url:
                raise VendorError("registry source redirected away from the pinned URL")
            payload = response.read(MAX_ARCHIVE_BYTES + 1)
    except VendorError:
        raise
    except OSError as exc:
        raise VendorError("could not download a pinned registry tarball") from exc
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise VendorError("registry tarball exceeds the vendoring byte limit")
    _verify_sri(payload, spec.integrity)
    target = destination / spec.archive_filename
    target.write_bytes(payload)
    return target


def _archive_path(
    spec: PackageSpec, archive_dir: Path | None, download_dir: Path
) -> Path:
    if archive_dir is None:
        return _download(spec, download_dir)
    path = archive_dir / spec.archive_filename
    if not path.is_file():
        raise VendorError(f"missing pinned archive: {spec.archive_filename}")
    return path


def _write_notices(workspace: Path, destination: Path) -> None:
    sections = [
        "Canvas runtime third-party notices\n",
        "Generated from the exact integrity-pinned registry inputs listed in runtime-manifest.json.\n",
    ]
    for spec in ALL_INPUTS:
        license_path = (
            workspace / spec.install_path / PurePosixPath(spec.license_member).name
        )
        sections.append(
            f"\n===== {spec.name}@{spec.version} ({spec.license}) =====\n\n"
            f"Source: {spec.source_url}\n\n"
            f"{license_path.read_text(encoding='utf-8').rstrip()}\n"
        )
    destination.write_text("".join(sections), encoding="utf-8", newline="\n")


def _escape_embedded_wasm_whitespace(bundle_path: Path) -> None:
    """Encode raw line-breaking whitespace in Emscripten's WASM template literal."""

    bundle = bundle_path.read_bytes()
    marker = b"`\\0asm"
    if bundle.count(marker) != 1:
        raise VendorError("the built bundle does not contain one embedded WASM module")
    start = bundle.index(marker)
    end = start + 1
    while True:
        end = bundle.find(b"`", end + 1)
        if end < 0:
            raise VendorError("the embedded WASM template literal is unterminated")
        preceding_backslashes = 0
        cursor = end - 1
        while cursor >= start and bundle[cursor] == ord("\\"):
            preceding_backslashes += 1
            cursor -= 1
        if preceding_backslashes % 2 == 0:
            break
    embedded = bundle[start + 1 : end]
    escaped = (
        embedded.replace(b"\r", b"\\r").replace(b"\n", b"\\n").replace(b"\t", b"\\t")
    )
    bundle_path.write_bytes(bundle[: start + 1] + escaped + bundle[end:])


def _run_esbuild(workspace: Path, output_path: Path, node_path: str | None) -> None:
    node = Path(node_path).resolve() if node_path else Path(shutil.which("node") or "")
    if not node.is_file():
        raise VendorError(
            "Node.js >=18 is required only to regenerate Canvas runtime assets"
        )
    version = subprocess.run(
        [str(node), "--version"],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": str(node.parent)},
    )
    if version.returncode != 0:
        raise VendorError("could not verify the build-only Node.js executable")
    try:
        major = int(version.stdout.strip().removeprefix("v").split(".", 1)[0])
    except ValueError as exc:
        raise VendorError("could not parse the build-only Node.js version") from exc
    if major < 18:
        raise VendorError(
            "Node.js >=18 is required only to regenerate Canvas runtime assets"
        )

    entry = workspace / "entry.mjs"
    entry.write_text(ENTRY_SOURCE, encoding="utf-8", newline="\n")
    esbuild = workspace / BUILD_TOOL.install_path / "bin" / "esbuild"
    tool_version = subprocess.run(
        [str(node), str(esbuild), "--version"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": str(node.parent), "NO_COLOR": "1", "SOURCE_DATE_EPOCH": "0"},
    )
    if (
        tool_version.returncode != 0
        or tool_version.stdout.strip() != BUILD_TOOL.version
    ):
        raise VendorError("the extracted build tool version does not match its pin")
    command = [
        str(node),
        str(esbuild),
        str(entry),
        "--bundle",
        "--format=esm",
        "--platform=browser",
        "--target=es2020",
        "--minify",
        "--legal-comments=none",
        "--charset=utf8",
        "--log-level=warning",
        f"--outfile={output_path}",
    ]
    completed = subprocess.run(
        command,
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": str(node.parent), "NO_COLOR": "1", "SOURCE_DATE_EPOCH": "0"},
    )
    if completed.returncode != 0:
        raise VendorError("the pinned esbuild-wasm bundle step failed")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise VendorError("the pinned build produced no Canvas runtime bundle")
    _escape_embedded_wasm_whitespace(output_path)


def _manifest_entry(
    spec: PackageSpec, extracted_files: dict[str, str]
) -> dict[str, object]:
    return {
        "name": spec.name,
        "version": spec.version,
        "source_url": spec.source_url,
        "integrity": spec.integrity,
        "license": spec.license,
        "extracted_files": extracted_files,
    }


def vendor(
    *, output_dir: Path, archive_dir: Path | None, node_path: str | None
) -> None:
    """Create the declared runtime outputs from exact verified registry inputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="canvas-runtime-vendor-") as temp_name:
        workspace = Path(temp_name)
        downloads = workspace / "downloads"
        downloads.mkdir()
        extracted_digests: dict[str, dict[str, str]] = {}
        for spec in ALL_INPUTS:
            archive = _archive_path(spec, archive_dir, downloads)
            destination = workspace / spec.install_path
            extracted_digests[spec.name] = extract_verified_package(
                archive_path=archive,
                expected_integrity=spec.integrity,
                expected_name=spec.name,
                expected_version=spec.version,
                expected_license=spec.license,
                allowed_members=spec.allowed_members,
                destination=destination,
                expected_dependencies=spec.expected_dependencies,
            )

        staged = workspace / "generated"
        staged.mkdir()
        bundle = staged / "quickjs-runtime.js"
        notices = staged / "THIRD_PARTY_LICENSES.txt"
        _run_esbuild(workspace, bundle, node_path)
        _write_notices(workspace, notices)
        outputs = {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
            for path in (bundle, notices)
        }
        manifest = {
            "schema_version": 1,
            "runtime_profile": "canvas-v1",
            "runtime_layout": {
                "javascript": "quickjs-runtime.js",
                "wasm": "embedded",
                "wasm_fetch_required": False,
            },
            "quickjs": {
                "source_revision": "2025-09-13+f1139494",
                "module_format": "esm",
                "build_mode": "release-sync",
                "filesystem": False,
                "single_file": True,
            },
            "packages": [
                _manifest_entry(spec, extracted_digests[spec.name])
                for spec in RUNTIME_PACKAGES
            ],
            "build_tool": _manifest_entry(
                BUILD_TOOL, extracted_digests[BUILD_TOOL.name]
            ),
            "reproducible_command": "python scripts/vendor_canvas_runtime.py",
            "outputs": outputs,
        }
        (staged / "runtime-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        for name in GENERATED_ASSETS:
            os.replace(staged / name, output_dir / name)


def verify(output_dir: Path) -> None:
    """Verify generated assets without importing Node or using the network."""

    try:
        manifest = json.loads(
            (output_dir / "runtime-manifest.json").read_text(encoding="utf-8")
        )
        outputs = manifest["outputs"]
        if set(outputs) != {"quickjs-runtime.js", "THIRD_PARTY_LICENSES.txt"}:
            raise VendorError("Canvas runtime output inventory mismatch")
        for name, expected in outputs.items():
            path = output_dir / name
            actual = {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
            if actual != expected:
                raise VendorError("Canvas runtime output integrity mismatch")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise VendorError("Canvas runtime manifest is invalid") from exc


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-dir", type=Path, help="Use already-downloaded pinned tarballs"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--node", help="Build-only Node.js executable (>=18)")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify committed outputs without Node/network",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    try:
        if args.verify:
            verify(args.output_dir)
            print("Canvas runtime assets verified")
        else:
            vendor(
                output_dir=args.output_dir,
                archive_dir=args.archive_dir,
                node_path=args.node,
            )
            verify(args.output_dir)
            print("Canvas runtime assets generated and verified")
    except VendorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
