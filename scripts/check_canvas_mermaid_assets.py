"""Rebuild and compare the complete Canvas Mermaid derived-artifact closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, build_opener

from vendor_canvas_mermaid import SOURCE, STATIC, build
from vendor_canvas_runtime import VendorError, _RejectRedirects

INPUT_MANIFEST = SOURCE / "inputs.json"
EXPECTED_OUTPUTS = frozenset(
    {
        "canvas_runtime_worker_v2.js",
        "canvas_renderer_v2.js",
        "mermaid-subset.json",
        "MERMAID_THIRD_PARTY_LICENSES.txt",
        "mermaid-runtime-manifest.json",
        "profile-catalog.json",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


class MermaidAssetCheckError(RuntimeError):
    """Raised when pinned inputs or regenerated outputs fail closed checks."""


def _validated_downloads(downloads: object) -> dict[str, dict[str, object]]:
    if not isinstance(downloads, Mapping) or not downloads:
        raise MermaidAssetCheckError("input manifest has no downloads")
    validated: dict[str, dict[str, object]] = {}
    for name, record in downloads.items():
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or "/" in name
            or "\\" in name
        ):
            raise MermaidAssetCheckError("input manifest contains an unsafe name")
        if not isinstance(record, Mapping) or set(record) != {
            "url",
            "bytes",
            "sha256",
        }:
            raise MermaidAssetCheckError(f"input declaration is incomplete: {name}")
        url = record["url"]
        size = record["bytes"]
        sha256 = record["sha256"]
        parsed = urlsplit(url) if isinstance(url, str) else None
        if (
            parsed is None
            or parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise MermaidAssetCheckError(f"input URL is not public HTTPS: {name}")
        if type(size) is not int or size <= 0:
            raise MermaidAssetCheckError(f"input size pin is invalid: {name}")
        if not isinstance(sha256, str) or _SHA256.fullmatch(sha256) is None:
            raise MermaidAssetCheckError(f"input SHA-256 pin is invalid: {name}")
        validated[name] = {"url": url, "bytes": size, "sha256": sha256}
    return validated


def _verified_input(data: bytes, record: Mapping[str, object], name: str) -> bytes:
    size = record["bytes"]
    sha256 = record["sha256"]
    if len(data) != size or hashlib.sha256(data).hexdigest() != sha256:
        raise MermaidAssetCheckError(f"declared input integrity mismatch: {name}")
    return data


def acquire_declared_inputs(
    downloads: object,
    destination: Path,
    *,
    input_dir: Path | None = None,
) -> None:
    """Copy or download only declared, exact pinned inputs into isolation.

    Args:
        downloads: Manifest mapping of leaf name to HTTPS URL, byte size and
            SHA-256 digest.
        destination: Empty, checker-owned directory receiving authenticated
            inputs.
        input_dir: Optional offline cache. When omitted, inputs are downloaded
            from their exact declared public URLs.

    Returns:
        None.

    Raises:
        MermaidAssetCheckError: If a declaration, input, redirect, or isolated
            destination violates the pinned-input contract.
    """

    records = _validated_downloads(downloads)
    destination.mkdir(parents=True, exist_ok=False)
    opener = None if input_dir is not None else build_opener(_RejectRedirects())
    for name, record in records.items():
        size = record["bytes"]
        assert isinstance(size, int)
        try:
            if input_dir is not None:
                source = input_dir / name
                if source.is_symlink() or not source.is_file():
                    raise MermaidAssetCheckError(f"declared input is missing: {name}")
                with source.open("rb") as stream:
                    data = stream.read(size + 1)
            else:
                assert opener is not None
                url = record["url"]
                assert isinstance(url, str)
                with opener.open(Request(url), timeout=30) as response:
                    if response.geturl() != url:
                        raise MermaidAssetCheckError(
                            f"declared input redirect refused: {name}"
                        )
                    data = response.read(size + 1)
        except MermaidAssetCheckError:
            raise
        except OSError as exc:
            raise MermaidAssetCheckError(
                f"declared input is unavailable: {name}"
            ) from exc
        (destination / name).write_bytes(_verified_input(data, record, name))


def compare_generated_outputs(
    rebuilt_dir: Path,
    committed_dir: Path,
    generated: object,
) -> None:
    """Require the entire generated closure to match packaged bytes exactly.

    Args:
        rebuilt_dir: Isolated output directory populated by the existing build.
        committed_dir: Packaged Canvas static directory to compare against.
        generated: Build result mapping output names to size and hash metadata.

    Returns:
        None.

    Raises:
        MermaidAssetCheckError: If the generated inventory is incomplete,
            contains extras, disagrees with its metadata, or differs from the
            committed packaged output.
    """

    if not isinstance(generated, Mapping) or set(generated) != EXPECTED_OUTPUTS:
        raise MermaidAssetCheckError("builder reported an incomplete output set")
    try:
        rebuilt_names = {path.name for path in rebuilt_dir.iterdir()}
    except OSError as exc:
        raise MermaidAssetCheckError("rebuilt output directory is unavailable") from exc
    if rebuilt_names != EXPECTED_OUTPUTS:
        raise MermaidAssetCheckError("rebuilt directory has missing or extra outputs")
    for name in sorted(EXPECTED_OUTPUTS):
        rebuilt = rebuilt_dir / name
        committed = committed_dir / name
        if rebuilt.is_symlink() or committed.is_symlink():
            raise MermaidAssetCheckError(
                f"generated output cannot be a symlink: {name}"
            )
        try:
            rebuilt_bytes = rebuilt.read_bytes()
            committed_bytes = committed.read_bytes()
        except OSError as exc:
            raise MermaidAssetCheckError(
                f"generated output is missing: {name}"
            ) from exc
        record = generated[name]
        expected_record = {
            "bytes": len(rebuilt_bytes),
            "sha256": hashlib.sha256(rebuilt_bytes).hexdigest(),
        }
        if record != expected_record:
            raise MermaidAssetCheckError(f"builder metadata mismatch: {name}")
        if rebuilt_bytes != committed_bytes:
            raise MermaidAssetCheckError(f"committed generated output drift: {name}")


def check_assets(
    *,
    input_dir: Path | None = None,
    manifest_path: Path = INPUT_MANIFEST,
    committed_dir: Path = STATIC,
) -> Mapping[str, object]:
    """Authenticate inputs, rebuild offline, and compare packaged outputs.

    Args:
        input_dir: Optional local cache containing only source input candidates.
            If omitted, exact manifest URLs are downloaded before the offline
            build begins.
        manifest_path: Pinned Mermaid input manifest.
        committed_dir: Packaged generated-output directory.

    Returns:
        The existing builder's verified output metadata mapping.

    Raises:
        MermaidAssetCheckError: If inputs or outputs violate the reproduction
            contract.
        VendorError: If the existing offline build rejects authenticated input.
    """

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        downloads = manifest["downloads"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise MermaidAssetCheckError(
            "input manifest is unavailable or invalid"
        ) from exc
    with tempfile.TemporaryDirectory(prefix="canvas-mermaid-check-") as temporary:
        root = Path(temporary)
        isolated_inputs = root / "inputs"
        rebuilt = root / "rebuilt"
        acquire_declared_inputs(downloads, isolated_inputs, input_dir=input_dir)
        generated = build(isolated_inputs, rebuilt)
        compare_generated_outputs(rebuilt, committed_dir, generated)
        return generated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        input_dir = args.input_dir
        configured = os.environ.get("TLDW_CANVAS_MERMAID_INPUT_DIR")
        if input_dir is None and configured is not None:
            if not configured:
                raise MermaidAssetCheckError(
                    "TLDW_CANVAS_MERMAID_INPUT_DIR cannot be empty"
                )
            input_dir = Path(configured)
        generated = check_assets(input_dir=input_dir)
    except (MermaidAssetCheckError, VendorError) as exc:
        print(f"Canvas Mermaid assets do not reproduce: {exc}", file=sys.stderr)
        return 1
    print(f"Canvas Mermaid assets reproduce: {len(generated)} outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
