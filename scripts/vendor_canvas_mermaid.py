#!/usr/bin/env python3
"""Build inert Mermaid source from authenticated, closed offline inputs only."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib.request import Request, build_opener

from vendor_canvas_runtime import (
    MAX_ARCHIVE_BYTES,
    MAX_EXTRACTED_BYTES,
    VendorError,
    _RejectRedirects,
    _safe_member_name,
    _verify_sri,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tldw_chatbook/Canvas/mermaid"
STATIC = ROOT / "tldw_chatbook/Canvas/static"
AUTHORED = ("budget.js", "text.js", "semantic.js", "entry.js")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def pretty(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def verify_input(data: bytes, record: dict) -> bytes:
    if len(data) != record["bytes"] or digest(data) != record["sha256"]:
        raise VendorError("declared input integrity mismatch")
    return data


def selected_archive(payload: bytes, inputs: dict) -> dict[str, bytes]:
    """Authenticate full inventory; allocate only four selected inert members."""
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise VendorError("archive byte limit")
    _verify_sri(payload, inputs["mermaid_integrity"])
    selected = {}
    seen = set()
    selected_bytes = 0
    scanned_bytes = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            # Iteration is bounded before getmembers could allocate an unbounded list.
            for member in archive:
                if len(seen) >= 1200:
                    raise VendorError("archive member count limit")
                if not _safe_member_name(member.name) or not member.isfile():
                    raise VendorError("unsafe archive member")
                scanned_bytes += member.size
                if member.size < 0 or scanned_bytes > 96 * 1024 * 1024:
                    raise VendorError("archive scan byte limit")
                if member.name in seen:
                    raise VendorError("duplicate archive member")
                seen.add(member.name)
                if member.name not in inputs["members"]:
                    raise VendorError("unexpected archive member")
                if member.name not in inputs["selected"]:
                    continue
                selected_bytes += member.size
                if member.size < 0 or selected_bytes > MAX_EXTRACTED_BYTES:
                    raise VendorError("selected member byte limit")
                stream = archive.extractfile(member)
                if stream is None:
                    raise VendorError("missing selected archive member")
                data = stream.read(member.size + 1)
                if (
                    len(data) != member.size
                    or digest(data) != inputs["selected"][member.name]
                ):
                    raise VendorError("selected member integrity mismatch")
                selected[member.name] = data
    except tarfile.TarError as exc:
        raise VendorError("invalid archive") from exc
    if seen != set(inputs["members"]) or set(selected) != set(inputs["selected"]):
        raise VendorError("missing archive member")
    metadata = json.loads(selected["package/package.json"])
    if (metadata.get("name"), metadata.get("version"), metadata.get("license")) != (
        "mermaid",
        "11.17.2",
        "MIT",
    ):
        raise VendorError("package identity mismatch")
    return selected


def grammar_source(selected: dict[str, bytes], rule: dict) -> str:
    source_map = json.loads(selected[rule["member"]])
    sources, contents = source_map["sources"], source_map["sourcesContent"]
    if len(sources) != len(contents) or sources.count(rule["source"]) != 1:
        raise VendorError("grammar source-map entry mismatch")
    source = contents[sources.index(rule["source"])]
    if not isinstance(source, str) or digest(source.encode()) != rule["sha256"]:
        raise VendorError("grammar source integrity mismatch")
    # The hash authenticates every byte. This anchored ESM suffix check validates
    # the exact export contract; no marker-based search/extraction or JS evaluation.
    suffix = (
        "\tparser.parser = parser;\n\texport { parser };\n\texport default parser;\n\t"
    )
    if not source.endswith(suffix) or len(re.findall(r"\bexport\b", source)) != 2:
        raise VendorError("unexpected generated parser exports")
    body = source[: -len(suffix)]
    return f"const {rule['name']} = (() => {{\n{body}\nreturn parser;\n}})();\n"


def unicode_tables(files: dict[str, bytes]) -> dict:
    tables: dict[str, list] = {}
    for filename in (
        "GraphemeBreakProperty.txt",
        "emoji-data.txt",
        "EastAsianWidth.txt",
        "DerivedCoreProperties.txt",
    ):
        for line in files[filename].decode().splitlines():
            fields = [part.strip() for part in line.split("#", 1)[0].split(";")]
            if len(fields) < 2:
                continue
            prop = fields[1]
            if filename == "DerivedCoreProperties.txt":
                if prop != "InCB" or len(fields) != 3:
                    continue
                prop = "InCB_" + fields[2]
            elif filename == "GraphemeBreakProperty.txt":
                prop = "GCB_" + prop
            elif filename == "EastAsianWidth.txt":
                if prop not in {"W", "F"}:
                    continue
                prop = "Wide"
            elif prop not in {"Extended_Pictographic", "Emoji_Presentation"}:
                continue
            span = fields[0].split("..")
            lo, hi = int(span[0], 16), int(span[-1], 16)
            if not 0 <= lo <= hi <= 0x10FFFF:
                raise VendorError("invalid Unicode range")
            tables.setdefault(prop, []).append([lo, hi])
    for key, spans in tables.items():
        merged = []
        for lo, hi in sorted(spans):
            if merged and lo <= merged[-1][1] + 1:
                merged[-1][1] = max(hi, merged[-1][1])
            else:
                merged.append([lo, hi])
        tables[key] = merged
    return tables


def build(input_dir: Path, output_dir: Path) -> dict:
    inputs_bytes = (SOURCE / "inputs.json").read_bytes()
    inputs = json.loads(inputs_bytes)
    if ".".join(map(str, sys.version_info[:3])) != inputs["python"]:
        raise VendorError("use the pinned Python build version")
    files = {}
    for name, row in inputs["downloads"].items():
        with (input_dir / name).open("rb") as stream:
            data = stream.read(min(row["bytes"], MAX_ARCHIVE_BYTES) + 1)
        files[name] = verify_input(data, row)
    selected = selected_archive(files["mermaid-11.17.2.tgz"], inputs)
    tables = canonical(unicode_tables(files))
    authored = {name: (SOURCE / name).read_bytes() for name in AUTHORED}
    source = "(() => {\n'use strict';\n"
    source += "".join(grammar_source(selected, rule) for rule in inputs["grammar"])
    source += "const unicodeTables = " + tables.decode() + ";\n"
    source += "\n".join(data.decode() for data in authored.values()) + "\n})();\n"
    encoded = source.encode()
    if len(encoded) > 256 * 1024:
        raise VendorError("library exceeds unchanged evaluated-script byte ceiling")
    library = pretty(
        {
            "schema_version": 1,
            "profile_id": "canvas-v2-mermaid-1",
            "source": source,
            "source_bytes": len(encoded),
            "source_sha256": digest(encoded),
            "inputs_sha256": digest(inputs_bytes),
            "inventory": {name: digest(data) for name, data in authored.items()},
        }
    )
    notices = (
        b"Canvas Mermaid candidate: upstream Mermaid 11.17.2 (MIT), generated Jison 0.4.18 parsers.\n\n"
        + selected["package/LICENSE"]
        + b"\nUnicode 16.0.0 / UAX29 revision45 (Unicode-3.0):\n\n"
        + files["unicode-license.txt"]
    )
    manifest = json.loads((STATIC / "runtime-manifest.json").read_bytes())
    manifest["runtime_profile"] = "canvas-v2-mermaid-1"
    manifest["reproducible_command"] = (
        "python scripts/vendor_canvas_mermaid.py --input-dir INPUTS"
    )
    contract = manifest["profile_contract"]
    contract["quotas"].update(
        {
            "id": "canvas-v2-mermaid-quotas-1",
            "document_declarations": 4,
            "diagram_input_bytes": 8192,
            "document_input_bytes": 16384,
            "diagram_nodes": 16,
            "document_nodes": 24,
            "diagram_edges": 24,
            "document_edges": 32,
            "diagram_participants": 6,
            "document_participants": 8,
            "diagram_messages": 16,
            "document_messages": 24,
            "diagram_notes": 8,
            "document_notes": 12,
            "label_bytes": 512,
            "diagram_label_bytes": 4096,
            "document_label_bytes": 8192,
            "diagram_svg_elements": 250,
            "document_svg_elements": 400,
            "diagram_output_bytes": 49152,
            "document_output_bytes": 65536,
            "diagram_work_units": 10000,
            "document_work_units": 20000,
            "diagram_width": 2048,
            "diagram_height": 4096,
            "diagram_area": 4194304,
            "document_area": 8388608,
        }
    )
    contract["grammar"] = {
        "id": "mermaid-11.17.2-jison-0.4.18-canvas-subset-1",
        "sha256": digest(encoded),
    }
    contract["unicode"] = {
        "version": "16.0.0",
        "segmentation": inputs["unicode"]["segmentation"],
        "width": "canvas-cell-width-1",
        "sha256": digest(tables + authored["text.js"]),
    }
    # Existing engine/facade/plan/layout are exact real V1 assets until later
    # slices implement V2. This candidate is deliberately never executable.
    manifest["mermaid_candidate"] = {
        "inputs_sha256": digest(inputs_bytes),
        "source_bytes": len(encoded),
        "source_sha256": digest(encoded),
        "qualification": "not-qualified",
    }
    manifest_bytes = pretty(manifest)
    catalog = json.loads((STATIC / "profile-catalog.json").read_bytes())
    catalog["profiles"] = [
        row for row in catalog["profiles"] if row["profile_id"] != "canvas-v2-mermaid-1"
    ]
    catalog["profiles"].append(
        {
            "profile_id": "canvas-v2-mermaid-1",
            "manifest": "mermaid-runtime-manifest.json",
            "manifest_sha256": digest(manifest_bytes),
            "executable": False,
            "reason": "profile-unavailable",
            "library": {
                "bytes": len(library) + len(notices),
                "files": {
                    "mermaid-subset.json": {
                        "bytes": len(library),
                        "sha256": digest(library),
                    },
                    "MERMAID_THIRD_PARTY_LICENSES.txt": {
                        "bytes": len(notices),
                        "sha256": digest(notices),
                    },
                },
            },
        }
    )
    catalog["default_diagram_profile"] = None
    build_projection = [
        {key: row[key] for key in ("profile_id", "manifest_sha256", "library")}
        for row in catalog["profiles"]
    ]
    policy_projection = {
        "default_diagram_profile": None,
        "profiles": [
            {key: row[key] for key in ("profile_id", "executable", "reason")}
            for row in catalog["profiles"]
        ],
    }
    catalog["build_id"] = digest(canonical(build_projection))
    catalog["policy_id"] = digest(canonical(policy_projection))
    outputs = {
        "mermaid-subset.json": library,
        "MERMAID_THIRD_PARTY_LICENSES.txt": notices,
        "mermaid-runtime-manifest.json": manifest_bytes,
        "profile-catalog.json": pretty(catalog),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in outputs.items():
        (output_dir / name).write_bytes(data)
    return {
        name: {"bytes": len(data), "sha256": digest(data)}
        for name, data in outputs.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=STATIC)
    args = parser.parse_args()
    if args.input_dir:
        print(json.dumps(build(args.input_dir, args.output_dir), sort_keys=True))
        return
    inputs = json.loads((SOURCE / "inputs.json").read_bytes())
    with tempfile.TemporaryDirectory(prefix="canvas-mermaid-vendor-") as temp:
        input_dir = Path(temp)
        for name, row in inputs["downloads"].items():
            url = row["url"]
            if not url.startswith("https://"):
                raise VendorError("input URL must use HTTPS")
            with build_opener(_RejectRedirects()).open(
                Request(url), timeout=30
            ) as response:
                if response.geturl() != url:
                    raise VendorError("input redirect refused")
                data = response.read(min(row["bytes"], MAX_ARCHIVE_BYTES) + 1)
            (input_dir / name).write_bytes(verify_input(data, row))
        print(json.dumps(build(input_dir, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
