"""Transcribe every saved successful live-TTS clip with an explicit local ASR model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import unicodedata
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def success_audio(evidence: dict) -> list[dict]:
    """Keep the denominator fixed to every success, including retries/repetitions."""
    rows = []
    for row in evidence["phases"]:
        if row.get("outcome") != "success":
            continue
        audio = row.get("audio")
        if not audio or not audio.get("path"):
            raise ValueError(f"Successful phase {row['id']} has no audio evidence")
        path = Path(audio["path"])
        if not path.is_file() or sha256(path) != audio["sha256"]:
            raise ValueError(
                f"Audio hash changed or file is missing for phase {row['id']}"
            )
        rows.append(row)
    if not rows:
        raise ValueError("No successful audio to transcribe")
    return rows


def normalize(text: str) -> str:
    # Combining vowel/nasal marks carry meaning in languages such as Hindi.
    return "".join(
        char
        for char in text.casefold()
        if not char.isspace() and not unicodedata.category(char).startswith("P")
    )


def compare_content(expected: str, transcript: str, anchors: list[str]) -> dict:
    normalized = normalize(transcript)
    position = 0
    positions = []
    for anchor in anchors:
        wanted = normalize(anchor)
        found = normalized.find(wanted, position) if wanted else -1
        positions.append(found)
        if found >= 0:
            position = found + len(wanted)
    return {
        "transcript": transcript,
        "expected_text": expected,
        "anchors": anchors,
        "anchor_positions": positions,
        "anchors_complete": bool(anchors) and all(p >= 0 for p in positions),
        "normalized_exact": normalize(expected) == normalized,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Existing local faster-whisper model directory",
    )
    parser.add_argument(
        "--language", help="ASR language code; omitted means multilingual detection"
    )
    parser.add_argument(
        "--anchor",
        action="append",
        default=[],
        help="Repeat for beginning/middle/end phrases",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("output must be new, with an existing parent directory")
    if not args.model.is_dir() or not (args.model / "model.bin").is_file():
        parser.error("model must be an existing local faster-whisper snapshot")
    evidence_hash = sha256(args.evidence)
    evidence = json.loads(args.evidence.read_text())
    rows = success_audio(evidence)
    for row in rows:
        anchors = args.anchor or row.get("content_anchors", [])
        if len(anchors) < 3:
            parser.error(
                "Provide at least three --anchor phrases for custom/multilingual text"
            )
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from faster_whisper import WhisperModel

    model = WhisperModel(
        str(args.model.resolve()),
        device="cpu",
        compute_type="int8",
        local_files_only=True,
    )
    report = {
        "evidence_path": str(args.evidence.resolve()),
        "evidence_sha256": evidence_hash,
        "model": str(args.model.resolve()),
        "model_files": {
            str(path.relative_to(args.model)): sha256(path)
            for path in sorted(args.model.rglob("*"))
            if path.is_file()
        },
        "runtime": {
            name: importlib.metadata.version(name)
            for name in ("faster-whisper", "ctranslate2")
        },
        "normalization": "Unicode case, punctuation and whitespace only; lexical differences preserved",
        "device": "cpu",
        "compute_type": "int8",
        "success_audio_count": len(rows),
        "clips": [],
    }
    for row in rows:
        segments, info = model.transcribe(
            row["audio"]["path"], language=args.language, beam_size=5
        )
        saved = [
            {"start": item.start, "end": item.end, "text": item.text}
            for item in segments
        ]
        transcript = " ".join(item["text"].strip() for item in saved)
        result = compare_content(
            row["expected_text"], transcript, args.anchor or row["content_anchors"]
        )
        result.update(
            phase_id=row["id"],
            audio_sha256=row["audio"]["sha256"],
            segments=saved,
            detected_language=info.language,
        )
        report["clips"].append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    if sha256(args.evidence) != evidence_hash:
        raise ValueError("Runtime evidence changed during transcription")
    success_audio(evidence)
    passed = all(
        row["normalized_exact"] and row["anchors_complete"] for row in report["clips"]
    )
    report["status"] = "content_passed" if passed else "content_review_required"
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, ensure_ascii=False)
        output.write("\n")
    args.output.chmod(0o600)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
