"""Transcribe every saved successful live-TTS clip with an explicit local ASR model."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import stat
import tempfile
import unicodedata
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Literal, NamedTuple


class AudioClip(NamedTuple):
    """Validated successful phase and its run-confined audio identity."""

    phase_id: str
    expected_text: str
    content_anchors: list[str]
    path: Path
    audio_sha256: str


def sha256(path: Path) -> str:
    """Hash a caller-validated local file in bounded blocks.

    Args:
        path: Validated file to read.

    Returns:
        Lowercase SHA256 digest of the complete file.

    Raises:
        OSError: The file cannot be read.
    """
    with path.open("rb") as source:
        return _stream_sha256(source)


def _stream_sha256(source: BinaryIO) -> str:
    digest = hashlib.sha256()
    source.seek(0)
    for block in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(block)
    source.seek(0)
    return digest.hexdigest()


def _validated_clips(evidence: object, run_root: Path) -> list[AudioClip]:
    # Keep imports/help stdlib-only. Validate only consumed fields, retaining
    # compatibility with the runner's richer provenance and partial phases.
    from pydantic import BaseModel, ConfigDict, Field, ValidationError

    class AudioEvidence(BaseModel):
        model_config = ConfigDict(strict=True)
        path: str = Field(min_length=1)
        sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    class PhaseEvidence(BaseModel):
        model_config = ConfigDict(strict=True)
        id: str = Field(min_length=1)
        outcome: (
            Literal[
                "success", "cancelled", "running", "failed", "expected_server_failure"
            ]
            | None
        ) = None
        audio: AudioEvidence | None = None
        expected_text: str | None = None
        content_anchors: list[str] = Field(default_factory=list)

    class Evidence(BaseModel):
        model_config = ConfigDict(strict=True)
        phases: list[PhaseEvidence]

    try:
        validated = Evidence.model_validate(evidence)
    except ValidationError:
        raise ValueError("Invalid evidence structure or audio hash") from None
    rows = []
    seen = set()
    for phase in validated.phases:
        if phase.id in seen:
            raise ValueError("Evidence phase IDs must be unique")
        seen.add(phase.id)
        if phase.outcome != "success":
            continue
        if phase.audio is None:
            raise ValueError("Successful phase has no audio evidence")
        if not phase.expected_text or not phase.expected_text.strip():
            raise ValueError("Successful phase has no expected text")
        rows.append(
            AudioClip(
                phase.id,
                phase.expected_text,
                phase.content_anchors,
                _audio_path(phase.audio.path, run_root),
                phase.audio.sha256,
            )
        )
    if not rows:
        raise ValueError("No successful audio to transcribe")
    return rows


def _audio_path(raw: str, run_root: Path) -> Path:
    from tldw_chatbook.Utils.path_validation import validate_path

    validated = validate_path(raw, run_root, allow_hidden=True, redact_paths=True)
    supplied = Path(raw)
    if not supplied.is_absolute():
        supplied = run_root / supplied
    # Do not normalize away a link or traversal before no-follow opening.
    if supplied != validated:
        raise ValueError("Audio path must not contain symlinks or traversal")
    return validated


@contextmanager
def _open_audio(path: Path, run_root: Path) -> Iterator[BinaryIO]:
    # Descriptor-relative traversal prevents a swapped descendant from
    # redirecting a validated path between validation and ASR consumption.
    if os.open not in os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW"):
        raise ValueError(
            "Audio verification requires descriptor-relative no-follow file access"
        )
    relative = path.relative_to(run_root)
    if not relative.parts or ".." in relative.parts:
        raise ValueError("Audio must be a file inside the evidence run")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    with ExitStack() as stack:
        parent = os.open(run_root, flags)
        stack.callback(os.close, parent)
        for part in relative.parts[:-1]:
            parent = os.open(part, flags, dir_fd=parent)
            stack.callback(os.close, parent)
        fd = os.open(
            relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
        )
        source = stack.enter_context(os.fdopen(fd, "rb"))
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise ValueError("Audio must be a regular file")
        yield source


def success_audio(evidence: object, run_root: Path) -> list[AudioClip]:
    """Validate and hash every successful phase without reducing the denominator.

    Args:
        evidence: Parsed runtime JSON, before structural validation.
        run_root: Canonical parent of the CLI-selected evidence file.

    Returns:
        Validated successful clips, including retries and repetitions.

    Raises:
        ValueError: Shape, path confinement, complete audio, or hashes are invalid.
        OSError: A no-follow regular-file read fails.
    """
    rows = _validated_clips(evidence, run_root)
    for row in rows:
        with _open_audio(row.path, run_root) as source:
            if _stream_sha256(source) != row.audio_sha256:
                raise ValueError("Audio hash changed")
    return rows


def normalize(text: str) -> str:
    """Remove case, whitespace and punctuation differences while retaining words.

    Args:
        text: Expected or recognized text.

    Returns:
        Comparable Unicode text, retaining meaningful combining marks.
    """
    return "".join(
        char
        for char in text.casefold()
        if not char.isspace() and not unicodedata.category(char).startswith("P")
    )


def compare_content(
    expected: str, transcript: str, anchors: list[str]
) -> dict[str, Any]:
    """Compare the complete utterance and the order of its content anchors.

    Args:
        expected: Complete intended utterance.
        transcript: Complete recognized utterance, preserved without rewriting.
        anchors: Ordered beginning, middle and end phrases.

    Returns:
        Original text, anchor positions and exact/ordered comparison results.
    """
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


def _load_whisper() -> Any:
    # optional_deps consults app configuration when first imported outside
    # pytest. Keep that lookup inside a disposable profile, never the user's.
    with tempfile.TemporaryDirectory(prefix="tts-content-profile-") as directory:
        root = Path(directory)
        overrides = {
            "TLDW_CONFIG_PATH": str(root / "config.toml"),
            "XDG_CONFIG_HOME": str(root / "config"),
            "XDG_DATA_HOME": str(root / "data"),
            "XDG_CACHE_HOME": str(root / "cache"),
            "TLDW_EAGER_DEPENDENCY_CHECK": "false",
            "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring",
        }
        original = {name: os.environ.get(name) for name in overrides}
        os.environ.update(overrides)
        try:
            from tldw_chatbook.Utils.optional_deps import require_dependency

            return require_dependency("faster_whisper", "transcription_faster_whisper")
        finally:
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _verify(args: argparse.Namespace) -> int:
    from tldw_chatbook.Utils.path_validation import validate_path_simple

    for name in ("evidence", "model"):
        path = validate_path_simple(
            getattr(args, name).expanduser(), require_exists=True
        )
        setattr(args, name, path.resolve())
    output = validate_path_simple(args.output.expanduser(), probe_existing=False)
    if os.path.lexists(output) or not output.parent.is_dir():
        raise ValueError("output must be new, with an existing parent directory")
    args.output = output.parent.resolve() / output.name
    if not args.evidence.is_file():
        raise ValueError("evidence must be a regular JSON file")
    if not args.model.is_dir() or not (args.model / "model.bin").is_file():
        raise ValueError("model must be an existing local faster-whisper snapshot")
    evidence_bytes = args.evidence.read_bytes()
    evidence_hash = hashlib.sha256(evidence_bytes).hexdigest()
    evidence = json.loads(evidence_bytes)
    run_root = args.evidence.parent
    rows = success_audio(evidence, run_root)
    for row in rows:
        anchors = args.anchor or row.content_anchors
        if len(anchors) < 3 or any(not normalize(anchor) for anchor in anchors):
            raise ValueError(
                "Provide at least three nonempty --anchor phrases for custom/multilingual text"
            )
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    whisper = _load_whisper()
    model = whisper.WhisperModel(
        str(args.model), device="cpu", compute_type="int8", local_files_only=True
    )
    report = {
        "evidence_path": str(args.evidence),
        "evidence_sha256": evidence_hash,
        "model": str(args.model),
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
        with _open_audio(row.path, run_root) as source:
            if _stream_sha256(source) != row.audio_sha256:
                raise ValueError("Audio hash changed before transcription")
            segments, info = model.transcribe(
                source, language=args.language, beam_size=5
            )
            saved = [
                {"start": item.start, "end": item.end, "text": item.text}
                for item in segments
            ]
            if _stream_sha256(source) != row.audio_sha256:
                raise ValueError("Audio hash changed during transcription")
        transcript = " ".join(item["text"].strip() for item in saved)
        result = compare_content(
            row.expected_text, transcript, args.anchor or row.content_anchors
        )
        result.update(
            phase_id=row.phase_id,
            audio_sha256=row.audio_sha256,
            segments=saved,
            detected_language=info.language,
        )
        report["clips"].append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    if sha256(args.evidence) != evidence_hash:
        raise ValueError("Runtime evidence changed during transcription")
    success_audio(evidence, run_root)
    passed = all(
        row["normalized_exact"] and row["anchors_complete"] for row in report["clips"]
    )
    report["status"] = "content_passed" if passed else "content_review_required"
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as output:
        json.dump(report, output, indent=2, ensure_ascii=False)
        output.write("\n")
    return 0 if passed else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Validate inputs and run optional local full-content transcription.

    Args:
        argv: CLI arguments, or None to read the process arguments.

    Returns:
        Zero for exact complete content, one for content requiring review.

    Raises:
        SystemExit: Help was requested or input/dependency validation failed.
        RuntimeError: The local recognizer fails during inference.
    """
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
    try:
        return _verify(args)
    except (OSError, ValueError, ImportError) as error:
        parser.error(str(error))
    return 2  # argparse.error always exits.


if __name__ == "__main__":
    raise SystemExit(main())
