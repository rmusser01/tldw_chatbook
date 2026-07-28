#!/usr/bin/env python3
"""Run an indicative local macOS comparison of five speech-to-text models."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
import gc
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import unicodedata
import wave


REPORT_LABEL = "indicative_macos_comparison"
MODEL_IDS = (
    "parakeet_v2_int8",
    "parakeet_v2_f32",
    "parakeet_v3_int8",
    "parakeet_v3_f32",
    "faster_whisper",
)
V3_LANGUAGES = frozenset(
    {
        "bg",
        "hr",
        "cs",
        "da",
        "nl",
        "et",
        "fi",
        "fr",
        "de",
        "el",
        "hu",
        "it",
        "lv",
        "lt",
        "mt",
        "pl",
        "pt",
        "ro",
        "sk",
        "sl",
        "es",
        "sv",
        "ru",
        "uk",
    }
)
REQUIRED_MODEL_FILES = {
    "v2_int8": frozenset(
        {
            "config.json",
            "vocab.txt",
            "encoder-model.int8.onnx",
            "decoder_joint-model.int8.onnx",
        }
    ),
    "v2_f32": frozenset(
        {
            "config.json",
            "vocab.txt",
            "encoder-model.onnx",
            "encoder-model.onnx.data",
            "decoder_joint-model.onnx",
        }
    ),
    "v3_int8": frozenset(
        {
            "config.json",
            "vocab.txt",
            "encoder-model.int8.onnx",
            "decoder_joint-model.int8.onnx",
        }
    ),
    "v3_f32": frozenset(
        {
            "config.json",
            "vocab.txt",
            "encoder-model.onnx",
            "encoder-model.onnx.data",
            "decoder_joint-model.onnx",
        }
    ),
    "faster_whisper": frozenset(
        {
            "config.json",
            "model.bin",
            "tokenizer.json",
            "vocabulary.txt",
        }
    ),
}

Case = dict[str, object]
Row = dict[str, object]
Transcriber = Callable[[Case], str]
ModelLoader = Callable[[], Transcriber]


def normalize_text(text: str) -> str:
    """Normalize transcript text for Unicode-aware WER and CER."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    without_punctuation = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in normalized
    )
    return " ".join(without_punctuation.split())


def edit_distance(
    reference: Sequence[str],
    hypothesis: Sequence[str],
) -> int:
    """Return the Levenshtein edit distance between two unit sequences."""

    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_unit in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_unit in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1]
                    + (reference_unit != hypothesis_unit),
                )
            )
        previous = current
    return previous[-1]


def _wav_identity(path: Path, case_id: str) -> tuple[float, str]:
    """Validate one WAV and return its duration and SHA-256 digest."""

    if not path.is_file():
        raise ValueError(f"case {case_id!r} audio does not exist: {path}")
    try:
        with wave.open(str(path), "rb") as audio:
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            frame_rate = audio.getframerate()
            frames = audio.getnframes()
            compression = audio.getcomptype()
            payload_bytes = len(audio.readframes(frames))
    except (EOFError, OSError, wave.Error) as error:
        raise ValueError(f"case {case_id!r} is not a readable WAV: {path}") from error
    if compression != "NONE":
        raise ValueError(f"case {case_id!r} WAV must use uncompressed PCM")
    if sample_width != 2:
        raise ValueError(f"case {case_id!r} WAV must use signed 16-bit samples")
    if frame_rate != 16_000:
        raise ValueError(f"case {case_id!r} WAV must use a 16 kHz sample rate")
    if channels != 1:
        raise ValueError(f"case {case_id!r} WAV must be mono")
    if frames < 1:
        raise ValueError(f"case {case_id!r} WAV must contain audio frames")
    expected_payload_bytes = frames * channels * sample_width
    if payload_bytes < expected_payload_bytes:
        raise ValueError(f"case {case_id!r} WAV payload is truncated")

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return frames / frame_rate, digest.hexdigest()


def _required_string(
    record: Mapping[str, object],
    field: str,
    line_number: int,
    *,
    allow_empty: bool = False,
) -> str:
    value = record.get(field)
    if not isinstance(value, str):
        raise ValueError(f"case line {line_number} field {field!r} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"case line {line_number} field {field!r} cannot be empty")
    return value


def load_cases(path: Path) -> list[dict[str, object]]:
    """Load and validate local JSONL speech cases."""

    path = Path(path)
    if not path.is_file():
        raise ValueError(f"case file does not exist: {path}")
    case_directory = path.resolve().parent
    seen_ids: set[str] = set()
    cases: list[Case] = []

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read case file: {path}") from error

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"case line {line_number} is not valid JSON: {error.msg}"
            ) from error
        if not isinstance(record, dict):
            raise ValueError(f"case line {line_number} must be a JSON object")

        case_id = _required_string(record, "id", line_number).strip()
        if case_id in seen_ids:
            raise ValueError(f"duplicate case id: {case_id!r}")
        seen_ids.add(case_id)

        audio_value = _required_string(record, "audio", line_number).strip()
        reference = _required_string(
            record,
            "reference",
            line_number,
            allow_empty=True,
        )
        language = _required_string(record, "language", line_number).strip().casefold()
        tag = _required_string(record, "tag", line_number).strip().casefold()
        if tag != "silence" and not normalize_text(reference):
            raise ValueError(
                f"case {case_id!r} reference may be empty only for tag 'silence'"
            )

        audio_path = Path(audio_value).expanduser()
        if not audio_path.is_absolute():
            audio_path = case_directory / audio_path
        audio_path = audio_path.resolve()
        duration, audio_sha256 = _wav_identity(audio_path, case_id)
        relative_audio = Path(
            os.path.relpath(audio_path, start=case_directory)
        ).as_posix()
        cases.append(
            {
                "id": case_id,
                "audio": audio_path,
                "audio_relative": relative_audio,
                "reference": reference,
                "language": language,
                "tag": tag,
                "audio_duration_seconds": duration,
                "audio_sha256": audio_sha256,
            }
        )

    if not cases:
        raise ValueError(f"case file contains no cases: {path}")
    return cases


def scheduled_models(case: dict[str, object]) -> tuple[str, ...]:
    """Return model IDs scheduled for one validated case."""

    if case.get("tag") == "silence":
        return MODEL_IDS
    language = case.get("language")
    if language == "en":
        return (
            "parakeet_v2_int8",
            "parakeet_v2_f32",
            "faster_whisper",
        )
    if language in V3_LANGUAGES:
        return (
            "parakeet_v3_int8",
            "parakeet_v3_f32",
            "faster_whisper",
        )
    return ("faster_whisper",)


def _error_text(error: BaseException) -> str:
    message = " ".join(str(error).split())
    detail = f": {message}" if message else ""
    return f"{type(error).__name__}{detail}"[:500]


def _base_row(case: Case, model_id: str) -> Row:
    normalized_reference = normalize_text(str(case["reference"]))
    reference_words = normalized_reference.split()
    reference_characters = [
        character for character in normalized_reference if not character.isspace()
    ]
    return {
        "case_id": str(case["id"]),
        "model_id": model_id,
        "reference": normalized_reference,
        "hypothesis": "",
        "word_edits": None,
        "word_reference_units": len(reference_words),
        "character_edits": None,
        "character_reference_units": len(reference_characters),
        "elapsed_seconds": None,
        "audio_duration_seconds": float(case["audio_duration_seconds"]),
        "rtf": None,
        "audio_sha256": str(case["audio_sha256"]),
        "error": None,
        "tag": str(case["tag"]),
    }


def _failure_row(
    case: Case,
    model_id: str,
    error: BaseException,
    *,
    elapsed_seconds: float | None = None,
) -> Row:
    row = _base_row(case, model_id)
    duration = float(row["audio_duration_seconds"])
    row.update(
        {
            "elapsed_seconds": elapsed_seconds,
            "rtf": elapsed_seconds / duration if elapsed_seconds is not None else None,
            "error": _error_text(error),
        }
    )
    return row


def _success_row(
    case: Case,
    model_id: str,
    hypothesis: str,
    elapsed_seconds: float,
) -> Row:
    row = _base_row(case, model_id)
    normalized_hypothesis = normalize_text(hypothesis)
    duration = float(row["audio_duration_seconds"])
    row.update(
        {
            "hypothesis": normalized_hypothesis,
            "elapsed_seconds": elapsed_seconds,
            "rtf": elapsed_seconds / duration,
        }
    )
    if case["tag"] == "silence":
        return row

    reference = str(row["reference"])
    reference_words = reference.split()
    hypothesis_words = normalized_hypothesis.split()
    reference_characters = [
        character for character in reference if not character.isspace()
    ]
    hypothesis_characters = [
        character for character in normalized_hypothesis if not character.isspace()
    ]
    row["word_edits"] = edit_distance(reference_words, hypothesis_words)
    row["character_edits"] = edit_distance(
        reference_characters,
        hypothesis_characters,
    )
    return row


def _summarize(rows: Sequence[Row]) -> dict[str, object]:
    model_summaries: dict[str, dict[str, object]] = {}
    for model_id in MODEL_IDS:
        model_rows = [row for row in rows if row["model_id"] == model_id]
        if not model_rows:
            continue
        successful_rows = [row for row in model_rows if row["error"] is None]
        attempted_rows = [
            row for row in model_rows if row["elapsed_seconds"] is not None
        ]
        quality_rows = [row for row in successful_rows if row["tag"] != "silence"]
        word_edits = sum(int(row["word_edits"]) for row in quality_rows)
        word_units = sum(int(row["word_reference_units"]) for row in quality_rows)
        character_edits = sum(int(row["character_edits"]) for row in quality_rows)
        character_units = sum(
            int(row["character_reference_units"]) for row in quality_rows
        )
        elapsed = sum(float(row["elapsed_seconds"]) for row in attempted_rows)
        audio_duration = sum(
            float(row["audio_duration_seconds"]) for row in attempted_rows
        )
        model_summaries[model_id] = {
            "scheduled_cases": len(model_rows),
            "successful_cases": len(successful_rows),
            "error_cases": len(model_rows) - len(successful_rows),
            "word_edits": word_edits,
            "word_reference_units": word_units,
            "wer": word_edits / word_units if word_units else None,
            "character_edits": character_edits,
            "character_reference_units": character_units,
            "cer": character_edits / character_units if character_units else None,
            "elapsed_seconds": elapsed,
            "audio_duration_seconds": audio_duration,
            "rtf": elapsed / audio_duration if audio_duration else None,
        }
    return {
        "models": model_summaries,
        "silence": [row for row in rows if row["tag"] == "silence"],
    }


def run_comparison(
    cases: Sequence[dict[str, object]],
    model_runners: Mapping[str, ModelLoader],
) -> tuple[list[dict], dict, bool]:
    """Load models sequentially and run every scheduled case.

    ``model_runners`` maps each model ID to a zero-argument loader. A loader
    returns a callable that accepts one case and returns transcript text.
    """

    rows: list[Row] = []
    for model_id in MODEL_IDS:
        scheduled_cases = [case for case in cases if model_id in scheduled_models(case)]
        if not scheduled_cases:
            continue
        loader = model_runners.get(model_id)
        if loader is None:
            load_error: BaseException = KeyError(
                f"no model loader configured for {model_id}"
            )
            for case in scheduled_cases:
                row = _failure_row(case, model_id, load_error)
                rows.append(row)
                print(
                    f"{model_id}/{case['id']}: {row['error']}",
                    file=sys.stderr,
                )
            continue
        try:
            transcribe = loader()
            if not callable(transcribe):
                raise TypeError("model loader did not return a callable")
        except Exception as error:
            load_error = RuntimeError(f"model load failed: {_error_text(error)}")
            for case in scheduled_cases:
                row = _failure_row(case, model_id, load_error)
                rows.append(row)
                print(
                    f"{model_id}/{case['id']}: {row['error']}",
                    file=sys.stderr,
                )
            continue

        try:
            for case in scheduled_cases:
                started = time.perf_counter()
                try:
                    hypothesis = transcribe(case)
                    if not isinstance(hypothesis, str):
                        raise TypeError("transcriber did not return text")
                except Exception as error:
                    elapsed = time.perf_counter() - started
                    row = _failure_row(
                        case,
                        model_id,
                        error,
                        elapsed_seconds=elapsed,
                    )
                    rows.append(row)
                    print(
                        f"{model_id}/{case['id']}: {row['error']}",
                        file=sys.stderr,
                    )
                    continue
                elapsed = time.perf_counter() - started
                rows.append(
                    _success_row(
                        case,
                        model_id,
                        hypothesis,
                        elapsed,
                    )
                )
        finally:
            del transcribe
            gc.collect()

    summary = _summarize(rows)
    has_errors = any(row["error"] is not None for row in rows)
    return rows, summary, has_errors


def _build_model_runners(
    directories: Mapping[str, Path],
) -> dict[str, ModelLoader]:
    """Build local-only loader closures for the five compared models."""

    def parakeet_loader(
        model_name: str,
        directory: Path,
        quantization: str | None,
    ) -> ModelLoader:
        def load() -> Transcriber:
            import onnx_asr

            model = onnx_asr.load_model(
                model_name,
                path=directory,
                quantization=quantization,
                providers=["CPUExecutionProvider"],
            )

            def transcribe(case: Case) -> str:
                return model.recognize(str(case["audio"]))

            return transcribe

        return load

    def load_faster_whisper() -> Transcriber:
        from faster_whisper import WhisperModel

        model = WhisperModel(
            str(directories["faster_whisper"]),
            device="cpu",
            compute_type="int8",
            local_files_only=True,
        )

        def transcribe(case: Case) -> str:
            options = (
                {} if case["tag"] == "silence" else {"language": str(case["language"])}
            )
            segments, _information = model.transcribe(
                str(case["audio"]),
                **options,
            )
            materialized_segments = list(segments)
            return " ".join(
                str(segment.text).strip()
                for segment in materialized_segments
                if str(segment.text).strip()
            )

        return transcribe

    return {
        "parakeet_v2_int8": parakeet_loader(
            "nemo-parakeet-tdt-0.6b-v2",
            directories["v2_int8"],
            "int8",
        ),
        "parakeet_v2_f32": parakeet_loader(
            "nemo-parakeet-tdt-0.6b-v2",
            directories["v2_f32"],
            None,
        ),
        "parakeet_v3_int8": parakeet_loader(
            "nemo-parakeet-tdt-0.6b-v3",
            directories["v3_int8"],
            "int8",
        ),
        "parakeet_v3_f32": parakeet_loader(
            "nemo-parakeet-tdt-0.6b-v3",
            directories["v3_f32"],
            None,
        ),
        "faster_whisper": load_faster_whisper,
    }


def _looks_like_repository_id(path: Path) -> bool:
    return (
        not path.is_absolute()
        and len(path.parts) == 2
        and all(
            part not in {".", ".."} and not part.startswith(".") for part in path.parts
        )
    )


def _validate_model_directories(
    directories: Mapping[str, Path],
) -> dict[str, Path]:
    validated: dict[str, Path] = {}
    for name, raw_path in directories.items():
        path = Path(raw_path)
        if _looks_like_repository_id(path) and not path.exists():
            raise ValueError(
                f"{name} must be a local directory, not a repository ID: {path}"
            )
        if not path.exists():
            raise ValueError(f"{name} model path does not exist: {path}")
        if not path.is_dir():
            raise ValueError(f"{name} model path is not a directory: {path}")
        validated[name] = path.resolve()
    for name, path in validated.items():
        for filename in sorted(REQUIRED_MODEL_FILES[name]):
            if not (path / filename).is_file():
                raise ValueError(
                    f"{name} model directory is missing required file: {filename}"
                )
    return validated


def _selected_filenames(model_id: str) -> set[str] | None:
    if model_id == "faster_whisper":
        return {
            "config.json",
            "model.bin",
            "preprocessor_config.json",
            "tokenizer.json",
            "vocabulary.json",
            "vocabulary.txt",
        }
    if not model_id.startswith("parakeet_"):
        return None
    quantized = model_id.endswith("_int8")
    suffix = ".int8.onnx" if quantized else ".onnx"
    filenames = {
        "config.json",
        "vocab.txt",
        f"encoder-model{suffix}",
        f"decoder_joint-model{suffix}",
    }
    if not quantized:
        filenames.add("encoder-model.onnx.data")
    return filenames


def _model_identity(directory: Path, model_id: str) -> dict[str, object]:
    selected_filenames = _selected_filenames(model_id)
    files = [
        {
            "name": file.relative_to(directory).as_posix(),
            "size_bytes": file.stat().st_size,
        }
        for file in sorted(directory.rglob("*"))
        if file.is_file()
        and (
            selected_filenames is None
            or file.relative_to(directory).as_posix() in selected_filenames
        )
    ]
    return {"directory_name": directory.name, "files": files}


def _environment_identity() -> dict[str, object]:
    packages = {}
    for package in ("onnx-asr", "onnxruntime", "faster-whisper", "ctranslate2"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "system": {
            "name": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "mac_version": platform.mac_ver()[0] or None,
        },
        "packages": packages,
    }


def write_report(path: Path, report: dict[str, object]) -> None:
    """Atomically replace ``path`` with a UTF-8 JSON report."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(
                report,
                temporary,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path, help="Local JSONL cases")
    parser.add_argument(
        "--v2-int8",
        required=True,
        type=Path,
        help="Local Parakeet v2 INT8 model directory",
    )
    parser.add_argument(
        "--v2-f32",
        required=True,
        type=Path,
        help="Local Parakeet v2 F32 model directory",
    )
    parser.add_argument(
        "--v3-int8",
        required=True,
        type=Path,
        help="Local Parakeet v3 INT8 model directory",
    )
    parser.add_argument(
        "--v3-f32",
        required=True,
        type=Path,
        help="Local Parakeet v3 F32 model directory",
    )
    parser.add_argument(
        "--faster-whisper",
        required=True,
        type=Path,
        help="Local faster-whisper model directory",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output JSON report")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the comparison and return a process status code."""

    args = _parse_args(argv)
    try:
        cases_path = args.cases.expanduser().resolve()
        output_path = args.output.expanduser().resolve()
        cases = load_cases(cases_path)
        directories = _validate_model_directories(
            {
                "v2_int8": args.v2_int8,
                "v2_f32": args.v2_f32,
                "v3_int8": args.v3_int8,
                "v3_f32": args.v3_f32,
                "faster_whisper": args.faster_whisper,
            }
        )
        if output_path == cases_path:
            raise ValueError("--output cannot replace the case JSONL file")
        if any(output_path == Path(case["audio"]) for case in cases):
            raise ValueError("--output cannot replace a case audio file")
        for name, directory in directories.items():
            if output_path.is_relative_to(directory):
                raise ValueError(
                    f"--output cannot be inside the {name} model directory"
                )
        if output_path.exists() and not output_path.is_file():
            raise ValueError("--output must be a file path")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model_directories = {
            "parakeet_v2_int8": directories["v2_int8"],
            "parakeet_v2_f32": directories["v2_f32"],
            "parakeet_v3_int8": directories["v3_int8"],
            "parakeet_v3_f32": directories["v3_f32"],
            "faster_whisper": directories["faster_whisper"],
        }
        models = {
            model_id: _model_identity(model_directories[model_id], model_id)
            for model_id in MODEL_IDS
        }
        runners = _build_model_runners(directories)
        rows, summary, has_errors = run_comparison(cases, runners)
        report: dict[str, object] = {
            "report_label": REPORT_LABEL,
            "generated_at": datetime.now(UTC).isoformat(),
            "environment": _environment_identity(),
            "models": models,
            "cases": [
                {
                    "id": case["id"],
                    "audio": case["audio_relative"],
                    "reference": case["reference"],
                    "normalized_reference": normalize_text(str(case["reference"])),
                    "language": case["language"],
                    "tag": case["tag"],
                    "audio_duration_seconds": case["audio_duration_seconds"],
                    "audio_sha256": case["audio_sha256"],
                }
                for case in cases
            ],
            "rows": rows,
            "aggregates": summary["models"],
            "silence": summary["silence"],
            "has_errors": has_errors,
        }
        write_report(output_path, report)
    except (OSError, ValueError) as error:
        print(f"stt model comparison: {_error_text(error)}", file=sys.stderr)
        return 2
    return 1 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
