from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import wave
import weakref

import pytest

from Helper_Scripts.Benchmarks import stt_model_comparison as comparison


MODEL_IDS = (
    "parakeet_v2_int8",
    "parakeet_v2_f32",
    "parakeet_v3_int8",
    "parakeet_v3_f32",
    "faster_whisper",
)
MODEL_FLAGS = (
    "v2_int8",
    "v2_f32",
    "v3_int8",
    "v3_f32",
    "faster_whisper",
)
SUPPORTED_V3_LANGUAGES = {
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


def _write_wav(
    path: Path,
    *,
    channels: int = 1,
    sample_width: int = 2,
    frame_rate: int = 16_000,
    seconds: int = 1,
) -> Path:
    frame = b"\0" * sample_width * channels
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(sample_width)
        audio.setframerate(frame_rate)
        audio.writeframes(frame * frame_rate * seconds)
    return path


def _write_cases(path: Path, cases: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(case) + "\n" for case in cases),
        encoding="utf-8",
    )
    return path


def _case(
    case_id: str,
    *,
    audio: str = "audio.wav",
    reference: str = "hello world",
    language: str = "en",
    tag: str = "clean",
) -> dict[str, object]:
    return {
        "id": case_id,
        "audio": audio,
        "reference": reference,
        "language": language,
        "tag": tag,
    }


def _loaded_case(
    case_id: str,
    *,
    reference: str,
    language: str,
    tag: str = "clean",
    duration: float = 2.0,
) -> dict[str, object]:
    return {
        "id": case_id,
        "audio": Path(f"/audio/{case_id}.wav"),
        "audio_relative": f"audio/{case_id}.wav",
        "reference": reference,
        "language": language,
        "tag": tag,
        "audio_duration_seconds": duration,
        "audio_sha256": f"hash-{case_id}",
    }


def _model_directories(tmp_path: Path) -> dict[str, Path]:
    directories = {}
    for flag in MODEL_FLAGS:
        directory = tmp_path / flag
        directory.mkdir()
        if flag == "faster_whisper":
            for filename in (
                "config.json",
                "model.bin",
                "tokenizer.json",
                "preprocessor_config.json",
                "vocabulary.txt",
            ):
                (directory / filename).write_bytes(filename.encode())
        else:
            quantization = "int8" if flag.endswith("int8") else "f32"
            suffix = ".int8.onnx" if quantization == "int8" else ".onnx"
            for filename in (
                "config.json",
                "vocab.txt",
                f"encoder-model{suffix}",
                f"decoder_joint-model{suffix}",
            ):
                (directory / filename).write_bytes(filename.encode())
            if quantization == "f32":
                (directory / "encoder-model.onnx.data").write_bytes(b"external")
            (directory / "unused.bin").write_bytes(b"unused")
        directories[flag] = directory
    return directories


def _main_args(
    cases: Path,
    directories: dict[str, Path],
    output: Path,
) -> list[str]:
    return [
        "--cases",
        str(cases),
        "--v2-int8",
        str(directories["v2_int8"]),
        "--v2-f32",
        str(directories["v2_f32"]),
        "--v3-int8",
        str(directories["v3_int8"]),
        "--v3-f32",
        str(directories["v3_f32"]),
        "--faster-whisper",
        str(directories["faster_whisper"]),
        "--output",
        str(output),
    ]


def test_normalize_text_is_unicode_aware() -> None:
    assert comparison.normalize_text(" Héllo—МИР! ") == "héllo мир"


def test_edit_distance_counts_insertions() -> None:
    assert comparison.edit_distance(["a", "b"], ["a", "x", "b"]) == 1


def test_load_cases_resolves_audio_relative_to_jsonl(tmp_path: Path) -> None:
    audio = _write_wav(tmp_path / "sample.wav")
    cases_path = _write_cases(
        tmp_path / "cases.jsonl",
        [_case("case-1", audio=audio.name)],
    )

    cases = comparison.load_cases(cases_path)

    assert cases[0]["audio"] == audio.resolve()
    assert cases[0]["audio_relative"] == "sample.wav"
    assert cases[0]["audio_duration_seconds"] == pytest.approx(1.0)
    assert cases[0]["audio_sha256"] == hashlib.sha256(audio.read_bytes()).hexdigest()


def test_load_cases_rejects_duplicate_ids(tmp_path: Path) -> None:
    _write_wav(tmp_path / "audio.wav")
    cases_path = _write_cases(
        tmp_path / "cases.jsonl",
        [_case("duplicate"), _case("duplicate")],
    )

    with pytest.raises(ValueError, match="duplicate"):
        comparison.load_cases(cases_path)


@pytest.mark.parametrize(
    ("wav_options", "message"),
    [
        ({"channels": 2}, "mono"),
        ({"sample_width": 1}, "16-bit"),
        ({"frame_rate": 8_000}, "16 kHz"),
    ],
)
def test_load_cases_validates_wav_format(
    tmp_path: Path,
    wav_options: dict[str, int],
    message: str,
) -> None:
    _write_wav(tmp_path / "audio.wav", **wav_options)
    cases_path = _write_cases(tmp_path / "cases.jsonl", [_case("invalid-wav")])

    with pytest.raises(ValueError, match=message):
        comparison.load_cases(cases_path)


def test_load_cases_rejects_truncated_wav_payload(tmp_path: Path) -> None:
    audio = _write_wav(tmp_path / "audio.wav")
    audio.write_bytes(audio.read_bytes()[:-10])
    cases_path = _write_cases(tmp_path / "cases.jsonl", [_case("truncated")])

    with pytest.raises(ValueError, match="truncated"):
        comparison.load_cases(cases_path)


def test_empty_reference_is_allowed_only_for_silence(tmp_path: Path) -> None:
    _write_wav(tmp_path / "audio.wav")
    invalid_path = _write_cases(
        tmp_path / "invalid.jsonl",
        [_case("empty-clean", reference="")],
    )
    silence_path = _write_cases(
        tmp_path / "silence.jsonl",
        [_case("silence", reference="", tag="silence")],
    )

    with pytest.raises(ValueError, match="reference"):
        comparison.load_cases(invalid_path)
    assert comparison.load_cases(silence_path)[0]["reference"] == ""


def test_non_silence_reference_must_have_normalized_metric_units(
    tmp_path: Path,
) -> None:
    _write_wav(tmp_path / "audio.wav")
    cases_path = _write_cases(
        tmp_path / "cases.jsonl",
        [_case("punctuation-only", reference="—!")],
    )

    with pytest.raises(ValueError, match="reference"):
        comparison.load_cases(cases_path)


@pytest.mark.parametrize("model_flag", MODEL_FLAGS)
@pytest.mark.parametrize("invalid_kind", ("missing", "file", "repository_id"))
def test_main_rejects_each_invalid_model_input_before_touching_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_flag: str,
    invalid_kind: str,
) -> None:
    _write_wav(tmp_path / "audio.wav")
    cases_path = _write_cases(tmp_path / "cases.jsonl", [_case("case")])
    directories = _model_directories(tmp_path)
    if invalid_kind == "missing":
        directories[model_flag] = tmp_path / "missing"
    elif invalid_kind == "file":
        not_a_directory = tmp_path / f"{model_flag}.bin"
        not_a_directory.write_bytes(b"model")
        directories[model_flag] = not_a_directory
    else:
        monkeypatch.chdir(tmp_path)
        directories[model_flag] = Path("owner/repository")
    output = tmp_path / "report.json"
    output.write_text("sentinel", encoding="utf-8")

    status = comparison.main(_main_args(cases_path, directories, output))

    assert status == 2
    assert output.read_text(encoding="utf-8") == "sentinel"


@pytest.mark.parametrize(
    ("model_flag", "missing_file"),
    [
        ("v2_int8", "config.json"),
        ("v2_int8", "vocab.txt"),
        ("v2_int8", "encoder-model.int8.onnx"),
        ("v2_int8", "decoder_joint-model.int8.onnx"),
        ("v2_f32", "config.json"),
        ("v2_f32", "vocab.txt"),
        ("v2_f32", "encoder-model.onnx"),
        ("v2_f32", "encoder-model.onnx.data"),
        ("v2_f32", "decoder_joint-model.onnx"),
        ("v3_int8", "config.json"),
        ("v3_int8", "vocab.txt"),
        ("v3_int8", "encoder-model.int8.onnx"),
        ("v3_int8", "decoder_joint-model.int8.onnx"),
        ("v3_f32", "config.json"),
        ("v3_f32", "vocab.txt"),
        ("v3_f32", "encoder-model.onnx"),
        ("v3_f32", "encoder-model.onnx.data"),
        ("v3_f32", "decoder_joint-model.onnx"),
        ("faster_whisper", "config.json"),
        ("faster_whisper", "model.bin"),
        ("faster_whisper", "tokenizer.json"),
        ("faster_whisper", "vocabulary.txt"),
    ],
)
def test_main_rejects_incomplete_model_before_touching_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_flag: str,
    missing_file: str,
) -> None:
    _write_wav(tmp_path / "audio.wav")
    cases_path = _write_cases(tmp_path / "cases.jsonl", [_case("case")])
    directories = _model_directories(tmp_path)
    (directories[model_flag] / missing_file).unlink()
    monkeypatch.setattr(
        comparison,
        "_build_model_runners",
        lambda _directories: {
            model_id: lambda: lambda case: str(case["reference"])
            for model_id in MODEL_IDS
        },
    )
    output = tmp_path / "report.json"
    output.write_text("sentinel", encoding="utf-8")

    status = comparison.main(_main_args(cases_path, directories, output))

    assert status == 2
    assert output.read_text(encoding="utf-8") == "sentinel"


@pytest.mark.parametrize(
    "collision",
    (
        "cases",
        "audio",
        *(f"{model_flag}_equal" for model_flag in MODEL_FLAGS),
        *(f"{model_flag}_inside" for model_flag in MODEL_FLAGS),
    ),
)
def test_main_rejects_output_input_collisions_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    collision: str,
) -> None:
    audio = _write_wav(tmp_path / "audio.wav")
    cases_path = _write_cases(tmp_path / "cases.jsonl", [_case("case")])
    directories = _model_directories(tmp_path)
    if collision == "cases":
        output = cases_path
        sentinel = cases_path.read_bytes()
        sentinel_path = cases_path
    elif collision == "audio":
        output = audio
        sentinel = audio.read_bytes()
        sentinel_path = audio
    else:
        model_flag, position = collision.rsplit("_", 1)
        model_directory = directories[model_flag]
        sentinel_path = model_directory / "config.json"
        sentinel = sentinel_path.read_bytes()
        if position == "equal":
            output = model_directory
        else:
            output = model_directory / "report.json"
            output.write_text("old report", encoding="utf-8")
            sentinel_path = output
            sentinel = output.read_bytes()

    monkeypatch.setattr(
        comparison,
        "_build_model_runners",
        lambda _directories: pytest.fail("model execution started"),
    )

    status = comparison.main(_main_args(cases_path, directories, output))

    assert status == 2
    assert sentinel_path.read_bytes() == sentinel


@pytest.mark.parametrize("destination_kind", ("directory", "file_parent"))
def test_main_rejects_invalid_output_destination_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    destination_kind: str,
) -> None:
    _write_wav(tmp_path / "audio.wav")
    cases_path = _write_cases(tmp_path / "cases.jsonl", [_case("case")])
    directories = _model_directories(tmp_path)
    if destination_kind == "directory":
        output = tmp_path / "existing-directory"
        output.mkdir()
        sentinel_path = output / "sentinel.txt"
    else:
        malformed_parent = tmp_path / "regular-file-parent"
        output = malformed_parent / "report.json"
        sentinel_path = malformed_parent
    sentinel = b"do not replace"
    sentinel_path.write_bytes(sentinel)

    monkeypatch.setattr(
        comparison,
        "_build_model_runners",
        lambda _directories: pytest.fail("model execution started"),
    )

    status = comparison.main(_main_args(cases_path, directories, output))

    assert status == 2
    assert sentinel_path.read_bytes() == sentinel


def test_scheduled_models_match_language_matrix() -> None:
    assert comparison.scheduled_models({"language": "en", "tag": "clean"}) == (
        "parakeet_v2_int8",
        "parakeet_v2_f32",
        "faster_whisper",
    )
    for language in SUPPORTED_V3_LANGUAGES:
        assert comparison.scheduled_models({"language": language, "tag": "clean"}) == (
            "parakeet_v3_int8",
            "parakeet_v3_f32",
            "faster_whisper",
        )
    assert comparison.scheduled_models({"language": "ja", "tag": "clean"}) == (
        "faster_whisper",
    )
    assert (
        comparison.scheduled_models({"language": "en", "tag": "silence"}) == MODEL_IDS
    )


def test_local_model_loaders_use_exact_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directories = _model_directories(tmp_path)
    onnx_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    whisper_init_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    whisper_transcribe_calls: list[tuple[object, dict[str, object]]] = []

    class FakeParakeet:
        def recognize(self, audio: object) -> str:
            return f"recognized {audio}"

    def fake_load_model(*args: object, **kwargs: object) -> FakeParakeet:
        onnx_calls.append((args, kwargs))
        return FakeParakeet()

    class FakeWhisper:
        def __init__(self, *args: object, **kwargs: object) -> None:
            whisper_init_calls.append((args, kwargs))

        def transcribe(
            self,
            audio: object,
            **kwargs: object,
        ) -> tuple[object, object]:
            whisper_transcribe_calls.append((audio, kwargs))
            segments = (SimpleNamespace(text=text) for text in (" one ", "two"))
            return segments, object()

    onnx_module = ModuleType("onnx_asr")
    onnx_module.load_model = fake_load_model  # type: ignore[attr-defined]
    whisper_module = ModuleType("faster_whisper")
    whisper_module.WhisperModel = FakeWhisper  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnx_asr", onnx_module)
    monkeypatch.setitem(sys.modules, "faster_whisper", whisper_module)

    loaders = comparison._build_model_runners(directories)
    for model_id in MODEL_IDS[:4]:
        loaders[model_id]()
    whisper_runner = loaders["faster_whisper"]()
    assert (
        whisper_runner(
            {"audio": Path("/audio/fr.wav"), "language": "fr", "tag": "clean"}
        )
        == "one two"
    )
    assert (
        whisper_runner(
            {"audio": Path("/audio/silence.wav"), "language": "en", "tag": "silence"}
        )
        == "one two"
    )

    assert onnx_calls == [
        (
            ("nemo-parakeet-tdt-0.6b-v2",),
            {
                "path": directories["v2_int8"],
                "quantization": "int8",
                "providers": ["CPUExecutionProvider"],
            },
        ),
        (
            ("nemo-parakeet-tdt-0.6b-v2",),
            {
                "path": directories["v2_f32"],
                "quantization": None,
                "providers": ["CPUExecutionProvider"],
            },
        ),
        (
            ("nemo-parakeet-tdt-0.6b-v3",),
            {
                "path": directories["v3_int8"],
                "quantization": "int8",
                "providers": ["CPUExecutionProvider"],
            },
        ),
        (
            ("nemo-parakeet-tdt-0.6b-v3",),
            {
                "path": directories["v3_f32"],
                "quantization": None,
                "providers": ["CPUExecutionProvider"],
            },
        ),
    ]
    assert whisper_init_calls == [
        (
            (str(directories["faster_whisper"]),),
            {
                "device": "cpu",
                "compute_type": "int8",
                "local_files_only": True,
            },
        )
    ]
    assert whisper_transcribe_calls == [
        (str(Path("/audio/fr.wav")), {"language": "fr"}),
        (str(Path("/audio/silence.wav")), {}),
    ]


def test_run_comparison_uses_micro_wer_and_cer() -> None:
    cases = [
        _loaded_case("one", reference="a b", language="ja"),
        _loaded_case("two", reference="c", language="ja"),
    ]
    hypotheses = {"one": "a x", "two": ""}
    loaders = {"faster_whisper": lambda: lambda case: hypotheses[str(case["id"])]}

    rows, summary, has_errors = comparison.run_comparison(cases, loaders)

    aggregate = summary["models"]["faster_whisper"]
    assert len(rows) == 2
    assert aggregate["word_edits"] == 2
    assert aggregate["word_reference_units"] == 3
    assert aggregate["wer"] == pytest.approx(2 / 3)
    assert aggregate["character_edits"] == 2
    assert aggregate["character_reference_units"] == 3
    assert aggregate["cer"] == pytest.approx(2 / 3)
    assert not has_errors


def test_model_load_failure_creates_every_scheduled_error_row() -> None:
    cases = [
        _loaded_case("one", reference="one", language="en"),
        _loaded_case("two", reference="two", language="en"),
    ]

    def fail_load() -> object:
        raise RuntimeError("cannot load")

    loaders = {
        "parakeet_v2_int8": fail_load,
        "parakeet_v2_f32": lambda: lambda case: str(case["reference"]),
        "faster_whisper": lambda: lambda case: str(case["reference"]),
    }

    rows, _summary, has_errors = comparison.run_comparison(cases, loaders)

    failed_rows = [row for row in rows if row["model_id"] == "parakeet_v2_int8"]
    assert [row["case_id"] for row in failed_rows] == ["one", "two"]
    assert all("model load failed" in str(row["error"]) for row in failed_rows)
    assert all(row["elapsed_seconds"] is None for row in failed_rows)
    assert all(row["rtf"] is None for row in failed_rows)
    assert has_errors
    assert len(rows) == 6


def test_transcription_error_does_not_stop_or_hide_attempt_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = [
        _loaded_case("bad", reference="bad", language="ja"),
        _loaded_case("good", reference="good", language="ja"),
    ]
    attempted: list[str] = []

    def transcribe(case: dict[str, object]) -> str:
        case_id = str(case["id"])
        attempted.append(case_id)
        if case_id == "bad":
            raise RuntimeError("decode failed")
        return "good"

    clock = iter((10.0, 12.0, 20.0, 24.0))
    monkeypatch.setattr(comparison.time, "perf_counter", lambda: next(clock))
    rows, summary, has_errors = comparison.run_comparison(
        cases,
        {"faster_whisper": lambda: transcribe},
    )

    assert attempted == ["bad", "good"]
    assert rows[0]["error"] == "RuntimeError: decode failed"
    assert rows[0]["elapsed_seconds"] == pytest.approx(2.0)
    assert rows[0]["rtf"] == pytest.approx(1.0)
    assert rows[1]["hypothesis"] == "good"
    assert rows[1]["error"] is None
    aggregate = summary["models"]["faster_whisper"]
    assert aggregate["elapsed_seconds"] == pytest.approx(6.0)
    assert aggregate["audio_duration_seconds"] == pytest.approx(4.0)
    assert aggregate["rtf"] == pytest.approx(1.5)
    assert has_errors


def test_main_writes_identity_timings_hashes_and_separate_silence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speech = _write_wav(tmp_path / "speech.wav", seconds=2)
    silence = _write_wav(tmp_path / "silence.wav")
    cases_path = _write_cases(
        tmp_path / "cases.jsonl",
        [
            _case(
                "speech",
                audio=speech.name,
                reference="bonjour",
                language="ja",
            ),
            _case(
                "silence",
                audio=silence.name,
                reference="",
                language="en",
                tag="silence",
            ),
        ],
    )
    directories = _model_directories(tmp_path)

    def fake_loaders(
        _directories: dict[str, Path],
    ) -> dict[str, object]:
        return {
            model_id: (
                lambda: (
                    lambda case: (
                        "" if case["tag"] == "silence" else str(case["reference"])
                    )
                )
            )
            for model_id in MODEL_IDS
        }

    monkeypatch.setattr(comparison, "_build_model_runners", fake_loaders)
    output = tmp_path / "report.json"

    status = comparison.main(_main_args(cases_path, directories, output))
    report = json.loads(output.read_text(encoding="utf-8"))

    assert status == 0
    assert report["report_label"] == "indicative_macos_comparison"
    assert report["environment"]["python"]["version"]
    assert "onnx-asr" in report["environment"]["packages"]
    assert report["cases"][0]["audio"] == "speech.wav"
    assert (
        report["cases"][0]["audio_sha256"]
        == hashlib.sha256(speech.read_bytes()).hexdigest()
    )
    v2_int8_files = report["models"]["parakeet_v2_int8"]["files"]
    assert {
        "name": "encoder-model.int8.onnx",
        "size_bytes": len(b"encoder-model.int8.onnx"),
    } in v2_int8_files
    assert "unused.bin" not in {file["name"] for file in v2_int8_files}
    assert {
        "name": "preprocessor_config.json",
        "size_bytes": len(b"preprocessor_config.json"),
    } in report["models"]["faster_whisper"]["files"]
    assert {
        "name": "vocabulary.txt",
        "size_bytes": len(b"vocabulary.txt"),
    } in report["models"]["faster_whisper"]["files"]
    speech_row = next(row for row in report["rows"] if row["case_id"] == "speech")
    assert speech_row["elapsed_seconds"] >= 0
    assert speech_row["audio_duration_seconds"] == pytest.approx(2.0)
    assert speech_row["rtf"] == pytest.approx(speech_row["elapsed_seconds"] / 2.0)
    aggregate = report["aggregates"]["faster_whisper"]
    assert aggregate["rtf"] == pytest.approx(
        aggregate["elapsed_seconds"] / aggregate["audio_duration_seconds"]
    )
    assert {row["model_id"] for row in report["silence"]} == set(MODEL_IDS)
    assert all(row["word_edits"] is None for row in report["silence"])


def test_write_report_replaces_existing_file_atomically(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    output.write_text('{"old": true}\n', encoding="utf-8")

    comparison.write_report(output, {"new": True})

    assert json.loads(output.read_text(encoding="utf-8")) == {"new": True}
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_write_report_failure_preserves_old_report_and_removes_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "report.json"
    old_report = b'{"old": true}\n'
    output.write_bytes(old_report)

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(comparison.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        comparison.write_report(output, {"new": True})

    assert output.read_bytes() == old_report
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_run_comparison_releases_each_model_before_loading_next() -> None:
    cases = [_loaded_case("english", reference="hello", language="en")]
    events: list[str] = []

    class FakeModel:
        pass

    def loader(name: str, previous: str | None) -> object:
        def load() -> object:
            if previous is not None:
                assert f"released:{previous}" in events
            model = FakeModel()
            weakref.finalize(model, events.append, f"released:{name}")

            def transcribe(_case: dict[str, object]) -> str:
                assert model is not None
                return "hello"

            return transcribe

        return load

    comparison.run_comparison(
        cases,
        {
            "parakeet_v2_int8": loader("v2_int8", None),
            "parakeet_v2_f32": loader("v2_f32", "v2_int8"),
            "faster_whisper": loader("faster_whisper", "v2_f32"),
        },
    )

    assert events == [
        "released:v2_int8",
        "released:v2_f32",
        "released:faster_whisper",
    ]


def test_main_returns_one_when_published_report_has_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_wav(tmp_path / "audio.wav")
    cases_path = _write_cases(
        tmp_path / "cases.jsonl",
        [_case("failure", language="ja")],
    )
    directories = _model_directories(tmp_path)

    def fail_transcription(_case: dict[str, object]) -> str:
        raise RuntimeError("failed")

    monkeypatch.setattr(
        comparison,
        "_build_model_runners",
        lambda _directories: {
            model_id: lambda: fail_transcription for model_id in MODEL_IDS
        },
    )
    output = tmp_path / "report.json"

    status = comparison.main(_main_args(cases_path, directories, output))
    report = json.loads(output.read_text(encoding="utf-8"))

    assert status == 1
    assert report["rows"][0]["error"] == "RuntimeError: failed"
