from __future__ import annotations

import copy
import json
import os
import stat
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.stt_eval.io import (
    atomic_write_json,
    atomic_write_jsonl,
    resolve_contained_path,
    verify_file,
)
from scripts.stt_eval.schema import (
    APPROVED_V3_LANGUAGES,
    ArtifactFile,
    EffectiveExecutionSettings,
    ExperimentManifest,
    MeasurementProfile,
    RunIdentityInputs,
    canonical_json,
    experiment_fingerprint,
    run_fingerprint,
)


FIXTURE = Path(__file__).parent / "fixtures" / "minimal-experiment.json"


def minimal_experiment() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_minimal_experiment_fixture_is_valid_and_frozen() -> None:
    manifest = ExperimentManifest.model_validate(minimal_experiment())

    assert manifest.schema_version == 1
    with pytest.raises(ValidationError):
        manifest.harness_revision = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ((), 2),
        (("corpus",), 2),
        (("models",), 2),
    ],
)
def test_manifest_rejects_unknown_schema_versions(
    path: tuple[str, ...], value: int
) -> None:
    raw = minimal_experiment()
    target = raw
    for component in path:
        target = target[component]  # type: ignore[index,assignment]
    target["schema_version"] = value  # type: ignore[index]

    with pytest.raises(ValidationError):
        ExperimentManifest.model_validate(raw)


@pytest.mark.parametrize(
    "path",
    [
        (),
        ("corpus", "samples", 0),
        ("models", "models", 0),
        ("matrix", 0),
        ("runtime",),
    ],
)
def test_manifest_rejects_unknown_fields_at_every_level(
    path: tuple[str | int, ...],
) -> None:
    raw = minimal_experiment()
    target = raw
    for component in path:
        target = target[component]  # type: ignore[index,assignment]
    target["surprise"] = True  # type: ignore[index]

    with pytest.raises(ValidationError):
        ExperimentManifest.model_validate(raw)


@pytest.mark.parametrize(
    "filename",
    [
        "",
        ".",
        "..",
        "/absolute.onnx",
        "nested/model.onnx",
        r"nested\model.onnx",
        "../model.onnx",
        r"..\model.onnx",
        "C:\\model.onnx",
        "model\x00.onnx",
        "model\n.onnx",
        " model.onnx",
        "model.onnx ",
    ],
)
def test_artifact_file_rejects_unsafe_filenames(filename: str) -> None:
    with pytest.raises(ValidationError):
        ArtifactFile(filename=filename, size_bytes=1, sha256="a" * 64)


@pytest.mark.parametrize(
    ("size_bytes", "sha256"),
    [
        (0, "a" * 64),
        (-1, "a" * 64),
        (2**63, "a" * 64),
        (1, "a" * 63),
        (1, "A" * 64),
        (1, "g" * 64),
    ],
)
def test_artifact_file_rejects_bad_size_or_sha256(size_bytes: int, sha256: str) -> None:
    with pytest.raises(ValidationError):
        ArtifactFile(filename="model.onnx", size_bytes=size_bytes, sha256=sha256)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("model", "repository"),
        ("model", "revision"),
        ("model", "license"),
        ("vad", "repository"),
        ("vad", "revision"),
        ("vad", "license"),
        ("vad", "precision"),
    ],
)
def test_model_and_vad_provenance_is_required(section: str, field: str) -> None:
    raw = minimal_experiment()
    item = (
        raw["models"]["models"][0]  # type: ignore[index]
        if section == "model"
        else raw["models"]["vad_variants"][0]  # type: ignore[index]
    )
    del item[field]

    with pytest.raises(ValidationError):
        ExperimentManifest.model_validate(raw)


@pytest.mark.parametrize("section", ["model", "vad"])
def test_model_and_vad_file_digests_are_required(section: str) -> None:
    raw = minimal_experiment()
    item = (
        raw["models"]["models"][0]  # type: ignore[index]
        if section == "model"
        else raw["models"]["vad_variants"][0]  # type: ignore[index]
    )
    del item["files"][0]["sha256"]

    with pytest.raises(ValidationError):
        ExperimentManifest.model_validate(raw)


def test_referenced_vad_variant_must_exist() -> None:
    raw = minimal_experiment()
    raw["models"]["vad_variants"] = []  # type: ignore[index]

    with pytest.raises(ValueError, match="unknown VAD variant"):
        ExperimentManifest.model_validate(raw)


def test_closed_matrix_rejects_duplicate_cells() -> None:
    raw = minimal_experiment()
    raw["matrix"].append(copy.deepcopy(raw["matrix"][0]))  # type: ignore[union-attr,index]

    with pytest.raises(ValueError, match="duplicate comparison matrix cell"):
        ExperimentManifest.model_validate(raw)


def test_closed_matrix_rejects_incomplete_cells() -> None:
    raw = minimal_experiment()
    raw["matrix"] = [  # type: ignore[index]
        cell
        for cell in raw["matrix"]  # type: ignore[union-attr]
        if cell["measurement_profile"] != "memory_reuse"
    ]

    with pytest.raises(ValueError, match="closed comparison matrix"):
        ExperimentManifest.model_validate(raw)


def test_closed_matrix_requires_every_v3_language_even_if_declaration_is_removed() -> (
    None
):
    raw = minimal_experiment()
    raw["requirements"] = [  # type: ignore[index]
        requirement
        for requirement in raw["requirements"]  # type: ignore[union-attr]
        if requirement["language"] != "uk"
    ]
    raw["matrix"] = [  # type: ignore[index]
        cell
        for cell in raw["matrix"]  # type: ignore[union-attr]
        if cell["language"] != "uk"
    ]

    with pytest.raises(ValueError, match="v3 language/profile coverage"):
        ExperimentManifest.model_validate(raw)


def test_closed_matrix_rejects_undeclared_cells() -> None:
    raw = minimal_experiment()
    extra = copy.deepcopy(raw["matrix"][0])  # type: ignore[index]
    extra["population_id"] = "other"
    raw["matrix"].append(extra)  # type: ignore[union-attr]

    with pytest.raises(ValueError, match="closed comparison matrix|unknown population"):
        ExperimentManifest.model_validate(raw)


def test_matrix_minimums_are_positive() -> None:
    raw = minimal_experiment()
    raw["matrix"][0]["min_audio_duration_seconds"] = 0  # type: ignore[index]

    with pytest.raises(ValidationError):
        ExperimentManifest.model_validate(raw)


def test_measurement_profile_is_closed() -> None:
    raw = minimal_experiment()
    raw["matrix"][0]["measurement_profile"] = "debug"  # type: ignore[index]

    with pytest.raises(ValidationError):
        ExperimentManifest.model_validate(raw)


def test_runtime_requires_every_qualification_package_identity() -> None:
    raw = minimal_experiment()
    raw["runtime"]["packages"] = [  # type: ignore[index]
        package
        for package in raw["runtime"]["packages"]  # type: ignore[index]
        if package["name"] != "faster-whisper"
    ]

    with pytest.raises(ValueError, match="runtime packages"):
        ExperimentManifest.model_validate(raw)


def test_v3_language_set_is_exactly_the_approved_24() -> None:
    raw = minimal_experiment()
    manifest = ExperimentManifest.model_validate(raw)
    approved = {
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

    assert len(APPROVED_V3_LANGUAGES) == 24
    assert APPROVED_V3_LANGUAGES == approved
    assert set(manifest.v3_languages) == approved


@pytest.mark.parametrize("change", ["remove", "add", "duplicate"])
def test_v3_language_set_rejects_any_change(change: str) -> None:
    raw = minimal_experiment()
    languages = raw["v3_languages"]  # type: ignore[assignment]
    if change == "remove":
        languages.pop()
    elif change == "add":
        languages.append("en")
    else:
        languages.append(languages[0])

    with pytest.raises(ValueError, match="v3 language set"):
        ExperimentManifest.model_validate(raw)


def test_canonical_json_is_utf8_sorted_compact_and_deterministic() -> None:
    left = {"z": 1, "é": "café", "a": [2, 3]}
    right = {"a": [2, 3], "é": "café", "z": 1}

    assert canonical_json(left) == canonical_json(right)
    assert canonical_json(left) == '{"a":[2,3],"z":1,"é":"café"}'.encode()


def test_canonical_json_rejects_nan() -> None:
    with pytest.raises(ValueError):
        canonical_json({"bad": float("nan")})


def test_experiment_fingerprint_is_stable_across_variant_run_inputs() -> None:
    manifest = ExperimentManifest.model_validate(minimal_experiment())
    fingerprint = experiment_fingerprint(manifest)
    int8_quality = RunIdentityInputs(
        model_variant_id="parakeet-v2-int8",
        measurement_profile=MeasurementProfile.QUALITY,
        effective_settings=EffectiveExecutionSettings(
            execution_provider="CPUExecutionProvider",
            device="cpu",
            intra_op_threads=4,
            inter_op_threads=1,
            vad_batch_size=1,
        ),
    )
    int8_throughput = RunIdentityInputs(
        model_variant_id="parakeet-v2-int8",
        measurement_profile=MeasurementProfile.THROUGHPUT,
        effective_settings=int8_quality.effective_settings,
    )
    int8_quality_two_threads = RunIdentityInputs(
        model_variant_id="parakeet-v2-int8",
        measurement_profile=MeasurementProfile.QUALITY,
        effective_settings=EffectiveExecutionSettings(
            execution_provider="CPUExecutionProvider",
            device="cpu",
            intra_op_threads=2,
            inter_op_threads=1,
            vad_batch_size=1,
        ),
    )
    baseline_quality = RunIdentityInputs(
        model_variant_id="faster-whisper-base-int8",
        measurement_profile=MeasurementProfile.QUALITY,
        effective_settings=int8_quality.effective_settings,
    )
    baseline_throughput = RunIdentityInputs(
        model_variant_id="faster-whisper-base-int8",
        measurement_profile=MeasurementProfile.THROUGHPUT,
        effective_settings=EffectiveExecutionSettings(
            execution_provider="CPUExecutionProvider",
            device="cpu",
            intra_op_threads=2,
            inter_op_threads=1,
            vad_batch_size=None,
        ),
    )

    assert experiment_fingerprint(manifest) == fingerprint
    assert run_fingerprint(fingerprint, int8_quality) != run_fingerprint(
        fingerprint, int8_throughput
    )
    assert run_fingerprint(fingerprint, int8_quality) != run_fingerprint(
        fingerprint, baseline_quality
    )
    assert run_fingerprint(fingerprint, int8_quality) != run_fingerprint(
        fingerprint, int8_quality_two_threads
    )
    assert run_fingerprint(fingerprint, int8_quality) != run_fingerprint(
        fingerprint, baseline_throughput
    )


def test_fingerprint_contract_has_fixed_golden_values() -> None:
    manifest = ExperimentManifest.model_validate(minimal_experiment())
    experiment = experiment_fingerprint(manifest)
    run = RunIdentityInputs(
        model_variant_id="parakeet-v2-int8",
        measurement_profile="quality",
        effective_settings=EffectiveExecutionSettings(
            execution_provider="CPUExecutionProvider",
            device="cpu",
            intra_op_threads=4,
            inter_op_threads=1,
            vad_batch_size=1,
        ),
    )

    assert experiment == (
        "e7a99647bff0024b8191ba7ca99476ed4535b552fe7e99b16f0c2f4996ae217a"
    )
    assert run_fingerprint(experiment, run) == (
        "c33e0ae286923963581bafa13722698bb0fb3d0e44969233e4ce0608c9814749"
    )


def test_non_variant_content_changes_only_experiment_identity_source() -> None:
    original = ExperimentManifest.model_validate(minimal_experiment())
    changed_raw = minimal_experiment()
    changed_raw["harness_revision"] = "new-harness-revision"
    changed = ExperimentManifest.model_validate(changed_raw)
    run = RunIdentityInputs(
        model_variant_id="parakeet-v2-int8",
        measurement_profile="quality",
        effective_settings=EffectiveExecutionSettings(
            execution_provider="CPUExecutionProvider",
            device="cpu",
            intra_op_threads=4,
            inter_op_threads=1,
            vad_batch_size=1,
        ),
    )

    original_experiment = experiment_fingerprint(original)
    changed_experiment = experiment_fingerprint(changed)
    assert original_experiment != changed_experiment
    assert run_fingerprint(original_experiment, run) != run_fingerprint(
        changed_experiment, run
    )


def test_verify_file_streams_and_matches_size_and_sha256(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"test")

    verify_file(
        path,
        expected_size=4,
        expected_sha256=(
            "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
        ),
        chunk_size=2,
    )


@pytest.mark.parametrize(
    ("expected_size", "expected_sha256", "message"),
    [
        (5, "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08", "size"),
        (4, "a" * 64, "SHA-256"),
    ],
)
def test_verify_file_rejects_mismatches(
    tmp_path: Path, expected_size: int, expected_sha256: str, message: str
) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"test")

    with pytest.raises(ValueError, match=message):
        verify_file(
            path,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )


def test_verify_file_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"test")
    link = tmp_path / "artifact.bin"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        verify_file(link, expected_size=4, expected_sha256="a" * 64)


def test_verify_file_rejects_intermediate_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "artifact.bin").write_bytes(b"test")
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        verify_file(
            linked / "artifact.bin",
            expected_size=4,
            expected_sha256=(
                "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
            ),
        )


def test_verify_file_rejects_non_regular_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="regular file"):
        verify_file(tmp_path, expected_size=4, expected_sha256="a" * 64)


def test_resolve_contained_path_rejects_escape_and_symlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="contained"):
        resolve_contained_path(root, "../outside/file")
    with pytest.raises(ValueError, match="symlink"):
        resolve_contained_path(root, "linked/file")


def test_resolve_contained_path_accepts_nonexistent_contained_target(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    (root / "nested").mkdir(parents=True)

    assert resolve_contained_path(root, "nested/file.json") == (
        root / "nested" / "file.json"
    )


def test_atomic_json_writer_publishes_canonical_content(tmp_path: Path) -> None:
    destination = tmp_path / "result.json"

    atomic_write_json(destination, {"z": 1, "a": "é"})

    assert destination.read_bytes() == '{"a":"é","z":1}'.encode()
    assert list(tmp_path.glob(".result.json.*.tmp")) == []


def test_atomic_jsonl_writer_publishes_complete_records(tmp_path: Path) -> None:
    destination = tmp_path / "raw.jsonl"

    atomic_write_jsonl(destination, [{"z": 1}, {"a": "é"}])

    assert destination.read_bytes() == b'{"z":1}\n{"a":"\xc3\xa9"}\n'
    assert list(tmp_path.glob(".raw.jsonl.*.tmp")) == []


def test_atomic_writer_fsyncs_file_then_replaces_then_fsyncs_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "result.json"
    events: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def record_fsync(descriptor: int) -> None:
        mode = os.fstat(descriptor).st_mode
        events.append("directory_fsync" if stat.S_ISDIR(mode) else "file_fsync")
        real_fsync(descriptor)

    def record_replace(
        source: str,
        target: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        events.append("replace")
        real_replace(
            source,
            target,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(os, "fsync", record_fsync)
    monkeypatch.setattr(os, "replace", record_replace)

    atomic_write_json(destination, {"complete": True})

    assert events == ["file_fsync", "replace", "directory_fsync"]


def test_atomic_writer_rejects_symlinked_parent(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        atomic_write_json(linked / "result.json", {"complete": True})

    assert not (real / "result.json").exists()
    assert list(real.glob(".result.json.*.tmp")) == []


@pytest.mark.parametrize("writer", [atomic_write_json, atomic_write_jsonl])
def test_atomic_writers_clean_staging_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writer: object
) -> None:
    destination = tmp_path / "result.json"
    destination.write_bytes(b"previous-complete-result")

    def fail_replace(
        source: str,
        target: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        if writer is atomic_write_json:
            atomic_write_json(destination, {"complete": True})
        else:
            atomic_write_jsonl(destination, [{"complete": True}])

    assert destination.read_bytes() == b"previous-complete-result"
    assert list(tmp_path.glob(".result.json.*.tmp")) == []


def test_atomic_jsonl_writer_cleans_staging_when_iteration_fails(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "raw.jsonl"

    def broken_records() -> object:
        yield {"complete": True}
        raise RuntimeError("record generation failed")

    with pytest.raises(RuntimeError, match="record generation failed"):
        atomic_write_jsonl(destination, broken_records())

    assert not destination.exists()
    assert list(tmp_path.glob(".raw.jsonl.*.tmp")) == []
