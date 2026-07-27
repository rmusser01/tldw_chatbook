from __future__ import annotations

import hashlib
import io
import math
import random
import subprocess
import tarfile
import wave
from pathlib import Path
from typing import Callable, Iterable

import httpx
import pytest
from pydantic import ValidationError

from scripts.stt_eval.prepare import (
    PrepareRequest,
    PreparationError,
    PreparationPreflight,
    prepare,
    preflight,
)
from scripts.stt_eval.schema import (
    ArchiveMember,
    ArtifactFile,
    ConcatenationRecipe,
    NoiseRecipe,
    NormalizedSampleRecipe,
    PreparationLimits,
    PreparationManifest,
    PreparationReceipt,
    PreparationSource,
    SilenceRecipe,
)


FIXTURES = Path(__file__).parent / "fixtures"
FFMPEG_FIXTURE = FIXTURES / "fake_ffmpeg.py"
SHA = "a" * 64
EXPERIMENT_ID = "e" * 64
AMPLE_SPACE = 1 << 40


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def artifact(filename: str, data: bytes) -> ArtifactFile:
    return ArtifactFile(filename=filename, size_bytes=len(data), sha256=digest(data))


def wav_bytes(
    samples: Iterable[int],
    *,
    sample_rate: int = 16_000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(sample_width)
        output.setframerate(sample_rate)
        if sample_width == 2:
            frames = b"".join(
                int(sample).to_bytes(2, "little", signed=True) for sample in samples
            )
        else:
            frames = bytes(samples)
        output.writeframes(frames)
    return buffer.getvalue()


def pcm_samples(data: bytes) -> list[int]:
    with wave.open(io.BytesIO(data), "rb") as stream:
        frames = stream.readframes(stream.getnframes())
    return [
        int.from_bytes(frames[index : index + 2], "little", signed=True)
        for index in range(0, len(frames), 2)
    ]


def noise_bytes(
    source: bytes, *, seed: int, amplitude: float, source_gain: float
) -> bytes:
    generator = random.Random(seed)
    peak = round(32767 * amplitude)
    samples = []
    for sample in pcm_samples(source):
        noisy = round(sample * source_gain) + generator.randint(-peak, peak)
        samples.append(max(-32768, min(32767, noisy)))
    return wav_bytes(samples)


def concat_bytes(inputs: list[bytes], gaps: list[float]) -> bytes:
    samples: list[int] = []
    for index, input_data in enumerate(inputs):
        if index:
            samples.extend([0] * round(gaps[index - 1] * 16_000))
        samples.extend(pcm_samples(input_data))
    return wav_bytes(samples)


def make_tar(
    path: Path,
    entries: list[tuple[str, bytes, bytes | None]],
    *,
    gzip: bool = False,
) -> bytes:
    """Create a tar; the third tuple item is an optional TarInfo type."""

    raw = io.BytesIO()
    mode = "w:gz" if gzip else "w"
    with tarfile.open(fileobj=raw, mode=mode) as archive:
        for name, data, member_type in entries:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o600
            if member_type is not None:
                info.type = member_type
                if member_type in {tarfile.SYMTYPE, tarfile.LNKTYPE}:
                    info.linkname = "target.wav"
                    info.size = 0
                    data = b""
                elif member_type != tarfile.REGTYPE:
                    info.size = 0
                    data = b""
            archive.addfile(info, io.BytesIO(data))
    result = raw.getvalue()
    path.write_bytes(result)
    return result


def member(name: str, data: bytes, *, selected: bool = True) -> ArchiveMember:
    return ArchiveMember(
        name=name,
        file_type="regular_file",
        size_bytes=len(data),
        sha256=digest(data),
        selected_for_preparation=selected,
    )


def limits(
    *,
    member_count: int = 20,
    file_bytes: int = 1 << 20,
    uncompressed_bytes: int = 1 << 21,
    headroom_bytes: int = 1,
) -> PreparationLimits:
    return PreparationLimits(
        max_member_count=member_count,
        max_file_bytes=file_bytes,
        max_uncompressed_bytes=uncompressed_bytes,
        staging_headroom_bytes=headroom_bytes,
    )


def local_source(
    archive_data: bytes,
    members: tuple[ArchiveMember, ...],
    *,
    archive_name: str = "common-voice.tar",
) -> PreparationSource:
    return PreparationSource(
        source_id="common-voice-en",
        repository="datacollective.mozillafoundation.org/common-voice",
        revision="cv-23.0",
        source_url="local://mozilla-data-collective",
        license="CC0-1.0",
        acquisition_mode="local_file",
        archive=artifact(archive_name, archive_data),
        members=members,
    )


def download_source(
    archive_data: bytes,
    members: tuple[ArchiveMember, ...],
) -> PreparationSource:
    return PreparationSource(
        source_id="fleurs-en",
        repository="huggingface.co/datasets/google/fleurs",
        revision="1" * 40,
        source_url="https://example.test/fleurs-en.tar",
        license="CC-BY-4.0",
        acquisition_mode="verified_download",
        archive=artifact("fleurs-en.tar", archive_data),
        members=members,
    )


def manifest_for(
    source: PreparationSource,
    normalized: tuple[NormalizedSampleRecipe, ...] = (),
    derived: tuple[SilenceRecipe | NoiseRecipe | ConcatenationRecipe, ...] = (),
    *,
    preparation_limits: PreparationLimits | None = None,
) -> PreparationManifest:
    return PreparationManifest(
        schema_version=1,
        sources=(source,),
        limits=preparation_limits or limits(),
        normalized_samples=normalized,
        derived_recipes=derived,
    )


def executable_ffmpeg(tmp_path: Path) -> Path:
    target = tmp_path / "fake ffmpeg"
    target.write_bytes(FFMPEG_FIXTURE.read_bytes())
    target.chmod(0o700)
    return target


def request_for(
    tmp_path: Path,
    manifest: PreparationManifest,
    archive_path: Path | None,
    **kwargs: object,
) -> PrepareRequest:
    local_inputs = (
        {manifest.sources[0].source_id: archive_path}
        if archive_path is not None
        else {}
    )
    defaults: dict[str, object] = {
        "free_space": lambda _path: AMPLE_SPACE,
    }
    defaults.update(kwargs)
    return PrepareRequest(
        manifest=manifest,
        experiment_fingerprint=EXPERIMENT_ID,
        destination=tmp_path / "prepared corpus",
        local_inputs=local_inputs,
        ffmpeg_executable=executable_ffmpeg(tmp_path),
        **defaults,
    )


def one_source_manifest(
    tmp_path: Path,
    *,
    archive_entries: list[tuple[str, bytes, bytes | None]] | None = None,
    declared_members: tuple[ArchiveMember, ...] | None = None,
    preparation_limits: PreparationLimits | None = None,
    source_factory: Callable[
        [bytes, tuple[ArchiveMember, ...]], PreparationSource
    ] = local_source,
) -> tuple[PreparationManifest, Path, bytes]:
    entries = archive_entries or [("audio/input.wav", wav_bytes([1, -2, 3]), None)]
    archive_path = tmp_path / "source.tar"
    archive_data = make_tar(archive_path, entries)
    if declared_members is None:
        declared_members = tuple(
            member(name, data)
            for name, data, entry_type in entries
            if entry_type is None
        )
    source = source_factory(archive_data, declared_members)
    return (
        manifest_for(source, preparation_limits=preparation_limits),
        archive_path,
        archive_data,
    )


class NetworkBomb:
    def stream(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("network must not be called")


def test_preflight_is_read_only_and_reports_full_resource_plan(tmp_path: Path) -> None:
    local_manifest, archive_path, _ = one_source_manifest(tmp_path)
    remote_archive = make_tar(
        tmp_path / "remote.tar", [("fleurs.wav", wav_bytes([4, 5]), None)]
    )
    remote = download_source(remote_archive, (member("fleurs.wav", wav_bytes([4, 5])),))
    combined = local_manifest.model_copy(
        update={"sources": local_manifest.sources + (remote,)}
    )
    destination = tmp_path / "never-created"
    request = PrepareRequest(
        manifest=combined,
        experiment_fingerprint=EXPERIMENT_ID,
        destination=destination,
        local_inputs={"common-voice-en": archive_path},
        ffmpeg_executable=tmp_path / "missing-ffmpeg",
        http_client=NetworkBomb(),
        free_space=lambda path: 123_456,
    )

    result = prepare(request)

    assert isinstance(result, PreparationPreflight)
    assert result.source_ids == ("common-voice-en", "fleurs-en")
    assert result.transfer_bytes == len(remote_archive)
    assert result.licenses == ("CC0-1.0", "CC-BY-4.0")
    assert result.required_local_inputs == (archive_path,)
    assert result.destination == destination
    assert result.available_bytes == 123_456
    assert result.staging_bytes > result.transfer_bytes
    assert result.required_free_bytes == result.staging_bytes + 1
    assert result.missing_tools == (str(tmp_path / "missing-ffmpeg"),)
    assert result.missing_local_inputs == ()
    assert not destination.exists()
    assert not any(
        path.name.startswith(".never-created") for path in tmp_path.iterdir()
    )


def test_missing_common_voice_and_space_block_before_network_or_extraction(
    tmp_path: Path,
) -> None:
    manifest, _, _ = one_source_manifest(tmp_path)
    request = request_for(tmp_path, manifest, None, http_client=NetworkBomb())
    request = request.with_free_space(0)

    plan = preflight(request)

    assert plan.missing_local_inputs == (Path("common-voice-en"),)
    with pytest.raises(PreparationError, match="missing local input.*free space"):
        prepare(request, execute=True)
    assert not request.destination.exists()


def test_destination_parent_cannot_traverse_a_symlink(tmp_path: Path) -> None:
    manifest, archive_path, _ = one_source_manifest(tmp_path)
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    request = request_for(tmp_path, manifest, archive_path).with_destination(
        linked_parent / "corpus"
    )

    with pytest.raises(PreparationError, match="symlink"):
        prepare(request, execute=True)

    assert not (real_parent / "corpus").exists()


def test_execute_gate_keeps_verified_download_off_network(tmp_path: Path) -> None:
    manifest, _, _ = one_source_manifest(tmp_path, source_factory=download_source)
    request = request_for(tmp_path, manifest, None, http_client=NetworkBomb())

    assert isinstance(prepare(request, execute=False), PreparationPreflight)


def test_local_file_never_calls_network(tmp_path: Path) -> None:
    manifest, archive_path, _ = one_source_manifest(tmp_path)
    request = request_for(tmp_path, manifest, archive_path, http_client=NetworkBomb())

    result = prepare(request, execute=True)

    assert result == request.destination


@pytest.mark.parametrize("failure", ["overflow", "hash"])
def test_download_verification_failure_cleans_staging(
    tmp_path: Path, failure: str
) -> None:
    manifest, _, archive_data = one_source_manifest(
        tmp_path, source_factory=download_source
    )
    payload = archive_data + b"x" if failure == "overflow" else archive_data[:-1] + b"x"
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.headers["user-agent"] == "tldw-stt-eval/1"
        assert "authorization" not in request.headers
        return httpx.Response(200, content=payload)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    request = request_for(tmp_path, manifest, None, http_client=client)

    with pytest.raises(PreparationError, match="size|SHA-256"):
        prepare(request, execute=True)

    assert calls == 1
    assert not request.destination.exists()
    assert not any(
        path.name.startswith(".prepared corpus.") for path in tmp_path.iterdir()
    )
    client.close()


@pytest.mark.parametrize(
    ("name", "match"),
    [
        ("../escape.wav", "unsafe archive member"),
        ("/absolute.wav", "unsafe archive member"),
        (r"nested\\windows.wav", "unsafe archive member"),
        ("nested//ambiguous.wav", "unsafe archive member"),
        ("./ambiguous.wav", "unsafe archive member"),
        ("CON/device.wav", "unsafe archive member"),
        ("nested/trailing./file.wav", "unsafe archive member"),
    ],
)
def test_archive_rejects_unsafe_member_names(
    tmp_path: Path, name: str, match: str
) -> None:
    data = wav_bytes([1])
    manifest, archive_path, _ = one_source_manifest(
        tmp_path,
        archive_entries=[(name, data, None)],
        declared_members=(member("safe.wav", data),),
    )
    request = request_for(tmp_path, manifest, archive_path)

    with pytest.raises(PreparationError, match=match):
        prepare(request, execute=True)

    assert not request.destination.exists()


def test_archive_schema_rejects_cross_platform_ambiguous_name() -> None:
    with pytest.raises(ValidationError, match="unambiguous"):
        member("nested/CON.wav", b"x")


def test_archive_schema_rejects_case_colliding_parent_paths() -> None:
    data = b"x"
    with pytest.raises(ValidationError, match="unambiguous"):
        PreparationSource(
            source_id="source",
            repository="repo",
            revision="revision",
            source_url="local://source",
            license="CC0",
            acquisition_mode="local_file",
            archive=ArtifactFile(filename="source.tar", size_bytes=1, sha256=SHA),
            members=(
                member("Speaker/a.wav", data),
                member("speaker/b.wav", data),
            ),
        )


def test_archive_rejects_duplicate_members(tmp_path: Path) -> None:
    data = wav_bytes([1])
    manifest, archive_path, _ = one_source_manifest(
        tmp_path,
        archive_entries=[("a.wav", data, None), ("a.wav", data, None)],
        declared_members=(member("a.wav", data),),
    )

    with pytest.raises(PreparationError, match="duplicate archive member"):
        prepare(request_for(tmp_path, manifest, archive_path), execute=True)


@pytest.mark.parametrize("case", ["unknown", "missing"])
def test_archive_rejects_unknown_and_missing_members(tmp_path: Path, case: str) -> None:
    data = wav_bytes([1])
    entries = [("a.wav", data, None)]
    declared = (
        (member("b.wav", data),)
        if case == "unknown"
        else (member("a.wav", data), member("b.wav", data))
    )
    manifest, archive_path, _ = one_source_manifest(
        tmp_path, archive_entries=entries, declared_members=declared
    )

    with pytest.raises(PreparationError, match=case):
        prepare(request_for(tmp_path, manifest, archive_path), execute=True)


@pytest.mark.parametrize(
    "member_type",
    [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE],
    ids=["symlink", "hardlink", "fifo", "device"],
)
def test_archive_rejects_non_regular_members(
    tmp_path: Path, member_type: bytes
) -> None:
    data = wav_bytes([1])
    manifest, archive_path, _ = one_source_manifest(
        tmp_path,
        archive_entries=[("a.wav", data, member_type)],
        declared_members=(member("a.wav", data),),
    )

    with pytest.raises(PreparationError, match="regular file"):
        prepare(request_for(tmp_path, manifest, archive_path), execute=True)


def test_archive_rejects_sparse_metadata(tmp_path: Path) -> None:
    data = wav_bytes([1])
    archive_path = tmp_path / "sparse.tar"
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        info = tarfile.TarInfo("a.wav")
        info.size = len(data)
        info.pax_headers = {"GNU.sparse.name": "a.wav"}
        archive.addfile(info, io.BytesIO(data))
    archive_data = raw.getvalue()
    archive_path.write_bytes(archive_data)
    manifest = manifest_for(local_source(archive_data, (member("a.wav", data),)))

    with pytest.raises(PreparationError, match="sparse|regular file"):
        prepare(request_for(tmp_path, manifest, archive_path), execute=True)


def test_archive_rejects_member_count_limit(tmp_path: Path) -> None:
    data = wav_bytes([1])
    manifest, archive_path, _ = one_source_manifest(
        tmp_path,
        archive_entries=[("a.wav", data, None), ("b.wav", data, None)],
        declared_members=(member("a.wav", data),),
        preparation_limits=limits(member_count=1),
    )

    with pytest.raises(PreparationError, match="member count"):
        prepare(request_for(tmp_path, manifest, archive_path), execute=True)


@pytest.mark.parametrize(
    ("file_limit", "total_limit", "match"),
    [(50, 1 << 20, "per-file"), (1 << 20, 50, "uncompressed byte")],
)
def test_archive_rejects_declared_size_limits(
    tmp_path: Path, file_limit: int, total_limit: int, match: str
) -> None:
    data = wav_bytes([1, 2, 3, 4])
    manifest, archive_path, _ = one_source_manifest(
        tmp_path,
        archive_entries=[("a.wav", data, None)],
        preparation_limits=limits(
            file_bytes=file_limit, uncompressed_bytes=total_limit
        ),
    )

    with pytest.raises(PreparationError, match=match):
        prepare(request_for(tmp_path, manifest, archive_path), execute=True)


def test_archive_rechecks_staging_space_during_scan(tmp_path: Path) -> None:
    manifest, archive_path, _ = one_source_manifest(tmp_path)
    checks = 0

    def shrinking_space(_path: Path) -> int:
        nonlocal checks
        checks += 1
        return AMPLE_SPACE if checks == 1 else 0

    request = request_for(tmp_path, manifest, archive_path, free_space=shrinking_space)

    with pytest.raises(PreparationError, match="staging free space"):
        prepare(request, execute=True)


def test_archive_rejects_actual_stream_overflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, archive_path, _ = one_source_manifest(tmp_path)
    original = tarfile.TarFile.extractfile

    def overflowing(
        archive: tarfile.TarFile, item: tarfile.TarInfo | str
    ) -> io.BytesIO | tarfile.ExFileObject | None:
        stream = original(archive, item)
        assert stream is not None
        return io.BytesIO(stream.read() + b"x")

    monkeypatch.setattr(tarfile.TarFile, "extractfile", overflowing)

    with pytest.raises(PreparationError, match="actual.*overflow|size mismatch"):
        prepare(request_for(tmp_path, manifest, archive_path), execute=True)


def test_local_archive_digest_failure_precedes_extraction(tmp_path: Path) -> None:
    manifest, archive_path, _ = one_source_manifest(tmp_path)
    archive_path.write_bytes(archive_path.read_bytes() + b"x")
    request = request_for(tmp_path, manifest, archive_path)

    with pytest.raises(PreparationError, match="size mismatch"):
        prepare(request, execute=True)

    assert not request.destination.exists()


def test_source_id_cannot_escape_private_staging(tmp_path: Path) -> None:
    manifest, archive_path, _ = one_source_manifest(tmp_path)
    unsafe_source = manifest.sources[0].model_copy(
        update={"source_id": "a/../../../../outside"}
    )
    unsafe_manifest = manifest.model_copy(update={"sources": (unsafe_source,)})
    request = request_for(tmp_path, unsafe_manifest, archive_path)

    prepare(request, execute=True)

    assert not (tmp_path / "outside").exists()


def test_download_rejects_injected_client_credentials_before_request(
    tmp_path: Path,
) -> None:
    manifest, _, _ = one_source_manifest(tmp_path, source_factory=download_source)
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    client = httpx.Client(
        headers={"Authorization": "Bearer must-not-leak"},
        transport=httpx.MockTransport(handler),
    )
    request = request_for(tmp_path, manifest, None, http_client=client)

    with pytest.raises(PreparationError, match="credential"):
        prepare(request, execute=True)

    assert calls == 0
    assert not request.destination.exists()
    client.close()


def complete_recipe_manifest(
    tmp_path: Path,
    *,
    source_wav: bytes | None = None,
    wrong_normalized_digest: bool = False,
    wrong_derived_digest: bool = False,
) -> tuple[PreparationManifest, Path, dict[str, bytes]]:
    source_wav = source_wav or wav_bytes([100, -100, 32760, -32760])
    archive_path = tmp_path / "source.tar.gz"
    archive_data = make_tar(
        archive_path, [("clips/source file.wav", source_wav, None)], gzip=True
    )
    normalized = source_wav
    silence = wav_bytes([0] * 16)
    noisy = noise_bytes(normalized, seed=593, amplitude=0.01, source_gain=1.0)
    concatenated = concat_bytes(
        [normalized, silence, noisy],
        [0.001, 0.0],
    )
    normalized_file = artifact("base.wav", normalized)
    if wrong_normalized_digest:
        normalized_file = normalized_file.model_copy(update={"sha256": SHA})
    silence_file = artifact("silence.wav", silence)
    noise_file = artifact("noise.wav", noisy)
    concat_file = artifact("long-form.wav", concatenated)
    if wrong_derived_digest:
        noise_file = noise_file.model_copy(update={"sha256": SHA})
    source = local_source(
        archive_data,
        (member("clips/source file.wav", source_wav),),
        archive_name="common-voice.tar.gz",
    )
    result = manifest_for(
        source,
        normalized=(
            NormalizedSampleRecipe(
                recipe_type="normalize",
                recipe_revision="pcm16-16khz-mono-v1",
                sample_id="base",
                source_id=source.source_id,
                source_member="clips/source file.wav",
                prepared_file=normalized_file,
            ),
        ),
        derived=(
            SilenceRecipe(
                recipe_type="silence",
                recipe_revision="synthetic-v1",
                sample_id="silence",
                duration_seconds=0.001,
                prepared_file=silence_file,
            ),
            NoiseRecipe(
                recipe_type="noise",
                recipe_revision="synthetic-v1",
                sample_id="noise",
                source_sample_id="base",
                seed=593,
                noise_amplitude=0.01,
                source_gain=1.0,
                prepared_file=noise_file,
            ),
            ConcatenationRecipe(
                recipe_type="concatenation",
                recipe_revision="concatenation-v1",
                sample_id="long-form",
                source_sample_ids=("base", "silence", "noise"),
                silence_gaps_seconds=(0.001, 0.0),
                prepared_file=concat_file,
            ),
        ),
    )
    return (
        result,
        archive_path,
        {
            "base.wav": normalized,
            "silence.wav": silence,
            "noise.wav": noisy,
            "long-form.wav": concatenated,
        },
    )


def test_ffmpeg_vector_version_and_all_deterministic_recipes(
    tmp_path: Path,
) -> None:
    manifest, archive_path, expected = complete_recipe_manifest(tmp_path)
    calls: list[list[str]] = []

    def recording_runner(
        arguments: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        return subprocess.run(arguments, **kwargs)

    request = request_for(
        tmp_path, manifest, archive_path, command_runner=recording_runner
    )

    destination = prepare(request, execute=True)

    assert destination == request.destination
    assert calls[0] == [str(request.ffmpeg_executable), "-version"]
    conversion = calls[1]
    assert conversion[0] == str(request.ffmpeg_executable)
    assert conversion[1:6] == [
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
    ]
    assert conversion[6].endswith("clips/source file.wav")
    assert conversion[7:-1] == [
        "-map_metadata",
        "-1",
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-n",
    ]
    assert conversion[-1].endswith("base.wav")
    for filename, data in expected.items():
        assert (destination / "audio" / filename).read_bytes() == data
    receipt = PreparationReceipt.model_validate_json(
        (destination / "receipt.json").read_bytes()
    )
    assert receipt.status == "complete"
    assert receipt.ffmpeg_version == "ffmpeg version fake-593 exact test build"
    assert tuple(file.filename for file in receipt.prepared_files) == tuple(expected)


def test_noise_and_concatenation_are_byte_identical_across_runs(
    tmp_path: Path,
) -> None:
    first_manifest, archive_path, _ = complete_recipe_manifest(tmp_path)
    first = request_for(tmp_path, first_manifest, archive_path)
    first_destination = prepare(first, execute=True)
    second = first.with_destination(tmp_path / "prepared-again")

    second_destination = prepare(second, execute=True)

    for filename in ("noise.wav", "long-form.wav"):
        assert (first_destination / "audio" / filename).read_bytes() == (
            second_destination / "audio" / filename
        ).read_bytes()


@pytest.mark.parametrize(
    "invalid_value",
    [math.nan, math.inf, -math.inf],
)
def test_recipe_parameters_reject_non_finite_values(invalid_value: float) -> None:
    with pytest.raises(ValidationError):
        SilenceRecipe(
            recipe_type="silence",
            recipe_revision="v1",
            sample_id="silence",
            duration_seconds=invalid_value,
            prepared_file=ArtifactFile(filename="x.wav", size_bytes=1, sha256=SHA),
        )


def test_manifest_rejects_unknown_recipe_input() -> None:
    source = PreparationSource(
        source_id="source",
        repository="repo",
        revision="rev",
        source_url="local://source",
        license="CC0",
        acquisition_mode="local_file",
        archive=ArtifactFile(filename="source.tar", size_bytes=1, sha256=SHA),
        members=(
            ArchiveMember(
                name="a.wav",
                file_type="regular_file",
                size_bytes=1,
                sha256=SHA,
                selected_for_preparation=True,
            ),
        ),
    )
    with pytest.raises(ValidationError, match="unknown input"):
        manifest_for(
            source,
            derived=(
                NoiseRecipe(
                    recipe_type="noise",
                    recipe_revision="v1",
                    sample_id="noise",
                    source_sample_id="missing",
                    seed=1,
                    noise_amplitude=0.1,
                    source_gain=1.0,
                    prepared_file=ArtifactFile(
                        filename="noise.wav", size_bytes=1, sha256=SHA
                    ),
                ),
            ),
        )


def test_normalized_output_rejects_audio_format_mismatch(tmp_path: Path) -> None:
    invalid_wav = wav_bytes([0, 1], sample_width=1)
    manifest, archive_path, _ = complete_recipe_manifest(
        tmp_path, source_wav=invalid_wav
    )
    request = request_for(tmp_path, manifest, archive_path)

    with pytest.raises(PreparationError, match="PCM16 mono 16000 Hz"):
        prepare(request, execute=True)

    assert not request.destination.exists()


@pytest.mark.parametrize(
    ("kind", "match"),
    [
        ("normalized", "SHA-256 mismatch"),
        ("derived", "SHA-256 mismatch"),
        ("ffmpeg", "ffmpeg conversion failed"),
    ],
)
def test_normalization_and_derivation_failures_clean_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    match: str,
) -> None:
    manifest, archive_path, _ = complete_recipe_manifest(
        tmp_path,
        wrong_normalized_digest=kind == "normalized",
        wrong_derived_digest=kind == "derived",
    )
    if kind == "ffmpeg":
        monkeypatch.setenv("FAKE_FFMPEG_FAIL", "1")
    request = request_for(tmp_path, manifest, archive_path)

    with pytest.raises(PreparationError, match=match):
        prepare(request, execute=True)

    assert not request.destination.exists()
    assert not any(
        path.name.startswith(".prepared corpus.") for path in tmp_path.iterdir()
    )


def test_prepared_output_bound_is_enforced(tmp_path: Path) -> None:
    manifest, archive_path, _ = complete_recipe_manifest(tmp_path)
    constrained = manifest.model_copy(
        update={"limits": limits(file_bytes=60, uncompressed_bytes=1 << 21)}
    )

    with pytest.raises(PreparationError, match="prepared output.*per-file"):
        prepare(request_for(tmp_path, constrained, archive_path), execute=True)


def test_publish_failure_cleans_staging(tmp_path: Path) -> None:
    manifest, archive_path, _ = complete_recipe_manifest(tmp_path)

    def fail_publish(_source: Path, _destination: Path) -> None:
        raise OSError("synthetic publish failure")

    request = request_for(tmp_path, manifest, archive_path, publisher=fail_publish)

    with pytest.raises(PreparationError, match="synthetic publish failure"):
        prepare(request, execute=True)

    assert not request.destination.exists()
    assert not any(
        path.name.startswith(".prepared corpus.") for path in tmp_path.iterdir()
    )


def test_existing_destination_is_reused_only_when_receipt_and_files_verify(
    tmp_path: Path,
) -> None:
    manifest, archive_path, _ = complete_recipe_manifest(tmp_path)
    request = request_for(tmp_path, manifest, archive_path)
    destination = prepare(request, execute=True)
    archive_path.unlink()
    reuse = request.with_local_inputs({})

    assert prepare(reuse, execute=True) == destination

    (destination / "audio" / "base.wav").write_bytes(b"corrupt")
    with pytest.raises(PreparationError, match="existing destination"):
        prepare(reuse, execute=True)


def test_existing_destination_rejects_unexpected_empty_directory(
    tmp_path: Path,
) -> None:
    manifest, archive_path, _ = complete_recipe_manifest(tmp_path)
    request = request_for(tmp_path, manifest, archive_path)
    destination = prepare(request, execute=True)
    (destination / "unexpected").mkdir()

    with pytest.raises(PreparationError, match="existing destination"):
        prepare(request, execute=True)


def test_existing_receipt_is_read_from_a_no_follow_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, archive_path, _ = complete_recipe_manifest(tmp_path)
    request = request_for(tmp_path, manifest, archive_path)
    destination = prepare(request, execute=True)
    receipt_path = destination / "receipt.json"
    decoy = tmp_path / "decoy-receipt.json"
    decoy.write_bytes(receipt_path.read_bytes())
    original_read_bytes = Path.read_bytes
    swapped = False

    def swap_before_read(path: Path) -> bytes:
        nonlocal swapped
        if path == receipt_path and not swapped:
            swapped = True
            path.unlink()
            path.symlink_to(decoy)
            data = original_read_bytes(path)
            path.unlink()
            path.write_bytes(data)
            return data
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", swap_before_read)

    assert prepare(request, execute=True) == destination
    assert not swapped


def test_receipt_and_persisted_schemas_are_strict_and_frozen(tmp_path: Path) -> None:
    manifest, archive_path, _ = complete_recipe_manifest(tmp_path)
    request = request_for(tmp_path, manifest, archive_path)
    destination = prepare(request, execute=True)
    receipt = PreparationReceipt.model_validate_json(
        (destination / "receipt.json").read_bytes()
    )

    with pytest.raises(ValidationError):
        PreparationReceipt.model_validate(
            {**receipt.model_dump(mode="json"), "surprise": True}
        )
    with pytest.raises(ValidationError):
        receipt.status = "partial"  # type: ignore[misc]
