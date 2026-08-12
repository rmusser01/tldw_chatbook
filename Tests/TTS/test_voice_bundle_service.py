"""Retained ownership tests for explicit clone-voice bundle portability."""

from __future__ import annotations

import asyncio
import copy
import os
import pickle
import stat
import struct
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

import tldw_chatbook.TTS.voice_bundle_service as bundle_service
from tldw_chatbook.TTS.TTS_Generation import AudioCppGuidedDependencySnapshot
from tldw_chatbook.TTS.profile_portability import PortableTTSProfile
from tldw_chatbook.TTS.profile_reference_audio import validate_canonical_reference_wav
from tldw_chatbook.TTS.profile_reference_types import (
    CanonicalTTSCloneReference,
    TTSCloneRecipeRequirement,
    TTSCloneReferenceSummary,
)
from tldw_chatbook.TTS.profile_repository import (
    TTSBundleImportRepositoryFacts,
    TTSBundleImportResult,
)
from tldw_chatbook.TTS.profile_types import (
    ProfileStoreResult,
    TTSProfileCollisionSnapshot,
    TTSProfileDraft,
)
from tldw_chatbook.TTS.voice_bundle_codec import (
    TTSCloneVoiceBundle,
    TTSVoiceBundleError,
    encode_clone_voice_bundle,
)
from tldw_chatbook.TTS.voice_bundle_service import (
    TTSVoiceBundleImportChoice,
    TTSVoiceBundlePortabilityService,
)


PROFILE_ID = UUID("01234567-89ab-4cde-8fab-0123456789ab")
COPY_ID = UUID("11234567-89ab-4cde-8fab-0123456789ab")


def _canonical_wav(*, sample: int = 0) -> bytes:
    frames = struct.pack("<h", sample) * 160
    fmt = struct.pack("<HHIIHH", 1, 1, 16_000, 32_000, 2, 16)
    body = b"WAVEfmt " + struct.pack("<I", 16) + fmt
    body += b"data" + struct.pack("<I", len(frames)) + frames
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _requirement() -> TTSCloneRecipeRequirement:
    return TTSCloneRecipeRequirement(
        recipe_id="pocket-tts",
        recipe_revision=7,
        model_id="model-a",
    )


def _bundle(*, sample: int = 0, name: str = "Imported voice") -> TTSCloneVoiceBundle:
    wav = _canonical_wav(sample=sample)
    metadata = validate_canonical_reference_wav(wav)
    return TTSCloneVoiceBundle(
        profile=PortableTTSProfile(
            profile_id=PROFILE_ID,
            draft=TTSProfileDraft(
                display_name=name,
                provider_id="audio_cpp",
                model_id="model-a",
                voice_id=None,
                response_format="wav",
                speed=1.0,
                options={},
            ),
        ),
        reference=CanonicalTTSCloneReference(
            wav_bytes=wav,
            reference_text="Exact private transcript.",
            sha256=sha256(wav).hexdigest(),
            byte_length=metadata.byte_length,
            duration_ms=metadata.duration_ms,
            sample_rate_hz=metadata.sample_rate_hz,
            channels=metadata.channels,
            sample_encoding=metadata.sample_encoding,
        ),
        recipe_requirement=_requirement(),
    )


def _dependency(*, state: str = "exact", revision: int = 4):
    requirement = _requirement() if state == "exact" else None
    return AudioCppGuidedDependencySnapshot(
        state=state,  # type: ignore[arg-type]
        provider_configuration_revision=revision,
        saved_generation=2,
        applied_generation=2,
        pending_configuration=state == "pending",
        saved_requirement=requirement,
        applied_requirement=requirement,
    )


class _Repository:
    def __init__(self) -> None:
        self.generation = 3
        self.collisions = TTSProfileCollisionSnapshot(None, None)
        self.commits: list[object] = []
        self.reference_result = None

    async def get_profile_collisions(self, profile_id, draft):
        collisions = (
            self.collisions
            if profile_id == PROFILE_ID and draft.normalized_name == "imported voice"
            else TTSProfileCollisionSnapshot(None, None)
        )
        return ProfileStoreResult(self.generation, collisions)

    async def get_reference(
        self, profile_id, *, expected_revision, expected_generation
    ):
        del profile_id, expected_revision, expected_generation
        if self.reference_result is None:
            raise AssertionError("unexpected private reference read")
        return ProfileStoreResult(self.generation, self.reference_result)

    async def get_profile(self, profile_id):
        del profile_id
        raise AssertionError("unexpected profile read")

    async def commit_bundle_import(self, command):
        self.commits.append(command)
        result = TTSBundleImportResult(
            kind="created", profile=command_to_profile(command)
        )
        return ProfileStoreResult(self.generation, result)


def command_to_profile(command):
    from tldw_chatbook.TTS.profile_types import TTSGenerationProfile

    profile_id = (
        command.copy_profile_id
        if command.choice == "copy"
        else command.source_profile_id
    )
    display_name = (
        command.copy_display_name
        if command.choice == "copy"
        else command.source_draft.display_name
    )
    now = datetime(2026, 8, 12, tzinfo=UTC)
    return TTSGenerationProfile(
        profile_id=profile_id,
        display_name=display_name,
        normalized_name=display_name.casefold(),
        provider_id=command.source_draft.provider_id,
        model_id=command.source_draft.model_id,
        voice_id=command.source_draft.voice_id,
        response_format=command.source_draft.response_format,
        speed=command.source_draft.speed,
        options=command.source_draft.options,
        revision=2,
        created_at=now,
        updated_at=now,
        reference=TTSCloneReferenceSummary(
            reference_id=UUID("21234567-89ab-4cde-8fab-0123456789ab"),
            byte_length=command.canonical_reference.byte_length,
            duration_ms=command.canonical_reference.duration_ms,
            sample_rate_hz=command.canonical_reference.sample_rate_hz,
            channels=command.canonical_reference.channels,
            sample_encoding=command.canonical_reference.sample_encoding,
            created_at=now,
            updated_at=now,
            recipe_requirement=command.recipe_requirement,
        ),
    )


class _DependencyService:
    def __init__(self) -> None:
        self.snapshot = _dependency()
        self.calls = 0

    async def audio_cpp_guided_dependency_snapshot(self, requirement):
        assert requirement == _requirement()
        self.calls += 1
        return self.snapshot


def _service(tmp_path: Path, **kwargs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    repository = _Repository()
    dependency = _DependencyService()
    service = TTSVoiceBundlePortabilityService(
        tmp_path / "owned-portability",
        repository,
        dependency,
        uuid_factory=lambda: COPY_ID,
        **kwargs,
    )
    return service, repository, dependency


def _write_source(path: Path, bundle: TTSCloneVoiceBundle | None = None) -> None:
    path.write_bytes(encode_clone_voice_bundle(bundle or _bundle()))
    path.chmod(0o644)


@pytest.mark.asyncio
async def test_inspection_narrows_source_and_cleans_private_operation_root(
    tmp_path: Path,
) -> None:
    source = tmp_path / "selected.tldw-voice.zip"
    _write_source(source)
    service, repository, dependency = _service(tmp_path)

    review = await service.inspect(source)

    assert stat.S_IMODE(source.stat().st_mode) == 0o600
    assert review.profile_id == PROFILE_ID
    assert review.profile_name == "Imported voice"
    assert review.dependency_state == "exact"
    assert review.allowed_choices == ("create",)
    assert "selected" not in repr(review)
    assert list((tmp_path / "owned-portability").iterdir()) == []
    assert repository.commits == []
    assert dependency.calls == 1
    await service.close()
    await service.wait_closed()


@pytest.mark.asyncio
async def test_source_mutation_during_copy_fails_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "selected.tldw-voice.zip"
    _write_source(source)
    service, _repository, _dependency = _service(tmp_path)

    def mutate(boundary: str) -> None:
        if boundary == "source_copy_complete":
            source.write_bytes(encode_clone_voice_bundle(_bundle(sample=1)))

    monkeypatch.setattr(bundle_service, "_test_boundary", mutate)

    with pytest.raises(TTSVoiceBundleError, match="source_changed"):
        await service.inspect(source)

    assert list((tmp_path / "owned-portability").iterdir()) == []
    await service.close()


@pytest.mark.asyncio
async def test_source_replacement_during_review_blocks_commit_before_repository(
    tmp_path: Path,
) -> None:
    source = tmp_path / "selected.tldw-voice.zip"
    replacement = tmp_path / "replacement.tldw-voice.zip"
    _write_source(source)
    _write_source(replacement, _bundle(sample=1))
    service, repository, _dependency = _service(tmp_path)
    review = await service.inspect(source)
    os.replace(replacement, source)

    with pytest.raises(TTSVoiceBundleError, match="source_changed"):
        await service.commit(
            review.handle,
            TTSVoiceBundleImportChoice(choice="create", inactive_consent=False),
        )

    assert repository.commits == []
    assert list((tmp_path / "owned-portability").iterdir()) == []
    await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary",
    (
        "source_initial_open",
        "source_copy_progress",
        "source_copy_complete",
        "source_post_copy",
        "source_post_inspection",
        "source_pre_fingerprint",
    ),
)
async def test_commit_revalidates_source_at_every_worker_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    source = tmp_path / "selected.tldw-voice.zip"
    _write_source(source)
    service, repository, _dependency = _service(tmp_path)
    review = await service.inspect(source)
    replaced = False

    def substitute(observed: str) -> None:
        nonlocal replaced
        if observed != boundary or replaced:
            return
        replaced = True
        replacement = tmp_path / "replacement.zip"
        _write_source(replacement, _bundle(sample=1))
        os.replace(replacement, source)

    monkeypatch.setattr(bundle_service, "_test_boundary", substitute)
    with pytest.raises(TTSVoiceBundleError, match="source_changed"):
        await service.commit(
            review.handle,
            TTSVoiceBundleImportChoice("create", False),
        )
    assert replaced is True
    assert repository.commits == []
    assert list((tmp_path / "owned-portability").iterdir()) == []
    await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary",
    (
        "source_initial_open",
        "source_copy_progress",
        "source_copy_complete",
        "source_post_copy",
        "source_post_inspection",
        "source_pre_fingerprint",
    ),
)
async def test_source_replacement_at_every_worker_boundary_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    source = tmp_path / "selected.tldw-voice.zip"
    _write_source(source)
    service, repository, _dependency = _service(tmp_path)
    replaced = False

    def substitute(observed: str) -> None:
        nonlocal replaced
        if observed != boundary or replaced:
            return
        replaced = True
        replacement = tmp_path / "replacement.zip"
        _write_source(replacement, _bundle(sample=1))
        os.replace(replacement, source)

    monkeypatch.setattr(bundle_service, "_test_boundary", substitute)

    with pytest.raises(TTSVoiceBundleError, match="source_changed"):
        await service.inspect(source)
    assert replaced is True
    assert repository.commits == []
    assert list((tmp_path / "owned-portability").iterdir()) == []
    await service.close()


@pytest.mark.asyncio
async def test_source_symlink_and_hardlink_are_refused_before_bundle_parse(
    tmp_path: Path,
) -> None:
    source = tmp_path / "selected.tldw-voice.zip"
    _write_source(source)
    service, repository, dependency = _service(tmp_path)
    symlink = tmp_path / "linked.zip"
    symlink.symlink_to(source)
    hardlink = tmp_path / "hardlinked.zip"
    os.link(source, hardlink)

    for unsafe in (symlink, hardlink):
        with pytest.raises(TTSVoiceBundleError, match="source_changed"):
            await service.inspect(unsafe)
    assert repository.commits == []
    assert dependency.calls == 0
    assert list((tmp_path / "owned-portability").iterdir()) == []
    await service.close()


def test_source_owner_type_link_and_size_policy_is_exact(monkeypatch) -> None:
    owner = os.geteuid()

    def info(*, mode=stat.S_IFREG | 0o600, uid=owner, links=1, size=1):
        return SimpleNamespace(st_mode=mode, st_uid=uid, st_nlink=links, st_size=size)

    assert bundle_service._owned_source(info()) is True
    assert bundle_service._owned_source(info(mode=stat.S_IFDIR | 0o700)) is False
    assert bundle_service._owned_source(info(uid=owner + 1)) is False
    assert bundle_service._owned_source(info(links=2)) is False
    assert bundle_service._owned_source(info(size=0)) is False
    assert (
        bundle_service._owned_source(
            info(size=bundle_service.MAX_BUNDLE_ARCHIVE_BYTES + 1)
        )
        is False
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("occupant", ("file", "symlink"))
async def test_operation_root_refuses_non_directory_or_symlink(
    tmp_path: Path,
    occupant: str,
) -> None:
    source = tmp_path / "selected.tldw-voice.zip"
    _write_source(source)
    root = tmp_path / "owned-portability"
    if occupant == "file":
        root.write_bytes(b"foreign")
    else:
        target = tmp_path / "foreign-root"
        target.mkdir()
        root.symlink_to(target, target_is_directory=True)
    repository = _Repository()
    dependency = _DependencyService()
    service = TTSVoiceBundlePortabilityService(root, repository, dependency)

    with pytest.raises(TTSVoiceBundleError, match="operation_failed"):
        await service.inspect(source)
    assert dependency.calls == 0
    assert repository.commits == []


@pytest.mark.asyncio
async def test_operation_file_substitution_is_preserved_and_fails_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "selected.tldw-voice.zip"
    _write_source(source)
    service, repository, _dependency = _service(tmp_path)
    foreign_payload = b"foreign occupant"
    substituted: Path | None = None

    def substitute(boundary: str) -> None:
        nonlocal substituted
        if boundary != "source_post_inspection" or substituted is not None:
            return
        operation = next((tmp_path / "owned-portability").iterdir())
        target = operation / "reference.txt"
        target.unlink()
        target.write_bytes(foreign_payload)
        target.chmod(0o600)
        substituted = target

    monkeypatch.setattr(bundle_service, "_test_boundary", substitute)

    with pytest.raises(TTSVoiceBundleError, match="cleanup_failed"):
        await service.inspect(source)
    assert substituted is not None
    assert substituted.read_bytes() == foreign_payload
    assert repository.commits == []


@pytest.mark.asyncio
async def test_partial_codec_sink_failure_discards_owned_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "selected.tldw-voice.zip"
    _write_source(source)
    service, repository, dependency = _service(tmp_path)

    def fail_after_partial(payload, *, sinks):
        del payload
        sinks.reference_wav.write(b"private partial")
        raise TTSVoiceBundleError("bundle_invalid")

    monkeypatch.setattr(
        bundle_service, "inspect_clone_voice_bundle", fail_after_partial
    )
    with pytest.raises(TTSVoiceBundleError, match="bundle_invalid"):
        await service.inspect(source)
    assert list((tmp_path / "owned-portability").iterdir()) == []
    assert repository.commits == []
    assert dependency.calls == 0
    await service.close()


@pytest.mark.asyncio
async def test_handle_is_single_use_foreign_expiring_and_uncopyable(
    tmp_path: Path,
) -> None:
    source = tmp_path / "selected.tldw-voice.zip"
    _write_source(source)
    now = [10.0]
    service, repository, _dependency = _service(tmp_path, clock=lambda: now[0])
    other, _other_repository, _other_dependency = _service(
        tmp_path / "other", clock=lambda: now[0]
    )
    review = await service.inspect(source)

    assert review.handle.is_redacted is True
    assert "source" not in repr(review.handle)
    for operation in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError):
            operation(review.handle)
    with pytest.raises(TTSVoiceBundleError, match="stale_inspection"):
        await other.commit(
            review.handle,
            TTSVoiceBundleImportChoice("create", False),
        )

    result = await service.commit(
        review.handle,
        TTSVoiceBundleImportChoice("create", False),
    )
    assert result.status == "created"
    assert len(repository.commits) == 1
    with pytest.raises(TTSVoiceBundleError, match="stale_inspection"):
        await service.commit(
            review.handle,
            TTSVoiceBundleImportChoice("create", False),
        )

    expiring = await service.inspect(source)
    now[0] += 601
    with pytest.raises(TTSVoiceBundleError, match="stale_inspection"):
        await service.commit(
            expiring.handle,
            TTSVoiceBundleImportChoice("create", False),
        )
    await other.close()
    await service.close()


@pytest.mark.asyncio
async def test_service_caps_live_sessions_at_four(tmp_path: Path) -> None:
    source = tmp_path / "selected.tldw-voice.zip"
    _write_source(source)
    service, _repository, _dependency = _service(tmp_path)

    for _ in range(4):
        await service.inspect(source)
    with pytest.raises(TTSVoiceBundleError, match="operation_failed"):
        await service.inspect(source)
    await service.close()


@pytest.mark.asyncio
async def test_stale_repository_result_returns_brand_new_review(tmp_path: Path) -> None:
    source = tmp_path / "selected.tldw-voice.zip"
    _write_source(source)
    service, repository, dependency = _service(tmp_path)
    review = await service.inspect(source)

    async def stale(command):
        repository.commits.append(command)
        repository.collisions = TTSProfileCollisionSnapshot(
            command_to_profile(command),
            command_to_profile(command),
        )
        dependency.snapshot = _dependency(state="missing", revision=5)
        return ProfileStoreResult(
            repository.generation,
            TTSBundleImportResult(
                kind="stale_inspection",
                profile=None,
                repository_facts=TTSBundleImportRepositoryFacts(
                    source_collisions=repository.collisions,
                    copy_collisions=None,
                ),
            ),
        )

    repository.commit_bundle_import = stale
    result = await service.commit(
        review.handle,
        TTSVoiceBundleImportChoice("create", False),
    )

    assert result.status == "stale_inspection"
    assert result.review is not None
    assert result.review.handle is not review.handle
    assert result.review.dependency_state == "missing"
    assert result.review.allowed_choices == ("copy",)
    assert result.review.copy_profile_id == COPY_ID
    assert dependency.calls >= 3
    with pytest.raises(TTSVoiceBundleError, match="stale_inspection"):
        await service.commit(
            review.handle,
            TTSVoiceBundleImportChoice("create", False),
        )
    await service.close()


@pytest.mark.asyncio
async def test_export_requires_ack_and_publishes_create_only_private_file(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "portable.tldw-voice.zip"
    service, repository, _dependency = _service(tmp_path)
    bundle = _bundle()
    profile = command_to_profile(
        SimpleNamespace(
            choice="create",
            source_profile_id=PROFILE_ID,
            copy_profile_id=None,
            copy_display_name=None,
            source_draft=bundle.profile.draft,
            canonical_reference=bundle.reference,
            recipe_requirement=bundle.recipe_requirement,
        )
    )

    async def get_profile(profile_id):
        assert profile_id == PROFILE_ID
        return ProfileStoreResult(repository.generation, profile)

    async def get_reference(profile_id, *, expected_revision, expected_generation):
        assert profile_id == PROFILE_ID
        assert expected_revision == profile.revision
        assert expected_generation == repository.generation
        from tldw_chatbook.TTS.profile_reference_types import TTSCloneReference

        return ProfileStoreResult(
            repository.generation,
            TTSCloneReference(
                summary=profile.reference,
                wav_bytes=bundle.reference.wav_bytes,
                reference_text=bundle.reference.reference_text,
                sha256=bundle.reference.sha256,
                recipe_requirement=bundle.recipe_requirement,
            ),
        )

    repository.get_profile = get_profile
    repository.get_reference = get_reference

    with pytest.raises(TTSVoiceBundleError, match="acknowledgement_required"):
        await service.export(
            PROFILE_ID,
            destination,
            expected_generation=repository.generation,
            expected_revision=profile.revision,
            acknowledged=False,
        )
    assert not destination.exists()

    await service.export(
        PROFILE_ID,
        destination,
        expected_generation=repository.generation,
        expected_revision=profile.revision,
        acknowledged=True,
    )
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert destination.read_bytes() == encode_clone_voice_bundle(bundle)
    with pytest.raises(TTSVoiceBundleError, match="destination_changed"):
        await service.export(
            PROFILE_ID,
            destination,
            expected_generation=repository.generation,
            expected_revision=profile.revision,
            acknowledged=True,
        )
    assert destination.read_bytes() == encode_clone_voice_bundle(bundle)
    await service.close()


@pytest.mark.asyncio
async def test_export_fsyncs_deterministic_bytes_and_refuses_symlink_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "portable.tldw-voice.zip"
    service, repository, _dependency = _service(tmp_path)
    bundle = _bundle()
    profile = command_to_profile(
        SimpleNamespace(
            choice="create",
            source_profile_id=PROFILE_ID,
            copy_profile_id=None,
            copy_display_name=None,
            source_draft=bundle.profile.draft,
            canonical_reference=bundle.reference,
            recipe_requirement=bundle.recipe_requirement,
        )
    )

    async def get_profile(_profile_id):
        return ProfileStoreResult(repository.generation, profile)

    from tldw_chatbook.TTS.profile_reference_types import TTSCloneReference

    reference = TTSCloneReference(
        summary=profile.reference,
        wav_bytes=bundle.reference.wav_bytes,
        reference_text=bundle.reference.reference_text,
        sha256=bundle.reference.sha256,
        recipe_requirement=bundle.recipe_requirement,
    )

    async def get_reference(*_args, **_kwargs):
        return ProfileStoreResult(repository.generation, reference)

    repository.get_profile = get_profile
    repository.get_reference = get_reference
    real_fsync = bundle_service.os.fsync
    fsync_calls: list[int] = []

    def record_fsync(descriptor: int) -> None:
        fsync_calls.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr(bundle_service.os, "fsync", record_fsync)
    await service.export(
        PROFILE_ID,
        destination,
        expected_generation=repository.generation,
        expected_revision=profile.revision,
        acknowledged=True,
    )
    assert destination.read_bytes() == encode_clone_voice_bundle(bundle)
    assert len(fsync_calls) >= 2

    symlink_destination = tmp_path / "symlink.tldw-voice.zip"
    symlink_destination.symlink_to(destination)
    with pytest.raises(TTSVoiceBundleError, match="destination_changed"):
        await service.export(
            PROFILE_ID,
            symlink_destination,
            expected_generation=repository.generation,
            expected_revision=profile.revision,
            acknowledged=True,
        )
    assert symlink_destination.is_symlink()
    assert destination.read_bytes() == encode_clone_voice_bundle(bundle)
    await service.close()


@pytest.mark.asyncio
async def test_export_preserves_substituted_temporary_and_publishes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "portable.tldw-voice.zip"
    service, repository, _dependency = _service(tmp_path)
    bundle = _bundle()
    profile = command_to_profile(
        SimpleNamespace(
            choice="create",
            source_profile_id=PROFILE_ID,
            copy_profile_id=None,
            copy_display_name=None,
            source_draft=bundle.profile.draft,
            canonical_reference=bundle.reference,
            recipe_requirement=bundle.recipe_requirement,
        )
    )
    from tldw_chatbook.TTS.profile_reference_types import TTSCloneReference

    reference = TTSCloneReference(
        summary=profile.reference,
        wav_bytes=bundle.reference.wav_bytes,
        reference_text=bundle.reference.reference_text,
        sha256=bundle.reference.sha256,
        recipe_requirement=bundle.recipe_requirement,
    )

    async def get_profile(_profile_id):
        return ProfileStoreResult(repository.generation, profile)

    async def get_reference(*_args, **_kwargs):
        return ProfileStoreResult(repository.generation, reference)

    repository.get_profile = get_profile
    repository.get_reference = get_reference
    foreign = b"foreign temporary occupant"
    substituted: Path | None = None

    def substitute(boundary: str) -> None:
        nonlocal substituted
        if boundary != "destination_pre_publish":
            return
        temporary = next(tmp_path.glob(".portable.tldw-voice.zip.*.tmp"))
        temporary.unlink()
        temporary.write_bytes(foreign)
        temporary.chmod(0o600)
        substituted = temporary

    monkeypatch.setattr(bundle_service, "_test_boundary", substitute)
    with pytest.raises(TTSVoiceBundleError, match="cleanup_failed"):
        await service.export(
            PROFILE_ID,
            destination,
            expected_generation=repository.generation,
            expected_revision=profile.revision,
            acknowledged=True,
        )
    assert not destination.exists()
    assert substituted is not None
    assert substituted.read_bytes() == foreign
    await service.close()


@pytest.mark.asyncio
async def test_cancellation_waits_for_retained_worker_and_close_joins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "selected.tldw-voice.zip"
    _write_source(source)
    service, _repository, _dependency = _service(tmp_path)
    entered = asyncio.Event()
    release = asyncio.Event()

    original = bundle_service._copy_and_inspect_sync

    def slow(*args, **kwargs):
        loop = kwargs.pop("test_loop")
        loop.call_soon_threadsafe(entered.set)
        while not release.is_set():
            pass
        return original(*args, **kwargs)

    loop = asyncio.get_running_loop()

    def wrapped(*args, **kwargs):
        return slow(*args, **kwargs, test_loop=loop)

    monkeypatch.setattr(bundle_service, "_copy_and_inspect_sync", wrapped)
    inspection = asyncio.create_task(service.inspect(source))
    await entered.wait()
    inspection.cancel()
    await asyncio.sleep(0)
    settled_before_worker = inspection.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await inspection
    assert settled_before_worker is False
    await service.close()
    await service.wait_closed()
    assert list((tmp_path / "owned-portability").iterdir()) == []


@pytest.mark.asyncio
async def test_close_waits_for_active_inspection_before_returning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "selected.tldw-voice.zip"
    _write_source(source)
    service, _repository, _dependency = _service(tmp_path)
    entered = asyncio.Event()
    release = asyncio.Event()
    original = bundle_service._copy_and_inspect_sync
    loop = asyncio.get_running_loop()

    def slow(*args, **kwargs):
        del kwargs
        loop.call_soon_threadsafe(entered.set)
        while not release.is_set():
            pass
        return original(*args)

    monkeypatch.setattr(bundle_service, "_copy_and_inspect_sync", slow)
    inspection = asyncio.create_task(service.inspect(source))
    await entered.wait()
    closing = asyncio.create_task(service.close())
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    with pytest.raises(TTSVoiceBundleError, match="operation_failed"):
        await inspection
    await closing
    await service.wait_closed()
    assert list((tmp_path / "owned-portability").iterdir()) == []
