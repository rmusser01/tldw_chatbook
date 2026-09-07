"""Retained ownership tests for explicit clone-voice bundle portability."""

from __future__ import annotations

import asyncio
import copy
import os
import pickle
import stat
import struct
import threading
from contextlib import asynccontextmanager
from dataclasses import fields, is_dataclass, replace
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
    TTSGenerationProfile,
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


def _producer_dependency_cases():
    requirement = _requirement()
    return (
        (
            "exact",
            AudioCppGuidedDependencySnapshot(
                "exact", 4, 2, 2, False, requirement, requirement
            ),
            False,
        ),
        (
            "exact-pending-settings",
            AudioCppGuidedDependencySnapshot(
                "exact", 5, 3, 2, True, requirement, requirement
            ),
            False,
        ),
        (
            "applied-exact-saved-drift",
            AudioCppGuidedDependencySnapshot("exact", 5, 3, 2, True, None, requirement),
            False,
        ),
        (
            "missing-pending-settings",
            AudioCppGuidedDependencySnapshot("missing", 5, 3, 2, True, None, None),
            True,
        ),
        (
            "mismatch-stable-settings",
            AudioCppGuidedDependencySnapshot("mismatch", 4, 2, 2, False, None, None),
            True,
        ),
        (
            "pending-saved-exact",
            AudioCppGuidedDependencySnapshot(
                "pending", 5, 3, 2, True, requirement, None
            ),
            True,
        ),
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


@asynccontextmanager
async def _profile_mutation_fence():
    yield


def _service(tmp_path: Path, **kwargs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    repository = _Repository()
    dependency = _DependencyService()
    kwargs.setdefault("profile_mutation_fence", _profile_mutation_fence)
    service = TTSVoiceBundlePortabilityService(
        tmp_path / "owned-portability",
        repository,
        dependency,
        uuid_factory=lambda: COPY_ID,
        **kwargs,
    )
    return service, repository, dependency


class _ArtifactLeaseCoordinator:
    def __init__(self) -> None:
        self.active = False
        self.calls: list[tuple[object, ...]] = []

    @asynccontextmanager
    async def lease_consumers(self, consumers):
        self.calls.append(tuple(consumers))
        self.active = True
        try:
            yield
        finally:
            self.active = False


class _ProfileMutationFence:
    def __init__(self) -> None:
        self.active = False

    @asynccontextmanager
    async def hold(self):
        self.active = True
        try:
            yield
        finally:
            self.active = False


@pytest.mark.asyncio
async def test_bundle_import_holds_artifact_lease_through_repository_commit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "selected.tldw-voice.zip"
    _write_source(source)
    coordinator = _ArtifactLeaseCoordinator()
    mutation_fence = _ProfileMutationFence()
    service, repository, _dependency = _service(
        tmp_path,
        artifact_lease_coordinator=coordinator,
        profile_mutation_fence=mutation_fence.hold,
    )
    repository_commit = repository.commit_bundle_import

    async def guarded_commit(command):
        assert coordinator.active is True
        assert mutation_fence.active is True
        return await repository_commit(command)

    repository.commit_bundle_import = guarded_commit
    review = await service.inspect(source)

    await service.commit(
        review.handle,
        TTSVoiceBundleImportChoice("create", False),
    )

    assert coordinator.active is False
    assert mutation_fence.active is False
    assert len(coordinator.calls) == 1
    requirement = coordinator.calls[0][0]
    assert requirement.provider_id == "audio_cpp"
    assert requirement.model_id == "model-a"
    assert requirement.recipe_requirement == _requirement()
    await service.close()


def _write_source(path: Path, bundle: TTSCloneVoiceBundle | None = None) -> None:
    path.write_bytes(encode_clone_voice_bundle(bundle or _bundle()))
    path.chmod(0o644)


def _install_export_record(
    repository: _Repository,
) -> tuple[TTSCloneVoiceBundle, TTSGenerationProfile]:
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
    return bundle, profile


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
@pytest.mark.parametrize(
    ("uuid_conflict", "name_conflict"),
    ((False, False), (True, False), (False, True), (True, True)),
)
async def test_review_exposes_only_safe_collision_booleans(
    tmp_path: Path,
    uuid_conflict: bool,
    name_conflict: bool,
) -> None:
    source = tmp_path / "selected.tldw-voice.zip"
    _write_source(source)
    service, repository, _dependency_service = _service(tmp_path)
    bundle = _bundle()
    candidate = command_to_profile(
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
    id_match = (
        replace(
            candidate,
            display_name="CANARY uuid collision",
            normalized_name="canary uuid collision",
        )
        if uuid_conflict
        else None
    )
    name_match = (
        replace(
            candidate,
            profile_id=COPY_ID,
            display_name="CANARY name collision",
            normalized_name="canary name collision",
        )
        if name_conflict
        else None
    )
    repository.collisions = TTSProfileCollisionSnapshot(id_match, name_match)

    review = await service.inspect(source)

    assert review.uuid_conflict is uuid_conflict
    assert review.name_conflict is name_conflict
    assert "CANARY" not in repr(review)
    await service.close()


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
    service = TTSVoiceBundlePortabilityService(
        root,
        repository,
        dependency,
        profile_mutation_fence=_profile_mutation_fence,
    )

    with pytest.raises(TTSVoiceBundleError, match="operation_failed"):
        await service.inspect(source)
    assert dependency.calls == 0
    assert repository.commits == []


@pytest.mark.asyncio
@pytest.mark.parametrize("residue", ("recognized", "unrecognized", "symlink", "mode"))
async def test_restart_preserves_operation_root_residue_and_reports_cleanup_failed(
    tmp_path: Path,
    residue: str,
) -> None:
    root = tmp_path / "owned-portability"
    root.mkdir(mode=0o700)
    if residue == "recognized":
        occupant = root / "operation-crash-residue"
        occupant.mkdir(mode=0o700)
    elif residue == "unrecognized":
        occupant = root / "foreign-residue"
        occupant.write_bytes(b"private residue")
        occupant.chmod(0o600)
    elif residue == "symlink":
        target = tmp_path / "foreign-target"
        target.write_bytes(b"foreign")
        occupant = root / "operation-symlink"
        occupant.symlink_to(target)
    else:
        occupant = root / "operation-wrong-mode"
        occupant.mkdir(mode=0o755)

    source = tmp_path / "selected.tldw-voice.zip"
    _write_source(source)
    for _ in range(2):
        service, _repository, _dependency_service = _service(tmp_path)
        with pytest.raises(TTSVoiceBundleError, match="cleanup_failed") as caught:
            await service.inspect(source)
        assert str(caught.value) == "cleanup_failed"
        assert occupant.exists() or occupant.is_symlink()
        await service.close()


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
        current = command_to_profile(command)
        repository.collisions = TTSProfileCollisionSnapshot(
            current,
            None,
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
@pytest.mark.parametrize("failure", ("read", "validation"))
async def test_exact_public_collision_private_reference_failure_blocks_review(
    tmp_path: Path,
    failure: str,
) -> None:
    source = tmp_path / "selected.tldw-voice.zip"
    _write_source(source)
    service, repository, _dependency_service = _service(tmp_path)
    bundle = _bundle()
    exact = command_to_profile(
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
    repository.collisions = TTSProfileCollisionSnapshot(exact, exact)

    if failure == "read":

        async def fail_reference(*_args, **_kwargs):
            raise RuntimeError("CANARY-private-reference-read")

    else:

        async def fail_reference(*_args, **_kwargs):
            return ProfileStoreResult(repository.generation, object())

    repository.get_reference = fail_reference
    with pytest.raises(TTSVoiceBundleError, match="operation_failed"):
        await service.inspect(source)
    assert service._sessions == {}
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
    with pytest.raises(TTSVoiceBundleError, match="destination_changed"):
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


def _exception_graph_values(error: BaseException) -> list[object]:
    pending: list[object] = [error]
    values: list[object] = []
    seen: set[int] = set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        values.append(value)
        if isinstance(value, BaseException):
            pending.extend(value.args)
            pending.extend(
                item
                for item in (value.__cause__, value.__context__)
                if item is not None
            )
            traceback = value.__traceback__
            while traceback is not None:
                if (
                    traceback.tb_frame.f_globals.get("__name__")
                    == bundle_service.__name__
                ):
                    pending.extend(traceback.tb_frame.f_locals.values())
                traceback = traceback.tb_next
        elif type(value) is dict:
            pending.extend(value.keys())
            pending.extend(value.values())
        elif type(value) in (list, tuple, set, frozenset):
            pending.extend(value)
        elif not isinstance(value, type) and is_dataclass(value):
            pending.extend(getattr(value, field.name) for field in fields(value))
    return values


class _PrivateBaseException(BaseException):
    pass


def _assert_bounded_public_error(error: BaseException, canary: str) -> None:
    assert error.__cause__ is None
    assert error.__context__ is None
    service_frames: list[str] = []
    traceback = error.__traceback__
    while traceback is not None:
        if traceback.tb_frame.f_globals.get("__name__") == bundle_service.__name__:
            service_frames.append(traceback.tb_frame.f_code.co_name)
            assert all(
                canary not in repr(value)
                for value in traceback.tb_frame.f_locals.values()
            )
        traceback = traceback.tb_next
    assert len(service_frames) == 1
    assert all(canary not in repr(value) for value in _exception_graph_values(error))


@pytest.mark.asyncio
@pytest.mark.parametrize("collaborator", ("repository", "dependency"))
async def test_public_inspect_severs_private_collaborator_exception_graph(
    tmp_path: Path,
    collaborator: str,
) -> None:
    source = tmp_path / "CANARY-selected.tldw-voice.zip"
    _write_source(source)
    service, repository, dependency = _service(tmp_path)
    canary = f"CANARY-{collaborator}-private-value"

    if collaborator == "repository":

        async def fail(*_args, **_kwargs):
            raise RuntimeError(canary)

        repository.get_profile_collisions = fail
    else:

        async def fail(_requirement):
            raise RuntimeError(canary)

        dependency.audio_cpp_guided_dependency_snapshot = fail

    with pytest.raises(TTSVoiceBundleError, match="operation_failed") as caught:
        await service.inspect(source)

    error = caught.value
    assert str(error) == "operation_failed"
    assert error.__cause__ is None
    assert error.__context__ is None
    service_frames: list[str] = []
    traceback = error.__traceback__
    while traceback is not None:
        if traceback.tb_frame.f_globals.get("__name__") == bundle_service.__name__:
            service_frames.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    assert service_frames == ["inspect"]
    assert all(canary not in repr(value) for value in _exception_graph_values(error))
    assert all(
        "CANARY-selected" not in repr(value) for value in _exception_graph_values(error)
    )
    await service.close()


@pytest.mark.asyncio
async def test_inspect_preserves_unsupported_platform_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "selected.tldw-voice.zip"
    _write_source(source)
    service, _repository, _dependency = _service(tmp_path)
    monkeypatch.setattr(bundle_service, "_posix_supported", lambda: False)

    with pytest.raises(TTSVoiceBundleError, match="unsupported_platform"):
        await service.inspect(source)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("state", "unknown"),
        ("provider_configuration_revision", -1),
        ("saved_generation", True),
        ("state", "pending"),
        ("applied_requirement", TTSCloneRecipeRequirement("other", 1, "other")),
    ),
)
async def test_invalid_dependency_snapshot_fails_before_review_publication(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    source = tmp_path / "selected.tldw-voice.zip"
    _write_source(source)
    service, repository, dependency = _service(tmp_path)
    forged = _dependency()
    object.__setattr__(forged, field, value)
    dependency.snapshot = forged

    with pytest.raises(TTSVoiceBundleError, match="operation_failed"):
        await service.inspect(source)
    assert repository.commits == []
    await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("_case", "snapshot", "inactive_consent"),
    _producer_dependency_cases(),
)
async def test_real_dependency_snapshot_cross_product_inspects_and_commits(
    tmp_path: Path,
    _case: str,
    snapshot: AudioCppGuidedDependencySnapshot,
    inactive_consent: bool,
) -> None:
    source = tmp_path / "selected.tldw-voice.zip"
    _write_source(source)
    service, repository, dependency = _service(tmp_path)
    dependency.snapshot = snapshot

    review = await service.inspect(source)
    assert review.dependency_state == snapshot.state
    result = await service.commit(
        review.handle,
        TTSVoiceBundleImportChoice("create", inactive_consent),
    )

    assert result.status == "created"
    assert len(repository.commits) == 1
    assert dependency.calls == 2
    await service.close()


@pytest.mark.asyncio
async def test_invalidate_is_idempotent_and_foreign_handles_are_bounded(
    tmp_path: Path,
) -> None:
    source = tmp_path / "selected.tldw-voice.zip"
    _write_source(source)
    service, repository, _dependency = _service(tmp_path)
    foreign, _foreign_repository, _foreign_dependency = _service(tmp_path / "other")
    review = await service.inspect(source)

    await service.invalidate(review.handle)
    await service.invalidate(review.handle)
    with pytest.raises(TTSVoiceBundleError, match="stale_inspection") as foreign_error:
        await foreign.invalidate(review.handle)
    assert foreign_error.value.__cause__ is None
    assert foreign_error.value.__context__ is None
    with pytest.raises(TTSVoiceBundleError, match="stale_inspection") as malformed:
        await foreign.invalidate(object())  # type: ignore[arg-type]
    assert malformed.value.__cause__ is None
    assert malformed.value.__context__ is None
    with pytest.raises(TTSVoiceBundleError, match="stale_inspection"):
        await service.commit(
            review.handle,
            TTSVoiceBundleImportChoice("create", False),
        )
    assert repository.commits == []
    await service.close()
    with pytest.raises(TTSVoiceBundleError, match="operation_failed"):
        await service.invalidate(review.handle)
    await foreign.close()


@pytest.mark.asyncio
async def test_commit_detects_same_size_mutation_with_restored_mtime_before_submit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "selected.tldw-voice.zip"
    first = encode_clone_voice_bundle(_bundle(sample=0))
    second = encode_clone_voice_bundle(_bundle(sample=1))
    assert len(first) == len(second)
    source.write_bytes(first)
    source.chmod(0o600)
    service, repository, _dependency = _service(tmp_path)
    review = await service.inspect(source)
    original_mtime = source.stat().st_mtime_ns

    def mutate(boundary: str) -> None:
        if boundary != "commit_pre_repository":
            return
        source.write_bytes(second)
        source.chmod(0o600)
        os.utime(source, ns=(original_mtime, original_mtime))

    monkeypatch.setattr(bundle_service, "_test_boundary", mutate)
    with pytest.raises(TTSVoiceBundleError, match="source_changed"):
        await service.commit(
            review.handle,
            TTSVoiceBundleImportChoice("create", False),
        )
    assert repository.commits == []
    await service.close()


@pytest.mark.asyncio
async def test_source_parent_substitution_during_review_fails_closed(
    tmp_path: Path,
) -> None:
    selected_parent = tmp_path / "selected-parent"
    selected_parent.mkdir()
    source = selected_parent / "selected.tldw-voice.zip"
    _write_source(source)
    service, repository, _dependency = _service(tmp_path)
    review = await service.inspect(source)
    moved_parent = tmp_path / "moved-parent"
    os.rename(selected_parent, moved_parent)
    selected_parent.mkdir()
    _write_source(selected_parent / source.name)

    with pytest.raises(TTSVoiceBundleError, match="source_changed"):
        await service.commit(
            review.handle,
            TTSVoiceBundleImportChoice("create", False),
        )
    assert repository.commits == []
    await service.close()


@pytest.mark.asyncio
async def test_repository_commit_is_retained_across_cancellation_and_close(
    tmp_path: Path,
) -> None:
    source = tmp_path / "selected.tldw-voice.zip"
    _write_source(source)
    service, repository, _dependency = _service(tmp_path)
    review = await service.inspect(source)
    entered = asyncio.Event()
    commit_future: asyncio.Future[ProfileStoreResult[TTSBundleImportResult]] = (
        asyncio.get_running_loop().create_future()
    )

    async def blocked(command):
        repository.commits.append(command)
        entered.set()
        return await commit_future

    repository.commit_bundle_import = blocked
    committing = asyncio.create_task(
        service.commit(
            review.handle,
            TTSVoiceBundleImportChoice("create", False),
        )
    )
    await entered.wait()
    committing.cancel()
    closing = asyncio.create_task(service.close())
    await asyncio.sleep(0)
    assert not commit_future.cancelled()
    assert not committing.done()
    assert not closing.done()
    command = repository.commits[0]
    commit_future.set_result(
        ProfileStoreResult(
            repository.generation,
            TTSBundleImportResult("created", command_to_profile(command)),
        )
    )
    with pytest.raises(asyncio.CancelledError):
        await committing
    await closing


@pytest.mark.asyncio
async def test_export_cancellation_after_durable_publication_returns_success(
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
    real_publish = bundle_service._publish_sync
    entered = threading.Event()
    release = threading.Event()

    def publish_then_wait(*args):
        outcome = real_publish(*args)
        entered.set()
        release.wait(5)
        return outcome

    monkeypatch.setattr(bundle_service, "_publish_sync", publish_then_wait)
    exporting = asyncio.create_task(
        service.export(
            PROFILE_ID,
            destination,
            expected_generation=repository.generation,
            expected_revision=profile.revision,
            acknowledged=True,
        )
    )
    await asyncio.to_thread(entered.wait, 5)
    assert destination.exists()
    exporting.cancel()
    await asyncio.sleep(0)
    assert not exporting.done()
    release.set()
    await exporting
    assert destination.read_bytes() == encode_clone_voice_bundle(bundle)
    await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault_boundary", ("destination_post_link", "destination_post_fsync")
)
async def test_export_post_ponr_fault_converges_to_reported_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_boundary: str,
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

    def fault(boundary: str) -> None:
        if boundary == fault_boundary:
            raise OSError("CANARY-post-link-private")

    monkeypatch.setattr(bundle_service, "_test_boundary", fault)
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
async def test_export_never_unlinks_final_path_after_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "portable.tldw-voice.zip"
    service, repository, _dependency_service = _service(tmp_path)
    bundle, profile = _install_export_record(repository)
    real_unlink = bundle_service.os.unlink
    final_unlinks: list[str] = []

    def refuse_final_unlink(path, *args, **kwargs):
        if os.fspath(path) == destination.name:
            final_unlinks.append(os.fspath(path))
            raise AssertionError("published final must never be unlinked")
        return real_unlink(path, *args, **kwargs)

    def post_link_fault(boundary: str) -> None:
        if boundary == "destination_post_link":
            raise OSError("transient after PONR")

    monkeypatch.setattr(bundle_service.os, "unlink", refuse_final_unlink)
    monkeypatch.setattr(bundle_service, "_test_boundary", post_link_fault)
    await service.export(
        PROFILE_ID,
        destination,
        expected_generation=repository.generation,
        expected_revision=profile.revision,
        acknowledged=True,
    )

    assert final_unlinks == []
    assert destination.read_bytes() == encode_clone_voice_bundle(bundle)
    await service.close()


@pytest.mark.asyncio
async def test_export_preserves_foreign_substitution_after_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "portable.tldw-voice.zip"
    service, repository, _dependency_service = _service(tmp_path)
    _bundle_value, profile = _install_export_record(repository)
    foreign = b"foreign destination occupant"

    def substitute(boundary: str) -> None:
        if boundary != "destination_post_link":
            return
        destination.unlink()
        destination.write_bytes(foreign)
        destination.chmod(0o600)

    monkeypatch.setattr(bundle_service, "_test_boundary", substitute)
    with pytest.raises(TTSVoiceBundleError, match="cleanup_failed"):
        await service.export(
            PROFILE_ID,
            destination,
            expected_generation=repository.generation,
            expected_revision=profile.revision,
            acknowledged=True,
        )
    assert destination.read_bytes() == foreign
    await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation", ("replace", "overwrite", "overwrite_during_read", "chmod_on_fsync")
)
async def test_export_reverifies_after_every_post_ponr_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    destination = tmp_path / "portable.tldw-voice.zip"
    service, repository, _dependency_service = _service(tmp_path)
    bundle, profile = _install_export_record(repository)
    expected = encode_clone_voice_bundle(bundle)
    foreign = b"F" * len(expected)

    if mutation == "overwrite_during_read":
        real_read = bundle_service.os.read
        destination_reads = 0

        def mutate_after_read(descriptor: int, size: int) -> bytes:
            nonlocal destination_reads
            data = real_read(descriptor, size)
            if (
                destination.exists()
                and stat.S_ISREG(os.fstat(descriptor).st_mode)
                and os.fstat(descriptor).st_ino == destination.stat().st_ino
            ):
                destination_reads += 1
                if destination_reads == 3:
                    original_mtime = destination.stat().st_mtime_ns
                    destination.write_bytes(foreign)
                    destination.chmod(0o600)
                    os.utime(destination, ns=(original_mtime, original_mtime))
            return data

        monkeypatch.setattr(bundle_service.os, "read", mutate_after_read)
    elif mutation == "chmod_on_fsync":
        real_fsync = bundle_service.os.fsync
        mutated = False

        def mutate_during_parent_fsync(descriptor: int) -> None:
            nonlocal mutated
            real_fsync(descriptor)
            if (
                not mutated
                and destination.exists()
                and stat.S_ISDIR(os.fstat(descriptor).st_mode)
            ):
                destination.chmod(0o644)
                mutated = True

        monkeypatch.setattr(bundle_service.os, "fsync", mutate_during_parent_fsync)
    else:

        def mutate_after_fsync(boundary: str) -> None:
            if boundary != "destination_post_fsync":
                return
            if mutation == "replace":
                replacement = tmp_path / "foreign-replacement"
                replacement.write_bytes(foreign)
                replacement.chmod(0o600)
                os.replace(replacement, destination)
            else:
                destination.write_bytes(foreign)
                destination.chmod(0o600)

        monkeypatch.setattr(bundle_service, "_test_boundary", mutate_after_fsync)

    with pytest.raises(TTSVoiceBundleError, match="cleanup_failed"):
        await service.export(
            PROFILE_ID,
            destination,
            expected_generation=repository.generation,
            expected_revision=profile.revision,
            acknowledged=True,
        )
    if mutation == "replace":
        assert destination.read_bytes() == foreign
    await service.close()


@pytest.mark.asyncio
async def test_export_path_never_uses_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "portable.tldw-voice.zip"
    service, repository, _dependency_service = _service(tmp_path)
    bundle, profile = _install_export_record(repository)
    unlink_calls: list[object] = []

    def forbidden_unlink(*args, **kwargs):
        unlink_calls.append((args, kwargs))
        raise AssertionError("export must not unlink any pathname")

    monkeypatch.setattr(bundle_service.os, "unlink", forbidden_unlink)
    await service.export(
        PROFILE_ID,
        destination,
        expected_generation=repository.generation,
        expected_revision=profile.revision,
        acknowledged=True,
    )
    assert unlink_calls == []
    assert destination.read_bytes() == encode_clone_voice_bundle(bundle)
    await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary", ("destination_pre_publish", "destination_post_link")
)
async def test_export_temp_substitution_is_preserved_without_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    destination = tmp_path / "portable.tldw-voice.zip"
    service, repository, _dependency_service = _service(tmp_path)
    bundle, profile = _install_export_record(repository)
    foreign = b"foreign randomized temporary occupant"
    substituted: Path | None = None

    def substitute(observed: str) -> None:
        nonlocal substituted
        if boundary == "destination_post_link" and observed == "destination_pre_rename":
            substituted = next(tmp_path.glob(".portable.tldw-voice.zip.*.tmp"))
            return
        if observed != boundary:
            return
        temporary = (
            substituted
            if substituted is not None
            else next(tmp_path.glob(".portable.tldw-voice.zip.*.tmp"))
        )
        replacement = tmp_path / "replacement-temp"
        replacement.write_bytes(foreign)
        replacement.chmod(0o600)
        os.replace(replacement, temporary)
        substituted = temporary

    def forbidden_unlink(*_args, **_kwargs):
        raise AssertionError("export must not unlink substituted temp paths")

    monkeypatch.setattr(bundle_service, "_test_boundary", substitute)
    monkeypatch.setattr(bundle_service.os, "unlink", forbidden_unlink)
    if boundary == "destination_pre_publish":
        with pytest.raises(TTSVoiceBundleError, match="destination_changed"):
            await service.export(
                PROFILE_ID,
                destination,
                expected_generation=repository.generation,
                expected_revision=profile.revision,
                acknowledged=True,
            )
        assert not destination.exists()
    else:
        await service.export(
            PROFILE_ID,
            destination,
            expected_generation=repository.generation,
            expected_revision=profile.revision,
            acknowledged=True,
        )
        assert destination.read_bytes() == encode_clone_voice_bundle(bundle)
    assert substituted is not None
    assert substituted.read_bytes() == foreign
    await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary",
    ("destination_pre_publish", "destination_pre_rename", "destination_post_link"),
)
async def test_export_cancellation_respects_atomic_publication_ponr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    destination = tmp_path / "portable.tldw-voice.zip"
    service, repository, _dependency_service = _service(tmp_path)
    bundle, profile = _install_export_record(repository)
    entered = threading.Event()
    release = threading.Event()

    def pause(observed: str) -> None:
        if observed != boundary:
            return
        entered.set()
        release.wait(5)

    monkeypatch.setattr(bundle_service, "_test_boundary", pause)
    exporting = asyncio.create_task(
        service.export(
            PROFILE_ID,
            destination,
            expected_generation=repository.generation,
            expected_revision=profile.revision,
            acknowledged=True,
        )
    )
    assert await asyncio.to_thread(entered.wait, 5)
    exporting.cancel()
    await asyncio.sleep(0)
    release.set()

    if boundary != "destination_post_link":
        with pytest.raises(asyncio.CancelledError):
            await exporting
        assert not destination.exists()
        retained = tuple(tmp_path.glob(".portable.tldw-voice.zip.*.tmp"))
        assert len(retained) == 1
        assert stat.S_IMODE(retained[0].stat().st_mode) == 0o600
    else:
        await exporting
        assert destination.read_bytes() == encode_clone_voice_bundle(bundle)
        assert not tuple(tmp_path.glob(".portable.tldw-voice.zip.*.tmp"))
    await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "seam",
    (
        "source_hook",
        "commit_repository",
        "export_repository",
        "codec",
        "destination_hook",
    ),
)
async def test_public_operations_sever_all_collaborator_exception_graphs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seam: str,
) -> None:
    service, repository, _dependency_service = _service(tmp_path)
    source = tmp_path / "selected.tldw-voice.zip"
    _write_source(source)
    canary = f"CANARY-{seam}-private"

    if seam == "source_hook":

        def fail_source(_boundary: str) -> None:
            raise RuntimeError(canary)

        monkeypatch.setattr(bundle_service, "_test_boundary", fail_source)
        operation = service.inspect(source)
    elif seam == "commit_repository":
        review = await service.inspect(source)

        async def fail_commit(_command):
            raise RuntimeError(canary)

        repository.commit_bundle_import = fail_commit
        operation = service.commit(
            review.handle, TTSVoiceBundleImportChoice("create", False)
        )
    else:
        _bundle_value, profile = _install_export_record(repository)
        if seam == "export_repository":

            async def fail_profile(_profile_id):
                raise RuntimeError(canary)

            repository.get_profile = fail_profile
        elif seam == "codec":

            def fail_codec(_bundle_value):
                raise RuntimeError(canary)

            monkeypatch.setattr(bundle_service, "encode_clone_voice_bundle", fail_codec)
        else:

            def fail_destination(boundary: str) -> None:
                if boundary == "destination_pre_publish":
                    raise RuntimeError(canary)

            monkeypatch.setattr(bundle_service, "_test_boundary", fail_destination)
        operation = service.export(
            PROFILE_ID,
            tmp_path / "exported.tldw-voice.zip",
            expected_generation=repository.generation,
            expected_revision=profile.revision,
            acknowledged=True,
        )

    with pytest.raises(TTSVoiceBundleError, match="operation_failed") as caught:
        await operation
    error = caught.value
    assert error.__cause__ is None
    assert error.__context__ is None
    assert all(canary not in repr(value) for value in _exception_graph_values(error))
    await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation_name", "signal_type"),
    (
        ("inspect_dependency", asyncio.CancelledError),
        ("inspect_dependency", _PrivateBaseException),
        ("inspect_dependency", KeyboardInterrupt),
        ("inspect_dependency", SystemExit),
        ("commit_repository", _PrivateBaseException),
        ("export_repository", _PrivateBaseException),
        ("export_codec", _PrivateBaseException),
        ("export_hook", _PrivateBaseException),
        ("invalidate", _PrivateBaseException),
        ("close", _PrivateBaseException),
    ),
)
async def test_public_operations_classify_sever_and_reconstruct_baseexceptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation_name: str,
    signal_type: type[BaseException],
) -> None:
    service, repository, dependency = _service(tmp_path)
    source = tmp_path / "CANARY-selected.tldw-voice.zip"
    _write_source(source)
    canary = f"CANARY-{operation_name}-private"

    def signal() -> BaseException:
        return signal_type(canary)

    if operation_name == "inspect_dependency":

        async def fail_dependency(_requirement):
            raise signal()

        dependency.audio_cpp_guided_dependency_snapshot = fail_dependency
        operation = service.inspect(source)
    elif operation_name in {"commit_repository", "invalidate"}:
        review = await service.inspect(source)
        if operation_name == "commit_repository":

            async def fail_commit(_command):
                raise signal()

            repository.commit_bundle_import = fail_commit
            operation = service.commit(
                review.handle, TTSVoiceBundleImportChoice("create", False)
            )
        else:

            async def fail_invalidate(_handle):
                raise signal()

            monkeypatch.setattr(service, "_invalidate_impl", fail_invalidate)
            operation = service.invalidate(review.handle)
    elif operation_name == "close":

        async def fail_close():
            raise signal()

        monkeypatch.setattr(service, "_complete_close", fail_close)
        operation = service.close()
    else:
        _bundle_value, profile = _install_export_record(repository)
        if operation_name == "export_repository":

            async def fail_profile(_profile_id):
                raise signal()

            repository.get_profile = fail_profile
        elif operation_name == "export_codec":

            def fail_codec(_bundle_value):
                raise signal()

            monkeypatch.setattr(bundle_service, "encode_clone_voice_bundle", fail_codec)
        else:

            def fail_hook(boundary: str) -> None:
                if boundary == "destination_pre_publish":
                    raise signal()

            monkeypatch.setattr(bundle_service, "_test_boundary", fail_hook)
        operation = service.export(
            PROFILE_ID,
            tmp_path / "CANARY-destination.tldw-voice.zip",
            expected_generation=repository.generation,
            expected_revision=profile.revision,
            acknowledged=True,
        )

    expected = (
        signal_type
        if signal_type in {asyncio.CancelledError, KeyboardInterrupt, SystemExit}
        else TTSVoiceBundleError
    )
    with pytest.raises(expected) as caught:
        await operation
    error = caught.value
    if type(error) is TTSVoiceBundleError:
        assert str(error) == "operation_failed"
    else:
        assert error.args == ()
    _assert_bounded_public_error(error, canary)
    if not service._closed:
        await service.close()


@pytest.mark.asyncio
async def test_close_atomically_seals_every_public_admission(
    tmp_path: Path,
) -> None:
    service, repository, _dependency_service = _service(tmp_path)
    source = tmp_path / "selected.tldw-voice.zip"
    _write_source(source)
    review = await service.inspect(source)
    _bundle_value, profile = _install_export_record(repository)
    await service.close()

    operations = (
        service.inspect(source),
        service.commit(review.handle, TTSVoiceBundleImportChoice("create", False)),
        service.export(
            PROFILE_ID,
            tmp_path / "after-close.tldw-voice.zip",
            expected_generation=repository.generation,
            expected_revision=profile.revision,
            acknowledged=True,
        ),
        service.invalidate(review.handle),
    )
    for operation in operations:
        with pytest.raises(TTSVoiceBundleError, match="operation_failed"):
            await operation
    assert repository.commits == []


@pytest.mark.asyncio
async def test_export_preserves_unsupported_platform_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, repository, _dependency_service = _service(tmp_path)
    _bundle_value, profile = _install_export_record(repository)
    monkeypatch.setattr(bundle_service, "_posix_supported", lambda: False)
    with pytest.raises(TTSVoiceBundleError, match="unsupported_platform"):
        await service.export(
            PROFILE_ID,
            tmp_path / "unsupported.tldw-voice.zip",
            expected_generation=repository.generation,
            expected_revision=profile.revision,
            acknowledged=True,
        )
    await service.close()


@pytest.mark.asyncio
async def test_dependency_refresh_mutation_is_detected_before_commit_hook(
    tmp_path: Path,
) -> None:
    source = tmp_path / "selected.tldw-voice.zip"
    _write_source(source)
    service, repository, dependency_service = _service(tmp_path)
    review = await service.inspect(source)
    replacement = encode_clone_voice_bundle(_bundle(sample=1))
    original_mtime = source.stat().st_mtime_ns
    original = dependency_service.audio_cpp_guided_dependency_snapshot

    async def mutate_then_snapshot(requirement):
        source.write_bytes(replacement)
        source.chmod(0o600)
        os.utime(source, ns=(original_mtime, original_mtime))
        return await original(requirement)

    dependency_service.audio_cpp_guided_dependency_snapshot = mutate_then_snapshot
    with pytest.raises(TTSVoiceBundleError, match="source_changed"):
        await service.commit(review.handle, TTSVoiceBundleImportChoice("create", False))
    assert repository.commits == []
    await service.close()


@pytest.mark.asyncio
async def test_export_destination_parent_substitution_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "selected-parent"
    parent.mkdir()
    service, repository, _dependency_service = _service(tmp_path / "service")
    _bundle_value, profile = _install_export_record(repository)
    destination = parent / "portable.tldw-voice.zip"
    moved = tmp_path / "moved-parent"

    def substitute(boundary: str) -> None:
        if boundary != "destination_pre_publish":
            return
        os.rename(parent, moved)
        parent.mkdir()
        (parent / "foreign-marker").write_bytes(b"foreign")

    monkeypatch.setattr(bundle_service, "_test_boundary", substitute)
    with pytest.raises(TTSVoiceBundleError, match="destination_changed"):
        await service.export(
            PROFILE_ID,
            destination,
            expected_generation=repository.generation,
            expected_revision=profile.revision,
            acknowledged=True,
        )
    assert not destination.exists()
    assert (parent / "foreign-marker").read_bytes() == b"foreign"
    retained = tuple(moved.glob(".portable.tldw-voice.zip.*.tmp"))
    assert len(retained) == 1
    assert stat.S_IMODE(retained[0].stat().st_mode) == 0o600
    await service.close()
