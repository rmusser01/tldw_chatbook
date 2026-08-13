"""Atomic repository contracts for private clone-reference payloads."""

from __future__ import annotations

import asyncio
import sqlite3
import struct
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest

import tldw_chatbook.TTS.profile_repository as profile_repository
import tldw_chatbook.TTS.profile_reference_storage as reference_storage
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
from tldw_chatbook.TTS.profile_reference_types import (
    CanonicalTTSCloneReference,
    TTSCloneRecipeRequirement,
)
from tldw_chatbook.TTS.profile_types import CharacterRef, TTSProfileDraft


PROFILE_A = UUID("20000000-0000-4000-8000-000000000001")
PROFILE_B = UUID("20000000-0000-4000-8000-000000000002")
REFERENCE_A = UUID("30000000-0000-4000-8000-000000000001")
REFERENCE_B = UUID("30000000-0000-4000-8000-000000000002")
REFERENCE_C = UUID("30000000-0000-4000-8000-000000000003")
NOW = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)


class _ControlFlow(BaseException):
    """Test-only operation-lane control flow."""


def _canonical(*, sample: int = 1, frames: int = 32) -> CanonicalTTSCloneReference:
    channels = 1
    sample_rate_hz = 16_000
    pcm = struct.pack("<h", sample) * frames
    byte_rate = sample_rate_hz * channels * 2
    fmt = struct.pack("<HHIIHH", 1, channels, sample_rate_hz, byte_rate, 2, 16)
    body = (
        b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(pcm))
        + pcm
    )
    wav = b"RIFF" + struct.pack("<I", len(body)) + body
    return CanonicalTTSCloneReference(
        wav_bytes=wav,
        reference_text=f"Reference {sample}",
        sha256=sha256(wav).hexdigest(),
        byte_length=len(wav),
        duration_ms=(frames * 1_000 + sample_rate_hz - 1) // sample_rate_hz,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        sample_encoding="pcm_s16le",
    )


def _draft(name: str) -> TTSProfileDraft:
    return TTSProfileDraft(
        display_name=name,
        provider_id="openai",
        model_id="tts-1",
        voice_id="alloy",
        response_format="mp3",
        speed=1.0,
        options=cast(Any, {}),
    )


def _audio_cpp_draft(name: str, *, model_id: str = "clone-model") -> TTSProfileDraft:
    return TTSProfileDraft(
        display_name=name,
        provider_id="audio_cpp",
        model_id=model_id,
        voice_id=None,
        response_format="wav",
        speed=1.0,
        options={},
    )


def _hostile_audio_cpp_draft(
    name: str, field_name: str, value: object
) -> TTSProfileDraft:
    """Bypass the frozen value object's constructor to probe command validation."""

    draft = _audio_cpp_draft(name)
    object.__setattr__(draft, field_name, value)
    return draft


def _requirement(*, model_id: str = "clone-model") -> TTSCloneRecipeRequirement:
    return TTSCloneRecipeRequirement(
        recipe_id="audio-cpp-0.5.1.pocket_tts.pocket_tts",
        recipe_revision=1,
        model_id=model_id,
    )


def _requirements_for_tests(*, model_id: str) -> TTSCloneRecipeRequirement:
    """Supply exact v4 provenance to legacy repository contract fixtures."""

    return _requirement(model_id=model_id)


class _UUIDSequence:
    def __init__(self, values: Iterator[UUID]) -> None:
        self._values = values

    def __call__(self) -> UUID:
        return next(self._values)


@asynccontextmanager
async def _opened_repository(
    path: Path,
    *,
    reference_ids: tuple[UUID, ...] = (REFERENCE_A, REFERENCE_B, REFERENCE_C),
) -> AsyncIterator[profile_repository.TTSProfileRepository]:
    repository = profile_repository.TTSProfileRepository(
        path,
        _clock=lambda: NOW,
        _uuid_factory=_UUIDSequence(iter(reference_ids)),
    )
    await repository.open()
    try:
        yield repository
    finally:
        await repository.close()


async def _create(
    repository: profile_repository.TTSProfileRepository,
    profile_id: UUID,
    name: str,
) -> tuple[int, int]:
    created = await repository.create_profile(_draft(name), profile_id)
    return created.generation, created.value.revision


@pytest.mark.asyncio
async def test_create_profile_with_reference_commits_one_revision_two_profile(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    canonical = _canonical(sample=7)
    async with _opened_repository(path) as repository:
        created = await repository.create_profile_with_reference(
            _draft("Clone voice"),
            PROFILE_A,
            canonical,
            _requirements_for_tests(model_id="tts-1"),
            expected_generation=repository.generation,
        )

        assert created.generation == repository.generation
        assert created.value.profile_id == PROFILE_A
        assert created.value.revision == 2
        assert created.value.reference is not None
        assert created.value.reference.reference_id == REFERENCE_A
        exact = await repository.get_reference(
            PROFILE_A,
            expected_revision=2,
            expected_generation=created.generation,
        )
        assert exact.value.wav_bytes == canonical.wav_bytes
        assert exact.value.reference_text == canonical.reference_text


@pytest.mark.asyncio
async def test_new_reference_write_persists_exact_recipe_requirement(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    requirement = _requirement()
    async with _opened_repository(path) as repository:
        created = await repository.create_profile_with_reference(
            _audio_cpp_draft("Provenance voice"),
            PROFILE_A,
            _canonical(),
            requirement,
            expected_generation=repository.generation,
        )

        assert created.value.reference is not None
        assert created.value.reference.recipe_requirement == requirement
        exact = await repository.get_reference(
            PROFILE_A,
            expected_revision=created.value.revision,
            expected_generation=created.generation,
        )
        assert exact.value.recipe_requirement == requirement


@pytest.mark.asyncio
async def test_new_reference_write_rejects_model_incoherent_provenance_before_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    async with _opened_repository(path) as repository:
        queued = False

        async def forbidden_queue(*_args: object, **_kwargs: object) -> object:
            nonlocal queued
            queued = True
            raise AssertionError("repository work was queued")

        monkeypatch.setattr(repository, "_submit_operation", forbidden_queue)
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.create_profile_with_reference(
                _audio_cpp_draft("Wrong model"),
                PROFILE_A,
                _canonical(),
                _requirement(model_id="other-model"),
                expected_generation=repository.generation,
            )

        _assert_error(caught.value, "operation_failed")
        assert queued is False


@pytest.mark.asyncio
async def test_set_reference_replaces_recipe_requirement_atomically(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    requirement = _requirement()
    async with _opened_repository(path) as repository:
        created = await repository.create_profile(
            _audio_cpp_draft("Attach provenance"),
            PROFILE_A,
        )
        attached = await repository.set_reference(
            PROFILE_A,
            _canonical(sample=9),
            requirement,
            expected_revision=created.value.revision,
            expected_generation=created.generation,
        )

        assert attached.value.reference is not None
        assert attached.value.reference.recipe_requirement == requirement


@pytest.mark.asyncio
async def test_reference_profile_rejects_generation_field_edit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    async with _opened_repository(path) as repository:
        created = await repository.create_profile_with_reference(
            _audio_cpp_draft("Immutable clone"),
            PROFILE_A,
            _canonical(),
            _requirement(),
            expected_generation=repository.generation,
        )
        changed_selection = TTSProfileDraft(
            display_name="Immutable clone",
            provider_id="audio_cpp",
            model_id="clone-model",
            voice_id="other-voice",
            response_format="wav",
            speed=1.0,
            options={},
        )

        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.update_profile(
                PROFILE_A,
                created.value.revision,
                changed_selection,
                expected_generation=created.generation,
            )

        _assert_error(caught.value, "conflict")
        stored = await repository.get_profile(PROFILE_A)
        assert stored.value == created.value


@pytest.mark.asyncio
async def test_reference_profile_allows_display_name_only_edit(tmp_path: Path) -> None:
    path = tmp_path / "profiles.sqlite3"
    async with _opened_repository(path) as repository:
        created = await repository.create_profile_with_reference(
            _audio_cpp_draft("Original clone"),
            PROFILE_A,
            _canonical(),
            _requirement(),
            expected_generation=repository.generation,
        )
        renamed = await repository.update_profile(
            PROFILE_A,
            created.value.revision,
            _audio_cpp_draft("Renamed clone"),
            expected_generation=created.generation,
        )

        assert renamed.value.display_name == "Renamed clone"
        assert renamed.value.reference == created.value.reference


@pytest.mark.asyncio
async def test_migrated_null_provenance_keeps_display_only_edit_boundary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    async with _opened_repository(path) as repository:
        created = await repository.create_profile_with_reference(
            _audio_cpp_draft("Legacy clone"),
            PROFILE_A,
            _canonical(),
            _requirement(),
            expected_generation=repository.generation,
        )

    connection = sqlite3.connect(path)
    connection.execute(
        f"UPDATE {reference_storage.REFERENCE_TABLE} "
        "SET recipe_id = NULL, recipe_revision = NULL"
    )
    connection.commit()
    connection.close()

    async with _opened_repository(path) as repository:
        legacy = await repository.get_profile(PROFILE_A)
        assert legacy.value.reference is not None
        assert legacy.value.reference.recipe_requirement is None
        renamed = await repository.update_profile(
            PROFILE_A,
            created.value.revision,
            _audio_cpp_draft("Legacy renamed"),
            expected_generation=repository.generation,
        )
        assert renamed.value.display_name == "Legacy renamed"

        changed_voice = TTSProfileDraft(
            display_name="Legacy renamed",
            provider_id="audio_cpp",
            model_id="clone-model",
            voice_id="other-voice",
            response_format="wav",
            speed=1.0,
            options={},
        )
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.update_profile(
                PROFILE_A,
                renamed.value.revision,
                changed_voice,
                expected_generation=repository.generation,
            )
        _assert_error(caught.value, "conflict")


@pytest.mark.asyncio
async def test_bundle_import_create_commits_profile_recipe_reference_atomically(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    async with _opened_repository(path) as repository:
        command = profile_repository.TTSBundleImportCommand(
            choice="create",
            source_profile_id=PROFILE_A,
            source_draft=_audio_cpp_draft("Imported clone"),
            recipe_requirement=_requirement(),
            canonical_reference=_canonical(),
            expected_generation=repository.generation,
            reviewed_source_collisions=profile_repository.TTSProfileCollisionSnapshot(
                None,
                None,
            ),
            copy_profile_id=None,
            copy_display_name=None,
            dependency_state="exact",
            inactive_consent=False,
        )

        result = await repository.commit_bundle_import(command)

        assert result.value.kind == "created"
        assert result.value.profile is not None
        assert result.value.profile.profile_id == PROFILE_A
        assert result.value.profile.reference is not None
        assert result.value.profile.reference.recipe_requirement == _requirement()
        assert (await repository.assignment_count(PROFILE_A)).value == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source_draft",
    (
        TTSProfileDraft(
            display_name="Wrong provider",
            provider_id="openai",
            model_id="clone-model",
            voice_id="alloy",
            response_format="mp3",
            speed=1.0,
            options={},
        ),
        _hostile_audio_cpp_draft(
            "Wrong format",
            "response_format",
            "mp3",
        ),
        _hostile_audio_cpp_draft(
            "Wrong speed",
            "speed",
            0.9,
        ),
        _hostile_audio_cpp_draft(
            "Wrong options",
            "options",
            {"language": "en"},
        ),
    ),
    ids=("provider", "format", "speed", "options"),
)
async def test_bundle_import_rejects_non_clone_profile_contract_before_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_draft: TTSProfileDraft,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    async with _opened_repository(path) as repository:
        queued = False

        async def forbidden_queue(*_args: object, **_kwargs: object) -> object:
            nonlocal queued
            queued = True
            raise AssertionError("repository work was queued")

        monkeypatch.setattr(repository, "_submit_operation", forbidden_queue)
        command = profile_repository.TTSBundleImportCommand(
            choice="create",
            source_profile_id=PROFILE_A,
            source_draft=source_draft,
            recipe_requirement=_requirement(),
            canonical_reference=_canonical(),
            expected_generation=repository.generation,
            reviewed_source_collisions=profile_repository.TTSProfileCollisionSnapshot(
                None,
                None,
            ),
            copy_profile_id=None,
            copy_display_name=None,
            dependency_state="exact",
            inactive_consent=False,
        )

        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.commit_bundle_import(command)

        _assert_error(caught.value, "operation_failed")
        assert queued is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("choice", "dependency_state", "inactive_consent", "allowed"),
    (
        ("create", "exact", False, True),
        ("create", "exact", True, False),
        ("create", "missing", False, False),
        ("create", "missing", True, True),
        ("reuse", "exact", False, True),
        ("reuse", "exact", True, False),
        ("reuse", "missing", False, True),
        ("reuse", "missing", True, False),
        ("copy", "exact", False, True),
        ("copy", "exact", True, False),
        ("copy", "missing", False, False),
        ("copy", "missing", True, True),
    ),
)
async def test_bundle_import_enforces_exact_inactive_consent_matrix_before_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    choice: str,
    dependency_state: str,
    inactive_consent: bool,
    allowed: bool,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    async with _opened_repository(path) as repository:
        queued = False
        marker = object()

        async def observe_queue(*_args: object, **_kwargs: object) -> object:
            nonlocal queued
            queued = True
            return marker

        monkeypatch.setattr(repository, "_submit_operation", observe_queue)
        is_copy = choice == "copy"
        command = profile_repository.TTSBundleImportCommand(
            choice=cast(Any, choice),
            source_profile_id=PROFILE_A,
            source_draft=_audio_cpp_draft("Consent matrix"),
            recipe_requirement=_requirement(),
            canonical_reference=_canonical(),
            expected_generation=repository.generation,
            reviewed_source_collisions=profile_repository.TTSProfileCollisionSnapshot(
                None,
                None,
            ),
            copy_profile_id=PROFILE_B if is_copy else None,
            copy_display_name="Consent matrix copy" if is_copy else None,
            dependency_state=cast(Any, dependency_state),
            inactive_consent=inactive_consent,
        )

        if allowed:
            assert await repository.commit_bundle_import(command) is marker
            assert queued is True
        else:
            with pytest.raises(ProfileRepositoryError) as caught:
                await repository.commit_bundle_import(command)
            _assert_error(caught.value, "operation_failed")
            assert queued is False


@pytest.mark.asyncio
async def test_bundle_import_normalizes_mutated_copy_name_before_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    async with _opened_repository(path) as repository:
        queued = False

        async def forbidden_queue(*_args: object, **_kwargs: object) -> object:
            nonlocal queued
            queued = True
            raise AssertionError("repository work was queued")

        monkeypatch.setattr(repository, "_submit_operation", forbidden_queue)
        command = profile_repository.TTSBundleImportCommand(
            choice="copy",
            source_profile_id=PROFILE_A,
            source_draft=_audio_cpp_draft("Copy source"),
            recipe_requirement=_requirement(),
            canonical_reference=_canonical(),
            expected_generation=repository.generation,
            reviewed_source_collisions=profile_repository.TTSProfileCollisionSnapshot(
                None,
                None,
            ),
            copy_profile_id=PROFILE_B,
            copy_display_name="Valid copy",
            dependency_state="exact",
            inactive_consent=False,
        )
        private_value = "\n/private/voice/source"
        object.__setattr__(command, "copy_display_name", private_value)

        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.commit_bundle_import(command)

        _assert_error(caught.value, "operation_failed")
        assert private_value not in str(caught.value)
        assert private_value not in repr(caught.value)
        assert queued is False


@pytest.mark.asyncio
async def test_bundle_import_copy_validation_preserves_base_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    async with _opened_repository(path) as repository:
        command = profile_repository.TTSBundleImportCommand(
            choice="copy",
            source_profile_id=PROFILE_A,
            source_draft=_audio_cpp_draft("Copy source"),
            recipe_requirement=_requirement(),
            canonical_reference=_canonical(),
            expected_generation=repository.generation,
            reviewed_source_collisions=profile_repository.TTSProfileCollisionSnapshot(
                None,
                None,
            ),
            copy_profile_id=PROFILE_B,
            copy_display_name="Valid copy",
            dependency_state="exact",
            inactive_consent=False,
        )

        def interrupt(**_kwargs: object) -> object:
            raise _ControlFlow()

        monkeypatch.setattr(profile_repository, "_validate_draft", lambda value: value)
        monkeypatch.setattr(profile_repository, "TTSProfileDraft", interrupt)

        with pytest.raises(_ControlFlow):
            await repository.commit_bundle_import(command)


@pytest.mark.asyncio
async def test_bundle_import_result_enforces_exact_sum_type(tmp_path: Path) -> None:
    path = tmp_path / "profiles.sqlite3"
    async with _opened_repository(path) as repository:
        created = await repository.create_profile(
            _audio_cpp_draft("Result profile"),
            PROFILE_A,
        )
        profile = created.value
        facts = profile_repository.TTSBundleImportRepositoryFacts(
            source_collisions=profile_repository.TTSProfileCollisionSnapshot(
                None,
                None,
            ),
            copy_collisions=None,
        )

        assert profile_repository.TTSBundleImportResult("created", profile).profile
        assert profile_repository.TTSBundleImportResult("reused", profile).profile
        assert (
            profile_repository.TTSBundleImportResult(
                "stale_inspection",
                None,
                facts,
            ).repository_facts
            is facts
        )

        invalid = (
            ("created", None, None),
            ("created", profile, facts),
            ("reused", None, None),
            ("reused", profile, facts),
            ("stale_inspection", profile, facts),
            ("stale_inspection", None, None),
            ("unknown", profile, None),
        )
        for kind, candidate_profile, candidate_facts in invalid:
            with pytest.raises(ProfileRepositoryError) as caught:
                profile_repository.TTSBundleImportResult(
                    cast(Any, kind),
                    candidate_profile,
                    candidate_facts,
                )
            _assert_error(caught.value, "operation_failed")


@pytest.mark.asyncio
@pytest.mark.parametrize("dependency_state", ("exact", "missing"))
async def test_bundle_import_reuses_only_exact_public_private_profile(
    tmp_path: Path,
    dependency_state: str,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    draft = _audio_cpp_draft("Existing clone")
    canonical = _canonical(sample=4)
    requirement = _requirement()
    async with _opened_repository(path) as repository:
        existing = await repository.create_profile_with_reference(
            draft,
            PROFILE_A,
            canonical,
            requirement,
            expected_generation=repository.generation,
        )
        collisions = await repository.get_profile_collisions(PROFILE_A, draft)
        command = profile_repository.TTSBundleImportCommand(
            choice="reuse",
            source_profile_id=PROFILE_A,
            source_draft=draft,
            recipe_requirement=requirement,
            canonical_reference=canonical,
            expected_generation=repository.generation,
            reviewed_source_collisions=collisions.value,
            copy_profile_id=None,
            copy_display_name=None,
            dependency_state=cast(Any, dependency_state),
            inactive_consent=False,
        )

        result = await repository.commit_bundle_import(command)

        assert result.value.kind == "reused"
        assert result.value.profile == existing.value
        assert (await repository.list_profiles()).value.total == 1
        assert (await repository.assignment_count(PROFILE_A)).value == 0


@pytest.mark.asyncio
async def test_bundle_import_copy_uses_reviewed_collision_free_destination(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    source_draft = _audio_cpp_draft("Colliding clone")
    async with _opened_repository(path) as repository:
        await repository.create_profile_with_reference(
            source_draft,
            PROFILE_A,
            _canonical(sample=1),
            _requirement(),
            expected_generation=repository.generation,
        )
        collisions = await repository.get_profile_collisions(PROFILE_A, source_draft)
        command = profile_repository.TTSBundleImportCommand(
            choice="copy",
            source_profile_id=PROFILE_A,
            source_draft=source_draft,
            recipe_requirement=_requirement(),
            canonical_reference=_canonical(sample=8),
            expected_generation=repository.generation,
            reviewed_source_collisions=collisions.value,
            copy_profile_id=PROFILE_B,
            copy_display_name="Colliding clone (imported)",
            dependency_state="exact",
            inactive_consent=False,
        )

        result = await repository.commit_bundle_import(command)

        assert result.value.kind == "created"
        assert result.value.profile is not None
        assert result.value.profile.profile_id == PROFILE_B
        assert result.value.profile.display_name == "Colliding clone (imported)"
        assert (await repository.list_profiles()).value.total == 2
        assert (await repository.assignment_count(PROFILE_B)).value == 0


@pytest.mark.asyncio
async def test_bundle_import_returns_refreshed_repository_facts_when_stale(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    draft = _audio_cpp_draft("Stale import")
    async with _opened_repository(path) as repository:
        reviewed = await repository.get_profile_collisions(PROFILE_A, draft)
        await repository.create_profile(draft, PROFILE_A)
        command = profile_repository.TTSBundleImportCommand(
            choice="create",
            source_profile_id=PROFILE_A,
            source_draft=draft,
            recipe_requirement=_requirement(),
            canonical_reference=_canonical(),
            expected_generation=repository.generation,
            reviewed_source_collisions=reviewed.value,
            copy_profile_id=None,
            copy_display_name=None,
            dependency_state="exact",
            inactive_consent=False,
        )

        result = await repository.commit_bundle_import(command)

        assert result.value.kind == "stale_inspection"
        assert result.value.profile is None
        assert result.value.repository_facts is not None
        assert (
            result.value.repository_facts.source_collisions.profile_id_match is not None
        )
        assert (await repository.list_profiles()).value.total == 1


@pytest.mark.asyncio
async def test_bundle_import_rechecks_copy_destination_inside_transaction(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    source_draft = _audio_cpp_draft("Source collision")
    copy_draft = _audio_cpp_draft("Reviewed copy")
    async with _opened_repository(path) as repository:
        await repository.create_profile(source_draft, PROFILE_A)
        reviewed = await repository.get_profile_collisions(PROFILE_A, source_draft)
        await repository.create_profile(copy_draft, PROFILE_B)
        command = profile_repository.TTSBundleImportCommand(
            choice="copy",
            source_profile_id=PROFILE_A,
            source_draft=source_draft,
            recipe_requirement=_requirement(),
            canonical_reference=_canonical(),
            expected_generation=repository.generation,
            reviewed_source_collisions=reviewed.value,
            copy_profile_id=PROFILE_B,
            copy_display_name=copy_draft.display_name,
            dependency_state="exact",
            inactive_consent=False,
        )

        result = await repository.commit_bundle_import(command)

        assert result.value.kind == "stale_inspection"
        assert result.value.repository_facts is not None
        copy_facts = result.value.repository_facts.copy_collisions
        assert copy_facts is not None
        assert copy_facts.profile_id_match is not None
        assert (await repository.list_profiles()).value.total == 2


@pytest.mark.asyncio
async def test_bundle_import_collision_recheck_holds_immediate_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    async with _opened_repository(path) as repository:
        real_read = repository._worker_read_profile_collisions
        escaped_transaction = False

        def probe_lock(*args: object, **kwargs: object) -> object:
            nonlocal escaped_transaction
            competitor = sqlite3.connect(path, timeout=0)
            try:
                competitor.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError:
                pass
            else:
                escaped_transaction = True
                competitor.rollback()
            finally:
                competitor.close()
            return real_read(*args, **kwargs)

        monkeypatch.setattr(repository, "_worker_read_profile_collisions", probe_lock)
        command = profile_repository.TTSBundleImportCommand(
            choice="create",
            source_profile_id=PROFILE_A,
            source_draft=_audio_cpp_draft("Serialized import"),
            recipe_requirement=_requirement(),
            canonical_reference=_canonical(),
            expected_generation=repository.generation,
            reviewed_source_collisions=profile_repository.TTSProfileCollisionSnapshot(
                None,
                None,
            ),
            copy_profile_id=None,
            copy_display_name=None,
            dependency_state="exact",
            inactive_consent=False,
        )

        result = await repository.commit_bundle_import(command)

        assert result.value.kind == "created"
        assert escaped_transaction is False


@pytest.mark.asyncio
async def test_bundle_import_missing_dependency_requires_inactive_consent_before_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    async with _opened_repository(path) as repository:
        queued = False

        async def forbidden_queue(*_args: object, **_kwargs: object) -> object:
            nonlocal queued
            queued = True
            raise AssertionError("repository work was queued")

        monkeypatch.setattr(repository, "_submit_operation", forbidden_queue)
        command = profile_repository.TTSBundleImportCommand(
            choice="create",
            source_profile_id=PROFILE_A,
            source_draft=_audio_cpp_draft("Inactive import"),
            recipe_requirement=_requirement(),
            canonical_reference=_canonical(),
            expected_generation=repository.generation,
            reviewed_source_collisions=profile_repository.TTSProfileCollisionSnapshot(
                None,
                None,
            ),
            copy_profile_id=None,
            copy_display_name=None,
            dependency_state="missing",
            inactive_consent=False,
        )

        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.commit_bundle_import(command)

        _assert_error(caught.value, "operation_failed")
        assert queued is False


@pytest.mark.asyncio
async def test_bundle_import_missing_dependency_with_consent_creates_unassigned(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    async with _opened_repository(path) as repository:
        command = profile_repository.TTSBundleImportCommand(
            choice="create",
            source_profile_id=PROFILE_A,
            source_draft=_audio_cpp_draft("Inactive import"),
            recipe_requirement=_requirement(),
            canonical_reference=_canonical(),
            expected_generation=repository.generation,
            reviewed_source_collisions=profile_repository.TTSProfileCollisionSnapshot(
                None,
                None,
            ),
            copy_profile_id=None,
            copy_display_name=None,
            dependency_state="missing",
            inactive_consent=True,
        )

        result = await repository.commit_bundle_import(command)

        assert result.value.kind == "created"
        assert result.value.profile is not None
        assert (await repository.assignment_count(PROFILE_A)).value == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("inactive_consent", (False, True))
async def test_bundle_import_copy_missing_dependency_requires_separate_consent(
    tmp_path: Path,
    inactive_consent: bool,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    source_draft = _audio_cpp_draft("Source conflict")
    async with _opened_repository(path) as repository:
        await repository.create_profile(source_draft, PROFILE_A)
        reviewed = await repository.get_profile_collisions(PROFILE_A, source_draft)
        command = profile_repository.TTSBundleImportCommand(
            choice="copy",
            source_profile_id=PROFILE_A,
            source_draft=source_draft,
            recipe_requirement=_requirement(),
            canonical_reference=_canonical(),
            expected_generation=repository.generation,
            reviewed_source_collisions=reviewed.value,
            copy_profile_id=PROFILE_B,
            copy_display_name="Source conflict (imported)",
            dependency_state="missing",
            inactive_consent=inactive_consent,
        )

        if not inactive_consent:
            with pytest.raises(ProfileRepositoryError) as caught:
                await repository.commit_bundle_import(command)
            _assert_error(caught.value, "operation_failed")
            assert (await repository.list_profiles()).value.total == 1
            return

        result = await repository.commit_bundle_import(command)
        assert result.value.kind == "created"
        assert result.value.profile is not None
        assert result.value.profile.profile_id == PROFILE_B
        assert (await repository.assignment_count(PROFILE_B)).value == 0


@pytest.mark.asyncio
async def test_bundle_import_reuse_rejects_private_or_unknown_provenance_difference(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    draft = _audio_cpp_draft("Existing clone")
    async with _opened_repository(path) as repository:
        await repository.create_profile_with_reference(
            draft,
            PROFILE_A,
            _canonical(sample=1),
            _requirement(),
            expected_generation=repository.generation,
        )
        collisions = await repository.get_profile_collisions(PROFILE_A, draft)
        different = profile_repository.TTSBundleImportCommand(
            choice="reuse",
            source_profile_id=PROFILE_A,
            source_draft=draft,
            recipe_requirement=_requirement(),
            canonical_reference=_canonical(sample=2),
            expected_generation=repository.generation,
            reviewed_source_collisions=collisions.value,
            copy_profile_id=None,
            copy_display_name=None,
            dependency_state="exact",
            inactive_consent=False,
        )
        result = await repository.commit_bundle_import(different)
        assert result.value.kind == "stale_inspection"

    connection = sqlite3.connect(path)
    connection.execute(
        f"UPDATE {reference_storage.REFERENCE_TABLE} "
        "SET recipe_id = NULL, recipe_revision = NULL"
    )
    connection.commit()
    connection.close()

    async with _opened_repository(path) as repository:
        legacy_collisions = await repository.get_profile_collisions(PROFILE_A, draft)
        legacy = profile_repository.TTSBundleImportCommand(
            choice="reuse",
            source_profile_id=PROFILE_A,
            source_draft=draft,
            recipe_requirement=_requirement(),
            canonical_reference=_canonical(sample=1),
            expected_generation=repository.generation,
            reviewed_source_collisions=legacy_collisions.value,
            copy_profile_id=None,
            copy_display_name=None,
            dependency_state="exact",
            inactive_consent=False,
        )
        result = await repository.commit_bundle_import(legacy)
        assert result.value.kind == "stale_inspection"


@pytest.mark.asyncio
async def test_bundle_import_failure_between_profile_and_reference_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    canary = "CANARY-private-transcript-checksum-staging-path"
    async with _opened_repository(path) as repository:
        real_put = repository._worker_put_reference

        def fail_after_reference(*args: object, **kwargs: object) -> object:
            real_put(*args, **kwargs)
            raise RuntimeError(canary)

        monkeypatch.setattr(repository, "_worker_put_reference", fail_after_reference)
        command = profile_repository.TTSBundleImportCommand(
            choice="create",
            source_profile_id=PROFILE_A,
            source_draft=_audio_cpp_draft("Rollback import"),
            recipe_requirement=_requirement(),
            canonical_reference=_canonical(),
            expected_generation=repository.generation,
            reviewed_source_collisions=profile_repository.TTSProfileCollisionSnapshot(
                None,
                None,
            ),
            copy_profile_id=None,
            copy_display_name=None,
            dependency_state="exact",
            inactive_consent=False,
        )

        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.commit_bundle_import(command)

        _assert_error(caught.value, "operation_failed")
        assert canary not in str(caught.value)
        assert canary not in repr(caught.value)
        assert (await repository.list_profiles()).value.total == 0


@pytest.mark.asyncio
async def test_cancelled_bundle_import_is_retained_and_commits_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    repository = profile_repository.TTSProfileRepository(
        path,
        _clock=lambda: NOW,
        _uuid_factory=_UUIDSequence(iter((REFERENCE_A,))),
    )
    await repository.open()
    entered = threading.Event()
    release = threading.Event()
    real_put = repository._worker_put_reference

    def blocked_put(*args: object, **kwargs: object) -> object:
        entered.set()
        if not release.wait(1.0):
            raise AssertionError("test did not release bundle import")
        return real_put(*args, **kwargs)

    monkeypatch.setattr(repository, "_worker_put_reference", blocked_put)
    command = profile_repository.TTSBundleImportCommand(
        choice="create",
        source_profile_id=PROFILE_A,
        source_draft=_audio_cpp_draft("Cancelled import"),
        recipe_requirement=_requirement(),
        canonical_reference=_canonical(),
        expected_generation=repository.generation,
        reviewed_source_collisions=profile_repository.TTSProfileCollisionSnapshot(
            None,
            None,
        ),
        copy_profile_id=None,
        copy_display_name=None,
        dependency_state="exact",
        inactive_consent=False,
    )
    task = asyncio.create_task(repository.commit_bundle_import(command))
    try:
        assert await asyncio.to_thread(entered.wait, 1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        await repository.close()
    finally:
        release.set()
        if repository.state is not profile_repository.ProfileRepositoryState.CLOSED:
            await repository.close()

    async with _opened_repository(path) as reopened:
        page = await reopened.list_profiles()
        assert page.value.total == 1
        assert page.value.profiles[0].reference is not None
        assert page.value.profiles[0].reference.recipe_requirement == _requirement()


@pytest.mark.asyncio
async def test_create_profile_with_reference_rolls_back_both_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    async with _opened_repository(path) as repository:
        real_put = repository._worker_put_reference

        def fail_after_profile_insert(*args: Any, **kwargs: Any) -> Any:
            real_put(*args, **kwargs)
            raise ProfileRepositoryError("operation_failed")

        monkeypatch.setattr(
            repository,
            "_worker_put_reference",
            fail_after_profile_insert,
        )
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.create_profile_with_reference(
                _draft("Rollback clone"),
                PROFILE_A,
                _canonical(),
                _requirements_for_tests(model_id="tts-1"),
                expected_generation=repository.generation,
            )
        _assert_error(caught.value, "operation_failed")

        page = await repository.list_profiles()
        assert page.value.total == 0
        with pytest.raises(ProfileRepositoryError) as missing:
            await repository.get_profile(PROFILE_A)
        _assert_error(missing.value, "missing")


@pytest.mark.asyncio
async def test_create_profile_with_reference_rejects_stale_generation_before_write(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    async with _opened_repository(path) as repository:
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.create_profile_with_reference(
                _draft("Stale clone"),
                PROFILE_A,
                _canonical(),
                _requirements_for_tests(model_id="tts-1"),
                expected_generation=repository.generation + 1,
            )
        _assert_error(caught.value, "stale")
        assert (await repository.list_profiles()).value.total == 0


@pytest.mark.asyncio
async def test_create_profile_with_reference_enforces_quota_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    monkeypatch.setattr(profile_repository, "MAX_REFERENCE_COUNT", 0)
    async with _opened_repository(path) as repository:
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.create_profile_with_reference(
                _draft("Quota clone"),
                PROFILE_A,
                _canonical(),
                _requirements_for_tests(model_id="tts-1"),
                expected_generation=repository.generation,
            )
        _assert_error(caught.value, "reference_quota")
        assert (await repository.list_profiles()).value.total == 0


@pytest.mark.asyncio
async def test_create_profile_with_reference_collision_commits_no_reference(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    async with _opened_repository(path) as repository:
        await repository.create_profile(_draft("Existing"), PROFILE_A)

        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.create_profile_with_reference(
                _draft(" existing "),
                PROFILE_B,
                _canonical(),
                _requirements_for_tests(model_id="tts-1"),
                expected_generation=repository.generation,
            )
        _assert_error(caught.value, "conflict")
        page = await repository.list_profiles()
        assert page.value.total == 1
        assert page.value.profiles[0].profile_id == PROFILE_A
        assert page.value.profiles[0].reference is None


@pytest.mark.asyncio
async def test_cancelled_create_profile_with_reference_never_leaves_one_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    repository = profile_repository.TTSProfileRepository(
        path,
        _clock=lambda: NOW,
        _uuid_factory=_UUIDSequence(iter((REFERENCE_A,))),
    )
    await repository.open()
    entered = threading.Event()
    release = threading.Event()
    real_put = repository._worker_put_reference

    def blocked_put(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        if not release.wait(1.0):
            raise AssertionError("test did not release reference write")
        return real_put(*args, **kwargs)

    monkeypatch.setattr(repository, "_worker_put_reference", blocked_put)
    task = asyncio.create_task(
        repository.create_profile_with_reference(
            _draft("Cancelled clone"),
            PROFILE_A,
            _canonical(),
            _requirements_for_tests(model_id="tts-1"),
            expected_generation=repository.generation,
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        await repository.close()
    finally:
        release.set()
        if repository.state is not profile_repository.ProfileRepositoryState.CLOSED:
            await repository.close()

    async with _opened_repository(path) as reopened:
        page = await reopened.list_profiles()
        assert page.value.total in (0, 1)
        if page.value.total == 1:
            profile = page.value.profiles[0]
            assert profile.profile_id == PROFILE_A
            assert profile.revision == 2
            assert profile.reference is not None


def _assert_error(error: ProfileRepositoryError, code: str) -> None:
    assert type(error) is ProfileRepositoryError
    assert error.code == code
    assert error.__cause__ is None
    assert error.__context__ is None


def test_reference_qualification_deadline_interrupts_aggregate_sql() -> None:
    progress_callbacks: list[object] = []
    deadline_checks = 0

    class AggregateScanConnection:
        def set_progress_handler(
            self,
            callback: object,
            opcode_interval: int,
        ) -> None:
            progress_callbacks.append((callback, opcode_interval))

        def execute(self, _sql: str) -> object:
            callback, _interval = cast(tuple[Any, int], progress_callbacks[-1])
            assert callable(callback)
            assert callback() == 1
            raise sqlite3.OperationalError("private interrupted query detail")

    def check_deadline() -> None:
        nonlocal deadline_checks
        deadline_checks += 1
        if deadline_checks >= 2:
            raise ProfileRepositoryError("restore_failed")

    with pytest.raises(ProfileRepositoryError) as caught:
        reference_storage.validate_reference_rows(
            cast(sqlite3.Connection, AggregateScanConnection()),
            check_deadline=check_deadline,
        )

    _assert_error(caught.value, "restore_failed")
    assert len(progress_callbacks) == 2
    assert progress_callbacks[0][1] == 1_000  # type: ignore[index]
    assert progress_callbacks[1] == (None, 0)


@pytest.mark.asyncio
async def test_set_reference_rejects_digest_valid_noncanonical_wav(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    pseudo_wav = b"not-a-canonical-wave"
    forged = CanonicalTTSCloneReference(
        wav_bytes=pseudo_wav,
        reference_text="Reference transcript",
        sha256=sha256(pseudo_wav).hexdigest(),
        byte_length=len(pseudo_wav),
        duration_ms=1,
        sample_rate_hz=16_000,
        channels=1,
        sample_encoding="pcm_s16le",
    )
    async with _opened_repository(path) as repository:
        generation, revision = await _create(repository, PROFILE_A, "Narrator")

        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.set_reference(
                PROFILE_A,
                forged,
                _requirements_for_tests(model_id="tts-1"),
                expected_revision=revision,
                expected_generation=generation,
            )
        _assert_error(caught.value, "operation_failed")
        unchanged = await repository.get_profile(PROFILE_A)

        assert unchanged.value.revision == revision
        assert unchanged.value.reference is None


@pytest.mark.asyncio
async def test_attach_stream_read_and_metadata_only_profile_surfaces(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    canonical = _canonical()
    async with _opened_repository(path) as repository:
        generation, revision = await _create(repository, PROFILE_A, "Narrator")
        attached = await repository.set_reference(
            PROFILE_A,
            canonical,
            _requirements_for_tests(model_id="tts-1"),
            expected_revision=revision,
            expected_generation=generation,
        )

        assert attached.value.revision == 2
        assert attached.value.reference is not None
        assert attached.value.reference.reference_id == REFERENCE_A
        exact = await repository.get_reference(
            PROFILE_A,
            expected_revision=2,
            expected_generation=generation,
        )
        assert exact.value.wav_bytes == canonical.wav_bytes
        assert exact.value.reference_text == canonical.reference_text
        assert exact.value.sha256 == canonical.sha256

        loaded = await repository.get_profile(PROFILE_A)
        page = await repository.list_profiles()
        assert loaded.value.reference == attached.value.reference
        assert page.value.profiles[0].reference == attached.value.reference

        character = CharacterRef(
            source="local", authority_id="library", character_id="narrator"
        )
        await repository.set_assignment(
            character,
            PROFILE_A,
            expected_generation=generation,
            expected_profile_revision=2,
            expected_current_profile_id=None,
            expected_profile=attached.value,
        )
        assigned = await repository.get_assigned_profile(character)
        assert assigned.value is not None
        assert assigned.value.profile.reference == attached.value.reference


@pytest.mark.asyncio
async def test_replace_uses_new_identity_and_remove_recovers_reference_free_profile(
    tmp_path: Path,
) -> None:
    async with _opened_repository(tmp_path / "profiles.sqlite3") as repository:
        generation, revision = await _create(repository, PROFILE_A, "Narrator")
        first = await repository.set_reference(
            PROFILE_A,
            _canonical(sample=1),
            _requirements_for_tests(model_id="tts-1"),
            expected_revision=revision,
            expected_generation=generation,
        )
        second = await repository.set_reference(
            PROFILE_A,
            _canonical(sample=2),
            _requirements_for_tests(model_id="tts-1"),
            expected_revision=first.value.revision,
            expected_generation=generation,
        )

        assert first.value.reference is not None
        assert second.value.reference is not None
        assert first.value.reference.reference_id == REFERENCE_A
        assert second.value.reference.reference_id == REFERENCE_B
        assert second.value.revision == 3

        removed = await repository.remove_reference(
            PROFILE_A,
            expected_revision=3,
            expected_generation=generation,
        )
        assert removed.value.revision == 4
        assert removed.value.reference is None
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.get_reference(
                PROFILE_A,
                expected_revision=4,
                expected_generation=generation,
            )
        _assert_error(caught.value, "missing")


@pytest.mark.asyncio
async def test_profile_update_preserves_attached_reference_summary(
    tmp_path: Path,
) -> None:
    async with _opened_repository(tmp_path / "profiles.sqlite3") as repository:
        generation, revision = await _create(repository, PROFILE_A, "Narrator")
        attached = await repository.set_reference(
            PROFILE_A,
            _canonical(),
            _requirements_for_tests(model_id="tts-1"),
            expected_revision=revision,
            expected_generation=generation,
        )

        updated = await repository.update_profile(
            PROFILE_A,
            attached.value.revision,
            _draft("Renamed"),
            expected_generation=generation,
        )

        assert updated.value.display_name == "Renamed"
        assert updated.value.revision == 3
        assert updated.value.reference == attached.value.reference


@pytest.mark.asyncio
async def test_reference_mutations_enforce_generation_revision_and_parent_presence(
    tmp_path: Path,
) -> None:
    async with _opened_repository(tmp_path / "profiles.sqlite3") as repository:
        generation, revision = await _create(repository, PROFILE_A, "Narrator")
        for expected_revision, expected_generation, code in (
            (revision + 1, generation, "conflict"),
            (revision, generation + 1, "stale"),
        ):
            with pytest.raises(ProfileRepositoryError) as caught:
                await repository.set_reference(
                    PROFILE_A,
                    _canonical(),
                    _requirements_for_tests(model_id="tts-1"),
                    expected_revision=expected_revision,
                    expected_generation=expected_generation,
                )
            _assert_error(caught.value, code)

        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.set_reference(
                PROFILE_B,
                _canonical(),
                _requirements_for_tests(model_id="tts-1"),
                expected_revision=1,
                expected_generation=generation,
            )
        _assert_error(caught.value, "missing")


@pytest.mark.asyncio
async def test_profile_delete_cascades_reference_row(tmp_path: Path) -> None:
    path = tmp_path / "profiles.sqlite3"
    async with _opened_repository(path) as repository:
        generation, revision = await _create(repository, PROFILE_A, "Narrator")
        await repository.set_reference(
            PROFILE_A,
            _canonical(),
            _requirements_for_tests(model_id="tts-1"),
            expected_revision=revision,
            expected_generation=generation,
        )
        await repository.delete_profile(PROFILE_A, expected_generation=generation)

    connection = sqlite3.connect(path)
    try:
        assert (
            connection.execute(
                "SELECT count(*) FROM tts_profile_clone_references"
            ).fetchone()[0]
            == 0
        )
    finally:
        connection.close()


@pytest.mark.asyncio
async def test_delete_clears_damage_marker_before_same_uuid_is_reused(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    async with _opened_repository(path) as repository:
        generation, revision = await _create(repository, PROFILE_A, "Narrator")
        attached = await repository.set_reference(
            PROFILE_A,
            _canonical(),
            _requirements_for_tests(model_id="tts-1"),
            expected_revision=revision,
            expected_generation=generation,
        )
        connection = sqlite3.connect(path)
        try:
            connection.execute(
                "UPDATE tts_profile_clone_references SET sha256 = ?",
                ("0" * 64,),
            )
            connection.commit()
        finally:
            connection.close()
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.get_reference(
                PROFILE_A,
                expected_revision=attached.value.revision,
                expected_generation=generation,
            )
        _assert_error(caught.value, "reference_unavailable")

        await repository.delete_profile(PROFILE_A, expected_generation=generation)
        recreated = await repository.create_profile(
            _draft("Narrator"),
            PROFILE_A,
            expected_generation=generation,
        )
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.get_reference(
                PROFILE_A,
                expected_revision=recreated.value.revision,
                expected_generation=generation,
            )

        _assert_error(caught.value, "missing")


@pytest.mark.asyncio
async def test_count_and_byte_quotas_include_replacement_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    small = _canonical(sample=1, frames=16)
    larger = _canonical(sample=2, frames=32)
    monkeypatch.setattr(profile_repository, "MAX_REFERENCE_COUNT", 2)
    monkeypatch.setattr(
        profile_repository, "MAX_REFERENCE_TOTAL_BYTES", larger.byte_length
    )
    async with _opened_repository(tmp_path / "profiles.sqlite3") as repository:
        generation, revision_a = await _create(repository, PROFILE_A, "A")
        _, revision_b = await _create(repository, PROFILE_B, "B")
        first = await repository.set_reference(
            PROFILE_A,
            small,
            _requirements_for_tests(model_id="tts-1"),
            expected_revision=revision_a,
            expected_generation=generation,
        )
        replacement = await repository.set_reference(
            PROFILE_A,
            larger,
            _requirements_for_tests(model_id="tts-1"),
            expected_revision=first.value.revision,
            expected_generation=generation,
        )
        assert replacement.value.reference is not None
        assert replacement.value.reference.byte_length == larger.byte_length

        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.set_reference(
                PROFILE_B,
                small,
                _requirements_for_tests(model_id="tts-1"),
                expected_revision=revision_b,
                expected_generation=generation,
            )
        _assert_error(caught.value, "reference_quota")

        monkeypatch.setattr(profile_repository, "MAX_REFERENCE_COUNT", 1)
        monkeypatch.setattr(
            profile_repository,
            "MAX_REFERENCE_TOTAL_BYTES",
            larger.byte_length + small.byte_length,
        )
        with pytest.raises(ProfileRepositoryError) as count_caught:
            await repository.set_reference(
                PROFILE_B,
                small,
                _requirements_for_tests(model_id="tts-1"),
                expected_revision=revision_b,
                expected_generation=generation,
            )
        _assert_error(count_caught.value, "reference_quota")


@pytest.mark.asyncio
async def test_concurrent_reference_mutations_admit_only_one_revision(
    tmp_path: Path,
) -> None:
    async with _opened_repository(tmp_path / "profiles.sqlite3") as repository:
        generation, revision = await _create(repository, PROFILE_A, "Narrator")

        async def attach(sample: int) -> object:
            try:
                return await repository.set_reference(
                    PROFILE_A,
                    _canonical(sample=sample),
                    _requirements_for_tests(model_id="tts-1"),
                    expected_revision=revision,
                    expected_generation=generation,
                )
            except ProfileRepositoryError as error:
                return error

        outcomes = await asyncio.gather(attach(1), attach(2))
        successes = [value for value in outcomes if not isinstance(value, Exception)]
        failures = [value for value in outcomes if isinstance(value, Exception)]
        assert len(successes) == 1
        assert len(failures) == 1
        assert isinstance(failures[0], ProfileRepositoryError)
        _assert_error(failures[0], "conflict")
        loaded = await repository.get_profile(PROFILE_A)
        assert loaded.value.revision == 2
        assert loaded.value.reference is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [RuntimeError("PRIVATE write detail"), _ControlFlow()]
)
async def test_reference_write_failure_rolls_back_payload_and_parent_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    path = tmp_path / "profiles.sqlite3"

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(profile_repository, "write_reference_blob", fail_write)
    async with _opened_repository(path) as repository:
        generation, revision = await _create(repository, PROFILE_A, "Narrator")
        if isinstance(failure, Exception):
            with pytest.raises(ProfileRepositoryError) as caught:
                await repository.set_reference(
                    PROFILE_A,
                    _canonical(),
                    _requirements_for_tests(model_id="tts-1"),
                    expected_revision=revision,
                    expected_generation=generation,
                )
            _assert_error(caught.value, "operation_failed")
        else:
            with pytest.raises(_ControlFlow):
                await repository.set_reference(
                    PROFILE_A,
                    _canonical(),
                    _requirements_for_tests(model_id="tts-1"),
                    expected_revision=revision,
                    expected_generation=generation,
                )
        loaded = await repository.get_profile(PROFILE_A)
        assert loaded.value.revision == 1
        assert loaded.value.reference is None

    connection = sqlite3.connect(path)
    try:
        assert (
            connection.execute(
                "SELECT count(*) FROM tts_profile_clone_references"
            ).fetchone()[0]
            == 0
        )
    finally:
        connection.close()


@pytest.mark.asyncio
async def test_later_parent_failure_rolls_back_completed_blob_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "profiles.sqlite3"

    def fail_parent(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("PRIVATE parent failure")

    monkeypatch.setattr(
        profile_repository.TTSProfileRepository,
        "_worker_bump_reference_revision",
        fail_parent,
    )
    async with _opened_repository(path) as repository:
        generation, revision = await _create(repository, PROFILE_A, "Narrator")
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.set_reference(
                PROFILE_A,
                _canonical(),
                _requirements_for_tests(model_id="tts-1"),
                expected_revision=revision,
                expected_generation=generation,
            )
        _assert_error(caught.value, "operation_failed")

    connection = sqlite3.connect(path)
    try:
        assert (
            connection.execute(
                "SELECT count(*) FROM tts_profile_clone_references"
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT revision FROM tts_generation_profiles"
            ).fetchone()[0]
            == 1
        )
    finally:
        connection.close()


@pytest.mark.asyncio
async def test_backup_and_restore_round_trip_exact_reference_payload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    backup = tmp_path / "profiles-backup.sqlite3"
    canonical = _canonical(sample=7)
    async with _opened_repository(path) as repository:
        generation, revision = await _create(repository, PROFILE_A, "Narrator")
        attached = await repository.set_reference(
            PROFILE_A,
            canonical,
            _requirements_for_tests(model_id="tts-1"),
            expected_revision=revision,
            expected_generation=generation,
        )
        await repository.backup_to(backup, timeout_seconds=2.0)
        await repository.remove_reference(
            PROFILE_A,
            expected_revision=attached.value.revision,
            expected_generation=generation,
        )
        restored = await repository.restore_from(backup, timeout_seconds=2.0)
        profile = await repository.get_profile(PROFILE_A)
        exact = await repository.get_reference(
            PROFILE_A,
            expected_revision=profile.value.revision,
            expected_generation=restored.generation,
        )

        assert exact.value.wav_bytes == canonical.wav_bytes
        assert exact.value.reference_text == canonical.reference_text
        assert exact.value.sha256 == canonical.sha256


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("corruption_sql", "parameters", "ignore_checks"),
    (
        (
            "UPDATE tts_profile_clone_references SET wav_bytes = zeroblob(byte_length)",
            (),
            False,
        ),
        (
            "UPDATE tts_profile_clone_references SET sha256 = ?",
            ("0" * 64,),
            False,
        ),
        (
            "UPDATE tts_profile_clone_references SET duration_ms = duration_ms + 1",
            (),
            False,
        ),
        (
            "UPDATE tts_profile_clone_references SET reference_text = ''",
            (),
            True,
        ),
    ),
)
async def test_backup_rejects_corrupt_reference_without_publication(
    tmp_path: Path,
    corruption_sql: str,
    parameters: tuple[object, ...],
    ignore_checks: bool,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    destination = tmp_path / "must-not-publish.sqlite3"
    async with _opened_repository(path) as repository:
        generation, revision = await _create(repository, PROFILE_A, "Narrator")
        await repository.set_reference(
            PROFILE_A,
            _canonical(),
            _requirements_for_tests(model_id="tts-1"),
            expected_revision=revision,
            expected_generation=generation,
        )
        connection = sqlite3.connect(path)
        try:
            if ignore_checks:
                connection.execute("PRAGMA ignore_check_constraints = ON")
            connection.execute(corruption_sql, parameters)
            connection.commit()
        finally:
            connection.close()
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.backup_to(destination, timeout_seconds=2.0)
        _assert_error(caught.value, "backup_failed")
    assert destination.exists() is False


@pytest.mark.asyncio
async def test_exact_damage_is_isolated_and_replacement_recovers_profile(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    async with _opened_repository(path) as repository:
        generation, revision_a = await _create(repository, PROFILE_A, "A")
        await _create(repository, PROFILE_B, "B")
        attached = await repository.set_reference(
            PROFILE_A,
            _canonical(),
            _requirements_for_tests(model_id="tts-1"),
            expected_revision=revision_a,
            expected_generation=generation,
        )

    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "UPDATE tts_profile_clone_references SET duration_ms = duration_ms + 1"
        )
        connection.commit()
    finally:
        connection.close()

    async with _opened_repository(path) as repository:
        generation = repository.generation
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.get_reference(
                PROFILE_A,
                expected_revision=attached.value.revision,
                expected_generation=generation,
            )
        _assert_error(caught.value, "reference_unavailable")
        assert PROFILE_A in repository._damaged_reference_profile_ids
        assert repository.state.value == "open"
        assert (await repository.get_profile(PROFILE_B)).value.display_name == "B"

        repaired = await repository.set_reference(
            PROFILE_A,
            _canonical(sample=9),
            _requirements_for_tests(model_id="tts-1"),
            expected_revision=attached.value.revision,
            expected_generation=generation,
        )
        exact = await repository.get_reference(
            PROFILE_A,
            expected_revision=repaired.value.revision,
            expected_generation=generation,
        )
        assert exact.value.reference_text == "Reference 9"
        assert PROFILE_A not in repository._damaged_reference_profile_ids


@pytest.mark.asyncio
async def test_structural_reference_read_failure_makes_repository_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    repository = profile_repository.TTSProfileRepository(path, _clock=lambda: NOW)
    await repository.open()
    generation, revision = await _create(repository, PROFILE_A, "Narrator")
    attached = await repository.set_reference(
        PROFILE_A,
        _canonical(),
        _requirements_for_tests(model_id="tts-1"),
        expected_revision=revision,
        expected_generation=generation,
    )

    def fail_structurally(*_args: object, **_kwargs: object) -> bytes:
        raise sqlite3.DatabaseError("private structural detail")

    monkeypatch.setattr(profile_repository, "read_reference_blob", fail_structurally)
    try:
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.get_reference(
                PROFILE_A,
                expected_revision=attached.value.revision,
                expected_generation=generation,
            )
        _assert_error(caught.value, "schema_corrupt")
        assert repository.state.value == "unavailable"
        assert PROFILE_A not in repository._damaged_reference_profile_ids
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_close_clears_generation_local_damage_markers(tmp_path: Path) -> None:
    path = tmp_path / "profiles.sqlite3"
    repository = profile_repository.TTSProfileRepository(path, _clock=lambda: NOW)
    await repository.open()
    generation, revision = await _create(repository, PROFILE_A, "Narrator")
    attached = await repository.set_reference(
        PROFILE_A,
        _canonical(),
        _requirements_for_tests(model_id="tts-1"),
        expected_revision=revision,
        expected_generation=generation,
    )
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "UPDATE tts_profile_clone_references SET sha256 = ?",
            ("0" * 64,),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(ProfileRepositoryError):
        await repository.get_reference(
            PROFILE_A,
            expected_revision=attached.value.revision,
            expected_generation=generation,
        )
    assert PROFILE_A in repository._damaged_reference_profile_ids

    await repository.close()

    assert repository._damaged_reference_profile_ids == set()


@pytest.mark.asyncio
async def test_restore_setup_failure_clears_prior_generation_damage_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    repository = profile_repository.TTSProfileRepository(path, _clock=lambda: NOW)
    await repository.open()
    generation, revision = await _create(repository, PROFILE_A, "Narrator")
    attached = await repository.set_reference(
        PROFILE_A,
        _canonical(),
        _requirements_for_tests(model_id="tts-1"),
        expected_revision=revision,
        expected_generation=generation,
    )
    await repository.backup_to(candidate)
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "UPDATE tts_profile_clone_references SET sha256 = ?",
            ("0" * 64,),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(ProfileRepositoryError):
        await repository.get_reference(
            PROFILE_A,
            expected_revision=attached.value.revision,
            expected_generation=generation,
        )
    assert PROFILE_A in repository._damaged_reference_profile_ids

    real_create_task = profile_repository.asyncio.create_task

    def fail_task_creation(coroutine: Any) -> None:
        coroutine.close()
        raise RuntimeError("private setup failure")

    with monkeypatch.context() as scoped:
        scoped.setattr(profile_repository.asyncio, "create_task", fail_task_creation)
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.restore_from(candidate)
    _assert_error(caught.value, "restore_failed")
    assert repository.state.value == "open"
    assert repository.generation == generation + 1
    assert repository._damaged_reference_profile_ids == set()

    assert profile_repository.asyncio.create_task is real_create_task
    await repository.close()


@pytest.mark.asyncio
async def test_restore_rejects_corrupt_reference_and_preserves_live_store(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    async with _opened_repository(path) as repository:
        generation, revision = await _create(repository, PROFILE_A, "Live")
        await repository.set_reference(
            PROFILE_A,
            _canonical(),
            _requirements_for_tests(model_id="tts-1"),
            expected_revision=revision,
            expected_generation=generation,
        )
        await repository.backup_to(candidate)

    connection = sqlite3.connect(candidate)
    try:
        connection.execute(
            "UPDATE tts_profile_clone_references SET sha256 = ?",
            ("0" * 64,),
        )
        connection.commit()
    finally:
        connection.close()

    async with _opened_repository(path) as repository:
        before = await repository.get_profile(PROFILE_A)
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.restore_from(candidate, timeout_seconds=2.0)
        _assert_error(caught.value, "restore_failed")
        after = await repository.get_profile(PROFILE_A)

        assert after.value == before.value
        assert repository.state.value == "open"


@pytest.mark.asyncio
async def test_restore_qualifies_stage_recovery_and_both_live_handles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    qualification_calls = 0
    real_validate = profile_repository.validate_reference_rows

    def count_qualification(
        connection: sqlite3.Connection,
        *,
        check_deadline: object = None,
    ) -> None:
        nonlocal qualification_calls
        qualification_calls += 1
        real_validate(connection, check_deadline=cast(Any, check_deadline))

    async with _opened_repository(path) as repository:
        generation, revision = await _create(repository, PROFILE_A, "Narrator")
        await repository.set_reference(
            PROFILE_A,
            _canonical(),
            _requirements_for_tests(model_id="tts-1"),
            expected_revision=revision,
            expected_generation=generation,
        )
        await repository.backup_to(candidate)
        qualification_calls = 0
        monkeypatch.setattr(
            profile_repository,
            "validate_reference_rows",
            count_qualification,
        )

        await repository.restore_from(candidate, timeout_seconds=2.0)

    assert qualification_calls == 4


@pytest.mark.asyncio
async def test_backup_deadline_interrupts_reference_scan_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    destination = tmp_path / "must-not-publish.sqlite3"
    now = 0.0
    real_read = reference_storage.read_reference_blob

    def expire_during_blob_scan(
        connection: sqlite3.Connection,
        rowid: int,
        byte_length: int,
        *,
        progress_guard: object = None,
    ) -> bytes:
        nonlocal now
        now = 2.0
        return real_read(
            connection,
            rowid,
            byte_length,
            progress_guard=cast(Any, progress_guard),
        )

    monkeypatch.setattr(profile_repository, "_monotonic", lambda: now)
    monkeypatch.setattr(
        reference_storage,
        "read_reference_blob",
        expire_during_blob_scan,
    )
    async with _opened_repository(path) as repository:
        generation, revision = await _create(repository, PROFILE_A, "Narrator")
        await repository.set_reference(
            PROFILE_A,
            _canonical(),
            _requirements_for_tests(model_id="tts-1"),
            expected_revision=revision,
            expected_generation=generation,
        )
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.backup_to(destination, timeout_seconds=1.0)
        _assert_error(caught.value, "backup_failed")

    assert destination.exists() is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "timeout",
    (None, True, 0, -1, float("inf"), float("-inf"), float("nan"), "5"),
)
async def test_backup_rejects_invalid_timeout_before_file_mutation(
    tmp_path: Path,
    timeout: object,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    destination = tmp_path / "must-not-publish.sqlite3"
    async with _opened_repository(path) as repository:
        before_generation = repository.generation
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.backup_to(
                destination,
                timeout_seconds=cast(Any, timeout),
            )
        _assert_error(caught.value, "backup_failed")
        assert repository.generation == before_generation
        assert repository.state.value == "open"

    assert destination.exists() is False


@pytest.mark.asyncio
async def test_ordinary_reads_never_project_sensitive_reference_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "profiles.sqlite3"
    observed_columns: list[set[str]] = []
    real_profile_decode = profile_repository._decode_profile_with_reference_row
    real_assigned_decode = profile_repository._decode_assigned_with_reference_row

    def inspect_profile(row: sqlite3.Row) -> object:
        observed_columns.append(set(row.keys()))
        return real_profile_decode(row)

    def inspect_assigned(row: sqlite3.Row) -> object:
        observed_columns.append(set(row.keys()))
        return real_assigned_decode(row)

    async with _opened_repository(path) as repository:
        generation, revision = await _create(repository, PROFILE_A, "Narrator")
        attached = await repository.set_reference(
            PROFILE_A,
            _canonical(),
            _requirements_for_tests(model_id="tts-1"),
            expected_revision=revision,
            expected_generation=generation,
        )
        character = CharacterRef("local", "library", "narrator")
        await repository.set_assignment(
            character,
            PROFILE_A,
            expected_generation=generation,
            expected_profile_revision=attached.value.revision,
            expected_current_profile_id=None,
        )
        monkeypatch.setattr(
            profile_repository, "_decode_profile_with_reference_row", inspect_profile
        )
        monkeypatch.setattr(
            profile_repository, "_decode_assigned_with_reference_row", inspect_assigned
        )
        monkeypatch.setattr(
            profile_repository,
            "read_reference_blob",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("ordinary read opened a reference BLOB")
            ),
        )

        await repository.get_profile(PROFILE_A)
        await repository.list_profiles()
        await repository.get_assigned_profile(character)

    assert observed_columns
    for columns in observed_columns:
        assert "wav_bytes" not in columns
        assert "reference_text" not in columns
        assert "sha256" not in columns


class _RecordingBlob:
    def __init__(
        self,
        blob: sqlite3.Blob,
        writes: list[int],
        reads: list[int],
        *,
        close_error: BaseException | None = None,
    ) -> None:
        self._blob = blob
        self._writes = writes
        self._reads = reads
        self._close_error = close_error
        self.closed = False

    def __len__(self) -> int:
        return len(self._blob)

    def write(self, payload: bytes) -> None:
        self._writes.append(len(payload))
        self._blob.write(payload)

    def read(self, size: int = -1) -> bytes:
        self._reads.append(size)
        return self._blob.read(size)

    def tell(self) -> int:
        return self._blob.tell()

    def close(self) -> None:
        self._blob.close()
        self.closed = True
        if self._close_error is not None:
            raise self._close_error


class _RecordingBlobConnection:
    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        close_error: BaseException | None = None,
    ) -> None:
        self.connection = connection
        self.close_error = close_error
        self.writes: list[int] = []
        self.reads: list[int] = []
        self.blobs: list[_RecordingBlob] = []

    def blobopen(self, *args: object, **kwargs: object) -> _RecordingBlob:
        blob = _RecordingBlob(
            self.connection.blobopen(*args, **kwargs),
            self.writes,
            self.reads,
            close_error=self.close_error,
        )
        self.blobs.append(blob)
        return blob


def test_blob_seam_streams_bounded_chunks_and_closes_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE tts_profile_clone_references(profile_id TEXT, wav_bytes BLOB)"
    )
    payload = b"0123456789abcdefghijklmnopqrstuvwxyz"
    cursor = connection.execute(
        "INSERT INTO tts_profile_clone_references VALUES ('p', zeroblob(?))",
        (len(payload),),
    )
    rowid = cursor.lastrowid
    assert type(rowid) is int
    recording = _RecordingBlobConnection(connection)
    monkeypatch.setattr(reference_storage, "REFERENCE_BLOB_CHUNK_BYTES", 7)

    reference_storage.write_reference_blob(cast(Any, recording), rowid, payload)
    restored = reference_storage.read_reference_blob(
        cast(Any, recording), rowid, len(payload)
    )

    assert restored == payload
    assert recording.writes == [7, 7, 7, 7, 7, 1]
    assert recording.reads == [7, 7, 7, 7, 7, 1, 1]
    assert all(blob.closed for blob in recording.blobs)
    connection.close()


@pytest.mark.parametrize("operation", ["read", "write"])
def test_blob_close_failure_is_safe_and_bounded(
    operation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE tts_profile_clone_references(profile_id TEXT, wav_bytes BLOB)"
    )
    payload = b"bounded payload"
    rowid = connection.execute(
        "INSERT INTO tts_profile_clone_references VALUES ('p', zeroblob(?))",
        (len(payload),),
    ).lastrowid
    assert type(rowid) is int
    recording = _RecordingBlobConnection(
        connection, close_error=RuntimeError("PRIVATE blob close detail")
    )
    monkeypatch.setattr(reference_storage, "REFERENCE_BLOB_CHUNK_BYTES", 7)

    with pytest.raises(ProfileRepositoryError) as caught:
        if operation == "write":
            reference_storage.write_reference_blob(cast(Any, recording), rowid, payload)
        else:
            reference_storage.read_reference_blob(
                cast(Any, recording), rowid, len(payload)
            )

    _assert_error(
        caught.value,
        "operation_failed" if operation == "write" else "reference_unavailable",
    )
    connection.close()
