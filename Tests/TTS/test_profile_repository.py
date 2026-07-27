"""Transactional CRUD tests for the serialized TTS profile repository."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import traceback
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest

import tldw_chatbook.TTS.profile_repository as profile_repository
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
from tldw_chatbook.TTS.profile_schema import encode_assignment
from tldw_chatbook.TTS.profile_types import (
    CharacterRef,
    CharacterTTSAssignment,
    ProfileRepositoryState,
    ProfileStoreResult,
    TTSGenerationProfile,
    TTSProfileDraft,
    TTSProfilePage,
    canonical_json_options,
)


CREATED_AT = datetime(2026, 7, 26, 12, 0, 0, 123456, tzinfo=UTC)
UPDATED_AT = datetime(2026, 7, 26, 13, 30, 0, 654321, tzinfo=UTC)
LATER_AT = datetime(2026, 7, 26, 14, 45, 0, 111111, tzinfo=UTC)
GENERATED_ID = UUID("10000000-0000-4000-8000-000000000001")
CALLER_ID = UUID("20000000-0000-0000-8000-000000000002")


class _ControlFlow(BaseException):
    """Test-only caller control-flow signal."""


class _SequenceCallable:
    """Return a deterministic sequence and fail if code consumes too much."""

    def __init__(self, values: Iterator[Any]) -> None:
        self._values = values

    def __call__(self) -> Any:
        return next(self._values)


class _TextSubclass(str):
    """An inexact public-boundary string."""


class _FalseyCallable:
    """A private seam whose truth value must not be inspected."""

    def __init__(self, value: object) -> None:
        self._value = value

    def __call__(self) -> object:
        return self._value

    def __bool__(self) -> bool:
        raise AssertionError("constructor inspected callable truthiness")


def _draft(
    display_name: str = "Narrator",
    *,
    provider_id: str = "openai",
    model_id: str = "tts-1-hd/音声",
    voice_id: str | None = "alloy/声",
    response_format: str = "mp3",
    speed: float = 1.25,
    options: object | None = None,
) -> TTSProfileDraft:
    return TTSProfileDraft(
        display_name=display_name,
        provider_id=provider_id,
        model_id=model_id,
        voice_id=voice_id,
        response_format=response_format,
        speed=speed,
        options=cast(
            Any,
            {
                "nested": {"items": [True, 2, 3.5, None]},
                "locale": "日本語",
            }
            if options is None
            else options,
        ),
    )


@asynccontextmanager
async def _opened_repository(
    database_path: Path,
    *,
    clock: Callable[[], datetime] | None = None,
    uuid_factory: Callable[[], UUID] | None = None,
) -> AsyncIterator[profile_repository.TTSProfileRepository]:
    repository = profile_repository.TTSProfileRepository(
        database_path,
        _clock=clock or (lambda: CREATED_AT),
        _uuid_factory=uuid_factory or (lambda: GENERATED_ID),
    )
    await repository.open()
    try:
        yield repository
    finally:
        await repository.close()


def _assert_safe_error(
    error: ProfileRepositoryError,
    code: str,
    *secrets: str,
) -> None:
    assert type(error) is ProfileRepositoryError
    assert error.code == code
    assert str(error) == f"TTS profile repository failed: {code}"
    assert error.__cause__ is None
    assert error.__context__ is None
    visible = " ".join(
        (
            str(error),
            repr(error),
            "".join(traceback.format_exception(error)),
            *(str(note) for note in getattr(error, "__notes__", ())),
        )
    )
    for secret in secrets:
        if secret:
            assert secret not in visible


def _external_execute(
    database_path: Path,
    statement: str,
    parameters: tuple[object, ...] = (),
) -> None:
    connection = sqlite3.connect(database_path, isolation_level=None)
    try:
        connection.execute(statement, parameters)
    finally:
        connection.close()


def _fail_next_commit(
    monkeypatch: pytest.MonkeyPatch,
    repository: profile_repository.TTSProfileRepository,
    error: BaseException,
) -> threading.Event:
    original_commit = repository._commit_transaction
    failed = False
    attempted = threading.Event()

    def fail_once(connection: sqlite3.Connection) -> None:
        nonlocal failed
        if not failed:
            failed = True
            attempted.set()
            raise error
        original_commit(connection)

    monkeypatch.setattr(repository, "_commit_transaction", fail_once)
    return attempted


def test_private_clock_and_uuid_seams_remain_constructor_pure(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "nested" / "profiles.sqlite3"

    repository = profile_repository.TTSProfileRepository(
        database_path,
        _clock=cast(Callable[[], datetime], _FalseyCallable(CREATED_AT)),
        _uuid_factory=cast(Callable[[], UUID], _FalseyCallable(GENERATED_ID)),
    )

    assert repository.state.value == "closed"
    assert repository.generation == 0
    assert not database_path.parent.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "args", "kwargs", "secret"),
    [
        ("create_profile", (object(),), {}, ""),
        (
            "create_profile",
            (_draft(),),
            {"profile_id": "secret-profile-id"},
            "secret-profile-id",
        ),
        ("get_profile", ("secret-get-id",), {}, "secret-get-id"),
        ("get_profile", (GENERATED_ID.bytes,), {}, ""),
        ("list_profiles", (), {"search": object()}, ""),
        (
            "list_profiles",
            (),
            {"search": _TextSubclass("secret-subclass")},
            "secret-subclass",
        ),
        (
            "list_profiles",
            (),
            {"search": "\x00secret-null-search"},
            "secret-null-search",
        ),
        ("list_profiles", (), {"search": "s" * 129}, "s" * 129),
        ("list_profiles", (), {"limit": True}, ""),
        ("list_profiles", (), {"limit": 0}, ""),
        ("list_profiles", (), {"limit": 101}, ""),
        ("list_profiles", (), {"offset": True}, ""),
        ("list_profiles", (), {"offset": -1}, ""),
        ("update_profile", ("secret-update-id", 1, _draft()), {}, "secret-update-id"),
        ("update_profile", (GENERATED_ID, True, _draft()), {}, ""),
        ("update_profile", (GENERATED_ID, 0, _draft()), {}, ""),
        ("update_profile", (GENERATED_ID, 1.0, _draft()), {}, ""),
        ("update_profile", (GENERATED_ID, 1, object()), {}, ""),
        ("delete_profile", ("secret-delete-id",), {}, "secret-delete-id"),
    ],
)
async def test_invalid_public_inputs_fail_before_worker_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    secret: str,
) -> None:
    repository = profile_repository.TTSProfileRepository(
        tmp_path / "must-not-exist" / "profiles.sqlite3"
    )
    submitted = False

    async def forbidden_submission(
        _operation: Callable[[sqlite3.Connection], object],
    ) -> ProfileStoreResult[object]:
        nonlocal submitted
        submitted = True
        raise AssertionError("invalid input reached worker submission")

    monkeypatch.setattr(repository, "_submit_operation", forbidden_submission)

    with pytest.raises(ProfileRepositoryError) as caught:
        await getattr(repository, method_name)(*args, **kwargs)

    _assert_safe_error(caught.value, "operation_failed", secret)
    assert submitted is False
    assert not tmp_path.joinpath("must-not-exist").exists()


@pytest.mark.asyncio
async def test_valid_crud_uses_lifecycle_state_lane_before_open(tmp_path: Path) -> None:
    repository = profile_repository.TTSProfileRepository(tmp_path / "profiles.sqlite3")

    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.get_profile(GENERATED_ID)

    _assert_safe_error(caught.value, "closed")
    assert not (tmp_path / "profiles.sqlite3").exists()


@pytest.mark.asyncio
async def test_hostile_search_normalizer_fails_safely_before_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "secret-normalizer-store.sqlite3"
    secret = "secret-hostile-unicode-normalizer"
    repository = profile_repository.TTSProfileRepository(database_path)
    submitted = False

    async def forbidden_submission(
        _operation: Callable[[sqlite3.Connection], object],
    ) -> ProfileStoreResult[object]:
        nonlocal submitted
        submitted = True
        raise AssertionError("failed normalization reached worker submission")

    def hostile_normalizer(_form: str, _value: str) -> str:
        try:
            raise RuntimeError(secret)
        except RuntimeError:
            raise ValueError(secret)

    monkeypatch.setattr(repository, "_submit_operation", forbidden_submission)
    monkeypatch.setattr(
        profile_repository,
        "_unicode_normalize",
        hostile_normalizer,
    )

    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.list_profiles(search="ordinary")

    _assert_safe_error(
        caught.value,
        "operation_failed",
        secret,
        str(database_path),
    )
    assert submitted is False
    assert not database_path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw_search", "visible_secret"),
    [
        pytest.param(
            "\tsecret-leading-tab",
            "secret-leading-tab",
            id="leading-tab",
        ),
        pytest.param(
            "\nsecret-leading-newline",
            "secret-leading-newline",
            id="leading-newline",
        ),
        pytest.param(
            "\x01secret-leading-control",
            "secret-leading-control",
            id="leading-control",
        ),
        pytest.param(
            "secret-trailing-tab\t",
            "secret-trailing-tab",
            id="trailing-tab",
        ),
        pytest.param(
            "secret-trailing-newline\n",
            "secret-trailing-newline",
            id="trailing-newline",
        ),
        pytest.param(
            "secret-trailing-control\x1f",
            "secret-trailing-control",
            id="trailing-control",
        ),
        pytest.param("\t \n", "", id="control-only-mixed-space"),
        pytest.param(
            "\u200bsecret-format",
            "secret-format",
            id="format-control",
        ),
        pytest.param(
            "\ud800secret-surrogate",
            "secret-surrogate",
            id="surrogate",
        ),
        pytest.param(
            "\ufdd0secret-noncharacter-start",
            "secret-noncharacter-start",
            id="noncharacter-fdd0",
        ),
        pytest.param(
            "\ufdefsecret-noncharacter-end",
            "secret-noncharacter-end",
            id="noncharacter-fdef",
        ),
        pytest.param(
            "\ufffesecret-plane0-fffe",
            "secret-plane0-fffe",
            id="plane0-fffe",
        ),
        pytest.param(
            "\uffffsecret-plane0-ffff",
            "secret-plane0-ffff",
            id="plane0-ffff",
        ),
        pytest.param(
            "\U0001fffesecret-plane1-fffe",
            "secret-plane1-fffe",
            id="plane1-fffe",
        ),
        pytest.param(
            "\U0010ffffsecret-plane16-ffff",
            "secret-plane16-ffff",
            id="plane16-ffff",
        ),
    ],
)
async def test_raw_unsafe_search_fails_safely_before_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_search: str,
    visible_secret: str,
) -> None:
    database_path = tmp_path / "secret-unsafe-search-store.sqlite3"
    repository = profile_repository.TTSProfileRepository(database_path)
    submitted = False

    async def forbidden_submission(
        _operation: Callable[[sqlite3.Connection], object],
    ) -> ProfileStoreResult[object]:
        nonlocal submitted
        submitted = True
        raise AssertionError("unsafe raw search reached worker submission")

    monkeypatch.setattr(repository, "_submit_operation", forbidden_submission)

    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.list_profiles(search=raw_search)

    _assert_safe_error(
        caught.value,
        "operation_failed",
        visible_secret,
        str(database_path),
    )
    assert repr(raw_search) not in "".join(traceback.format_exception(caught.value))
    assert submitted is False
    assert not database_path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("helper_name", "secret"),
    [
        ("_unicode_category", "secret-hostile-unicode-category"),
        ("_unicode_ord", "secret-hostile-unicode-ord"),
    ],
)
async def test_hostile_search_character_inspection_fails_safely_before_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    helper_name: str,
    secret: str,
) -> None:
    database_path = tmp_path / "secret-character-inspector-store.sqlite3"
    repository = profile_repository.TTSProfileRepository(database_path)
    submitted = False

    async def forbidden_submission(
        _operation: Callable[[sqlite3.Connection], object],
    ) -> ProfileStoreResult[object]:
        nonlocal submitted
        submitted = True
        raise AssertionError("failed character inspection reached submission")

    def hostile_inspector(_character: str) -> object:
        try:
            raise RuntimeError(secret)
        except RuntimeError:
            raise ValueError(secret)

    monkeypatch.setattr(repository, "_submit_operation", forbidden_submission)
    monkeypatch.setattr(
        profile_repository,
        helper_name,
        hostile_inspector,
        raising=False,
    )

    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.list_profiles(search="ordinary")

    _assert_safe_error(
        caught.value,
        "operation_failed",
        secret,
        str(database_path),
    )
    assert submitted is False
    assert not database_path.exists()


@pytest.mark.asyncio
async def test_create_generates_uuid4_or_retains_exact_caller_uuid_and_round_trips(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    generated_draft = _draft("  Ｎａｒｒａｔｏｒ 音声  ")
    caller_draft = _draft(
        "Caller Profile",
        model_id="model/exact/2",
        voice_id=None,
        response_format="ogg",
        speed=0.75,
        options={"seed": 7, "labels": ["声", False]},
    )
    generated_factory = _SequenceCallable(iter((GENERATED_ID,)))

    async with _opened_repository(
        database_path,
        clock=lambda: CREATED_AT,
        uuid_factory=cast(Callable[[], UUID], generated_factory),
    ) as repository:
        generated_result = await repository.create_profile(generated_draft)
        caller_result = await repository.create_profile(
            caller_draft,
            profile_id=CALLER_ID,
        )

        assert generated_result.generation == 1
        assert caller_result.generation == 1
        generated = generated_result.value
        caller = caller_result.value
        assert type(generated) is TTSGenerationProfile
        assert generated.profile_id == GENERATED_ID
        assert generated.profile_id.version == 4
        assert generated.display_name == "Ｎａｒｒａｔｏｒ 音声"
        assert generated.normalized_name == "narrator 音声"
        assert generated.provider_id == generated_draft.provider_id
        assert generated.model_id == generated_draft.model_id
        assert generated.voice_id == generated_draft.voice_id
        assert generated.response_format == generated_draft.response_format
        assert generated.speed == generated_draft.speed
        assert canonical_json_options(generated.options) == canonical_json_options(
            generated_draft.options
        )
        assert generated.revision == 1
        assert generated.created_at == CREATED_AT
        assert generated.updated_at == CREATED_AT
        assert generated.created_at.tzinfo is UTC

        assert caller.profile_id == CALLER_ID
        assert caller.display_name == caller_draft.display_name
        assert caller.provider_id == caller_draft.provider_id
        assert caller.model_id == caller_draft.model_id
        assert caller.voice_id is None
        assert caller.response_format == caller_draft.response_format
        assert caller.speed == caller_draft.speed
        assert canonical_json_options(caller.options) == canonical_json_options(
            caller_draft.options
        )
        assert caller.revision == 1
        assert caller.created_at == caller.updated_at == CREATED_AT

        with pytest.raises((AttributeError, TypeError)):
            setattr(generated, "display_name", "mutated")
        with pytest.raises(TypeError):
            cast(Any, generated.options)["new"] = "mutated"


@pytest.mark.asyncio
async def test_create_conflicts_do_not_overwrite_uuid_or_normalized_name(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    first_id = UUID("30000000-0000-4000-8000-000000000003")
    other_id = UUID("40000000-0000-4000-8000-000000000004")
    third_id = UUID("50000000-0000-4000-8000-000000000005")

    async with _opened_repository(database_path) as repository:
        first = (
            await repository.create_profile(
                _draft("  Ｓｔｒａｓｓｅ ｶﾀｶﾅ  "),
                profile_id=first_id,
            )
        ).value

        with pytest.raises(ProfileRepositoryError) as duplicate_id:
            await repository.create_profile(
                _draft("A different display name"),
                profile_id=first_id,
            )
        _assert_safe_error(duplicate_id.value, "conflict")

        with pytest.raises(ProfileRepositoryError) as duplicate_name:
            await repository.create_profile(
                _draft("STRASSE カタカナ"),
                profile_id=other_id,
            )
        _assert_safe_error(duplicate_name.value, "conflict")

        with pytest.raises(ProfileRepositoryError) as casefold_name:
            await repository.create_profile(
                _draft("strasse カタカナ"),
                profile_id=third_id,
            )
        _assert_safe_error(casefold_name.value, "conflict")

        assert (await repository.get_profile(first_id)).value == first
        page = (await repository.list_profiles()).value
        assert page.profiles == (first,)
        assert page.total == 1


@pytest.mark.asyncio
async def test_get_missing_and_corrupt_rows_fail_with_safe_specific_codes(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "secret-profile-path.sqlite3"
    profile_id = UUID("60000000-0000-4000-8000-000000000006")
    corrupt_secret = "secret-corrupt-options"

    async with _opened_repository(database_path) as repository:
        with pytest.raises(ProfileRepositoryError) as missing:
            await repository.get_profile(profile_id)
        _assert_safe_error(missing.value, "missing", str(database_path))

        await repository.create_profile(_draft(), profile_id=profile_id)
        await asyncio.to_thread(
            _external_execute,
            database_path,
            """
            UPDATE tts_generation_profiles
            SET options_json = ?
            WHERE profile_id = ?
            """,
            (f'{{"{corrupt_secret}":NaN}}', str(profile_id)),
        )

        with pytest.raises(ProfileRepositoryError) as corrupt:
            await repository.get_profile(profile_id)
        _assert_safe_error(
            corrupt.value,
            "corrupt_data",
            corrupt_secret,
            str(database_path),
        )


@pytest.mark.asyncio
async def test_list_is_stable_paginated_immutable_and_reports_filtered_total(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    entries = (
        (UUID("00000000-0000-4000-8000-000000000004"), "Zulu"),
        (UUID("00000000-0000-4000-8000-000000000002"), "beta"),
        (UUID("00000000-0000-4000-8000-000000000003"), "Äther"),
        (UUID("00000000-0000-4000-8000-000000000001"), "Alpha"),
    )

    async with _opened_repository(database_path) as repository:
        for profile_id, display_name in entries:
            await repository.create_profile(
                _draft(display_name),
                profile_id=profile_id,
            )

        first_page = (await repository.list_profiles(limit=2, offset=0)).value
        second_page = (await repository.list_profiles(limit=2, offset=2)).value
        beyond = (await repository.list_profiles(limit=100, offset=20)).value
        single = (await repository.list_profiles(limit=1, offset=1)).value

        assert type(first_page) is TTSProfilePage
        assert type(first_page.profiles) is tuple
        expected_names = tuple(
            sorted(
                (display_name for _, display_name in entries),
                key=lambda name: (
                    _draft(name).normalized_name,
                    str(
                        next(profile_id for profile_id, item in entries if item == name)
                    ),
                ),
            )
        )
        assert (
            tuple(
                profile.display_name
                for profile in first_page.profiles + second_page.profiles
            )
            == expected_names
        )
        assert (
            first_page.total == second_page.total == beyond.total == single.total == 4
        )
        assert len(first_page.profiles) == len(second_page.profiles) == 2
        assert beyond.profiles == ()
        assert single.profiles == (first_page.profiles[1],)


@pytest.mark.asyncio
async def test_list_search_normalizes_case_and_treats_like_metacharacters_literal(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    names = (
        "100% Real",
        "under_score",
        "Bang! Voice",
        "Ｃａｆé 音声",
        "Straße",
        "Ordinary",
    )

    async with _opened_repository(database_path) as repository:
        for index, name in enumerate(names, start=1):
            await repository.create_profile(
                _draft(name),
                profile_id=UUID(f"70000000-0000-4000-8000-{index:012d}"),
            )

        percent = (await repository.list_profiles(search="%")).value
        underscore = (await repository.list_profiles(search="_")).value
        escape = (await repository.list_profiles(search="!")).value
        unicode_case = (await repository.list_profiles(search="  CAFÉ 音声  ")).value
        casefold = (await repository.list_profiles(search="STRASSE")).value
        empty = (await repository.list_profiles(search="")).value
        spaces_only = (await repository.list_profiles(search="   ")).value

        assert tuple(profile.display_name for profile in percent.profiles) == (
            "100% Real",
        )
        assert percent.total == 1
        assert tuple(profile.display_name for profile in underscore.profiles) == (
            "under_score",
        )
        assert underscore.total == 1
        assert tuple(profile.display_name for profile in escape.profiles) == (
            "Bang! Voice",
        )
        assert escape.total == 1
        assert tuple(profile.display_name for profile in unicode_case.profiles) == (
            "Ｃａｆé 音声",
        )
        assert unicode_case.total == 1
        assert tuple(profile.display_name for profile in casefold.profiles) == (
            "Straße",
        )
        assert casefold.total == 1
        assert empty == spaces_only
        assert empty.total == len(names)


@pytest.mark.asyncio
async def test_update_uses_optimistic_revision_and_preserves_winner_exactly(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    profile_id = UUID("80000000-0000-4000-8000-000000000008")
    clock = _SequenceCallable(iter((CREATED_AT, UPDATED_AT)))

    async with _opened_repository(
        database_path,
        clock=cast(Callable[[], datetime], clock),
    ) as repository:
        created = (
            await repository.create_profile(_draft("Shared"), profile_id=profile_id)
        ).value
        editor_a = (await repository.get_profile(profile_id)).value
        editor_b = (await repository.get_profile(profile_id)).value
        winner_draft = _draft(
            "Shared Updated",
            model_id="winner-model",
            voice_id=None,
            response_format="flac",
            speed=2.0,
            options={"winner": {"exact": [1, "声"]}},
        )
        loser_draft = _draft(
            "Loser Value",
            model_id="loser-model",
            options={"must": "not persist"},
        )

        winner = (
            await repository.update_profile(
                profile_id,
                editor_a.revision,
                winner_draft,
            )
        ).value
        with pytest.raises(ProfileRepositoryError) as stale:
            await repository.update_profile(
                profile_id,
                editor_b.revision,
                loser_draft,
            )
        _assert_safe_error(stale.value, "conflict")

        stored = (await repository.get_profile(profile_id)).value
        assert winner.revision == created.revision + 1 == 2
        assert winner.created_at == created.created_at == CREATED_AT
        assert winner.updated_at == UPDATED_AT
        assert winner.display_name == winner_draft.display_name
        assert winner.model_id == winner_draft.model_id
        assert winner.voice_id == winner_draft.voice_id
        assert winner.response_format == winner_draft.response_format
        assert winner.speed == winner_draft.speed
        assert canonical_json_options(winner.options) == canonical_json_options(
            winner_draft.options
        )
        assert stored == winner
        assert stored != editor_b


@pytest.mark.asyncio
async def test_update_allows_display_spelling_change_with_same_normalized_key(
    tmp_path: Path,
) -> None:
    profile_id = UUID("81000000-0000-4000-8000-000000000008")
    clock = _SequenceCallable(iter((CREATED_AT, UPDATED_AT)))

    async with _opened_repository(
        tmp_path / "profiles.sqlite3",
        clock=cast(Callable[[], datetime], clock),
    ) as repository:
        original = (
            await repository.create_profile(
                _draft("Straße"),
                profile_id=profile_id,
            )
        ).value
        updated = (
            await repository.update_profile(
                profile_id,
                original.revision,
                _draft("STRASSE"),
            )
        ).value

        assert original.normalized_name == updated.normalized_name == "strasse"
        assert updated.display_name == "STRASSE"
        assert updated.revision == 2


@pytest.mark.asyncio
async def test_update_missing_or_name_collision_rolls_back_without_partial_changes(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    first_id = UUID("82000000-0000-4000-8000-000000000008")
    second_id = UUID("83000000-0000-4000-8000-000000000008")
    missing_id = UUID("84000000-0000-4000-8000-000000000008")
    clock = _SequenceCallable(iter((CREATED_AT, CREATED_AT, UPDATED_AT, LATER_AT)))

    async with _opened_repository(
        database_path,
        clock=cast(Callable[[], datetime], clock),
    ) as repository:
        first = (
            await repository.create_profile(_draft("First"), profile_id=first_id)
        ).value
        second = (
            await repository.create_profile(_draft("Second"), profile_id=second_id)
        ).value

        with pytest.raises(ProfileRepositoryError) as missing:
            await repository.update_profile(
                missing_id,
                1,
                _draft("Missing"),
            )
        _assert_safe_error(missing.value, "missing")

        with pytest.raises(ProfileRepositoryError) as collision:
            await repository.update_profile(
                second_id,
                second.revision,
                _draft("Ｆｉｒｓｔ"),
            )
        _assert_safe_error(collision.value, "conflict")

        assert (await repository.get_profile(first_id)).value == first
        assert (await repository.get_profile(second_id)).value == second
        recovered = (
            await repository.update_profile(
                second_id,
                second.revision,
                _draft("Third"),
            )
        ).value
        assert recovered.revision == 2
        assert recovered.display_name == "Third"


@pytest.mark.asyncio
async def test_delete_removes_exactly_one_profile_and_missing_is_safe(
    tmp_path: Path,
) -> None:
    profile_id = UUID("90000000-0000-4000-8000-000000000009")
    other_id = UUID("91000000-0000-4000-8000-000000000009")

    async with _opened_repository(tmp_path / "profiles.sqlite3") as repository:
        await repository.create_profile(_draft("Delete"), profile_id=profile_id)
        other = (
            await repository.create_profile(_draft("Keep"), profile_id=other_id)
        ).value

        deleted = await repository.delete_profile(profile_id)

        assert deleted == ProfileStoreResult(generation=1, value=None)
        with pytest.raises(ProfileRepositoryError) as missing:
            await repository.get_profile(profile_id)
        _assert_safe_error(missing.value, "missing")
        assert (await repository.get_profile(other_id)).value == other

        with pytest.raises(ProfileRepositoryError) as missing_delete:
            await repository.delete_profile(profile_id)
        _assert_safe_error(missing_delete.value, "missing")


@pytest.mark.asyncio
async def test_delete_rejects_corrupt_target_without_discarding_it(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "secret-corrupt-delete.sqlite3"
    profile_id = UUID("91500000-0000-4000-8000-000000000009")
    corrupt_secret = "secret-delete-corrupt-options"

    async with _opened_repository(database_path) as repository:
        original = (
            await repository.create_profile(
                _draft("Corrupt Delete"),
                profile_id=profile_id,
            )
        ).value
        await asyncio.to_thread(
            _external_execute,
            database_path,
            """
            UPDATE tts_generation_profiles
            SET options_json = ?
            WHERE profile_id = ?
            """,
            (f'{{"{corrupt_secret}":NaN}}', str(profile_id)),
        )

        with pytest.raises(ProfileRepositoryError) as corrupt:
            await repository.delete_profile(profile_id)
        _assert_safe_error(
            corrupt.value,
            "corrupt_data",
            corrupt_secret,
            str(database_path),
        )

        await asyncio.to_thread(
            _external_execute,
            database_path,
            """
            UPDATE tts_generation_profiles
            SET options_json = ?
            WHERE profile_id = ?
            """,
            (canonical_json_options(original.options), str(profile_id)),
        )
        assert (await repository.get_profile(profile_id)).value == original


@pytest.mark.asyncio
async def test_foreign_key_restricted_delete_maps_to_conflict_without_policy(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    profile_id = UUID("92000000-0000-4000-8000-000000000009")
    assignment = CharacterTTSAssignment(
        character_ref=CharacterRef(
            source="local",
            authority_id="authority-1",
            character_id="character-1",
        ),
        profile_id=profile_id,
    )

    async with _opened_repository(database_path) as repository:
        profile = (
            await repository.create_profile(
                _draft("Assigned"),
                profile_id=profile_id,
            )
        ).value
        encoded = encode_assignment(
            assignment,
            created_at=CREATED_AT,
            updated_at=CREATED_AT,
        )
        await asyncio.to_thread(
            _external_execute,
            database_path,
            """
            INSERT INTO character_tts_assignments (
                source, authority_id, character_id, profile_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                encoded["source"],
                encoded["authority_id"],
                encoded["character_id"],
                encoded["profile_id"],
                encoded["created_at"],
                encoded["updated_at"],
            ),
        )

        with pytest.raises(ProfileRepositoryError) as conflict:
            await repository.delete_profile(profile_id)

        _assert_safe_error(conflict.value, "conflict")
        assert (await repository.get_profile(profile_id)).value == profile


@pytest.mark.asyncio
async def test_create_failure_before_commit_rolls_back_and_repository_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "secret-create-store.sqlite3"
    profile_id = UUID("a0000000-0000-4000-8000-00000000000a")
    secret = "secret-create-commit-failure"

    async with _opened_repository(database_path) as repository:
        attempted = _fail_next_commit(
            monkeypatch,
            repository,
            sqlite3.OperationalError(secret),
        )

        with pytest.raises(ProfileRepositoryError) as failed:
            await repository.create_profile(
                _draft("Rollback Create"),
                profile_id=profile_id,
            )
        _assert_safe_error(
            failed.value,
            "operation_failed",
            secret,
            str(database_path),
        )
        assert attempted.is_set()

        with pytest.raises(ProfileRepositoryError) as missing:
            await repository.get_profile(profile_id)
        _assert_safe_error(missing.value, "missing")
        recovered = (
            await repository.create_profile(
                _draft("Rollback Create"),
                profile_id=profile_id,
            )
        ).value
        assert recovered.revision == 1


@pytest.mark.asyncio
async def test_rollback_failure_quarantines_connection_and_retry_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "secret-rollback-store.sqlite3"
    profile_id = UUID("a5000000-0000-4000-8000-00000000000a")
    commit_secret = "secret-commit-before-rollback-failure"
    rollback_secret = "secret-rollback-failure"
    repository = profile_repository.TTSProfileRepository(
        database_path,
        _clock=lambda: CREATED_AT,
        _uuid_factory=lambda: GENERATED_ID,
    )
    await repository.open()
    try:
        _fail_next_commit(
            monkeypatch,
            repository,
            sqlite3.OperationalError(commit_secret),
        )

        def failed_rollback(_connection: sqlite3.Connection) -> None:
            raise sqlite3.OperationalError(rollback_secret)

        monkeypatch.setattr(
            repository,
            "_rollback_transaction",
            failed_rollback,
            raising=False,
        )

        with pytest.raises(ProfileRepositoryError) as failed:
            await repository.create_profile(
                _draft("Rollback Quarantine"),
                profile_id=profile_id,
            )
        _assert_safe_error(
            failed.value,
            "operation_failed",
            commit_secret,
            rollback_secret,
            str(database_path),
        )
        assert repository.state is ProfileRepositoryState.UNAVAILABLE

        with pytest.raises(ProfileRepositoryError) as unavailable:
            await repository.get_profile(profile_id)
        _assert_safe_error(unavailable.value, "unavailable")

        reopened = await repository.open()
        assert reopened.generation == 2
        assert repository.state is ProfileRepositoryState.OPEN
        with pytest.raises(ProfileRepositoryError) as missing:
            await repository.get_profile(profile_id)
        _assert_safe_error(missing.value, "missing")
        recovered = (
            await repository.create_profile(
                _draft("Rollback Quarantine"),
                profile_id=profile_id,
            )
        ).value
        assert recovered.revision == 1
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_update_failure_before_commit_rolls_back_and_repository_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "secret-update-store.sqlite3"
    profile_id = UUID("b0000000-0000-4000-8000-00000000000b")
    secret = "secret-update-commit-failure"
    clock = _SequenceCallable(iter((CREATED_AT, UPDATED_AT, LATER_AT)))

    async with _opened_repository(
        database_path,
        clock=cast(Callable[[], datetime], clock),
    ) as repository:
        original = (
            await repository.create_profile(
                _draft("Original"),
                profile_id=profile_id,
            )
        ).value
        attempted = _fail_next_commit(
            monkeypatch,
            repository,
            sqlite3.OperationalError(secret),
        )

        with pytest.raises(ProfileRepositoryError) as failed:
            await repository.update_profile(
                profile_id,
                original.revision,
                _draft("Must Roll Back", model_id="rolled-back-model"),
            )
        _assert_safe_error(
            failed.value,
            "operation_failed",
            secret,
            str(database_path),
        )
        assert attempted.is_set()
        assert (await repository.get_profile(profile_id)).value == original

        recovered = (
            await repository.update_profile(
                profile_id,
                original.revision,
                _draft("Recovered", model_id="recovered-model"),
            )
        ).value
        assert recovered.revision == 2
        assert recovered.display_name == "Recovered"
        assert recovered.updated_at == LATER_AT


@pytest.mark.asyncio
async def test_delete_failure_before_commit_rolls_back_and_repository_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "secret-delete-store.sqlite3"
    profile_id = UUID("c0000000-0000-4000-8000-00000000000c")
    secret = "secret-delete-commit-failure"

    async with _opened_repository(database_path) as repository:
        original = (
            await repository.create_profile(
                _draft("Survivor"),
                profile_id=profile_id,
            )
        ).value
        attempted = _fail_next_commit(
            monkeypatch,
            repository,
            sqlite3.OperationalError(secret),
        )

        with pytest.raises(ProfileRepositoryError) as failed:
            await repository.delete_profile(profile_id)
        _assert_safe_error(
            failed.value,
            "operation_failed",
            secret,
            str(database_path),
        )
        assert attempted.is_set()
        assert (await repository.get_profile(profile_id)).value == original

        assert (await repository.delete_profile(profile_id)).value is None
        with pytest.raises(ProfileRepositoryError) as missing:
            await repository.get_profile(profile_id)
        _assert_safe_error(missing.value, "missing")


@pytest.mark.asyncio
async def test_mutation_control_flow_is_preserved_after_transaction_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_id = UUID("d0000000-0000-4000-8000-00000000000d")
    signal = _ControlFlow()

    async with _opened_repository(tmp_path / "profiles.sqlite3") as repository:
        _fail_next_commit(monkeypatch, repository, signal)

        with pytest.raises(_ControlFlow) as caught:
            await repository.create_profile(
                _draft("Control Flow"),
                profile_id=profile_id,
            )

        assert caught.value is signal
        with pytest.raises(ProfileRepositoryError) as missing:
            await repository.get_profile(profile_id)
        _assert_safe_error(missing.value, "missing")
        assert (
            await repository.create_profile(
                _draft("Control Flow"),
                profile_id=profile_id,
            )
        ).value.profile_id == profile_id


@pytest.mark.asyncio
async def test_hostile_codec_failure_is_recreated_without_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "secret-codec-store.sqlite3"
    profile_id = UUID("e0000000-0000-4000-8000-00000000000e")
    secret = "secret-hostile-codec-context"

    async with _opened_repository(database_path) as repository:
        await repository.create_profile(_draft(), profile_id=profile_id)

        def hostile_decode(_row: object) -> TTSGenerationProfile:
            try:
                raise RuntimeError(secret)
            except RuntimeError:
                raise ValueError(secret)

        monkeypatch.setattr(profile_repository, "decode_profile", hostile_decode)

        with pytest.raises(ProfileRepositoryError) as failed:
            await repository.get_profile(profile_id)

        _assert_safe_error(
            failed.value,
            "operation_failed",
            secret,
            str(database_path),
        )


@pytest.mark.asyncio
async def test_two_open_repositories_resolve_sqlite_constraint_race_safely(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    first = profile_repository.TTSProfileRepository(
        database_path,
        _clock=lambda: CREATED_AT,
        _uuid_factory=lambda: GENERATED_ID,
    )
    second = profile_repository.TTSProfileRepository(
        database_path,
        _clock=lambda: CREATED_AT,
        _uuid_factory=lambda: CALLER_ID,
    )
    await first.open()
    await second.open()
    try:
        outcomes = await asyncio.gather(
            first.create_profile(
                _draft("Ｒａｃｅ Ｎａｍｅ"),
                profile_id=GENERATED_ID,
            ),
            second.create_profile(
                _draft("race name"),
                profile_id=CALLER_ID,
            ),
            return_exceptions=True,
        )

        successes = [
            outcome for outcome in outcomes if type(outcome) is ProfileStoreResult
        ]
        failures = [
            outcome for outcome in outcomes if type(outcome) is ProfileRepositoryError
        ]
        assert len(successes) == len(failures) == 1
        _assert_safe_error(cast(ProfileRepositoryError, failures[0]), "conflict")
        winning_page = (await first.list_profiles()).value
        assert winning_page.total == 1
        assert winning_page.profiles[0].normalized_name == "race name"
    finally:
        await first.close()
        await second.close()


@pytest.mark.asyncio
async def test_crud_sql_runs_only_on_repository_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    event_loop_thread = threading.get_ident()
    traced_threads: list[int] = []
    real_open = profile_repository.open_profile_store

    def traced_open(path: Path) -> sqlite3.Connection:
        connection = real_open(path)
        connection.set_trace_callback(
            lambda _statement: traced_threads.append(threading.get_ident())
        )
        return connection

    monkeypatch.setattr(profile_repository, "open_profile_store", traced_open)
    clock = _SequenceCallable(iter((CREATED_AT, UPDATED_AT)))

    async with _opened_repository(
        database_path,
        clock=cast(Callable[[], datetime], clock),
    ) as repository:
        created = (
            await repository.create_profile(
                _draft("Worker"),
                profile_id=GENERATED_ID,
            )
        ).value
        await repository.get_profile(created.profile_id)
        await repository.list_profiles(search="work", limit=1, offset=0)
        await repository.update_profile(
            created.profile_id,
            created.revision,
            _draft("Worker Updated"),
        )
        await repository.delete_profile(created.profile_id)

    assert traced_threads
    assert len(set(traced_threads)) == 1
    assert traced_threads[0] != event_loop_thread
