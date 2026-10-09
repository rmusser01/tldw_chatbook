"""A CRLF file without a final newline survives its note shrinking to one line (TASK-34000.48).

A text with no line ending at all carries no evidence about its line-ending
convention: ``_parse_supported_text`` reports it as ``lf`` because the profile
has no "indeterminate" value. A synced CRLF file whose note is edited down to
one newline-less line was therefore re-observed as ``lf`` right after its own
correct write, and the FIRST fence was the filesystem's own post-write check:
``PosixNotesSyncFilesystem.replace`` raised a PARTIAL
``replacement_postcondition_failed`` whose cleanup handle named the real file
with no private identity. Recovery then refused that handle
(``recovery_authority_changed``) and Check refused the open entry: the folder
was held for good, with the written file never deleted.

The rule now: an observation whose logical text has no ``"\\n"`` proves nothing
about ``newline`` and inherits the recorded convention
(``notes_sync_filesystem.proven_profile``); ``final_newline``, ``utf8_bom`` and
``mode`` are always determinate and are never carried. Every profile comparison
and every binding commit goes through it, so the recorded CRLF convention is
never flipped by such an observation and the next multi-line write is CRLF.

Everything here is the production stack -- the runtime owner, the real executor
and POSIX filesystem, a real ChaChaNotes database and a real ``.md`` in a temp
vault. The bytes on disk and the rows in the stores are the evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from Tests.Notes.notes_sync_tail_edit_support import Vault, build_owner
from tldw_chatbook.Notes import notes_sync_filesystem
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.notes_device_state_store import (
    NotesDeviceStateStore,
    NotesSyncBindingRecord,
)
from tldw_chatbook.Notes.notes_scope_service import ScopeType
from tldw_chatbook.Notes.notes_sync_authority import NotesScopeSyncAuthority
from tldw_chatbook.Notes.notes_sync_executor import (
    NotesSyncExecutionRequest,
    NotesSyncExecutor,
)
from tldw_chatbook.Notes.notes_sync_filesystem import (
    NotesSyncFilesystemPartialError,
    PosixNotesSyncFilesystem,
)
from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncActionKind,
    NotesSyncDirection,
    NotesSyncOperationState,
    NotesSyncSerializationProfile,
)
from tldw_chatbook.Notes.notes_sync_reconciler import ManagedPlacementEffectKind
from tldw_chatbook.Notes.notes_sync_runtime import NotesSyncRuntimeOwner

pytestmark = [pytest.mark.unit, pytest.mark.asyncio, pytest.mark.bootstrap_profile]

#: The note's logical text (LF) ...
NOTE_TEXT = "line one\nline two\nline three"
#: ... and the file as a Windows editor leaves it: CRLF, no final newline.
CRLF_BYTES = b"line one\r\nline two\r\nline three"
#: An ordinary multi-line edit: round-trips under the file's own convention.
MULTI_LINE_EDIT = "line one\nline two edited\nline three"
#: The corner: one line, no newline anywhere -- nothing to observe a convention from.
SINGLE_LINE = "just one line"
#: The edit after the corner: must come out CRLF (AC#2).
BACK_TO_TWO_LINES = "back to\ntwo lines"


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def vault(tmp_path: Path):
    selected = Vault(tmp_path, NOTE_TEXT, file_bytes=CRLF_BYTES)
    try:
        yield selected
    finally:
        selected.close()


def _root(owner: NotesSyncRuntimeOwner, root_id: str = "root-1"):
    return next(root for root in owner.snapshot().roots if root.root_id == root_id)


def _binding(vault: Vault, binding_id: str = "binding-1") -> NotesSyncBindingRecord:
    store = NotesDeviceStateStore(vault.state_path)
    try:
        return store.get_binding(binding_id)
    finally:
        store.close()


def _recovery_metadata(vault: Vault, operation_id: str) -> dict:
    store = NotesDeviceStateStore(vault.state_path)
    try:
        return json.loads(
            store.load_operation_recovery(operation_id).metadata.decode("utf-8")
        )
    finally:
        store.close()


async def _save(vault: Vault, owner: NotesSyncRuntimeOwner, text: str) -> None:
    vault.edit_note(text)
    assert await owner.note_changed("note-1") == ("root-1",)
    await owner.settle()


def _healthy(vault: Vault, owner: NotesSyncRuntimeOwner, file_bytes: bytes) -> None:
    """No open entry, nothing held, the file holds ``file_bytes``, CRLF recorded."""

    assert vault.file.read_bytes() == file_bytes
    assert vault.incomplete() == [], "an operation was left open for Recovery"
    root = _root(owner)
    assert (root.status, root.next_action, root.action_id) == (
        "up_to_date",
        "sync_now",
        None,
    ), "the folder was left held"
    binding = _binding(vault)
    assert binding.serialization.newline == "crlf", (
        "the file's recorded line-ending convention was flipped"
    )
    assert binding.serialization.final_newline is False
    # One baseline digest serves both sides: the LOGICAL text's.
    assert binding.content_digest == _digest(file_bytes.replace(b"\r\n", b"\n"))


def _refresh_held(vault: Vault, owner: NotesSyncRuntimeOwner, plan) -> bool:
    """The plan names a representation refresh for the pair, no content action,
    and the root is held for review on it -- the base's behaviour for a real
    representation change, which the newline rule must leave alone."""

    effects = [(effect.kind, effect.binding_id) for effect in plan.managed_placement_effects]
    kinds = [action.kind for action in plan.safe_actions]
    root = _root(owner)
    return (
        effects == [(ManagedPlacementEffectKind.REPRESENTATION_REFRESH, "binding-1")]
        and kinds == [NotesSyncActionKind.NO_CHANGE]
        and plan.attention == ()
        and (root.status, root.next_action) == ("needs_attention", "review_changes")
        and vault.incomplete() == []
    )


# --- AC#1 / AC#4: the shrink to one line syncs and the folder stays healthy ------


async def test_a_crlf_file_without_a_final_newline_survives_a_single_line_note(
    vault: Vault,
) -> None:
    owner = build_owner(vault)
    await owner.start()
    try:
        assert _binding(vault).serialization == NotesSyncSerializationProfile(
            utf8_bom=False, newline="crlf", final_newline=False, mode=0o644
        )
        # The ordinary case first: a multi-line edit round-trips as CRLF.
        await _save(vault, owner, MULTI_LINE_EDIT)
        _healthy(vault, owner, b"line one\r\nline two edited\r\nline three")

        await _save(vault, owner, SINGLE_LINE)

        _healthy(vault, owner, b"just one line")
    finally:
        await owner.shutdown()


# --- AC#2: the convention is never flipped by a newline-free observation ---------


async def test_the_next_multi_line_edit_writes_crlf(vault: Vault) -> None:
    owner = build_owner(vault)
    await owner.start()
    try:
        await _save(vault, owner, SINGLE_LINE)
        _healthy(vault, owner, b"just one line")

        await _save(vault, owner, BACK_TO_TWO_LINES)

        _healthy(vault, owner, b"back to\r\ntwo lines")
    finally:
        await owner.shutdown()


async def test_a_relaunch_after_the_shrink_still_writes_crlf(vault: Vault) -> None:
    """The recorded convention survives a restart: nothing in memory carries it."""

    owner = build_owner(vault)
    await owner.start()
    try:
        await _save(vault, owner, SINGLE_LINE)
        _healthy(vault, owner, b"just one line")
    finally:
        await owner.shutdown()

    relaunched = build_owner(vault)
    await relaunched.start()
    try:
        await _save(vault, relaunched, BACK_TO_TWO_LINES)
        _healthy(vault, relaunched, b"back to\r\ntwo lines")
    finally:
        await relaunched.shutdown()


# --- The reconciler trace: the next pass holds nothing ---------------------------


async def test_the_next_pass_after_a_single_line_write_does_not_hold_the_root(
    vault: Vault,
) -> None:
    owner = build_owner(vault)
    await owner.start()
    try:
        await _save(vault, owner, SINGLE_LINE)

        plan = await owner.check_root("root-1")

        assert plan.attention == ()
        assert plan.managed_placement_effects == (), (
            "the planner raised a representation refresh over the one-line file"
        )
        assert [action.kind for action in plan.safe_actions] == [
            NotesSyncActionKind.NO_CHANGE
        ]
        assert plan.deletion_groups == ()
        _healthy(vault, owner, b"just one line")
    finally:
        await owner.shutdown()


# --- Negative controls: a genuine representation change is still one -----------


async def test_a_crlf_to_lf_rewrite_with_line_endings_present_is_still_a_change(
    vault: Vault,
) -> None:
    """The text carries newlines, so its convention IS observed: LF != CRLF."""

    owner = build_owner(vault)
    await owner.start()
    try:
        vault.file.write_bytes(NOTE_TEXT.encode("utf-8"))
        assert owner.schedule_hint("root-1") is not None

        plan = await owner.check_root("root-1")

        assert _refresh_held(vault, owner, plan)
        assert vault.file.read_bytes() == NOTE_TEXT.encode("utf-8")
        assert _binding(vault).serialization.newline == "crlf"
    finally:
        await owner.shutdown()


async def test_a_mode_change_on_a_newline_free_file_is_never_masked(
    vault: Vault,
) -> None:
    """FAT32/SMB shape (Review Focus 4): ``mode`` is never inherited.

    After the shrink the file's newline is unproven and inherited; its mode is
    a fact. A mode that differs from the recorded one is still a
    representation change the planner surfaces, not something the helper hides.
    """

    owner = build_owner(vault)
    await owner.start()
    try:
        await _save(vault, owner, SINGLE_LINE)
        _healthy(vault, owner, b"just one line")
        os.chmod(vault.file, 0o600)
        assert stat.S_IMODE(vault.file.stat().st_mode) == 0o600
        assert owner.schedule_hint("root-1") is not None

        plan = await owner.check_root("root-1")

        assert _refresh_held(vault, owner, plan)
        assert _binding(vault).serialization.mode == 0o644
    finally:
        await owner.shutdown()


@pytest.mark.parametrize(
    ("observed", "text", "recorded", "expected"),
    [
        pytest.param(
            NotesSyncSerializationProfile(False, "lf", False, 0o644),
            "just one line",
            NotesSyncSerializationProfile(False, "crlf", False, 0o644),
            NotesSyncSerializationProfile(False, "crlf", False, 0o644),
            id="no-line-ending-inherits-newline",
        ),
        pytest.param(
            NotesSyncSerializationProfile(False, "lf", False, 0o644),
            "two\nlines",
            NotesSyncSerializationProfile(False, "crlf", False, 0o644),
            NotesSyncSerializationProfile(False, "lf", False, 0o644),
            id="a-line-ending-is-proof",
        ),
        pytest.param(
            NotesSyncSerializationProfile(False, "lf", True, 0o644),
            "one line\n",
            NotesSyncSerializationProfile(False, "crlf", True, 0o644),
            NotesSyncSerializationProfile(False, "lf", True, 0o644),
            id="a-final-newline-alone-is-proof",
        ),
        pytest.param(
            NotesSyncSerializationProfile(False, "lf", False, 0o644),
            "just one line",
            None,
            NotesSyncSerializationProfile(False, "lf", False, 0o644),
            id="nothing-recorded-nothing-inherited",
        ),
        pytest.param(
            NotesSyncSerializationProfile(False, "lf", False, 0o600),
            "just one line",
            NotesSyncSerializationProfile(False, "crlf", False, 0o644),
            NotesSyncSerializationProfile(False, "crlf", False, 0o600),
            id="mode-is-never-carried",
        ),
        pytest.param(
            NotesSyncSerializationProfile(True, "lf", False, 0o644),
            "just one line",
            NotesSyncSerializationProfile(False, "crlf", False, 0o644),
            NotesSyncSerializationProfile(True, "crlf", False, 0o644),
            id="bom-is-never-carried",
        ),
        pytest.param(
            NotesSyncSerializationProfile(False, "lf", False, 0o644),
            "just one line",
            NotesSyncSerializationProfile(False, "crlf", True, 0o644),
            NotesSyncSerializationProfile(False, "crlf", False, 0o644),
            id="final-newline-is-never-carried",
        ),
        pytest.param(
            NotesSyncSerializationProfile(False, "lf", False, 0o644),
            "",
            NotesSyncSerializationProfile(False, "crlf", False, 0o644),
            NotesSyncSerializationProfile(False, "crlf", False, 0o644),
            id="empty-text-inherits-newline",
        ),
    ],
)
def test_proven_profile_carries_newline_only(
    observed: NotesSyncSerializationProfile,
    text: str,
    recorded: NotesSyncSerializationProfile | None,
    expected: NotesSyncSerializationProfile,
) -> None:
    assert notes_sync_filesystem.proven_profile(observed, text, recorded) == expected


def test_the_filesystem_post_write_check_keeps_the_reviewed_newline(
    tmp_path: Path,
) -> None:
    """The first fence on dev: ``replace`` itself. Now it proves the bytes."""

    root = tmp_path / "root"
    root.mkdir()
    target = root / "note.md"
    target.write_bytes(CRLF_BYTES)
    with PosixNotesSyncFilesystem(root) as filesystem:
        before = filesystem.observe("note.md")
        assert before.observation.serialization.newline == "crlf"

        after = filesystem.replace("note.md", SINGLE_LINE, expected=before)

        assert target.read_bytes() == b"just one line"
        assert after.recovery_bytes == CRLF_BYTES
        # And a write asked for under the recorded profile over that
        # newline-free file comes out CRLF.
        again = filesystem.replace(
            "note.md",
            BACK_TO_TWO_LINES,
            expected=after,
            profile=before.observation.serialization,
        )
        assert target.read_bytes() == b"back to\r\ntwo lines"
        assert again.observation.serialization.newline == "crlf"


def test_the_filesystem_create_check_keeps_the_candidate_newline(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    with PosixNotesSyncFilesystem(root) as filesystem:
        created = filesystem.create(
            "new.md",
            SINGLE_LINE,
            profile=NotesSyncSerializationProfile(False, "crlf", False, 0o644),
        )
    assert (root / "new.md").read_bytes() == b"just one line"
    assert created.observation.serialization.final_newline is False


def test_the_filesystem_refuses_a_write_profile_the_reviewed_file_cannot_carry(
    tmp_path: Path,
) -> None:
    """A caller may only name a profile the reviewed state proves: never a
    different mode, BOM or final-newline rule, and never a different newline
    when the file carries line endings."""

    root = tmp_path / "root"
    root.mkdir()
    target = root / "note.md"
    target.write_bytes(CRLF_BYTES)
    with PosixNotesSyncFilesystem(root) as filesystem:
        before = filesystem.observe("note.md")
        for unprovable in (
            NotesSyncSerializationProfile(False, "lf", False, 0o644),
            NotesSyncSerializationProfile(False, "crlf", False, 0o600),
            NotesSyncSerializationProfile(True, "crlf", False, 0o644),
            NotesSyncSerializationProfile(False, "crlf", True, 0o644),
        ):
            with pytest.raises(
                notes_sync_filesystem.NotesSyncFilesystemError
            ) as raised:
                filesystem.replace(
                    "note.md", SINGLE_LINE, expected=before, profile=unprovable
                )
            assert raised.value.reason_code == "replacement_profile_unproven"
            assert not isinstance(raised.value, NotesSyncFilesystemPartialError)
    assert target.read_bytes() == CRLF_BYTES, "nothing was written"


# --- AC#5: a root already wedged on dev by this corner heals through Recovery ----


async def _wedge_as_dev_does(
    vault: Vault, owner: NotesSyncRuntimeOwner, monkeypatch: pytest.MonkeyPatch
) -> str:
    """Shrink the note with the newline rule disabled; return the open entry's id.

    This is the exact durable state dev leaves: the write landed, the
    filesystem's own post-write check raised a PARTIAL
    ``replacement_postcondition_failed`` with a cleanup handle naming the real
    file and no identity, and the binding still carries the old baseline.
    """

    with monkeypatch.context() as patch:
        # On a build that still ships the raw comparison (the RED run) the
        # defect wedges the root by itself; nothing needs disabling.
        if hasattr(notes_sync_filesystem, "proven_profile"):
            patch.setattr(
                notes_sync_filesystem,
                "proven_profile",
                lambda observed, _text, _recorded: observed,
            )
        await _save(vault, owner, SINGLE_LINE)
    assert vault.file.read_bytes() == b"just one line", "the write landed"
    assert vault.incomplete() == [
        ("update_file", "needs_attention", "replacement_postcondition_failed")
    ]
    root = _root(owner)
    assert (root.status, root.next_action) == ("needs_attention", "resolve_cleanup")
    assert root.action_id is not None
    metadata = _recovery_metadata(vault, root.action_id)
    assert metadata["cleanup_pending"] is True
    assert metadata["cleanup_relative_path"] == "quotes.md"
    assert metadata["cleanup_reason_code"] == "replacement_postcondition_failed"
    assert metadata["cleanup_identity"] is None
    binding = _binding(vault)
    assert binding.content_digest == _digest(NOTE_TEXT.encode("utf-8"))
    assert binding.serialization.newline == "crlf"
    with pytest.raises(RuntimeError, match="sync_recovery_unresolved"):
        await owner.check_root("root-1")
    return root.action_id


async def test_a_root_wedged_on_dev_by_this_corner_heals_through_recovery(
    vault: Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = build_owner(vault)
    await owner.start()
    try:
        operation_id = await _wedge_as_dev_does(vault, owner, monkeypatch)

        await owner.resolve_cleanup("root-1", operation_id)

        # Settled at the proven baseline, re-planned to healthy, nothing
        # rewritten and nothing deleted.
        _healthy(vault, owner, b"just one line")
        assert vault.note()["content"] == SINGLE_LINE
        assert vault.file.exists()
        # Sync is alive again in both directions and the convention held.
        await _save(vault, owner, BACK_TO_TWO_LINES)
        _healthy(vault, owner, b"back to\r\ntwo lines")
        vault.file.write_bytes(b"back to\r\ntwo lines\r\nand a disk edit")
        assert owner.schedule_hint("root-1") is not None
        await owner.settle()
        assert vault.note()["content"] == "back to\ntwo lines\nand a disk edit"
        assert vault.incomplete() == []
    finally:
        await owner.shutdown()


async def test_a_wedged_root_whose_file_changed_meanwhile_settles_to_a_review(
    vault: Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two-sided control: the heal never picks a winner.

    The file was edited on disk after the wedge and the note moved on too.
    Recovery settles the entry without proving the write (the bytes are not
    ours), keeps the reviewed baseline, and the next plan is an ordinary
    ``both_sides_changed`` review with nothing written either way.
    """

    owner = build_owner(vault)
    await owner.start()
    try:
        operation_id = await _wedge_as_dev_does(vault, owner, monkeypatch)
        vault.file.write_bytes(b"edited on disk while held")
        vault.edit_note("edited in Notes while held")
        assert await owner.note_changed("note-1") == ()

        await owner.resolve_cleanup("root-1", operation_id)

        assert vault.incomplete() == []
        root = _root(owner)
        assert (root.status, root.next_action) == ("needs_attention", "review_changes")
        assert vault.file.read_bytes() == b"edited on disk while held"
        assert vault.note()["content"] == "edited in Notes while held"
        plan = await owner.check_root("root-1")
        assert [(item.kind.value, item.reason_code) for item in plan.attention] == [
            ("conflict", "both_sides_changed")
        ]
        assert plan.safe_actions == ()
    finally:
        await owner.shutdown()


# --- Fix round 1 (review Important 2): the landed-target rule is scoped ---------
#
# ``_cleanup_names_the_landed_target`` applies to the two kinds a settle can
# close. A create_file whose post-write check raised the same handle shape
# keeps the Recovery path 9413696ea1 has, exactly: the cleanup stays pending
# and Recovery refuses with ``recovery_authority_changed``. The production
# runtime plans a CREATE_FILE only from a stored CANDIDATE row, which the
# executor's ``_require_new_candidate_owner`` then refuses, so this pin runs
# the real executor over the real store, filesystem and note authority
# (the shape ``test_notes_sync_tail_edit.py`` uses for its executor pin).


def _disabled_rule(monkeypatch: pytest.MonkeyPatch):
    """Context: the newline rule off (a no-op on a build without it)."""

    patch = monkeypatch.context()
    context = patch.__enter__()
    if hasattr(notes_sync_filesystem, "proven_profile"):
        context.setattr(
            notes_sync_filesystem,
            "proven_profile",
            lambda observed, _text, _recorded: observed,
        )
    return patch


async def _executor_wedge(
    vault: Vault,
    monkeypatch: pytest.MonkeyPatch,
    *,
    kind: NotesSyncActionKind,
):
    """Run one request with the rule off; return (executor, store, filesystem, result)."""

    if kind is NotesSyncActionKind.CREATE_FILE:
        assert vault.database.add_note("second", SINGLE_LINE, "note-2") == "note-2"
        LocalNoteFolderRepository(vault.database).reconcile_managed(
            owner_id="root-1",
            desired=(("folder-1", "note-1"), ("folder-1", "note-2")),
        )
    else:
        vault.edit_note(SINGLE_LINE)
    store = NotesDeviceStateStore(vault.state_path)
    filesystem = PosixNotesSyncFilesystem(vault.root)
    filesystem.__enter__()
    authority = NotesScopeSyncAuthority(
        vault.scope_service,
        scope=ScopeType.LOCAL_NOTE,
        user_id="user-1",
        note_scope_id="local_note",
    )
    executor = NotesSyncExecutor(
        store, authority, filesystem, recovery_capacity_bytes=1024 * 1024
    )
    if kind is NotesSyncActionKind.CREATE_FILE:
        note = await authority.observe("note-2")
        request = NotesSyncExecutionRequest(
            operation_id="operation-create",
            root_id="root-1",
            logical_folder_id="folder-1",
            direction=NotesSyncDirection.BIDIRECTIONAL,
            binding_id="binding-2",
            observation_token="observation-create",
            action_kind=NotesSyncActionKind.CREATE_FILE,
            note=note,
            file=None,
            desired_title=note.title,
            recovery_id="recovery-operation-create",
            recovery_expires_at=2**62,
            candidate_relative_path="second.md",
            candidate_serialization=NotesSyncSerializationProfile(
                False, "crlf", False, 0o644
            ),
        )
    else:
        note = await authority.observe("note-1")
        request = NotesSyncExecutionRequest(
            operation_id="operation-update",
            root_id="root-1",
            logical_folder_id="folder-1",
            direction=NotesSyncDirection.BIDIRECTIONAL,
            binding_id="binding-1",
            observation_token="observation-update",
            action_kind=NotesSyncActionKind.UPDATE_FILE,
            note=note,
            file=filesystem.observe("quotes.md"),
            desired_title=note.title,
            recovery_id="recovery-operation-update",
            recovery_expires_at=2**62,
        )
    patch = _disabled_rule(monkeypatch)
    try:
        result = await executor.execute(request)
    finally:
        patch.__exit__(None, None, None)
    assert (result.state, result.reason_code) == (
        NotesSyncOperationState.NEEDS_ATTENTION,
        "replacement_postcondition_failed",
    )
    metadata = _recovery_metadata(vault, request.operation_id)
    assert metadata["cleanup_pending"] is True
    assert metadata["cleanup_identity"] is None
    assert metadata["cleanup_relative_path"] == metadata["file_relative_path"]
    return executor, store, filesystem, request.operation_id


async def test_a_create_wedged_by_the_same_handle_keeps_todays_recovery_path(
    vault: Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identical on 9413696ea1: the create's cleanup is still pending and
    Recovery refuses it; the created file is never deleted."""

    executor, store, filesystem, operation_id = await _executor_wedge(
        vault, monkeypatch, kind=NotesSyncActionKind.CREATE_FILE
    )
    try:
        created = vault.root / "second.md"
        assert created.read_bytes() == b"just one line"
        assert vault.incomplete() == [
            ("create_file", "needs_attention", "replacement_postcondition_failed")
        ]

        assert executor.cleanup_pending(operation_id) is True
        with pytest.raises(RuntimeError, match="recovery_authority_changed"):
            await executor.resolve_filesystem_cleanup(operation_id)
        with pytest.raises(RuntimeError, match="recovery_authority_changed"):
            await executor.settle_attention(operation_id)

        assert vault.incomplete() == [
            ("create_file", "needs_attention", "replacement_postcondition_failed")
        ]
        assert created.read_bytes() == b"just one line"
        assert executor.cleanup_pending(operation_id) is True
    finally:
        filesystem.__exit__(None, None, None)
        store.close()


async def test_an_update_wedged_by_the_same_handle_is_settled_not_cleaned_up(
    vault: Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The settleable twin at the executor level (AC#5): no cleanup pending,
    the settle proves the bytes and closes the entry; nothing deleted."""

    executor, store, filesystem, operation_id = await _executor_wedge(
        vault, monkeypatch, kind=NotesSyncActionKind.UPDATE_FILE
    )
    try:
        assert vault.file.read_bytes() == b"just one line"
        assert executor.cleanup_pending(operation_id) is False

        result = await executor.settle_attention(operation_id)

        assert result.state is NotesSyncOperationState.COMPLETED
        assert vault.incomplete() == []
        assert vault.file.read_bytes() == b"just one line"
        binding = store.get_binding("binding-1")
        assert binding.serialization.newline == "crlf"
        assert binding.content_digest == _digest(b"just one line")
    finally:
        filesystem.__exit__(None, None, None)
        store.close()
