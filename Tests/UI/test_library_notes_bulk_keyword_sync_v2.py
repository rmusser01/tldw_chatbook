"""Bulk Add/Remove keyword goes through the Notes scope service (PR #3423 I-2).

The select-mode Actions… ▸ Add keyword… / Remove keyword… worker linked and
unlinked keywords straight through ``NotesInteropService``: plain database
link writes. The editor's keyword save goes through
``NotesScopeService._sync_local_note_keywords`` instead, which on a ready
Sync v2 local-first profile records the organization intents another device
syncs from -- a ``notes.keyword`` resource for a new keyword and a
``notes.keyword_link`` intent per link -- and which sits behind the scope
service's policy gate. So a bulk change read "Added 'travel' to 3 notes" and
never reached the user's other device, and skipped the policy gate the bulk
Move does not skip.

Two boots of the real ``LibraryScreen`` with the production stylesheet over
the bulk seed (a real ChaChaNotes database behind the real scope service,
``library_notes_bulk_support``):

- A ready Sync v2 local-first profile, wired by the app's own
  ``_wire_notes_sync_services``: a bulk add of a new keyword records one
  ``notes.keyword`` resource and a ``notes.keyword_link`` upsert per note; a
  bulk remove records a tombstone per note. Receipts as without Sync v2.
- The policy gate refusing local note updates: nothing is linked, no keyword
  is created, and the receipt counts every note as not changed, in the
  existing partial-failure style, with the notes still checked.

The receipts without Sync v2 stay pinned by
``test_library_notes_select_mode_counts_and_bulk_extended.py``.

Base (``bd01c0e46b``): the Sync v2 boot records no intent at all, and the
policy boot links all three notes ("Added 'food' to 3 notes").
"""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import pytest

from Tests.UI.library_notes_bulk_support import (
    enter_select_mode,
    mounted,
    notes_ready,
    painted,
    until,
)
from Tests.UI.test_library_notes_select_mode_counts_and_bulk_extended import (
    _check,
    _keyword,
)
from Tests.UI.test_library_notes_trash_pages_and_purges_extended import (
    _painted_toasts,
)
from tldw_chatbook.app import _wire_notes_sync_services
from tldw_chatbook.runtime_policy.types import PolicyDeniedError
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

WIDE = (160, 45)
PICKED = ("Ideas inbox", "Reading list", "Trip Oslo")
SERVER = "server-a"
DATASET = "dataset-server-a"


def _ready_sync_v2(seed, tmp_path) -> SyncStateRepository:
    """A ready local-first Sync v2 profile, wired the way the app wires it."""
    db = seed.profile.db
    state = SyncStateRepository(tmp_path / "sync-state.sqlite3", client_id="bulk-kw")
    state.set_sync_v2_profile_state(
        server_profile_id=SERVER,
        authenticated_principal_id=None,
        workspace_scope=None,
        profile_mode="local_first",
        device_id="device-a",
        dataset_id=DATASET,
    )
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO notes_organization_sync_checkpoints("
            "server_profile_id, dataset_id, local_state, server_state, "
            "inventory_phase, updated_at) VALUES "
            "(?, ?, 'ready', 'ready', 'complete', '2026-11-05T00:00:00+00:00')",
            (SERVER, DATASET),
        )
    _wire_notes_sync_services(
        SimpleNamespace(
            active_server_id=SERVER,
            chachanotes_db=db,
            sync_state_repository=state,
            sync_v2_dataset_keys={},
            notes_scope_service=seed.profile.scope_service,
            local_first_sync_service=SimpleNamespace(
                notes_organization_repository=None,
                notes_organization_sync_service=None,
            ),
            sync_restore_service=SimpleNamespace(notes_organization_repository=None),
        )
    )
    assert seed.profile.scope_service.organization_sync_service is not None
    return state


def _intents(seed) -> list[tuple[str, str, str]]:
    """Every organization intent as ``(domain, operation, subject)``, oldest
    first, through a second connection to the test's own database file."""
    with sqlite3.connect(seed.tmp_path / "notes.sqlite3") as connection:
        rows = connection.execute(
            "SELECT domain, operation, payload_json FROM "
            "notes_organization_sync_intents ORDER BY intent_sequence"
        ).fetchall()
    out = []
    for domain, operation, payload in rows:
        data = json.loads(payload or "{}")
        out.append(
            (domain, operation, str(data.get("subject_id") or data.get("keyword") or ""))
        )
    return out


class _DenyNoteUpdates:
    """The runtime policy refusing local note updates, and nothing else."""

    def __init__(self) -> None:
        self.refused: list[str] = []

    def require_allowed(self, *, action_id: str, **_kwargs) -> None:
        if action_id == "notes.update.local":
            self.refused.append(action_id)
            raise PolicyDeniedError(
                action_id=action_id,
                reason_code="authority_denied",
                user_message="Local note changes are not allowed right now.",
                effective_source="server",
                authority_owner="server",
            )


async def test_a_bulk_keyword_change_records_sync_v2_intents_per_note(
    tmp_path,
) -> None:
    host, seed = mounted(tmp_path)
    _ready_sync_v2(seed, tmp_path)
    ids = [seed.note_ids[title] for title in PICKED]
    async with host.run_test(size=WIDE) as pilot:
        screen = await notes_ready(host, pilot)
        before = _intents(seed)
        await enter_select_mode(host, pilot, screen)
        await _check(host, pilot, screen, PICKED)
        await _keyword(
            host,
            pilot,
            screen,
            "library-notes-select-action-add-keyword",
            "travel",
            "Add a keyword to 3 notes",
        )
        await until(
            pilot,
            lambda: "Added 'travel' to 3 notes" in painted(host),
            lambda: f"the add receipt: {painted(host)}",
        )
        assert seed.keyword_links("travel") == set(ids)
        added = _intents(seed)[len(before) :]
        # One resource for the new keyword, one link intent per note.
        assert [row for row in added if row[0] == "notes.keyword"] == [
            ("notes.keyword", "upsert", "travel")
        ], added
        assert sorted(
            row[2] for row in added if row[0] == "notes.keyword_link"
        ) == sorted(ids), added
        assert {row[1] for row in added if row[0] == "notes.keyword_link"} == {
            "upsert"
        }, added
        assert not screen._notes_state.select_mode

        await enter_select_mode(host, pilot, screen)
        await _check(host, pilot, screen, PICKED)
        marker = len(_intents(seed))
        await _keyword(
            host,
            pilot,
            screen,
            "library-notes-select-action-remove-keyword",
            "Travel",
            "Remove a keyword from 3 notes",
        )
        await until(
            pilot,
            lambda: "Removed 'travel' from 3 notes" in painted(host),
            lambda: f"the remove receipt: {painted(host)}",
        )
        assert seed.keyword_links("travel") == set()
        removed = _intents(seed)[marker:]
        assert sorted(removed) == sorted(
            ("notes.keyword_link", "tombstone", note_id) for note_id in ids
        ), removed


async def test_a_bulk_keyword_change_the_policy_refuses_changes_nothing(
    tmp_path,
) -> None:
    host, seed = mounted(tmp_path)
    async with host.run_test(size=WIDE, notifications=True) as pilot:
        screen = await notes_ready(host, pilot)
        # The Library notifies through ``app_instance``; route its toasts to
        # the running host so they paint.
        screen.app_instance.notify = host.notify
        policy = _DenyNoteUpdates()
        seed.profile.scope_service.policy_enforcer = policy
        before = seed.keyword_state()
        await enter_select_mode(host, pilot, screen)
        await _check(host, pilot, screen, PICKED)
        await _keyword(
            host,
            pilot,
            screen,
            "library-notes-select-action-add-keyword",
            "food",
            "Add a keyword to 3 notes",
        )
        # PR #3423 re-review N-3: the refusal's own sentence follows the count.
        receipt = (
            "Added 'food' to 0 notes · 3 could not be changed — still checked, "
            "so you can try again. Local note changes are not allowed right now."
        )
        await until(
            pilot,
            lambda: any(receipt in toast for toast in _painted_toasts(host)),
            lambda: f"the refused receipt: {_painted_toasts(host)!r} "
            f"(painted: {painted(host)})",
        )
        assert policy.refused == ["notes.update.local"] * 3
        assert seed.keyword_state() == before
        assert seed.keyword_links("food") == set()
        # The existing partial-failure shape: select mode kept, the refused
        # notes still checked.
        assert screen._notes_state.select_mode
        assert screen._notes_state.row_selection.count == 3
