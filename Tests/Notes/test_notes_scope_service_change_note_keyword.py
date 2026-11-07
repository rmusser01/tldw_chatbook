"""``NotesScopeService.change_local_note_keyword``: one keyword, one note.

PR #3423 review I-2. The Library's bulk Add/Remove keyword now changes each
note through this seam, which applies the note's current keyword set plus or
minus one keyword through the editor's own ``_sync_local_note_keywords``:
the policy gate first, then -- on a ready Sync v2 local-first profile -- the
organization intents another device syncs from.

Real stores throughout: a ``CharactersRAGDB`` behind the real
``NotesInteropService``, and for Sync v2 a real ``SyncStateRepository`` wired
by the app's own ``_wire_notes_sync_services``. The mounted bulk arms are in
``Tests/UI/test_library_notes_bulk_keyword_sync_v2.py``.

TASK-34000.116: ``resolve_local_note_keyword_profile`` resolves the Sync v2
profile once, and checks its group's readiness, for a run of changes that pass
it back as ``resolved_profile``; those changes read no profile at all. The
default (no resolution passed) still resolves per change, as the editor's save
does.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.app import _wire_notes_sync_services
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.runtime_policy.types import PolicyDeniedError
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository

#: The interop service opens its per-user database through the real
#: source-bound config readers.
pytestmark = pytest.mark.bootstrap_profile

USER = "default_user"
SERVER = "server-a"
DATASET = "dataset-server-a"


class _Profile:
    def __init__(self, tmp_path: Path) -> None:
        self.db = CharactersRAGDB(tmp_path / "notes.sqlite3", client_id="kw-seam")
        self.interop = NotesInteropService(
            base_db_directory=tmp_path,
            api_client_id="kw-seam",
            global_db_to_use=self.db,
        )
        self.service = NotesScopeService(
            local_notes_service=self.interop, server_service=None
        )
        self.note_id = self.db.add_note("Trip Oslo", "itinerary")
        self.db.link_note_to_keyword(self.note_id, self.db.add_keyword("planning"))

    def change(
        self, keyword: str, *, remove: bool = False, **resolved: object
    ) -> bool:
        with self.interop.note_transaction(USER) as cursor:
            return self.service.change_local_note_keyword(
                user_id=USER,
                note_id=self.note_id,
                keyword=keyword,
                remove=remove,
                cursor=cursor,
                **resolved,
            )

    def count_profile_reads(self) -> list[str]:
        """Record every ``SyncStateRepository`` profile read, by name."""
        reads: list[str] = []
        for name in ("list_sync_v2_profile_states", "get_sync_v2_profile_state"):
            method = getattr(self.state, name)

            def counted(*args, _name=name, _method=method, **kwargs):
                reads.append(_name)
                return _method(*args, **kwargs)

            setattr(self.state, name, counted)
        return reads

    def keywords(self) -> list[str]:
        return sorted(
            row["keyword"] for row in self.db.get_keywords_for_note(self.note_id)
        )

    def intents(self) -> list[tuple[str, str]]:
        return [
            (row["domain"], row["operation"])
            for row in self.db.get_connection()
            .execute(
                "SELECT domain, operation FROM notes_organization_sync_intents "
                "ORDER BY intent_sequence"
            )
            .fetchall()
        ]

    def wire_sync_v2(self, tmp_path: Path, *, ready: bool = True) -> None:
        state = self.state = SyncStateRepository(
            tmp_path / "sync-state.sqlite3", client_id="kw"
        )
        state.set_sync_v2_profile_state(
            server_profile_id=SERVER,
            authenticated_principal_id=None,
            workspace_scope=None,
            profile_mode="local_first",
            device_id="device-a",
            dataset_id=DATASET,
        )
        with self.db.transaction() as cursor:
            cursor.execute(
                "INSERT INTO notes_organization_sync_checkpoints("
                "server_profile_id, dataset_id, local_state, server_state, "
                "inventory_phase, updated_at) VALUES "
                "(?, ?, ?, 'ready', 'complete', '2026-11-05T00:00:00+00:00')",
                (SERVER, DATASET, "ready" if ready else "pulling"),
            )
        _wire_notes_sync_services(
            SimpleNamespace(
                active_server_id=SERVER,
                chachanotes_db=self.db,
                sync_state_repository=state,
                sync_v2_dataset_keys={},
                notes_scope_service=self.service,
                local_first_sync_service=SimpleNamespace(
                    notes_organization_repository=None,
                    notes_organization_sync_service=None,
                ),
                sync_restore_service=SimpleNamespace(
                    notes_organization_repository=None
                ),
            )
        )
        assert self.service.organization_sync_service is not None


def test_without_sync_v2_one_keyword_changes_and_the_rest_stay(tmp_path) -> None:
    profile = _Profile(tmp_path)

    assert profile.change("travel") is True
    assert profile.keywords() == ["planning", "travel"]
    # Matching ignores case: already there.
    assert profile.change("Travel") is False
    assert profile.change("TRAVEL", remove=True) is True
    assert profile.keywords() == ["planning"]
    assert profile.change("travel", remove=True) is False
    assert profile.keywords() == ["planning"]
    assert profile.intents() == []


def test_on_a_ready_sync_v2_profile_each_change_records_its_intents(
    tmp_path,
) -> None:
    profile = _Profile(tmp_path)
    profile.wire_sync_v2(tmp_path)

    assert profile.change("travel") is True
    assert profile.intents() == [
        ("notes.keyword", "upsert"),
        ("notes.keyword_link", "upsert"),
    ]
    assert profile.change("travel", remove=True) is True
    assert profile.intents()[2:] == [("notes.keyword_link", "tombstone")]
    # The note's other keyword was never touched.
    assert profile.keywords() == ["planning"]
    assert profile.change("travel", remove=True) is False
    assert len(profile.intents()) == 3


def test_a_sync_v2_group_that_is_not_ready_refuses_without_a_write(tmp_path) -> None:
    profile = _Profile(tmp_path)
    profile.wire_sync_v2(tmp_path, ready=False)

    with pytest.raises(ValueError, match="not ready"):
        profile.change("travel")
    assert profile.keywords() == ["planning"]
    assert profile.intents() == []


def test_the_policy_gate_refuses_before_anything_is_read_or_written(
    tmp_path,
) -> None:
    profile = _Profile(tmp_path)
    asked: list[str] = []

    class _Deny:
        def require_allowed(self, *, action_id: str, **_kwargs) -> None:
            asked.append(action_id)
            raise PolicyDeniedError(
                action_id=action_id,
                reason_code="authority_denied",
                user_message="Local note changes are not allowed right now.",
                effective_source="server",
                authority_owner="server",
            )

    profile.service.policy_enforcer = _Deny()
    with pytest.raises(PolicyDeniedError):
        profile.change("travel")
    assert asked == ["notes.update.local"]
    assert profile.keywords() == ["planning"]
    assert profile.db.get_keyword_by_text("travel") is None


def test_a_resolved_profile_spares_every_change_its_profile_reads(tmp_path) -> None:
    profile = _Profile(tmp_path)
    profile.wire_sync_v2(tmp_path)
    reads = profile.count_profile_reads()

    # The default path, the editor's: each change resolves the profile itself.
    assert profile.change("travel") is True
    per_change = len(reads)
    assert per_change > 0, reads

    reads.clear()
    resolved = profile.service.resolve_local_note_keyword_profile(user_id=USER)
    once = len(reads)
    assert once > 0, reads
    reads.clear()
    assert profile.change("travel", remove=True, resolved_profile=resolved) is True
    assert profile.change("food", resolved_profile=resolved) is True
    assert profile.change("food", remove=True, resolved_profile=resolved) is True
    assert reads == []
    # The same intents as the default path records.
    assert profile.intents() == [
        ("notes.keyword", "upsert"),
        ("notes.keyword_link", "upsert"),
        ("notes.keyword_link", "tombstone"),
        ("notes.keyword", "upsert"),
        ("notes.keyword_link", "upsert"),
        ("notes.keyword_link", "tombstone"),
    ]
    assert profile.keywords() == ["planning"]


def test_resolving_for_a_group_that_is_not_ready_refuses_before_any_write(
    tmp_path,
) -> None:
    profile = _Profile(tmp_path)
    profile.wire_sync_v2(tmp_path, ready=False)

    with pytest.raises(ValueError, match="not ready"):
        profile.service.resolve_local_note_keyword_profile(user_id=USER)
    assert profile.keywords() == ["planning"]
    assert profile.intents() == []


def test_a_group_that_stops_being_ready_mid_run_refuses_without_a_profile_read(
    tmp_path,
) -> None:
    profile = _Profile(tmp_path)
    profile.wire_sync_v2(tmp_path)
    resolved = profile.service.resolve_local_note_keyword_profile(user_id=USER)
    reads = profile.count_profile_reads()
    assert profile.change("travel", resolved_profile=resolved) is True
    with profile.db.transaction() as cursor:
        cursor.execute(
            "UPDATE notes_organization_sync_checkpoints SET local_state = "
            "'pulling' WHERE server_profile_id = ? AND dataset_id = ?",
            (SERVER, DATASET),
        )

    # Each change re-checks readiness from the Notes database's checkpoint
    # row inside its own transaction.
    with pytest.raises(ValueError, match="not ready"):
        profile.change("food", resolved_profile=resolved)
    assert reads == []
    assert profile.keywords() == ["planning", "travel"]
    assert [domain for domain, _ in profile.intents()] == [
        "notes.keyword",
        "notes.keyword_link",
    ]


def test_without_sync_v2_wiring_there_is_nothing_to_resolve(tmp_path) -> None:
    profile = _Profile(tmp_path)

    assert profile.service.resolve_local_note_keyword_profile(user_id=USER) is None
    assert profile.change("travel") is True
    assert profile.keywords() == ["planning", "travel"]
