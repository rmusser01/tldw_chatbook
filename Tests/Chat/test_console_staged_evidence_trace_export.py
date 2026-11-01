"""TASK-33621.8: a staged-Library turn's trace exports under every profile.

Every production send is admitted through ``ConsoleRuntime.accept_turn`` and
takes the frozen staged-evidence route. That route never records the
``retrieval_*`` events the live search route does, yet the turn's
``context_attached`` event named ``retrieval_completed`` as its parent and
source. The exporter refused the whole bundle over that one orphan, so any
conversation with a staged-Library turn could not be exported at all.

The send here is the real one -- runtime custody, controller, store, SQLite
persistence and the production trace snapshot builder -- with only the
provider stream and the Library search boundary answered locally.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_staged_evidence_dispatch_pin import (
    _admit,
    _library_search,
    _runtime,
)
from Tests.private_profile import private_profile_test
from tldw_chatbook.Chat.trajectory_export import (
    TraceExportProfile,
    build_trace_export,
    preflight_trace_export,
)
from tldw_chatbook.Chat.trajectory_import import load_imported_trace
from tldw_chatbook.UI.Console_Modules.review_selection import (
    _build_trajectory_snapshot,
)

_STAGED = "Orchid project codename: HELIX-7731."


@pytest.mark.asyncio
@private_profile_test
async def test_staged_library_turn_records_only_resolvable_links_and_exports(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, store, controller, session_id = _runtime(tmp_path, ephemeral=False)
    _library_search(monkeypatch)
    runtime.stage_console_staged_evidence(SimpleNamespace(evidence=_STAGED))

    await runtime.wait_for_turn(
        _admit(runtime, controller, session_id, "What is the Orchid codename?")
    )

    conversation_id = store._sessions[session_id].persisted_conversation_id
    assert conversation_id
    snapshot = _build_trajectory_snapshot(store, conversation_id)
    records = [record for turn in snapshot.turns for record in turn.records]
    kinds = {record.kind for record in records}
    # The staged evidence reached the request and the trace says so.
    assert {"context_attached", "context_injected"} <= kinds
    for profile in TraceExportProfile:
        preflight = preflight_trace_export(snapshot, profile=profile)
        payload = build_trace_export(
            snapshot,
            preflight=preflight,
            confirm_full=profile is TraceExportProfile.FULL_TRACE,
        )
        assert payload["manifest"]["profile"] == profile.value
        # Nothing had to be repaired on the way out for a new turn.
        assert not [
            entry
            for entry in payload["manifest"]["redaction_provenance"]
            if entry["reason"] == "reference_not_recorded"
        ]
        load_imported_trace(payload)

    # AC#2: the producer itself now records only links that resolve.
    recorded = {record.event_id for record in records}
    unresolved = [
        (record.kind, field, getattr(record, field))
        for record in records
        for field in ("parent_event_id", "source_event_id", "replacement_event_id")
        if getattr(record, field) and getattr(record, field) not in recorded
    ]
    assert unresolved == []
