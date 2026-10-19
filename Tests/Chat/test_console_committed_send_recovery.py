"""A durably committed send keeps its own recovery when a resume finds no provider.

TASK-33621.20 pre-merge re-review. A send whose trace record could not be
saved after its durable commit pauses as TRACE_PROVENANCE, and its card offers
Retry / Send without capture / Cancel -- each aware that the user and assistant
rows, the dispatch checkpoint and the postcommit continuation already exist.

"Send without capture" with a provider that was not ready used to re-pause it
as DESTINATION_CHANGED. The card then offered the pre-commit Retry send /
Cancel send: Cancel said "nothing was sent", put the text back as the draft,
and orphaned the committed rows, checkpoint and continuation (a resend
duplicated the message); Retry dropped the user's capture-off choice.

Real controller, store and SQLite database; only the provider resolver is
stubbed not-ready, and the trace repository fails once to cause the pause.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_trace_first_send_atomicity import (
    _controller,
    _fail_trace_request_once,
    _force_capture_on,
    _record_provider_entries,
)
from tldw_chatbook.Chat.console_trace_provenance import ConsoleTraceCaptureMode
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsoleTurnPreparationState,
)
from tldw_chatbook.Chat.console_unsent_turn import shelf_releasable
from tldw_chatbook.UI.Console_Modules.provider_continuation_recovery import (
    trace_call_recovery_state,
)

pytestmark = pytest.mark.bootstrap_profile


def _counts(db) -> tuple[int, int]:
    connection = db.get_connection()
    messages = connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    checkpoints = connection.execute(
        "SELECT COUNT(*) FROM console_dispatch_checkpoints"
    ).fetchone()[0]
    return messages, checkpoints


@pytest.mark.asyncio
async def test_send_without_capture_on_a_provider_that_is_not_ready_keeps_the_committed_pause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, store, controller, _gateway = _controller(tmp_path)
    _force_capture_on(monkeypatch)
    entries = _record_provider_entries(controller, monkeypatch)
    _fail_trace_request_once("missing_repository", store=store, monkeypatch=monkeypatch)

    first = await controller.submit_draft("secret durable body", session_id="session-1")
    paused = store.preparation_for_session("session-1")
    assert first.accepted is True, first.visible_copy
    assert paused.pause_kind is ConsolePreparationPauseKind.TRACE_PROVENANCE
    committed = _counts(db)
    assert committed == (2, 1)

    ready_resolver = controller._resolve_for_send_bounded

    async def not_ready(_selection):
        return SimpleNamespace(ready=False, visible_copy="Provider is not ready.")

    controller._resolve_for_send_bounded = not_ready
    refused = await controller.send_without_capture(paused.preparation_id)

    current = store.preparation_for_session("session-1")
    assert refused.accepted is False
    assert entries == []
    # The committed send keeps its own pause and its own card ...
    assert current.preparation_id == paused.preparation_id
    assert current.state is ConsoleTurnPreparationState.PAUSED
    assert current.pause_kind is ConsolePreparationPauseKind.TRACE_PROVENANCE
    state = trace_call_recovery_state(current)
    assert state.provenance is True and state.resend_pause is None
    # ... never the pre-commit Retry send / Cancel send, nor the shelf release.
    assert shelf_releasable(current, current.preparation_id) is False
    assert paused.preparation_id in controller._durable_postcommit_continuations
    assert _counts(db) == committed

    # The user's capture-off choice still stands, and sends once, exactly.
    controller._resolve_for_send_bounded = ready_resolver
    resumed = await controller.send_without_capture(paused.preparation_id)

    assert resumed.accepted is True, resumed.visible_copy
    assert len(entries) == 1
    assert entries[0]["capture_mode_override"] is ConsoleTraceCaptureMode.CAPTURE_OFF
    assert _counts(db)[0] == committed[0]
