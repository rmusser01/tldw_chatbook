"""TASK-33940.3: a Capture-On send with a system prompt must reach the provider.

Live, every send in a new workspace chat (whose persona sets a system prompt)
ended BLOCKED with "Trace provenance could not be saved" and nothing visible:
``_build_durable_trace_request`` labelled the unsaved leading system row
ACTIVE_REQUEST, the request's ``system`` category accepts only RENDERED_SYSTEM,
and the resulting TraceProvenanceAlignmentError was swallowed without a log.
The pause's Retry / Send without capture / Cancel card was also never shown,
because the transcript projection only admitted TRACE_CALL pauses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from Tests.Chat.test_console_first_send_atomicity import _controller
from Tests.Chat.test_console_trace_first_send_atomicity import (
    _fail_trace_request_once,
    _force_capture_on,
    _record_provider_entries,
)
from Tests.private_profile import private_profile_test
from tldw_chatbook.Chat import console_chat_controller as controller_module
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_trace_provenance import ConsoleTraceCaptureMode
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsoleTurnPreparationState,
)
from tldw_chatbook.UI.Console_Modules.provider_continuation_recovery import (
    trace_call_recovery_state,
)

SYSTEM_PROMPT = "You are the My Project workspace agent."


@pytest.mark.asyncio
@private_profile_test
async def test_capture_on_send_with_a_system_prompt_reaches_the_provider(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _db, store, controller, _gateway = _controller(
        tmp_path,
        initial_settings=ConsoleSessionSettings(
            provider="llama_cpp", model="test-model", system_prompt=SYSTEM_PROMPT
        ),
    )
    _force_capture_on(monkeypatch)
    entries = _record_provider_entries(controller, monkeypatch)

    result = await controller.submit_draft("hello", session_id="session-1")

    paused = store.preparation_for_session("session-1")
    assert paused is None or paused.state is not ConsoleTurnPreparationState.PAUSED
    assert result.accepted is True
    assert len(entries) == 1
    assert entries[0]["capture_mode_override"] is ConsoleTraceCaptureMode.CAPTURE_ON
    assert entries[0]["trace_request"] is not None


@pytest.mark.asyncio
@private_profile_test
async def test_a_genuine_provenance_failure_is_logged_and_offered_as_a_card(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _db, store, controller, _gateway = _controller(tmp_path)
    _force_capture_on(monkeypatch)
    entries = _record_provider_entries(controller, monkeypatch)
    _fail_trace_request_once("build", store=store, monkeypatch=monkeypatch)
    records: list[str] = []
    sink = controller_module.logger.add(
        lambda message: records.append(message.record["message"]), level="WARNING"
    )
    try:
        result = await controller.submit_draft(
            "PRIVATE-DRAFT-BODY", session_id="session-1"
        )
    finally:
        controller_module.logger.remove(sink)

    paused = store.preparation_for_session("session-1")
    assert result.accepted is True
    assert entries == []
    assert paused is not None
    assert paused.pause_kind is ConsolePreparationPauseKind.TRACE_PROVENANCE
    logged = [line for line in records if "trace provenance" in line.lower()]
    assert logged, "the swallowed failure must leave a log line"
    assert "TraceProvenancePersistenceError" in logged[0]
    assert "PRIVATE-DRAFT-BODY" not in "\n".join(records)
    # The pause is actionable, so the transcript's recovery card must show it.
    state = trace_call_recovery_state(paused)
    assert state is not None
    assert state.preparation_id == paused.preparation_id
    assert state.temporary_capture is False
