"""TASK-33621.23 AC2: a blocked Console submit never ends silently.

The reported send stopped at ``controller_submit status=blocked`` and the log
said nothing else. The "Console submit requires attention" warning added since
logs only ``accepted=True status=blocked``, the same line for every cause. These
tests pin three things:

- every BLOCKED result names a code-owned blocker;
- that code reaches both the app log warning and the persistent send-stage line;
- the user sees the same block (run chip and recovery card).
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest

from Tests.Chat.test_console_first_send_atomicity import _controller
from Tests.Chat.test_console_trace_first_send_atomicity import (
    _fail_trace_request_once,
    _force_capture_on,
    _record_provider_entries,
)
from tldw_chatbook.Chat import console_chat_controller as controller_module
from tldw_chatbook.Chat.console_chat_models import ConsoleRunStatus
from tldw_chatbook.UI.Console_Modules.provider_continuation_recovery import (
    trace_call_recovery_state,
)
from tldw_chatbook.Utils.persistent_diagnostics import PersistentDiagnosticFilter

pytestmark = pytest.mark.bootstrap_profile

_PRIVATE_DRAFT = "PRIVATE-DRAFT-33621-23"
_CHAT = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "Chat"


class _Lines(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


@pytest.fixture
def logs():
    """Collect the persistent diagnostic lines and the controller's warnings."""
    persistent = _Lines()
    persistent.addFilter(PersistentDiagnosticFilter())
    root = logging.getLogger()
    old_level = root.level
    root.setLevel(logging.INFO)
    root.addHandler(persistent)
    warnings: list[str] = []
    sink = controller_module.logger.add(
        lambda message: warnings.append(message.record["message"]), level="WARNING"
    )
    try:
        yield persistent.lines, warnings
    finally:
        controller_module.logger.remove(sink)
        root.removeHandler(persistent)
        root.setLevel(old_level)


def _attention(warnings: list[str]) -> str:
    lines = [line for line in warnings if "Console submit requires attention" in line]
    assert lines, "a blocked submit must log the attention warning"
    return lines[-1]


def _submit_end(persistent: list[str]) -> str:
    lines = [
        line
        for line in persistent
        if "phase=controller_submit" in line and "status=blocked" in line
    ]
    assert lines, "the controller_submit stage must end blocked"
    return lines[-1]


@pytest.mark.asyncio
async def test_a_trace_provenance_block_logs_its_blocker_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, logs
) -> None:
    persistent, warnings = logs
    _db, store, controller, _gateway = _controller(tmp_path)
    _force_capture_on(monkeypatch)
    entries = _record_provider_entries(controller, monkeypatch)
    _fail_trace_request_once("build", store=store, monkeypatch=monkeypatch)

    result = await controller.submit_draft(_PRIVATE_DRAFT, session_id="session-1")

    assert result.terminal_status is ConsoleRunStatus.BLOCKED
    assert entries == []
    assert "blocker=trace_provenance" in _attention(warnings)
    assert "error_category=trace_provenance" in _submit_end(persistent)
    assert (
        getattr(getattr(result, "blocker", None), "value", None) == "trace_provenance"
    )
    # The user sees the same block: the run chip and the recovery card.
    state = controller.run_state_for("session-1")
    assert state.status is ConsoleRunStatus.BLOCKED
    assert "Trace provenance could not be saved" in state.visible_copy
    assert trace_call_recovery_state(store.preparation_for_session("session-1"))
    assert _PRIVATE_DRAFT not in "\n".join([*persistent, *warnings])


@pytest.mark.asyncio
async def test_a_trace_call_block_logs_its_blocker_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, logs
) -> None:
    """The missing-request pause (test_console_trace_request_missing_pause)."""
    persistent, warnings = logs
    _db, _store, controller, _gateway = _controller(tmp_path)
    _force_capture_on(monkeypatch)
    _record_provider_entries(controller, monkeypatch)
    monkeypatch.setattr(
        controller, "_build_durable_trace_request", lambda **_kwargs: None
    )

    result = await controller.submit_draft(_PRIVATE_DRAFT, session_id="session-1")

    assert result.terminal_status is ConsoleRunStatus.BLOCKED
    assert "blocker=trace_call" in _attention(warnings)
    assert "error_category=trace_call" in _submit_end(persistent)
    assert getattr(getattr(result, "blocker", None), "value", None) == "trace_call"
    assert _PRIVATE_DRAFT not in "\n".join([*persistent, *warnings])


def _blocked_results_without_a_blocker(path: Path) -> list[int]:
    missing: list[int] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        name = (
            callee.attr
            if isinstance(callee, ast.Attribute)
            else getattr(callee, "id", "")
        )
        if name != "ConsoleSubmitResult":
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        status = keywords.get("terminal_status")
        if (
            isinstance(status, ast.Attribute)
            and status.attr == "BLOCKED"
            and "blocker" not in keywords
        ):
            missing.append(node.lineno)
    return missing


@pytest.mark.parametrize(
    "module", ["console_chat_controller.py", "console_native_commit.py"]
)
def test_every_blocked_submit_result_names_its_blocker(module: str) -> None:
    """A new literal BLOCKED exit must name its own code.

    Without one it would log the generic ``turn_refused``, which says less.
    Sites whose status is read from the run state are covered at runtime by
    the refusal-inside-the-turn tests above.
    """
    path = _CHAT / module
    assert "ConsoleRunStatus.BLOCKED" in path.read_text(encoding="utf-8")
    assert _blocked_results_without_a_blocker(path) == []


def _refuse_inside_the_stream(controller, monkeypatch, refuse) -> None:
    """Return a refusal from inside the turn, as ``_run_agent_reply`` does.

    The refusal itself comes from the controller's own helper; only the call
    that reaches it is replaced.
    """

    async def stream(**kwargs):
        return refuse(kwargs["assistant_message_id"])

    monkeypatch.setattr(controller, "_stream_assistant_response", stream)


@pytest.mark.asyncio
async def test_a_project_setup_refusal_in_a_temporary_chat_names_its_blocker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, logs
) -> None:
    """The folder picker's Cancel in a temporary (non-durable) chat."""
    persistent, warnings = logs
    _db, _store, controller, _gateway = _controller(tmp_path)
    session = controller.new_session(ephemeral=True)
    _refuse_inside_the_stream(
        controller,
        monkeypatch,
        lambda assistant_id: controller._refuse_turn_for_project_setup(
            assistant_id, session.id, "project_binding_required"
        ),
    )

    result = await controller.submit_draft(_PRIVATE_DRAFT, session_id=session.id)

    assert result.terminal_status is ConsoleRunStatus.BLOCKED
    assert "blocker=project_setup" in _attention(warnings)
    assert "error_category=project_setup" in _submit_end(persistent)
    assert _PRIVATE_DRAFT not in "\n".join([*persistent, *warnings])


@pytest.mark.asyncio
@pytest.mark.parametrize("temporary", [True, False], ids=["temporary", "saved"])
async def test_any_other_refusal_inside_the_turn_still_names_a_blocker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, logs, temporary: bool
) -> None:
    """A BLOCKED result whose site named no code must not log ``blocker=-``.

    Both chat kinds read the status from the run state at a ``replace`` site,
    which no literal code can reach.
    """
    persistent, warnings = logs
    _db, _store, controller, _gateway = _controller(tmp_path)
    session_id = (
        controller.new_session(ephemeral=True).id if temporary else "session-1"
    )
    _refuse_inside_the_stream(
        controller,
        monkeypatch,
        lambda _assistant_id: controller._block(
            session_id, "Private scratch space is unavailable."
        ),
    )

    result = await controller.submit_draft(_PRIVATE_DRAFT, session_id=session_id)

    assert result.terminal_status is ConsoleRunStatus.BLOCKED
    assert "blocker=turn_refused" in _attention(warnings)
    assert "error_category=turn_refused" in _submit_end(persistent)
    assert controller.run_state_for(session_id).visible_copy == (
        "Private scratch space is unavailable."
    )


@pytest.mark.asyncio
async def test_a_failed_submit_never_logs_a_blocker_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, logs
) -> None:
    """A code is about a block; a site whose status settled FAILED logs ``-``."""
    persistent, warnings = logs
    _db, _store, controller, _gateway = _controller(tmp_path)

    async def lifecycle(*_args, **_kwargs):
        return controller_module.ConsoleSubmitResult(
            True,
            True,
            "Accepted turn failed before provider dispatch.",
            terminal_status=ConsoleRunStatus.FAILED,
            blocker=controller_module.ConsoleSubmitBlocker.SETTLEMENT_FAILED,
        )

    monkeypatch.setattr(controller, "_submit_draft_lifecycle", lifecycle)

    await controller.submit_draft(_PRIVATE_DRAFT, session_id="session-1")

    assert "status=failed blocker=-" in _attention(warnings)
    final = [line for line in persistent if "phase=controller_submit" in line]
    assert final and "error_category" not in final[-1]
