# Console Screen Feedback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement three native console-screen improvements: click negative space to clear message selection, harden local agent turn-control finalization, and add a two-tab context viewer modal.

**Architecture:** Keep changes scoped to the native console (`ChatScreen` / `ConsoleTranscript` / `ConsoleChatController`). Add a guarded click handler to the transcript, normalize `RunOutcome` handling in `_finalize_agent_reply`, and expose a read-only async snapshot method that a new modal renders in Current / Next-Send tabs.

**Tech Stack:** Python 3.11+, Textual, Pydantic-style dataclasses, `loguru`, pytest with Textual's `run_test`/`pilot`.

---

## File Structure

| File | Responsibility |
|---|---|
| `tldw_chatbook/Widgets/Console/console_transcript.py` | Add negative-space click handler; keep existing selection API unchanged. |
| `tldw_chatbook/Chat/console_chat_controller.py` | Harden `_finalize_agent_reply()` for empty `final_text`, unknown statuses, and diagnostics; add `build_context_snapshot()`. |
| `tldw_chatbook/Chat/console_chat_models.py` | Add `ConsoleContextSnapshot` dataclass. |
| `tldw_chatbook/Widgets/Console/console_context_modal.py` | New two-tab modal for current transcript and next-send payload. |
| `tldw_chatbook/UI/console_command_provider.py` | Register **Console: View chat context** command. |
| `tldw_chatbook/UI/Screens/chat_screen.py` | Bind `ctrl+shift+p` to open the context modal. |
| `Tests/UI/test_console_native_transcript.py` | Tests for selection clearing. |
| `Tests/Chat/test_console_chat_controller.py` | Tests for agent finalization and context snapshot. |
| `Tests/Chat/test_console_agent_bridge.py` | Bridge-level tests for `ConsoleAgentBridge.run_reply()`. |
| `Tests/UI/test_console_context_modal.py` | Tests for the context viewer modal. |
| `Tests/UI/test_chat_screen_context_modal.py` | Tests that `ChatScreen` keybinding and command palette open the modal. |

---

## Task 1: Clear selection on negative-space click

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py`
- Test: `Tests/UI/test_console_native_transcript.py`

### Step 1: Write the failing test

Append to `Tests/UI/test_console_native_transcript.py`:

```python
@pytest.mark.asyncio
async def test_console_transcript_click_background_clears_selection():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message("m2")
        await pilot.pause()
        assert transcript.selected_message_id == "m2"

        # Click the scroll container background, not a message row.
        await pilot.click("#console-native-transcript")
        await pilot.pause()

        assert transcript.selected_message_id is None
```

Run:

```bash
pytest Tests/UI/test_console_native_transcript.py::test_console_transcript_click_background_clears_selection -v
```

Expected: FAIL (`selected_message_id` still `"m2"`).

### Step 2: Implement the click handler

In `ConsoleTranscript` (`tldw_chatbook/Widgets/Console/console_transcript.py`), add:

```python
from textual.events import Click

# Widget IDs / classes that count as negative space for deselection.
_NEGATIVE_SPACE_IDS = {
    "console-native-transcript",
}
# Classes whose clicks must NOT clear selection even if they bubble.
_INTERACTIVE_CLASSES = {
    "console-transcript-message",
    "console-transcript-action-row",
    "console-transcript-action-button",
    "console-transcript-rule",
    "console-transcript-action-guide",
    "console-transcript-empty-state",
    "console-transcript-empty-panel",
}


def on_click(self, event: Click) -> None:
    if self.selected_message_id is None:
        return
    control = event.control
    if control is None:
        return
    if control.id in _NEGATIVE_SPACE_IDS:
        self.action_clear_selection()
        return
    if any(cls in control.classes for cls in _INTERACTIVE_CLASSES):
        return
    # Scrollbar widgets have classes containing "scrollbar".
    if "scrollbar" in " ".join(control.classes):
        return
```

Place it next to the existing `action_clear_selection` method.

Run:

```bash
pytest Tests/UI/test_console_native_transcript.py::test_console_transcript_click_background_clears_selection -v
```

Expected: PASS.

### Step 3: Add negative tests for interactive backgrounds

Append to `Tests/UI/test_console_native_transcript.py`:

```python
@pytest.mark.asyncio
async def test_console_transcript_click_action_row_background_preserves_selection():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message("m2")
        await pilot.pause()
        assert transcript.selected_message_id == "m2"

        await pilot.click("#console-message-actions-m2")
        await pilot.pause()

        assert transcript.selected_message_id == "m2"


@pytest.mark.asyncio
async def test_console_transcript_click_rule_preserves_selection():
    app = TranscriptHarness()

    async with app.run_test(size=(100, 32)) as pilot:
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message("m2")
        await pilot.pause()

        await pilot.click(".console-transcript-rule")
        await pilot.pause()

        assert transcript.selected_message_id == "m2"
```

Run:

```bash
pytest Tests/UI/test_console_native_transcript.py::test_console_transcript_click_action_row_background_preserves_selection -v
```

Expected: PASS (the action row background does not match the negative-space allow-list).

### Step 4: Commit

```bash
git add Tests/UI/test_console_native_transcript.py tldw_chatbook/Widgets/Console/console_transcript.py
git commit -m "feat(console): clear message selection on negative-space click"
```

---

## Task 2: Harden agent turn-control finalization

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Test: `Tests/Chat/test_console_chat_controller.py`
- Test: `Tests/Chat/test_console_agent_bridge.py` (new file)

### Step 2.1: Inspect current `_finalize_agent_reply`

Open `tldw_chatbook/Chat/console_chat_controller.py:1661-1735`. Confirm the existing branches:

- `RUN_CANCELLED` → `_mark_stream_stopped`
- non-`RUN_DONE` → `mark_message_failed` + `_append_failure_system_row`
- `RUN_DONE` → `finalize_variant_stream` / `mark_message_complete`

No special handling exists for empty `final_text`, placeholder-missing fallback, or explicit unknown-status logging.

### Step 2.2: Write failing test for empty `final_text` fallback

Append to `Tests/Chat/test_console_chat_controller.py`:

```python
@pytest.mark.asyncio
async def test_finalize_agent_reply_empty_final_text_uses_fallback():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session(title="Chat 1")
    from tldw_chatbook.Agents.agent_models import RUN_DONE

    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
    )

    outcome = RunOutcome(status=RUN_DONE, steps=[], final_text="")
    result = controller._finalize_agent_reply(
        assistant.id,
        session.id,
        outcome,
        variant_mode=False,
    )

    updated = store.get_message(assistant.id)
    assert updated.status == "complete"
    assert "No response was generated" in updated.content
    assert result.accepted is True
```

> Replace `StreamingGateway()` with the actual gateway class used by existing tests.

Run:

```bash
pytest Tests/Chat/test_console_chat_controller.py::test_finalize_agent_reply_empty_final_text_uses_fallback -v
```

Expected: FAIL (message is complete but empty, or assertion on content fails).

### Step 2.3: Implement empty `final_text` fallback

In `_finalize_agent_reply`, replace the `RUN_DONE` branch:

```python
# Before:
# if variant_mode:
#     completed = self.store.finalize_variant_stream(assistant_message_id)
# else:
#     completed = self.store.mark_message_complete(assistant_message_id)

# After:
completed = self._complete_agent_message(
    assistant_message_id, variant_mode=variant_mode, outcome=outcome
)
```

Add helper method:

```python
def _complete_agent_message(
    self,
    assistant_message_id: str,
    *,
    variant_mode: bool,
    outcome: Any,
) -> ConsoleChatMessage:
    from tldw_chatbook.Agents.agent_models import RUN_DONE

    if outcome.status == RUN_DONE and not outcome.final_text.strip():
        # Placeholder is pending/streaming; bypass the store's post-stream guard.
        message = self.store.get_message(assistant_message_id)
        message.content = "No response was generated."

    if variant_mode:
        return self.store.finalize_variant_stream(assistant_message_id)
    return self.store.mark_message_complete(assistant_message_id)
```

Run:

```bash
pytest Tests/Chat/test_console_chat_controller.py::test_finalize_agent_reply_empty_final_text_uses_fallback -v
```

Expected: PASS.

### Step 2.4: Write failing test for placeholder-missing fallback

Append to `Tests/Chat/test_console_chat_controller.py`:

```python
@pytest.mark.asyncio
async def test_finalize_agent_reply_missing_placeholder_appends_message():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session(title="Chat 1")
    from tldw_chatbook.Agents.agent_models import RUN_DONE

    outcome = RunOutcome(status=RUN_DONE, steps=[], final_text="final answer")
    result = controller._finalize_agent_reply(
        "non-existent-id",
        session.id,
        outcome,
        variant_mode=False,
    )

    messages = list(store.messages_for_session(session.id))
    assert len(messages) == 1
    assert messages[0].role == ConsoleMessageRole.ASSISTANT
    assert messages[0].content == "final answer"
    assert result.accepted is True
```

Run:

```bash
pytest Tests/Chat/test_console_chat_controller.py::test_finalize_agent_reply_missing_placeholder_appends_message -v
```

Expected: FAIL (`KeyError` or `_session_closed_result`).

### Step 2.5: Implement placeholder-missing fallback

Refactor `_finalize_agent_reply` to delegate terminal handling to helpers. Keep the existing stopped/cancelled guard intact, then delegate:

```python
# Keep the existing task-227 guard unchanged:
if current.status == "stopped" or (cancel_event is not None and cancel_event.is_set()):
    self._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STOPPED, "Response stopped.")
    )
    return ConsoleSubmitResult(True, True, current.content)

if outcome.status == RUN_CANCELLED:
    return self._finalize_agent_cancelled(assistant_message_id, session_id)

if outcome.status != RUN_DONE:
    return self._finalize_agent_failure(assistant_message_id, session_id, outcome)

return self._finalize_agent_success(
    assistant_message_id, session_id, outcome, variant_mode=variant_mode
)
```

Add helpers:

```python
def _ensure_assistant_placeholder(self, assistant_message_id: str, session_id: str) -> ConsoleChatMessage | None:
    try:
        return self.store.get_message(assistant_message_id)
    except KeyError:
        return None


def _find_runtime_written_assistant(self, session_id: str) -> ConsoleChatMessage | None:
    """Return the most recent assistant message if the runtime already wrote one."""
    for message in reversed(list(self.store.messages_for_session(session_id))):
        if message.role == ConsoleMessageRole.ASSISTANT:
            return message
    return None


def _finalize_agent_cancelled(
    self,
    assistant_message_id: str,
    session_id: str,
) -> ConsoleSubmitResult:
    try:
        stopped = self._mark_stream_stopped(
            assistant_message_id, visible_copy="Response stopped.")
    except KeyError:
        # Placeholder gone; treat as a stopped session.
        return self._session_closed_result()
    return ConsoleSubmitResult(True, True, stopped.content)


def _finalize_agent_success(
    self,
    assistant_message_id: str,
    session_id: str,
    outcome: Any,
    *,
    variant_mode: bool,
) -> ConsoleSubmitResult:
    from tldw_chatbook.Agents.agent_models import RUN_DONE

    placeholder = self._ensure_assistant_placeholder(assistant_message_id, session_id)
    if placeholder is None:
        runtime_message = self._find_runtime_written_assistant(session_id)
        if runtime_message is not None:
            if not runtime_message.content:
                runtime_message.content = outcome.final_text or "No response was generated."
            runtime_message.status = "complete"
            self._set_run_state(ConsoleRunState(ConsoleRunStatus.COMPLETED, "Response complete."))
            return ConsoleSubmitResult(True, True, runtime_message.content)
        content = outcome.final_text or "No response was generated."
        message = self.store.append_message(
            session_id,
            role=ConsoleMessageRole.ASSISTANT,
            content=content,
        )
        completed = self.store.mark_message_complete(message.id)
        self._set_run_state(ConsoleRunState(ConsoleRunStatus.COMPLETED, "Response complete."))
        return ConsoleSubmitResult(True, True, completed.content)

    completed = self._complete_agent_message(
        assistant_message_id, variant_mode=variant_mode, outcome=outcome
    )
    self._set_run_state(ConsoleRunState(ConsoleRunStatus.COMPLETED, "Response complete."))
    return ConsoleSubmitResult(True, True, completed.content)


def _finalize_agent_failure(
    self,
    assistant_message_id: str,
    session_id: str,
    outcome: Any,
) -> ConsoleSubmitResult:
    visible_copy = self._agent_failure_visible_copy(outcome)
    placeholder = self._ensure_assistant_placeholder(assistant_message_id, session_id)
    if placeholder is None:
        runtime_message = self._find_runtime_written_assistant(session_id)
        if runtime_message is not None:
            runtime_message.status = "failed"
            if not runtime_message.content:
                runtime_message.content = visible_copy
            self._append_failure_system_row(session_id, visible_copy)
            self._set_run_state(ConsoleRunState(ConsoleRunStatus.FAILED, visible_copy))
            return ConsoleSubmitResult(True, True, runtime_message.content)
        message = self.store.append_message(
            session_id,
            role=ConsoleMessageRole.ASSISTANT,
            content=visible_copy,
        )
        failed = self.store.mark_message_failed(message.id)
        self._append_failure_system_row(session_id, visible_copy)
        self._set_run_state(ConsoleRunState(ConsoleRunStatus.FAILED, visible_copy))
        return ConsoleSubmitResult(True, True, failed.content)

    failed = self.store.mark_message_failed(assistant_message_id)
    self._append_failure_system_row(session_id, visible_copy)
    self._set_run_state(ConsoleRunState(ConsoleRunStatus.FAILED, visible_copy))
    return ConsoleSubmitResult(True, True, failed.content)
```

Run:

```bash
pytest Tests/Chat/test_console_chat_controller.py::test_finalize_agent_reply_missing_placeholder_appends_message Tests/Chat/test_console_chat_controller.py::test_finalize_agent_reply_empty_final_text_uses_fallback -v
```

Expected: PASS.

### Step 2.6: Add tests for failure, cancel, and exception branches

Append to `Tests/Chat/test_console_chat_controller.py`:

```python
@pytest.mark.asyncio
async def test_finalize_agent_reply_error_marks_failed():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session(title="Chat 1")
    from tldw_chatbook.Agents.agent_models import RUN_ERROR

    assistant = store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="partial")
    outcome = RunOutcome(status=RUN_ERROR, steps=[], final_text="")
    result = controller._finalize_agent_reply(assistant.id, session.id, outcome, variant_mode=False)

    updated = store.get_message(assistant.id)
    assert updated.status == "failed"
    assert result.accepted is True


@pytest.mark.asyncio
async def test_finalize_agent_reply_cancelled_marks_stopped():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session(title="Chat 1")
    from tldw_chatbook.Agents.agent_models import RUN_CANCELLED

    assistant = store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="")
    outcome = RunOutcome(status=RUN_CANCELLED, steps=[], final_text="")
    result = controller._finalize_agent_reply(assistant.id, session.id, outcome, variant_mode=False)

    updated = store.get_message(assistant.id)
    assert updated.status == "stopped"
    assert result.accepted is True


@pytest.mark.asyncio
async def test_finalize_agent_reply_unknown_status_marks_failed():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session(title="Chat 1")

    assistant = store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="partial")
    outcome = RunOutcome(status="RUN_SUPERSEDED", steps=[], final_text="")
    result = controller._finalize_agent_reply(assistant.id, session.id, outcome, variant_mode=False)

    updated = store.get_message(assistant.id)
    assert updated.status == "failed"
    assert "RUN_SUPERSEDED" in str(store.messages_for_session(session.id))
```

> Replace `StreamingGateway()` with the actual gateway class used by existing tests.

Run:

```bash
pytest Tests/Chat/test_console_chat_controller.py::test_finalize_agent_reply_error_marks_failed Tests/Chat/test_console_chat_controller.py::test_finalize_agent_reply_cancelled_marks_stopped Tests/Chat/test_console_chat_controller.py::test_finalize_agent_reply_unknown_status_marks_failed -v
```

Expected: PASS.

### Step 2.7: Add bridge-level tests

Create `Tests/Chat/test_console_agent_bridge.py`:

```python
import pytest
from unittest.mock import MagicMock

from tldw_chatbook.Agents.agent_models import RUN_DONE, RUN_ERROR, RunOutcome
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge


class FakeAgentService:
    def __init__(self, outcome: RunOutcome):
        self.outcome = outcome

    def run_turn(self, *args, **kwargs) -> RunOutcome:
        return self.outcome


def test_run_reply_returns_runoutcome_done():
    outcome = RunOutcome(status=RUN_DONE, steps=[], final_text="done")
    bridge = ConsoleAgentBridge(agent_service=FakeAgentService(outcome))
    result = bridge.run_reply(
        conversation_id="c1",
        session_id="s1",
        resolution=None,
        assistant_message_id="a1",
        model="gpt-4",
        session_system_prompt="sys",
        agent_messages=[{"role": "user", "content": "hi"}],
        should_cancel=lambda: False,
    )
    assert result.status == RUN_DONE
    assert result.final_text == "done"


def test_run_reply_returns_runoutcome_error():
    outcome = RunOutcome(status=RUN_ERROR, steps=[], final_text="")
    bridge = ConsoleAgentBridge(agent_service=FakeAgentService(outcome))
    result = bridge.run_reply(
        conversation_id="c1",
        session_id="s1",
        resolution=None,
        assistant_message_id="a1",
        model="gpt-4",
        session_system_prompt="sys",
        agent_messages=[{"role": "user", "content": "hi"}],
        should_cancel=lambda: False,
    )
    assert result.status == RUN_ERROR
```

> The constructor signature for `ConsoleAgentBridge` may differ; adjust the test setup to match the actual class.

Run:

```bash
pytest Tests/Chat/test_console_agent_bridge.py -v
```

Expected: PASS.

### Step 2.8: Add structured diagnostics logging

In `_run_agent_reply`, before and after the `asyncio.to_thread` call:

```python
from loguru import logger

logger.info(
    "console_agent_run_start",
    assistant_message_id=assistant_message_id,
    model=self.model or self.configured_model,
    conversation_id=conversation_id,
)

outcome = await asyncio.to_thread(...)

logger.info(
    "console_agent_run_end",
    assistant_message_id=assistant_message_id,
    status=outcome.status,
    steps=len(outcome.steps),
    final_text_len=len(outcome.final_text),
)
```

In `ConsoleAgentBridge.run_reply`, add at entry and exit, and wrap the tool-invocation closure so each tool call is logged:

```python
from loguru import logger

logger.info(
    "console_agent_bridge_run",
    conversation_id=conversation_id,
    model=model,
    allowed_tools_count=len(allowed_tools),
)

# If the bridge builds an invoke_tool closure for AgentService.run_turn,
# wrap it to log each tool invocation:
original_invoke_tool = invoke_tool

def logged_invoke_tool(tool_call):
    logger.info(
        "console_agent_tool_call",
        tool_name=getattr(tool_call, "name", "unknown"),
        tool_id=getattr(tool_call, "id", "unknown"),
    )
    result = original_invoke_tool(tool_call)
    logger.info(
        "console_agent_tool_result",
        tool_name=getattr(tool_call, "name", "unknown"),
        success=getattr(result, "success", True),
    )
    return result

# Pass logged_invoke_tool to AgentService.run_turn in place of invoke_tool.
```

After `AgentService.run_turn()` returns, log each step in the outcome:

```python
for i, step in enumerate(outcome.steps):
    logger.info(
        "console_agent_step",
        step_index=i,
        step_kind=getattr(step, "kind", "unknown"),
        step_summary=getattr(step, "summary", "")[:200],
    )
```

> The exact `AgentStep` shape is in `tldw_chatbook/Agents/agent_models.py`. Adjust attribute access (`step.kind`, `step.summary`, etc.) to match the actual fields.

Run the existing controller tests:

```bash
pytest Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_agent_bridge.py -v
```

Expected: all PASS.

### Step 2.9: Commit

```bash
git add Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_agent_bridge.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_agent_bridge.py
git commit -m "feat(console): harden agent turn-control finalization, fallback, and diagnostics"
```

---

## Task 3: Add context snapshot method

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_models.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Test: `Tests/Chat/test_console_chat_controller.py`

### Step 3.1: Define the snapshot dataclass

In `tldw_chatbook/Chat/console_chat_models.py`, add:

```python
from typing import Any


@dataclass
class ConsoleContextSnapshot:
    """Read-only snapshot of current transcript and next-send provider payload."""

    current_messages: list[ConsoleChatMessage]
    next_send_payload: dict[str, Any]
```

### Step 3.2: Implement `build_context_snapshot`

In `ConsoleChatController`, add:

```python
from tldw_chatbook.Chat.console_chat_models import ConsoleContextSnapshot


async def build_context_snapshot(
    self,
    draft: str,
    attachments: Iterable[MessageAttachment] | None = None,
    staged_sources: Iterable[ConsoleStagedSource] | None = None,
) -> ConsoleContextSnapshot:
    """Return a read-only snapshot of the current transcript and the assembled next-send payload.

    Skills with side effects are NOT executed; only chat dictionaries are applied.
    """
    session_id = self.store.active_session_id
    if not session_id:
        return ConsoleContextSnapshot(current_messages=[], next_send_payload={})

    current_messages = list(self.store.messages_for_session(session_id))

    # Build the next-send payload as submit_draft would, but do not persist.
    provider_messages = self._provider_messages_for_session(session_id)

    # Append a synthetic user turn for the draft so the preview matches what would be sent.
    if draft.strip():
        synthetic_user = self._provider_message_payloads(
            [
                ConsoleChatMessage(
                    role=ConsoleMessageRole.USER,
                    content=draft,
                    attachments=tuple(attachments or ()),
                )
            ],
            skip_failed=True,
        )
        # Replace image data with placeholders for the preview.
        synthetic_user = self._replace_image_data_with_placeholders(synthetic_user)
        provider_messages.extend(synthetic_user)

    # Do NOT call _apply_skill_substitution because it may execute skills with side effects.
    # Instead, annotate the final user message if it starts with a skill command.
    provider_messages = self._annotate_skill_commands(provider_messages)

    # Chat dictionaries are safe to apply (string replacements only).
    provider_messages = await self._apply_chat_dictionaries(provider_messages, session_id)

    # Gather native tool schemas and MCP note.
    tools_info = self._build_tools_info_for_snapshot()

    # Redact secrets before returning.
    redacted_messages = self._redact_secrets(provider_messages)

    # Deep-copy messages so the snapshot is independent of the store.
    from dataclasses import replace
    copied_messages = [replace(msg) for msg in current_messages]

    return ConsoleContextSnapshot(
        current_messages=copied_messages,
        next_send_payload={
            "model": self.model or self.configured_model,
            "messages": redacted_messages,
            "system": self._leading_system_message(),
            "staged_sources": [
                {"source_id": s.source_id, "label": s.label, "type": s.source_type}
                for s in (staged_sources or ())
            ],
            "tools": tools_info,
        },
    )
```

> Note: `_provider_messages_for_session` already includes the leading system message; the `system` field above is duplicated for clarity in the viewer. If the project prefers a single `messages` list, omit the separate `system` key.

Add helpers:

```python
@staticmethod
def _replace_image_data_with_placeholders(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import copy
    result = copy.deepcopy(messages)
    for message in result:
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    part["image_url"] = {"url": "[image: data redacted for preview]"}
                if isinstance(part, dict) and part.get("type") == "image":
                    part["image"] = "[image: data redacted for preview]"
    return result


@staticmethod
def _annotate_skill_commands(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import copy
    result = copy.deepcopy(messages)
    if result and result[-1].get("role") == "user":
        text = result[-1].get("content", "")
        if isinstance(text, str) and text.startswith("/"):
            result[-1]["content"] = (
                f"{text}\n\n[Skill command not resolved in preview; "
                "actual substitution happens at send time.]"
            )
    return result


def _build_tools_info_for_snapshot(self) -> dict[str, Any]:
    """Return native tool schemas and an MCP note for the snapshot."""
    tools: list[dict[str, Any]] = []
    if self._agent_bridge is not None:
        # Native tools only; live MCP catalog composition is out of scope.
        tools = getattr(self._agent_bridge, "native_tool_schemas", lambda: [])()
    mcp_note = "MCP tools are configured but live catalog composition is not shown in this preview."
    return {
        "native_schemas": tools,
        "mcp_note": mcp_note if self._mcp_provider else None,
    }
```

> Adjust attribute names (`_agent_bridge`, `_mcp_provider`) to match the actual controller fields.

Add a static helper for secrets redaction:

```python
_SECRET_REDACTION_KEYS = {"api_key", "apikey", "token", "password", "secret", "bearer"}


@staticmethod
def _redact_secrets(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a deep-copied payload with likely secret values replaced."""
    import copy
    redacted = copy.deepcopy(payload)

    def _redact_obj(obj: Any) -> Any:
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                lowered = key.lower()
                if any(secret_key in lowered for secret_key in ConsoleChatController._SECRET_REDACTION_KEYS):
                    result[key] = "[redacted]"
                else:
                    result[key] = _redact_obj(value)
            return result
        if isinstance(obj, list):
            return [_redact_obj(item) for item in obj]
        return obj

    return _redact_obj(redacted)
```

### Step 3.3: Write tests for context snapshot

Append to `Tests/Chat/test_console_chat_controller.py`:

```python
@pytest.mark.asyncio
async def test_build_context_snapshot_returns_current_and_next_send():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session(title="Chat 1")

    store.append_message(session.id, role=ConsoleMessageRole.USER, content="Hello")
    store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="Hi there")

    snapshot = await controller.build_context_snapshot(draft="Explain tools")

    assert len(snapshot.current_messages) == 2
    assert snapshot.current_messages[0].role == ConsoleMessageRole.USER
    assert snapshot.next_send_payload["messages"][-1]["content"].startswith("Explain tools")


@pytest.mark.asyncio
async def test_build_context_snapshot_does_not_execute_skills():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session(title="Chat 1")

    store.append_message(session.id, role=ConsoleMessageRole.USER, content="Hello")

    snapshot = await controller.build_context_snapshot(draft="/search tools")
    final_content = snapshot.next_send_payload["messages"][-1]["content"]
    assert "/search tools" in final_content
    assert "Skill command not resolved in preview" in final_content


@pytest.mark.asyncio
async def test_build_context_snapshot_redacts_secrets():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session(title="Chat 1")

    store.append_message(session.id, role=ConsoleMessageRole.USER, content="run")
    controller.system_prompt = "Use api_key=secret123"

    snapshot = await controller.build_context_snapshot(draft="ok")
    payload_text = str(snapshot.next_send_payload)
    assert "secret123" not in payload_text
    assert "[redacted]" in payload_text


@pytest.mark.asyncio
async def test_build_context_snapshot_is_immutable():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session(title="Chat 1")

    msg = store.append_message(session.id, role=ConsoleMessageRole.USER, content="Hello")

    snapshot = await controller.build_context_snapshot(draft="Follow up")
    original_content = snapshot.current_messages[0].content
    snapshot.current_messages[0].content = "mutated"

    reloaded = store.get_message(msg.id)
    assert reloaded.content == original_content
```

> Replace `StreamingGateway()` with the actual gateway class used by existing tests.

Run:

```bash
pytest Tests/Chat/test_console_chat_controller.py::test_build_context_snapshot_returns_current_and_next_send Tests/Chat/test_console_chat_controller.py::test_build_context_snapshot_redacts_secrets Tests/Chat/test_console_chat_controller.py::test_build_context_snapshot_is_immutable -v
```

Expected: PASS.

### Step 3.4: Commit

```bash
git add tldw_chatbook/Chat/console_chat_models.py tldw_chatbook/Chat/console_chat_controller.py Tests/Chat/test_console_chat_controller.py
git commit -m "feat(console): add build_context_snapshot for next-send context preview"
```

---

## Task 4: Build the context viewer modal

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_context_modal.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/UI/console_command_provider.py`
- Test: `Tests/UI/test_console_context_modal.py`

### Step 4.1: Create the modal widget (v1)

Create `tldw_chatbook/Widgets/Console/console_context_modal.py`:

```python
from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    Label,
    LoadingIndicator,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)
from textual.worker import Worker, WorkerState

from tldw_chatbook.Chat.console_chat_models import ConsoleContextSnapshot


SIZE_THRESHOLD_BYTES = 1 * 1024 * 1024


class ConsoleContextModal(ModalScreen[None]):
    DEFAULT_CSS = """
    ConsoleContextModal { align: center middle; }
    #console-context-modal { width: 95; height: 40; border: tall gray; }
    #console-context-header { height: auto; }
    #console-context-warning { height: auto; color: yellow; }
    #console-context-loading { display: none; }
    #console-context-loading.loading { display: block; }
    #console-context-tabs { height: 1fr; }
    #console-context-actions { height: auto; }
    """

    BINDINGS = [("escape", "dismiss", "Close"), ("r", "refresh", "Refresh")]

    snapshot = reactive(ConsoleContextSnapshot(current_messages=[], next_send_payload={}))
    raw_json = reactive(False)
    in_progress = reactive(False)
    token_estimate = reactive(None)
    loading = reactive(False)

    def __init__(
        self,
        snapshot: ConsoleContextSnapshot,
        snapshot_factory: Callable[[], Awaitable[ConsoleContextSnapshot]],
        *,
        token_estimate: int | None = None,
        in_progress: bool = False,
    ) -> None:
        super().__init__()
        self.snapshot = snapshot
        self._snapshot_factory = snapshot_factory
        self.token_estimate = token_estimate
        self.in_progress = in_progress

    def compose(self) -> ComposeResult:
        with Vertical(id="console-context-modal"):
            yield Static("Chat Context", id="console-context-header")
            yield Static("", id="console-context-warning")
            yield LoadingIndicator(id="console-context-loading")

            with TabbedContent(id="console-context-tabs"):
                with TabPane("Current", id="console-context-current"):
                    yield Vertical(id="console-context-current-body")
                with TabPane("Next Send", id="console-context-next-send"):
                    yield Vertical(id="console-context-next-send-body")

            with Horizontal(id="console-context-actions"):
                yield Checkbox("Raw JSON", id="console-context-raw")
                yield Button("Refresh", id="console-context-refresh", disabled=self.in_progress)
                yield Button("Copy JSON", id="console-context-copy")
                yield Button("Save to File", id="console-context-save")
                yield Button("Close", id="console-context-close")

    def on_mount(self) -> None:
        self._render()

    def watch_snapshot(self) -> None:
        self._render()

    def watch_raw_json(self) -> None:
        self._render()

    def watch_loading(self) -> None:
        loading = self.query_one("#console-context-loading", LoadingIndicator)
        if self.loading:
            loading.add_class("loading")
        else:
            loading.remove_class("loading")

    def _render(self) -> None:
        warning = self.query_one("#console-context-warning", Static)
        if self.in_progress:
            warning.update("A response is in progress; snapshot may change.")
        else:
            warning.update("")

        header = self.query_one("#console-context-header", Static)
        header_text = "Chat Context"
        if self.token_estimate is not None:
            header_text += f" (~{self.token_estimate} tokens)"
        header.update(header_text)

        current_container = self.query_one("#console-context-current-body", Vertical)
        current_container.remove_children()
        current_container.mount(self._build_current_context_view())

        next_container = self.query_one("#console-context-next-send-body", Vertical)
        next_container.remove_children()
        next_container.mount(self._build_next_send_view())

    def _build_current_context_view(self) -> Vertical:
        container = Vertical()
        if not self.snapshot.current_messages:
            container.mount(Label("No conversation context."))
            return container
        for msg in self.snapshot.current_messages:
            collapsible = Collapsible(title=f"[{msg.role}] {msg.status}", collapsed=True)
            collapsible.mount(TextArea(msg.content, read_only=True))
            container.mount(collapsible)
        return container

    def _build_next_send_view(self) -> Vertical:
        container = Vertical()
        payload = self.snapshot.next_send_payload
        text = self._format_next_send_text()

        if len(text.encode("utf-8")) > SIZE_THRESHOLD_BYTES:
            container.mount(Label(
                "Context exceeds 1 MiB. Use Save to File to view the full payload."
            ))
            return container

        if self.raw_json:
            container.mount(TextArea(text, read_only=True))
            return container

        model_collapsible = Collapsible(title="Model", collapsed=False)
        model_collapsible.mount(Label(str(payload.get("model", "unknown"))))
        container.mount(model_collapsible)

        system_collapsible = Collapsible(title="System", collapsed=True)
        system_collapsible.mount(TextArea(self._json_block(payload.get("system")), read_only=True))
        container.mount(system_collapsible)

        messages_collapsible = Collapsible(title="Messages", collapsed=False)
        for i, msg in enumerate(payload.get("messages", [])):
            msg_collapsible = Collapsible(title=f"Message {i}", collapsed=True)
            msg_collapsible.mount(TextArea(self._json_block(msg), read_only=True))
            messages_collapsible.mount(msg_collapsible)
        container.mount(messages_collapsible)

        tools = payload.get("tools")
        if tools:
            tools_collapsible = Collapsible(title="Tools", collapsed=True)
            tools_collapsible.mount(TextArea(self._json_block(tools), read_only=True))
            container.mount(tools_collapsible)

        staged = payload.get("staged_sources")
        if staged:
            staged_collapsible = Collapsible(title="Staged Sources", collapsed=True)
            staged_collapsible.mount(TextArea(self._json_block(staged), read_only=True))
            container.mount(staged_collapsible)

        return container

    def _format_next_send_text(self) -> str:
        import json
        return json.dumps(self.snapshot.next_send_payload, indent=2, default=str)

    @staticmethod
    def _json_block(obj: Any) -> str:
        import json
        return json.dumps(obj, indent=2, default=str)

    @on(Button.Pressed, "#console-context-close")
    def _close(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    @on(Button.Pressed, "#console-context-refresh")
    def _refresh(self, event: Button.Pressed) -> None:
        event.stop()
        self.run_worker(self._load_snapshot, exclusive=True)

    async def _load_snapshot(self) -> None:
        self.loading = True
        try:
            self.snapshot = await self._snapshot_factory()
        finally:
            self.loading = False

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.state == WorkerState.ERROR:
            self.loading = False
            self.notify("Failed to refresh context.", severity="error")

    @on(Checkbox.Changed, "#console-context-raw")
    def _toggle_raw(self, event: Checkbox.Changed) -> None:
        event.stop()
        self.raw_json = event.value

    @on(Button.Pressed, "#console-context-copy")
    def _copy_json(self, event: Button.Pressed) -> None:
        event.stop()
        import json

        text = json.dumps(self.snapshot.next_send_payload, indent=2, default=str)
        try:
            import pyperclip
            pyperclip.copy(text)
            self.notify("JSON copied to clipboard.")
        except Exception:
            self.notify("Copy failed: pyperclip unavailable.", severity="warning")

    @on(Button.Pressed, "#console-context-save")
    def _save_json(self, event: Button.Pressed) -> None:
        event.stop()
        import json
        from pathlib import Path
        from datetime import datetime

        text = json.dumps(self.snapshot.next_send_payload, indent=2, default=str)
        filename = f"chatbook_context_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = Path.home() / "Downloads" / filename
        path.write_text(text, encoding="utf-8")
        self.notify(f"Saved to {path}")

    def action_refresh(self) -> None:
        self.run_worker(self._load_snapshot, exclusive=True)
```

> Note: `LoadingIndicator` requires Textual ≥0.47. If the project uses an older version, replace it with a `Static("Loading…")` toggled via CSS.

### Step 4.2: Wire up keybinding and command palette

In `tldw_chatbook/UI/Screens/chat_screen.py`, add to `BINDINGS`:

```python
Binding("ctrl+shift+p", "view_chat_context", "View context", show=True),
```

Add the action method to `ChatScreen`:

```python
from tldw_chatbook.Widgets.Console.console_context_modal import ConsoleContextModal
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

async def action_view_chat_context(self) -> None:
    controller = self._console_chat_controller
    session_id = controller.store.active_session_id
    if not session_id:
        self.notify("No active conversation.", severity="warning")
        return

    composer = self.query_one("#console-native-composer", ConsoleComposerBar)
    draft = composer.value if composer else ""
    staged_sources = controller.store.workspace_context.allowed_sources
    attachments = controller.store.pending_attachments(session_id)

    async def _factory() -> ConsoleContextSnapshot:
        return await controller.build_context_snapshot(
            draft=draft,
            attachments=attachments,
            staged_sources=staged_sources,
        )

    try:
        snapshot = await _factory()
    except Exception as exc:
        self.notify(f"Could not build context snapshot: {exc}", severity="error")
        return

    token_estimate = self._estimate_tokens(snapshot.next_send_payload)
    in_progress = controller.run_state.status in (
        ConsoleRunStatus.STREAMING,
        ConsoleRunStatus.AGENT_RUNNING,
    )
    self.push_screen(ConsoleContextModal(
        snapshot,
        _factory,
        token_estimate=token_estimate,
        in_progress=in_progress,
    ))


def _estimate_tokens(self, payload: dict[str, Any]) -> int | None:
    text = str(payload)
    return int(len(text.split()) * 1.3)
```

> Replace `run_state`, `ConsoleRunStatus`, and store methods with the actual attributes/methods used by `ChatScreen`.

In `tldw_chatbook/UI/console_command_provider.py`, add to `_commands`:

```python
(
    "Console: View chat context",
    screen.action_view_chat_context,
    "Show current and next-send context (Ctrl+Shift+P)",
),
```

### Step 4.3: Write modal tests

Create `Tests/UI/test_console_context_modal.py`:

```python
import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Collapsible, Label, Static, TextArea

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleContextSnapshot,
    ConsoleMessageRole,
)
from tldw_chatbook.Widgets.Console.console_context_modal import ConsoleContextModal


SNAPSHOT = ConsoleContextSnapshot(
    current_messages=[
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="Hello"),
        ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="Hi"),
    ],
    next_send_payload={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
)

EMPTY_SNAPSHOT = ConsoleContextSnapshot(current_messages=[], next_send_payload={})


async def _snapshot_factory() -> ConsoleContextSnapshot:
    return SNAPSHOT


async def _empty_factory() -> ConsoleContextSnapshot:
    return EMPTY_SNAPSHOT


class ModalHarness(App):
    def compose(self) -> ComposeResult:
        yield Static("background")

    def on_mount(self) -> None:
        self.push_screen(ConsoleContextModal(SNAPSHOT, _snapshot_factory, token_estimate=42))


@pytest.mark.asyncio
async def test_context_modal_renders_tabs():
    app = ModalHarness()

    async with app.run_test(size=(100, 40)) as pilot:
        modal = app.screen
        header = modal.query_one("#console-context-header", Static)
        header_text = str(header.renderable)
        assert "Chat Context" in header_text
        assert "42 tokens" in header_text

        current_container = modal.query_one("#console-context-current-body", Vertical)
        text_areas = current_container.query(TextArea)
        assert any("Hello" in ta.text for ta in text_areas)

        next_container = modal.query_one("#console-context-next-send-body", Vertical)
        labels = list(next_container.query(Label))
        assert any("gpt-4" in str(label.renderable) for label in labels)


@pytest.mark.asyncio
async def test_context_modal_empty_state():
    app = ModalHarness()
    app._push_empty = lambda: app.push_screen(
        ConsoleContextModal(EMPTY_SNAPSHOT, _empty_factory)
    )

    async with app.run_test(size=(100, 40)) as pilot:
        app._push_empty()
        await pilot.pause()
        modal = app.screen
        current_container = modal.query_one("#console-context-current-body", Vertical)
        labels = list(current_container.query(Label))
        assert any("No conversation context" in str(label.renderable) for label in labels)


@pytest.mark.asyncio
async def test_context_modal_in_progress_warning():
    app = ModalHarness()
    app._push_in_progress = lambda: app.push_screen(
        ConsoleContextModal(SNAPSHOT, _snapshot_factory, in_progress=True)
    )

    async with app.run_test(size=(100, 40)) as pilot:
        app._push_in_progress()
        await pilot.pause()
        modal = app.screen
        warning = modal.query_one("#console-context-warning", Static)
        assert "in progress" in str(warning.renderable)
        refresh_button = modal.query_one("#console-context-refresh", Button)
        assert refresh_button.disabled
```

> The `query` method requires Textual 0.41+. If unavailable, use `query_one` with a descendant selector.

Run:

```bash
pytest Tests/UI/test_console_context_modal.py -v
```

Expected: PASS.

### Step 4.4: Add ChatScreen wiring tests

Create `Tests/UI/test_chat_screen_context_modal.py`:

```python
import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage, ConsoleMessageRole
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_context_modal import ConsoleContextModal


class ChatScreenHarness(App):
    def compose(self) -> ComposeResult:
        yield ChatScreen()


@pytest.mark.asyncio
async def test_chat_screen_keybinding_opens_context_modal():
    app = ChatScreenHarness()

    async with app.run_test(size=(120, 40)) as pilot:
        # Ensure an active session and a message exist.
        screen = app.screen
        controller = screen._console_chat_controller
        session = controller.store.ensure_session(title="Test")
        controller.store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="Hello",
        )
        await pilot.pause()

        await pilot.press("ctrl+shift+p")
        await pilot.pause()

        assert isinstance(app.screen, ConsoleContextModal)


@pytest.mark.asyncio
async def test_chat_screen_command_palette_opens_context_modal():
    app = ChatScreenHarness()

    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        controller = screen._console_chat_controller
        session = controller.store.ensure_session(title="Test")
        controller.store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="Hello",
        )
        await pilot.pause()

        # Open command palette and select the context command.
        await pilot.press("ctrl+backslash")
        await pilot.pause()
        await pilot.press("Console: View chat context")
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, ConsoleContextModal)
```

> Adjust the command palette key (`ctrl+backslash`) to the actual binding used by the project.

Run:

```bash
pytest Tests/UI/test_chat_screen_context_modal.py -v
```

Expected: PASS.

### Step 4.5: Commit

```bash
git add tldw_chatbook/Widgets/Console/console_context_modal.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/console_command_provider.py Tests/UI/test_console_context_modal.py Tests/UI/test_chat_screen_context_modal.py
git commit -m "feat(console): add context viewer modal and keybinding"
```

---

## Task 5: Integration and smoke tests

### Step 5.1: Run the full console test suites

```bash
pytest Tests/UI/test_console_native_transcript.py Tests/UI/test_console_context_modal.py Tests/UI/test_chat_screen_context_modal.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_agent_bridge.py -v
```

Expected: all PASS.

### Step 5.2: Manual smoke test checklist

1. Launch the app: `python3 -m tldw_chatbook.app`
2. Open the console, send a message, select it, then click empty space below the last message — action row should disappear.
3. Enable agent mode, ask "explain all tools available" — response should complete without a control error.
4. Press `Ctrl+Shift+P` (or run **Console: View chat context** from the command palette) — verify both tabs render.

### Step 5.3: Final commit

```bash
git add Tests/UI/test_console_native_transcript.py Tests/UI/test_console_context_modal.py Tests/UI/test_chat_screen_context_modal.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_agent_bridge.py
git commit -m "test(console): add integration tests and smoke checklist for console feedback"
```

---

## Notes & Risks

- `ConsoleTranscript` is nested inside `ConsoleTranscriptSurface` → `ConsoleSessionSurface`. The plan uses `#console-native-transcript` to query it, which matches the existing ID.
- `_provider_messages_for_session()` may include the leading system message inside `messages`. The `next_send_payload["system"]` field is included for viewer clarity; if the project convention is to keep system inside `messages`, remove the duplicate.
- The context modal intentionally does not execute skills with side effects. If a future requirement asks for fully resolved skill output in the preview, that should be a separate task.
- `pyperclip` may not be available; verify clipboard handling against existing utilities before implementing `_copy_json`.
