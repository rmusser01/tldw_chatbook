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
| `Tests/UI/test_console_context_modal.py` | Tests for the context viewer modal. |

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

NEGATIVE_SPACE_WIDGET_IDS = {
    "console-native-transcript",
}


def on_click(self, event: Click) -> None:
    if self.selected_message_id is None:
        return
    control = event.control
    if control is None:
        return
    if control.id in NEGATIVE_SPACE_WIDGET_IDS:
        self.action_clear_selection()
```

Place it next to the existing `action_clear_selection` method.

Run:

```bash
pytest Tests/UI/test_console_native_transcript.py::test_console_transcript_click_background_clears_selection -v
```

Expected: PASS.

### Step 3: Add negative test for action-row background

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
- Test: `Tests/Chat/test_console_chat_controller.py`

### Step 2.1: Inspect current `_finalize_agent_reply`

Open `tldw_chatbook/Chat/console_chat_controller.py:1661-1735`. Confirm the existing branches:

- `RUN_CANCELLED` → `_mark_stream_stopped`
- non-`RUN_DONE` → `mark_message_failed` + `_append_failure_system_row`
- `RUN_DONE` → `finalize_variant_stream` / `mark_message_complete`

No special handling exists for empty `final_text` or unknown statuses.

### Step 2.2: Write failing test for empty `final_text` fallback

Append to `Tests/Chat/test_console_chat_controller.py`:

```python
@pytest.mark.asyncio
async def test_finalize_agent_reply_empty_final_text_uses_fallback(controller, sample_session):
    from tldw_chatbook.Agents.agent_models import RUN_DONE

    session_id = sample_session.id
    assistant = controller.store.append_message(
        session_id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
    )

    outcome = RunOutcome(status=RUN_DONE, steps=[], final_text="")
    result = controller._finalize_agent_reply(
        assistant.id,
        session_id,
        outcome,
        variant_mode=False,
    )

    updated = controller.store.get_message(assistant.id)
    assert updated.status == "complete"
    assert "No response was generated" in updated.content
    assert result.success is True
```

Run:

```bash
pytest Tests/Chat/test_console_chat_controller.py::test_finalize_agent_reply_empty_final_text_uses_fallback -v
```

Expected: FAIL (status may be `complete` with empty content, or assertion fails).

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
        self.store.update_message_content(
            assistant_message_id,
            "No response was generated.",
        )

    if variant_mode:
        return self.store.finalize_variant_stream(assistant_message_id)
    return self.store.mark_message_complete(assistant_message_id)
```

> If `update_message_content` does not exist, use the store's existing mutation method (e.g., `set_message_content` or direct field update).

Run:

```bash
pytest Tests/Chat/test_console_chat_controller.py::test_finalize_agent_reply_empty_final_text_uses_fallback -v
```

Expected: PASS.

### Step 2.4: Write failing test for unknown/superseded status

Append to `Tests/Chat/test_console_chat_controller.py`:

```python
@pytest.mark.asyncio
async def test_finalize_agent_reply_unknown_status_is_failure(controller, sample_session):
    from tldw_chatbook.Agents.agent_models import RUN_ERROR

    session_id = sample_session.id
    assistant = controller.store.append_message(
        session_id,
        role=ConsoleMessageRole.ASSISTANT,
        content="partial",
    )

    outcome = RunOutcome(status="UNKNOWN_STATUS", steps=[], final_text="")
    result = controller._finalize_agent_reply(
        assistant.id,
        session_id,
        outcome,
        variant_mode=False,
    )

    updated = controller.store.get_message(assistant.id)
    assert updated.status == "failed"
    assert result.success is True  # finalize succeeded, message is failed
```

Run:

```bash
pytest Tests/Chat/test_console_chat_controller.py::test_finalize_agent_reply_unknown_status_is_failure -v
```

Expected: FAIL (unknown status may not enter the failure branch).

### Step 2.5: Implement unknown status handling

In `_finalize_agent_reply`, after the `RUN_CANCELLED` branch and before the `outcome.status != RUN_DONE` branch, add an explicit guard:

```python
from tldw_chatbook.Agents.agent_models import RUN_DONE, RUN_CANCELLED

# ... existing RUN_CANCELLED branch ...

if outcome.status != RUN_DONE:
    # RUN_ERROR, RUN_STUCK, RUN_SUPERSEDED, or any future status.
    visible_copy = self._agent_failure_visible_copy(outcome)
    try:
        failed = self.store.mark_message_failed(assistant_message_id)
    except KeyError:
        return self._session_closed_result()
    self._append_failure_system_row(session_id, visible_copy)
    self._set_run_state(ConsoleRunState(ConsoleRunStatus.FAILED, visible_copy))
    return ConsoleSubmitResult(True, True, failed.content)
```

Update `_agent_failure_visible_copy` to handle unknown statuses gracefully:

```python
@staticmethod
def _agent_failure_visible_copy(outcome: Any) -> str:
    from tldw_chatbook.Agents.agent_models import RUN_STUCK, STEP_ERROR
    reason = ""
    for step in reversed(getattr(outcome, "steps", None) or []):
        if getattr(step, "kind", None) == STEP_ERROR and getattr(step, "summary", ""):
            reason = step.summary
            break
    if outcome.status == RUN_STUCK:
        return f"Agent run stuck: {reason or 'budget or loop limit reached'}."
    if reason:
        return f"Agent run failed: {reason}."
    return f"Agent run failed: {outcome.status}."
```

Run:

```bash
pytest Tests/Chat/test_console_chat_controller.py::test_finalize_agent_reply_unknown_status_is_failure Tests/Chat/test_console_chat_controller.py::test_finalize_agent_reply_empty_final_text_uses_fallback -v
```

Expected: PASS.

### Step 2.6: Add structured diagnostics logging

In `_run_agent_reply`, before and after the `asyncio.to_thread` call, add log lines:

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

Inside `ConsoleAgentBridge.run_reply`, after each model turn is impractical without restructuring; instead log at the bridge entry/exit:

```python
logger.info(
    "console_agent_bridge_run",
    conversation_id=conversation_id,
    model=model,
    allowed_tools_count=len(allowed_tools),
)
```

Run the existing controller tests to ensure no regressions:

```bash
pytest Tests/Chat/test_console_chat_controller.py -v
```

Expected: all existing + new tests PASS.

### Step 2.7: Commit

```bash
git add Tests/Chat/test_console_chat_controller.py tldw_chatbook/Chat/console_chat_controller.py
git commit -m "feat(console): harden agent turn-control finalization and add diagnostics"
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
    """Return a read-only snapshot of the current transcript and the assembled next-send payload."""
    session = self.active_session
    if session is None:
        return ConsoleContextSnapshot(current_messages=[], next_send_payload={})

    current_messages = list(self.store.messages_for_session(session.id))

    # Build the next-send payload as submit_draft would, but do not persist.
    provider_messages = self._provider_messages_for_session(session.id)

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
        provider_messages.extend(synthetic_user)

    provider_messages, _ = await self._apply_skill_substitution(provider_messages)
    provider_messages = await self._apply_chat_dictionaries(provider_messages, session.id)

    # Redact secrets before returning.
    redacted = self._redact_secrets(provider_messages)

    return ConsoleContextSnapshot(
        current_messages=list(current_messages),
        next_send_payload={
            "model": self.model or self.configured_model,
            "messages": redacted,
            "system": self._leading_system_message(),
            "staged_sources": [
                {"source_id": s.source_id, "label": s.label, "type": s.source_type}
                for s in (staged_sources or ())
            ],
        },
    )
```

> Note: `_provider_messages_for_session` already includes the leading system message; the `system` field above is duplicated for clarity in the viewer. If the project prefers a single `messages` list, omit the separate `system` key.

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
async def test_build_context_snapshot_returns_current_and_next_send(controller, sample_session):
    controller.store.append_message(
        sample_session.id,
        role=ConsoleMessageRole.USER,
        content="Hello",
    )
    controller.store.append_message(
        sample_session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Hi there",
    )

    snapshot = await controller.build_context_snapshot(draft="Explain tools")

    assert len(snapshot.current_messages) == 2
    assert snapshot.current_messages[0].role == ConsoleMessageRole.USER
    assert snapshot.next_send_payload["messages"][-1]["content"] == "Explain tools"


@pytest.mark.asyncio
async def test_build_context_snapshot_redacts_secrets(controller, sample_session):
    controller.store.append_message(
        sample_session.id,
        role=ConsoleMessageRole.USER,
        content="run",
    )
    controller.session_settings.system_prompt = "Use api_key=secret123"

    snapshot = await controller.build_context_snapshot(draft="ok")
    payload_text = str(snapshot.next_send_payload)
    assert "secret123" not in payload_text
    assert "[redacted]" in payload_text


@pytest.mark.asyncio
async def test_build_context_snapshot_is_immutable(controller, sample_session):
    controller.store.append_message(
        sample_session.id,
        role=ConsoleMessageRole.USER,
        content="Hello",
    )

    snapshot = await controller.build_context_snapshot(draft="Follow up")
    original_content = snapshot.current_messages[0].content
    snapshot.current_messages[0].content = "mutated"

    reloaded = controller.store.get_message(snapshot.current_messages[0].id)
    assert reloaded.content == original_content
```

> Adjust the test fixtures (`controller`, `sample_session`) to match the actual fixture names in `Tests/Chat/test_console_chat_controller.py`.

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
- Test: `Tests/UI/test_console_context_modal.py`

### Step 4.1: Create the modal widget

Create `tldw_chatbook/Widgets/Console/console_context_modal.py`:

```python
from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static, TabbedContent, TabPane, TextArea

from tldw_chatbook.Chat.console_chat_models import ConsoleContextSnapshot


class ConsoleContextModal(ModalScreen[None]):
    DEFAULT_CSS = """
    ConsoleContextModal { align: center middle; }
    #console-context-modal { width: 95; height: 35; border: tall gray; }
    #console-context-header { height: auto; }
    #console-context-tabs { height: 1fr; }
    #console-context-actions { height: auto; }
    """

    BINDINGS = [("escape", "dismiss", "Close")]

    def __init__(
        self,
        snapshot: ConsoleContextSnapshot,
        *,
        token_estimate: int | None = None,
    ) -> None:
        super().__init__()
        self._snapshot = snapshot
        self._token_estimate = token_estimate

    def compose(self) -> ComposeResult:
        with Vertical(id="console-context-modal"):
            header_text = "Chat Context"
            if self._token_estimate is not None:
                header_text += f" (~{self._token_estimate} tokens)"
            yield Static(header_text, id="console-context-header")

            with TabbedContent(id="console-context-tabs"):
                with TabPane("Current", id="console-context-current"):
                    yield self._render_current_context()
                with TabPane("Next Send", id="console-context-next-send"):
                    yield self._render_next_send()

            with Horizontal(id="console-context-actions"):
                yield Button("Copy JSON", id="console-context-copy")
                yield Button("Save to File", id="console-context-save")
                yield Button("Close", id="console-context-close")

    def _render_current_context(self) -> TextArea:
        lines: list[str] = []
        for msg in self._snapshot.current_messages:
            lines.append(f"[{msg.role}] {msg.status}")
            lines.append(msg.content)
            lines.append("")
        if not lines:
            lines.append("No conversation context.")
        return TextArea("\n".join(lines), read_only=True)

    def _render_next_send(self) -> TextArea:
        import json

        payload = self._snapshot.next_send_payload
        text = json.dumps(payload, indent=2, default=str)
        return TextArea(text, read_only=True)

    @on(Button.Pressed, "#console-context-close")
    def _close(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    @on(Button.Pressed, "#console-context-copy")
    def _copy_json(self, event: Button.Pressed) -> None:
        event.stop()
        import json
        import pyperclip

        pyperclip.copy(json.dumps(self._snapshot.next_send_payload, indent=2, default=str))

    @on(Button.Pressed, "#console-context-save")
    def _save_json(self, event: Button.Pressed) -> None:
        event.stop()
        # Save-to-file implementation is a follow-up task; for now, raise a warning.
        self.notify("Save to file is not yet implemented.", severity="warning")
```

> If `pyperclip` is not a project dependency, replace the copy action with a call to the existing clipboard utility (search for `copy_to_clipboard` in the codebase) or remove the button.

### Step 4.2: Wire up keybinding and command palette

In `tldw_chatbook/UI/Screens/chat_screen.py`, add to `BINDINGS`:

```python
Binding("ctrl+shift+p", "view_chat_context", "View context", show=True),
```

Add the action method to `ChatScreen`:

```python
from tldw_chatbook.Widgets.Console.console_context_modal import ConsoleContextModal

async def action_view_chat_context(self) -> None:
    controller = self._controller
    composer = self.query_one("#console-composer", ConsoleComposer)
    draft = composer.value if composer else ""
    staged_sources = self._staged_sources  # or however the screen exposes staged sources
    attachments = self._pending_attachments  # or however the screen exposes pending attachments

    snapshot = await controller.build_context_snapshot(
        draft=draft,
        attachments=attachments,
        staged_sources=staged_sources,
    )

    token_estimate = self._estimate_tokens(snapshot.next_send_payload)
    self.push_screen(ConsoleContextModal(snapshot, token_estimate=token_estimate))


def _estimate_tokens(self, payload: dict[str, Any]) -> int | None:
    text = str(payload)
    return int(len(text.split()) * 1.3)
```

> Replace `#console-composer`, `_staged_sources`, and `_pending_attachments` with the actual widget IDs / attributes used by `ChatScreen`.

In `tldw_chatbook/UI/console_command_provider.py`, add to `_commands`:

```python
(
    "Console: View chat context",
    screen.action_view_chat_context,
    "Show current and next-send context (Ctrl+Shift+P)",
),
```

### Step 4.3: Write a modal test

Create `Tests/UI/test_console_context_modal.py`:

```python
import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleContextSnapshot,
    ConsoleMessageRole,
)
from tldw_chatbook.Widgets.Console.console_context_modal import ConsoleContextModal


class ModalHarness(App):
    def compose(self) -> ComposeResult:
        snapshot = ConsoleContextSnapshot(
            current_messages=[
                ConsoleChatMessage(role=ConsoleMessageRole.USER, content="Hello"),
                ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="Hi"),
            ],
            next_send_payload={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
        )
        yield ConsoleContextModal(snapshot, token_estimate=42)


@pytest.mark.asyncio
async def test_context_modal_renders_tabs():
    app = ModalHarness()

    async with app.run_test(size=(100, 40)) as pilot:
        text = app.screen.query_one("#console-context-modal").display_text
        assert "Current" in text
        assert "Next Send" in text
        assert "42 tokens" in text
```

> `display_text` may need to be replaced with the actual way to extract visible text from the modal (e.g., `pilot.screen.display_text` or a helper).

Run:

```bash
pytest Tests/UI/test_console_context_modal.py -v
```

Expected: PASS.

### Step 4.4: Commit

```bash
git add tldw_chatbook/Widgets/Console/console_context_modal.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/console_command_provider.py Tests/UI/test_console_context_modal.py
git commit -m "feat(console): add context viewer modal and keybinding"
```

---

## Task 5: Integration and smoke tests

### Step 5.1: Run the full console test suites

```bash
pytest Tests/UI/test_console_native_transcript.py Tests/UI/test_console_context_modal.py Tests/Chat/test_console_chat_controller.py -v
```

Expected: all PASS.

### Step 5.2: Manual smoke test checklist

1. Launch the app: `python3 -m tldw_chatbook.app`
2. Open the console, send a message, select it, then click empty space below the last message — action row should disappear.
3. Enable agent mode, ask "explain all tools available" — response should complete without a control error.
4. Press `Ctrl+Shift+P` (or run **Console: View chat context** from the command palette) — verify both tabs render.

### Step 5.3: Final commit

```bash
git add -A
git commit -m "test(console): add integration tests and smoke checklist for console feedback"
```

---

## Notes & Risks

- `ConsoleTranscript` is nested inside `ConsoleTranscriptSurface` → `ConsoleSessionSurface`. The plan uses `#console-native-transcript` to query it, which matches the existing ID.
- `_provider_messages_for_session()` may include the leading system message inside `messages`. The `next_send_payload["system"]` field is included for viewer clarity; if the project convention is to keep system inside `messages`, remove the duplicate.
- The context modal intentionally does not execute skills with side effects. If a future requirement asks for fully resolved skill output in the preview, that should be a separate task.
- `pyperclip` may not be available; verify clipboard handling against existing utilities before implementing `_copy_json`.
