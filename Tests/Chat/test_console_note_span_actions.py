"""Controller-level tests for the TASK-31759 More-menu note actions.

Covers ``ConsoleChatController.build_transcript_note`` and
``summarize_span_as_note``: span slicing (inclusive, active-path only,
USER/ASSISTANT rows), note formatting, gates, and the core contract that
the summarize path is STATELESS with respect to compaction -- no attempt
ledger rows, no context-summary boundary move. Reuses the fake-gateway
harness shape from ``test_console_rewind_summarize.py``.
"""

import asyncio

import pytest

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_controller import (
    ConsoleChatController,
    ConsoleNoteDraft,
    ConsoleSubmitResult,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_provider_gateway import (
    AuxiliaryCompletionResult,
    ConsoleProviderResolution,
)
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from Tests.console_provider_doubles import provider_resolution


class NoteSummaryGateway:
    """Fake gateway capturing the auxiliary summary call."""

    def __init__(self, summary: str = "NOTE SUMMARY", ready: bool = True) -> None:
        self.summary = summary
        self.ready = ready
        self.captured_auxiliary = None
        self.calls = 0

    async def resolve_for_send(self, selection):
        destination = provider_resolution(
            ready=True,
            provider="llama_cpp",
            model="test-model",
            base_url="http://127.0.0.1:9099",
        ).resolved_destination
        return ConsoleProviderResolution(
            ready=self.ready,
            provider="llama_cpp",
            model="test-model",
            base_url="http://127.0.0.1:9099",
            max_tokens=512,
            visible_copy="" if self.ready else "Provider blocked: no key.",
            resolved_destination=destination if self.ready else None,
        )

    async def complete_auxiliary(self, request):
        self.calls += 1
        self.captured_auxiliary = request
        return AuxiliaryCompletionResult(
            provider=request.resolution.provider,
            model=request.resolution.model or "test-model",
            text=self.summary,
            usage=ProviderUsage(
                uncached_input=20,
                output=5,
                provider=request.resolution.provider,
                model=request.resolution.model or "test-model",
            ),
        )

    async def stream_chat(self, resolution, messages, **kwargs):  # pragma: no cover
        yield ""


def _note_controller(tmp_path, *, gateway=None):
    db = CharactersRAGDB(tmp_path / "note-span.sqlite", "note-span")
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.create_session()
    store.persist_session_if_needed(session.id)
    resolved_gateway = gateway or NoteSummaryGateway()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=resolved_gateway,
    )
    return controller, store, session, resolved_gateway, db


def _seed_rows(store, session_id):
    """U1/A1/U2/A2 then a TOOL row then U3/A3; returns the appended rows."""
    rows = []
    for role, content in (
        (ConsoleMessageRole.USER, "first question"),
        (ConsoleMessageRole.ASSISTANT, "first answer"),
        (ConsoleMessageRole.USER, "second question"),
        (ConsoleMessageRole.ASSISTANT, "second answer"),
        (ConsoleMessageRole.TOOL, "tool noise"),
        (ConsoleMessageRole.USER, "third question"),
        (ConsoleMessageRole.ASSISTANT, "third answer"),
    ):
        rows.append(store.append_message(session_id, role=role, content=content))
    return tuple(rows)


@pytest.mark.asyncio
async def test_build_transcript_note_is_inclusive_and_user_assistant_only(
    tmp_path,
):
    controller, store, session, _gateway, _db = _note_controller(tmp_path)
    rows = _seed_rows(store, session.id)

    draft = controller.build_transcript_note(rows[3].id)

    assert isinstance(draft, ConsoleNoteDraft)
    assert draft.title == "Transcript: first question"
    assert draft.content.startswith("> Generated: ")
    assert "**User:** first question" in draft.content
    assert "**Assistant:** first answer" in draft.content
    assert "**Assistant:** second answer" in draft.content
    # Inclusive boundary: nothing AFTER the selected row, and the TOOL row
    # never appears.
    assert "third" not in draft.content
    assert "tool noise" not in draft.content


@pytest.mark.asyncio
async def test_build_transcript_note_accepts_assistant_target(tmp_path):
    controller, store, session, _gateway, _db = _note_controller(tmp_path)
    rows = _seed_rows(store, session.id)

    draft = controller.build_transcript_note(rows[6].id)

    assert isinstance(draft, ConsoleNoteDraft)
    assert "**Assistant:** third answer" in draft.content


@pytest.mark.asyncio
async def test_note_actions_block_off_path_and_unknown_targets(tmp_path):
    controller, store, session, _gateway, _db = _note_controller(tmp_path)
    _rows = _seed_rows(store, session.id)

    blocked_transcript = controller.build_transcript_note("no-such-message")
    assert isinstance(blocked_transcript, ConsoleSubmitResult)
    assert blocked_transcript.accepted is False
    assert blocked_transcript.visible_copy

    blocked_summary = await controller.summarize_span_as_note("no-such-message")
    assert isinstance(blocked_summary, ConsoleSubmitResult)
    assert blocked_summary.accepted is False
    assert blocked_summary.visible_copy


@pytest.mark.asyncio
async def test_summarize_span_as_note_returns_draft_without_compaction_writes(
    tmp_path,
):
    controller, store, session, gateway, db = _note_controller(tmp_path)
    rows = _seed_rows(store, session.id)

    draft = await controller.summarize_span_as_note(rows[3].id)

    assert isinstance(draft, ConsoleNoteDraft)
    assert draft.title == "Summary: first question"
    assert "NOTE SUMMARY" in draft.content
    assert draft.content.startswith("> Generated: ")
    # The auxiliary call carried the note-summarize system prompt and the
    # span text (inclusive, tool-free) as the user message.
    assert gateway.calls == 1
    sent = list(gateway.captured_auxiliary.messages)
    assert sent[0]["role"] == "system"
    assert "saved as a note" in sent[0]["content"]
    user_text = sent[1]["content"]
    assert "User: first question" in user_text
    assert "Assistant: second answer" in user_text
    assert "third" not in user_text
    assert "tool noise" not in user_text
    # Stateless contract: the rewind context summary and boundary are
    # untouched, and no auxiliary attempt ledger rows were written.
    assert store.session_context_summary(session.id) == (None, None)
    with db.get_connection() as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM console_auxiliary_attempts"
        ).fetchone()[0]
    assert count == 0


@pytest.mark.asyncio
async def test_summarize_span_as_note_blocks_when_provider_not_ready(tmp_path):
    controller, store, session, _gateway, _db = _note_controller(
        tmp_path, gateway=NoteSummaryGateway(ready=False)
    )
    rows = _seed_rows(store, session.id)

    blocked = await controller.summarize_span_as_note(rows[3].id)

    assert isinstance(blocked, ConsoleSubmitResult)
    assert blocked.accepted is False
    assert "Provider blocked" in blocked.visible_copy


@pytest.mark.asyncio
async def test_summarize_span_as_note_blocks_oversized_span(tmp_path, monkeypatch):
    controller, store, session, gateway, _db = _note_controller(tmp_path)
    rows = _seed_rows(store, session.id)
    # Shrink the span budget so the seeded span blows past it; the action
    # must BLOCK (user-visible copy), never silently trim oldest turns --
    # a note that quietly omits turns is worse than no note.
    monkeypatch.setattr(controller, "_SUMMARY_SPAN_TOKEN_BUDGET", 3)

    blocked = await controller.summarize_span_as_note(rows[3].id)

    assert isinstance(blocked, ConsoleSubmitResult)
    assert blocked.accepted is False
    assert "too large" in blocked.visible_copy
    assert gateway.calls == 0

# --- UI dispatch wiring (TASK-31759) ---------------------------------------


def _build_screen():
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "test-model"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "test-model"}
    }
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    return app, ChatScreen(app)


def _dispatch_event(action_id: str, message_id: str):
    from types import SimpleNamespace

    return SimpleNamespace(
        button=SimpleNamespace(
            id=f"console-message-action-{action_id}-{message_id}"
        ),
        stop=lambda: None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("action_id", ["summarize-note", "save-transcript-note"])
async def test_note_actions_dispatch_one_exclusive_note_worker(action_id):
    """Each More-menu note action routes to the shared console-note-actions
    worker group (never console-run, so it can never cancel a live stream)."""
    import asyncio
    from types import SimpleNamespace

    app, screen = _build_screen()
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="q")
    completed = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="done."
    )

    spawned = []

    def fake_run_worker(work, **kwargs):
        spawned.append(kwargs)
        if asyncio.iscoroutine(work):
            work.close()
        return SimpleNamespace(cancel=lambda: None)

    screen.run_worker = fake_run_worker

    handled = await screen.handle_console_message_action(
        _dispatch_event(action_id, completed.id)
    )

    assert handled is True
    assert len(spawned) == 1
    assert spawned[0].get("group") == "console-note-actions"
    assert spawned[0].get("exclusive") is True


@pytest.mark.asyncio
@pytest.mark.parametrize("action_id", ["summarize-note", "save-transcript-note"])
async def test_note_actions_mid_run_notify_instead_of_summarizing(action_id):
    """With a run streaming, the note actions must not reach a provider
    call: the worker resolves to the controller's active-run rejection and
    surfaces it as a notice."""
    import asyncio
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleRunState,
        ConsoleRunStatus,
    )

    app, screen = _build_screen()
    store = screen._ensure_console_chat_store()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="q")
    completed = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="done."
    )
    notices: list[str] = []

    tasks: list[asyncio.Future] = []

    def capture_worker(work, **kwargs):
        notices.append("spawned")
        tasks.append(asyncio.ensure_future(work))
        return SimpleNamespace(cancel=lambda: None)

    screen.run_worker = capture_worker
    app.notify = lambda message, **kwargs: notices.append(str(message))
    screen._ensure_console_chat_controller()._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STREAMING, "Streaming response.")
    )

    handled = await screen.handle_console_message_action(
        _dispatch_event(action_id, completed.id)
    )

    assert handled is True
    for task in tasks:
        await task
    # The worker ran to completion and the controller's run gate produced
    # the user-visible rejection.
    assert any("run" in note.lower() for note in notices), notices
