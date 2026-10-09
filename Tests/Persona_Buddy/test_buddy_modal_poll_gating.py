"""Buddy modal poll gating: fingerprint-first ticks and adaptive cadence (task-10).

Hermetic doubles only: the store/controller/coordinator behind the modal are
fakes with spy counters, so the gate/interval/dedup contracts below observe
the modal's poll path directly instead of a full Console runtime.
"""

from types import SimpleNamespace

import pytest
from textual.app import App
from textual.widgets import Static

from tldw_chatbook.Persona_Buddy.interaction import BuddyBinding
from tldw_chatbook.Widgets.Persona_Widgets.buddy_conversation_modal import (
    BuddyConversationModal,
)

BINDING = BuddyBinding(kind="conversation", target_id="sess-1")


def _message(role: str, content: str) -> SimpleNamespace:
    return SimpleNamespace(role=SimpleNamespace(value=role), content=content)


class FakeStore:
    """Transcript double exposing the store's two read seams with spies."""

    def __init__(self, messages):
        self.messages = list(messages)
        self.snapshot_calls = 0
        self.fingerprint_calls = 0

    def messages_for_session(self, session_id: str):
        self.snapshot_calls += 1
        return list(self.messages)

    def session_fingerprint(self, session_id: str):
        self.fingerprint_calls += 1
        if not self.messages:
            return (0, 0)
        return (len(self.messages), len(self.messages[-1].content))


class FakeController:
    def __init__(self, store):
        self.store = store
        self.run_state = SimpleNamespace(
            status=SimpleNamespace(value="idle"), is_send_allowed=True
        )

    def run_state_for(self, session_id: str):
        return self.run_state


class FakeCoordinator:
    def __init__(self, app, store, controller, session):
        self.app = app
        self.store = store
        self.controller = controller
        self.session = session
        self.drafts = {}
        self.notices = {}
        self.submitting = set()
        self.payloads = {}
        self.show_decisions_calls = 0
        self.show_decision_ids = []
        self.close_voice_calls = 0

    def resolve(self, binding):
        return self.session

    def show_decisions(self, owner, binding, *, decision_id=None):
        self.show_decisions_calls += 1
        self.show_decision_ids.append(decision_id)

    def close_voice(self, owner):
        self.close_voice_calls += 1

    def can_open_console(self, binding):
        return True

    def decision_payloads(self, binding):
        return self.payloads

    def voice_status(self, owner):
        return "idle"


class ModalShellApp(App):
    def __init__(self, messages=()):
        super().__init__()
        self.store = FakeStore(messages)
        self.controller = FakeController(self.store)
        self.session = SimpleNamespace(id="sess-1", title="Bound conversation")
        self.coordinator = FakeCoordinator(
            self, self.store, self.controller, self.session
        )

    def on_mount(self) -> None:
        self.push_screen(
            BuddyConversationModal(self.coordinator, BINDING, allow_voice=False)
        )


def _spy_static_updates(modal) -> list:
    """Wrap update() on every projection Static; returns [counter]."""
    counter = [0]
    originals = []

    def spy(widget_id):
        widget = modal.query_one(widget_id, Static)
        original = widget.update
        originals.append((widget, original))

        def counting_update(*args, **kwargs):
            counter[0] += 1
            return original(*args, **kwargs)

        widget.update = counting_update

    for widget_id in (
        "#buddy-conversation-title",
        "#buddy-activity",
        "#buddy-transcript",
        "#buddy-reply-notice",
    ):
        spy(widget_id)
    return counter


async def test_idle_ticks_perform_no_snapshot_and_no_dom_work():
    app = ModalShellApp([_message("user", "hello" * 200) for _ in range(80)])
    async with app.run_test(size=(80, 24)) as pilot:
        modal = app.screen
        await pilot.pause()
        app.store.snapshot_calls = 0
        app.store.fingerprint_calls = 0
        app.coordinator.show_decisions_calls = 0
        updates = _spy_static_updates(modal)

        for _ in range(5):
            modal._on_poll_tick()

        assert app.store.snapshot_calls == 0
        assert app.store.fingerprint_calls == 5
        assert app.coordinator.show_decisions_calls == 0
        assert updates[0] == 0


async def test_streaming_ticks_render_once_per_fingerprint_change():
    messages = [_message("user", "question"), _message("assistant", "answer")]
    app = ModalShellApp(messages)
    async with app.run_test(size=(80, 24)) as pilot:
        modal = app.screen
        await pilot.pause()
        app.store.snapshot_calls = 0
        app.coordinator.show_decisions_calls = 0
        transcript_updates = _spy_static_updates(modal)

        for token in ("one", "two", "three"):
            messages[-1].content += f" {token}"
            modal._on_poll_tick()

        assert app.store.snapshot_calls == 3
        assert app.coordinator.show_decisions_calls == 3
        assert "three" in str(modal.query_one("#buddy-transcript", Static).render())
        assert transcript_updates[0] >= 1


async def test_decisions_coordinator_invoked_at_most_once_per_tick():
    messages = [_message("user", "q"), _message("assistant", "a")]
    app = ModalShellApp(messages)
    async with app.run_test(size=(80, 24)) as pilot:
        modal = app.screen
        await pilot.pause()
        app.coordinator.show_decisions_calls = 0

        for index in range(4):
            messages[-1].content += f" more{index}"
            modal._on_poll_tick()

        assert app.coordinator.show_decisions_calls == 4


async def test_poll_interval_decays_when_idle_and_tightens_when_active():
    app = ModalShellApp([_message("user", "hello")])
    async with app.run_test(size=(80, 24)) as pilot:
        modal = app.screen
        await pilot.pause()
        # Fresh mount: the fast cadence trails the initial render so armed
        # decisions and started runs are picked up at the legacy latency.
        assert modal._poll_interval == pytest.approx(0.2)

        # Quiet idle ticks decay to the slow cadence...
        for _ in range(6):
            modal._on_poll_tick()
        assert modal._poll_interval == pytest.approx(1.0)
        assert modal._timer._interval == pytest.approx(1.0)

        # ...a busy run state tightens immediately, even on a gated tick.
        app.controller.run_state = SimpleNamespace(
            status=SimpleNamespace(value="streaming"), is_send_allowed=False
        )
        modal._on_poll_tick()
        assert modal._poll_interval == pytest.approx(0.2)
        assert modal._timer._interval == pytest.approx(0.2)

        # Back to idle with an observed store change: fast trail first...
        app.controller.run_state = SimpleNamespace(
            status=SimpleNamespace(value="idle"), is_send_allowed=True
        )
        app.store.messages[0].content += " changed"
        modal._on_poll_tick()
        assert modal._poll_interval == pytest.approx(0.2)
        # ...then the quiet decay settles the modal back to slow.
        for _ in range(6):
            modal._on_poll_tick()
        assert modal._poll_interval == pytest.approx(1.0)

        # A pending decision holds the fast cadence while it awaits the user.
        app.coordinator.payloads = {
            "approval": {"_decision_id": "d1", "round_id": "r1", "calls": []}
        }
        modal._on_poll_tick()
        assert modal._poll_interval == pytest.approx(0.2)


async def test_transcript_rendering_matches_legacy_projection():
    messages = [_message("user", f"Question {n}\nline {n}") for n in range(45)] + [
        _message("assistant", f"Answer {n}") for n in range(35)
    ]
    expected = "\n\n".join(
        f"{message.role.value.title()}: {message.content}" for message in messages[-60:]
    )[-64000:]
    app = ModalShellApp(messages)
    async with app.run_test(size=(80, 24)) as pilot:
        modal = app.screen
        await pilot.pause()
        assert str(modal.query_one("#buddy-transcript", Static).render()) == expected

        app.store.messages.append(_message("assistant", "final word"))
        expected_final = "\n\n".join(
            f"{message.role.value.title()}: {message.content}"
            for message in app.store.messages[-60:]
        )[-64000:]
        modal._on_poll_tick()
        assert (
            str(modal.query_one("#buddy-transcript", Static).render()) == expected_final
        )


async def test_empty_transcript_keeps_no_messages_yet_placeholder():
    app = ModalShellApp([])
    async with app.run_test(size=(80, 24)) as pilot:
        modal = app.screen
        await pilot.pause()
        assert "No messages yet" in str(
            modal.query_one("#buddy-transcript", Static).render()
        )


async def test_forced_refresh_renders_despite_stable_fingerprint():
    app = ModalShellApp([_message("user", "hello")])
    async with app.run_test(size=(80, 24)) as pilot:
        modal = app.screen
        await pilot.pause()
        app.store.snapshot_calls = 0
        updates = _spy_static_updates(modal)

        modal.refresh_projection()

        assert app.store.snapshot_calls == 1
        assert updates[0] >= 1


async def test_failed_render_releases_rendered_decision_claim():
    """A mid-body render failure must leave no rendered claim (task-10 dedup).

    The pre-change refresh registered the claim-less decision view before
    rendering; dropping that early call required encoding the same end
    state as an explicit failure path.
    """
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard

    app = ModalShellApp([_message("user", "hello")])
    app.coordinator.payloads = {
        "approval": {
            "_decision_id": "d1",
            "round_id": "r1",
            "calls": [
                {
                    "llm_name": "write_file",
                    "server_key": "agent:builtin",
                    "tool_name": "write_file",
                    "server_label": "Built-in",
                    "arguments": {},
                    "reason": "risk_floored",
                }
            ],
        }
    }
    async with app.run_test(size=(80, 24)) as pilot:
        modal = app.screen
        await pilot.pause()
        assert app.coordinator.show_decision_ids[-1] == "d1"

        def fail_render(*args, **kwargs):
            raise RuntimeError("card render failed")

        card = modal.query_one(ChatApprovalCard)
        card.set_batch = fail_render
        with pytest.raises(RuntimeError, match="card render failed"):
            modal.refresh_projection()
        assert app.coordinator.show_decision_ids[-1] is None


# -- store seam: the fingerprint the gate reads (real in-memory store) --


def test_session_fingerprint_tracks_count_and_streamed_length():
    from tldw_chatbook.Chat.console_chat_store import (
        ConsoleChatStore,
        ConsoleMessageRole,
    )

    store = ConsoleChatStore()
    session = store.create_session()
    assert store.session_fingerprint(session.id) == (0, 0)

    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hello")
    assert store.session_fingerprint(session.id) == (1, 5)

    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    assert store.session_fingerprint(session.id) == (2, 0)

    # Buffered stream growth is visible WITHOUT folding the buffer: the
    # live message body stays untouched until a real read materializes it.
    store.append_stream_chunk(assistant.id, "abc")
    store.append_stream_chunk(assistant.id, "defg")
    assert store.session_fingerprint(session.id) == (2, 7)
    assert store.messages_for_session(session.id)[-1].content == "abcdefg"


def test_session_fingerprint_unknown_session_raises():
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    store = ConsoleChatStore()
    with pytest.raises(KeyError):
        store.session_fingerprint("missing-session")
