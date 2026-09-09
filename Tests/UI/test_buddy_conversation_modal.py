"""Mounted exact-owner Buddy commands over the real Console controller/store."""

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.app import ComposeResult
from textual.widgets import Button, Input, TextArea

from Tests.Chat.test_console_runtime_lifetime import (
    ConsoleChatStore,
    _pending_call,
    _StalledGateway,
)
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Persona_Buddy.interaction import BuddyBinding


def pending_image():
    from tldw_chatbook.Chat.attachment_core import PendingAttachment

    return PendingAttachment(
        file_path="/tmp/buddy-private.png",
        display_name="private.png",
        file_type="image",
        insert_mode="attachment",
        data=b"private-image",
        mime_type="image/png",
        original_size=13,
        processed_size=13,
    )


async def until(predicate):
    deadline = asyncio.get_running_loop().time() + 5
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline
        await asyncio.sleep(0.01)


class Harness(ConsolidatedCSSApp):
    CSS_PATH = str(
        Path(__file__).resolve().parents[2] / "tldw_chatbook/css/tldw_cli_modular.tcss"
    )

    def __init__(self):
        super().__init__()
        self.gateway = _StalledGateway()
        self.store = ConsoleChatStore()
        self.target = self.store.create_session()
        self.target.title = "Bound conversation"
        self.other = self.store.create_session()
        self.other.draft = "Unrelated Console draft"
        self.controller = ConsoleChatController(
            store=self.store, provider_gateway=self.gateway
        )
        self.controller.app = self
        self.controller.on_submission_accepted = lambda: setattr(
            self.other, "draft", ""
        )
        self.controller.set_pending_approval = lambda payload: None
        self.controller._interrupt_host.POLL_SECONDS = 0.01
        self.console_runtime = ConsoleRuntime(self)
        self.console_runtime.set_chat_store(self.store)
        self.console_runtime.set_chat_controller(self.controller)

    def compose(self) -> ComposeResult:
        yield Input(id="underlying-input")

    async def on_unmount(self):
        await self.console_runtime.dispose()


@pytest.mark.asyncio
async def test_buddy_send_targets_bound_session_and_survives_modal_close():
    from tldw_chatbook.UI.Navigation.buddy_conversation import open_buddy_conversation

    app = Harness()
    async with app.run_test(size=(100, 36)) as pilot:
        opener = app.query_one(Input)
        opener.focus()
        base = app.screen
        binding = BuddyBinding.for_session(app.target)
        other_attachment = pending_image()
        app.store.add_pending_attachment(app.other.id, other_attachment)
        modal = open_buddy_conversation(app, binding, allow_voice=False)
        await pilot.pause()
        modal.query_one("#buddy-reply", TextArea).load_text("Directed reply")
        modal.query_one("#buddy-send", Button).press()
        await until(app.gateway.started.is_set)
        assert app.store.active_session_id == app.other.id
        assert app.other.draft == "Unrelated Console draft"
        assert app.store.pending_attachments(app.other.id) == [other_attachment]
        assert (
            app.store.messages_for_session(app.target.id)[0].content == "Directed reply"
        )
        modal.query_one("#buddy-close", Button).press()
        await until(lambda: app.screen is base)
        assert app.controller.in_flight_run_count() == 1
        assert app.focused is opener
        app.gateway.never_release.set()
        await until(lambda: app.controller.in_flight_run_count() == 0)
        assert (
            app.store.messages_for_session(app.target.id)[-1].content == "partialnever"
        )
        assert not app.store.messages_for_session(app.other.id)


@pytest.mark.asyncio
async def test_buddy_drafts_and_approval_use_exact_binding():
    from tldw_chatbook.UI.Navigation.buddy_conversation import open_buddy_conversation

    app = Harness()
    async with app.run_test(size=(100, 36)) as pilot:
        binding = BuddyBinding.for_session(app.target)
        modal = open_buddy_conversation(app, binding, allow_voice=False)
        await pilot.pause()
        modal.query_one("#buddy-reply", TextArea).load_text("Keep this draft")
        modal.query_one("#buddy-close", Button).press()
        await until(lambda: app.screen is not modal)
        modal = open_buddy_conversation(app, binding, allow_voice=False)
        await pilot.pause()
        assert modal.query_one("#buddy-reply", TextArea).text == "Keep this draft"
        pending = asyncio.create_task(
            asyncio.to_thread(
                app.controller.request_mcp_approvals,
                [_pending_call()],
                session_id=app.target.id,
            )
        )
        try:
            await until(lambda: app.controller._interrupt_host.pending_total() == 1)
            await pilot.pause(0.3)
            from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
                ChatApprovalCard,
            )

            card = modal.query_one(ChatApprovalCard)
            assert card.display
            round_id = app.controller._interrupt_host.head_round_payload(
                "approval", app.target.id
            )["round_id"]
            # A forged round belonging to the other conversation never resolves.
            assert (
                modal.coordinator.resolve_decision(
                    binding, "approval", "wrong-round", {}
                )
                is False
            )
            card.post_message(
                ChatApprovalCard.ApprovalDecided({"c1": "deny"}, round_id=round_id)
            )
            await asyncio.wait_for(pending, 3)
            assert app.store.active_session_id == app.other.id
        finally:
            if not pending.done():
                app.controller.begin_shutdown()
                await pending


@pytest.mark.asyncio
async def test_closed_owner_does_not_fall_back_or_start_microphone():
    from tldw_chatbook.UI.Navigation.buddy_conversation import open_buddy_conversation

    app = Harness()
    async with app.run_test(size=(90, 28)) as pilot:
        binding = BuddyBinding.for_session(app.target)
        app.store.close_session(app.target.id)
        modal = open_buddy_conversation(app, binding)
        await pilot.pause()
        assert modal.query_one("#buddy-send", Button).disabled
        assert modal.query_one("#buddy-mic", Button).disabled
        assert app.store.active_session_id == app.other.id


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("question", "skill_install", "skill_script"))
async def test_buddy_existing_decision_cards_resolve_only_the_bound_round(kind):
    from Tests.Chat.test_console_ask_user_round import _questions
    from tldw_chatbook.UI.Navigation.buddy_conversation import open_buddy_conversation
    from tldw_chatbook.Widgets.Chat_Widgets.chat_question_card import ChatQuestionCard
    from tldw_chatbook.Widgets.Chat_Widgets.chat_task_cards import ChatTaskCards
    from tldw_chatbook.Widgets.Chat_Widgets.skill_install_confirm_card import (
        SkillInstallConfirmCard,
    )
    from tldw_chatbook.Widgets.Chat_Widgets.skill_script_confirm_card import (
        SkillScriptConfirmCard,
    )

    app = Harness()
    async with app.run_test(size=(100, 36)) as pilot:
        setattr(app.controller, "set_pending_" + kind, lambda payload: None)
        requests = {
            "question": lambda: app.controller.request_user_questions(
                _questions(), session_id=app.target.id
            ),
            "skill_install": lambda: app.controller.request_skill_install_confirm(
                "https://example.com/skill", session_id=app.target.id
            ),
            "skill_script": lambda: app.controller.request_skill_script_confirm(
                {"skill_name": "Demo", "script_path": "demo.py"},
                session_id=app.target.id,
            ),
        }
        pending = asyncio.create_task(asyncio.to_thread(requests[kind]))
        try:
            await until(lambda: app.controller._interrupt_host.pending_total() == 1)
            binding = BuddyBinding.for_session(app.target)
            modal = open_buddy_conversation(app, binding, allow_voice=False)
            await pilot.pause()
            modal.refresh_projection()
            payload = app.controller._interrupt_host.head_round_payload(
                kind, app.target.id
            )
            request_id = payload["request_id"]
            assert not modal.coordinator.resolve_decision(
                BuddyBinding.for_session(app.other), kind, request_id, False
            )
            if kind == "question":
                card = modal.query_one(ChatQuestionCard)
                event = ChatTaskCards.QuestionAnswered([], request_id)
            elif kind == "skill_install":
                card = modal.query_one(SkillInstallConfirmCard)
                event = SkillInstallConfirmCard.InstallDecided(False, request_id)
            else:
                card = modal.query_one(SkillScriptConfirmCard)
                event = SkillScriptConfirmCard.ScriptDecided(False, False, request_id)
            assert card.display
            card.post_message(event)
            await asyncio.wait_for(pending, 3)
            assert app.store.active_session_id == app.other.id
        finally:
            if not pending.done():
                app.controller.begin_shutdown()
                await pending


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("question", "skill_install", "skill_script"))
async def test_explicit_buddy_retains_headless_decisions_after_close(kind):
    from Tests.Chat.test_console_ask_user_round import _questions
    from tldw_chatbook.UI.Navigation.buddy_conversation import open_buddy_conversation

    app = Harness()
    async with app.run_test(size=(100, 36)) as pilot:
        assert getattr(app.controller, "set_pending_" + kind) is None
        binding = BuddyBinding.for_session(app.target)
        modal = open_buddy_conversation(app, binding, allow_voice=False)
        await pilot.pause()
        modal.query_one("#buddy-close", Button).press()
        await until(lambda: app.screen is not modal)

        def request(session_id):
            if kind == "question":
                return app.controller.request_user_questions(
                    _questions(), session_id=session_id
                )
            if kind == "skill_install":
                return app.controller.request_skill_install_confirm(
                    "https://example.com/skill", session_id=session_id
                )
            return app.controller.request_skill_script_confirm(
                {"skill_name": "Demo", "script_path": "demo.py"}, session_id=session_id
            )

        # A wake-only sibling with no explicit interaction still fails closed.
        result = request(app.other.id)
        assert result in (
            False,
            {"answered": False, "reason": "cancelled"},
            {"allow": False, "remember": False},
        )
        pending = asyncio.create_task(asyncio.to_thread(request, app.target.id))
        try:
            await until(
                lambda: (
                    app.controller._interrupt_host.pending_total() == 1
                    or pending.done()
                )
            )
            assert not pending.done()
            reopened = open_buddy_conversation(app, binding, allow_voice=False)
            await pilot.pause()
            payload = reopened.coordinator.decision_payloads(binding)[kind]
            decision = (
                []
                if kind == "question"
                else (False, False)
                if kind == "skill_script"
                else False
            )
            assert reopened.coordinator.resolve_decision(
                binding, kind, payload["request_id"], decision
            )
            await asyncio.wait_for(pending, 3)
            assert app.store.active_session_id == app.other.id
            app.target.conversation_binding_revision += 1
            assert not app.controller._interrupt_host.has_retained_decision_target(
                app.target.id
            )
        finally:
            if not pending.done():
                app.controller.begin_shutdown()
                await pending


@pytest.mark.asyncio
async def test_draft_follows_same_conversation_canonical_binding():
    from tldw_chatbook.UI.Navigation.buddy_conversation import open_buddy_conversation

    app = Harness()
    app.target.ephemeral = False
    async with app.run_test(size=(90, 28)) as pilot:
        binding = BuddyBinding.for_session(app.target)
        modal = open_buddy_conversation(app, binding, allow_voice=False)
        await pilot.pause()
        modal.query_one("#buddy-reply", TextArea).load_text("Keep this private draft")
        await pilot.pause()
        modal.query_one("#buddy-close", Button).press()
        await until(lambda: app.screen is not modal)
        app.target.persisted_conversation_id = "now-saved"
        canonical = BuddyBinding.for_session(app.target)
        reopened = open_buddy_conversation(app, canonical, allow_voice=False)
        await pilot.pause()
        assert (
            reopened.query_one("#buddy-reply", TextArea).text
            == "Keep this private draft"
        )
        assert app.store.active_session_id == app.other.id


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ("deleted", "profile"))
async def test_saved_load_revalidates_after_io_without_selecting(monkeypatch, change):
    import tldw_chatbook.Chat.console_conversation_hydration as hydration
    from tldw_chatbook.UI.Navigation.buddy_conversation import open_buddy_conversation

    app = Harness()
    record = {"id": "saved-conversation", "runtime_backend": "local", "title": "Saved"}
    app.local_chat_conversation_service = SimpleNamespace(
        get_conversation_metadata=lambda _: record
    )
    started, release = asyncio.Event(), asyncio.Event()

    async def load(_app, target):
        assert target == "saved-conversation"
        started.set()
        await release.wait()
        return {"conversation": record, "roots": []}

    monkeypatch.setattr(hydration, "load_console_conversation_tree", load)
    async with app.run_test(size=(90, 28)) as pilot:
        binding = BuddyBinding(
            "conversation",
            "saved:saved-conversation",
            conversation_id="saved-conversation",
        )
        modal = open_buddy_conversation(app, binding, allow_voice=False)
        await started.wait()
        if change == "deleted":
            record["deleted"] = True
        else:
            app.chachanotes_db = object()
        release.set()
        await until(lambda: not modal.coordinator.restoring)
        await pilot.pause()
        assert modal.coordinator.resolve(binding) is None
        assert app.store.active_session_id == app.other.id
        assert len(app.store.sessions()) == 2
        assert not app.gateway.started.is_set()


@pytest.mark.asyncio
async def test_failed_cold_bootstrap_keeps_explicit_console_recovery(monkeypatch):
    import tldw_chatbook.Chat.console_launch_wake as wake
    from tldw_chatbook.UI.Navigation.buddy_conversation import open_buddy_conversation

    app = Harness()
    app.console_runtime.set_chat_store(None)
    app.console_runtime.set_chat_controller(None)
    app.local_chat_conversation_service = SimpleNamespace(
        get_conversation_metadata=lambda _: {"runtime_backend": "local"}
    )
    monkeypatch.setattr(wake, "_ensure_launch_runtime", lambda _app: None)
    async with app.run_test(size=(90, 28)) as pilot:
        binding = BuddyBinding(
            "conversation", "saved:available", conversation_id="available"
        )
        modal = open_buddy_conversation(app, binding, allow_voice=False)
        await until(lambda: not modal.coordinator.restoring)
        await pilot.pause()
        assert modal.query_one("#buddy-send", Button).disabled
        assert not modal.query_one("#buddy-open-console", Button).disabled
        assert "Open Console" in modal.coordinator.notices[binding]
        assert not app.gateway.started.is_set()


@pytest.mark.asyncio
async def test_concurrent_saved_loader_retains_the_exact_buddy_decision_owner(
    monkeypatch,
):
    import tldw_chatbook.Chat.console_conversation_hydration as hydration
    from Tests.Chat.test_console_ask_user_round import _questions
    from tldw_chatbook.UI.Navigation.buddy_conversation import open_buddy_conversation

    app = Harness()
    conversation_id = "concurrent-saved"
    record = {"id": conversation_id, "runtime_backend": "local", "title": "Saved"}
    app.local_chat_conversation_service = SimpleNamespace(
        get_conversation_metadata=lambda _: record
    )
    started, release = asyncio.Event(), asyncio.Event()

    async def load(_app, target):
        assert target == conversation_id
        started.set()
        await release.wait()
        return {"conversation": record, "roots": []}

    monkeypatch.setattr(hydration, "load_console_conversation_tree", load)
    async with app.run_test(size=(100, 36)) as pilot:
        binding = BuddyBinding(
            "conversation", "saved:" + conversation_id, conversation_id=conversation_id
        )
        modal = open_buddy_conversation(app, binding, allow_voice=False)
        await started.wait()
        restored = app.store.create_session(ephemeral=False, activate=False)
        app.store.rebind_persisted_conversation(restored.id, conversation_id)
        release.set()
        await until(lambda: not modal.coordinator.restoring)
        await pilot.pause()
        assert modal.coordinator.resolve(binding) is restored
        modal.query_one("#buddy-close", Button).press()
        await until(lambda: app.screen is not modal)
        pending = asyncio.create_task(
            asyncio.to_thread(
                app.controller.request_user_questions,
                _questions(),
                session_id=restored.id,
            )
        )
        try:
            await until(
                lambda: (
                    app.controller._interrupt_host.pending_total() == 1
                    or pending.done()
                )
            )
            assert not pending.done()
            reopened = open_buddy_conversation(app, binding, allow_voice=False)
            await pilot.pause()
            payload = reopened.coordinator.decision_payloads(binding)["question"]
            assert reopened.coordinator.resolve_decision(
                binding, "question", payload["request_id"], []
            )
            await asyncio.wait_for(pending, 3)
            assert app.store.active_session_id == app.other.id
        finally:
            if not pending.done():
                app.controller.begin_shutdown()
                await pending


@pytest.mark.asyncio
async def test_dictation_finishes_into_reviewable_draft_without_sending():
    from tldw_chatbook.UI.Navigation.buddy_conversation import open_buddy_conversation

    app = Harness()

    class Voice:
        discarded = False

        def start(self, **kwargs):
            assert app.buddy_speech_coordinator.queue.state.input_active

        def stop_and_transcribe(self):
            return "Dictated words"

        def discard(self):
            self.discarded = True

    voice = Voice()
    async with app.run_test(size=(90, 28)) as pilot:
        binding = BuddyBinding.for_session(app.target)
        modal = open_buddy_conversation(app, binding)
        modal.coordinator.voice_factory = lambda **kwargs: voice
        modal.coordinator.voice_availability = lambda: SimpleNamespace(ok=True)
        await pilot.pause()
        modal.query_one("#buddy-mic", Button).press()
        await until(lambda: modal.coordinator.voice_status(modal) == "recording")
        await until(lambda: not modal.query_one("#buddy-mic", Button).disabled)
        modal.query_one("#buddy-mic", Button).press()
        await until(lambda: modal.coordinator.drafts.get(binding) == "Dictated words")
        await until(
            lambda: modal.query_one("#buddy-reply", TextArea).text == "Dictated words"
        )
        assert voice.discarded
        assert not app.buddy_speech_coordinator.queue.state.input_active
        assert not app.gateway.started.is_set()


@pytest.mark.asyncio
async def test_closing_buddy_discards_microphone_and_ignores_late_result():
    from tldw_chatbook.UI.Navigation.buddy_conversation import open_buddy_conversation

    app = Harness()

    class Voice:
        discarded = False

        def start(self, **kwargs):
            pass

        def discard(self):
            self.discarded = True

    voice = Voice()
    async with app.run_test(size=(90, 28)) as pilot:
        modal = open_buddy_conversation(app, BuddyBinding.for_session(app.target))
        modal.coordinator.voice_factory = lambda **kwargs: voice
        modal.coordinator.voice_availability = lambda: SimpleNamespace(ok=True)
        await pilot.pause()
        modal.query_one("#buddy-mic", Button).press()
        await until(lambda: modal.coordinator.voice_status(modal) == "recording")
        modal.query_one("#buddy-close", Button).press()
        await until(lambda: app.screen is not modal)
        await until(lambda: voice.discarded)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "refusal", ("attachment", "repurposed", "deleted", "busy", "command")
)
async def test_buddy_refuses_unseen_input_and_unavailable_owners(refusal):
    from tldw_chatbook.Chat.console_chat_models import ConsoleRunState, ConsoleRunStatus
    from tldw_chatbook.UI.Navigation.buddy_conversation import open_buddy_conversation

    app = Harness()
    async with app.run_test(size=(80, 24)) as pilot:
        if refusal == "deleted":
            app.target.ephemeral = False
            app.target.persisted_conversation_id = "deleted-conversation"
            app.local_chat_conversation_service = SimpleNamespace(
                get_conversation_metadata=lambda _: None
            )
        binding = BuddyBinding.for_session(app.target)
        if refusal == "attachment":
            app.store.add_pending_attachment(app.target.id, pending_image())
        elif refusal == "repurposed":
            app.target.conversation_binding_revision += 1
        elif refusal == "busy":
            app.controller._set_run_state(
                ConsoleRunState(ConsoleRunStatus.STREAMING), session_id=app.target.id
            )
        modal = open_buddy_conversation(app, binding, allow_voice=False)
        await pilot.pause()
        reply = "/new" if refusal == "command" else "Do not redirect me"
        modal.coordinator.request_send(binding, reply)
        await until(lambda: binding not in modal.coordinator.submitting)
        assert modal.coordinator.notices[binding]
        assert modal.coordinator.drafts[binding] == reply
        assert not app.gateway.started.is_set()
        assert not app.store.messages_for_session(app.other.id)
        if refusal == "attachment":
            assert (
                app.store.pending_attachments(app.target.id)[0].data == b"private-image"
            )
        assert not modal.query("#buddy-mic")
        assert modal.query_one("#buddy-close").region.bottom <= app.size.height


@pytest.mark.asyncio
@pytest.mark.parametrize("late_phase", ("starting", "transcribing"))
async def test_late_voice_completion_after_close_cannot_change_the_draft(late_phase):
    from tldw_chatbook.UI.Navigation.buddy_conversation import open_buddy_conversation

    gate = threading.Event()
    started = threading.Event()

    class Voice:
        discards = 0

        def start(self, **kwargs):
            if late_phase == "starting":
                started.set()
                gate.wait(5)

        def stop_and_transcribe(self):
            started.set()
            gate.wait(5)
            return "late private speech"

        def discard(self):
            self.discards += 1

    voice = Voice()
    app = Harness()
    async with app.run_test(size=(90, 28)) as pilot:
        binding = BuddyBinding.for_session(app.target)
        modal = open_buddy_conversation(app, binding)
        modal.coordinator.voice_factory = lambda **kwargs: voice
        modal.coordinator.voice_availability = lambda: SimpleNamespace(ok=True)
        await pilot.pause()
        modal.query_one("#buddy-reply", TextArea).load_text("Keep typed words")
        await pilot.pause()
        modal.coordinator.request_voice(modal, binding, allowed=True)
        if late_phase == "transcribing":
            await until(lambda: modal.coordinator.voice_status(modal) == "recording")
            modal.coordinator.request_voice(modal, binding, allowed=True)
        await until(started.is_set)
        modal.query_one("#buddy-close", Button).press()
        await until(lambda: app.screen is not modal)
        assert voice.discards
        gate.set()
        await pilot.pause(0.1)
        assert modal.coordinator.drafts[binding] == "Keep typed words"
        assert not app.gateway.started.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("saved", (False, True))
async def test_real_console_buddy_send_preserves_both_durable_console_drafts(
    monkeypatch, tmp_path, saved
):
    from Tests.UI.test_console_native_chat_flow import (
        _configure_native_ready_console,
        _ReadyResolutionGateway,
        _select_llamacpp_console,
    )
    from Tests.UI.test_console_screen_reuse import (
        _boot_settled,
        _press_until_screen,
        _scratch_env,
    )
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.Navigation.buddy_conversation import open_buddy_conversation
    from tldw_chatbook.Widgets.Console import ConsoleComposerBar

    _scratch_env(monkeypatch, tmp_path)

    class Gateway(_ReadyResolutionGateway):
        started = threading.Event()
        release = threading.Event()

        async def stream_chat(self, resolution, messages, **kwargs):
            self.started.set()
            yield "reply"
            while not self.release.is_set():
                await asyncio.sleep(0.01)

        async def aclose(self):
            pass

    gateway = Gateway()
    app = TldwCli()
    _configure_native_ready_console(app)
    app.console_provider_gateway_factory = lambda: gateway
    async with app.run_test(size=(170, 48)) as pilot:
        await _boot_settled(app, pilot)
        await _press_until_screen(pilot, "ctrl+2", "ChatScreen")
        console = app.screen
        _select_llamacpp_console(console)
        controller = console._ensure_console_chat_controller()
        target = controller.store.ensure_session()
        binding = BuddyBinding.for_session(target)
        reply = "Same durable Console draft"
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft(reply)
        target.draft = reply
        other = controller.new_session()
        await console._sync_native_console_chat_ui()
        composer.load_draft("Unrelated visible composer")
        other.draft = "Unrelated visible composer"
        await _press_until_screen(pilot, "ctrl+1", "HomeScreen")
        prior_workspace = controller.store.workspace_context.active_workspace_id
        if saved:
            conversation_id = app.local_chat_conversation_service.create_conversation(
                title="Unloaded saved conversation",
                runtime_backend="local",
                scope_type="global",
            )
            # A saved Console turn carries its durable Library policy; a raw
            # legacy Chat row is deliberately refused by the commit guard.
            controller.store.persistence.console_library_policy_repository.insert(
                conversation_id,
                controller.store.session_library_policy_candidate(target.id),
            )
            app.chachanotes_db.add_message(
                {
                    "conversation_id": conversation_id,
                    "sender": "user",
                    "role": "user",
                    "content": "Earlier saved message",
                }
            )
            binding = BuddyBinding(
                "conversation",
                "saved:" + conversation_id,
                conversation_id=conversation_id,
            )
            assert binding.resolve_session(controller.store.sessions()) is None
        modal = open_buddy_conversation(app, binding, allow_voice=False)
        await pilot.pause()
        try:
            if saved:
                await until(lambda: modal.coordinator.resolve(binding) is not None)
                target = modal.coordinator.resolve(binding)
                target.draft = reply
                assert target.persisted_conversation_id == conversation_id
                assert (
                    controller.store.messages_for_session(target.id)[0].content
                    == "Earlier saved message"
                )
                assert (
                    controller.store.workspace_context.active_workspace_id
                    == prior_workspace
                )
            modal.query_one("#buddy-reply", TextArea).load_text(reply)
            modal.query_one("#buddy-send", Button).press()
            await until(gateway.started.is_set)
            assert target.persisted_conversation_id
            assert modal.coordinator.resolve(binding) is target
            assert target.draft == reply
            assert other.draft == "Unrelated visible composer"
            assert composer.draft_text() == "Unrelated visible composer"
            assert controller.store.active_session_id == other.id
            modal.query_one("#buddy-close", Button).press()
            await until(lambda: type(app.screen).__name__ == "HomeScreen")
            assert controller.in_flight_run_count() == 1
            gateway.release.set()
            await until(lambda: controller.in_flight_run_count() == 0)
            assert target.draft == reply
            assert not controller.store.messages_for_session(other.id)
            reopened = open_buddy_conversation(
                app, BuddyBinding.for_session(target), allow_voice=False
            )
            await pilot.pause()
            reopened.query_one("#buddy-open-console", Button).press()
            await until(lambda: app.screen.__class__.__name__ == "ChatScreen")
            await until(
                lambda: (
                    controller.store.active_session_id == target.id
                    and composer.draft_text() == reply
                )
            )
        finally:
            gateway.release.set()
            await controller.shutdown()


@pytest.mark.asyncio
async def test_cold_home_saved_buddy_bootstraps_only_after_explicit_open(
    monkeypatch, tmp_path
):
    from Tests.UI.test_console_native_chat_flow import (
        _configure_native_ready_console,
        _ReadyResolutionGateway,
    )
    from Tests.UI.test_console_screen_reuse import _boot_settled, _scratch_env
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_library_policy import (
        ConsoleAssistantLibraryAccess,
        ConsoleAutoRetrieve,
        ConsoleLibraryPolicyCandidate,
    )
    from tldw_chatbook.UI.Navigation.buddy_conversation import open_buddy_conversation

    _scratch_env(monkeypatch, tmp_path)
    config_path = tmp_path / "config" / "tldw_cli" / "config.toml"
    config_path.write_text(
        config_path.read_text() + '\n[general]\ndefault_tab = "home"\n'
    )

    class Gateway(_ReadyResolutionGateway):
        started = threading.Event()

        async def stream_chat(self, resolution, messages, **kwargs):
            self.started.set()
            yield "Cold reply"

        async def aclose(self):
            pass

    gateway = Gateway()
    app = TldwCli()
    _configure_native_ready_console(app)
    app.console_provider_gateway_factory = lambda: gateway
    async with app.run_test(size=(150, 45)) as pilot:
        await _boot_settled(app, pilot)
        assert app.screen.__class__.__name__ == "HomeScreen"
        runtime = app.console_runtime
        assert runtime.chat_store is None and runtime.chat_controller is None
        conversation_id = app.local_chat_conversation_service.create_conversation(
            title="Cold saved Console", runtime_backend="local", scope_type="global"
        )
        ChatPersistenceService(
            app.chachanotes_db
        ).console_library_policy_repository.insert(
            conversation_id,
            ConsoleLibraryPolicyCandidate(
                ConsoleAutoRetrieve.NEVER, ConsoleAssistantLibraryAccess.BLOCKED
            ),
        )
        app.chachanotes_db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "assistant",
                "role": "assistant",
                "content": "Saved result to review",
            }
        )
        binding = BuddyBinding(
            "conversation", "saved:" + conversation_id, conversation_id=conversation_id
        )
        modal = open_buddy_conversation(app, binding, allow_voice=False)
        try:
            await until(lambda: modal.coordinator.resolve(binding) is not None)
            await until(lambda: not modal.query_one("#buddy-send", Button).disabled)
            store = runtime.chat_store
            target = modal.coordinator.resolve(binding)
            assert store.active_session_id is None
            assert (
                store.messages_for_session(target.id)[0].content
                == "Saved result to review"
            )
            modal.query_one("#buddy-reply", TextArea).load_text("Follow up from Home")
            modal.query_one("#buddy-send", Button).press()
            await until(gateway.started.is_set)
            modal.query_one("#buddy-close", Button).press()
            await until(lambda: app.screen.__class__.__name__ == "HomeScreen")
            assert store.active_session_id is None
            assert runtime.view is None
            await until(lambda: runtime.chat_controller.in_flight_run_count() == 0)
            from Tests.Chat.test_console_ask_user_round import _questions

            for kind in ("question", "approval"):
                controller = runtime.chat_controller
                assert controller.set_pending_question is None
                assert controller._ask_user_wiring(target.id)
                if kind == "question":
                    pending = asyncio.create_task(
                        asyncio.to_thread(
                            controller.request_user_questions,
                            _questions(),
                            session_id=target.id,
                        )
                    )
                else:
                    pending = asyncio.create_task(
                        asyncio.to_thread(
                            controller.request_mcp_approvals,
                            [_pending_call()],
                            session_id=target.id,
                        )
                    )
                try:
                    await until(
                        lambda controller=controller, pending=pending: (
                            controller._interrupt_host.pending_total() == 1
                            or pending.done()
                        )
                    )
                    assert not pending.done()
                    reopened = open_buddy_conversation(app, binding, allow_voice=False)
                    await pilot.pause()
                    payload = reopened.coordinator.decision_payloads(binding)[kind]
                    assert reopened.coordinator.resolve_decision(
                        binding,
                        kind,
                        payload.get("request_id") or payload.get("round_id"),
                        [] if kind == "question" else {"c1": "deny"},
                    )
                    await asyncio.wait_for(pending, 3)
                    reopened.query_one("#buddy-close", Button).press()
                    await until(lambda: app.screen.__class__.__name__ == "HomeScreen")
                    assert store.active_session_id is None
                finally:
                    if not pending.done():
                        controller.begin_shutdown()
                        await pending
        finally:
            if runtime.chat_controller is not None:
                await runtime.chat_controller.shutdown()


@pytest.mark.asyncio
async def test_recorder_buffer_callback_runs_on_textual_app_thread():
    from tldw_chatbook.UI.Navigation.buddy_conversation import (
        BuddyConversationCoordinator,
        _VoiceCapture,
    )

    app = Harness()
    callback = []
    threads = []
    finished = asyncio.Event()

    class Recorder:
        def start(self, *, on_buffer_limit):
            callback.append(on_buffer_limit)

        def discard(self):
            pass

    class Speech:
        async def wait_silent(self):
            pass

    async with app.run_test():
        coordinator = BuddyConversationCoordinator(app)
        binding = BuddyBinding.for_session(app.target)
        owner = object()
        capture = _VoiceCapture(binding, Recorder(), "", Speech())
        coordinator._voices[owner] = capture

        def voice(*args, **kwargs):
            threads.append(threading.get_ident())
            finished.set()

        coordinator.request_voice = voice
        await coordinator._start_voice(owner, capture)
        await asyncio.to_thread(callback[0])
        await asyncio.wait_for(finished.wait(), 2)
        assert threads == [threading.get_ident()]
        assert capture.status == "recording"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phase", ["tree", "attachments", "cursor", "continuations", "generation"]
)
@pytest.mark.parametrize("change", [False, True])
async def test_saved_buddy_bulk_reads_allow_loop_progress_and_recheck_owner(
    tmp_path, monkeypatch, phase, change
):
    from tldw_chatbook.Chat.chat_conversation_scope_service import (
        ChatConversationScopeService,
    )
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.UI.Navigation.buddy_conversation import open_buddy_conversation

    db = CharactersRAGDB(tmp_path / "restore.db", "buddy-restore")
    local = ChatConversationService(db)
    target = local.create_conversation(
        title="Saved", runtime_backend="local", scope_type="global"
    )
    db.add_message(
        {
            "conversation_id": target,
            "sender": "user",
            "role": "user",
            "content": "Saved input",
        }
    )
    app = Harness()
    app.chachanotes_db = db
    app.local_chat_conversation_service = local
    app.chat_conversation_scope_service = ChatConversationScopeService(
        local_service=local, server_service=None
    )
    persistence = ChatPersistenceService(db)
    app.store.persistence = persistence
    service, name = {
        "tree": (local, "get_conversation_tree"),
        "attachments": (db, "get_attachments_for_messages"),
        "cursor": (db, "get_conversation_active_cursor"),
        "continuations": (db, "get_messages_for_conversation"),
        "generation": (persistence, "get_generation_metadata_for_messages"),
    }[phase]
    original = getattr(service, name)
    entered, release = threading.Event(), threading.Event()
    reads = []
    main_thread = threading.get_ident()

    def delayed(*args, **kwargs):
        reads.append(threading.get_ident())
        entered.set()
        assert release.wait(3), "bulk read blocked the event loop"
        return original(*args, **kwargs)

    monkeypatch.setattr(service, name, delayed)
    closed = []
    close = db.close_connection

    def close_worker():
        close()
        closed.append((threading.get_ident(), getattr(db._local, "conn", None)))

    monkeypatch.setattr(db, "close_connection", close_worker)
    published = []
    restore = app.store.restore_persisted_session

    def restore_on_app(**kwargs):
        published.append(threading.get_ident())
        return restore(**kwargs)

    monkeypatch.setattr(app.store, "restore_persisted_session", restore_on_app)
    try:
        async with app.run_test():
            binding = BuddyBinding(
                "conversation", "saved:" + target, conversation_id=target
            )
            modal = open_buddy_conversation(app, binding, allow_voice=False)
            await until(entered.is_set)
            assert reads == [reads[0]] and reads[0] != main_thread
            assert not published
            if change:
                app.chachanotes_db = object()
            release.set()
            await until(lambda: not modal.coordinator.restoring)
            if change:
                assert modal.coordinator.resolve(binding) is None
                assert not published
            else:
                restored = modal.coordinator.resolve(binding)
                assert restored is not None
                assert published == [main_thread]
                assert (
                    app.store.messages_for_session(restored.id)[0].content
                    == "Saved input"
                )
            assert app.store.active_session_id == app.other.id
            assert any(
                thread != main_thread and connection is None
                for thread, connection in closed
            )
            assert db.get_conversation_by_id(target)["id"] == target
            assert len(reads) == 1 and reads[0] != main_thread
    finally:
        release.set()
        db.close_connection()
