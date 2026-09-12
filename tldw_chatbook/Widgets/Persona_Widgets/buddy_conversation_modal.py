"""A disposable transcript and decision projection for one explicit Buddy target."""

from __future__ import annotations

from typing import Any, ClassVar

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static, TextArea

from tldw_chatbook.Persona_Buddy.interaction import BuddyBinding
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard
from tldw_chatbook.Widgets.Chat_Widgets.chat_question_card import ChatQuestionCard
from tldw_chatbook.Widgets.Chat_Widgets.chat_task_cards import ChatTaskCards
from tldw_chatbook.Widgets.Chat_Widgets.skill_install_confirm_card import (
    SkillInstallConfirmCard,
)
from tldw_chatbook.Widgets.Chat_Widgets.skill_script_confirm_card import (
    SkillScriptConfirmCard,
)
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin


class BuddyConversationModal(SafeModalDismissMixin, ModalScreen[None]):
    """Close only the projection and microphone, leaving accepted runtime work intact."""

    SAFE_MODAL_CONTENT = "#buddy-conversation"
    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("escape", "request_safe_cancel", "Close")
    ]
    DEFAULT_CSS = """
    BuddyConversationModal { align: center middle; }
    #buddy-conversation { width: 90; max-width: 96%; height: 90%; max-height: 52;
        border: round $accent; background: $panel; padding: 0 1; }
    #buddy-conversation-title { height: 2; text-style: bold; }
    #buddy-activity, #buddy-reply-notice { height: auto; }
    #buddy-conversation-body { height: 1fr; min-height: 2; }
    #buddy-transcript-navigation { height: 1; }
    #buddy-transcript-navigation Button { height: 1; min-width: 8; width: auto; }
    #buddy-transcript { height: auto; padding: 1 0; }
    #buddy-reply { height: 5; min-height: 3; }
    BuddyConversationModal.compact #buddy-reply { height: 3; }
    BuddyConversationModal.compact #buddy-conversation-title { height: 1; }
    #buddy-actions { height: 3; align-horizontal: right; }
    #buddy-actions Button { min-width: 8; width: auto; margin-left: 1; }
    #buddy-reply-notice { color: $text-muted; }
    """

    def __init__(
        self, coordinator: Any, binding: BuddyBinding, *, allow_voice: bool = True
    ) -> None:
        super().__init__()
        self.coordinator = coordinator
        self.binding = binding
        self.allow_voice = allow_voice
        self._syncing_draft = False
        self._visible = True
        self._timer: Any = None
        self._last_transcript: str | None = None
        self._last_decisions = ""
        self._new_updates = False
        self._last_draft = coordinator.drafts.get(binding, "")

    def compose(self) -> ComposeResult:
        with Vertical(id="buddy-conversation"):
            yield Static("Conversation", id="buddy-conversation-title", markup=False)
            yield Static("", id="buddy-activity", markup=False)
            with VerticalScroll(id="buddy-conversation-body"):
                yield Static("", id="buddy-transcript", markup=False)
                yield ChatApprovalCard(id="buddy-approval")
                yield SkillInstallConfirmCard(id="buddy-install")
                yield SkillScriptConfirmCard(id="buddy-script")
                yield ChatQuestionCard(id="buddy-question")
            with Horizontal(id="buddy-transcript-navigation"):
                yield Button("Latest", id="buddy-latest", compact=True)
                yield Button("Pending decision", id="buddy-pending", compact=True)
            yield Static("", id="buddy-reply-notice", markup=False)
            yield TextArea(
                self.coordinator.drafts.get(self.binding, ""), id="buddy-reply"
            )
            from tldw_chatbook.UI.Navigation.buddy_speech import ensure_buddy_speech
            from tldw_chatbook.Widgets.Persona_Widgets.buddy_speech_controls import (
                BuddySpeechControls,
            )

            yield BuddySpeechControls(ensure_buddy_speech(self.coordinator.app))
            with Horizontal(id="buddy-actions"):
                if self.allow_voice:
                    yield Button("Dictate", id="buddy-mic")
                yield Button("Send", id="buddy-send", variant="primary")
                yield Button("Open Console", id="buddy-open-console")
                yield Button("Close", id="buddy-close")

    def on_resize(self) -> None:
        self.set_class(self.size.height <= 24, "compact")

    def on_mount(self) -> None:
        super().on_mount()
        self.query_one("#buddy-conversation-body", VerticalScroll).anchor()
        self._timer = self.set_interval(0.2, self.refresh_projection)
        self.call_after_refresh(self.refresh_projection)
        self.query_one("#buddy-reply", TextArea).focus()

    def refresh_projection(self) -> None:
        if not self.is_mounted or not self._visible:
            return
        body = self.query_one("#buddy-conversation-body", VerticalScroll)
        first = self._last_transcript is None
        follow = first or body.scroll_y >= body.max_scroll_y - 1
        changed = False
        coordinator = self.coordinator
        session = coordinator.resolve(self.binding)
        controller = coordinator.controller
        available = session is not None and controller is not None
        coordinator.show_decisions(self, self.binding if available else None)
        if not available:
            coordinator.close_voice(self)
        title = session.title if session is not None else "Conversation unavailable"
        self.query_one("#buddy-conversation-title", Static).update(str(title))
        busy = not available or not controller.run_state_for(session.id).is_send_allowed
        if busy:
            coordinator.close_voice(self)
        activity = (
            controller.run_state_for(session.id).status.value.replace("_", " ")
            if available
            else "This target is missing or unavailable. No other conversation will be used."
        )
        self.query_one("#buddy-activity", Static).update(activity)
        self.query_one("#buddy-send", Button).disabled = (
            busy or self.binding in coordinator.submitting
        )
        self.query_one(
            "#buddy-open-console", Button
        ).disabled = not coordinator.can_open_console(self.binding)
        self.query_one("#buddy-reply", TextArea).disabled = not available
        if available:
            messages = controller.store.messages_for_session(session.id)[-60:]
            transcript = "\n\n".join(
                f"{message.role.value.title()}: {message.content}"
                for message in messages
            )[-64000:]
            if transcript != self._last_transcript:
                changed = True
                self.query_one("#buddy-transcript", Static).update(
                    transcript or "No messages yet."
                )
                self._last_transcript = transcript
        else:
            self.query_one("#buddy-transcript", Static).update("")
        payloads = coordinator.decision_payloads(self.binding) if available else {}
        decisions = repr(payloads)
        changed = changed or decisions != self._last_decisions
        self._last_decisions = decisions
        self.query_one("#buddy-pending", Button).display = bool(payloads)
        if changed:
            if follow:
                self.call_after_refresh(self._scroll_latest)
            else:
                self._new_updates = True
        self.query_one("#buddy-latest", Button).label = (
            "Latest · new updates" if self._new_updates else "Latest"
        )
        approval = payloads.get("approval")
        card = self.query_one(ChatApprovalCard)
        if approval:
            card.set_batch(
                approval.get("calls", []),
                timeout_seconds=approval.get("timeout_seconds", 0),
                round_id=approval.get("round_id"),
                phase=approval.get("phase", "approval"),
                summary=approval.get("summary"),
            )
        else:
            card.display = False
        self.query_one(SkillInstallConfirmCard).set_install(
            payloads.get("skill_install")
        )
        self.query_one(SkillScriptConfirmCard).set_script(payloads.get("skill_script"))
        self.query_one(ChatQuestionCard).set_questions(payloads.get("question"))
        rendered_id = None
        for kind, card_type in (
            ("approval", ChatApprovalCard),
            ("skill_install", SkillInstallConfirmCard),
            ("skill_script", SkillScriptConfirmCard),
        ):
            payload = payloads.get(kind)
            if payload and self.query_one(card_type).display:
                rendered_id = payload.get("_decision_id")
        coordinator.show_decisions(
            self, self.binding if available else None, decision_id=rendered_id
        )
        notice = coordinator.notices.get(self.binding, "")
        if "worktree_merge" in payloads:
            notice = "A worktree decision needs review. Open Console to continue."
        self.query_one("#buddy-reply-notice", Static).update(notice)
        draft = coordinator.drafts.get(self.binding, "")
        composer = self.query_one("#buddy-reply", TextArea)
        if self._last_draft != draft:
            self._syncing_draft = True
            composer.load_text(draft)
            self._last_draft = draft
            self._syncing_draft = False
        if self.allow_voice:
            mic = self.query_one("#buddy-mic", Button)
            status = coordinator.voice_status(self)
            mic.disabled = (
                not available or busy or status in {"starting", "transcribing"}
            )
            mic.label = (
                "Finish dictation"
                if status == "recording"
                else status.title()
                if status != "idle"
                else "Dictate"
            )

    def _scroll_latest(self) -> None:
        self.query_one("#buddy-conversation-body", VerticalScroll).anchor()
        self._new_updates = False
        self.query_one("#buddy-latest", Button).label = "Latest"

    @on(Button.Pressed, "#buddy-latest")
    def latest(self, event: Button.Pressed) -> None:
        event.stop()
        self._scroll_latest()

    @on(Button.Pressed, "#buddy-pending")
    def pending(self, event: Button.Pressed) -> None:
        event.stop()
        for card_type in (
            ChatApprovalCard,
            SkillInstallConfirmCard,
            SkillScriptConfirmCard,
            ChatQuestionCard,
        ):
            card = self.query_one(card_type)
            if card.display:
                body = self.query_one("#buddy-conversation-body", VerticalScroll)
                body.release_anchor()
                controls = [widget for widget in card.query(Button) if widget.focusable]
                if controls:
                    controls[0].focus(scroll_visible=False)
                    # Nested scroll_to_widget clips through the tall card's
                    # ancestors at small sizes. Target the action's measured
                    # position directly in the transcript viewport instead.
                    body.scroll_to(
                        y=body.scroll_y
                        + controls[0].region.bottom
                        - body.content_region.bottom,
                        animate=False,
                        immediate=True,
                    )
                else:
                    body.scroll_to_widget(card, animate=False, top=True, immediate=True)
                return
        self.query_one("#buddy-open-console", Button).focus()

    @on(TextArea.Changed, "#buddy-reply")
    def draft_changed(self, event: TextArea.Changed) -> None:
        if not self._syncing_draft:
            self.coordinator.drafts[self.binding] = event.text_area.text
            self._last_draft = event.text_area.text

    @on(Button.Pressed, "#buddy-send")
    def send_reply(self, event: Button.Pressed) -> None:
        event.stop()
        self.coordinator.request_send(
            self.binding, self.query_one("#buddy-reply", TextArea).text
        )
        self.refresh_projection()

    @on(Button.Pressed, "#buddy-mic")
    def dictate(self, event: Button.Pressed) -> None:
        event.stop()
        self.coordinator.request_voice(self, self.binding, allowed=self.allow_voice)
        self.refresh_projection()

    @on(Button.Pressed, "#buddy-close")
    def close(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss_safe_once(None)

    @on(Button.Pressed, "#buddy-open-console")
    def open_console(self, event: Button.Pressed) -> None:
        event.stop()
        if self.coordinator.can_open_console(self.binding):
            self.dismiss_safe_once(None)
            self.coordinator.open_console(self.binding)

    @on(ChatApprovalCard.ApprovalDecided)
    def approval_decided(self, event: ChatApprovalCard.ApprovalDecided) -> None:
        event.stop()
        self.coordinator.resolve_decision(
            self.binding, "approval", event.round_id, event.decisions
        )

    @on(SkillInstallConfirmCard.InstallDecided)
    def install_decided(self, event: SkillInstallConfirmCard.InstallDecided) -> None:
        event.stop()
        self.coordinator.resolve_decision(
            self.binding, "skill_install", event.request_id, event.allow
        )

    @on(SkillScriptConfirmCard.ScriptDecided)
    def script_decided(self, event: SkillScriptConfirmCard.ScriptDecided) -> None:
        event.stop()
        self.coordinator.resolve_decision(
            self.binding,
            "skill_script",
            event.request_id,
            (event.allow, event.remember),
        )

    @on(ChatTaskCards.QuestionAnswered)
    def question_answered(self, event: ChatTaskCards.QuestionAnswered) -> None:
        event.stop()
        self.coordinator.resolve_decision(
            self.binding, "question", event.request_id, event.answers
        )

    def on_screen_suspend(self) -> None:
        self._visible = False
        self.coordinator.show_decisions(self, None)
        self.coordinator.close_voice(self)

    def on_screen_resume(self) -> None:
        self._visible = True
        self.call_after_refresh(self.refresh_projection)

    def on_unmount(self) -> None:
        self.coordinator.show_decisions(self, None)
        self.coordinator.close_voice(self)
        if self._timer is not None:
            self._timer.stop()
        super().on_unmount()

    def dismiss_safe_once(self, result: object) -> bool:
        if self.is_mounted:
            text = self.query_one("#buddy-reply", TextArea).text
            if text != self._last_draft:
                self.coordinator.drafts[self.binding] = text
        return super().dismiss_safe_once(result)
