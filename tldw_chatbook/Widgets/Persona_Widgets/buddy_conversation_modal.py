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
    #: Poll cadence while the bound session is executing a generation (task-10).
    _POLL_ACTIVE_SECONDS: ClassVar[float] = 0.2
    #: Poll cadence while the bound session is idle; ticks then only compare
    #: the store fingerprint and cheap coordinator state before returning.
    _POLL_IDLE_SECONDS: ClassVar[float] = 1.0
    #: Unchanged quiet ticks allowed at the active cadence before decaying to
    #: the idle cadence: one second of fast polling trails every observed
    #: change so freshly armed decisions and just-started runs are picked up
    #: at the historical 0.2 s latency.
    _POLL_FAST_DECAY_TICKS: ClassVar[int] = 5
    BUNDLED_CSS = """
    BuddyConversationModal { align: center middle; }
    #buddy-conversation { width: 90; max-width: 96%; height: 90%; max-height: 52;
        border: round $accent; background: $panel; padding: 0 1; }
    #buddy-conversation-title { height: 2; text-style: bold; }
    #buddy-activity, #buddy-reply-notice { height: auto; }
    #buddy-conversation-body { height: 1fr; min-height: 2; }
    #buddy-transcript-navigation { height: 1; }
    #buddy-transcript-navigation Button.buddy-conversation-modal-button { height: 1; min-width: 8; width: auto; }
    #buddy-transcript { height: auto; padding: 1 0; }
    #buddy-reply { height: 5; min-height: 3; }
    BuddyConversationModal.compact #buddy-reply { height: 3; }
    BuddyConversationModal.compact #buddy-conversation-title { height: 1; }
    #buddy-actions { height: 3; align-horizontal: right; }
    #buddy-actions Button.buddy-conversation-modal-button { min-width: 8; width: auto; margin-left: 1; }
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
        self._poll_interval: float = self._POLL_ACTIVE_SECONDS
        self._quiet_ticks = 0
        self._last_gate_key: tuple[Any, ...] | None = None
        self._last_transcript: str | None = None
        self._last_decision_key: tuple[Any, ...] = ()
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
                yield Button(
                    "Latest",
                    id="buddy-latest",
                    compact=True,
                    classes="buddy-conversation-modal-button",
                )
                yield Button(
                    "Pending decision",
                    id="buddy-pending",
                    compact=True,
                    classes="buddy-conversation-modal-button",
                )
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
                    yield Button(
                        "Dictate",
                        id="buddy-mic",
                        classes="buddy-conversation-modal-button",
                    )
                yield Button(
                    "Send",
                    id="buddy-send",
                    variant="primary",
                    classes="buddy-conversation-modal-button",
                )
                yield Button(
                    "Open Console",
                    id="buddy-open-console",
                    classes="buddy-conversation-modal-button",
                )
                yield Button(
                    "Close", id="buddy-close", classes="buddy-conversation-modal-button"
                )

    def on_resize(self) -> None:
        self.set_class(self.size.height <= 24, "compact")

    def on_mount(self) -> None:
        super().on_mount()
        self.query_one("#buddy-conversation-body", VerticalScroll).anchor()
        self._set_poll_interval(self._POLL_ACTIVE_SECONDS)
        self.call_after_refresh(self.refresh_projection)
        self.query_one("#buddy-reply", TextArea).focus()

    def _set_poll_interval(self, seconds: float) -> None:
        """Re-arm the poll timer only when the cadence class changes.

        Stopping and recreating is safe even from inside a tick callback:
        Textual delivers the cancellation to the timer task after the
        callback returns. Re-arming on every call would reset the phase, so
        unchanged cadences are a no-op.
        """
        if self._timer is not None and seconds == self._poll_interval:
            return
        self._poll_interval = seconds
        if self._timer is not None:
            self._timer.stop()
        self._timer = self.set_interval(seconds, self._on_poll_tick)

    def _on_poll_tick(self) -> None:
        """Timer entry point: a change-gated projection poll (task-10)."""
        self.refresh_projection(force=False)

    def refresh_projection(self, *, force: bool = True) -> None:
        """Re-render the projection.

        Args:
            force: Render even when the cheap change gate sees no movement.
                Timer ticks pass ``False``; direct calls (mount, resume,
                send, dictate, tests) keep the historical always-render
                default.
        """
        if not self.is_mounted or not self._visible:
            return
        coordinator = self.coordinator
        session = coordinator.resolve(self.binding)
        controller = coordinator.controller
        available = session is not None and controller is not None
        # Cheap reads only (task-10): the store fingerprint first, then the
        # equally small decision identity tuple, one O(1) run-state lookup
        # and coordinator ephemerals. Together they gate the whole render --
        # an unchanged idle tick costs no snapshot, no transcript build, no
        # card sync, no decision-view registration and no DOM writes.
        fingerprint = (
            controller.store.session_fingerprint(session.id) if available else None
        )
        payloads = coordinator.decision_payloads(self.binding) if available else {}
        # Identity tuple replaces the old O(payload) ``repr`` compare: kinds,
        # decision ids, phases and the approval batch size move the key; the
        # per-tick countdown inside ``timeout_seconds`` snapshots does not
        # (the cards own their visible countdowns).
        decision_key = tuple(
            (
                kind,
                payload.get("_decision_id"),
                payload.get("round_id") or payload.get("request_id"),
                payload.get("phase"),
                len(payload.get("calls") or ()),
            )
            for kind, payload in sorted(payloads.items())
        )
        run_state = controller.run_state_for(session.id) if available else None
        busy = not available or not run_state.is_send_allowed
        gate_key = (
            available,
            fingerprint,
            run_state.status.value if available else None,
            decision_key,
            self.binding in coordinator.submitting,
            coordinator.notices.get(self.binding, ""),
            coordinator.drafts.get(self.binding, ""),
            coordinator.voice_status(self),
        )
        changed_state = force or gate_key != self._last_gate_key
        # Cadence (task-10): fast while executing, while a decision awaits
        # the user, or briefly after any observed change; the quiet-idle
        # decay keeps a settled modal at one cheap tick per second. This
        # runs on gated ticks too, which is what keeps the timer from
        # sticking fast or slow.
        self._quiet_ticks = 0 if changed_state else self._quiet_ticks + 1
        fast = (
            busy or bool(payloads) or self._quiet_ticks <= self._POLL_FAST_DECAY_TICKS
        )
        self._set_poll_interval(
            self._POLL_ACTIVE_SECONDS if fast else self._POLL_IDLE_SECONDS
        )
        if not changed_state:
            return
        self._last_gate_key = gate_key
        try:
            body = self.query_one("#buddy-conversation-body", VerticalScroll)
            first = self._last_transcript is None
            follow = first or body.scroll_y >= body.max_scroll_y - 1
            changed = False
            if not available:
                coordinator.close_voice(self)
            title = session.title if session is not None else "Conversation unavailable"
            title_widget = self.query_one("#buddy-conversation-title", Static)
            if force or str(title_widget.renderable) != str(title):
                title_widget.update(str(title))
            if busy:
                coordinator.close_voice(self)
            activity = (
                run_state.status.value.replace("_", " ")
                if available
                else "This target is missing or unavailable. No other conversation will be used."
            )
            activity_widget = self.query_one("#buddy-activity", Static)
            if force or str(activity_widget.renderable) != activity:
                activity_widget.update(activity)
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
                transcript_widget = self.query_one("#buddy-transcript", Static)
                if force or str(transcript_widget.renderable):
                    transcript_widget.update("")
                # The cleared view must republish even if the same messages return.
                self._last_transcript = None
            changed = changed or decision_key != self._last_decision_key
            self._last_decision_key = decision_key
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
            self.query_one(SkillScriptConfirmCard).set_script(
                payloads.get("skill_script")
            )
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
        except Exception:
            # A render that failed mid-body must not leave a stale rendered
            # claim spending the decision's visible allowance (pinned by
            # test_buddy_clock_claim_keeps_render_and_owner_boundaries): the
            # pre-task-10 refresh registered the claim-less view BEFORE
            # rendering, so any raise left the owner unclaimed. Reproduce
            # that end state here, then surface the failure.
            coordinator.show_decisions(self, self.binding if available else None)
            raise
        notice = coordinator.notices.get(self.binding, "")
        if "worktree_merge" in payloads:
            notice = "A worktree decision needs review. Open Console to continue."
        notice_widget = self.query_one("#buddy-reply-notice", Static)
        if force or str(notice_widget.renderable) != notice:
            notice_widget.update(notice)
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
        if not self.is_mounted or not self._visible:
            return
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
