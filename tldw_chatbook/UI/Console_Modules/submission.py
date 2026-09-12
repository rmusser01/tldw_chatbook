"""Console composer submission admission and send-time draft capture."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ...Chat.console_command_grammar import (
    KIND_COMMAND,
    KIND_NOT_COMMAND,
    KIND_UNKNOWN,
    REWIND_COMMAND_NAME,
    CommandParse,
)
from ...Chat.console_display_state import build_console_evidence_display_state
from ...Widgets.Console.console_composer_bar import ConsoleDraftStash
from . import raw_cli as raw_cli_ui
from .prompt_queue import ConsolePromptDispatchStatus
from .retrieval import source_mentions_rag as _source_mentions_rag


@dataclass(frozen=True, slots=True)
class _ConsolePendingSend:
    """One keyboard capture claimable only by its scheduled callback."""

    session_id: str
    stash: ConsoleDraftStash | None
    token: object


class ConsoleSubmissionController:
    """Own console composer admission and send-time draft capture.

    App identity is stable for this controller lifetime. All other dependencies
    are explicit callables resolved by wiring at use time. No DOM is owned here.
    """

    def __init__(
        self,
        *,
        app_instance_accessor: Callable[[], Any],
        _ui_responsiveness_monitor: Callable[[], Any],
        _active_console_settings_readiness: Callable[..., Any],
        _pending_image_attachment: Callable[..., Any],
        _attachment_block_reason: Callable[..., Any],
        _answer_pending_question_with_draft: Callable[..., Any],
        _append_native_console_system_message: Callable[..., Any],
        _blocked_skill_summaries: Callable[..., Any],
        _clear_console_composer_draft: Callable[..., Any],
        _console_command_rewind: Callable[..., Any],
        _console_composer_or_none: Callable[..., Any],
        _consume_pending_console_launch: Callable[..., Any],
        _current_console_conversation_id: Callable[[], str | None],
        _ensure_active_console_session_settings: Callable[..., Any],
        _sync_console_session_draft: Callable[..., Any],
        _dismiss_console_guidance: Callable[..., Any],
        _dispatch_console_command: Callable[..., Any],
        _dispatch_draft: Callable[..., Any],
        _dispatch_prompt: Callable[..., Any],
        _fetch_skill_context: Callable[..., Any],
        _focus_console_composer_if_needed: Callable[..., Any],
        _respond_to_blocked_skill: Callable[..., Any],
        _start_raw_command: Callable[..., Any],
        _unknown_command_hint: Callable[..., Any],
        _console_chat_store_accessor: Callable[[], Any],
        _console_command_registry_accessor: Callable[[], Any],
        _console_visible_draft_session_id_accessor: Callable[[], Any],
    ) -> None:
        self._app_instance_accessor = app_instance_accessor
        self._ui_responsiveness_monitor = _ui_responsiveness_monitor
        self._active_console_settings_readiness = _active_console_settings_readiness
        self._pending_image_attachment = _pending_image_attachment
        self._attachment_block_reason = _attachment_block_reason
        self._answer_pending_question_with_draft = _answer_pending_question_with_draft
        self._append_native_console_system_message = (
            _append_native_console_system_message
        )
        self._blocked_skill_summaries = _blocked_skill_summaries
        self._clear_console_composer_draft = _clear_console_composer_draft
        self._console_command_rewind = _console_command_rewind
        self._console_composer_or_none = _console_composer_or_none
        self._consume_pending_console_launch = _consume_pending_console_launch
        self._current_console_conversation_id = _current_console_conversation_id
        self._ensure_active_console_session_settings = (
            _ensure_active_console_session_settings
        )
        self._sync_console_session_draft = _sync_console_session_draft
        self._dismiss_console_guidance = _dismiss_console_guidance
        self._dispatch_console_command = _dispatch_console_command
        self._dispatch_draft = _dispatch_draft
        self._dispatch_prompt = _dispatch_prompt
        self._fetch_skill_context = _fetch_skill_context
        self._focus_console_composer_if_needed = _focus_console_composer_if_needed
        self._respond_to_blocked_skill = _respond_to_blocked_skill
        self._start_raw_command = _start_raw_command
        self._unknown_command_hint = _unknown_command_hint
        self._console_chat_store_accessor = _console_chat_store_accessor
        self._console_command_registry_accessor = _console_command_registry_accessor
        self._console_visible_draft_session_id_accessor = (
            _console_visible_draft_session_id_accessor
        )
        self._console_pending_send: _ConsolePendingSend | None = None
        self._console_unknown_send_armed = None

    @property
    def _console_chat_store(self) -> Any:
        return self._console_chat_store_accessor()

    @property
    def _console_command_registry(self) -> Any:
        return self._console_command_registry_accessor()

    @property
    def _console_visible_draft_session_id(self) -> Any:
        return self._console_visible_draft_session_id_accessor()

    @property
    def app_instance(self) -> Any:
        return self._app_instance_accessor()

    def _console_pending_image_attachment(self):
        """Return a staged image attachment, if any staged item qualifies.

        Scans the whole staged list (not just the first item) so a
        multi-attachment session still gates vision-capability/blocked-send
        checks correctly when the qualifying image isn't staged first.
        """
        store = self._console_chat_store
        if store is None or store.active_session_id is None:
            return None
        try:
            pendings = store.pending_attachments(store.active_session_id)
        except KeyError:
            return None
        for pending in pendings:
            if (
                pending is not None
                and pending.insert_mode == "attachment"
                and pending.file_type == "image"
                and pending.data is not None
            ):
                return pending
        return None

    def _console_attachment_blocked_reason(self) -> str:
        """Return blocked-send copy when a staged image can't reach the model."""
        from tldw_chatbook.Chat.attachment_core import vision_block_reason

        if self._pending_image_attachment() is None:
            return ""
        effective_settings, _readiness = self._active_console_settings_readiness()
        return (
            vision_block_reason(effective_settings.provider, effective_settings.model)
            or ""
        )

    def _console_send_blocked_reason(self) -> str:
        """Return a user-facing reason if Console send cannot safely run."""
        conversation_id = self._current_console_conversation_id()
        if conversation_id in getattr(
            self.app_instance, "_conversation_archive_inflight", ()
        ):
            return "Archive change in progress. Your draft is preserved."
        # A cached archive flag may predate a restore by another writer.
        # The awaited submit boundary checks durable state before any send.
        pending_launch = self._consume_pending_console_launch()
        if pending_launch is not None and _source_mentions_rag(pending_launch.source):
            evidence_state = build_console_evidence_display_state(pending_launch)
            if evidence_state is None or evidence_state.available_count == 0:
                return (
                    "Console send blocked: Library search has no available evidence. "
                    "Review source authority before sending."
                )
        _readiness_settings, readiness = self._active_console_settings_readiness()
        if (
            readiness.operability == "not_ready"
            and readiness.recovery_action != "wait_for_active_run"
        ):
            # Active-run admission belongs to the prompt queue. It refuses
            # while the turn is preparing and admits Queue after acceptance;
            # only actual provider setup gaps belong in this gate.
            if readiness.recovery_action == "configure_credential":
                provider = readiness.provider_display_name or "this provider"
                return (
                    f"Console send blocked: Add an API key for {provider} before "
                    "sending."
                )
            if readiness.recovery_action == "select_model":
                return "Console send blocked: Select a model before sending."
            if readiness.recovery_action == "save_endpoint":
                return (
                    "Console send blocked: Save the provider endpoint before sending."
                )
            if readiness.recovery_action == "configure_endpoint":
                return "Console send blocked: Enter a valid provider endpoint before sending."
            if readiness.recovery_action == "retry_connection":
                return "Console send blocked: Retry the provider connection before sending."
            return "Console send blocked: Finish provider setup before sending."
        attachment_reason = self._attachment_block_reason()
        if attachment_reason:
            return attachment_reason
        return ""

    def _console_visible_send_session_id(self) -> str | None:
        """Return the exact session represented by the mounted composer."""

        session_id = self._console_visible_draft_session_id
        if session_id is not None:
            return session_id
        self._ensure_active_console_session_settings()
        self._sync_console_session_draft()
        return self._console_visible_draft_session_id

    async def _send_console_message_from_visible_action(
        self,
        *,
        session_id: str | None = None,
        pending_send_token: object | None = None,
    ) -> bool:
        """Observe the visible action before command parsing and send gating."""
        from tldw_chatbook.Chat.console_send_diagnostics import send_diagnostic_scope

        async with send_diagnostic_scope(
            "ui_action", self._ui_responsiveness_monitor()
        ) as diagnostic:
            sent = await self._send_console_message_from_visible_action_observed(
                session_id=session_id, pending_send_token=pending_send_token
            )
            diagnostic.outcome = "dispatched" if sent else "not_dispatched"
            return sent

    async def _send_console_message_from_visible_action_observed(
        self,
        *,
        session_id: str | None = None,
        pending_send_token: object | None = None,
    ) -> bool:
        """Route the visible Console send action through the native controller.

        Returns:
            True once the draft has been queued as a user turn; False on every
            refusal -- an empty draft with no attachment, a `/`-command or
            unknown-command dispatch (which never sends by design), and every
            gate inside `_dispatch_console_draft_send`. Each refusal has
            already shown its own toast or system row.
        """
        # A scheduled Enter callback may consume only its own capture.
        # Mouse/Workbench sends have no token and always read the live draft.
        stash = None
        if pending_send_token is not None:
            pending_send = self._console_pending_send
            if pending_send is None or pending_send.token is not pending_send_token:
                return False
            self._console_pending_send = None
            session_id = pending_send.session_id
            stash = pending_send.stash
        if session_id is None:
            session_id = self._console_visible_send_session_id()
        if (
            pending_send_token is None
            and self._console_pending_send is not None
            and self._console_pending_send.session_id == session_id
        ):
            return False
        if session_id is None or self._console_visible_draft_session_id != session_id:
            self.app_instance.notify(
                "Console chat changed before send; the draft was kept in its original chat.",
                severity="warning",
            )
            return False
        stash, composer, draft, raw_cli_handled = raw_cli_ui.prepare_visible_send(
            stash, self._console_composer_or_none, self._start_raw_command
        )
        if raw_cli_handled:
            return False
        if pending_send_token is None and composer is not None:
            stash = composer.capture_draft_for_send()
            draft = stash.text if stash is not None else draft
        if not draft.strip() and self._pending_image_attachment() is None:
            self._focus_console_composer_if_needed(force=True)
            return False
        self._dismiss_console_guidance()

        # Command parsing runs before any readiness/blocked gating: a
        # recognized command dispatch (or an unknown-command hint) never
        # sends, so it must work even while Send is blocked. Draft text
        # carrying any real paste-originated segment (regardless of its
        # current collapse/confirm/expanded display state) is never treated
        # as command input -- Task 9's grammar module deliberately leaves
        # that gating to the caller, since only the composer knows the real
        # segment state.
        has_paste = (
            stash.has_paste
            if stash is not None
            else (composer is not None and composer.has_paste_segments())
        )
        if composer is not None and not has_paste:
            parse = self._console_command_registry.parse(draft)
        else:
            parse = CommandParse(kind=KIND_NOT_COMMAND)

        argument_free_rewind = (
            parse.kind == KIND_COMMAND
            and parse.name == REWIND_COMMAND_NAME
            and parse.args == ""
        )
        if argument_free_rewind:
            self._console_unknown_send_armed = None
            opening_composer = composer if pending_send_token is None else None
            opening_revision = None
            if opening_composer is not None:
                opening_revision = (
                    opening_composer.edit_serial,
                    opening_composer.capture_draft_snapshot().generation,
                    draft,
                )
            opened = await self._console_command_rewind(parse)
            if opened and opening_composer is not None and opening_revision is not None:
                current = self._console_composer_or_none()
                current_snapshot = (
                    current.capture_draft_snapshot()
                    if current is opening_composer
                    else None
                )
                if (
                    current is opening_composer
                    and current.edit_serial == opening_revision[0]
                    and current_snapshot is not None
                    and current_snapshot.generation == opening_revision[1]
                    and current.draft_text() == opening_revision[2]
                ):
                    self._clear_console_composer_draft()
            return False

        if parse.kind == KIND_COMMAND:
            # Captured drafts remain in the composer until runtime custody.
            self._console_unknown_send_armed = None
            await self._dispatch_console_command(parse)
            return False

        if parse.kind == KIND_UNKNOWN:
            # Fold-in (Task 9 fix-wave review; hard removal Task 4 -- there
            # is no fallback resolver at all anymore, so EVERY unmatched
            # `/word` reaches here as KIND_UNKNOWN): a typed `/name` that
            # matches ONLY needs-review (trust-blocked) skills would
            # otherwise fall through to the generic "Unknown command" hint
            # just like any other unrecognized word. Checking against a
            # FRESH context surfaces the same needs-review response instead,
            # before the unknown-command arm/hint logic ever runs. This
            # never arms the unknown-command escape: a blocked match is a
            # known-but-blocked command, not an unrecognized one, so a
            # repeated Enter shows the same response again rather than
            # silently falling through to a literal send.
            context = await self._fetch_skill_context()
            blocked_summaries = self._blocked_skill_summaries(context)
            if await self._respond_to_blocked_skill(parse.name, blocked_summaries):
                return False
            if self._console_unknown_send_armed == draft:
                # Second consecutive Enter on the *same* unmodified draft:
                # disarm and fall through to a normal send below.
                self._console_unknown_send_armed = None
            else:
                self._console_unknown_send_armed = draft
                await self._append_native_console_system_message(
                    self._unknown_command_hint(parse.name)
                )
                return False

        if self._answer_pending_question_with_draft(draft):
            return False
        if self._console_visible_draft_session_id != session_id:
            self.app_instance.notify(
                "Console chat changed before send; the draft was kept in its original chat.",
                severity="warning",
            )
            return False
        return await self._dispatch_draft(draft, stash=stash, session_id=session_id)

    async def _dispatch_console_draft_send(
        self,
        draft: str,
        stash: "ConsoleDraftStash | None" = None,
        *,
        session_id: str | None = None,
    ) -> bool:
        """Compatibility delegate for the one typed queue-aware dispatcher."""

        from tldw_chatbook.Chat.console_send_diagnostics import send_diagnostic_scope

        async with send_diagnostic_scope(
            "ui_dispatch", self._ui_responsiveness_monitor()
        ) as diagnostic:
            if session_id is None:
                session_id = self._console_visible_send_session_id()
            result = await self._dispatch_prompt(
                draft, session_id=session_id, stash=stash
            )
            diagnostic.outcome = result.status.value
            return result.status is not ConsolePromptDispatchStatus.REFUSED
