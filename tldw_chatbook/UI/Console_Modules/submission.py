"""Console composer submission admission, accepted-draft barriers, and send stash recovery."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from ...Widgets.Console.console_composer_bar import ConsoleDraftStash
import asyncio
from loguru import logger
from textual.css.query import QueryError
from .prompt_queue import ConsolePromptDispatchStatus
from .retrieval import source_mentions_rag as _source_mentions_rag
from ..Console_Modules import raw_cli as raw_cli_ui
from ...Chat.console_command_grammar import (
    KIND_COMMAND,
    KIND_NOT_COMMAND,
    KIND_UNKNOWN,
    REWIND_COMMAND_NAME,
    CommandParse,
)
from ...Chat.console_chat_models import ConsoleRunStatus
from ...Chat.console_display_state import build_console_evidence_display_state


logger = logger.bind(module="ChatScreen")


class ConsoleSubmissionController:
    """Own console composer submission admission, accepted-draft barriers, and send stash recovery.

    App identity is stable for this controller lifetime. All other dependencies
    are explicit callables resolved by wiring at use time. No DOM is owned here.
    """

    def __init__(
        self,
        *,
        app_instance_accessor: Callable[[], Any],
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
        _dismiss_console_guidance: Callable[..., Any],
        _dispatch_console_command: Callable[..., Any],
        _dispatch_draft: Callable[..., Any],
        _dispatch_prompt: Callable[..., Any],
        _ensure_console_chat_controller: Callable[..., Any],
        _ensure_console_chat_store: Callable[..., Any],
        _fetch_skill_context: Callable[..., Any],
        _focus_console_composer_if_needed: Callable[..., Any],
        _invalidate_persisted_rows_cache: Callable[..., Any],
        _query_composer: Callable[..., Any],
        _record_console_first_send: Callable[..., Any],
        _respond_to_blocked_skill: Callable[..., Any],
        _restore_stash: Callable[..., Any],
        _start_console_transcript_sync_timer: Callable[..., Any],
        _start_raw_command: Callable[..., Any],
        _stop_draft_spend_refresh: Callable[..., Any],
        _sync_console_command_popup: Callable[..., Any],
        _sync_native_console_chat_ui: Callable[..., Any],
        _unknown_command_hint: Callable[..., Any],
        run_worker: Callable[..., Any],
        _console_chat_store_accessor: Callable[[], Any],
        _console_command_registry_accessor: Callable[[], Any],
        _console_undo_histories_accessor: Callable[[], Any],
        _console_visible_draft_session_id_accessor: Callable[[], Any],
        is_mounted_accessor: Callable[[], Any],
        _watchdog_seconds_accessor: Callable[[], Any],
    ) -> None:
        self._app_instance_accessor = app_instance_accessor
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
        self._dismiss_console_guidance = _dismiss_console_guidance
        self._dispatch_console_command = _dispatch_console_command
        self._dispatch_draft = _dispatch_draft
        self._dispatch_prompt = _dispatch_prompt
        self._ensure_console_chat_controller = _ensure_console_chat_controller
        self._ensure_console_chat_store = _ensure_console_chat_store
        self._fetch_skill_context = _fetch_skill_context
        self._focus_console_composer_if_needed = _focus_console_composer_if_needed
        self._invalidate_persisted_rows_cache = _invalidate_persisted_rows_cache
        self._query_composer = _query_composer
        self._record_console_first_send = _record_console_first_send
        self._respond_to_blocked_skill = _respond_to_blocked_skill
        self._restore_stash = _restore_stash
        self._start_console_transcript_sync_timer = _start_console_transcript_sync_timer
        self._start_raw_command = _start_raw_command
        self._stop_draft_spend_refresh = _stop_draft_spend_refresh
        self._sync_console_command_popup = _sync_console_command_popup
        self._sync_native_console_chat_ui = _sync_native_console_chat_ui
        self._unknown_command_hint = _unknown_command_hint
        self.run_worker = run_worker
        self._console_chat_store_accessor = _console_chat_store_accessor
        self._console_command_registry_accessor = _console_command_registry_accessor
        self._console_undo_histories_accessor = _console_undo_histories_accessor
        self._console_visible_draft_session_id_accessor = (
            _console_visible_draft_session_id_accessor
        )
        self.is_mounted_accessor = is_mounted_accessor
        self._watchdog_seconds_accessor = _watchdog_seconds_accessor
        self._console_pending_send_stash = None
        self._console_inflight_send_stashes = {}
        self._console_submit_session_by_task = {}
        self._console_unknown_send_armed = None

    @property
    def _console_chat_store(self) -> Any:
        return self._console_chat_store_accessor()

    @property
    def _console_command_registry(self) -> Any:
        return self._console_command_registry_accessor()

    @property
    def _console_undo_histories(self) -> Any:
        return self._console_undo_histories_accessor()

    @property
    def _console_visible_draft_session_id(self) -> Any:
        return self._console_visible_draft_session_id_accessor()

    @property
    def is_mounted(self) -> Any:
        return self.is_mounted_accessor()

    @property
    def _watchdog_seconds(self) -> Any:
        return self._watchdog_seconds_accessor()

    @property
    def app_instance(self) -> Any:
        return self._app_instance_accessor()

    async def _submit_console_native_draft(
        self, draft: str, session_id: str | None = None
    ) -> None:
        controller = self._ensure_console_chat_controller()
        self._stop_draft_spend_refresh()
        self._start_console_transcript_sync_timer()
        # Task 3b: `session_id` is the session THIS worker was dispatched
        # for (`_dispatch_console_draft_send` already resolved it via the
        # `console-run-{session_id}` group). Defaulted to the currently
        # active session only for direct-call test idioms that predate the
        # per-session stash map -- equivalent to the old singular-slot
        # behavior for the (overwhelmingly common) single-session case.
        if session_id is None:
            session_id = controller.store.active_session_id or ""
        dispatch_composer = self._console_composer_or_none()
        dispatch_snapshot = (
            dispatch_composer.capture_draft_snapshot()
            if dispatch_composer is not None
            and self._console_visible_draft_session_id == session_id
            else None
        )
        dispatch_history = (
            dispatch_composer.export_undo_history()
            if dispatch_snapshot is not None
            else None
        )
        dispatch_draft_revision = (
            (
                dispatch_composer.edit_serial,
                dispatch_composer.capture_draft_snapshot().generation,
            )
            if dispatch_composer is not None
            and self._console_visible_draft_session_id == session_id
            else None
        )
        task = asyncio.current_task()
        if task is not None:
            # See `_on_console_submission_accepted`: it fires synchronously
            # from deep inside the `submit_draft` await below, on this SAME
            # task, and has no session id of its own to key by.
            self._console_submit_session_by_task[task] = session_id
        # TASK-340: a keyboard send already cleared the composer at the Enter
        # keypress. The accepted-hook consumes this slot; a refusal below
        # restores it instead. Snapshot before submit_draft so the hook's
        # consumption is observable here.
        inflight_stash = self._console_inflight_send_stashes.get(session_id)
        try:
            # F4 fix (Qodo wave): thread the session THIS worker was
            # dispatched for all the way into the controller -- previously
            # `submit_draft` re-resolved "the session to submit into" via
            # `store.active_session_id` at execution time, so a tab switch
            # racing the scheduling gap between `run_worker(...)` and this
            # coroutine body actually running could submit into whichever
            # session the user switched TO instead of the dispatching one.
            result = await controller.run_prompt_chain(draft, session_id=session_id)
        except Exception:
            # An unexpected submit crash must not eat the keypress-cleared
            # draft — and must not escape the worker (exit_on_error would
            # take the whole app down with it).
            leaked_stash = self._console_inflight_send_stashes.pop(session_id, None)
            if leaked_stash is not None:
                self._restore_stash(leaked_stash)
            logger.exception("Console submit failed unexpectedly")
            self.app_instance.notify(
                "Console send failed unexpectedly — your draft was restored.",
                severity="error",
            )
            return
        finally:
            if task is not None:
                self._console_submit_session_by_task.pop(task, None)
        # TASK-251: a submit may have created/updated a persisted
        # conversation (title, updated_at) -- invalidate so the browser
        # reflects it on the very next sync instead of the TTL window.
        self._invalidate_persisted_rows_cache()
        try:
            composer = self._query_composer()
        except QueryError:
            composer = None
        # Task 3b: only the composer that STILL SHOWS this session gets
        # mutated on its behalf. A background session's dispatch can
        # complete long after the user switched away -- restoring an
        # abandoned draft (or clearing should_clear_draft below) into
        # whatever composer happens to be visible would leak this
        # session's text into a DIFFERENT session's tab.
        composer_reflects_session = (
            composer is not None and controller.store.active_session_id == session_id
        )
        # TASK-1281 review NEW-5: `clear_draft`/`clear_history` below must
        # only ever touch the composer when it PROVABLY shows this exact
        # session's draft right now, not merely when the store's active
        # session id happens to match -- `composer_reflects_session` above
        # is Task 3b's pre-existing (looser) check, kept as-is for
        # `restore_stashed_draft` below, but during the TASK-339
        # session-switch settle window `active_session_id` can already
        # equal `session_id` while the composer still visibly shows a
        # DIFFERENT session (see F1) -- clearing on that weaker guard would
        # wipe the wrong session's on-screen draft. Unified with
        # `_on_console_submission_accepted`'s own guard shape.
        composer_visible_for_session = (
            composer is not None
            and self._console_visible_draft_session_id == session_id
        )
        stash = (
            self._console_inflight_send_stashes.pop(session_id, None) or inflight_stash
        )
        if (
            not result.accepted
            and stash is not None
            and composer_reflects_session
            and composer_visible_for_session
        ):
            # Controller-level refusal of a keyboard send: the composer was
            # cleared at the keypress, so hand the draft back (ahead of any
            # keystrokes typed since).
            if (
                dispatch_snapshot is not None
                and composer.edit_serial == dispatch_snapshot.edit_serial
            ):
                composer.restore_undo_history(dispatch_history)
            composer.restore_stashed_draft(stash)
        elif (
            not result.accepted
            and composer_visible_for_session
            and dispatch_snapshot is not None
            and composer.edit_serial == dispatch_snapshot.edit_serial
            and not composer.draft_text()
        ):
            # Setup may refuse after the accepted hook cleared this draft.
            # A newer edit or another visible session retains ownership.
            composer.restore_snapshot(dispatch_snapshot)
            composer.restore_undo_history(dispatch_history)
        if result.session_closed:
            # Task 4 (D2 fix wave): `_session_closed_result` is `accepted`
            # (see its own docstring) so the restore above never fires, and
            # its owning session no longer exists to hold a SYSTEM row --
            # there is nothing left to write into and nowhere to restore a
            # keypress-cleared draft TO. A toast is the one surface still
            # available: without it this outcome was completely silent
            # (composer already cleared, no row, no notification).
            # Fix-round-2 (I2/M2): `session_closed` is now set ONLY at the
            # dispatch-gap call site (the OTHER ~19 `_session_closed_result`
            # sites -- mid-run closes the user already confirmed -- leave it
            # `False`), and that ONE site's `visible_copy` is always the
            # informative "...before your message could send." string, not
            # the generic "Session closed." every other site uses -- so
            # `result.visible_copy` is used directly, with no dead fallback.
            self.app_instance.notify(result.visible_copy, severity="warning")
        if (
            result.should_clear_draft
            and composer_visible_for_session
            and inflight_stash is None
            and (
                dispatch_draft_revision is None
                or (
                    composer.edit_serial,
                    composer.capture_draft_snapshot().generation,
                )
                == dispatch_draft_revision
            )
        ):
            # Stashed sends were cleared at the keypress — clearing again
            # here would eat keystrokes typed after Enter (the next draft).
            composer.clear_draft()
            # TASK-1281 review F2: send is a history barrier -- see
            # `_on_console_submission_accepted`'s identical comment. This
            # site covers the same "content is genuinely gone" moment for
            # sends that reach here without an inflight keypress stash
            # (e.g. the mouse-click Send path).
            composer.clear_history()
            self._sync_console_command_popup()
        if result.accepted:
            # TASK-1281 review NEW-5: only an ACCEPTED send makes this
            # session's pre-send history genuinely stale -- a refusal
            # (blocked/failed/canceled) sent nothing, so a background
            # session's banked undo/redo history must survive it exactly
            # as it would have survived never attempting the send at all.
            self._console_undo_histories.pop(session_id, None)
        if (
            result.accepted
            and controller.run_state.status is ConsoleRunStatus.COMPLETED
        ):
            # Retry/continue/regenerate paths intentionally don't record the flag here —
            # they require an existing message, so ``has_messages`` already keeps the
            # card hidden and the flag was set by the originating submit.
            # Failed/stopped first sends must NOT set the one-time flag: the
            # setup card should return until a send completes with content.
            self._record_console_first_send()
        await self._sync_native_console_chat_ui()

    def _on_console_submission_accepted(self) -> None:
        """Clear the composer as soon as a submit is accepted, not at run end.

        Keeping the sent text in the composer for the whole run reads as
        "not sent" during long local-model generations; blocked submits never
        reach this hook, so their draft is preserved for correction.
        ``ConsoleChatController.submit_draft`` invokes this hook only once
        its own skill-substitution/trust re-check has confirmed the turn
        actually proceeds (Qodo finding 3, PR #636 bot review) -- a
        substitution refusal, like any other blocked submit, never reaches
        it, so a refused draft stays in the composer too.

        Task 3b: this fires synchronously from deep inside ``submit_draft``,
        on the SAME task as the ``_submit_console_native_draft`` worker that
        awaited it -- ``_console_submit_session_by_task`` resolves which
        session's stash entry (if any) is this call's own, without changing
        this hook's public no-arg ``Callable[[], None]`` contract (still
        assignable via ``controller.on_submission_accepted = ...`` exactly
        as before). A lookup miss (direct-call test idioms, or no wrapping
        task) falls back to the active session -- the pre-Task-3b behavior.
        """
        try:
            composer = self._query_composer()
        except QueryError:
            composer = None
        task = asyncio.current_task()
        session_id = (
            self._console_submit_session_by_task.get(task) if task is not None else None
        )
        active_session_id = self._ensure_console_chat_store().active_session_id or ""
        if session_id is None:
            session_id = active_session_id
        if session_id in self._console_inflight_send_stashes:
            # TASK-340: this submit's draft was captured and cleared at the
            # Enter keypress — clearing now would eat keystrokes typed since
            # (they are the NEXT draft). Consume the stash instead.
            self._console_inflight_send_stashes.pop(session_id, None)
        elif composer is not None and active_session_id == session_id:
            composer.clear_draft()
            self._sync_console_command_popup()
        # TASK-1281 review F2: this hook fires ONLY once submit_draft has
        # confirmed the turn actually proceeds (never for a blocked/refused
        # send -- see the docstring above), so every call here represents a
        # draft that is genuinely, irrevocably gone. Clearing just the
        # draft text (above) is not enough: the mutations that PRODUCED it
        # stay reachable on the undo stack either way (a `clear_draft()`
        # with no `record_history=True` records nothing, so it doesn't
        # cover them), and Ctrl+Z would resurrect already-sent content back
        # into the composer -- and, via the undo/redo re-persist, right
        # back into the store as the "live" draft for a message that has
        # already shipped. Drops the banked history unconditionally (a sent
        # session can never be usefully switched back into with anything
        # from before the send), and the composer's own live stacks too
        # when it still shows this exact session.
        self._console_undo_histories.pop(session_id, None)
        if (
            composer is not None
            and self._console_visible_draft_session_id == session_id
        ):
            composer.clear_history()
        # A send can finish while navigation is tearing this screen down. Do
        # not create a coroutine that Textual will reject after unmounting;
        # the next mounted view rebuilds from the durable chat store.
        if not self.is_mounted:
            return
        # task-351(a): echo the just-appended USER message immediately rather
        # than waiting up to a full 0.2s transcript-poll cycle (and a heavy
        # first poll after it). The composer clears here at acceptance, so
        # without this the transcript still read "No messages yet" for ~600ms
        # after the text vanished — reading as "not sent". This hook only fires
        # once submit_draft has confirmed the turn actually proceeds (never for
        # a blocked/refused send), so the USER row is already in the store.
        # `_sync_native_console_chat_ui` coalesces against a running sync via
        # its own `_console_sync_in_progress`/`_console_sync_requested` guard
        # (a concurrent call sets "requested" and the in-progress run re-fires
        # from its `finally`), so the echo still lands. NOT `exclusive=True`:
        # that would CANCEL a console-sync worker mid-flight, and a sync
        # cancelled after it advanced a scope sentinel but before its awaited
        # refresh completed would leave inspector/summary caches stale until the
        # scope next changes (Qodo #2). Coalescing gives the echo without that
        # cancellation. `exit_on_error=False`: best-effort acknowledgment — if
        # the screen is tearing down (or a send races a navigation away) the
        # sync can hit a removed widget and raise `NoMatches`; the poll runs the
        # same coroutine from a timer whose exceptions Textual already absorbs,
        # so a transient failure here must likewise never crash the app (default
        # `exit_on_error=True` would) — the next poll re-renders regardless.
        self.run_worker(
            self._sync_native_console_chat_ui(),
            group="console-sync",
            exit_on_error=False,
        )

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

    async def _send_console_message_from_visible_action(self) -> bool:
        """Route the visible Console send action through the native controller.

        Returns:
            True once the draft has been queued as a user turn; False on every
            refusal -- an empty draft with no attachment, a `/`-command or
            unknown-command dispatch (which never sends by design), and every
            gate inside `_dispatch_console_draft_send`. Each refusal has
            already shown its own toast or system row.
        """
        # TASK-340: a keyboard send captured its payload at the Enter
        # keypress; the mouse path still reads the live draft here.
        stash = self._console_pending_send_stash
        self._console_pending_send_stash = None
        stash, composer, draft, raw_cli_handled = raw_cli_ui.prepare_visible_send(
            stash, self._console_composer_or_none, self._start_raw_command
        )
        if raw_cli_handled:
            return False
        if not draft.strip() and self._pending_image_attachment() is None:
            if composer is not None:
                composer.restore_stashed_draft(stash)
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
            opening_composer = composer if stash is None else None
            opening_revision = None
            if opening_composer is not None:
                opening_revision = (
                    opening_composer.edit_serial,
                    opening_composer.capture_draft_snapshot().generation,
                    draft,
                )
            opened = False
            try:
                opened = await self._console_command_rewind(parse)
            finally:
                if not opened and composer is not None:
                    composer.restore_stashed_draft(stash)
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
            # Commands operate on the live composer draft (`/prompt` replaces
            # it wholesale, unrecognized handlers leave it untouched) — put
            # the stash back first so their semantics stay identical.
            if composer is not None:
                composer.restore_stashed_draft(stash)
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
                if composer is not None:
                    composer.restore_stashed_draft(stash)
                return False
            if self._console_unknown_send_armed == draft:
                # Second consecutive Enter on the *same* unmodified draft:
                # disarm and fall through to a normal send below.
                self._console_unknown_send_armed = None
            else:
                self._console_unknown_send_armed = draft
                if composer is not None:
                    composer.restore_stashed_draft(stash)
                await self._append_native_console_system_message(
                    self._unknown_command_hint(parse.name)
                )
                return False

        if self._answer_pending_question_with_draft(draft):
            return False
        return await self._dispatch_draft(draft, stash=stash)

    async def _dispatch_console_draft_send(
        self, draft: str, stash: "ConsoleDraftStash | None" = None
    ) -> bool:
        """Compatibility delegate for the one typed queue-aware dispatcher."""

        result = await self._dispatch_prompt(draft, stash=stash)
        return result.status is not ConsolePromptDispatchStatus.REFUSED

    def _restore_console_send_stash(self, stash: "ConsoleDraftStash | None") -> None:
        """Hand a keypress-captured draft back to the composer (TASK-340)."""
        if stash is None:
            return
        try:
            composer = self._query_composer()
        except QueryError:
            return
        composer.restore_stashed_draft(stash)

    def _recover_stuck_console_send_stash(
        self, stash: "ConsoleDraftStash | None"
    ) -> None:
        """Recover a keypress-captured draft `Button.Pressed` never consumed.

        Task 4 fix-round-2 (I3): the Enter handler's own no-op-press check
        (``send_button.disabled or not send_button.display`` right before
        ``.press()``) only catches the case where the button was ALREADY
        disabled/hidden at that instant. ``.press()`` itself just POSTS
        ``Button.Pressed`` for the message pump to deliver later -- if the
        button (or its composer) is pruned in the gap between that post and
        the pump actually delivering it, the message is dropped and
        ``handle_console_send_message``/``_send_console_message_from_
        visible_action`` -- the ONLY code that consumes ``_console_pending_
        send_stash`` -- never runs. Without this recovery, that leaves the
        stash slot permanently non-``None``, and the duplicate-send guard at
        the top of the ``"enter"`` branch swallows every subsequent Enter
        forever (D2's exact shape, via a narrower door than the no-op-press
        check alone closes).

        Scheduled once per send via ``set_timer`` right after ``.press()``;
        a no-op in the overwhelmingly common case where the Pressed handler
        already consumed the slot (or a later send's own stash superseded
        this one -- blocked from happening while this slot is still set by
        the duplicate guard itself, but checked by identity anyway as a
        cheap belt-and-suspenders).

        Args:
            stash: The exact stash object this watchdog was scheduled for.
        """
        if self._console_pending_send_stash is not stash:
            return
        logger.warning(
            "Console send Enter: pending stash was never consumed by the "
            "Pressed handler after {:.2f}s -- recovering the draft instead "
            "of leaving the duplicate-send guard latched shut.",
            self._watchdog_seconds,
        )
        self._console_pending_send_stash = None
        self._restore_stash(stash)
