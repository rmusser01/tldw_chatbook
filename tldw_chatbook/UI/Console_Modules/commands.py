"""Console command routing, prompt transformations, and rewind orchestration."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from collections.abc import Mapping
import inspect
from typing import Optional
from loguru import logger
from textual.css.query import QueryError
from ...Chat.console_context_compaction import (
    EffectiveMemoryKind,
    complete_durable_units,
)
from ...Chat.console_command_grammar import (
    FEWER_PERMISSION_PROMPTS_COMMAND_HANDLER_ID,
    FEWER_PERMISSION_PROMPTS_COMMAND_NAME,
    GENERATE_IMAGE_COMMAND_HANDLER_ID,
    GENERATE_IMAGE_COMMAND_NAME,
    GENERATE_VIDEO_COMMAND_HANDLER_ID,
    GENERATE_VIDEO_COMMAND_NAME,
    PREFILL_COMMAND_HANDLER_ID,
    PREFILL_COMMAND_NAME,
    PROMPT_COMMAND_HANDLER_ID,
    PROMPT_COMMAND_NAME,
    RESEARCH_COMMAND_HANDLER_ID,
    RESEARCH_COMMAND_NAME,
    HELP_COMMAND_HANDLER_ID,
    HELP_COMMAND_NAME,
    CONSOLE_ACTION_COMMAND_HANDLER_ID,
    CONSOLE_ACTION_COMMANDS,
    DOCTOR_COMMAND_HANDLER_ID,
    DOCTOR_COMMAND_NAME,
    REWIND_COMMAND_HANDLER_ID,
    EMERGENCY_STOP_COMMAND_HANDLER_ID,
    EMERGENCY_STOP_COMMAND_NAME,
    REDIRECT_COMMAND_HANDLER_ID,
    REDIRECT_COMMAND_NAME,
    STEER_COMMAND_HANDLER_ID,
    STEER_COMMAND_NAME,
    REWIND_COMMAND_NAME,
    SKILLS_COMMAND_HANDLER_ID,
    SKILLS_COMMAND_NAME,
    STREAM_VIDEO_COMMAND_HANDLER_ID,
    STREAM_VIDEO_COMMAND_NAME,
    SYSTEM_COMMAND_HANDLER_ID,
    SYSTEM_COMMAND_NAME,
    CommandParse,
    default_console_registry,
)
from ...MCP.permission_prompt_reducer import format_permission_prompt_report
from ...Chat.console_prefill import (
    ACTION_CLEAR,
    ACTION_ERROR,
    ACTION_ONE_SHOT,
    ACTION_PIN,
    ACTION_STATUS,
    describe_prefill_preview,
    parse_prefill_args,
)
from ...Chat.console_generate_image import insert_style_token_into_draft
from ...Chat.console_chat_models import ConsoleMessageRole, derive_console_session_title
from ...Chat.console_ephemeral import blocked_reason
from ...Chat.console_command_suggestions import _COMMAND_DESCRIPTIONS

if TYPE_CHECKING:
    from ...Chat.console_chat_controller import ConsoleChatController
    from ...Widgets.Console.console_rewind_modal import (
        ConsoleRewindChoice,
        RewindPromptRow,
    )


logger = logger.bind(module="ChatScreen")


class ConsoleCommandsController:
    """Own console command routing, prompt transformations, and rewind orchestration.

    App identity is stable for this controller lifetime. All other dependencies
    are explicit callables resolved by wiring at use time. No DOM is owned here.
    """

    def __init__(
        self,
        *,
        app_instance_accessor: Callable[[], Any],
        _academic_research_enabled: Callable[..., Any],
        _active_session_settings: Callable[..., Any],
        _append_native_console_system_message: Callable[..., Any],
        _apply_rewind_choice: Callable[..., Any],
        _apply_rewind_position: Callable[..., Any],
        _clear_composer: Callable[..., Any],
        _console_active_session_is_ephemeral: Callable[..., Any],
        _console_command_apply_system: Callable[..., Any],
        _console_command_generate_image: Callable[..., Any],
        _console_command_generate_video: Callable[..., Any],
        _console_command_insert_prompt: Callable[..., Any],
        _console_command_skills: Callable[..., Any],
        _console_command_stream_video: Callable[..., Any],
        _console_composer_or_none: Callable[..., Any],
        _current_console_conversation_id: Callable[..., Any],
        _default_session_settings: Callable[..., Any],
        _ensure_console_chat_controller: Callable[..., Any],
        _ensure_console_chat_store: Callable[..., Any],
        _focus_console_composer_if_needed: Callable[..., Any],
        _insert_prompt_text: Callable[..., Any],
        _local_research_service: Callable[..., Any],
        _query_composer: Callable[..., Any],
        _request_rewind: Callable[..., Any],
        _research_database: Callable[..., Any],
        _resolve_action: Callable[..., Any],
        _summarize_console_from: Callable[..., Any],
        _summarize_console_up_to: Callable[..., Any],
        _sync_console_chat_core_state: Callable[..., Any],
        _sync_console_command_popup: Callable[..., Any],
        _sync_console_settings_summary: Callable[..., Any],
        _sync_native_console_chat_ui: Callable[..., Any],
        push_screen: Callable[..., Any],
        run_worker: Callable[..., Any],
    ) -> None:
        self._app_instance_accessor = app_instance_accessor
        self._academic_research_enabled = _academic_research_enabled
        self._active_session_settings = _active_session_settings
        self._append_native_console_system_message = (
            _append_native_console_system_message
        )
        self._apply_rewind_choice = _apply_rewind_choice
        self._apply_rewind_position = _apply_rewind_position
        self._clear_composer = _clear_composer
        self._console_active_session_is_ephemeral = _console_active_session_is_ephemeral
        self._console_command_apply_system = _console_command_apply_system
        self._console_command_generate_image = _console_command_generate_image
        self._console_command_generate_video = _console_command_generate_video
        self._console_command_insert_prompt = _console_command_insert_prompt
        self._console_command_skills = _console_command_skills
        self._console_command_stream_video = _console_command_stream_video
        self._console_composer_or_none = _console_composer_or_none
        self._current_console_conversation_id = _current_console_conversation_id
        self._default_session_settings = _default_session_settings
        self._ensure_console_chat_controller = _ensure_console_chat_controller
        self._ensure_console_chat_store = _ensure_console_chat_store
        self._focus_console_composer_if_needed = _focus_console_composer_if_needed
        self._insert_prompt_text = _insert_prompt_text
        self._local_research_service = _local_research_service
        self._query_composer = _query_composer
        self._request_rewind = _request_rewind
        self._research_database = _research_database
        self._resolve_action = _resolve_action
        self._summarize_console_from = _summarize_console_from
        self._summarize_console_up_to = _summarize_console_up_to
        self._sync_console_chat_core_state = _sync_console_chat_core_state
        self._sync_console_command_popup = _sync_console_command_popup
        self._sync_console_settings_summary = _sync_console_settings_summary
        self._sync_native_console_chat_ui = _sync_native_console_chat_ui
        self.push_screen = push_screen
        self.run_worker = run_worker
        self._console_command_registry = default_console_registry()

    @property
    def app_instance(self) -> Any:
        return self._app_instance_accessor()

    _CONSOLE_ACTION_COMMAND_TARGETS = {
        "model": "action_open_console_model_popover",
        "sessions": "action_open_console_session_switcher",
        "workspace": "action_open_console_workspace_switcher",
        "new": "action_new_console_tab",
        "temp": "action_new_temporary_console_tab",
        "settings": "action_open_console_session_settings",
        "context": "action_view_chat_context",
    }

    _CONSOLE_COMMAND_NAME_TO_HANDLER_ID = {
        PROMPT_COMMAND_NAME: PROMPT_COMMAND_HANDLER_ID,
        SYSTEM_COMMAND_NAME: SYSTEM_COMMAND_HANDLER_ID,
        SKILLS_COMMAND_NAME: SKILLS_COMMAND_HANDLER_ID,
        FEWER_PERMISSION_PROMPTS_COMMAND_NAME: (
            FEWER_PERMISSION_PROMPTS_COMMAND_HANDLER_ID
        ),
        PREFILL_COMMAND_NAME: PREFILL_COMMAND_HANDLER_ID,
        GENERATE_IMAGE_COMMAND_NAME: GENERATE_IMAGE_COMMAND_HANDLER_ID,
        GENERATE_VIDEO_COMMAND_NAME: GENERATE_VIDEO_COMMAND_HANDLER_ID,
        STREAM_VIDEO_COMMAND_NAME: STREAM_VIDEO_COMMAND_HANDLER_ID,
        REWIND_COMMAND_NAME: REWIND_COMMAND_HANDLER_ID,
        STEER_COMMAND_NAME: STEER_COMMAND_HANDLER_ID,
        REDIRECT_COMMAND_NAME: REDIRECT_COMMAND_HANDLER_ID,
        EMERGENCY_STOP_COMMAND_NAME: EMERGENCY_STOP_COMMAND_HANDLER_ID,
        RESEARCH_COMMAND_NAME: RESEARCH_COMMAND_HANDLER_ID,
        HELP_COMMAND_NAME: HELP_COMMAND_HANDLER_ID,
        DOCTOR_COMMAND_NAME: DOCTOR_COMMAND_HANDLER_ID,
        **{
            _name: CONSOLE_ACTION_COMMAND_HANDLER_ID
            for _name, _hint in CONSOLE_ACTION_COMMANDS
        },
    }

    _CONSOLE_REWIND_PREVIEW_MAX_LENGTH = 60

    def _console_unknown_command_hint(self, name: str) -> str:
        """Return the Enter-again hint copy for an unrecognized `/name` draft.

        Derived from the registry's own ``available_names()`` (Task 9) rather
        than a hardcoded list, so a newly-registered command (e.g. `/skills`)
        is reflected here automatically.
        """
        available = ", ".join(
            f"/{name}" for name in self._console_command_registry.available_names()
        )
        return f"Unknown command /{name} — available: {available}. Press Enter again to send as text."

    async def _dispatch_console_command(self, parse: CommandParse) -> None:
        """Dispatch a parsed Console slash command to its handler.

        A ``handler_id`` that resolves to nothing (an unrecognized command
        name) is consumed silently: nothing is sent and the draft is left
        untouched.
        """
        handler_id = self._CONSOLE_COMMAND_NAME_TO_HANDLER_ID.get(parse.name)
        # F5 (task-9 review): the composer-menu entry that BUILDS a
        # /generate-image draft was gated, but typing (or pasting) the
        # command itself was not -- this is the actual choke point every
        # path to running it passes through, so gating here closes all of
        # them at once rather than chasing each way a draft gets composed.
        if handler_id == GENERATE_IMAGE_COMMAND_HANDLER_ID:
            image_blocked = blocked_reason(
                GENERATE_IMAGE_COMMAND_HANDLER_ID,
                ephemeral=self._console_active_session_is_ephemeral(),
            )
            if image_blocked is not None:
                await self._append_native_console_system_message(image_blocked)
                return
        # task-3401.5: the video twin of the image gate above -- typing or
        # pasting /generate-video hits the same choke point, so a temporary
        # chat cannot reach the disk-writing sink through the command either.
        if handler_id == GENERATE_VIDEO_COMMAND_HANDLER_ID:
            video_blocked = blocked_reason(
                GENERATE_VIDEO_COMMAND_HANDLER_ID,
                ephemeral=self._console_active_session_is_ephemeral(),
            )
            if video_blocked is not None:
                await self._append_native_console_system_message(video_blocked)
                return
        dispatch_map = {
            "insert-prompt": self._console_command_insert_prompt,
            "apply-system": self._console_command_apply_system,
            SKILLS_COMMAND_HANDLER_ID: self._console_command_skills,
            FEWER_PERMISSION_PROMPTS_COMMAND_HANDLER_ID: (
                self._console_command_fewer_permission_prompts
            ),
            PREFILL_COMMAND_HANDLER_ID: self._console_command_prefill,
            GENERATE_IMAGE_COMMAND_HANDLER_ID: self._console_command_generate_image,
            GENERATE_VIDEO_COMMAND_HANDLER_ID: self._console_command_generate_video,
            STREAM_VIDEO_COMMAND_HANDLER_ID: self._console_command_stream_video,
            REWIND_COMMAND_HANDLER_ID: self._request_rewind,
            STEER_COMMAND_HANDLER_ID: self._console_command_steer,
            REDIRECT_COMMAND_HANDLER_ID: self._console_command_redirect,
            EMERGENCY_STOP_COMMAND_HANDLER_ID: self._console_command_emergency_stop,
            RESEARCH_COMMAND_HANDLER_ID: self._console_command_research,
            HELP_COMMAND_HANDLER_ID: self._console_command_help,
            DOCTOR_COMMAND_HANDLER_ID: self._console_command_doctor,
            CONSOLE_ACTION_COMMAND_HANDLER_ID: self._console_command_run_action,
        }
        handler = dispatch_map.get(handler_id)
        if handler is None:
            return
        await handler(parse)

    async def _console_command_help(self, parse: CommandParse) -> None:
        """TASK-25908: list console commands, or detail one, from the live
        registry. Output is one bounded block appended to the scrollable
        transcript; gated commands are marked with their unavailability."""
        ephemeral = self._console_active_session_is_ephemeral()

        def _availability(name: str) -> str | None:
            handler = self._CONSOLE_COMMAND_NAME_TO_HANDLER_ID.get(name)
            if handler is None:
                return None
            return blocked_reason(handler, ephemeral=ephemeral)

        from ...Chat.console_help import (  # ADR-097 boot ratchet: deferred off the boot path (loads on first use).
            build_command_detail,
            build_help_listing,
        )

        commands = self._console_command_registry.commands()
        query = (parse.args or "").strip()
        if query:
            text = build_command_detail(
                commands, _COMMAND_DESCRIPTIONS, query, availability_fn=_availability
            )
        else:
            text = build_help_listing(
                commands, _COMMAND_DESCRIPTIONS, availability_fn=_availability
            )
        await self._append_native_console_system_message(text)

    async def _console_command_doctor(self, parse: CommandParse) -> None:
        """TASK-25906: run the aggregate health checks and print the report.

        The DB integrity PRAGMA can be slow, so the checks run off the event
        loop. Network probes are opt-in via `/doctor network`.
        """
        import asyncio as _asyncio

        from ...Utils.doctor import run_doctor, format_doctor_report

        include_network = "network" in (parse.args or "").lower()
        try:
            checks = await _asyncio.to_thread(
                run_doctor, include_network=include_network
            )
            report = format_doctor_report(checks)
        except Exception as exc:  # noqa: BLE001 - a command must not crash the screen
            report = f"Doctor could not complete: {exc}"
        await self._append_native_console_system_message(report)

    async def _console_command_run_action(self, parse: CommandParse) -> None:
        """TASK-25909: dispatch a typed action command to the existing screen
        action method that already implements it. Refuses honestly when the
        action is unavailable rather than failing silently (AC#4)."""
        method_name = self._CONSOLE_ACTION_COMMAND_TARGETS.get(parse.name)
        method = self._resolve_action(method_name) if method_name else None
        if method is None:
            await self._append_native_console_system_message(
                f"/{parse.name} is not available in this context."
            )
            return
        try:
            result = method()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # noqa: BLE001 - a command must not crash the screen
            logger.opt(exception=True).warning(
                "Console action command /{} failed", parse.name
            )
            await self._append_native_console_system_message(
                f"/{parse.name} could not run: {exc}"
            )

    async def _open_console_style_picker_for_insert(self) -> None:
        """Open the image-style picker, inserting whatever style is chosen."""
        from ...Widgets.Console.console_style_picker_modal import (
            ConsoleStylePickerModal,
        )

        def _apply_picker_choice(record: Optional[Mapping[str, Any]]) -> None:
            self._focus_console_composer_if_needed(force=True)
            if record is None:
                return
            style_id = str(record.get("id") or "").strip()
            if not style_id:
                return
            self._insert_console_style_token_into_composer(style_id)

        self.push_screen(ConsoleStylePickerModal(), callback=_apply_picker_choice)

    def _insert_console_style_token_into_composer(self, style_id: str) -> bool:
        """Compose an ``@<style_id>`` token into a valid `/generate-image` draft.

        Delegates the actual composition to `insert_style_token_into_draft`
        (the pure grammar-aware helper `/generate-image`'s own parser is
        built from) rather than blindly prepending the token: a bare
        prepend would land the token BEFORE the command word on an
        unedited draft like ``/generate-image a dragon``, producing
        ``@style_anime /generate-image a dragon`` -- text `parse_generate_
        image_args` never sees as a command at all (`ConsoleCommandRegistry
        .parse` only recognizes drafts that START with `/`), so the whole
        thing would ship to the LLM as plain chat text instead of
        generating anything.

        The whole draft is replaced wholesale with the composed result --
        same clear-then-paste idiom `_insert_prompt_text_into_composer`
        uses for `replace=True` -- since the composed draft may reorder or
        drop text relative to the original (e.g. replacing an existing
        leading `@style` token), so a plain in-place insert cannot express
        it.

        Args:
            style_id: The resolved style template's `id` (e.g. "style_anime").

        Returns:
            ``True`` when the composer widget was found and the insert
            applied, ``False`` when no native composer is mounted.
        """
        try:
            composer = self._query_composer()
        except QueryError:
            return False
        new_draft = insert_style_token_into_draft(composer.draft_text(), style_id)
        composer.clear_draft()
        composer.insert_text_as_paste(new_draft)
        return True

    def _insert_prompt_text_into_composer(self, text: str, *, replace: bool) -> bool:
        """Insert resolved prompt text into the Console composer via paste semantics.

        Args:
            text: The prompt's ``user_prompt`` body to insert.
            replace: ``True`` replaces the whole draft wholesale (the
                `/prompt` command's own draft IS the command being replaced
                by its result). ``False`` appends onto whatever draft
                already exists (Library's "Use in Console" handoff) -- an
                already-empty draft still gets a clean insert with no
                separator, but existing draft text is never clobbered.

        Returns:
            ``True`` when the composer widget was found and the insert
            applied, ``False`` when no native composer is mounted.
        """
        try:
            composer = self._query_composer()
        except QueryError:
            return False
        if replace:
            composer.clear_draft()
            composer.insert_text_as_paste(text)
        elif composer.draft_text():
            # Appending onto an existing draft must never mash the two
            # payloads together with no boundary between them. The composer
            # caret is editable now, so seek the end first to keep this an
            # append rather than a mid-draft splice. TASK-1281 review N1:
            # the separator and the body are inserted as ONE
            # `insert_text_as_paste` call (rather than a separate
            # `insert_text("\n")` followed by the paste) so they also
            # record as a single undo entry -- previously one Ctrl+Z only
            # removed the pasted body and left a stray blank line behind.
            #
            # Review NEW-4 (known, deliberately unfixed): when `text` is
            # long enough to collapse, the leading "\n" lands INSIDE that
            # one collapsed segment, so the collapsed token itself carries
            # no visible boundary marker between "existing draft" and the
            # pasted body (canonical text is still correct either way).
            # Splitting this back into two calls to restore a literal,
            # always-visible separator would reintroduce the exact N1 bug
            # this comment describes (two undo entries, a stray blank line
            # surviving one Ctrl+Z) -- not a trivial fix, so left as a
            # documented display-only limitation rather than reverted.
            composer.move_cursor_end()
            composer.insert_text_as_paste(f"\n{text}")
        else:
            composer.insert_text_as_paste(text)
        self._sync_console_command_popup()
        return True

    async def _console_command_prefill(self, parse: CommandParse) -> None:
        """Set, pin, clear, or report the Console response prefill (`/prefill`).

        One-shot (`/prefill <text>`) applies to the next normal send only
        and wins over pinned; `/prefill pin <text>` applies to every
        submit/retry/regenerate until cleared and write-throughs to
        conversation metadata when the session is persisted. Errors leave
        the draft in place for correction (mirrors `/system`'s
        no-system-part behavior); handled outcomes clear it.
        """
        action = parse_prefill_args(parse.args)
        store = self._ensure_console_chat_store()
        self._active_session_settings()
        session = store.ensure_session()
        if session.settings is None:
            # `ensure_session` only applies `settings=` when it CREATES the
            # session; one created earlier without settings (e.g. by a bare
            # system-message append) would make the pinned-prefill update a
            # silent no-op in `set_session_pinned_prefill` (PR #729 Qodo
            # finding 3), so seed defaults before any pin/clear below.
            session = store.replace_session_settings(
                session.id, self._default_session_settings()
            )
        if action.kind == ACTION_ERROR:
            await self._append_native_console_system_message(action.error)
            return
        if action.kind == ACTION_STATUS:
            one_shot = store.session_one_shot_prefill(session.id)
            settings = store.session_settings(session.id)
            pinned = getattr(settings, "pinned_prefill", None) if settings else None
            lines = []
            if one_shot:
                lines.append(
                    f"Prefill (next send only): '{describe_prefill_preview(one_shot)}'"
                )
            if pinned:
                lines.append(f"Prefill (pinned): '{describe_prefill_preview(pinned)}'")
            if not lines:
                lines.append("No prefill armed.")
            # Clear before the message-append await: `_append_native_console_
            # system_message` triggers nested syncs (transcript render among
            # them) that make the confirmation visible to a polling caller
            # before this coroutine resumes past the `await` -- clearing
            # first closes that window instead of racing it.
            self._clear_composer()
            await self._append_native_console_system_message("\n".join(lines))
            return
        if action.kind == ACTION_CLEAR:
            store.set_session_one_shot_prefill(session.id, None)
            _session, persisted = store.set_session_pinned_prefill(session.id, None)
            self._sync_console_chat_core_state()
            self._sync_console_settings_summary()
            copy = "Prefill cleared."
            if not persisted:
                copy += " (Warning: saved conversation not updated.)"
            self._clear_composer()
            await self._append_native_console_system_message(copy)
            return
        # Deliberately NOT markup-escaped: native Console transcript rows
        # render as literal Content (never parsed as Rich markup), so an
        # escape here would surface as a visible backslash in previews
        # containing '[' — matching every neighboring system-row handler.
        preview = describe_prefill_preview(action.text)
        if action.kind == ACTION_PIN:
            _session, persisted = store.set_session_pinned_prefill(
                session.id, action.text
            )
            self._sync_console_chat_core_state()
            self._sync_console_settings_summary()
            copy = (
                f"Prefill pinned: '{preview}'. Applies to every send, retry, and "
                "regenerate until /prefill clear. The reply continues directly "
                "from the last character; tool calling is skipped on prefilled sends."
            )
            if not persisted:
                copy += " (Warning: saved conversation not updated.)"
            self._clear_composer()
            await self._append_native_console_system_message(copy)
            return
        if action.kind == ACTION_ONE_SHOT:
            store.set_session_one_shot_prefill(session.id, action.text)
            self._sync_console_chat_core_state()
            self._sync_console_settings_summary()
            self._clear_composer()
            await self._append_native_console_system_message(
                f"Prefill armed for next send: '{preview}'. The reply continues "
                "directly from the last character; tool calling is skipped on "
                "prefilled sends."
            )
        return

    async def _console_command_research(self, parse: CommandParse) -> None:
        """``/research <question>``: launch a local deep-research run whose
        completed report is delivered back into THIS conversation (task-16481).

        The run executes in a worker; the handoff inserts an assistant
        message on completion, and the existing terminal-run notification
        remains the fallback when insertion is impossible.
        """
        from tldw_chatbook.UI.Console_Modules.research_command import (
            parse_research_command,
        )

        try:
            intent = parse_research_command(parse.args or "")
        except ValueError as usage_error:
            await self._append_native_console_system_message(
                f"/research: {usage_error}"
            )
            return
        question = intent.question
        source_policy = intent.source_policy
        provider_overrides = intent.provider_overrides()
        conversation_id = self._current_console_conversation_id()
        if not conversation_id:
            await self._append_native_console_system_message(
                "Deep research needs an active conversation to deliver its report into."
            )
            return
        local_service = self._local_research_service()
        if local_service is None:
            await self._append_native_console_system_message(
                "Local research service is unavailable; cannot start a run."
            )
            return

        from tldw_chatbook.Research_Interop.chat_handoff import (
            insert_research_completion_message,
        )
        from tldw_chatbook.Research_Interop.local_research_engine import (
            LocalResearchEngine,
        )

        search_params: dict = {}
        paper_search_fn = None
        try:
            from tldw_chatbook.Tools.web_tool_impls import deep_search_pipeline_params

            search_params = deep_search_pipeline_params()
            if self._academic_research_enabled():
                from tldw_chatbook.Research_Interop.academic_providers import (
                    search_papers,
                )

                paper_search_fn = search_papers
        except Exception:
            pass

        db = self._research_database()

        async def _run_research() -> None:
            launch_kwargs: dict = {
                "query": question,
                "chat_handoff": {
                    "conversation_id": conversation_id,
                    "origin": "console",
                },
                "source_policy": source_policy,
            }
            if provider_overrides:
                launch_kwargs["provider_overrides"] = provider_overrides
            engine = LocalResearchEngine(
                local_service,
                search_params=search_params,
                paper_search_fn=paper_search_fn,
                completion_handoff=(
                    (lambda payload: insert_research_completion_message(db, payload))
                    if db is not None
                    else None
                ),
            )
            try:
                # TASK-21105 review fix: launch_run is the store's FIRST USE
                # now that the research DB opens lazily, so a corrupt or
                # unreadable database raises HERE rather than at app boot
                # (where construction failure used to null the service and
                # trip the unavailable-guard above). It must sit inside the
                # worker's guard: this worker runs with Textual's default
                # exit_on_error=True, so an unhandled raise tears down the
                # whole app.
                run = local_service.launch_run(**launch_kwargs)
                await engine.execute_run(run["id"])
            except Exception as exc:  # noqa: BLE001 - worker must not crash the screen
                logger.warning(f"Console research run failed: {exc}")

        self.run_worker(
            _run_research(),
            group="console-research",
            exclusive=False,
            description=f"Console research: {question[:60]}",
        )
        policy_note = (
            f" [policy: {source_policy}]" if source_policy != "balanced" else ""
        )
        await self._append_native_console_system_message(
            f"Deep research started: {question}{policy_note}\n"
            "The report will be added to this conversation when the run "
            "completes."
        )

    def _console_rewind_prompt_rows(
        self, session_id: str
    ) -> tuple[RewindPromptRow, ...]:
        """Build newest-first `/rewind` menu rows for a session's USER turns.

        Args:
            session_id: Native Console session id whose active-path USER
                messages become rows.

        Returns:
            One `RewindPromptRow` per USER message in `messages_for_session`
            (which walks the active path; a message not on the active path
            never appears here), ordered newest first. `index_label` is the
            USER turn's 1-based chronological position ("#1", "#2", ...);
            `preview` is a collapsed, truncated single-line preview of its
            content.
        """
        from ...Widgets.Console.console_rewind_modal import RewindPromptRow

        store = self._ensure_console_chat_store()
        user_messages = [
            message
            for message in store.messages_for_session(session_id)
            if message.role is ConsoleMessageRole.USER
        ]
        rows = [
            RewindPromptRow(
                message_id=message.id,
                index_label=f"#{position}",
                preview=derive_console_session_title(
                    message.content,
                    max_length=self._CONSOLE_REWIND_PREVIEW_MAX_LENGTH,
                )
                or "(empty prompt)",
            )
            for position, message in enumerate(user_messages, start=1)
        ]
        rows.reverse()
        return tuple(rows)

    async def _console_command_steer(self, parse: CommandParse) -> bool:
        """`/steer <text>`: deliver guidance into the ACTIVE running turn.

        TASK-25903. Plain submission still queues for the next turn; this is
        the explicit per-message opt-in. A refusal (no active run, finished
        run, empty or over-cap text) surfaces as a notify rather than being
        silently dropped -- the AC#5 contract, end to end.
        """
        text = (parse.args or "").strip()
        controller = self._ensure_console_chat_controller()
        refusal = controller.steer_active_run(text)
        if refusal is not None:
            self.app_instance.notify(f"Not steered: {refusal}", severity="warning")
            return False
        # Review I-3: dispatch restores the stash before the handler, so
        # without this the full `/steer <text>` stays in the composer and one
        # habitual extra Enter delivers the SAME guidance twice into the live
        # run. Siblings (/prefill, /fewer-permission-prompts) clear too.
        self._clear_composer()
        self.app_instance.notify("Steered into the running turn.")
        return True

    async def _console_command_redirect(self, parse: CommandParse) -> bool:
        """`/redirect <text>`: cut off the current response, keep completed
        tool results and the streamed partial, re-run the turn with the
        correction as a plain user message.

        TASK-28227. /steer lets the current response finish; this is for
        when it is already wrong. Stop is untouched and remains terminal.
        Refusals surface as a notify, never a silent drop.
        """
        text = (parse.args or "").strip()
        controller = self._ensure_console_chat_controller()
        refusal = controller.redirect_active_run(text)
        if refusal is not None:
            self.app_instance.notify(f"Not redirected: {refusal}", severity="warning")
            return False
        # Same double-delivery guard as /steer (review I-3): dispatch
        # restores the stash, so the command would sit in the composer and
        # one habitual extra Enter would redirect the run twice.
        self._clear_composer()
        self.app_instance.notify("Redirect sent — correcting the running turn.")
        return True

    async def _console_command_emergency_stop(self, parse: CommandParse) -> bool:
        """`/emergency-stop [clear|<reason>]`: the global stop (TASK-26004).

        Holds all NEW agent runs and scheduled dispatches; in-flight work is
        untouched, and the state survives a restart. `clear`/`off`/`resume`
        lifts it. A stopped state is reported plainly on the next send/dispatch.
        """
        from tldw_chatbook.emergency_stop import (
            clear_emergency_stop,
            default_emergency_stop_path,
            set_emergency_stop,
        )

        arg = (parse.args or "").strip()
        path = default_emergency_stop_path()
        if arg.lower() in {"clear", "off", "resume"}:
            clear_emergency_stop(path)
            self._clear_composer()
            self.app_instance.notify("Emergency stop cleared — new work may start.")
            return True
        set_emergency_stop(path, reason=arg)
        self._clear_composer()
        self.app_instance.notify(
            "Emergency stop ACTIVE — new runs and scheduled dispatches are held. "
            "Run /emergency-stop clear to resume.",
            severity="warning",
        )
        return True

    async def _console_command_rewind(self, parse: CommandParse) -> bool:
        """Open the `/rewind` menu over the active session's prior USER prompts.

        Collects the active path's USER-turn rows (newest first) and pushes
        `ConsoleRewindModal`; a session with no USER turns yet (or no active
        session at all) is a no-op notify rather than an empty modal.

        Returns:
            True when the modal opened; False when no USER prompt rows exist.
        """
        from ...Widgets.Console.console_rewind_modal import ConsoleRewindModal

        store = self._ensure_console_chat_store()
        session_id = store.active_session_id
        rows = self._console_rewind_prompt_rows(session_id) if session_id else ()
        if not rows:
            self.app_instance.notify("Nothing to rewind.", severity="warning")
            return False

        controller = self._ensure_console_chat_controller()
        summary_disabled_reason = self._console_rewind_summary_disabled_reason(
            controller, session_id
        )
        try:
            active_path_identity = tuple(store.active_path_message_ids(session_id))
        except KeyError:
            self.app_instance.notify("Nothing to rewind.", severity="warning")
            return False
        has_effective_memory = False
        try:
            _local, _global, effective_memory = controller.context_control_inputs(
                session_id
            )
            has_effective_memory = effective_memory.kind is not EffectiveMemoryKind.RAW
        except Exception:
            # Keep the modal available, but fail closed for disclosure: the
            # lookup may have failed while replacement memory exists.
            has_effective_memory = True
            logger.warning(
                "Console rewind effective-memory lookup failed; "
                "showing conservative replacement warning."
            )

        async def _apply_choice(choice: "ConsoleRewindChoice | None") -> None:
            await self._apply_rewind_choice(
                session_id,
                choice,
                active_path_identity=active_path_identity,
            )

        self.push_screen(
            ConsoleRewindModal(
                prompts=rows,
                has_effective_memory=has_effective_memory,
                summary_disabled_reason=summary_disabled_reason,
            ),
            callback=_apply_choice,
        )
        return True

    async def _apply_console_rewind_choice(
        self,
        session_id: str,
        choice: "ConsoleRewindChoice | None",
        *,
        active_path_identity: tuple[str, ...] | None = None,
    ) -> None:
        """Apply a `/rewind` modal result.

        `None` (Escape / "Never mind") just returns focus to the composer.
        `"restore"` is pure tree navigation: gated on
        `controller.send_refusal_copy(...)` (mirrors regenerate/resend --
        never mutates while a run is streaming; returns non-empty refusal
        copy exactly when a send would currently be blocked, parallel-
        agents spec §4), the new active leaf is the selected prompt's
        PARENT found by an id lookup in `active_path_message_ids` (never
        positional -- display-only TOOL rows can pad
        `messages_for_session`'s view without being tree nodes), with
        `None` (empty transcript) when the selected prompt was the root.
        The selected prompt's own text is written back into the composer
        via the same paste-semantics seam `/prompt` uses.
        Both summary kinds run their controller flow on the same exclusive
        `console-run-{session_id}` worker group, gated on `send_refusal_copy`
        the same way restore is (never mutates while a run is active).

        A `ModalScreen` blocks session switching while the rewind modal is up,
        so today this is theoretical -- but the callback still re-checks the
        store's active session against the one captured when the modal opened
        (`session_id`) before doing anything, and no-ops with a notify on a
        mismatch. This keeps the flow robust against future modal/timing
        changes (e.g. a background auto-switch or a non-modal rewind surface)
        that could let the active session change out from under a pending
        choice.

        Args:
            session_id: Native Console session id the modal was opened for.
            choice: The modal's result, or `None`.
            active_path_identity: Exact ordered path captured for the modal.
        """
        from ...Widgets.Console.console_rewind_modal import (
            KIND_RESTORE,
            KIND_SUMMARIZE_FROM,
            KIND_SUMMARIZE_UP_TO,
        )

        store = self._ensure_console_chat_store()
        try:
            if store.active_session_id != session_id:
                self.app_instance.notify(
                    "Console session changed — rewind cancelled.",
                    severity="warning",
                )
                return
            if choice is None:
                return
            if choice.kind in {KIND_SUMMARIZE_UP_TO, KIND_SUMMARIZE_FROM}:
                try:
                    current_path_identity = tuple(
                        store.active_path_message_ids(session_id)
                    )
                except KeyError:
                    current_path_identity = ()
                if (
                    active_path_identity is not None
                    and current_path_identity != active_path_identity
                ):
                    self.app_instance.notify(
                        "Conversation changed before summarization could start.",
                        severity="warning",
                    )
                    return
                captured_path_identity = (
                    active_path_identity
                    if active_path_identity is not None
                    else current_path_identity
                )
                controller = self._ensure_console_chat_controller()
                # Gate BEFORE spawning: an exclusive console-run worker cancels
                # any in-flight run at creation time, before the controller's
                # own rejection can run -- refuse first, like regenerate.
                target_session_id = controller.store.active_session_id or ""
                refusal = controller.send_refusal_copy(target_session_id)
                if refusal:
                    self.app_instance.notify(refusal, severity="warning")
                    return
                target_is_current = choice.message_id in captured_path_identity
                if not target_is_current:
                    self.app_instance.notify(
                        "Console message action target no longer exists.",
                        severity="error",
                    )
                    return
                worker = (
                    self._summarize_console_from
                    if choice.kind == KIND_SUMMARIZE_FROM
                    else self._summarize_console_up_to
                )
                self.run_worker(
                    worker(
                        controller,
                        session_id,
                        choice.message_id,
                        captured_path_identity,
                    ),
                    exclusive=True,
                    group=f"console-run-{target_session_id}",
                )
                return
            if choice.kind != KIND_RESTORE:
                return
            controller = self._ensure_console_chat_controller()
            refusal = controller.send_refusal_copy(controller.store.active_session_id)
            if refusal:
                self.app_instance.notify(refusal, severity="warning")
                return
            try:
                path = store.active_path_message_ids(session_id)
                index = path.index(choice.message_id)
            except (KeyError, ValueError):
                self.app_instance.notify(
                    "Console message action target no longer exists.",
                    severity="error",
                )
                return
            self._apply_rewind_position(session_id, choice.message_id, path, index)
            # The lookup above proves `choice.message_id` is a live message, so
            # this can't raise -- fetch the FULL text rather than reusing
            # `choice.prompt_text`, which is only the modal row's truncated
            # display preview (see `ConsoleRewindChoice`/`RewindPromptRow`).
            full_text = store.get_message(choice.message_id).content
            self._insert_prompt_text(full_text, replace=True)
            await self._sync_native_console_chat_ui()
        finally:
            self._focus_console_composer_if_needed(force=True)

    def _clear_console_composer_draft(self) -> None:
        """Clear the native Console composer's draft text, if mounted.

        Shared by any handled-command success path that applies a side
        effect (rather than inserting replacement text) but must still not
        leave its own invocation text sitting in the composer afterward --
        e.g. a successful named `/system <name>` apply. `/prompt`'s
        equivalent success path instead replaces the draft with the
        resolved prompt body via ``_insert_prompt_text_into_composer``,
        which already clears via the same ``clear_draft()`` seam.
        """
        composer = self._console_composer_or_none()
        if composer is not None:
            composer.clear_draft()
            self._sync_console_command_popup()

    async def _console_command_fewer_permission_prompts(
        self, parse: CommandParse
    ) -> None:
        """Render local MCP prompt-reduction recommendations."""
        del parse
        self._clear_composer()
        service = getattr(self.app_instance, "unified_mcp_service", None)
        loader = getattr(service, "permission_prompt_recommendations", None)
        if not callable(loader):
            await self._append_native_console_system_message(
                "MCP prompt recommendations unavailable - MCP service is not ready."
            )
            return
        try:
            report = await loader()
        except Exception as exc:  # noqa: BLE001 -- render recovery, never send command
            logger.warning(
                "MCP prompt recommendations command failed (exception_type={})",
                type(exc).__name__,
            )
            await self._append_native_console_system_message(
                "MCP prompt recommendations unavailable - local analysis failed."
            )
            return
        await self._append_native_console_system_message(
            format_permission_prompt_report(report)
        )

    @staticmethod
    def _console_rewind_summary_disabled_reason(
        controller: ConsoleChatController,
        session_id: str,
    ) -> str:
        """Return only synchronously known run/tip refusal guidance."""
        if not controller.run_state_for(session_id).is_send_allowed:
            return "A run is already running in this tab."
        try:
            path = controller.store.active_path_message_ids(session_id)
            messages = {
                message.id: message
                for message in controller.store.messages_for_session(session_id)
            }
            tip = messages.get(path[-1]) if path else None
        except KeyError:
            tip = None
        if tip is not None and (
            tip.role is ConsoleMessageRole.USER
            or (tip.role is ConsoleMessageRole.ASSISTANT and tip.status != "complete")
        ):
            return "Finish the current exchange before summarizing."
        snapshots = controller._durable_context_snapshots(session_id)
        if snapshots:
            units = complete_durable_units(snapshots)
            if not units or units[-1].boundary_message_id != snapshots[-1].message_id:
                return "Finish the current exchange before summarizing."
        return ""
