"""Current Console Context/spend display ownership (TASK34406 AC51)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from loguru import logger
from textual.css.query import NoMatches, QueryError

from ...Chat.console_chat_controller import ConsoleChatController
from ...Chat.console_chat_models import (
    FEEDBACK_ACTIVE_RUN_STATUSES,
    ConsoleChatMessage,
    ConsoleMessageRole,
    ConsoleRunStatus,
)
from ...Chat.console_chat_store import ConsoleChatStore
from ...Chat.console_context_policy import ConsoleContextPolicyOverrides
from ...Chat.console_cost_tracker import (
    ConsoleCacheState,
    ConsoleCostRow,
    ConsoleCostRowTotals,
    ConsoleCostState,
    TokenEstimateCache,
    build_cost_rows,
    build_cost_rows_totals,
)
from ...Chat.console_display_state import (
    console_prompted_evidence_text,
    estimate_console_next_send_tokens,
)
from ...Chat.console_exchange_capture import ExchangeCapture
from ...Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
    ConsoleSettingsSummaryState,
    build_console_settings_summary_state,
)
from ...Chat.console_trace_projection import ProjectedTraceCall
from ...Chat.provider_readiness import provider_config_key
from ...Widgets.Console import ConsoleSettingsSummary
from ...Widgets.Console.console_context_controls import (
    ConsoleContextControlState,
    build_console_context_control_state,
    compaction_forecast_rows,
)
from ...Widgets.Console.console_status_chips import ConsoleStatusChips
from ..Console_Modules import console_spend_projection as spend
from ..Console_Modules.left_rail import ConsoleLeftRail

if TYPE_CHECKING:
    from ...Widgets.Console.console_conversation_inspector import InspectorTurn

CONSOLE_ACTIVE_RUN_STATUSES: tuple[ConsoleRunStatus, ...] = tuple(
    sorted(FEEDBACK_ACTIVE_RUN_STATUSES, key=lambda status: status.value)
)


def _console_inspector_turn_preview(content: Any) -> str:
    """Best-effort short text preview for one Conversation Inspector
    Costs-tab turn row (task-8 review finding 5).

    ``ConsoleChatMessage.content`` is declared ``str``, but a multimodal
    (structured, OpenAI-style content-block list) message is not
    guaranteed to have been coerced to text by the time it reaches here --
    several other modules in this codebase (``Chat/Chat_Functions.py``,
    ``console_provider_gateway.py``) carry their own ``isinstance(content,
    str)`` guards for exactly this reason. Slicing a list with ``[:60]``
    would silently yield up to 60 LIST ELEMENTS, not characters -- not a
    preview, and not obviously wrong-looking in a diff either. Falls back
    to the first text block's text (bounded to 60 chars, matching the str
    path), or ``""`` when nothing text-shaped is found -- never a
    fabricated summary.
    """
    if isinstance(content, str):
        return content[:60]
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    return text[:60]
    return ""


def _build_console_inspector_exchanges_loader(
    messages_by_native_id: Mapping[str, Any],
    projected_calls_reader: Callable[[str], Sequence[ProjectedTraceCall]],
    abandoned_run_tags_for: Callable[[str], AbstractSet[str]] | None = None,
) -> Callable[[str], Awaitable[list[tuple[ExchangeCapture, bool]]]]:
    """Build the Costs-tab ``exchanges_loader`` for
    ``ConsoleConversationInspector`` (task-8, extended task-9).

    A standalone function rather than a method-local closure specifically
    so it is unit-testable without mounting a ``ChatScreen`` (review
    finding 6) -- pure extraction, no behavior change from the closure
    this replaced in ``ChatScreen._build_console_inspector_cost_data``.

    Args:
        messages_by_native_id: ``ConsoleChatMessage.id`` -> the matching
            in-memory message, for the native-first check.
        projected_calls_reader: Store-owned persisted-message reader returning
            discriminated normalized/legacy calls. Called lazily only on the
            durable fallback path, so this Textual helper never receives a DB
            handle and an ephemeral session never performs durable I/O.
        abandoned_run_tags_for: Optional ``native_message_id ->
            {run_tag, ...}`` lookup (task-9; ``ConsoleChatStore.
            abandoned_exchange_run_tags`` in production) used ONLY on the
            native-capture path to resolve each capture's real
            ``abandoned`` flag. Defaults to ``None``, which preserves the
            task-8 behavior of reporting ``abandoned=False`` for every
            native capture -- kept optional (rather than required) so the
            existing unit tests in
            ``Tests/UI/test_chat_screen_console_inspector_loader.py``,
            which construct this loader with just the first two
            positional args, are unaffected.

    Returns:
        An async ``native_message_id -> [(capture, abandoned), ...]``
        callable (see ``console_conversation_inspector``'s module
        docstring for the pair contract and the ordering caveat -- callers
        must NOT trust the returned order, only ``(created_at, seq)``).
        Prefers ``message.exchanges`` (native, in-memory captures resolve
        ``abandoned`` via ``abandoned_run_tags_for`` when supplied, else
        always ``False``) and only falls back to a threaded
        ``get_message_exchanges`` + ``capture_from_blob`` read when there
        is no native capture AND the message has a
        ``persisted_message_id`` (an ephemeral session has neither, so it
        returns ``[]`` without any durable read). Corrupt legacy isolation and
        normalized-first selection belong to the injected projection.
    """

    async def _exchanges_loader(
        native_message_id: str,
    ) -> list[tuple[ExchangeCapture, bool]]:
        message = messages_by_native_id.get(native_message_id)
        if message is not None and message.exchanges:
            # Native captures win when present -- they are fresher than
            # whatever was last flushed to the DB.
            abandoned_tags: AbstractSet[str] = (
                abandoned_run_tags_for(native_message_id)
                if abandoned_run_tags_for is not None
                else frozenset()
            )
            return [
                (capture, capture.run_tag in abandoned_tags)
                for capture in message.exchanges
            ]
        persisted_id = message.persisted_message_id if message is not None else None
        if not persisted_id:
            return []

        def _read() -> list[tuple[ExchangeCapture, bool]]:
            return [
                (
                    replace(
                        projected.capture,
                        trace_provenance=projected.provenance,
                        trace_chronology=projected.chronology,
                        trace_uncertainty=projected.uncertainty_codes,
                    ),
                    projected.abandoned,
                )
                for projected in projected_calls_reader(persisted_id)
            ]

        return await asyncio.to_thread(_read)

    return _exchanges_loader


class ConsoleContextSpendController:
    """Display-only Context/spend state; no DOM or independent authority."""

    def __init__(
        self,
        screen: Any,
        *,
        ensure_chat_store: Callable[..., Any],
        current_chat_store: Callable[..., Any],
        ensure_chat_controller: Callable[..., Any],
        current_chat_controller: Callable[..., Any],
        composer: Callable[..., Any],
        pending_launch: Callable[..., Any],
        build_staged_context_state: Callable[..., Any],
        active_settings_readiness: Callable[..., Any],
        provider_model_display: Callable[..., Any],
        active_session_settings: Callable[..., Any],
        active_native_session: Callable[..., Any],
        workspace_context: Callable[..., Any],
        fleet_token_total: Callable[..., Any],
        response_rule_usage: Callable[..., Any],
        sync_rail_system_line: Callable[..., Any],
        sync_agent_section: Callable[..., Any],
        request_context_allocation_reconcile: Callable[..., Any],
        record_timer_created: Callable[..., Any],
        record_timer_stopped: Callable[..., Any],
        push_inspector: Callable[..., Any],
        context_window_for_display: Callable[..., Any],
        context_inputs: Callable[..., Any],
        display_pricing_catalog: Callable[..., Any],
        build_cost_snapshot: Callable[..., Any],
        build_context_estimate: Callable[..., Any],
        estimate_tokens_locally: Callable[..., Any],
        next_send_factories: Callable[..., Any],
        cost_ttl_seconds: Callable[..., Any],
        estimate_ttl_seconds: Callable[..., Any],
        refresh_checked_cost_chip: Callable[..., Any],
    ) -> None:
        """Own display state; every non-state dependency is a named live callable.

        The Screen is retained only for framework queries/timers/notifications.
        Context and pricing callbacks explicitly retain that actual Screen as
        their proof owner; module aliases and sibling owners resolve at call time.
        """
        self._screen = screen
        self._ensure_console_chat_store = ensure_chat_store
        self._current_chat_store = current_chat_store
        self._ensure_console_chat_controller = ensure_chat_controller
        self._current_chat_controller = current_chat_controller
        self._console_composer_or_none = composer
        self._pending_launch = pending_launch
        self._build_console_staged_context_state = build_staged_context_state
        self._active_console_settings_readiness = active_settings_readiness
        self._active_console_provider_model_display = provider_model_display
        self._active_session_settings = active_session_settings
        self._active_native_session = active_native_session
        self._workspace_context = workspace_context
        self._fleet_token_total = fleet_token_total
        self._response_rule_usage = response_rule_usage
        self._sync_console_rail_system_line = sync_rail_system_line
        self._sync_console_agent_section = sync_agent_section
        self._request_console_context_allocation_reconcile = (
            request_context_allocation_reconcile
        )
        self._record_ui_timer_created = record_timer_created
        self._record_ui_timer_stopped = record_timer_stopped
        self._push_console_inspector = push_inspector
        self._context_window_for_display = context_window_for_display
        self._context_inputs = context_inputs
        self._display_pricing_catalog = display_pricing_catalog
        self._build_cost_snapshot = build_cost_snapshot
        self._build_context_estimate = build_context_estimate
        self._estimate_tokens_locally = estimate_tokens_locally
        self._console_inspector_next_send_factories = next_send_factories
        self._cost_ttl_seconds = cost_ttl_seconds
        self._estimate_ttl_seconds = estimate_ttl_seconds
        self._sync_console_cost_chip = refresh_checked_cost_chip
        self._console_cost_estimate_cache = None
        self._console_display_history_cache = None
        self._console_estimate_cache = None
        self._console_estimate_cache_owners = None
        self._console_settled_context_cache = None
        self._console_compaction_forecast_cache = None
        self._console_settled_spend_cache = None
        self._console_cost_projection_tokens_cache = None
        self._last_console_context_control_state = None
        self._last_console_cost_state = None
        self._console_cost_cache_state = ConsoleCacheState.NONE
        self._console_cost_fp_revisions = {}
        self._console_cost_break_reasons = {}
        self._console_cost_ttl_timer = None

    @property
    def query_one(self):
        return self._screen.query_one

    @property
    def query_one_optional(self):
        return self._screen.query_one_optional

    @property
    def set_interval(self):
        return self._screen.set_interval

    @property
    def notify(self):
        return self._screen.notify

    def _console_next_send_token_estimate(
        self, snapshot: Any, session_id: str | None = None
    ) -> int | None:
        """Estimate the tokens the snapshot's next-send payload will ship.

        task-25886: the Next Send header's count must answer "what is this
        message about to send", which on a first message is dominated by the
        system prompt, project-instruction bodies, tool schemas, and staged
        evidence -- none of which the draft-only estimate sees. Counted via
        the shared pure estimator (:func:`estimate_console_next_send_tokens`)
        over the assembled payload, plus the staged evidence text the
        preview payload carries only as label-only metadata (the same
        zero-I/O seam ``console_prompted_evidence_text`` the settings
        estimate and cost chip already read). Degrades to ``None`` on an
        assembly-error payload -- no count is better than a wrong one.

        ``session_id`` is the session the snapshot was captured for; the
        screen-global staged launch is folded in ONLY while that session is
        still the active one, mirroring
        ``_console_settings_context_estimate_for_session``'s
        ``include_active_staging`` gate -- a session switch behind an open
        inspector must not splice the new session's staged evidence into
        the old session's estimate.
        """
        payload = getattr(snapshot, "next_send_payload", None)
        if not isinstance(payload, dict) or payload.get("error"):
            return None
        include_staged = True
        if session_id is not None:
            store = self._current_chat_store()
            include_staged = store is not None and store.active_session_id == session_id
        provider, model, _settings = self._active_console_provider_model_display()
        return estimate_console_next_send_tokens(
            payload_messages=payload.get("messages") or [],
            payload_system=payload.get("system"),
            tools_info=payload.get("tools"),
            extra_texts=(console_prompted_evidence_text(self._pending_launch()),)
            if include_staged
            else (),
            model=model or "",
            provider=provider or "",
        )

    def _console_cost_estimate_cache_or_new(self) -> TokenEstimateCache:
        """Return this screen's token-estimate memo, creating it on first use.

        Held per screen rather than per session: entries are keyed by message
        id and every hit is re-verified against the row's own text, so the
        two sessions of a switched-between pair share the cache safely (see
        :class:`TokenEstimateCache`).
        """
        cache = self._console_cost_estimate_cache
        if cache is None:
            cache = TokenEstimateCache()
            self._console_cost_estimate_cache = cache
        return cache

    def _active_console_settings_context_estimate(
        self,
    ) -> ConsoleSettingsContextEstimate:
        """Return context usage for the active native Console settings snapshot."""
        from .execution_state import selected_context_summary

        projected = selected_context_summary(self._screen)
        if projected is not None:
            return projected.estimate

        store = self._ensure_console_chat_store()
        session_id = store.active_session_id
        if session_id is None:
            settings = self._active_session_settings()
            if not spend.console_tokenizer_display_ready(
                self._screen, settings.model, settings.provider
            ):
                return ConsoleSettingsContextEstimate(None, None, "Context unavailable")
            return self._build_context_estimate(
                [],
                settings.provider,
                settings.model,
                max_tokens_response=settings.max_tokens,
                system_prompt=settings.system_prompt,
            )
        return self._console_settings_context_estimate_for_session(session_id)

    def _console_display_history(
        self,
        store: ConsoleChatStore,
        session_id: str,
        controller: ConsoleChatController,
    ) -> tuple[
        tuple[object, ...],
        list[ConsoleChatMessage],
        spend.ConsoleSpendHistoryProjection,
    ]:
        """Reuse detached settled rows until store or run ownership changes."""
        recovery = store.dispatch_recovery_for_session(session_id)
        preparation = store.preparation_for_session(session_id)
        run_status = controller.run_state_for(session_id).status
        has_submit_task = bool(controller._submit_tasks_for_session(session_id))
        key = (
            id(store),
            session_id,
            store.display_projection_revision(session_id),
            id(recovery),
            id(preparation),
            run_status,
            has_submit_task,
        )
        cached = getattr(self, "_console_display_history_cache", None)
        if cached is not None and cached[0] == key:
            return key, cached[2], cached[3]
        messages = store.messages_for_session(session_id)
        history = spend.build_console_spend_history_projection(
            messages, recovery, preparation, run_status, has_submit_task
        )
        # Retain the owner objects so their IDs cannot be reused while this
        # screen still holds the corresponding projection.
        self._console_display_history_cache = (
            key,
            store,
            messages,
            history,
            recovery,
            preparation,
        )
        return key, messages, history

    def _console_settings_context_estimate_for_session(
        self,
        session_id: str,
        *,
        settings: ConsoleSessionSettings | None = None,
    ) -> ConsoleSettingsContextEstimate:
        """Return settings context derived from one captured session only."""
        store = self._ensure_console_chat_store()
        settings = settings or store.session_settings(session_id)
        if settings is None:
            raise KeyError(session_id)
        if not spend.console_tokenizer_display_ready(
            self._screen, settings.model, settings.provider
        ):
            return ConsoleSettingsContextEstimate(None, None, "Context unavailable")
        include_active_staging = store.active_session_id == session_id
        controller = self._ensure_console_chat_controller()
        run_status = controller.run_state_for(session_id).status
        recovery = store.dispatch_recovery_for_session(session_id)
        preparation = store.preparation_for_session(session_id)
        composer = self._console_composer_or_none() if include_active_staging else None
        draft_text = composer.draft_text() if composer is not None else ""
        pending_owner = self._pending_launch() if include_active_staging else None
        # TASK-33081: keep the one-second streaming bound, but check its key
        # before any transcript snapshots. Ordinary edits invalidate through
        # payload/display revisions; growing stream text waits at most one second.
        estimate_cache_key = (
            store,
            session_id,
            store.message_count(session_id),
            store.payload_revision(session_id),
            None
            if run_status in CONSOLE_ACTIVE_RUN_STATUSES
            else store.display_projection_revision(session_id),
            store.session_settings_revision(session_id),
            settings.provider,
            settings.model,
            settings.max_tokens,
            settings.system_prompt,
            draft_text,
            id(pending_owner),
            id(recovery),
            id(preparation),
            run_status,
            bool(controller._submit_tasks_for_session(session_id)),
        )
        now = time.monotonic()
        cached_estimate = getattr(self, "_console_estimate_cache", None)
        if (
            cached_estimate is not None
            and cached_estimate[0] == estimate_cache_key
            and now - cached_estimate[1] < self._estimate_ttl_seconds()
        ):
            return cached_estimate[2]

        def remember(
            estimate: ConsoleSettingsContextEstimate,
        ) -> ConsoleSettingsContextEstimate:
            self._console_estimate_cache = (estimate_cache_key, now, estimate)
            # Keep immutable owners alive while their IDs are in the cache key.
            self._console_estimate_cache_owners = (pending_owner, recovery, preparation)
            return estimate

        workspace_context = (
            self._workspace_context() if include_active_staging else None
        )
        pending_launch = self._pending_launch() if include_active_staging else None
        staged_context_state = self._build_console_staged_context_state(pending_launch)
        greeting = ""
        composer = self._console_composer_or_none() if include_active_staging else None
        if include_active_staging:
            controller = self._ensure_console_chat_controller()
            history_key, session_messages, history = self._console_display_history(
                store, session_id, controller
            )
            staged_text = console_prompted_evidence_text(pending_launch)
            context_window = self._context_window_for_display(settings)
            cache_key = (
                history_key,
                settings.provider,
                settings.model,
                settings.max_tokens,
                settings.system_prompt,
                len(workspace_context.staged_sources)
                if workspace_context is not None
                else 0,
                staged_context_state.summary,
                staged_text,
                context_window,
            )
            cached_context = getattr(self, "_console_settled_context_cache", None)
            if cached_context is None or cached_context[0] != cache_key:
                greeting = controller._seeded_greeting_text(
                    session_id, session_messages
                )
                settled_messages = spend.build_console_context_messages(
                    session_messages, history.request_ids, ""
                )
                settled_estimate = self._build_context_estimate(
                    settled_messages,
                    settings.provider,
                    settings.model,
                    staged_source_count=cache_key[5],
                    staged_context_summary=staged_context_state.summary,
                    max_tokens_response=settings.max_tokens,
                    system_prompt=spend.fold_system_prompt(
                        settings.system_prompt, greeting
                    ),
                    staged_text=staged_text,
                    context_window=context_window,
                )
                self._console_settled_context_cache = (cache_key, settled_estimate)
            else:
                settled_estimate = cached_context[1]
            draft = composer.draft_text() if composer is not None else ""
            if not draft.strip() or settled_estimate.used_tokens is None:
                return remember(settled_estimate)
            estimate = self._build_context_estimate(
                [{"role": "user", "content": draft}],
                settings.provider,
                settings.model,
                staged_source_count=cache_key[5],
                staged_context_summary=staged_context_state.summary,
                max_tokens_response=settings.max_tokens,
                context_window=context_window,
                history_used_tokens=settled_estimate.used_tokens,
            )
            return remember(estimate)
        else:
            try:
                session_messages = store.messages_for_session(session_id)
            except KeyError:
                session_messages = []
            messages = spend.build_console_context_messages(session_messages, None, "")
        estimate = self._build_context_estimate(
            messages,
            settings.provider,
            settings.model,
            staged_source_count=(
                len(workspace_context.staged_sources)
                if workspace_context is not None
                else 0
            ),
            staged_context_summary=staged_context_state.summary,
            max_tokens_response=settings.max_tokens,
            system_prompt=spend.fold_system_prompt(settings.system_prompt, greeting),
            # task-6: staged evidence used to move only the label's "; N
            # sources staged" suffix (`staged_source_count` above) while
            # `used_tokens` silently reported zero for content the send
            # is likely to carry. `console_prompted_evidence_text` reads
            # the same in-memory, zero-I/O staged bundle and produces the
            # formatted pre-authority estimate
            # `_current_console_workspace_context` already parses above --
            # no extra DB round trip. The actual send may shrink this after
            # its authority check.
            staged_text=console_prompted_evidence_text(pending_launch),
            context_window=self._context_window_for_display(settings),
        )
        return remember(estimate)

    def _active_console_context_control_state(
        self,
        *,
        estimate: ConsoleSettingsContextEstimate | None = None,
        thinking_history_effective_policy: str | None = None,
    ) -> ConsoleContextControlState:
        """Build the shared quick/full context snapshot for the active session."""
        store = self._ensure_console_chat_store()
        session_id = store.active_session_id
        if session_id is None:
            settings = self._active_session_settings()
            estimate = estimate or self._active_console_settings_context_estimate()
            return build_console_context_control_state(
                settings=settings,
                estimate=estimate,
                overrides=ConsoleContextPolicyOverrides(),
                global_overrides=None,
                active_memory=None,
            )
        return self._console_context_control_state_for_session(
            session_id,
            estimate=estimate,
            thinking_history_effective_policy=thinking_history_effective_policy,
        )

    def _console_context_control_state_for_session(
        self,
        session_id: str,
        *,
        estimate: ConsoleSettingsContextEstimate | None = None,
        settings: ConsoleSessionSettings | None = None,
        thinking_history_effective_policy: str | None = None,
    ) -> ConsoleContextControlState:
        """Build context controls from one captured session binding."""
        store = self._ensure_console_chat_store()
        settings = settings or store.session_settings(session_id)
        if settings is None:
            raise KeyError(session_id)
        estimate = estimate or self._console_settings_context_estimate_for_session(
            session_id,
            settings=settings,
        )
        overrides = ConsoleContextPolicyOverrides()
        global_overrides = None
        memory = None
        controller = self._ensure_console_chat_controller()
        try:
            overrides, global_overrides, memory = self._context_inputs(
                controller, session_id
            )
        except (KeyError, ValueError):
            pass
        accounting = None
        try:
            accounting = controller.context_breakdown_accounting(session_id)
        except Exception:  # noqa: BLE001 - Preserve optional accounting display fallback.
            accounting = None
        forecast_tokens, forecast_units = self._console_compaction_forecast_inputs(
            session_id, settings, overrides, global_overrides, memory
        )
        mandatory = self._console_compaction_mandatory_estimate(
            session_id,
            settings,
            estimate,
            controller=controller,
        )
        caps = spend.model_token_caps_for_display(self._screen, settings)
        unavailable = ""
        estimated_history = forecast_tokens
        if forecast_units and (mandatory is None or caps is None):
            unavailable = "Current mandatory input or tool overhead is not measured."
            forecast_tokens = None
        state = build_console_context_control_state(
            settings=settings,
            estimate=estimate,
            overrides=overrides,
            global_overrides=global_overrides,
            active_memory=memory,
            busy=memory is None,
            status_message="Loading context…" if memory is None else "",
            accounting=accounting,
            thinking_history_policy=(
                store.session_thinking_history_policy(session_id)
                if session_id is not None
                else "auto"
            ),
            thinking_history_effective_policy=thinking_history_effective_policy,
            forecast_tokens=forecast_tokens,
            forecast_units=forecast_units,
            request_overhead_tokens=mandatory,
            provider_input_cap_tokens=caps[0] if caps is not None else None,
            provider_output_cap_tokens=caps[1] if caps is not None else None,
            forecast_unavailable_reason=unavailable,
        )
        trigger = state.compaction_trigger_tokens
        if (
            unavailable
            and estimated_history is not None
            and trigger is not None
            and estimated_history >= trigger
        ):
            # An advisory about the displayed estimate, never an ASK verdict:
            # unknown mandatory material may make compaction impossible.
            state = replace(
                state,
                forecast_high_context=True,
                forecast_unavailable_reason=(
                    "Estimated history reaches the displayed compaction trigger; "
                    "next-send check unavailable."
                ),
            )
        return state

    def _console_compaction_mandatory_estimate(
        self,
        session_id: str,
        settings: ConsoleSessionSettings,
        estimate: ConsoleSettingsContextEstimate,
        *,
        controller: ConsoleChatController,
    ) -> int | None:
        """Separate current static input from history; unknown tools stay unknown.

        Count only the current prompt, greeting, draft and staged evidence,
        with the existing display estimator's single framing base. Historical
        text is not re-tokenized here. Last-request accounting cannot qualify
        an agent's current tool surface.
        This never runs hooks, permission readers, or request previews.
        """
        if estimate.used_tokens is None or not spend.console_tokenizer_display_ready(
            self._screen, settings.model, settings.provider
        ):
            return None
        if (
            controller._agent_runtime_enabled and controller._agent_bridge is not None
        ) or controller._resolve_submit_prefill(session_id)[0]:
            return None
        store = self._ensure_console_chat_store()
        if store.pending_attachments(session_id):
            return None
        _key, messages, _history = self._console_display_history(
            store, session_id, controller
        )
        if any(message.attachments for message in messages):
            return None
        active = store.active_session_id == session_id
        composer = self._console_composer_or_none() if active else None
        draft = composer.draft_text() if composer is not None else ""
        mandatory = self._build_context_estimate(
            [{"role": "user", "content": draft}] if draft.strip() else [],
            settings.provider,
            settings.model,
            system_prompt=spend.fold_system_prompt(
                settings.system_prompt,
                controller._seeded_greeting_text(session_id, messages),
            ),
            staged_text=console_prompted_evidence_text(self._pending_launch())
            if active
            else "",
        )
        return mandatory.used_tokens

    def _console_compaction_forecast_inputs(
        self,
        session_id: str,
        settings: ConsoleSessionSettings,
        overrides: Any,
        global_overrides: Any,
        memory: Any,
    ) -> tuple[int | None, int]:
        """The next send's conversation tokens and turns, under Ask only.

        TASK-33621.4 AC#3. Counted from the rows ``compaction_forecast_rows``
        selects (memory text plus turns after its boundary, no system prompt,
        sources or draft), once per transcript/memory/model change. Nothing
        is measured until the memory read has warmed (``memory`` None).
        """
        from ...Chat.console_context_compaction import EffectiveMemoryResult
        from ...Chat.console_context_policy import (
            ContextCompactionMode,
            merge_context_policy,
        )

        if not spend.console_tokenizer_display_ready(
            self._screen, settings.model, settings.provider
        ):
            return None, 0
        if not isinstance(memory, EffectiveMemoryResult) or (
            merge_context_policy(
                global_overrides=global_overrides, conversation_overrides=overrides
            ).compaction_mode
            is not ContextCompactionMode.ASK
        ):
            return None, 0
        store = self._ensure_console_chat_store()
        history_key, messages, history = self._console_display_history(
            store, session_id, self._ensure_console_chat_controller()
        )
        record = memory.memory
        key = (
            history_key,
            memory.kind,
            getattr(record, "memory_id", None),
            getattr(record, "revision", None),
            settings.provider,
            settings.model,
        )
        cached = self._console_compaction_forecast_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        admitted = [
            message
            for message in messages
            if history.request_ids is None or message.id in history.request_ids
        ]
        projected = compaction_forecast_rows(admitted, memory)
        result: tuple[int | None, int] = (None, 0)
        if projected is not None:
            rows, units = projected
            estimate = self._build_context_estimate(
                list(rows), settings.provider, settings.model
            )
            result = (estimate.used_tokens, units)
        self._console_compaction_forecast_cache = (key, result)
        return result

    def compaction_pause_expected(self) -> bool:
        """Whether the active chat's next send will pause for compaction."""
        from .execution_state import selected_context_summary

        projected = selected_context_summary(self._screen)
        if projected is not None:
            return projected.compaction_pause_expected

        state = self._last_console_context_control_state
        return state is not None and state.compaction_pause_forecast is not None

    def compaction_forecast_unavailable(self) -> bool:
        """Whether current tool/input facts cannot qualify a forecast."""
        from .execution_state import selected_context_summary

        projected = selected_context_summary(self._screen)
        if projected is not None:
            return projected.compaction_forecast_unavailable

        state = self._last_console_context_control_state
        return state is not None and bool(state.forecast_unavailable_reason)

    def _build_console_settings_summary_state(self) -> ConsoleSettingsSummaryState:
        """Build compact summary state for the active Console session settings."""
        from .execution_state import selected_context_summary

        projected = selected_context_summary(self._screen)
        if projected is not None:
            settings, readiness = self._active_console_settings_readiness()
            return build_console_settings_summary_state(
                settings, projected.estimate, readiness
            )

        settings, readiness = self._active_console_settings_readiness()
        estimate = self._active_console_settings_context_estimate()
        try:
            self._last_console_context_control_state = (
                self._active_console_context_control_state(estimate=estimate)
            )
        except (KeyError, ValueError):
            self._last_console_context_control_state = None
        return build_console_settings_summary_state(
            settings,
            estimate,
            readiness,
        )

    def _refresh_settings_summary(self) -> None:
        """Refresh the mounted Console settings summary surfaces if present."""
        from .execution_state import selected_client

        if selected_client(self._screen) is not None:
            self._apply_console_settings_summary_state(
                self._build_console_settings_summary_state()
            )
            return
        from tldw_chatbook.Chat.console_parent_settings import parent_settings_render

        with parent_settings_render(self._screen):
            from tldw_chatbook.Chat.console_parent_settings import (
                defer_parent_settings_render,
            )

            if defer_parent_settings_render(self._screen):
                return
            self._apply_console_settings_summary_state(
                self._build_console_settings_summary_state()
            )

    def _apply_console_settings_summary_state(
        self,
        summary_state: ConsoleSettingsSummaryState,
    ) -> None:
        """Apply one already-derived settings summary to its mounted owners."""
        try:
            summary = self.query_one(
                "#console-settings-summary", ConsoleSettingsSummary
            )
        except (NoMatches, QueryError):
            pass
        else:
            # The child owns its bounded-body and rail invalidation.
            summary.sync_state(summary_state)
        # The Model section's sampling values and readiness word; the rail
        # finds its own rows (TASK-32811.7, TASK-33005.3, TASK-33628.5.2).
        rail = self.query_one_optional("#console-left-rail", ConsoleLeftRail)
        if rail is not None:
            rail.sync_model_section_values(summary_state)

        self._sync_console_rail_system_line()
        self._sync_console_agent_section()
        self._request_console_context_allocation_reconcile()

    def _build_console_cost_state(self) -> ConsoleCostState | None:
        """Build the cost chip's display state for the active session (task-5).

        Returns ``None`` when there is no active NATIVE Console session (the
        chip renders hidden) -- this is a normal condition, not a failure,
        so it is not subject to the best-effort fallback below.

        Everything past that point is best-effort: an unexpected failure is
        logged and this returns the last computed state (``None`` if there
        never was one) rather than raising into the sync path -- a stale or
        missing chip is fine, a broken send is not.
        """
        from ...Chat.console_control_settings import (
            ConsoleDefaultConvergenceRequired,
            ConsoleSettingsPending,
        )
        from ...Chat.console_parent_settings import (
            ParentSettingsDeferred,
            ParentSettingsOriginChanged,
            ParentSettingsSourceChanged,
        )

        try:
            from .execution_state import selected_context_summary

            projected = selected_context_summary(self._screen)
            if projected is not None:
                self._console_cost_cache_state = projected.cost_cache_state
                return projected.cost
            session = self._active_native_session()
            store = self._current_chat_store()
            if session is None or store is None:
                self._console_cost_cache_state = ConsoleCacheState.NONE
                return None
            from .config_sync import (
                run_console_config_sync,
                run_console_presentation_sync,
            )
            from .console_spend_projection import (
                ConsoleReadinessConfigProjection,
                _readiness_runtime,
                _same_readiness_key,
            )

            screen = self._screen
            projection = getattr(screen, "_console_readiness_config_projection", None)
            native_key = None
            if (
                type(projection) is ConsoleReadinessConfigProjection
                and projection.screen is screen
                and projection.value is not None
                and projection._runtime_owner is _readiness_runtime(screen)
            ):
                current_key = projection._key()
                if _same_readiness_key(
                    current_key, projection.key, ignore_settings=True
                ) and not _same_readiness_key(current_key, projection.key):
                    # A settings-only revision must not leave draft spend stale
                    # while this same owner's config mapping is being renewed.
                    native_key = current_key

            cost_state = self._last_console_cost_state
            previous_display = (
                self._console_cost_cache_state,
                self._console_settled_spend_cache,
                self._console_cost_projection_tokens_cache,
            )

            native_current = True

            def derive() -> None:
                nonlocal cost_state, native_current
                cost_state = self._build_console_cost_state_under_config(session, store)
                if native_key is not None and (
                    projection.screen is not screen
                    or getattr(screen, "_closing", False)
                    or _readiness_runtime(screen) is not projection._runtime_owner
                    or getattr(projection._runtime_owner, "_disposed", False)
                    or not _same_readiness_key(native_key, projection._key())
                ):
                    native_current = False
                    screen._request_console_control_bar_sync(delayed=True)

            if native_key is None:
                accepted = run_console_presentation_sync(screen, derive)
            else:
                accepted = run_console_config_sync(
                    derive,
                    maintenance_paused=(
                        getattr(screen, "_console_sync_maintenance_paused", False)
                        or getattr(screen, "_closing", False)
                        or getattr(projection._runtime_owner, "_disposed", False)
                    ),
                    defer=lambda: screen._request_console_control_bar_sync(delayed=True),
                    checked_projection=projection,
                )
            if accepted is False or not native_current:
                # A refused projection cannot own display or TTL state.
                (
                    self._console_cost_cache_state,
                    self._console_settled_spend_cache,
                    self._console_cost_projection_tokens_cache,
                ) = previous_display
                return self._last_console_cost_state
            return cost_state
        except (
            ConsoleDefaultConvergenceRequired,
            ConsoleSettingsPending,
            ParentSettingsDeferred,
            ParentSettingsOriginChanged,
            ParentSettingsSourceChanged,
        ):
            raise
        except Exception:  # noqa: BLE001 - Preserve last valid optional cost presentation.
            logger.opt(exception=True).warning("cost_chip_state_failed")
            return self._last_console_cost_state

    def _build_console_cost_state_under_config(
        self, session: Any, store: ConsoleChatStore
    ) -> ConsoleCostState | None:
        """Derive one synchronous display using the already selected owner pair."""
        from .cost_display import build_cost_display, build_cost_facts

        session_id = session.id
        provider, model, _settings = self._active_console_provider_model_display()
        tokenizer_ready = spend.console_tokenizer_display_ready(self._screen, model, provider)
        controller = self._ensure_console_chat_controller()
        history_key, messages, history = self._console_display_history(store, session_id, controller)
        facts = build_cost_facts(
            self, session_id=session_id, store=store, controller=controller,
            cost_controller=self._current_chat_controller(), provider=provider, model=model,
            tokenizer_ready=tokenizer_ready, history_key=history_key, messages=messages,
            history=history, catalog=self._display_pricing_catalog(),
            helper_usage=self._response_rule_usage(session_id), fleet_tokens=self._fleet_token_total(),
        )
        if facts is None:
            return None
        composer = self._console_composer_or_none()
        context_state = self._last_console_context_control_state if tokenizer_ready else None
        if context_state is None and tokenizer_ready:
            try:
                context_state = self._active_console_context_control_state()
            except (KeyError, ValueError):
                pass
        return build_cost_display(
            facts, context_state=context_state,
            pending_attachments=bool(store.pending_attachments(session_id)),
            draft_text=composer.draft_text() if composer is not None else "",
            rate_limit_line=spend.console_rate_limit_line(provider_config_key(provider)),
        )

    def _refresh_cost_chip(self) -> None:
        """Refresh the cost chip from freshly built state (task-5).

        Called at the END of ``_sync_console_control_bar`` -- deliberately
        OUTSIDE that method's ``control_state_changed``/
        ``workbench_state_changed`` guard (mirroring the unconditional
        inspector build right above it there), since cost/cache state can
        change independently of whether the control labels changed -- and
        from the 10s TTL repaint timer below so a WARM cache that goes
        EXPIRED with no other sync call in between still repaints.
        """
        from tldw_chatbook.Chat.console_parent_settings import parent_settings_render
        from .execution_state import selected_client

        if selected_client(self._screen) is not None:
            self._render_cost_chip()
            return

        with parent_settings_render(self._screen):
            from tldw_chatbook.Chat.console_parent_settings import (
                defer_parent_settings_render,
            )

            if defer_parent_settings_render(self._screen):
                return
            self._render_cost_chip()

    def _render_cost_chip(self) -> None:
        cost_state = self._build_console_cost_state()
        if cost_state != self._last_console_cost_state:
            self._last_console_cost_state = cost_state
            try:
                status_chips = self.query_one("#console-status-chips", ConsoleStatusChips)
            except QueryError:
                status_chips = None
            if status_chips is not None:
                status_chips.sync_cost_state(cost_state)
        if self._console_cost_cache_state == ConsoleCacheState.WARM:
            self._start_console_cost_ttl_timer()
        else:
            self._stop_console_cost_ttl_timer()

    def _start_console_cost_ttl_timer(self) -> None:
        """Start the 10s WARM->EXPIRED cost-chip repaint timer (task-5).

        No-ops when already running. Mirrors
        ``_start_console_transcript_sync_timer``'s audit pairing, but on its
        own cadence and stop condition -- see that method's docstring for
        why the 0.2s tick alone cannot cover this repaint.
        """
        if self._console_cost_ttl_timer is not None:
            return
        self._console_cost_ttl_timer = self.set_interval(
            self._cost_ttl_seconds(), self._sync_console_cost_chip
        )
        self._record_ui_timer_created("console-cost-ttl")

    def _stop_console_cost_ttl_timer(self) -> None:
        """Stop the cost-chip TTL repaint timer, if running."""
        if self._console_cost_ttl_timer is None:
            return
        try:
            self._console_cost_ttl_timer.stop()
        finally:
            self._record_ui_timer_stopped("console-cost-ttl")
            self._console_cost_ttl_timer = None

    def _open_console_cost_breakdown(self) -> None:
        """Push the Conversation Inspector on the Costs tab for the active
        native session (task-5/8).

        task-8: this used to push a standalone modal (retired in task-10);
        it now pushes the shared ``ConsoleConversationInspector`` (same
        modal Ctrl+Shift+P opens, just starting on a different tab). The
        Next Send factories are built too (via ``_console_inspector_next_
        send_factories``) so switching to that tab after opening from the
        chip renders real content, not stale/empty data.
        """
        from ...Widgets.Console.console_conversation_inspector import TAB_COSTS

        controller = self._ensure_console_chat_controller()
        session_id = controller.store.active_session_id
        if not session_id:
            self.notify("No active conversation.", severity="warning")
            return
        factory, estimate_factory, token_estimate, in_progress = (
            self._console_inspector_next_send_factories(controller, session_id)
        )
        self._push_console_inspector(
            initial_tab=TAB_COSTS,
            snapshot_factory=factory,
            estimate_factory=estimate_factory,
            token_estimate=token_estimate,
            in_progress=in_progress,
            payload_estimate=lambda snapshot: self._console_next_send_token_estimate(
                snapshot, session_id=session_id
            ),
        )

    def _build_console_inspector_cost_data(
        self,
    ) -> tuple[
        list[ConsoleCostRow],
        ConsoleCostRowTotals,
        list[InspectorTurn],
        Callable[[str], Awaitable[list[tuple[ExchangeCapture, bool]]]],
    ]:
        """Shared Costs-tab inputs for ``ConsoleConversationInspector``
        (task-8, extended task-9 for the Exchange tab).

        Returns:
            ``(rows, totals, turns, exchanges_loader)`` -- ``rows``/
            ``totals`` are ``build_cost_rows``/``build_cost_rows_totals``'s
            output; ``turns`` is one :class:`InspectorTurn` per transcript
            message (NOT filtered to contributing ones -- the
            contributing-only property is enforced downstream, in
            ``ConsoleConversationInspector``); ``exchanges_loader`` is
            called by the modal with one turn's ``native_message_id`` and
            returns ``(capture, abandoned)`` pairs (see
            ``console_conversation_inspector``'s module docstring for the
            pair contract and the ordering caveat -- callers must NOT trust
            the returned order, only ``(created_at, seq)``).

            The loader checks the NATIVE (in-memory) store first --
            ``message.exchanges`` on the matching ``ConsoleChatMessage`` --
            and only falls back to a DB read when there is none, so an
            EPHEMERAL session (no ``persisted_message_id``, no DB row at
            all) still resolves its captures; a native capture wins over a
            DB one when both exist (the native copy is fresher -- see
            ``ConsoleChatStore.attach_message_exchanges``). task-9 closed
            the former "known gap" here: a native capture's ``abandoned``
            flag is now resolved through ``store.abandoned_exchange_run_
            tags`` (the store's new public accessor over its private
            ``_abandoned_exchange_run_tags`` bookkeeping) rather than
            always reporting ``False``.
        """
        from ...Widgets.Console.console_conversation_inspector import InspectorTurn

        store = self._current_chat_store()
        messages: list[Any] = []
        if store is not None and store.active_session_id is not None:
            try:
                messages = store.messages_for_session(store.active_session_id)
            except KeyError:
                messages = []
        provider, model, _settings = self._active_console_provider_model_display()
        try:
            rows = build_cost_rows(messages, provider=provider, model=model)
        except Exception:  # noqa: BLE001 - Preserve optional Inspector row error policy.
            logger.opt(exception=True).warning("cost_breakdown_rows_failed")
            rows = []
        totals: ConsoleCostRowTotals = build_cost_rows_totals(rows)

        turns = [
            InspectorTurn(
                message_id=message.persisted_message_id or "",
                native_message_id=message.id,
                index=index,
                role=(
                    message.role.value
                    if isinstance(message.role, ConsoleMessageRole)
                    else str(message.role)
                ),
                preview=_console_inspector_turn_preview(message.content),
            )
            for index, message in enumerate(messages)
        ]
        messages_by_native_id = {message.id: message for message in messages}
        exchanges_loader = _build_console_inspector_exchanges_loader(
            messages_by_native_id,
            store.projected_trace_calls
            if store is not None
            else lambda _message_id: (),
            store.abandoned_exchange_run_tags if store is not None else None,
        )

        return rows, totals, turns, exchanges_loader
