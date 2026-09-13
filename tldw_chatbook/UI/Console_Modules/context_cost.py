"""Console context and spend projections, including token and fingerprint memoization."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import replace
import asyncio
import time
from typing import Any, Optional
from loguru import logger
from textual.css.query import NoMatches, QueryError
from ..Console_Modules import console_spend_projection as spend
from ...Chat.console_context_policy import ConsoleContextPolicyOverrides
from ...Chat.console_cost_tracker import (
    ConsoleCacheState,
    ConsoleCostRow,
    ConsoleCostRowTotals,
    ConsoleCostState,
    TokenEstimateCache,
    build_cost_rows,
    build_cost_rows_totals,
    build_cost_snapshot,
    fingerprint_break_reason,
    token_estimate_signature,
)
from ...Chat.console_exchange_capture import ExchangeCapture
from ...Chat.console_trace_projection import ProjectedTraceCall
from ...LLM_Calls.pricing_catalog import get_pricing_catalog
from ...Chat.console_chat_models import (
    ConsoleContextSnapshot,
    ConsoleMessageRole,
    ConsoleRunStatus,
    MessageAttachment,
)
from ...Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
    ConsoleSettingsSummaryState,
    _estimate_tokens_locally,
    build_console_context_estimate,
    build_console_settings_summary_state,
)
from ...Chat.console_display_state import (
    console_prompted_evidence_text,
    estimate_console_next_send_tokens,
)
from ...Chat.provider_readiness import provider_config_key
from ...Utils.token_counter import estimate_tokens
from ...Widgets.Console.console_context_controls import (
    ConsoleContextControlState,
    build_console_context_control_state,
)
from ...Widgets.Console.console_conversation_inspector import InspectorTurn

from ...Chat.console_chat_models import FEEDBACK_ACTIVE_RUN_STATUSES


logger = logger.bind(module="ChatScreen")
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


class ConsoleContextCostController:
    """Own console context and spend projections, including token and fingerprint memoization.

    App identity is stable for this controller lifetime. All other dependencies
    are explicit callables resolved by wiring at use time. No DOM is owned here.
    """

    def __init__(
        self,
        *,
        app_instance_accessor: Callable[[], Any],
        _active_console_provider_model_display: Callable[..., Any],
        _active_console_settings_readiness: Callable[..., Any],
        _active_control_state: Callable[..., Any],
        _active_estimate: Callable[..., Any],
        _active_session: Callable[..., Any],
        _active_session_settings: Callable[..., Any],
        _agent_fleet_tokens: Callable[..., Any],
        _build_console_staged_context_state: Callable[..., Any],
        _console_composer_or_none: Callable[..., Any],
        _ensure_console_chat_controller: Callable[..., Any],
        _ensure_console_chat_store: Callable[..., Any],
        _query_composer: Callable[..., Any],
        _session_control_state: Callable[..., Any],
        _session_estimate: Callable[..., Any],
        _workspace_context: Callable[..., Any],
        _console_chat_controller_accessor: Callable[[], Any],
        _console_chat_store_accessor: Callable[[], Any],
        _last_console_cost_state_accessor: Callable[[], Any],
        _pending_console_launch_context_accessor: Callable[[], Any],
    ) -> None:
        self._app_instance_accessor = app_instance_accessor
        self._active_console_provider_model_display = (
            _active_console_provider_model_display
        )
        self._active_console_settings_readiness = _active_console_settings_readiness
        self._active_control_state = _active_control_state
        self._active_estimate = _active_estimate
        self._active_session = _active_session
        self._active_session_settings = _active_session_settings
        self._agent_fleet_tokens = _agent_fleet_tokens
        self._build_console_staged_context_state = _build_console_staged_context_state
        self._console_composer_or_none = _console_composer_or_none
        self._ensure_console_chat_controller = _ensure_console_chat_controller
        self._ensure_console_chat_store = _ensure_console_chat_store
        self._query_composer = _query_composer
        self._session_control_state = _session_control_state
        self._session_estimate = _session_estimate
        self._workspace_context = _workspace_context
        self._console_chat_controller_accessor = _console_chat_controller_accessor
        self._console_chat_store_accessor = _console_chat_store_accessor
        self._last_console_cost_state_accessor = _last_console_cost_state_accessor
        self._pending_console_launch_context_accessor = (
            _pending_console_launch_context_accessor
        )
        self._console_cost_cache_state = ConsoleCacheState.NONE
        self._console_cost_estimate_cache = None
        self._last_console_context_control_state = None
        self._console_cost_fp_revisions = {}
        self._console_cost_break_reasons = {}

    @property
    def _console_chat_controller(self) -> Any:
        return self._console_chat_controller_accessor()

    @property
    def _console_chat_store(self) -> Any:
        return self._console_chat_store_accessor()

    @property
    def _last_console_cost_state(self) -> Any:
        return self._last_console_cost_state_accessor()

    @property
    def _pending_console_launch_context(self) -> Any:
        return self._pending_console_launch_context_accessor()

    @property
    def app_instance(self) -> Any:
        return self._app_instance_accessor()

    def _console_inspector_next_send_factories(
        self, controller: Any, session_id: str
    ) -> tuple[
        Callable[[], Awaitable[ConsoleContextSnapshot]],
        Callable[[], int | None],
        int | None,
        bool,
    ]:
        """Build the Next Send tab's snapshot/estimate factories (task-18300).

        Shared by BOTH ``ConsoleConversationInspector`` entry points (the
        cost chip and Ctrl+Shift+P) -- the two push the SAME modal
        instance, and the user can switch to the Next Send tab regardless
        of which tab it opened on, so a caller that skipped this would
        leave the tab showing nothing. Building the closures themselves is
        cheap (no I/O happens until one is actually CALLED); the Next Send
        pane calls ``snapshot_factory`` once on mount regardless of
        ``initial_tab`` (see ``ConsoleConversationInspector.on_mount``).

        ``session_id`` is threaded in rather than re-read from the store
        because the composer only reflects the ACTIVE session; see
        ``_captured_draft``.
        """

        def _captured_draft() -> str:
            if controller.store.active_session_id == session_id:
                try:
                    return self._query_composer().draft_text()
                except (NoMatches, QueryError):
                    pass
            session = next(
                (item for item in controller.store.sessions() if item.id == session_id),
                None,
            )
            return session.draft if session is not None else ""

        async def _factory() -> ConsoleContextSnapshot:
            current_draft = _captured_draft()
            pending = controller.store.pending_attachments(session_id)
            current_attachments = tuple(
                MessageAttachment(
                    data=pending_attachment.data,
                    mime_type=pending_attachment.mime_type or "image/png",
                    display_name=pending_attachment.display_name,
                    position=index,
                )
                for index, pending_attachment in enumerate(pending)
            )
            current_staged_sources = controller.store.workspace_context.allowed_sources

            return await controller.build_context_snapshot(
                draft=current_draft,
                attachments=current_attachments,
                staged_sources=current_staged_sources,
                session_id=session_id,
            )

        def _estimate_factory() -> int | None:
            return self._estimate_tokens({"draft": _captured_draft()})

        token_estimate = _estimate_factory()
        in_progress = controller.run_state.status in CONSOLE_ACTIVE_RUN_STATUSES
        return _factory, _estimate_factory, token_estimate, in_progress

    def _estimate_tokens(self, payload: dict[str, Any]) -> int | None:
        """Return a token estimate for the current draft text."""
        text = payload.get("draft", "")
        if not text:
            return None
        return estimate_tokens(text, "", "")

    def _console_next_send_token_estimate(
        self, snapshot: Any, session_id: Optional[str] = None
    ) -> Optional[int]:
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
            store = self._console_chat_store
            include_staged = store is not None and store.active_session_id == session_id
        provider, model, _settings = self._active_console_provider_model_display()
        return estimate_console_next_send_tokens(
            payload_messages=payload.get("messages") or [],
            payload_system=payload.get("system"),
            tools_info=payload.get("tools"),
            extra_texts=(
                console_prompted_evidence_text(self._pending_console_launch_context),
            )
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
        store = self._ensure_console_chat_store()
        session_id = store.active_session_id
        if session_id is None:
            settings = self._active_session_settings()
            return build_console_context_estimate(
                [],
                settings.provider,
                settings.model,
                max_tokens_response=settings.max_tokens,
                system_prompt=settings.system_prompt,
            )
        return self._session_estimate(session_id)

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
        include_active_staging = store.active_session_id == session_id
        workspace_context = (
            self._workspace_context() if include_active_staging else None
        )
        pending_launch = (
            self._pending_console_launch_context if include_active_staging else None
        )
        staged_context_state = self._build_console_staged_context_state(pending_launch)
        try:
            session_messages = store.messages_for_session(session_id)
        except KeyError:
            session_messages = []
        greeting = ""
        composer = self._console_composer_or_none() if include_active_staging else None
        if include_active_staging:
            controller = self._ensure_console_chat_controller()
            history = spend.build_console_spend_history_projection(
                session_messages,
                store.dispatch_recovery_for_session(session_id),
                store.preparation_for_session(session_id),
                controller.run_state_for(session_id).status,
                bool(controller._submit_tasks_for_session(session_id)),
            )
            messages = spend.build_console_context_messages(
                session_messages,
                history.request_ids,
                composer.draft_text() if composer is not None else "",
            )
            greeting = controller._seeded_greeting_text(session_id, session_messages)
        else:
            messages = spend.build_console_context_messages(session_messages, None, "")
        return build_console_context_estimate(
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
        )

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
            estimate = estimate or self._active_estimate()
            return build_console_context_control_state(
                settings=settings,
                estimate=estimate,
                overrides=ConsoleContextPolicyOverrides(),
                global_overrides=None,
                active_memory=None,
            )
        return self._session_control_state(
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
        estimate = estimate or self._session_estimate(
            session_id,
            settings=settings,
        )
        overrides = ConsoleContextPolicyOverrides()
        global_overrides = None
        memory = None
        controller = self._ensure_console_chat_controller()
        try:
            overrides, global_overrides, memory = controller.context_control_inputs(
                session_id
            )
        except (KeyError, ValueError):
            pass
        accounting = None
        try:
            accounting = controller.context_breakdown_accounting(session_id)
        except Exception:
            accounting = None
        return build_console_context_control_state(
            settings=settings,
            estimate=estimate,
            overrides=overrides,
            global_overrides=global_overrides,
            active_memory=memory,
            accounting=accounting,
            thinking_history_policy=(
                store.session_thinking_history_policy(session_id)
                if session_id is not None
                else "auto"
            ),
            thinking_history_effective_policy=thinking_history_effective_policy,
        )

    def _build_console_settings_summary_state(self) -> ConsoleSettingsSummaryState:
        """Build compact summary state for the active Console session settings."""
        settings, readiness = self._active_console_settings_readiness()
        estimate = self._active_estimate()
        try:
            self._last_console_context_control_state = self._active_control_state(
                estimate=estimate
            )
        except (KeyError, ValueError):
            self._last_console_context_control_state = None
        return build_console_settings_summary_state(
            settings,
            estimate,
            readiness,
        )

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
        try:
            session = self._active_session()
            store = self._console_chat_store
            if session is None or store is None:
                self._console_cost_cache_state = ConsoleCacheState.NONE
                return None
            session_id = session.id
            try:
                messages = store.messages_for_session(session_id)
            except KeyError:
                self._console_cost_cache_state = ConsoleCacheState.NONE
                return None

            controller = self._ensure_console_chat_controller()
            history = spend.build_console_spend_history_projection(
                messages,
                store.dispatch_recovery_for_session(session_id),
                store.preparation_for_session(session_id),
                controller.run_state_for(session_id).status,
                bool(controller._submit_tasks_for_session(session_id)),
            )
            snapshot_messages = spend.build_console_current_cost_messages(
                messages, history.current_ids
            )
            provider, model, _settings = self._active_console_provider_model_display()
            # PR2b Task 5 (cost rollup): the active conversation's LIVE
            # sub-agent fleet spend, folded into the snapshot's token total
            # (never priced -- see `ConsoleCostSnapshot.fleet_tokens`'s
            # docstring for why). Read straight off the SAME live source
            # the fleet rail rows themselves read
            # (`_console_agent_fleet_token_total` sums `bridge.fleet_
            # snapshot(...)`'s `FleetHandle.total_tokens`), so the chip and
            # the rail can never disagree about a conversation's fleet
            # spend.
            fleet_tokens = self._agent_fleet_tokens()
            # PR3a-1 Task 6b (audit F3): plus whatever a SURVIVING child
            # billed after its turn's usage was attached to the assistant
            # message. PR3a-2 Task 3 (tasks 15660/15667): that spend is now
            # INTERIM, not lost -- when the conversation's last fleet child
            # settles, the controller's "usage-reattach" drain consumer
            # folds the whole turn (survivors included) back onto the
            # message's own usage row and this line falls to zero. Until
            # that drain, the money is named here on the chip's unpriced
            # sub-agent line. No double count: `unattributed_fleet_tokens`
            # counts ONLY payloads closed out after the latest attach (the
            # fold resets its watermark), and a live handle's
            # `FleetHandle.total_tokens` -- what `_console_agent_fleet_token_
            # total` sums -- is 0 until it finishes, by which point it has
            # left `fleet_snapshot`.
            cost_controller = self._console_chat_controller
            if cost_controller is not None:
                unattributed = getattr(
                    cost_controller, "unattributed_fleet_tokens", None
                )
                if callable(unattributed):
                    fleet_tokens += int(unattributed(session_id) or 0)
            # task-15451: this method runs on the 0.2s tick for the whole
            # duration of a run (plus every control-bar sync pass and the
            # 10s TTL timer), and the equality guard in
            # `_sync_console_cost_chip` gates only the REPAINT -- the build
            # itself always ran. Without the memo below every usage-less row
            # (all user/system rows, legacy assistant rows, the staged
            # evidence pseudo-row) was re-tokenized by a per-character
            # Python loop 5x/s: ~28ms/tick on a 99KB transcript, measured.
            # The memo re-verifies each row's own text before serving a hit,
            # so it can change how long this takes but not what it returns.
            #
            # Gating the whole snapshot on `store.payload_revision` instead
            # was considered and rejected: usage is not payload-affecting, so
            # `ConsoleChatStore.set_message_usage` never bumps that counter --
            # a real priced usage landing on an ALREADY-terminal row (the
            # documented Stop-path ordering) would leave the chip showing the
            # estimated total until some unrelated edit moved the revision.
            estimate_cache = self._console_cost_estimate_cache_or_new()
            snapshot = build_cost_snapshot(
                snapshot_messages,
                provider=provider,
                model=model,
                fleet_tokens=fleet_tokens,
                estimate_cache=estimate_cache,
            )

            controller = self._console_chat_controller
            run_status = (
                controller.run_state_for(session_id).status
                if controller is not None
                else ConsoleRunStatus.IDLE
            )
            # Fingerprint compare is the expensive step (rebuilds the
            # pre-compaction provider payload) -- only pay it when the
            # session isn't actively streaming AND its payload has actually
            # changed since the last check (revision `!=`, not `>`: a
            # restore can reset the store's counter back down).
            break_reason = self._console_cost_break_reasons.get(session_id)
            if controller is not None and run_status not in CONSOLE_ACTIVE_RUN_STATUSES:
                current_revision = store.payload_revision(session_id)
                if current_revision != self._console_cost_fp_revisions.get(session_id):
                    baseline = controller.payload_fingerprint_baseline(session_id)
                    break_reason = None
                    if baseline is not None:
                        current_fp = controller.compute_current_fingerprint(session_id)
                        break_reason = fingerprint_break_reason(baseline, current_fp)
                    self._console_cost_fp_revisions[session_id] = current_revision
                    self._console_cost_break_reasons[session_id] = break_reason

            cache_state = ConsoleCacheState.NONE
            ttl_remaining_s: float | None = None
            if controller is not None:
                warm_until, had_activity = controller.cache_ttl_snapshot(session_id)
                if had_activity and warm_until is not None:
                    now = time.monotonic()
                    if now < warm_until:
                        cache_state = ConsoleCacheState.WARM
                        ttl_remaining_s = warm_until - now
                    else:
                        cache_state = ConsoleCacheState.EXPIRED
            self._console_cost_cache_state = cache_state

            projected_delta_usd: float | None = None
            pricing_as_of: str | None = None
            catalog = get_pricing_catalog()
            provider_key = provider_config_key(provider)
            pricing = catalog.get_pricing(provider_key, model or "")
            if pricing is not None:
                pricing_as_of = pricing.as_of
                if (
                    cache_state == ConsoleCacheState.WARM
                    and break_reason
                    and pricing.cache_write_per_mtok is not None
                    and pricing.cache_read_per_mtok is not None
                ):
                    # `break_reason` is required here (Qodo round, finding
                    # 3): `build_cost_state`/`_cache_state_line` only ever
                    # read `projected_delta_usd` inside their own
                    # `break_reason`-gated branches (the alert suffix on the
                    # label, and the "~+$" clause in the tooltip's cache
                    # line) -- with no break reason the value is built and
                    # then silently discarded every call. Skipping the
                    # (expensive: `_estimate_tokens_locally` over the WHOLE
                    # transcript) computation here avoids that on every
                    # 0.2s/10s sync tick for a long-running WARM session
                    # that never alerts.
                    #
                    # `snapshot_messages` (not `messages`): same mid-stream-
                    # animation guard as the snapshot above -- an in-flight
                    # row's growing content must not grow the projected
                    # break-delta either, e.g. when a NEW turn starts
                    # streaming while a PRIOR turn's alert is still showing
                    # (fingerprint recompute -- and so `break_reason` /
                    # `alert` -- is frozen during the run, but this
                    # projection is computed fresh every call).
                    #
                    # task-15451: gated, but not cheap -- an alerting
                    # session pays a WHOLE-transcript estimate on every tick
                    # for as long as the alert stands. Same memo, same
                    # guarantee: the hit is verified against every row's
                    # (role, content) before it is served.
                    projection_rows = tuple(
                        (
                            str(getattr(message.role, "value", message.role)),
                            message.content,
                        )
                        for message in snapshot_messages
                    )

                    def _estimate_projection() -> int:
                        return _estimate_tokens_locally(
                            [
                                {"role": role, "content": content}
                                for role, content in projection_rows
                            ],
                            model or "",
                            provider_key,
                        )

                    projection_cache = self._console_cost_estimate_cache_or_new()
                    estimated_tokens = projection_cache.estimate(
                        ("#cost-projection", session_id),
                        token_estimate_signature(
                            projection_rows, model or "", provider_key
                        ),
                        _estimate_projection,
                    )
                    rate_delta = (
                        pricing.cache_write_per_mtok - pricing.cache_read_per_mtok
                    ) / 1_000_000
                    projected_delta_usd = round(estimated_tokens * rate_delta, 6)

            composer = self._console_composer_or_none()
            context_state = self._last_console_context_control_state
            if context_state is None:
                try:
                    context_state = self._active_control_state()
                except (KeyError, ValueError):
                    pass
            return spend.build_console_spend_cost_state(
                snapshot,
                cache_state,
                break_reason,
                projected_delta_usd,
                ttl_remaining_s,
                pricing_as_of,
                pricing is not None,
                context_state,
                # Configuration capture resolves RAG defaults; text-only
                # display refreshes must not pull that work onto first paint.
                any(
                    message.role is ConsoleMessageRole.USER
                    and message.attachments
                    and message.id in history.request_ids
                    for message in messages
                )
                and any(
                    row.attachments
                    for row in controller._lightweight_provider_message_rows(
                        [
                            message
                            for message in messages
                            if message.id in history.request_ids
                        ],
                        skip_failed=True,
                        session_id=session_id,
                        turn_context=controller.resolve_turn_configuration_snapshot(
                            session_id
                        ),
                    )
                ),
                bool(store.pending_attachments(session_id)),
                pricing.input_per_mtok if pricing is not None else None,
                composer.draft_text() if composer is not None else "",
            )
        except Exception:
            logger.opt(exception=True).warning("cost_chip_state_failed")
            return self._last_console_cost_state

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
        store = self._console_chat_store
        messages: list[Any] = []
        if store is not None and store.active_session_id is not None:
            try:
                messages = store.messages_for_session(store.active_session_id)
            except KeyError:
                messages = []
        provider, model, _settings = self._active_console_provider_model_display()
        try:
            rows = build_cost_rows(messages, provider=provider, model=model)
        except Exception:
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
