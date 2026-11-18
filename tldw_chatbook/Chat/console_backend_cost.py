"""Original cost projection owned by the selected readiness read."""

from __future__ import annotations

from types import FunctionType
from loguru import logger

from .console_chat_models import FEEDBACK_ACTIVE_RUN_STATUSES, ConsoleMessageRole
from .console_cost_tracker import ConsoleCacheState, TokenEstimateCache, build_cost_snapshot
from .console_domain_services import _definition
from .console_session_settings import _estimate_tokens_locally
from ..UI.Console_Modules import console_spend_projection as spend
from ..UI.Console_Modules import cost_display, pricing_display
from ..UI.Console_Modules import tokenizer_display


class ConsoleBackendCost:
    """Keep the same revision memos as the local cost display, per backend."""

    def __init__(self, backend):
        self.backend = backend
        self.catalog = None
        self.facts = None
        self.unavailable = False
        self._last_display = None
        self.cache_state = ConsoleCacheState.NONE
        self._tokenizer_preparation = None
        self._tokenizer_ready = False
        self._history_cache = None
        self._console_cost_estimate_cache = TokenEstimateCache()
        self._console_cost_cache_state = ConsoleCacheState.NONE
        self._console_settled_spend_cache = None
        self._console_cost_projection_tokens_cache = None
        self._console_cost_fp_revisions = {}
        self._console_cost_break_reasons = {}
        self._build_cost_snapshot = build_cost_snapshot
        self._estimate_tokens_locally = _estimate_tokens_locally

    def _console_cost_estimate_cache_or_new(self):
        return self._console_cost_estimate_cache

    def _inputs(self, session, effective, tokenizer_ready):
        store, controller = self.backend.store, self.backend.controller
        recovery = store.dispatch_recovery_for_session(session.id)
        preparation = store.preparation_for_session(session.id)
        status = controller.run_state_for(session.id).status
        submitted = bool(controller._submit_tasks_for_session(session.id))
        history_key = (id(store), session.id, store.display_projection_revision(session.id),
                       id(recovery), id(preparation), status, submitted)
        if self._history_cache is None or self._history_cache[0] != history_key:
            messages = tuple(store.read_only_messages_for_session(session.id))
            history = spend.build_console_spend_history_projection(
                messages, recovery, preparation, status, submitted,
            )
            # Keep the actual owners alive, not only their IDs, across retirement.
            self._history_cache = history_key, messages, history, store, recovery, preparation
        _, messages, history, *_ = self._history_cache
        helper_usage = self.backend.runtime.response_rule_usage(session.id)
        return dict(
            session_id=session.id, store=store, controller=controller,
            cost_controller=controller, provider=effective.provider, model=effective.model,
            tokenizer_ready=tokenizer_ready, history_key=history_key,
            messages=messages, history=history, catalog=self.catalog,
            helper_usage=helper_usage,
            fleet_tokens=(sum(handle.total_tokens for handle in
                self.backend.runtime.agent_bridge.fleet_snapshot(
                    session.persisted_conversation_id)) if session.persisted_conversation_id else 0),
        )

    def can_derive(self, session, effective):
        """Prove the shared builder cannot enter a native media/fingerprint read."""
        if self.unavailable:
            # No price/native fact is reused after failure. The optional chip
            # keeps its last display until the existing readiness age expires.
            return True
        if not pricing_display.pricing_display_current(self.catalog):
            return False
        if effective.model and (
            self._tokenizer_preparation is None
            or not tokenizer_display.preparation_current(self._tokenizer_preparation)
        ):
            return False
        tokenizer_ready = self._tokenizer_ready
        inputs = self._inputs(session, effective, tokenizer_ready)
        controller, store = inputs["controller"], inputs["store"]
        key = (inputs["history_key"], effective.provider, effective.model,
               store.session_settings_revision(session.id), self.catalog,
               inputs["helper_usage"], tokenizer_ready)
        cached = self._console_settled_spend_cache
        if tokenizer_ready and (cached is None or cached[0] != key) and any(
            row.role is ConsoleMessageRole.USER and row.attachments
            and row.id in inputs["history"].request_ids for row in inputs["messages"]
        ):
            return False
        if (tokenizer_ready
            and controller.run_state_for(session.id).status not in FEEDBACK_ACTIVE_RUN_STATUSES
            and store.payload_revision(session.id) != self._console_cost_fp_revisions.get(session.id)
            and controller.payload_fingerprint_baseline(session.id) is not None):
            return False
        return True

    def prepare(self, session, effective, tokenizer_ready, preparation):
        """Called only in the existing finite, observed readiness worker."""
        self.unavailable = False
        self.catalog = pricing_display.prepare_pricing_display()
        if not pricing_display.pricing_display_current(self.catalog):
            raise RuntimeError("Console cost pricing source is unavailable")
        if effective.model != session.settings.model:
            preparation = (tokenizer_display.capture_tokenizer_preparation(
                effective.model, effective.provider,
            ) if effective.model else None)
            tokenizer_ready = preparation is None or preparation()
        self._tokenizer_preparation = preparation
        self._tokenizer_ready = tokenizer_ready
        self.derive(session, effective, native=True)

    def derive(self, session, effective, *, native=False):
        if self.unavailable:
            return True
        if not native and not self.can_derive(session, effective):
            return False
        try:
            self.facts = cost_display.build_cost_facts(
                self, **self._inputs(session, effective, self._tokenizer_ready),
            )
        except Exception:
            self.failed()
        return True

    def failed(self):
        self.unavailable = True
        logger.opt(exception=True).warning("cost_chip_state_failed")

    def project(self, session, context_state):
        identity = session.id, session.incarnation_id
        if not self.unavailable and not pricing_display.pricing_display_current(self.catalog):
            self.unavailable = True
        if not self.unavailable:
            try:
                state = cost_display.build_cost_display(
                    self.facts, context_state=context_state,
                    pending_attachments=bool(self.backend.store.pending_attachments(session.id)),
                    draft_text=session.draft,
                    rate_limit_line=spend.console_rate_limit_line(session.settings.provider),
                )
                cache = self._console_cost_cache_state
                self._last_display = identity, state, cache
            except Exception:
                self.failed()
        if self._last_display is not None and self._last_display[0] == identity:
            self.cache_state = self._last_display[2]
            return self._last_display[1]
        self.cache_state = ConsoleCacheState.NONE
        return None


_METHODS = tuple((name, _definition(value)) for name, value in vars(ConsoleBackendCost).items()
                 if type(value) is FunctionType)
