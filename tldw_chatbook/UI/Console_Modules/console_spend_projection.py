"""Cheap, UI-neutral projections for Console current and next-send spend."""

from __future__ import annotations

import asyncio
import copy
import threading
import time
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import wraps
from typing import Any

from ...Chat.assistant_generation_state import assistant_state_allows_provider_history
from ...Chat.console_chat_controller import _is_empty_transcript_row
from ...Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleDispatchRecoveryKind,
    ConsoleDispatchRecoveryState,
    ConsoleMessageRole,
    ConsoleRunStatus,
    fold_greeting_into_system_prompt,
)
from ...Chat.console_context_policy import (
    ConsoleContextPolicyOverrides,
    context_policy_overrides_from_console_config,
)
from ...Chat.console_cost_tracker import (
    ConsoleCacheState,
    ConsoleCostSnapshot,
    ConsoleCostState,
    build_cost_state,
)
from ...Chat.console_turn_preparation import (
    ConsoleTurnPreparation,
    ConsoleTurnPreparationState,
)
from ...Chat.provider_continuation import ProviderContinuationCheckpoint
from ...Utils.egress import latest_rate_limit_headers
from ...Utils.input_validation import validate_console_draft
from ...Widgets.Console.console_context_controls import (
    ConsoleContextControlState,
    ConsoleNextSendSpendState,
    build_console_context_cost_state,
    build_console_next_send_spend_state,
)

_ACCEPTED_RECOVERY_KINDS = frozenset(
    {
        ConsoleDispatchRecoveryKind.ACCEPTED,
        ConsoleDispatchRecoveryKind.DISPATCH_STARTED,
        ConsoleDispatchRecoveryKind.EPHEMERAL_ACCEPTED,
        ConsoleDispatchRecoveryKind.EPHEMERAL_DISPATCH_STARTED,
        ConsoleDispatchRecoveryKind.REMOTE_ACCEPTED,
        ConsoleDispatchRecoveryKind.REMOTE_DISPATCH_STARTED,
    }
)
_ACCEPTED_PREPARATION_STATES = frozenset(
    {
        ConsoleTurnPreparationState.ACCEPTED,
        ConsoleTurnPreparationState.DISPATCH_STARTED,
        ConsoleTurnPreparationState.DISPATCHED,
    }
)
_REMOTE_RECOVERY_KINDS = frozenset(
    {
        ConsoleDispatchRecoveryKind.REMOTE_ACCEPTED,
        ConsoleDispatchRecoveryKind.REMOTE_DISPATCH_STARTED,
    }
)


def _screen_readiness_config(screen: Any) -> Any:
    from textual._context import NoActiveAppError

    app_instance = getattr(screen, "app_instance", None)
    if app_instance is None:
        return {}
    try:
        return getattr(screen.app, "app_config") or {}
    except (AttributeError, NoActiveAppError):
        return getattr(app_instance, "app_config", {}) or {}


def provider_readiness_app_config(
    screen: Any, load_current: Callable[[], Any], *, memoize: bool = True
) -> Any:
    """Keep live action reads; synchronous presentation supplies owned data."""
    active = getattr(screen, "_console_readiness_projection_active", None)
    if active is not None and active[0] == threading.get_ident():
        return active[1]
    memo = getattr(screen, "_console_derivation_memo", None) if memoize else None
    if memo is not None and "app_config" in memo:
        return memo["app_config"]
    resolved = _screen_readiness_config(screen)
    if screen._console_config_snapshot_is_disk_loaded(resolved):
        try:
            fresh = load_current()
        except Exception:  # noqa: BLE001 - preserve the existing snapshot fallback.
            pass
        else:
            if isinstance(fresh, Mapping) and fresh:
                resolved = fresh
    if memo is not None:
        memo["app_config"] = resolved
    return resolved


@dataclass(frozen=True)
class ConsoleReadinessConfigRead:
    """Source tags captured inside the same checked finite mapping read."""

    source_before: tuple[int, str]
    value: Mapping
    source_after: tuple[int, str]
    context_policy: ConsoleContextPolicyOverrides | None = None


class ConsoleReadinessConfigProjection:
    """Own finite config reads shared only by synchronous presentation work."""

    def __init__(
        self, screen: Any, *, read_current: Callable[[], Any], max_age: float = 1.0
    ) -> None:
        self.screen = screen
        self.read_current = read_current
        self.max_age = max_age
        self.key = self.value = None
        self.at = 0.0
        self.pending = False
        self.context_policy = None
        self._settled = asyncio.Event()
        self._settled.set()

    @classmethod
    def for_screen(cls, screen: Any) -> ConsoleReadinessConfigProjection:
        projection = getattr(screen, "_console_readiness_config_projection", None)
        if projection is None:

            def read_current():
                from tldw_chatbook import config
                from tldw_chatbook.Backup_Recovery.config_participants import (
                    checked_config_identity,
                    operation,
                )
                from ..Screens.chat_screen import load_settings

                with operation(config) as active:
                    before = checked_config_identity(config, active)
                    value = copy.deepcopy(load_settings())
                    # Match live get_cli_setting's sparse CLI lookup, rather
                    # than deriving policy from the merged application map.
                    console = config.load_cli_config_and_ensure_existence().get(
                        "console"
                    )
                    try:
                        policy = context_policy_overrides_from_console_config(
                            console if isinstance(console, dict) else None
                        )
                    except (TypeError, ValueError):
                        # The existing display reader treats invalid policy as
                        # unavailable; provider readiness still has its mapping.
                        policy = None
                    return ConsoleReadinessConfigRead(
                        before, value, checked_config_identity(config, active), policy
                    )

            projection = cls(screen, read_current=read_current)
            screen._console_readiness_config_projection = projection
        return projection

    def _key(self) -> tuple:
        from tldw_chatbook import config

        screen = self.screen
        app = getattr(screen, "app_instance", None)
        store = getattr(screen, "_console_chat_store", None)
        session_id = getattr(store, "active_session_id", None)
        owner = (
            next((item for item in store.sessions() if item.id == session_id), None)
            if store is not None
            else None
        )
        return (
            config.current_config_identity(),
            app,
            id(getattr(app, "app_config", None)),
            getattr(app, "chachanotes_db", None),
            store,
            session_id,
            getattr(owner, "workspace_id", None),
            store.session_settings_revision(session_id) if owner is not None else None,
            id(owner),
            owner,
            getattr(app, "app_config", None),
        )

    def run(self, body: Callable[[], Any]) -> bool:
        """Defer a cold owner; use only its own last mapping during expiry."""
        screen = self.screen
        key = self._key()
        current = key == self.key and self.value is not None
        if (
            not current or time.monotonic() - self.at >= self.max_age
        ) and not self.pending:
            self.pending = True
            self._settled.clear()
            screen.run_worker(
                self._refresh(key), exclusive=False, group="console-readiness-config"
            )
        if not current:
            return False
        previous = getattr(screen, "_console_readiness_projection_active", None)
        screen._console_readiness_projection_active = threading.get_ident(), self.value
        try:
            with screen._console_derivation_scope():
                return body() is not False
        finally:
            screen._console_readiness_projection_active = previous

    async def warm(self) -> bool:
        """Wait for the same checked owner when a modal needs cold display data."""
        key = self._key()
        self.run(lambda: None)
        if self.pending:
            await self._settled.wait()
        return (
            key == self._key() == self.key
            and self.value is not None
            and time.monotonic() - self.at < self.max_age
        )

    async def _refresh(self, key: tuple) -> None:
        worker = asyncio.create_task(asyncio.to_thread(self.read_current))
        try:
            try:
                result = await asyncio.shield(worker)
            except asyncio.CancelledError:
                # The native read still owns its resources until its callback
                # exits; cancellation may not publish or finish ownership early.
                while not worker.done():
                    try:
                        await asyncio.shield(worker)
                    except asyncio.CancelledError:
                        continue
                    except Exception:  # noqa: BLE001 - cancellation takes precedence.
                        break
                if not worker.cancelled():
                    worker.exception()
                raise
            if (
                not isinstance(result, ConsoleReadinessConfigRead)
                or result.source_before != key[0]
                or result.source_after != key[0]
                or key != self._key()
            ):
                return
            value = result.value
            controller = getattr(self.screen, "_session", None)
            store = getattr(self.screen, "_console_chat_store", None)
            owner = (
                next((item for item in store.sessions() if item.id == key[5]), None)
                if store is not None
                else None
            )
            if controller is not None and owner is not None:
                controller._maybe_refresh_stale_default_console_settings(
                    store, owner, checked_config=value
                )
            # Convergence may change this same owner's settings revision. It
            # cannot authorize publishing to a different profile or owner.
            current_key = self._key()
            if (*current_key[:7], *current_key[8:]) != (*key[:7], *key[8:]):
                return
            changed = (
                self.key != current_key
                or self.value != value
                or self.context_policy != result.context_policy
            )
            self.key, self.value, self.at = current_key, value, time.monotonic()
            self.context_policy = result.context_policy
        except Exception:  # noqa: BLE001 - remain cold and retryable.
            return
        finally:
            self.pending = False
            self._settled.set()
        if changed:
            self.screen.run_worker(
                self.screen._sync_native_console_chat_ui(),
                exclusive=False,
                group="console-readiness-publication",
            )


def console_readiness_presentation(function: Callable) -> Callable:
    """Limit disposable configuration data to named synchronous UI refreshes."""

    @wraps(function)
    def wrapped(screen, *args, **kwargs):
        config = _screen_readiness_config(screen)
        if not config or not screen._console_config_snapshot_is_disk_loaded(config):
            return function(screen, *args, **kwargs)
        return ConsoleReadinessConfigProjection.for_screen(screen).run(
            lambda: function(screen, *args, **kwargs)
        )

    return wrapped


def run_console_config_sync(
    sync: Callable[[], None],
    *,
    maintenance_paused: bool,
    request_retry: Callable[[], None],
) -> bool:
    """Keep one checked native config lifetime through a synchronous projection."""
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Backup_Recovery.config_participants import (
        ConfigOperationBusy,
        operation,
    )

    if maintenance_paused:
        request_retry()
        return False
    failure: BaseException | None = None
    entered = False
    try:
        with operation(config, wait_for_locks=False):
            entered = True
            try:
                sync()
            except BaseException as error:  # noqa: BLE001 - re-raised after native owner exit.
                # A UI error must not mark config persistence as failed.
                # Nested config failures retain their own failure state.
                failure = error
    except BaseException as error:
        if not entered and (
            type(error) is ConfigOperationBusy
            or type(error) is RecoveryRequired
            and error.args == ("storage_locally_paused",)
        ):
            request_retry()
            return False
        if failure is not None and error is not failure:
            raise error from failure
        raise
    if failure is not None:
        raise failure
    return True


class ConsoleContextReadSnapshot:
    """Share disposable presentation inputs behind exact owner/revision fences.

    Presentation callers outside the general refresh (spend and credentials)
    share this memo too. Live controller action/dispatch reads never use it.
    """

    def __init__(
        self,
        *,
        max_age: float = 1.0,
        schedule: Callable | None = None,
        refresh: Callable | None = None,
        config_projection: ConsoleReadinessConfigProjection | None = None,
    ) -> None:
        self.config_projection = config_projection
        self.task = None
        self.key = None
        self.value = None
        self.at = 0.0
        self.max_age = max_age
        self.schedule = schedule
        self.refresh = refresh
        self.pending_key = None
        self.lock = asyncio.Lock()

    @classmethod
    def for_screen(cls, screen: Any, *, max_age: float) -> ConsoleContextReadSnapshot:
        """Reuse one screen-owned disposable presentation reader."""
        snapshot = getattr(screen, "_console_context_read_snapshot", None)
        if snapshot is None:
            snapshot = cls(
                max_age=max_age,
                schedule=screen.run_worker,
                refresh=screen._sync_native_console_chat_ui,
                config_projection=(
                    ConsoleReadinessConfigProjection.for_screen(screen)
                    if _screen_readiness_config(screen)
                    and screen._console_config_snapshot_is_disk_loaded(
                        _screen_readiness_config(screen)
                    )
                    else None
                ),
            )
            screen._console_context_read_snapshot = snapshot
        return snapshot

    @contextmanager
    def scope(self) -> Iterator[None]:
        """Keep the refresh owner while retaining its fenced presentation memo."""
        previous = self.task
        self.task = asyncio.current_task()
        try:
            yield
        finally:
            self.task = previous

    def _key(self, controller: Any, session_id: str) -> tuple | None:
        try:
            store = controller.store
            owner = next(item for item in store.sessions() if item.id == session_id)
            return (
                controller,
                store,
                getattr(store, "persistence", None),
                getattr(controller, "_context_repository", None),
                getattr(getattr(controller, "_context_repository", None), "db", None),
                getattr(getattr(store, "persistence", None), "db", None),
                getattr(getattr(controller, "app", None), "chachanotes_db", None),
                store.active_session_id,
                session_id,
                id(owner),
                owner,
                owner.persisted_conversation_id,
                owner.workspace_id,
                getattr(owner, "active_run_id", None),
                owner.context_policy_overrides,
                store.payload_revision(session_id),
                store.display_projection_revision(session_id),
                store.conversation_context_epoch(session_id),
                store.session_settings_revision(session_id),
                store.session_context_summary(session_id),
                controller.run_state_for(session_id).status,
                self.config_projection._key() if self.config_projection else None,
            )
        except (AttributeError, KeyError, StopIteration):
            return None

    def _cache_key(self, controller: Any, session_id: str) -> tuple | None:
        owner = self._key(controller, session_id)
        projection = self.config_projection
        if owner is None or projection is None:
            return owner
        if (
            projection.key != projection._key()
            or projection.value is None
            or getattr(projection.screen, "_console_chat_store", None)
            is not controller.store
            or projection.key[5] != session_id
        ):
            return None
        return (*owner, projection.context_policy)

    async def warm(self, controller: Any, session_id: str) -> bool:
        """Publish only a result whose captured owner survived the await."""
        async with self.lock:
            return await self._warm(controller, session_id)

    async def _warm(self, controller: Any, session_id: str) -> bool:
        projection = self.config_projection
        if projection is not None and not await projection.warm():
            return False
        key = self._cache_key(controller, session_id)
        if key is None:
            return False
        if (
            key is not None
            and key == self.key
            and self.value is not None
            and time.monotonic() - self.at < self.max_age
        ):
            return True
        if key != self.key:
            self.key = self.value = None
        read = getattr(controller, "context_control_presentation_inputs", None)
        if not callable(read):
            return False
        try:
            if projection is not None:
                value = await read(
                    session_id, _presentation_global_overrides=projection.context_policy
                )
            else:
                value = await read(session_id)
        except Exception:  # noqa: BLE001 - the existing live reader owns error policy.
            return False
        if key is None or key != self._cache_key(controller, session_id):
            return False
        self.key, self.value = key, value
        self.at = time.monotonic()
        return True

    def inputs(self, controller: Any, session_id: str) -> tuple:
        """Read cheap UI state, scheduling finite work for cold/expired owners."""
        if self.config_projection is not None:
            self.config_projection.run(lambda: None)
        key = self._cache_key(controller, session_id)
        read = getattr(controller, "context_control_presentation_inputs", None)
        if not callable(read):
            return controller.context_control_inputs(session_id)
        current = key is not None and self.key == key and self.value is not None
        if not current or time.monotonic() - self.at >= self.max_age:
            if (
                self.schedule is not None
                and key is not None
                and self.pending_key != key
            ):
                self.pending_key = key
                self.schedule(
                    self._refresh(controller, session_id, key),
                    exclusive=False,
                    group="console-context-presentation",
                )
        if current:
            # Expiry keeps the last rendered presentation for this exact owner
            # until its finite refresh publishes. It is never send authority.
            return self.value
        owner = next(
            (item for item in controller.store.sessions() if item.id == session_id),
            None,
        )
        if owner is None:
            raise KeyError(session_id)
        return owner.context_policy_overrides, None, None

    async def _refresh(self, controller: Any, session_id: str, key: tuple) -> None:
        previous = self.key, self.value
        try:
            async with self.lock:
                published = key == self._cache_key(
                    controller, session_id
                ) and await self._warm(controller, session_id)
        finally:
            if self.pending_key == key:
                self.pending_key = None
        if (
            published
            and previous != (self.key, self.value)
            and self.refresh is not None
        ):
            self.schedule(
                self.refresh(), exclusive=False, group="console-context-publication"
            )


def fold_system_prompt(system_prompt: str | None, greeting: str) -> str:
    """Fold the seeded greeting exactly as the provider send path does.

    Args:
        system_prompt: Configured system text, if any.
        greeting: Seeded assistant greeting folded into system context.

    Returns:
        Combined system text.
    """
    return fold_greeting_into_system_prompt(system_prompt or "", greeting)


@dataclass(frozen=True, slots=True)
class ConsoleSpendHistoryProjection:
    """Message ownership split for the next request and realized Current."""

    request_ids: frozenset[str]
    current_ids: frozenset[str]


def _remote_active_user_id(
    messages: Sequence[ConsoleChatMessage],
    recovery: ConsoleDispatchRecoveryState | None,
) -> str | None:
    if recovery is None or recovery.kind not in _REMOTE_RECOVERY_KINDS:
        return None
    assistant = next(
        (
            message
            for message in messages
            if message.role is ConsoleMessageRole.ASSISTANT
            and message.persisted_message_id == recovery.assistant_message_id
        ),
        None,
    )
    if assistant is None or assistant.parent_message_id is None:
        return None
    return next(
        (
            message.id
            for message in messages
            if message.role is ConsoleMessageRole.USER
            and message.persisted_message_id == assistant.parent_message_id
        ),
        None,
    )


def build_console_spend_history_projection(
    messages: Sequence[ConsoleChatMessage],
    recovery: ConsoleDispatchRecoveryState | None,
    preparation: ConsoleTurnPreparation | None,
    run_status: ConsoleRunStatus,
    has_submit_task: bool,
) -> ConsoleSpendHistoryProjection:
    """Project provider-request and billed-history rows without reading media.

    Args:
        messages: Ordered transcript rows for the active session.
        recovery: Recovery ownership, whose checkpoint uses persisted IDs.
        preparation: Local preparation ownership using transient message IDs.
        run_status: Current provider-run lifecycle state.
        has_submit_task: Whether a submission is currently in progress.

    Returns:
        Transient IDs admitted to request context and settled spend history.
    """
    request_excluded_user_id: str | None = None
    current_excluded_user_id: str | None = None
    checkpoint = recovery.checkpoint if recovery is not None else None
    remote_user = _remote_active_user_id(messages, recovery)
    if checkpoint is not None:
        current_excluded_user_id = next(
            (
                message.id
                for message in messages
                if message.role is ConsoleMessageRole.USER
                and message.persisted_message_id == checkpoint.user_message_id
            ),
            checkpoint.user_message_id,
        )
        if recovery.kind not in _ACCEPTED_RECOVERY_KINDS:
            request_excluded_user_id = current_excluded_user_id
    elif remote_user is not None:
        current_excluded_user_id = remote_user
    elif preparation is not None and preparation.transient_user_message_id is not None:
        current_excluded_user_id = preparation.transient_user_message_id
        if preparation.state not in _ACCEPTED_PREPARATION_STATES:
            request_excluded_user_id = preparation.transient_user_message_id
    elif (
        run_status is ConsoleRunStatus.VALIDATING
        and has_submit_task
        and messages
        and messages[-1].role is ConsoleMessageRole.USER
    ):
        request_excluded_user_id = current_excluded_user_id = messages[-1].id

    request_ids: set[str] = set()
    current_ids: set[str] = set()
    seen_user = False
    for message in messages:
        if message.role not in {ConsoleMessageRole.USER, ConsoleMessageRole.ASSISTANT}:
            continue
        excluded_request = message.id == request_excluded_user_id
        excluded_current = message.id == current_excluded_user_id
        if message.status == "failed":
            if (
                not excluded_current
                and message.role is ConsoleMessageRole.ASSISTANT
                and seen_user
            ):
                current_ids.add(message.id)
            continue
        if _is_empty_transcript_row(message):
            continue
        if not seen_user and message.role is ConsoleMessageRole.ASSISTANT:
            continue
        if message.role is ConsoleMessageRole.USER:
            seen_user = True
        provider_eligible = not (
            message.role is ConsoleMessageRole.ASSISTANT
            and not assistant_state_allows_provider_history(
                state=message.assistant_generation_state,
                has_valid_continuation=(
                    isinstance(
                        message.provider_continuation, ProviderContinuationCheckpoint
                    )
                    and message.provider_continuation.state == "active"
                ),
                content=message.content,
            )
        )
        if (
            not excluded_current
            and message.role is ConsoleMessageRole.ASSISTANT
            and message.status == "stopped"
        ):
            current_ids.add(message.id)
        if not provider_eligible:
            continue
        if not excluded_request:
            request_ids.add(message.id)
        if not excluded_current:
            current_ids.add(message.id)
    return ConsoleSpendHistoryProjection(frozenset(request_ids), frozenset(current_ids))


def build_console_current_cost_messages(
    messages: Sequence[ConsoleChatMessage],
    current_ids: Collection[str],
) -> list[ConsoleChatMessage]:
    """Return settled cost rows without estimating input already in real usage.

    Args:
        messages: Ordered session transcript.
        current_ids: Transient IDs admitted to settled spend history.

    Returns:
        Settled rows, omitting user estimates covered by assistant usage.
    """
    rows = [
        message
        for message in messages
        if message.id in current_ids
        and getattr(message, "status", "complete") not in {"pending", "streaming"}
    ]
    accounted_users: set[str] = set()
    current_user: ConsoleChatMessage | None = None
    for message in rows:
        if message.role is ConsoleMessageRole.USER:
            current_user = message
        elif (
            message.role is ConsoleMessageRole.ASSISTANT
            and message.usage is not None
            and current_user is not None
        ):
            accounted_users.add(current_user.id)
    return [message for message in rows if message.id not in accounted_users]


def build_console_context_messages(
    messages: Sequence[ConsoleChatMessage],
    request_ids: Collection[str] | None,
    draft_text: str,
) -> list[dict[str, str]]:
    """Return lifecycle-filtered text rows plus the mounted draft.

    Args:
        messages: Ordered session transcript.
        request_ids: Admitted transient IDs, or None to include all rows.
        draft_text: Current composer text.

    Returns:
        Role/content dictionaries with a nonblank draft appended.
    """
    rows = [
        {
            "role": str(getattr(message.role, "value", message.role)),
            "content": message.content,
        }
        for message in messages
        if request_ids is None or message.id in request_ids
    ]
    if draft_text.strip():
        rows.append({"role": "user", "content": draft_text})
    return rows


def build_console_next_send_projection(
    has_historical_media: bool,
    has_pending_attachments: bool,
    request_tokens: int | None,
    input_per_mtok: float | None,
    draft_text: str,
) -> ConsoleNextSendSpendState:
    """Build the input-only forecast from text and admitted media metadata.

    Args:
        has_historical_media: Whether provider-admitted history carries media,
            after lifecycle filtering, capability checks, and image budgeting.
        has_pending_attachments: Whether the draft carries staged media.
        request_tokens: Estimated next-request text tokens, if available.
        input_per_mtok: Uncached input price per million tokens, if known.
        draft_text: Composer text before canonical validation.

    Returns:
        Additional input-charge estimate or an explicit unavailable state.
    """
    validated_draft, validation_error = validate_console_draft(
        draft_text, allow_empty=True
    )
    if validation_error is not None:
        return ConsoleNextSendSpendState(
            "unavailable",
            "On next send: unavailable\n"
            "This message cannot be sent until the draft is corrected.",
        )
    return build_console_next_send_spend_state(
        request_tokens=request_tokens,
        input_per_mtok=input_per_mtok,
        sendable_text=bool(validated_draft.strip()),
        has_media=has_pending_attachments or has_historical_media,
    )


def console_rate_limit_line(provider_key: str) -> str | None:
    """The remaining rate limit the provider's last response reported (TASK-28229).

    Args:
        provider_key: The session provider's config key.

    Returns:
        The tooltip line, or ``None`` when the provider sent no rate-limit
        headers in this process.
    """
    entry = latest_rate_limit_headers(provider_key)
    if entry is None:
        return None
    # Imported only once an entry exists, so it is never loaded before the
    # UI is ready (Tests/Performance/test_ui_ready_module_census.py).
    from ...Chat.provider_rate_limits import format_rate_limit_line

    captured_at, headers = entry
    return format_rate_limit_line(headers, captured_at)


def build_console_spend_cost_state(
    snapshot: ConsoleCostSnapshot,
    cache_state: ConsoleCacheState,
    break_reason: str | None,
    projected_delta_usd: float | None,
    ttl_remaining_s: float | None,
    pricing_as_of: str | None,
    pricing_available: bool,
    context_state: ConsoleContextControlState | None,
    has_historical_media: bool,
    has_pending_attachments: bool,
    input_per_mtok: float | None,
    draft_text: str,
    rate_limit_line: str | None = None,
) -> ConsoleCostState:
    """Compose Current and next-send display state from captured pure inputs.

    Args:
        snapshot: Settled current-spend totals and availability.
        cache_state: Observed prompt-cache status.
        break_reason: Safe description of a cache invalidation, if any.
        projected_delta_usd: Estimated extra charge from cache invalidation.
        ttl_remaining_s: Remaining cache lifetime in seconds, if known.
        pricing_as_of: Pricing catalog timestamp, if known.
        pricing_available: Whether the selected model has known pricing.
        context_state: Context usage presentation, if available.
        has_historical_media: Whether admitted request history includes media.
        has_pending_attachments: Whether the draft carries staged media.
        input_per_mtok: Uncached input price per million tokens, if known.
        draft_text: Current composer text before canonical validation.
        rate_limit_line: The provider's remaining rate limit, if reported.

    Returns:
        Current spend, optionally combined with context and next-send copy.
    """
    empty_priced = (
        snapshot.available
        and snapshot.row_count == 0
        and snapshot.fleet_tokens == 0
        and pricing_available
    )
    if empty_priced:
        snapshot = replace(snapshot, total_usd=0.0, pricing_known=True)
    current = build_cost_state(
        snapshot,
        cache_state=cache_state,
        break_reason=break_reason,
        projected_delta_usd=projected_delta_usd,
        ttl_remaining_s=ttl_remaining_s,
        pricing_as_of=pricing_as_of,
    )
    if empty_priced:
        current = replace(current, label="$0.00", compact_label="$0.00")
    if context_state is None:
        return current
    next_send = build_console_next_send_projection(
        has_historical_media,
        has_pending_attachments,
        context_state.request_tokens,
        input_per_mtok,
        draft_text,
    )
    return build_console_context_cost_state(
        context_state, current, next_send, rate_limit_line
    )


@dataclass(slots=True, kw_only=True)
class ConsoleDraftSpendRefresh:
    """Own the single coalesced timer used for idle draft spend refreshes."""

    schedule_timer: Callable[[float, Callable[[], None]], Any]
    sync_settings_summary: Callable[[], None]
    sync_cost_chip: Callable[[], None]
    delay_seconds: float = 0.2
    timer: Any | None = None

    def schedule(self) -> None:
        """Replace any pending timer with one coalesced refresh."""
        self.stop()
        self.timer = self.schedule_timer(self.delay_seconds, self.refresh)

    def route_edit(self, *, run_active: bool) -> None:
        """Schedule an idle edit or cancel refreshes during an active run.

        Args:
            run_active: Whether provider work currently owns the draft.
        """
        if run_active:
            self.stop()
        else:
            self.schedule()

    def stop(self) -> None:
        """Cancel the pending refresh timer, if one exists."""
        if self.timer is not None:
            self.timer.stop()
        self.timer = None

    def refresh(self) -> None:
        """Clear timer ownership and refresh context and spend displays."""
        self.timer = None
        self.sync_settings_summary()
        self.sync_cost_chip()
