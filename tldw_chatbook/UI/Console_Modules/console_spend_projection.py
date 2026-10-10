"""Cheap, UI-neutral projections for Console current and next-send spend."""

from __future__ import annotations

from _thread import LockType

import asyncio
import copy
import sys
import threading
import time
import weakref
from collections import OrderedDict
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import partial, wraps
from inspect import getattr_static
from types import FunctionType, MethodType, ModuleType
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

from . import pricing_display as _pricing_display

_PRICING_DISPLAY_SOURCE = _pricing_display._SOURCE_CAPSULE
_DISPLAY_PRICING_CLASS = _pricing_display.DisplayPricingCatalog


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


@dataclass(frozen=True, eq=False)
class _CheckedDisplayProof:
    """Detached display provenance; no operation, lease, or disk authority."""

    projection: Any
    screen: Any
    reader: Callable
    key_reader: Callable
    value: Mapping
    source: tuple[int, str]
    owner: tuple
    config: Any
    participants: Any
    raw: Any
    storage: Any
    aliases: tuple
    participant: Any
    participant_state: Any
    loop: Any
    thread: Any
    at: float
    pricing: Any = None


_checked_display_proofs = weakref.WeakSet()
_standard_readiness_projections = weakref.WeakSet()


@dataclass(frozen=True)
class ConsoleReadinessConfigRead:
    """Source tags captured inside the same checked finite mapping read."""

    source_before: tuple[int, str]
    value: Mapping
    source_after: tuple[int, str]
    context_policy: ConsoleContextPolicyOverrides | None = None
    _display_proof: _CheckedDisplayProof | None = None
    pricing: Any = None


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
        self.pricing = None
        self._display_proof = None
        self._read_request = None
        self._settled = asyncio.Event()
        self._settled.set()

    @classmethod
    def for_screen(cls, screen: Any) -> ConsoleReadinessConfigProjection:
        projection = getattr(screen, "_console_readiness_config_projection", None)
        if projection is None or (
            type(projection) is ConsoleReadinessConfigProjection
            and projection.screen is not screen
        ):

            def read_current():
                from tldw_chatbook import config
                from tldw_chatbook.Backup_Recovery import (
                    config_participants as participants,
                    raw_participants as raw,
                    storage_admission as storage,
                )
                from ..Screens import chat_screen

                request = projection._read_request
                loader = chat_screen.load_settings
                aliases = (
                    config.current_config_identity,
                    config._get_effective_config_path,
                    participants.binding,
                    raw._participant_identity,
                    loader,
                )
                with participants.operation(config) as active:
                    before = participants.checked_config_identity(config, active)
                    value = copy.deepcopy(loader())
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
                    pricing = (
                        _pricing_display.prepare_pricing_display()
                        if _pricing_display_sources_current()
                        else None
                    )
                    after = participants.checked_config_identity(config, active)
                    with storage._lock:
                        participant = raw._states[active].participant
                        participant_state = raw._participant_identity(participant)
                proof = None
                if (
                    type(projection) is ConsoleReadinessConfigProjection
                    and projection in _standard_readiness_projections
                    and projection.read_current is read_current
                    and request is not None
                    and loader is config.load_settings
                    and before == after == request[2][0]
                ):
                    proof = _CheckedDisplayProof(
                        projection,
                        screen,
                        read_current,
                        _standard_readiness_key,
                        value,
                        before,
                        request[2],
                        config,
                        participants,
                        raw,
                        storage,
                        aliases,
                        participant,
                        participant_state,
                        request[0],
                        request[1],
                        time.monotonic(),
                        pricing,
                    )
                    _checked_display_proofs.add(proof)
                return ConsoleReadinessConfigRead(
                    before, value, after, policy, proof, pricing
                )

            projection = cls(screen, read_current=read_current)
            if type(projection) is ConsoleReadinessConfigProjection:
                _standard_readiness_projections.add(projection)
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
        pending_at_entry = self.pending
        key = self._key()
        current = _same_readiness_key(key, self.key) and self.value is not None
        if (
            not current or time.monotonic() - self.at >= self.max_age
        ) and not self.pending:
            self.pending = True
            self._settled.clear()
            refresh = self._refresh(key)
            try:
                screen.run_worker(
                    refresh, exclusive=False, group="console-readiness-config"
                )
            except BaseException as error:
                # Scheduling did not take ownership of this coroutine. A cold
                # presentation read remains retryable; cancellation propagates.
                refresh.close()
                self.pending = False
                self._settled.set()
                if not isinstance(error, Exception):
                    raise
        if not current:
            return False
        previous = getattr(screen, "_console_readiness_projection_active", None)
        previous_pending = getattr(self, "_display_refresh_pending_at_entry", None)
        screen._console_readiness_projection_active = (
            threading.get_ident(),
            self.value,
            self,
        )
        try:
            self._display_refresh_pending_at_entry = pending_at_entry
            with screen._console_derivation_scope():
                return body() is not False
        finally:
            screen._console_readiness_projection_active = previous
            self._display_refresh_pending_at_entry = previous_pending

    async def warm(self) -> bool:
        """Wait for the same checked owner when a modal needs cold display data."""
        key = self._key()
        self.run(lambda: None)
        if self.pending:
            await self._settled.wait()
        return (
            _same_readiness_key(key, self._key())
            and _same_readiness_key(key, self.key)
            and self.value is not None
            and time.monotonic() - self.at < self.max_age
        )

    async def _refresh(self, key: tuple) -> None:
        reader = self.read_current
        self._read_request = asyncio.get_running_loop(), threading.current_thread(), key
        worker = asyncio.create_task(asyncio.to_thread(reader))
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
                or not _same_readiness_key(key, self._key())
            ):
                return
            # Pricing equality is user-replaceable code too. Refuse before
            # comparing or publishing an adapter whose source changed in flight.
            if (self.pricing is not None or result.pricing is not None) and (
                not _pricing_display_sources_current()
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
            if not _same_readiness_key(current_key, key, ignore_settings=True):
                return
            changed = (
                not _same_readiness_key(self.key, current_key)
                or self.value != value
                or self.context_policy != result.context_policy
                or self.pricing != result.pricing
            )
            self.key, self.value, self.at = current_key, value, time.monotonic()
            self.context_policy = result.context_policy
            self.pricing = result.pricing
            proof = result._display_proof
            self._display_proof = None
            if (
                type(proof) is _CheckedDisplayProof
                and proof in _checked_display_proofs
                and proof.projection is self
                and proof.screen is self.screen
                and proof.reader is reader is self.read_current
                and proof.value is value
                and _same_readiness_key(proof.owner, key)
            ):
                # The existing pristine-default handoff may change only this
                # owner's settings revision. Bind the issued display copy to
                # that final exact owner and its original one-second clock.
                published = replace(proof, owner=current_key, at=self.at)
                _checked_display_proofs.add(published)
                self._display_proof = published
        except Exception:  # noqa: BLE001 - remain cold and retryable.
            return
        finally:
            self._read_request = None
            self.pending = False
            self._settled.set()
        if changed:
            publication = self.screen._sync_native_console_chat_ui()
            try:
                self.screen.run_worker(
                    publication,
                    exclusive=False,
                    group="console-readiness-publication",
                )
            except BaseException as error:
                publication.close()
                if not isinstance(error, Exception):
                    raise


def _same_readiness_key(
    left: Any, right: Any, *, ignore_settings: bool = False
) -> bool:
    """Compare semantic tokens only after exact retained owner references."""
    if (
        type(left) is not tuple
        or type(right) is not tuple
        or len(left) != 11
        or len(right) != 11
    ):
        return False
    # App, database, store, session owner, and app mapping are receivers,
    # not values. Keep revision index 7's existing pristine-default exception.
    if any(left[index] is not right[index] for index in (1, 3, 4, 9, 10)):
        return False
    if ignore_settings:
        return (*left[:7], *left[8:]) == (*right[:7], *right[8:])
    return left == right


_standard_readiness_key = ConsoleReadinessConfigProjection._key


def _checked_display_status(projection: Any) -> bool | None:
    """Return fresh display acceptance, owner refusal, or native fallback.

    All checks here are in-memory identity or lexical source checks. The
    coordinator lock protects only installed participant mappings, never IO.
    """
    if type(projection) is not ConsoleReadinessConfigProjection:
        return None
    proof = projection._display_proof
    key_reader = projection._key
    if (
        type(proof) is not _CheckedDisplayProof
        or proof not in _checked_display_proofs
        or projection not in _standard_readiness_projections
        or proof.projection is not projection
        or proof.reader is not projection.read_current
        or proof.value is not projection.value
        or type(key_reader) is not MethodType
        or key_reader.__func__ is not proof.key_reader
        or key_reader.__self__ is not projection
        or proof.thread is not threading.current_thread()
    ):
        return None
    try:
        if asyncio.get_running_loop() is not proof.loop:
            return None
    except RuntimeError:
        return None
    active = getattr(projection.screen, "_console_readiness_projection_active", None)
    if (
        active is None
        or len(active) != 3
        or active[0] != threading.get_ident()
        or active[1] is not proof.value
        or active[2] is not projection
    ):
        return None
    if projection.screen is not proof.screen or getattr(
        projection.screen, "_closing", False
    ):
        return False
    from ..Screens import chat_screen

    config, participants, raw, storage = (
        proof.config,
        proof.participants,
        proof.raw,
        proof.storage,
    )
    if (
        sys.modules.get("tldw_chatbook.config") is not config
        or sys.modules.get("tldw_chatbook.Backup_Recovery.config_participants")
        is not participants
        or sys.modules.get("tldw_chatbook.Backup_Recovery.raw_participants") is not raw
        or sys.modules.get("tldw_chatbook.Backup_Recovery.storage_admission")
        is not storage
        or config._config_participants is not participants
        or not all(
            current is issued
            for current, issued in zip(
                (
                    config.current_config_identity,
                    config._get_effective_config_path,
                    participants.binding,
                    raw._participant_identity,
                    chat_screen.load_settings,
                ),
                proof.aliases,
            )
        )
        or chat_screen.load_settings is not config.load_settings
    ):
        return False
    if not _checked_pricing_current(projection, proof):
        return None
    # These lexical checks include the current generation and raw selected path.
    # An expired same-owner mapping may use the original native fallback; a
    # different owner/source may never enter a body around that old mapping.
    if (
        not _same_readiness_key(projection._key(), projection.key)
        or not _same_readiness_key(projection.key, proof.owner)
        or projection.key[0] != proof.source
    ):
        return False
    bound = participants.binding(config)
    if bound is None or str(bound[1]) != proof.source[1]:
        return False
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    with storage._lock:
        try:
            state = raw._participant_identity(proof.participant)
        except RecoveryRequired as error:
            if error.args != ("raw_participant_not_installed",):
                raise
            return False
        if (
            state is not proof.participant_state
            or state.source() is not config
            or state.owner != "config"
            or str(state.selected) != proof.source[1]
            or state.closed
            or storage._pause is not None
        ):
            return False
    if not _same_readiness_key(projection._key(), projection.key):
        return False
    if projection.at != proof.at:
        return None
    if time.monotonic() - proof.at >= min(1.0, projection.max_age):
        # Defer only a refresh pending before this synchronous display entry;
        # a direct call that starts its own refresh keeps its native fallback.
        return (
            False
            if projection.pending is True
            and getattr(projection, "_display_refresh_pending_at_entry", None) is True
            else None
        )
    return True


class _ContextCapacityDisplayRefused(Exception):
    """A captured metadata source changed before display publication."""


def _original_capacity_method(owner: Any, module: Any, entry: tuple) -> Any:
    name, function, code = entry
    if (
        type(function) is not FunctionType
        or function.__globals__ is not vars(module)
        or function.__code__ is not code
        or getattr_static(type(owner), "__getattribute__")
        is not object.__getattribute__
        or getattr_static(owner, name, None) is not function
    ):
        return None
    method = getattr(owner, name, None)
    if (
        type(method) is not MethodType
        or method.__self__ is not owner
        or method.__func__ is not function
    ):
        return None
    return method


def _context_capacity_display_sources(gateway: Any, app: Any) -> Any:
    """Capture only the standard synchronous metadata route, without IO."""
    module = sys.modules.get("tldw_chatbook.Chat.console_provider_gateway")
    runtime = sys.modules.get("tldw_chatbook.Chat.console_runtime")
    if type(module) is not ModuleType or type(runtime) is not ModuleType:
        return None
    namespace, runtime_namespace = vars(module), vars(runtime)
    originals = namespace.get("_CONTEXT_CAPACITY_DISPLAY_ORIGINALS")
    provider_original = runtime_namespace.get("_PROVIDER_CONFIG_FOR_APP_ORIGINAL")
    if (
        type(originals) is not tuple
        or len(originals) != 3
        or originals[0] is not vars(module)
        or originals[1] is not namespace.get("ConsoleProviderGateway")
        or type(gateway) is not originals[1]
        or getattr_static(type(gateway), "__getattribute__")
        is not object.__getattribute__
        or type(provider_original) is not tuple
        or len(provider_original) != 3
        or provider_original[0] is not vars(runtime)
    ):
        return None
    provider_function, provider_code = provider_original[1:]
    missing = object()
    provider = getattr_static(gateway, "_config_provider", missing)
    cache_lock = getattr_static(gateway, "_context_windows_lock", missing)
    cache = getattr_static(gateway, "_context_windows", missing)
    environ = getattr_static(gateway, "_environ", missing)
    if (
        type(provider_function) is not FunctionType
        or runtime_namespace.get("_provider_config_for_app") is not provider_function
        or provider_function.__globals__ is not vars(runtime)
        or provider_function.__code__ is not provider_code
        or type(provider) is not partial
        or provider.func is not provider_function
        or len(provider.args) != 1
        or provider.args[0] is not app
        or provider.keywords
        or type(cache_lock) is not LockType
        or (environ is not None and type(environ) is not dict)  # noqa: E721 - custom maps stay fresh.
    ):
        return None
    methods = tuple(
        _original_capacity_method(gateway, module, entry) for entry in originals[2]
    )
    if len(methods) != 3 or any(method is None for method in methods):
        return None
    refs = (
        module,
        runtime,
        originals,
        provider_original,
        provider,
        provider.keywords,
        cache_lock,
        environ,
        cache,
    )
    read_cached = None
    if cache is not None:
        cache_module = sys.modules.get("tldw_chatbook.Chat.console_context_window")
        if type(cache_module) is not ModuleType:
            return None
        cache_namespace = vars(cache_module)
        cache_original = cache_namespace.get("_CONTEXT_CAPACITY_CACHE_ORIGINAL")
        if (
            type(cache_original) is not tuple
            or len(cache_original) != 4
            or cache_original[0] is not vars(cache_module)
            or cache_original[1] is not cache_namespace.get("ContextWindowCache")
            or type(cache) is not cache_original[1]
            or getattr_static(type(cache), "__getattribute__")
            is not object.__getattribute__
        ):
            return None
        read_cached = _original_capacity_method(
            cache, cache_module, ("cached", cache_original[2], cache_original[3])
        )
        if read_cached is None:
            return None
        serving_lock = getattr_static(cache, "_lock", missing)
        serving_records = getattr_static(cache, "_cache", missing)
        if (
            type(serving_lock) is not LockType
            or type(serving_records) is not OrderedDict
        ):
            return None
        refs += (cache_module, cache_original, serving_lock, serving_records)
    # Bound method objects are ephemeral; retain their exact function/receiver.
    refs += tuple(
        item for method in methods for item in (method.__func__, method.__self__)
    )
    if read_cached is not None:
        refs += (read_cached.__func__, read_cached.__self__)
    return refs, methods[2], methods[1], cache, read_cached


def cached_context_window_for_display(screen: Any, gateway: Any, settings: Any) -> Any:
    """Use an issued copy for metadata only; custom/direct routes stay fresh."""
    projection = getattr(screen, "_console_readiness_config_projection", None)
    if _checked_display_status(projection) is not True:
        return gateway.cached_context_window(settings)
    proof = projection._display_proof
    captured = _context_capacity_display_sources(gateway, proof.owner[1])
    if captured is None:
        return gateway.cached_context_window(settings)
    refs, display, project_target, cache, read_cached = captured
    value = display(
        settings,
        proof.value,
        cache=cache,
        project_target=project_target,
        read_cached=read_cached,
    )
    current = _context_capacity_display_sources(gateway, proof.owner[1])
    if (
        current is None
        or len(current[0]) != len(refs)
        or any(now is not before for now, before in zip(current[0], refs))
        or projection._display_proof is not proof
        or _checked_display_status(projection) is not True
    ):
        raise _ContextCapacityDisplayRefused()
    return value


def _pricing_display_sources_current() -> bool:
    # Inspect retained original objects before invoking any mutable helper.
    if (
        sys.modules.get(_pricing_display.__name__) is not _pricing_display
        or _pricing_display._SOURCE_CAPSULE is not _PRICING_DISPLAY_SOURCE
        or _pricing_display.DisplayPricingCatalog is not _DISPLAY_PRICING_CLASS
    ):
        return False
    for owner, name, original, code, defaults, kwdefaults in _PRICING_DISPLAY_SOURCE:
        if (
            getattr_static(owner or _pricing_display, name, None) is not original
            or original.__code__ is not code
            or original.__defaults__ is not defaults
            or original.__kwdefaults__ is not kwdefaults
        ):
            return False
    return True


def _checked_pricing_current(projection, proof) -> bool:
    from ...Chat import console_cost_tracker
    from ...LLM_Calls import pricing_catalog
    from ..Screens import chat_screen

    if not _pricing_display_sources_current():
        return False

    snapshot, code, defaults, kwdefaults = console_cost_tracker._COST_SNAPSHOT_SOURCE
    return (
        projection.pricing is proof.pricing
        and _pricing_display.pricing_display_current(proof.pricing)
        and chat_screen.get_pricing_catalog is pricing_catalog.get_pricing_catalog
        and console_cost_tracker.get_pricing_catalog
        is pricing_catalog.get_pricing_catalog
        and chat_screen.build_cost_snapshot
        is snapshot
        is console_cost_tracker.build_cost_snapshot
        and snapshot.__code__ is code
        and snapshot.__defaults__ is defaults
        and snapshot.__kwdefaults__ is kwdefaults
    )


def pricing_catalog_for_display(screen: Any, get_catalog: Callable) -> Any:
    """Supply detached prices only inside the qualified synchronous UI route."""
    projection = getattr(screen, "_console_readiness_config_projection", None)
    if _checked_display_status(projection) is True:
        return projection.pricing
    return get_catalog()


def pricing_snapshot_options(catalog: Any) -> dict[str, Any]:
    """Keep original snapshot/custom calls unchanged outside checked display."""
    from .pricing_display import DisplayPricingCatalog

    return (
        {"pricing_catalog": catalog} if type(catalog) is DisplayPricingCatalog else {}
    )


def _run_checked_display_sync(
    projection: Any,
    sync: Callable[[], None],
    request_retry: Callable[[], None],
) -> bool | None:
    status = _checked_display_status(projection)
    if status is None:
        return None
    if not status:
        request_retry()
        return False
    failure = None
    try:
        sync()
    except BaseException as error:  # noqa: BLE001 - preserve the body exception.
        failure = error
    try:
        accepted = _checked_display_status(projection) is True
        if not accepted:
            request_retry()
    except BaseException as error:
        if failure is not None and error is not failure:
            raise error from failure
        raise
    if type(failure) is _ContextCapacityDisplayRefused:
        if accepted:
            request_retry()
        return False
    if failure is not None:
        raise failure
    return accepted


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
    checked_projection: ConsoleReadinessConfigProjection | None = None,
) -> bool:
    """Keep fresh native authority; accept only issued warm display provenance."""
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Backup_Recovery.config_participants import (
        ConfigOperationBusy,
        operation,
    )

    if maintenance_paused:
        request_retry()
        return False
    displayed = _run_checked_display_sync(checked_projection, sync, request_retry)
    if displayed is not None:
        return displayed
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
            full_refresh = screen._sync_native_console_chat_ui
            poll_refresh = getattr(screen, "_sync_console_poll_display_ui", None)
            refresh = full_refresh
            if callable(poll_refresh):

                async def refresh():
                    current_full = getattr(screen, "_sync_native_console_chat_ui", None)
                    same_full = current_full is full_refresh or (
                        type(current_full) is MethodType
                        and type(full_refresh) is MethodType
                        and current_full.__self__ is full_refresh.__self__
                        and current_full.__func__ is full_refresh.__func__
                    )
                    # A running pass may already have consumed the old context.
                    # Preserve its trailing FULL demand and captured custom callback.
                    if not same_full or getattr(
                        screen, "_console_sync_in_progress", False
                    ):
                        return await full_refresh()
                    return await poll_refresh()

            snapshot = cls(
                max_age=max_age,
                schedule=screen.run_worker,
                refresh=refresh,
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
        from ...Chat.console_chat_controller import (
            _CONSOLE_CONTEXT_PRESENTATION_READERS,
            ConsoleChatController,
        )
        from ...Chat.console_received_dispatch import stock_native_methods

        try:
            store = controller.store
            owner = next(item for item in store.sessions() if item.id == session_id)
            dependency = (
                ("stock-echo", controller._held_send_echo_id(session_id))
                if type(controller) is ConsoleChatController
                and stock_native_methods(
                    controller, _CONSOLE_CONTEXT_PRESENTATION_READERS
                )
                else (
                    "legacy-lifecycle",
                    getattr(owner, "active_run_id", None),
                    controller.run_state_for(session_id).status,
                )
            )
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
                dependency,
                owner.context_policy_overrides,
                store.payload_revision(session_id),
                store.display_projection_revision(session_id),
                store.conversation_context_epoch(session_id),
                store.session_settings_revision(session_id),
                store.session_context_summary(session_id),
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
            not _same_readiness_key(projection.key, projection._key())
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
