"""Finite lazy skill-trust source and publication for the explicit domain owner."""

from __future__ import annotations

import asyncio
import inspect
import os
import sys
import threading
from dataclasses import dataclass
from types import MethodType
from typing import Any

from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

from .console_domain_services import (
    ConsoleDomainServices,
    _definition,
    _definition_current,
)


@dataclass(frozen=True, slots=True)
class DomainSkillTrustSource:
    domain_services: ConsoleDomainServices
    scope: Any
    local: Any
    source_current: Any
    current: Any
    loop: Any
    thread: Any
    runtime_reads: set


def capture_domain_skill_trust_source(domain, scope):
    """Capture exact stock local/config collaborators without consulting App."""
    from tldw_chatbook import config
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
    from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService

    from .console_configuration_preparation import _resident_skill_wiring

    checked = _resident_skill_wiring()
    if checked is None:
        return None
    wiring, metadata = checked
    domain.require_current()
    sources = domain.sources
    if (
        sources.skill_wiring_source is not wiring
        or type(scope) is not SkillsScopeService
    ):
        return None
    local = scope.local_service
    if type(local) is not LocalSkillsService or sources.local_skills is not local:
        return None
    factory = vars(local).get("_trust_service_factory")
    if (
        type(factory) is not MethodType
        or factory.__self__ is not domain
        or factory.__func__ is not ConsoleDomainServices.read_skill_trust
        or type(domain._trust_lock) is not asyncio.Lock
        or scope.policy_enforcer is not local.policy_enforcer
        or scope.policy_enforcer is not sources.policy_enforcer
    ):
        return None
    runtime = domain.runtime
    if runtime is None or runtime._disposed or runtime.domain_services is not domain:
        raise RuntimeError("console_skill_trust_runtime_closed")
    source_records = (
        wiring._CONSOLE_SKILL_WIRING_SOURCE,
        wiring._CONSOLE_SKILL_HELPER_SOURCE,
        sys.modules[
            "tldw_chatbook.Skills_Interop.skills_scope_service"
        ]._CONSOLE_SKILL_CONTEXT_SOURCE,
        sys.modules[
            "tldw_chatbook.Skills_Interop.local_skills_service"
        ]._CONSOLE_SKILL_CONTEXT_SOURCE,
        config._COMPACT_MODEL_CONFIG_SOURCE,
    )
    if not all(
        wiring._console_skill_source_current(metadata, row) for row in source_records
    ):
        return None
    scope_values = vars(scope)
    local_values = vars(local)
    if any(name in scope_values for name in wiring._CONSOLE_SKILL_SCOPE_SLOTS) or any(
        name in local_values for name in wiring._CONSOLE_SKILL_LOCAL_SLOTS
    ):
        return None
    scope_refs = tuple(
        (name, scope_values.get(name))
        for name in ("local_service", "server_service", "policy_enforcer")
    )
    local_refs = tuple(
        (name, local_values.get(name))
        for name in (
            "_trust_service_factory",
            "store_dir",
            "skills_dir",
            "policy_enforcer",
        )
    )
    lock, reads = domain._trust_lock, runtime._preparation_reads
    identity = config.current_config_identity()
    cache, posture = (
        vars(config).get("_SETTINGS_CACHE"),
        vars(config).get("_SETTINGS_CACHE_POSTURE"),
    )
    selector = tuple(os.environ.get(key) for key in _SELECTOR_KEYS)
    loop, thread = asyncio.get_running_loop(), threading.current_thread()
    association = domain.presentation.association

    def source_current():
        try:
            domain.require_current(sources)
            domain.presentation.require_association(association)
        except RecoveryRequired:
            return False
        return (
            domain_skill_definitions_current()
            and _resident_skill_wiring() == checked
            and all(
                wiring._console_skill_source_current(metadata, row)
                for row in source_records
            )
            and domain.runtime is runtime
            and runtime.domain_services is domain
            and not runtime._disposed
            and runtime._preparation_reads is reads
            and domain._trust_lock is lock
            and sources.skills_scope is scope
            and sources.local_skills is local
            and vars(scope) is scope_values
            and vars(local) is local_values
            and all(scope_values.get(name) is value for name, value in scope_refs)
            and all(local_values.get(name) is value for name, value in local_refs)
            and not any(
                name in scope_values for name in wiring._CONSOLE_SKILL_SCOPE_SLOTS
            )
            and not any(
                name in local_values for name in wiring._CONSOLE_SKILL_LOCAL_SLOTS
            )
            and sys.modules.get("tldw_chatbook.config") is config
            and config.current_config_identity() == identity
            and vars(config).get("_SETTINGS_CACHE") is cache
            and vars(config).get("_SETTINGS_CACHE_POSTURE") is posture
            and tuple(os.environ.get(key) for key in _SELECTOR_KEYS) == selector
        )

    def current():
        try:
            return (
                asyncio.get_running_loop() is loop
                and threading.current_thread() is thread
                and source_current()
            )
        except (AttributeError, TypeError, ValueError):
            return False

    if not current():
        return None
    return DomainSkillTrustSource(
        domain, scope, local, source_current, current, loop, thread, reads
    )


def record_domain_skill_trust_candidate(domain, candidate):
    """Witness only the extracted initializer's actual just-published candidate."""
    from tldw_chatbook.Skills_Interop.skill_trust_models import (
        _SkillTrustPublisher,
        _skill_publisher_identity,
        _skill_trust_publisher,
    )

    marker = _skill_trust_publisher.get()
    if type(marker) is not _SkillTrustPublisher or not marker.active:
        return
    if _skill_publisher_identity() != (marker.thread_id, marker.task):
        return
    source = marker.preparation.trust_source
    if (
        type(source) is not DomainSkillTrustSource
        or source.domain_services is not domain
        or candidate is None
    ):
        return
    frame = sys._getframe(1)
    if not (
        frame.f_locals.get("self") is domain
        and (
            (
                frame.f_code is ConsoleDomainServices.read_skill_trust.__code__
                and frame.f_locals.get("candidate") is candidate
            )
            or (
                frame.f_code is ConsoleDomainServices.ensure_skill_trust.__code__
                and frame.f_locals.get("service") is candidate
            )
        )
        and domain._trust is candidate
    ):
        return
    marker.candidate = candidate


def domain_skill_publisher_guard(source):
    domain, local = source.domain_services, source.local
    bindings = tuple(
        (owner, name, inspect.getattr_static(owner, name))
        for owner, name in (
            (ConsoleDomainServices, "read_skill_trust"),
            (ConsoleDomainServices, "build_skill_trust"),
            (ConsoleDomainServices, "ensure_skill_trust"),
            (type(local), "trust_service"),
        )
    )
    functions = tuple(
        _definition(value.fget if isinstance(value, property) else value)
        for _, _, value in bindings
    )
    fields = tuple(
        (vars(local), name, vars(local).get(name))
        for name in (
            "_trust_service_factory",
            "store_dir",
            "skills_dir",
            "policy_enforcer",
        )
    )
    environment = os.environ
    data = vars(environment).get("_data")
    if type(environment) is not os._Environ or type(data) is not dict:
        return None
    selector = tuple(
        (environment.encodekey(key), data.get(environment.encodekey(key)))
        for key in _SELECTOR_KEYS
    )
    return (
        domain.sources,
        domain._trust_lock,
        bindings,
        functions,
        fields,
        environment,
        data,
        selector,
    )


def domain_skill_publisher_guard_current(source, guard):
    sources, lock, bindings, functions, fields, environment, data, selector = guard
    return (
        source.domain_services.sources is sources
        and source.domain_services._trust_lock is lock
        and all(
            inspect.getattr_static(owner, name) is value
            for owner, name, value in bindings
        )
        and all(_definition_current(record) for record in functions)
        and all(container.get(name) is value for container, name, value in fields)
        and os.environ is environment
        and vars(environment).get("_data") is data
        and all(data.get(key) == value for key, value in selector)
    )


_SELECTOR_KEYS = (
    "TLDW_CONFIG_PATH",
    "HOME",
    "USERPROFILE",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
)


def domain_skill_definitions_current():
    return all(
        globals().get(name) is record[0] and _definition_current(record)
        for name, record in _DOMAIN_SKILL_DEFINITIONS
    )


# The extracted builder and its finite definition records are installed below.


def build_domain_skill_trust(_source_wiring) -> Any:
    """Build the skill trust service. Performs OS keyring discovery.

    Split out of the eager wiring (TASK-21111(b)): it is the only part
    of the local skills stack that touches the keyring -- twice, once
    for the rollback marker store's secure-backend probe and once for
    the trust key cache -- and nothing at startup asks a trust question.
    Deferring the whole SERVICE was not enough on its own: the Console's
    agent bridge takes the skills scope facade during Chat screen mount,
    which merely relocated the discovery from ``__init__`` to mount.
    ``LocalSkillsService`` therefore takes this as a FACTORY and calls it
    on the first trust decision.
    """
    local_skills_store_dir = _source_wiring.default_local_skills_store_dir(
        _source_wiring.get_user_data_dir()
    )
    trust_store_dir = _source_wiring.default_trust_store_dir(local_skills_store_dir)
    from ..Skills_Interop.recovery_activation import is_recovered

    recovered_skills = is_recovered(local_skills_store_dir / "skills", trust_store_dir)
    if recovered_skills:
        from ..Skills_Interop.skill_trust_store import (
            FileSkillTrustGenerationMarkerStore,
        )

        skill_trust_marker_store = FileSkillTrustGenerationMarkerStore(
            trust_store_dir / _source_wiring._SKILL_TRUST_MARKER_FILENAME,
            store_dir=trust_store_dir,
        )
        reduced_rollback_protection = True
        skill_key_cache = None
    else:
        trust_account_scope = _source_wiring.skill_trust_account_scope(trust_store_dir)
        skill_trust_marker_store, reduced_rollback_protection = (
            _source_wiring.build_skill_trust_marker_store_with_fallback(
                fallback_marker_path=trust_store_dir
                / _source_wiring._SKILL_TRUST_MARKER_FILENAME,
                store_dir=trust_store_dir,
                account_scope=trust_account_scope,
            )
        )
        skill_key_cache = _source_wiring.build_default_skill_trust_key_cache(
            account_scope=trust_account_scope
        )
    return _source_wiring.SkillTrustService(
        skills_dir=local_skills_store_dir / "skills",
        trust_store=_source_wiring.SkillTrustStore(
            store_dir=trust_store_dir,
            marker_store=skill_trust_marker_store,
        ),
        key_cache=skill_key_cache,
        keyring_convenience_enabled=False,
        reduced_rollback_protection=reduced_rollback_protection,
    )


_DOMAIN_SKILL_DEFINITIONS = tuple(
    (name, _definition(globals()[name]))
    for name in (
        "capture_domain_skill_trust_source",
        "record_domain_skill_trust_candidate",
        "domain_skill_publisher_guard",
        "domain_skill_publisher_guard_current",
        "build_domain_skill_trust",
    )
)
