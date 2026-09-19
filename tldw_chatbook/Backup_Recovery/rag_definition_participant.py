"""Finite profile/config operations and read-only retained definition checks."""

import asyncio
import sys
import threading
import time
from contextlib import contextmanager
from functools import wraps

from .bootstrap import RecoveryRequired
from .profile_paths import effective_config_path, lexical_path, user_data_dir


class _DefinitionToken:
    """Explicit accepted continuation, retained through every entered scope."""

    def __init__(self, owner):
        self._owner = owner
        self._scopes = 0
        self._closing = False

    @contextmanager
    def scope(self):
        owner = self._owner
        with owner._condition:
            if self not in owner._tokens or self._closing:
                raise RecoveryRequired("rag_definition_action_inactive")
            self._scopes += 1
        previous = getattr(owner._local, "accepted", False)
        owner._local.accepted = True
        try:
            yield
        finally:
            owner._local.accepted = previous
            with owner._condition:
                self._scopes -= 1
                if self._closing:
                    self.close()

    def close(self):
        owner = self._owner
        with owner._condition:
            self._closing = True
            if self in owner._tokens and not self._scopes:
                owner._tokens.remove(self)
                owner._active -= 1
                owner._condition.notify_all()


class DefinitionParticipant:
    """Settle installed definition writes and protect experiment memory."""

    def __init__(self):
        self._condition = threading.Condition()
        self._local = threading.local()
        self._active = 0
        self._tokens = set()
        self._experiment_transitions = 0
        self._closed = False

    def reserve(self, parent=None):
        """Admit root work or an explicit still-live accepted descendant."""
        with self._condition:
            if parent is not None:
                if (
                    type(parent) is not _DefinitionToken
                    or parent not in self._tokens
                    or parent._closing
                ):
                    raise RecoveryRequired("rag_definition_action_inactive")
            elif self._closed:
                raise RecoveryRequired("rag_definition_operations_paused")
            token = _DefinitionToken(self)
            self._tokens.add(token)
            self._active += 1
            return token

    def operation(self, function):
        @wraps(function)
        def invoke(*args, **kwargs):
            if getattr(self._local, "accepted", False):
                return function(*args, **kwargs)
            with self._condition:
                if self._closed:
                    raise RecoveryRequired("rag_definition_operations_paused")
                self._active += 1
            self._local.accepted = True
            try:
                return function(*args, **kwargs)
            finally:
                self._local.accepted = False
                with self._condition:
                    self._active -= 1
                    self._condition.notify_all()

        return invoke

    def experiment_operation(self, function):
        """Keep experiment transitions atomic with the maintenance intake check."""

        @wraps(function)
        def invoke(*args, **kwargs):
            if getattr(self._local, "experiment", False):
                return function(*args, **kwargs)
            with self._condition:
                # Ordinary accepted work does not authorize new experiment state.
                if self._closed:
                    raise RecoveryRequired("rag_definition_operations_paused")
                self._experiment_transitions += 1
            self._local.experiment = True
            try:
                return function(*args, **kwargs)
            finally:
                self._local.experiment = False
                with self._condition:
                    self._experiment_transitions -= 1
                    self._condition.notify_all()

        return invoke

    def _maintenance_close_admission(self):
        with self._condition:
            loaded = sys.modules.get("tldw_chatbook.RAG_Search.config_profiles")
            manager_type = vars(loaded).get("ConfigProfileManager") if loaded else None
            if self._experiment_transitions or (
                manager_type is not None
                and any(
                    manager._current_experiment is not None
                    for manager in manager_type.live_instances()
                )
            ):
                # Do not seal intake and thereby reject an ongoing search's
                # metric delivery. The user must end/save the experiment first.
                raise RecoveryRequired("rag_definition_experiment_unsaved")
            self._closed = True

    async def _maintenance_drain(self, deadline):
        while True:
            with self._condition:
                if not self._closed:
                    raise RecoveryRequired("rag_definition_intake_open")
                if not self._active:
                    return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(remaining, 0.01))

    def _maintenance_resume(self):
        with self._condition:
            self._closed = False


participant = DefinitionParticipant()
definition_operation = participant.operation
definition_experiment_operation = participant.experiment_operation


def retained_issues(app):
    """Inspect loaded owners without constructing managers or securing paths."""
    from .unsaved_editors import UnsavedEditor

    label = "RAG definitions"
    loaded = sys.modules.get("tldw_chatbook.RAG_Search.config_profiles")
    if loaded is None:
        return ()
    manager_type = vars(loaded).get("ConfigProfileManager")
    managers = list(manager_type.live_instances()) if manager_type is not None else []
    manager = vars(loaded).get("_GLOBAL_PROFILE_MANAGER")
    if manager is not None:
        managers.append(manager)
    ingestion = sys.modules.get("tldw_chatbook.RAG_Search.ingestion_indexing")
    services = [vars(app).get("_rag_service")]
    if ingestion is not None:
        services.append(vars(ingestion).get("_shared_service"))
    enhanced = sys.modules.get(
        "tldw_chatbook.RAG_Search.simplified.enhanced_rag_service_v2"
    )
    for service in services:
        if service is None:
            continue
        if enhanced is None or type(service) is not vars(enhanced).get(
            "EnhancedRAGServiceV2"
        ):
            plain = sys.modules.get("tldw_chatbook.RAG_Search.simplified.rag_service")
            legacy = sys.modules.get(
                "tldw_chatbook.RAG_Search.simplified.enhanced_rag_service"
            )
            if (
                plain is not None
                and type(service) is vars(plain).get("RAGService")
                or legacy is not None
                and type(service) is vars(legacy).get("EnhancedRAGService")
            ):
                continue
            return (UnsavedEditor(label, "unknown"),)
        managers.append(vars(service).get("profile_manager"))
    if not managers:
        return ()
    config = sys.modules.get("tldw_chatbook.config")
    if config is None:
        return (UnsavedEditor(label, "unknown"),)
    values = vars(config).get("_CONFIG_CACHE")
    if (
        not isinstance(values, dict)
        or vars(config).get("_CONFIG_CACHE_SOURCE") != effective_config_path()
    ):
        return (UnsavedEditor(label, "unknown"),)
    try:
        expected = user_data_dir(values) / "rag_profiles"
        issues = []
        for manager in {id(value): value for value in managers}.values():
            if type(manager) is not vars(loaded).get("ConfigProfileManager"):
                return (UnsavedEditor(label, "unknown"),)
            if not isinstance(manager._experiment_results, dict):
                return (UnsavedEditor(label, "unknown"),)
            if lexical_path(manager.profiles_dir) != expected:
                issues.append(UnsavedEditor(label, "alternate_source_unqualified"))
            experiment = manager._current_experiment
            if experiment is not None:
                if type(experiment) is not vars(loaded).get("ExperimentConfig"):
                    return (UnsavedEditor(label, "unknown"),)
                issues.append(UnsavedEditor(label, "needs_user_save_discard"))
                if (
                    experiment.results_dir is not None
                    and expected / "experiments"
                    not in lexical_path(experiment.results_dir).parents
                ):
                    issues.append(UnsavedEditor(label, "alternate_source_unqualified"))
        return tuple(issues)
    except (AttributeError, TypeError, ValueError, RuntimeError, OSError):
        return (UnsavedEditor(label, "unknown"),)
