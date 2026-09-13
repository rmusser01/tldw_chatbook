"""Concrete chat core/sidecar source pairs; never transferable IO authority."""

import copy
import stat
import sys
import threading
import weakref
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

from tldw_chatbook.Utils.platform_files import os

from . import bootstrap, profile_paths
from . import storage_admission as storage

ROUTES = {"persona_sidecar", "dictionary_history", "citation_sidecar"}
_BINDINGS = weakref.WeakKeyDictionary()
_LOCKS = weakref.WeakValueDictionary()
_LOCAL = threading.local()


def _classes():
    result = {}
    for module, name, route, owner, attribute, leaf in (
        (
            "Character_Chat.local_character_persona_service",
            "LocalCharacterPersonaService",
            "persona_sidecar",
            "personas",
            "persona_store_path",
            "tldw_chatbook_personas.json",
        ),
        (
            "Character_Chat.local_chat_dictionary_service",
            "LocalChatDictionaryService",
            "dictionary_history",
            "chat.dictionary_history",
            "history_store_path",
            "tldw_chatbook_chat_dictionary_history.json",
        ),
        (
            "Chat.chat_conversation_service",
            "ChatConversationService",
            "citation_sidecar",
            "chat.rag_context",
            "rag_context_store_path",
            "tldw_chatbook_chat_rag_context.json",
        ),
        (
            "Chat.citation_legacy_migration",
            "CitationLegacyMigrationService",
            "citation_sidecar",
            "chat.rag_context",
            "sidecar_path",
            "tldw_chatbook_chat_rag_context.json",
        ),
    ):
        cls = getattr(sys.modules.get("tldw_chatbook." + module), name, None)
        if cls is not None:
            result[cls] = (route, owner, attribute, leaf)
    return result


def _source_kind(source):
    for cls, description in _classes().items():
        if isinstance(source, cls):
            return cls, description
    raise bootstrap.RecoveryRequired("chat_source_not_supported")


@dataclass
class _Binding:
    source_type: type
    db: object
    path: Path
    config: object
    config_path: Path
    companions: tuple = ()


def _validate(source):
    bound = _BINDINGS.get(source)
    if bound is None:
        return None
    cls, (_, _, attribute, leaf) = _source_kind(source)
    config = bound.config
    data = config._CONFIG_CACHE
    from .participants import _repository_participant

    if (
        type(source) is not bound.source_type
        or cls is not bound.source_type
        or source.db is not bound.db
        or getattr(source, attribute) is None
        or profile_paths.lexical_path(getattr(source, attribute)) != bound.path
        or sys.modules.get("tldw_chatbook.config") is not config
        or profile_paths.lexical_path(config._get_effective_config_path())
        != bound.config_path
        or config._CONFIG_CACHE_SOURCE != bound.config_path
        or data is None
        or profile_paths.lexical_path(profile_paths.user_data_dir(data) / leaf)
        != bound.path
        or profile_paths.database_path(data, "chachanotes_db_path") != bound.db.db_path
    ):
        raise bootstrap.RecoveryRequired("chat_source_selection_changed")
    _repository_participant(bound.db)
    if cls.__name__ == "ChatConversationService":
        expected = bound.companions[0]() if bound.companions else None
        if source.citation_legacy_migration is not expected:
            raise bootstrap.RecoveryRequired("chat_source_relationship_changed")
    if (
        cls.__name__ == "CitationLegacyMigrationService"
        and source.repository.db is not bound.db
    ):
        raise bootstrap.RecoveryRequired("chat_source_relationship_changed")
    return bound


def check_citation_attachment(source, migration):
    bound = _validate(source)
    if bound is not None:
        expected = bound.companions[0]() if bound.companions else None
        if migration is not expected:
            raise bootstrap.RecoveryRequired("chat_source_relationship_changed")


def binding(source):
    if source not in _BINDINGS:
        return None
    bound = _validate(source)
    return _source_kind(source)[1][1], bound.path, True


def selection(source, route, target=None):
    _, (actual_route, _, attribute, _) = _source_kind(source)
    if route != actual_route:
        raise bootstrap.RecoveryRequired("chat_source_not_supported")
    bound = _validate(source)
    value = getattr(source, attribute)
    if value is None:
        raise bootstrap.RecoveryRequired("chat_source_has_no_file")
    selected = profile_paths.lexical_path(value)
    if target is not None and selected != profile_paths.lexical_path(target):
        raise bootstrap.RecoveryRequired("chat_source_selection_changed")
    return selected, bound is not None, False


def _install(source, db, config, selected):
    from ..DB.ChaChaNotes_DB import CharactersRAGDB
    from .participants import _repository_participant

    # An arbitrary injected DB remains ordinary. No owner/path flag installs it.
    if type(db) is not CharactersRAGDB or db.is_memory_db:
        return
    if (
        profile_paths.database_path(config._CONFIG_CACHE, "chachanotes_db_path")
        != db.db_path
    ):
        return
    _repository_participant(db)
    _BINDINGS[source] = _Binding(
        type(source),
        db,
        selected,
        config,
        profile_paths.lexical_path(config._get_effective_config_path()),
    )


def build_persona_service(db):
    from .. import config
    from ..Character_Chat.local_character_persona_service import (
        LocalCharacterPersonaService,
    )

    with closing(storage._Acquisition()) as attempt:
        selected = config.get_user_data_dir() / "tldw_chatbook_personas.json"
        attempt.check()
        source = LocalCharacterPersonaService.__new__(LocalCharacterPersonaService)
        _install(source, db, config, selected)
        LocalCharacterPersonaService.__init__(source, db, persona_store_path=selected)
        return source


def build_dictionary_service(db):
    from .. import config
    from ..Character_Chat.local_chat_dictionary_service import (
        LocalChatDictionaryService,
    )

    with closing(storage._Acquisition()) as attempt:
        selected = (
            config.get_user_data_dir() / "tldw_chatbook_chat_dictionary_history.json"
        )
        attempt.check()
        source = LocalChatDictionaryService.__new__(LocalChatDictionaryService)
        _install(source, db, config, selected)
        LocalChatDictionaryService.__init__(source, db, history_store_path=selected)
        return source


def bind_citation_services(service, migration):
    """Bind the concrete factory's already-composed DB/repository relationship."""
    from .. import config
    from ..Chat.chat_conversation_service import ChatConversationService
    from ..Chat.citation_legacy_migration import CitationLegacyMigrationService

    with closing(storage._Acquisition()) as attempt:
        selected = config.get_user_data_dir() / "tldw_chatbook_chat_rag_context.json"
        attempt.check()
        if service in _BINDINGS or migration in _BINDINGS:
            _validate(service)
            _validate(migration)
            return
        if (
            type(service) is not ChatConversationService
            or type(migration) is not CitationLegacyMigrationService
            or service.citation_legacy_migration is not migration
            or service.db is not migration.db
            or migration.repository.db is not service.db
            or service.rag_context_store_path != selected
            or migration.sidecar_path != selected
        ):
            return
        _install(service, service.db, config, selected)
        _install(migration, migration.db, config, selected)
        if service in _BINDINGS:
            _BINDINGS[service].companions = (weakref.ref(migration),)
            _BINDINGS[migration].companions = (weakref.ref(service),)


def _cache_names(source):
    name = _source_kind(source)[0].__name__
    if name == "LocalCharacterPersonaService":
        return (
            "_persona_profiles",
            "_persona_store_extras",
            "_persona_exemplars",
            "_character_exemplars",
            "_chat_settings",
            "_chat_greeting_selections",
            "_chat_presets",
            "_character_memories",
        )
    if name == "LocalChatDictionaryService":
        return ("_history",)
    if name == "ChatConversationService":
        return ("_rag_context_store",)
    return ()


def drain_ready(source):
    _validate(source)
    return getattr(source, "_chat_persistence_error", None) is None


@contextmanager
def operation(source):
    """Preadmit the fixed actual pair, then discover both existing scopes."""
    from . import raw_participants as raw
    from .participants import _core_operation

    bound = _validate(source)
    route, _, attribute, _ = _source_kind(source)[1]
    active = getattr(_LOCAL, "active", None)
    if active is not None and any(item is source for item, _ in active):
        for item, token in active:
            _validate(item)
            raw._check(token)
        core = getattr(storage._operation_local, "operation", None)
        if core is not None:
            storage._check_operation(core, source.db.db_path)
        previous = getattr(raw._local, "operation", None)
        raw._local.operation = next(token for item, token in active if item is source)
        try:
            yield raw._local.operation
        finally:
            raw._local.operation = previous
        return
    if getattr(source, attribute) is None:
        # Caller-selected no-file services keep their existing memory semantics.
        yield None
        return
    previous_core = getattr(storage._operation_local, "operation", None)
    previous_raw = getattr(raw._local, "operation", None)
    # A different pair acquires independent ordinary admission while gates open.
    storage._operation_local.operation = None
    raw._local.operation = None
    _LOCAL.active = None
    attempt = None
    acquired = False
    before = {}
    entered = False
    changes_before = 0
    tokens = []
    lock = (
        threading.RLock()
        if _source_kind(source)[0].__name__ == "CitationLegacyMigrationService"
        else getattr(source, "_history_lock", None)
    )
    try:
        attempt = storage._Acquisition()
        selected = selection(source, route)[0]
        if lock is None:
            key = str(selected.resolve())
            with storage._lock:
                lock = _LOCKS.get(key)
                if lock is None:
                    lock = threading.RLock()
                    _LOCKS[key] = lock
        while not lock.acquire(timeout=0.05):
            attempt.check()
        acquired = True
        attempt.check()
        members = (source,)
        if bound is not None:
            members += tuple(ref() for ref in bound.companions)
        before = {
            item: {
                name: copy.deepcopy(getattr(item, name)) for name in _cache_names(item)
            }
            for item in (source,)
        }
        with ExitStack() as scopes:
            core = None
            from ..DB.ChaChaNotes_DB import CharactersRAGDB

            if isinstance(source.db, CharactersRAGDB):
                scopes.enter_context(_core_operation(source.db))
                core = getattr(storage._operation_local, "operation", None)
            tokens = []
            for item in members:
                item_route = _source_kind(item)[1][0]
                token = scopes.enter_context(raw._scope(item, item_route, writing=True))
                tokens.append((item, token))
            # Both native scopes exist before exposure. No pause-time acquisition.
            attempt.check()
            for item, token in tokens:
                _validate(item)
                state = raw._check(token)
                if bound is not None:
                    state.observed_files[state.selected] = _sidecar_identity(
                        state.selected
                    )
                if (
                    state.participant is not None
                    and raw._participant_state(state.participant).closed
                ):
                    raise bootstrap.RecoveryRequired("storage_locally_paused")
            if core is not None:
                storage._check_operation(core, source.db.db_path)
                if core.participant.closed:
                    raise bootstrap.RecoveryRequired("storage_locally_paused")
            qualified = (
                bound is not None
                and core is not None
                and core.hold is not None
                and all(all(raw._states[token].holds) for _, token in tokens)
            )
            with storage._changed:
                attempt.check()
                if any(
                    raw._states[token].participant is not None
                    and raw._participant_state(raw._states[token].participant).closed
                    for _, token in tokens
                ) or (core is not None and core.participant.closed):
                    raise bootstrap.RecoveryRequired("storage_locally_paused")
                if qualified:
                    storage._operation_local.operation = core
                elif bound is not None:
                    source._chat_persistence_error = "chat_scope_unqualified"
                raw._local.operation = tokens[0][1]
                _LOCAL.active = tuple(tokens)
            changes_before = _native_changes(source.db)
            entered = True
            try:
                yield tokens[0][1]
            finally:
                _LOCAL.active = None
                storage._operation_local.operation = None
                raw._local.operation = tokens[-1][1]
        if any(token in raw._states for _, token in tokens):
            raise bootstrap.RecoveryRequired("raw_resources_not_retired")
    except BaseException:
        for item, snapshot in before.items():
            for name, value in snapshot.items():
                setattr(item, name, value)
            if entered and (
                _native_changes(source.db) != changes_before
                or any(
                    raw._states.get(token) is not None and raw._states[token].uncertain
                    for _, token in tokens
                )
            ):
                item._chat_persistence_error = "chat_publication_incomplete"
        raise
    finally:
        if acquired:
            lock.release()
        if attempt is not None:
            attempt.close()
        _LOCAL.active = active
        storage._operation_local.operation = previous_core
        raw._local.operation = previous_raw
        if previous_core is not None:
            storage._check_operation(previous_core, previous_core.path)
        if previous_raw is not None:
            raw._check(previous_raw)


def guarded(function):
    @wraps(function)
    def wrapped(source, *args, **kwargs):
        cls, _ = _source_kind(source)
        if (
            cls.__dict__.get(function.__name__) is not wrapped
            or function.__module__ != cls.__module__
        ):
            raise bootstrap.RecoveryRequired("chat_source_not_supported")
        with operation(source):
            return function(source, *args, **kwargs)

    return wrapped


def read_text(source):
    from . import raw_participants as raw

    with operation(source) as token:
        path = selection(source, _source_kind(source)[1][0])[0]
        with raw._file(token, path, "r") as stream:
            return stream.read()


def write_text(source, text):
    from . import raw_participants as raw

    with operation(source) as token:
        selected = selection(source, _source_kind(source)[1][0])[0]
        temporary = selected.with_suffix(selected.suffix + ".tmp")
        raw._mkdirs(token)
        try:
            with raw._file(token, temporary, "w") as stream:
                stream.write(text)
            state = raw._check(token, selected, writing=True)
            if _validate(source) is not None and (
                _sidecar_identity(selected, writing=True)
                != state.observed_files.get(selected)
            ):
                raise bootstrap.RecoveryRequired("chat_sidecar_identity_changed")
            raw._replace(token, temporary, selected)
            if _validate(source) is not None:
                state.observed_files[selected] = _sidecar_identity(selected)
        finally:
            state = raw._states[token]
            if temporary in state.created_files and not state.uncertain:
                raw._remove_temporary(token, temporary)


def _sidecar_identity(path, *, writing=False):
    try:
        info = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if writing and not stat.S_ISREG(info.st_mode):
        raise bootstrap.RecoveryRequired("chat_sidecar_identity_changed")
    return info.st_dev, info.st_ino


def _native_changes(db):
    """Observe only the actual source thread's existing native connection."""
    from ..DB.ChaChaNotes_DB import CharactersRAGDB

    if not isinstance(db, CharactersRAGDB):
        return 0
    connection = getattr(db._local, "conn", None)
    return connection.total_changes if connection is not None else 0


@contextmanager
def sidecar_descriptor(source):
    """Keep the canonical legacy reader's actual descriptor until positive close."""
    from ..Chat.citation_legacy_migration import CitationLegacyMigrationService
    from . import raw_participants as raw

    if not isinstance(source, CitationLegacyMigrationService):
        raise bootstrap.RecoveryRequired("chat_source_not_supported")
    with operation(source) as token:
        selected = profile_paths.lexical_path(source.sidecar_path)
        state = raw._check(token, selected)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        if state.pinned:
            descriptor = os.open(
                selected.name, flags, dir_fd=state.pins[selected.parent]
            )
        else:
            descriptor = os.open(selected, flags)
        state.descriptors.add(descriptor)
        try:
            yield descriptor
        finally:
            raw._close_descriptor(state, descriptor)
