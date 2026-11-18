"""One selected owner's original scope summaries, refreshed only when due."""

from __future__ import annotations

import asyncio
from inspect import getattr_static
from types import MethodType

from loguru import logger

from ..Backup_Recovery import chat_source_participants
from ..Character_Chat.chat_dictionary_scope_service import ChatDictionaryScopeService
from ..Character_Chat.local_chat_dictionary_service import LocalChatDictionaryService
from ..DB.base_db import operation_owned_connection
from .console_control_dispatch import require_control_source
from .console_domain_services import _definition, _definition_current
from .console_execution_inspector_values import build_inspector_scopes, inspector_scopes
from .console_execution_retirement import _ReadResources
from .console_preparation_reads import preparation_read_for, run_preparation_read
from . import console_execution_inspector_values as _values
from ..UI.Console_Modules import inspector_scope_display as _display


class ConsoleBackendInspectorScopes:
    """Retain original scope-change caches; no cached value authorizes an effect."""

    def __init__(self, backend):
        self.backend = backend
        self.service = None
        self.key = None
        self.dictionaries = None
        self.world_books = None
        self.value = None
        self.reads = set()
        self.receipts = []
        self.observations = []
        self.uncertain = False

    def _key(self, session):
        return (
            self.backend.profile, self.backend.profile.storage.chachanotes,
            session.id, session.incarnation_id, session.persisted_conversation_id,
            session.conversation_binding_revision, self.backend.controls.source_revision,
        )

    def _current(self, source, session, key):
        backend = self.backend
        backend.require_sources()
        backend.presentation.require_association(backend.presentation.association)
        if (
            self.uncertain or type(self) is not ConsoleBackendInspectorScopes
            or backend.presentation.inspector_scopes is not self
            or source.sources is not backend.domain.sources
            or backend.store._sessions.get(session.id) is not session
            or backend.store.active_session_id != session.id
            or source.identity.session_id != session.id
            or source.identity.incarnation_id != session.incarnation_id
            or source.identity.source_revision != backend.controls.source_revision
            or self._key(session) != key
        ):
            raise RuntimeError("Console Inspector scope owner changed")
        for owner, name, descriptor, record in _SOURCES:
            receiver = self if owner is ConsoleBackendInspectorScopes else owner
            if (getattr_static(owner, name, None) is not descriptor
                or getattr_static(receiver, name, None) is not descriptor
                or not _definition_current(record)):
                raise RuntimeError("Console Inspector scope source changed")
        if self.service is not None:
            local = self.service.local_service
            if (
                type(self.service) is not ChatDictionaryScopeService
                or type(local) is not LocalChatDictionaryService
                or local.db is not key[1]
                or self.service.policy_enforcer is not backend.profile.policy_enforcer
                or self.service.server_service is not None
            ):
                raise RuntimeError("Console Inspector dictionary service changed")
            for receiver, names in ((self.service, _SCOPE_METHODS), (local, _LOCAL_METHODS)):
                for name in names:
                    bound = getattr(receiver, name)
                    original = getattr_static(type(receiver), name)
                    valid = (
                        bound is original.__func__ if isinstance(original, staticmethod)
                        else type(bound) is MethodType and bound.__self__ is receiver and bound.__func__ is original
                    )
                    if not valid:
                        raise RuntimeError("Console Inspector dictionary method changed")

    async def warm(self, identity):
        """Join the original summary work before publishing this selected scope."""
        from .console_dictionary_summary_read import DictionarySummaryRead, dictionary_summary_retired

        backend = self.backend
        backend._require_owner_loop()
        source = require_control_source(backend.runtime)
        message = backend.adapter._receipts[identity.request_id][0]
        session = backend.store._sessions.get(identity.session_id)
        if message.operation != "ReadProjection" or source.identity is not identity or session is None:
            raise RuntimeError("Console Inspector scope command changed")
        key = self._key(session)
        self._current(source, session, key)
        refresh = message.payload.get("inspector_refresh")
        changed = key != self.key
        if not changed and refresh is None:
            return
        conversation_id = session.persisted_conversation_id
        dictionaries, world_books = self.dictionaries, self.world_books
        if conversation_id is None:
            # The original no-scope branches perform no service or DB work.
            dictionaries, world_books = {"dictionaries": []}, {"world_books": []}
        else:
            task = asyncio.current_task()
            observations = self.observations = []
            receipts = []
            self.receipts = receipts
            dictionary_receipt = DictionarySummaryRead(self)
            world_receipt = DictionarySummaryRead(self)
            effect = backend.controls._begin_effect("inspector_scope_summary", self, source)

            def current():
                self._current(source, session, key)

            def read():
                handoff = preparation_read_for(self)
                if handoff is None or handoff.creator is not self or handoff.task is not task:
                    raise RuntimeError("Console Inspector scope read changed")
                current()
                dictionary_value, world_value = dictionaries, world_books
                if changed or refresh == "dictionary":
                    try:
                        if self.service is None:
                            observation = _ReadResources(None, handoff)
                            observations.append(observation)
                            with observation.observe(), operation_owned_connection(key[1]):
                                local = chat_source_participants.build_dictionary_service(key[1])
                                self.service = ChatDictionaryScopeService(
                                    local_service=local, server_service=None,
                                    policy_enforcer=backend.profile.policy_enforcer,
                                )
                        current()
                        receipt = dictionary_receipt
                        receipts.append(receipt)
                        dictionary_value = receipt.read(
                            self.service, conversation_id, None, handoff=handoff,
                        )
                    except Exception:
                        logger.opt(exception=True).warning("Could not summarize active chat dictionaries for the Console inspector.")
                        dictionary_value = {"dictionaries": []}
                    current()
                    if not isinstance(dictionary_value, dict):
                        dictionary_value = {"dictionaries": []}
                if changed or refresh == "world_book":
                    try:
                        receipt = world_receipt
                        receipts.append(receipt)
                        world_value = receipt.read_world_books(key[1], conversation_id, handoff=handoff)
                    except Exception:
                        logger.opt(exception=True).warning("Could not summarize active world books for the Console inspector.")
                        world_value = {"world_books": []}
                    current()
                    if not isinstance(world_value, dict):
                        world_value = {"world_books": []}
                return dictionary_value, world_value

            def retired():
                return all(item.confirmed() for item in observations) and all(
                    dictionary_summary_retired(item) for item in receipts
                )

            try:
                dictionaries, world_books = await run_preparation_read(
                    read, creator=self, session_id=session.id, reads=self.reads,
                    observers=(backend.runtime._preparation_reads,),
                    require_current=current,
                    submission_slot=backend.runtime._display_submission_slot,
                )
                current()
                if not retired():
                    raise RuntimeError("Console Inspector scope retirement is unproven")
            finally:
                settled = retired()
                if not settled:
                    self.uncertain = True
                backend.controls._finish_effect(effect, retired=settled)
        self._current(source, session, key)
        value = build_inspector_scopes(conversation_id, dictionaries, world_books)
        inspector_scopes(value, False)
        self._current(source, session, key)
        self.key, self.value = key, value
        self.dictionaries, self.world_books = dictionaries, world_books

    def current_value(self, identity):
        source = require_control_source(self.backend.runtime)
        session = self.backend.store._sessions.get(identity.session_id)
        if session is None or source.identity is not identity:
            raise RuntimeError("Console Inspector scope command changed")
        self._current(source, session, self.key)
        if self.value is None or self.value.conversation_id != session.persisted_conversation_id:
            raise RuntimeError("Console Inspector scope publication is unavailable")
        return self.value


_SCOPE_METHODS = (
    "summarize_active_dictionaries", "_normalize_mode", "_invoke", "_enforce_policy",
    "_backend", "_call_backend", "_statistics_action", "_is_file_backed_local", "_maybe_await",
)
_LOCAL_METHODS = ("summarize_active_dictionaries", "_require_db")
_SOURCES = tuple(
    (owner, name, descriptor,
     _definition(descriptor.__func__ if isinstance(descriptor, (staticmethod, classmethod)) else descriptor))
    for owner, names in (
        (ConsoleBackendInspectorScopes, ("__init__", "_key", "_current", "warm", "current_value")),
        (ChatDictionaryScopeService, _SCOPE_METHODS + ("__init__",)),
        (LocalChatDictionaryService, _LOCAL_METHODS + ("__init__",)),
        (chat_source_participants, ("build_dictionary_service",)),
        (_values, ("build_inspector_scopes", "inspector_scopes", "_row", "_action")),
        (_display, ("dictionary_rows", "dictionary_actions", "world_book_rows", "world_book_actions")),
    )
    for name in names for descriptor in (getattr_static(owner, name),)
)
