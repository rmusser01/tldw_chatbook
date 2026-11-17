"""Finite selected settings apply over the original store and native writer."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from inspect import getattr_static
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .console_control_dispatch import ConsoleLocalControls
    from .console_parent_settings import ParentSettingsCapture

from .console_domain_services import _definition, _definition_current


class ConsoleSettingsPending(RuntimeError):
    """A synchronous selected capture cannot observe an unfinished apply."""


class ConsoleDefaultConvergenceRequired(RuntimeError):
    """The original selected render owes an admitted asynchronous replacement."""


def settings_pending(store, session_id):
    """Return the exact retained selected mutation, without blocking controls."""
    pending = getattr(store, "_selected_settings_pending", {}).get(session_id)
    return (
        pending
        if pending is not None and store._sessions.get(session_id) is pending.session
        else None
    )


def require_settings_settled(store, session_id):
    if settings_pending(store, session_id) is not None:
        raise ConsoleSettingsPending("Console settings apply is still pending")


async def wait_settings_settled(store, session_id):
    """Wait for the exact pre/native/post operation outside all store locks."""
    while (pending := settings_pending(store, session_id)) is not None:
        await asyncio.shield(pending.finished)
        if not pending.retired or pending.commit is None:
            raise ConsoleSettingsPending("Console settings completion is unproven")


@dataclass(slots=True, eq=False)
class _SettingsTransition:
    store: Any
    session_id: str
    released: bool = False

    @classmethod
    def acquire(cls, store, session_id):
        # Same preparation → voice → fork order as the original public apply.
        # Only the global count survives; thread-local nested admission never
        # crosses the native await or becomes visible to an unrelated Task.
        with store._voice_promotion_lock:
            if (
                store._voice_promotion_state_replacement is not None
                and not store._voice_promotion_replacement_owned_here()
            ):
                raise RuntimeError("Console state replacement is in progress.")
            if session_id in store._voice_promotion_leases:
                raise RuntimeError("A voice promotion owns this session.")
            store._voice_promotion_mutation_admissions[session_id] = (
                store._voice_promotion_mutation_admissions.get(session_id, 0) + 1
            )
        try:
            store._begin_fork_source_transition(session_id)
        except BaseException:
            cls(store, session_id)._release_voice()
            raise
        return cls(store, session_id)

    def _release_voice(self):
        with self.store._voice_promotion_lock:
            remaining = (
                self.store._voice_promotion_mutation_admissions[self.session_id] - 1
            )
            if remaining:
                self.store._voice_promotion_mutation_admissions[self.session_id] = (
                    remaining
                )
            else:
                del self.store._voice_promotion_mutation_admissions[self.session_id]

    def release(self):
        if self.released:
            raise RuntimeError("Console settings transition already retired")
        self.released = True
        self.store._finish_fork_source_transition(self.session_id)
        self._release_voice()


class PendingSettingsApply:
    """One exact admitted pre/native/post operation, retained through cancel."""

    def __init__(self, controls, command, source, *, convergence=None, mutation=None):
        from .console_chat_store import _ConsoleSettingsPersistenceLifecycle
        from .console_settings_durability import ConsoleSettingsDurabilityOwner
        from .console_parent_settings import (
            ParentSettingsCapture,
            require_parent_settings_capture,
        )

        self.controls, self.command, self.source = controls, command, source
        self.convergence = convergence
        self.mutation = mutation
        self.pre_state = None
        self.live_before_state = None
        self.native_required = None
        self.owned_state = None
        self.actual_task = None
        self.actual_coroutine = None
        self.apply_started = False
        self.native_entered = False
        self.authority_failed = False
        self.error = None
        self.failed_settled = False
        self.store = controls._store
        self.parent_capture = source if type(source) is ParentSettingsCapture else None
        self.guarded_compensation = False
        if self.parent_capture is not None:
            if source.purpose == "guarded-prompt" and convergence is None:
                from .console_guarded_prompt import require_guarded_operation

                self.guarded_compensation = require_guarded_operation(source, mutation)
            require_parent_settings_capture(
                source, controls, presentation=not self.guarded_compensation
            )
            allowed = (
                convergence is not None
                and mutation is None
                and source.purpose
                in {
                    "initial-compose",
                    "early-full-defaults",
                    "readiness-defaults",
                    "compact-controls",
                    "system-prompt",
                    "prompt-modal",
                    "prompt-picker-result",
                    "prompt-editor-result",
                    "handsfree-entry",
                    "sidechat-open",
                    "prompt-resolution",
                    "realtime-ready",
                    "realtime-instructions",
                    "attachment-completion",
                    "guarded-prompt",
                    "prompt-target",
                    "identity-surfaces",
                    "prompt-editor-open",
                    "prefill-command",
                    "workbench-help",
                    "recovery-prepare",
                    "settings-open-prepare",
                    "settings-context-current",
                    "settings-display-prepare",
                    "settings-modal-prepare",
                    "tabs-settings",
                    "visible-send",
                    "prompt-component",
                }
            ) or (
                convergence is None
                and mutation is not None
                and (
                    (
                        mutation[0] == "system_prompt"
                        and source.purpose
                        in {
                            "system-prompt",
                            "prompt-picker-result",
                            "prompt-editor-result",
                            "prompt-component",
                            "guarded-prompt",
                        }
                    )
                    or (
                        mutation[0] == "pinned_prefill"
                        and source.purpose == "prefill-command"
                    )
                    or (
                        mutation[0] == "replacement"
                        and (
                            source.purpose in {"compact-controls", "prefill-command"}
                            or self.guarded_compensation
                        )
                    )
                )
            )
            prior = source.pending_operation
            if (
                not allowed
                or command is not source
                or (
                    prior is not None
                    and (
                        not prior.retired
                        or not prior.finished.done()
                        or (
                            prior.commit is None
                            and not (self.guarded_compensation and prior.failed_settled)
                        )
                    )
                )
            ):
                raise RuntimeError(
                    "Parent settings capture cannot authorize this operation"
                )
            self.session = source.session
        else:
            self.session = self.store._sessions[command.identity.session_id]
            if self.session.incarnation_id != source.identity.incarnation_id:
                raise RuntimeError("Console settings command incarnation changed")
        self.incarnation = self.session.incarnation_id
        self.binding = self.session.conversation_binding_revision
        self.conversation = self.session.persisted_conversation_id
        self.persistence = self.store.persistence
        self.database = getattr(self.persistence, "db", None)
        self.retirement = None
        self.worker = None
        self.transition = None
        self.commit = None
        self.lifecycle = self.store._settings_persistence_lifecycles.setdefault(
            self.session.id, _ConsoleSettingsPersistenceLifecycle()
        )
        self.durability = source.sources.settings_durability_owner
        if type(self.durability) is not ConsoleSettingsDurabilityOwner:
            raise TypeError("Selected settings require the original durability owner")
        self.admission = self.durability.try_acquire()
        if self.admission is None:
            raise ValueError("Application is closing; nothing applied.")
        try:
            self.effect = (
                source.effect
                if self.parent_capture is not None
                else controls._begin_effect("session_settings", self, source)
            )
        except BaseException:
            self.admission.release()
            raise
        self.finished = asyncio.get_running_loop().create_future()
        self.lifecycle.pending_live.add(self.finished)
        if self.parent_capture is not None:
            source.pending_operation = self

    def current(self, *, presentation=True):
        try:
            return self._require_current(presentation=presentation)
        except BaseException:
            self.authority_failed = True
            raise

    def _require_current(self, *, presentation=True):
        from .chat_persistence_service import ChatPersistenceService
        from .console_parent_settings import require_parent_settings_capture

        self.controls._domain.require_current(self.source.sources)
        if self.mutation is not None and self.mutation[0] == "system_prompt":
            for name, record in _PROMPT_PERSISTENCE_DEFINITIONS:
                if getattr_static(self.persistence, name) is not record[
                    0
                ] or not _definition_current(record):
                    raise RuntimeError("Console prompt persistence reader changed")
        if self.mutation is not None and self.mutation[0] == "pinned_prefill":
            if getattr_static(
                self.persistence, "update_conversation_pinned_prefill"
            ) is not _PREFILL_WRITER_DEFINITION[0] or not _definition_current(
                _PREFILL_WRITER_DEFINITION
            ):
                raise RuntimeError("Console pinned prefill writer changed")
        if self.parent_capture is not None:
            require_parent_settings_capture(
                self.parent_capture, self.controls, presentation=presentation
            )
        if (
            self.store is not self.controls._store
            or self.store._sessions.get(self.session.id) is not self.session
            or self.session.incarnation_id != self.incarnation
            or self.session.conversation_binding_revision != self.binding
            or self.session.persisted_conversation_id != self.conversation
            or self.store.persistence is not self.persistence
            or getattr(self.persistence, "db", None) is not self.database
        ):
            raise RuntimeError("Console settings owner changed")
        for name, record in _STORE_DEFINITIONS:
            installed = getattr_static(self.store, name)
            if isinstance(installed, staticmethod):
                installed = installed.__func__
            if installed is not record[0] or not _definition_current(record):
                raise RuntimeError("Console settings reader changed")
        if self.conversation is not None and (
            type(self.persistence) is not ChatPersistenceService
            or getattr_static(
                self.persistence, "update_conversation_console_session_settings"
            )
            is not _WRITER_DEFINITION[0]
            or not _definition_current(_WRITER_DEFINITION)
            or self.database is not self.source.sources.chat_database
            or self.database.is_memory_db
        ):
            raise RuntimeError("Selected settings require the original file writer")
        if self.pre_state is not None and (
            self.session.settings is not self.pre_state[0]
            or (
                self.session.generation_settings_revision,
                self.session.settings_revision,
                self.session.context_policy_revision,
            )
            != self.pre_state[1:]
        ):
            raise RuntimeError("Console pending settings changed")

    async def apply(self):
        from .console_native_commit import (
            _ConsoleNativeBinding,
            _ConsolePlanNativeRetirement,
            _invoke_plan_native,
        )
        from .chat_persistence_service import ChatPersistenceService
        from .console_preparation_reads import run_preparation_read

        self.apply_started = True
        async with self.lifecycle.lock:
            with self.store._preparation_lock:
                self.current(presentation=not self.guarded_compensation)
                if self.mutation is not None:
                    session, previous = self.session, self.session.settings
                    settings = (
                        self.mutation[1]
                        if self.mutation[0] == "replacement"
                        else previous
                    )
                elif self.convergence is None:
                    session, previous, settings = (
                        self.store._prepare_console_settings_live(
                            self.command.submission
                        )
                    )
                    self.store._validate_console_settings_live(
                        self.command.submission, session
                    )
                else:
                    from tldw_chatbook.UI.Console_Modules.session import (
                        follows_saved_defaults,
                    )

                    self.controls._domain.presentation.require_default_convergence(
                        self.convergence
                    )
                    prepared = self.convergence.prepared
                    session, previous, settings = (
                        self.session,
                        prepared.settings_before,
                        prepared.settings,
                    )
                    if (
                        prepared.store is not self.store
                        or prepared.session is not session
                        or session.settings is not previous
                        or session.settings_revision != prepared.revision_before
                        or not follows_saved_defaults(
                            self.store, session, prepared.default_generation
                        )
                    ):
                        raise RuntimeError("Console default replacement source changed")
                pending = getattr(self.store, "_selected_settings_pending", None)
                if pending is None:
                    pending = self.store._selected_settings_pending = {}
                if session.id in pending:
                    raise RuntimeError("Console settings mutation already pending")
                self.live_before_state = (
                    session.settings,
                    session.generation_settings_revision,
                    session.settings_revision,
                    session.context_policy_revision,
                )
                self.transition = _SettingsTransition.acquire(self.store, session.id)
                pending[session.id] = self
                try:
                    if self.convergence is None and self.mutation is None:
                        session.applied_settings_submission_ids.append(
                            self.command.submission.submission_id
                        )
                    prompt = (
                        self.mutation is not None
                        and self.mutation[0] == "system_prompt"
                    )
                    prefill = (
                        self.mutation is not None
                        and self.mutation[0] == "pinned_prefill"
                    )
                    component = prompt or prefill
                    try:
                        if component:
                            prepare = (
                                self.store._prepare_session_system_prompt
                                if prompt
                                else self.store._prepare_session_pinned_prefill
                            )
                            prompt_write = prepare(session.id, self.mutation[1])
                            settings, changed = session.settings, False
                        else:
                            _, changed = (
                                self.store._replace_session_settings_before_snapshot(
                                    session.id,
                                    settings,
                                    mark_user_work=self.convergence is None,
                                    canonical_settings_baseline=settings
                                    if self.convergence is not None
                                    else None,
                                )
                            )
                    finally:
                        # This is the exact locked mutation phase, including a
                        # partial original mutation that raises before return.
                        self.owned_state = (
                            session.settings,
                            session.generation_settings_revision,
                            session.settings_revision,
                            session.context_policy_revision,
                        )
                    self.pre_state = (
                        settings,
                        session.generation_settings_revision,
                        session.settings_revision,
                        session.context_policy_revision,
                    )
                    applicable = prompt_write[-1] if component else False
                    write = (
                        prompt_write
                        if component and applicable and self.conversation is not None
                        else None
                        if component
                        else self.store._capture_session_settings_snapshot_write(
                            session, settings, changed
                        )
                    )
                    self.native_required = write is not None
                except BaseException:
                    del pending[session.id]
                    self.transition.release()
                    self.transition = None
                    raise
            try:
                if write is not None:
                    writer_definition = (
                        _PROMPT_WRITER_DEFINITION
                        if prompt
                        else _PREFILL_WRITER_DEFINITION
                        if prefill
                        else _WRITER_DEFINITION
                    )
                    writer = (
                        self.persistence.update_conversation_system_prompt
                        if prompt
                        else self.persistence.update_conversation_pinned_prefill
                        if prefill
                        else write[0]
                    )
                    if (
                        type(self.persistence) is not ChatPersistenceService
                        or getattr(writer, "__self__", None) is not self.persistence
                        or getattr(writer, "__func__", None) is not writer_definition[0]
                        or not _definition_current(writer_definition)
                        or self.database is not self.source.sources.chat_database
                        or self.database.is_memory_db
                    ):
                        raise RuntimeError(
                            "Selected settings require the original file writer"
                        )
                    self.retirement = _ConsolePlanNativeRetirement()
                    binding = _ConsoleNativeBinding(
                        self.store,
                        self,
                        self.persistence,
                        self.database,
                        retirement=self.retirement,
                    )

                    def write_snapshot():
                        self.current(presentation=False)
                        if not _definition_current(_WRITER_DEFINITION):
                            raise RuntimeError("Console settings writer changed")
                        self.native_entered = True
                        return _invoke_plan_native(
                            binding,
                            lambda: (
                                self.store._write_session_system_prompt(*write)
                                if prompt
                                else self.store._write_session_pinned_prefill(*write)
                                if prefill
                                else self.store._write_session_settings_snapshot(*write)
                            ),
                        )

                    write_result = await run_preparation_read(
                        write_snapshot,
                        creator=self,
                        session_id=session.id,
                        reads=self.controls._runtime._preparation_reads,
                        require_current=lambda: self.current(presentation=False),
                    )
                    if not self.retirement.confirmed:
                        raise RuntimeError(
                            "Console settings native retirement is unproven"
                        )
                with self.store._preparation_lock:
                    self.current(presentation=False)
                    if changed:
                        self.store._bump_settings_revision(session.id)
                    if self.mutation is not None:
                        self.commit = (
                            (
                                session,
                                write_result if write is not None else applicable,
                            )
                            if component
                            else session
                        )
                    elif self.convergence is None:
                        self.commit = self.store._finish_console_settings_live(
                            self.command.submission, session, previous, settings
                        )
                    else:
                        self.commit = settings
                        self.controls._domain.presentation.finish_default_convergence(
                            self.convergence
                        )
                    self.owned_state = (
                        session.settings,
                        session.generation_settings_revision,
                        session.settings_revision,
                        session.context_policy_revision,
                    )
                    return self.commit
            finally:
                with self.store._preparation_lock:
                    if (
                        self.retired
                        and self.commit is not None
                        and pending.get(session.id) is self
                    ):
                        del pending[session.id]
                    if self.retired and self.transition is not None:
                        self.transition.release()
                        self.transition = None

    @property
    def retired(self):
        return self.retirement is None or self.retirement.confirmed


async def converge_default_settings(
    controls: ConsoleLocalControls,
    session_id: str,
    checked_config: Any,
    *,
    _parent_capture: ParentSettingsCapture | None = None,
) -> PendingSettingsApply | None:
    """Settle original default convergence under its exact retained source.

    Args:
        controls: Actual controls associated with the original settings store.
        session_id: The captured original session.
        checked_config: The original checked configuration result.
        _parent_capture: Registered parent owner for autonomous presentation.

    Returns:
        The completed operation, or None when original policy needs no change.

    Raises:
        RuntimeError: Source, settings, definition, or physical custody changed.
    """
    from .console_parent_settings import settings_inline_source

    if _parent_capture is None:
        source = settings_inline_source(controls._runtime)
        if source is None or source.identity.session_id != session_id:
            raise RuntimeError("Default convergence lost its original command source")
    else:
        from .console_parent_settings import require_parent_settings_capture

        source = require_parent_settings_capture(_parent_capture, controls)
        if source.session_id != session_id or not source.started:
            raise RuntimeError("Default convergence lost its admitted parent source")
    checkpoint = controls._domain.presentation.prepare_default_convergence(
        session_id, checked_config
    )
    if checkpoint is None:
        return
    operation = PendingSettingsApply(controls, source, source, convergence=checkpoint)
    await retain_settings_operation(operation)
    return operation


async def retain_settings_operation(operation):
    """Join one original settings operation through native and post settlement."""
    task = None
    try:
        operation.actual_coroutine = operation.apply()
        task = operation.durability.launch(
            operation.admission,
            operation.actual_coroutine,
            name="console-default-convergence",
        )
        operation.actual_task = task
        operation.lifecycle.tasks.add(task)
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
            except BaseException:
                break
        value = task.result()
        if cancelled:
            raise asyncio.CancelledError
        return value
    except BaseException as error:
        operation.error = error
        raise
    finally:
        if task is not None:
            operation.lifecycle.tasks.discard(task)
        operation.admission.release()
        operation.lifecycle.pending_live.discard(operation.finished)
        if not operation.finished.done():
            operation.finished.set_result(None)
        if operation.parent_capture is None:
            operation.controls._finish_effect(
                operation.effect, retired=operation.retired
            )


from .console_chat_store import ConsoleChatStore
from .chat_persistence_service import ChatPersistenceService

_STORE_DEFINITIONS = tuple(
    (name, _definition(getattr(ConsoleChatStore, name)))
    for name in (
        "_prepare_console_settings_live",
        "_validate_console_settings_live",
        "_finish_console_settings_live",
        "_replace_session_settings_before_snapshot",
        "_capture_session_settings_snapshot_write",
        "_write_session_settings_snapshot",
        "_replace_session_context_policy_live",
        "_bump_settings_revision",
        "_begin_fork_source_transition",
        "_finish_fork_source_transition",
        "_prepare_session_system_prompt",
        "_write_session_system_prompt",
        "_prepare_session_pinned_prefill",
        "_write_session_pinned_prefill",
        "_persist_roleplay_context",
        "_persistence_accepts_kwarg",
    )
)
_WRITER_DEFINITION = _definition(
    ChatPersistenceService.update_conversation_console_session_settings
)
_PROMPT_WRITER_DEFINITION = _definition(
    ChatPersistenceService.update_conversation_system_prompt
)
_PREFILL_WRITER_DEFINITION = _definition(
    ChatPersistenceService.update_conversation_pinned_prefill
)
_PROMPT_PERSISTENCE_DEFINITIONS = tuple(
    (name, _definition(getattr(ChatPersistenceService, name)))
    for name in (
        "update_conversation_system_prompt",
        "update_conversation_roleplay_context",
    )
)
