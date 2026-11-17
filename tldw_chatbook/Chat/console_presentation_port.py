"""Concrete in-process presentation over the actual Console attachment."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import getattr_static
from typing import Any

from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired


@dataclass(frozen=True, slots=True)
class ConsoleParentAssociation:
    """Opaque association metadata; never a source or permission substitute."""

    identity: object
    revision: int


@dataclass(slots=True, eq=False)
class _ArchiveSendCheckpoint:
    port: Any
    association: ConsoleParentAssociation
    record: Any
    conversation_id: str
    generation: int
    consumed: bool = False


@dataclass(slots=True, eq=False)
class _DefaultConvergenceCheckpoint:
    port: Any
    stamp: tuple
    prepared: Any
    definitions: tuple
    consumed: bool = False


@dataclass(slots=True, eq=False)
class _DisplayFullEntry:
    port: object
    view: object
    task: object
    stamp: object
    full: object = None
    body_returned: bool = False
    character_demand: str | None = None


class ConsolePresentationPort:
    """Keep App and view callbacks exclusively on the presentation side."""

    def __init__(self, app: Any) -> None:
        self._app = app
        self._configuration = getattr(app, "app_config", None)
        self._runtime = None
        self._controller = None
        self._view = None
        self._visit = 0
        self._attachment_generation = None
        self.association = ConsoleParentAssociation(object(), 0)
        self._app_runtime = getattr_static(app, "console_runtime", None)
        self._controller_app = None
        self._hook_sources = ()
        self._display_full_entries = {}
        self._display_context_demands = {}
        self._display_character_demands = {}
        from types import FunctionType
        from .console_domain_services import _definition

        self._parent_callbacks = tuple(
            (
                name,
                callback,
                _definition(callback) if type(callback) is FunctionType else None,
            )
            for name in ("notify", "call_from_thread")
            for callback in (getattr_static(app, name, None),)
        )

    def bind_runtime(self, runtime: Any) -> None:
        if self._runtime is not None and self._runtime is not runtime:
            raise RecoveryRequired("console_snapshot_owner_changed")
        self._runtime = runtime

    def definitions_current(self) -> bool:
        from .console_domain_services import _definition_current

        return type(self) is ConsolePresentationPort and all(
            getattr_static(self, name, None) is record[0]
            and _definition_current(record)
            for name, record in _PORT_DEFINITIONS
        )

    def wire_wake(self, wake, projection) -> None:
        wake.wire(
            app=self._app,
            startup_ready=lambda: bool(getattr(self._app, "_ui_ready", True)),
        )
        wake._recovery_projection = projection

    def register_fleet_attention(self, bridge, receipt_service) -> None:
        from .console_fleet_attention import register_fleet_attention

        register_fleet_attention(bridge, self._app, receipt_service=receipt_service)

    def require_association(self, association: ConsoleParentAssociation) -> None:
        """Check real parent objects, independent of backend source checks."""
        from .console_domain_services import _definition_current

        runtime = self._runtime
        if (
            not self.definitions_current()
            or any(
                getattr_static(self._app, name, None) is not callback
                or (record is not None and not _definition_current(record))
                for name, callback, record in self._parent_callbacks
            )
            or association is not self.association
            or getattr(self._app, "app_config", None) is not self._configuration
            or runtime is None
            or runtime._app is not self._app
            or runtime._disposed
            or getattr_static(self._app, "console_runtime", None)
            is not self._app_runtime
            or (
                self._controller is not None
                and runtime._chat_controller is not self._controller
            )
            or (
                self._controller is not None
                and self._controller.app is not self._controller_app
            )
        ):
            raise RecoveryRequired("console_snapshot_owner_changed")

    def _stamp(self):
        self.require_association(self.association)
        return self.association, self._view, self._visit, self._attachment_generation

    def _require_display_owner(self, owner):
        from .console_domain_display import ConsoleDomainDisplayOwner
        from .console_display_phases import ConsoleDisplayPhaseRefused

        self.require_association(self.association)
        if (
            type(owner) is not ConsoleDomainDisplayOwner
            or owner.domain.presentation is not self
            or owner.runtime is not self._runtime
            or self._runtime.domain_services is not owner.domain
            or getattr(self._app, "chachanotes_db", None)
            is not owner.sources.chat_database
            or getattr(self._app, "app_config", None) is not owner.sources.configuration
        ):
            raise ConsoleDisplayPhaseRefused(
                "Display parent/source association changed"
            )
        view = self._view
        if view is None:
            if self._runtime.view is not None:
                raise ConsoleDisplayPhaseRefused("Display attachment changed")
            return None
        from tldw_chatbook.UI.Screens.chat_screen import (
            ChatScreen,
            _CONSOLE_DISPLAY_PARENT_READERS,
            _CONSOLE_DISPLAY_PARENT_DEFINITIONS,
            _console_screen_is_torn_down,
        )
        from .console_received_dispatch import stock_native_methods

        from tldw_chatbook.UI.Console_Modules.character_context import (
            _character_reader_current,
        )

        if (
            type(view) is not ChatScreen
            or not all(
                _character_reader_current(record, view)
                for record in _CONSOLE_DISPLAY_PARENT_DEFINITIONS
            )
            or not stock_native_methods(view, _CONSOLE_DISPLAY_PARENT_READERS)
            or self._runtime.view is not view
            or view.app_instance is not self._app
            or view._console_chat_store is not owner.store
            or view._console_chat_controller is not owner.controller
            or view._character_context is not owner.character
            or getattr(view, "_console_chat_tearing_down", False)
            or _console_screen_is_torn_down(view)
        ):
            raise ConsoleDisplayPhaseRefused("Display attached owner changed")
        return view

    def capture_display(self, owner):
        # None means no attached witness. It is never converted into True.
        return None if self._require_display_owner(owner) is None else self._stamp()

    def display_run_active(self, owner):
        view = self._require_display_owner(owner)
        return None if view is None else view._console_run_active()

    def display_character_key(self, owner):
        view = self._require_display_owner(owner)
        if view is None:
            return None
        # The original App/config/native-session identity members remain real.
        # Parent run/idle state is observed from its original complete body.
        return (
            self._stamp(),
            self._app,
            id(self._app.app_config),
            self._app.app_config,
            self._app.chachanotes_db,
            view._console_chat_store,
            view._console_chat_controller,
            getattr(view, "_console_chat_tearing_down", False),
        )

    def require_display_checkpoint(self, owner, request):
        from .console_domain_display import _DisplayCheckpoint
        from .console_display_phases import ConsoleDisplayPhaseRefused

        view = self._require_display_owner(owner)
        full = owner._full
        if (
            type(request) is not _DisplayCheckpoint
            or full is None
            or request.full is not full
            or full.pending_checkpoint is not request
            or request.consumed
            or request.source is not owner.sources
            or request.session is not full.session
            or request.incarnation != full.incarnation
            or request.runner.full is not full
            or full._runner_ordinals.get(request.runner) != request.ordinal
            or request.kind
            not in {
                "pair-ambient",
                "scope-owner",
                "scope-error",
                "post-groups-owner",
            }
            or not full.physically_retired
        ):
            raise ConsoleDisplayPhaseRefused("Display fixed checkpoint changed")
        if request.stamp is None:
            if view is not None:
                raise ConsoleDisplayPhaseRefused(
                    "Detached display acquired another view"
                )
            return None
        if view is None or not self._current_visit(request.stamp):
            raise ConsoleDisplayPhaseRefused("Display checkpoint visit changed")
        return None

    def require_display_publication(self, owner, stamp, full):
        from .console_display_phases import ConsoleDisplayPhaseRefused

        view = self._require_display_owner(owner)
        if owner._full is not full or not full.physically_retired:
            raise ConsoleDisplayPhaseRefused("Display publication has unsettled owners")
        if stamp is None:
            if view is not None:
                raise ConsoleDisplayPhaseRefused("Detached display view changed")
            return None
        if view is None or not self._current_visit(stamp):
            raise ConsoleDisplayPhaseRefused("Display publication visit changed")
        return None

    def publish_display_character(self, owner, state):
        view = self._require_display_owner(owner)
        if view is None:
            return
        from tldw_chatbook.UI.Console_Modules.wiring import (
            _sync_character_context_presentation,
        )

        _sync_character_context_presentation(view, state)

    def display_appearance_current(self, owner, stamp):
        from .console_received_dispatch import stock_native_methods
        from tldw_chatbook.UI.Console_Modules.workspace import (
            ConsoleWorkspaceController,
            _APPEARANCE_DISPLAY_PUBLISHERS,
        )

        view = self._require_display_owner(owner)
        if stamp is None:
            return None  # A detached read has no screen-running witness.
        if view is None or not self._current_visit(stamp):
            return False
        workspace = view._workspace
        if (
            type(workspace) is not ConsoleWorkspaceController
            or not stock_native_methods(workspace, _APPEARANCE_DISPLAY_PUBLISHERS)
            or owner.domain._conversation_owner is None
            or getattr(self._app, "local_chat_conversation_service", None)
            is not owner.domain._conversation_owner[1]
        ):
            return False
        return workspace._screen_running_accessor()

    def publish_display_appearance(self, owner, stamp):
        self.require_display_publication(owner, stamp, owner._full)
        view = self._require_display_owner(owner)
        if view is not None:
            view._workspace._publish_display_reads()
    def archive_send_inflight(self, record) -> str | None:
        """Check the real parent's reservation before qualifying the reader."""
        from .conversation_archive_actions import archive_send_inflight_refusal

        self.require_association(self.association)
        return archive_send_inflight_refusal(self._app, record.archive_conversation_id)

    def capture_archive_send(self, record) -> _ArchiveSendCheckpoint:
        """Capture the original generation after native reader qualification."""
        self.require_association(self.association)
        return _ArchiveSendCheckpoint(
            self,
            self.association,
            record,
            record.archive_conversation_id,
            getattr(self._app, "_conversation_archive_generation", 0),
        )

    def finish_archive_send(self, checkpoint, record, states) -> str | None:
        """Publish only the matching complete read, after physical retirement."""
        from .conversation_archive_actions import publish_archive_send_state

        if (
            type(checkpoint) is not _ArchiveSendCheckpoint
            or checkpoint.port is not self
            or checkpoint.record is not record
            or checkpoint.consumed
            or checkpoint.conversation_id != record.archive_conversation_id
        ):
            raise RuntimeError("Console archive checkpoint changed")
        self.require_association(checkpoint.association)
        checkpoint.consumed = True
        return publish_archive_send_state(
            self._app, checkpoint.conversation_id, checkpoint.generation, states
        )

    def prepare_default_convergence(self, session_id, checked_config):
        """Run the original real-parent eligibility/memo phase after config read."""
        from tldw_chatbook.UI.Console_Modules.session import (
            ConsoleDefaultConvergence,
            ConsoleSessionController,
            _CONSOLE_DEFAULT_CONVERGENCE_READERS,
        )
        from .console_domain_services import _definition_current

        stamp = self._stamp()
        view = self._view
        if view is None:
            return None
        owner = getattr(view, "_session", None)
        if type(owner) is not ConsoleSessionController:
            raise RuntimeError("Console defaults presentation owner changed")
        store = self._runtime._chat_store
        if getattr(view, "_console_chat_store", None) is not store:
            raise RuntimeError("Console defaults store changed")
        session = store._sessions[session_id]
        definitions = _CONSOLE_DEFAULT_CONVERGENCE_READERS
        if any(
            getattr_static(owner, name) is not record[0]
            or not _definition_current(record)
            for name, record in definitions
        ):
            raise RuntimeError("Console defaults defining callbacks changed")
        if checked_config is None:
            # Early FULL has not reached readiness preparation. Preserve the
            # original core entry's nonblocking cold-config guard and order;
            # only its already-readable parent map may prepare defaults here.
            from contextlib import ExitStack
            from tldw_chatbook import config
            from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
                defer_cold_live_read,
            )
            from tldw_chatbook.UI.Console_Modules.config_sync import (
                sync_config_unreadable_notice,
            )

            sync_config_unreadable_notice(view)
            with ExitStack() as held:
                for lock in (
                    config._settings_rebuild_lock(),
                    config._config_file_lock(),
                ):
                    if not lock.acquire(blocking=False):
                        return None
                    held.callback(lock.release)
                if defer_cold_live_read(view, "sync_console_core_state"):
                    return None
                from .console_control_settings import ConsoleDefaultConvergenceRequired

                try:
                    prepared = owner._prepare_stale_default_console_settings(
                        store, session
                    )
                except ConsoleDefaultConvergenceRequired:
                    return None  # Original core/readiness order owns this cold replay.
        else:
            prepared = owner._prepare_stale_default_console_settings(
                store,
                session,
                checked_config=checked_config,
            )
        if type(prepared) is not ConsoleDefaultConvergence:
            return None
        return _DefaultConvergenceCheckpoint(self, stamp, prepared, definitions)

    def require_default_convergence(self, checkpoint) -> None:
        """Validate the exact parent memo/visit without certifying domain state."""
        from .console_domain_services import _definition_current
        from tldw_chatbook.UI.Console_Modules import session as source

        if (
            type(checkpoint) is not _DefaultConvergenceCheckpoint
            or checkpoint.port is not self
            or checkpoint.consumed
            or not self._current_visit(checkpoint.stamp)
            or self._runtime._chat_store.active_session_id
            != checkpoint.prepared.session.id
        ):
            raise RuntimeError("Console defaults checkpoint changed")
        prepared = checkpoint.prepared
        if (
            self._view._session is not prepared.owner
            or prepared.owner._pristine_defaults_checked is not prepared.checked
            or prepared.owner._console_new_chat_default_generation()
            != prepared.default_generation
            or any(
                getattr_static(prepared.owner, name) is not record[0]
                or not _definition_current(record)
                for name, record in checkpoint.definitions
            )
            or any(
                getattr(source, name) is not record[0]
                or not _definition_current(record)
                for name, record in source._CONSOLE_DEFAULT_CONVERGENCE_SOURCES
            )
        ):
            raise RuntimeError("Console defaults source changed")

    def finish_default_convergence(self, checkpoint) -> None:
        """Publish original span/notice only after exact replacement settlement."""
        self.require_default_convergence(checkpoint)
        prepared = checkpoint.prepared
        if (
            prepared.store._sessions.get(prepared.session.id) is not prepared.session
            or prepared.session.incarnation_id != prepared.incarnation
            or prepared.session.conversation_binding_revision
            != prepared.binding_revision
            or prepared.session.settings is not prepared.settings
            or prepared.store.session_settings_revision(prepared.session.id)
            != prepared.revision_before + 1
        ):
            raise RuntimeError("Console default replacement changed before publication")
        prepared.owner._finish_stale_default_console_settings(prepared)
        checkpoint.consumed = True

    def _current_visit(self, stamp) -> bool:
        try:
            self.require_association(stamp[0])
        except RecoveryRequired:
            return False
        return (
            self._view is stamp[1]
            and self._visit == stamp[2]
            and self._runtime.view is stamp[1]
            and self._runtime._attached_generation == stamp[3]
            and all(
                getattr(target, name, None) is callback
                for target, name, callback in self._hook_sources
            )
        )

    def notice(self, message: str, *, severity: str = "warning") -> bool:
        """Presentation failure cannot retract accepted domain work."""
        try:
            self.require_association(self.association)
            self._app.notify(message, severity=severity)
        except Exception:  # noqa: BLE001 - notices are advisory.
            return False
        return True

    def bind_view_hooks(self) -> None:
        runtime = self._runtime
        view = runtime.view
        self._controller = runtime._chat_controller
        self._controller_app = getattr(self._controller, "app", None)
        generation = runtime._attached_generation
        if view is not self._view or generation != self._attachment_generation:
            self._visit += 1
            self._view = view
            self._attachment_generation = generation
        if view is None:
            return
        provider = getattr(view, "console_view_hooks", None)
        hooks = provider() if callable(provider) else {}
        sources = []
        for slot in runtime._execution_service._runtime_source.CONSOLE_VIEW_HOOK_SLOTS:
            target = runtime._hook_target(slot.target)
            if target is not None:
                callback = hooks.get(slot.name, slot.viewless_default)
                setattr(target, slot.name, callback)
                sources.append((target, slot.name, callback))
        self._hook_sources = tuple(sources)
        owner = runtime._execution_service._display_owner
        if owner is not None:
            if view._character_context is not owner.character:
                raise RuntimeError("Display character attachment changed")
            for name, value in view._console_character_view_bindings.items():
                setattr(owner.character, name, value)

    def clear_view_hooks(self, *, only: str | None = None) -> None:
        runtime = self._runtime
        self._controller = runtime._chat_controller
        self._controller_app = getattr(self._controller, "app", None)
        self._visit += 1
        self._view = None
        self._attachment_generation = None
        self._hook_sources = ()
        owner = runtime._execution_service._display_owner
        if owner is not None:
            from tldw_chatbook.UI.Console_Modules.character_context import (
                ConsoleCharacterQueryHandoffCapability,
            )

            for name in (
                "_activate_target",
                "_navigate_roleplay",
                "_navigate_repair",
                "_navigate_inspection",
                "_navigate_unavailable_browse",
                "_navigate_roleplay_home",
                "_navigate_library_home",
                "_start_console",
            ):
                setattr(owner.character, name, owner.no_view_action)
            owner.character._progress_counts = None
            owner.character._query_handoff = None
            owner.character._query_handoff_capability = (
                ConsoleCharacterQueryHandoffCapability()
            )
        for slot in runtime._execution_service._runtime_source.CONSOLE_VIEW_HOOK_SLOTS:
            if only is not None and slot.target != only:
                continue
            target = runtime._hook_target(slot.target)
            if target is not None:
                setattr(target, slot.name, slot.viewless_default)

    def _decision_setter(self, kind: str):
        controller = self._controller
        if controller is None:
            return None
        return {
            "approval": controller.set_pending_approval,
            "question": controller.set_pending_question,
            "chat_create": controller.set_pending_chat_create,
            "skill_install": controller.set_pending_skill_install,
            "skill_script": controller.set_pending_skill_script,
            "worktree_merge": controller.set_pending_worktree_merge,
        }[kind]

    def decision_available(self, kind: str) -> bool:
        return self._decision_setter(kind) is not None

    def pending_router_available(self) -> bool:
        return (
            self._controller is not None
            and self._controller.set_pending_decision is not None
        )

    def present_pending_projection(self, projection) -> bool:
        """UI-thread adoption preserves the original setter's mounted result."""
        stamp = self._stamp()
        if not self._current_visit(stamp) or self._controller is None:
            return False
        setter = self._controller.set_pending_decision
        if setter is not None:
            return setter(projection) is True
        if projection is None:
            for kind in ("approval", "skill_install", "skill_script"):
                setter = self._decision_setter(kind)
                if setter is not None:
                    setter(None)
            return False
        setter = self._decision_setter(projection.decision_type)
        if setter is None:
            return False
        setter(dict(projection.payload))
        return True

    def present_chat_create(self, host, payload) -> None:
        stamp = self._stamp()

        def apply():
            if not self._current_visit(stamp):
                return
            setter = self._decision_setter("chat_create")
            if setter is None:
                return
            active_session_id = host.read_controller_store().active_session_id or ""
            if payload is None:
                setter(
                    host.read_controller__head_round_payload()(
                        host.read_controller__parked_chat_create_payloads(),
                        active_session_id,
                    )
                )
                return
            owning_session_id = str(payload.get("session_id") or "")
            if owning_session_id in host.read_controller__session_close_generations():
                return
            with host.read_controller__pending_chat_create_lock():
                state = host.read_controller__pending_chat_create_rounds().get(
                    payload.get("request_id")
                )
                if state is None or state.get("revoked") or state["event"].is_set():
                    return
                if (
                    state.get("session_scoped", True)
                    and owning_session_id != active_session_id
                ):
                    return
            setter(payload)

        self._dispatch(apply)

    def project_task_panel(self, session_id, tasks) -> None:
        stamp = self._stamp()

        def apply():
            if self._current_visit(stamp) and self._controller is not None:
                setter = self._controller.set_task_panel
                if setter is not None:
                    setter(session_id, tasks)

        self._dispatch(apply)

    def deliver_permission_summary(self, host, round_id, payload, text) -> None:
        stamp = self._stamp()

        def apply():
            if self._current_visit(stamp):
                host._deliver_permission_summary(round_id, payload, text)

        self._dispatch(apply)

    def project_attention(self, host, *, ring: bool) -> None:
        stamp = self._stamp()
        ring = ring and not bool(getattr(self._app, "is_headless", False))

        def apply():
            if not self._current_visit(stamp):
                return
            from tldw_chatbook.UI.Navigation.main_navigation import (
                set_console_attention,
            )

            set_console_attention(self._app, host.pending_total())
            bell = getattr(self._app, "bell", None) if ring else None
            if callable(bell):
                bell()

        try:
            self._dispatch(apply)
        except Exception:  # noqa: BLE001 - attention cannot fail a decision.
            pass

    def _dispatch(self, callback) -> None:
        """Adopt on the App loop; marshal only actual worker callers."""
        from threading import get_ident

        self.require_association(self.association)
        if (
            getattr(self._app, "_loop", None) is not None
            and getattr(self._app, "_thread_id", None) == get_ident()
        ):
            callback()
        else:
            self._app.call_from_thread(callback)

    def present_decision(self, kind: str, payload: dict[str, Any] | None) -> None:
        stamp = self._stamp()

        def apply():
            if not self._current_visit(stamp):
                return
            setter = self._decision_setter(kind)
            if setter is not None:
                setter(payload)

        self._dispatch(apply)

    def clear_decision(self, kind: str) -> None:
        self.present_decision(kind, None)

    def project_pending(self) -> None:
        stamp = self._stamp()

        def apply():
            if self._current_visit(stamp) and self._controller is not None:
                self._controller.project_pending_decision_for_active_session()

        self._dispatch(apply)

    def announce_hidden(self, message: str) -> bool:
        return self.notice(message, severity="warning")

    def park_decision(self, session_id: str) -> None:
        stamp = self._stamp()

        def apply():
            if not self._current_visit(stamp) or self._controller is None:
                return
            callback = self._controller.park_pending_approval
            if callback is not None:
                callback(session_id)

        self._dispatch(apply)

    def remount_decision(self, host, kind, session_id, *, after=None) -> None:
        """Re-derive the head on the UI thread, at the original read position."""
        stamp = self._stamp()

        def apply():
            if not self._current_visit(stamp):
                return
            setter = self._decision_setter(kind)
            if setter is None:
                return
            target = session_id
            if target is None:
                target = host._active_session_id()
            elif target != host._active_session_id():
                return
            host.refresh_decision_clocks()
            payload = host.head_round_payload(kind, target)
            setter(payload)
            hook = after if after is not None else host.after_remount.get(kind)
            if hook is not None and isinstance(payload, dict):
                hook(payload)

        self._dispatch(apply)

    def begin_display_full(self, view):
        """Register the actual original FULL Task, before its first await."""
        import asyncio
        from .console_display_phases import ConsoleDisplayPhaseRefused
        from .console_received_dispatch import stock_native_methods
        from tldw_chatbook.UI.Screens.chat_screen import (
            ChatScreen,
            _CONSOLE_DISPLAY_PARENT_READERS,
        )

        self.require_association(self.association)
        task = asyncio.current_task()
        if (
            task is None
            or type(view) is not ChatScreen
            or self._view is not view
            or self._runtime.view is not view
            or view.app_instance is not self._app
            or not stock_native_methods(view, _CONSOLE_DISPLAY_PARENT_READERS)
            or not view._console_sync_in_progress
            or task in self._display_full_entries
        ):
            raise ConsoleDisplayPhaseRefused("Original FULL Task entry changed")
        entry = _DisplayFullEntry(self, view, task, self._stamp())
        entry.character_demand = self._display_character_demands.get(task)
        self._display_full_entries[task] = entry
        demand = self._display_context_demands.get(task)
        if demand is not None:
            demand.append(entry)
        return entry

    def require_display_full_entry(self, entry, owner, task):
        """Only the real registered FULL owns autonomous display admission."""
        from .console_display_phases import ConsoleDisplayPhaseRefused

        if (
            type(entry) is not _DisplayFullEntry
            or entry.port is not self
            or self._display_full_entries.get(task) is not entry
            or entry.task is not task
            or entry.body_returned
            or task.done()
            or entry.full is not None
            or owner.domain.presentation is not self
            or owner.runtime is not self._runtime
            or owner.store is not self._runtime._chat_store
            or owner.controller is not self._runtime._chat_controller
            or self._require_display_owner(owner) is not entry.view
            or not self._current_visit(entry.stamp)
            or not entry.view._console_sync_in_progress
        ):
            raise ConsoleDisplayPhaseRefused("Original FULL registration changed")

    def end_display_full(self, entry):
        """Do not erase an uncertain native registration when its Task exits."""
        import asyncio

        if (
            type(entry) is not _DisplayFullEntry
            or entry.port is not self
            or entry.task is not asyncio.current_task()
            or self._display_full_entries.get(entry.task) is not entry
        ):
            raise RuntimeError("Display parent entry retirement changed")
        entry.body_returned = True
        full = entry.full
        if full is None:
            # Pre-selection decline / entry never admitted native display.
            del self._display_full_entries[entry.task]
            return
        effect = full.effect
        if (
            effect is not None
            and effect.completed.done()
            and effect.physically_retired is True
        ):
            del self._display_full_entries[entry.task]

    async def _run_display_context_demand(
        self, owner, stamp, entries, *, character_demand=None, blocked_entries=None
    ):
        """Observe only this original FULL invocation; never grant it authority."""
        import asyncio

        task = asyncio.current_task()
        if task in self._display_context_demands or not self._current_visit(stamp):
            return
        view = self._require_display_owner(owner)
        if view is not stamp[1]:
            return
        if character_demand not in (None, "scope", "forced-scope", "refresh"):
            raise RuntimeError("Unknown fixed Character display demand")
        self._display_context_demands[task] = entries
        if character_demand is not None:
            self._display_character_demands[task] = character_demand
        try:
            await view._sync_native_console_chat_ui()
            if not entries and blocked_entries is not None:
                blocked_entries.extend(
                    entry
                    for entry in self._display_full_entries.values()
                    if entry.task is not task
                    and entry.view is view
                    and entry.stamp == stamp
                    and not entry.body_returned
                )
        finally:
            del self._display_context_demands[task]
            self._display_character_demands.pop(task, None)

    async def _join_display_operation(self, operation):
        """Retain an actual FULL through observer cancellation and physical exit."""
        import asyncio

        cancelled = None
        while not operation.done():
            try:
                await asyncio.shield(operation)
            except asyncio.CancelledError as error:
                cancelled = error
            except BaseException:
                break
        if operation.cancelled():
            if cancelled is not None:
                raise cancelled
            return False
        error = operation.exception()
        if cancelled is not None:
            raise cancelled
        if error is not None:
            raise error
        return True

    async def refresh_display_character(self, owner, character, kind):
        """Carry one original Character action through its exact FULL owner."""
        import asyncio
        from .console_display_phases import ConsoleDisplayPhaseRefused

        owner.require_current()
        owner.require_character_sources(character)
        if kind not in ("scope", "forced-scope", "refresh"):
            raise ConsoleDisplayPhaseRefused("Character demand kind changed")
        view = self._require_display_owner(owner)
        stamp = self.capture_display(owner)
        session = owner.active_session()
        if view is None or stamp is None or session is None:
            return False
        incarnation = session.incarnation_id
        current = asyncio.current_task()
        if current in self._display_full_entries or current in owner._full_tasks:
            raise ConsoleDisplayPhaseRefused(
                "Character demand would await its own FULL"
            )
        entries = [
            entry
            for entry in self._display_full_entries.values()
            if entry.view is view and entry.stamp == stamp and not entry.body_returned
        ]
        if len(entries) > 1:
            raise ConsoleDisplayPhaseRefused(
                "Character demand has competing FULL owners"
            )
        if entries:
            await self._join_display_operation(entries[0].task)
        owner.require_character_sources(character)
        if (
            not self._current_visit(stamp)
            or self._require_display_owner(owner) is not view
            or owner.active_session() is not session
            or session.incarnation_id != incarnation
        ):
            return False
        for relayed in (False, True):
            entries, blocked = [], []
            body = self._run_display_context_demand(
                owner, stamp, entries, character_demand=kind, blocked_entries=blocked
            )
            try:
                worker = view.run_worker(
                    body, exclusive=False, group="console-character-presentation"
                )
            except BaseException:
                body.close()
                raise
            operation = worker._task
            if not await self._join_display_operation(operation):
                return False
            if entries or kind == "scope" or relayed or len(blocked) != 1:
                break
            # No entry means this invocation admitted no Character/native work.
            # Only its observed competing entry can carry the one owed relay.
            predecessor = blocked[0]
            await self._join_display_operation(predecessor.task)
            prior = predecessor.full
            owner.require_character_sources(character)
            if (
                prior is None
                or not predecessor.body_returned
                or prior.selection.parent_entry is not predecessor
                or prior.owner is not owner
                or prior.session is not session
                or prior.stamp != stamp
                or not prior.phase_issuance_closed
                or not prior.physically_retired
                or prior.effect is None
                or not prior.effect.completed.done()
                or prior.effect.physically_retired is not True
                or not self._current_visit(stamp)
                or owner.active_session() is not session
                or session.incarnation_id != incarnation
            ):
                return False
        owner.require_character_sources(character)
        if (
            len(entries) != 1
            or not self._current_visit(stamp)
            or owner.active_session() is not session
            or session.incarnation_id != incarnation
        ):
            return False
        entry = entries[0]
        full = entry.full
        if full is None:
            return False
        effect = full.effect
        if not (
            entry.port is self
            and entry.task is operation
            and entry.body_returned
            and entry.character_demand == kind
            and full.owner is owner
            and full.session is session
            and full.selection.parent_entry is entry
            and full.stamp == stamp
            and full.logical is not None
            and full.logical.callback_returned
            and full.phase_issuance_closed
            and full.physically_retired
            and effect is not None
            and effect.completed.done()
            and effect.physically_retired is True
        ):
            raise ConsoleDisplayPhaseRefused("Character demand retirement is unproven")
        return full.character_result

    async def warm_display_context(self, owner, snapshot, session_id, key, stamp):
        """Join an exact existing or scheduled FULL, outside settings/native locks."""
        import asyncio
        from .console_display_phases import ConsoleDisplayPhaseRefused

        view = self._require_display_owner(owner)
        if view is not stamp[1] or not self._current_visit(stamp):
            return False
        session = owner.store._sessions.get(session_id)
        if session is None or owner.store.active_session_id != session_id:
            return False
        current = asyncio.current_task()
        if current in self._display_full_entries or current in owner._full_tasks:
            raise ConsoleDisplayPhaseRefused("Context demand would await its own FULL")
        entries = [
            entry
            for entry in self._display_full_entries.values()
            if entry.view is view and entry.stamp == stamp and not entry.body_returned
        ]
        if len(entries) > 1:
            raise ConsoleDisplayPhaseRefused("Context demand has competing FULL owners")
        worker = None
        if entries:
            operation = entries[0].task
        else:
            operation_body = self._run_display_context_demand(owner, stamp, entries)
            try:
                worker = view.run_worker(
                    operation_body,
                    exclusive=False,
                    group="console-context-presentation",
                )
            except BaseException:
                operation_body.close()
                raise
            operation = worker._task
        cancelled = None
        while not operation.done():
            try:
                await asyncio.shield(operation)
            except asyncio.CancelledError as error:
                cancelled = error
            except BaseException:
                break
        if operation.cancelled():
            if cancelled is not None:
                raise cancelled
            return False
        error = operation.exception()
        if cancelled is not None:
            raise cancelled
        if error is not None:
            raise error
        owner.require_current()
        if (
            len(entries) != 1
            or not self._current_visit(stamp)
            or owner.context is not snapshot
            or snapshot.screen is not view
            or owner.store._sessions.get(session_id) is not session
            or owner.store.active_session_id != session_id
            or key != snapshot._cache_key(owner.controller, session_id)
        ):
            return False
        entry = entries[0]
        full = entry.full
        if full is None:
            return False
        effect = full.effect
        return bool(
            entry.port is self
            and entry.task is operation
            and entry.body_returned
            and full.owner is owner
            and full.session is session
            and full.selection.parent_entry is entry
            and full.stamp == stamp
            and full.logical is not None
            and full.logical.callback_returned
            and full.phase_issuance_closed
            and full.physically_retired
            and effect is not None
            and effect.completed.done()
            and effect.physically_retired is True
        )

        # Otherwise the exact entry, effect, logical observer and original
        # resource records remain retained. Fencing/recovery owns that refusal.


# Capture actual defining functions, defaults, globals and closures once.
from .console_domain_services import _definition

_PORT_DEFINITIONS = tuple(
    (name, _definition(value))
    for name, value in vars(ConsolePresentationPort).items()
    if callable(value)
)
