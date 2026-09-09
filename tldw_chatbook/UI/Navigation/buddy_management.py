"""One app-owned entry point for Buddy management from any primary destination."""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from typing import Any

from tldw_chatbook.Persona_Buddy.interaction import (
    BuddyBinding,
    BuddyInteractionPreferences,
    parse_preferences,
    serialize_preferences,
)


class BuddyManagementCoordinator:
    """Own staged-management application and scope; never own Console execution."""

    def __init__(
        self, app: Any, *, controller: Any = None, library: Any = None
    ) -> None:
        self.app = app
        self._controller = controller
        self._library = library
        self.preferences = parse_preferences(
            app.app_config.get("buddy_interaction", {})
        )
        self._apply_lock = asyncio.Lock()
        self._opening = False
        self._modal: Any = None
        self._scope_timer: Any = None
        self._scope_configured = "buddy_interaction" in app.app_config
        self._promotion_pending = False
        self._promotion_retry_after = 0.0

    @property
    def controller(self) -> Any:
        if self._controller is None:
            self._controller = self.app.ensure_persona_buddy_controller()
        if self._controller is None:
            raise ValueError("Buddy storage is unavailable. Check the active profile.")
        return self._controller

    @property
    def library(self) -> Any:
        if self._library is None:
            from tldw_chatbook.config import get_user_data_dir
            from tldw_chatbook.Persona_Buddy.library import BuddyLibrary

            db = getattr(self.app, "chachanotes_db", None)
            if db is None:
                raise ValueError(
                    "Buddy storage is unavailable. Check the active profile."
                )
            personas = getattr(self.app, "local_character_persona_service", None)
            self._library = BuddyLibrary(
                db,
                get_user_data_dir(),
                persona_reader=(
                    personas.get_persona_profile if personas is not None else None
                ),
            )
        return self._library

    def request_open(self) -> None:
        """Open at most one management surface through an app-owned worker."""
        if self._opening or (self._modal is not None and self._modal.is_mounted):
            return
        self._opening = True
        self.app.run_worker(self._open(), group="buddy-management-open", exclusive=True)

    def request_visibility(self, action: str) -> None:
        """Control the independent Buddy without a Persona selection."""
        if action == "manage":
            self.request_open()
        elif action in {"show", "close", "disable"}:
            self.app.run_worker(
                self._set_visibility(action), group="buddy-visibility", exclusive=False
            )

    async def _set_visibility(self, action: str) -> None:
        async with self._apply_lock:
            controller = self.controller
            previous = controller.current_preferences()
            if action == "show" and previous.selection is None:
                self.request_open()
                return
            changes = (
                {"enabled": True, "open": True}
                if action == "show"
                else {"open": False}
                if action == "close"
                else {"enabled": False}
            )
            revision = controller.apply_preferences_patch(**changes)
            try:
                if not await controller.persist_preferences_revision(revision):
                    raise ValueError("Could not save Buddy visibility. Retry.")
            except Exception:  # noqa: BLE001 - keep current settings on storage failure
                controller.rollback_preferences_revision(revision, previous)
                self.app.notify(
                    "Could not save Buddy visibility. Retry.", severity="error"
                )
                return
            from tldw_chatbook.Persona_Buddy.preferences import (
                serialize_persona_buddy_preferences,
            )

            self.app.app_config["persona_buddy"] = serialize_persona_buddy_preferences(
                controller.current_preferences()
            )
            self.reconcile_scope()
            pending = self.app.reconcile_persona_buddy_view()
            if asyncio.iscoroutine(pending):
                await pending

    def request_interaction(self) -> None:
        """Open the explicitly followed target without selecting a Console tab."""
        binding = self.preferences.binding
        if binding is None:
            self.request_open()
        elif binding.kind == "conversation":
            from .buddy_conversation import open_buddy_conversation

            open_buddy_conversation(self.app, binding)
        else:
            from .buddy_workspace import open_buddy_workspace

            open_buddy_workspace(self.app, binding)

    async def _open(self) -> None:
        from tldw_chatbook.Widgets.Persona_Widgets.buddy_management_modal import (
            BuddyManagementChoice,
            BuddyManagementModal,
        )

        try:
            controller = self.controller
            # Selection/startup publication has its own authority. Merely opening
            # this form lists installed content and never imports or seeds it.
            buddies = await asyncio.to_thread(self.library.list_buddies)
            personas = await asyncio.to_thread(self._persona_choices)
            targets = await self._target_choices()
            saved = controller.current_preferences()
            buddy_id = getattr(saved.selection, "buddy_id", None)
            selected_buddy = (
                await asyncio.to_thread(self.library.get_buddy, buddy_id)
                if buddy_id
                else None
            )

            async def artwork_page(offset: int) -> tuple[tuple[str, str], ...]:
                page = await asyncio.to_thread(
                    self.library.list_buddies, limit=100, offset=offset
                )
                return tuple((record.name, record.id) for record in page)

            initial = BuddyManagementChoice(
                enabled=saved.enabled and saved.open,
                buddy_id=buddy_id,
                binding=self._binding_for_form(targets),
                animated=self.preferences.animated,
                speak_responses=self.preferences.speak_responses,
                width=saved.geometry.width,
                height=saved.geometry.height,
            )
            revision = controller.snapshot().preferences_generation
            imports: dict[str, str] = {}

            async def commit(choice: Any) -> None:
                nonlocal revision
                try:
                    await self.apply_choice(
                        choice, expected_revision=revision, imports=imports
                    )
                except ValueError as exc:
                    if getattr(exc, "buddy_retry_revision", None) is not None:
                        revision = exc.buddy_retry_revision
                    raise
                self.app.notify("Buddy settings applied.")
                try:
                    pending = self.app.reconcile_persona_buddy_view()
                    if asyncio.iscoroutine(pending):
                        await pending
                except Exception:  # noqa: BLE001 - settings already saved
                    self.app.notify(
                        "Buddy settings saved, but the view could not refresh. Try Show Buddy again.",
                        severity="warning",
                    )

            self._modal = BuddyManagementModal(
                buddies=tuple((record.name, record.id) for record in buddies),
                targets=targets,
                personas=personas,
                initial=initial,
                preview=self._preview,
                apply=commit,
                artwork_page=artwork_page,
                selected_buddy=(selected_buddy.name, selected_buddy.id)
                if selected_buddy
                else None,
            )
            self.app.push_screen(self._modal, self._closed)
        except Exception:  # noqa: BLE001 - app boundary keeps storage faults out of the message pump
            self.app.notify(
                "Could not open the Buddy library. Check the active profile and retry.",
                severity="error",
            )
        finally:
            self._opening = False

    def _persona_choices(self) -> tuple[tuple[str, str], ...]:
        service = getattr(self.app, "local_character_persona_service", None)
        if service is None:
            return ()
        rows = []
        offset = 0
        while True:
            page = service.list_persona_profiles(
                active_only=True, limit=100, offset=offset
            )
            rows.extend(
                (str(row.get("name") or "Unnamed Persona"), str(row["id"]))
                for row in page
            )
            if len(page) < 100:
                return tuple(rows)
            offset += len(page)

    def _runtime_sessions(self) -> tuple[Any, ...]:
        runtime = getattr(self.app, "console_runtime", None)
        store = getattr(runtime, "chat_store", None)
        return tuple(store.sessions()) if store is not None else ()

    def _binding_for_form(self, targets: tuple[Any, ...]) -> BuddyBinding | None:
        """Match a restored/first-saved conversation while preserving stale-slot guards."""
        binding = self.preferences.binding
        if binding is None or binding.kind != "conversation":
            return binding
        session = binding.resolve_session(self._runtime_sessions())
        if session is not None:
            for target in targets:
                if target.binding.resolve_session((session,)) is session:
                    return target.binding
        return binding

    async def _promote_saved_binding(self, expected: BuddyBinding) -> bool:
        """Persist a first-save identity only while that exact pin still owns settings."""
        async with self._apply_lock:
            if (
                self.preferences.binding != expected
                or expected.ephemeral
                or expected.conversation_id
            ):
                return False
            session = expected.resolve_session(self._runtime_sessions())
            if session is None or not session.persisted_conversation_id:
                return False
            promoted = replace(
                self.preferences, binding=BuddyBinding.for_session(session)
            )
            encoded = serialize_preferences(promoted)
            revision = self.controller.snapshot().preferences_generation
            saved = await self.controller.persist_preferences_revision(
                revision, extra_sections={"buddy_interaction": encoded}
            )
            if not saved:
                raise ValueError(
                    "Could not save the Buddy’s conversation link. It will retry while this app is open."
                )
            self.preferences = promoted
            self.app.app_config["buddy_interaction"] = dict(encoded)
            return True

    async def _finish_binding_promotion(self, expected: BuddyBinding) -> None:
        try:
            await self._promote_saved_binding(expected)
        except Exception:  # noqa: BLE001 - failed preference I/O must not affect a conversation
            self._promotion_retry_after = time.monotonic() + 30.0
            self.app.notify(
                "Could not save the Buddy’s conversation link. It will retry while this app is open.",
                severity="warning",
            )
        finally:
            self._promotion_pending = False

    def _persona_label(self, persona_id: str | None) -> str:
        if not persona_id:
            return "None"
        service = getattr(self.app, "local_character_persona_service", None)
        try:
            row = service.get_persona_profile(persona_id) if service else None
            if not row or row.get("deleted"):
                return "Unavailable"
            return str(row.get("name") or "Unnamed Persona")
        except Exception:  # noqa: BLE001 - display-only lookup degrades independently
            return "Unavailable"

    def _workspace_persona_label(self, workspace: Any) -> str:
        from tldw_chatbook.Workspaces.assistant_defaults import (
            resolve_effective_assistant_default,
        )

        service = getattr(self.app, "local_character_persona_service", None)
        try:
            effective = resolve_effective_assistant_default(
                workspace.assistant_defaults,
                service.get_persona_profile if service else lambda _: None,
            )
            if effective.status == "available":
                return effective.label or "Unnamed Persona"
            return (
                "Unavailable (new conversations use None)"
                if effective.status == "unavailable"
                else "None"
            )
        except Exception:  # noqa: BLE001 - display-only lookup cannot prevent management
            return "Unavailable (new conversations use None)"

    async def _target_choices(self) -> tuple[Any, ...]:
        from tldw_chatbook.Widgets.Persona_Widgets.buddy_management_modal import (
            BuddyTargetChoice,
        )
        from tldw_chatbook.Workspaces.models import DEFAULT_WORKSPACE_ID

        runtime = getattr(self.app, "console_runtime", None)
        controller = getattr(runtime, "chat_controller", None)
        targets = []
        for session in self._runtime_sessions():
            if session.runtime_backend != "local":
                continue
            busy = (
                controller is not None
                and not controller.run_state_for(session.id).is_send_allowed
            )
            is_character = session.assistant_kind == "character"
            targets.append(
                BuddyTargetChoice(
                    f"conversation:{session.id}",
                    f"Conversation: {session.title} · {session.id[:8]}",
                    BuddyBinding.for_session(session),
                    current_persona=await asyncio.to_thread(
                        self._persona_label,
                        session.assistant_id
                        if session.assistant_kind == "persona"
                        else None,
                    ),
                    persona_editable=not busy and not is_character,
                    persona_unavailable_reason=(
                        "Wait for this conversation’s current run to finish before changing its Persona."
                        if busy
                        else "This conversation uses a character. Manage its character settings in Console."
                        if is_character
                        else ""
                    ),
                )
            )
        registry = getattr(self.app, "workspace_registry_service", None)
        if registry is not None:
            workspaces = await asyncio.to_thread(registry.list_workspaces)
            for workspace in workspaces:
                authority = str(
                    getattr(workspace.authority, "value", workspace.authority)
                )
                if (
                    workspace.archived
                    or workspace.workspace_id == DEFAULT_WORKSPACE_ID
                    or authority != "local-only"
                ):
                    continue
                targets.append(
                    BuddyTargetChoice(
                        f"workspace:{workspace.workspace_id}",
                        f"Workspace: {workspace.name}",
                        BuddyBinding(
                            kind="workspace", target_id=workspace.workspace_id
                        ),
                        current_persona=await asyncio.to_thread(
                            self._workspace_persona_label, workspace
                        ),
                    )
                )
        return tuple(targets)

    async def _preview(self, buddy_id: str, state: str) -> Any:
        from rich.text import Text

        from tldw_chatbook.Persona_Buddy.rendering import prepare_persona_buddy_frame

        def render() -> Any:
            resolution = self.library.resolve_preview(
                buddy_id, state=state, reduced_motion=True
            )
            if not resolution.frames:
                return Text("No preview is available for this expression.")
            return prepare_persona_buddy_frame(
                resolution.frames[0],
                resolution_cache_identity=resolution.cache_identity,
                cols=24,
                lines=10,
            ).renderable

        return await asyncio.to_thread(render)

    def _closed(self, choice: Any) -> None:
        self._modal = None
        if choice is not None:
            self.app.run_worker(
                self._apply_and_report(choice),
                group="buddy-management-apply",
                exclusive=False,
            )

    async def _apply_and_report(self, choice: Any) -> None:
        try:
            await self.apply_choice(choice)
        except ValueError as exc:
            self.app.notify(str(exc), severity="error")
        except Exception:  # noqa: BLE001 - app boundary keeps storage faults out of the message pump
            self.app.notify(
                "Could not finish applying Buddy settings. Reopen Buddy settings to review the current selection and retry.",
                severity="error",
            )
        else:
            self.app.notify("Buddy settings applied.")
            pending = self.app.reconcile_persona_buddy_view()
            if asyncio.iscoroutine(pending):
                await pending

    async def _validate_binding(self, binding: BuddyBinding | None) -> Any:
        if binding is None:
            return None
        if binding.kind == "conversation":
            session = binding.resolve_session(self._runtime_sessions())
            if session is None:
                raise ValueError(
                    "The bound conversation is unavailable. Open it in Console or choose another target."
                )
            return session
        registry = getattr(self.app, "workspace_registry_service", None)
        workspace = (
            await asyncio.to_thread(registry.get_workspace, binding.target_id)
            if registry
            else None
        )
        if (
            workspace is None
            or workspace.archived
            or str(getattr(workspace.authority, "value", workspace.authority))
            != "local-only"
        ):
            raise ValueError(
                "The bound workspace is unavailable. Choose another target."
            )
        return workspace

    async def apply_choice(
        self,
        choice: Any,
        *,
        expected_revision: int | None = None,
        imports: dict[str, str] | None = None,
    ) -> None:
        """Validate, publish requested artwork, then batch preference persistence.

        Import failure cannot change selection. A failed preference write rolls back
        only this exact in-memory revision, preserving any newer user changes.
        """
        from tldw_chatbook.Persona_Buddy.preferences import BuddySelection
        from tldw_chatbook.Widgets.Persona_Widgets.buddy_management_modal import (
            PERSONA_UNCHANGED,
        )

        async with self._apply_lock:

            def require_current() -> None:
                if (
                    expected_revision is not None
                    and self.controller.snapshot().preferences_generation
                    != expected_revision
                ):
                    raise ValueError(
                        "Buddy settings changed elsewhere. Cancel and reopen to review the current selection before applying."
                    )

            require_current()
            target = await self._validate_binding(choice.binding)
            if choice.persona_choice != PERSONA_UNCHANGED:
                # Admission runs before artwork publication or preference mutation.
                from tldw_chatbook.Chat.console_persona_assignment import (
                    prepare_buddy_persona_assignment,
                )

                assignment = prepare_buddy_persona_assignment(
                    self.app, choice.binding, target, choice.persona_choice
                )
            else:
                assignment = None
            selected_id = choice.buddy_id
            if choice.import_path:
                selected_id = (
                    imports.get(choice.import_path) if imports is not None else None
                )
                if selected_id is None:
                    try:
                        review = await asyncio.to_thread(
                            self.library.review_archive, choice.import_path
                        )
                        require_current()
                        record = await asyncio.to_thread(
                            self.library.publish_review, review
                        )
                    except Exception as exc:
                        raise ValueError(
                            "Could not import this Buddy pack. Check the path, pack format and profile storage, then retry."
                        ) from exc
                    selected_id = record.id
                    if imports is not None:
                        imports[choice.import_path] = selected_id
                elif (
                    await asyncio.to_thread(self.library.get_buddy, selected_id) is None
                ):
                    raise ValueError(
                        "The imported Buddy was removed. Cancel and reopen to import it again."
                    )
            elif selected_id is not None:
                record = await asyncio.to_thread(self.library.get_buddy, selected_id)
                if record is None:
                    raise ValueError(
                        "Selected artwork is unavailable. Choose another Buddy or import its pack."
                    )
            # I/O above can outlive a deleted or repurposed target.
            await self._validate_binding(choice.binding)
            require_current()
            controller = self.controller
            previous = controller.current_preferences()
            selected = (
                BuddySelection(selected_id)
                if selected_id is not None
                else previous.selection
            )
            if choice.enabled and selected is None:
                raise ValueError("Choose artwork before enabling the Buddy.")
            interaction = BuddyInteractionPreferences(
                binding=choice.binding,
                animated=choice.animated,
                speak_responses=choice.speak_responses,
            )
            encoded = serialize_preferences(interaction)
            revision = controller.apply_preferences_patch(
                enabled=choice.enabled,
                open=choice.enabled or previous.open,
                selection=selected,
                geometry=replace(
                    previous.geometry, width=choice.width, height=choice.height
                ),
            )
            try:
                saved = await controller.persist_preferences_revision(
                    revision, extra_sections={"buddy_interaction": encoded}
                )
            except Exception as exc:
                restored = controller.rollback_preferences_revision(revision, previous)
                error = ValueError(
                    "Could not save Buddy settings. Check profile storage and retry."
                )
                if restored:
                    error.buddy_retry_revision = (
                        controller.snapshot().preferences_generation
                    )
                raise error from exc
            if not saved:
                restored = controller.rollback_preferences_revision(revision, previous)
                error = ValueError(
                    "Could not save Buddy settings. Your previous settings are retained; retry."
                    if restored
                    else "Buddy settings changed elsewhere while saving. Cancel and reopen to review the current selection."
                )
                if restored:
                    error.buddy_retry_revision = (
                        controller.snapshot().preferences_generation
                    )
                raise error
            self.preferences = interaction
            self._scope_configured = True
            self.app.app_config["buddy_interaction"] = dict(encoded)
            from tldw_chatbook.Persona_Buddy.preferences import (
                serialize_persona_buddy_preferences,
            )

            self.app.app_config["persona_buddy"] = serialize_persona_buddy_preferences(
                controller.current_preferences()
            )
            controller.invalidate_profile()
            self.start_scope_tracking()
            if assignment is not None:
                # This is a separate assistant-domain commit, never hidden inside
                # artwork selection. Its own version guard retains the old Persona
                # on failure and reports the exact partial Apply outcome.
                try:
                    await assignment.apply()
                except Exception as exc:
                    error = ValueError(
                        "Buddy settings were saved, but the Persona changed or could not be saved. Your staged Persona choice is retained; review it and retry."
                    )
                    error.buddy_retry_revision = revision
                    raise error from exc

    def start_scope_tracking(self) -> None:
        """Resume persisted scope when an enabled Buddy first becomes visible."""
        self.reconcile_scope()
        if self._scope_timer is None and hasattr(self.app, "set_interval"):
            self._scope_timer = self.app.set_interval(0.5, self.reconcile_scope)

    def reconcile_scope(self) -> None:
        """Filter trusted lifecycle events and replay current state after a rebind."""
        if not self._scope_configured:
            return
        speech = getattr(self.app, "buddy_speech_coordinator", None)
        if speech is not None or self.preferences.speak_responses:
            from .buddy_speech import ensure_buddy_speech

            presentation = self.controller.current_preferences()
            ensure_buddy_speech(self.app).configure(
                self.preferences.binding,
                self.preferences.speak_responses
                and presentation.enabled
                and presentation.open,
            )
        runtime = getattr(self.app, "console_runtime", None)
        if runtime is None:
            return
        controller = getattr(runtime, "chat_controller", None)
        if controller is None:
            return
        binding = self.preferences.binding
        sessions = self._runtime_sessions()
        if binding is None:
            selected = ()
        elif binding.kind == "conversation":
            resolved = binding.resolve_session(sessions)
            selected = (resolved,) if resolved is not None else ()
            if (
                resolved is not None
                and resolved.persisted_conversation_id
                and not binding.ephemeral
                and binding.conversation_id is None
                and not self._promotion_pending
                and time.monotonic() >= self._promotion_retry_after
            ):
                self._promotion_pending = True
                self.app.run_worker(
                    self._finish_binding_promotion(binding),
                    group="buddy-binding-promotion",
                    exclusive=False,
                    exit_on_error=False,
                )
        else:
            selected = tuple(
                session for session in sessions if binding.includes(session)
            )
        sink = runtime.persona_buddy_sink
        changed = sink.set_scope(
            session_ids=frozenset(session.id for session in selected),
            conversation_ids=frozenset(
                session.persisted_conversation_id
                for session in selected
                if session.persisted_conversation_id
            ),
        )
        if changed:
            for session in selected:
                sink.run_state(
                    session.id, controller.run_state_for(session.id).status, replay=True
                )
                for kind in (
                    "approval",
                    "skill_install",
                    "skill_script",
                    "question",
                    "worktree_merge",
                ):
                    for payload in controller._interrupt_host.session_round_payloads(
                        kind, session.id
                    ):
                        request_id = payload.get("request_id") or payload.get(
                            "round_id"
                        )
                        if request_id:
                            sink.approval_round(
                                session.id, str(request_id), pending=True
                            )


def get_buddy_management(app: Any) -> BuddyManagementCoordinator:
    """Lazily compose the one app-owned Buddy interaction/management coordinator."""
    coordinator = getattr(app, "_buddy_management", None)
    if coordinator is None:
        coordinator = BuddyManagementCoordinator(app)
        app._buddy_management = coordinator
    return coordinator


def open_buddy_management(app: Any) -> None:
    """Shared composer-menu and floating-Buddy settings action."""
    get_buddy_management(app).request_open()


def open_buddy_interaction(app: Any) -> None:
    """Shared click/keyboard action on the current floating Buddy."""
    get_buddy_management(app).request_interaction()
