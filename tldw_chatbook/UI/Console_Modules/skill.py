"""DOM-free Console skill discovery, trust, and decision policy."""

from __future__ import annotations

import asyncio
import sys
import threading
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from functools import partial
from types import FunctionType, ModuleType
from typing import Any

from loguru import logger

from ...Chat.console_command_grammar import CommandParse
from ...Chat.console_skill_resolver import (
    SKILL_UNTRUSTED_REFUSE,
    SkillCommandCandidate,
    format_skills_list,
)
from ..Screens.chat_screen_state import TaskResumeState

CONSOLE_SKILL_NEEDS_REVIEW_HINT_TEMPLATE = (
    "{count} matching skill(s) need review in Library ▸ Skills before running."
)


class ConsoleSkillController:
    """Own live Console skill policy without owning Textual presentation."""

    def __init__(
        self,
        *,
        app_instance: object,
        append_native_console_system_message: Callable[[str], Awaitable[None]],
        sync_console_command_popup: Callable[[], None],
        task_resume_state: Callable[[], TaskResumeState],
        set_task_resume_state: Callable[[TaskResumeState], None],
        current_chat_controller: Callable[[], Any | None],
    ) -> None:
        """Bind the live app and narrow screen/controller callbacks.

        Args:
            app_instance: Application object exposing ``skills_scope_service``.
            append_native_console_system_message: Append one Console system row.
            sync_console_command_popup: Refresh the open command popup, if any.
            task_resume_state: Return the current immutable resume state.
            set_task_resume_state: Replace the current resume state.
            current_chat_controller: Return the active chat controller, if any.
        """
        self.app_instance = app_instance
        self._append_native_console_system_message = (
            append_native_console_system_message
        )
        self._sync_console_command_popup = sync_console_command_popup
        self._task_resume_state = task_resume_state
        self._set_task_resume_state = set_task_resume_state
        self._current_chat_controller = current_chat_controller
        self._console_skill_candidates: tuple[SkillCommandCandidate, ...] = ()

    async def _fetch_console_skill_context(self) -> Mapping[str, Any]:
        """Fetch a fresh skill context, failing closed to an empty mapping."""
        service = getattr(self.app_instance, "skills_scope_service", None)
        get_context = getattr(service, "get_context", None)
        if not callable(get_context):
            return {}
        try:
            owner = _capture_console_skill_discovery(self)
            prepared = None
            if owner is not None:
                current, source, metadata = owner
                if not current():
                    raise RuntimeError("console_skill_trust_owner_changed")
                wiring = sys.modules.get("tldw_chatbook.app_service_wiring")
                if type(wiring) is ModuleType:
                    values = vars(wiring)
                    wiring_source = values.get("_CONSOLE_SKILL_WIRING_SOURCE")
                    helper_source = values.get("_CONSOLE_SKILL_HELPER_SOURCE")
                    entry = values.get("_CONSOLE_SKILL_ENTRY")
                    if (
                        type(entry) is tuple and len(entry) == 2
                        and entry[0] is values.get("_capture_console_skill_trust_preparation")
                        and entry[1] is values.get("_prepare_console_skill_trust_service")
                        and metadata._source_current(wiring_source)
                        and metadata._source_current(helper_source)
                    ):
                        prepared = entry[0](self.app_instance, service, current, source)
                        if prepared is not None:
                            if (not current() or not metadata._source_current(wiring_source)
                                or not metadata._source_current(helper_source)):
                                raise RuntimeError("console_skill_trust_source_changed")
                            await entry[1](prepared)
                            if not current():
                                raise RuntimeError("console_skill_trust_owner_changed")
                            if (
                                not metadata._source_current(wiring_source)
                                or not metadata._source_current(helper_source)
                                or not prepared[1]()
                            ):
                                raise RuntimeError("console_skill_trust_source_changed")
            context = await get_context(mode="local")
            if owner is not None:
                if not current():
                    raise RuntimeError("console_skill_trust_owner_changed")
                if prepared is not None and (
                    not metadata._source_current(wiring_source)
                    or not metadata._source_current(helper_source)
                    or not prepared[1]()
                ):
                    raise RuntimeError("console_skill_trust_source_changed")
        except Exception:
            logger.opt(exception=True).warning("Console skill context fetch failed.")
            return {}
        return context if isinstance(context, Mapping) else {}

    @staticmethod
    def _console_skill_trusted_candidates_from_context(
        context: Mapping[str, Any],
    ) -> tuple[SkillCommandCandidate, ...]:
        """Project trusted, user-invocable candidates in stable name order."""
        available = context.get("available_skills")
        candidates = [
            SkillCommandCandidate(
                name=str(item.get("name")),
                description=str(item.get("description") or ""),
            )
            for item in (available or [])
            if isinstance(item, Mapping)
            and item.get("name")
            and item.get("user_invocable", True)
            and not item.get("trust_blocked", False)
        ]
        candidates.sort(key=lambda candidate: candidate.name.casefold())
        return tuple(candidates)

    @staticmethod
    def _console_skill_blocked_summaries(
        context: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], ...]:
        """Return named trust-blocked skill summaries."""
        blocked = context.get("blocked_skills")
        return tuple(
            item
            for item in (blocked or [])
            if isinstance(item, Mapping) and item.get("name")
        )

    async def _refresh_console_skill_candidates(self) -> None:
        """Refresh the popup's cached trusted-candidate snapshot."""
        from tldw_chatbook.config import config_read_would_wait

        # TASK-35836: the first trust-service build reads config on the UI loop.
        while config_read_would_wait():
            await asyncio.sleep(0.2)
        owner = _capture_console_skill_discovery(self)
        context = await self._fetch_console_skill_context()
        if owner is not None and not owner[0]():
            return
        candidates = self._console_skill_trusted_candidates_from_context(context)
        if owner is not None and not owner[0]():
            return
        self._console_skill_candidates = candidates
        self._sync_console_command_popup()

    @staticmethod
    def _split_console_skill_name_args(text: str) -> tuple[str, str]:
        """Split stripped text into its leading word and remaining text."""
        for index, character in enumerate(text):
            if character.isspace():
                return text[:index], text[index + 1 :]
        return text, ""

    async def _console_command_skills(self, parse: CommandParse) -> None:
        """List trusted skills or show the static ``$name`` run hint."""
        args = parse.args.strip()
        if not args:
            context = await self._fetch_console_skill_context()
            candidates = self._console_skill_trusted_candidates_from_context(context)
            # Lazy: keeps command_handoff off the boot path (ADR-097). After
            # the fetch, answer in the chat /skills came from (TASK-33622.16).
            from .command_handoff import append_command_output

            await append_command_output(
                self._append_native_console_system_message,
                format_skills_list(candidates),
            )
            return
        name, _rest = self._split_console_skill_name_args(args)
        await self._append_native_console_system_message(
            f"Run skills by typing ${name} — /skills only lists them."
        )

    async def _console_skill_blocked_match_response(
        self, name: str, blocked_summaries: tuple[Mapping[str, Any], ...]
    ) -> bool:
        """Append a refusal or review hint when a blocked skill matches."""
        name_lower = name.lower()
        exact_blocked = next(
            (
                item
                for item in blocked_summaries
                if str(item.get("name") or "").lower() == name_lower
            ),
            None,
        )
        if exact_blocked is not None:
            reason = str(
                exact_blocked.get("trust_reason_code")
                or exact_blocked.get("trust_status")
                or "needs review"
            )
            await self._append_skill_refuse_row(
                str(exact_blocked.get("name") or name), reason
            )
            return True
        prefix_blocked = [
            item
            for item in blocked_summaries
            if str(item.get("name") or "").lower().startswith(name_lower)
        ]
        if prefix_blocked:
            await self._append_native_console_system_message(
                CONSOLE_SKILL_NEEDS_REVIEW_HINT_TEMPLATE.format(
                    count=len(prefix_blocked)
                )
            )
            return True
        return False

    async def _append_skill_refuse_row(self, name: str, reason: str) -> None:
        """Append the stable untrusted-skill refusal transcript row."""
        await self._append_native_console_system_message(
            SKILL_UNTRUSTED_REFUSE.format(name=name, reason=reason)
        )

    def _set_console_pending_skill_install(
        self, payload: dict[str, Any] | None
    ) -> bool:
        """Replace only the pending skill-install task state."""
        current = self._task_resume_state()
        return bool(
            self._set_task_resume_state(
                replace(current, pending_skill_install=payload)
            )
        )

    def _set_console_pending_skill_script(self, payload: dict[str, Any] | None) -> bool:
        """Replace only the pending skill-script task state."""
        current = self._task_resume_state()
        return bool(
            self._set_task_resume_state(
                replace(current, pending_skill_script=payload)
            )
        )

    def _set_console_pending_chat_create(
        self, payload: dict[str, Any] | None
    ) -> None:
        """Replace only the pending chat-create task state."""
        current = self._task_resume_state()
        self._set_task_resume_state(replace(current, pending_chat_create=payload))

    def handle_console_skill_install_decided(
        self, allow: bool, request_id: str | None
    ) -> None:
        """Forward an install decision to the current chat controller.

        Args:
            allow: Whether the user approved installation.
            request_id: Identifier of the pending confirmation round.
        """
        controller = self._current_chat_controller()
        if controller is not None:
            controller.resolve_pending_skill_install(allow, request_id=request_id)

    def handle_console_skill_script_decided(
        self, allow: bool, remember: bool, request_id: str | None
    ) -> None:
        """Forward a script decision to the current chat controller.

        Args:
            allow: Whether the user approved script execution.
            remember: Whether to persist the approval decision.
            request_id: Identifier of the pending confirmation round.
        """
        controller = self._current_chat_controller()
        if controller is not None:
            controller.resolve_pending_skill_script(
                allow, remember, request_id=request_id
            )

    def handle_console_chat_create_decided(
        self, allow: bool, remember: bool, request_id: str | None
    ) -> None:
        """Forward a chat-create decision to the current chat controller.

        Args:
            allow: Whether the user approved creating the chat.
            remember: Whether to also grant standing session permission.
            request_id: Identifier of the pending confirmation round.
        """
        controller = self._current_chat_controller()
        if controller is not None:
            controller.resolve_pending_chat_create(
                allow, remember, request_id=request_id
            )


def _sync_console_skill_popup(screen: Any) -> None:
    """Keep the original popup lookup late-bound at its existing call site."""
    screen._sync_console_command_popup()


_DISCOVERY_SOURCE = None
_DISCOVERY_VIEW = None


def bind_console_skill_discovery(view_type: type, constructor: Callable) -> None:
    """Bind the actual view and constructor after their definitions complete.

    Args:
        view_type: Original ChatScreen at its definition boundary.
        constructor: Original controller constructor from Console wiring.
    """
    global _DISCOVERY_SOURCE, _DISCOVERY_VIEW
    if _DISCOVERY_SOURCE is not None:
        return
    metadata = sys.modules["tldw_chatbook.Widgets.compact_model_bar"]
    view_namespace = vars(sys.modules[view_type.__module__])
    wiring_namespace = constructor.__globals__
    _DISCOVERY_VIEW = view_type
    functions = (
        (globals(), "_sync_console_skill_popup", _sync_console_skill_popup),
        (globals(), "_capture_console_skill_discovery", _capture_console_skill_discovery),
        (globals(), "bind_console_skill_discovery", bind_console_skill_discovery),
        *((vars(ConsoleSkillController), name, vars(ConsoleSkillController)[name])
          for name in ("__init__", "_fetch_console_skill_context", "_refresh_console_skill_candidates")),
        (vars(ConsoleSkillController), "_console_skill_trusted_candidates_from_context",
         vars(ConsoleSkillController)["_console_skill_trusted_candidates_from_context"].__func__),
        (vars(view_type), "_sync_console_command_popup", vars(view_type)["_sync_console_command_popup"]),
        (wiring_namespace, "build_console_controllers", constructor),
        *((vars(metadata), name, vars(metadata)[name])
          for name in ("_plain_fields", "_source_current", "_function_current")),
    )
    # The staticmethod slot retains its descriptor; its function is checked below.
    records = tuple(
        (namespace, name, function, function.__code__, function.__globals__,
         function.__defaults__, function.__kwdefaults__,
         tuple((function.__kwdefaults__ or {}).items()), function.__closure__,
         tuple((cell, cell.cell_contents) for cell in function.__closure__ or ()),
         vars(function).get("__wrapped__"))
        for namespace, name, function in functions
        if name != "_console_skill_trusted_candidates_from_context"
    )
    projection = ConsoleSkillController._console_skill_trusted_candidates_from_context
    _DISCOVERY_SOURCE = (
        globals(), __file__, __spec__, __spec__.origin,
        (
            (globals(), "ConsoleSkillController", ConsoleSkillController),
            (globals(), "_DISCOVERY_VIEW", view_type),
            (globals(), "partial", partial),
            (globals(), "asyncio", asyncio), (globals(), "threading", threading),
            (globals(), "sys", sys), (globals(), "FunctionType", FunctionType),
            (globals(), "ModuleType", ModuleType),
            (vars(asyncio), "get_running_loop", asyncio.get_running_loop),
            (vars(threading), "current_thread", threading.current_thread),
            (view_namespace, "ChatScreen", view_type),
            (wiring_namespace, "ConsoleSkillController", ConsoleSkillController),
            (wiring_namespace, "_sync_console_skill_popup", _sync_console_skill_popup),
            (wiring_namespace, "partial", partial),
            (vars(ConsoleSkillController), "_console_skill_trusted_candidates_from_context",
             vars(ConsoleSkillController)["_console_skill_trusted_candidates_from_context"]),
            *((namespace, key, namespace[key])
              for namespace in (view_namespace, wiring_namespace, vars(metadata))
              for key in ("__file__", "__spec__")),
            *((vars(namespace["__spec__"]), "origin", namespace["__spec__"].origin)
              for namespace in (view_namespace, wiring_namespace, vars(metadata))),
            *((sys.modules, namespace["__name__"], sys.modules[namespace["__name__"]])
              for namespace in (view_namespace, wiring_namespace, vars(metadata))),
        ),
        records + (({"projection": projection}, "projection", projection,
                    projection.__code__, projection.__globals__, projection.__defaults__,
                    projection.__kwdefaults__, tuple((projection.__kwdefaults__ or {}).items()),
                    projection.__closure__,
                    tuple((cell, cell.cell_contents) for cell in projection.__closure__ or ()),
                    vars(projection).get("__wrapped__")),),
    )


def _capture_console_skill_discovery(
    controller: Any,
) -> tuple[Callable[[], bool], tuple, ModuleType] | None:
    """Capture only the original view callback and its current attachment."""
    source = globals().get("_DISCOVERY_SOURCE")
    metadata = sys.modules.get("tldw_chatbook.Widgets.compact_model_bar")
    if (type(source) is not tuple or len(source) != 6
        or type(source[5]) is not tuple or type(metadata) is not ModuleType
        or any(type(row) is not tuple or len(row) != 11 or type(row[1]) is not str
               for row in source[5])):
        return None
    # Check the reused checker before invoking it, including in-place changes.
    proofs = tuple(row for row in source[5] if row[0] is vars(metadata))
    if len(proofs) != 3 or any(
        type(row[2]) is not FunctionType
        or vars(metadata).get(row[1]) is not row[2]
        or row[2].__code__ is not row[3]
        or row[2].__globals__ is not row[4]
        or row[2].__defaults__ is not row[5]
        or row[2].__kwdefaults__ is not row[6]
        or row[2].__closure__ is not row[8]
        for row in proofs
    ):
        return None
    source_current = metadata._source_current
    plain_fields = metadata._plain_fields
    if not source_current(source):
        return None
    fields = plain_fields(controller, ConsoleSkillController, ("app_instance", "_sync_console_command_popup"))
    if fields is None or any(name in fields for name in (
        "_fetch_console_skill_context", "_refresh_console_skill_candidates",
        "_console_skill_trusted_candidates_from_context",
    )):
        return None
    callback = fields.get("_sync_console_command_popup")
    if (type(callback) is not partial or callback.func is not _sync_console_skill_popup
        or len(callback.args) != 1 or callback.keywords):
        return None
    view = callback.args[0]
    view_names = ("app_instance", "_skill", "_console_runtime_attachment_generation", "_closing", "_closed")
    view_fields = plain_fields(view, _DISCOVERY_VIEW, view_names)
    if view_fields is None or "_sync_console_command_popup" in view_fields:
        return None
    app = fields.get("app_instance")
    app_module = sys.modules.get("tldw_chatbook.app")
    if type(app_module) is not ModuleType:
        return None
    app_record = vars(app_module).get("_CONSOLE_SKILL_APP_SOURCE")
    if type(app_record) is not tuple or len(app_record) != 5:
        return None
    app_type, _namespace, _factory, _code, runtime_type = app_record
    app_names = ("console_runtime", "_shutting_down", "_exit", "_console_runtime_shutdown_task")
    app_fields = plain_fields(app, app_type, app_names)
    if app_fields is None:
        return None
    runtime = app_fields.get("console_runtime")
    runtime_names = ("view", "_attached_generation", "_disposed", "_app")
    runtime_fields = plain_fields(runtime, runtime_type, runtime_names)
    if runtime_fields is None:
        return None
    generation = runtime_fields.get("_attached_generation")
    loop, thread = asyncio.get_running_loop(), threading.current_thread()

    def current():
        if any(
            vars(metadata).get(row[1]) is not row[2]
            or row[2].__code__ is not row[3]
            or row[2].__globals__ is not row[4]
            or row[2].__defaults__ is not row[5]
            or row[2].__kwdefaults__ is not row[6]
            or row[2].__closure__ is not row[8]
            for row in proofs
        ):
            return False
        return (
            globals().get("_DISCOVERY_SOURCE") is source and source_current(source)
            and asyncio.get_running_loop() is loop and threading.current_thread() is thread
            and plain_fields(controller, ConsoleSkillController, ("app_instance", "_sync_console_command_popup")) is fields
            and not any(name in fields for name in (
                "_fetch_console_skill_context", "_refresh_console_skill_candidates",
                "_console_skill_trusted_candidates_from_context",
            ))
            and fields.get("app_instance") is app
            and fields.get("_sync_console_command_popup") is callback
            and callback.func is _sync_console_skill_popup
            and len(callback.args) == 1 and callback.args[0] is view and not callback.keywords
            and plain_fields(view, _DISCOVERY_VIEW, view_names) is view_fields
            and "_sync_console_command_popup" not in view_fields
            and view_fields.get("app_instance") is app and view_fields.get("_skill") is controller
            and type(generation) is int
            and type(view_generation := view_fields.get("_console_runtime_attachment_generation")) is int
            and view_generation == generation
            and view_fields.get("_closing") is False and view_fields.get("_closed") is False
            and sys.modules.get("tldw_chatbook.app") is app_module
            and vars(app_module).get("_CONSOLE_SKILL_APP_SOURCE") is app_record
            and vars(app_module).get("TldwCli") is app_type
            and vars(app_module).get("ConsoleRuntime") is runtime_type
            and plain_fields(app, app_type, app_names) is app_fields
            and app_fields.get("console_runtime") is runtime
            and app_fields.get("_shutting_down") is False and app_fields.get("_exit") is False
            and app_fields.get("_console_runtime_shutdown_task") is None
            and plain_fields(runtime, runtime_type, runtime_names) is runtime_fields
            and runtime_fields.get("_app") is app and runtime_fields.get("_disposed") is False
            and runtime_fields.get("view") is view
            and type(runtime_generation := runtime_fields.get("_attached_generation")) is int
            and runtime_generation == generation
        )

    return current, source, metadata
