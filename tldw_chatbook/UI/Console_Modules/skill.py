"""DOM-free Console skill discovery, trust, and decision policy."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
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
            preparation = _stock_console_skill_preparation(
                self, self.app_instance, service
            )
            if preparation is None:
                context = await get_context(mode="local")
            else:
                context = await get_context(
                    mode="local", _console_trust_preparation=preparation
                )
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
        context = await self._fetch_console_skill_context()
        self._console_skill_candidates = (
            self._console_skill_trusted_candidates_from_context(context)
        )
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
            self._set_task_resume_state(replace(current, pending_skill_install=payload))
        )

    def _set_console_pending_skill_script(self, payload: dict[str, Any] | None) -> bool:
        """Replace only the pending skill-script task state."""
        current = self._task_resume_state()
        return bool(
            self._set_task_resume_state(replace(current, pending_skill_script=payload))
        )

    def _set_console_pending_chat_create(self, payload: dict[str, Any] | None) -> None:
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


def _stock_console_skill_preparation(controller, app, service):
    """Select existing stock preparation without importing its heavy services."""
    import inspect
    import sys
    from types import (
        FunctionType,
        GetSetDescriptorType,
        MemberDescriptorType,
        ModuleType,
    )

    cls = _CONSOLE_SKILL_CONTROLLER_ORIGINAL
    missing = _CONSOLE_SKILL_MISSING
    if (
        type(controller) is not cls
        or inspect.getattr_static(cls, "__getattribute__")
        is not object.__getattribute__
        or inspect.getattr_static(cls, "__getattr__", missing) is not missing
        or inspect.getattr_static(cls, "app_instance", missing) is not missing
    ):
        return None
    dictionary = inspect.getattr_static(cls, "__dict__", missing)
    if (
        type(dictionary) is not GetSetDescriptorType
        and type(dictionary) is not MemberDescriptorType
    ):
        return None
    fields = vars(controller)
    if type(fields) is not dict or fields.get("app_instance") is not app:  # noqa: E721 -- exact stock metadata; no custom dispatch
        return None
    module = sys.modules.get("tldw_chatbook.app_service_wiring")
    if type(module) is not ModuleType:
        return None
    namespace = vars(module)
    ensure = inspect.getattr_static(type(app), "ensure_local_skill_trust_service", None)
    if type(ensure) is not FunctionType or ensure.__globals__ is not namespace:
        return None
    entry = namespace.get("_CONSOLE_SKILL_ENTRY")
    if type(entry) is not tuple or len(entry) != 2:
        return None
    capture, prepare = entry
    if type(capture) is not FunctionType or type(prepare) is not FunctionType:
        return None
    source = namespace.get("_CONSOLE_SKILL_WIRING_SOURCE")
    # These two original callback bodies must be present in the defining table
    # before any optional callback is dispatched. No custom iterable/key used.
    if type(source) is not tuple or len(source) != 6 or type(source[5]) is not tuple:
        return None
    controller_module = sys.modules.get(__name__)
    if (
        type(controller_module) is not ModuleType
        or vars(controller_module) is not globals()
    ):
        return None
    controller_source = globals().get("_CONSOLE_SKILL_CONTEXT_SOURCE")
    if type(controller_source) is not tuple or len(controller_source) != 6:
        return None
    fetch = inspect.getattr_static(cls, "_fetch_console_skill_context", None)
    controller_records = controller_source[5]
    if type(controller_records) is not tuple:
        return None
    functions = (
        capture,
        prepare,
        namespace.get("_console_skill_metadata_current"),
        namespace.get("_console_skill_source_current"),
    )
    for function in functions:
        if type(function) is not FunctionType:
            return None
        match = None
        for row in source[5]:
            if type(row) is not tuple or len(row) != 11:
                return None
            if row[2] is function:
                match = row
                break
        if (
            match is None
            or function.__globals__ is not namespace
            or function.__code__ is not match[3]
            or function.__defaults__ is not match[5]
            or function.__kwdefaults__ is not match[6]
            or function.__closure__ is not match[8]
        ):
            return None
        keyword_defaults = function.__kwdefaults__
        if keyword_defaults is not None and type(keyword_defaults) is not dict:  # noqa: E721 -- exact stock metadata; no custom dispatch
            return None
        if keyword_defaults is not None and any(
            type(key) is not str  # noqa: E721 -- exact stock metadata; no custom dispatch
            for key in keyword_defaults  # noqa: E721 -- exact stock metadata; no custom dispatch
        ):
            return None
        if type(match[7]) is not tuple or type(match[9]) is not tuple:
            return None
        if len(keyword_defaults or {}) != len(match[7]):
            return None
        for item in match[7]:
            if type(item) is not tuple or len(item) != 2 or type(item[0]) is not str:  # noqa: E721 -- exact stock metadata; no custom dispatch
                return None
            if (keyword_defaults or {}).get(item[0]) is not item[1]:
                return None
        if len(function.__closure__ or ()) != len(match[9]):
            return None
        for cell, item in zip(function.__closure__ or (), match[9]):
            if (
                type(item) is not tuple
                or len(item) != 2
                or item[0] is not cell
                or cell.cell_contents is not item[1]
            ):
                return None

    # Qualify this caller and fetch before dispatching any optional helper.
    for name, function in (
        ("_stock_console_skill_preparation", _stock_console_skill_preparation),
        ("_fetch_console_skill_context", fetch),
    ):
        if type(function) is not FunctionType:
            return None
        matches = [
            row
            for row in controller_records
            if type(row) is tuple and len(row) == 11 and row[2] is function
        ]
        if len(matches) != 1:
            return None
        row = matches[0]
        if (
            function.__globals__ is not globals()
            or function.__code__ is not row[3]
            or function.__defaults__ is not row[5]
            or function.__kwdefaults__ is not row[6]
            or function.__closure__ is not row[8]
        ):
            return None

    def owner_current():
        return (
            type(controller) is cls
            and inspect.getattr_static(cls, "__getattribute__")
            is object.__getattribute__
            and inspect.getattr_static(cls, "__getattr__", missing) is missing
            and inspect.getattr_static(cls, "app_instance", missing) is missing
            and vars(controller) is fields
            and fields.get("app_instance") is app
        )

    try:
        captured = capture(app, service, owner_current, controller_source)
    except (AttributeError, TypeError, ValueError):
        return None
    return None if captured is None else (prepare, captured)


_CONSOLE_SKILL_CONTROLLER_ORIGINAL = ConsoleSkillController
_CONSOLE_SKILL_MISSING = object()


# Definition-time originals for stock Console trust preparation only.
from types import FunctionType as _SkillFunctionType  # noqa: E402


_CONSOLE_SKILL_FUNCTIONS = {
    "__init__": ConsoleSkillController.__dict__["__init__"],
    "_fetch_console_skill_context": ConsoleSkillController.__dict__[
        "_fetch_console_skill_context"
    ],
    "_stock_console_skill_preparation": _stock_console_skill_preparation,
}
_CONSOLE_SKILL_CONTEXT_SOURCE = (
    globals(),
    __file__,
    __spec__,
    getattr(__spec__, "origin", None),
    (
        (globals(), "ConsoleSkillController", ConsoleSkillController),
        (globals(), "_CONSOLE_SKILL_FUNCTIONS", _CONSOLE_SKILL_FUNCTIONS),
        (
            ConsoleSkillController.__dict__,
            "__init__",
            ConsoleSkillController.__dict__["__init__"],
        ),
        (
            ConsoleSkillController.__dict__,
            "_fetch_console_skill_context",
            ConsoleSkillController.__dict__["_fetch_console_skill_context"],
        ),
        (
            globals(),
            "_stock_console_skill_preparation",
            _stock_console_skill_preparation,
        ),
        (
            globals(),
            "_CONSOLE_SKILL_CONTROLLER_ORIGINAL",
            _CONSOLE_SKILL_CONTROLLER_ORIGINAL,
        ),
        (globals(), "_CONSOLE_SKILL_MISSING", _CONSOLE_SKILL_MISSING),
    ),
    tuple(
        (
            _CONSOLE_SKILL_FUNCTIONS,
            _skill_name,
            _skill_function,
            _skill_function.__code__,
            _skill_function.__globals__,
            _skill_function.__defaults__,
            _skill_function.__kwdefaults__,
            tuple((_skill_function.__kwdefaults__ or {}).items()),
            _skill_function.__closure__,
            tuple(
                (cell, cell.cell_contents) for cell in _skill_function.__closure__ or ()
            ),
            vars(_skill_function).get("__wrapped__"),
        )
        for _skill_name, _skill_function in _CONSOLE_SKILL_FUNCTIONS.items()
        if type(_skill_function) is _SkillFunctionType
    ),
)
