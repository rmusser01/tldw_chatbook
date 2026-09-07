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
            context = await get_context(mode="local")
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
            await self._append_native_console_system_message(
                format_skills_list(candidates)
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
    ) -> None:
        """Replace only the pending skill-install task state."""
        current = self._task_resume_state()
        self._set_task_resume_state(replace(current, pending_skill_install=payload))

    def _set_console_pending_skill_script(self, payload: dict[str, Any] | None) -> None:
        """Replace only the pending skill-script task state."""
        current = self._task_resume_state()
        self._set_task_resume_state(replace(current, pending_skill_script=payload))

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
