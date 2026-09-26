"""Built-in skills in Library ▸ Skills: read-only preview, Customize, Enabled.

TASK-32954 (spec §3.5). Split out of ``LibrarySkillsController`` so that
controller does not grow (ruling R14). This controller keeps no state of its
own: the preview lives in ``LibrarySkillsState.builtin_preview`` and every
shared binding (worker runner, service-call boundary, detail generation,
editor reset, canvas sync) is read through the owning skills controller.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping
from typing import Any

from loguru import logger
from textual.widgets import Button, Switch

from ...config import save_setting_to_cli_config
from ...Library.library_shell_state import LIBRARY_ROW_BROWSE_SKILLS
from ...Skills_Interop.builtin_skills import (
    BUILTIN_SKILL_DIGESTS,
    disabled_builtins_from_config,
)
from ...Skills_Interop.local_skills_service import LocalSkillsService
from .canvas_sync import _sync_library_canvas

#: Serializes ``[skills] disabled_builtins`` writes (TASK-32954).
_DISABLED_BUILTINS_SAVE_LOCK = threading.Lock()


class LibrarySkillsBuiltinController:
    """Own the built-in skill preview, Customize, and Enabled switch.

    Args:
        skills: The owning ``LibrarySkillsController``; its state properties
            and framework bindings are read and written through it.
    """

    def __init__(self, skills: Any) -> None:
        self._skills = skills

    def preview_work_pane_values(self, values: dict[str, Any]) -> dict[str, Any]:
        """Work-pane values while a built-in is open: read-only, never the editor."""
        s = self._skills
        preview = s._library_skill_builtin_preview
        if preview is None:
            values["mode"] = "loading"
            values["detail_notice"] = s._library_skill_detail_error or "Loading skill…"
        else:
            values["mode"] = "preview"
            values["builtin_preview"] = preview
        return values

    def _open_library_skill_builtin_preview(self, skill_name: str) -> None:
        """Open a built-in skill read-only in the Work pane.

        The editor (dirty tracking, Save/Discard vetoes, trust review) has no
        read-only mode, so a built-in row never reaches it.
        """
        s = self._skills
        s._reset_library_skill_editor_state()
        s._selected_skill_name = skill_name
        s._library_selected_row_id = LIBRARY_ROW_BROWSE_SKILLS
        s._library_skills_view = "preview"
        s.run_worker(
            self._refresh_library_skill_builtin_preview(
                skill_name, s._claim_library_skill_detail_generation()
            ),
            exclusive=True,
            group="library_skill_detail",
        )
        _sync_library_canvas(s, "skills")

    def _library_skill_preview_is_current(self, skill_name: str, generation: int) -> bool:
        s = self._skills
        return bool(
            generation == s._library_skill_detail_generation
            and skill_name == s._selected_skill_name
            and s._library_skills_view == "preview"
        )

    async def _refresh_library_skill_builtin_preview(
        self, skill_name: str, generation: int
    ) -> None:
        """Fetch the built-in's content and render the read-only preview."""
        s = self._skills
        service = getattr(s.app_instance, "skills_scope_service", None)
        detail: Any = None
        try:
            detail = await s._run_library_service_call(
                service.get_skill,
                skill_name,
                mode="local",
                include_disabled_builtins=True,
                isolate_in_worker=True,
            )
        except Exception:  # noqa: BLE001 -- UI boundary: failure becomes a notice
            logger.opt(exception=True).warning("Failed to load a built-in skill preview.")
        if not self._library_skill_preview_is_current(skill_name, generation):
            return
        s._library_skill_detail_loading = False
        if not isinstance(detail, Mapping):
            s._library_skill_detail_error = "Couldn’t load this built-in skill."
        else:
            _front, body = LocalSkillsService._parse_front_matter(
                str(detail.get("content") or "")
            )
            s._library_skill_builtin_preview = {
                "name": skill_name,
                "content": body,
                "enabled": skill_name
                not in disabled_builtins_from_config(s.app_instance.app_config),
            }
        if s.is_mounted:
            _sync_library_canvas(s, "skills")

    async def _library_skill_is_builtin_only(self, skill_name: str) -> bool:
        """Whether ``skill_name`` is a built-in with no user copy."""
        if skill_name not in BUILTIN_SKILL_DIGESTS:
            return False
        s = self._skills
        service = getattr(s.app_instance, "skills_scope_service", None)
        try:
            detail = await s._run_library_service_call(
                service.get_skill,
                skill_name,
                mode="local",
                include_disabled_builtins=True,
                isolate_in_worker=True,
            )
        except Exception:  # noqa: BLE001 -- unknown: the preview reports the error
            return True
        return isinstance(detail, Mapping) and detail.get("source") == "builtin"

    async def open_preview_if_builtin_only(self, skill_name: str) -> bool:
        """Review link: a built-in has no editor, so open its preview instead.

        Returns:
            ``True`` when the preview was opened (the caller must stop).
        """
        if not await self._library_skill_is_builtin_only(skill_name):
            return False
        self._open_library_skill_builtin_preview(skill_name)
        return True

    def handle_library_skill_builtin_customize(self, event: Button.Pressed) -> None:
        """Customize: copy the open built-in into the user's skills."""
        event.stop()
        s = self._skills
        preview = s._library_skill_builtin_preview
        if preview is None:
            return
        s.run_worker(
            self._customize_library_skill_builtin(str(preview["name"])),
            exclusive=True,
            group="library_skill_builtin_customize",
            exit_on_error=False,
        )

    async def _customize_library_skill_builtin(self, skill_name: str) -> None:
        """Seed only this built-in, return to the list, and say what happened."""
        s = self._skills
        service = getattr(s.app_instance, "skills_scope_service", None)
        try:
            result = await s._run_library_service_call(
                service.seed_builtin_skills,
                names=[skill_name],
                mode="local",
                isolate_in_worker=True,
            )
            copy = await s._run_library_service_call(
                service.get_skill, skill_name, mode="local", isolate_in_worker=True
            )
        except Exception:  # noqa: BLE001 -- UI boundary: failure becomes a notice
            logger.opt(exception=True).warning("Customize of a built-in skill failed.")
            s.app.notify("Couldn’t copy this built-in skill.", severity="error")
            return
        seeded = result.get("seeded") if isinstance(result, Mapping) else None
        if not seeded or skill_name not in seeded:
            # e.g. a folder of that name exists in the store but is not indexed.
            s.app.notify(
                "Nothing was copied — a skill with this name may already exist "
                "in your skills folder.",
                severity="warning",
            )
            return
        notice = "Copied to your skills — edit your copy."
        if isinstance(copy, Mapping) and copy.get("trust_blocked"):
            notice += " It needs review before the assistant can use it."
        s._reset_library_skill_editor_state()
        s._refresh_library_skills_after_committed_mutation()
        s.app.notify(notice)

    def handle_library_skill_builtin_enabled(self, event: Switch.Changed) -> None:
        """Enabled switch: hide or show the open built-in for every reader."""
        event.stop()
        s = self._skills
        preview = s._library_skill_builtin_preview
        if preview is None or bool(preview.get("enabled")) == event.value:
            return
        s.run_worker(
            self._set_library_skill_builtin_enabled(str(preview["name"]), event.value),
            exclusive=True,
            group="library_skill_builtin_enabled",
            exit_on_error=False,
        )

    async def _set_library_skill_builtin_enabled(
        self, skill_name: str, enabled: bool
    ) -> None:
        """Apply ``[skills] disabled_builtins`` in memory now, then persist it.

        The skills service's built-in loader reads the in-memory app config,
        so the change applies without a restart.
        """
        s = self._skills
        config = s.app_instance.app_config
        skills_config = config.setdefault("skills", {})
        current = skills_config.get("disabled_builtins")
        disabled = [
            name for name in (current if isinstance(current, list) else [])
            if isinstance(name, str) and name != skill_name
        ]
        if not enabled:
            disabled.append(skill_name)
        skills_config["disabled_builtins"] = disabled
        preview = s._library_skill_builtin_preview
        if preview is not None and preview.get("name") == skill_name:
            s._library_skill_builtin_preview = {**preview, "enabled": enabled}
        s._refresh_library_skills_after_committed_mutation()
        if s.is_mounted:
            _sync_library_canvas(s, "skills")

        def persist_current() -> bool:
            # Serialized, and reads memory at write time: a cancelled earlier
            # toggle's thread can never land a stale list after a newer one.
            with _DISABLED_BUILTINS_SAVE_LOCK:
                current = config.get("skills", {}).get("disabled_builtins", [])
                return save_setting_to_cli_config(
                    "skills", "disabled_builtins", list(current)
                )

        try:
            saved = await asyncio.to_thread(persist_current)
        except Exception:  # noqa: BLE001 -- UI boundary: failure becomes a notice
            saved = False
        if saved is not True:
            s.app.notify(
                "Changed for this session, but could not be saved.",
                severity="warning",
            )
