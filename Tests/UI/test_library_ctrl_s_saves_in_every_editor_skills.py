"""Ctrl+S in the skill editor follows the one Library rule (TASK-34000.40 N-33).

The skill editor had owned Ctrl+S since task-424: it saved the open skill
from ANYWHERE on the screen while a save was available, and did nothing,
silently, on a clean skill -- though the editor's own hint line said
"ctrl+s Save". Under the owner's rule (ADR-031 refinement 2026-10-03; settled
2026-10-27) the key belongs to the editor that holds focus: it saves a dirty
skill, says "Saved — no changes." on a clean one, and does nothing from
the Skills list beside it.

The production ``TldwCli`` booted into Library over a real
``LocalSkillsService`` store in ``tmp_path``; the data claim reads the
skill's ``SKILL.md`` on disk. Real keys and clicks, painted text.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.app_module_patches import patch_app_global
from Tests.UI.library_quit_guard_support import (
    SIZE,
    _library,
    _library_app,
    _record_toasts,
    _settings_without_splash,
    _type,
    _until,
)
from Tests.UI.test_library_shell import _wait_for_condition
from Tests.UI.test_library_ctrl_s_saves_in_every_editor import (
    NO_CHANGES,
    _footer,
    _painted,
    _within,
)
from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

_SKILL = "summarize-notes"
_CONTENT = "---\ndescription: Summarize notes\n---\n# Summarize\nSummarize body text.\n"


async def test_ctrl_s_saves_the_focused_skill_and_nothing_from_its_list(
    tmp_path, monkeypatch
):
    store = tmp_path / "skills"
    store.mkdir()
    local = LocalSkillsService(
        store_dir=store,
        trust_service=None,
        allow_untrusted_without_trust_service=True,
        policy_enforcer=None,
    )
    await local.create_skill(name=_SKILL, content=_CONTENT)
    skill_file = next(store.rglob("SKILL.md"))
    app, _profile = _library_app(tmp_path, monkeypatch, [], lambda _p: None, notes=False)
    app.skills_scope_service = SkillsScopeService(
        local_service=local, server_service=None, policy_enforcer=None
    )
    toasts = _record_toasts(monkeypatch, app)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            screen.query_one("#library-row-browse-skills", Button).press()
            await _until(
                pilot,
                lambda: bool(screen.query(f"#library-skill-row-{_SKILL}")),
                "the skill row",
            )
            screen.query_one(f"#library-skill-row-{_SKILL}", Button).press()
            await _until(
                pilot,
                lambda: screen._skills_state.detail is not None
                and screen._skills_state.view == "editor",
                "the skill editor",
            )
            await _until(
                pilot, lambda: bool(screen.query("#library-skill-mode-edit")), "Edit mode"
            )
            screen.query_one("#library-skill-mode-edit", Button).press()
            await _until(
                pilot,
                lambda: bool(screen.query("#library-skill-description"))
                and screen.query_one("#library-skill-description").region.height > 0,
                "the Description field",
            )

            def _in_description() -> bool:
                return getattr(screen.focused, "id", None) == "library-skill-description"

            await pilot.pause(0.2)
            await pilot.click("#library-skill-description")
            await _until(pilot, _in_description, "focus in Description")

            def _status() -> str:
                return _painted(app, screen.query_one("#library-skill-save-status"))

            # Clean: Save is hidden, so no chip (ruling T9-j), and the key
            # answers on the status line (was: silence).
            await _until(
                pilot,
                lambda: "back to skills list" in _footer(app),
                "the skill editor's footer",
            )
            assert "ctrl+s" not in _footer(app), f"clean skill chipped: {_footer(app)}"
            await pilot.press("ctrl+s")
            await _until(
                pilot,
                lambda: NO_CHANGES in _status(),
                "a clean skill's Ctrl+S to say there were no changes",
            )

            await pilot.press("end")
            await _type(pilot, " fast")
            await _until(pilot, lambda: screen._skills_state.dirty, "a dirty skill")
            await _until(
                pilot,
                lambda: "ctrl+s save skill" in _footer(app),
                "a dirty skill (Save shown) to chip Ctrl+S",
            )
            # Off the editor -- the Skills list beside it -- the key does nothing.
            await pilot.press("shift+f6")
            await _until(
                pilot,
                lambda: screen.focused is not None
                and not _within(screen.focused, "library-skill-work-pane"),
                "Shift+F6 to leave the skill editor",
            )
            moved_to = screen.focused
            await pilot.press("ctrl+s")
            await pilot.pause(0.5)
            assert "fast" not in skill_file.read_text(encoding="utf-8"), (
                f"Ctrl+S on {moved_to!r} saved the skill open beside it"
            )
            assert screen._skills_state.dirty
            assert screen.focused is moved_to
            assert "ctrl+s" not in _footer(app), _footer(app)

            await pilot.click("#library-skill-description")
            await _until(pilot, _in_description, "focus back")
            await pilot.press("ctrl+s")
            await _until(
                pilot,
                lambda: "Summarize notes fast" in skill_file.read_text(encoding="utf-8"),
                "Ctrl+S to write the skill",
                timeout=5.0,
            )
            await _wait_for_condition(
                pilot,
                lambda: "Saved." in _status(),
                message=lambda: f"no painted Saved receipt: {_status()!r}",
            )
            assert _in_description(), screen.focused
            assert not [toast for toast in toasts if "Save" in toast], toasts
