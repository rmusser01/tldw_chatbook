"""TASK-32954 Task 7: Library ▸ Skills built-in badge and read-only preview.

State/controller level: the handlers are exercised unbound against small
stand-ins, and the preview pane is mounted in a bare Textual app -- the
editor (dirty tracking, Save/Discard vetoes, trust review) must never mount
for a built-in row.

``tldw_chatbook.app`` is imported at module scope so the heavy import runs at
collection time, not inside the per-test sandbox that trips this checkout's
ADR-126 ``RecoveryRequired`` fixture gate (see
``Tests/UI/test_personas_character_changed.py``).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Markdown, Switch

import tldw_chatbook.app  # noqa: F401  -- collection-time import, see docstring
from tldw_chatbook.Library.library_skills_state import build_skills_list_state
from tldw_chatbook.UI.Library_Modules import library_skills_controller as lsc
from tldw_chatbook.UI.Library_Modules.library_skills_controller import (
    LibrarySkillsController,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_skill_work_pane import (
    LibrarySkillWorkPane,
)

NAME = "character-creator"


def _rows(summaries):
    state = build_skills_list_state(
        {"available_skills": summaries}, query="", sort="name"
    )
    return {row.name: row for row in state.rows}


def test_builtin_row_badged():
    rows = _rows([{"name": NAME, "source": "builtin", "trust_blocked": False}])
    assert rows[NAME].is_builtin and not rows[NAME].overridden


def test_user_copy_marks_override():
    rows = _rows([{"name": NAME, "source": "local", "overrides_builtin": True}])
    assert rows[NAME].overridden and not rows[NAME].is_builtin


def test_plain_user_row_has_no_badge():
    rows = _rows([{"name": "mine"}])
    assert not rows["mine"].is_builtin and not rows["mine"].overridden


def test_builtin_activation_opens_preview_not_editor():
    calls: list[str] = []

    async def flush() -> bool:
        return True

    fake = SimpleNamespace(
        _flush_library_skill_save=flush,
        _skills_controller=SimpleNamespace(
            _open_library_skill_builtin_preview=lambda n: calls.append(f"preview:{n}")
        ),
        _reset_library_skill_editor_state=lambda: calls.append("editor-reset"),
        run_worker=lambda *a, **k: calls.append("editor-fetch"),
    )
    button = SimpleNamespace(skill_name=NAME, skill_builtin=True)
    event = SimpleNamespace(button=button, stop=lambda: None)
    asyncio.run(LibraryScreen.handle_library_skill_row(fake, event))
    assert calls == [f"preview:{NAME}"]


def _controller_stand_in(**extra: Any) -> SimpleNamespace:
    calls: list[Any] = []
    fake = SimpleNamespace(
        calls=calls,
        _selected_skill_name="",
        _library_selected_row_id="",
        _library_skills_view="list",
        _library_skill_reader_mode="edit",
        _library_skill_builtin_preview=None,
        _library_skill_detail_generation=0,
        _reset_library_skill_editor_state=lambda: calls.append("reset"),
        run_worker=lambda coro, **k: (coro.close(), calls.append(("worker", k))),
        _refresh_library_skills_after_committed_mutation=lambda: calls.append(
            "refresh"
        ),
        app=SimpleNamespace(
            notify=lambda message, **k: calls.append(("notify", message))
        ),
        is_mounted=False,
        **extra,
    )
    fake._refresh_library_skill_builtin_preview = (
        lambda *a, **k: LibrarySkillsController._refresh_library_skill_builtin_preview(
            fake, *a, **k
        )
    )
    fake._claim_library_skill_detail_generation = (
        lambda: LibrarySkillsController._claim_library_skill_detail_generation(fake)
    )
    return fake


def test_open_preview_sets_preview_view_and_never_the_editor(monkeypatch):
    monkeypatch.setattr(lsc, "_sync_library_canvas", lambda *a, **k: True)
    fake = _controller_stand_in()
    fake._library_skill_detail_loading = False
    fake._library_skill_detail_error = ""
    fake._library_skill_detail_retryable = False
    LibrarySkillsController._open_library_skill_builtin_preview(fake, NAME)
    assert fake._library_skills_view == "preview"
    assert fake._selected_skill_name == NAME
    assert fake.calls[0] == "reset"
    assert ("worker", {"exclusive": True, "group": "library_skill_detail"}) in fake.calls


def _service(**methods: Any) -> SimpleNamespace:
    return SimpleNamespace(**methods)


async def _direct_call(fn, *args, isolate_in_worker=False, **kwargs):
    return await fn(*args, **kwargs)


@pytest.mark.parametrize(
    ("blocked", "needs_review"), [(False, False), (True, True)]
)
def test_customize_seeds_only_that_builtin_and_refreshes(blocked, needs_review):
    seeded: list[Any] = []

    async def seed_builtin_skills(**kwargs):
        seeded.append(kwargs)
        return {"seeded": [NAME], "count": 1}

    async def get_skill(name, **kwargs):
        return {"name": name, "trust_blocked": blocked}

    fake = _controller_stand_in(
        app_instance=SimpleNamespace(
            skills_scope_service=_service(
                seed_builtin_skills=seed_builtin_skills, get_skill=get_skill
            )
        ),
        _run_library_service_call=_direct_call,
    )
    asyncio.run(LibrarySkillsController._customize_library_skill_builtin(fake, NAME))
    assert seeded == [{"names": [NAME], "mode": "local"}]
    assert "reset" in fake.calls and "refresh" in fake.calls
    notice = next(m for m in fake.calls if isinstance(m, tuple) and m[0] == "notify")[1]
    assert notice.startswith("Copied to your skills — edit your copy.")
    assert ("review" in notice) is needs_review


def test_enabled_switch_updates_memory_and_persists(monkeypatch):
    saved: list[Any] = []
    monkeypatch.setattr(
        lsc, "save_setting_to_cli_config", lambda *a: saved.append(a) or True
    )
    monkeypatch.setattr(lsc, "_sync_library_canvas", lambda *a, **k: True)
    config: dict[str, Any] = {"skills": {"disabled_builtins": ["other"]}}
    fake = _controller_stand_in(app_instance=SimpleNamespace(app_config=config))
    fake._library_skill_builtin_preview = {"name": NAME, "enabled": True}

    asyncio.run(
        LibrarySkillsController._set_library_skill_builtin_enabled(fake, NAME, False)
    )
    assert config["skills"]["disabled_builtins"] == ["other", NAME]
    assert saved == [("skills", "disabled_builtins", ["other", NAME])]
    assert fake._library_skill_builtin_preview["enabled"] is False
    assert "refresh" in fake.calls

    asyncio.run(
        LibrarySkillsController._set_library_skill_builtin_enabled(fake, NAME, True)
    )
    assert config["skills"]["disabled_builtins"] == ["other"]
    assert saved[-1] == ("skills", "disabled_builtins", ["other"])


class _PaneApp(App):
    def compose(self) -> ComposeResult:
        yield LibrarySkillWorkPane(
            mode="preview",
            builtin_preview={
                "name": NAME,
                "content": "---\nname: character-creator\n---\n# Character Creator\n",
                "enabled": True,
            },
            id="library-skill-work-pane",
        )


def test_preview_pane_is_read_only_markdown_with_customize_and_switch():
    async def run() -> None:
        app = _PaneApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            preview = app.query_one("#library-skill-builtin-preview")
            assert preview.query(Markdown)
            assert isinstance(
                app.query_one("#library-skill-builtin-customize"), Button
            )
            switch = app.query_one("#library-skill-builtin-enabled", Switch)
            assert switch.value is True
            # Never the editor: no body TextArea, no Save.
            assert not app.query("#library-skill-body")
            assert not app.query("#library-skill-save")

    asyncio.run(run())
