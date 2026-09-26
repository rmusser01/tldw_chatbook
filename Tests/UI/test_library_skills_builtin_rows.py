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
from tldw_chatbook.UI.Library_Modules import library_skills_builtin_controller as lsbc
from tldw_chatbook.UI.Library_Modules import library_skills_controller as lsc
from tldw_chatbook.UI.Library_Modules.library_skills_builtin_controller import (
    LibrarySkillsBuiltinController,
)
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
            builtin=SimpleNamespace(
                _open_library_skill_builtin_preview=lambda n: calls.append(
                    f"preview:{n}"
                )
            )
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
    # R14: the built-in cluster lives on its own controller, bound to the
    # skills controller (here, this stand-in).
    fake.builtin = LibrarySkillsBuiltinController(fake)
    fake._claim_library_skill_detail_generation = (
        lambda: LibrarySkillsController._claim_library_skill_detail_generation(fake)
    )
    return fake


def test_open_preview_sets_preview_view_and_never_the_editor(monkeypatch):
    monkeypatch.setattr(lsc, "_sync_library_canvas", lambda *a, **k: True)
    monkeypatch.setattr(lsbc, "_sync_library_canvas", lambda *a, **k: True)
    fake = _controller_stand_in()
    fake._library_skill_detail_loading = False
    fake._library_skill_detail_error = ""
    fake._library_skill_detail_retryable = False
    fake.builtin._open_library_skill_builtin_preview(NAME)
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
    asyncio.run(fake.builtin._customize_library_skill_builtin(NAME))
    assert seeded == [{"names": [NAME], "mode": "local"}]
    assert "reset" in fake.calls and "refresh" in fake.calls
    notice = next(m for m in fake.calls if isinstance(m, tuple) and m[0] == "notify")[1]
    assert notice.startswith("Copied to your skills — edit your copy.")
    assert ("review" in notice) is needs_review


def test_enabled_switch_updates_memory_and_persists(monkeypatch):
    saved: list[Any] = []
    monkeypatch.setattr(
        lsbc, "save_setting_to_cli_config", lambda *a: saved.append(a) or True
    )
    monkeypatch.setattr(lsc, "_sync_library_canvas", lambda *a, **k: True)
    monkeypatch.setattr(lsbc, "_sync_library_canvas", lambda *a, **k: True)
    config: dict[str, Any] = {"skills": {"disabled_builtins": ["other"]}}
    fake = _controller_stand_in(app_instance=SimpleNamespace(app_config=config))
    fake._library_skill_builtin_preview = {"name": NAME, "enabled": True}

    asyncio.run(
        fake.builtin._set_library_skill_builtin_enabled(NAME, False)
    )
    assert config["skills"]["disabled_builtins"] == ["other", NAME]
    assert saved == [("skills", "disabled_builtins", ["other", NAME])]
    assert fake._library_skill_builtin_preview["enabled"] is False
    assert "refresh" in fake.calls

    asyncio.run(
        fake.builtin._set_library_skill_builtin_enabled(NAME, True)
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


# --- Fix round 1 -----------------------------------------------------------


def test_disabled_builtin_row_is_badged_disabled():
    rows = _rows(
        [{"name": NAME, "source": "builtin", "builtin_disabled": True}]
    )
    assert rows[NAME].is_builtin and rows[NAME].builtin_disabled


def test_preview_enabled_state_comes_from_config_and_reads_disabled(monkeypatch):
    monkeypatch.setattr(lsc, "_sync_library_canvas", lambda *a, **k: True)
    monkeypatch.setattr(lsbc, "_sync_library_canvas", lambda *a, **k: True)
    calls: list[Any] = []

    async def get_skill(name, **kwargs):
        calls.append(kwargs)
        return {"name": name, "content": "---\nname: x\n---\n# Body\n"}

    fake = _controller_stand_in(
        app_instance=SimpleNamespace(
            skills_scope_service=_service(get_skill=get_skill),
            app_config={"skills": {"disabled_builtins": [NAME]}},
        ),
        _run_library_service_call=_direct_call,
    )
    fake._selected_skill_name = NAME
    fake._library_skills_view = "preview"
    fake._library_skill_detail_generation = 3
    asyncio.run(
        fake.builtin._refresh_library_skill_builtin_preview(NAME, 3)
    )
    assert calls == [{"mode": "local", "include_disabled_builtins": True}]
    assert fake._library_skill_builtin_preview == {
        "name": NAME,
        "content": "# Body\n",
        "enabled": False,
    }


def test_customize_that_copied_nothing_says_so():
    async def seed_builtin_skills(**kwargs):
        return {"seeded": [], "count": 0}

    async def get_skill(name, **kwargs):
        return {"name": name}

    fake = _controller_stand_in(
        app_instance=SimpleNamespace(
            skills_scope_service=_service(
                seed_builtin_skills=seed_builtin_skills, get_skill=get_skill
            )
        ),
        _run_library_service_call=_direct_call,
    )
    asyncio.run(fake.builtin._customize_library_skill_builtin(NAME))
    notices = [m[1] for m in fake.calls if isinstance(m, tuple) and m[0] == "notify"]
    assert notices and not notices[0].startswith("Copied")
    assert "Nothing was copied" in notices[0]
    assert "reset" not in fake.calls and "refresh" not in fake.calls


def test_customize_of_a_modified_builtin_says_it_failed_integrity():
    # Qodo #2 (PR #2842): the service refuses a tampered built-in; the
    # notice must say why, not blame an existing folder.
    async def seed_builtin_skills(**kwargs):
        return {"seeded": [], "count": 0, "blocked": {NAME: "builtin_modified"}}

    async def get_skill(name, **kwargs):
        return {"name": name}

    fake = _controller_stand_in(
        app_instance=SimpleNamespace(
            skills_scope_service=_service(
                seed_builtin_skills=seed_builtin_skills, get_skill=get_skill
            )
        ),
        _run_library_service_call=_direct_call,
    )
    asyncio.run(fake.builtin._customize_library_skill_builtin(NAME))
    notices = [m[1] for m in fake.calls if isinstance(m, tuple) and m[0] == "notify"]
    assert notices and "integrity check" in notices[0]
    assert "reset" not in fake.calls and "refresh" not in fake.calls


def test_rapid_enabled_toggles_leave_disk_matching_memory(monkeypatch):
    import itertools
    import time

    saved: list[list[str]] = []
    entered = itertools.count()

    def slow_first_save(section, key, value):
        if next(entered) == 0:
            time.sleep(0.3)  # the first (soon stale) write is the slow one
        saved.append(list(value))
        return True

    monkeypatch.setattr(lsbc, "save_setting_to_cli_config", slow_first_save)
    monkeypatch.setattr(lsc, "_sync_library_canvas", lambda *a, **k: True)
    monkeypatch.setattr(lsbc, "_sync_library_canvas", lambda *a, **k: True)
    config: dict[str, Any] = {"skills": {"disabled_builtins": []}}
    fake = _controller_stand_in(app_instance=SimpleNamespace(app_config=config))
    fake._library_skill_builtin_preview = {"name": NAME, "enabled": True}

    async def both() -> None:
        await asyncio.gather(
            fake.builtin._set_library_skill_builtin_enabled(NAME, False),
            fake.builtin._set_library_skill_builtin_enabled(NAME, True),
        )

    asyncio.run(both())
    assert config["skills"]["disabled_builtins"] == []
    assert saved[-1] == config["skills"]["disabled_builtins"]


@pytest.mark.parametrize(("source", "expected"), [("builtin", "preview"), ("local", "editor")])
def test_review_link_routes_builtin_only_names_to_preview(monkeypatch, source, expected):
    monkeypatch.setattr(lsc, "_sync_library_canvas", lambda *a, **k: True)
    monkeypatch.setattr(lsbc, "_sync_library_canvas", lambda *a, **k: True)
    opened: list[str] = []

    async def flush() -> bool:
        return True

    async def get_skill(name, **kwargs):
        return {"name": name, "source": source}

    fake = _controller_stand_in(
        app_instance=SimpleNamespace(skills_scope_service=_service(get_skill=get_skill)),
        _run_library_service_call=_direct_call,
        _flush_library_skill_save=flush,
        _refresh_library_skill_detail=lambda name: asyncio.sleep(0),
    )
    fake.builtin._open_library_skill_builtin_preview = lambda n: opened.append("preview")
    asyncio.run(
        LibrarySkillsController._open_library_skill_editor_for_review(fake, NAME)
    )
    if expected == "preview":
        assert opened == ["preview"] and fake._library_skills_view != "editor"
    else:
        assert opened == [] and fake._library_skills_view == "editor"
