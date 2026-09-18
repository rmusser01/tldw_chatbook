"""Keyboard journeys through retained Prompt history under production CSS."""

from __future__ import annotations

import pytest
from textual.widgets import Button, TextArea

from Tests.UI.test_library_prompt_action_journeys import _activate_more_action
from Tests.UI.test_library_prompts_canvas import (
    _build_test_app,
    _open_prompt_editor,
    _real_prompt_scope_service,
    _wire_empty_non_prompt_services,
)
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _painted_text,
    _wait_for_condition,
    _wait_for_library_shell,
)


def _history_host(tmp_path, theme, versions=12):
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _, _ = db.add_prompt(
        name="History [bold] café",
        author="Zoë",
        details="A retained message",
        system_prompt="",
        user_prompt="Version 1\n\nKeep [bold] literal text.",
        keywords=["history"],
    )
    for version in range(2, versions + 1):
        db.update_prompt_by_id(
            prompt_id,
            {"user_prompt": f"Version {version}\n\nKeep [bold] literal text."},
            expected_version=version - 1,
        )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryProductionCSSHarness(app)
    host.theme = theme
    app.notify = host.notify
    return db, service, prompt_id, host


async def _history_loaded(screen, pilot):
    await _wait_for_condition(
        pilot,
        lambda: (
            screen._library_prompt_history_state.page_status == "loaded"
            and bool(screen.query(".library-prompt-history-row"))
        ),
        message="History page did not load",
    )


def _row(screen, version):
    return next(
        row
        for row in screen.query(".library-prompt-history-row")
        if row.source_version == version
    )


async def _focused(screen, host, pilot, selector, label):
    await _wait_for_condition(
        pilot,
        lambda: (
            screen.focused is screen.query_one(selector)
            and label in _painted_text(host, screen.focused.region)
        ),
        message=lambda: (
            f"History focus={screen.focused!r}; target={selector}; "
            f"painted={_painted_text(host, screen.query_one(selector).region)!r}"
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_history_menu_selection_and_older_page_preserve_keyboard_position(
    tmp_path, size, theme
):
    db, _, prompt_id, host = _history_host(tmp_path, theme)
    original = db.fetch_prompt_details(prompt_id)
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        editor_fields = tuple(screen.query(TextArea))
        await _activate_more_action(screen, pilot, "more-history")
        await _history_loaded(screen, pilot)
        await _focused(
            screen,
            host,
            pilot,
            "#library-prompt-history-collapsible > CollapsibleTitle",
            "Retained history (12)",
        )
        assert [
            row.source_version for row in screen.query(".library-prompt-history-row")
        ] == list(range(12, 2, -1))

        await pilot.press("tab")
        assert screen.focused is _row(screen, 12)
        await pilot.press("enter")
        await _focused(screen, host, pilot, f"#{_row(screen, 12).id}", "v12")
        preview = screen.query_one("#library-prompt-history-user", TextArea)
        assert preview.text == original["user_prompt"]
        assert preview.read_only
        assert all(field.is_attached for field in editor_fields)

        for version in range(11, 2, -1):
            await pilot.press("tab")
            assert screen.focused is _row(screen, version)
        await pilot.press("tab")
        assert screen.focused.id == "library-prompt-history-load-older"
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: len(screen.query(".library-prompt-history-row")) == 12,
            message="Older retained versions did not appear",
        )
        await _focused(screen, host, pilot, f"#{_row(screen, 2).id}", "v2")
        await pilot.press("tab", "enter")
        await _focused(screen, host, pilot, f"#{_row(screen, 1).id}", "v1")
        assert (
            screen.query_one("#library-prompt-history-user", TextArea).text
            == "Version 1\n\nKeep [bold] literal text."
        )
        assert db.fetch_prompt_details(prompt_id) == original
        assert all(field.is_attached for field in editor_fields)


async def _tab_to(screen, host, pilot, target_id, label):
    for _ in range(8):
        if screen.focused is screen.query_one(f"#{target_id}"):
            break
        await pilot.press("tab")
    await _focused(screen, host, pilot, f"#{target_id}", label)


async def _confirm_restore(host, screen, pilot, *, confirm):
    from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

    await _wait_for_condition(
        pilot,
        lambda: not screen.focused.has_class("-active"),
        message="Restore button is still completing its prior press",
    )
    await pilot.press("enter")
    await _wait_for_condition(
        pilot,
        lambda: isinstance(host.screen, ConfirmationDialog),
        message="Restore did not open confirmation",
    )
    dialog = host.screen
    assert "creates a new current version" in dialog.message
    assert dialog.focused is dialog.query_one("#cancel-button", Button)
    assert "Cancel" in _painted_text(host, dialog.focused.region)
    if confirm:
        await pilot.press("tab")
        assert dialog.focused is dialog.query_one("#confirm-button", Button)
        assert "Restore" in _painted_text(host, dialog.focused.region)
        await pilot.press("enter")
    else:
        await pilot.press("escape")
    await _wait_for_condition(
        pilot, lambda: host.screen is screen, message="Restore dialog did not close"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_history_restore_cancel_no_change_and_new_version_are_readable(
    tmp_path, size, theme
):
    db, _, prompt_id, host = _history_host(tmp_path, theme, versions=2)
    original = db.fetch_prompt_details(prompt_id)
    retained = db.get_prompt_history_entries(original["uuid"], page_size=10)
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        await _activate_more_action(screen, pilot, "more-history")
        await _history_loaded(screen, pilot)
        await _focused(
            screen,
            host,
            pilot,
            "#library-prompt-history-collapsible > CollapsibleTitle",
            "Retained history (2)",
        )
        await pilot.press("tab", "enter")
        await _focused(screen, host, pilot, f"#{_row(screen, 2).id}", "v2")
        await _tab_to(
            screen,
            host,
            pilot,
            "library-prompt-history-restore",
            "Restore selected version",
        )
        await _confirm_restore(host, screen, pilot, confirm=False)
        await _focused(
            screen,
            host,
            pilot,
            "#library-prompt-history-restore",
            "Restore selected version",
        )
        assert db.fetch_prompt_details(prompt_id) == original

        await _confirm_restore(host, screen, pilot, confirm=True)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_prompt_history_state.restore_outcome is not None,
            message="Identical restore did not report its outcome",
        )
        assert screen._library_prompt_history_state.restore_outcome.kind == "no_change"
        await _focused(
            screen,
            host,
            pilot,
            "#library-prompt-history-restore",
            "Restore selected version",
        )
        assert (
            "Retained v2 already matches current v2; no new version was created."
            == " ".join(
                _painted_text(
                    host, screen.query_one("#library-prompt-history-outcome").region
                ).split()
            )
        )
        assert db.fetch_prompt_details(prompt_id) == original

        # Re-enter from the disclosure to walk to the oldest retained row.
        screen.query_one(
            "#library-prompt-history-collapsible > CollapsibleTitle"
        ).focus()
        await pilot.press("tab", "tab", "enter")
        await _focused(screen, host, pilot, f"#{_row(screen, 1).id}", "v1")
        await _tab_to(
            screen,
            host,
            pilot,
            "library-prompt-history-restore",
            "Restore selected version",
        )
        await _confirm_restore(host, screen, pilot, confirm=True)
        await _wait_for_condition(
            pilot,
            lambda: screen._prompts_state.version == 3,
            message="Restore did not adopt its new current version",
        )
        await _focused(
            screen,
            host,
            pilot,
            "#library-prompt-history-collapsible > CollapsibleTitle",
            "Retained history (3)",
        )
        restored = db.fetch_prompt_details(prompt_id)
        assert restored["user_prompt"] == "Version 1\n\nKeep [bold] literal text."
        assert restored["version"] == 3
        history = db.get_prompt_history_entries(original["uuid"], page_size=10)
        assert history["items"][1:] == retained["items"]


@pytest.mark.asyncio
async def test_history_page_completion_respects_newer_keyboard_focus(
    tmp_path, monkeypatch
):
    import asyncio

    _, _, prompt_id, host = _history_host(tmp_path, "textual-dark")
    async with host.run_test(size=(80, 24)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        await _activate_more_action(screen, pilot, "more-history")
        await _history_loaded(screen, pilot)
        started, release = asyncio.Event(), asyncio.Event()
        run_service = screen._run_library_service_call

        async def held_page(method, **kwargs):
            if kwargs.get("before_change_id") is not None:
                started.set()
                await release.wait()
            return await run_service(method, **kwargs)

        monkeypatch.setattr(screen, "_run_library_service_call", held_page)
        screen.query_one("#library-prompt-history-load-older", Button).focus()
        await pilot.press("enter")
        await _wait_for_condition(
            pilot, started.is_set, message="Older page was not requested"
        )
        try:
            opener = screen.query_one("#library-prompt-more-actions", Button)
            opener.focus()
            await _focused(
                screen, host, pilot, "#library-prompt-more-actions", "More actions"
            )
        finally:
            release.set()
        await _wait_for_condition(
            pilot,
            lambda: len(screen.query(".library-prompt-history-row")) == 12,
            message="Held page did not settle",
        )
        await pilot.pause()
        assert screen.focused is opener


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
async def test_history_keyboard_retry_and_dirty_gate_preserve_saved_content(
    tmp_path, monkeypatch, size
):
    from textual.widgets import Input

    from tldw_chatbook.DB.Prompts_DB import DatabaseError

    db, service, prompt_id, host = _history_host(tmp_path, "textual-light", versions=2)
    original = db.fetch_prompt_details(prompt_id)
    list_versions = service.list_prompt_versions
    restore_entry = db.restore_prompt_history_entry
    page_attempts = restore_attempts = 0

    def fail_first_page(**kwargs):
        nonlocal page_attempts
        page_attempts += 1
        if page_attempts == 1:
            raise RuntimeError("Private backend failure")
        return list_versions(**kwargs)

    def fail_first_restore(*args, **kwargs):
        nonlocal restore_attempts
        restore_attempts += 1
        if restore_attempts == 1:
            raise DatabaseError("Private backend failure")
        return restore_entry(*args, **kwargs)

    monkeypatch.setattr(service, "list_prompt_versions", fail_first_page)
    monkeypatch.setattr(db, "restore_prompt_history_entry", fail_first_restore)
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        await _activate_more_action(screen, pilot, "more-history")
        await _wait_for_condition(
            pilot,
            lambda: screen._library_prompt_history_state.page_status == "error",
            message="Initial page failure did not appear",
        )
        await pilot.press("tab")
        await _focused(
            screen, host, pilot, "#library-prompt-history-retry-page", "Retry"
        )
        assert "Couldn't load retained history" in _painted_text(
            host, screen.query_one("#library-prompt-history-page-error").region
        )
        await pilot.press("enter")
        await _history_loaded(screen, pilot)
        await _focused(screen, host, pilot, f"#{_row(screen, 2).id}", "v2")
        await pilot.press("enter")
        await _tab_to(
            screen,
            host,
            pilot,
            "library-prompt-history-restore",
            "Restore selected version",
        )
        await _confirm_restore(host, screen, pilot, confirm=True)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_prompt_history_state.restore_outcome is not None,
            message="Restore failure did not appear",
        )
        assert screen._library_prompt_history_state.restore_outcome.kind == "error"
        await _focused(
            screen,
            host,
            pilot,
            "#library-prompt-history-restore",
            "Restore selected version",
        )
        assert "Couldn't restore retained history." in _painted_text(
            host, screen.query_one("#library-prompt-history-outcome").region
        )
        await _confirm_restore(host, screen, pilot, confirm=True)
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_prompt_history_state.restore_outcome is not None
                and screen._library_prompt_history_state.restore_outcome.kind
                == "no_change"
            ),
            message="Restore retry did not complete",
        )
        await _focused(
            screen,
            host,
            pilot,
            "#library-prompt-history-restore",
            "Restore selected version",
        )
        assert page_attempts == 2 and restore_attempts == 2
        name = screen.query_one("#library-prompt-name", Input)
        name.focus()
        await pilot.press("end", "x")
        await _wait_for_condition(
            pilot,
            lambda: screen._prompts_state.dirty,
            message="Editing the name did not mark the draft dirty",
        )
        assert screen.query_one("#library-prompt-history-restore", Button).disabled
        assert screen.query_one("#library-prompt-history-user", TextArea).read_only
        assert db.fetch_prompt_details(prompt_id) == original
