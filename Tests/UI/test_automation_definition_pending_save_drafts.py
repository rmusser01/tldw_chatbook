"""Real Save acknowledgements must retire only their submitted raw draft."""

from __future__ import annotations

import threading
from unittest.mock import AsyncMock

import pytest
from textual.widgets import Button, Checkbox, Input, Select, Static, TextArea

from Tests.UI import test_automation_definition_form as form_tests
from Tests.UI.test_automation_definition_form import (
    _FormHost,
    _fill_minimal_valid_form,
    _until,
)
from tldw_chatbook.Scheduling.services.scheduling_service import SchedulingService
from tldw_chatbook.Scheduling.services.server_client import ServerUnavailableError
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

# Reuse the same real database/service fixture owners as adjacent form tests.
db = form_tests.db
local_service = form_tests.local_service


@pytest.fixture
def held_writes(db, monkeypatch):
    """Hold real physical writes; never synthesize service outcomes."""
    entered, release = threading.Event(), threading.Event()
    calls, settled = [], []
    for method in ("create_automation_definition", "update_automation_definition"):
        original = getattr(db, method)

        def held(*args, _original=original, _method=method, **kwargs):
            calls.append(_method)
            try:
                entered.set()
                assert release.wait(8), "physical write was not released"
                return _original(*args, **kwargs)
            finally:
                settled.append(True)

        monkeypatch.setattr(db, method, held)
    try:
        yield entered, release, calls, settled
    finally:
        release.set()


async def _form_kwargs(service, database, edit):
    if not edit:
        return {}
    initial = await service.save_definition(
        {
            "family": "recurring_question",
            "name": "Seed",
            "input": {"question": "What changed?"},
            "schedule": {"kind": "cron", "cron": "0 9 * * *", "timezone": "UTC"},
        },
        "local",
    )
    assert initial.status == "saved"
    return {
        "definition_id": initial.definition_id,
        "definition_row": database.get_automation_definition(initial.definition_id),
    }


async def _keyboard(pilot, screen, selector, *keys):
    widget = screen.query_one(selector)
    widget.scroll_visible(immediate=True)
    await pilot.pause()
    widget.focus()
    await pilot.press(*keys)
    await pilot.pause()
    assert pilot.app.focused is widget


async def _release_save(pilot, screen, gate):
    _, release, calls, settled = gate
    release.set()
    await _until(lambda: len(settled) == len(calls))
    await _until(lambda: not screen._save_pending)
    await pilot.pause()


@pytest.mark.parametrize("edit", [False, True], ids=["create", "edit"])
@pytest.mark.parametrize(
    "field,keys,expected",
    [
        (
            "name",
            ("end", "space", "L", "A", "T", "E", "R"),
            "Daily standup digest LATER",
        ),
        ("question", ("f7", "space", "space"), "  "),
        ("run-at", ("home", "shift+end", "b", "a", "d"), "bad"),
        ("provider", ("space", "space"), "  "),
        ("generation-mode", ("enter", "down", "enter"), "required"),
        ("notify", ("space",), False),
    ],
    ids=["name", "question", "run-at", "provider", "generation-mode", "notify"],
)
async def test_success_retains_later_raw_draft(
    local_service, db, monkeypatch, edit, field, keys, expected
):
    """Catches unconditional dismissal and normalized-payload comparisons."""
    kwargs = await _form_kwargs(local_service, db, edit)
    # Install the hold after the seed has actually committed.
    entered, release = threading.Event(), threading.Event()
    calls, settled = [], []
    method = "update_automation_definition" if edit else "create_automation_definition"
    original = getattr(db, method)

    def held(*args, **kwargs):
        calls.append(True)
        try:
            entered.set()
            assert release.wait(8)
            return original(*args, **kwargs)
        finally:
            settled.append(True)

    monkeypatch.setattr(db, method, held)
    gate = entered, release, calls, settled
    app = _FormHost(local_service, **kwargs)
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        screen = app.screen
        await _fill_minimal_valid_form(screen)
        screen.query_one("#automation-notify", Checkbox).value = True
        await pilot.pause()
        try:
            await pilot.click("#automation-save")
            await _until(entered.is_set)
            await _keyboard(pilot, screen, f"#automation-{field}", *keys)
            widget = screen.query_one(f"#automation-{field}")
            raw = widget.text if isinstance(widget, TextArea) else widget.value
            assert raw == expected
            assert not widget.disabled
            assert screen.query_one("#automation-save", Button).disabled
        finally:
            await _release_save(pilot, screen, gate)
        rows = db.list_automation_definitions(owner_id="local")
        assert len(rows) == 1
        assert rows[0]["name"] == "Daily standup digest"
        assert rows[0]["input"]["question"] == "What shipped today?"
        assert rows[0]["input"].get("provider") is None
        assert len(calls) == 1
        assert screen.is_attached, "successful acknowledgement removed the newer draft"
        assert app.screen is screen
        assert app.result == "not-yet-dismissed"
        assert screen._dirty
        assert (
            widget.text if isinstance(widget, TextArea) else widget.value
        ) == expected
        assert not screen.query_one("#automation-save", Button).disabled


@pytest.mark.parametrize("edit", [False, True], ids=["create", "edit"])
async def test_unchanged_success_closes_with_real_saved_result(local_service, db, edit):
    """Catches retaining every successful Save instead of only later drafts."""
    kwargs = await _form_kwargs(local_service, db, edit)
    app = _FormHost(local_service, **kwargs)
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        screen = app.screen
        await _fill_minimal_valid_form(screen)
        await pilot.pause()
        await pilot.click("#automation-save")
        await _until(lambda: app.result != "not-yet-dismissed")
        await _until(lambda: not screen.is_attached)
        assert app.result.status == "saved"
        rows = db.list_automation_definitions(owner_id="local")
        assert len(rows) == 1
        assert rows[0]["id"] == app.result.definition_id
        if edit:
            assert app.result.definition_id == kwargs["definition_id"]


async def test_retained_create_next_explicit_save_updates_same_owner_and_row(
    local_service, db, held_writes
):
    """Catches duplicate creates and retargeting the adopted record's owner."""
    entered, _, calls, _ = held_writes
    app = _FormHost(
        local_service,
        available_owners=[("This device", "local"), ("Server", "server:1")],
    )
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        screen = app.screen
        await _fill_minimal_valid_form(screen)
        screen.query_one("#automation-notify", Checkbox).value = True
        await pilot.pause()
        try:
            await pilot.click("#automation-save")
            await _until(entered.is_set)
            await _keyboard(pilot, screen, "#automation-name", "end", "X")
        finally:
            await _release_save(pilot, screen, held_writes)
        assert screen.is_attached
        first_id = db.list_automation_definitions(owner_id="local")[0]["id"]
        owner = screen.query_one("#automation-runs-on", Select)
        assert owner.disabled
        assert owner.value == "local"
        assert "Existing automations keep the owner" in " ".join(
            widget.visual.plain for widget in screen.query(Static)
        )
        assert not screen.query_one("#automation-question", TextArea).disabled
        save = screen.query_one("#automation-save", Button)
        await _until(lambda: not save.has_class("-active"))
        await pilot.click("#automation-save")
        await _until(lambda: app.result != "not-yet-dismissed")
        assert app.result.status == "saved"
        assert app.result.definition_id == first_id
        rows = db.list_automation_definitions(owner_id="local")
        assert len(rows) == 1
        assert rows[0]["name"] == "Daily standup digestX"
        assert calls == ["create_automation_definition", "update_automation_definition"]
        assert db.list_automation_definitions(owner_id="server:1") == []


@pytest.mark.parametrize("owner,status", [("local", "saved"), ("server:1", "queued")])
async def test_discard_later_draft_returns_last_real_success(
    local_service, db, held_writes, owner, status
):
    """Catches dropping workbench refresh by returning None after a real save."""
    if owner != "local":
        client = AsyncMock()
        client.preview_automation_definition.side_effect = ServerUnavailableError(
            "offline"
        )
        local_service = SchedulingService(
            db=db, server_client=client, runtime_source=owner
        )
    entered, _, calls, _ = held_writes
    app = _FormHost(
        local_service, available_owners=[("Owner", owner)], default_owner=owner
    )
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        screen = app.screen
        await _fill_minimal_valid_form(screen)
        screen.query_one("#automation-notify", Checkbox).value = True
        await pilot.pause()
        try:
            await pilot.click("#automation-save")
            await _until(entered.is_set)
            await _keyboard(pilot, screen, "#automation-name", "end", "X")
        finally:
            await _release_save(pilot, screen, held_writes)
        assert screen.is_attached
        await pilot.click("#automation-cancel")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmationDialog)
        await pilot.click("#cancel-button")
        await pilot.pause()
        assert app.screen is screen
        assert screen.query_one("#automation-name", Input).value.endswith("X")
        # Exercise the real quit guard and choose Keep editing too.
        quitting = app.run_worker(screen.confirm_quit())
        await _until(lambda: isinstance(app.screen, ConfirmationDialog))
        await _until(lambda: bool(app.screen.query("#cancel-button")))
        await pilot.pause()
        await pilot.click("#cancel-button")
        await quitting.wait()
        assert quitting.result is False
        assert app.screen is screen
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmationDialog)
        await pilot.click("#confirm-button")
        await _until(lambda: app.result != "not-yet-dismissed")
        assert app.result.status == status
        rows = db.list_automation_definitions(owner_id=owner)
        assert len(rows) == 1
        assert app.result.definition_id == rows[0]["id"]
        assert rows[0]["name"] == "Daily standup digest"
        assert len(calls) == 1
        pending = db.get_pending_mutations(owner, primitive="automation_definition")
        assert len(pending) == (1 if status == "queued" else 0)


async def test_failed_create_unlocks_owner_and_keeps_newer_draft_for_retry(
    local_service, db, held_writes
):
    """Catches an identity lock surviving definite rollback or draft overwrite."""
    with db.transaction() as conn:
        conn.execute("""CREATE TRIGGER fail_pending_create BEFORE INSERT ON automation_definitions
            BEGIN SELECT RAISE(ABORT, 'test refused write'); END""")
    entered, _, calls, _ = held_writes
    app = _FormHost(
        local_service, available_owners=[("Local", "local"), ("Server", "server:1")]
    )
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        screen = app.screen
        await _fill_minimal_valid_form(screen)
        screen.query_one("#automation-notify", Checkbox).value = True
        await pilot.pause()
        try:
            await pilot.click("#automation-save")
            await _until(entered.is_set)
            assert screen.query_one("#automation-runs-on", Select).disabled
            await _keyboard(pilot, screen, "#automation-name", "end", "X")
        finally:
            await _release_save(pilot, screen, held_writes)
        assert screen.is_attached
        assert screen._dirty
        assert not screen.query_one("#automation-runs-on", Select).disabled
        assert (
            screen.query_one("#automation-name", Input).value == "Daily standup digestX"
        )
        assert (
            "Save failed"
            in screen.query_one("#automation-form-errors", Static).visual.plain
        )
        assert db.list_automation_definitions(owner_id="local") == []
        with db.transaction() as conn:
            conn.execute("DROP TRIGGER fail_pending_create")
        save = screen.query_one("#automation-save", Button)
        await _until(lambda: not save.has_class("-active"))
        await pilot.click("#automation-save")
        await _until(lambda: app.result != "not-yet-dismissed")
        assert app.result.status == "saved"
        assert (
            db.get_automation_definition(app.result.definition_id)["name"]
            == "Daily standup digestX"
        )
        assert calls == ["create_automation_definition", "create_automation_definition"]


@pytest.mark.parametrize(
    "read_failure", [False, True], ids=["held-read", "failed-read"]
)
async def test_saved_identity_survives_metadata_read_and_later_keyboard_input(
    local_service, db, held_writes, monkeypatch, read_failure
):
    """Catches stale pre-await comparisons and losing durable identity on read failure."""
    entered, release, calls, settled = held_writes
    read_entered, read_release = threading.Event(), threading.Event()
    original_read = db.get_automation_definition

    def read_saved(*args, **kwargs):
        read_entered.set()
        assert read_release.wait(8), "metadata read was not released"
        if read_failure:
            raise OSError("test post-commit metadata read failed")
        return original_read(*args, **kwargs)

    app = _FormHost(local_service)
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        screen = app.screen
        await _fill_minimal_valid_form(screen)
        screen.query_one("#automation-notify", Checkbox).value = True
        await pilot.pause()
        try:
            await pilot.click("#automation-save")
            await _until(entered.is_set)
            await _keyboard(pilot, screen, "#automation-name", "end", "X")
            # Partial teardown may remove presentation chrome while the raw
            # editor remains attached and the admitted write still commits.
            chrome = "#automation-owner-help" if read_failure else ".form-title"
            await screen.query_one(chrome).remove()
            monkeypatch.setattr(db, "get_automation_definition", read_saved)
            release.set()
            await _until(lambda: len(settled) == len(calls))
            await _until(lambda: read_entered.is_set() or not screen._save_pending)
            assert screen.is_attached
            assert read_entered.is_set()
            # Edit again during the additional metadata await, reverting the
            # first late edit but leaving a new raw Question draft.
            await _keyboard(pilot, screen, "#automation-name", "end", "backspace")
            await _keyboard(
                pilot, screen, "#automation-question", "f7", "space", "space"
            )
        finally:
            release.set()
            read_release.set()
            await _until(lambda: not screen._save_pending)
            monkeypatch.setattr(db, "get_automation_definition", original_read)
        assert screen.is_attached
        assert screen._dirty
        assert screen.query_one("#automation-question", TextArea).text == "  "
        rows = db.list_automation_definitions(owner_id="local")
        assert len(rows) == 1
        assert rows[0]["name"] == "Daily standup digest"
        # Correct the invalid draft using real keys, then Save must UPDATE even
        # after the post-commit read failed, never create a second definition.
        await _keyboard(
            pilot, screen, "#automation-question", "f7", "R", "e", "t", "r", "y", "?"
        )
        await pilot.click("#automation-save")
        await _until(lambda: app.result != "not-yet-dismissed")
        assert app.result.status == "saved"
        assert app.result.definition_id == rows[0]["id"]
        assert len(db.list_automation_definitions(owner_id="local")) == 1
        assert (
            db.get_automation_definition(rows[0]["id"])["input"]["question"] == "Retry?"
        )
        assert calls == ["create_automation_definition", "update_automation_definition"]
