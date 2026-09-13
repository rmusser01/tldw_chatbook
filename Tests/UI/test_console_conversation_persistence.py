"""Database write paths behind the conversation action menu (TASK-23200).

Qodo review on PR #2233: the menu shipped with coverage of its shape,
navigation and the favourite write, but the three handlers it added that
actually touch the database -- change status, rename, delete -- had none.
These exercise the write, the refusal, the failure and the confirmation for
each, because every one of them is a path a user can reach from one click.
"""

from __future__ import annotations

import pytest

from Tests.UI.test_console_left_rail import make_console_pilot


class _FakeDB:
    """Minimal stand-in recording writes, with switchable failure modes."""

    def __init__(self, *, state: str = "in-progress", missing: bool = False) -> None:
        self.record = None if missing else {
            "id": "conv-1",
            "title": "Chat 1",
            "state": state,
            "version": 3,
        }
        self.updates: list[tuple[str, dict, int]] = []
        self.deletes: list[tuple[str, int]] = []
        self.raise_on_write: Exception | None = None

    def get_conversation_by_id(self, conversation_id: str):
        return self.record

    def update_conversation(self, conversation_id, update_data, expected_version):
        if self.raise_on_write is not None:
            raise self.raise_on_write
        self.updates.append((conversation_id, dict(update_data), expected_version))
        return True

    def soft_delete_conversation(self, conversation_id, expected_version):
        if self.raise_on_write is not None:
            raise self.raise_on_write
        self.deletes.append((conversation_id, expected_version))
        return True


async def _console_with_db(pilot, db):
    console = pilot.app.screen
    console.app_instance.chachanotes_db = db
    notes: list[str] = []
    console.app_instance.notify = lambda message, **kw: notes.append(str(message))
    return console, notes


async def _settle(pilot, console):
    await console.workers.wait_for_complete()
    await pilot.pause(0.2)


@pytest.mark.asyncio
async def test_change_status_writes_the_canonical_state_and_confirms() -> None:
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        db = _FakeDB()
        console, notes = await _console_with_db(pilot, db)

        console._workspace.set_console_conversation_state(
            "conv-1", "backlog", conversation_title="Chat 1"
        )
        await _settle(pilot, console)

        assert db.updates == [("conv-1", {"state": "backlog"}, 3)]
        assert any("Backlog" in note for note in notes), notes


@pytest.mark.asyncio
async def test_archive_writes_resolved_rather_than_a_separate_flag() -> None:
    """Archive is a state, not a column -- pin that it stays one."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        db = _FakeDB()
        console, _notes = await _console_with_db(pilot, db)

        from tldw_chatbook.Chat.console_conversation_actions import ARCHIVED_STATE

        console._workspace.set_console_conversation_state(
            "conv-1", ARCHIVED_STATE, conversation_title="Chat 1"
        )
        await _settle(pilot, console)

        assert db.updates == [("conv-1", {"state": "resolved"}, 3)]


@pytest.mark.asyncio
async def test_a_missing_conversation_warns_instead_of_crashing() -> None:
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        db = _FakeDB(missing=True)
        console, notes = await _console_with_db(pilot, db)

        console._workspace.set_console_conversation_state(
            "gone", "backlog", conversation_title="Ghost"
        )
        await _settle(pilot, console)

        assert not db.updates
        assert any("Ghost" in note for note in notes), notes


@pytest.mark.asyncio
async def test_a_failing_write_reports_rather_than_going_silent() -> None:
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        db = _FakeDB()
        db.raise_on_write = RuntimeError("locked")
        console, notes = await _console_with_db(pilot, db)

        console._workspace.set_console_conversation_state(
            "conv-1", "backlog", conversation_title="Chat 1"
        )
        await _settle(pilot, console)

        assert not db.updates
        assert any("Could not change" in note for note in notes), notes


@pytest.mark.asyncio
async def test_rename_writes_the_new_title() -> None:
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        db = _FakeDB()
        console, notes = await _console_with_db(pilot, db)

        console._workspace._rename_console_conversation("conv-1", "Planning notes")
        await _settle(pilot, console)

        assert db.updates == [("conv-1", {"title": "Planning notes"}, 3)]
        assert any("Planning notes" in note for note in notes), notes


@pytest.mark.asyncio
async def test_rename_refuses_a_title_the_shared_validator_rejects() -> None:
    """Qodo review: user text must not reach persistence on a bare strip()."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        db = _FakeDB()
        console, notes = await _console_with_db(pilot, db)

        captured = {}

        def _fake_push(screen, callback=None):
            captured["callback"] = callback

        console.app_instance.push_screen = _fake_push
        console._workspace.open_console_conversation_rename("conv-1", "Chat 1")
        assert "callback" in captured, "the rename prompt was never opened"

        captured["callback"]("x" * 5000)
        await _settle(pilot, console)

        assert not db.updates, "an over-long title reached the database"
        assert any("cannot be used" in note for note in notes), notes


@pytest.mark.asyncio
async def test_rename_to_the_same_title_is_a_no_op() -> None:
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        db = _FakeDB()
        console, _notes = await _console_with_db(pilot, db)

        captured = {}
        console.app_instance.push_screen = lambda screen, callback=None: captured.update(
            callback=callback
        )
        console._workspace.open_console_conversation_rename("conv-1", "Chat 1")
        captured["callback"]("  Chat 1  ")
        await _settle(pilot, console)

        assert not db.updates


@pytest.mark.asyncio
async def test_delete_soft_deletes_and_confirms() -> None:
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        db = _FakeDB()
        console, notes = await _console_with_db(pilot, db)

        console._workspace._delete_console_conversation("conv-1", "Chat 1")
        await _settle(pilot, console)

        assert db.deletes == [("conv-1", 3)]
        assert any("Deleted" in note for note in notes), notes


@pytest.mark.asyncio
async def test_delete_asks_before_it_writes() -> None:
    """A one-click delete must not reach the database unconfirmed."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        db = _FakeDB()
        console, _notes = await _console_with_db(pilot, db)

        captured = {}
        console.app_instance.push_screen = lambda screen, callback=None: captured.update(
            callback=callback, screen=screen
        )
        console._workspace.confirm_console_conversation_delete("conv-1", "Chat 1")

        assert "callback" in captured, "delete did not ask for confirmation"
        await _settle(pilot, console)
        assert not db.deletes, "delete wrote before the user confirmed"

        captured["callback"](False)
        await _settle(pilot, console)
        assert not db.deletes, "declining the dialog still deleted"

        captured["callback"](True)
        await _settle(pilot, console)
        assert db.deletes == [("conv-1", 3)]


@pytest.mark.asyncio
async def test_the_menu_reads_canonical_state_not_the_row_display_copy() -> None:
    """Qodo review, PR #2233: row.status is display copy, not a DB state.

    Rows reach the browser carrying "active", "open", "workspace" or
    "workspace-thread" as well as real states, and every non-canonical value
    normalises to in-progress -- so a RESOLVED conversation shown as an
    "active session" row offered Archive instead of Unarchive.
    """
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        db = _FakeDB(state="resolved")
        console, _notes = await _console_with_db(pilot, db)

        assert console._row_actions._console_conversation_state("conv-1") == "resolved"

        from tldw_chatbook.Chat.console_conversation_actions import (
            ACTION_UNARCHIVE,
            ConversationMenuTarget,
            build_conversation_menu,
        )

        target = ConversationMenuTarget(
            conversation_id="conv-1",
            title="Chat 1",
            state=console._row_actions._console_conversation_state("conv-1"),
        )
        assert target.is_archived
        assert build_conversation_menu(target)[2].action_id == ACTION_UNARCHIVE


@pytest.mark.asyncio
async def test_state_lookup_falls_back_rather_than_blocking_the_menu() -> None:
    """An unsaved chat, a missing record or a broken DB must still open."""
    from tldw_chatbook.Chat.console_conversation_actions import (
        DEFAULT_CONVERSATION_STATE,
    )

    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        console, _notes = await _console_with_db(pilot, _FakeDB(missing=True))
        assert console._row_actions._console_conversation_state(None) == DEFAULT_CONVERSATION_STATE
        assert console._row_actions._console_conversation_state("gone") == DEFAULT_CONVERSATION_STATE

        class _Broken:
            def get_conversation_by_id(self, conversation_id):
                raise RuntimeError("db down")

        console.app_instance.chachanotes_db = _Broken()
        assert (
            console._row_actions._console_conversation_state("conv-1") == DEFAULT_CONVERSATION_STATE
        )
