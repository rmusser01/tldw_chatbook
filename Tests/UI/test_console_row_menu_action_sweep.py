"""Every row-menu action id, dispatched against a live Console (TASK-33621.12).

G3-02 (2026-09-29 Console UX review): conversation row ▸ Copy as ▸ Save .md…
called ``push_screen`` on the ``ChatScreen`` -- a ``Screen``, which has no
such method -- so the worker's ``AttributeError`` ended the whole app. The
only test of that action replaced the handler with a fake, so the broken call
had never run.

This sweep closes the class, not the one instance: every action id the
conversation-row and workspace-row menus can emit is posted to the real
``ChatScreen`` handler -- real ChaChaNotes database, real workspace registry,
real workers and modal pushes, no monkeypatched seam -- and the app must still
be running afterwards.

The id lists are DERIVED from the pure menu models over every target shape and
page, not copied by hand, so a new menu entry is swept automatically. A guard
also checks that every module-level ``ACTION_*`` command constant is among
them: an entry reachable only from a target shape this file does not build
fails loudly instead of going unswept. Page ids (``page:*``) are included --
the menus consume them, but the screen handlers must tolerate them too.
"""

from __future__ import annotations

from itertools import product
from typing import get_args

import pytest

from Tests.UI.test_console_left_rail import make_console_pilot
from tldw_chatbook.Chat import console_conversation_actions as conversation_actions
from tldw_chatbook.Chat import console_workspace_actions as workspace_actions

# The mounted cases build and reload app config; under the per-test sandbox
# redirect that trips ADR-126 admission (`raw_source_selection_changed`)
# before any body runs -- see lessons-testing-evidence.md, "Tests/UI
# `RecoveryRequired` at setup is a profile-selection trip". The collection-time
# test profile is still a private temp root, never the user's real config.
pytestmark = pytest.mark.bootstrap_profile

_SETTLE_SECONDS = 1.0


def _command_constants(module) -> set[str]:
    """Module-level ``ACTION_*`` ids, minus the ``prefix:`` builders."""
    return {
        value
        for name, value in vars(module).items()
        if name.startswith("ACTION_")
        and isinstance(value, str)
        and not value.endswith(":")
    }


def _conversation_action_ids() -> list[str]:
    ids: set[str] = set()
    for starred, state, manual_unread in product(
        (False, True),
        conversation_actions.CONVERSATION_STATES,
        (False, True),
    ):
        target = conversation_actions.ConversationMenuTarget(
            conversation_id="conv-sweep",
            title="Sweep chat",
            state=state,
            starred=starred,
            has_messages=True,
            manual_unread=manual_unread,
        )
        for page in get_args(conversation_actions.MenuPage):
            ids.update(
                item.action_id
                for item in conversation_actions.build_conversation_menu(target, page)
            )
    return sorted(ids)


def _workspace_action_ids() -> list[str]:
    ids: set[str] = set()
    for is_active in (False, True):
        target = workspace_actions.WorkspaceMenuTarget(
            workspace_id="ws-sweep", name="Sweep workspace", is_active=is_active
        )
        for page in get_args(workspace_actions.MenuPage):
            ids.update(
                item.action_id
                for item in workspace_actions.build_workspace_menu(target, page)
            )
    return sorted(ids)


CONVERSATION_ACTION_IDS = _conversation_action_ids()
WORKSPACE_ACTION_IDS = _workspace_action_ids()


def test_the_sweep_covers_every_declared_action_constant() -> None:
    """A new ``ACTION_*`` the derived shapes never emit must fail here."""
    missing_conversation = _command_constants(conversation_actions) - set(
        CONVERSATION_ACTION_IDS
    )
    missing_workspace = _command_constants(workspace_actions) - set(
        WORKSPACE_ACTION_IDS
    )
    assert not missing_conversation, (
        "conversation menu actions no swept target shape emits: "
        f"{sorted(missing_conversation)}"
    )
    assert not missing_workspace, (
        "workspace menu actions no swept target shape emits: "
        f"{sorted(missing_workspace)}"
    )
    # Anchor the derivation to the ids G3-02 was about, so a broken
    # enumeration cannot pass by producing nothing.
    assert "save-markdown" in CONVERSATION_ACTION_IDS
    assert "archive" in WORKSPACE_ACTION_IDS


def _record_started_workers(app) -> list:
    """Remember every worker the app starts from now on.

    A worker started with ``exit_on_error=True`` (the default) that raises
    ends the app, which ``app._exception`` records. A non-fatal worker's
    failure is quieter, and Textual drops every finished worker from
    ``app.workers`` the moment it ends, so looking there afterwards never sees
    it -- and a fast worker (the save's render finishes inside one frame)
    slips between any polling interval. This wraps the registry's
    ``add_worker`` on this one app instance as a pure pass-through spy: it
    records the worker and calls straight through, so no scheduling, state or
    product seam changes.
    """
    started: list = []
    register = app.workers.add_worker

    def add_worker(worker, start=True, exclusive=True):
        started.append(worker)
        return register(worker, start=start, exclusive=exclusive)

    app.workers.add_worker = add_worker
    return started


async def _settle_and_assert_alive(pilot, action_id: str, started: list) -> None:
    """Let the action's workers run, then require a living app and no failed
    worker among those it started."""
    from textual.worker import WorkerState

    await pilot.pause(_SETTLE_SECONDS)
    app = pilot.app
    assert app._exception is None, f"{action_id!r} ended the app: {app._exception!r}"
    assert app.is_running, f"{action_id!r} stopped the app"
    failed = [
        (worker.name or worker.group, repr(worker.error))
        for worker in started
        if worker.state == WorkerState.ERROR
    ]
    assert not failed, f"{action_id!r} had failed workers: {failed}"


@pytest.mark.asyncio
@pytest.mark.parametrize("action_id", CONVERSATION_ACTION_IDS)
async def test_conversation_row_action_never_ends_the_app(
    tmp_path, action_id: str
) -> None:
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Widgets.Console.console_conversation_action_menu import (
        ConversationActionChosen,
    )

    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        db = CharactersRAGDB(str(tmp_path / "sweep.db"), "sweep")
        screen.app_instance.chachanotes_db = db
        conversation_id = db.add_conversation({"title": "Sweep chat"})
        for sender, content in (("user", "sweep question"), ("assistant", "sweep answer")):
            db.add_message(
                {
                    "conversation_id": conversation_id,
                    "sender": sender,
                    "content": content,
                }
            )
        # Shape the target the way the menu would have been opened for this
        # entry to be offered (Remove favourite only on a starred row, etc.).
        target = conversation_actions.ConversationMenuTarget(
            conversation_id=conversation_id,
            title="Sweep chat",
            state=(
                conversation_actions.ARCHIVED_STATE
                if action_id == conversation_actions.ACTION_UNARCHIVE
                else conversation_actions.DEFAULT_CONVERSATION_STATE
            ),
            starred=action_id == conversation_actions.ACTION_UNFAVORITE,
            has_messages=True,
            manual_unread=action_id == conversation_actions.ACTION_MARK_READ,
        )

        started = _record_started_workers(pilot.app)
        screen.post_message(ConversationActionChosen(action_id, target))

        await _settle_and_assert_alive(pilot, action_id, started)
        if action_id == conversation_actions.ACTION_SAVE_MARKDOWN:
            # Non-vacuity: the spy must actually see the action's worker run
            # to completion, or the failed-worker check proves nothing.
            from textual.worker import WorkerState

            from tldw_chatbook.UI.Console_Modules.markdown_export import (
                SAVE_MARKDOWN_WORKER_GROUP,
            )

            assert [
                worker.state
                for worker in started
                if worker.group == SAVE_MARKDOWN_WORKER_GROUP
            ] == [WorkerState.SUCCESS]


@pytest.mark.asyncio
@pytest.mark.parametrize("action_id", WORKSPACE_ACTION_IDS)
async def test_workspace_row_action_never_ends_the_app(action_id: str) -> None:
    from tldw_chatbook.Widgets.Console.console_workspace_action_menu import (
        WorkspaceActionChosen,
    )

    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        registry = screen.app_instance.workspace_registry_service
        record = registry.create_workspace(
            workspace_id="ws-sweep", name="Sweep workspace"
        )
        # RAG scope is only offered on the active workspace; everything else
        # is offered on an inactive one, which also exercises activation.
        is_active = action_id == workspace_actions.ACTION_RAG_SCOPE
        if is_active:
            registry.set_active_workspace(record.workspace_id)
        target = workspace_actions.WorkspaceMenuTarget(
            workspace_id=record.workspace_id,
            name=record.name,
            is_active=is_active,
        )

        started = _record_started_workers(pilot.app)
        screen.post_message(WorkspaceActionChosen(action_id, target))

        await _settle_and_assert_alive(pilot, action_id, started)
