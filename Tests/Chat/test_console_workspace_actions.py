"""Menu model for the Console workspace action menu (TASK-25710)."""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat.console_workspace_actions import (
    ACTION_ACTIVATE,
    ACTION_ARCHIVE,
    ACTION_BACK,
    ACTION_NEW_CHAT,
    ACTION_RAG_SCOPE,
    ACTION_RENAME,
    WorkspaceMenuTarget,
    build_workspace_menu,
    page_from_action,
)


def _workspace(**overrides) -> WorkspaceMenuTarget:
    base = {"workspace_id": "ws-1", "name": "Research", "is_active": False}
    base.update(overrides)
    return WorkspaceMenuTarget(**base)


@pytest.mark.unit
def test_root_menu_offers_the_five_approved_entries():
    labels = [item.label for item in build_workspace_menu(_workspace())]
    assert labels == [
        "Activate",
        "New chat",
        "Rename…",
        "RAG scope…",
        "More",
    ]


@pytest.mark.unit
def test_activate_marks_and_disables_the_active_workspace():
    active = build_workspace_menu(_workspace(is_active=True))[0]
    assert active.is_current is True
    assert active.enabled is False
    assert active.disabled_reason, "the disabled Activate must state its reason"

    inactive = build_workspace_menu(_workspace(is_active=False))[0]
    assert inactive.is_current is False
    assert inactive.enabled is True
    assert inactive.disabled_reason == ""


@pytest.mark.unit
def test_rag_scope_gates_on_active_workspace_with_a_stated_reason():
    inactive = build_workspace_menu(_workspace(is_active=False))[3]
    assert inactive.action_id == ACTION_RAG_SCOPE
    assert inactive.enabled is False
    assert "Activate" in inactive.disabled_reason

    active = build_workspace_menu(_workspace(is_active=True))[3]
    assert active.enabled is True
    assert active.disabled_reason == ""


@pytest.mark.unit
def test_more_page_offers_back_and_archive_only():
    items = build_workspace_menu(_workspace(), page="more")
    assert [(i.action_id, i.label) for i in items] == [
        (ACTION_BACK, "‹ Back"),
        (ACTION_ARCHIVE, "Archive"),
    ]


@pytest.mark.unit
def test_every_disabled_entry_states_its_precondition():
    for page in ("root", "more"):
        for item in build_workspace_menu(_workspace(is_active=True), page=page):
            if not item.enabled:
                assert item.disabled_reason, (
                    f"{item.action_id} on {page} is disabled with no reason"
                )


@pytest.mark.unit
def test_root_commands_carry_no_page_navigation():
    for item in build_workspace_menu(_workspace()):
        if item.opens_page is None:
            assert item.action_id in {
                ACTION_ACTIVATE,
                ACTION_NEW_CHAT,
                ACTION_RENAME,
                ACTION_RAG_SCOPE,
            }


@pytest.mark.unit
def test_page_from_action_recognises_only_workspace_pages():
    assert page_from_action("page:more") == "more"
    assert page_from_action("page:root") == "root"
    assert page_from_action(ACTION_ACTIVATE) is None
    # Pages from the sibling conversation menu must not leak in.
    assert page_from_action("page:status") is None
