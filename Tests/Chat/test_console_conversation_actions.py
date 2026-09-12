"""Menu model for the Console conversation action menu (TASK-23200)."""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat.console_conversation_actions import (
    ACTION_ARCHIVE,
    ACTION_BACK,
    ACTION_DELETE,
    ACTION_FAVORITE,
    ACTION_RENAME,
    ACTION_UNARCHIVE,
    ACTION_UNFAVORITE,
    ARCHIVED_STATE,
    CONVERSATION_STATES,
    ConversationMenuTarget,
    build_conversation_menu,
    conversation_state_label,
    page_from_action,
    state_from_action,
)


def _saved(**overrides) -> ConversationMenuTarget:
    base = {"conversation_id": "conv-1", "title": "Chat 1"}
    base.update(overrides)
    return ConversationMenuTarget(**base)


@pytest.mark.unit
def test_state_vocabulary_matches_the_database():
    """The menu must never offer a state the DB would reject."""
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    assert CONVERSATION_STATES == CharactersRAGDB._ALLOWED_CONVERSATION_STATES
    assert ARCHIVED_STATE in CharactersRAGDB._ALLOWED_CONVERSATION_STATES


@pytest.mark.unit
def test_root_menu_offers_the_six_requested_entries():
    labels = [item.label for item in build_conversation_menu(_saved())]
    assert labels == [
        "Favourite",
        "Change status",
        "Archive",
        "Rename…",
        "Copy as",
        "More",
    ]


@pytest.mark.unit
def test_favourite_toggles_label_and_action_with_current_state():
    plain = build_conversation_menu(_saved(starred=False))[0]
    assert (plain.action_id, plain.label) == (ACTION_FAVORITE, "Favourite")

    starred = build_conversation_menu(_saved(starred=True))[0]
    assert (starred.action_id, starred.label) == (ACTION_UNFAVORITE, "Remove favourite")


@pytest.mark.unit
def test_archive_is_the_resolved_state_and_toggles_to_unarchive():
    plain = build_conversation_menu(_saved(state="in-progress"))[2]
    assert (plain.action_id, plain.label) == (ACTION_ARCHIVE, "Archive")

    archived = build_conversation_menu(_saved(state=ARCHIVED_STATE))[2]
    assert (archived.action_id, archived.label) == (ACTION_UNARCHIVE, "Unarchive")


@pytest.mark.unit
def test_unsaved_conversation_disables_actions_with_a_stated_reason():
    """A disabled control with no explanation is the defect being removed."""
    items = build_conversation_menu(ConversationMenuTarget(conversation_id=None))
    actionable = [item for item in items if not item.action_id.startswith("page:")]
    assert actionable, "expected some gated entries"
    for item in actionable:
        assert not item.enabled
        assert item.disabled_reason, f"{item.action_id} is disabled but unexplained"


@pytest.mark.unit
def test_favourites_unavailable_explains_itself_instead_of_a_jargon_line():
    item = build_conversation_menu(_saved(favorites_available=False))[0]
    assert not item.enabled
    assert "unavailable" in item.disabled_reason.lower()
    assert "star" not in item.disabled_reason.lower()


@pytest.mark.unit
def test_status_page_lists_every_state_and_marks_the_current_one():
    items = build_conversation_menu(_saved(state="backlog"), page="status")
    assert items[0].action_id == ACTION_BACK

    states = items[1:]
    assert [state_from_action(item.action_id) for item in states] == list(
        CONVERSATION_STATES
    )
    current = [item for item in states if item.is_current]
    assert len(current) == 1
    assert state_from_action(current[0].action_id) == "backlog"
    # The current state cannot be re-chosen: picking it would do nothing.
    assert not current[0].enabled
    assert current[0].disabled_reason


@pytest.mark.unit
def test_more_page_holds_delete_behind_the_back_entry():
    items = build_conversation_menu(_saved(), page="more")
    assert [item.action_id for item in items] == [ACTION_BACK, ACTION_DELETE]


@pytest.mark.unit
@pytest.mark.parametrize("page", ["root", "status", "more"])
def test_every_page_renders_something(page):
    assert build_conversation_menu(_saved(), page=page)


@pytest.mark.unit
def test_state_from_action_rejects_states_the_database_would_reject():
    assert state_from_action("set-state:resolved") == "resolved"
    assert state_from_action("set-state:bogus") is None
    assert state_from_action(ACTION_RENAME) is None


@pytest.mark.unit
def test_page_from_action_distinguishes_navigation_from_commands():
    assert page_from_action("page:status") == "status"
    assert page_from_action("page:nope") is None
    assert page_from_action(ACTION_DELETE) is None


@pytest.mark.unit
def test_unknown_state_falls_back_to_the_default_rather_than_raising():
    target = _saved(state="something-else")
    assert target.normalized_state == "in-progress"
    assert not target.is_archived
    assert conversation_state_label("something-else") == "In progress"
    assert conversation_state_label(None) == "In progress"


@pytest.mark.unit
def test_every_state_has_a_human_label():
    for state in CONVERSATION_STATES:
        label = conversation_state_label(state)
        assert label and label != state
