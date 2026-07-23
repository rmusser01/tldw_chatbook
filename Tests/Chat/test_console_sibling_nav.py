"""Sibling-navigation tests for Console conversation branching (TASK-7).

Phase A wires `<`/`>` to move the active leaf across PERSISTED sibling
branches (Task 6's regenerate fork) rather than cycling the old in-memory
``ConsoleVariantSet``. These tests pin ``ChatScreen._select_console_message_
variant``'s contract directly: given a ``message_id`` (the transcript ROW a
swipe button was clicked on -- not necessarily the current active leaf, since
a previous swipe may have landed deep inside a sibling's own branch), it
resolves siblings via ``store.siblings_at``, computes the target index by
direction, and retargets the active leaf at the target sibling's
most-recent-descendant leaf (``store._leaf_under``).

Built via the same unmounted-``ChatScreen(app)`` pattern as
``Tests/UI/test_console_run_gate.py`` -- the method under test only touches
``self._ensure_console_chat_store()``, never widget queries, so no Textual
mount is required.
"""

from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


def _build_screen():
    app = _build_test_app()
    screen = ChatScreen(app)
    return app, screen


def test_sibling_nav_moves_active_leaf_between_two_siblings():
    """Given two assistant siblings a1, a2 (a2 active), variant-previous
    moves the active leaf back to a1; variant-next returns to a2."""
    app, screen = _build_screen()
    store = screen._ensure_console_chat_store()
    session = store.ensure_session(title="t")
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    a1 = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="one"
    )
    a2 = store.create_sibling(a1.id, role=ConsoleMessageRole.ASSISTANT, content="two")
    assert store.active_leaf(session.id) == a2.id

    screen._select_console_message_variant(a2.id, direction="variant-previous")

    assert store.active_leaf(session.id) == a1.id
    _, index, count = store.siblings_at(store.active_leaf(session.id))
    assert (index, count) == (0, 2)

    screen._select_console_message_variant(a1.id, direction="variant-next")

    assert store.active_leaf(session.id) == a2.id


def test_sibling_nav_previous_restores_deepest_descendant_of_sibling_subtree():
    """Swiping to a sibling that has its OWN downstream conversation lands on
    its most-recent descendant, not on the fork point itself."""
    app, screen = _build_screen()
    store = screen._ensure_console_chat_store()
    session = store.ensure_session(title="t")
    user1 = store.append_message(session.id, role=ConsoleMessageRole.USER, content="q1")
    a1 = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="one"
    )
    user2 = store.append_message(session.id, role=ConsoleMessageRole.USER, content="q2")
    a1_reply2 = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="one-b"
    )
    # Fork alongside a1 (NOT a1_reply2) -- a2 is a's sibling, parented at
    # user1, regardless of how far a1's own branch has since continued.
    a2 = store.create_sibling(a1.id, role=ConsoleMessageRole.ASSISTANT, content="two")
    assert store.active_leaf(session.id) == a2.id

    screen._select_console_message_variant(a2.id, direction="variant-previous")

    assert store.active_leaf(session.id) == a1_reply2.id
    assert store.active_path_message_ids(session.id) == [
        user1.id,
        a1.id,
        user2.id,
        a1_reply2.id,
    ]

    # The swipe row for the fork point is still a1 (not the leaf it resolved
    # to) -- pressing variant-next there returns to a2.
    screen._select_console_message_variant(a1.id, direction="variant-next")

    assert store.active_leaf(session.id) == a2.id


def test_sibling_nav_previous_is_a_noop_at_the_first_sibling():
    app, screen = _build_screen()
    store = screen._ensure_console_chat_store()
    session = store.ensure_session(title="t")
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    a1 = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="one"
    )
    store.create_sibling(a1.id, role=ConsoleMessageRole.ASSISTANT, content="two")

    screen._select_console_message_variant(a1.id, direction="variant-previous")

    # a1 is the FIRST sibling (index 0) -- pressing "previous" on its own row
    # must not move the active leaf (which is still a2, off a1's row).
    assert store.active_leaf(session.id) != a1.id


def test_sibling_nav_next_is_a_noop_at_the_last_sibling():
    app, screen = _build_screen()
    store = screen._ensure_console_chat_store()
    session = store.ensure_session(title="t")
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    a1 = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="one"
    )
    a2 = store.create_sibling(a1.id, role=ConsoleMessageRole.ASSISTANT, content="two")
    assert store.active_leaf(session.id) == a2.id

    screen._select_console_message_variant(a2.id, direction="variant-next")

    assert store.active_leaf(session.id) == a2.id


def test_sibling_nav_is_a_noop_for_a_linear_single_child_message():
    """A message with no siblings must not crash or move the active leaf --
    the UI never shows `<`/`>` for it, but the handler stays defensive."""
    app, screen = _build_screen()
    store = screen._ensure_console_chat_store()
    session = store.ensure_session(title="t")
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    a1 = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="only"
    )
    assert store.active_leaf(session.id) == a1.id

    screen._select_console_message_variant(a1.id, direction="variant-previous")
    assert store.active_leaf(session.id) == a1.id

    screen._select_console_message_variant(a1.id, direction="variant-next")
    assert store.active_leaf(session.id) == a1.id
