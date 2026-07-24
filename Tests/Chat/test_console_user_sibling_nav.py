"""User-message sibling-navigation tests (Console branching Phase B, Task 4).

Phase A's swipe/counter gate is role-generic (``sibling_count > 1``): the
transcript counter, the `<`/`>` action row, and ``ChatScreen.
_select_console_message_variant`` were all written against the store's
persisted-sibling tree, not against a hard-coded "assistant only" rule. The
sibling-nav tests filed under Task 7 (``Tests/Chat/test_console_sibling_
nav.py``, ``Tests/UI/test_console_native_transcript.py``) only ever exercise
ASSISTANT siblings (regenerate's own use case), so this file pins the same
contracts for USER rows -- e.g. an edited user message that keeps its
original alongside it as a sibling -- to confirm no assistant-only
assumption slipped in anywhere along the path.

Built via the same unmounted-``ChatScreen(app)`` pattern as
``Tests/Chat/test_console_sibling_nav.py``: the store and
``_select_console_message_variant`` only touch
``self._ensure_console_chat_store()``, never widget queries, so no Textual
mount is required.
"""

from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage, ConsoleMessageRole
from tldw_chatbook.Chat.console_message_actions import ConsoleMessageActionService
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_transcript import _message_render_text


def _build_screen():
    app = _build_test_app()
    screen = ChatScreen(app)
    return app, screen


def _build_two_user_siblings():
    """Build session: user1(u1) -> assistant(a1), then fork u1 into u2.

    Mirrors an edit-and-resend: ``u1`` keeps its own downstream reply ``a1``
    off to the side, while the new ``u2`` sibling becomes the active leaf --
    the same "anchor with its own descendants" shape Task 7's assistant
    tests exercise, just rooted at a USER node instead.
    """
    app, screen = _build_screen()
    store = screen._ensure_console_chat_store()
    session = store.ensure_session(title="t")
    u1 = store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    a1 = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="reply to hi"
    )
    u2 = store.create_sibling(u1.id, role=ConsoleMessageRole.USER, content="edited hi")
    return app, screen, store, session, u1, a1, u2


# --- Assertion 1: store.siblings_at reports two USER children -------------


def test_siblings_at_reports_two_user_siblings():
    _app, _screen, store, _session, u1, _a1, u2 = _build_two_user_siblings()

    siblings, index, count = store.siblings_at(u1.id)

    assert count == 2
    assert [s.id for s in siblings] == [u1.id, u2.id]
    assert index == 0
    assert all(s.role == ConsoleMessageRole.USER for s in siblings)

    # Resolves from the new sibling too (off-anchor lookup, same contract
    # ``siblings_at``'s docstring promises for off-path nodes).
    _siblings2, index2, count2 = store.siblings_at(u2.id)
    assert count2 == 2
    assert index2 == 1


# --- Assertion 2: available_actions offers variant-previous/-next on a ----
# --- USER message snapshot with sibling_count == 2, not gated on role ----


def test_available_actions_offers_variant_nav_for_user_message_with_siblings():
    _app, _screen, store, session, u1, _a1, u2 = _build_two_user_siblings()

    # u2 is the freshly retargeted active leaf, so its transient
    # sibling_index/sibling_count were just filled in by the store's single
    # active-path writer (_recompute_active_path) -- fetch the live snapshot
    # rather than constructing one by hand.
    active_user_message = store.get_message(u2.id)
    assert active_user_message.role == ConsoleMessageRole.USER
    assert active_user_message.sibling_count == 2
    assert active_user_message.sibling_index == 1

    service = ConsoleMessageActionService()
    actions = {
        action.action_id: action
        for action in service.available_actions(active_user_message)
    }

    assert "variant-previous" in actions
    assert "variant-next" in actions
    # index 1 of 2: previous is enabled (there is a sibling before it), next
    # is disabled (it's the last sibling) -- same enablement rule Task 7
    # pinned for assistant rows, unchanged by role.
    assert actions["variant-previous"].enabled is True
    assert actions["variant-next"].enabled is False
    assert actions["variant-next"].disabled_reason

    # Not gated on assistant-only: contrast with "regenerate", which IS
    # intentionally assistant-only (nothing to regenerate on a user turn) --
    # that stays blocked while variant nav stays offered.
    assert actions["regenerate"].enabled is False

    # session id sanity: both messages belong to the same session.
    assert store.session_id_for_message(u2.id) == session.id


def test_available_actions_also_offers_variant_nav_for_off_path_user_sibling():
    """The FIRST user sibling (off the active path after the fork) still
    reports sibling_count == 2 via ``siblings_at`` -- ``available_actions``
    should offer nav for it too when a snapshot carries that count, since
    the action row's gate is the message's own field, not "is this the
    active leaf" or "is this an assistant message"."""
    _app, _screen, store, _session, u1, _a1, _u2 = _build_two_user_siblings()

    off_path_siblings, off_path_index, off_path_count = store.siblings_at(u1.id)
    off_path_user_message = off_path_siblings[off_path_index]
    assert off_path_user_message.id == u1.id
    assert off_path_count == 2

    # siblings_at() snapshots don't carry the active-path-only sibling_count
    # field (that's filled by _recompute_active_path for on-path nodes
    # only) -- build the equivalent snapshot explicitly to isolate the
    # available_actions() contract from the store's active-path bookkeeping.
    snapshot_with_count = ConsoleChatMessage(
        role=off_path_user_message.role,
        content=off_path_user_message.content,
        id=off_path_user_message.id,
        sibling_index=off_path_index,
        sibling_count=off_path_count,
    )

    service = ConsoleMessageActionService()
    action_ids = [
        action.action_id
        for action in service.available_actions(snapshot_with_count)
    ]

    assert "variant-previous" in action_ids
    assert "variant-next" in action_ids


# --- Assertion 3: transcript counter renders (n/m) for a USER row --------


def test_transcript_counter_renders_for_user_row_with_siblings():
    """Mirrors ``test_sibling_counter_rendered_for_message_with_siblings``
    in ``Tests/UI/test_console_native_transcript.py`` (which pins this for
    an ASSISTANT row) -- same helper, USER role."""
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="edited hi",
        id="u2",
        sibling_index=1,
        sibling_count=2,
    )

    rendered = _message_render_text(message, selected=False)

    assert "(2/2)" in rendered.plain


def test_transcript_counter_renders_for_live_user_sibling_snapshot():
    """Same assertion, but sourced from the real store (not a hand-built
    snapshot) -- end-to-end from ``create_sibling`` through to the
    renderer."""
    _app, _screen, store, _session, _u1, _a1, u2 = _build_two_user_siblings()

    active_user_message = store.get_message(u2.id)
    rendered = _message_render_text(active_user_message, selected=False)

    assert "(2/2)" in rendered.plain


# --- Assertion 4: _select_console_message_variant moves the active leaf --
# --- across USER siblings -------------------------------------------------


def test_select_console_message_variant_moves_active_leaf_between_user_siblings():
    _app, screen, store, session, u1, _a1, u2 = _build_two_user_siblings()
    assert store.active_leaf(session.id) == u2.id

    screen._select_console_message_variant(u2.id, direction="variant-previous")

    # u1 has its own downstream reply (a1), so swiping back lands on u1's
    # most-recent descendant -- the same "resume mid-conversation" contract
    # Task 7 pins for assistant forks (test_sibling_nav_previous_restores_
    # deepest_descendant_of_sibling_subtree), now exercised on a user fork.
    assert store.active_leaf(session.id) == _a1_id(store, u1)

    screen._select_console_message_variant(u1.id, direction="variant-next")

    assert store.active_leaf(session.id) == u2.id


def _a1_id(store, u1):
    """Return the id of u1's own most-recent descendant."""
    return store._leaf_under(u1.id)


def test_select_console_message_variant_noop_at_either_end_for_user_siblings():
    _app, screen, store, session, u1, _a1, u2 = _build_two_user_siblings()

    # u1 is the FIRST user sibling (index 0) -- pressing "previous" on its
    # row must not move the active leaf (still u2, off u1's row).
    screen._select_console_message_variant(u1.id, direction="variant-previous")
    assert store.active_leaf(session.id) == u2.id

    # u2 is the LAST user sibling -- pressing "next" on it is a no-op too.
    screen._select_console_message_variant(u2.id, direction="variant-next")
    assert store.active_leaf(session.id) == u2.id
