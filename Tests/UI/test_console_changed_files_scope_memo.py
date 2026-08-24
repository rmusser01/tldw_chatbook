"""TASK-21121: the changed-files guard must not snapshot the session per tick.

`ChatScreen._console_changed_files_scope()` runs on every 0.2s Console run
tick. It used to call `ConsoleChatStore.messages_for_session()` and
reverse-scan the result -- but that call `dataclasses.replace`-copies EVERY
message in the session BEFORE the scan looks at one of them, so the early
break bought nothing and a long session paid a full O(messages) copy pass
five times a second. Worst case per the old docstring was "no marker
anywhere", which is the common case.

Two halves are tested here, and BOTH are load-bearing:

* the COST half -- a counter probe over
  `ConsoleChatStore._snapshot`/`messages_for_session` across simulated
  ticks during a streamed reply, asserted to be flat in the session size;
* the CORRECTNESS (control-arm) half -- the scope must still report the
  right run id when markers ARE present, across every way a marker can
  arrive or leave (live append, resume overlay, branch switch, delete,
  session switch, restore). Without this half, "fix" the cost by always
  returning `None` and the perf assertions all pass.

Deliberately a `ChatScreen.__new__` harness rather than the mounted
`ConsoleHarness` used by `test_console_changed_files_wiring.py`: the
subject is one pure derivation over the store, and a mounted app's own
per-tick `messages_for_session` traffic (transcript render, cost chip)
would drown the counter this probe exists to read.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

RUN_A = "run-aaaa"
RUN_B = "run-bbbb"
MARKER_TEXT = "✎ Edited 1 file  +1 −1 — review with `v`"
CONV_ID = "conv-21121"


def _build_screen(store: ConsoleChatStore, *, conversation_id=CONV_ID) -> ChatScreen:
    """A `ChatScreen` shell with only what `_console_changed_files_scope` reads.

    `_console_chat_store` is a property over the runtime handle
    (`_console_runtime()`), so the runtime ref is pre-seeded rather than the
    attribute -- setting the attribute would go through the setter and
    construct a real runtime.
    """
    screen = ChatScreen.__new__(ChatScreen)
    screen._console_runtime_ref = SimpleNamespace(chat_store=store)
    screen._character = SimpleNamespace(
        _current_console_rail_conversation_id=lambda: conversation_id
    )
    return screen


def _seed_session(store: ConsoleChatStore, *, messages: int) -> str:
    """Create a session and fill it with `messages` alternating chat turns."""
    session = store.create_session()
    for index in range(messages):
        store.append_message(
            session.id,
            role=(
                ConsoleMessageRole.USER
                if index % 2 == 0
                else ConsoleMessageRole.ASSISTANT
            ),
            content=f"turn {index}",
        )
    return session.id


class _StoreCallCounter:
    """Counts the two full-session passes this task is about.

    ARMED explicitly, because the store snapshots on its own account too --
    `append_message` and `append_stream_chunk` each return one. Counting
    those would bury the number under test (they are exactly the 26 that a
    first cut of this probe mistook for the guard's own cost).
    """

    def __init__(self, monkeypatch) -> None:
        self.snapshots = 0
        self.list_calls = 0
        self._armed = False
        original_snapshot = ConsoleChatStore._snapshot
        original_list = ConsoleChatStore.messages_for_session

        def counting_snapshot(message):
            if self._armed:
                self.snapshots += 1
            return original_snapshot(message)

        def counting_list(inner_self, session_id):
            if self._armed:
                self.list_calls += 1
            return original_list(inner_self, session_id)

        monkeypatch.setattr(
            ConsoleChatStore, "_snapshot", staticmethod(counting_snapshot)
        )
        monkeypatch.setattr(ConsoleChatStore, "messages_for_session", counting_list)

    def measure(self, call):
        """Run `call` with the counters live; return its result."""
        self._armed = True
        try:
            return call()
        finally:
            self._armed = False


def _tick_a_streamed_reply(
    screen: ChatScreen,
    store: ConsoleChatStore,
    session_id: str,
    counter: "_StoreCallCounter",
    *,
    ticks: int,
) -> list[tuple[str | None, str | None]]:
    """Simulate `ticks` run ticks while an assistant reply streams in.

    One chunk per tick, exactly as the live 0.2s poll observes a stream:
    the store's message OBJECTS mutate in place while the view list stays
    the same object, which is precisely the case the memo has to serve.
    Only the guard call itself is measured.
    """
    reply = store.append_message(
        session_id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    scopes = []
    for index in range(ticks):
        store.append_stream_chunk(reply.id, f"token-{index} ")
        scopes.append(counter.measure(screen._console_changed_files_scope))
    return scopes


#
# -- Cost half: the per-tick pass -------------------------------------------
#


@pytest.mark.parametrize("session_size", [40, 400])
@pytest.mark.parametrize("marker_run_id", [None, RUN_A])
def test_streamed_reply_ticks_never_snapshot_the_session(
    monkeypatch, session_size, marker_run_id
):
    """The probe: N ticks over a streaming reply cost ZERO snapshot passes.

    Before TASK-21121 this was `ticks * (session_size + 1)` snapshot copies
    plus one `messages_for_session` per tick. Parametrized over two session
    sizes so the assertion states the SHAPE (flat in session size) rather
    than a number that happens to hold for one fixture -- and over marker
    presence, so "make it free by always answering `None`" fails the COST
    half too, not only the control arm below.
    """
    store = ConsoleChatStore()
    session_id = _seed_session(store, messages=session_size)
    if marker_run_id is not None:
        store.append_message(
            session_id,
            role=ConsoleMessageRole.TOOL,
            content=MARKER_TEXT,
            change_review_run_id=marker_run_id,
        )
    screen = _build_screen(store)

    # Warm the guard exactly as the first live tick would, then count.
    screen._console_changed_files_scope()
    counter = _StoreCallCounter(monkeypatch)
    scopes = _tick_a_streamed_reply(screen, store, session_id, counter, ticks=25)

    assert counter.list_calls == 0, (
        "the changed-files guard must not re-materialize the whole session "
        f"per tick -- got {counter.list_calls} messages_for_session call(s)"
    )
    assert counter.snapshots == 0, (
        "a run tick's changed-files guard must copy no messages at all -- "
        f"got {counter.snapshots} snapshot(s) for a {session_size}-message session"
    )
    # ...and it still answered the RIGHT thing, identically, every tick.
    assert scopes == [(CONV_ID, marker_run_id)] * 25


def test_appending_the_session_costs_one_rescan_not_one_per_tick(monkeypatch):
    """A view change re-derives ONCE; the ticks after it are free again."""
    store = ConsoleChatStore()
    session_id = _seed_session(store, messages=60)
    screen = _build_screen(store)
    screen._console_changed_files_scope()

    counter = _StoreCallCounter(monkeypatch)
    store.append_message(session_id, role=ConsoleMessageRole.USER, content="next")
    for _ in range(10):
        counter.measure(screen._console_changed_files_scope)

    assert counter.snapshots == 0
    assert counter.list_calls == 0


#
# -- Control arm: the answer is still right ---------------------------------
#


def test_scope_reports_the_newest_marker_run_id():
    """Two markers: the guard names the LAST one on the path, not the first."""
    store = ConsoleChatStore()
    session_id = _seed_session(store, messages=6)
    screen = _build_screen(store)

    assert screen._console_changed_files_scope() == (CONV_ID, None)

    store.append_message(
        session_id,
        role=ConsoleMessageRole.TOOL,
        content=MARKER_TEXT,
        change_review_run_id=RUN_A,
    )
    assert screen._console_changed_files_scope() == (CONV_ID, RUN_A)

    store.append_message(session_id, role=ConsoleMessageRole.USER, content="more")
    store.append_message(
        session_id,
        role=ConsoleMessageRole.TOOL,
        content=MARKER_TEXT,
        change_review_run_id=RUN_B,
    )
    assert screen._console_changed_files_scope() == (CONV_ID, RUN_B)


def test_scope_survives_an_ordinary_message_appended_after_the_marker():
    """`_recompute_active_path` re-splices markers -- the guard must follow.

    The rebuild installs a brand-new view list, so this is also the test
    that the memo re-derives on list REPLACEMENT rather than serving the
    pre-rebuild answer by identity accident.
    """
    store = ConsoleChatStore()
    session_id = _seed_session(store, messages=4)
    screen = _build_screen(store)
    store.append_message(
        session_id,
        role=ConsoleMessageRole.TOOL,
        content=MARKER_TEXT,
        change_review_run_id=RUN_A,
    )
    screen._console_changed_files_scope()

    store.append_message(
        session_id, role=ConsoleMessageRole.ASSISTANT, content="after the marker"
    )

    assert screen._console_changed_files_scope() == (CONV_ID, RUN_A)


def test_scope_drops_a_marker_whose_branch_was_switched_away():
    """A marker anchored off the active path is gone -- so is its run id."""
    store = ConsoleChatStore()
    session = store.create_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="ask")
    answer = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="first answer"
    )
    store.append_message(
        session.id,
        role=ConsoleMessageRole.TOOL,
        content=MARKER_TEXT,
        change_review_run_id=RUN_A,
    )
    screen = _build_screen(store)
    assert screen._console_changed_files_scope() == (CONV_ID, RUN_A)

    sibling = store.create_sibling(
        answer.id, role=ConsoleMessageRole.ASSISTANT, content="regenerated"
    )
    store.set_active_leaf(session.id, sibling.id)

    assert screen._console_changed_files_scope() == (CONV_ID, None)

    store.set_active_leaf(session.id, answer.id)
    assert screen._console_changed_files_scope() == (CONV_ID, RUN_A)


def test_scope_re_derives_when_a_same_length_branch_replaces_the_view():
    """The identity half of the memo's signature, isolated.

    A branch switch that lands on a path the SAME LENGTH as the one it
    replaced changes the answer without changing `len(view)`. Caught only
    because the memo also checks that the view is the same list OBJECT --
    dropping that check leaves every other test in this file green
    (verified by mutation), which is exactly why this one exists.
    """
    store = ConsoleChatStore()
    session = store.create_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="ask")
    answer = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="first answer"
    )
    store.append_message(
        session.id,
        role=ConsoleMessageRole.TOOL,
        content=MARKER_TEXT,
        change_review_run_id=RUN_A,
    )
    screen = _build_screen(store)
    assert screen._console_changed_files_scope() == (CONV_ID, RUN_A)
    marked_view_length = len(store.messages_for_session(session.id))

    # Regenerate onto a sibling branch, then extend it back to the same
    # length -- no marker anywhere on the new path.
    sibling = store.create_sibling(
        answer.id, role=ConsoleMessageRole.ASSISTANT, content="regenerated"
    )
    store.set_active_leaf(session.id, sibling.id)
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="follow up")
    assert len(store.messages_for_session(session.id)) == marked_view_length

    assert screen._console_changed_files_scope() == (CONV_ID, None)


def test_scope_sees_resume_derived_markers_from_the_overlay():
    """Resume writes the view DIRECTLY (`apply_resume_marker_overlay`).

    That path bypasses `append_message` entirely, so an
    invalidate-on-append design would miss it. Verification does not.
    """
    from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage

    store = ConsoleChatStore()
    session = store.create_session()
    user = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="resume me"
    )
    screen = _build_screen(store)
    assert screen._console_changed_files_scope() == (CONV_ID, None)

    store.apply_resume_marker_overlay(
        session.id,
        [
            user,
            ConsoleChatMessage(
                role=ConsoleMessageRole.TOOL,
                content=MARKER_TEXT,
                status="complete",
                change_review_run_id=RUN_B,
            ),
        ],
    )

    assert screen._console_changed_files_scope() == (CONV_ID, RUN_B)


def test_scope_follows_a_deleted_marker_anchor():
    """Deleting the anchored turn purges the marker; the guard notices."""
    store = ConsoleChatStore()
    session = store.create_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="ask")
    answer = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="answer"
    )
    store.append_message(
        session.id,
        role=ConsoleMessageRole.TOOL,
        content=MARKER_TEXT,
        change_review_run_id=RUN_A,
    )
    screen = _build_screen(store)
    assert screen._console_changed_files_scope() == (CONV_ID, RUN_A)

    store.delete_message(answer.id)

    assert screen._console_changed_files_scope() == (CONV_ID, None)


#
# -- Store-level: memo lifetime, teardown, cross-session --------------------
#


def test_memo_never_bleeds_between_two_sessions():
    """The single memo slot is keyed by session -- an alternating read is exact."""
    store = ConsoleChatStore()
    marked = store.create_session()
    plain = store.create_session()
    store.append_message(
        marked.id,
        role=ConsoleMessageRole.TOOL,
        content=MARKER_TEXT,
        change_review_run_id=RUN_A,
    )
    store.append_message(plain.id, role=ConsoleMessageRole.USER, content="nothing here")

    for _ in range(3):
        assert store.newest_change_review_run_id(marked.id) == RUN_A
        assert store.newest_change_review_run_id(plain.id) is None


def test_unknown_session_raises_keyerror_like_messages_for_session():
    """Same contract as the call it replaces -- the caller catches KeyError."""
    store = ConsoleChatStore()
    with pytest.raises(KeyError):
        store.newest_change_review_run_id("no-such-session")


def test_closing_the_memoized_session_does_not_strand_a_stale_answer():
    """Teardown: a closed session's slot can never be served to a live one."""
    store = ConsoleChatStore()
    closing = store.create_session()
    survivor = store.create_session()
    store.append_message(
        closing.id,
        role=ConsoleMessageRole.TOOL,
        content=MARKER_TEXT,
        change_review_run_id=RUN_A,
    )
    assert store.newest_change_review_run_id(closing.id) == RUN_A

    store.close_session(closing.id)

    with pytest.raises(KeyError):
        store.newest_change_review_run_id(closing.id)
    assert store.newest_change_review_run_id(survivor.id) is None


def test_restore_state_is_not_served_the_pre_restore_answer():
    """`restore_state` replaces the view wholesale without any memo hook.

    The marker is anchored to a real turn here, deliberately. An
    anchor-`None` marker (one appended before the session had any node)
    SURVIVES a restore into the rebuilt view, because `restore_state`
    clears `_messages_by_session` but not `_tool_markers_by_session` and
    `_with_tool_markers` leads every rebuilt view with the anchor-`None`
    ones. That is a pre-existing store defect, reproduced identically by
    the pre-TASK-21121 reverse scan (verified against base `fb0a9601e`)
    and filed as TASK-21311 -- not something this memo may paper over,
    and not something it introduces.
    """
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession

    store = ConsoleChatStore()
    session = store.create_session(session_id="restored-1")
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="ask")
    store.append_message(
        session.id,
        role=ConsoleMessageRole.TOOL,
        content=MARKER_TEXT,
        change_review_run_id=RUN_A,
    )
    assert store.newest_change_review_run_id(session.id) == RUN_A

    store.restore_state(
        sessions=[ConsoleChatSession(id="restored-1", title="Restored")],
        messages_by_session={},
        active_session_id="restored-1",
    )

    assert store.newest_change_review_run_id("restored-1") is None


def test_screen_scope_tolerates_a_store_without_the_active_session():
    """The guard's own KeyError path still degrades to `(conversation, None)`."""
    store = ConsoleChatStore()
    session = store.create_session()
    screen = _build_screen(store)
    store._sessions.pop(session.id)

    assert screen._console_changed_files_scope() == (CONV_ID, None)
