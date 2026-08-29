"""Console resume reconstructs the active branch from the stored leaf pointer.

Task 8 (Phase A conversation branching): resuming a saved conversation must
load the WHOLE persisted tree (every branch, on- and off-path) and derive the
visible transcript from the stored ``active_leaf_message_id`` pointer -- not
from a ``children[-1]`` latest-branch walk. Loading all branches is what makes
off-path siblings navigable (swipe) immediately after resume.

Real DB round-trips: a real ``CharactersRAGDB`` behind the real
``ChatConversationService``/``ChatPersistenceService`` and the real ChatScreen
full-tree flatten -- no hand-rolled fakes for the pieces under test.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_agent_bridge import (
    ConsoleAgentBridge,
    inject_resume_agent_markers,
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession, ConsoleChatStore
from tldw_chatbook.Chat.console_cost_tracker import console_cost_snapshot_messages
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_turn_grouping import (
    group_console_transcript_messages,
    visual_messages,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Console_Modules.review_selection import (
    _build_trajectory_snapshot,
)
from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController
from tldw_chatbook.UI.Screens.chat_screen import (
    ChatScreen,
)

from Tests.UI.test_destination_shells import _build_test_app


def _append_local_command_run(
    db: AgentRunsDB,
    *,
    conversation_id: str,
    anchor: str | None,
    command: str = "printf LOCAL_COMMAND_SECRET",
    output: str = "[stdout]\nLOCAL_OUTPUT_SECRET\n",
) -> str:
    run_id = db.create_run(
        conversation_id=conversation_id,
        agent_kind="local_command",
        task="Local command",
        assistant_message_id=anchor,
    )
    db.append_steps(
        run_id,
        [
            {
                "index": 0,
                "kind": "tool_call",
                "summary": "Local command started",
                "tool_name": "raw_cli",
                "args": {
                    "command": command,
                    "shell": "/bin/zsh",
                    "cwd": "/private/tmp",
                    "invocation_id": "local-invocation",
                },
            },
            {
                "index": 1,
                "kind": "tool_result",
                "summary": "Local command completed",
                "tool_name": "raw_cli",
                "result": output,
                "args": {
                    "invocation_id": "local-invocation",
                    "shell": "/bin/zsh",
                    "cwd": "/private/tmp",
                    "stdout_preview": "LOCAL_OUTPUT_SECRET\n",
                    "stderr_preview": "",
                    "elapsed_seconds": 0.25,
                    "exit_code": 7,
                    "terminal_state": "exited",
                    "truncated": False,
                    "cleanup_proven": True,
                },
                "status": "done",
                "tool_outcome": "failed",
            },
        ],
    )
    db.set_status(run_id, "done")
    return run_id


def _persist_branched_conversation(db: CharactersRAGDB):
    """Persist ``U1 -> {A1 (older), A1' (newer)}`` and return the ids.

    Explicit, strictly increasing timestamps pin the sibling order the tree
    service reads (``ORDER BY timestamp ASC``), so ``A1`` is unambiguously the
    older sibling and ``A1'`` the ``children[-1]`` most-recent one.
    """
    service = ChatConversationService(db)
    conversation_id = service.create_conversation(
        id="branch-conv-1",
        title="Branchy",
        scope_type="global",
        state="in-progress",
    )
    u1 = db.add_message(
        {
            "id": "m-u1",
            "conversation_id": conversation_id,
            "sender": "user",
            "role": "user",
            "content": "u1",
            "timestamp": "2026-01-01T00:00:00.000000+00:00",
        }
    )
    a1 = db.add_message(
        {
            "id": "m-a1",
            "conversation_id": conversation_id,
            "parent_message_id": u1,
            "sender": "assistant",
            "role": "assistant",
            "content": "a1",
            "timestamp": "2026-01-01T00:00:01.000000+00:00",
        }
    )
    a1_prime = db.add_message(
        {
            "id": "m-a1-prime",
            "conversation_id": conversation_id,
            "parent_message_id": u1,
            "sender": "assistant",
            "role": "assistant",
            "content": "a1-prime",
            "timestamp": "2026-01-01T00:00:02.000000+00:00",
        }
    )
    return conversation_id, u1, a1, a1_prime


def _persist_flat_legacy_conversation(db: CharactersRAGDB):
    """Persist a legacy FLAT conversation: every message a NULL-parent root.

    Mimics pre-branching persistence, where the base ``_persist_new_message``
    hardcoded ``parent_message_id=None`` for every message. The four rows
    therefore load on resume as four separate roots (all siblings under
    ``None``), not one linear thread. Strictly increasing timestamps pin the
    DB's ``ORDER BY timestamp ASC`` root order.
    """
    service = ChatConversationService(db)
    conversation_id = service.create_conversation(
        id="flat-conv-1",
        title="Flat",
        scope_type="global",
        state="in-progress",
    )
    for i, (content, sender) in enumerate(
        [("u1", "user"), ("a1", "assistant"), ("u2", "user"), ("a2", "assistant")]
    ):
        db.add_message(
            {
                "id": f"m-flat-{i}",
                "conversation_id": conversation_id,
                "sender": sender,
                "role": sender,
                "content": content,
                "timestamp": f"2026-01-01T00:00:0{i}.000000+00:00",
            }
        )
    return conversation_id


def _persist_mixed_legacy_then_branched_conversation(db: CharactersRAGDB):
    """Flat legacy prefix ``[u1,a1,u2,a2]`` (NULL parents) then a post-feature
    continuation ``u3 -> a3`` genuinely parented onto ``a2``.

    Reproduces an old conversation that gained new messages after branching
    landed: the prefix is four roots, the continuation is a real subtree
    hanging off the last flat row.
    """
    service = ChatConversationService(db)
    conversation_id = service.create_conversation(
        id="mixed-conv-1",
        title="Mixed",
        scope_type="global",
        state="in-progress",
    )
    ids = []
    for i, (content, sender) in enumerate(
        [("u1", "user"), ("a1", "assistant"), ("u2", "user"), ("a2", "assistant")]
    ):
        ids.append(
            db.add_message(
                {
                    "id": f"m-mixed-{i}",
                    "conversation_id": conversation_id,
                    "sender": sender,
                    "role": sender,
                    "content": content,
                    "timestamp": f"2026-01-01T00:00:0{i}.000000+00:00",
                }
            )
        )
    a2_id = ids[-1]
    u3 = db.add_message(
        {
            "id": "m-mixed-u3",
            "conversation_id": conversation_id,
            "parent_message_id": a2_id,
            "sender": "user",
            "role": "user",
            "content": "u3",
            "timestamp": "2026-01-01T00:00:05.000000+00:00",
        }
    )
    db.add_message(
        {
            "id": "m-mixed-a3",
            "conversation_id": conversation_id,
            "parent_message_id": u3,
            "sender": "assistant",
            "role": "assistant",
            "content": "a3",
            "timestamp": "2026-01-01T00:00:06.000000+00:00",
        }
    )
    return conversation_id


def _persist_genuine_multi_root_conversation(db: CharactersRAGDB):
    """Persist a genuine root-level fork: two independent USER-rooted threads.

    Mirrors editing-and-resending the conversation's very FIRST user message
    (Phase B ``edit_and_resend_message``): ``create_sibling`` parents the fork
    at the anchor's own parent, which is ``None`` for a root message, so the
    edit becomes a SECOND root-level USER sibling with its own subtree. Both
    roots are USER (role-homogeneous), the signal
    ``ConsoleChatStore._chain_legacy_flat_roots`` uses to distinguish a
    genuine multi-root fork (left un-chained, both roots independently
    navigable) from legacy flat data (mixed USER/ASSISTANT roots, chained
    into one spine) -- see that method's docstring.
    """
    service = ChatConversationService(db)
    conversation_id = service.create_conversation(
        id="multi-root-conv-1",
        title="Multi-root",
        scope_type="global",
        state="in-progress",
    )
    u1 = db.add_message(
        {
            "id": "m-u1",
            "conversation_id": conversation_id,
            "sender": "user",
            "role": "user",
            "content": "u1",
            "timestamp": "2026-01-01T00:00:00.000000+00:00",
        }
    )
    a1 = db.add_message(
        {
            "id": "m-a1",
            "conversation_id": conversation_id,
            "parent_message_id": u1,
            "sender": "assistant",
            "role": "assistant",
            "content": "a1",
            "timestamp": "2026-01-01T00:00:01.000000+00:00",
        }
    )
    u1_prime = db.add_message(
        {
            "id": "m-u1-prime",
            "conversation_id": conversation_id,
            "sender": "user",
            "role": "user",
            "content": "u1-prime",
            "timestamp": "2026-01-01T00:00:02.000000+00:00",
        }
    )
    a1_prime = db.add_message(
        {
            "id": "m-a1-prime",
            "conversation_id": conversation_id,
            "parent_message_id": u1_prime,
            "sender": "assistant",
            "role": "assistant",
            "content": "a1-prime",
            "timestamp": "2026-01-01T00:00:03.000000+00:00",
        }
    )
    return conversation_id, u1, a1, u1_prime, a1_prime


def _resume_into_store(db: CharactersRAGDB, conversation_id: str):
    """Mirror the production resume plumbing end to end.

    Full-tree flatten via the REAL ChatScreen helper + the stored cursor pair,
    fed into ``restore_persisted_session`` exactly as
    ``_resume_console_workspace_conversation`` does.
    """
    service = ChatConversationService(db)
    tree = service.get_conversation_tree(
        conversation_id, depth_cap=10_000, root_limit=10_000
    )
    screen = ChatScreen(_build_test_app())
    screen.app_instance.chachanotes_db = db
    all_nodes = screen._console_messages_from_conversation_tree(tree)
    active_leaf_id, before_message_id = db.get_conversation_active_cursor(
        conversation_id
    )
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.restore_persisted_session(
        title="Branchy",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=all_nodes,
        active_leaf_persisted_id=active_leaf_id,
        active_leaf_before_persisted_id=before_message_id,
    )
    return store, session


def test_console_messages_from_conversation_tree_flattens_all_branches():
    """The flatten returns EVERY node (both siblings), each carrying its
    persisted id and persisted parent id -- not just the latest branch."""
    screen = ChatScreen(_build_test_app())
    tree = {
        "conversation": {"title": "Saved"},
        "root_threads": [
            {
                "id": "u1",
                "sender": "user",
                "content": "u1",
                "parent_message_id": None,
                "children": [
                    {
                        "id": "a1",
                        "sender": "assistant",
                        "content": "a1",
                        "parent_message_id": "u1",
                        "children": [],
                    },
                    {
                        "id": "a1-prime",
                        "sender": "assistant",
                        "content": "a1-prime",
                        "parent_message_id": "u1",
                        "children": [],
                    },
                ],
            }
        ],
    }

    messages = screen._console_messages_from_conversation_tree(tree)

    by_pid = {m.persisted_message_id: m for m in messages}
    assert set(by_pid) == {"u1", "a1", "a1-prime"}
    assert by_pid["u1"].parent_message_id is None
    assert by_pid["a1"].parent_message_id == "u1"
    assert by_pid["a1-prime"].parent_message_id == "u1"


def test_resume_reconstructs_older_branch_from_active_leaf():
    """Pointer at the OLDER sibling -> transcript is the older branch, not the
    ``children[-1]`` latest one the pre-Task-8 walk would have shown."""
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        conversation_id, _u1, a1, _a1_prime = _persist_branched_conversation(db)
        db.set_conversation_active_leaf(conversation_id, a1)

        store, session = _resume_into_store(db, conversation_id)

        view = [m.content for m in store.messages_for_session(session.id)]
        assert view == ["u1", "a1"]
    finally:
        db.close_connection()


def test_resume_loads_off_path_siblings_for_swipe():
    """After resuming onto the older branch, the off-path sibling is loaded and
    navigable: ``siblings_at`` reports 2 and ``set_active_leaf`` swaps the view."""
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        conversation_id, _u1, a1, _a1_prime = _persist_branched_conversation(db)
        db.set_conversation_active_leaf(conversation_id, a1)

        store, session = _resume_into_store(db, conversation_id)

        restored_a1 = store.messages_for_session(session.id)[-1]
        assert restored_a1.content == "a1"

        snapshots, index, count = store.siblings_at(restored_a1.id)
        assert count == 2
        assert index == 0
        other = next(s for s in snapshots if s.id != restored_a1.id)
        assert other.content == "a1-prime"

        store.set_active_leaf(session.id, other.id)
        view = [m.content for m in store.messages_for_session(session.id)]
        assert view == ["u1", "a1-prime"]
    finally:
        db.close_connection()


def test_resume_falls_back_to_recent_leaf_and_repairs_pointer_when_missing():
    """No stored pointer -> resume the most-recent ``children[-1]`` branch AND
    repair the durable pointer so the next resume is exact."""
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        conversation_id, _u1, _a1, a1_prime = _persist_branched_conversation(db)
        assert db.get_conversation_active_leaf(conversation_id) is None

        store, session = _resume_into_store(db, conversation_id)

        view = [m.content for m in store.messages_for_session(session.id)]
        assert view == ["u1", "a1-prime"]
        assert db.get_conversation_active_leaf(conversation_id) == a1_prime
    finally:
        db.close_connection()


def test_resume_chains_legacy_flat_roots_into_full_transcript():
    """C1 regression: legacy flat data (every message a NULL-parent root, no
    active-leaf pointer) resumes as the FULL transcript, not truncated to the
    last row, and every message reports a single sibling (no phantom counter).

    Before the fix the active-leaf fallback walked only the LAST root, so the
    transcript collapsed to ``['a2']`` and each row rendered a bogus ``4/4``.
    """
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        conversation_id = _persist_flat_legacy_conversation(db)
        assert db.get_conversation_active_leaf(conversation_id) is None

        store, session = _resume_into_store(db, conversation_id)

        view = store.messages_for_session(session.id)
        assert [m.content for m in view] == ["u1", "a1", "u2", "a2"]
        for message in view:
            _snapshots, _index, count = store.siblings_at(message.id)
            assert count == 1
    finally:
        db.close_connection()


def test_resume_chains_flat_prefix_then_preserves_real_continuation():
    """C1 mixed case: a flat legacy prefix followed by a genuinely-parented
    continuation resumes as the full linear transcript, real subtree intact."""
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        conversation_id = _persist_mixed_legacy_then_branched_conversation(db)

        store, session = _resume_into_store(db, conversation_id)

        view = [m.content for m in store.messages_for_session(session.id)]
        assert view == ["u1", "a1", "u2", "a2", "u3", "a3"]
    finally:
        db.close_connection()


def test_resume_falls_back_when_pointer_dangles():
    """A pointer at a message that no longer exists -> same most-recent-leaf
    fallback and pointer repair as a missing pointer."""
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        conversation_id, _u1, _a1, a1_prime = _persist_branched_conversation(db)
        db.set_conversation_active_leaf(conversation_id, "deleted-message-id")

        store, session = _resume_into_store(db, conversation_id)

        view = [m.content for m in store.messages_for_session(session.id)]
        assert view == ["u1", "a1-prime"]
        assert db.get_conversation_active_leaf(conversation_id) == a1_prime
    finally:
        db.close_connection()


def test_resume_valid_leaf_wins_over_marker_and_repairs_cursor_pair():
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        conversation_id, u1, a1, _a1_prime = _persist_branched_conversation(db)
        assert db.set_conversation_active_cursor(
            conversation_id,
            active_leaf_message_id=a1,
            before_message_id=u1,
        )

        store, session = _resume_into_store(db, conversation_id)

        assert [
            message.content for message in store.messages_for_session(session.id)
        ] == ["u1", "a1"]
        assert db.get_conversation_active_cursor(conversation_id) == (a1, None)
    finally:
        db.close_connection()


def test_resume_dangling_leaf_ignores_valid_marker_and_repairs_to_newest():
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        conversation_id, u1, _a1, a1_prime = _persist_branched_conversation(db)
        assert db.set_conversation_active_cursor(
            conversation_id,
            active_leaf_message_id="missing-leaf",
            before_message_id=u1,
        )

        store, session = _resume_into_store(db, conversation_id)

        assert [
            message.content for message in store.messages_for_session(session.id)
        ] == ["u1", "a1-prime"]
        assert store.session_draft(session.id) == ""
        assert db.get_conversation_active_cursor(conversation_id) == (
            a1_prime,
            None,
        )
    finally:
        db.close_connection()


def test_resume_invalid_marker_on_empty_tree_clears_cursor_pair():
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        conversation_id = ChatConversationService(db).create_conversation(
            id="empty-invalid-marker",
            title="Empty",
            scope_type="global",
            state="in-progress",
        )
        assert db.set_conversation_active_cursor(
            conversation_id,
            active_leaf_message_id=None,
            before_message_id="missing-root",
        )

        store, session = _resume_into_store(db, conversation_id)

        assert store.active_path_message_ids(session.id) == []
        assert store.session_draft(session.id) == ""
        assert db.get_conversation_active_cursor(conversation_id) == (None, None)
    finally:
        db.close_connection()


def test_resume_marker_reads_current_durable_prompt_content():
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        conversation_id, u1, _a1, _a1_prime = _persist_branched_conversation(db)
        assert db.set_conversation_active_cursor(
            conversation_id,
            active_leaf_message_id=None,
            before_message_id=u1,
        )
        row = db.get_message_by_id(u1)
        assert row is not None
        assert db.update_message(
            u1,
            {"content": "u1 changed in durable storage"},
            expected_version=row["version"],
            preserve_descendants=True,
        )

        store, session = _resume_into_store(db, conversation_id)

        assert store.active_path_message_ids(session.id) == []
        assert store.session_draft(session.id) == "u1 changed in durable storage"
        assert db.get_conversation_active_cursor(conversation_id) == (None, u1)
    finally:
        db.close_connection()


def test_legacy_flat_before_first_then_new_root_restart_preserves_all_rows():
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        conversation_id = _persist_flat_legacy_conversation(db)
        original_ids = {
            row["id"] for row in db.get_messages_for_conversation(conversation_id)
        }
        assert original_ids == {
            "m-flat-0",
            "m-flat-1",
            "m-flat-2",
            "m-flat-3",
        }

        store, session = _resume_into_store(db, conversation_id)
        first_prompt = store.messages_for_session(session.id)[0]
        assert first_prompt.persisted_message_id == "m-flat-0"
        assert store.set_active_path_before(session.id, first_prompt.id) is True
        assert db.get_conversation_active_cursor(conversation_id) == (
            None,
            "m-flat-0",
        )

        new_root_id = db.add_message(
            {
                "id": "m-flat-new-root",
                "conversation_id": conversation_id,
                "sender": "user",
                "role": "user",
                "content": "u1 edited",
                "timestamp": "2026-01-01T00:00:04.000000+00:00",
            }
        )
        db.set_conversation_active_leaf(conversation_id, new_root_id)

        _restarted_store, _restarted_session = _resume_into_store(
            db, conversation_id
        )
        durable_rows = db.get_messages_for_conversation(conversation_id)
        durable_ids = {row["id"] for row in durable_rows}
        assert durable_ids == original_ids | {new_root_id}
        assert len(durable_rows) == len(original_ids) + 1
        assert all(
            db.get_message_by_id(message_id) is not None
            for message_id in original_ids
        )
    finally:
        db.close_connection()


def test_resume_second_root_loads_off_path_but_is_not_shown():
    """Genuine multi-root resume: active leaf under the first root shows only
    that root's branch; the second root loads too (navigable via
    ``siblings_at``) but is not part of the visible transcript.

    Unlike the legacy-flat-roots cases above (mixed USER/ASSISTANT roots,
    chained into one spine), two role-homogeneous all-USER roots are the
    genuine Phase-B root-level fork shape and are correctly left un-chained.
    """
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        conversation_id, _u1, a1, _u1_prime, _a1_prime = (
            _persist_genuine_multi_root_conversation(db)
        )
        db.set_conversation_active_leaf(conversation_id, a1)

        store, session = _resume_into_store(db, conversation_id)

        view = [m.content for m in store.messages_for_session(session.id)]
        assert view == ["u1", "a1"]

        root_message = store.messages_for_session(session.id)[0]
        assert root_message.persisted_message_id == "m-u1"
        snapshots, index, count = store.siblings_at(root_message.id)
        assert count == 2
        assert index == 0
        other_root = next(s for s in snapshots if s.id != root_message.id)
        assert other_root.content == "u1-prime"
        assert other_root.persisted_message_id == "m-u1-prime"
    finally:
        db.close_connection()


def test_resume_clears_stale_persisted_summary_with_dangling_boundary():
    """TASK-550: a persisted `/rewind` summary whose boundary id maps to no
    message on the just-loaded tree (e.g. the boundary message's branch was
    hard-deleted, or a foreign client rewrote history) is permanently
    orphaned. Resume leaves the in-memory summary unset (fail-open, as
    before Task-550) AND best-effort clears the stale persisted pair so it
    doesn't linger in the DB row indefinitely.
    """
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        conversation_id, _u1, _a1, _a1_prime = _persist_branched_conversation(db)
        db.set_conversation_context_summary(
            conversation_id, "stale recap", "deleted-boundary-id"
        )

        store, session = _resume_into_store(db, conversation_id)

        assert store.session_context_summary(session.id) == (None, None)
        assert db.get_conversation_context_summary(conversation_id) == (None, None)
    finally:
        db.close_connection()


def test_resume_leaves_valid_persisted_summary_boundary_untouched():
    """A persisted summary whose boundary DOES resolve on the loaded tree
    restores unchanged, and the DB row is left exactly as persisted (only
    the dangling case in the test above clears it)."""
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        conversation_id, u1, _a1, _a1_prime = _persist_branched_conversation(db)
        db.set_conversation_context_summary(conversation_id, "earlier recap", u1)

        store, session = _resume_into_store(db, conversation_id)

        summary, boundary_native_id = store.session_context_summary(session.id)
        assert summary == "earlier recap"
        assert boundary_native_id is not None
        assert db.get_conversation_context_summary(conversation_id) == (
            "earlier recap",
            u1,
        )
    finally:
        db.close_connection()


def _persist_degenerate_all_user_legacy_conversation(db: CharactersRAGDB):
    """Legacy FLAT conversation whose user turns got NO assistant reply.

    Reachable in the flat era via repeated failed/blocked sends: every row is
    a NULL-parent USER root with no children. task-572's target shape -- the
    role-homogeneity signal alone mistakes it for a genuine Phase-B root fork
    and leaves it un-chained (phantom counter, truncated view).
    """
    service = ChatConversationService(db)
    conversation_id = service.create_conversation(
        id="degenerate-flat-conv-1",
        title="Degenerate",
        scope_type="global",
        state="in-progress",
    )
    for i, content in enumerate(["u1", "u2", "u3"]):
        db.add_message(
            {
                "id": f"m-degen-{i}",
                "conversation_id": conversation_id,
                "sender": "user",
                "role": "user",
                "content": content,
                "timestamp": f"2026-01-01T00:00:0{i}.000000+00:00",
            }
        )
    return conversation_id


def test_resume_chains_degenerate_all_user_legacy_conversation():
    """task-572: a degenerate all-USER legacy conversation (multiple
    parentless user rows, no replies anywhere) resumes as the full ordered
    sequence with no phantom sibling counter.

    The stronger fingerprint: an all-USER root set whose roots are ALL
    childless is degenerate legacy (a genuine first-message edit-&-resend
    fork always carries at least one reply subtree under a root) and is
    chained; an all-USER root set with any subtree stays un-chained (pinned
    by ``test_resume_second_root_loads_off_path_but_is_not_shown``).
    """
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        conversation_id = _persist_degenerate_all_user_legacy_conversation(db)
        assert db.get_conversation_active_leaf(conversation_id) is None

        store, session = _resume_into_store(db, conversation_id)

        view = store.messages_for_session(session.id)
        assert [m.content for m in view] == ["u1", "u2", "u3"]
        for message in view:
            _snapshots, _index, count = store.siblings_at(message.id)
            assert count == 1
    finally:
        db.close_connection()


def test_resume_restores_usage_from_usage_json():
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        service = ChatConversationService(db)
        conversation_id = service.create_conversation(
            id="usage-conv-1",
            title="Usage",
            scope_type="global",
            state="in-progress",
        )
        u1 = db.add_message(
            {
                "id": "m-usage-u1",
                "conversation_id": conversation_id,
                "sender": "user",
                "role": "user",
                "content": "u1",
                "timestamp": "2026-01-01T00:00:00.000000+00:00",
            }
        )
        db.add_message(
            {
                "id": "m-usage-a1",
                "conversation_id": conversation_id,
                "parent_message_id": u1,
                "sender": "assistant",
                "role": "assistant",
                "content": "a1",
                "timestamp": "2026-01-01T00:00:01.000000+00:00",
                "usage_json": (
                    '{"uncached_input": 10, "cache_read": 0, "cache_write": 0,'
                    ' "output": 5, "provider": "openai", "model": "gpt-4o",'
                    ' "partial": false}'
                ),
            }
        )

        store, session = _resume_into_store(db, conversation_id)

        assistant = store.messages_for_session(session.id)[-1]
        assert assistant.content == "a1"
        assert assistant.usage is not None
        assert assistant.usage.uncached_input == 10
        assert assistant.usage.output == 5
        assert assistant.usage.provider == "openai"
    finally:
        db.close_connection()


def test_resume_tolerates_null_and_garbage_usage_json():
    # Legacy rows (NULL) and corrupt JSON must load with usage=None, never raise.
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        service = ChatConversationService(db)
        conversation_id = service.create_conversation(
            id="usage-conv-2",
            title="UsageLegacy",
            scope_type="global",
            state="in-progress",
        )
        u1 = db.add_message(
            {
                "id": "m-legacy-u1",
                "conversation_id": conversation_id,
                "sender": "user",
                "role": "user",
                "content": "u1",
                "timestamp": "2026-01-01T00:00:00.000000+00:00",
            }
        )
        db.add_message(
            {
                "id": "m-legacy-a1",
                "conversation_id": conversation_id,
                "parent_message_id": u1,
                "sender": "assistant",
                "role": "assistant",
                "content": "a1",
                "timestamp": "2026-01-01T00:00:01.000000+00:00",
                "usage_json": "{broken",
            }
        )

        store, session = _resume_into_store(db, conversation_id)

        assert all(m.usage is None for m in store.messages_for_session(session.id))
    finally:
        db.close_connection()


def test_screen_state_round_trip_preserves_usage():
    """F6: navigating away and back serializes the transcript to a JSON-safe
    snapshot. That snapshot dropped `usage`, so a session that had already
    recorded real spend came back reading $0 / "no usage" until the
    conversation was reloaded from the DB (and never, for unsaved sessions).
    """
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleChatMessage,
        ConsoleMessageRole,
    )
    from tldw_chatbook.Chat.provider_usage import ProviderUsage

    usage = ProviderUsage(
        uncached_input=904,
        cache_read=4096,
        cache_write=128,
        output=42,
        provider="anthropic",
        model="claude-sonnet-5",
        partial=True,
    )
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="answer",
        status="complete",
        usage=usage,
    )

    payload = ChatScreen._serialize_console_message(message)
    assert payload["usage_json"] is not None

    restored = ChatScreen._restore_console_message(payload)

    assert restored is not None
    assert restored.usage == usage


def test_screen_state_round_trip_tolerates_missing_and_broken_usage():
    """Legacy snapshots have no `usage_json` key at all; a corrupt one must
    degrade to "no usage known", never raise mid-restore."""
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleChatMessage,
        ConsoleMessageRole,
    )

    without_usage = ChatScreen._serialize_console_message(
        ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="a")
    )
    assert without_usage["usage_json"] is None
    assert ChatScreen._restore_console_message(without_usage).usage is None

    legacy = {"role": "assistant", "content": "a", "status": "complete"}
    assert ChatScreen._restore_console_message(legacy).usage is None

    corrupt = {**legacy, "usage_json": "{not json"}
    assert ChatScreen._restore_console_message(corrupt).usage is None


def test_resume_restores_metadata_from_metadata_json():
    """task-2364: without the read-back a resumed conversation loses every
    structured fact -- and the reseed builder, which now reads the
    interrupted FLAG, would replay the visible marker into the model."""
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        service = ChatConversationService(db)
        conversation_id = service.create_conversation(
            id="meta-conv-1",
            title="Metadata",
            scope_type="global",
            state="in-progress",
        )
        u1 = db.add_message(
            {
                "id": "m-meta-u1",
                "conversation_id": conversation_id,
                "sender": "user",
                "role": "user",
                "content": "u1",
                "timestamp": "2026-01-01T00:00:00.000000+00:00",
                "metadata_json": (
                    '{"engine": "realtime", "provider": "openai",'
                    ' "model": "gpt-4o-transcribe", "interrupted": false,'
                    ' "transcript_status": "final"}'
                ),
            }
        )
        db.add_message(
            {
                "id": "m-meta-a1",
                "conversation_id": conversation_id,
                "parent_message_id": u1,
                "sender": "assistant",
                "role": "assistant",
                "content": "a1 ⏹ interrupted",
                "timestamp": "2026-01-01T00:00:01.000000+00:00",
                "metadata_json": (
                    '{"engine": "realtime", "provider": "openai",'
                    ' "model": "gpt-realtime", "interrupted": true,'
                    ' "transcript_status": ""}'
                ),
            }
        )

        store, session = _resume_into_store(db, conversation_id)

        user, assistant = store.messages_for_session(session.id)
        assert user.metadata is not None
        assert user.metadata.transcript_status == "final"
        assert user.metadata.model == "gpt-4o-transcribe"
        assert assistant.metadata is not None
        assert assistant.metadata.interrupted is True
        assert assistant.metadata.engine == "realtime"
    finally:
        db.close_connection()


def test_resume_restores_an_empty_transcript_row_and_its_explanation():
    """task-2391: a committed voice turn whose transcript came back with no
    words must still be there -- and still explained -- after a restart.
    Unlike the `pending`/`final` case above, this row was never real user
    words in the first place; its content is the placeholder task-2391
    writes so the row can be durably created at all (the DB layer refuses a
    message with neither text nor an image, so a metadata-only "empty"
    record could never survive to be resumed)."""
    from tldw_chatbook.UI.Console_Modules.realtime import (
        CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER,
    )

    db = CharactersRAGDB(":memory:", "test_client")
    try:
        service = ChatConversationService(db)
        conversation_id = service.create_conversation(
            id="meta-conv-2",
            title="Empty transcript",
            scope_type="global",
            state="in-progress",
        )
        db.add_message(
            {
                "id": "m-meta-empty",
                "conversation_id": conversation_id,
                "sender": "user",
                "role": "user",
                "content": CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER,
                "timestamp": "2026-01-01T00:00:00.000000+00:00",
                "metadata_json": (
                    '{"engine": "realtime", "provider": "openai",'
                    ' "model": "gpt-4o-transcribe", "interrupted": false,'
                    ' "transcript_status": "empty"}'
                ),
            }
        )

        store, session = _resume_into_store(db, conversation_id)

        (user,) = store.messages_for_session(session.id)
        assert user.content == CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER
        assert user.metadata is not None
        assert user.metadata.transcript_status == "empty"
    finally:
        db.close_connection()


def test_resume_tolerates_null_and_garbage_metadata_json():
    # Legacy rows (NULL) and corrupt JSON load with metadata=None, never raise.
    db = CharactersRAGDB(":memory:", "test_client")
    try:
        service = ChatConversationService(db)
        conversation_id = service.create_conversation(
            id="meta-conv-2",
            title="MetadataLegacy",
            scope_type="global",
            state="in-progress",
        )
        u1 = db.add_message(
            {
                "id": "m-meta-legacy-u1",
                "conversation_id": conversation_id,
                "sender": "user",
                "role": "user",
                "content": "u1",
                "timestamp": "2026-01-01T00:00:00.000000+00:00",
            }
        )
        db.add_message(
            {
                "id": "m-meta-legacy-a1",
                "conversation_id": conversation_id,
                "parent_message_id": u1,
                "sender": "assistant",
                "role": "assistant",
                "content": "a1",
                "timestamp": "2026-01-01T00:00:01.000000+00:00",
                "metadata_json": "{broken",
            }
        )

        store, session = _resume_into_store(db, conversation_id)

        assert all(m.metadata is None for m in store.messages_for_session(session.id))
    finally:
        db.close_connection()


def test_screen_state_round_trip_preserves_metadata():
    """Navigating away and back must not silently drop the interrupted flag
    -- the same class of loss F6 fixed for usage."""
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleChatMessage,
        ConsoleMessageRole,
    )
    from tldw_chatbook.Chat.message_metadata import MessageMetadata

    metadata = MessageMetadata(
        engine="realtime",
        provider="openai",
        model="gpt-realtime",
        interrupted=True,
    )
    payload = ChatScreen._serialize_console_message(
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT,
            content="answer ⏹ interrupted",
            status="complete",
            metadata=metadata,
        )
    )
    assert payload["metadata_json"] is not None

    restored = ChatScreen._restore_console_message(payload)

    assert restored is not None
    assert restored.metadata == metadata


def test_screen_state_round_trip_tolerates_missing_and_broken_metadata():
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleChatMessage,
        ConsoleMessageRole,
    )

    without_metadata = ChatScreen._serialize_console_message(
        ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="a")
    )
    without_metadata.pop("metadata_json", None)
    assert ChatScreen._restore_console_message(without_metadata).metadata is None

    broken = ChatScreen._serialize_console_message(
        ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="a")
    )
    broken["metadata_json"] = "{nope"
    assert ChatScreen._restore_console_message(broken).metadata is None


def test_character_screen_state_round_trips_roleplay_identity_fields_exactly():
    """Omitting an explicit field silently drops it during screen navigation."""
    original = ConsoleChatSession(
        settings=ConsoleSessionSettings(
            provider="openai",
            model="gpt-4.1",
            system_prompt="Speak with Captain Rowan.",
        ),
        assistant_kind="character",
        character_name="Alraune",
        user_display_name_override="Captain Rowan",
        character_system_template="Speak with {{user}}.",
        identity_revision=7,
    )
    controller = ConsoleSessionController.__new__(ConsoleSessionController)

    payload = controller._console_session_to_state(original)
    restored = controller._console_session_from_state(payload)

    assert payload["user_display_name_override"] == "Captain Rowan"
    assert payload["character_system_template"] == "Speak with {{user}}."
    assert payload["identity_revision"] == 7
    assert restored.user_display_name_override == "Captain Rowan"
    assert restored.character_system_template == "Speak with {{user}}."
    assert restored.identity_revision == original.identity_revision


def test_console_task_state_round_trip_preserves_holes_and_id_high_water():
    """Navigation restores task records without reusing a deleted task ID."""
    original = ConsoleChatSession(title="Task state")
    original.todo_store.create(content="One")
    original.todo_store.create(content="Two")
    original.todo_store.create(content="Three")
    original.todo_store.update(task_id="2", expected_version=1, status="deleted")
    controller = ConsoleSessionController.__new__(ConsoleSessionController)

    payload = controller._console_session_to_state(original)
    restored = controller._console_session_from_state(payload)

    assert restored.todo_store.export_snapshot() == {
        "next_id": 4,
        "tasks": [
            {"id": "1", "version": 1, "content": "One", "status": "pending"},
            {
                "id": "3",
                "version": 1,
                "content": "Three",
                "status": "pending",
            },
        ],
    }
    assert restored.todo_store.create(content="Four")["id"] == "4"


def test_console_task_state_uses_one_named_projection_key(monkeypatch):
    """Serialization and restoration share the same screen-state key."""
    from tldw_chatbook.UI.Console_Modules import session as session_module

    monkeypatch.setattr(
        session_module,
        "_CONSOLE_TODO_STATE_KEY",
        "task_state_contract_probe",
        raising=False,
    )
    original = ConsoleChatSession(title="Task key")
    original.todo_store.create(content="One")
    controller = ConsoleSessionController.__new__(ConsoleSessionController)

    payload = controller._console_session_to_state(original)
    restored = controller._console_session_from_state(payload)

    assert "task_state_contract_probe" in payload
    assert "todo_state" not in payload
    assert restored.todo_store.list_after(None) == [
        {"id": "1", "version": 1, "content": "One", "status": "pending"}
    ]


def test_console_task_state_missing_legacy_key_starts_empty_without_warning():
    """Pre-task screen state is a normal legacy payload, not corruption."""
    from loguru import logger as loguru_logger

    controller = ConsoleSessionController.__new__(ConsoleSessionController)
    payload = controller._console_session_to_state(ConsoleChatSession())
    payload.pop("todo_state", None)
    warnings: list[str] = []
    sink_id = loguru_logger.add(
        lambda message: warnings.append(message.record["message"]), level="WARNING"
    )
    try:
        restored = controller._console_session_from_state(payload)
    finally:
        loguru_logger.remove(sink_id)

    assert restored.todo_store.list_after(None) == []
    assert restored.todo_store.create(content="First")["id"] == "1"
    assert warnings == []


def test_console_task_state_malformed_key_starts_empty_with_fixed_warning():
    """Corrupt state emits one structured, payload-free Loguru warning."""
    from loguru import logger as loguru_logger

    controller = ConsoleSessionController.__new__(ConsoleSessionController)
    payload = controller._console_session_to_state(ConsoleChatSession())
    private_values = (
        "private-task-payload",
        "/Users/private/workspace/tasks.json",
        "private-api-key",
    )
    payload["todo_state"] = {
        "sentinel": private_values[0],
        "private_path": private_values[1],
        "api_key": private_values[2],
    }
    records: list[dict[str, object]] = []
    formatted: list[str] = []

    def capture(message) -> None:
        records.append(message.record)
        formatted.append(str(message))

    sink_id = loguru_logger.add(
        capture,
        level="WARNING",
        format="{name}:{function}:{message}",
    )
    try:
        restored = controller._console_session_from_state(payload)
    finally:
        loguru_logger.remove(sink_id)

    assert restored.todo_store.list_after(None) == []
    assert restored.todo_store.create(content="First")["id"] == "1"
    assert len(records) == 1
    record = records[0]
    assert record["message"] == "Console task state invalid; starting empty."
    assert record["exception"] is None
    assert record["extra"] == {"module": "ChatScreen"}
    assert record["name"] == "tldw_chatbook.UI.Console_Modules.session"
    assert record["module"] == "session"
    assert record["function"] == "_console_session_from_state"
    assert formatted == [
        "tldw_chatbook.UI.Console_Modules.session:_console_session_from_state:"
        "Console task state invalid; starting empty.\n"
    ]
    for private_value in private_values:
        assert private_value not in formatted[0]


@pytest.mark.parametrize(
    "invalid_name",
    [
        "bad\x00name",
        "界" * 25,
        "bad\u202ename",
    ],
    ids=["control", "overwide", "bidi-control"],
)
def test_character_screen_state_restore_discards_invalid_user_display_name(
    invalid_name: str,
) -> None:
    """Corrupt saved identity must not prevent the rest of the chat restoring."""
    original = ConsoleChatSession(
        title="Saved roleplay chat",
        assistant_kind="character",
        character_name="Alraune",
        user_display_name_override="Captain Rowan",
        character_system_template="Speak with {{user}}.",
    )
    controller = ConsoleSessionController.__new__(ConsoleSessionController)
    payload = controller._console_session_to_state(original)
    payload["user_display_name_override"] = invalid_name

    restored = controller._console_session_from_state(payload)

    assert restored.id == original.id
    assert restored.title == "Saved roleplay chat"
    assert restored.user_display_name_override is None
    assert restored.character_system_template == "Speak with {{user}}."


@pytest.mark.asyncio
async def test_durable_resume_restores_only_guarded_roleplay_context():
    """Absent, invalid, and future metadata must never invent template provenance."""
    from Tests.UI.test_console_native_chat_flow import (
        StaticConversationTreeService,
        _configure_native_ready_console,
    )
    from Tests.UI.test_destination_shells import _wait_for_selector
    from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
        ConsoleHarness,
    )

    conversations = {
        "valid-roleplay": {
            "conversation": {
                "id": "valid-roleplay",
                "title": "Valid roleplay",
                "system_prompt": "Speak with Captain Rowan.",
                "runtime_backend": "local",
                "assistant_kind": "character",
                "assistant_id": "7",
                "character_id": 7,
                "metadata": {
                    "console_roleplay_context": {
                        "version": 2,
                        "user_name_override": "Captain Rowan",
                        "character_system_template": "Speak with {{user}}.",
                        "character_name_snapshot": "Alraune",
                    }
                },
            },
            "root_threads": [],
        },
        "invalid-roleplay": {
            "conversation": {
                "id": "invalid-roleplay",
                "title": "Invalid roleplay",
                "system_prompt": "Ordinary safe invalid fallback.",
                "metadata": {
                    "console_roleplay_context": {
                        "version": 1,
                        "user_name_override": "bad\nname",
                        "character_system_template": "Ignored {{user}}.",
                    }
                },
            },
            "root_threads": [],
        },
        "absent-roleplay": {
            "conversation": {
                "id": "absent-roleplay",
                "title": "Absent roleplay",
                "system_prompt": "Ordinary safe absent fallback.",
                "metadata": {"unrelated": "keep"},
            },
            "root_threads": [],
        },
        "future-roleplay": {
            "conversation": {
                "id": "future-roleplay",
                "title": "Future roleplay",
                "system_prompt": "Ordinary safe future fallback.",
                "metadata": {
                    "console_roleplay_context": {
                        "version": 3,
                        "user_name_override": "Future Name",
                        "character_system_template": "Future {{user}}.",
                    }
                },
            },
            "root_threads": [],
        },
    }
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.chat_conversation_scope_service = StaticConversationTreeService(conversations)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        console._refresh_active_dictionaries_summary_if_scope_changed = AsyncMock()
        store = console._ensure_console_chat_store()

        assert await console._workspace._resume_console_workspace_conversation(
            "valid-roleplay"
        )
        valid = store.switch_session(store.active_session_id)
        assert valid.user_display_name_override == "Captain Rowan"
        assert valid.character_system_template == "Speak with {{user}}."
        assert valid.settings.system_prompt == "Speak with Captain Rowan."
        assert valid.character_name == "Alraune"
        assert valid.settings.character_label == "Alraune"

        assert await console._workspace._resume_console_workspace_conversation(
            "invalid-roleplay"
        )
        invalid = store.switch_session(store.active_session_id)
        assert invalid.user_display_name_override is None
        assert invalid.character_system_template is None
        assert invalid.settings.system_prompt == "Ordinary safe invalid fallback."

        assert await console._workspace._resume_console_workspace_conversation(
            "absent-roleplay"
        )
        absent = store.switch_session(store.active_session_id)
        assert absent.user_display_name_override is None
        assert absent.character_system_template is None
        assert absent.settings.system_prompt == "Ordinary safe absent fallback."

        assert await console._workspace._resume_console_workspace_conversation(
            "future-roleplay"
        )
        future = store.switch_session(store.active_session_id)
        assert future.user_display_name_override is None
        assert future.character_system_template is None
        assert future.settings.system_prompt == "Ordinary safe future fallback."


def test_local_command_resume_restores_one_anchored_display_only_marker(tmp_path):
    db = AgentRunsDB(tmp_path / "agent-runs.db")
    conversation_id = "local-resume"
    _append_local_command_run(
        db, conversation_id=conversation_id, anchor="assistant-leaf"
    )
    _append_local_command_run(
        db, conversation_id=conversation_id, anchor="deleted-leaf"
    )
    store = ConsoleChatStore()
    session = store.create_session(session_id=conversation_id)
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="normal user prompt",
    )
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="normal assistant reply",
    )
    assistant.persisted_message_id = "assistant-leaf"
    controller = ConsoleChatController(
        store=store,
        provider_gateway=SimpleNamespace(),
    )
    before = json.dumps(
        controller._provider_messages_for_session(session.id),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=SimpleNamespace(),
    )
    blocks = bridge.resume_marker_messages(conversation_id)
    resumed = inject_resume_agent_markers(
        [
            ConsoleChatMessage(role=ConsoleMessageRole.USER, content="normal user prompt"),
            ConsoleChatMessage(
                role=ConsoleMessageRole.ASSISTANT,
                content="normal assistant reply",
                persisted_message_id="assistant-leaf",
            ),
        ],
        blocks,
    )

    markers = [message for message in resumed if message.role is ConsoleMessageRole.TOOL]
    assert len(markers) == 1
    marker = markers[0]
    assert resumed.index(marker) == 2
    assert marker.raw_cli_presentation is not None
    assert marker.raw_cli_presentation.exit_code == 7
    assert marker.raw_cli_presentation.lifecycle_state == "exited"
    units = group_console_transcript_messages(resumed)
    assert units[1].assistant_turn is not None
    assert units[1].assistant_turn.assistant is resumed[1]
    assert units[1].assistant_turn.activities == ()
    assert units[2].standalone is marker
    assert tuple(visual_messages(units)) == tuple(resumed)

    store.append_message(
        session.id,
        role=ConsoleMessageRole.TOOL,
        content=marker.content,
        tool_output_full=marker.tool_output_full,
        activity_presentation=marker.activity_presentation,
        raw_cli_presentation=marker.raw_cli_presentation,
        record_trajectory=False,
    )
    after = json.dumps(
        controller._provider_messages_for_session(session.id),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert after == before


@pytest.mark.parametrize("local_query_fails", [False, True])
def test_resume_drops_poison_local_rows_without_breaking_primary_markers(
    tmp_path,
    monkeypatch,
    local_query_fails,
):
    db = AgentRunsDB(tmp_path / "agent-runs.db")
    primary_id = db.create_run(
        conversation_id="poison-local-resume",
        agent_kind="primary",
        assistant_message_id="assistant-leaf",
    )
    db.append_steps(
        primary_id,
        [
            {
                "index": 0,
                "kind": "tool_result",
                "tool_name": "calculator",
                "result": "primary marker survived",
                "args": None,
            }
        ],
    )
    db.set_status(primary_id, "done")
    projection_calls: list[str] = []

    def local_command_resume_records(conversation_id):
        projection_calls.append(conversation_id)
        if local_query_fails:
            raise ValueError("corrupt local-command steps")
        return [
            42,
            {
                "id": "poison-local-run",
                "agent_kind": "local_command",
                "assistant_message_id": "assistant-leaf",
                "status": "done",
                "steps": "not-a-step-sequence",
            },
        ]

    monkeypatch.setattr(
        db,
        "local_command_resume_records",
        local_command_resume_records,
        raising=False,
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=None,
        provider_gateway=None,
    )

    blocks = bridge.resume_marker_messages("poison-local-resume")
    markers = [marker for _anchor, block in blocks for marker in block]

    assert projection_calls == ["poison-local-resume"]
    assert len(markers) == 1
    assert markers[0].raw_cli_presentation is None
    assert "primary marker survived" in markers[0].content


def test_local_command_is_ignored_by_trajectory_rails_fleet_and_cost(tmp_path):
    db = AgentRunsDB(tmp_path / "agent-runs.db")
    conversation_id = "local-exclusions"
    run_id = _append_local_command_run(
        db, conversation_id=conversation_id, anchor="assistant-leaf"
    )
    store = ConsoleChatStore()
    store.create_session(session_id=conversation_id)
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=SimpleNamespace(),
    )

    assert bridge.latest_primary_run_id(conversation_id) is None
    assert bridge.subagent_count(conversation_id) == 0
    assert bridge.subagent_runs(conversation_id) == []
    assert bridge.fleet_snapshot(conversation_id) == []
    historical = bridge.historical_snapshot(conversation_id)
    assert historical.status == "idle"
    assert historical.steps == ()
    assert historical.subagents == ()

    class _MessageDB:
        def get_messages_for_conversation(self, *_args, **_kwargs):
            return []

        def get_trajectory_rows(self, *_args, **_kwargs):
            return []

        def get_conversation_active_leaf(self, *_args, **_kwargs):
            return None

    trajectory_store = SimpleNamespace(
        persistence=SimpleNamespace(db=_MessageDB()),
        variant_sets_for_conversation=lambda _conversation_id: (),
    )
    snapshot = _build_trajectory_snapshot(
        trajectory_store,
        conversation_id,
        agent_runs_db=db,
    )
    assert run_id not in repr(snapshot)
    assert "LOCAL_COMMAND_SECRET" not in repr(snapshot)
    assert "LOCAL_OUTPUT_SECRET" not in repr(snapshot)

    ordinary = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="ordinary",
    )
    (marker,) = bridge.resume_marker_messages(conversation_id)[0][1]
    assert console_cost_snapshot_messages([ordinary, marker]) == [ordinary]
