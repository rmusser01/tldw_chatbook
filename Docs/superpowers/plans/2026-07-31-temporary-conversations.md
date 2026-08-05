# Temporary Conversations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a Console chat run entirely in memory — no conversation row, no message rows, no derived local files — with an explicit "Save this chat" that promotes it to a normal conversation.

**Architecture:** One `ephemeral: bool` flag on `ConsoleChatSession`, enforced at exactly one choke point: `ConsoleChatStore.persist_session_if_needed` returns `None` for a temporary session, so it never acquires a `persisted_conversation_id`, and every downstream durable write no-ops along the branch it already takes when no persistence adapter is configured. Artifact-producing UI actions (image generation, chatbook export) are blocked from a single registry module. Promotion clears the flag and writes everything inside one DB transaction.

**Tech Stack:** Python 3.11+, Textual, SQLite (`CharactersRAGDB`), pytest.

**Spec:** `Docs/superpowers/specs/2026-07-31-temporary-conversations-design.md`

## Global Constraints

- **Worktree:** `/private/tmp/ephemeral`, branch `docs/temporary-conversations-spec` (off `origin/dev` @ `0fa4d2ca5`). All work happens here.
- **Interpreter:** `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`. Always invoke as `python -m pytest` **with cwd set to the worktree** — `sys.path[0]` is then the worktree, so the worktree's `tldw_chatbook` wins over the editable install pointing at the main checkout. Verified: `import tldw_chatbook` from the worktree resolves to `/private/tmp/ephemeral/tldw_chatbook/__init__.py`.
- **Never run `git stash`.** The stash stack is shared across every worktree in this repo.
- **Never hand-edit `tldw_chatbook/css/tldw_cli_modular.tcss`.** It is a generated bundle. Edit `tldw_chatbook/css/components/_agentic_terminal.tcss` and regenerate with `python tldw_chatbook/css/build_css.py`.
- **The promise is "not saved locally"** — never "private", "untracked", "anonymous", or anything implying provider-side behavior. Copy that overstates the guarantee is a defect.
- **The tab marker is presentation-only.** `session.title` must never contain marker text; promotion would otherwise save a conversation literally titled with the marker.
- **Backlog task IDs collide across concurrent sessions.** Before filing any backlog task for this work, sweep every worktree (`os.listdir` + regex over `backlog/tasks/` in all checkouts), then leapfrog with headroom, and re-verify at merge time.
- The word "ephemeral" is the internal/code term. The user-facing word is **"temporary"**, everywhere.

---

### Task 1: Local-write sink registry and audit

The spec's residual risk is not the gate — it is whether the list of artifact-producing actions is complete. This task creates the single place that list lives, and audits the codebase to fill it.

**Files:**
- Create: `tldw_chatbook/Chat/console_ephemeral.py`
- Create: `Tests/Chat/test_console_ephemeral.py`
- Modify: `Docs/superpowers/specs/2026-07-31-temporary-conversations-design.md` (append audit findings)

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `TEMPORARY_LABEL: str` — the chip label, `"Temporary — not saved"`.
  - `TEMPORARY_TOOLTIP: str`
  - `EPHEMERAL_BLOCKED_ACTIONS: dict[str, str]` — action id → reason sentence.
  - `blocked_reason(action_id: str, *, ephemeral: bool) -> str | None`
  - `ACTION_SAVE_CHAT: str` — `"save-chat"`, the promote action id.

- [ ] **Step 1: Audit the local-write sinks reachable from a Console chat**

Run these searches from the worktree and read each hit, deciding for every one whether a temporary chat can reach it and whether it writes to disk:

```bash
cd /private/tmp/ephemeral
grep -rn "generate-image\|save-chatbook\|save_chatbook" tldw_chatbook/UI/Screens/chat_screen.py
grep -rn "def .*write\|open(.*[\"']w\|\.write_bytes\|\.write_text\|save_image\|export" tldw_chatbook/Chat/ tldw_chatbook/Widgets/Console/
grep -rn "index_conversation\|index_message\|embed\|add_to_index" tldw_chatbook/RAG_Search/ | grep -i "conversation\|chat"
grep -rn "media_db\|Client_Media_DB" tldw_chatbook/Chat/attachment_core.py tldw_chatbook/UI/Screens/chat_screen.py
```

Write the findings as a table appended to the spec under a new `## Sink audit (task 1)` heading: one row per sink, columns `Sink | Reachable from a temporary chat? | Writes what | Decision`. Decision is one of `blocked`, `allowed (no write)`, or `no-op (needs a conversation id)`.

The two sinks already known from the spec are Generate Image (writes a PNG) and Save Chatbook (writes an export file). RAG indexing of the chat's own content is a `no-op` — it needs a conversation id. Anything else the searches surface is a new row.

- [ ] **Step 2: Write the failing test**

Create `Tests/Chat/test_console_ephemeral.py`:

```python
"""Temporary (non-persisted) Console conversations: vocabulary and blocked actions."""

import pytest

from tldw_chatbook.Chat.console_ephemeral import (
    EPHEMERAL_BLOCKED_ACTIONS,
    TEMPORARY_LABEL,
    blocked_reason,
)


@pytest.mark.unit
def test_blocked_reason_only_applies_to_temporary_sessions():
    """A normal chat blocks nothing; a temporary one blocks the audited sinks."""
    for action_id in EPHEMERAL_BLOCKED_ACTIONS:
        assert blocked_reason(action_id, ephemeral=False) is None
        reason = blocked_reason(action_id, ephemeral=True)
        assert isinstance(reason, str) and reason.strip()

    assert blocked_reason("send", ephemeral=True) is None


@pytest.mark.unit
def test_blocked_reasons_name_the_artifact_not_the_feature():
    """Each reason says what would hit disk -- 'disabled' alone teaches nothing."""
    for action_id, reason in EPHEMERAL_BLOCKED_ACTIONS.items():
        assert "temporary chat" in reason, action_id
        assert reason == reason.strip()


@pytest.mark.unit
def test_user_facing_copy_never_overstates_the_guarantee():
    """The promise is local durability only -- not privacy, not anonymity."""
    forbidden = ("private", "anonym", "untracked", "incognito", "secure")
    copy = " ".join([TEMPORARY_LABEL, *EPHEMERAL_BLOCKED_ACTIONS.values()]).lower()
    for word in forbidden:
        assert word not in copy, f"copy overstates the guarantee: {word!r}"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /private/tmp/ephemeral && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_ephemeral.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tldw_chatbook.Chat.console_ephemeral'`

- [ ] **Step 4: Write the module**

Create `tldw_chatbook/Chat/console_ephemeral.py`:

```python
"""Temporary (non-persisted) Console conversations: shared vocabulary.

A temporary session never acquires a ``persisted_conversation_id`` (see
``ConsoleChatStore.persist_session_if_needed``), so every durable write in
the store no-ops on its own. What this module owns is the OTHER half of the
guarantee: the UI actions that would write a derived artifact to disk even
though no conversation row exists.

The registry below is the single place that list lives. Adding a new
artifact-producing Console action means adding a row here -- the enumeration
test in ``Tests/Chat/test_console_ephemeral.py`` is what keeps that honest.

The promise is LOCAL DURABILITY only: "not saved locally". Nothing here may
imply privacy or provider-side behavior.
"""

from __future__ import annotations

#: Composer-menu action id for promoting a temporary chat ("Save this chat").
ACTION_SAVE_CHAT = "save-chat"

#: Chip label shown in the Console status strip while a chat is temporary.
TEMPORARY_LABEL = "Temporary — not saved"

#: Chip tooltip. Says what survives and what does not, without implying more.
TEMPORARY_TOOLTIP = (
    "This chat is not saved locally. It is lost when the tab closes or the "
    "app restarts. Activate to save it."
)

#: Action id -> why it is unavailable while the chat is temporary. Keyed by
#: the ids the Console workbench and composer menu already use, so a lookup
#: needs no translation layer.
EPHEMERAL_BLOCKED_ACTIONS: dict[str, str] = {
    "generate-image": (
        "Generating an image writes a file to disk — not available in a "
        "temporary chat."
    ),
    "save-chatbook": (
        "Saving a Chatbook exports a file to disk — not available in a "
        "temporary chat."
    ),
}


def blocked_reason(action_id: str, *, ephemeral: bool) -> str | None:
    """Return why ``action_id`` is unavailable, or ``None`` when it is available.

    Args:
        action_id: Console action id (workbench action or composer menu entry).
        ephemeral: Whether the active session is temporary.

    Returns:
        The reason sentence to show on the disabled control, or ``None`` when
        the action is available (which is always the case outside a temporary
        chat).
    """
    if not ephemeral:
        return None
    return EPHEMERAL_BLOCKED_ACTIONS.get(action_id)
```

If Step 1's audit surfaced additional sinks, add one `EPHEMERAL_BLOCKED_ACTIONS` row per sink here, using the same phrasing shape ("<verb>ing … writes/exports … — not available in a temporary chat."), and add its UI block to Task 9 following the pattern shown there.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /private/tmp/ephemeral && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_ephemeral.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
cd /private/tmp/ephemeral
git add tldw_chatbook/Chat/console_ephemeral.py Tests/Chat/test_console_ephemeral.py Docs/superpowers/specs/2026-07-31-temporary-conversations-design.md
git commit -m "feat: temporary-chat vocabulary and blocked-action registry

Single place the artifact-producing action list lives, with an
enumeration test so a new sink cannot be added silently. Spec gains the
audit table of every local-write sink reachable from a Console chat."
```

---

### Task 2: The `ephemeral` flag and the persistence gate

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_store.py:247-273` (dataclass), `:413-445` (`create_session`), `:447` (`restore_persisted_session`), `:2187-2197` (`persist_session_if_needed`)
- Test: `Tests/Chat/test_console_ephemeral.py` (append)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `ConsoleChatSession.ephemeral: bool` (default `False`)
  - `ConsoleChatStore.create_session(..., ephemeral: bool = False)`

- [ ] **Step 1: Write the failing test**

Append to `Tests/Chat/test_console_ephemeral.py`. This is **the proof test**, and it carries its own control: the same harness, the same calls, a normal session — which MUST write rows. Without the control the assertion "no rows appeared" passes against a completely broken gate.

```python
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _row_counts(db: CharactersRAGDB) -> tuple[int, int]:
    """Return (conversations, messages) row counts straight from SQLite."""
    conn = db.get_connection()
    conversations = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
    messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    return conversations, messages


def _run_a_chat(store: ConsoleChatStore, session_id: str) -> None:
    """Drive one complete exchange through the store."""
    store.append_message(
        session_id, role=ConsoleMessageRole.USER, content="hello", persist=True
    )
    store.append_message(
        session_id, role=ConsoleMessageRole.ASSISTANT, content="hi there", persist=True
    )
    store.persist_session_if_needed(session_id)


@pytest.mark.unit
def test_temporary_session_writes_no_rows_while_a_normal_one_does(tmp_path):
    """The gate holds -- proven against a control that DOES write.

    A harness with ``persistence=None`` would pass the "no rows" half of
    this trivially, which is why the normal-session half runs first in the
    same database with the same calls.
    """
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))

        # CONTROL: a normal session must write rows here.
        baseline = _row_counts(db)
        normal = store.create_session(title="Normal chat")
        _run_a_chat(store, normal.id)
        after_normal = _row_counts(db)
        assert after_normal[0] == baseline[0] + 1, "control wrote no conversation row"
        assert after_normal[1] > baseline[1], "control wrote no message rows"
        assert normal.persisted_conversation_id is not None

        # SUBJECT: a temporary session must write nothing.
        temporary = store.create_session(title="Temporary chat", ephemeral=True)
        _run_a_chat(store, temporary.id)
        assert _row_counts(db) == after_normal
        assert temporary.persisted_conversation_id is None
        assert store.persist_session_if_needed(temporary.id) is None

        # The transcript is still fully present in memory.
        assert [m.content for m in store.messages_for_session(temporary.id)] == [
            "hello",
            "hi there",
        ]

        # Closing the tab -- the ordinary way a temporary chat ends -- must
        # not flush anything on the way out.
        store.close_session(temporary.id)
        assert _row_counts(db) == after_normal
    finally:
        db.close()


@pytest.mark.unit
def test_restore_persisted_session_refuses_to_open_the_second_door(tmp_path):
    """``restore_persisted_session`` assigns the id directly -- it must not
    be reachable with ``ephemeral`` set, or the gate has a bypass."""
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        with pytest.raises(ValueError, match="temporary"):
            store.restore_persisted_session(
                title="Restored",
                workspace_id=None,
                persisted_conversation_id="conv-1",
                all_nodes=[],
                ephemeral=True,
            )
    finally:
        db.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /private/tmp/ephemeral && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_ephemeral.py -v`
Expected: FAIL — `TypeError: create_session() got an unexpected keyword argument 'ephemeral'`

- [ ] **Step 3: Add the field**

In `tldw_chatbook/Chat/console_chat_store.py`, after `character_name: str | None = None` in `ConsoleChatSession` (currently line 273):

```python
    #: Temporary conversation (spec 2026-07-31): this session is never written
    #: to local storage. Enforced in exactly one place --
    #: ``persist_session_if_needed`` refuses to mint a
    #: ``persisted_conversation_id`` -- so every durable write downstream
    #: no-ops along the branch it already takes with no persistence adapter.
    #: A write site that forgets about this flag therefore fails toward NOT
    #: writing, which is the whole reason the guard lives at the id and not
    #: at the 43 sites that consult ``self.persistence``.
    ephemeral: bool = False
```

- [ ] **Step 4: Thread it through `create_session`**

In `create_session` (line 413), add the keyword after `character_name`:

```python
        character_name: str | None = None,
        ephemeral: bool = False,
    ) -> ConsoleChatSession:
        """Create and activate a new native Console session.

        Args:
            ephemeral: When True the session is temporary -- never written to
                local storage until ``promote_ephemeral_session`` clears the
                flag.
        """
```

and in the `ConsoleChatSession(...)` construction inside it:

```python
            character_name=character_name,
            ephemeral=ephemeral,
        )
```

- [ ] **Step 5: Close the gate**

In `persist_session_if_needed` (line 2187), replace the docstring's Returns block and insert the guard immediately after `session = self._session_or_raise(session_id)`:

```python
        Returns:
            The persisted conversation ID; ``None`` when no persistence
            adapter is configured, or when the session is temporary.
        """
        session = self._session_or_raise(session_id)
        # Temporary conversations (spec 2026-07-31) stop here, BEFORE the
        # already-persisted check and before the adapter is consulted. This
        # single early return is the entire durability mechanism: with no
        # conversation id, `persist_message_if_needed` and every other
        # conversation-scoped write returns early on its own.
        if session.ephemeral:
            return None
        if session.persisted_conversation_id is not None:
            return session.persisted_conversation_id
```

- [ ] **Step 6: Guard the second door**

`restore_persisted_session` (line 447) assigns `persisted_conversation_id` directly, bypassing the gate. It is unreachable for temporary sessions today, but the guard is what keeps a later change from quietly opening it. Add the keyword to its signature after `all_nodes`:

```python
        ephemeral: bool = False,
```

and as the first statement in its body:

```python
        # A restored session comes FROM durable storage, so it is by
        # definition not temporary. Refuse rather than silently produce a
        # session that is both temporary and persisted -- the one state the
        # gate's invariant does not allow.
        if ephemeral:
            raise ValueError(
                "Cannot restore a persisted session as temporary: a temporary "
                "session has no persisted conversation."
            )
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd /private/tmp/ephemeral && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_ephemeral.py -v`
Expected: PASS (5 tests)

- [ ] **Step 8: Run the existing store suite for regressions**

Run: `cd /private/tmp/ephemeral && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_chat_store.py Tests/Chat/test_console_chat_store_tree.py Tests/Chat/test_console_chat_store_parent_persist.py -q`
Expected: PASS, no new failures.

- [ ] **Step 9: Commit**

```bash
cd /private/tmp/ephemeral
git add tldw_chatbook/Chat/console_chat_store.py Tests/Chat/test_console_ephemeral.py
git commit -m "feat: gate Console persistence on a session ephemeral flag

persist_session_if_needed returns None for a temporary session, so it
never acquires a conversation id and every downstream durable write
no-ops. Proven with a real SQLite database against a normal-session
control that writes rows in the same harness."
```

---

### Task 3: Promotion — "Save this chat" in one transaction

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_store.py` (new method after `persist_session_if_needed`, plus a new private helper `_tree_nodes_parent_first`)
- Test: `Tests/Chat/test_console_ephemeral.py` (append)

**Interfaces:**
- Consumes: `ConsoleChatSession.ephemeral`, `create_session(ephemeral=...)` from Task 2; `session.rag_scope_holder` (task-9's `SessionScopeHolder`); `_nodes_by_session` / `_children_by_parent` / `_native_parent_by_message` (the full-tree structures Phase A's branching tasks maintain).
- Produces: `ConsoleChatStore.promote_ephemeral_session(session_id: str) -> str | None` — returns the new conversation id, or `None` when the session was not temporary or no adapter is configured. Raises whatever the persistence layer raises (or an internal `RuntimeError` for one defensive, currently-unreachable case — see below), after restoring the session to its temporary state, including its held RAG retrieval scope.

**Implementation history (why this section looks different from a first pass):** this task shipped in three rounds. The initial pass matched the prescription below almost exactly, but persisted only the active-path view (`_messages_by_session`) inside a DB transaction, with a plain rollback of `ephemeral`/the persisted ids. An independent review then found four gaps in round 1 (a held RAG scope was destroyed by a failed save; the no-transaction fallback dropped atomicity silently; an internal `None` return bypassed the rollback; the invariant was momentarily false inside the except block) and, in round 2, one gap that changed the task's actual behavior: **promotion was saving only the active path**, silently dropping off-path branches left behind by `create_sibling` (regenerate / edit-and-resend) — reachable by swiping back in a normal conversation, but gone forever from a promoted temporary one. The code and tests below are the FINAL, whole-tree-promoting state; the full round-by-round record (RED/GREEN output, mutation-testing sanity checks, and what was ruled out and why) lives in `.superpowers/sdd/2026-07-31-temporary-conversations/task-3-report.md`.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/Chat/test_console_ephemeral.py` (needs one new import alongside the existing ones: `from tldw_chatbook.Chat.rag_scope import RagScope, ScopeItem, SOURCE_TYPE_MEDIA`):

```python
@pytest.mark.unit
def test_promotion_writes_every_message_in_order(tmp_path):
    """Saving a temporary chat persists exactly what is on screen.

    This session's tree has no branches, so the active path IS the whole
    tree and this test cannot by itself distinguish active-path-only
    promotion from whole-tree promotion -- see
    ``test_promotion_writes_every_node_including_off_path_branches`` below
    for the branching case that does.
    """
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        session = store.create_session(title="Temporary chat", ephemeral=True)
        _run_a_chat(store, session.id)
        assert _row_counts(db) == (0, 0)

        conversation_id = store.promote_ephemeral_session(session.id)

        assert conversation_id is not None
        assert session.ephemeral is False
        assert session.persisted_conversation_id == conversation_id
        assert _row_counts(db) == (1, 2)
        persisted = [
            db.get_message_by_id(m.persisted_message_id)["content"]
            for m in store.messages_for_session(session.id)
        ]
        assert persisted == ["hello", "hi there"]
    finally:
        db.close()


@pytest.mark.unit
def test_promotion_writes_every_node_including_off_path_branches(tmp_path):
    """Saving must not silently drop history still reachable by swiping back.

    Regenerating (``create_sibling``) leaves the previous assistant reply
    off the active path but still a real tree node, reachable via
    ``set_active_leaf``. A normal (never-temporary) conversation persists
    that node like any other; a promoted temporary one must come out the
    same way -- otherwise regenerating twice and then saving would silently
    erase the earlier answers, making the promoted conversation unlike one
    that had been saved from the start.
    """
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        session = store.create_session(title="Temporary chat", ephemeral=True)
        store.append_message(
            session.id, role=ConsoleMessageRole.USER, content="hello", persist=True
        )
        original_reply = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="hi there",
            persist=True,
        )
        regenerated_reply = store.create_sibling(
            original_reply.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="hi again",
            persist=True,
        )
        assert _row_counts(db) == (0, 0)
        # The active view follows the regenerated branch; the original
        # reply is off-path but still a tree node.
        assert [m.content for m in store.messages_for_session(session.id)] == [
            "hello",
            "hi again",
        ]

        conversation_id = store.promote_ephemeral_session(session.id)

        assert conversation_id is not None
        # 1 conversation, 3 messages: "hello" plus BOTH assistant replies.
        assert _row_counts(db) == (1, 3)
        all_nodes = store._nodes_by_session[session.id]
        assert len(all_nodes) == 3
        assert all(
            node.persisted_message_id is not None for node in all_nodes.values()
        ), "every tree node must be persisted, not just the active path"
        persisted_contents = {
            db.get_message_by_id(node.persisted_message_id)["content"]
            for node in all_nodes.values()
        }
        assert persisted_contents == {"hello", "hi there", "hi again"}

        # Swipe-back must still work after saving: switching the active leaf
        # to the off-path (now-persisted) original reply must not raise and
        # must surface its persisted content.
        store.set_active_leaf(session.id, original_reply.id)
        assert [m.content for m in store.messages_for_session(session.id)] == [
            "hello",
            "hi there",
        ]
    finally:
        db.close()


@pytest.mark.unit
def test_promotion_preserves_persisted_parent_child_structure(tmp_path):
    """The persisted tree must connect exactly like the in-memory one.

    Builds a tree deep enough (root -> reply -> user turn -> two sibling
    replies) that writing a node before its parent is persisted would
    either strand it as a bogus root or -- worse -- silently attach it to
    the wrong, already-persisted ancestor further up the chain. Comparing
    every persisted row's ``parent_message_id`` against the in-memory
    native parent's OWN persisted id (translated through the same
    node) catches both failure modes; a same-order-as-creation write
    produces neither. (Verified by temporarily reversing
    ``_tree_nodes_parent_first``'s output and confirming this exact
    assertion fails -- see the task-3 report.)
    """
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        session = store.create_session(title="Temporary chat", ephemeral=True)
        root = store.append_message(
            session.id, role=ConsoleMessageRole.USER, content="U1", persist=True
        )
        reply = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="A1", persist=True
        )
        turn2 = store.append_message(
            session.id, role=ConsoleMessageRole.USER, content="U2", persist=True
        )
        branch_a = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="A2a", persist=True
        )
        branch_b = store.create_sibling(
            branch_a.id, role=ConsoleMessageRole.ASSISTANT, content="A2b", persist=True
        )

        conversation_id = store.promote_ephemeral_session(session.id)

        assert conversation_id is not None
        assert _row_counts(db) == (1, 5)
        rows_by_persisted_id = {
            row["id"]: row
            for row in db.get_messages_for_conversation(conversation_id, limit=100)
        }
        assert len(rows_by_persisted_id) == 5

        all_nodes = store._nodes_by_session[session.id]
        for native_id, node in all_nodes.items():
            assert node.persisted_message_id in rows_by_persisted_id
            native_parent_id = store._native_parent_by_message[native_id]
            expected_parent_persisted_id = (
                all_nodes[native_parent_id].persisted_message_id
                if native_parent_id is not None
                else None
            )
            actual_parent_persisted_id = rows_by_persisted_id[
                node.persisted_message_id
            ]["parent_message_id"]
            assert actual_parent_persisted_id == expected_parent_persisted_id, (
                f"node {node.content!r} persisted with the wrong parent"
            )

        # Spot-check the branch point explicitly (the case an ordering bug
        # would most likely get wrong -- turn2 has TWO persisted children).
        # ``root``/``reply``/``turn2`` are snapshots taken BEFORE promotion
        # (``append_message`` returns a point-in-time copy via
        # ``dataclasses.replace`` -- see ``_snapshot``), so their own
        # ``persisted_message_id`` is stale; only their stable native ``.id``
        # is reused here, looked up fresh through the live ``all_nodes``
        # mapping captured after promotion above.
        turn2_persisted_id = all_nodes[turn2.id].persisted_message_id
        children_of_turn2 = {
            row["content"]
            for row in rows_by_persisted_id.values()
            if row["parent_message_id"] == turn2_persisted_id
        }
        assert children_of_turn2 == {"A2a", "A2b"}
        assert (
            rows_by_persisted_id[all_nodes[reply.id].persisted_message_id][
                "parent_message_id"
            ]
            == all_nodes[root.id].persisted_message_id
        )
        assert (
            rows_by_persisted_id[all_nodes[turn2.id].persisted_message_id][
                "parent_message_id"
            ]
            == all_nodes[reply.id].persisted_message_id
        )
    finally:
        db.close()


@pytest.mark.unit
def test_promotion_is_idempotent(tmp_path):
    """A second Save writes nothing more and does not raise."""
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        session = store.create_session(title="Temporary chat", ephemeral=True)
        _run_a_chat(store, session.id)
        first = store.promote_ephemeral_session(session.id)
        after_first = _row_counts(db)

        assert store.promote_ephemeral_session(session.id) is None
        assert _row_counts(db) == after_first
        assert session.persisted_conversation_id == first
    finally:
        db.close()


@pytest.mark.unit
def test_failed_promotion_rolls_back_and_stays_temporary(tmp_path, monkeypatch):
    """A half-saved conversation must never be left in history.

    The failure is injected on the SECOND message write, so the conversation
    row and the first message are already in the transaction when it blows
    up -- exactly the partial state the rollback exists to undo.
    """
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        persistence = ChatPersistenceService(db)
        store = ConsoleChatStore(persistence=persistence)
        session = store.create_session(title="Temporary chat", ephemeral=True)
        _run_a_chat(store, session.id)

        calls = {"n": 0}
        real_create = persistence.create_message

        def failing_create(**kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("disk full")
            return real_create(**kwargs)

        monkeypatch.setattr(persistence, "create_message", failing_create)

        with pytest.raises(RuntimeError, match="disk full"):
            store.promote_ephemeral_session(session.id)

        assert _row_counts(db) == (0, 0), "partial conversation survived"
        assert session.ephemeral is True, "failed save left the chat persisting"
        assert session.persisted_conversation_id is None
        assert all(
            m.persisted_message_id is None
            for m in store.messages_for_session(session.id)
        )
    finally:
        db.close()


@pytest.mark.unit
def test_failed_promotion_restores_the_held_rag_scope(tmp_path, monkeypatch):
    """A failed save must not silently drop the user's scope selection.

    ``persist_session_if_needed`` flushes (and empties) the session's held
    RAG scope as soon as the conversation row is created -- before either
    message write can fail. If promotion rolls back the database but
    leaves the now-empty holder alone, the user's scope selection vanishes
    even though the chat correctly stays temporary. This is reachable in
    normal use: the Console screen puts a scope in the holder precisely
    when there is no persisted conversation, which is always true for a
    temporary chat.
    """
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        persistence = ChatPersistenceService(db)
        store = ConsoleChatStore(persistence=persistence)
        session = store.create_session(title="Temporary chat", ephemeral=True)
        _run_a_chat(store, session.id)

        scope = RagScope(items=(ScopeItem(SOURCE_TYPE_MEDIA, "doc-1"),), updated_at="t1")
        session.rag_scope_holder.set(scope)

        calls = {"n": 0}
        real_create = persistence.create_message

        def failing_create(**kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("disk full")
            return real_create(**kwargs)

        monkeypatch.setattr(persistence, "create_message", failing_create)

        with pytest.raises(RuntimeError, match="disk full"):
            store.promote_ephemeral_session(session.id)

        assert session.ephemeral is True
        assert session.rag_scope_holder.scope == scope, (
            "failed promotion must restore the held scope, not leave it empty"
        )
    finally:
        db.close()


@pytest.mark.unit
def test_promotion_restores_ephemeral_flag_if_persist_returns_none_unexpectedly(
    tmp_path, monkeypatch
):
    """Defensive: an unexpected None from persist_session_if_needed must not
    silently leave the session non-ephemeral with no persisted conversation.

    That state is exactly what the docstring warns about -- a failed save
    that silently starts persisting on the next send. Nothing in
    ``persist_session_if_needed`` reaches this today (its only None-return
    branches are already ruled out once ``ephemeral`` is cleared and
    ``self.persistence`` is known non-None), so this test forces the case
    directly to prove the rollback still fires if a future change ever adds
    one.
    """
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        session = store.create_session(title="Temporary chat", ephemeral=True)
        _run_a_chat(store, session.id)

        monkeypatch.setattr(store, "persist_session_if_needed", lambda session_id: None)

        with pytest.raises(RuntimeError):
            store.promote_ephemeral_session(session.id)

        assert session.ephemeral is True
        assert session.persisted_conversation_id is None
        assert _row_counts(db) == (0, 0)
    finally:
        db.close()
```

`create_message` is the adapter call the message flush reaches (verified: `console_chat_store.py:2673` at the time Task 3 started) — not `add_message`, which does not exist on `ChatPersistenceService`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /private/tmp/ephemeral && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_ephemeral.py -v -k promotion`
Expected: FAIL. For a from-scratch implementation this is `AttributeError: 'ConsoleChatStore' object has no attribute 'promote_ephemeral_session'` on all seven; when only the round-2 (whole-tree) tests are new against an already-existing active-path-only implementation, the two branching tests fail on the row-count assertions instead (`(1, 2) != (1, 3)`, `(1, 4) != (1, 5)`) — both are the "fails for the right reason" signal to look for.

- [ ] **Step 3: Implement promotion**

Add a private helper (near the other tree-walk helpers, e.g. beside `_subtree_ids`/`_leaf_under`):

```python
    def _tree_nodes_parent_first(self, session_id: str) -> list[ConsoleChatMessage]:
        """Return EVERY tree node for a session, guaranteed parent-before-child.

        Used by ``promote_ephemeral_session`` (task-3), which must persist the
        whole conversation tree -- every off-path branch left behind by
        ``create_sibling`` (regenerate / edit-and-resend), not just the
        active-path view -- so a promoted temporary chat comes out
        indistinguishable from one that had been saved from the start,
        swipe-back included.

        Ordering is load-bearing, not cosmetic: ``_persist_new_message``
        resolves each node's persisted parent via
        ``_nearest_persisted_ancestor_id``, which walks up
        ``_native_parent_by_message`` looking for the nearest ANCESTOR that
        already has a ``persisted_message_id``. Persisting a child before its
        parent would leave that walk with nothing to find (a stray root) or,
        worse, silently resolve to some unrelated already-persisted ancestor
        further up the chain -- a misparented row that looks fine until a
        later resume walks the wrong branch.

        A breadth-first walk from the roots (``_children_by_parent[session_id]
        [None]``) down through ``_children_by_parent`` guarantees this by
        construction: a node is only enqueued once its parent has already been
        dequeued and emitted. This does NOT rely on ``_nodes_by_session``'s
        dict insertion/iteration order for correctness -- that order is
        unspecified by this method's contract even though CPython dicts
        happen to preserve insertion order today.

        Returns:
            Every node, in an order where each node's parent (if any)
            precedes it. TOOL markers are excluded -- they are display-only
            and never become tree nodes (see ``_register_tree_node``).
        """
        nodes = self._nodes_by_session.get(session_id, {})
        children_map = self._children_by_parent.get(session_id, {})
        ordered: list[ConsoleChatMessage] = []
        queue: deque[str] = deque(children_map.get(None, []))
        while queue:
            node_id = queue.popleft()
            node = nodes.get(node_id)
            if node is not None:
                ordered.append(node)
            queue.extend(children_map.get(node_id, []))
        return ordered
```

Needs `from collections import deque` added to the module's stdlib imports.

Add to `ConsoleChatStore`, immediately after `persist_session_if_needed`:

```python
    def promote_ephemeral_session(self, session_id: str) -> str | None:
        """Save a temporary conversation to durable storage, all or nothing.

        Clears ``ephemeral`` first -- that is what opens the gate in
        ``persist_session_if_needed`` -- then mints the conversation and
        flushes every node in the FULL conversation tree, not just the
        active-path view: off-path branches left behind by
        ``create_sibling`` (regenerate / edit-and-resend) are still reachable
        by swiping back, and a normal (never-temporary) conversation persists
        them, so a promoted one must too -- otherwise saving would silently
        discard history the user could see a moment before clicking Save.
        Nodes are written parent-before-child (``_tree_nodes_parent_first``)
        since each node's persisted parent is resolved from its
        already-persisted ancestors. The whole sequence runs inside one
        database transaction when the adapter exposes a real database, so a
        failure part-way through leaves NO conversation in history rather
        than a truncated one.

        On any failure the session is restored to its temporary state
        (``ephemeral`` back to True, ids cleared, any held RAG retrieval
        scope restored). A failed save that left the flag cleared would
        silently start persisting on the next send -- the opposite of what
        the user asked for. Restoring the RAG scope matters for the same
        reason: ``persist_session_if_needed`` flushes (and empties) the
        session's held scope as soon as the conversation row exists, before
        any message write can fail, so a rollback that only undid the DB
        write would still leave the user's scope selection gone.

        Args:
            session_id: Id of the temporary session to save.

        Returns:
            The new persisted conversation id, or ``None`` when the session
            was not temporary (already saved -- this is idempotent) or no
            persistence adapter is configured.

        Raises:
            Exception: Whatever the persistence layer raises, re-raised after
                the in-memory rollback. Also raised (as ``RuntimeError``) if
                ``persist_session_if_needed`` unexpectedly returns ``None``
                after ``ephemeral`` has already been cleared -- today
                unreachable, but treated as a failure rather than silently
                leaving the session non-ephemeral with no persisted
                conversation.
        """
        session = self._session_or_raise(session_id)
        if not session.ephemeral:
            return None
        if self.persistence is None:
            return None

        messages = self._tree_nodes_parent_first(session_id)
        db = getattr(self.persistence, "db", None)
        transaction = getattr(db, "transaction", None)
        # Captured BEFORE any write -- persist_session_if_needed empties the
        # holder on a successful flush, so this is the only chance to learn
        # what was held and restore it if the save fails partway through.
        held_scope = session.rag_scope_holder.scope

        def _write() -> str:
            conversation_id = self.persist_session_if_needed(session_id)
            if conversation_id is None:
                # Unreachable today: persist_session_if_needed's only
                # None-return branches (ephemeral, already-persisted,
                # no adapter) are all ruled out by the checks above and by
                # clearing `ephemeral` before this call. Raising rather than
                # returning None keeps this on the SAME rollback path as
                # every other failure, instead of silently leaving the
                # session non-ephemeral with no persisted conversation.
                raise RuntimeError(
                    "promote_ephemeral_session: persist_session_if_needed "
                    "unexpectedly returned None after ephemeral was cleared; "
                    "aborting the save."
                )
            for message in messages:
                self.persist_message_if_needed(message.id)
            return conversation_id

        session.ephemeral = False
        try:
            if callable(transaction):
                with transaction():
                    return _write()
            # No real database seam to wrap in a transaction (e.g. a
            # narrower persistence fake) -- production wiring always builds
            # ChatPersistenceService with a real CharactersRAGDB, so this
            # branch is not reachable there today, but the loss of the
            # all-or-nothing guarantee it causes must still be observable
            # rather than silent, matching the RAG-scope-flush warning just
            # above in persist_session_if_needed.
            logger.bind(session_id=session_id).warning(
                "Saving Console session {} without a database transaction "
                "-- the persistence adapter exposes no `db.transaction()` "
                "seam. A failure part-way through this save may leave a "
                "partial conversation in history instead of the "
                "all-or-nothing guarantee this method normally provides.",
                session_id,
            )
            return _write()
        except Exception:
            # persisted_conversation_id cleared BEFORE ephemeral is set back
            # to True so the two are never simultaneously in the one
            # combination the rest of the codebase treats as forbidden
            # (ephemeral=True with a non-None persisted_conversation_id),
            # even momentarily between statements.
            session.persisted_conversation_id = None
            session.ephemeral = True
            session.rag_scope_holder.set(held_scope)
            for message in messages:
                message.persisted_message_id = None
            logger.bind(session_id=session_id).exception(
                "Saving a temporary Console conversation failed; it stays temporary."
            )
            raise
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /private/tmp/ephemeral && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_ephemeral.py -v`
Expected: PASS (13 tests — 6 carried over from Tasks 1/2's vocabulary and gate tests, plus the 7 promotion tests above).

Also run the store/tree/sibling/branching regression suites, since this task touches shared tree-walk and persistence code:

```
cd /private/tmp/ephemeral && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Chat/test_console_chat_store.py \
  Tests/Chat/test_console_chat_store_parent_persist.py \
  Tests/Chat/test_console_chat_store_sibling.py \
  Tests/Chat/test_console_chat_store_summary.py \
  Tests/Chat/test_console_chat_store_tree.py \
  Tests/Chat/test_chat_persistence_service.py \
  Tests/Chat/test_console_terminal_citation_persistence.py \
  Tests/Chat/test_console_ephemeral.py \
  Tests/Chat/test_rag_scope_storage.py \
  Tests/Chat/test_console_regenerate_branching.py \
  Tests/Chat/test_console_edit_resend.py \
  -q
```

- [ ] **Step 5: Commit**

Shipped as three commits (initial implementation, then two review-fix rounds), not one:

```bash
cd /private/tmp/ephemeral
git add tldw_chatbook/Chat/console_chat_store.py Tests/Chat/test_console_ephemeral.py
git commit -m "feat: promote a temporary Console chat in one transaction

Clearing the flag is what opens the gate, so promotion clears it first,
then mints the conversation and flushes every message inside one DB
transaction. Any failure restores the temporary state so a failed save
never leaves a chat that silently starts persisting."
# round-1 review fixes (RAG scope restoration, no-transaction warning,
# defensive RuntimeError, except-block ordering): a second commit
# "fix: close four review gaps in temporary-chat promotion rollback"
# round-2 review fix (whole-tree promotion, this section's final state):
# a third commit -- see task-3-report.md for the exact SHAs.
```

---

### Task 4: `ephemeral` round-trips through screen state

The highest-severity risk in the spec. `save_state` serializes sessions as an explicit field list; omitting `ephemeral` means navigating Console → another screen → Console silently converts a temporary chat into a persisting one, and the next send writes it to the database.

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:12592-12613` (`_serialize_native_console_state`), `:12643-12715` (`_restore_native_console_state`)
- Test: `Tests/UI/test_console_native_chat_flow.py` (append)

**Interfaces:**
- Consumes: `ConsoleChatSession.ephemeral` from Task 2.
- Produces:
  - `ChatScreen._console_session_to_state(session) -> dict[str, Any]`
  - `ChatScreen._console_session_from_state(raw_session: dict[str, Any]) -> ConsoleChatSession`
  - the `"ephemeral"` key carried by both.

- [ ] **Step 1: Write the failing test**

Append to `Tests/UI/test_console_native_chat_flow.py`. This tests the two pure serialization halves directly rather than driving a screen, so it stays fast and has no app fixture:

```python
def test_console_screen_state_round_trips_the_temporary_flag():
    """A temporary chat must not become a persisting one by navigating away.

    `_serialize_native_console_state` writes an explicit field list; a field
    missing from it is silently dropped on restore. For `ephemeral` that
    drop is not cosmetic -- the next send would write the chat to the
    database.
    """
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    screen = ChatScreen.__new__(ChatScreen)

    # A REAL round trip: the serializer's own output feeds the restorer.
    # Asserting on a hand-built dict would test neither half.
    temporary = ConsoleChatSession(title="Temporary chat", ephemeral=True)
    payload = screen._console_session_to_state(temporary)
    assert payload["ephemeral"] is True
    assert screen._console_session_from_state(payload).ephemeral is True

    normal = ConsoleChatSession(title="Normal chat")
    assert screen._console_session_from_state(
        screen._console_session_to_state(normal)
    ).ephemeral is False

    # Legacy payloads predate the key entirely.
    assert screen._console_session_from_state(
        {"id": normal.id, "title": normal.title}
    ).ephemeral is False, "a payload with no key must default to saved"
```

This requires extracting **both** per-session halves into helpers. That extraction is the point: the round trip is currently untestable without a running app, which is exactly why a field is easy to drop from one side of it.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /private/tmp/ephemeral && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py -v -k temporary_flag`
Expected: FAIL — `AttributeError: 'ChatScreen' object has no attribute '_console_session_to_state'`

- [ ] **Step 3: Extract both per-session halves into helpers**

In `chat_screen.py`, extract the per-session **serialize** dict literal out of `_serialize_native_console_state` (lines 12597-12611). Keep every field byte-for-byte — this is a move, not a rewrite:

```python
    @staticmethod
    def _console_session_to_state(session: ConsoleChatSession) -> dict[str, Any]:
        """Serialize one ConsoleChatSession for screen-state restoration.

        Extracted from `_serialize_native_console_state` so the round trip
        is testable without a running app. This is an explicit field list:
        a field missing from it is silently dropped on the way back.
        """
        return {
            "id": session.id,
            "title": session.title,
            "workspace_id": session.workspace_id,
            "persisted_conversation_id": session.persisted_conversation_id,
            "draft": session.draft,
            "settings": ChatScreen._serialize_console_settings(session.settings),
            "updated_at": session.updated_at,
            "runtime_backend": session.runtime_backend,
            "assistant_kind": session.assistant_kind,
            "assistant_id": session.assistant_id,
            "assistant_authority_id": session.assistant_authority_id,
            "character_id": session.local_character_id(),
            "character_name": session.character_name,
        }
```

If `_serialize_console_settings` is an instance method rather than a static one, make `_console_session_to_state` an instance method too and call it as `self._serialize_console_settings(...)`; the test constructs the screen via `ChatScreen.__new__(ChatScreen)`, which supports either.

The comprehension in `_serialize_native_console_state` then reads:

```python
            "sessions": [
                self._console_session_to_state(session)
                for session in store.sessions()
            ],
```

Symmetrically, move the body of the `for raw_session in raw_sessions:` loop that builds `session_kwargs` and returns `ConsoleChatSession(**session_kwargs)` (lines 12642-12715) into:

```python
    def _console_session_from_state(self, raw_session: dict[str, Any]) -> ConsoleChatSession:
        """Rebuild one ConsoleChatSession from its serialized screen state.

        The mirror of `_console_session_to_state`. Every legacy-payload
        branch below exists because older saved states omit keys that newer
        ones carry -- keep them.
        """
```

The loop then reads:

```python
        for raw_session in raw_sessions:
            if not isinstance(raw_session, dict):
                continue
            session = self._console_session_from_state(raw_session)
```

- [ ] **Step 4: Add the field to both halves**

In `_console_session_to_state`, after `"character_name": session.character_name,`:

```python
            # Temporary conversations: without this key a temporary chat
            # comes back as a persisting one after any screen navigation,
            # and the next send writes it to the DB.
            "ephemeral": session.ephemeral,
```

In `_console_session_from_state`, before the `ConsoleChatSession(**session_kwargs)` construction:

```python
        # Legacy payloads predate the key; absent means saved, never temporary.
        session_kwargs["ephemeral"] = raw_session.get("ephemeral") is True
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /private/tmp/ephemeral && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py -q`
Expected: PASS, no new failures (the extraction must not change any existing behavior).

- [ ] **Step 6: Commit**

```bash
cd /private/tmp/ephemeral
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_native_chat_flow.py
git commit -m "fix: round-trip the temporary flag through Console screen state

save_state serializes an explicit field list, so an omitted field is
silently dropped -- for ephemeral that would convert a temporary chat
into a persisting one on any screen navigation. Extracts the per-session
restore so the round trip is testable without an app."
```

---

### Task 5: Born-temporary entry points

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py:1659-1670` (`new_session`)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:1386` (BINDINGS), `:1954` (new action), `:2177` (creator)
- Modify: `tldw_chatbook/UI/console_command_provider.py:42-46`
- Modify: `tldw_chatbook/Widgets/Console/console_session_surface.py:231-241`
- Test: `Tests/UI/test_console_composer_menu.py` (append — it already covers Console UI wiring for this feature family)

**Interfaces:**
- Consumes: `create_session(ephemeral=...)` from Task 2.
- Produces:
  - `ConsoleChatController.new_session(..., ephemeral: bool = False)`
  - `ChatScreen.action_new_temporary_console_tab() -> None`
  - `ChatScreen._create_native_console_session_from_active_context(*, ephemeral: bool = False)`
  - Button id `console-new-temporary-tab`

- [ ] **Step 1: Write the failing test**

Append to `Tests/UI/test_console_composer_menu.py`:

```python
@pytest.mark.unit
def test_temporary_tab_has_a_free_chord_and_a_palette_entry():
    """Alt+T must not collide, and the palette path must exist regardless.

    A chord that a terminal swallows is not a guaranteed path; the palette
    entry is. Both are asserted so neither can quietly disappear.
    """
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    keys = [b.key for b in ChatScreen.BINDINGS]
    assert keys.count("alt+t") == 1
    assert [b.action for b in ChatScreen.BINDINGS if b.key == "alt+t"] == [
        "new_temporary_console_tab"
    ]
    assert callable(ChatScreen.action_new_temporary_console_tab)


@pytest.mark.unit
def test_controller_new_session_can_be_born_temporary():
    """`ephemeral` reaches the store, not just the controller signature."""
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    store = ConsoleChatStore()
    assert store.create_session(title="A").ephemeral is False
    assert store.create_session(title="B", ephemeral=True).ephemeral is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /private/tmp/ephemeral && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_composer_menu.py -v -k temporary`
Expected: FAIL — `AttributeError: type object 'ChatScreen' has no attribute 'action_new_temporary_console_tab'`

- [ ] **Step 3: Thread `ephemeral` through the controller**

In `console_chat_controller.py`, `new_session` (line 1659):

```python
    def new_session(
        self,
        *,
        title: str | None = None,
        settings: ConsoleSessionSettings | None = None,
        ephemeral: bool = False,
    ) -> ConsoleChatSession:
        """Create and activate a new native Console session.

        Args:
            ephemeral: Create the session temporary -- never written to local
                storage until explicitly saved.
        """
        next_number = len(self.store.sessions()) + 1
        session = self.store.create_session(
            title=title or f"Chat {next_number}",
            settings=settings,
            ephemeral=ephemeral,
        )
```

- [ ] **Step 4: Add the screen action and binding**

In `chat_screen.py`, add after `action_new_console_tab` (line 1960):

```python
    def action_new_temporary_console_tab(self) -> None:
        """Open a temporary Console tab: never saved locally (Alt+T).

        Born temporary rather than converted: a chat that persists its first
        exchange and is made temporary afterwards has already written rows.
        """
        if self._console_setup_modal_blocking():
            return
        self.run_worker(
            self._create_native_console_session_from_active_context(ephemeral=True),
            exclusive=False,
        )
```

Add to `BINDINGS` immediately after the `ctrl+t` line (line 1386):

```python
        Binding("alt+t", "new_temporary_console_tab", "Temporary tab", show=False),
```

Change `_create_native_console_session_from_active_context` (line 2177) to accept the flag:

```python
    async def _create_native_console_session_from_active_context(
        self, *, ephemeral: bool = False
    ) -> None:
        """Create and focus a native Console session in the active workspace context.

        Args:
            ephemeral: Create the session temporary (never saved locally).
        """
```

and pass it through:

```python
        self._ensure_console_chat_controller().new_session(
            settings=(
                self._active_console_session_settings()
                or self._default_console_session_settings()
            ),
            ephemeral=ephemeral,
        )
```

- [ ] **Step 5: Add the palette entry**

In `console_command_provider.py`, after the "Console: New chat tab" tuple:

```python
            (
                "Console: New temporary chat",
                screen.action_new_temporary_console_tab,
                "Open a chat that is never saved locally (Alt+T)",
            ),
```

- [ ] **Step 6: Add the tab-strip button**

In `console_session_surface.py`, add after `_build_new_tab_button` (line 241):

```python
    def _build_new_temporary_tab_button(self) -> Button:
        """Return the tab-strip control for a chat that is never saved."""
        button = Button("Temporary", id="console-new-temporary-tab", compact=True)
        button.tooltip = "New temporary Console tab — not saved locally"
        for style, value in (
            ("width", CONSOLE_NEW_TAB_BUTTON_WIDTH),
            ("min_width", CONSOLE_NEW_TAB_BUTTON_WIDTH),
            ("max_width", CONSOLE_NEW_TAB_BUTTON_WIDTH),
            ("height", CONSOLE_NEW_TAB_BUTTON_HEIGHT),
            ("min_height", CONSOLE_NEW_TAB_BUTTON_HEIGHT),
            ("max_height", CONSOLE_NEW_TAB_BUTTON_HEIGHT),
        ):
            setattr(button.styles, style, value)
        return button
```

Mount it right after the new-tab button (line 476):

```python
            await tab_strip.mount(self._build_new_tab_button())
            await tab_strip.mount(self._build_new_temporary_tab_button())
```

Handle the press in `chat_screen.py`, beside the existing Console button handlers (near line 2195):

```python
    @on(Button.Pressed, "#console-new-temporary-tab")
    def on_console_new_temporary_tab(self, event: Button.Pressed) -> None:
        """Open a temporary Console tab from the tab strip."""
        event.stop()
        self.action_new_temporary_console_tab()
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd /private/tmp/ephemeral && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_composer_menu.py Tests/Chat/test_console_chat_controller.py -q`
Expected: PASS, no new failures.

- [ ] **Step 8: Commit**

```bash
cd /private/tmp/ephemeral
git add tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/console_command_provider.py tldw_chatbook/Widgets/Console/console_session_surface.py Tests/UI/test_console_composer_menu.py
git commit -m "feat: three ways to start a temporary Console chat

Command palette, a tab-strip button, and Alt+T. Born temporary rather
than converted -- a chat made temporary after its first exchange has
already written rows."
```

---

### Task 6: Presentation-only tab marker

**Spec deviation, deliberate.** The spec asks for `Temporary · <title>`. Console tabs render at `CONSOLE_SESSION_TAB_DISPLAY_CHARS = 19`, so a 12-character prefix would leave 7 characters of title and make every temporary tab look identical. This task uses a single glyph instead, decoded in the tooltip — the same mechanism `CONSOLE_RUN_MARKER_GLYPHS` already uses for fleet run markers, whose meanings `_session_tab_tooltip` already spells out. The spec's intent (the tab says it is temporary) is preserved; only the width cost changes.

**Files:**
- Modify: `tldw_chatbook/Chat/console_glyphs.py`
- Modify: `tldw_chatbook/Widgets/Console/console_session_surface.py:46-81` (`_session_tab_tooltip`), `:270-315` (`_build_session_tab_button`, `_tab_label`)
- Test: `Tests/UI/test_console_native_chat_flow.py` (append)

**Interfaces:**
- Consumes: `ConsoleChatSession.ephemeral` from Task 2.
- Produces: `GLYPH_TEMPORARY: str`; `_tab_label(title, *, marker=..., ephemeral=False)`.

- [ ] **Step 1: Write the failing test**

Append to `Tests/UI/test_console_native_chat_flow.py`:

```python
def test_temporary_tab_marker_is_presentation_only():
    """The marker must never enter session.title.

    Promotion saves `session.title` verbatim, so a marker written into the
    title would produce a saved conversation literally named after it -- and
    renaming would then fight the marker on every render.
    """
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession
    from tldw_chatbook.Chat.console_glyphs import GLYPH_TEMPORARY
    from tldw_chatbook.Widgets.Console.console_session_surface import (
        CONSOLE_SESSION_TAB_DISPLAY_CHARS,
        ConsoleSessionSurface,
        _session_tab_tooltip,
    )

    session = ConsoleChatSession(title="Vector store notes", ephemeral=True)
    label = ConsoleSessionSurface._tab_label(session.title, ephemeral=True)

    assert label.startswith(GLYPH_TEMPORARY)
    assert "Vector store" in label
    assert len(label) <= CONSOLE_SESSION_TAB_DISPLAY_CHARS + 2  # glyph + space
    assert GLYPH_TEMPORARY not in session.title

    plain = ConsoleSessionSurface._tab_label(session.title, ephemeral=False)
    assert GLYPH_TEMPORARY not in plain

    tooltip = _session_tab_tooltip(session, active=False)
    assert "not saved" in tooltip.lower()
    assert "not saved" not in _session_tab_tooltip(
        ConsoleChatSession(title="Normal"), active=False
    ).lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /private/tmp/ephemeral && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py -v -k presentation_only`
Expected: FAIL — `ImportError: cannot import name 'GLYPH_TEMPORARY'`

- [ ] **Step 3: Add the glyph**

In `tldw_chatbook/Chat/console_glyphs.py`, after `GLYPH_SOURCE_NOTE`:

```python
#: Temporary (never-saved) Console session tab marker. A dotted ring reads as
#: "outline of a thing, not the thing" -- deliberately unlike the solid run
#: markers above, which mean a run is happening. Decoded in the tab tooltip;
#: the 19-cell tab label has no room for a word.
GLYPH_TEMPORARY = "◌"
```

- [ ] **Step 4: Render it**

In `console_session_surface.py`, `_tab_label` (line 299):

```python
    @classmethod
    def _tab_label(
        cls,
        title: str,
        *,
        marker: ConsoleRunMarker = ConsoleRunMarker.NONE,
        ephemeral: bool = False,
    ) -> str:
```

and at the end of its body, replacing the current return:

```python
        label = cls._display_title(title)
        glyph = CONSOLE_RUN_MARKER_GLYPHS.get(marker, "")
        if glyph:
            label = f"{glyph} {label}"
        # Presentation only: never written into session.title, which
        # promotion saves verbatim.
        if ephemeral:
            label = f"{GLYPH_TEMPORARY} {label}"
        return label
```

Import `GLYPH_TEMPORARY` alongside the existing `GLYPH_CLOSE` import (line 22):

```python
from tldw_chatbook.Chat.console_glyphs import GLYPH_CLOSE, GLYPH_TEMPORARY
```

In `_build_session_tab_button` (line 282):

```python
            self._tab_label(
                session.title, marker=marker, ephemeral=session.ephemeral
            ),
```

- [ ] **Step 5: Decode it in the tooltip**

In `_session_tab_tooltip` (line 75), replace the assembly with:

```python
    meaning = CONSOLE_RUN_MARKER_MEANINGS.get(marker, "")
    tail = f" — {meaning}." if meaning else "."
    if active:
        text = f"Active Console tab: {session.title}{tail} Click again to rename."
    else:
        text = f"Switch to Console tab: {session.title}{tail}"
    if session.ephemeral:
        # The ◌ glyph carries no meaning on its own; this is where it is
        # decoded, exactly like the run-marker meanings above.
        text = f"{text} Temporary — not saved locally."
    return _escape_markup(text)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /private/tmp/ephemeral && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py -q`
Expected: PASS, no new failures.

- [ ] **Step 7: Commit**

```bash
cd /private/tmp/ephemeral
git add tldw_chatbook/Chat/console_glyphs.py tldw_chatbook/Widgets/Console/console_session_surface.py Tests/UI/test_console_native_chat_flow.py
git commit -m "feat: mark temporary Console tabs with a glyph, decoded in the tooltip

Deviates from the spec's 'Temporary · <title>' prefix: tab labels render
at 19 cells, so a 12-char prefix leaves 7 chars of title and every
temporary tab looks alike. Presentation only -- the marker never enters
session.title, which promotion saves verbatim."
```

---

### Task 7: The Temporary chip

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_status_chips.py:203-251`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (push the state where `sync_scope_chip` is already pushed)
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Test: `Tests/UI/test_console_composer_menu.py` (append)

**Interfaces:**
- Consumes: `TEMPORARY_LABEL`, `TEMPORARY_TOOLTIP` from Task 1; `session.ephemeral` from Task 2.
- Produces:
  - `ConsoleTemporaryChip` (posts `ConsoleTemporaryChip.SaveRequested`)
  - `ConsoleStatusChips.sync_temporary_chip(ephemeral: bool) -> None`
  - `ConsoleStatusChips._temporary_chip_render(ephemeral: bool) -> tuple[str, str, bool]`
  - chip id `console-temporary-chip`
  - `ChatScreen._console_active_session_is_ephemeral() -> bool` — the shared accessor Tasks 8 and 9 also use.

The strip widget is `ConsoleStatusChips(Horizontal)` (`console_status_chips.py:145`). Follow `ConsoleScopeChip` exactly: it is the established pattern for a chip that hides entirely when it has nothing to say, and it is synced by its own pushed method rather than through `ConsoleControlState` (whose equality-gated `sync_state` would need a new field).

- [ ] **Step 1: Write the failing test**

Append to `Tests/UI/test_console_composer_menu.py`:

```python
@pytest.mark.unit
def test_temporary_chip_is_hidden_outside_a_temporary_chat():
    """The chip says one thing; when it does not apply it vanishes."""
    from tldw_chatbook.Chat.console_ephemeral import TEMPORARY_LABEL
    from tldw_chatbook.Widgets.Console.console_status_chips import (
        ConsoleStatusChips,
    )

    label, tooltip, hidden = ConsoleStatusChips._temporary_chip_render(True)
    assert label == TEMPORARY_LABEL
    assert hidden is False
    assert "not saved" in tooltip.lower()

    _label, _tooltip, hidden_normal = ConsoleStatusChips._temporary_chip_render(False)
    assert hidden_normal is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /private/tmp/ephemeral && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_composer_menu.py -v -k temporary_chip`
Expected: FAIL — `AttributeError: ... has no attribute '_temporary_chip_render'`

- [ ] **Step 3: Add the chip class**

In `console_status_chips.py`, after `ConsoleScopeChip`:

```python
class ConsoleTemporaryChip(ConsoleChip):
    """Temporary-chat chip that doubles as the "Save this chat" action.

    Same activation contract as the sibling action chips: Enter/Space while
    focused, or a click. The chip is the marker AND the escape hatch, so the
    user never has to remember where saving lives.
    """

    BINDINGS = [
        Binding("enter", "save_chat", "Save this chat", show=False),
        Binding("space", "save_chat", "Save this chat", show=False),
    ]

    class SaveRequested(Message):
        """Posted when the temporary chip is activated."""

    def action_save_chat(self) -> None:
        self.post_message(self.SaveRequested())

    def _on_click(self, event: events.Click) -> None:
        self.post_message(self.SaveRequested())
```

- [ ] **Step 4: Compose and sync it**

Add the import at the top of `console_status_chips.py`:

```python
from tldw_chatbook.Chat.console_ephemeral import TEMPORARY_LABEL, TEMPORARY_TOOLTIP
```

Add an `ephemeral` attribute initialised to `False` alongside `scope_state` in the widget's `__init__`, then yield the chip first in `compose` — the marker belongs before the settings chips, not after them:

```python
    def compose(self) -> ComposeResult:
        # First: this is a property of the whole chat, not one setting.
        yield self._temporary_chip()
        yield self._chip(
            self.state.provider_label,
            id="console-provider-chip",
            chip_class=ConsoleModelChip,
        )
```

and add:

```python
    def _temporary_chip(self) -> ConsoleTemporaryChip:
        label, tooltip, hidden = self._temporary_chip_render(self.ephemeral)
        chip = self._chip(
            label,
            id="console-temporary-chip",
            chip_class=ConsoleTemporaryChip,
        )
        chip.tooltip = tooltip
        chip.display = not hidden
        return chip

    @staticmethod
    def _temporary_chip_render(ephemeral: bool) -> tuple[str, str, bool]:
        """Pure ``(label, tooltip, hidden)`` render for the temporary chip.

        Args:
            ephemeral: Whether the active session is temporary.

        Returns:
            ``label``: chip text. ``tooltip``: hover/focus text, which is
            where the save affordance is spelled out. ``hidden``: ``True``
            for a normal chat -- a "Saved" chip on every ordinary
            conversation would be noise, and the strip is width-bounded.
        """
        if not ephemeral:
            return TEMPORARY_LABEL, TEMPORARY_TOOLTIP, True
        return TEMPORARY_LABEL, TEMPORARY_TOOLTIP, False

    def sync_temporary_chip(self, ephemeral: bool) -> None:
        """Refresh the temporary chip from the active session's flag.

        Separate from ``sync_state`` for the same reason ``sync_scope_chip``
        is: this is pushed from the screen when the active session changes,
        not on every control-bar sync tick.

        Args:
            ephemeral: Whether the active session is temporary.
        """
        if ephemeral == self.ephemeral:
            return
        self.ephemeral = ephemeral
        try:
            chip = self.query_one("#console-temporary-chip", ConsoleTemporaryChip)
        except NoMatches:
            return
        label, tooltip, hidden = self._temporary_chip_render(ephemeral)
        chip.update(label)
        chip.tooltip = tooltip
        chip.display = not hidden
```

- [ ] **Step 5: Push the state from the screen**

`ConsoleChatStore` has no public single-session getter — `_session_or_raise` is private — so add this accessor to `chat_screen.py` beside `_current_console_session_id` (line 9193). Tasks 8 and 9 use it too; define it once here:

```python
    def _console_active_session_is_ephemeral(self) -> bool:
        """Return whether the active Console session is temporary.

        Public-API only (`sessions()` + `active_session_id`): the store has
        no single-session getter that is not private. The scan is over open
        tabs, so it is a handful of items.
        """
        store = self._console_chat_store
        if store is None:
            return False
        active_id = store.active_session_id
        if not active_id:
            return False
        return any(
            session.id == active_id and session.ephemeral
            for session in store.sessions()
        )
```

Then find every call site of `sync_scope_chip` (`grep -n "sync_scope_chip" tldw_chatbook/UI/Screens/chat_screen.py`) and add a `sync_temporary_chip` call beside each:

```python
            strip.sync_temporary_chip(self._console_active_session_is_ephemeral())
```

- [ ] **Step 6: Style it**

In `tldw_chatbook/css/components/_agentic_terminal.tcss`, beside the existing `.console-control-chip` rules:

```css
#console-temporary-chip {
    color: $warning;
    text-style: italic;
}
```

Then regenerate the bundle — **never hand-edit `tldw_cli_modular.tcss`**:

```bash
cd /private/tmp/ephemeral && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd /private/tmp/ephemeral && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_composer_menu.py Tests/Chat/test_console_display_state.py -q`
Expected: PASS, no new failures.

- [ ] **Step 8: Commit**

```bash
cd /private/tmp/ephemeral
git add tldw_chatbook/Widgets/Console/console_status_chips.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_console_composer_menu.py
git commit -m "feat: Temporary chip in the Console status strip

Hidden entirely in a normal chat, following ConsoleScopeChip. Doubles as
the save action so the escape hatch sits on the marker itself."
```

---

### Task 8: Promotion UI

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_composer_menu_modal.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:5100-5168` (menu open + dispatch)
- Test: `Tests/UI/test_console_composer_menu.py` (append)

**Interfaces:**
- Consumes: `ACTION_SAVE_CHAT` (Task 1), `promote_ephemeral_session` (Task 3), `ConsoleTemporaryChip.SaveRequested` (Task 7).
- Produces: `build_composer_menu_entries(..., ephemeral: bool = False)`; `ChatScreen._promote_console_temporary_session()`.

- [ ] **Step 1: Write the failing test**

Append to `Tests/UI/test_console_composer_menu.py`:

```python
@pytest.mark.unit
def test_save_this_chat_appears_only_in_a_temporary_chat():
    """The entry is meaningless in a normal chat, so it is absent, not disabled.

    This is the one case where hiding beats disabling: a disabled "Save this
    chat" on an already-saved conversation would read as a failure.
    """
    from tldw_chatbook.Chat.console_ephemeral import ACTION_SAVE_CHAT

    normal = [e.action_id for e in build_composer_menu_entries()]
    assert ACTION_SAVE_CHAT not in normal

    temporary = build_composer_menu_entries(ephemeral=True)
    ids = [e.action_id for e in temporary]
    assert ids[0] == ACTION_SAVE_CHAT, "the escape hatch goes first"
    entry = temporary[0]
    assert entry.enabled is True
    assert "not saved" in entry.description.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /private/tmp/ephemeral && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_composer_menu.py -v -k save_this_chat`
Expected: FAIL — `TypeError: build_composer_menu_entries() got an unexpected keyword argument 'ephemeral'`

- [ ] **Step 3: Add the menu entry**

In `console_composer_menu_modal.py`, import the action id and extend the builder:

```python
from tldw_chatbook.Chat.console_ephemeral import ACTION_SAVE_CHAT
```

```python
def build_composer_menu_entries(
    *, attachment_kind: str = "none", ephemeral: bool = False
) -> tuple[ComposerMenuEntry, ...]:
    """Build the menu rows for the current composer state.

    Generate Caption is disabled -- never hidden -- when it cannot act, and
    the row says which case applies: nothing staged, or a staged file that
    is not an image. Explicit unavailable states beat vanishing entries.

    "Save this chat" is the exception: it is ABSENT outside a temporary
    chat rather than disabled, because a disabled save on an already-saved
    conversation reads as a failure rather than as "already done".

    Args:
        attachment_kind: ``"image"``, ``"other"``, or ``"none"``.
        ephemeral: Whether the active session is temporary.

    Returns:
        The menu entries in display order.
    """
```

Build the base tuple as today, then prepend when temporary:

```python
    entries = (
        ComposerMenuEntry(
            ACTION_GENERATE_IMAGE,
            "Generate Image",
            "Build a /generate-image command",
        ),
        ...  # unchanged
    )
    if not ephemeral:
        return entries
    return (
        ComposerMenuEntry(
            ACTION_SAVE_CHAT,
            "Save this chat",
            "This chat is not saved locally — save it now",
        ),
        *entries,
    )
```

Pass the flag at the open site in `chat_screen.py` (line ~5110), reading the active session's flag the same way `_console_pending_attachment_kind` reads the store.

- [ ] **Step 4: Add the dispatch and the handler**

Add the two new imports to `chat_screen.py` alongside the existing composer-menu action imports:

```python
from tldw_chatbook.Chat.console_ephemeral import ACTION_SAVE_CHAT
from tldw_chatbook.Widgets.Console.console_status_chips import ConsoleTemporaryChip
```

In `_handle_console_composer_menu_choice` (line 5146), add before the `ACTION_GENERATE_IMAGE` branch:

```python
        if action_id == ACTION_SAVE_CHAT:
            self._promote_console_temporary_session()
            return
```

Add the handler beside it:

```python
    def _promote_console_temporary_session(self) -> None:
        """Save the active temporary chat, then refresh its marker and chip.

        Both entry points (the composer menu row and the Temporary chip) land
        here, so the save behaves identically however it was reached.
        """
        store = self._ensure_console_chat_store()
        session_id = getattr(store, "active_session_id", None)
        if not session_id:
            return
        try:
            conversation_id = store.promote_ephemeral_session(session_id)
        except Exception:
            logger.opt(exception=True).warning("Saving the temporary chat failed")
            self.app_instance.notify(
                "Could not save this chat. It is still temporary.",
                severity="error",
            )
            return
        if conversation_id is None:
            return
        self._invalidate_console_persisted_rows_cache()
        self.run_worker(self._sync_native_console_chat_ui(), exclusive=False)
        self.app_instance.notify("Chat saved.", severity="information")

    @on(ConsoleTemporaryChip.SaveRequested)
    def on_console_temporary_chip_save(
        self, event: ConsoleTemporaryChip.SaveRequested
    ) -> None:
        """Save the temporary chat from its status chip."""
        event.stop()
        self._promote_console_temporary_session()
```

`_sync_native_console_chat_ui` is a coroutine — it must be run as a worker, never called bare. A previously shipped no-op in this file came from awaiting it incorrectly.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /private/tmp/ephemeral && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_composer_menu.py -q`
Expected: PASS, no new failures.

- [ ] **Step 6: Commit**

```bash
cd /private/tmp/ephemeral
git add tldw_chatbook/Widgets/Console/console_composer_menu_modal.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_composer_menu.py
git commit -m "feat: Save this chat, from the composer menu or the Temporary chip

Both entry points land in one handler. The menu row is absent rather than
disabled outside a temporary chat -- a disabled save on an already-saved
conversation reads as a failure, not as 'already done'."
```

---

### Task 9: Block the artifact-producing actions

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_composer_menu_modal.py` (Generate Image)
- Modify: `tldw_chatbook/Widgets/Console/console_workbench_state.py:16-95` (Save Chatbook)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (pass the flag into `build_console_workbench_state`; Save as Note/Media/Prompt/Chatbook destinations; Save Context button)
- Modify: `tldw_chatbook/Chat/console_message_actions.py` (Save Image, task 1 audit addendum below)
- Modify: `tldw_chatbook/Widgets/Console/console_context_modal.py` (Save Context, task 1 audit addendum below)
- Test: `Tests/UI/test_console_composer_menu.py` (append)
- Test: `Tests/Chat/test_console_message_actions.py` (append, task 1 audit addendum below)
- Test: `Tests/UI/test_console_native_chat_flow.py` (append, task 1 audit addendum below)
- Test: `Tests/UI/test_console_context_modal.py` (append, task 1 audit addendum below)

**Interfaces:**
- Consumes: `blocked_reason` (Task 1), `session.ephemeral` (Task 2).
- Produces: `build_console_workbench_state(..., ephemeral: bool = False)`.

**Scope note (task 1 audit):** the sink audit in Task 1 found six more
local-write sinks beyond Generate Image and Save Chatbook —
`EPHEMERAL_BLOCKED_ACTIONS` now has eight entries, not two. Steps 1-7 below
(as originally planned) cover only the first two. Steps 8-11 below are the
addendum that covers the other six: `save-image`, `save-as-note`,
`save-as-media`, `save-as-prompt`, `save-as-chatbook`, `save-context`. This
task is not complete until all eight are wired.

- [ ] **Step 1: Write the failing test**

Append to `Tests/UI/test_console_composer_menu.py`:

```python
@pytest.mark.unit
def test_artifact_actions_are_disabled_with_a_reason_in_a_temporary_chat():
    """Disabled and explained, never hidden -- and still enabled normally.

    The second half is the control: an assertion that an action is disabled
    proves nothing unless the same call proves it is enabled otherwise.
    """
    from tldw_chatbook.Chat.console_ephemeral import blocked_reason
    from tldw_chatbook.Widgets.Console.console_workbench_state import (
        build_console_workbench_state,
    )
    from tldw_chatbook.Chat.console_display_state import ConsoleControlState

    menu = {
        e.action_id: e
        for e in build_composer_menu_entries(ephemeral=True)
    }
    image = menu[ACTION_GENERATE_IMAGE]
    assert image.enabled is False
    assert image.description == blocked_reason("generate-image", ephemeral=True)

    normal = {e.action_id: e for e in build_composer_menu_entries()}
    assert normal[ACTION_GENERATE_IMAGE].enabled is True

    # ConsoleControlState has seven required label fields and no defaults.
    control_state = ConsoleControlState(
        provider_label="Provider: stub",
        model_label="Model: stub",
        assistant_label="Assistant: General",
        rag_label="RAG: off",
        sources_label="Sources: 0",
        tools_label="Tools: 0",
        approvals_label="Approvals: 0",
    )

    def chatbook_action(**kwargs):
        state = build_console_workbench_state(
            control_state=control_state, can_save_chatbook=True, **kwargs
        )
        return {a.id: a for a in state.actions}["save-chatbook"]

    blocked = chatbook_action(ephemeral=True)
    assert blocked.disabled is True
    assert blocked.tooltip == blocked_reason("save-chatbook", ephemeral=True)
    assert chatbook_action().disabled is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /private/tmp/ephemeral && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_composer_menu.py -v -k artifact_actions`
Expected: FAIL — `TypeError: build_console_workbench_state() got an unexpected keyword argument 'ephemeral'`

- [ ] **Step 3: Block Generate Image in the menu**

In `console_composer_menu_modal.py`, import `blocked_reason` and use it for the Generate Image entry:

```python
from tldw_chatbook.Chat.console_ephemeral import ACTION_SAVE_CHAT, blocked_reason
```

```python
    image_blocked = blocked_reason(ACTION_GENERATE_IMAGE, ephemeral=ephemeral)
    entries = (
        ComposerMenuEntry(
            ACTION_GENERATE_IMAGE,
            "Generate Image",
            image_blocked or "Build a /generate-image command",
            enabled=image_blocked is None,
        ),
        ...
```

`ACTION_GENERATE_IMAGE` is already the string `"generate-image"`, which is the registry key — no translation needed.

- [ ] **Step 4: Block Save Chatbook in the workbench state**

In `console_workbench_state.py`, add the keyword and use it:

```python
    run_active: bool = False,
    ephemeral: bool = False,
) -> WorkbenchState:
```

Document it in the docstring's Args:

```python
        ephemeral: Whether the active session is temporary, which blocks the
            actions that would write a derived artifact to disk.
```

and in the `actions` tuple:

```python
    chatbook_blocked = blocked_reason("save-chatbook", ephemeral=ephemeral)
```

```python
        WorkbenchAction(
            id="save-chatbook",
            label="Save Chatbook",
            tooltip=chatbook_blocked or "Save this run as a Chatbook",
            disabled=chatbook_blocked is not None or not can_save_chatbook,
        ),
```

with the import at the top:

```python
from tldw_chatbook.Chat.console_ephemeral import blocked_reason
```

- [ ] **Step 5: Pass the flag from the screen**

In `chat_screen.py`, `_build_console_workbench_state` (line 10677) ends in a single `build_console_workbench_state(...)` call (line 10695). Add one keyword to it, using the accessor defined in Task 7 Step 5:

```python
            run_active=self._console_run_active(),
            ephemeral=self._console_active_session_is_ephemeral(),
        )
```

Also pass it at the composer-menu open site from Task 8 Step 3, so both consumers read the same accessor.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /private/tmp/ephemeral && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_composer_menu.py Tests/Chat/test_console_ephemeral.py -q`
Expected: PASS, no new failures.

- [ ] **Step 7: Commit**

```bash
cd /private/tmp/ephemeral
git add tldw_chatbook/Widgets/Console/console_composer_menu_modal.py tldw_chatbook/Widgets/Console/console_workbench_state.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_composer_menu.py
git commit -m "feat: block artifact-producing actions in a temporary chat

Generate Image and Save Chatbook disable with the reason on the control,
sourced from the single blocked-action registry so a new sink means one
new row rather than a new hunt through call sites."
```

**Addendum (task 1 audit): the other six sinks**

- [ ] **Step 8: Block Save Image (message-action row)**

`ConsoleMessageActionService.available_actions` (`console_message_actions.py:116`)
has no `ephemeral` parameter today; `_action_enabled` and
`_action_disabled_reason` (lines 427, 444) are the two static methods that
decide, per `action_id`, whether a row action is enabled and what its
disabled reason is — the `"variant-previous"`/`"variant-next"` handling
there is the existing pattern to extend, not replace.

Write a failing test in `Tests/Chat/test_console_message_actions.py` that
builds a message with an image attachment, calls `available_actions(...,
ephemeral=True)`, and asserts the `"save-image"` entry has `enabled=False`
and `disabled_reason == blocked_reason("save-image", ephemeral=True)`, plus
the control: `ephemeral=False` (or omitted) leaves it enabled. Run it, watch
it fail on the unexpected keyword, then thread `ephemeral: bool = False`
through `available_actions` → `_action_enabled` → `_action_disabled_reason`,
consulting `blocked_reason("save-image", ephemeral=ephemeral)` the same way
`_action_disabled_reason` already special-cases `"regenerate"`. `chat_screen.py`
calls `available_actions(...)` when building the message-action row; pass
`ephemeral=self._console_active_session_is_ephemeral()` (Task 7's accessor)
at that call site. Run the test again to confirm it passes.

- [ ] **Step 9: Block the four Save as... destinations**

`_console_save_as_destinations` (`chat_screen.py:16622`) builds
`available_destinations`/`unavailable_save_reasons` keyed by label
(`"Chatbook"`, `"Note"`, `"Media"`, `"Prompt"`) before constructing
`ConsoleMessageActionService(...).save_as_destinations(message)`. In a
temporary chat every destination is unavailable regardless of service
readiness — write a failing test in `Tests/UI/test_console_native_chat_flow.py`
(or wherever the existing save-as destination tests live; `grep -rn
"save_as_destinations\|_console_save_as_destinations"` finds them) asserting
that with an ephemeral session, all four destinations come back
`available=False` with `reason == blocked_reason("save-as-<label
lowercased>", ephemeral=True)` (e.g. `"save-as-note"` for `"Note"`), and the
control: a non-ephemeral session with all three services wired still returns
the pre-existing availability. Watch it fail, then in
`_console_save_as_destinations`, when
`self._console_active_session_is_ephemeral()` is true, skip the
service-readiness checks and populate `unavailable_reasons` for all four
labels from the registry instead (service readiness is moot if the write
itself is blocked). Run the test again to confirm it passes.

- [ ] **Step 10: Block Save Context**

`ConsoleContextModal` (`console_context_modal.py`) has no `ephemeral`
constructor argument; `_save_json` (line 287) is the handler behind the
`#console-context-save` button. Write a failing test in
`Tests/UI/test_console_context_modal.py` constructing the modal with
`ephemeral=True` and asserting the save button is `disabled` with its
`tooltip` equal to `blocked_reason("save-context", ephemeral=True)` (mirror
the disabled-with-reason pattern the composer menu already uses for
Generate Caption), plus the control: `ephemeral=False` leaves it enabled.
Watch it fail, then add the `ephemeral: bool = False` constructor kwarg,
apply it to the `#console-context-save` button in `compose()`, and pass
`ephemeral=self._console_active_session_is_ephemeral()` from
`action_view_chat_context` (`chat_screen.py:2009`) where the modal is
constructed. Run the test again to confirm it passes.

- [ ] **Step 11: Run the full addendum test set and commit**

Run: `cd /private/tmp/ephemeral && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_message_actions.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_context_modal.py Tests/Chat/test_console_ephemeral.py -q`
Expected: PASS, no new failures.

```bash
cd /private/tmp/ephemeral
git add tldw_chatbook/Chat/console_message_actions.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Widgets/Console/console_context_modal.py Tests/Chat/test_console_message_actions.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_context_modal.py
git commit -m "feat: block the six sinks the task-1 audit added

Save Image, the four Save as... destinations, and Save Context all
disable with a reason in a temporary chat, closing the gap between the
two originally-known sinks and the audited EPHEMERAL_BLOCKED_ACTIONS
registry."
```

---

### Task 10: Full suite, then live verification

Nothing in tasks 1-9 proves the feature works in a running terminal. Every prior program in this repo that skipped this step shipped at least one defect the tests could not see — most recently a chip that opened its modal correctly and then discarded the result.

**Files:**
- Modify: `Docs/superpowers/specs/2026-07-31-temporary-conversations-design.md` (record verification results)

- [ ] **Step 1: Run the full Console and Chat suites**

```bash
cd /private/tmp/ephemeral
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/ Tests/UI/ -q -n 8
```
Expected: PASS with no new failures against the `origin/dev` baseline. If anything fails, compare against a clean `origin/dev` run before assuming it is yours.

- [ ] **Step 2: Launch the app against a scratch config**

The app writes to the real user config and databases. Use an isolated profile so a temporary-chat experiment can never touch live data:

```bash
cd /private/tmp/ephemeral
mkdir -p /private/tmp/ephemeral-verify
printf '[general]\nusers_name = "verify_ephemeral"\n' > /private/tmp/ephemeral-verify/config.toml
tmux -L ephemeral new-session -d -x 235 -y 52 \
  'TLDW_CONFIG_PATH=/private/tmp/ephemeral-verify/config.toml /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m tldw_chatbook.app'
sleep 12
tmux -L ephemeral capture-pane -p | head -8
```

- [ ] **Step 3: Verify Alt+T actually reaches the app**

`Ctrl+<digit>` cannot be sent through tmux, but `Alt+<letter>` can:

```bash
tmux -L ephemeral send-keys M-t
sleep 1
tmux -L ephemeral capture-pane -p | grep -i "temporary\|◌"
```

Expected: a new tab carrying the `◌` marker, and the `Temporary — not saved` chip in the status strip.

**If Alt+T does not reach the app**, that is a real finding, not a test-harness problem: remove the binding, keep the palette and tab-strip paths, and record the result in the spec. Do not leave a binding in the code that does nothing.

- [ ] **Step 4: Verify no rows are written, against the live database**

Send a message in the temporary tab, then query the scratch database directly:

```bash
tmux -L ephemeral send-keys -l "hello from a temporary chat"
tmux -L ephemeral send-keys Enter
sleep 5
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python - <<'PY'
import sqlite3, pathlib
root = pathlib.Path.home() / ".local/share/tldw_cli/verify_ephemeral"
for db in root.rglob("*.sqlite*"):
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "conversations" in tables:
            print(db.name,
                  "conversations:", conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0],
                  "messages:", conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
    except sqlite3.Error as exc:
        print(db.name, "unreadable:", exc)
PY
```

Expected: `conversations: 0 messages: 0`. **Run the same check after a normal chat in the same session** — it must show non-zero. A zero-vs-zero result proves nothing.

- [ ] **Step 5: Verify promotion end to end**

Activate the Temporary chip (click via SGR mouse escape at the chip's column from a `capture-pane` dump, press and release ~0.3s apart), then re-run the row-count script. Expected: the conversation and its messages now exist, the `◌` marker is gone, and the chip has disappeared.

- [ ] **Step 6: Verify what stays available, and what the switcher shows**

Two spec claims that only a running app can settle:

1. **RAG retrieval still works.** In a temporary chat, run a retrieval (`Run Library RAG`, or a `/rag` query) and confirm results come back. Reading the index stores nothing, so it must not be blocked — a temporary chat that cannot retrieve is a regression, not a safety feature.
2. **The switcher does not list it.** Open the session switcher (`Ctrl+K`) and confirm the temporary chat is absent from the persisted-conversation list while its tab is still open and switchable. There is no conversation row for it to appear as.

- [ ] **Step 7: Tear down and record**

```bash
tmux -L ephemeral send-keys C-q
tmux -L ephemeral kill-server
rm -rf ~/.local/share/tldw_cli/verify_ephemeral /private/tmp/ephemeral-verify
```

Append a `## Live verification` section to the spec recording, for each of steps 3-6: what was run, what was observed, and — where a control existed — the control's result. Record failures as failures; a step that could not be completed is recorded as not completed, not omitted.

- [ ] **Step 8: Commit**

```bash
cd /private/tmp/ephemeral
git add Docs/superpowers/specs/2026-07-31-temporary-conversations-design.md
git commit -m "docs: record live verification of temporary conversations"
```

---

## Notes for the implementer

**The one invariant.** A session is never both `ephemeral` and holding a `persisted_conversation_id`. Two places can violate it: `persist_session_if_needed` (guarded in Task 2) and `restore_persisted_session` (guarded in Task 2 Step 6). If you add a third assignment to `persisted_conversation_id`, guard it there too.

**Assertions about absence need a control.** "No rows were written" and "the action is disabled" both pass trivially against a broken harness. Every such assertion in this plan is paired with a positive case in the same test, in the same harness. Keep that pairing if you restructure the tests.

**"The UI responded" is not evidence the action took effect.** This repo has shipped three separate defects where a modal opened, a chip highlighted, or a menu closed while the underlying write silently did nothing. Task 10 checks the database, not the screen.
