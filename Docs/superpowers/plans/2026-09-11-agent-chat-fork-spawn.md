# Agent Chat Fork & Spawn Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `fork_chat` and `new_chat` runtime agent tools that let a Console agent create confirmed, background workstream chats for the user (fork = verbatim active-path copy with lineage columns; new = fresh chat).

**Architecture:** Both tools ride the existing injected-callable runtime-tool seam (`install_skill`/`run_skill_script` pattern): bridge closures in `run_reply` do policy validation → blocking per-call confirm → controller-side execution; `AgentService` pins schemas for primary runs and `agent_runtime` dispatches to injected `LoopDeps` callables. Fork copying is a new pure-ish service method over the existing conversation tree read; sessions surface via a non-activating restore plus the standard console-sync worker.

**Tech Stack:** Python ≥3.11, Textual 8.x, SQLite (FTS5) via `ChaChaNotes_DB`, pytest with real in-memory SQLite.

**Spec:** `Docs/superpowers/specs/2026-09-11-agent-chat-fork-spawn-design.md` — read it first; this plan argues from it.
**ADR:** `backlog/decisions/150-agent-chat-fork-and-spawn.md` · **Task:** TASK-32482 · **Follow-ups:** TASK-32480 (sub-agents), preset/provider integration (post ADR-147).

## Global Constraints

- **No DB schema migration** in either database. The lineage columns `conversations.parent_conversation_id` / `forked_from_message_id` already exist (`DB/ChaChaNotes_DB.py:468-469`).
- **Advertised must equal usable** (the #847 lesson, restated at `console_chat_controller.py:12114-12126`): only inject a confirm/execute callable when a UI sink is wired, so the bridge never builds a tool the model can never succeed with.
- **Primary agents only**: every schema pin and `LoopDeps` population for these tools is gated on `agent_kind == AGENT_KIND_PRIMARY` (spec §Architecture; sub-agents are TASK-32480).
- **Fail closed**: any UI error, no-UI, cancel, or timeout in a confirm round denies the call and returns a `ToolResult(ok=False, ...)`; never an exception across the tool seam.
- **The working tree carries unrelated in-flight WIP** (provider-routing PR). Every commit step stages ONLY the files its task lists — never `git add -A`.
- **Targeted tests only**: run the test files each task names. Do not run the full suite unless the user asks (AGENTS.md testing rule).
- Tool result payloads are JSON strings; opening prompts are drafts (`store.set_session_draft`), never auto-sent.
- The fork copy read must use raised caps (`root_limit=10_000, depth_cap=10_000` — the same policy as `console_conversation_hydration.load_console_conversation_tree`); silent truncation is a bug.

**Anchor-drift warning:** line numbers below were verified against the working tree on 2026-09-11 with the uncommitted provider-routing WIP present. If anchors drift, locate the named symbol (`grep -n`) and apply the change there; the structures are stable.

---

### Task 1: Lineage passthrough in `ChatPersistenceService.create_conversation`

**Files:**
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py:216-331` (`create_conversation`)
- Test: `Tests/Chat/test_chat_persistence_service.py` (reuse its existing real in-memory-DB fixture)

**Interfaces:**
- Consumes: `self.db.add_conversation(conv_data)` which already accepts `forked_from_message_id` / `parent_conversation_id` as passthrough keys (`DB/ChaChaNotes_DB.py:7725-7764`).
- Produces: `create_conversation(..., parent_conversation_id: str | None = None, forked_from_message_id: str | None = None) -> str` — later tasks (7) call it with lineage kwargs. The `ConsoleChatPersistence` protocol (`Chat/console_chat_store.py:205-220`) declares `create_conversation(self, **kwargs) -> str`, so no protocol change.

- [ ] **Step 1: Write the failing test** (append to `Tests/Chat/test_chat_persistence_service.py`; reuse that file's existing db/service fixture names — check the top of the file and use its pattern for constructing the service)

```python
def test_create_conversation_records_fork_lineage(persistence_service_with_db):
    db, service = persistence_service_with_db  # adapt names to the file's fixture
    parent_id = service.create_conversation(conversation_title="Source")
    child_id = service.create_conversation(
        conversation_title="Fork of Source",
        parent_conversation_id=parent_id,
        forked_from_message_id="msg-abc",
    )
    row = db.get_conversation_by_id(child_id)
    assert row is not None
    assert row["parent_conversation_id"] == parent_id
    assert row["forked_from_message_id"] == "msg-abc"
    assert row["title"] == "Fork of Source"


def test_create_conversation_without_lineage_unchanged(persistence_service_with_db):
    db, service = persistence_service_with_db
    conv_id = service.create_conversation(conversation_title="Plain")
    row = db.get_conversation_by_id(conv_id)
    assert row["parent_conversation_id"] is None
    assert row["forked_from_message_id"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest Tests/Chat/test_chat_persistence_service.py -k lineage -v`
Expected: FAIL with `TypeError: create_conversation() got an unexpected keyword argument 'parent_conversation_id'`

- [ ] **Step 3: Implement** — in `create_conversation`'s signature add after `speech_preferences`:

```python
        parent_conversation_id: Optional[str] = None,
        forked_from_message_id: Optional[str] = None,
```

and inside the body, where `conversation_data` is built (before the `metadata` json.dumps line), add:

```python
        if parent_conversation_id is not None:
            conversation_data["parent_conversation_id"] = parent_conversation_id
        if forked_from_message_id is not None:
            conversation_data["forked_from_message_id"] = forked_from_message_id
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest Tests/Chat/test_chat_persistence_service.py -v`
Expected: PASS (all, including pre-existing)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/chat_persistence_service.py Tests/Chat/test_chat_persistence_service.py
git commit -m "feat: fork lineage passthrough in console persistence create_conversation"
```

---

### Task 2: `copy_conversation_active_path` on `ChatConversationService`

**Files:**
- Modify: `tldw_chatbook/Chat/chat_conversation_service.py` (new method near `get_conversation_tree`, ~line 1112)
- Test: `Tests/Chat/test_chat_conversation_service.py`

**Interfaces:**
- Consumes: `self.get_conversation_tree(conversation_id, root_limit=..., depth_cap=...)` (same file, :1059) returning `{"conversation": dict, "root_threads": [nodes]}` where each node carries `id, parent_message_id, sender, content, role, timestamp, image_data, image_mime_type, usage_json, metadata_json, provider_continuation_json, children`; `self.db.get_conversation_active_leaf`, `self.db.add_message`, `self.db.set_conversation_active_leaf`.
- Produces: `copy_conversation_active_path(self, source_conversation_id: str, target_conversation_id: str) -> dict` returning `{"copied": int, "leaf_message_id": str | None}`; raises `ValueError("empty_history")` when the source has no messages. Task 7's executor calls this.

- [ ] **Step 1: Write the failing tests** (append to `Tests/Chat/test_chat_conversation_service.py`, reusing its existing db/service fixture pattern)

```python
def _seed_chain(db, service, conv_id, texts):
    """Seed a linear parent->child chain; returns message ids in order."""
    ids = []
    parent = None
    for sender, text in texts:
        mid = db.add_message(
            {
                "conversation_id": conv_id,
                "sender": sender,
                "content": text,
                "parent_message_id": parent,
            }
        )
        ids.append(str(mid))
        parent = mid
    db.set_conversation_active_leaf(conv_id, ids[-1])
    return ids


def test_copy_active_path_copies_and_remaps(service_with_db):
    db, service = service_with_db  # adapt to the file's fixture names
    src = service.create_conversation(title="Src")
    ids = _seed_chain(db, service, src, [("user", "hello"), ("assistant", "hi"), ("user", "go")])
    dst = service.create_conversation(title="Dst")

    outcome = service.copy_conversation_active_path(src, dst)

    assert outcome["copied"] == 3
    copied = db.get_messages_for_conversation(dst)
    assert [m["content"] for m in copied] == ["hello", "hi", "go"]
    assert [m["sender"] for m in copied] == ["user", "assistant", "user"]
    # parents remapped: each copied message's parent is the previous copied one
    assert copied[0]["parent_message_id"] is None
    assert copied[1]["parent_message_id"] == copied[0]["id"]
    assert copied[2]["parent_message_id"] == copied[1]["id"]
    # ids are fresh, not the source's
    assert {m["id"] for m in copied}.isdisjoint(set(ids))
    # active leaf points at the copied leaf
    assert db.get_conversation_active_leaf(dst) == copied[-1]["id"] == outcome["leaf_message_id"]


def test_copy_active_path_ignores_inactive_branch(service_with_db):
    db, service = service_with_db
    src = service.create_conversation(title="Src")
    root = db.add_message({"conversation_id": src, "sender": "user", "content": "root"})
    kept = db.add_message({"conversation_id": src, "sender": "assistant", "content": "kept",
                           "parent_message_id": root})
    db.add_message({"conversation_id": src, "sender": "assistant", "content": "dropped",
                    "parent_message_id": root})
    db.set_conversation_active_leaf(src, kept)
    dst = service.create_conversation(title="Dst")

    outcome = service.copy_conversation_active_path(src, dst)

    assert outcome["copied"] == 2
    contents = [m["content"] for m in db.get_messages_for_conversation(dst)]
    assert contents == ["root", "kept"]


def test_copy_active_path_falls_back_to_latest_when_no_leaf(service_with_db):
    db, service = service_with_db
    src = service.create_conversation(title="Src")
    _seed_chain(db, service, src, [("user", "a"), ("assistant", "b")])
    db.set_conversation_active_leaf(src, None)  # API-created conversation, never opened
    dst = service.create_conversation(title="Dst")

    outcome = service.copy_conversation_active_path(src, dst)

    assert outcome["copied"] == 2


def test_copy_active_path_empty_history_raises(service_with_db):
    db, service = service_with_db
    src = service.create_conversation(title="Empty")
    dst = service.create_conversation(title="Dst")
    with pytest.raises(ValueError, match="empty_history"):
        service.copy_conversation_active_path(src, dst)


def test_copy_active_path_preserves_fields(service_with_db):
    db, service = service_with_db
    src = service.create_conversation(title="Src")
    mid = db.add_message({
        "conversation_id": src, "sender": "assistant", "content": "tool stuff",
        "role": "tool", "metadata_json": '{"k": 1}', "usage_json": '{"tokens": 5}',
        "provider_continuation_json": '{"ckpt": true}',
    })
    db.set_conversation_active_leaf(src, mid)
    dst = service.create_conversation(title="Dst")

    service.copy_conversation_active_path(src, dst)

    copied = db.get_messages_for_conversation(dst)[0]
    assert copied["role"] == "tool"
    assert copied["metadata_json"] == '{"k": 1}'
    assert copied["usage_json"] == '{"tokens": 5}'
    assert copied["provider_continuation_json"] == '{"ckpt": true}'


def test_copy_active_path_atomic_rollback(service_with_db, monkeypatch):
    db, service = service_with_db
    src = service.create_conversation(title="Src")
    _seed_chain(db, service, src, [("user", "a"), ("assistant", "b"), ("user", "c")])
    dst = service.create_conversation(title="Dst")

    real_add_message = db.add_message
    calls = {"n": 0}

    def flaky_add_message(msg_data):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("boom mid-copy")
        return real_add_message(msg_data)

    monkeypatch.setattr(db, "add_message", flaky_add_message)
    with pytest.raises(RuntimeError, match="boom mid-copy"):
        service.copy_conversation_active_path(src, dst)
    monkeypatch.undo()
    assert db.get_messages_for_conversation(dst) == []  # nothing created
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Chat/test_chat_conversation_service.py -k copy_active_path -v`
Expected: FAIL with `AttributeError: ... has no attribute 'copy_conversation_active_path'`

- [ ] **Step 3: Implement** — add to `ChatConversationService` after `get_conversation_tree`:

```python
    def copy_conversation_active_path(
        self, source_conversation_id: str, target_conversation_id: str
    ) -> dict[str, Any]:
        """Copy the source conversation's active path into the target, verbatim.

        Walks the active-leaf ancestry (root -> leaf) of ``source_conversation_id``
        and re-inserts each message into ``target_conversation_id`` with fresh ids
        and remapped parents, preserving sender/role/content, images, usage,
        metadata, provider-continuation payloads, and timestamps. Runs inside a
        single transaction: a mid-copy failure rolls back everything. Sibling
        branches off the active path are NOT copied (fork = the path the user
        sees; combine with rewind for mid-conversation forks).

        Returns:
            ``{"copied": <int>, "leaf_message_id": <new id of the copied leaf>}``

        Raises:
            ValueError: ``empty_history`` when the source has no messages.
        """
        tree = self.get_conversation_tree(
            source_conversation_id, root_limit=10_000, depth_cap=10_000
        )
        nodes: dict[str, dict[str, Any]] = {}

        def _walk(node: dict[str, Any]) -> None:
            nodes[str(node["id"])] = node
            for child in node.get("children") or []:
                _walk(child)

        for root in tree.get("root_threads") or []:
            _walk(root)
        if not nodes:
            raise ValueError("empty_history")

        leaf_id = self.db.get_conversation_active_leaf(source_conversation_id)
        if leaf_id is None or leaf_id not in nodes:
            # Conversations created outside the Console have no leaf pointer;
            # fall back to the most recent message by timestamp.
            leaf_id = max(nodes, key=lambda i: str(nodes[i].get("timestamp") or ""))

        path: list[dict[str, Any]] = []
        cursor: str | None = leaf_id
        while cursor is not None and cursor in nodes:
            path.append(nodes[cursor])
            cursor = nodes[cursor].get("parent_message_id")
        path.reverse()

        id_map: dict[str, str] = {}
        with self.db.transaction():
            for node in path:
                old_id = str(node["id"])
                new_id = self.db.add_message(
                    {
                        "conversation_id": target_conversation_id,
                        "sender": node.get("sender") or node.get("role") or "user",
                        "content": node.get("content") or "",
                        "role": node.get("role"),
                        "parent_message_id": id_map.get(str(node.get("parent_message_id"))),
                        "image_data": node.get("image_data"),
                        "image_mime_type": node.get("image_mime_type"),
                        "timestamp": node.get("timestamp"),
                        "usage_json": node.get("usage_json"),
                        "metadata_json": node.get("metadata_json"),
                        "provider_continuation_json": node.get("provider_continuation_json"),
                    }
                )
                if new_id is None:
                    raise RuntimeError("copy_active_path: message insert failed")
                id_map[old_id] = str(new_id)

        new_leaf = id_map.get(str(leaf_id))
        self.db.set_conversation_active_leaf(target_conversation_id, new_leaf)
        return {"copied": len(path), "leaf_message_id": new_leaf}
```

Note: `add_message` opens its own nested transaction; the outer `self.db.transaction()` makes the whole copy atomic (nested-depth tracking is built into `TransactionContextManager`, `DB/ChaChaNotes_DB.py:3350-3378`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/Chat/test_chat_conversation_service.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/chat_conversation_service.py Tests/Chat/test_chat_conversation_service.py
git commit -m "feat: copy_conversation_active_path fork primitive with parent remap and atomicity"
```

---

### Task 3: Tool name constants and schemas

**Files:**
- Modify: `tldw_chatbook/Agents/agent_models.py:75-123` (constants + `RUNTIME_TOOL_NAMES`)
- Modify: `tldw_chatbook/Agents/tool_catalog.py` (schemas after `RUN_SKILL_SCRIPT_TOOL_SCHEMA`, ~line 320)
- Test: `Tests/Agents/test_agent_chat_create_tools.py` (new)

**Interfaces:**
- Produces: `FORK_CHAT_TOOL_NAME = "fork_chat"`, `NEW_CHAT_TOOL_NAME = "new_chat"` (in `RUNTIME_TOOL_NAMES`); `FORK_CHAT_TOOL_SCHEMA` / `NEW_CHAT_TOOL_SCHEMA` (`ToolSchema(id="runtime:fork_chat"|"runtime:new_chat", ...)`). Tasks 4 and 7 import these names.

- [ ] **Step 1: Write the failing test**

```python
"""fork_chat / new_chat runtime-tool constants and schema shapes."""
from tldw_chatbook.Agents.agent_models import (
    FORK_CHAT_TOOL_NAME,
    NEW_CHAT_TOOL_NAME,
    RUNTIME_TOOL_NAMES,
)
from tldw_chatbook.Agents.tool_catalog import FORK_CHAT_TOOL_SCHEMA, NEW_CHAT_TOOL_SCHEMA


def test_names_are_runtime_tools():
    assert FORK_CHAT_TOOL_NAME == "fork_chat"
    assert NEW_CHAT_TOOL_NAME == "new_chat"
    assert FORK_CHAT_TOOL_NAME in RUNTIME_TOOL_NAMES
    assert NEW_CHAT_TOOL_NAME in RUNTIME_TOOL_NAMES


def test_fork_chat_schema_shape():
    assert FORK_CHAT_TOOL_SCHEMA.id == "runtime:fork_chat"
    assert FORK_CHAT_TOOL_SCHEMA.name == FORK_CHAT_TOOL_NAME
    props = FORK_CHAT_TOOL_SCHEMA.parameters["properties"]
    assert set(props) == {"title", "opening_prompt", "instructions"}
    assert FORK_CHAT_TOOL_SCHEMA.parameters["required"] == []
    for text in (FORK_CHAT_TOOL_SCHEMA.description,):
        assert "user" in text and "confirm" in text  # documents the confirm contract


def test_new_chat_schema_shape():
    assert NEW_CHAT_TOOL_SCHEMA.id == "runtime:new_chat"
    assert NEW_CHAT_TOOL_SCHEMA.name == NEW_CHAT_TOOL_NAME
    props = NEW_CHAT_TOOL_SCHEMA.parameters["properties"]
    assert set(props) == {"title", "opening_prompt", "instructions"}
    assert NEW_CHAT_TOOL_SCHEMA.parameters["required"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest Tests/Agents/test_agent_chat_create_tools.py -v`
Expected: FAIL with `ImportError: cannot import name 'FORK_CHAT_TOOL_NAME'`

- [ ] **Step 3: Implement.** In `agent_models.py`, after `READ_AGENT_MESSAGES_TOOL_NAME` (line ~103):

```python
FORK_CHAT_TOOL_NAME = "fork_chat"
NEW_CHAT_TOOL_NAME = "new_chat"
```

and add both to the `RUNTIME_TOOL_NAMES` frozenset (lines 106-123). In `tool_catalog.py`, after `RUN_SKILL_SCRIPT_TOOL_SCHEMA`:

```python
FORK_CHAT_TOOL_SCHEMA = ToolSchema(
    id="runtime:fork_chat",
    name=FORK_CHAT_TOOL_NAME,
    description=(
        "Fork the current chat into a new chat so the user can pursue a parallel "
        "workstream: the conversation's active message history is copied verbatim "
        "into a brand-new chat (nothing is removed from the current chat). The "
        "user is asked to confirm every fork. The copy is a snapshot at the "
        "moment of this call — your current in-progress reply is NOT included, "
        "so put the workstream's framing into opening_prompt. opening_prompt is "
        "placed in the new chat's input box as a draft the user reviews and "
        "sends themselves; it is never sent automatically. instructions, when "
        "given, become the new chat's standing system prompt (refused for "
        "character chats). The new chat opens in the background; the user "
        "switches to it when ready. Use sparingly — each call shows the user an "
        "approval card, and do not retry after the user declines."
    ),
    parameters={
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short title for the new chat, e.g. 'Workstream: DB migration'.",
            },
            "opening_prompt": {
                "type": "string",
                "description": (
                    "First message for the workstream, delivered as a draft in "
                    "the new chat's input box for the user to review, edit, and send."
                ),
            },
            "instructions": {
                "type": "string",
                "description": (
                    "Optional standing system prompt for the new chat, replacing "
                    "the forked chat's system prompt. Not allowed when the current "
                    "chat is bound to a character."
                ),
            },
        },
        "required": [],
    },
)

NEW_CHAT_TOOL_SCHEMA = ToolSchema(
    id="runtime:new_chat",
    name=NEW_CHAT_TOOL_NAME,
    description=(
        "Create a brand-new, empty chat for a parallel workstream unrelated to "
        "the current conversation's history. The user is asked to confirm every "
        "creation. opening_prompt is placed in the new chat's input box as a "
        "draft the user reviews and sends themselves; it is never sent "
        "automatically. instructions, when given, become the new chat's standing "
        "system prompt. The new chat opens in the same workspace, in the "
        "background; the user switches to it when ready. Use sparingly — each "
        "call shows the user an approval card, and do not retry after the user "
        "declines."
    ),
    parameters={
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short title for the new chat.",
            },
            "opening_prompt": {
                "type": "string",
                "description": "Draft first message placed in the new chat's input box.",
            },
            "instructions": {
                "type": "string",
                "description": "Optional standing system prompt for the new chat.",
            },
        },
        "required": [],
    },
)
```

Import the two name constants in `tool_catalog.py`'s existing `agent_models` import block.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest Tests/Agents/test_agent_chat_create_tools.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/agent_models.py tldw_chatbook/Agents/tool_catalog.py Tests/Agents/test_agent_chat_create_tools.py
git commit -m "feat: fork_chat/new_chat tool names and schemas"
```

---

### Task 4: AgentService injection, schema pins, LoopDeps, and dispatch

**Files:**
- Modify: `tldw_chatbook/Agents/agent_service.py` (`__init__` ~1031-1151; `_run_one` pin block ~2500-2556; `LoopDeps` population ~4650-4701; `build_first_request_schema_plan` ~588-636)
- Modify: `tldw_chatbook/Agents/agent_runtime.py` (dispatch branches after the `run_skill_script` branch, ~1663)
- Test: `Tests/Agents/test_agent_runtime.py` (extend `make_deps`, ~line 41); `Tests/Agents/test_agent_chat_create_tools.py` (extend)

**Interfaces:**
- Consumes: `FORK_CHAT_TOOL_SCHEMA` / `NEW_CHAT_TOOL_SCHEMA` (Task 3); the recipe comment at `agent_models.py:84` ("name-constant + RUNTIME_TOOL_NAMES + tool_catalog schema + LoopDeps field + dispatch-branch + primary-agent-only-service-gate").
- Produces:
  - `AgentService.__init__(..., fork_chat_tool: Callable[[dict], ToolResult] | None = None, new_chat_tool: Callable[[dict], ToolResult] | None = None)` stored as `self._fork_chat_tool` / `self._new_chat_tool`.
  - `LoopDeps.fork_chat: Callable[[dict], ToolResult] | None = None` and `LoopDeps.new_chat: Callable[[dict], ToolResult] | None = None`.
  - `build_first_request_schema_plan(..., fork_chat_enabled: bool = False, new_chat_enabled: bool = False)` (Task 7 passes these at the bridge call site, `console_agent_bridge.py:3343-3346` pattern).
  - Dispatch contract: a tool call named `fork_chat`/`new_chat` with the deps field non-None calls `deps.fork_chat(dict(call.args))` / `deps.new_chat(dict(call.args))` after emitting `STEP_TOOL_CALL`; when the field is None the call falls through to `deps.invoke_tool(call)` (existing fallback, `agent_runtime.py:1682-1684`).

- [ ] **Step 1: Write the failing dispatch tests** (extend `Tests/Agents/test_agent_runtime.py`; the file already imports `LoopDeps`, `run_agent_loop`, `ModelTurn`, `ToolCall`, `SPAWN_TOOL_NAME` and defines `fence`/`make_deps`/`run`/`CFG`)

Add to `make_deps` two optional kwargs forwarded into `LoopDeps`:

```python
def make_deps(turns, *, invoke=None, spawn=None, cancel=None, clock=None,
              fork_chat=None, new_chat=None):
    ...
    return LoopDeps(
        ...,
        fork_chat=fork_chat,
        new_chat=new_chat,
    )
```

Then append tests:

```python
def test_fork_chat_dispatches_to_injected_callable():
    seen = []

    def fake_fork(args):
        seen.append(args)
        return ToolResult(ok=True, content='{"conversation_id": "c2"}')

    out = run(
        [
            ModelTurn(text=fence("fork_chat", {"title": "W: db"})),
            ModelTurn(text="Forked."),
        ],
        fork_chat=fake_fork,
    )
    assert out.status == RUN_DONE and out.final_text == "Forked."
    assert seen == [{"title": "W: db"}]
    kinds = [s.kind for s in out.steps]
    assert kinds == ["model", "tool_call", "tool_result", "model"]
    assert out.steps[1].tool_name == "fork_chat"


def test_new_chat_dispatches_to_injected_callable():
    seen = []

    def fake_new(args):
        seen.append(args)
        return ToolResult(ok=False, error="user_denied")

    out = run(
        [
            ModelTurn(text=fence("new_chat", {"title": "W: api"})),
            ModelTurn(text="Declined."),
        ],
        new_chat=fake_new,
    )
    assert out.status == RUN_DONE
    assert seen == [{"title": "W: api"}]


def test_chat_create_tools_fall_through_when_not_wired():
    # No fork_chat/new_chat in deps: the generic invoke_tool path handles it,
    # exactly like any other unknown runtime tool name.
    calls = []
    out = run(
        [
            ModelTurn(text=fence("new_chat", {"title": "x"})),
            ModelTurn(text="done"),
        ],
        invoke=lambda c: calls.append(c) or ToolResult(ok=True, content="ok"),
    )
    assert out.status == RUN_DONE
    assert calls[0].name == "new_chat"
```

Note: `CFG.allowed_tools` governs only catalog tools, not runtime names (`agent_service.py:2491-2499` comment) — no `allowed_tools` change needed.

Also add the pin-gating matrix test (spec §Testing: "schema pinned for primary runs and absent for subagent kinds") via a small pure helper introduced in Step 3:

```python
def test_chat_create_pin_gating_matrix():
    from tldw_chatbook.Agents.agent_service import _chat_create_runtime_schemas
    from tldw_chatbook.Agents.agent_models import AGENT_KIND_PRIMARY, AGENT_KIND_SUBAGENT

    tool = lambda args: ToolResult(ok=True, content="{}")
    primary = _chat_create_runtime_schemas(AGENT_KIND_PRIMARY, tool, tool)
    assert [s.name for s in primary] == ["fork_chat", "new_chat"]
    assert _chat_create_runtime_schemas(AGENT_KIND_SUBAGENT, tool, tool) == []
    assert _chat_create_runtime_schemas(AGENT_KIND_PRIMARY, None, tool) == [
        s for s in primary if s.name == "new_chat"
    ]
    assert _chat_create_runtime_schemas(AGENT_KIND_PRIMARY, None, None) == []
```

And extend `Tests/Agents/test_agent_chat_create_tools.py` for the first-plan flags (real signature: `build_first_request_schema_plan(registry, allowed_tools, budget, *, skill_file_enabled, install_skill_enabled, run_skill_script_enabled, run_log_active, ...) -> FirstRequestSchemaPlan` with a `.runtime_schemas` tuple; `RunBudget` and `ToolCatalogRegistry` construct with defaults — crib from existing callers via `grep -n "build_first_request_schema_plan(" tldw_chatbook/ Tests/`):

```python
def test_first_request_schema_plan_includes_chat_create_tools():
    from tldw_chatbook.Agents.agent_service import build_first_request_schema_plan
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.Agents.agent_models import RunBudget

    registry = ToolCatalogRegistry()
    budget = RunBudget()

    def _names(**flags):
        plan = build_first_request_schema_plan(
            registry, (), budget,
            skill_file_enabled=False,
            install_skill_enabled=False,
            run_skill_script_enabled=False,
            run_log_active=False,
            **flags,
        )
        return {s.name for s in plan.runtime_schemas}

    on = _names(fork_chat_enabled=True, new_chat_enabled=True)
    assert "fork_chat" in on and "new_chat" in on
    off = _names(fork_chat_enabled=False, new_chat_enabled=False)
    assert "fork_chat" not in off and "new_chat" not in off
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Agents/test_agent_runtime.py -k "fork_chat or new_chat or chat_create" Tests/Agents/test_agent_chat_create_tools.py -v`
Expected: FAIL — `TypeError: make_deps() got an unexpected keyword argument 'fork_chat'`, then dispatch falls through, and plan flags missing.

- [ ] **Step 3: Implement.**

`agent_runtime.py` — add two fields to `LoopDeps` after `run_skill_script` (mirror its comment style):

```python
    fork_chat: Callable[[dict], ToolResult] | None = None
    new_chat: Callable[[dict], ToolResult] | None = None
```

Dispatch branches after the `run_skill_script` branch (~1663), before the generic fallback:

```python
                elif call.name == FORK_CHAT_TOOL_NAME and deps.fork_chat is not None:
                    add(STEP_TOOL_CALL, tool_name=call.name, args=dict(call.args))
                    result = deps.fork_chat(dict(call.args))
                elif call.name == NEW_CHAT_TOOL_NAME and deps.new_chat is not None:
                    add(STEP_TOOL_CALL, tool_name=call.name, args=dict(call.args))
                    result = deps.new_chat(dict(call.args))
```

Import `FORK_CHAT_TOOL_NAME` / `NEW_CHAT_TOOL_NAME` in `agent_runtime.py`'s existing `agent_models` import.

`agent_service.py` — `__init__`: add after `run_skill_script_tool`:

```python
        fork_chat_tool: Callable[[dict], ToolResult] | None = None,
        new_chat_tool: Callable[[dict], ToolResult] | None = None,
```

stored after `self._run_skill_script_tool = run_skill_script_tool`:

```python
        # Primary-only by design (ADR-150): children never create chats in v1.
        self._fork_chat_tool = fork_chat_tool
        self._new_chat_tool = new_chat_tool
```

`_run_one` pin block — introduce a module-level pure helper (unit-testable; keeps `_run_one` thin) and call it. Append immediately after the `RUN_SKILL_SCRIPT_TOOL_SCHEMA` pin (~2538), before `progress_inbox = None`:

```python
        runtime_schemas.extend(
            _chat_create_runtime_schemas(agent_kind, self._fork_chat_tool, self._new_chat_tool)
        )
```

with the helper at module level (near `build_first_request_schema_plan`):

```python
def _chat_create_runtime_schemas(
    agent_kind: str,
    fork_chat_tool: "Callable[[dict], ToolResult] | None",
    new_chat_tool: "Callable[[dict], ToolResult] | None",
) -> list[ToolSchema]:
    """ADR-150: fork_chat/new_chat are primary-only in v1 (sub-agents: TASK-32480)."""
    if agent_kind != AGENT_KIND_PRIMARY:
        return []
    schemas: list[ToolSchema] = []
    if fork_chat_tool is not None:
        schemas.append(FORK_CHAT_TOOL_SCHEMA)
    if new_chat_tool is not None:
        schemas.append(NEW_CHAT_TOOL_SCHEMA)
    return schemas
```

`LoopDeps` population (~4689-4701) — after `run_skill_script=self._run_skill_script_tool`:

```python
            fork_chat=(
                self._fork_chat_tool
                if agent_kind == AGENT_KIND_PRIMARY
                and self._fork_chat_tool is not None
                else None
            ),
            new_chat=(
                self._new_chat_tool
                if agent_kind == AGENT_KIND_PRIMARY
                and self._new_chat_tool is not None
                else None
            ),
```

`build_first_request_schema_plan` (~588-636) — add parameters `fork_chat_enabled: bool = False, new_chat_enabled: bool = False` next to `install_skill_enabled` / `run_skill_script_enabled` (lines 616-619), and append the same two schema appends gated on the flags wherever `INSTALL_SKILL_TOOL_SCHEMA` is appended in that function. Import both schemas in `agent_service.py`'s existing `tool_catalog` import.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/Agents/test_agent_runtime.py Tests/Agents/test_agent_chat_create_tools.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/agent_service.py tldw_chatbook/Agents/agent_runtime.py Tests/Agents/test_agent_runtime.py Tests/Agents/test_agent_chat_create_tools.py
git commit -m "feat: agent service seam for fork_chat/new_chat (pins, deps, dispatch, first-plan flags)"
```

---

### Task 5: Controller confirm rounds (`request_chat_create_confirm`)

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (state ~2188-2236; methods modeled on `request_skill_script_confirm` :6253-6438, `resolve_pending_skill_script` :6472-6509, `_marshal_pending_skill_script` :6463-6470, `_remount_parked_skill_script` :6440-6461)
- Test: `Tests/Chat/test_console_chat_create_confirm.py` (new; crib the `make_controller` fixture from `Tests/Chat/test_console_skill_script_confirm.py:60-116`)

**Interfaces:**
- Produces (consumed by Task 7):
  - `request_chat_create_confirm(payload: dict, *, session_id: str | None = None) -> dict` returning `{"allow": bool, "remember": bool}`. Session-scoped remember: a prior `remember=True` decision for `(session_id, tool)` short-circuits subsequent rounds for that tool in that session with `{"allow": True, "remember": True}` (no card).
  - `resolve_pending_chat_create(allow: bool, remember: bool, request_id: str | None = None) -> None` (UI thread).
  - `pending_chat_create_ids() -> list[str]` (tests).
  - `set_pending_chat_create: Callable[[dict | None], None] | None = None` attribute (wired by Task 6).
  - `self._chat_create_session_grants: dict[str, set[str]]` (session_id → tool names), cleared for a session in `close_session`.
- Payload contract in/out: request payload keys `tool` ("fork_chat"|"new_chat"), `title`, `opening_prompt`, `instructions`, `fork_source_title` (fork only), `fork_message_count` (fork only). The card payload additionally carries `session_id`, `request_id`, `timeout_seconds`, `deadline_monotonic` (added by the controller, mirroring the skill-script card payload).

- [ ] **Step 1: Write the failing tests** (new file; the fixture mirrors `test_console_skill_script_confirm.py`)

```python
"""Confirm rounds for agent-initiated chat creation (fork_chat / new_chat)."""
import threading

import pytest

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from Tests.Chat.test_console_skill_script_confirm import _FakeApp, _wait_until  # reuse fakes


@pytest.fixture
def make_controller():
    made = []

    def _make() -> ConsoleChatController:
        store = ConsoleChatStore()
        controller = ConsoleChatController(store=store, provider_gateway=object())
        controller.app = _FakeApp()
        controller.pending_chat_create_payloads = []
        controller.set_pending_chat_create = controller.pending_chat_create_payloads.append
        made.append(controller)
        return controller

    yield _make
    for controller in made:
        controller.begin_shutdown()


def _payload(tool="fork_chat", **extra):
    return {"tool": tool, "title": "W: db", "opening_prompt": "go", "instructions": ""}


def test_allow_round_trip(make_controller):
    controller = make_controller()
    result = {}
    t = threading.Thread(target=lambda: result.update(
        decision=controller.request_chat_create_confirm(_payload(), session_id="s1")))
    t.start()
    _wait_until(lambda: bool(controller.pending_chat_create_ids()))
    controller.resolve_pending_chat_create(True, False, request_id=controller.pending_chat_create_ids()[0])
    t.join(timeout=5)
    assert result["decision"] == {"allow": True, "remember": False}


def test_deny_round_trip(make_controller):
    controller = make_controller()
    result = {}
    t = threading.Thread(target=lambda: result.update(
        decision=controller.request_chat_create_confirm(_payload(), session_id="s1")))
    t.start()
    _wait_until(lambda: bool(controller.pending_chat_create_ids()))
    controller.resolve_pending_chat_create(False, False, request_id=controller.pending_chat_create_ids()[0])
    t.join(timeout=5)
    assert result["decision"] == {"allow": False, "remember": False}


def test_remember_grants_session_scope(make_controller):
    controller = make_controller()
    results = []

    def first():
        results.append(controller.request_chat_create_confirm(_payload(), session_id="s1"))
    t = threading.Thread(target=first)
    t.start()
    _wait_until(lambda: bool(controller.pending_chat_create_ids()))
    controller.resolve_pending_chat_create(True, True, request_id=controller.pending_chat_create_ids()[0])
    t.join(timeout=5)

    # Second call in the same session: no card, straight allow.
    decision = controller.request_chat_create_confirm(_payload(), session_id="s1")
    assert decision == {"allow": True, "remember": True}
    assert controller.pending_chat_create_payloads == [controller.pending_chat_create_payloads[0]]

    # Different tool in the same session still confirms.
    t2 = threading.Thread(target=lambda: results.append(
        controller.request_chat_create_confirm(_payload(tool="new_chat"), session_id="s1")))
    t2.start()
    _wait_until(lambda: len(controller.pending_chat_create_ids()) > 0)
    controller.resolve_pending_chat_create(True, False, request_id=controller.pending_chat_create_ids()[-1])
    t2.join(timeout=5)
    assert results[-1] == {"allow": True, "remember": False}


def test_no_ui_fails_closed_immediately(make_controller):
    controller = make_controller()
    controller.app = None
    controller.set_pending_chat_create = None
    decision = controller.request_chat_create_confirm(_payload())
    assert decision == {"allow": False, "remember": False}
```

Adapt `_wait_until` import: if it is a private helper in the skill-script test module, copy it into this file instead of importing (check how that module defines it).

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Chat/test_console_chat_create_confirm.py -v`
Expected: FAIL with `AttributeError: 'ConsoleChatController' object has no attribute 'request_chat_create_confirm'`

- [ ] **Step 3: Implement** — in `console_chat_controller.py`, mirroring the skill-script machinery piece for piece:

State (next to `_pending_skill_script_rounds`, ~2230):

```python
        self._pending_chat_create_lock = threading.Lock()
        self._pending_chat_create_rounds: dict[str, dict[str, Any]] = {}
        self._parked_chat_create_payloads: dict[str, dict[str, Any]] = {}
        self._chat_create_session_grants: dict[str, set[str]] = {}
        self.set_pending_chat_create: Callable[[dict | None], None] | None = None
        self.chat_create_confirm_timeout_seconds: Callable[[], int] | None = None
```

`request_chat_create_confirm` — copy `request_skill_script_confirm` (:6253-6438) and change: the rounds/parked maps to the ones above; the marshal target to `set_pending_chat_create`; the parked-remount target map; AND insert the grant short-circuit at the top, after the no-UI guard:

```python
        tool = str(payload.get("tool") or "")
        if tool in self._chat_create_session_grants.get(owning_session_id, set()):
            return {"allow": True, "remember": True}
```

(compute `owning_session_id` first, exactly as the skill-script method does), and in the success path, when the decision is allow+remember:

```python
                    if decision.get("remember", False):
                        self._chat_create_session_grants.setdefault(owning_session_id, set()).add(tool)
```

`resolve_pending_chat_create`, `pending_chat_create_ids`, `_marshal_pending_chat_create`, `_remount_parked_chat_create`: direct mirrors of the four skill-script counterparts (:6440-6520), renamed. Register `_remount_parked_chat_create` wherever `_remount_parked_skill_script` is registered (session switch/new/close hooks — `grep -n "_remount_parked_skill_script" tldw_chatbook/Chat/console_chat_controller.py` and add the sibling call at each site). In `close_session`, add `self._chat_create_session_grants.pop(session_id, None)` (locate `def close_session` and add the line alongside its existing per-session teardown).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/Chat/test_console_chat_create_confirm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_controller.py Tests/Chat/test_console_chat_create_confirm.py
git commit -m "feat: chat-create confirm rounds with session-scoped remember and parking"
```

---

### Task 6: Confirm card widget + UI wiring

**Files:**
- Create: `tldw_chatbook/Widgets/Chat_Widgets/chat_create_confirm_card.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen_state.py:8` (`TaskResumeState`)
- Modify: `tldw_chatbook/Widgets/Chat_Widgets/chat_task_cards.py` (mount branch, ~line 67)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (wiring kwargs ~4837-4856; `@on` handlers ~16638-16650)
- Modify: `tldw_chatbook/UI/Console_Modules/skill.py` (setter next to `_set_console_pending_skill_script`, ~183-186, and a decision forwarder next to its handler, ~201-215)
- Test: `Tests/Chat/test_chat_create_confirm_card.py` (new)

**Interfaces:**
- Consumes: `TaskResumeState` (`chat_screen_state.py:8`) currently holding `pending_skill_install` / `pending_skill_script`; the mount pattern `script_card.set_script(task_state.pending_skill_script)` (`chat_task_cards.py:67`); the screen-wiring pattern `"set_pending_skill_script": getattr(skill, "_set_console_pending_skill_script", None)` (`chat_screen.py:4854-4856`).
- Produces: `ChatCreateConfirmCard` widget with `set_payload(payload: dict | None)` and a `ChatCreateConfirmCard.ChatCreateDecided` message carrying `allow: bool, remember: bool, request_id: str`; `TaskResumeState.pending_chat_create: dict[str, Any] | None = None` + `has_pending_chat_create()`; screen kwargs entry `"set_pending_chat_create"`; `@on(ChatCreateConfirmCard.ChatCreateDecided)` handler forwarding to `controller.resolve_pending_chat_create`.

- [ ] **Step 1: Write the failing test**

```python
"""ChatCreateConfirmCard renders the payload and emits decisions."""
import pytest

from tldw_chatbook.Widgets.Chat_Widgets.chat_create_confirm_card import ChatCreateConfirmCard


def _payload(**over):
    base = {
        "tool": "fork_chat",
        "title": "W: DB migration",
        "opening_prompt": "Please plan the schema migration.",
        "instructions": "Focus only on the DB migration.",
        "fork_source_title": "API redesign",
        "fork_message_count": 12,
        "request_id": "r-1",
    }
    base.update(over)
    return base


@pytest.mark.parametrize(
    "allow,remember",
    [(True, False), (True, True), (False, False)],
)
def test_decisions_carry_request_id(allow, remember):
    card = ChatCreateConfirmCard()
    card.set_payload(_payload())
    messages = []
    card.post_message = lambda m: messages.append(m)  # capture instead of pump
    card._decide(allow=allow, remember=remember)
    assert len(messages) == 1
    decided = messages[0]
    assert decided.allow is allow
    assert decided.remember is remember
    assert decided.request_id == "r-1"


def test_clear_payload_hides_card():
    card = ChatCreateConfirmCard()
    card.set_payload(_payload())
    card.set_payload(None)
    assert card.display is False


def test_header_reflects_tool():
    card = ChatCreateConfirmCard()
    card.set_payload(_payload(tool="new_chat"))
    assert "new chat" in card._header_text().lower()
    card.set_payload(_payload())
    assert "fork" in card._header_text().lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest Tests/Chat/test_chat_create_confirm_card.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tldw_chatbook.Widgets.Chat_Widgets.chat_create_confirm_card'`

- [ ] **Step 3: Implement.** New card (read `Widgets/Chat_Widgets/skill_script_confirm_card.py` first and match its class structure, styling constants, and markup helpers; the skeleton below is the behavioral contract):

```python
"""Approval card for agent-initiated chat creation (fork_chat / new_chat)."""
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Button, Static


class ChatCreateConfirmCard(Widget):
    """Show one pending chat-creation request; emit the user's decision."""

    DEFAULT_CSS = """
    ChatCreateConfirmCard { border: round $accent; padding: 0 1; }
    ChatCreateConfirmCard .chat-create-body { margin: 0 1; }
    """

    class ChatCreateDecided(Message):  # import Message from textual.message
        def __init__(self, allow: bool, remember: bool, request_id: str) -> None:
            super().__init__()
            self.allow = allow
            self.remember = remember
            self.request_id = request_id

    def __init__(self) -> None:
        super().__init__()
        self._payload: dict | None = None

    def compose(self) -> ComposeResult:
        yield Static("", id="chat-create-header")
        yield Static("", id="chat-create-body", classes="chat-create-body")
        with Horizontal(id="chat-create-actions"):
            yield Button("Allow", id="chat-create-allow", variant="success")
            yield Button("Allow for this session", id="chat-create-allow-remember")
            yield Button("Deny", id="chat-create-deny", variant="error")

    def set_payload(self, payload: dict | None) -> None:
        self._payload = payload
        self.display = payload is not None
        if payload is None:
            return
        self.query_one("#chat-create-header", Static).update(self._header_text())
        self.query_one("#chat-create-body", Static).update(self._body_text())

    def _header_text(self) -> str:
        assert self._payload is not None
        verb = "Fork this chat" if self._payload.get("tool") == "fork_chat" else "Create new chat"
        return f"[b]{verb}[/b] — “{self._payload.get('title', '')}”"

    def _body_text(self) -> str:
        assert self._payload is not None
        lines: list[str] = []
        if self._payload.get("tool") == "fork_chat":
            lines.append(
                f"Copies {self._payload.get('fork_message_count', '?')} messages from "
                f"“{self._payload.get('fork_source_title', '')}” into a new chat."
            )
        if self._payload.get("opening_prompt"):
            lines.append(f"[b]Opening prompt (draft for the input box):[/b]\n{self._payload['opening_prompt']}")
        if self._payload.get("instructions"):
            lines.append(f"[b]System prompt:[/b]\n{self._payload['instructions']}")
        return "\n\n".join(lines)

    @on(Button.Pressed, "#chat-create-allow")
    def _allow(self) -> None:
        self._decide(allow=True, remember=False)

    @on(Button.Pressed, "#chat-create-allow-remember")
    def _allow_remember(self) -> None:
        self._decide(allow=True, remember=True)

    @on(Button.Pressed, "#chat-create-deny")
    def _deny(self) -> None:
        self._decide(allow=False, remember=False)

    def _decide(self, *, allow: bool, remember: bool) -> None:
        if self._payload is None:
            return
        request_id = str(self._payload.get("request_id", ""))
        self.set_payload(None)
        self.post_message(self.ChatCreateDecided(allow, remember, request_id))
```

(Fix the `Message` import: `from textual.message import Message`. Match the sibling cards' CSS/markup-escape conventions — escape user/model text with the same helper the skill card uses.)

Then:
1. `chat_screen_state.py` `TaskResumeState`: add `pending_chat_create: dict[str, Any] | None = None` and `has_pending_chat_create()` mirroring `has_pending_skill_script()` (see `chat_task_cards.py:72` for how the sibling helper is consumed).
2. `chat_task_cards.py`: mirror the skill-script mount branch at ~67 — when `task_state.has_pending_chat_create()`, mount/update a `ChatCreateConfirmCard` with `task_state.pending_chat_create`, and tear it down when absent, exactly as the sibling branch does for the script card.
3. `UI/Console_Modules/skill.py`: add `_set_console_pending_chat_create` directly after `_set_console_pending_skill_script` (:183-186), same `replace(current, pending_chat_create=payload)` body; add `handle_console_chat_create_decided(self, allow, remember, request_id)` next to `handle_console_skill_script_decided` (:201-215) forwarding to `controller.resolve_pending_chat_create(allow, remember, request_id=request_id)`.
4. `chat_screen.py`: add to the controller-construction kwargs (next to `"set_pending_skill_script"`, :4854-4856): `"set_pending_chat_create": getattr(skill, "_set_console_pending_chat_create", None),` and a handler next to the skill-script one (:16638-16650):

```python
    @on(ChatCreateConfirmCard.ChatCreateDecided)
    def handle_console_chat_create_decided(self, event: Any) -> None:
        event.stop()
        self._skill.handle_console_chat_create_decided(
            event.allow, event.remember, request_id=event.request_id
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/Chat/test_chat_create_confirm_card.py Tests/Chat/test_console_skill_script_confirm.py -v`
Expected: PASS (new file passes; skill-script card suite unchanged)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Chat_Widgets/chat_create_confirm_card.py tldw_chatbook/UI/Screens/chat_screen_state.py tldw_chatbook/Widgets/Chat_Widgets/chat_task_cards.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Console_Modules/skill.py Tests/Chat/test_chat_create_confirm_card.py
git commit -m "feat: chat-create confirmation card and Console wiring"
```

---

### Task 7: Bridge closures, controller executor, session completion

**Files:**
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py` (`run_reply` signature ~3385-3441; closures after `run_skill_script_tool` ~3865; flags helper ~2823-2880 + call site ~3343-3346; `AgentService(...)` construction ~4165-4179)
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (`execute_agent_chat_create` + partials at the `run_reply` invocation ~12108-12128)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (`_complete_agent_chat_create` + wiring kwargs)
- Test: `Tests/Chat/test_console_chat_create_integration.py` (new)

**Interfaces:**
- Consumes: Task 4 (`fork_chat_tool`/`new_chat_tool` AgentService params, first-plan flags), Task 5 (`request_chat_create_confirm`, `resolve_pending_chat_create`, grants), Tasks 1-2 (`create_conversation` lineage kwargs, `copy_conversation_active_path`).
- Produces:
  - `run_reply(..., request_chat_create_confirm: Callable[[dict], dict] | None = None, execute_agent_chat_create: Callable[[dict], dict] | None = None)`; closures `fork_chat_tool(payload: dict) -> ToolResult` and `new_chat_tool(payload: dict) -> ToolResult` built only when BOTH callables are non-None (advertised-equals-usable).
  - `ConsoleChatController.execute_agent_chat_create(payload: dict) -> dict` — payload `{tool, session_id, title, opening_prompt, instructions}`; returns `{"ok": True, "title", "conversation_id", "workspace_id", "copied_messages", "draft_set": True}` or `{"ok": False, "kind": <error kind>, "error": <message>}` with kinds: `source_not_persisted`, `empty_history`, `character_conflict`, `payload_too_large`, `session_gone`, `execution_failed`.
  - Screen-side `complete_agent_chat_create(**kwargs)` (UI thread): creates the non-activated session, sets the draft, invalidates the persisted-rows cache, triggers console-sync, toasts.
  - Conversation metadata key `console_agent_handoff` = `{"draft": str, "created_via": "fork_chat"|"new_chat", "source_run_id": str}` (Task 8 rehydrates).
  - Caps (module-level in the bridge, next to the closures): `CHAT_CREATE_TITLE_MAX = 120`, `CHAT_CREATE_PAYLOAD_MAX = 20_000`, denial terminal threshold `2`.

- [ ] **Step 1: Write the failing integration tests** (new file; crib the fake-app/controller fixture from Task 5 and the real in-memory-DB service pattern from `Tests/Chat/test_chat_conversation_service.py`)

```python
"""End-to-end bridge closures for fork_chat / new_chat with fakes + real SQLite."""
import json

import pytest

from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_agent_bridge import (
    build_chat_create_tool_closures,
)


class _FakeConfirm:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(payload)
        return self.decisions.pop(0)


class _FakeExecutor:
    def __init__(self, result=None):
        self.calls = []
        self.result = result or {"ok": True, "title": "W", "conversation_id": "c2",
                                 "workspace_id": None, "copied_messages": 3, "draft_set": True}

    def __call__(self, payload):
        self.calls.append(payload)
        return dict(self.result)


def test_fork_chat_tool_happy_path():
    confirm, executor = _FakeConfirm([{"allow": True, "remember": False}]), _FakeExecutor()
    fork_tool, new_tool = build_chat_create_tool_closures(
        confirm=confirm, execute=executor, session_id="s1", run_id="r1"
    )
    result = fork_tool({"title": "W: db", "opening_prompt": "go", "instructions": ""})
    assert result.ok
    data = json.loads(result.content)
    assert data["conversation_id"] == "c2" and data["draft_set"] is True
    assert "user" in data["note"].lower()
    assert executor.calls[0]["tool"] == "fork_chat"


def test_deny_returns_error_without_execution():
    confirm, executor = _FakeConfirm([{"allow": False, "remember": False}]), _FakeExecutor()
    fork_tool, _ = build_chat_create_tool_closures(confirm=confirm, execute=executor,
                                                   session_id="s1", run_id="r1")
    result = fork_tool({"title": "x"})
    assert not result.ok and "declined" in result.error.lower()
    assert executor.calls == []


def test_denial_guard_terminals_after_two():
    confirm = _FakeConfirm([{"allow": False, "remember": False}, {"allow": False, "remember": False}])
    executor = _FakeExecutor()
    fork_tool, _ = build_chat_create_tool_closures(confirm=confirm, execute=executor,
                                                   session_id="s1", run_id="r1")
    assert not fork_tool({}).ok
    assert not fork_tool({}).ok
    third = fork_tool({})
    assert not third.ok and "denied_repeatedly" in third.error
    assert confirm.payloads.__len__() == 2  # no third card


def test_remember_skips_confirm_for_that_tool_only():
    confirm, executor = _FakeConfirm([{"allow": True, "remember": True}]), _FakeExecutor()
    fork_tool, new_tool = build_chat_create_tool_closures(confirm=confirm, execute=executor,
                                                          session_id="s1", run_id="r1")
    assert fork_tool({}).ok                      # card shown, remembered
    assert fork_tool({}).ok                      # no card (session grant path via executor note)
    assert not new_tool({}).ok                   # confirm has no decisions left -> deny fail-closed
```

(The remember behavior above is closure-side memo of the controller's grant: the closure records `remember=True` per tool and skips calling confirm on later calls in the same run — the controller grants dict covers cross-run-within-session. Assert accordingly.)

Executor-level tests with a real in-memory DB (fixture pattern from `Tests/Chat/test_chat_conversation_service.py`), driving `ConsoleChatController.execute_agent_chat_create` with `complete_agent_chat_create` stubbed to record kwargs:

```python
def test_execute_fork_copies_history_and_lineage(real_db_controller):
    controller, db = real_db_controller
    src_session = controller.store.create_session(title="Src", activate=True)
    src_conv = controller.store.persistence.create_conversation(conversation_title="Src")
    src_session.persisted_conversation_id = src_conv
    ChatConversationService(db).create_conversation  # sanity import
    svc = ChatConversationService(db)
    ids = [db.add_message({"conversation_id": src_conv, "sender": "user", "content": "hi"})]
    db.set_conversation_active_leaf(src_conv, str(ids[0]))

    outcome = controller.execute_agent_chat_create(
        {"tool": "fork_chat", "session_id": src_session.id, "title": "W: db",
         "opening_prompt": "go", "instructions": ""}
    )
    assert outcome["ok"], outcome
    forked = db.get_conversation_by_id(outcome["conversation_id"])
    assert forked["parent_conversation_id"] == src_conv
    msgs = db.get_messages_for_conversation(outcome["conversation_id"])
    assert [m["content"] for m in msgs] == ["hi"]
    assert outcome["copied_messages"] == 1


def test_execute_fork_refuses_instructions_on_character_chat(real_db_controller):
    controller, db = real_db_controller
    session = controller.store.create_session(title="Char", character_id=7, character_name="Rex")
    conv = controller.store.persistence.create_conversation(conversation_title="Char", character_id=7)
    session.persisted_conversation_id = conv
    db.add_message({"conversation_id": conv, "sender": "user", "content": "hi"})
    outcome = controller.execute_agent_chat_create(
        {"tool": "fork_chat", "session_id": session.id, "title": "x",
         "opening_prompt": "", "instructions": "be terse"}
    )
    assert not outcome["ok"] and outcome["kind"] == "character_conflict"


def test_execute_fork_ephemeral_source_error(real_db_controller):
    controller, db = real_db_controller
    session = controller.store.create_session(title="Tmp", ephemeral=True)
    outcome = controller.execute_agent_chat_create(
        {"tool": "fork_chat", "session_id": session.id, "title": "x"})
    assert not outcome["ok"] and outcome["kind"] == "source_not_persisted"


def test_execute_fork_empty_history_error(real_db_controller):
    controller, db = real_db_controller
    session = controller.store.create_session(title="Empty")
    conv = controller.store.persistence.create_conversation(conversation_title="Empty")
    session.persisted_conversation_id = conv
    outcome = controller.execute_agent_chat_create(
        {"tool": "fork_chat", "session_id": session.id, "title": "x"})
    assert not outcome["ok"] and outcome["kind"] == "empty_history"


def test_execute_payload_too_large(real_db_controller):
    controller, db = real_db_controller
    session = controller.store.create_session(title="S")
    outcome = controller.execute_agent_chat_create(
        {"tool": "new_chat", "session_id": session.id, "title": "x",
         "opening_prompt": "y" * 20_001, "instructions": ""})
    assert not outcome["ok"] and outcome["kind"] == "payload_too_large"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Chat/test_console_chat_create_integration.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_chat_create_tool_closures'` and missing `execute_agent_chat_create`.

- [ ] **Step 3: Implement.**

`console_agent_bridge.py` — module-level helper (exported for tests; called from `run_reply`), placed after the `run_skill_script_tool` closure region:

```python
CHAT_CREATE_TITLE_MAX = 120
CHAT_CREATE_PAYLOAD_MAX = 20_000
_CHAT_CREATE_DENIAL_LIMIT = 2


def build_chat_create_tool_closures(
    *,
    confirm: Callable[[dict], dict],
    execute: Callable[[dict], dict],
    session_id: str,
    run_id: str,
) -> tuple[Callable[[dict], ToolResult], Callable[[dict], ToolResult]]:
    """Build the fork_chat/new_chat runtime-tool closures for one run.

    Both share one per-run denial counter (terminal after two denials) and one
    per-run remember memo (the controller's session grants cover cross-run
    remembers inside the same session).
    """
    denials = {"fork_chat": 0, "new_chat": 0}
    remembered: set[str] = set()

    def _run(tool: str, args: dict) -> ToolResult:
        if denials[tool] >= _CHAT_CREATE_DENIAL_LIMIT:
            return ToolResult(
                ok=False,
                error=(
                    "denied_repeatedly: the user declined twice; chat creation "
                    "is disabled for the rest of this run. Do not retry."
                ),
            )
        title = str(args.get("title") or "").strip()[:CHAT_CREATE_TITLE_MAX]
        opening_prompt = str(args.get("opening_prompt") or "")
        instructions = str(args.get("instructions") or "")
        if len(opening_prompt) > CHAT_CREATE_PAYLOAD_MAX or len(instructions) > CHAT_CREATE_PAYLOAD_MAX:
            return ToolResult(ok=False, error="payload_too_large: opening_prompt/instructions exceed 20000 chars")
        payload = {
            "tool": tool,
            "session_id": session_id,
            "run_id": run_id,
            "title": title,
            "opening_prompt": opening_prompt,
            "instructions": instructions,
        }
        if tool not in remembered:
            try:
                decision = confirm(dict(payload))
            except Exception:  # noqa: BLE001 — a UI error fails closed
                decision = {"allow": False, "remember": False}
            if not isinstance(decision, Mapping) or not decision.get("allow", False):
                denials[tool] += 1
                return ToolResult(
                    ok=False,
                    error="The user declined. Do not retry this turn.",
                )
            if decision.get("remember", False):
                remembered.add(tool)
        outcome = execute(dict(payload))
        if not isinstance(outcome, dict) or not outcome.get("ok"):
            kind = str(outcome.get("kind", "execution_failed")) if isinstance(outcome, dict) else "execution_failed"
            return ToolResult(ok=False, error=f"{kind}: {outcome.get('error', 'chat creation failed') if isinstance(outcome, dict) else 'chat creation failed'}")
        copied = outcome.get("copied_messages")
        note = (
            "The new chat opened in the background (same workspace) with your "
            "opening prompt as a draft in its input box; the user reviews and "
            "sends it themselves. Do not send messages into the new chat."
        )
        content = json.dumps(
            {
                "title": outcome.get("title"),
                "conversation_id": outcome.get("conversation_id"),
                "workspace_id": outcome.get("workspace_id"),
                "copied_messages": copied,
                "draft_set": bool(outcome.get("draft_set")),
                "note": note,
            }
        )
        return ToolResult(ok=True, content=content)

    def fork_chat_tool(args: dict) -> ToolResult:
        return _run("fork_chat", args)

    def new_chat_tool(args: dict) -> ToolResult:
        return _run("new_chat", args)

    return fork_chat_tool, new_chat_tool
```

In `run_reply`: add the two parameters to the signature (after `request_skill_script_confirm`); build the closures near where `run_skill_script_tool` is built:

```python
        fork_chat_tool = new_chat_tool = None
        if request_chat_create_confirm is not None and execute_agent_chat_create is not None:
            fork_chat_tool, new_chat_tool = build_chat_create_tool_closures(
                confirm=request_chat_create_confirm,
                execute=execute_agent_chat_create,
                session_id=session_id,
                run_id=str(run_id),
            )
```

(`run_id` — use the run identifier available in `run_reply`'s scope; if only `conversation_id`/`assistant_message_id` are in scope, use `assistant_message_id` and name it accordingly.) Pass `fork_chat_tool=fork_chat_tool, new_chat_tool=new_chat_tool` into the `AgentService(...)` construction (~4165-4179). Extend the first-plan call path: the flags helper at :2823-2880 and its call site at :3343-3346 gain `fork_chat_enabled=bool(fork_chat_tool is not None)` and `new_chat_enabled=bool(new_chat_tool is not None)` (the call-site booleans are computed where `script_tool_enabled` is, ~3548).

`console_chat_controller.py` — the executor (module import of `ChatConversationService` at top):

```python
    def execute_agent_chat_create(self, payload: dict) -> dict:
        """WORKER THREAD: create the confirmed chat (conversation + copy) and
        marshal UI completion. Returns an outcome dict (see TASK-32482 spec)."""
        tool = str(payload.get("tool") or "")
        session_id = str(payload.get("session_id") or "")
        session = next((s for s in self.store.sessions() if s.id == session_id), None)
        if session is None:
            return {"ok": False, "kind": "session_gone", "error": "source session not found"}
        title = str(payload.get("title") or "").strip()
        opening_prompt = str(payload.get("opening_prompt") or "")
        instructions = str(payload.get("instructions") or "")
        if len(opening_prompt) > 20_000 or len(instructions) > 20_000:
            return {"ok": False, "kind": "payload_too_large", "error": "payload exceeds 20000 chars"}
        persistence = self.store.persistence
        db = getattr(persistence, "db", None)
        if persistence is None or db is None:
            return {"ok": False, "kind": "execution_failed", "error": "persistence unavailable"}

        source_conv: str | None = None
        source_row: dict[str, Any] | None = None
        copied = 0
        if tool == "fork_chat":
            source_conv = session.persisted_conversation_id
            if session.ephemeral or not source_conv:
                return {"ok": False, "kind": "source_not_persisted",
                        "error": "the current chat is temporary; nothing to fork"}
            conversation_service = ChatConversationService(db)
            tree = conversation_service.get_conversation_tree(
                source_conv, root_limit=10_000, depth_cap=10_000
            )
            source_row = dict(tree.get("conversation") or {})
            if not title:
                title = f"Fork of {source_row.get('title') or 'chat'}"[:120]
            if instructions and source_row.get("character_id"):
                return {"ok": False, "kind": "character_conflict",
                        "error": "character chats keep their persona; fork without instructions"}
        elif not title:
            title = "New Chat"

        handoff_metadata = {
            "console_agent_handoff": {
                "draft": opening_prompt,
                "created_via": tool,
                "source_run_id": str(payload.get("run_id") or ""),
            }
        }
        try:
            if tool == "fork_chat":
                new_conv = persistence.create_conversation(
                    conversation_title=title,
                    scope_type=source_row.get("scope_type") or "global",
                    workspace_id=source_row.get("workspace_id"),
                    system_prompt=instructions or source_row.get("system_prompt"),
                    character_id=source_row.get("character_id"),
                    assistant_kind=source_row.get("assistant_kind"),
                    assistant_id=source_row.get("assistant_id"),
                    assistant_authority_id=source_row.get("assistant_authority_id"),
                    metadata=handoff_metadata,
                    parent_conversation_id=source_conv,
                    forked_from_message_id=source_row.get("active_leaf_message_id"),
                )
                copy_outcome = ChatConversationService(db).copy_conversation_active_path(
                    str(source_conv), new_conv
                )
                copied = int(copy_outcome["copied"])
            else:
                workspace_id = session.workspace_id
                scope = "global" if workspace_id in (None, CONSOLE_GLOBAL_WORKSPACE_ID) else "workspace"
                new_conv = persistence.create_conversation(
                    conversation_title=title,
                    scope_type=scope,
                    workspace_id=None if scope == "global" else workspace_id,
                    system_prompt=instructions or None,
                    metadata=handoff_metadata,
                )
        except ValueError as exc:
            if "empty_history" in str(exc):
                return {"ok": False, "kind": "empty_history",
                        "error": "nothing to fork yet; use new_chat"}
            return {"ok": False, "kind": "copy_failed", "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "kind": "copy_failed", "error": str(exc)}

        if self.app is not None and self.complete_agent_chat_create is not None:
            self.app.call_from_thread(
                self.complete_agent_chat_create,
                session_id=session_id,
                conversation_id=new_conv,
                title=title,
                tool=tool,
                opening_prompt=opening_prompt,
                workspace_id=(source_row or {}).get("workspace_id") if tool == "fork_chat" else session.workspace_id,
            )
        return {
            "ok": True, "title": title, "conversation_id": new_conv,
            "workspace_id": (source_row or {}).get("workspace_id") if tool == "fork_chat" else session.workspace_id,
            "copied_messages": copied, "draft_set": bool(opening_prompt),
        }
```

Add `self.complete_agent_chat_create: Callable[..., None] | None = None` to controller state (Task 5's block). Import `CONSOLE_GLOBAL_WORKSPACE_ID` from `Chat/console_chat_store.py` (it defines it) if not already imported.

Define the `real_db_controller` fixture concretely at the top of the integration test file:

```python
@pytest.fixture
def real_db_controller():
    """Controller + real in-memory SQLite store, with UI completion stubbed."""
    from Tests.Chat.test_console_skill_script_confirm import _FakeApp
    from tldw_chatbook.DB.ChaChaNotes_DB import ChaChaNotesDB  # adapt to the class name/
    # constructor pattern used by Tests/Chat/test_chat_persistence_service.py's db fixture
    db = <in-memory ChaChaNotes DB per that fixture>
    persistence = ChatPersistenceService(db)
    store = ConsoleChatStore(persistence=persistence)
    controller = ConsoleChatController(store=store, provider_gateway=object())
    controller.app = _FakeApp()
    completed = []
    controller.complete_agent_chat_create = lambda **kw: completed.append(kw)
    yield controller, db
    controller.begin_shutdown()
```

(Crib the exact in-memory DB construction from the persistence-service test fixture — the class name and args differ across DB modules; reuse whatever that suite does so schema/migrations initialize identically. `ChatPersistenceService` additionally needs its `workspace_registry` argument if its tests pass one — mirror them.)

Controller `run_reply` invocation (~12108-12128), next to the skill confirms (same advertised-equals-usable guard):

```python
                request_chat_create_confirm=(
                    functools.partial(self.request_chat_create_confirm, session_id=session_id)
                    if self.set_pending_chat_create is not None
                    else None
                ),
                execute_agent_chat_create=(
                    self.execute_agent_chat_create
                    if self.complete_agent_chat_create is not None
                    else None
                ),
```

`chat_screen.py` — the UI-thread completion, next to `_park_console_approval`:

```python
    def _complete_agent_chat_create(
        self, *, session_id: str, conversation_id: str, title: str,
        tool: str, opening_prompt: str, workspace_id,
    ) -> None:
        controller = self._console_chat_controller
        if controller is None:
            return
        store = controller.store
        nodes: list = []
        leaf = None
        if tool == "fork_chat":
            from tldw_chatbook.Chat.console_conversation_hydration import (
                console_messages_from_conversation_tree,
            )
            service = getattr(self.app, "chat_conversation_scope_service", None)
            maybe_tree = service.get_conversation_tree(
                conversation_id, mode="local", depth_cap=10_000, root_limit=10_000
            )
            import inspect as _inspect
            tree = maybe_tree if not _inspect.isawaitable(maybe_tree) else None
            if tree is None:  # async facade: fall back to sync service read
                from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
                tree = ChatConversationService(store.persistence.db).get_conversation_tree(
                    conversation_id, root_limit=10_000, depth_cap=10_000
                )
            nodes = list(console_messages_from_conversation_tree(tree, db=self.app.chachanotes_db))
            leaf = self.app.chachanotes_db.get_conversation_active_leaf(conversation_id)
        session = store.restore_persisted_session(
            title=title,
            workspace_id=workspace_id,
            persisted_conversation_id=conversation_id,
            all_nodes=nodes,
            active_leaf_persisted_id=leaf,
            activate=False,
        )
        if opening_prompt:
            store.set_session_draft(session.id, opening_prompt)
        self._invalidate_console_persisted_rows_cache()
        self.run_worker(self._sync_native_console_chat_ui, exclusive=True, group="console-sync")
        verb = "Forked" if tool == "fork_chat" else "New"
        self.app_instance.notify(f"{verb} chat created: {title}")
```

(Prefer the sync service read directly and drop the awaitable branch if `chat_conversation_scope_service.get_conversation_tree` is async in this build — the excerpt shows the hydration path awaiting it; simplest is to always use the sync `ChatConversationService`. `activate=False` requires Task 8.) Wire the kwargs entry next to `set_pending_skill_script` (:4854-4856): `"complete_agent_chat_create": self._complete_agent_chat_create,`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/Chat/test_console_chat_create_integration.py Tests/Chat/test_console_chat_create_confirm.py Tests/Chat/test_chat_create_confirm_card.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/UI/Screens/chat_screen.py Tests/Chat/test_console_chat_create_integration.py
git commit -m "feat: fork_chat/new_chat bridge closures, executor, and background session completion"
```

---

### Task 8: Store — non-activating restore and one-shot draft rehydration

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_store.py` (`restore_persisted_session` :1210-1316; draft rehydrate inside it)
- Test: `Tests/Chat/test_console_chat_store.py` (extend; reuse `FakePersistence` :1376-1464 and the real-DB variant :1676-1678)

**Interfaces:**
- Consumes: Task 7's `console_agent_handoff` metadata key; `restore_persisted_session` currently calls `create_session` WITHOUT `activate=False` (activates — excerpt H(3)); `set_session_draft` (:1957).
- Produces: `restore_persisted_session(..., activate: bool = True)` forwarded to `create_session(activate=activate)`; when the restored conversation's metadata contains `console_agent_handoff.draft`, the session draft is set from it and the key is removed from persisted metadata (one-shot: it survives restarts until the chat is first opened).

- [ ] **Step 1: Write the failing tests**

```python
def test_restore_persisted_session_activate_false_keeps_current(real_db_store):
    store, db = real_db_store  # adapt: ConsoleChatStore(persistence=ChatPersistenceService(db))
    first = store.create_session(title="A")
    conv = store.persistence.create_conversation(conversation_title="B")
    restored = store.restore_persisted_session(
        title="B", workspace_id=None, persisted_conversation_id=conv,
        all_nodes=[], activate=False,
    )
    assert store.active_session_id == first.id  # NOT switched
    assert restored.persisted_conversation_id == conv


def test_restore_rehydrates_handoff_draft_once(real_db_store):
    store, db = real_db_store
    conv = store.persistence.create_conversation(
        conversation_title="C",
        metadata={"console_agent_handoff": {"draft": "please plan the migration",
                                            "created_via": "fork_chat", "source_run_id": "r9"}},
    )
    restored = store.restore_persisted_session(
        title="C", workspace_id=None, persisted_conversation_id=conv,
        all_nodes=[], activate=False,
    )
    assert restored.draft == "please plan the migration"
    # one-shot: the persisted key was cleared
    row = db.get_conversation_by_id(conv)
    assert "console_agent_handoff" not in (row.get("metadata") or "{}")


def test_restore_without_handoff_leaves_draft_alone(real_db_store):
    store, db = real_db_store
    conv = store.persistence.create_conversation(conversation_title="D")
    restored = store.restore_persisted_session(
        title="D", workspace_id=None, persisted_conversation_id=conv,
        all_nodes=[], activate=False,
    )
    assert restored.draft == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Chat/test_console_chat_store.py -k "restore_persisted or handoff" -v`
Expected: FAIL with `TypeError: restore_persisted_session() got an unexpected keyword argument 'activate'`

- [ ] **Step 3: Implement** — in `restore_persisted_session`: add `activate: bool = True` to the signature; forward `activate=activate` into its internal `create_session(...)` call; after `session.persisted_conversation_id = str(persisted_conversation_id)`, add:

```python
            handoff_draft = None
            db = getattr(self.persistence, "db", None)
            if db is not None:
                row = db.get_conversation_by_id(str(persisted_conversation_id))
                raw_metadata = (row or {}).get("metadata") or "{}"
                try:
                    metadata_obj = json.loads(raw_metadata) if isinstance(raw_metadata, str) else {}
                except ValueError:
                    metadata_obj = {}
                handoff = metadata_obj.pop("console_agent_handoff", None) if isinstance(metadata_obj, dict) else None
                if isinstance(handoff, dict) and handoff.get("draft"):
                    handoff_draft = str(handoff["draft"])
                    # one-shot: clear the key so a later restore does not re-fill
                    # the composer. Optimistic-locked update (version from the
                    # row we just read); on a version race, skip clearing — the
                    # worst case is the draft re-filling once on a later restore.
                    try:
                        db.update_conversation(
                            str(persisted_conversation_id),
                            {"metadata": json.dumps(metadata_obj, allow_nan=False)},
                            int((row or {}).get("version") or 1),
                        )
                    except Exception:  # noqa: BLE001 — ConflictError et al.
                        logger.opt(exception=True).debug(
                            "Failed to clear console_agent_handoff after rehydrate"
                        )
            ...
            if handoff_draft:
                self.set_session_draft(session.id, handoff_draft)
```

(`json` is already imported in `console_chat_store.py`; `db.update_conversation(conversation_id, update_data, expected_version)` is the optimistic-locked signature at `DB/ChaChaNotes_DB.py:8746` — `metadata` expects a JSON string, which is what we pass. `logger` is the module's existing loguru logger; if the module uses a different name, mirror it.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/Chat/test_console_chat_store.py -v`
Expected: PASS (all — including pre-existing restore tests with default `activate=True`)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_store.py Tests/Chat/test_console_chat_store.py
git commit -m "feat: non-activating persisted-session restore and one-shot agent handoff draft rehydration"
```

---

### Task 9: Docs, task hygiene, and verification

**Files:**
- Modify: `Docs/User_Guide/console/agent-runs-and-tools.md`
- Modify: `backlog/tasks/task-32482 - Agent-chat-fork-and-spawn-tools-fork_chat-new_chat.md`
- Modify: `Docs/User_Guide/console.md` (one-paragraph workstream blurb, only if it lists agent tools; otherwise skip)

**Interfaces:** none (documentation + closure).

- [ ] **Step 1: Document the tools** — in `agent-runs-and-tools.md`, next to the `run_skill_script`/`install_skill` documentation, add a `fork_chat` / `new_chat` section stating: what each does; that every call requires approval (Allow / Allow for this session / Deny); that "Allow for this session" is per-tool and ends with the session; that the opening prompt is a draft the user sends; the mid-turn snapshot boundary (the agent's current reply is not in the fork); that chats open in the background with a toast; character-chat and temporary-chat refusals; and that the copied chat records lineage (parent + fork point).
- [ ] **Step 2: Update the backlog task** — check off the ACs in TASK-32482, add the `## Implementation Notes` section (approach, files touched, decisions: no-migration, session-scoped remember, denial guard, one-shot draft rehydration, deferred follow-ups TASK-32480 + preset PR), set status via `backlog task edit 32482 -s Done --notes "..."` only after steps 3-4 pass.
- [ ] **Step 3: Run the full targeted verification list**

```bash
pytest Tests/Chat/test_chat_persistence_service.py \
       Tests/Chat/test_chat_conversation_service.py \
       Tests/Agents/test_agent_chat_create_tools.py \
       Tests/Agents/test_agent_runtime.py \
       Tests/Chat/test_console_chat_create_confirm.py \
       Tests/Chat/test_chat_create_confirm_card.py \
       Tests/Chat/test_console_chat_create_integration.py \
       Tests/Chat/test_console_chat_store.py \
       Tests/Chat/test_console_skill_script_confirm.py \
       Tests/Chat/test_console_rewind_summarize.py -v
```

Expected: PASS across the board (rewind suite included as a regression guard on active-leaf semantics).

- [ ] **Step 4: Live verification** (per `backlog/docs/lessons-live-verification.md`; requires the user or a driver session with the app running and a reachable provider): have an agent propose two workstreams and call `fork_chat` + `new_chat`; confirm the card shows full bodies; allow; verify toast, workspace listing (both membership + persisted paths), drafts in the composers; send in the fork and verify the source chat does not receive it; restart the app and verify both chats and (unopened) drafts persist. Record the outcome in the task notes.
- [ ] **Step 5: Commit**

```bash
git add Docs/User_Guide/console/agent-runs-and-tools.md "backlog/tasks/task-32482 - Agent-chat-fork-and-spawn-tools-fork_chat-new_chat.md"
git commit -m "docs: fork_chat/new_chat user guide + task closure"
```

---

## Sequencing note (vs. the provider-routing PR)

This plan is written against the working tree **including** the uncommitted provider-routing WIP (`agent_service.py` runtime-schema block at :2500-2556, `console_agent_bridge.py` flag plumbing at :2823-2880/:3343-3346). If that PR rebases, re-verify four anchors before starting Task 4: the `_run_one` runtime_schemas block end, the `LoopDeps` construction tail, the `build_first_request_schema_plan` flag list, and the bridge flags helper. All edits here are additive (new lines only) except the two `create_conversation`/`restore_persisted_session` signature extensions, which are in files the routing PR does not touch.
