# Local Todo Task API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the racy full-list `todo_write` tool with stable-ID create/update/get/list operations that remain correct across concurrent parent and fleet-child calls and ordinary in-process Console navigation.

**Architecture:** Add one stdlib-only `SessionTodoStore` that owns task records, ID allocation, compare-and-swap mutation, navigation snapshots, and the two-lock callback-ordering protocol. `LocalToolProvider` remains the permission/schema/JSON boundary and closes four tool handlers over that store; `ConsoleChatSession` owns one store, the controller injects it into each reconstructed provider, and the existing bridge renders defensive snapshots without persisting them durably.

**Tech Stack:** Python 3.11, standard-library `threading`/`json`/`dataclasses`, existing `LocalToolProvider` and `ToolResult` seams, Textual screen-state projection, pytest, Loguru only at existing application boundaries.

**Design:** `Docs/superpowers/specs/2026-08-11-local-todo-task-api-design.md`

**Task:** `backlog/tasks/task-13216 - LocalToolProvider-todo_write-is-last-write-wins-under-concurrent-fleet-children.md`

**ADR required:** yes

**ADR path:** `backlog/decisions/032-local-agent-tool-permission-boundary.md`

**Reason:** ADR-032 owns the local-tool permission and provider boundary; amend it with the stable-ID task API, CAS, and session-state ownership instead of creating a competing ADR.

---

## File structure and responsibility map

- **Create:** `tldw_chatbook/Agents/session_todo_store.py` — pure state owner: validation, stable IDs, CAS, two-lock mutation/callback protocol, defensive reads, and pure-data snapshot export/import. No Textual, permission-store, provider, config, or Chat imports.
- **Create:** `Tests/Agents/test_session_todo_store.py` — focused store behavior, mutation, snapshot, deterministic concurrency, callback/deadlock, and mutation-kill tests.
- **Modify:** `tldw_chatbook/Agents/local_tool_provider.py` — replace conditional `todo_write` registration with four schemas/handlers, strict raw-argument validation, and compact byte-aware JSON pages.
- **Modify:** `Tests/Agents/test_local_tool_provider.py` — exact catalog/schema/tags, raw argument rejection, JSON result shapes, pagination, privacy, and absence without a store.
- **Modify:** `tldw_chatbook/Chat/console_chat_store.py` — replace the mutable `todos` list with one `SessionTodoStore` per `ConsoleChatSession`.
- **Modify:** `tldw_chatbook/Chat/console_chat_controller.py` — inject the session store and defensive transcript callback into each local provider.
- **Modify:** `Tests/Chat/test_console_local_review_hook.py` — prove composition uses the session's exact store and emits markers only for successful mutations.
- **Modify:** `Tests/Chat/test_console_chat_store.py` — prove ordinary durable conversation persistence/resume does not serialize task records or the next-ID counter.
- **Modify:** `tldw_chatbook/UI/Console_Modules/session.py` — export/import the pure task snapshot in the existing explicit Console screen-state projection; fail soft and payload-free for malformed state.
- **Modify:** `Tests/UI/test_console_resume_active_path.py` — navigation round trip, legacy missing state, malformed state, and deleted-ID high-water coverage.
- **Modify:** `tldw_chatbook/Chat/console_agent_bridge.py` — update marker prose and sanitize terminal control characters at display time only.
- **Modify:** `Tests/Chat/test_console_agent_bridge.py` — marker rendering/control-character regression coverage.
- **Modify:** `Tests/Agents/test_local_tools_integration.py` — real find/load/permission flow for the replacement tools.
- **Modify:** `Tests/Agents/test_fleet_runtime.py` — real shared-provider parent/fleet concurrent-create path.
- **Modify:** `Tests/MCP/test_control_plane_permissions.py` — prove an obsolete `todo_write` grant does not authorize any replacement tool and pin create/update mutation floors versus get/list reads.
- **Modify:** `tldw_chatbook/MCP/local_server_tools.py`, `tldw_chatbook/MCP/server.py`, `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py` — update current prose to say all four task tools are absent without a Console session store.
- **Modify:** `Tests/MCP/test_local_server_tools.py`, `Tests/MCP/test_gateway_runtime_tools.py`, `Tests/UI/test_mcp_workbench.py` — exact external/Hub absence of all four tools and removal of `todo_write`.
- **Modify:** `Docs/superpowers/specs/2026-08-04-local-agent-tools-design.md`, `Docs/superpowers/specs/2026-08-05-local-agent-tools-phases-3-4-replan.md` — clearly supersede the old TodoWrite contract while preserving historical context.
- **Modify:** `backlog/decisions/032-local-agent-tool-permission-boundary.md` — required TASK-13216 addendum.
- **Modify:** live local-tool documentation discovered by the final stale-contract scan; do not rewrite historical implementation plans merely to make them look current.
- **Modify at closeout only:** TASK-13216, and `backlog/docs/lessons-testing-evidence.md` only if implementation produces a genuinely reusable incident.

### Shared constants and public shapes

Keep these in `session_todo_store.py` and import them into the provider/tests rather than duplicating literals:

```python
MAX_TODO_ITEMS = 50
MAX_TODO_CONTENT_CHARS = 500
TODO_STATUSES = ("pending", "in_progress", "completed")

TodoRecord = dict[str, object]
TodoChangeCallback = Callable[[list[TodoRecord]], None]
```

The store raises a store-local `TodoStoreError(ValueError)` containing short deterministic, non-reflective messages. `LocalToolProvider.invoke()` already turns handler exceptions into bounded failed `ToolResult`s; do not make the store depend on `Tools.local_tool_impls.LocalToolError` just to reuse an exception name.

## Task 1: Record the governing contract before production changes

**Files:**
- Modify: `backlog/decisions/032-local-agent-tool-permission-boundary.md`
- Modify: `Docs/superpowers/specs/2026-08-04-local-agent-tools-design.md`
- Modify: `Docs/superpowers/specs/2026-08-05-local-agent-tools-phases-3-4-replan.md`

- [ ] **Step 1: Add the ADR-032 TASK-13216 addendum**

Record these decisions, without implementation detail:

```markdown
**Addendum (TASK-13216, 2026-08-11): session tasks use item-oriented CAS.**
The Console-local `todo_write` full-list replacement is retired. A supplied
Console session store registers `todo_create`, `todo_update`, `todo_get`, and
`todo_list`; create/update remain permission-gated mutations, get/list are
read-only, and no task tool is registered without Console session state.
Stable session-local IDs, exact expected-version checks, and atomic mutation
preserve concurrent parent/fleet changes. State remains process-memory-only;
the Console screen snapshot carries pure task records and the next-ID
high-water mark solely across in-process navigation.
```

- [ ] **Step 2: Amend the older local-tool designs**

Add a prominent supersession note next to their `todo_write` sections pointing to the 2026-08-11 design. Preserve those documents as historical records; do not silently rewrite completed phase plans.

- [ ] **Step 3: Verify the governing references**

Run:

```bash
rg -n "TASK-13216|todo_create|todo_update|todo_get|todo_list" \
  backlog/decisions/032-local-agent-tool-permission-boundary.md \
  Docs/superpowers/specs/2026-08-04-local-agent-tools-design.md \
  Docs/superpowers/specs/2026-08-05-local-agent-tools-phases-3-4-replan.md
git diff --check
```

Expected: all three documents identify the stable-ID contract; diff check exits 0.

- [ ] **Step 4: Commit the governing change**

```bash
git add backlog/decisions/032-local-agent-tool-permission-boundary.md \
  Docs/superpowers/specs/2026-08-04-local-agent-tools-design.md \
  Docs/superpowers/specs/2026-08-05-local-agent-tools-phases-3-4-replan.md
git commit -m "docs(todo): amend local task permission contract"
```

## Task 2: Build the session store's records, validation, and navigation state

**Files:**
- Create: `tldw_chatbook/Agents/session_todo_store.py`
- Create: `Tests/Agents/test_session_todo_store.py`

- [ ] **Step 1: Write failing record and validation tests**

Cover create/get/ordered list, defensive copies, 50-task capacity, one `in_progress`, exact built-in integer versions, canonical decimal IDs, 500-character content/activeForm bounds, unknown statuses, blank content, and strict UTF-8 rejection. Pin atomic rejection:

```python
def test_create_rejects_lone_surrogate_without_allocating_an_id():
    store = SessionTodoStore()
    with pytest.raises(TodoStoreError, match="UTF-8"):
        store.create(content="bad\ud800")
    assert store.export_snapshot() == {"next_id": 1, "tasks": []}


def test_reads_are_defensive_and_keep_creation_order():
    store = SessionTodoStore()
    first = store.create(content="A")
    second = store.create(content="B", active_form="Doing B")
    first["content"] = "mutated outside"
    listed = store.list_after(None)
    listed[0]["content"] = "also outside"
    assert store.get("1")["content"] == "A"
    assert [item["id"] for item in store.list_after(None)] == ["1", "2"]
    assert second["version"] == 1
```

- [ ] **Step 2: Run the focused tests and capture RED**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Agents/test_session_todo_store.py -q
```

Expected: collection/import failure because `session_todo_store.py` does not exist.

- [ ] **Step 3: Implement the minimal store skeleton and exact validators**

Use a private record mapping and exact-type helpers. The public interface should be:

```python
class SessionTodoStore:
    def __init__(self) -> None: ...
    def create(
        self,
        *,
        content: object,
        active_form: object = _MISSING,
        on_change: TodoChangeCallback | None = None,
    ) -> TodoRecord: ...
    def update(
        self,
        *,
        task_id: object,
        expected_version: object,
        content: object = _MISSING,
        status: object = _MISSING,
        active_form: object = _MISSING,
        on_change: TodoChangeCallback | None = None,
    ) -> TodoRecord: ...
    def get(self, task_id: object) -> TodoRecord: ...
    def list_after(self, cursor: int | None) -> list[TodoRecord]: ...
    def export_snapshot(self) -> dict[str, object]: ...
    @classmethod
    def from_snapshot(cls, payload: object) -> "SessionTodoStore": ...
```

Validation helpers must use exact built-in types where required:

```python
def _canonical_id(value: object, *, field: str = "id") -> str:
    if type(value) is not str or not value.isascii() or not value.isdecimal():
        raise TodoStoreError(f"{field} must be a canonical positive decimal string")
    if value == "0" or value.startswith("0"):
        raise TodoStoreError(f"{field} must be a canonical positive decimal string")
    return value


def _version(value: object) -> int:
    if type(value) is not int or value < 1:
        raise TodoStoreError("expected_version must be an integer at least 1")
    return value


def _text(value: object, *, field: str, required: bool) -> str:
    if type(value) is not str or (required and not value.strip()):
        raise TodoStoreError(f"{field} must be a non-empty string")
    if len(value) > MAX_TODO_CONTENT_CHARS:
        raise TodoStoreError(f"{field} must be at most 500 characters")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise TodoStoreError(f"{field} must be valid UTF-8") from exc
    return value
```

Do not log or interpolate the rejected value or exception.

- [ ] **Step 4: Add snapshot import/export tests before implementing restore**

Test exact valid round trip, missing/deleted ID high-water preservation, duplicate IDs, noncanonical IDs, invalid types/fields, over-cap input, multiple `in_progress`, and `next_id <= max(live ids)`. Mutating exported input/output must not alias internal state.

- [ ] **Step 5: Implement snapshot import/export**

Validate into temporary local values first and assign only after the full payload passes. The accepted payload has exact keys `next_id` and `tasks`; each task has only public record keys. `next_id` is an exact positive integer and must exceed every live numeric ID.

- [ ] **Step 6: Run focused tests and mutation probes**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Agents/test_session_todo_store.py -q
```

Then temporarily weaken and restore each guard: return the live record instead of a copy; accept `isinstance(version, int)`; skip strict UTF-8; reset `next_id` from only current live count. Each corresponding focused test must fail before the code is restored.

- [ ] **Step 7: Commit the state model**

```bash
git add tldw_chatbook/Agents/session_todo_store.py \
  Tests/Agents/test_session_todo_store.py
git commit -m "feat(todo): add session task state model"
```

## Task 3: Add atomic CAS mutation and deterministic callback ordering

**Files:**
- Modify: `tldw_chatbook/Agents/session_todo_store.py`
- Modify: `Tests/Agents/test_session_todo_store.py`

- [ ] **Step 1: Write failing update/delete semantics tests**

Cover partial patch, same-value patch increments once, activeForm removal with `None`, delete-only behavior, delete result `{id, deleted, version}`, update-after-delete not found, stale conflict with fixed non-reflective message, and the one-`in_progress` invariant. Ensure validation/not-found/conflict emits no callback and changes no state.

- [ ] **Step 2: Write deterministic concurrency RED tests**

Use `threading.Barrier`/`Event`, never timing-only sleeps:

```python
def test_callback_serializes_mutation_commit_and_return():
    entered = threading.Event()
    release = threading.Event()
    second_waiting_on_mutation_lock = threading.Event()
    second_done = threading.Event()
    store = SessionTodoStore()

    # Wrap the real private lock with a test probe. The second thread signals
    # immediately before it attempts the real acquire. If production removes
    # the mutation-lock context entirely, this event is never set and the test
    # fails instead of passing vacuously on scheduler delay.
    store._mutation_lock = ObservedLock(
        store._mutation_lock,
        thread_name="second-mutation",
        before_acquire=second_waiting_on_mutation_lock.set,
    )

    def blocked(snapshot):
        entered.set()
        assert release.wait(2)

    first = Thread(target=lambda: store.create(content="A", on_change=blocked))
    second = Thread(
        target=lambda: (
            store.create(content="B"),
            second_done.set(),
        )
    )
    first.start()
    assert entered.wait(2)
    second.name = "second-mutation"
    second.start()
    assert second_waiting_on_mutation_lock.wait(2)
    assert not second_done.is_set()
    release.set()
    first.join(2)
    second.join(2)
    assert second_done.is_set()
```

`ObservedLock` is a test-only context-manager wrapper around the real lock; it must delegate `__enter__`/`__exit__` without changing production. Also force: two creates; two jointly-valid updates on different IDs; same-ID stale conflict; two different IDs racing to `in_progress`; and 49 tasks plus two creates (one success, one fixed capacity error, exactly 50 live tasks).

Add explicit callback contracts:

- successive successful mutations produce snapshots in commit order;
- mutating a received snapshot cannot mutate store state or a later snapshot;
- failed validation/conflict/not-found emits no callback;
- a raising callback leaves the mutation committed and returns the created/updated record; and
- a credential/path-shaped callback exception produces exactly the fixed payload-free warning with no structured exception field or private fragments.

- [ ] **Step 3: Run RED**

Run only the new test group with `-k "update or delete or concurrent or callback"`; expected failures should be missing atomic/CAS behavior, not fixture hangs.

- [ ] **Step 4: Implement the two-lock transaction**

Use one ordinary state lock and one ordinary mutation lock:

```python
def _mutate(self, commit, on_change):
    with self._mutation_lock:
        with self._state_lock:
            result = commit()
            snapshot = self._snapshot_locked()
        if on_change is not None:
            try:
                on_change(snapshot)
            except Exception:
                _LOG.warning("Session todo change callback failed.")
        return _copy_record(result)
```

The state lock must be released before callback execution. The mutation lock remains held through the callback. Reads take only the state lock. Do not expose either lock.

- [ ] **Step 5: Add the direct/cross-thread callback read deadlock test**

Run the callback test in a `multiprocessing` subprocess with a bounded join. Define the child entry point at module scope so it remains picklable under macOS/Windows `spawn`. Inside the callback, call `store.get()` directly and have another thread call `store.list_after(None)`; both must complete before callback returns. If the child is still alive at the deadline, terminate it and fail. This turns holding `_state_lock` across callback into a deterministic failure without hanging pytest. Also make a raising callback contain a credential/path-shaped sentinel; assert the mutation succeeds and the one fixed warning contains no exception, content, path, or credential fragments.

- [ ] **Step 6: Verify GREEN and kill the guards**

Run the whole store file. Then temporarily remove, one at a time, the mutation lock, state lock, version equality, defensive-copy operation, and one-in-progress check. Each required focused test must turn red; restore after each probe.

- [ ] **Step 7: Commit atomic mutation behavior**

```bash
git add tldw_chatbook/Agents/session_todo_store.py \
  Tests/Agents/test_session_todo_store.py
git commit -m "fix(todo): serialize versioned task mutations"
```

## Task 4: Replace `todo_write` with four strict provider tools

**Files:**
- Modify: `tldw_chatbook/Agents/local_tool_provider.py`
- Modify: `Tests/Agents/test_local_tool_provider.py`

- [ ] **Step 1: Replace old tests with failing catalog/schema tests**

Assert:

```python
TODO_TOOL_NAMES = ("todo_create", "todo_update", "todo_get", "todo_list")

def test_task_tools_are_conditional_and_todo_write_is_removed(tmp_path):
    without = make_provider(root=tmp_path)
    assert not ({*TODO_TOOL_NAMES, "todo_write"} & {e.name for e in without.list_catalog()})

    with_store = make_provider(root=tmp_path, todo_store=SessionTodoStore())
    names = [e.name for e in with_store.list_catalog()]
    assert all(name in names for name in TODO_TOOL_NAMES)
    assert "todo_write" not in names
    assert with_store.hub_tool_for("todo_create").tags == ("mutates",)
    assert with_store.hub_tool_for("todo_update").tags == ("mutates",)
    assert with_store.hub_tool_for("todo_get").tags == ()
    assert with_store.hub_tool_for("todo_list").tags == ()
```

Pin exact schema required fields, enums, nullable activeForm, bounds, and `additionalProperties: false` for all four tools.

- [ ] **Step 2: Write table-driven raw-boundary RED tests**

Invoke handlers directly through `provider.invoke()` with missing/unknown keys, bool expected_version, integer ID/cursor, zero/leading-zero/signed cursor, null activeForm on create, empty update, delete plus another mutation field, caller-supplied version/status/id on create, and lone surrogates. Assert failed `ToolResult`, fixed/bounded error, no state or callback.

Add a provider-level callback-failure test: inject a callback that raises a credential/path-shaped sentinel, invoke `todo_create`, and assert `ToolResult.ok is True`, the compact JSON result is the committed record, the store contains it, and the one fixed diagnostic contains none of the sentinel fragments. This proves the store's containment remains true at the public provider boundary.

- [ ] **Step 3: Run RED**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Agents/test_local_tool_provider.py \
  -k "todo or task_tools" -q
```

Expected: failures because the old `todo_write` catalog and handlers still exist.

- [ ] **Step 4: Implement strict handler projection and four schemas**

Change constructor/default-spec types from `list | None` to `SessionTodoStore | None`. Keep raw mapping validation in this module:

```python
def _exact_args(args: object, *, allowed: set[str], required: set[str]) -> dict:
    if type(args) is not dict:
        raise TodoStoreError("arguments must be an object")
    unknown = set(args) - allowed
    missing = required - set(args)
    if unknown:
        raise TodoStoreError("arguments contain unknown properties")
    if missing:
        raise TodoStoreError("required task arguments are missing")
    return args
```

Handlers call `store.create/update/get/list_after` and serialize every success using:

```python
def _todo_json(payload: object) -> str:
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(text.encode("utf-8")) > _MAX_RESULT_BYTES:
        raise TodoStoreError("task result exceeds the result limit")
    return text
```

No task response may pass through `_fit_result` in an oversized form; the handler must construct a complete valid response first.

- [ ] **Step 5: Implement byte-aware `todo_list` pages**

Build candidate pages in creation order and append a task only if the complete `{tasks, next_cursor}` candidate stays within `_MAX_RESULT_BYTES`. Use the returned task's ID as `next_cursor` only when more live tasks remain. Accept a canonical positive cursor as an exclusive numeric lower bound even if deleted/unissued/future.

- [ ] **Step 6: Add maximum ASCII/multibyte pagination tests**

Fill 50 records with 500-character content and activeForm, then traverse every page for ASCII and `é`/multibyte content. For each response:

```python
payload = json.loads(result.content)
assert len(result.content.encode("utf-8")) <= 32 * 1024
```

Assert exact once-only ID coverage; delete a page-ending task before continuation; create after the first page and assert it appears once; future cursor yields `{tasks: [], next_cursor: None}`.

- [ ] **Step 7: Verify provider tests and mutations**

Run all provider tests. Mutation-check removal of `additionalProperties`, bool rejection, complete-response byte measurement, and `ensure_ascii=False`; each corresponding test must fail, then restore.

- [ ] **Step 8: Commit the provider migration**

```bash
git add tldw_chatbook/Agents/local_tool_provider.py \
  Tests/Agents/test_local_tool_provider.py
git commit -m "feat(todo): expose stable task operations"
```

## Task 5: Wire the store into Console sessions and navigation

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/UI/Console_Modules/session.py`
- Modify: `Tests/Chat/test_console_local_review_hook.py`
- Modify: `Tests/Chat/test_console_chat_store.py`
- Modify: `Tests/UI/test_console_resume_active_path.py`

- [ ] **Step 1: Write controller wiring RED tests**

Replace old list assertions with the exact store object:

```python
provider, _ = controller._compose_local_provider(session_id=session.id)
created = provider.invoke("local:todo_create", {"content": "Ship it"})
assert created.ok
assert session.todo_store.get("1")["content"] == "Ship it"
assert markers == [(session.id, session.todo_store.list_after(None))]
```

No session, unknown session, or absent bridge must register none of the four task tools.

- [ ] **Step 2: Write navigation-state RED tests**

Create IDs 1–3, delete ID 2, serialize through `_console_session_to_state`, restore through `_console_session_from_state`, and assert records 1/3 plus `next_id == 4`; the next create must receive ID `4`. Also test a missing `todo_state` legacy payload (empty store) and malformed payload (empty store plus one exact fixed warning with no raw sentinel/path/key material).

In `Tests/Chat/test_console_chat_store.py`, persist a conversation whose live session store contains a task, then restore it through the existing durable `restore_persisted_session` seam. Assert the restored session has an empty store whose first create receives ID `1`. This pins the deliberate process-memory-only boundary rather than merely relying on the current persistence field list.

- [ ] **Step 3: Run RED**

Run the exact new nodes from `test_console_local_review_hook.py` and `test_console_resume_active_path.py`; expected failures are missing `todo_store` and missing `todo_state` projection.

Include the new durable non-persistence node from `Tests/Chat/test_console_chat_store.py` in this RED command; it must fail until `ConsoleChatSession` owns the replacement store.

- [ ] **Step 4: Change `ConsoleChatSession` ownership**

```python
from tldw_chatbook.Agents.session_todo_store import SessionTodoStore

todo_store: SessionTodoStore = field(default_factory=SessionTodoStore)
```

Remove `todos`. This store is still omitted from durable Chat persistence; it appears only in the in-process screen projection below.

- [ ] **Step 5: Update controller composition**

`_todo_wiring()` returns the exact session store plus a callback receiving defensive task snapshots:

```python
def _on_todo_change(tasks: list[dict[str, object]]) -> None:
    bridge.append_todo_marker(session_id, tasks)

return {"todo_store": session.todo_store, "on_todo_change": _on_todo_change}
```

Do not marshal through Textual or call back into the store.

- [ ] **Step 6: Add screen-state projection and fail-soft restore**

Serialize `"todo_state": session.todo_store.export_snapshot()`. On restore:

- key absent: create a fresh empty store with no warning;
- key present and valid: `SessionTodoStore.from_snapshot(raw)`;
- key present and invalid: create a fresh empty store and emit exactly `Console task state invalid; starting empty.` without exception interpolation.

Pass the resulting store in `session_kwargs` before constructing `ConsoleChatSession`.

- [ ] **Step 7: Verify navigation and composition**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/Chat/test_console_local_review_hook.py \
  Tests/Chat/test_console_chat_store.py \
  Tests/UI/test_console_resume_active_path.py -q
```

Mutation-check omission of `todo_state` from serialization and resetting `next_id` on restore; both navigation tests must fail.

- [ ] **Step 8: Commit Console ownership/navigation**

```bash
git add tldw_chatbook/Chat/console_chat_store.py \
  tldw_chatbook/Chat/console_chat_controller.py \
  tldw_chatbook/UI/Console_Modules/session.py \
  Tests/Chat/test_console_local_review_hook.py \
  Tests/Chat/test_console_chat_store.py \
  Tests/UI/test_console_resume_active_path.py
git commit -m "feat(todo): retain task state across console navigation"
```

## Task 6: Harden transcript rendering

**Files:**
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Modify: `Tests/Chat/test_console_agent_bridge.py`

- [ ] **Step 1: Write display-boundary RED tests**

Assert IDs/versions remain absent from the marker; create/update/delete snapshots still render creation order; empty list renders cleared. Add all control ranges:

```python
@pytest.mark.parametrize("control", ["\x00", "\x1f", "\x7f", "\x80", "\x9f"])
def test_todo_marker_replaces_terminal_controls_without_mutating_store(control):
    tasks = [{"id": "1", "version": 1, "content": f"left{control}right", "status": "pending"}]
    assert control not in format_todo_marker(tasks)
    assert tasks[0]["content"] == f"left{control}right"
```

- [ ] **Step 2: Run RED**

Run the marker tests; expected: controls survive current rendering.

- [ ] **Step 3: Implement display-only sanitization**

Add a small helper that replaces `ord(char) < 0x20`, `0x7F <= ord(char) <= 0x9F` with spaces, then flattens line breaks and truncates to 200 characters. Do not alter stored/JSON task text.

- [ ] **Step 4: Run bridge tests and commit**

```bash
../../.venv/bin/python -m pytest Tests/Chat/test_console_agent_bridge.py -q
git add tldw_chatbook/Chat/console_agent_bridge.py Tests/Chat/test_console_agent_bridge.py
git commit -m "fix(todo): sanitize transcript task markers"
```

## Task 7: Prove real permission, catalog, and parent/fleet paths

**Files:**
- Modify: `Tests/Agents/test_local_tools_integration.py`
- Modify: `Tests/Agents/test_fleet_runtime.py`
- Modify: `Tests/MCP/test_control_plane_permissions.py`

- [ ] **Step 1: Migrate the find/load permission integration test**

Script `find_tools("todo") -> load_tools("local:todo_create") -> todo_create`. Assert compact JSON result, task present in the injected store, and exactly one approval with `risk_floored=True`. Add a read-only `todo_get` or `todo_list` call proving its empty tags do not acquire the mutation floor.

- [ ] **Step 2: Run the integration test RED then GREEN**

Run the exact migrated nodes before changing their harness; capture RED from the missing new tool. Update `make_service(..., todo_store=SessionTodoStore())`, allowed tools, expected call order, and result parsing. Re-run to GREEN.

- [ ] **Step 3: Add a real fleet shared-provider concurrency test**

Use `make_fleet_service()` with one `LocalToolProvider` and one `SessionTodoStore` registered for both parent and child. Use a store/callback barrier so parent and child reach `todo_create` concurrently; after both complete, assert distinct IDs and both task contents. Assert run IDs differ and both tool results are successful. Do not replace this with two direct threads—the store unit tests already cover that lower seam.

- [ ] **Step 4: Add jointly-valid different-ID fleet updates**

Seed two tasks. Have parent and child update separate IDs with expected version 1 and jointly valid statuses; assert both version 2 records survive. Keep one-in-progress race at the store layer where the invariant can be driven deterministically without model-script timing.

- [ ] **Step 5: Pin permission migration and risk-floor behavior**

Using the real `UnifiedMCPControlPlaneService` test seam, persist an explicit `local:todo_write` allow entry, then resolve the four replacement `HubTool`s. Assert the obsolete entry authorizes none of them. Assert inherited server allow is risk-floored for create/update but remains allow for get/list; explicit per-tool grants still require each replacement tool's own current definition hash.

- [ ] **Step 6: Run focused fleet/integration tests and mutation probe**

Run the exact new nodes, then temporarily construct a separate store for the child provider path. The shared-state fleet test must fail; restore.

- [ ] **Step 7: Commit integration evidence**

```bash
git add Tests/Agents/test_local_tools_integration.py \
  Tests/Agents/test_fleet_runtime.py \
  Tests/MCP/test_control_plane_permissions.py
git commit -m "test(todo): prove shared parent fleet task state"
```

## Task 8: Update external absence contracts and live documentation

**Files:**
- Modify: `tldw_chatbook/MCP/local_server_tools.py`
- Modify: `tldw_chatbook/MCP/server.py`
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py`
- Modify: `Tests/MCP/test_local_server_tools.py`
- Modify: `Tests/MCP/test_gateway_runtime_tools.py`
- Modify: `Tests/UI/test_mcp_workbench.py`
- Modify: additional live docs found by the stale scan only when they describe the current runtime

- [ ] **Step 1: Write exact absence RED assertions**

Replace single-name assertions with:

```python
TASK_TOOL_NAMES = {"todo_create", "todo_update", "todo_get", "todo_list"}
assert "todo_write" not in names
assert TASK_TOOL_NAMES.isdisjoint(names)
```

Cover external MCP catalog, registrations, gateway-published locals, and Hub catalog view.

- [ ] **Step 2: Update production prose and live docs**

State that no Console `SessionTodoStore` is supplied outside the Console, so none of the four task tools is registered. Remove current-runtime prose implying `todo_write` remains the current API. Do not claim that old permission entries are migrated.

- [ ] **Step 3: Run focused MCP/UI tests**

```bash
../../.venv/bin/python -m pytest \
  Tests/MCP/test_local_server_tools.py \
  Tests/MCP/test_gateway_runtime_tools.py \
  Tests/UI/test_mcp_workbench.py -q
```

- [ ] **Step 4: Run the stale-contract scan**

```bash
rg -n "todo_write|full replacement todo|replaces.*todo|live.*todos.*list" \
  tldw_chatbook Docs README.md Tests \
  -g '!Docs/superpowers/plans/2026-08-05-local-agent-tools-phase3a.md' \
  -g '!backlog/tasks/task-2821*'
```

Classify every remaining hit. Allowed: explicit retirement/removal assertions, historical completed plans/tasks, and this task's motivating description. Forbidden: current production/docstrings, live user/developer docs, current inventory claims, or tests that still invoke `todo_write` successfully.

- [ ] **Step 5: Commit external/docs migration**

Stage exact reviewed files only and commit:

```bash
git add tldw_chatbook/MCP/local_server_tools.py \
  tldw_chatbook/MCP/server.py \
  tldw_chatbook/UI/MCP_Modules/mcp_workbench.py \
  Tests/MCP/test_local_server_tools.py \
  Tests/MCP/test_gateway_runtime_tools.py \
  Tests/UI/test_mcp_workbench.py
# Add only exact live-document paths changed after classifying the scan.
git commit -m "docs(todo): retire todo_write runtime contract"
```

## Task 9: Full verification, review, and backlog closeout

**Files:**
- Modify: `backlog/tasks/task-13216 - LocalToolProvider-todo_write-is-last-write-wins-under-concurrent-fleet-children.md`
- Modify only if an incident justifies it: `backlog/docs/lessons-testing-evidence.md`

- [ ] **Step 1: Run focused behavioral suites**

```bash
../../.venv/bin/python -m pytest \
  Tests/Agents/test_session_todo_store.py \
  Tests/Agents/test_local_tool_provider.py \
  Tests/Agents/test_local_tools_integration.py \
  Tests/Agents/test_fleet_runtime.py \
  Tests/Chat/test_console_local_review_hook.py \
  Tests/Chat/test_console_chat_store.py \
  Tests/Chat/test_console_agent_bridge.py \
  Tests/UI/test_console_resume_active_path.py \
  Tests/MCP/test_local_server_tools.py \
  Tests/MCP/test_control_plane_permissions.py \
  Tests/MCP/test_gateway_runtime_tools.py \
  Tests/UI/test_mcp_workbench.py -q
```

Expected: all selected tests pass; only already-characterized dependency warnings are acceptable.

- [ ] **Step 2: Run reachability suites**

```bash
../../.venv/bin/python -m pytest Tests/Agents Tests/Chat Tests/MCP -q
```

Then run the repository's full suite serially when no other repository-wide pytest is active:

```bash
../../.venv/bin/python -m pytest -q
```

Record exact totals and any independently reproduced upstream baseline; do not describe an incomplete or waived run as green.

No dependency, license expression, license file, or packaging-manifest change is planned. Verify that remains true with `git diff --name-only origin/dev...HEAD -- pyproject.toml requirements*.txt MANIFEST.in LICENSE`; expected output is empty. Then run the repository's applicable artifact/license contract once:

```bash
../../.venv/bin/python -m pytest \
  Tests/Packaging/test_installed_distribution.py::test_built_artifacts_match_distribution_contract -q
```

If a manifest/dependency/license file appears in the diff, stop and update the task scope/plan before treating the license gate as sufficient.

- [ ] **Step 3: Run static/security checks over every changed Python file**

First inventory the committed diff:

```bash
git diff --name-only origin/dev...HEAD
```

Then run:

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Agents/session_todo_store.py \
  tldw_chatbook/Agents/local_tool_provider.py \
  tldw_chatbook/Chat/console_chat_store.py \
  tldw_chatbook/Chat/console_chat_controller.py \
  tldw_chatbook/Chat/console_agent_bridge.py \
  tldw_chatbook/UI/Console_Modules/session.py \
  Tests/Agents/test_session_todo_store.py \
  Tests/Agents/test_local_tool_provider.py \
  Tests/Agents/test_local_tools_integration.py \
  Tests/Agents/test_fleet_runtime.py \
  Tests/Chat/test_console_local_review_hook.py \
  Tests/Chat/test_console_chat_store.py \
  Tests/Chat/test_console_agent_bridge.py \
  Tests/UI/test_console_resume_active_path.py \
  Tests/MCP/test_local_server_tools.py \
  Tests/MCP/test_control_plane_permissions.py \
  Tests/MCP/test_gateway_runtime_tools.py \
  Tests/UI/test_mcp_workbench.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Agents/session_todo_store.py \
  tldw_chatbook/Agents/local_tool_provider.py \
  tldw_chatbook/Chat/console_chat_store.py \
  tldw_chatbook/Chat/console_chat_controller.py \
  tldw_chatbook/Chat/console_agent_bridge.py \
  tldw_chatbook/UI/Console_Modules/session.py \
  Tests/Agents/test_session_todo_store.py \
  Tests/Agents/test_local_tool_provider.py \
  Tests/Agents/test_local_tools_integration.py \
  Tests/Agents/test_fleet_runtime.py \
  Tests/Chat/test_console_local_review_hook.py \
  Tests/Chat/test_console_chat_store.py \
  Tests/Chat/test_console_agent_bridge.py \
  Tests/UI/test_console_resume_active_path.py \
  Tests/MCP/test_local_server_tools.py \
  Tests/MCP/test_control_plane_permissions.py \
  Tests/MCP/test_gateway_runtime_tools.py \
  Tests/UI/test_mcp_workbench.py
../../.venv/bin/python -m mypy \
  tldw_chatbook/Agents/session_todo_store.py \
  tldw_chatbook/Agents/local_tool_provider.py \
  tldw_chatbook/Chat/console_chat_store.py \
  tldw_chatbook/Chat/console_chat_controller.py \
  tldw_chatbook/Chat/console_agent_bridge.py \
  tldw_chatbook/UI/Console_Modules/session.py
../../.venv/bin/python -m bandit -q \
  tldw_chatbook/Agents/session_todo_store.py \
  tldw_chatbook/Agents/local_tool_provider.py \
  tldw_chatbook/Chat/console_chat_store.py \
  tldw_chatbook/Chat/console_chat_controller.py \
  tldw_chatbook/Chat/console_agent_bridge.py \
  tldw_chatbook/UI/Console_Modules/session.py
../../.venv/bin/python -m compileall -q \
  tldw_chatbook/Agents tldw_chatbook/Chat tldw_chatbook/UI/Console_Modules
git diff --check origin/dev...HEAD
git diff --check
```

If implementation changes any additional Python file discovered by the stale-contract scan, add that exact path to every applicable command before running the gate.

- [ ] **Step 4: Perform file-by-file self-review and final mutation pass**

Review against every design §9 bullet. Re-run the required mutation kills: mutation lock, state lock, CAS comparison, defensive copy, one-in-progress guard, pagination byte counting, screen-state high-water projection, terminal control sanitization, and shared fleet store. Restore after every probe, then rerun the affected focused tests.

- [ ] **Step 5: Request independent code review**

Review the exact committed diff against the approved design and ADR-032. Any tracked review fix requires rerunning Steps 1–4 and a fresh review of the new exact commit.

- [ ] **Step 6: Close TASK-13216 only after all evidence is final**

Use the Backlog CLI to check all six ACs, add concise Implementation Notes, and mark Done. The notes must name the store/provider/navigation/marker changes, the two-lock trade-off, exact test/static evidence, mutation evidence, and ADR-032 addendum. Add a lessons entry only for a real reusable trap encountered during implementation.

- [ ] **Step 7: Commit closeout records**

```bash
git add 'backlog/tasks/task-13216 - LocalToolProvider-todo_write-is-last-write-wins-under-concurrent-fleet-children.md'
# Add lessons-testing-evidence.md only if Step 6 produced a justified entry.
git commit -m "docs(todo): close stable task API migration"
```

- [ ] **Step 8: Final cleanliness and ancestry check**

```bash
git status --short --branch
git diff --check origin/dev...HEAD
git log --oneline --decorate origin/dev..HEAD
```

Expected: clean worktree, no uncommitted diff, and only reviewed TASK-13216 commits ahead of the current integrated `origin/dev`.
