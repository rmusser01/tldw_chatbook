# Local Todo Task API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the racy full-list `todo_write` tool with stable-ID create/update/get/list operations that remain correct across concurrent parent and fleet-child calls and ordinary in-process Console navigation.

**Architecture:** Add one stdlib-only `SessionTodoStore` that owns task records, ID allocation, compare-and-swap mutation, navigation snapshots, and the two-lock callback-ordering protocol. `LocalToolProvider` remains the permission/canonical-schema/JSON boundary and closes four tool handlers over that store. Google and Cohere native adapters project fresh provider-compatible disclosure copies without mutating the strict canonical schemas; exact raw handlers remain final enforcement. `ConsoleChatSession` owns one store, the controller injects it into each reconstructed provider, and the existing bridge renders defensive snapshots without persisting them durably. Provider/native Task 4 and Console-composition Task 5 are one merge/deploy unit.

**Tech Stack:** Python 3.11, standard-library `threading`/`json`/`dataclasses`, existing `LocalToolProvider` and `ToolResult` seams, Textual screen-state projection, pytest, Loguru only at existing application boundaries.

**Design:** `Docs/superpowers/specs/2026-08-11-local-todo-task-api-design.md`

**Task:** `backlog/tasks/task-13216 - LocalToolProvider-todo_write-is-last-write-wins-under-concurrent-fleet-children.md`

**ADR required:** yes

**ADR path:** `backlog/decisions/032-local-agent-tool-permission-boundary.md`

**Reason:** ADR-032 owns the local-tool permission and provider boundary; amend it with the stable-ID task API, CAS, and session-state ownership instead of creating a competing ADR. Native schema projection repairs compatibility inside the existing Google/Cohere provider adapters and does not require another ADR.

---

## File structure and responsibility map

- **Create:** `tldw_chatbook/Agents/session_todo_store.py` — pure state owner: validation, stable IDs, CAS, two-lock mutation/callback protocol, defensive reads, and pure-data snapshot export/import. No Textual, permission-store, provider, config, or Chat imports.
- **Create:** `Tests/Agents/test_session_todo_store.py` — focused store behavior, mutation, snapshot, deterministic concurrency, callback/deadlock, and mutation-kill tests.
- **Modify:** `tldw_chatbook/Agents/local_tool_provider.py` — replace conditional `todo_write` registration with four schemas/handlers, strict raw-argument validation, and compact byte-aware JSON pages.
- **Modify:** `Tests/Agents/test_local_tool_provider.py` — exact catalog/schema/tags, raw argument rejection, JSON result shapes, pagination, privacy, and absence without a store.
- **Modify:** `tldw_chatbook/LLM_Calls/LLM_API_Calls.py` — project strict canonical schemas into Google `parametersJsonSchema` and Cohere's supported disclosure subset without aliasing or mutation.
- **Modify:** `Tests/Chat/test_google_native_tools.py`, `Tests/Chat/test_cohere_native_tools.py` — exact offline native payloads, unsupported-keyword stripping/lowering, and canonical-schema non-mutation proofs.
- **Modify:** `tldw_chatbook/Chat/console_chat_store.py` — replace the mutable `todos` list with one `SessionTodoStore` per `ConsoleChatSession`.
- **Modify:** `tldw_chatbook/Chat/console_chat_controller.py` — inject the session store and defensive transcript callback into each local provider.
- **Modify:** `Tests/Chat/test_console_local_review_hook.py` — prove composition uses the session's exact store and emits markers only for successful mutations.
- **Modify:** `Tests/Chat/test_console_chat_store.py` — prove ordinary durable conversation persistence/resume does not serialize task records or the next-ID counter.
- **Modify:** `tldw_chatbook/UI/Console_Modules/session.py` — export/import the pure task snapshot in the existing explicit Console screen-state projection; fail soft and payload-free for malformed state.
- **Modify:** `Tests/UI/test_console_resume_active_path.py` — navigation round trip, legacy missing state, malformed state, and deleted-ID high-water coverage.
- **Modify:** `tldw_chatbook/Chat/console_agent_bridge.py` — update marker prose and sanitize terminal control characters at display time only.
- **Modify:** `Tests/Chat/test_console_agent_bridge.py` — marker rendering/control-character regression coverage.
- **Modify:** `Tests/Agents/test_local_tools_integration.py` — migrate the stale minimal todo find/load workflow during Task 4, then add expanded permission/discovery evidence in Task 7.
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
MAX_TODO_NUMBER = (1 << 53) - 1
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
Public task-ID numeric values and versions stay in the portable JSON
exact-integer domain `1..2**53-1`; fixed atomic exhaustion applies when an ID
or version increment would exceed that domain.
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

Cover create/get/ordered list, defensive copies, 50-task capacity, one
`in_progress`, exact built-in integer versions in `1..MAX_TODO_NUMBER`,
canonical decimal IDs in the same numeric domain, 500-character
content/activeForm bounds, unknown statuses, blank content, and strict UTF-8
rejection. Test exact maximum, one-over, and 100,000-digit IDs; exact maximum,
one-over, and very-large versions/`next_id` integers; reject oversized ID text
lexically before decimal conversion. Pin atomic rejection:

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
    maximum = str(MAX_TODO_NUMBER)
    if type(value) is not str or not value or len(value) > len(maximum):
        raise TodoStoreError(f"invalid {field}")
    if (
        not value.isascii()
        or not value.isdecimal()
        or value[0] == "0"
        or (len(value) == len(maximum) and value > maximum)
    ):
        raise TodoStoreError(f"invalid {field}")
    return value


def _version(value: object) -> int:
    if type(value) is not int or not 1 <= value <= MAX_TODO_NUMBER:
        raise TodoStoreError("invalid expected_version")
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

Test exact valid round trip, missing/deleted ID high-water preservation,
duplicate IDs, noncanonical IDs, invalid types/fields, over-cap input, multiple
`in_progress`, and `next_id <= max(live ids)`. Add boundary snapshots proving:
live IDs and versions accept `MAX_TODO_NUMBER` but reject one-over and
very-long values; `next_id` accepts `1..MAX_TODO_NUMBER + 1` but rejects either
side; the upper sentinel may sit above a live maximum ID; and existing
creation-order and deleted-ID-gap semantics remain unchanged. Mutating exported
input/output must not alias internal state.

- [ ] **Step 5: Implement snapshot import/export**

Validate into temporary local values first and assign only after the full
payload passes. The accepted payload has exact keys `next_id` and `tasks`; each
task has only public record keys. Each live ID and version is bounded by
`MAX_TODO_NUMBER`. `next_id` is an exact built-in integer in
`1..MAX_TODO_NUMBER + 1` and must exceed every live numeric ID; the upper value
is the exhaustion sentinel, never a live public ID. Creation at
`next_id == MAX_TODO_NUMBER` issues the final ID once and commits the sentinel.
Any later create raises the fixed `task id space exhausted` error without
state change or callback; after full request validation, check exhaustion
before the live-task capacity condition so every sentinel-state create has that
fixed result.

- [ ] **Step 6: Run focused tests and mutation probes**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Agents/test_session_todo_store.py -q
```

Then temporarily weaken and restore each guard: return the live record instead
of a copy; accept `isinstance(version, int)`; skip strict UTF-8; reset `next_id`
from only current live count; remove the `MAX_TODO_NUMBER` lexical/numeric
checks; and allow create past the exhaustion sentinel. Each corresponding
focused test must fail before the code is restored.

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

Cover partial patch, same-value patch increments once, activeForm removal with
`None`, delete-only behavior, delete result `{id, deleted, version}`, update-
after-delete not found, stale conflict with fixed non-reflective message, and
the one-`in_progress` invariant. Seed boundary records through validated
snapshots: update and delete at `MAX_TODO_NUMBER - 1` must return version
`MAX_TODO_NUMBER`; either operation on a record already at the maximum must
raise fixed `task version exhausted` with no state change or callback. Pin the
existing precedence: complete input validation, not-found, conflict, version
exhaustion, then proposed-record/global-invariant validation. Ensure every
validation/not-found/conflict/exhaustion failure emits no callback and changes
no state.

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

`ObservedLock` is a test-only context-manager wrapper around the real lock; it
must delegate `__enter__`/`__exit__` without changing production. Also force:
two creates; two jointly-valid updates on different IDs; same-ID stale
conflict; two different IDs racing to `in_progress`; and 49 tasks plus two
creates. When ID space permits beyond the next allocation, assert one success,
one fixed capacity error, and exactly 50 live tasks. At the terminal boundary
(`next_id == MAX_TODO_NUMBER`), assert the final ID is issued once and the loser
receives fixed `task id space exhausted`, because exhaustion precedes capacity.
Use a test-only parking mapping to stop the first contender inside the selected
real `len`/`get`/`setitem` state operation, then observe the second immediately
before its real mutation-lock acquisition. This forced interleaving makes each
concurrency test fail if either lock disappears and avoids scheduler-delay
evidence.

Add explicit callback contracts:

- successive successful mutations produce snapshots in commit order;
- mutating a received snapshot cannot mutate store state or a later snapshot;
- failed validation/conflict/not-found emits no callback;
- a raising callback leaves the mutation committed and returns the created/updated record;
- a credential/path-shaped callback exception produces exactly the fixed payload-free warning with no structured exception field or private fragments;
- same-thread create/update/delete attempted inside a callback fail fast with
  one fixed payload-free error rather than deadlocking or mutating state; and
- callback containment covers `BaseException`, preserving the never-raise seam
  after commit.

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
            was_active = getattr(self._callback_context, "active", False)
            self._callback_context.active = True
            try:
                on_change(snapshot)
            except BaseException:
                _LOG.warning("Session todo change callback failed.")
            finally:
                self._callback_context.active = was_active
        return _copy_record(result)
```

Call `_reject_callback_mutation()` at the start of each public mutation before
validation or lock acquisition. The state lock must be released before
callback execution. The mutation lock remains held through the callback. Reads
take only the state lock. Do not expose either lock.

- [ ] **Step 5: Add the direct/cross-thread callback read deadlock test**

Run the callback read and reentrant-mutation tests with an explicit
`multiprocessing.get_context("spawn")` and bounded joins. Define every child
entry point at module scope so it remains picklable under macOS/Windows spawn.
Inside the read callback, call `store.get()` directly and have another thread
call `store.list_after(None)`; both must complete before callback returns. In a
`finally`, terminate and then kill a stuck child if needed, join it, and close
both process and queue handles on success, timeout, and start failure. Add
regressions proving that timeout reaps the child and start failure closes every
constructed handle. This turns holding `_state_lock` across callback into a
deterministic failure without hanging or leaking pytest children. Also make a
raising callback contain a credential/path-shaped sentinel; assert the mutation
succeeds and the one fixed warning contains no exception, content, path, or
credential fragments.

- [ ] **Step 6: Verify GREEN and kill the guards**

Run the whole store file. Then temporarily remove, one at a time, the mutation
lock, state lock, version equality, numeric-exhaustion checks, callback
reentrancy guard, defensive-copy operation, and one-in-progress check. Each
required focused test must turn red; restore after each probe. The Task 3 review
hardening—fail-fast callback mutation, forced real-operation interleavings, and
spawn cleanup on all exits—is part of the accepted plan, not disposable test
scaffolding.

- [ ] **Step 7: Commit atomic mutation behavior**

```bash
git add tldw_chatbook/Agents/session_todo_store.py \
  Tests/Agents/test_session_todo_store.py
git commit -m "fix(todo): serialize versioned task mutations"
```

## Task 4: Replace `todo_write` and project strict schemas to native providers

**Files:**
- Modify: `tldw_chatbook/Agents/local_tool_provider.py`
- Modify: `Tests/Agents/test_local_tool_provider.py`
- Modify: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py`
- Modify: `Tests/Chat/test_google_native_tools.py`
- Modify: `Tests/Chat/test_cohere_native_tools.py`
- Modify: `Tests/Agents/test_local_tools_integration.py`

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

Pin the full strict canonical JSON Schema: exact required fields, enums,
nullable update `activeForm`, string/numeric bounds, delete-only conditional,
and `additionalProperties: false` for all four tools. The `id` and `cursor`
string schemas must accept exactly canonical decimal values in
`1..MAX_TODO_NUMBER`—including an exact upper-bound pattern, not merely a
16-character limit—and `expected_version` must declare integer
`minimum: 1`/`maximum: MAX_TODO_NUMBER`. Schema tests accept the exact boundary
and reject one-over before handler invocation. These `ToolSchema` objects are
the authoritative local/MCP/UI contract; native projections must not replace,
weaken, alias, or mutate them.

- [ ] **Step 2: Write table-driven raw-boundary RED tests**

Invoke handlers directly through `provider.invoke()` with missing/unknown keys,
bool or `int`-subclass `expected_version`, integer ID/cursor,
zero/leading-zero/signed ID/cursor, null activeForm on create, empty update,
delete plus another mutation field, caller-supplied version/status/id on
create, and lone surrogates. For ID, expected version, and cursor, test the
exact `MAX_TODO_NUMBER` boundary, one-over, and a very-long string/integer.
Assert schema and raw invocation agree: only the exact boundary is accepted;
every failure returns a fixed bounded `ToolResult` with no state or callback.

Add a provider-level callback-failure test: inject a callback that raises a credential/path-shaped sentinel, invoke `todo_create`, and assert `ToolResult.ok is True`, the compact JSON result is the committed record, the store contains it, and the one fixed diagnostic contains none of the sentinel fragments. This proves the store's containment remains true at the public provider boundary.

- [ ] **Step 3: Write native-projection and stale-integration RED tests**

In `Tests/Chat/test_google_native_tools.py`, pass a canonical task schema that
contains the strict range, pattern, `additionalProperties`, nullable, and
delete-conditional keywords through the real request converter. Assert the
exact `FunctionDeclaration` contains `parametersJsonSchema` equal to the full
canonical schema and has no `parameters` field.

In `Tests/Chat/test_cohere_native_tools.py`, pass the same canonical fixture
through the real Cohere v2 converter. Assert the exact recursively projected
copy preserves object/property names, types, descriptions, `required`, `enum`,
supported `anyOf`, and `additionalProperties`; lowers a nullable union to the
supported Cohere nullable shape; and contains no `allOf`, `oneOf`, `not`,
numeric `minimum`/`maximum` variants, string `minLength`/`maxLength`, `pattern`,
anchor, or lookahead constraint.
For both converters, retain a deep copy of the canonical input, assert it is
unchanged after conversion, assert projected nested dictionaries do not alias
it, mutate the captured projection, and assert the canonical input remains
unchanged.

Invoke the exact provider handler with a value that the lowered Cohere
disclosure cannot express—such as an over-bound `expected_version` or
noncanonical ID—and assert the fixed corrective failure with no state change or
callback. The transport projection is never treated as final validation.

Migrate the minimal stale todo workflow in
`Tests/Agents/test_local_tools_integration.py` from
`find_tools("todo") -> load_tools("local:todo_write") -> todo_write` to
`find_tools("todo") -> load_tools("local:todo_create") -> todo_create` over an
injected `SessionTodoStore`. Assert the compact JSON created record, committed
store state, and one `risk_floored=True` mutation approval. Expanded read-only,
parent/fleet, and permission migration coverage remains in Task 7.

All native projection tests inspect mocked request payloads. They require no
live Google or Cohere network call.

- [ ] **Step 4: Run provider, native-projection, and integration RED**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Agents/test_local_tool_provider.py \
  -k "todo or task_tools" -q
../../.venv/bin/python -m pytest \
  Tests/Chat/test_google_native_tools.py \
  Tests/Chat/test_cohere_native_tools.py -q
../../.venv/bin/python -m pytest \
  Tests/Agents/test_local_tools_integration.py \
  -k "todo" -q
```

Expected: failures because the old `todo_write` catalog/handler/integration
expectation remains, the Google payload uses the incompatible `parameters`
field, and the Cohere payload retains keywords outside its documented
strict-tools subset.

- [ ] **Step 5: Implement strict handlers and four canonical schemas**

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

After exact-key validation, route `todo_get`/`todo_update` IDs and `todo_list`
cursors through the store's bounded canonical-ID validator before numeric
conversion, and route `expected_version` through the exact built-in bounded
integer validator. Mirror those ceilings in the schemas. Reject very-long ID
and cursor strings lexically before decimal conversion is attempted.

Handlers call `store.create/update/get/list_after` and serialize every success using:

```python
def _todo_json(payload: object) -> str:
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(text.encode("utf-8")) > _MAX_RESULT_BYTES:
        raise TodoStoreError("task result exceeds the result limit")
    return text
```

No task response may pass through `_fit_result` in an oversized form; the handler must construct a complete valid response first.

- [ ] **Step 6: Implement non-mutating native schema projections**

In `_google_tools_payload`, deep-copy the canonical `function.parameters` into
the converted declaration's `parametersJsonSchema` field and omit
`parameters`. Google declares those two fields mutually exclusive; do not
translate the full schema into the narrower OpenAPI subset.

In `_cohere_tools_payload`, recursively construct a fresh disclosure copy
bounded to Cohere's strict-tools keyword subset. Preserve object/property names,
types, descriptions, `required`, `enum`, supported `anyOf`, and
`additionalProperties`; lower the canonical nullable union to Cohere's
supported nullable shape. At every depth omit `allOf`, `oneOf`, `not`, numeric
`minimum`/`maximum` variants, string `minLength`/`maxLength`, and all `pattern`
regex constraints, including anchors and lookaheads. Keep Cohere's
OpenAI-like outer `function.parameters` envelope.

Neither converter may modify or retain mutable nested aliases into the
canonical tool schema. Do not weaken the canonical `ToolSchema` to fit either
transport. A call that satisfies a lowered Cohere disclosure but violates a
canonical bound, pattern, or conditional reaches the exact raw handler and
returns its bounded corrective validation error.

- [ ] **Step 7: Implement byte-aware `todo_list` pages**

Build candidate pages in creation order and append a task only if the complete
`{tasks, next_cursor}` candidate stays within `_MAX_RESULT_BYTES`. Use the
returned task's ID as `next_cursor` only when more live tasks remain. Accept a
bounded canonical cursor as an exclusive numeric lower bound even if
deleted/unissued/future; a never-issued/future cursor is valid only through
`MAX_TODO_NUMBER`.

- [ ] **Step 8: Add maximum ASCII/multibyte pagination tests**

Fill 50 records with 500-character content and activeForm, then traverse every page for ASCII and `é`/multibyte content. For each response:

```python
payload = json.loads(result.content)
assert len(result.content.encode("utf-8")) <= 32 * 1024
```

Assert exact once-only ID coverage; delete a page-ending task before
continuation; create after the first page and assert it appears once; the
maximum-domain future cursor yields `{tasks: [], next_cursor: None}`. Serialize
boundary ID/version/tombstone/list responses and parse each with `json.loads`;
assert every result is complete valid JSON, every public task number is in
`1..MAX_TODO_NUMBER`, and every UTF-8 result remains within the 32-KiB cap.

- [ ] **Step 9: Verify provider, native, and migrated integration tests**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/Agents/test_local_tool_provider.py \
  Tests/Chat/test_google_native_tools.py \
  Tests/Chat/test_cohere_native_tools.py \
  Tests/Agents/test_local_tools_integration.py -q
```

No live provider credential or network call is required. Mutation-check
removal of canonical `additionalProperties`, bool rejection, schema/raw
`MAX_TODO_NUMBER` checks, very-long lexical rejection, complete-response byte
measurement, and `ensure_ascii=False`. Also make Google emit `parameters`, let
one unsupported Cohere composition/range/regex keyword survive, skip nullable
lowering, and alias/mutate one canonical nested schema; each corresponding test
must fail, then restore.

- [ ] **Step 10: Commit the provider/native migration checkpoint**

```bash
git add tldw_chatbook/Agents/local_tool_provider.py \
  Tests/Agents/test_local_tool_provider.py \
  tldw_chatbook/LLM_Calls/LLM_API_Calls.py \
  Tests/Chat/test_google_native_tools.py \
  Tests/Chat/test_cohere_native_tools.py \
  Tests/Agents/test_local_tools_integration.py
git commit -m "feat(todo): expose stable task operations"
```

This intermediate commit is a review checkpoint only. Do not merge, release,
or deploy it without Task 5: the current Console still supplies the legacy
mutable list. Do not add a temporary `todo_write` or list compatibility shim.

## Task 5: Wire the store into Console sessions and navigation

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/UI/Console_Modules/session.py`
- Modify: `Tests/Chat/test_console_local_review_hook.py`
- Modify: `Tests/Chat/test_console_chat_store.py`
- Modify: `Tests/UI/test_console_resume_active_path.py`

**Atomic merge gate:** Task 4 and Task 5 are one deploy/merge unit. Task 5 must
remove the legacy list seam and pass the combined reachable suites below before
the two commits receive combined review.

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

Remove `todos` and pass only `SessionTodoStore` at the provider seam. Do not add
a temporary list adapter or restore `todo_write`. This store is still omitted
from durable Chat persistence; it appears only in the in-process screen
projection below.

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

- [ ] **Step 7: Verify the combined reachable provider/integration/Console unit**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/Agents/test_local_tool_provider.py \
  Tests/Agents/test_local_tools_integration.py \
  Tests/Chat/test_google_native_tools.py \
  Tests/Chat/test_cohere_native_tools.py \
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

Only the completed Task 4 + Task 5 commit range is eligible for merge, release,
or deployment. Review the combined range after the Step 7 suites pass; the Task
4 checkpoint alone is intentionally non-deployable.

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

- [ ] **Step 1: Expand the migrated find/load integration coverage**

Task 4 already migrates the minimal stale workflow to
`find_tools("todo") -> load_tools("local:todo_create") -> todo_create`. Build on
that replacement path with a read-only `todo_get` or `todo_list` call proving
its empty tags do not acquire the mutation floor, while the created compact
JSON result and injected-store state remain reachable through the real service.

- [ ] **Step 2: Run the migrated and expanded integration nodes**

Run the Task 4 migrated node plus the new read-only node. The harness already
uses `make_service(..., todo_store=SessionTodoStore())` and replacement-tool
allowed names from the atomic Task 4 migration; do not reintroduce the legacy
list or `todo_write` expectation.

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
- Modify: `tldw_chatbook/MCP/local_store.py` — reject the reserved external
  profile before persistence and filter pre-existing reserved profile/catalog
  state during load.
- Modify: `tldw_chatbook/MCP/hub_tool_catalog.py` — drop a raw reserved
  external-profile record before constructing a `HubTool`.
- Modify: `Tests/MCP/test_local_store.py` — pin save/load behavior, associated
  state filtering, valid-profile controls, and exact whitespace/case policy.
- Modify: `Tests/MCP/test_hub_tool_catalog.py` — pin the raw-projection guard
  and demonstrate the pre-fix permission-identity collision.
- Modify: `Docs/User_Guide/mcp.md` — replace the stale Hub session-todo claim
  with the live workspace/Git/web inventory and explicit Console-only task
  boundary.
- Modify: `Tests/MCP/test_mcp_documentation_contract.py` — reject both literal
  and synonymous claims that session todo/task tools are Hub inventory.

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

### Reserved identity quality follow-up

The normal external-profile save path trims surrounding whitespace and rejects
`:` or embedded whitespace, but it did not reject the normalized `__local__`
segment that ADR-032 already assigns to the synthetic Console workspace-tool
principal. Both hand-written store JSON and raw catalog records bypass that
validator. Before the fix, a discovered external `fs_write` can therefore
become `local:__local__::fs_write`, identical to the real workspace tool's Hub
and permission key.

- [ ] **Step 6: Commit the governing boundary before production changes**

Add unchecked measurable acceptance criteria to TASK-13216, amend ADR-032 and
the design with the reserved identity, and add these exact files and gates to
Task 8. Keep the task In Progress. Commit only those four governance files as
`docs(todo): reserve local tool identity`.

- [ ] **Step 7: Write reserved-identity RED tests**

Using the real `LocalMCPStore`, prove that `save_profile()` accepts
`__local__`. Hand-write a store containing that profile plus discovery and
runtime state and prove it loads into an external catalog. Directly pass the
same record to `local_tools_from_record()` and prove it yields
`local:__local__::fs_write`. Where the real permission resolver permits, copy
the genuine workspace `fs_write` schema/description into that spoof and prove
one permission identity resolves both. Add an ordinary valid-profile control.

The post-fix assertions must require rejection before persistence, complete
load-time removal of the reserved profile and its associated discovery/runtime
state, and empty raw projection. Explicitly test exact `__local__`, existing
surrounding-whitespace trimming, embedded-whitespace rejection, and case
variants as still-valid distinct ids. A space-wrapped `__local__` must
normalize to the exact reserved token and be rejected.

- [ ] **Step 8: Implement all three narrow guards**

Reject the exact reserved id from the save-time profile validator. Filter an
existing reserved profile and its associated discovery snapshot and runtime
state in `LocalMCPStoreState.from_dict()` so it is inert immediately after
load; do not rename, reinterpret, or persist a cleanup. Independently return no
tools for a raw reserved record in `local_tools_from_record()`. Preserve every
other current profile-id rule and error behavior.

- [ ] **Step 9: Correct and contract-test the live MCP user guide**

Describe the Hub local catalog as workspace, read-only Git, and web tools.
State explicitly that `todo_create`, `todo_update`, `todo_get`, and `todo_list`
require a Console `SessionTodoStore` and are not Hub tools. Do not claim any old
permission migration. The documentation contract must reject synonymous
session-todo/session-task Hub inventory wording, not only `todo_write`.

- [ ] **Step 10: Prove mutations and run focused security/static gates**

Temporarily remove each save, load, and raw-projection guard independently;
the matching test must turn RED, then be restored. Re-run the original exploit
and prove it no longer reproduces while the legitimate external-profile
control remains. Run the local-store, Hub catalog, workbench, documentation,
and permission suites plus Ruff format/check, focused mypy, Bandit, compile,
and diff checks for the exact changed files.

- [ ] **Step 11: Commit the reserved-identity fix**

Stage only `tldw_chatbook/MCP/local_store.py`,
`tldw_chatbook/MCP/hub_tool_catalog.py`, `Tests/MCP/test_local_store.py`,
`Tests/MCP/test_hub_tool_catalog.py`, `Docs/User_Guide/mcp.md`, and
`Tests/MCP/test_mcp_documentation_contract.py`. Commit as
`fix(mcp): reserve local workspace tool identity`.

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
  Tests/Chat/test_google_native_tools.py \
  Tests/Chat/test_cohere_native_tools.py \
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
  tldw_chatbook/LLM_Calls/LLM_API_Calls.py \
  tldw_chatbook/Chat/console_chat_store.py \
  tldw_chatbook/Chat/console_chat_controller.py \
  tldw_chatbook/Chat/console_agent_bridge.py \
  tldw_chatbook/UI/Console_Modules/session.py \
  tldw_chatbook/MCP/local_store.py \
  tldw_chatbook/MCP/hub_tool_catalog.py \
  Tests/Agents/test_session_todo_store.py \
  Tests/Agents/test_local_tool_provider.py \
  Tests/Agents/test_local_tools_integration.py \
  Tests/Agents/test_fleet_runtime.py \
  Tests/Chat/test_google_native_tools.py \
  Tests/Chat/test_cohere_native_tools.py \
  Tests/Chat/test_console_local_review_hook.py \
  Tests/Chat/test_console_chat_store.py \
  Tests/Chat/test_console_agent_bridge.py \
  Tests/UI/test_console_resume_active_path.py \
  Tests/MCP/test_local_server_tools.py \
  Tests/MCP/test_control_plane_permissions.py \
  Tests/MCP/test_gateway_runtime_tools.py \
  Tests/MCP/test_local_store.py \
  Tests/MCP/test_hub_tool_catalog.py \
  Tests/MCP/test_mcp_documentation_contract.py \
  Tests/UI/test_mcp_workbench.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Agents/session_todo_store.py \
  tldw_chatbook/Agents/local_tool_provider.py \
  tldw_chatbook/LLM_Calls/LLM_API_Calls.py \
  tldw_chatbook/Chat/console_chat_store.py \
  tldw_chatbook/Chat/console_chat_controller.py \
  tldw_chatbook/Chat/console_agent_bridge.py \
  tldw_chatbook/UI/Console_Modules/session.py \
  tldw_chatbook/MCP/local_store.py \
  tldw_chatbook/MCP/hub_tool_catalog.py \
  Tests/Agents/test_session_todo_store.py \
  Tests/Agents/test_local_tool_provider.py \
  Tests/Agents/test_local_tools_integration.py \
  Tests/Agents/test_fleet_runtime.py \
  Tests/Chat/test_google_native_tools.py \
  Tests/Chat/test_cohere_native_tools.py \
  Tests/Chat/test_console_local_review_hook.py \
  Tests/Chat/test_console_chat_store.py \
  Tests/Chat/test_console_agent_bridge.py \
  Tests/UI/test_console_resume_active_path.py \
  Tests/MCP/test_local_server_tools.py \
  Tests/MCP/test_control_plane_permissions.py \
  Tests/MCP/test_gateway_runtime_tools.py \
  Tests/MCP/test_local_store.py \
  Tests/MCP/test_hub_tool_catalog.py \
  Tests/MCP/test_mcp_documentation_contract.py \
  Tests/UI/test_mcp_workbench.py
../../.venv/bin/python -m mypy \
  tldw_chatbook/Agents/session_todo_store.py \
  tldw_chatbook/Agents/local_tool_provider.py \
  tldw_chatbook/LLM_Calls/LLM_API_Calls.py \
  tldw_chatbook/Chat/console_chat_store.py \
  tldw_chatbook/Chat/console_chat_controller.py \
  tldw_chatbook/Chat/console_agent_bridge.py \
  tldw_chatbook/UI/Console_Modules/session.py \
  tldw_chatbook/MCP/local_store.py \
  tldw_chatbook/MCP/hub_tool_catalog.py
../../.venv/bin/python -m bandit -q \
  tldw_chatbook/Agents/session_todo_store.py \
  tldw_chatbook/Agents/local_tool_provider.py \
  tldw_chatbook/LLM_Calls/LLM_API_Calls.py \
  tldw_chatbook/Chat/console_chat_store.py \
  tldw_chatbook/Chat/console_chat_controller.py \
  tldw_chatbook/Chat/console_agent_bridge.py \
  tldw_chatbook/UI/Console_Modules/session.py \
  tldw_chatbook/MCP/local_store.py \
  tldw_chatbook/MCP/hub_tool_catalog.py
../../.venv/bin/python -m compileall -q \
  tldw_chatbook/Agents tldw_chatbook/LLM_Calls tldw_chatbook/Chat \
  tldw_chatbook/UI/Console_Modules \
  tldw_chatbook/MCP/local_store.py \
  tldw_chatbook/MCP/hub_tool_catalog.py
git diff --check origin/dev...HEAD
git diff --check
```

If implementation changes any additional Python file discovered by the stale-contract scan, add that exact path to every applicable command before running the gate.

- [ ] **Step 4: Perform file-by-file self-review and final mutation pass**

Review against every design §9 bullet. Re-run the required mutation kills:
mutation lock, state lock, CAS comparison, defensive copy, one-in-progress
guard, pagination byte counting, Google `parametersJsonSchema`, Cohere
unsupported-keyword stripping/nullable lowering, canonical-schema non-mutation,
screen-state high-water projection, terminal control sanitization, and shared
fleet store. Restore after every probe, then rerun the affected focused tests.

- [ ] **Step 5: Request independent code review**

Review the exact committed diff against the approved design and ADR-032. Any tracked review fix requires rerunning Steps 1–4 and a fresh review of the new exact commit.

- [ ] **Step 6: Close TASK-13216 only after all evidence is final**

Use the Backlog CLI to check all seven ACs, add concise Implementation Notes, and mark Done. The notes must name the store/provider/navigation/marker changes, the two-lock trade-off, exact test/static evidence, mutation evidence, and ADR-032 addendum. Add a lessons entry only for a real reusable trap encountered during implementation.

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
