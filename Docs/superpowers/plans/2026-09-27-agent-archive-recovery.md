# Agent archive recovery implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user ask a Console agent to find a missing local archived conversation and explicitly restore that conversation, its workspace, or both without changing the current chat.

**Architecture:** Add a narrowly authenticated Console recovery capability alongside the selected Direct/RAG Library provider. A focused recovery package owns bounded search, immutable result references and confirmed operations; existing conversation/workspace services own persistence and the original-ID resume handoff owns opening. Keep timestamps separate from optimistic concurrency and report cross-store partial completion.

**Tech Stack:** Python >=3.12, Textual >=8.0.0,<9, SQLite/FTS5, Pydantic, asyncio/threading, pytest. No new dependency.

**Spec:** [Approved archive recovery design](../specs/2026-09-17-agent-archive-recovery-design.md). Read it and [ADR-166](../../../backlog/decisions/166-agent-assisted-archive-recovery.md) before execution.

## Global constraints

- Scope: local saved non-deleted conversations; interactive user-originated primary runs in ordinary local Console sessions. No remote, temporary, subagent, MCP or unattended execution.
- Assistant Library access must be Allowed for agent tools; the dedicated capability works in Direct and RAG-only modes. Manual Library recovery remains available when agent access is Blocked.
- Preserve frozen per-turn authority, destination disclosures, persona restrictions, kill switches and runtime execution gates. Tool names and actor provenance are app-owned.
- `search_archived_conversations`: query maximum 512 characters; default page 10, maximum 20; snippets maximum 320 characters; serialized response maximum 32 KiB; interruptible storage budget five seconds.
- Date intervals use timezone-qualified input, inclusive lower/exclusive upper bounds. Either current chat/workspace archive event may match. Unknown historical times remain unknown.
- Retain at most 10 result pages per source session for 30 minutes from creation. One recovery confirmation per session, expiring after five minutes. At most 20 completed presentation receipts per source session. Active operations stay owned until settlement.
- Restore only after one-shot exact-target UI confirmation. Neither stored Allow nor remembered approval substitutes for it. Restoration does not activate, send, resume agents, move ownership or copy messages.
- Use real SQLite for persistence evidence, actual mounted event dispatch for interaction evidence, and targeted tests only. No full suite without user opt-in.
- UI: read `backlog/docs/design-language.md`; compose `$ds-*` tokens; edit source TCSS and rebuild; do not hand-edit the generated bundle or shadow terminal/global shortcuts.
- Existing checkout contains unrelated edits. At execution, use the worktree skill and an isolated checkout containing the approved docs. Do not overwrite or commit unrelated changes. Do not use another process's profile/database for live checks.

ADR required: yes

ADR path: `backlog/decisions/166-agent-assisted-archive-recovery.md`

Reason: implement the approved agent contracts, Library-mode exception, local chronology and cross-store confirmation ownership. No additional ADR is needed unless execution changes these boundaries.

## Baseline and file responsibilities

Planning baseline: 2026-09-27. Conversation schema is 73 and workspace schema is 8. This plan names migrations v73→v74 and v8→v9; re-read both constants at execution and renumber only if another migration has landed, updating tests/artifacts together. Do not edit the earlier archive migration.

Existing seams:

- `DB/ChaChaNotes_DB.py`: `set_conversations_archived`, archive queries, versioned migrations and sync triggers.
- `DB/Workspace_DB.py`, `Workspaces/models.py`, `Workspaces/registry_service.py`: workspace lifecycle and name uniqueness. Current unarchive lacks an expected revision; add one.
- `Chat/conversation_archive_actions.py`: app-owned lifecycle reservations/cache publication and cancellation shielding. Reuse it for conversation writes.
- `Agents/tool_catalog.py`: `register_builtin_library_provider` accepts one exact Direct/RAG class/name set. Preserve it unchanged; recovery gets a separate exact registration.
- `Chat/console_runtime.py`: `ConsoleSubmissionOrigin`, queued/wake admission, view-hook slots, session close and disposal. Capture user origin here, not from model arguments or mere primary-agent status.
- `Agents/agent_runtime.py`, `agent_service.py`, `Chat/console_agent_bridge.py`: injected runtime-callable precedent for `fork_chat`/`new_chat`; waits must not be hidden behind the generic short tool timeout.
- `Widgets/Console/console_transcript.py`, `Widgets/Chat_Widgets/chat_task_cards.py`, `UI/Console_Modules/skill.py`: current Console presentation/confirmation patterns. `Widgets/tool_message_widgets.py` is the legacy character-chat surface and is not this feature's target.

New focused files:

| File | Responsibility |
| --- | --- |
| `DB/conversation_archive_queries.py` | Parameterized recovery query and bounded message projection, with no UI imports. |
| `Chat/archive_recovery/__init__.py` | Lightweight package, no startup construction. |
| `Chat/archive_recovery/contracts.py` | Validated requests, immutable snapshots, result/outcome types, constants. |
| `Chat/archive_recovery/search_service.py` | Workspace snapshot, date/search orchestration and page byte fitting. |
| `Chat/archive_recovery/result_store.py` | Session-bound reference/cursor retention and trusted card provenance. |
| `Chat/archive_recovery/operations.py` | Preflight and confirmed lifecycle effects, including post-write observation. |
| `Chat/archive_recovery/coordinator.py` | Confirmation, operation ownership, cancellation and deduplication. |
| `Chat/archive_recovery/runtime.py` | Main-loop/agent-worker bridge and session/run capability lifecycle. |
| `Agents/archive_recovery_tool_provider.py` | Exact reserved schemas and authenticated discovery/execution checks. |
| `Widgets/Chat_Widgets/archive_recovery_cards.py` | Native result and confirmation widgets driven only by typed projections. |
| `UI/Console_Modules/archive_recovery.py` | Thin view/event adapter, explicit Preview/Open navigation. |

Do not introduce a generic action framework, persistent replay journal or competing archive index. Define new module contents in the task that first needs them, not in an empty scaffolding commit.

## Execution discipline

Before each implementation task: read its Backlog file, set it In Progress, then add its Implementation Plan referencing this plan/ADR. The tasks below are delivery units; the numbered steps within them are the red/green/review/commit cycle. Run narrow failing tests before production changes, expand the matrix as each behavior lands, and commit only that task's files. Inspect failures against the baseline; do not relabel existing unrelated failures as passes.

Use `.venv/bin/python -m pytest` and the project's installed lint/formatter. After affected Python changes, run `ruff check` on those exact files and check formatting with the repository's configured formatter; record pre-existing diagnostics rather than formatting whole large modules. Documentation-only steps use link/whitespace/Backlog guards, not application tests.

## Backlog delivery tasks

| Plan task | Backlog task |
| --- | --- |
| 1 | [TASK-33014](../../../backlog/tasks/task-33014%20-%20Persist-archive-chronology-and-workspace-lifecycle-revisions.md) |
| 2 | [TASK-33015](../../../backlog/tasks/task-33015%20-%20Search-local-archive-recovery-candidates-with-bounded-results.md) |
| 3 | [TASK-33016](../../../backlog/tasks/task-33016%20-%20Retain-stable-archive-recovery-result-references.md) |
| 4 | [TASK-33017](../../../backlog/tasks/task-33017%20-%20Coordinate-confirmed-archive-restoration-without-navigation.md) |
| 5 | [TASK-33018](../../../backlog/tasks/task-33018%20-%20Expose-authorized-archive-recovery-tools-to-Console-agents.md) |
| 6 | [TASK-33019](../../../backlog/tasks/task-33019%20-%20Add-actionable-archive-recovery-results-and-confirmations-to-chat.md) |
| 7 | [TASK-33020](../../../backlog/tasks/task-33020%20-%20Qualify-and-document-the-complete-agent-archive-recovery-journey.md) |

## Task 1: Persist chronology and conditional workspace lifecycle

**Files**

- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py`, `DB/Workspace_DB.py`, `Workspaces/models.py`, `Workspaces/registry_service.py`, `Chat/chat_conversation_service.py`, `Chat/conversation_archive_actions.py`.
- Create: `tldw_chatbook/DB/migrations/chachanotes_v73_to_v74_archive_chronology.sql`, `workspaces_v8_to_v9_archive_chronology.sql`.
- Test: create `Tests/DB/test_archive_chronology.py`; extend `Tests/DB/test_conversation_archive.py`, `Tests/DB/test_workspace_db.py`, `Tests/Workspaces/test_workspace_registry_service.py`.

**Interfaces**

- Conversation metadata gains `archived_at: str | None`; existing versioned archive result format remains unchanged.
- `WorkspaceRecord` gains `archived_at: str | None = None`, `archive_revision: int = 0`.
- `archive_workspace(workspace_id: str, *, expected_archive_revision: int | None = None) -> WorkspaceRecord`.
- `unarchive_workspace(workspace_id: str, *, name: str | None = None, expected_archive_revision: int | None = None, expected_name: str | None = None) -> WorkspaceRecord`.
- Both methods raise new `WorkspaceArchiveConflict` for an expected-state mismatch; existing callers without an expected value still capture/check current state within their write transaction. Recovery always supplies the captured revision/name. No-op repeats retain timestamps/revisions and preserve existing public refusal semantics.

- [ ] **1. Add a failing real-store clock/revision regression.** Use existing constructors and deterministic registry clock; put this test in the new test file:

```python
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService


def test_workspace_cycle_uses_revision_not_clock(tmp_path):
    db = WorkspaceDB(tmp_path / "workspaces.sqlite")
    registry = LocalWorkspaceRegistryService(
        db, now_factory=lambda: "2026-09-20T12:00:00+00:00"
    )
    try:
        registry.create_workspace(workspace_id="w", name="Recovered")
        first = registry.archive_workspace("w")
        registry.unarchive_workspace("w")
        second = registry.archive_workspace("w")
        assert second.archived_at == first.archived_at
        assert second.archive_revision == first.archive_revision + 2
    finally:
        db.close()
```

- [ ] **2. Establish red.** Run `.venv/bin/python -m pytest Tests/DB/test_archive_chronology.py -q`. Expect missing chronology/revision fields, not a fixture/import failure. Add stale revision, backward clock and future-modification-time cases using the same store.
- [ ] **3. Add transactional migrations and lifecycle updates.** The schema additions are:

```sql
ALTER TABLE conversations ADD COLUMN archived_at TEXT;
CREATE INDEX idx_conversations_archive_chronology
    ON conversations(deleted, archived, archived_at DESC, id);
```

```sql
ALTER TABLE workspace_records ADD COLUMN archived_at TEXT;
ALTER TABLE workspace_records ADD COLUMN archive_revision INTEGER NOT NULL DEFAULT 0
    CHECK (archive_revision >= 0);
```

Read/recreate current conversation sync trigger definitions in the new migration, excluding local timestamp-only writes while retaining shared-payload changes, version movement and retained latest shared event. Register both migrations in the existing transactional runners; update projections and workspace row conversion. Use actual UTC time for chronology and `archive_revision = archive_revision + 1` for workspace transitions. The conditional restore statement is:

```sql
UPDATE workspace_records
SET archived = 0, archived_at = NULL,
    archive_revision = archive_revision + 1, name = ?, updated_at = ?
WHERE workspace_id = ? AND archived = 1
  AND archive_revision = ? AND name = ?;
```

Check duplicate names and perform this write inside one immediate transaction. Apply the symmetric state/revision condition to archiving. Search every writer of `workspace_records.archived`, including Default repair and Undo; route lifecycle transitions through the same invariant. Conversation archive writes set/clear `archived_at` in the existing version-checked statement. Do not add it to sync/export payloads.
- [ ] **4. Qualify migrations and existing behavior.** Add fresh/upgraded/rollback tests; old archives have NULL time; archive/no-op/restore/Undo cycles preserve IDs, messages, branches and sync event counts; stale workspace revisions fail even with equal dates. Run the four test files listed above plus `Tests/Chat/test_conversation_archive_actions.py` and `Tests/UI/test_workspace_archive_review_regressions.py`.
- [ ] **5. Review and commit.** Verify no old migration changed; scoped lint/format and `git diff --check`; update task AC/notes; commit `feat: persist archive chronology and workspace revisions` with explicit paths.

## Task 2: Search the local recovery inventory

**Files**

- Create: recovery `contracts.py`, `search_service.py`, package `__init__.py`, `DB/conversation_archive_queries.py`.
- Modify: narrow delegates in `Chat/chat_conversation_service.py`, `DB/ChaChaNotes_DB.py`.
- Test: create `Tests/Chat/test_archive_recovery_search.py`, `Tests/DB/test_archive_recovery_query.py`.

**Interfaces**

- `ArchiveSearchRequest` is a Pydantic model with `query: str = ""`, `workspace_id: str | None = None`, `archived_after: datetime | None = None`, `archived_before: datetime | None = None`, `include_unknown_archive_times: bool = False`, `limit: int = 10`; forbid extras, strict booleans/integers, timezone-naive dates, query over 512 chars and page limits outside 1..20. Tool-level `cursor` is resolved by Task 3, not accepted as SQL state.
- Frozen `ArchiveWorkspaceSnapshot(workspace_id: str, name: str, archived: bool, archived_at: str | None, archive_revision: int)`.
- Frozen `ArchiveTarget(conversation_id: str, conversation_version: int, title: str, conversation_archived: bool, workspace_id: str | None, workspace: ArchiveWorkspaceSnapshot | None)`; normalize Default using existing scope rules, never treat a missing named workspace as Default.
- Frozen `ArchiveCandidate(target: ArchiveTarget, conversation_archived_at: str | None, excerpt: str, match_basis: tuple[str, ...], sort_key: tuple[int, str, str], chronology_incomplete: bool, match_message_id: str | None, allowed_scopes: tuple[str, ...])`.
- Frozen `ArchiveSearchPage(items: tuple[ArchiveCandidate, ...], next_anchor: tuple[int, str, str] | None, workspace_digest: str, unknown_times_present: bool)`.
- `ArchiveSearchService(chat_service: ChatConversationService, workspace_registry: LocalWorkspaceRegistryService, *, monotonic: Callable[[], float] = time.monotonic)`; synchronous `search(request: ArchiveSearchRequest, *, after: tuple[int, str, str] | None = None, expected_workspace_digest: str | None = None) -> ArchiveSearchPage` runs on the owning storage worker.
- `ArchiveRecoveryError(code: str, message: str)` carries sanitized codes (`invalid_argument`, `stale`, `unavailable`, `search_incomplete`, `storage_error`); adapters translate it into structured outcomes.
- `query_archive_recovery_page(db: CharactersRAGDB, request_data: Mapping[str, object], workspace_rows: tuple[Mapping[str, object], ...], *, after: tuple[int, str, str] | None, deadline: float, monotonic: Callable[[], float]) -> dict[str, object]` in the DB query module returns `items`, `next_anchor` and `unknown_times_present`. Items contain Task-2 candidate fields using raw storage identities; `search_service.py` constructs immutable candidates and adds the workspace digest. The DB module never imports the higher-level recovery package.

- [ ] **1. Add failing workspace-only and chronology tests.** Construct `CharactersRAGDB(tmp_path / "chat.sqlite", client_id="recovery")`, `WorkspaceDB(tmp_path / "workspace.sqlite")`, `ChatConversationService` and `LocalWorkspaceRegistryService`. Seed chats with `scope_type="workspace"` and a durable `workspace_id`; insert visible messages with `db.add_message`. Archive only the workspace; search must find its still-active chat. Also exercise the event rule directly with a small pure helper introduced in this task:

```python
def test_date_window_matches_earlier_chat_event():
    from tldw_chatbook.Chat.archive_recovery.search_service import matching_archive_events

    assert matching_archive_events(
        chat_at="2026-09-21T10:00:00+00:00",
        workspace_at="2026-09-25T10:00:00+00:00",
        after="2026-09-21T00:00:00+00:00",
        before="2026-09-22T00:00:00+00:00",
    ) == ("chat",)
```

`matching_archive_events(*, chat_at: str | None, workspace_at: str | None, after: str | None, before: str | None) -> tuple[str, ...]` compares normalized aware datetimes. Add both, neither, undated, lower-inclusive and upper-exclusive tests.
- [ ] **2. Run red.** `.venv/bin/python -m pytest Tests/Chat/test_archive_recovery_search.py Tests/DB/test_archive_recovery_query.py -q`; expected failures are missing recovery query/service behavior. Prove system/tool/hidden-reasoning-only needles do not match, not merely that their snippets are omitted.
- [ ] **3. Implement the query with all filters before paging.** Capture a sorted workspace snapshot and SHA-256 digest (identity/name/archive revision/state/time); pass it as a bound JSON array into a SQLite `json_each` CTE, already supported by the database's JSON usage. Left-join on authoritative `conversations.workspace_id`. Eligible rows satisfy `deleted = 0 AND (c.archived = 1 OR w.archived = 1)`. Missing named workspaces retain only explicitly chat-archived candidates without mutation actions. An unknown explicit workspace or failed enumeration raises `unavailable`.

For date selection, use separate predicates for chat/workspace event times, then combine with OR. Use the latest qualifying date as the sort time; under no date interval use latest applicable known date. Unknown-only candidates form the trailing group. The keyset tuple is `(group ascending, normalized UTC sort_time descending, conversation ID ascending)`; encode all three in the cursor. Keep SQL parameterized and use escaped lexical/FTS terms, never raw MATCH syntax supplied by the model.

The pure event helper's core is:

```python
from datetime import datetime


def matching_archive_events(*, chat_at, workspace_at, after, before):
    lower = datetime.fromisoformat(after) if after else None
    upper = datetime.fromisoformat(before) if before else None
    matches = []
    for kind, value in (("chat", chat_at), ("workspace", workspace_at)):
        if value is None:
            continue
        event = datetime.fromisoformat(value)
        if (lower is None or event >= lower) and (upper is None or event < upper):
            matches.append(kind)
    return tuple(matches)
```

Normalize dates at the boundary; add public type hints/docstrings. Use this helper as an oracle for the SQL date matrix, not to filter an already-limited page. Role-restricted message predicates search only canonical visible user/assistant text; private serialized reasoning/control fields must not enter the query. Where legacy content cannot be proven to be visible text, exclude it from body matching and retain title/keyword discovery. Label inactive-branch provenance without changing the branch or promising a reader jump.

Install a SQLite progress handler only for the owned query connection, backed by a monotonic deadline covering workspace read/query/projection; clear it in `finally`. Bound lock waits by remaining budget. Fetch at most `limit + 1` candidates and bounded excerpt projections, not full transcripts. Fit serialized pages before issuing continuation; trim optional title/excerpt display text with explicit truncation, never identity fields. If no complete minimal row fits, fail explicitly; never return an endlessly repeating empty cursor page.
- [ ] **4. Run the full targeted search matrix.** Include >20 archived-workspace-only chats, mixed lifecycle states, Default, missing workspace, duplicate titles, Unicode, literal FTS operators, role restrictions, unknown dates, byte-boundary truncation, stable ties, changed workspace digest and query cancellation/deadline. Record `EXPLAIN QUERY PLAN` on representative data; do not claim that response limits bound scan cost. Run new query/search tests and `Tests/DB/test_conversation_archive_query_plan.py`.
- [ ] **5. Review and commit.** Ensure no embeddings import or temp persisted inventory, scoped lint/format and task closeout; commit `feat: add bounded cross-workspace archive search`.

## Task 3: Bind result references to source sessions

**Files**

- Create: `Chat/archive_recovery/result_store.py`, `Tests/Chat/test_archive_recovery_result_store.py`.
- Modify: `contracts.py` for `RecoveryCallKey` and the typed result projection.

**Interfaces**

- Frozen `RecoveryCallKey(session_id: str, run_id: str, call_id: str, argument_digest: str)`; digest is SHA-256 of canonical validated JSON arguments, never raw payload in logs.
- `ArchiveResultStore(*, monotonic: Callable[[], float] = time.monotonic)`.
- `publish(session_id: str, request: ArchiveSearchRequest, page: ArchiveSearchPage, *, model_visible: bool) -> dict[str, object]`: returns validated rows with opaque `result_ref`, `result_set_id` and optional cursor; retains immutable candidates internally.
- `resolve(session_id: str, result_ref: str, *, for_agent: bool = False) -> ArchiveCandidate`; raises `ArchiveRecoveryError` for expired/wrong-owner references or agent selection of a manual-only page.
- `resolve_cursor(session_id: str, cursor: str) -> tuple[ArchiveSearchRequest, tuple[int, str, str], str]`; opaque tokens point to app-owned normalized filters/anchor/digest, not deserialized arbitrary transcript arguments.
- `close_session(session_id: str) -> None` invalidates retained pages and cursors. Coordinator pins a copied immutable target separately while awaiting a decision.

- [ ] **1. Write failing retention/ownership tests.** Define the test's candidate builder using the exact Task-2 dataclasses; create an active Default workspace target with `conversation_archived=True`, an empty snippet and `allowed_scopes=("chat_only",)`. Publish two pages and prove the first reference still resolves to its original conversation. Use a mutable clock list instead of sleeping:

```python
clock = [100.0]
store = ArchiveResultStore(monotonic=lambda: clock[0])
published = store.publish("s1", request, page, model_visible=True)
ref = published["items"][0]["result_ref"]
assert store.resolve("s1", ref).target.conversation_id == "first"
clock[0] += 1801
with pytest.raises(ArchiveRecoveryError, match="expired"):
    store.resolve("s1", ref)
```

Define the input objects in that test with these exact constructors (import them from `contracts.py`):

```python
target = ArchiveTarget(
    conversation_id="first", conversation_version=3, title="First",
    conversation_archived=True, workspace_id=None, workspace=None,
)
candidate = ArchiveCandidate(
    target=target, conversation_archived_at="2026-09-21T10:00:00+00:00",
    excerpt="A remembered discussion", match_basis=("chat",),
    sort_key=(0, "2026-09-21T10:00:00+00:00", "first"),
    chronology_incomplete=False, match_message_id=None,
    allowed_scopes=("chat_only",),
)
request = ArchiveSearchRequest()
page = ArchiveSearchPage((candidate,), None, "digest", False)
```

Also test wrong session, 11th-page eviction, stale cursor filters and manual-only agent selection.
- [ ] **2. Run red.** `.venv/bin/python -m pytest Tests/Chat/test_archive_recovery_result_store.py -q`.
- [ ] **3. Implement the bounded registry.** Use per-session `OrderedDict` maps, `secrets.token_urlsafe`, monotonic expiries and a lock for worker/main-loop access. Snapshot supplied records; callers never receive mutable registry internals. Enforce limits at insert/resolve; eviction does not renumber visible rows. Store only live application provenance; importing a transcript never calls `publish`.

```python
import hashlib
import json


def argument_digest(arguments: dict[str, object]) -> str:
    encoded = json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
```

Add this function with validation performed before invocation. Result and cursor handles are random local references, not permission grants. A manual refresh creates `model_visible=False` results and does not append a tool message.
- [ ] **4. Verify and commit.** Run new result-store tests with deterministic expiry/close/restart cases; assert old/copy/imported JSON cannot reconstruct references. Scoped lint/format and task notes; commit `feat: retain session-owned archive recovery results`.

## Task 4: Coordinate exact-target restoration

**Files**

- Create: `Chat/archive_recovery/operations.py`, `coordinator.py`, `Tests/Chat/test_archive_recovery_operations.py`, `Tests/Chat/test_archive_recovery_coordinator.py`.
- Modify: recovery `contracts.py`; reuse `Chat/conversation_archive_actions.py` and workspace CAS APIs from Task 1.

**Interfaces**

- `RestoreScope = Literal["chat_only", "workspace_only", "chat_and_workspace"]`.
- Frozen `RecoveryConfirmation(request_id: str, operation_id: str, session_id: str, target: ArchiveTarget, allowed_scopes: tuple[RestoreScope, ...], selected_scope: RestoreScope | None, replacement_name: str | None, expires_at: float)`.
- Frozen `RecoveryOutcome(status: str, conversation_id: str, workspace_id: str | None, chat_changed: bool, workspace_changed: bool, chat_available: bool | None, next_action: str | None, message: str)`; `None` availability means observation failed, never false success.
- `allowed_restore_scopes(chat_archived: bool, workspace_archived: bool) -> tuple[RestoreScope, ...]` implements the approved four-state matrix.
- Async `read_recovery_target(app: Any, conversation_id: str) -> ArchiveTarget`; uses existing `local_conversation_service`/`storage_call` and rejects missing/deleted/unresolved ownership.
- Async `execute_restore(app: Any, target: ArchiveTarget, scope: RestoreScope, *, replacement_name: str | None, cancel_event: threading.Event) -> RecoveryOutcome`; internal coordinator-only effect function, never directly registered as an agent tool.
- `ArchiveRecoveryCoordinator(app: Any, results: ArchiveResultStore, *, monotonic: Callable[[], float] = time.monotonic)` exposes async `request(session_id, result_ref, *, call_key: RecoveryCallKey | None, scope: RestoreScope | None) -> RecoveryConfirmation | RecoveryOutcome`, `decide(session_id, request_id, *, allow: bool, scope: RestoreScope | None) -> RecoveryOutcome`, `revise_workspace_name(session_id, request_id, name) -> RecoveryConfirmation`, `wait_result(operation_id: str) -> RecoveryOutcome`, `close_session(session_id: str) -> None`, `shutdown() -> None`.
- UI binds the exact request ID/session to the action; agent closures can request/wait, but never call `decide`. Name revision invalidates the previous request and requires a newly displayed confirmation.

- [ ] **1. Pin the action matrix before effects.**

```python
import pytest
from tldw_chatbook.Chat.archive_recovery.operations import allowed_restore_scopes


@pytest.mark.parametrize("chat,workspace,expected", [
    (True, False, ("chat_only",)),
    (True, True, ("chat_only", "chat_and_workspace")),
    (False, True, ("workspace_only",)),
    (False, False, ()),
])
def test_only_meaningful_recovery_choices(chat, workspace, expected):
    assert allowed_restore_scopes(chat, workspace) == expected
```

Add real-store tests using Task-2 construction: `SimpleNamespace(local_chat_conversation_service=chats, workspace_registry_service=registry)` satisfies the shared storage lookup. Seed/archive a workspace and conversation, retain their target, mutate the conversation version, then assert the stale preflight leaves the workspace archived. Inject a failure only after the workspace write to distinguish a real partial result from failure before any write.
- [ ] **2. Run red.** `.venv/bin/python -m pytest Tests/Chat/test_archive_recovery_operations.py Tests/Chat/test_archive_recovery_coordinator.py -q`.
- [ ] **3. Implement effects with explicit phase ownership.** Matrix implementation:

```python
def allowed_restore_scopes(chat_archived, workspace_archived):
    if chat_archived and workspace_archived:
        return ("chat_only", "chat_and_workspace")
    if chat_archived:
        return ("chat_only",)
    if workspace_archived:
        return ("workspace_only",)
    return ()
```

Before showing a card compare current conversation version/ownership and workspace revision/name to the captured target. Repeat the checks at decision time. Restore a workspace using Task-1 CAS and the reviewed replacement name; restore the chat using `change_conversation_archive(app, [cid], archived=False, expected_versions={cid: version})`. After effects, re-read ownership/state and report actual changed resources separately from current availability. Do not call the restore-and-resume screen handler from this function.

The coordinator state progression is `awaiting_confirmation → admitted → workspace_write → chat_write → observed → settled`, skipping irrelevant write phases. Hold one active operation per source session until settlement; match duplicate button/model requests to it. Request IDs and confirmation snapshots are app-minted. A decision consumes authorization once under the coordinator lock, before an effect task is scheduled; concurrent duplicate decisions join the retained task. Do not keep locks across storage awaits.

Run effects in a strong-referenced `asyncio.Task`; waiters use `asyncio.shield`. Cancellation before admission changes no storage. Cancellation during a started SQLite write sets the operation's cancellation event, waits for that write to settle and prevents subsequent unstarted writes; it does not cancel the retained task. Close and shutdown refuse new operations, settle admitted work, then release database handles. When final observation fails, report known completed writes with unknown availability.

Key agent settlement by full `RecoveryCallKey`. Same session/run/call with different digest refuses. Retain consumed-call identity in the existing live run's settlement state even when the 20-entry presentation receipt cache evicts its text; replay after eviction is `expired`, not a new confirmation or write. After run teardown reject late callbacks rather than recreating its state. Manual action IDs are generated by the app and consumed once. Rename conflicts produce Restore as followed by a fresh confirmation; never retry a mutation automatically.
- [ ] **4. Exercise cancellation and concurrency using barriers.** Tests must hold the actual workspace write, cancel the waiting task, release the write, and assert durable workspace restoration plus unchanged chat and a partial receipt. Also test no-UI/timeout/deny, queued duplicate decisions, reused call IDs in different runs, changed arguments, evicted receipts, source-session close, shutdown ordering, stale archive revisions, concurrent rehoming/deletion during workspace-only restore, and post-write observation failure. No fixed sleeps; await visible state or explicit events. Run new operation/coordinator tests plus `Tests/Chat/test_conversation_archive_actions.py`, `Tests/UI/test_console_archive_cancellation.py`, `Tests/UI/test_console_archive_recovery_boundaries.py`.
- [ ] **5. Review and commit.** Inspect every path for ownership release and actual-outcome copy; scoped lint/format and task closeout; commit `feat: coordinate confirmed archive recovery without navigation`.

## Task 5: Integrate authenticated Console agent tools

**Files**

- Create: `Agents/archive_recovery_tool_provider.py`, recovery `runtime.py`, `Tests/Agents/test_archive_recovery_tools.py`, `Tests/Chat/test_console_archive_recovery_dispatch.py`.
- Modify: `Agents/tool_catalog.py`, `agent_service.py`, `agent_runtime.py`, `Chat/console_agent_bridge.py`, `console_runtime.py`, `console_chat_controller.py`; thread provenance through `console_turn_context.py` only if an immutable field is needed.
- Regressions: `Tests/Agents/test_library_name_reservation.py`, `test_library_tool_provider.py`, `test_agent_chat_create_tools.py`, `Tests/Chat/test_console_turn_library_authority.py`.

**Interfaces**

- Constants `SEARCH_ARCHIVED_CONVERSATIONS = "search_archived_conversations"`, `REQUEST_CONVERSATION_RESTORE = "request_conversation_restore"`, `RECOVERY_RESERVED_TOOL_NAMES` containing exactly those two names.
- `ArchiveRecoveryToolProvider` authenticates an issuer-owned in-memory capability; it advertises exact schemas and exposes no generic database or confirmation-decision API.
- `ToolCatalogRegistry.register_archive_recovery_provider(provider: ArchiveRecoveryToolProvider, authority: object) -> bool` verifies exact concrete type, exact issued authority, exact names, owning run/session and eligibility; it coexists with the unchanged Library registration method.
- `ArchiveRecoveryRuntime` owns one `ArchiveResultStore` and coordinator per application, partitioned by source session. Async `search(session_id, arguments: dict[str, object], *, model_visible: bool) -> dict[str, object]`, `request_restore(session_id, run_id, call_id, arguments) -> RecoveryOutcome`; synchronous closures bridge these to the app loop from the agent worker without nested `asyncio.run` on the UI thread.
- `build_archive_recovery_tool_closures` in recovery `runtime.py` binds admitted session/run/capability and returns `(search_callable, restore_callable)`, each taking validated `dict` and returning `ToolResult`. RuntimeDeps gains those two optional callables with default `None`.
- Search public schema is Task-2 request fields plus `cursor: str | None`; restore is `result_ref: str`, optional `scope: RestoreScope`. Reject unknown fields and cursor/filter mismatch.

- [ ] **1. Write catalog and production dispatch failures.** Use the existing real registry fixture pattern from `Tests/Agents/test_library_tool_provider.py`; register Direct/RAG normally, then recovery. Ensure both coexist. Test Blocked, foreign issuer, wrong exact class, forged source string, reserved-name impersonation and calls after owner teardown. Parameterize the admission matrix:

```python
@pytest.mark.parametrize("access,direct,interactive,primary,expected", [
    ("allowed", True, True, True, True),
    ("allowed", False, True, True, True),
    ("blocked", True, True, True, False),
    ("allowed", False, False, True, False),
    ("allowed", True, True, False, False),
])
def test_recovery_admission(access, direct, interactive, primary, expected):
    assert recovery_eligibility(
        assistant_access=access, direct_library_tools=direct,
        user_originated=interactive, primary=primary,
        temporary=False, local_session=True, answerable=True,
    ) is expected
```

Introduce `recovery_eligibility` as a pure helper in `runtime.py` with those keyword-only inputs; it is a tested decision helper, not an authority issuer. The capability can only be minted at the real runtime admission point. Test both layers.
- [ ] **2. Run red.** `.venv/bin/python -m pytest Tests/Agents/test_archive_recovery_tools.py Tests/Chat/test_console_archive_recovery_dispatch.py -q`.
- [ ] **3. Wire authority, schemas and runtime calls.** Derive Allowed from the executed turn's frozen `ConsoleTurnLibraryAuthority.policy`, independent of `direct_library_tools`. Preserve its destination and current tool/persona gates. Mint recovery authority only for an app-proven interactive manual/user-queued send. `ConsoleSubmissionOrigin.QUEUED` alone is insufficient: inspect its producer token to exclude scheduler/goal sources. AGENT_WAKE, restored unattended runs, missing provenance, temporary sessions and children receive no closure. When that provenance is absent on a legacy queue entry, refuse recovery rather than infer from text or primary status.

Register recovery names in the same collision checks that reserve existing runtime names, including when access is Blocked. Keep MCP/shared Library descriptors unchanged. Add optional closures to AgentService and RuntimeDeps and dispatch them under the existing execution checks. Confirmation waiting follows the runtime `fork_chat`/`new_chat` in-loop pattern so the generic tool timeout cannot orphan an approved effect; denial uses `ToolResult.blocked` and operational errors retain truthful status. Search callbacks publish the trusted structured card projection through an app-owned event, not by reparsing the model-visible JSON.

The eligibility helper's stable rule is:

```python
def recovery_eligibility(*, assistant_access, direct_library_tools,
                         user_originated, primary, temporary,
                         local_session, answerable):
    del direct_library_tools  # Retrieval choice does not grant authority.
    return bool(assistant_access == "allowed" and user_originated and primary
                and not temporary and local_session and answerable)
```

Use the actual enum value at its caller, add annotations/docstrings, and do not accept this predicate's booleans from a model. Agent restore must resolve a `model_visible=True` result before arming the coordinator; manual refresh references cannot be silently promoted. Unknown cursor handles and raw conversation IDs are refused.
- [ ] **4. Verify the real call chain.** A fake model returning the search call must pass through actual Console preparation, catalog composition, AgentService and RuntimeDeps, return the expected stored chat, and emit a trusted projection. A follow-up restore call must wait for the test-driven UI decision and return the real SQLite result. Repeat for Allowed/RAG-only with embeddings unavailable; Blocked must advertise neither tool and direct invocation must fail. Also test kill switch/persona denial, endpoint destination disclosure, queue execution after a policy change, scheduled primary origin and spoofed MCP collisions. Run new tests plus the listed regressions and `Tests/Chat/test_console_library_runtime_policy.py`.
- [ ] **5. Review and commit.** Check no broadening of Library exact-class validation, no generic local write grant and no new credentials/log bodies. Scoped lint/format and task closeout; commit `feat: expose authority-bound Console archive recovery tools`.

## Task 6: Render trusted results and exact recovery controls

**Files**

- Create: `Widgets/Chat_Widgets/archive_recovery_cards.py`, `UI/Console_Modules/archive_recovery.py`, `Tests/UI/test_console_archive_recovery_cards.py`, `Tests/UI/test_console_archive_recovery_flow.py`.
- Modify: `Widgets/Console/console_transcript.py`, `Widgets/Chat_Widgets/chat_task_cards.py`, `UI/Screens/chat_screen.py`, `chat_screen_state.py`, `Chat/console_runtime.py`, `css/screen_agentic_console.tcss`; add stylesheet-module registration only if splitting a new sheet is needed.
- Reuse: `UI/Console_Modules/archive.py` original-ID resume, current Library reader/navigation and workspace Restore as modal.

**Interfaces**

- `ArchiveRecoveryResultsCard` consumes the runtime's validated result-set projection; row actions carry `(source_session_id, result_set_id, result_ref)`.
- `ArchiveRecoveryConfirmCard` consumes `RecoveryConfirmation` and posts a typed `RecoveryDecided(session_id, request_id, allow, scope)` user event. It has no Remember/Allow always control.
- `ConsoleArchiveRecoveryView` is the thin screen adapter: `show_results(projection)`, `show_confirmation(confirmation | None)`, `show_receipt(outcome)`; it resolves actions through the live runtime, not JSON in transcript messages.
- Declare matching hooks in `CONSOLE_VIEW_HOOK_SLOTS` with explicit no-view refusal/projection behavior. Pending card ownership stays in the runtime across screen detach/remount.

- [ ] **1. Add mounted failing tests for both entry points.** Build on `Tests/UI/test_console_conversation_archive_flow.py` and `Tests/Chat/test_console_chat_create_integration.py` real-store construction. A fake provider emits the new tool call; the real runtime returns cards. Use Pilot to press the actual row Restore action and confirm; repeat with an agent restore request. Assert original conversation ID/message IDs, archive state and unchanged active draft, not just callback counts.

Widget-only scope test (the production-flow cases remain required):

```python
from textual.app import App, ComposeResult
from tldw_chatbook.Widgets.Chat_Widgets.archive_recovery_cards import ArchiveRecoveryConfirmCard


class RecoveryCardApp(App):
    def __init__(self, confirmation):
        super().__init__()
        self.confirmation = confirmation
        self.decisions = []

    def compose(self) -> ComposeResult:
        yield ArchiveRecoveryConfirmCard(self.confirmation)

    def on_archive_recovery_confirm_card_recovery_decided(self, event):
        self.decisions.append(event)
```

Instantiate with Task-4's frozen confirmation type. Define stable local widget IDs `recovery-confirm`, `recovery-cancel`, `recovery-scope`, and test `pilot.click("#recovery-confirm")` plus keyboard Tab/Enter. The emitted scope must match the visible choice and request identity; expired/superseded confirmations emit no accepted operation.
- [ ] **2. Run red.** `.venv/bin/python -m pytest Tests/UI/test_console_archive_recovery_cards.py Tests/UI/test_console_archive_recovery_flow.py -q`.
- [ ] **3. Compose token-based native controls.** Render title/workspace, separate lifecycle labels, event basis/date and snippet. Header shows all-local or explicit workspace scope and resolved date range. Render 10 rows by default with meaningful continuation/Refresh/Include unknown times controls; scrolling/keyboard selection preserves fixed result identities. Escape markup and control characters in titles/snippets; never render model-authored links as buttons.

Confirmation copy must be concrete:

```text
Restore chat only
The chat will be restored. Its workspace will remain archived.
Opening the chat will still require restoring that workspace.

Restore chat and workspace
The workspace and its other active chats will become visible.
Chats archived individually will stay archived.
```

Preview opens the current reader by exact ID without source staging/restoration; a deleted target yields a recoverable message. Restore as updates the coordinator request and presents the final replacement name for confirmation. Manual refresh remains UI-only; agent selection of that page requires a fresh authorized search. Post-restart/imported transcript cards are non-actionable unless app-owned provenance can validate their filters; otherwise offer manual Library search.

Explicit Open delegates to the existing exact resume handoff. Chat-only restore under an archived workspace offers its separate recovery before opening. Preserve active branch, current session draft and focus until the user chooses Open. Bind feedback to source session/request generation so a late result cannot paint another chat. Show parked pending decisions when returning to the owning chat; five-minute expiry still runs while backgrounded. Blocked Library access routes to the existing control without silently enabling it.
- [ ] **4. Verify real keyboard and responsive behavior.** Run the two new UI files at compact and wide sizes and `Tests/UI/test_console_runtime_ownership.py`, `Tests/UI/test_console_archive_recovery_boundaries.py`, `Tests/UI/test_console_conversation_archive_flow.py`. Rebuild with `.venv/bin/python tldw_chatbook/css/build_css.py`, then run `Tests/UI/test_design_token_governance.py` and `Tests/UI/test_css_bundle_sync_guard.py`. Assert readable compositor output, workspace effect copy, no remembered approval, expiration focus, pending-decision remount, duplicate click, no-UI refusal and preservation of drafts/branches.
- [ ] **5. Review and commit.** Confirm changes are on the Console surfaces only; scoped lint/format, generated-bundle check and task notes; commit `feat: add Console archive recovery cards and actions`.

## Task 7: Qualify the complete journey and document it

**Files**

- Extend: new search/coordinator/dispatch/UI test files from Tasks 2–6.
- Modify: `Docs/User_Guide/console/agent-runs-and-tools.md`, `context-and-rag.md`, `sessions-tabs-workspaces.md`.
- Create: `Docs/superpowers/qa/console/2026-09-27-agent-archive-recovery.md` with actual evidence only.

**Interfaces**

Consumes the six previous deliverables through real Console entry points. Produces a reproducible targeted qualification report and user-facing descriptions; no new production boundary or feature scope.

- [ ] **1. Close the integrated test matrix.** Add production-path tests for all four lifecycle combinations, unknown historical timestamps, RAG-only Allowed, Blocked, the Monday-chat/Friday-workspace date case, a stale result after rehoming, and cancellation after a workspace commit. Add an end-to-end assertion that recovery search cannot match an excluded system/tool-only needle. Use real SQLite and app controller; fake only the external model and explicit fault/clock inputs. Keep restoration/opening separate in assertions:

```python
assert store.active_session_id == original_session_id
assert next(s for s in store.sessions() if s.id == original_session_id).draft == original_draft
assert db.get_conversation_by_id(target_id)["archived"] == 0
assert [m["id"] for m in db.get_library_conversation_messages(target_id)["messages"]] == original_message_ids
```

Include the original active-leaf ID in the fixture. Trigger Open through its UI event, then assert `persisted_conversation_id == target_id` and the original branch is active. The fake provider's invocation counter must not increase merely from a manual Restore/Open; when an agent is awaiting its requested restoration, its ordinary post-tool continuation is allowed and must be distinguished from a new user send.
- [ ] **2. Run a single scoped integration pass after the final code changes.** Run exactly the new test files from Tasks 1–6 plus the existing archive-actions, conversation-archive, workspace-registry, library-name-reservation, runtime-ownership, runtime-shutdown, archive-boundary, design-token and CSS-bundle regression files. Record command, revision, interpreter and pass/fail/skip counts. Do not run a full suite. Re-run only tests affected by a subsequent correction.
- [ ] **3. Perform live app qualification in disposable local data.** Create two named workspaces and three distinct saved chats; archive one chat, one workspace with an active chat, and both for the third case. From another chat with an unsent draft, ask a configured tool-capable model to find recent archives. Exercise result-button restoration and conversational “restore the second one,” then explicit Open. Repeat with Direct off/RAG-only and no index. Verify the same persisted chat/branch, unchanged unrelated draft, explicit workspace effects and no unintended sends. Capture screenshots/interaction notes and sanitized storage assertions in the QA report. If no reachable authorized model or UI is available, record the missing live evidence and leave qualification open; fake-model tests are not relabeled as live-model proof.
- [ ] **4. Update user guidance and static evidence.** Explain Assistant Library access, the RAG-only exception, local/all-workspace scope, true event dates/unknown dates, result expiration, Preview, the action matrix, partial outcomes and separate Open. Include a concise example:

```text
You: Find conversations I archived today. I don't remember the workspace.
Assistant: Here are the matching local chats, with their workspaces and archive dates.
You: Restore the second one.
App: Confirm the named chat and the exact chat/workspace restoration scope.
```

Run scoped Python lint/formatter checks and document-link/whitespace checks; include any baseline diagnostic limitations. Review the implementation against every spec section and every modified archive writer. Only after all acceptance criteria/evidence are complete should each implementation task be marked Done through Backlog CLI.
- [ ] **5. Commit qualification.** Commit only relevant tests, user docs, QA report and task notes with `test: qualify agent-assisted archive recovery`. No push, merge or broad test sweep is implied by this plan.

## Coverage review

| Spec requirement | Delivery tasks |
| --- | --- |
| Purpose, local cross-workspace inventory, alternatives | 2, 5, 6, 7 |
| Library authority, destination and interactive-origin eligibility | 5, 7 |
| True chronology, either-event filtering, unknown dates | 1, 2, 7 |
| Role-restricted search, bounds, keyset continuation, query budget | 2, 7 |
| Immutable results, expiry, manual-only refresh, imported-card provenance | 3, 5, 6 |
| State-dependent confirmation, rename, transaction-level CAS | 1, 4, 6 |
| Cancellation, deduplication, partial writes, final observation, shutdown | 4, 5, 7 |
| Explicit Preview/Open, original identity/branch and current draft | 6, 7 |
| Local-only schema/sync invariants | 1, 7 |
| UI tokens, production hooks and bounded startup imports | 5, 6, 7 |

## Execution handoff

The user approved the written design. This document completes planning, not implementation. Before execution, choose inline execution or subagent-driven execution, prepare an isolated checkout through the worktree skill, and recheck task IDs/schema versions against the then-current base. All production tests, migrations and live checks above are prospective until an executor records their actual results.
