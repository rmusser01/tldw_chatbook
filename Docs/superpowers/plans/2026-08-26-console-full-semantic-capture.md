# Console Full Semantic Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user deliberately retain and inspect complete, bounded semantic provider exchanges for the next eligible send, one Console conversation, or the global Console default while Safe remains the default and Full records can be logically purged.

**Architecture:** Extend the existing `ExchangeCapture -> ConsoleProviderStreamSignals -> ConsoleChatStore -> message_exchanges` path; do not add a second trace pipeline. A pure Safe/Full resolver freezes detail at accepted-turn admission, the existing provider gateway captures the adapter-boundary request and observable response under that frozen detail, ChaChaNotes stores queryable matching provenance, and shared callbacks project the same policy into Conversation Inspector, live Trace, and F9 Settings.

**Tech Stack:** Python 3.11+, Textual 8.x, SQLite/FTS5 sidecars, zlib/JSON standard library, pytest/Hypothesis, existing TOML configuration primitives.

**Spec:** `Docs/superpowers/specs/2026-08-26-console-full-semantic-capture-design.md`

**Backlog decomposition:** `TASK-22507` tracks the architecture. Execute the atomic children in order: `TASK-22507.1` (capture/persistence), `TASK-22507.2` (runtime/provider threading), `TASK-22507.3` (quiescent purge), then `TASK-22507.4` (shared UI/export/docs).

**ADR required:** yes

**ADR path:** `backlog/decisions/089-console-full-semantic-capture-policy.md`

**Reason:** ADR-089 already governs the persisted privacy metadata, provider/runtime boundary, logical deletion semantics, and shared UI/storage contract. Do not create a second ADR unless implementation changes one of those approved boundaries.

## Global Constraints

- Safe is the application default; `[console] exchange_capture` remains the authoritative kill switch.
- Precedence is next eligible send, conversation override, global default, then application Safe.
- Freeze capture detail at accepted-turn admission; retries, tool loops, and surviving fleet calls retain the frozen value.
- Full means semantic provider-adapter input and observable output, not generic byte-literal HTTP; llama.cpp remains the literal-payload exception.
- Structured credentials and unknown kwargs never persist. URL userinfo, query, and fragment never persist.
- Request and response binary/data-URI/base64 values become deterministic size/hash stubs, including nested tool arguments and results.
- One call has a 64 MiB uncompressed UTF-8 JSON accumulation/decompression ceiling and the existing 16 MiB compressed ceiling.
- Capture is best-effort and must never fail or alter a model run; capture bodies and raw body-bearing exception values never enter logs.
- `capture_blob` is compressed, not encrypted.
- Conversation policy and exchange captures are local-only: no sync, FTS, server payload, conversation metadata, or Trace-event projection.
- Scoped purge is logical record deletion, not forensic secure erasure of WAL/free pages, snapshots, exports, or backups.
- Imported/shared Traces are read-only and never expose live capture controls.
- Capture detail and export profile are distinct. Full clipboard and filesystem disclosure is confirmed every time.
- No new tracing framework, database, settings subsystem, permission system, dependency, or raw-wire recorder.
- User-visible Full activation lands only after provenance, persistence, confirmation, and purge dependencies exist.
- Re-read `CharactersRAGDB._CURRENT_SCHEMA_VERSION` before migration work. This plan names v49 -> v50 because v49 is current at planning time; if another migration lands first, renumber the new migration, runner, and fixtures together before editing production logic.
- Run the complete DB migration suite after the schema change. Do not run the repository-wide full suite without explicit owner approval.

## File and Ownership Map

- `tldw_chatbook/Chat/console_exchange_capture.py`: Safe/Full types, pure resolution, bounded capture construction, blob compatibility, and provenance validation.
- `tldw_chatbook/Chat/console_project_instructions.py`: expose the incumbent credential-free endpoint canonicalizer for capture reuse.
- `tldw_chatbook/Chat/console_capture_policy_repository.py`: new, narrow local conversation-override repository; no runtime or UI ownership.
- `tldw_chatbook/DB/migrations/chachanotes_v49_to_v50_console_full_capture.sql`: queryable exchange detail and sparse conversation policy table.
- `tldw_chatbook/DB/ChaChaNotes_DB.py`: migration runner plus local exchange/policy/count/purge database methods.
- `tldw_chatbook/Chat/chat_persistence_service.py`: version-neutral local sidecar facade.
- `tldw_chatbook/Chat/console_chat_store.py`: per-session override/one-shot/revisions, exchange cache ownership, staged purge, and promotion/hydration.
- `tldw_chatbook/Chat/console_chat_controller.py`: admission-time resolution/consumption, immutable run signals, mutation lifecycle, and quiescence lease.
- `tldw_chatbook/Chat/console_provider_gateway.py`: generic, Anthropic, and llama.cpp adapter-boundary capture using the frozen detail and budget.
- `tldw_chatbook/Chat/console_exchange_export.py`: new per-call export governor reusing `TraceExportProfile`.
- `tldw_chatbook/Widgets/Console/console_capture_policy_dialog.py`: new shared scoped-policy/count/purge modal.
- `tldw_chatbook/Widgets/Console/console_exchange_export_dialog.py`: new one-call profile/destination export flow.
- `tldw_chatbook/Widgets/Console/console_conversation_inspector.py`: immutable target status, provenance, revision fences, and dialog launch.
- `tldw_chatbook/UI/Screens/trajectory_screen.py`: compact future-capture status/control only for live traces.
- `tldw_chatbook/UI/Screens/chat_screen.py`: immutable callback binding for Inspector and live Trace.
- `tldw_chatbook/UI/Screens/settings_screen.py`: canonical global kill-switch/detail owners using the same mutation result contract.
- `tldw_chatbook/Widgets/Console/trace_export_dialog.py`: publish the existing profile labels and Full-warning primitive for exchange-export reuse.
- `tldw_chatbook/css/components/_agentic_terminal.tcss`: responsive modal/status styling.
- `tldw_chatbook/css/tldw_cli_modular.tcss`: generated bundle; never hand-edit.
- `Docs/User_Guide/console/context-and-rag.md` and `Docs/User_Guide/console/chat-basics.md`: retention, provider boundary, export, and purge documentation.

---

### Task 1: Safe-First Capture Provenance and Persistence (`TASK-22507.1`)

**Files:**
- Modify: `tldw_chatbook/Chat/console_exchange_capture.py:1-258`
- Modify: `tldw_chatbook/Chat/console_project_instructions.py:144-245`
- Create: `tldw_chatbook/Chat/console_capture_policy_repository.py`
- Modify: `tldw_chatbook/config.py:3192-3195`
- Create: `tldw_chatbook/DB/migrations/chachanotes_v49_to_v50_console_full_capture.sql`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py:513,6493-6585,6738-6784,12210-12324`
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py:1064-1092`
- Modify: `tldw_chatbook/Chat/console_chat_store.py:9462-9525`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:1803-1885`
- Modify: `Tests/Chat/test_console_exchange_capture.py`
- Create: `Tests/Chat/test_console_capture_policy_repository.py`
- Create: `Tests/DB/test_chachanotes_full_capture_migration.py`
- Modify: `Tests/DB/test_chachanotes_message_exchanges.py`
- Modify: `Tests/Chat/test_console_chat_store_exchanges.py`
- Modify: `Tests/UI/test_chat_screen_console_inspector_loader.py`

**Interfaces:**
- Consumes: incumbent `ExchangeCapture`, `capture_to_blob`, `capture_from_blob`, `CAPTURE_REQUEST_ALLOWLIST`, `CharactersRAGDB.transaction()`, and `apply_settings_mutation_to_cli_config` semantics.
- Produces:
  - `CaptureDetail(str, Enum)` with `SAFE = "safe"` and `FULL = "full"`.
  - `CapturePolicySource(str, Enum)` with `DISABLED`, `NEXT_SEND`, `CONVERSATION`, `GLOBAL`, and `APPLICATION`.
  - `CapturePolicyResolution(enabled: bool, detail: CaptureDetail, source: CapturePolicySource, invalid_sources: tuple[str, ...])`.
  - `resolve_capture_policy(*, enabled: bool, next_send: object = None, conversation: object = None, global_default: object = None, allow_next_send: bool = True) -> CapturePolicyResolution`.
  - `CaptureBudget(limit_bytes: int = 64 * 1024 * 1024)` shared by request and response accumulation.
  - `build_request_capture(kwargs, *, capture_detail=CaptureDetail.SAFE, budget=None) -> tuple[dict, tuple[str, ...]]`.
  - `build_response_capture(*, content: str, tool_calls: Sequence[Mapping[str, Any]], synthetic_fallback: bool = False, budget: CaptureBudget | None = None) -> dict[str, Any]`.
  - `capture_from_storage(blob: bytes, declared_detail: object) -> ExchangeCapture`, which defaults a legacy blob to Safe and rejects a column/blob mismatch.
  - `ConversationCapturePolicy(conversation_id: str, detail: CaptureDetail, updated_at: str)`.
  - `CapturePolicyWriteStatus(str, Enum)` with `STORED`, `DELETED`, `UNCHANGED`, `MISSING_CONVERSATION`, and `UNAVAILABLE`; `CapturePolicyWriteResult(status: CapturePolicyWriteStatus, policy: ConversationCapturePolicy | None)`.
  - `ConsoleCapturePolicyRepository.read(conversation_id: str) -> ConversationCapturePolicy | None` and `.replace(conversation_id: str, detail: CaptureDetail | None) -> CapturePolicyWriteResult`; `None` deletes the row to mean Inherit.
  - ChaChaNotes exchange rows carrying `capture_detail`, always derived from `ExchangeCapture.capture_detail` at the same write site as `capture_blob`.

- [ ] **Step 1: Pin Safe/Full parsing and precedence with failing pure tests**

Add tests that make invalid values observable without ever enabling Full:

```python
def test_capture_policy_precedence_and_invalid_values_fail_safe():
    resolved = resolve_capture_policy(
        enabled=True,
        next_send="full",
        conversation="safe",
        global_default="full",
    )
    assert resolved == CapturePolicyResolution(
        enabled=True,
        detail=CaptureDetail.FULL,
        source=CapturePolicySource.NEXT_SEND,
        invalid_sources=(),
    )
    invalid = resolve_capture_policy(enabled=True, global_default="future-value")
    assert invalid.detail is CaptureDetail.SAFE
    assert invalid.source is CapturePolicySource.APPLICATION
    assert invalid.invalid_sources == ("global",)


def test_capture_off_wins_without_forgetting_dormant_detail():
    resolved = resolve_capture_policy(enabled=False, conversation="full")
    assert resolved.enabled is False
    assert resolved.detail is CaptureDetail.FULL
    assert resolved.source is CapturePolicySource.CONVERSATION
```

Run: `pytest Tests/Chat/test_console_exchange_capture.py -q`

Expected: FAIL because the capture policy types and resolver do not exist.

- [ ] **Step 2: Implement the pure policy types and legacy-Safe provenance**

Add the enum/resolution contract beside `ExchangeCapture`, and give the dataclass a backward-compatible trailing default:

```python
class CaptureDetail(str, Enum):
    SAFE = "safe"
    FULL = "full"


@dataclass(frozen=True)
class ExchangeCapture:
    # existing fields remain in their existing order
    capture_detail: CaptureDetail = CaptureDetail.SAFE
```

`capture_to_blob` must replace the enum in the `asdict()` payload with `capture.capture_detail.value` before JSON encoding. `capture_from_blob` parses only `"safe"`/`"full"`, defaults an absent legacy field to Safe, and rejects unknown stored values. `resolve_capture_policy` must preserve the dormant winning detail when disabled, ignore the one-shot when `allow_next_send=False`, collect only content-free source names for invalid values, and otherwise use next -> conversation -> global -> Safe.

Run: `pytest Tests/Chat/test_console_exchange_capture.py -q`

Expected: the new resolver tests PASS and incumbent blob tests still PASS.

- [ ] **Step 3: Pin endpoint sanitization, Full project instructions, response stubbing, and the shared 64 MiB budget**

Add tests with a real tagged instruction row, nested response tool result, credential-bearing URLs, and a deliberately tiny test budget:

```python
def test_safe_omits_but_full_retains_tagged_project_instruction_body():
    kwargs = {"messages_payload": [_project_instruction_row("AGENTS BODY")]}
    safe, safe_omitted = build_request_capture(kwargs, capture_detail=CaptureDetail.SAFE)
    full, full_omitted = build_request_capture(kwargs, capture_detail=CaptureDetail.FULL)
    assert "AGENTS BODY" not in json.dumps(safe)
    assert "messages_payload[0].content" in safe_omitted
    assert full["messages_payload"][0]["content"] == "AGENTS BODY"
    assert "messages_payload[0].content" not in full_omitted


def test_endpoint_identity_drops_credentials_query_and_fragment():
    request, _ = build_request_capture(
        {"api_base_url": "https://user:pass@example.test/v1?q=secret#fragment"},
        capture_detail=CaptureDetail.FULL,
    )
    assert request["api_base_url"] == "https://example.test/v1"


def test_request_and_response_share_one_bounded_budget():
    budget = CaptureBudget(limit_bytes=256)
    request, _ = build_request_capture(
        {"messages_payload": [{"role": "user", "content": "x" * 220}]},
        capture_detail=CaptureDetail.FULL,
        budget=budget,
    )
    response = build_response_capture(
        content="y" * 220,
        tool_calls=[{"function": {"arguments": "QUJD" * 2000}}],
        budget=budget,
    )
    assert request["truncation_inventory"] or response["truncation_inventory"]
    assert budget.used_bytes <= budget.limit_bytes
```

Run: `pytest Tests/Chat/test_console_exchange_capture.py -q`

Expected: FAIL until the common sanitizer/budget is used on both sides.

- [ ] **Step 4: Implement bounded semantic construction and content-free failures**

Expose a public `canonical_provider_endpoint_identity()` wrapper over the existing `_canonical_endpoint_identity()`; do not add another URL parser. In capture construction:

```python
def build_request_capture(
    kwargs: Mapping[str, Any],
    *,
    capture_detail: CaptureDetail = CaptureDetail.SAFE,
    budget: CaptureBudget | None = None,
) -> tuple[dict, tuple[str, ...]]:
    active_budget = budget or CaptureBudget()
    # allowlist first; sanitize endpoint-shaped values; redact tagged rows only
    # in Safe; stub binaries; retain through active_budget.
```

Use `json.JSONEncoder(...).iterencode()` for bounded UTF-8 serialization and `zlib.decompressobj().decompress(..., max_length=CAPTURE_JSON_MAX_BYTES + 1)` for bounded decode. `capture_from_blob` supplies `CaptureDetail.SAFE` when the field is absent. Any oversize legacy decode raises one content-free `CaptureUnavailableError("capture exceeds safe decode limit")`.

Run: `pytest Tests/Chat/test_console_exchange_capture.py -q`

Expected: PASS, including the existing compressed-cap and deterministic-stub tests.

- [ ] **Step 5: Write failing v49 -> v50 migration and repository tests**

First assert `CharactersRAGDB._CURRENT_SCHEMA_VERSION == 49`. Build a genuine v49 fixture, insert a pre-feature exchange blob without `capture_detail`, migrate, and assert:

```python
assert migrated._CURRENT_SCHEMA_VERSION == 50
row = connection.execute(
    "SELECT capture_detail FROM message_exchanges WHERE run_tag = 'legacy'"
).fetchone()
assert row[0] == "safe"
sql = connection.execute(
    "SELECT sql FROM sqlite_master WHERE name = 'console_conversation_capture_policy'"
).fetchone()[0]
assert "CHECK" in sql and "full" in sql and "safe" in sql
```

Repository tests must cover missing row -> Inherit, Safe/Full replace, Inherit deletion, deleted/missing conversation refusal, cascade deletion, corrupt checked value refusal, and no `sync_log` delta.

Run: `pytest Tests/DB/test_chachanotes_full_capture_migration.py Tests/Chat/test_console_capture_policy_repository.py -q`

Expected: FAIL because the migration, runner, and repository do not exist.

- [ ] **Step 6: Implement the migration, DB runner, and local repository**

The migration SQL is exactly local sidecar DDL:

```sql
ALTER TABLE message_exchanges
ADD COLUMN capture_detail TEXT NOT NULL DEFAULT 'safe'
CHECK (capture_detail IN ('safe', 'full'));

CREATE INDEX idx_message_exchanges_capture_detail
ON message_exchanges(capture_detail, message_id);

CREATE TABLE console_conversation_capture_policy(
  conversation_id TEXT PRIMARY KEY NOT NULL
    REFERENCES conversations(id) ON DELETE CASCADE,
  capture_detail TEXT NOT NULL CHECK (capture_detail IN ('safe', 'full')),
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

Advance `_CURRENT_SCHEMA_VERSION`, add `_migrate_from_v49_to_v50`, and register key `49` in `migration_steps`. The repository uses parameterized SQL under `db.transaction(immediate=True)` and returns only `CapturePolicyWriteStatus` values without logging policy values or SQL exception bodies.

Run: `pytest Tests/DB/test_chachanotes_full_capture_migration.py Tests/Chat/test_console_capture_policy_repository.py -q`

Expected: PASS.

- [ ] **Step 7: Pin matching column/blob provenance through the real store seam**

Extend existing DB/store tests so a Full capture produces both a Full blob and column, a hand-crafted mismatch is skipped as corrupt by the Inspector loader, and a legacy blob plus Safe column remains readable:

```python
stored = db.get_message_exchanges(message_id)[0]
assert stored["capture_detail"] == "full"
assert capture_from_storage(
    stored["capture_blob"], stored["capture_detail"]
).capture_detail is CaptureDetail.FULL

with pytest.raises(CaptureCorruptError):
    capture_from_storage(stored["capture_blob"], "safe")
```

Run: `pytest Tests/DB/test_chachanotes_message_exchanges.py Tests/Chat/test_console_chat_store_exchanges.py Tests/UI/test_chat_screen_console_inspector_loader.py -q`

Expected: FAIL until every write/read seam carries the column.

- [ ] **Step 8: Implement one-source provenance writes and fail-closed reads**

At `_persist_exchanges_only`, derive the row from the immutable capture:

```python
rows.append({
    "run_tag": capture.run_tag,
    "seq": capture.seq,
    "status": capture.status,
    "abandoned": capture.run_tag in abandoned_tags,
    "capture_detail": capture.capture_detail.value,
    "capture_blob": capture_to_blob(capture),
    "created_at": capture.created_at,
})
```

Add `capture_detail` to DB INSERT/UPSERT/SELECT. The loader must call `capture_from_storage`, log only the exception category plus permitted message/run identifiers, and skip the corrupt call without exposing blob data.

Run: `pytest Tests/Chat/test_console_exchange_capture.py Tests/DB/test_chachanotes_message_exchanges.py Tests/Chat/test_console_chat_store_exchanges.py Tests/UI/test_chat_screen_console_inspector_loader.py -q`

Expected: PASS.

- [ ] **Step 9: Run the complete DB migration gate**

Run: `pytest Tests/ChaChaNotesDB Tests/DB -q`

Expected: PASS. This is the required DB/migration sweep, not the repository-wide full suite.

- [ ] **Step 10: Commit Gate 1**

```bash
git add tldw_chatbook/Chat/console_exchange_capture.py tldw_chatbook/Chat/console_project_instructions.py tldw_chatbook/Chat/console_capture_policy_repository.py tldw_chatbook/config.py tldw_chatbook/DB/ChaChaNotes_DB.py tldw_chatbook/DB/migrations/chachanotes_v49_to_v50_console_full_capture.sql tldw_chatbook/Chat/chat_persistence_service.py tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/UI/Screens/chat_screen.py Tests/Chat/test_console_exchange_capture.py Tests/Chat/test_console_capture_policy_repository.py Tests/DB/test_chachanotes_full_capture_migration.py Tests/DB/test_chachanotes_message_exchanges.py Tests/Chat/test_console_chat_store_exchanges.py Tests/UI/test_chat_screen_console_inspector_loader.py backlog/tasks/task-22507.1\ -\ Add-Safe-first-semantic-capture-provenance-and-persistence.md
git commit -m "feat(console): persist bounded capture provenance"
```

---

### Task 2: Admission-Time Policy and Provider Threading (`TASK-22507.2`)

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_store.py:697-775,4232-4265,8473-8684`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py:1725-1760,4436-5457,5608-6075,13671-14066`
- Modify: `tldw_chatbook/Chat/console_provider_gateway.py:133-537,2472-2825,2961-3105`
- Modify: `tldw_chatbook/config.py:5500-5608,6079-6241`
- Modify: `Tests/Chat/test_console_chat_controller_exchanges.py`
- Modify: `Tests/Chat/test_console_chat_controller.py`
- Modify: `Tests/Chat/test_console_chat_store_exchanges.py`
- Modify: `Tests/Chat/test_console_provider_gateway.py`
- Modify: `Tests/test_config_save_settings_semantics.py`

**Interfaces:**
- Consumes: Task 1 `CaptureDetail`, `CapturePolicyResolution`, resolver, budget, repository, and matching persisted provenance.
- Produces:
  - `ConsoleChatSession.capture_detail_override: CaptureDetail | None`, `next_capture_detail: CaptureDetail | None`, `next_capture_detail_revision: int`, `capture_revision: int`, and `capture_policy_save_pending: bool`; the process-wide policy revision remains store/controller-owned rather than duplicated per session.
  - `CapturePolicyState(next_detail: CaptureDetail | None, conversation_detail: CaptureDetail | None, next_revision: int, policy_revision: int, capture_revision: int, save_pending: bool)`.
  - `ConsoleChatStore.capture_policy_state(session_id: str) -> CapturePolicyState`, `set_session_next_capture_detail(session_id: str, detail: CaptureDetail | None, *, expected_policy_revision: int) -> tuple[CaptureDetail | None, int, int]`, `consume_session_next_capture_detail(session_id: str, *, expected_next_revision: int) -> bool`, and `replace_session_capture_override(session_id: str, detail: CaptureDetail | None, *, expected_policy_revision: int, save_pending: bool = False) -> int`; stale expected revisions raise content-free `CapturePolicyStaleError`.
  - `ConsoleProviderStreamSignals.capture_detail: CaptureDetail`, inherited read-only by `ConsoleProviderCallSignals`.
  - `CapturePolicySnapshot(session_id: str, conversation_id: str | None, conversation_title: str, enabled: bool, next_detail: CaptureDetail | None, conversation_detail: CaptureDetail | None, global_detail: CaptureDetail, effective: CapturePolicyResolution, policy_revision: int, config_generation: int, capture_revision: int, active_run_detail: CaptureDetail | None, queued_consumer: bool, save_pending: bool, error_code: str | None)`.
  - `CapturePolicyMutationStatus(str, Enum)` with `APPLIED`, `SAFE_SESSION_ONLY`, `STALE`, `TARGET_MISSING`, and `FAILED`; `CapturePolicyMutationResult(status: CapturePolicyMutationStatus, snapshot: CapturePolicySnapshot, retryable: bool, reason_code: str | None, config_result: ConfigMutationResult | None = None)`.
  - `ConsoleChatController.capture_policy_snapshot(session_id: str) -> CapturePolicySnapshot`, `set_next_capture_detail(session_id: str, detail: CaptureDetail | None, *, expected_policy_revision: int) -> CapturePolicyMutationResult`, `async replace_conversation_capture_detail(session_id: str, detail: CaptureDetail | None, *, expected_policy_revision: int) -> CapturePolicyMutationResult`, and `apply_global_capture_settings(*, enabled: bool, detail: CaptureDetail, expected_config_generation: int, expected_policy_revision: int) -> CapturePolicyMutationResult`.
  - `_admit_capture_policy(session_id, origin) -> ConsoleProviderStreamSignals`, called once after acceptance ownership is established.
  - A capture-specific global config mutation that returns `ConfigMutationResult` plus honest runtime status and uses the existing config generation fence.

- [ ] **Step 1: Write the admission/consumption matrix as failing tests**

Parameterize manual, authorized queue, agent wake, readiness rejection, queue cancellation, local-command refusal, cancellation before acceptance, and cancellation after acceptance:

```python
@pytest.mark.parametrize(
    ("origin", "accepted", "consume"),
    [
        (ConsoleSubmissionOrigin.MANUAL, True, True),
        (ConsoleSubmissionOrigin.QUEUED, True, True),
        (ConsoleSubmissionOrigin.AGENT_WAKE, True, False),
        (ConsoleSubmissionOrigin.MANUAL, False, False),
    ],
)
async def test_next_capture_consumption_matrix(origin, accepted, consume):
    snapshot = controller.capture_policy_snapshot(session.id)
    controller.set_next_capture_detail(
        session.id,
        CaptureDetail.FULL,
        expected_policy_revision=snapshot.policy_revision,
    )
    result = await drive_submission(origin=origin, accepted=accepted)
    assert result.accepted is accepted
    assert (store.session_next_capture_detail(session.id) is None) is consume
```

Add a race test that re-arms a new one-shot while the accepted run is active; revision-gated consumption must leave the new slot intact.

Run: `pytest Tests/Chat/test_console_chat_controller_exchanges.py -q`

Expected: FAIL because no capture one-shot owner exists.

- [ ] **Step 2: Implement session policy state and exact-revision consumption**

Mirror the existing one-shot prefill pattern without sharing its slot:

```python
def set_session_next_capture_detail(
    self,
    session_id: str,
    detail: CaptureDetail | None,
    *,
    expected_policy_revision: int,
) -> tuple[CaptureDetail | None, int, int]:
    session = self._session_or_raise(session_id)
    if self._capture_policy_revision != expected_policy_revision:
        raise CapturePolicyStaleError
    session.next_capture_detail = detail
    session.next_capture_detail_revision += 1
    self._capture_policy_revision += 1
    return (
        session.next_capture_detail,
        session.next_capture_detail_revision,
        self._capture_policy_revision,
    )


def consume_session_next_capture_detail(
    self, session_id: str, *, expected_next_revision: int
) -> bool:
    session = self._session_or_raise(session_id)
    if session.next_capture_detail_revision != expected_next_revision:
        return False
    session.next_capture_detail = None
    session.next_capture_detail_revision += 1
    self._capture_policy_revision += 1
    return True
```

Hydrate the conversation override from the Task 1 repository and flush a staged ephemeral override on promotion. A failed staged Safe write over inherited Full stays Safe in memory with `save_pending=True`.

Run: `pytest Tests/Chat/test_console_chat_store_exchanges.py Tests/Chat/test_console_chat_controller_exchanges.py -q`

Expected: session-state tests PASS; admission tests still FAIL until signals freeze at acceptance.

- [ ] **Step 3: Freeze detail at the accepted owner boundary**

Create signals only after the turn has an admitted owner. Thread the same object through the ordinary path and `_DurablePostcommitContinuation`:

```python
signals = self._admit_capture_policy(session.id, origin)
# transition COMMITTING -> ACCEPTED and store `signals` on the continuation
stream_result = await self._stream_assistant_response(
    ...,
    stream_signals=signals,
)
```

`_admit_capture_policy` resolves under the session's admission serialization, captures the exact next-slot revision, consumes only manual/authorized queued turns after owner acceptance, and calls the resolver with `allow_next_send=False` for `AGENT_WAKE`. Post-acceptance cancellation does not restore the consumed slot.

Remove the late policy read from `_stream_assistant_response_inner`; its defensive direct-call fallback remains Safe unless a test explicitly supplies a resolution.

Run: `pytest Tests/Chat/test_console_chat_controller_exchanges.py Tests/Chat/test_console_chat_controller.py -q`

Expected: PASS for acceptance, cancellation ordering, queue, local-command, wake, durable continuation, and re-arm races.

- [ ] **Step 4: Pin frozen detail across call views, retries, tool loops, and fleet survivors**

Add tests that mutate global/conversation state after the first call and assert every capture on the original signals remains Full while a later run resolves Safe:

```python
signals = ConsoleProviderStreamSignals(
    exchange_capture_enabled=True,
    capture_detail=CaptureDetail.FULL,
)
first = signals.new_usage_call()
second = signals.new_usage_call()
assert first.capture_detail is second.capture_detail is CaptureDetail.FULL
```

Drive the existing agent multi-call and surviving-fleet harnesses with the same signal identity. Assert every `ExchangeCapture.capture_detail` equals the admission value.

Run: `pytest Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_console_chat_controller_exchanges.py Tests/Chat/test_console_chat_controller.py -q`

Expected: FAIL until call-scoped signals and `_flight_capture` carry provenance.

- [ ] **Step 5: Thread frozen detail through the generic, Anthropic, and llama.cpp capture seams**

Add a read-only property to `ConsoleProviderCallSignals`; pass detail and the common `CaptureBudget` into `begin_exchange`; build the immutable result with the same detail:

```python
@property
def capture_detail(self) -> CaptureDetail:
    return self._aggregate.capture_detail


return ExchangeCapture(
    ...,
    response=build_response_capture(
        content="".join(flight["content"]),
        tool_calls=flight["tool_calls"],
        budget=flight["capture_budget"],
    ),
    capture_detail=flight["capture_detail"],
)
```

The generic branch calls `build_request_capture(kwargs, capture_detail=call_signals.capture_detail, budget=budget)` immediately before `chat_api_call`. The Anthropic fixture must prove `system_message`, ordinary messages, tagged AGENTS/workspace/RAG injection, tool schemas, tool calls, and tool results survive in Full while Safe retains its approved redactions. The llama.cpp branch keeps `wire_payload`, but sanitizes/stubs it through the same budget and detail provenance. Replace `logger.opt(exception=True).warning("exchange_capture_begin_failed")` with a content-free category/type log; the frame contains request bodies.

Run: `pytest Tests/Chat/test_console_provider_gateway.py -q`

Expected: PASS for existing gateway behavior plus Safe/Full Anthropic, generic, llama.cpp, response-binary, retry, and never-break-send cases.

- [ ] **Step 6: Implement and test the canonical global mutation lifecycle**

Use the existing locked config snapshot/generation and structured result. Tests must inject failure before replacement and during general cache reload:

```python
safe = apply_console_capture_settings(
    enabled=True,
    detail=CaptureDetail.SAFE,
    expected_generation=snapshot.generation,
)
assert runtime_capture_policy().detail is CaptureDetail.SAFE

full = apply_console_capture_settings(
    enabled=True,
    detail=CaptureDetail.FULL,
    expected_generation=snapshot.generation,
)
assert full.file_replaced is True
```

Required ordering:

1. A Safe/Off result publishes the shared runtime projection before the disk attempt; a failed write reports `Safe for this app session — save failed`.
2. A Full/On result publishes only after atomic file replacement.
3. Replacement plus cache-refresh failure remains saved and active and reports `Saved and active — settings cache refresh degraded`.
4. Stale config generation rejects without changing an admitted run.
5. Turning capture Off disarms all live one-shots under admission serialization but leaves persistent Full detail dormant.
6. First-party Off -> On requires the Task 4 confirmation callback whenever any global/conversation Full detail could resume.

Run: `pytest Tests/test_config_save_settings_semantics.py Tests/Chat/test_console_chat_controller_exchanges.py -q`

Expected: PASS.

- [ ] **Step 7: Run the Gate 2 focused matrix**

Run: `pytest Tests/Chat/test_console_exchange_capture.py Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_console_chat_controller_exchanges.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_chat_store_exchanges.py Tests/test_config_save_settings_semantics.py -q`

Expected: PASS with no provider transcript-output changes and no capture body in captured logs.

- [ ] **Step 8: Commit Gate 2**

```bash
git add tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_provider_gateway.py tldw_chatbook/config.py Tests/Chat/test_console_chat_controller_exchanges.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_chat_store_exchanges.py Tests/Chat/test_console_provider_gateway.py Tests/test_config_save_settings_semantics.py backlog/tasks/task-22507.2\ -\ Freeze-scoped-capture-policy-across-Console-provider-runs.md
git commit -m "feat(console): freeze capture policy at admission"
```

---

### Task 3: Conversation-Scoped Logical Purge and Revision Fences (`TASK-22507.3`)

**Files:**
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py:12210-12324`
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py:1064-1092`
- Modify: `tldw_chatbook/Chat/console_chat_store.py:871-1085,7852-7904,9462-9525,10668-10877`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py:2050-2338,4279-4505,9580-9705,14068-14332`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:1803-1885,9644-9699`
- Modify: `tldw_chatbook/Widgets/Console/console_conversation_inspector.py:332-463,1030-1445`
- Modify: `Tests/DB/test_chachanotes_message_exchanges.py`
- Modify: `Tests/Chat/test_console_chat_store_exchanges.py`
- Create: `Tests/Chat/test_console_capture_purge.py`
- Modify: `Tests/Chat/test_console_chat_controller_exchanges.py`
- Modify: `Tests/UI/test_console_conversation_inspector.py`
- Modify: `Tests/UI/test_chat_screen_console_inspector_loader.py`

**Interfaces:**
- Consumes: Task 1 queryable `capture_detail`; Task 2 frozen signals, run/fleet ownership, and process-local policy revision.
- Produces:
  - `CharactersRAGDB.list_full_exchange_keys_for_conversation(conversation_id) -> set[tuple[str, str, int]]` and `delete_full_exchanges_for_conversation(conversation_id) -> int`.
  - `StagedCapturePurge(session_id: str, conversation_id: str | None, expected_revision: int, durable_keys: frozenset[tuple[str, str, int]], message_swaps: tuple[tuple[ConsoleMessage, tuple[ExchangeCapture, ...]], ...], blob_cache: Mapping[tuple[str, str, int], bytes], abandoned_tags: Mapping[str, frozenset[str]], capture_revisions: Mapping[str, int])`; every collection and message reference needed after commit is precomputed.
  - `ConsoleChatStore.capture_revision(session_id: str) -> int`, `stage_full_capture_purge(session_id: str) -> StagedCapturePurge`, and `commit_full_capture_purge(stage: StagedCapturePurge) -> int`; the commit method owns the durable delete followed by the authoritative swaps.
  - `CapturePurgeAvailability(can_purge: bool, reason_code: str | None)`.
  - `CapturePurgeStatus(str, Enum)` with `DELETED`, `BLOCKED`, `STALE`, and `FAILED`; `CapturePurgeResult(status: CapturePurgeStatus, removed_count: int, capture_revision: int, reason_code: str | None)` with content-free `blocked(...)` and `deleted(...)` constructors.
  - `ConsoleChatController.capture_purge_availability(session_id: str) -> CapturePurgeAvailability` and `async purge_full_captures(session_id: str, expected_capture_revision: int) -> CapturePurgeResult`.
  - Optional Inspector `capture_revision_provider: Callable[[], int]`; expansion/Copy/Save compare the open revision immediately before reading a cached capture.

- [ ] **Step 1: Write failing database scope and rollback tests**

Create one conversation containing an active-path message, off-path sibling, abandoned regeneration, and soft-deleted message, each with Safe and Full exchanges. Assert the Full query includes all four and deletion leaves every Safe row/message/usage value untouched:

```python
keys = db.list_full_exchange_keys_for_conversation(conversation_id)
assert keys == {
    (active_id, "active-full", 0),
    (sibling_id, "sibling-full", 0),
    (abandoned_id, "abandoned-full", 0),
    (soft_deleted_id, "deleted-full", 0),
}
assert db.delete_full_exchanges_for_conversation(conversation_id) == 4
```

Monkeypatch the delete transaction to raise before commit and assert rows are byte-identical afterward.

Run: `pytest Tests/DB/test_chachanotes_message_exchanges.py -q`

Expected: FAIL because conversation-wide Full queries do not exist.

- [ ] **Step 2: Implement immutable-conversation count/delete SQL**

Use no `messages.deleted` filter:

```sql
SELECT exchange.message_id, exchange.run_tag, exchange.seq
  FROM message_exchanges AS exchange
  JOIN messages AS message ON message.id = exchange.message_id
 WHERE message.conversation_id = ?
   AND exchange.capture_detail = 'full';

DELETE FROM message_exchanges
 WHERE capture_detail = 'full'
   AND message_id IN (
       SELECT id FROM messages WHERE conversation_id = ?
   );
```

Wrap deletion in one `BEGIN IMMEDIATE` transaction and expose it through `ChatPersistenceService` without logging conversation titles or capture content.

Run: `pytest Tests/DB/test_chachanotes_message_exchanges.py -q`

Expected: PASS.

- [ ] **Step 3: Write failing staged in-memory/cache purge tests**

Cover durable and ephemeral sessions, `_messages_by_session` plus the complete node graph, `_exchange_blob_cache`, `_abandoned_exchange_run_tags`, failure before DB commit, and a later terminal flush:

```python
stage = store.stage_full_capture_purge(session.id)
assert any(c.capture_detail is CaptureDetail.FULL for c in message.exchanges)
persistence.delete_full_exchanges_for_conversation.side_effect = RuntimeError("rollback")
with pytest.raises(RuntimeError):
    store.commit_full_capture_purge(stage)
assert message.exchanges == original_exchanges

persistence.delete_full_exchanges_for_conversation.side_effect = None
removed = store.commit_full_capture_purge(stage)
assert removed == expected_full_count
assert all(c.capture_detail is CaptureDetail.SAFE for c in message.exchanges)
store._persist_exchanges_only(message)
assert persistence.full_rows == []
```

Run: `pytest Tests/Chat/test_console_chat_store_exchanges.py Tests/Chat/test_console_capture_purge.py -q`

Expected: FAIL because staged purge/revision ownership does not exist.

- [ ] **Step 4: Implement fallible staging before the durable commit and authoritative swaps after it**

`StagedCapturePurge` contains message references plus replacement exchange tuples, complete replacement blob caches, abandoned-tag maps, durable key inventory, and a complete replacement capture-revision map. Build all replacements before SQL. After commit, perform only reference assignments:

```python
deleted = persistence.delete_full_exchanges_for_conversation(conversation_id)
self._exchange_blob_cache = stage.blob_cache
self._abandoned_exchange_run_tags = stage.abandoned_tags
for message, exchanges in stage.message_swaps:
    message.exchanges = exchanges
self._capture_revisions = stage.capture_revisions
return deleted
```

No decoding, allocation, callback, notification, or repaint may occur between durable commit and those swaps. Ephemeral sessions skip SQL and perform the same swaps. Ensure `attach_message_exchanges` and `_persist_exchanges_only` refuse a session holding the quiescence lease.

Run: `pytest Tests/Chat/test_console_chat_store_exchanges.py Tests/Chat/test_console_capture_purge.py -q`

Expected: PASS.

- [ ] **Step 5: Write failing controller quiescence tests**

Parameterize each blocker: active primary task, preparation/admission, surviving fleet child, retained post-turn signals, and in-flight exchange flush. Also race a new submit and a purge:

```python
availability = controller.capture_purge_availability(session.id)
assert availability.can_purge is False
assert availability.reason_code == "fleet_writer_active"

result = await controller.purge_full_captures(
    session.id,
    expected_capture_revision=controller.capture_revision(session.id),
)
assert result.status == "blocked"
assert result.removed_count == 0
```

Run: `pytest Tests/Chat/test_console_capture_purge.py Tests/Chat/test_console_chat_controller_exchanges.py -q`

Expected: FAIL until the controller owns the lease.

- [ ] **Step 6: Implement the controller-owned capture-quiescence lease**

Under the existing submit/preparation serialization, mark the exact session quiescent before any await. The lease blocks new admission and exchange flush, then re-checks every writer before staging. It releases only after commit/swaps or a pre-commit failure. Use bounded reason codes and user-facing copy; never include capture content.

```python
async with self._capture_quiescence(session_id) as lease:
    if not lease.available:
        return CapturePurgeResult.blocked(lease.reason_code)
    stage = self.store.stage_full_capture_purge(session_id)
    removed = await self._run_durable_db_call(
        self.store.commit_full_capture_purge, stage
    )
    return CapturePurgeResult.deleted(
        removed_count=removed,
        capture_revision=self.store.capture_revision(session_id),
    )
```

Run: `pytest Tests/Chat/test_console_capture_purge.py Tests/Chat/test_console_chat_controller_exchanges.py -q`

Expected: PASS for all blockers, race cases, rollback, ephemeral behavior, and later-flush non-reinsertion.

- [ ] **Step 7: Write failing stale Inspector expansion/Copy/Save tests**

Open two Inspectors at revision N, purge through one, then attempt expansion, clipboard, and file save through the other. All actions must clear cached Full bodies and require Refresh:

```python
inspector._capture_revision_at_open = 7
revision.value = 8
assert inspector._copy_exchange_capture(call_key) is False
assert inspector._exchange_capture_by_call_key == {}
assert "Refresh" in inspector.query_one("#console-inspector-capture-status").renderable
```

Run: `pytest Tests/UI/test_console_conversation_inspector.py Tests/UI/test_chat_screen_console_inspector_loader.py -q`

Expected: FAIL because cached call maps do not validate a revision.

- [ ] **Step 8: Implement revision fences without adding the purge UI yet**

Inject immutable `target_session_id`, `target_conversation_id`, and `capture_revision_provider` from `ChatScreen._push_console_inspector`. Before/after async loads and immediately before expansion/Copy/Save, compare with the open revision. On mismatch, clear `_exchange_capture_by_call_key`, loaded-call/section/message sets, and mounted Full bodies; display `Stored captures changed · Refresh required`.

Run: `pytest Tests/UI/test_console_conversation_inspector.py Tests/UI/test_chat_screen_console_inspector_loader.py -q`

Expected: PASS.

- [ ] **Step 9: Run Gate 3 focused verification**

Run: `pytest Tests/DB/test_chachanotes_message_exchanges.py Tests/Chat/test_console_chat_store_exchanges.py Tests/Chat/test_console_capture_purge.py Tests/Chat/test_console_chat_controller_exchanges.py Tests/UI/test_console_conversation_inspector.py Tests/UI/test_chat_screen_console_inspector_loader.py -q`

Expected: PASS.

- [ ] **Step 10: Commit Gate 3**

```bash
git add tldw_chatbook/DB/ChaChaNotes_DB.py tldw_chatbook/Chat/chat_persistence_service.py tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Widgets/Console/console_conversation_inspector.py Tests/DB/test_chachanotes_message_exchanges.py Tests/Chat/test_console_chat_store_exchanges.py Tests/Chat/test_console_capture_purge.py Tests/Chat/test_console_chat_controller_exchanges.py Tests/UI/test_console_conversation_inspector.py Tests/UI/test_chat_screen_console_inspector_loader.py backlog/tasks/task-22507.3\ -\ Purge-Full-captures-under-conversation-quiescence.md
git commit -m "feat(console): purge full captures under quiescence"
```

---

### Task 4: Shared Inspector/Trace/Settings UX and Governed Export (`TASK-22507.4`)

**Required design skill:** Before editing frontend behavior or styling, read and apply `impeccable` to the approved UI contract without redesigning its scope.

**Files:**
- Create: `tldw_chatbook/Chat/console_exchange_export.py`
- Create: `tldw_chatbook/Widgets/Console/console_capture_policy_dialog.py`
- Create: `tldw_chatbook/Widgets/Console/console_exchange_export_dialog.py`
- Modify: `tldw_chatbook/Widgets/Console/trace_export_dialog.py:31-49,287-301`
- Modify: `tldw_chatbook/Widgets/Console/console_conversation_inspector.py:231-485,869-1445`
- Modify: `tldw_chatbook/UI/Screens/trajectory_screen.py:183-343,406-453,1186-1215,1875-1888`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:1803-1885,9644-9755`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py:console behavior compose/save/sync regions near 22700-23060`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Generate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Modify: `Docs/User_Guide/console/context-and-rag.md`
- Modify: `Docs/User_Guide/console/chat-basics.md`
- Create: `Tests/Chat/test_console_exchange_export.py`
- Create: `Tests/UI/test_console_capture_policy_dialog.py`
- Create: `Tests/UI/test_console_exchange_export_dialog.py`
- Modify: `Tests/UI/test_console_conversation_inspector.py`
- Modify: `Tests/UI/test_trajectory_screen.py`
- Modify: `Tests/UI/test_trajectory_live.py`
- Modify: `Tests/UI/test_trajectory_import_ui.py`
- Modify: `Tests/UI/test_trace_export_ui.py`
- Modify: `Tests/UI/test_settings_configuration_hub.py`
- Modify: `Tests/UI/test_settings_narrow_layout.py`

**Interfaces:**
- Consumes: Task 2 `CapturePolicySnapshot`/mutation methods and config-generation results; Task 3 purge/count/revision callbacks; existing `TraceExportProfile`.
- Produces:
  - `project_exchange_export(capture: ExchangeCapture, profile: TraceExportProfile) -> ExchangeExportProjection`.
  - `ExchangeExportProjection(profile: TraceExportProfile, payload: Mapping[str, Any], json_text: str, full_available: bool, disabled_reason: str | None)`.
  - `CapturePolicyBindings(target_session_id: str, target_conversation_id: str | None, read: Callable[[], CapturePolicySnapshot], apply_next: Callable[[CaptureDetail | None, int], CapturePolicyMutationResult], apply_conversation: Callable[[CaptureDetail | None, int], Awaitable[CapturePolicyMutationResult]], apply_global: Callable[[bool, CaptureDetail, int, int], CapturePolicyMutationResult], count_full: Callable[[], Awaitable[int]], purge_full: Callable[[int], Awaitable[CapturePurgeResult]], capture_revision: Callable[[], int])` frozen when the parent screen opens.
  - One `ConsoleCapturePolicyDialog` used by Inspector and live Trace; Settings uses the same global mutation and confirmation helpers.
  - One `ConsoleExchangeExportDialog` for a selected call and Clipboard/File destination.

- [ ] **Step 1: Write failing per-call export projection tests**

Pin all three existing profiles against Safe and Full captures:

```python
safe_summary = project_exchange_export(full_capture, TraceExportProfile.SAFE_SUMMARY)
assert "AGENTS BODY" not in safe_summary.json_text
assert safe_summary.payload["capture_detail"] == "full"

redacted = project_exchange_export(full_capture, TraceExportProfile.REDACTED_DIAGNOSTIC)
assert "AGENTS BODY" not in redacted.json_text
assert "messages_payload[0].content" in redacted.payload["omitted_keys"]

with pytest.raises(ExchangeExportUnavailable):
    project_exchange_export(safe_capture, TraceExportProfile.FULL_TRACE)

full = project_exchange_export(full_capture, TraceExportProfile.FULL_TRACE)
assert "AGENTS BODY" in full.json_text
assert "sk-STRUCTURED" not in full.json_text
```

Run: `pytest Tests/Chat/test_console_exchange_export.py -q`

Expected: FAIL because the exchange-specific governor does not exist.

- [ ] **Step 2: Implement the source-specific exchange exporter**

Import `TraceExportProfile`; do not create a second enum. Safe summary contains provider/model/status/usage/capture detail and omission/truncation inventories only. Redacted diagnostic reapplies Safe project-instruction redaction to the stored Full request. Full trace is available only when `capture.capture_detail is FULL`. All profiles run the binary stubber again defensively.

Run: `pytest Tests/Chat/test_console_exchange_export.py -q`

Expected: PASS.

- [ ] **Step 3: Publish and reuse the existing profile labels/Full warning**

Rename `_PROFILE_LABEL` and `_PROFILE_COPY` to public `TRACE_EXPORT_PROFILE_LABELS` and `TRACE_EXPORT_PROFILE_COPY`. Extract one confirmation factory:

```python
def full_trace_confirmation(*, noun: str) -> ConfirmationDialog:
    return ConfirmationDialog(
        title=f"Export full {noun}?",
        message=(
            "Full trace may include prompts, injected instructions, tool arguments, "
            "outputs, and local paths. Credentials remain structurally blocked."
        ),
        confirm_label=f"Export full {noun.lower()}",
        cancel_label="Go back",
    )
```

Use it from both Trace and exchange dialogs so copy cannot drift.

Run: `pytest Tests/UI/test_trace_export_ui.py Tests/UI/test_console_exchange_export_dialog.py -q`

Expected: existing Trace tests PASS; new exchange-dialog tests still FAIL until the dialog exists.

- [ ] **Step 4: Build the one-call profile/destination dialog with mandatory Full confirmation**

The dialog accepts one immutable `ExchangeCapture`, an expected capture revision, and a revision provider. It defaults to Redacted diagnostic, offers Clipboard/File as destinations, disables Full for Safe captures with visible reason text, validates revision immediately before projection and immediately before disclosure, confirms every Full action, and uses `validate_path` plus atomic file write behavior already used by Trace export.

Run: `pytest Tests/UI/test_console_exchange_export_dialog.py -q`

Expected: PASS for Safe unavailability, clipboard, file, overwrite, confirmation cancellation, confirmation on every repeat, and stale revision refusal.

- [ ] **Step 5: Write failing shared policy-dialog behavior tests**

Cover single-scope Apply, resulting-effective escalation, stale policy/config revisions, missing target, queued consumer, Capture Off, dormant Full, Safe write failure, Full write failure, cache-refresh degraded success, purge confirmation copy, and focus restoration:

```python
await dialog.apply(CaptureScope.CONVERSATION, None)  # Inherit
assert dialog.preview.effective_detail is CaptureDetail.FULL
assert dialog.preview.requires_confirmation is True

result = await dialog.apply(CaptureScope.NEXT_SEND, CaptureDetail.FULL)
assert result.confirmation_count == 0

await dialog.delete_full_captures()
assert "logical record deletion" in host.confirmation_message
assert "WAL" in host.confirmation_message
assert "capture policy remains Full" in host.confirmation_message
```

Run: `pytest Tests/UI/test_console_capture_policy_dialog.py -q`

Expected: FAIL because the modal does not exist.

- [ ] **Step 6: Implement the shared scoped policy/count/purge modal**

Use a scrollable body and fixed Cancel/Apply actions. Each Apply mutates exactly one selected scope. Conversation/global Full and any Inherit/disarm revealing Full use the approved warnings; next-send Full has no secondary confirmation; global Full requires explicit acknowledgement. Full-enabling edits are disabled while Off. Duplicate actions are disabled while Applying. All status is literal text (`Safe`, `Full`, `Off`, `Applying`, `Failed`, `Saved and active — settings cache refresh degraded`). Escape cancels and the parent restores focus.

Run: `pytest Tests/UI/test_console_capture_policy_dialog.py -q`

Expected: PASS.

- [ ] **Step 7: Wire immutable Inspector status, provenance, purge, and export**

At `_push_console_inspector`, capture the exact session/conversation IDs and build `CapturePolicyBindings`; never read `active_session_id` from a later button handler. Add the pinned two-line status above tabs and `c` binding. Historical call titles append `capture: Safe|Full`. Replace per-call Copy/Save buttons with one `Export…` action opening `ConsoleExchangeExportDialog`. On purge success clear loaded call/body maps and refresh count; on refresh failure preserve successful deletion status.

Run: `pytest Tests/UI/test_console_conversation_inspector.py Tests/UI/test_chat_screen_console_inspector_loader.py -q`

Expected: PASS for immutable targeting, provenance, stale revisions, purge repaint, export launch, and focus return.

- [ ] **Step 8: Wire live Trace and imported/shared read-only states**

Add `c` only when `CapturePolicyBindings` is present. Live title area renders `Future exchange capture: Safe|Full|Off · c Change…`; when an active run differs, show both frozen and future detail. Imported/shared traces render `Capture policy unavailable for imported Trace`, omit the binding from contextual hints, and cannot open the dialog.

Run: `pytest Tests/UI/test_trajectory_screen.py Tests/UI/test_trajectory_live.py Tests/UI/test_trajectory_import_ui.py -q`

Expected: PASS, including the BINDINGS/footer 1:1 invariant.

- [ ] **Step 9: Wire the canonical F9 global controls**

Add `exchange_capture` and `exchange_capture_detail` to Console Behavior using the same snapshot generation, mutation function, Full confirmation, Off-to-On resume warning, structured partial-success result, and status text used by the policy dialog. Do not add controls to `Tools_Settings_Window.py` or `enhanced_settings_sidebar.py`.

Run: `pytest Tests/UI/test_settings_configuration_hub.py Tests/UI/test_settings_narrow_layout.py Tests/test_config_save_settings_semantics.py -q`

Expected: PASS for Full/Off confirmations, stale generation, Safe write failure, cache-reload degraded success, and reload equivalence.

- [ ] **Step 10: Add production-shaped 80x24 geometry and keyboard tests**

Use `ConsolidatedCSSApp` for Inspector, policy dialog, exchange export, live Trace, imported Trace, and F9 Console Behavior at `(80, 24)`. Assert status text and fixed actions have nonzero visible regions, content scrolls, disabled reasons are visible, `c` opens only eligible surfaces, Escape returns focus, and no binding shadows ADR-031 keys.

Run: `pytest Tests/UI/test_console_capture_policy_dialog.py Tests/UI/test_console_exchange_export_dialog.py Tests/UI/test_console_conversation_inspector.py Tests/UI/test_trajectory_screen.py Tests/UI/test_trajectory_live.py Tests/UI/test_trajectory_import_ui.py Tests/UI/test_settings_narrow_layout.py -q`

Expected: PASS.

- [ ] **Step 11: Add styles, regenerate the bundle, and verify integrity**

Put source rules in `_agentic_terminal.tcss`, use semantic tokens only, then generate and check:

Run: `python3 -m tldw_chatbook.css.build_css`

Expected: `tldw_chatbook/css/tldw_cli_modular.tcss` regenerated from modules.

Run: `python3 -m tldw_chatbook.css.check_bundle_sync`

Expected: PASS with no source/bundle drift.

- [ ] **Step 12: Update user documentation and privacy wording**

Document: Safe default; three scopes and one-shot expiry; Anthropic system/messages/tools and injected AGENTS/workspace/RAG/tool content in Full; semantic adapter boundary and llama.cpp exception; structured credential exclusion; ordinary text may contain secrets; 64 MiB/16 MiB bounds; compression-not-encryption; per-call export profiles; logical purge; WAL/free-page/snapshot/export/backup limits; Capture Off dormant Full/resume warning; and imported Trace read-only behavior.

Run: `rg -n "Safe|Full|Anthropic|AGENTS|compression|WAL|backup|logical|64 MiB|16 MiB" Docs/User_Guide/console/context-and-rag.md Docs/User_Guide/console/chat-basics.md`

Expected: every named boundary has explicit user-facing copy.

- [ ] **Step 13: Run the final targeted privacy and UI gate**

Run: `pytest Tests/Chat/test_console_exchange_capture.py Tests/Chat/test_console_exchange_export.py Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_console_chat_controller_exchanges.py Tests/Chat/test_console_capture_purge.py Tests/DB/test_chachanotes_message_exchanges.py Tests/UI/test_console_capture_policy_dialog.py Tests/UI/test_console_exchange_export_dialog.py Tests/UI/test_console_conversation_inspector.py Tests/UI/test_chat_screen_console_inspector_loader.py Tests/UI/test_trajectory_screen.py Tests/UI/test_trajectory_live.py Tests/UI/test_trajectory_import_ui.py Tests/UI/test_trace_export_ui.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_settings_narrow_layout.py -q`

Expected: PASS.

Run: `python3 -m tldw_chatbook.css.check_bundle_sync`

Expected: PASS.

Run: `git diff --check`

Expected: no whitespace errors.

- [ ] **Step 14: Perform a production-shaped privacy inspection**

Using a test database and log sink, send one Safe and one Full Anthropic-shaped exchange containing unique sentinels for system, AGENTS/workspace instruction, RAG snippet, tool schema, tool arguments/result, ordinary semantic secret, structured API key, endpoint userinfo/query/fragment, and nested base64 response. Inspect decoded ChaChaNotes rows, in-memory/cache owners, Redacted and Full exports, and configured filesystem logs. Expected: Full semantic sentinels exist only in the Full capture/confirmed Full export; Safe omits tagged instruction bodies; structured key/URL credentials/base64 never appear anywhere; logs contain only content-free categories.

- [ ] **Step 15: Close Backlog documentation and commit Gate 4**

Mark child acceptance criteria only after the evidence above exists. Add concise Implementation Notes to each completed child, including ADR-089, exact commands/results, modified files, and whether any generalizable lesson was discovered. Keep `TASK-22507` open until all four children are Done.

```bash
git add tldw_chatbook/Chat/console_exchange_export.py tldw_chatbook/Widgets/Console/console_capture_policy_dialog.py tldw_chatbook/Widgets/Console/console_exchange_export_dialog.py tldw_chatbook/Widgets/Console/trace_export_dialog.py tldw_chatbook/Widgets/Console/console_conversation_inspector.py tldw_chatbook/UI/Screens/trajectory_screen.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Docs/User_Guide/console/context-and-rag.md Docs/User_Guide/console/chat-basics.md Tests/Chat/test_console_exchange_export.py Tests/UI/test_console_capture_policy_dialog.py Tests/UI/test_console_exchange_export_dialog.py Tests/UI/test_console_conversation_inspector.py Tests/UI/test_trajectory_screen.py Tests/UI/test_trajectory_live.py Tests/UI/test_trajectory_import_ui.py Tests/UI/test_trace_export_ui.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_settings_narrow_layout.py backlog/tasks/task-22507.4\ -\ Expose-shared-Full-capture-controls-and-governed-exchange-export.md
git commit -m "feat(console): expose governed full capture controls"
```

## Final Integration Gate

- [ ] Re-run `pytest Tests/ChaChaNotesDB Tests/DB -q` after every migration conflict/rebase.
- [ ] Re-run the Task 4 targeted privacy/UI command after the four child commits are together.
- [ ] Re-run `python3 -m tldw_chatbook.css.check_bundle_sync` and `git diff --check`.
- [ ] Ask the owner whether to run the repository-wide full test suite; do not infer approval from this targeted plan.
- [ ] Review `git diff --stat` and `git status --short`; stage only files owned by this feature in the dirty worktree.
- [ ] Complete all child task ACs/notes/statuses, then complete `TASK-22507` with implementation notes linking ADR-089 and this plan.
