# Lock-safe Private SQLite Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve SQLite lock ownership while retaining private-file validation, backup identity checks and TTS lifecycle safety, then unblock the existing Canvas qualification task.

**Architecture:** Run original-inode file inspection in fresh exec'd, operation-owned helpers. Keep SQL transactions and connection factories in the parent; use fixed metadata-only helper proof for live TTS, explicit authority handoffs, and bounded terminal retention after live proof loss. Live exact-current TTS handles verify the public native no-checkpoint-on-close policy before SQL; healthy close explicitly checkpoints under proof, while exclusive migration/evidence artifacts retain their separate ownership rules.

**Tech Stack:** Python >=3.12, public SQLite Connection.setconfig/getconfig and SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE, standard-library subprocess/pipe/JSON, existing Textual workers, pytest and real SQLite. No new dependency or custom SQLite build.

**Spec:** [Approved design](../specs/2026-09-07-sqlite-lock-safe-private-validation-design.md).

**Backlog:** [TASK-31942](<../../../backlog/tasks/task-31942 - Resolve-native-SQLite-crash-blocking-Canvas-qualification.md>), In Progress. Numbered tasks below are implementation/review units within this task, not new Backlog IDs.

ADR required: yes
ADR path: backlog/decisions/125-lock-safe-private-sqlite-validation.md
Reason: This implements the accepted cross-process privacy/proof boundary, terminal-retention contract and approved native-close/runtime amendment. Preserve ADR-028, ADR-029 and ADR-051; no duplicate ADR is needed.

## Global Constraints

- Work only in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/canvas-v1`, branch `codex/canvas-v2-mermaid-design`. Original baseline is `9bc73ffb3`; Tasks1–4 remain reviewed through `8b4e5c1d4`. The amendment resumes from documentation checkpoint `f202b8090` plus this plan update. Preserve unrelated changes.
- Read the spec, TASK-31942, ADR-028/029/051/125, and `backlog/docs/lessons-testing-evidence.md`, `lessons-live-verification.md` and `lessons-backlog-hygiene.md` before implementation.
- Protocol version 1; length-prefixed JSON; maximum of 64 KiB per request or response; closed fields, one outstanding request per lease. No database contents, exception messages or raw traceback are returned.
- General launch/control waits: five seconds maximum; initial fixed TTS validation: 30-second maximum. Close: one second, then a further two-second terminate/kill/reap bound. Earlier operation deadlines govern admission and IPC together; cleanup has its separately reported bound.
- At most eight helpers: four retained TTS owners and four reserved transient slots. Normal preparation reserves one transient slot; source-pin backup/copy/restore reserves two. TTS admission reserves one retained permit plus one transient slot atomically. No nested wait for additional capacity while holding an insufficient reservation.
- TTS evidence retains the one-million metadata-row ceiling and 576-MiB artifact ceiling. No BLOB selection, full serialization, snapshot file or startup database copy.
- No parent opens, duplicates or receives original database/sidecar FDs for live inspection. Parent directory FDs are permitted. No pool, daemon, listener, generic SQL RPC, caller-selected import/executable, new dependency or global all-SQLite registry.
- Preserve owner registration, target-kind admission, custom factories, read-only source-mode exceptions, optional-sidecar generation policy, no-follow/type/owner/link/mode/identity checks and expected-identity semantics. Windows remains explicitly unverified; memory databases bypass helpers.
- Preserve WAL, transactions, restore safety snapshots and indeterminate reporting. No schema/mmap/journal-mode workaround or user-data/environment mutation.
- Python >=3.12 is approved. Before TTS initialization/admission, probe public native close-policy support on an owned in-memory handle; refuse with source-free `runtime_unsupported` if absent/rejected. Configure and verify again on each live exact-current TTS handle before first SQL. Never use a numeric fallback, private ABI, global factory toggle or runtime installation.
- Keep the live handle's native flag enabled through finalization. Healthy serialized cleanup rolls back pending work, revalidates each phase, settles tombstones and attempts one PASSIVE checkpoint. Valid partial/exact BUSY leaves WAL intact; other errors retain the cleanup owner and worker for guarded retry. Restore retains its stricter existing checkpoint and must not acquire a duplicate PASSIVE checkpoint. Initialization/exclusive evidence/publication/recovery are outside this live flag policy.
- Lost live TTS proof requires explicit restart-required terminal quarantine, including partial setup after a live SQLite handle exists. Do not force-close or remint proof. Healthy siblings remain usable; new TTS admission is latched off. Terminal owners retain their permits and resources until process exit.
- Canceling an async waiter does not abandon its shielded worker. Worker-observed cancellation stops new work and enters owned cleanup.
- Use repository pytest pre-import isolation for all app imports. Only owned temporary databases, subprocesses and browsers. Do not run ad hoc application imports against user configuration.
- Targeted tests only. No full repository sweep, PR, push, rebase or merge. Canvas V2 remains disabled until its existing independent admission gates pass; completion of this correction is not admission.
- Use `apply_patch` for manual edits. Commit only named task files, never blanket-stage the untracked diagnostic. Do not weaken test or performance budgets to obtain green results.

## Evidence and file map

The original untracked `Tests/DB/test_private_sqlite_lock_diagnosis.py` contained intentional failures: raw SHM close, actual private SHM preparation and the actual private connection seam released an existing writer lock. Its existence after worktree recovery is not assumed. Promoted committed regressions and preserved evidence below are the source of current claims; do not recreate or blanket-stage a historical diagnostic as a green test.

Historical evidence lives under `.superpowers/sdd/2026-09-06-chatbook-canvas-v2-mermaid-implementation/`: `sqlite-crash-diagnosis.md`, corrected `sqlite-descriptor-design-inventory.md` and `sqlite-lock-*.log`. Existing SIGBUS captures live under `output/playwright/mermaid-release/native-db/`. Do not claim this proves every historical crash's exact interleaving.

| File | Responsibility |
| --- | --- |
| New `tldw_chatbook/DB/private_sqlite_protocol.py` | Bounded frame codec, closed operation schemas, immutable identity projections; stdlib only. |
| New `tldw_chatbook/DB/private_sqlite_files.py` | Extracted existing descriptor-relative validation and source-pin logic; no SQLite connection factory or subprocess launcher. |
| New `tldw_chatbook/DB/private_sqlite_helper.py` | Fixed child dispatcher, pin ownership, parent-liveness checks; no parent-policy decisions. |
| New `tldw_chatbook/DB/private_sqlite_helper_entry.py` | Isolated installed-file entry point and fixed dependency-leaf bootstrap. |
| New `tldw_chatbook/DB/private_sqlite_process.py` | Parent admission reservations, pipe deadlines, child/lease ownership and bounded cleanup. |
| Existing `tldw_chatbook/DB/private_sqlite.py` | Registered policy, public connection/backup seams and exclusive candidate ownership; delegate live file inspection. |
| New `tldw_chatbook/TTS/profile_validation.py` | Extract shared schema/domain codecs and metadata validators without live opener, migration orchestration or app initialization. |
| New `tldw_chatbook/TTS/profile_sqlite_proof.py` | Fixed child TTS evidence initializer and retained-cohort operations. |
| New `tldw_chatbook/TTS/profile_sqlite_policy.py`; new `Tests/TTS/test_profile_sqlite_policy.py` | Public runtime capability/configuration checks; no file access, helper launch, checkpoint or app/config import. |
| Existing `tldw_chatbook/TTS/profile_schema.py` | Live SQLite wrapper/admission and compatibility exports for extracted validators. |
| Existing `tldw_chatbook/TTS/profile_repository.py` | Restore export, directory authority, shielded lifecycle and terminal quarantine ownership. |
| Existing `tldw_chatbook/TTS/profile_errors.py` | Distinct bounded runtime-unsupported and restart-required repository errors. |
| Existing `tldw_chatbook/DB/sql_validation.py`; new `tldw_chatbook/DB/sql_identifier_core.py` | Preserve public logging wrappers while extracting the exact identifier grammar/escaping needed by isolated schema checks. |
| Existing `tldw_chatbook/TTS/profile_migration_publication.py`, `profile_migration_recovery.py`, `profile_migration_namespace.py` | Exclusive ownership audit and close-failure retention. |
| Existing `pyproject.toml`, active runtime docs/scripts and CI contracts enumerated in Task5a; new `Tests/Packaging/test_python_runtime_floor.py` | Consistent approved Python floor without rewriting independent packages or historical evidence. |
| New `Tests/Packaging/test_private_sqlite_helper_distribution.py` | Installed-wheel entry/import isolation; further package discovery changes only if the wheel omits required files. |
| New `Tests/DB/test_private_sqlite_protocol.py`, `test_private_sqlite_process.py`, `test_private_sqlite_lock_preservation.py` | Frame/admission/process tests and actual lock regressions. |
| New `Tests/TTS/test_profile_sqlite_proof.py`, `test_profile_sqlite_helper_lifecycle.py` | Fixed proof and real repository/helper lifecycle tests. |

Do not relocate whole TTS domain/migration modules. Extract only the validator closure, preserving existing public imports and calling semantics. Do not implement a second weaker schema/privacy policy.

## Common verification and commit rules

Run commands from the worktree above. The interpreter is `../../.venv/bin/python`. Each task records the exact RED command/result before its implementation and the GREEN command/result afterward. An import error is sufficient only for a genuinely new API test; existing-lock and lifecycle regressions must fail for their asserted behavior.

After each task, run its listed focused selection, `git diff --check`, and Ruff check/format-check on its touched Python files. Record existing unrelated formatting debt separately; do not reformat entire large legacy modules or waive introduced errors. Do not install/change shared dependencies to make tooling available.

For example, Task 1's exact static selection is:

```bash
../../.venv/bin/python -m ruff check --output-format concise tldw_chatbook/DB/private_sqlite.py tldw_chatbook/DB/private_sqlite_protocol.py tldw_chatbook/DB/private_sqlite_files.py tldw_chatbook/DB/private_sqlite_helper.py tldw_chatbook/DB/private_sqlite_helper_entry.py Tests/DB/test_private_sqlite_protocol.py
../../.venv/bin/python -m ruff format --check tldw_chatbook/DB/private_sqlite.py tldw_chatbook/DB/private_sqlite_protocol.py tldw_chatbook/DB/private_sqlite_files.py tldw_chatbook/DB/private_sqlite_helper.py tldw_chatbook/DB/private_sqlite_helper_entry.py Tests/DB/test_private_sqlite_protocol.py
git diff --check
```

Subsequent tasks select their own enumerated files, including new files. Every task finishes with a scoped commit and independent review under the selected execution workflow; issues must be resolved before its dependent task. No task's intermediate green subset establishes whole-correction completion.

### Task 1: Extract leaf validation and define the bounded child protocol

**Files:** Create protocol/files/helper/entry modules and `Tests/DB/test_private_sqlite_protocol.py`; modify `DB/private_sqlite.py` only to share the extracted raw validation implementation, preserving its current orchestration. Read `Utils/private_paths.py`; do not change its policy.

**Interfaces:**

- `FileIdentity` is a frozen, `repr=False` projection with integer `dev`, `ino`, `mode`, `uid`, `gid`, `nlink`, `size`, `mtime_ns`, `ctime_ns`; `from_stat(value: os.stat_result) -> FileIdentity` and `same_inode(other: FileIdentity) -> bool`. Separate comparisons preserve parent-security and post-init contracts; never use one full-tuple equality for every policy.
- `encode_frame(payload: dict[str, object]) -> bytes` and `decode_frame(frame: bytes) -> dict[str, object]` validate framing and JSON structure. `ProtocolError` has source-free fixed text.
- `PrepareRequest` carries `path: str`, `writable: bool`, `create_if_missing: bool`, `preserve_source_mode: bool`; it does not accept owner policy, executable or suffix input. `PrepareResult` carries main identity plus existing privacy status/reason projections for the fixed cohort. Parent policy determines permitted request values.
- `prepare_batch(request: PrepareRequest) -> PrepareResult` in `private_sqlite_files.py` delegates the existing no-follow generation logic to each fixed artifact.
- Child operations initially support `prepare`, `pin_source`, `recheck_source`, `close`; source rechecks accept no new path. TTS operations are added only in Task 4.

- [x] Write codec tests with actual adversarial frames, including duplicate JSON keys, non-finite numbers, booleans in integer identity fields, unknown fields/version/operation, oversized length/body, invalid UTF-8, truncation and trailing bytes.

```python
import struct
import pytest
from tldw_chatbook.DB.private_sqlite_protocol import (
    ProtocolError, decode_frame, encode_frame,
)

def test_frame_round_trip_preserves_closed_request():
    value = {"version": 1, "operation": "close"}
    assert decode_frame(encode_frame(value)) == value

@pytest.mark.parametrize("body", [
    b'{"version":1,"version":2,"operation":"close"}',
    b'{"version":1,"operation":"close","unexpected":true}',
    b'{"version":1,"operation":"fetch"}',
])
def test_closed_frame_refuses_ambiguous_or_unknown_requests(body):
    with pytest.raises(ProtocolError):
        decode_frame(struct.pack("!I", len(body)) + body)

def test_oversize_length_is_rejected_before_body_allocation():
    with pytest.raises(ProtocolError):
        decode_frame(struct.pack("!I", 65537))
```

- [x] Run `../../.venv/bin/python -m pytest -q Tests/DB/test_private_sqlite_protocol.py` and record RED.
- [x] Implement the four-byte unsigned network-order length prefix, 65,536-byte body limit and closed per-direction schemas. The stream reader checks the length before allocating/reading the body; codec tests alone do not establish bounded pipe I/O.

```python
def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError()
        result[key] = value
    return result
```

Use this object hook with `json.loads`, reject non-finite constants and cap nesting before accepting a frame. Decode errors never include source/path/payload text. Response schemas distinguish existing private-path statuses from unavailable/protocol/timeout errors.

- [x] Move `_open_artifact_fd`, `_prepare_posix_artifact_generation` and their actual dependency closure into the file leaf. Keep the four-generation optional-sidecar policy and all double-open/fstat/fchmod postconditions. Existing private-seam callers delegate to the same implementation, not a copied algorithm. Run direct leaf tests for correct/wrong mode, missing main/optional sidecars, link/owner/parent substitution, read-only preservation and churn exhaustion.
- [x] Implement the fixed helper entry for `[sys.executable, '-I', '-S', absolute_entry_path]`. Use private stdin/stdout pipes and discarded stderr. Since importing `tldw_chatbook` executes startup code, bootstrap only fixed package namespaces for `tldw_chatbook`, `DB` and `Utils`, with `__path__` set from the installed entry's verified package location; do not execute their `__init__.py`. No caller-controlled module names or search paths. The only child imports at this stage are the named helper/protocol/files modules, `Utils.private_paths` and stdlib. Test from a hostile cwd containing fake package files; do not add cwd to `sys.path`.
- [x] Run the new tests plus `Tests/Utils/test_private_paths.py` and relevant existing artifact-generation cases in `Tests/DB/test_private_sqlite.py`. Parent monkeypatches cannot affect an exec child: direct leaf fault injection remains in-process, while future integration tests must invoke the real helper. Do not rewrite an integration test to mock away the process boundary.
- [x] Commit the five named new Python files and the modified private seam with message `refactor(db): isolate private SQLite file validation and protocol`; review the leaf extraction and child import boundary.

### Task 2: Own helper reservations, pipe deadlines and process cleanup

**Files:** Create `DB/private_sqlite_process.py`, `Tests/DB/test_private_sqlite_process.py`; extend the Task 1 helper/entry for retained pins and idle-parent detection.

**Interfaces:**

- `OperationDeadline(expires_at: float | None)` exposes `remaining(cap: float) -> float`, using monotonic time; expiration raises `HelperTimeoutError` before additional work.
- `HelperAdmission` exposes `reserve(*, transient: int, retained: int, deadline: OperationDeadline) -> HelperReservation`. The production instance is process-owned, reset/refused after fork; test instances allow isolated accounting, not configurable product ceilings.
- `HelperReservation` is an explicit context-managed owner. Nested calls receive it directly. It retains charges for unreaped children and terminal retained owners; it cannot release a live child's capacity accidentally on `__exit__`.
- `HelperLease.start(request: PrepareRequest, *, operation: str, reservation: HelperReservation, deadline: OperationDeadline) -> HelperLease`; `request(operation: str, *, deadline: OperationDeadline) -> dict[str, object]`; `close() -> None`. The parent call validates operation membership; rechecks cannot retarget the lease.
- `HelperUnavailableError`, `HelperTimeoutError`, `HelperProtocolError`, `HelperCleanupError` have closed source-free reasons. Cleanup results distinguish reaped, still-owned and terminal-retained resources.

Original-parent identity must be captured before exec: the launcher overwrites the fixed internal `_TLDW_PRIVATE_SQLITE_PARENT_PID` child-environment field, and the entry consumes/validates it before imports and file preparation. No caller-configurable environment patch or wire field is added. Direct-entry test fixtures supply this required launcher metadata; include parent death before delayed entry initialization, not only after successful initialization.

- [x] Add isolated admission tests, starting with capacity that a retained owner cannot borrow:

```python
from contextlib import ExitStack
import time
import pytest
from tldw_chatbook.DB.private_sqlite_process import (
    HelperAdmission, HelperTimeoutError, OperationDeadline,
)

def test_retained_admission_leaves_transient_headroom():
    admission = HelperAdmission()
    with ExitStack() as owned:
        for _ in range(4):
            owned.enter_context(admission.reserve(
                transient=0, retained=1,
                deadline=OperationDeadline(time.monotonic() + 1),
            ))
        with admission.reserve(
            transient=2, retained=0,
            deadline=OperationDeadline(time.monotonic() + 1),
        ):
            with pytest.raises(HelperTimeoutError):
                admission.reserve(
                    transient=1, retained=1,
                    deadline=OperationDeadline(time.monotonic()),
                )
```

- [x] Run `../../.venv/bin/python -m pytest -q Tests/DB/test_private_sqlite_process.py`, record RED, then implement reservation accounting under one condition/lock. Atomically acquire the entire requested pair or acquire nothing. Capacity tests use barriers, not scheduling sleeps; include two concurrent two-slot operations, four TTS initializers, an excess admission, nested reservation borrowing and oversized-envelope refusal.
- [x] Implement nonblocking pipe read/write with selector waits against the same absolute deadline. Check response length before reading its body. Use one request at a time, no background unbounded output collector. Partial header/body writes and reads consume the same budget.

```python
wait_seconds = deadline.remaining(cap=5.0)
events = selector.select(wait_seconds)
if not events:
    raise HelperTimeoutError()
```

- [x] Add actual-child tests for EOF, crash, malformed/oversized/truncated replies, stalled input/output, short caller deadlines, parent exit and an inherited-pipe orphan scenario. Fault child programs belong only to test fixtures; production never accepts a test executable/request option. Assert only the captured child is signaled and there are no descendants.
- [x] Implement close with one-second normal exit, then terminate/kill/reap within the further two-second bound. Retain the child object and capacity charge if reaping fails; report cleanup failure without masking `KeyboardInterrupt`/`SystemExit`/worker cancellation. Parent identity polling runs at least once per second while idle. Check captured parent/process identity after fork rather than reusing inherited reservations or pipes.
- [x] Run protocol/process tests together and repeat bounded create/close cycles while measuring live child counts and FD counts. Tests assert no growth on normal/early-failure paths, not on explicitly retained failure states.
- [x] Commit the parent process module, helper changes and process tests with message `feat(db): bound SQLite helper ownership and deadlines`; review capacity, cancellation and failure ownership before integration.

### Task 3: Repair normal connection and online-backup lock ownership

**Files:** Modify `DB/private_sqlite.py`, `Tests/DB/test_private_sqlite.py`, `test_private_sqlite_inventory.py`, `test_private_sqlite_interop_owners.py`; create `Tests/DB/test_private_sqlite_lock_preservation.py`. Promote, but do not blindly commit, `test_private_sqlite_lock_diagnosis.py`.

**Interfaces:**

- Add explicit keyword-only `operation_deadline: float | None = None` and internal `reservation: HelperReservation | None = None` at `_connect_registered_sqlite`; consume both before its existing `sqlite3.connect` call. Do not switch normal opens to the captured `_SQLITE_CONNECT`, which is currently reserved for descriptor views. Public `connect_private_sqlite` accepts the operation deadline without changing existing SQLite kwargs or `expected_identity: os.stat_result | None` callers.
- `_PinnedSQLiteSource` retains `selected`, `identity: FileIdentity`, `enforce_private_mode` and `lease: HelperLease`, not main/sidecar FDs. Alias comparisons accept validated identity projections internally while public stat-result input remains valid.
- Backup/copy/restore entry points acquire the two-slot envelope once and pass it plus the absolute deadline through pinning, every recheck and sequential connection preparation. Borrowed SQLite connections remain caller-owned.

- [x] Create a real cross-process writer oracle in the new test module using the diagnostic's separately exec'd stdlib SQLite contender. Its result is exactly `0` or `sqlite3.SQLITE_BUSY`; unexpected errors, timeout or startup failure fail the test. Add the actual private-seam regression:

```python
def test_private_connect_preserves_existing_wal_writer(tmp_path):
    path = tmp_path / "owned.sqlite"
    with contextlib.closing(sqlite3.connect(path, isolation_level=None)) as owner:
        owner.execute("PRAGMA journal_mode=WAL")
        owner.execute("CREATE TABLE probe(value INTEGER)")
        owner.execute("BEGIN IMMEDIATE")
        assert not _other_process_can_begin_write(path)
        with contextlib.closing(private_sqlite.connect_private_sqlite(
            "db.chachanotes.primary", path,
        )):
            assert owner.in_transaction
            assert not _other_process_can_begin_write(path)
        assert not _other_process_can_begin_write(path)
        owner.rollback()
```

Copy `_other_process_can_begin_write` from the preserved diagnostic with a fixed five-second child timeout and strict error-code assertion. Keep paths test-owned. Add rollback-journal writer/read-lock variants and a WAL snapshot/checkpoint oracle: WAL readers do not forbid all writers, so test snapshot stability/checkpoint exclusion rather than asserting an incorrect WAL-reader writer lock.

- [x] Run `../../.venv/bin/python -m pytest -q Tests/DB/test_private_sqlite_lock_preservation.py` and record the actual lock-exclusion RED before routing the seam.
- [x] Delegate main plus fixed sidecars to one helper batch before parent SQLite open. Retain local owner/target admission and the last expected-identity check; memory bypass and Windows policy remain unchanged. Never call the raw live-inode leaf locally after helper success. Ensure custom factory return values and borrowed raw SQLite handles do not acquire new ownership obligations.

```python
# Ordering inside the existing registered seam, after policy admission:
prepared = prepare_in_helper(request, reservation=reservation, deadline=deadline)
verify_expected_named_identity(selected, expected_identity)
connection = sqlite3.connect(database_argument, **sqlite_kwargs)
return connection
```

`prepare_in_helper(request: PrepareRequest, *, reservation: HelperReservation, deadline: OperationDeadline) -> PrepareResult` is a Task 3 parent adapter over `HelperLease.start(request, operation='prepare', reservation=reservation, deadline=deadline)` with owned close. `verify_expected_named_identity(selected: Path, expected_identity: os.stat_result | FileIdentity | None) -> None` is the extracted existing stat-only expected-identity comparison; do not introduce raw file opens or claim to eliminate the preexisting pathname-open race.

- [x] Replace source-pin FD ownership with `pin_source`/`recheck_source` calls. Inject helper failure before backup, during the existing progress guard, and on final recheck; assert callbacks, rollback/indeterminate errors and close ownership remain exact. Hold a separate live connection's transaction across successful and failing pin release.
- [x] Preserve the diagnostic raw-SHM-close negative control in an isolated child experiment with the opposite expected outcome; never run the destructive negative control against a database later used for product assertions. Only remove the old untracked diagnostic after every case and historical log is accounted for; preserve its bare-launch timing as historical evidence, not a performance claim.
- [x] Run the four targeted DB files and `Tests/Utils/test_private_paths.py`; include `test_restore_fails_promptly_and_unchanged_for_active_transactions` with its unchanged one-second assertion. Resolve test seams invalidated by exec boundaries using direct leaf tests plus actual integration tests, not parent-only monkeypatch success.
- [x] Commit the explicit DB files with message `fix(db): preserve SQLite locks across private opens and backup pins`; review actual cross-process lock evidence and borrowed-handle ownership.

Task 3 checkpoint: `0e5363446` passed independent spec and quality gates. Root
lock/protocol/process selection: 167 passed; remaining listed DB/private-path
selection: 372 passed, two Windows-only skips, three strict inventory failures.
The unchanged legacy Collections raw opener, maintenance owner-ID mismatch and
tracked `super().backup` census remain separate qualification gaps, not waivers.
Original diagnostic source and promotion map are preserved in the evidence directory.
No full qualification or Canvas admission is established. The quality reviewer
approved the scoped production work with the parser-spy integration correction
assigned to the already-planned Task 4 extraction below.

### Task 4: Extract and run the fixed metadata-only TTS proof

**Files:** Create `TTS/profile_validation.py`, `TTS/profile_sqlite_proof.py`, `DB/sql_identifier_core.py`, `Tests/TTS/test_profile_sqlite_proof.py`; modify `TTS/profile_schema.py`, `DB/sql_validation.py`, helper/entry/protocol/process modules, `Tests/DB/test_private_sqlite_protocol.py`, `Tests/DB/test_private_sqlite_process.py`, `Tests/DB/test_sql_validation.py` and `Tests/TTS/test_profile_schema.py`. Process changes are limited to fixed TTS operation membership and its existing planned 30-second initializer/normal five-second control budgets; no live repository lifecycle work yet.

Extraction lint treatment: twelve existing broad exception-normalization/cleanup
handlers move with the shared validator. Preserve their semantics, including
hostile mapping refusal and control-flow precedence. Line-specific `BLE001`
annotations with concrete safety rationales are permitted only for those proven
relocated boundaries; record original/new locations for independent review. No
file-wide/global suppression, trace logging, narrower exception list, or artificial
control-flow rewrite solely to satisfy lint. Other introduced diagnostics must be
fixed normally; a clean configured check does not mean the broad catches vanished.

**Interfaces:**

- Move `_validate_schema`, `_validate_schema_body`, their manifest/codec/domain helpers, `validate_profile_store_rows` and `_stream_exact_store_metadata_evidence` into `profile_validation.py`; keep compatibility imports at `profile_schema.py`. Keep signatures and error semantics, including `check_deadline`, unchanged.
- Extract exact identifier grammar/escaping into `sql_identifier_core.py`; `sql_validation.py` keeps public signatures and logging wrappers. The isolated validator imports the leaf, not loguru. Reuse existing stdlib-only domain types/errors, reference bounds and versioned DDL; do not duplicate constants.
- `TTSProof` child owner supports `initialize()`, `pin_sidecars()`, `recheck()`, `export_restore_authority()` and `close()` for its one original path. Wire operations are `tts_exact_current`, `tts_pin_sidecars`, `tts_recheck`, `tts_export_restore_authority`, `close`.
- `TTSRestoreAuthority` is a frozen, source-free record with `parent: FileIdentity`, `main: FileIdentity`, `wal: FileIdentity`, `shm: FileIdentity`. Repository generation is attached/checked by the parent adapter, not supplied as permission by generated/caller content.

- [x] Write shared-validator parity tests on real v4, unsupported-version, malformed-schema and invalid-domain databases. Exercise large references without selecting audio or text payloads. Add an exact TTS proof test using a closed current store built with existing `profile_schema.open_profile_store` under pytest isolation:

  Preserve the oversized-options pre-parser test by scoping its spy to the
  extracted domain parser, not the shared stdlib `json.loads` module. At the
  Task 3 checkpoint its candidate case correctly rejects corrupt data but the
  global spy also sees four valid helper response frames. Do not weaken the
  assertion that the actual oversized options value never reaches JSON parsing.

```python
def test_fixed_tts_proof_accepts_current_store_without_returning_rows(tmp_path):
    path = tmp_path / "profiles.sqlite3"
    connection = profile_schema.open_profile_store(path)
    connection.close()
    deadline = OperationDeadline(time.monotonic() + 30)
    with HelperAdmission().reserve(
        transient=1, retained=1, deadline=deadline,
    ) as reservation:
        lease = HelperLease.start(
            PrepareRequest(str(path), False, False, False),
            operation="tts_exact_current", reservation=reservation,
            deadline=deadline,
        )
        try:
            reply = lease.request("tts_recheck", deadline=deadline)
            assert set(reply) == {"version", "operation", "status", "identity"}
        finally:
            lease.close()
```

For the initializer's pre-live absent-sidecar state, `tts_recheck` validates that bound state; after `tts_pin_sidecars`, it requires the complete original cohort. Do not let repeated pin commands accept a replacement generation.

An ordinary post-initialization TTS authority refusal must preserve the healthy
helper, original pins and usable control channel: restoring the same exact
authority must permit recheck and eventual close. Test this through a real helper
lease, not only a direct `TTSProof` object. Initial proof refusal and actual
transport/protocol/helper failure remain fail-closed; this is not permission to
reprove a new path or continue SQLite work while authority is refused.

- [x] Run `../../.venv/bin/python -m pytest -q Tests/TTS/test_profile_sqlite_proof.py` and record RED. Extract the validator closure without changing existing schema versions/DDL or decoding behavior. Move only needed routines; migration orchestration stays in `profile_schema.py`.
- [x] Extend fixed namespace bootstrap to `TTS` and `TTS.migrations`. The import closure is restricted to the new validator/proof, existing profile types/errors/reference types, migration DDL modules, migration-journal bounds and stdlib leaf modules. No `profile_repository`, `profile_schema` live opener, config, providers, logging or package initializer. Test the actual child import graph and installed wheel in Task 7.
- [x] Implement child initialization: pin original parent/main/cohort, open its fixed read-only immutable descriptor view directly with sqlite3, configure/version/schema/domain/metadata checks, close SQL before raw pins, then retain only identity pins. Enforce 30 seconds and row/artifact ceilings without payload copies or recursive helpers. Refuse mixed/substituted sidecars; return only closed status/identity fields.

```python
validate_profile_store_rows(evidence, check_deadline=check_deadline)
_stream_exact_store_metadata_evidence(evidence)
# Do not return the digest's input rows or any reference payload across IPC.
evidence.close()
evidence = None
```

- [x] Run new proof tests, `Tests/TTS/test_profile_schema.py`, and `Tests/DB/test_sql_validation.py`. Add exact malformed/oversized TTS-operation frames to protocol tests. Existing normal exact-open no-BLOB/serialization and incremental metadata tests must still pass.
- [x] Commit the named leaf/validator/protocol/tests with message `refactor(tts): share isolated metadata-only SQLite proof`; review schema parity, import isolation and original-inode ownership before the live wrapper uses it.

Task 4 complete through `8b4e5c1d4`: initial implementation `2db961c4d`,
spec correction `d643bbc90`, staged capture `9fb1e3a4b`, typed privacy refusal
correction `8b4e5c1d4`. Independent spec and quality gates pass. Approved retry
retains/revalidates exact acquired WAL/main/directory pins and binds missing SHM
once; acquired pins are never reopened/replaced/reminted. Incomplete cohorts
refuse use/export. Known numeric and typed filesystem authority refusals preserve
the initialized healthy helper; unexpected/internal/transport failures remain
fatal. Actual-child tests cover no-pin/partial/complete refusal and exact
restoration, original child/pins/capacity, parent permission changes and initial
failure classification.

Fresh final implementer selection: 415 passed, two existing skips, one existing
dependency warning in 38.99s; separately authorized local parser parameters:
two passed in 0.80s. Root actual permission/initializer controls: three passed;
scoped static and immutable diff checks pass. Reviewer independently reproduced
refusal, same-child/main recovery on exact restoration and clean reap.
Task 5 may start; this is not whole-correction or Canvas qualification.

Earlier repository/lifecycle run: 360 passes and 11 stdlib multiprocessing
semaphore ENOSPC setup failures, reproduced without Chatbook imports outside
sandbox. A fresh probe after the user stopped the runaway worktree remover
still failed. Those tests and three unchanged strict inventory gaps remain
unqualified, not waived.

Recovery: the isolated checkout disappeared twice; user identified and stopped
a runaway removal process. Main checkout was untouched. Approval commit
`2a0b206dd` survived, the recovered product hash matched exactly, reconstructed
test ASTs were checked, and final verification reran after recovery. Historical
ignored logs are not assumed available. Current correction records belong to
`.superpowers/sdd/2026-09-07-sqlite-lock-safe-private-validation-implementation/`;
external recovery artifacts remain under `/private/tmp/sqlite-task4-evidence.RSZgc4/`.
Attribute historical evidence separately from fresh runs.

### Task 5a: Adopt the approved runtime floor and fail-closed TTS capability admission

**Approval and evidence:** The user approved Python >=3.12 and then the written
native-close amendment. The old [orderly-exit gate](../reviews/2026-09-07-sqlite-orderly-exit-gate.md)
remains a required production regression. The [native spike](../reviews/2026-09-07-sqlite-native-close-policy-spike.md)
passed 25 configured cases; eight default controls still reproduced deletion.
This task establishes runtime compatibility/admission, not live shutdown safety.
Task5b integrates that policy with the reviewed helper and owns the shutdown gate.

**Files:**

- Create `tldw_chatbook/TTS/profile_sqlite_policy.py`, `Tests/TTS/test_profile_sqlite_policy.py`, `Tests/Packaging/test_python_runtime_floor.py`.
- Modify `tldw_chatbook/TTS/profile_errors.py` and the pre-initialization admission point in `tldw_chatbook/TTS/profile_repository.py`; extend `Tests/TTS/test_profile_repository_lifecycle.py` for preflight refusal only. Do not replace live ownership in this task.
- Modify `pyproject.toml` (requires-python, classifiers, mypy target); `README.md`, `AGENTS.md`, `CLAUDE.md`, `Packaging/README.md`, `Packaging/windows/build_windows.py`, `scripts/preflight.sh`, `run_all_tests_with_report.py`, and active examples in `scripts/terminal_qualification/README.md`.
- Modify `.github/workflows/derived-artifacts.yml`, `.github/workflows/test.yml`, `.github/workflows/nightly-deep.yml`, `.github/workflows/css-bundle-guard.yml`; corresponding `Tests/CI/test_github_actions_test_workflow.py`, `test_ci_queue_pressure_contract.py`, `test_derived_artifacts_workflow.py` only where their qualification contracts change.
- Modify `Tests/Architecture/test_python_floor_syntax.py`; preserve the explicitly legacy 3.11 detector in `Tests/floor_syntax.py` unless a small docstring clarification is necessary. No general syntax-parser rewrite.

**Interfaces:**

- `require_native_close_policy_support() -> None` creates, configures, verifies and closes one owned `sqlite3.connect(":memory:")` probe. Check that the public methods and named constant exist before connecting; do not cache success across actual handles. Use no app/config imports, file paths, pragma, helper or numeric constant fallback.
- `configure_native_close_policy(connection: sqlite3.Connection) -> None` borrows the caller's handle, calls `setconfig(sqlite3.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE, True)`, then requires `getconfig(...) is True`. It executes no SQL and never closes the borrowed handle. Task5b's opener owns its failure cleanup.
- Missing/rejected public capability maps to `ProfileRepositoryError("runtime_unsupported") from None`. Add that closed code and fixed public text: `TTS profile repository unavailable: SQLite runtime lacks required close-policy support.` Preserve bounded error serialization; do not include SQLite exception text or paths. Do not translate unrelated programming errors with an indiscriminate catch-all.
- `_worker_open` performs the probe after any already-owned cleanup and before canonicalizing/initializing a new store or acquiring new store/helper ownership. Preserve existing cleanup owners and normal error propagation. A failed probe does not latch terminal proof loss.

- [ ] Add the new policy tests with a real native handle and missing-capability negative control:

```python
import sqlite3
import pytest
from tldw_chatbook.TTS import profile_sqlite_policy as policy
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

def test_configure_native_policy_verifies_flag_without_sql():
    connection = sqlite3.connect(":memory:")
    statements = []
    connection.set_trace_callback(statements.append)
    try:
        policy.configure_native_close_policy(connection)
        assert connection.getconfig(sqlite3.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE) is True
        assert statements == []
        connection.execute("SELECT 1")  # The borrowed handle remains open.
    finally:
        connection.close()

def test_missing_constant_refuses_before_opening_probe(monkeypatch):
    monkeypatch.delattr(policy.sqlite3, "SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE")
    def unexpected_connect(*args, **kwargs):
        pytest.fail("unsupported runtime opened a handle")
    monkeypatch.setattr(policy.sqlite3, "connect", unexpected_connect)
    with pytest.raises(ProfileRepositoryError) as failure:
        policy.require_native_close_policy_support()
    assert failure.value.code == "runtime_unsupported"
    assert "SQLite runtime lacks required close-policy support" in str(failure.value)
```

Native positive tests require the declared supported capability; report an unavailable build as an explicit qualification gap, not a fake pass. Add focused proxies for missing methods, rejected `setconfig`, rejected/false `getconfig`, and source-text sentinel suppression. Verify the owned probe closes on configuration failure and success; the borrowed configuration helper never closes. Test the closed code survives pickling and unknown codes still map to `operation_failed`.
- [ ] Run `../../.venv/bin/python -m pytest -q Tests/TTS/test_profile_sqlite_policy.py` and record RED for the absent API.
- [ ] Implement the leaf policy and closed error mapping. Keep configuration equivalent to the following, with explicit capability checks and the specified bounded mapping around SQLite's public capability refusals:

```python
option = sqlite3.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE
connection.setconfig(option, True)
if connection.getconfig(option) is not True:
    raise ProfileRepositoryError("runtime_unsupported")
```

- [ ] Add `test_unsupported_runtime_refuses_before_store_initialization` to the lifecycle tests. Patch the repository's imported preflight function to raise `runtime_unsupported`, then open a repository under a nonexistent `tmp_path / "not-created" / "profiles.sqlite3"`. Assert that exact error, no parent/store/lease/helper creation and bounded `close()`. Restore preflight and prove a fresh owned repository can open/close, with admission counters unchanged. Run that exact node RED before inserting the probe; run GREEN afterward.
- [ ] Add runtime-floor metadata tests before changing metadata. Parse TOML with `tomllib` and assert `requires-python == ">=3.12"`, no Python3.11 classifier and mypy target `3.12`. Check the named active build/preflight floors and qualification CI jobs, not every historical 3.11 string in the repository. Run `Tests/Packaging/test_python_runtime_floor.py` RED.
- [ ] Update the named metadata/docs/scripts and CI contracts. Minimum package/AST qualification jobs move to 3.12. Remove the unsupported Ubuntu3.11 nightly row instead of duplicating the existing Ubuntu3.12 row. Preserve Ubuntu3.12/3.13, macOS3.12 and Windows3.12/cp1252 rows, existing triggers, concurrency, sequencing and budgets. Leave standalone backlog guard and standalone CI-shape-only Python3.11 jobs unchanged because they do not install/parse Chatbook. Do not bump `packages/tldw_profile_core`, vendored projects, backend-specific environments or dependencies; retain historical Windows3.11 evidence.
- [ ] Correct the syntax guard's coupling to the old floor. Keep historical PEP701 verdicts explicitly labeled Python3.11, including an optional comparison with an actual 3.11 interpreter. The supported-floor guard must compile every shipped module on the actual declared interpreter, including when pytest itself is running on 3.12 (remove that early skip). On newer test runtimes, report missing 3.12 explicitly; minimum-runtime CI must execute the real compile check. Do not use `ast.parse(feature_version=...)` as tokenizer evidence or keep applying the legacy PEP701 detector as a 3.12 rejection rule.
- [ ] Add real-floor positive and negative controls to that guard: `x = f"{ {"k": 1}["k"] }"` compiles on 3.12; `def broken(: pass` does not. Feed these through the same subprocess compile harness used by the shipped-module check. This protects the guard when accepting now-valid PEP701 source; no runtime download during tests.
- [ ] Run the bounded selection below and `zsh -n scripts/preflight.sh`; syntax-check the two changed Python entry scripts without executing their build/full-suite bodies. Run static checks on the touched files and inspect the full active-runtime diff for accidental historical/vendor/dependency changes.

```bash
../../.venv/bin/python -m pytest -q --tb=short \
  Tests/TTS/test_profile_sqlite_policy.py \
  Tests/TTS/test_profile_repository_lifecycle.py::test_unsupported_runtime_refuses_before_store_initialization \
  Tests/Packaging/test_python_runtime_floor.py \
  Tests/Packaging/test_release_metadata.py \
  Tests/Architecture/test_python_floor_syntax.py \
  Tests/CI/test_github_actions_test_workflow.py \
  Tests/CI/test_ci_queue_pressure_contract.py \
  Tests/CI/test_derived_artifacts_workflow.py
```

- [ ] Commit only the enumerated changed files as `feat(tts): require native SQLite close-policy capability`; obtain independent spec and quality review. Document unavailable interpreter/platform evidence and existing unrelated failures. This task alone does not claim the live flag is applied or foreign-cohort finalization is fixed.

### Task 5b: Migrate live TTS authority, native close policy and terminal cleanup

**Depends on:** reviewed Tasks1–4 and Task5a. Resume the production shutdown gate,
not the discarded default-close approach. Read both preserved gate and native
spike reports before implementing. All new steps below remain unchecked.

**Files:** Modify `TTS/profile_schema.py`, `TTS/profile_repository.py`, `TTS/profile_errors.py`, `DB/private_sqlite_process.py`, and `TTS/profile_migration_namespace.py` only for the required validated metadata adaptation; create `Tests/TTS/test_profile_sqlite_helper_lifecycle.py`; extend `Tests/TTS/test_profile_repository_lifecycle.py` and focused namespace comparison tests as needed. Exclusive finalizer behavior remains Task 6.

**Interfaces:**

- `_ExactCurrentProfileConnection` owns parent live SQLite, helper lease, verified directory-only FD, identity metadata and retained admission permit. It owns no original main/WAL/SHM raw FD or parent immutable evidence SQL connection.
- Consume Task5a's `configure_native_close_policy(connection: sqlite3.Connection) -> None` immediately after the exact live factory returns and before query-only/version/WAL/metadata SQL. Do not call it through the generic or shared initialization factory. The wrapper's `close() -> None` rechecks proof, closes SQLite with the flag still set, then reaps/releases helper/directory/permit ownership; it does not perform PASSIVE SQL. Repository normal cleanup owns that explicit checkpoint, so restore's existing TRUNCATE sequence is not duplicated.
- `export_restore_authority(*, deadline: OperationDeadline) -> TTSRestoreAuthority` revalidates helper and local directory, then returns immutable metadata. `verified_parent_fd(*, deadline: OperationDeadline) -> int` borrows the wrapper-owned directory FD for immediate existing tombstone settlement; callers do not close it.
- Add `ExactProfileStoreProofLostError` as an `ExactProfileStoreAuthorityError` subtype in `profile_schema.py`; distinguish terminal helper loss from a healthy-helper namespace mismatch. Add closed repository code `restart_required` with user-facing text stating restart is required, without paths.
- The process-owned admission object exposes `latch_tts_proof_loss() -> None`; subsequent retained admissions refuse. It records no database paths/connections. Existing repository/application ownership retains terminal objects and permits; it does not use a generic SQLite registry.

- [ ] Write the real two-repository lock test before replacing the wrapper:

```python
@pytest.mark.asyncio
async def test_closing_one_repository_preserves_sibling_writer(tmp_path):
    path = tmp_path / "profiles.sqlite3"
    first = TTSProfileRepository(path)
    second = TTSProfileRepository(path)
    await first.open()
    await second.open()
    await second._submit_operation(lambda c: c.execute("BEGIN IMMEDIATE"))
    try:
        assert not _other_process_can_begin_write(path)
        await first.close()
        assert not _other_process_can_begin_write(path)
    finally:
        await second._submit_operation(lambda c: c.rollback())
        await second.close()
```

Define the strict contender in this test module from Task 3's diagnostic algorithm, or import a shared test-only support module after explicitly moving it there. Never import another test module just to acquire fixtures. Include partial sidecar-pin failure while a sibling owns a real transaction.

- [ ] Run the new test and record behavioral RED. Replace local pins/evidence SQL with the fixed helper. Preserve query-only-before-admission, exact version, metadata validation, post-init binding and named/parent/sidecar checks at every existing guarded use. Pass the live opener's absolute deadline through both helper and normal private seam, using the initial reserved capacity.
- [ ] Add `test_exact_live_policy_precedes_first_sql` using the actual live opener and a tracing factory proxy: record successful set/get and require it before the first execute, including `PRAGMA query_only`. Add rejected actual-handle set/get after successful memory preflight: assert no SQL, no usable publication and ordinary close/reap/release. Inject a first native close failure and assert `ExactProfileStoreCleanupError.connection` retains the complete owner for retry. An already-open but unused handle cannot be dropped merely because configuration failed.
- [ ] Add `test_live_policy_does_not_change_initialization_or_immutable_evidence`: observe exact-live handles configured, with creation/migration/evidence handles still following their existing ownership/checkpoint rules. Exercise fresh store, supported legacy-schema migration, completed schema publication and two real reopen cycles with committed residual WAL. Run these focused nodes RED, apply the scoped policy, then GREEN. No factory-wide monkeypatch is the product implementation.
- [ ] Replace both descriptor-field consumers explicitly. `_worker_close_for_restore` captures `TTSRestoreAuthority` before checkpoint and revalidates afterward; it installs generation-bound parent/sidecar identities before closing. `_worker_cleanup` calls `verified_parent_fd` for required reusable-tombstone settlement. Remove their permissive missing-field `getattr` branches. Preserve exact namespace removal under exclusive restore ownership; metadata export never authorizes a replacement cohort.

```python
authority = exact_connection.export_restore_authority(deadline=deadline)
self._restore_sidecar_identities = {
    "-wal": authority.wal,
    "-shm": authority.shm,
}
# Adapt ParentAuthority/internal comparison types explicitly; do not synthesize
# incomplete os.stat_result values or weaken existing exact-removal checks.
```

- [ ] Add real restore-with-retained-sidecars, failed export before close, migration/restore tombstones through final close, and integer parent gid/mode/identity substitution tests. Preserve the existing healthy-helper exact-namespace restoration and close-retry behavior.
- [ ] Add real healthy-cleanup tests before changing `_worker_cleanup`: pending writes are rolled back, committed writes recover after reopen, a pinned reader yields a valid partial PASSIVE checkpoint, and a live sibling writer remains excluded from an external contender after this repository closes. Observe exactly one `PRAGMA main.wal_checkpoint(PASSIVE)` on normal cleanup and none on restore's guarded TRUNCATE close path. Inject exact `sqlite_errorcode == sqlite3.SQLITE_BUSY` separately from IOERR/LOCKED/malformed results: only exact BUSY or a valid three-integer checkpoint result can continue under valid proof. A busy/partial result never authorizes unlinking remaining WAL/SHM.
- [ ] Implement normal cleanup on its existing serialized worker: proof check, roll back its pending transaction, proof check, verified-directory tombstone settlement, proof check, one PASSIVE checkpoint, post-proof check, wrapper native close, then normal remaining ownership release. Preserve deadlines/progress cancellation; do not add a retry loop or alter auto-checkpoint. On non-BUSY checkpoint or close failure retain the connection/lease/helper/worker for the existing guarded retry. Update `_finish_close` so an error with a retained cleanup owner cannot shut down its executor. Keep this separate from terminal proof-loss handling.

```python
# Within serialized normal cleanup, with the surrounding phase proof checks:
if connection.in_transaction:
    connection.rollback()  # Never commit merely to close.
try:
    checkpoint = connection.execute("PRAGMA main.wal_checkpoint(PASSIVE)").fetchone()
except sqlite3.Error as error:
    if getattr(error, "sqlite_errorcode", None) != sqlite3.SQLITE_BUSY:
        raise
else:
    # Require SQLite's three integer fields: busy in {0, 1}, and either
    # nonnegative log/checkpoint counts with checkpointed <= log or (-1, -1).
    # Invalid results are operation_failed with the owner retained, not BUSY.
    busy, log_frames, checkpointed_frames = checkpoint
```

The result-shape checks occur before unpacking; tests cover wrong length/types/ranges. Use existing deadline/progress machinery rather than an unbounded new SQL path. A newly lost helper at any phase switches to terminal retention, with no subsequent rollback/checkpoint/close.
- [ ] Add helper-loss tests in isolated owned processes, not the main pytest process, so retained SQLite handles/workers cannot contaminate later tests. Kill only the captured helper; assert use and close yield `restart_required`, no SQL/finalizer cleanup occurs in-process, repeated new repository construction cannot acquire retained capacity, and healthy sibling work continues. Include helper loss between live SQLite open and wrapper publication.
- [ ] Implement terminal handling before ordinary authority failure handling:

```python
except ExactProfileStoreProofLostError:
    self._exact_authority_quarantined = True
    self._helper_restart_required = True
    admission.latch_tts_proof_loss()
    raise ProfileRepositoryError("restart_required") from None
```

`_helper_restart_required` is initialized false on repository construction and never reset in-process. Keep references through the existing app-owned repository; do not release its SHARED lease, live SQLite, worker or retained permit. Reap the dead helper independently. Healthy-helper close failure retains the ordinary retry path. Pre-live helper failure releases ordinary ownership and does not latch. Admission latching checks existing waiters inside the reservation loop and wakes them; unrelated transient reservations and already-healthy siblings remain usable. Bound terminal owners by the existing four retained permits, not a new global connection registry.
- [ ] Qualify normal application shutdown and abrupt exit separately with private namespaces and externally held observer handles. During terminal retention, exclusive store acquisition must remain blocked. After owned process exit it must succeed, with foreign substituted cohorts unchanged. If orderly interpreter finalizers close SQLite unsafely or ownership cannot be retained without hanging shutdown, STOP: report the failed approved-design gate. Do not substitute `os._exit`, force-kill product behavior, restore foreign paths in the fixture to hide failure, or silently weaken the contract.
- [ ] Use an actual app-owned repository/worker for ordinary shutdown, not only a module-global sqlite handle. Seed stores in a separate child so the live parent never acquires legacy raw main/WAL/SHM descriptors. Exercise idle/read/write states, helper loss during partial publication, one/two live owners, outstanding statements and large transactions that actually spill pages. Audit whether production can expose a BLOB handle; test its real lifetime if reachable, otherwise document the concrete API boundary rather than inventing a new BLOB API.
- [ ] Keep foreign-cohort preservation and original-data recovery as separate observations: compare foreign names/inodes/link counts/bytes through external observer handles after exit, then recover the original owned store in a separately owned namespace without letting fixture cleanup mutate the observed foreign cohort. Verify committed values, rollback of uncommitted/spilled writes and integrity after ordinary native finalization. Retain default-close failure only as a diagnostic control, not a deliberately failing normal-suite test. Report unsupported platform coverage explicitly.
- [ ] Cancel an async waiter while its repository worker is paused at a barrier. Assert the worker retains its reservation/helper until settlement and no later operation reuses its capacity early. Verify deadline exhaustion prevents publication but permits bounded owned cleanup.
- [ ] Run the new lifecycle file plus `Tests/TTS/test_profile_repository_lifecycle.py` and `Tests/TTS/test_profile_repository.py`. Confirm all terminal tests are process-contained and normal repeated open/close cycles are leak-free.
- [ ] Commit only the named TTS/process/test files with message `fix(tts): preserve remote proof authority across repository lifecycle`; obtain independent ownership/security review before final qualification.

### Task 6: Correct exclusive descriptor-view finalizers

**Files:** Modify `DB/private_sqlite.py`, `TTS/profile_migration_publication.py`, `profile_migration_recovery.py`, `profile_errors.py`, and affected namespace ownership code; extend `Tests/DB/test_private_sqlite.py`, `Tests/TTS/test_profile_migration_publication.py`, `test_profile_migration_recovery.py`, `test_profile_repository_lifecycle.py`.

**Interfaces:** Preserve `connect_private_sqlite_descriptor(owner_id, descriptor_fd, **kwargs)` for the remaining registered exclusive owners. Borrow the verified FD during SQLite open; no `os.dup`/immediate raw close. Remove only the obsolete live `tts.profile_store_descriptor` owner after Task5b no longer uses it. Public caller-owned descriptors remain borrowed.

Add `ProfileMigrationCleanupError` in `profile_errors.py`, carrying a source-free retained `owner` whose `close() -> None` retries teardown only. Propagate that owner through existing migration/recovery callers to the repository; no broad error mapper may discard it. This is a specific operation owner, not a connection registry or permission to resume failed publication.

- [ ] Add close-failure tests at publication `_immutable_validate` and recovery `_validate_authoritative_targets`, observing actual raw-close attempts with their existing owned temporary artifacts. Inject a SQLite proxy whose first close raises and whose retry closes the real connection. Assert the wrapper retains the exact SQLite connection, file and parent FDs, and no raw close/hash/rename proceeds after the failed close.

```python
class CloseOnceFailure:
    def __init__(self, connection):
        self.connection = connection
        self.failed = False

    def __getattr__(self, name):
        return getattr(self.connection, name)

    @property
    def row_factory(self):
        return self.connection.row_factory

    @row_factory.setter
    def row_factory(self, value):
        self.connection.row_factory = value

    def close(self):
        if not self.failed:
            self.failed = True
            raise OSError("owned test close failure")
        self.connection.close()
```

Add this behavioral publication test alongside the existing `_store` and `_prepared` helpers; the recovery test exercises its actual journal-authorized target rather than bypassing journal admission:

```python
def test_immutable_validation_retains_pins_when_sqlite_close_fails(tmp_path, monkeypatch):
    module = _publication_module()
    path = tmp_path / module.PROFILE_MIGRATION_CANDIDATE_LEAVES[
        module.ProfileMigrationPublicationSlot.ACTIVE
    ]
    _store(path, version=4, marker="close-failure")
    identity = _prepared(module, path, module.ProfileMigrationPublicationSlot.ACTIVE, "")
    parent_fd, file_fd, leaf = module._open_exact(identity)
    real_connect = module.connect_private_sqlite_descriptor
    proxy = CloseOnceFailure(real_connect(
        "tts.profile_migration_publication_descriptor", file_fd,
        isolation_level=None,
    ))
    monkeypatch.setattr(module, "_open_exact", lambda value: (parent_fd, file_fd, leaf))
    monkeypatch.setattr(module, "connect_private_sqlite_descriptor", lambda *a, **k: proxy)
    try:
        with pytest.raises(Exception) as failure:
            module._immutable_validate(identity)
        assert proxy.failed
        assert os.fstat(file_fd).st_ino == path.stat().st_ino
        assert os.fstat(parent_fd).st_ino == path.parent.stat().st_ino
        assert failure.value.owner is not None
        failure.value.owner.close()
    finally:
        if proxy.failed:
            proxy.close()
        for descriptor in (file_fd, parent_fd):
            try:
                os.close(descriptor)
            except OSError:
                pass
```

On the existing implementation the fstat assertion fails with a closed FD after the injected SQL close failure. Once the ownership API exists, narrow the exception assertion to `ProfileMigrationCleanupError` and assert its retry closes exactly once; fixture fallback cleanup is not product ownership evidence.

- [ ] Run the new exact failure nodes and record RED. Use the existing `_close_profile_migration_destination` retained-owner pattern to settle SQL before raw pins. Each caller must retain cleanup authority in its existing operation/repository owner or a typed cleanup exception carrying that owner; a local variable lost on exception is not retention. Preserve control-flow exception precedence.
- [ ] Borrow the original descriptor in the exclusive immutable opener, and update inventory tests. Audit every raw original-inode close in private_sqlite and TTS schema/publication/recovery/namespace/repository. Record a concrete helper-owned or tested closed/exclusive lifetime for each; an owner string, `immutable=1`, `_RECOVERY_LOCK` or absent sidecars alone does not prove exclusivity.
- [ ] Verify actual repository entry points hold EXCLUSIVE ownership and have settled live handles before low-level migration/recovery publication. Test shared-sibling refusal and callbacks that fail/try to escape handles. Keep exact tombstone/content checks and refuse an unproven live consumer; do not expand helper RPC to hashing/publication or add a global connection registry.
- [ ] Run the listed DB/publication/recovery/lifecycle selections; commit with message `fix(db): retain exclusive SQLite artifacts after close failure`; independent review checks the completed descriptor census and failure paths.

### Task 7: Qualify installed isolation, storage behavior and actual Canvas children

**Files:** Create `Tests/Packaging/test_private_sqlite_helper_distribution.py`; extend appropriate tests in `Tests/Performance/test_app_startup_performance.py`; run the existing import-weight, UI-ready census and screen-preimport budget guards without changing their ceilings; update `Docs/Canvas/V2_VERIFICATION.md`, TASK-31942 implementation notes and the preserved evidence/progress records. Modify `pyproject.toml` only if the wheel test proves a missing helper/leaf file.

**Interfaces:** No new runtime API. Consume the actual launcher, fixed operations and repository lifecycle from Tasks 1–6. Preserve V2 disabled policy while running candidate fixtures.

- [ ] Add an installed-wheel test before treating checkout execution as packaging evidence. Build with existing offline packaging fixtures/toolchain into `tmp_path`, install the built artifact with no dependency resolution into an isolated temporary target, and launch the installed absolute entry with `-I -S` from an unrelated hostile cwd. Verify the entry and complete leaf closure are from that wheel, not the checkout/PYTHONPATH/user site. No shared-venv install or network fetch.

```python
completed = subprocess.run(
    [sys.executable, "-I", "-S", str(installed_entry)],
    input=encode_frame({"version": 1, "operation": "close"}),
    env={**os.environ, "_TLDW_PRIVATE_SQLITE_PARENT_PID": str(os.getpid())},
    capture_output=True, cwd=hostile_cwd, timeout=5, check=True,
)
assert decode_frame(completed.stdout)["operation"] == "close"
assert completed.stderr == b""
```

The test defines `installed_entry` by inspecting the built wheel's installed files, and creates `hostile_cwd` under `tmp_path`. Exercise a real prepare and fixed TTS initialization too; successful close alone does not prove the full import closure is packaged. Instrument imports in the test-owned child harness, not through a new production diagnostics operation.
- [ ] Run installed-wheel RED/GREEN tests and verify no imports of app/config/providers/keyring/loguru/Textual startup or user files. Confirm the approved Python3.12 minimum, native policy availability and actual macOS POSIX behavior; report unavailable OS/interpreter coverage explicitly, never as passing Windows/Linux qualification.
- [ ] Run the complete affected storage selection:

```bash
../../.venv/bin/python -m pytest -q --tb=short \
  Tests/DB/test_private_sqlite_protocol.py \
  Tests/DB/test_private_sqlite_process.py \
  Tests/DB/test_private_sqlite_lock_preservation.py \
  Tests/DB/test_private_sqlite.py \
  Tests/DB/test_private_sqlite_inventory.py \
  Tests/DB/test_private_sqlite_interop_owners.py \
  Tests/DB/test_sql_validation.py \
  Tests/Utils/test_private_paths.py \
  Tests/Utils/test_private_persistent_artifacts.py \
  Tests/TTS/test_profile_sqlite_proof.py \
  Tests/TTS/test_profile_sqlite_policy.py \
  Tests/TTS/test_profile_sqlite_helper_lifecycle.py \
  Tests/TTS/test_profile_schema.py \
  Tests/TTS/test_profile_repository.py \
  Tests/TTS/test_profile_repository_lifecycle.py \
  Tests/TTS/test_profile_migration_publication.py \
  Tests/TTS/test_profile_migration_recovery.py \
  Tests/Packaging/test_private_sqlite_helper_distribution.py \
  Tests/Packaging/test_python_runtime_floor.py \
  Tests/Architecture/test_python_floor_syntax.py \
  Tests/CI/test_github_actions_test_workflow.py \
  Tests/CI/test_ci_queue_pressure_contract.py \
  Tests/CI/test_derived_artifacts_workflow.py \
  Tests/Performance/test_app_startup_performance.py \
  Tests/Performance/test_app_import_weight.py \
  Tests/Performance/test_ui_ready_module_census.py \
  Tests/Performance/test_screen_preimport_payload_budget.py
```

- [ ] Benchmark identical synthetic workloads against the immutable baseline in a separate temporary checkout, never move this worktree's HEAD. Measure full helper launch/batch, threaded app startup, TTS metadata-only open, repeated proof rechecks, peak live helpers/FDs and real child readiness. Keep existing performance ceilings unchanged, including the 972-module UI-ready ceiling governed by ADR-097 (`backlog/decisions/097-boot-budget-ratchets.md`); read its measurement/exception rules before interpreting a breach. The earlier 21.34ms bare-interpreter median is only a historical launch floor.
- [ ] Run the exact previously failed actual-child nodes using their existing candidate fixtures:

```bash
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no \
  'Tests/Canvas/browser/test_canvas_served_flow.py::test_actual_child_control_refusal_keeps_terminal_usable[snapshot]' \
  'Tests/Canvas/browser/test_canvas_served_flow.py::test_actual_chatbook_console_finalizes_canvas_create_and_update[normal-True]' \
  'Tests/Canvas/browser/test_canvas_served_flow.py::test_actual_chatbook_console_finalizes_canvas_create_and_update[read-publication-False]' \
  Tests/Canvas/browser/test_canvas_served_flow.py::test_canonical_adversarial_corpus_stays_in_served_product_route \
  Tests/Canvas/browser/test_canvas_zero_egress.py::test_canonical_adversarial_corpus_stays_in_native_product_route
```

Preserve the existing source-free child fault/lifecycle captures. Do not reinterpret a disconnect as successful refusal, weaken fail-closed transport, disable maintenance, or silently rerun a crash until it disappears. Diagnose each failure on its evidence.
- [ ] Run final static/format checks on the explicit changed files and `git diff --check`. Obtain independent review of the whole TASK-31942 correction, including terminal shutdown evidence and all source/descriptor owners, not only the last test commit.
- [ ] Add concise implementation notes and exact evidence to TASK-31942 only after implementation/review. Check each AC and mark Done via Backlog CLI only if all gates pass; otherwise keep In Progress and document the precise failing gate. Record relevant hard-won lessons with their incident. Commit qualification/docs as `test(db): qualify lock-safe SQLite helper integration`.
- [ ] Hand back to Task 8 of `Docs/superpowers/plans/2026-09-06-chatbook-canvas-v2-mermaid-implementation.md`. Rerun its full required candidate/admitted qualification and independent reviews under that plan. Do not enable V2 as part of this correction commit or claim the previously admitted 1,346-pass/4-fail run was green.

## Spec coverage and handoff

| Approved contract | Implementation/review unit |
| --- | --- |
| Existing privacy checks, isolated fixed helper and installed code | 1, 3, 7 |
| Framing, failure classification, limits, parent identity, cleanup | 1, 2, 7 |
| Whole-operation reservations and deadline/cancellation ownership | 2, 3, 5b |
| Normal SQLite, factories, memory/Windows and borrowed backup handles | 3 |
| Fixed metadata-only TTS evidence and no payload transfer | 4 |
| Approved Python floor, supported native API, source-free pre-init refusal | 5a, 7 |
| Pre-SQL actual-handle policy; initialization/exclusive scope and retained failure cleanup | 5b, 6 |
| Healthy rollback/PASSIVE partial/BUSY/error, residual WAL and restore checkpoint distinction | 5b, 7 |
| Restore export, directory handoff and exact cohort checks | 5b |
| Terminal proof loss, bounded retention, healthy siblings, both exit modes and finalizer data recovery | 5b, 7 |
| Exclusive descriptor finalizers and complete consumer inventory | 6 |
| Real lock oracles, packaging, performance and actual Canvas regressions | 3, 5b, 7 |

Plan self-review checks interfaces across tasks, all approved spec sections, exact existing test names, bounded errors and unchecked work status. Tasks1–4 have already passed their independent gates; do not restart them. Task5a then Task5b resume the user's selected subagent-driven workflow, with separate spec/quality gates before Task6/7. None of the amendment implementation is complete at this plan checkpoint. Preserve the known semaphore ENOSPC and strict-inventory gaps as unqualified evidence, not passing tests or permission for host cleanup/unrelated repairs.
