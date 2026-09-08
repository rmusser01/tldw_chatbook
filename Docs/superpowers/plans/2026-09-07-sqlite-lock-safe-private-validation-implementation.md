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
- Modify `Tests/DB/test_private_sqlite_inventory.py` and `backlog/docs/sqlite-private-owner-inventory.md` only to account for the approved fixed, argument-free capability probe's literal `sqlite3.connect(":memory:")`. This creates no filesystem owner or new public connection seam. Keep all file/URI connections under the existing registration policy.

**Interfaces:**

- `require_native_close_policy_support() -> None` creates, configures, verifies and closes one owned `sqlite3.connect(":memory:")` probe. Check that the public methods and named constant exist before connecting; do not cache success across actual handles. Use no app/config imports, file paths, pragma, helper or numeric constant fallback.
- `configure_native_close_policy(connection: sqlite3.Connection) -> None` borrows the caller's handle, calls `setconfig(sqlite3.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE, True)`, then requires `getconfig(...) is True`. It executes no SQL and never closes the borrowed handle. Task5b's opener owns its failure cleanup.
- Missing/rejected public capability maps to `ProfileRepositoryError("runtime_unsupported") from None`. Add that closed code and fixed public text: `TTS profile repository unavailable: SQLite runtime lacks required close-policy support.` Preserve bounded error serialization; do not include SQLite exception text or paths. Do not translate unrelated programming errors with an indiscriminate catch-all.
- `_worker_open` performs the probe after any already-owned cleanup and before canonicalizing/initializing a new store or acquiring new store/helper ownership. Preserve existing cleanup owners and normal error propagation. A failed probe does not latch terminal proof loss.

- [x] Add the new policy tests with a real native handle and missing-capability negative control:

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
- [x] Run `../../.venv/bin/python -m pytest -q Tests/TTS/test_profile_sqlite_policy.py` and record RED for the absent API.
- [x] Implement the leaf policy and closed error mapping. Keep configuration equivalent to the following, with explicit capability checks and the specified bounded mapping around SQLite's public capability refusals:

```python
option = sqlite3.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE
connection.setconfig(option, True)
if connection.getconfig(option) is not True:
    raise ProfileRepositoryError("runtime_unsupported")
```

- [x] Add `test_unsupported_runtime_refuses_before_store_initialization` to the lifecycle tests. Patch the repository's imported preflight function to raise `runtime_unsupported`, then open a repository under a nonexistent `tmp_path / "not-created" / "profiles.sqlite3"`. Assert that exact error, no parent/store/lease/helper creation and bounded `close()`. Restore preflight and prove a fresh owned repository can open/close, with admission counters unchanged. Run that exact node RED before inserting the probe; run GREEN afterward.
- [x] Add runtime-floor metadata tests before changing metadata. Parse TOML with `tomllib` and assert `requires-python == ">=3.12"`, no Python3.11 classifier and mypy target `3.12`. Check the named active build/preflight floors and qualification CI jobs, not every historical 3.11 string in the repository. Run `Tests/Packaging/test_python_runtime_floor.py` RED.
- [x] Account for exactly one raw call at `TTS/profile_sqlite_policy.require_native_close_policy_support` in the existing raw census. Pair the exact count/site with a source guard requiring that argument-free function's call to be `sqlite3.connect(":memory:")` with no URI, factory, alternate argument or forwarded input. Exercise accepted literal memory plus refused file, URI, variable-target and duplicate-call synthetic controls; retain every prior alias/bypass control. Document the memory-only exception beside the census, not as a fictitious file owner. The existing unrelated raw Collections call must remain a census failure until separately corrected; no general raw-memory bypass is introduced.
- [x] Update the named metadata/docs/scripts and CI contracts. Minimum package/AST qualification jobs move to 3.12. Remove the unsupported Ubuntu3.11 nightly row instead of duplicating the existing Ubuntu3.12 row. Preserve Ubuntu3.12/3.13, macOS3.12 and Windows3.12/cp1252 rows, existing triggers, concurrency, sequencing and budgets. Leave standalone backlog guard and standalone CI-shape-only Python3.11 jobs unchanged because they do not install/parse Chatbook. Do not bump `packages/tldw_profile_core`, vendored projects, backend-specific environments or dependencies; retain historical Windows3.11 evidence.
- [x] Correct the syntax guard's coupling to the old floor. Keep historical PEP701 verdicts explicitly labeled Python3.11, including an optional comparison with an actual 3.11 interpreter. The supported-floor guard must compile every shipped module on the actual declared interpreter, including when pytest itself is running on 3.12 (remove that early skip). On newer test runtimes, report missing 3.12 explicitly; minimum-runtime CI must execute the real compile check. Do not use `ast.parse(feature_version=...)` as tokenizer evidence or keep applying the legacy PEP701 detector as a 3.12 rejection rule.
- [x] Add real-floor positive and negative controls to that guard: `x = f"{ {"k": 1}["k"] }"` compiles on 3.12; `def broken(: pass` does not. Feed these through the same subprocess compile harness used by the shipped-module check. This protects the guard when accepting now-valid PEP701 source; no runtime download during tests.
- [x] Run the bounded selection below and `zsh -n scripts/preflight.sh`; syntax-check the two changed Python entry scripts without executing their build/full-suite bodies. Run static checks on the touched files and inspect the full active-runtime diff for accidental historical/vendor/dependency changes.

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

Also run the focused new memory-probe census guard and existing transition/alias/bypass controls in `Tests/DB/test_private_sqlite_inventory.py`; the whole inventory may be run diagnostically to distinguish its known unrelated failures, but cannot be reported green while they remain.

- [x] Commit only the enumerated changed files as `feat(tts): require native SQLite close-policy capability`; obtain independent spec and quality review. Document unavailable interpreter/platform evidence and existing unrelated failures. This task alone does not claim the live flag is applied or foreign-cohort finalization is fixed.

Task5a checkpoint: commits `406f1e71f` and `c007b696d` passed independent spec
and quality review. Required selection: 106 passed, one optional historical
Python3.11 skip, one existing Requests dependency warning; focused memory-census
controls: 11 passed. Controller fresh critical admission/floor run: 21 passed;
focused Ruff check, task-owned format subset and immutable diff check passed.
Four inherited formatter failures and broader legacy lint debt remain explicit;
the full inventory is not green. Two minor review notes (a non-counting missing-
method proxy close assertion and baseline qualification noise) are retained for
final review. Actual remote CI/platform execution, the optional historical
interpreter comparison and Task5b shutdown qualification are not claimed.

### Task 5b: Migrate live TTS authority, native close policy and terminal cleanup

**Depends on:** reviewed Tasks1–4 and Task5a. Resume the production shutdown gate,
not the discarded default-close approach. Read both preserved gate and native
spike reports before implementing. Implementation and scoped review are complete
through c98ebfa61; the full spawned-test qualification remains open below.

**Files:** Modify `TTS/profile_schema.py`, `TTS/profile_repository.py`, `TTS/profile_errors.py`, `DB/private_sqlite_process.py`, and `TTS/profile_migration_namespace.py` only for the required validated metadata adaptation; create `Tests/TTS/test_profile_sqlite_helper_lifecycle.py`; extend `Tests/TTS/test_profile_repository_lifecycle.py` and focused namespace comparison tests as needed. Exclusive finalizer behavior remains Task 6.

**Interfaces:**

- `_ExactCurrentProfileConnection` owns parent live SQLite, helper lease, verified directory-only FD, identity metadata and retained admission permit. It owns no original main/WAL/SHM raw FD or parent immutable evidence SQL connection.
- Consume Task5a's `configure_native_close_policy(connection: sqlite3.Connection) -> None` immediately after the exact live factory returns and before query-only/version/WAL/metadata SQL. Do not call it through the generic or shared initialization factory. The wrapper's `close() -> None` rechecks proof, closes SQLite with the flag still set, then reaps/releases helper/directory/permit ownership; it does not perform PASSIVE SQL. Repository normal cleanup owns that explicit checkpoint, so restore's existing TRUNCATE sequence is not duplicated.
- `export_restore_authority(*, deadline: OperationDeadline) -> TTSRestoreAuthority` revalidates helper and local directory, then returns immutable metadata. `verified_parent_fd(*, deadline: OperationDeadline) -> int` borrows the wrapper-owned directory FD for immediate existing tombstone settlement; callers do not close it.
- Add `ExactProfileStoreProofLostError` as an `ExactProfileStoreAuthorityError` subtype in `profile_schema.py`; distinguish terminal helper loss from a healthy-helper namespace mismatch. Add closed repository code `restart_required` with user-facing text stating restart is required, without paths.
- The process-owned admission object exposes `latch_tts_proof_loss() -> None`; subsequent retained admissions refuse. It records no database paths/connections. Existing repository/application ownership retains terminal objects and permits; it does not use a generic SQLite registry.

- [x] Write the real two-repository lock test before replacing the wrapper:

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

- [x] Run the new test and record behavioral RED. Replace local pins/evidence SQL with the fixed helper. Preserve query-only-before-admission, exact version, metadata validation, post-init binding and named/parent/sidecar checks at every existing guarded use. Pass the live opener's absolute deadline through both helper and normal private seam, using the initial reserved capacity.
- [x] Add `test_exact_live_policy_precedes_first_sql` using the actual live opener and a tracing factory proxy: record successful set/get and require it before the first execute, including `PRAGMA query_only`. Add rejected actual-handle set/get after successful memory preflight: assert no SQL, no usable publication and ordinary close/reap/release. Inject a first native close failure and assert `ExactProfileStoreCleanupError.connection` retains the complete owner for retry. An already-open but unused handle cannot be dropped merely because configuration failed.
- [x] Add `test_live_policy_does_not_change_initialization_or_immutable_evidence`: observe exact-live handles configured, with creation/migration/evidence handles still following their existing ownership/checkpoint rules. Exercise fresh store, supported legacy-schema migration, completed schema publication and two real reopen cycles with committed residual WAL. Run these focused nodes RED, apply the scoped policy, then GREEN. No factory-wide monkeypatch is the product implementation.
- [x] Replace both descriptor-field consumers explicitly. `_worker_close_for_restore` captures `TTSRestoreAuthority` before checkpoint and revalidates afterward; it installs generation-bound parent/sidecar identities before closing. `_worker_cleanup` calls `verified_parent_fd` for required reusable-tombstone settlement. Remove their permissive missing-field `getattr` branches. Preserve exact namespace removal under exclusive restore ownership; metadata export never authorizes a replacement cohort.

```python
authority = exact_connection.export_restore_authority(deadline=deadline)
self._restore_sidecar_identities = {
    "-wal": authority.wal,
    "-shm": authority.shm,
}
# Adapt ParentAuthority/internal comparison types explicitly; do not synthesize
# incomplete os.stat_result values or weaken existing exact-removal checks.
```

- [x] Add real restore-with-retained-sidecars, failed export before close, migration/restore tombstones through final close, and integer parent gid/mode/identity substitution tests. Preserve the existing healthy-helper exact-namespace restoration and close-retry behavior.
- [x] Add real healthy-cleanup tests before changing `_worker_cleanup`: pending writes are rolled back, committed writes recover after reopen, a pinned reader yields a valid partial PASSIVE checkpoint, and a live sibling writer remains excluded from an external contender after this repository closes. Observe exactly one `PRAGMA main.wal_checkpoint(PASSIVE)` on normal cleanup and none on restore's guarded TRUNCATE close path. Inject exact `sqlite_errorcode == sqlite3.SQLITE_BUSY` separately from IOERR/LOCKED/malformed results: only exact BUSY or a valid three-integer checkpoint result can continue under valid proof. A busy/partial result never authorizes unlinking remaining WAL/SHM.
- [x] Implement normal cleanup on its existing serialized worker: proof check, roll back its pending transaction, proof check, verified-directory tombstone settlement, proof check, one PASSIVE checkpoint, post-proof check, wrapper native close, then normal remaining ownership release. Preserve deadlines/progress cancellation; do not add a retry loop or alter auto-checkpoint. On non-BUSY checkpoint or close failure retain the connection/lease/helper/worker for the existing guarded retry. Update `_finish_close` so an error with a retained cleanup owner cannot shut down its executor. Keep this separate from terminal proof-loss handling.

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
- [x] Add helper-loss tests in isolated owned processes, not the main pytest process, so retained SQLite handles/workers cannot contaminate later tests. Kill only the captured helper; assert use and close yield `restart_required`, no SQL/finalizer cleanup occurs in-process, repeated new repository construction cannot acquire retained capacity, and healthy sibling work continues. Include helper loss between live SQLite open and wrapper publication.
- [x] Implement terminal handling before ordinary authority failure handling:

```python
except ExactProfileStoreProofLostError:
    self._exact_authority_quarantined = True
    self._helper_restart_required = True
    admission.latch_tts_proof_loss()
    raise ProfileRepositoryError("restart_required") from None
```

`_helper_restart_required` is initialized false on repository construction and never reset in-process. Keep references through the existing app-owned repository; do not release its SHARED lease, live SQLite, worker or retained permit. Reap the dead helper independently. Healthy-helper close failure retains the ordinary retry path. Pre-live helper failure releases ordinary ownership and does not latch. Admission latching checks existing waiters inside the reservation loop and wakes them; unrelated transient reservations and already-healthy siblings remain usable. Bound terminal owners by the existing four retained permits, not a new global connection registry.
- [x] Qualify normal application shutdown and abrupt exit separately with private namespaces and externally held observer handles. During terminal retention, exclusive store acquisition must remain blocked. After owned process exit it must succeed, with foreign substituted cohorts unchanged. If orderly interpreter finalizers close SQLite unsafely or ownership cannot be retained without hanging shutdown, STOP: report the failed approved-design gate. Do not substitute `os._exit`, force-kill product behavior, restore foreign paths in the fixture to hide failure, or silently weaken the contract.
- [x] Use an actual app-owned repository/worker for ordinary shutdown, not only a module-global sqlite handle. Seed stores in a separate child so the live parent never acquires legacy raw main/WAL/SHM descriptors. Exercise idle/read/write states, helper loss during partial publication, one/two live owners, outstanding statements and large transactions that actually spill pages. Audit whether production can expose a BLOB handle; test its real lifetime if reachable, otherwise document the concrete API boundary rather than inventing a new BLOB API.
- [x] Keep foreign-cohort preservation and original-data recovery as separate observations: compare foreign names/inodes/link counts/bytes through external observer handles after exit, then recover the original owned store in a separately owned namespace without letting fixture cleanup mutate the observed foreign cohort. Verify committed values, rollback of uncommitted/spilled writes and integrity after ordinary native finalization. Retain default-close failure only as a diagnostic control, not a deliberately failing normal-suite test. Report unsupported platform coverage explicitly.
- [x] Cancel an async waiter while its repository worker is paused at a barrier. Assert the worker retains its reservation/helper until settlement and no later operation reuses its capacity early. Verify deadline exhaustion prevents publication but permits bounded owned cleanup.
- [ ] Run the new lifecycle file plus `Tests/TTS/test_profile_repository_lifecycle.py` and `Tests/TTS/test_profile_repository.py`. Confirm all terminal tests are process-contained and normal repeated open/close cycles are leak-free.
- [x] Commit only the named TTS/process/test files with message `fix(tts): preserve remote proof authority across repository lifecycle`; obtain independent ownership/security review before final qualification.

Task5b reviewed checkpoint: implementation 42e2766e9 and fix-round1 c98ebfa61
passed independent scoped ownership/spec/quality review; both Important findings
(first-close terminal projection and late-owner worker loss) are addressed, with
no new blocking fix findings. Parent live ownership uses fixed remote proof,
pre-SQL native close policy, explicit directory/restore handoffs and bounded
terminal retention. Actual-app 14-case ordinary/abrupt shutdown gates passed,
including foreign-cohort preservation and separate committed/uncommitted recovery.
The round1 two-file run (235 passed, one existing warning, 175.59s) plus the added
residual-only case establishes 236 distinct passing cases there; controller fresh
committed six-case review regression: 6 passed, one warning, 5.81s. Touched-file formats and
changed-span lint checks pass; aggregate lint debt remains disclosed. The earlier
three-file attempt: 382 passed then failed before body at SemLock ENOSPC; 32 explicit
unaffected tail cases passed, but the failed node and 10 subsequent spawned cases
remain unqualified. The unchecked combined-run gate transfers to Task7, not a
waiver. Installed-wheel/platform proof and the remaining descriptor census stay
Task6/7 obligations. Whole TASK-31942 remains In Progress; V2 stays disabled.

### Task 6: Correct exclusive descriptor-view finalizers

**Files:** Modify `DB/private_sqlite.py`, `TTS/profile_migration_publication.py`, `profile_migration_recovery.py`, `profile_errors.py`, `profile_repository.py`, `profile_migration_candidate.py` and `profile_schema.py` for the required retained-owner propagation and schema raw-close ordering, and affected namespace ownership code; extend `Tests/DB/test_private_sqlite.py`, `Tests/DB/test_private_sqlite_inventory.py`, `Tests/TTS/test_profile_migration_publication.py`, `test_profile_migration_recovery.py`, `test_profile_repository_lifecycle.py` and `test_profile_schema.py`. Update `backlog/docs/sqlite-private-owner-inventory.md` for the required descriptor/raw-close census; unrelated owner repairs remain out of scope.

**Interfaces:** Preserve `connect_private_sqlite_descriptor(owner_id, descriptor_fd, **kwargs)` for the remaining registered exclusive owners. Borrow the verified FD during SQLite open; no `os.dup`/immediate raw close. Remove only the obsolete live `tts.profile_store_descriptor` owner after Task5b no longer uses it. Public caller-owned descriptors remain borrowed.

Add `ProfileMigrationCleanupError` in `profile_errors.py`, carrying a source-free retained `owner` whose `close() -> None` retries teardown only. Propagate that owner through existing migration/recovery callers to the repository; no broad error mapper may discard it. This is a specific operation owner, not a connection registry or permission to resume failed publication.

- [x] Add close-failure tests at publication `_immutable_validate` and recovery `_validate_authoritative_targets`, observing actual raw-close attempts with their existing owned temporary artifacts. Inject a SQLite proxy whose first close raises and whose retry closes the real connection. Assert the wrapper retains the exact SQLite connection, file and parent FDs, and no raw close/hash/rename proceeds after the failed close.

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

- [x] Run the new exact failure nodes and record RED. Use the existing `_close_profile_migration_destination` retained-owner pattern to settle SQL before raw pins. Each caller must retain cleanup authority in its existing operation/repository owner or a typed cleanup exception carrying that owner; a local variable lost on exception is not retention. Preserve control-flow exception precedence.
- [x] Borrow the original descriptor in the exclusive immutable opener, and update inventory tests. Audit every raw original-inode close in private_sqlite and TTS schema/publication/recovery/namespace/repository. Record a concrete helper-owned or tested closed/exclusive lifetime for each; an owner string, `immutable=1`, `_RECOVERY_LOCK` or absent sidecars alone does not prove exclusivity.
- [x] Verify actual repository entry points hold EXCLUSIVE ownership and have settled live handles before low-level migration/recovery publication. Test shared-sibling refusal and callbacks that fail/try to escape handles. Keep exact tombstone/content checks and refuse an unproven live consumer; do not expand helper RPC to hashing/publication or add a global connection registry.
- [x] Run the listed DB/publication/recovery/lifecycle selections; commit with message `fix(db): retain exclusive SQLite artifacts after close failure`; independent review checks the completed descriptor census and failure paths.

Task6 reviewed checkpoint: implementation `21f19e743` and fix-round1 `39cd13ba7`
passed independent task-scoped review. The original review found two Important
publication defects: rollback could continue after failed native close and lose
its owner; the direct handoff could discard earlier deferred cancellation.
Both are addressed, with no new blocking fix findings. Four durable behavioral
regressions failed before the fix and passed after it, including real multi-slot
repository initialization retaining EXCLUSIVE lease and worker until teardown.
Final fix coverage: publication/lifecycle278passed, one existing warning,88.73s;
controller fresh committed confirmation4passed, one warning,1.40s. Before the fix,
the five-file752-pass run preceded the last callback amendment; final covering
DB265passes and schema/entry164passes qualify that amendment separately.
The complete raw-close census and actual closed/exclusive entry checks are reviewed.
The file-list reconciliation includes candidate/schema mappers already required
by the all-owner propagation and raw-close audit; no new helper-copy API or source
semantics were added. Exact control-flow identity uses the existing private
cleanup handoff; ordinary indeterminate rollback policy is unchanged.
Strict inventory remains29passed/3known unrelated failures; inherited Ruff debt,
Windows/live skips and earlier11SemLock-blocked cases remain open qualification
limits. Changed-line lint, format and immutable diff checks pass. Task7 now owns
the final installed-wheel, affected-selection, performance and Canvas gates;
whole TASK-31942 remains In Progress, all final ACs unchecked, V2 disabled.

### Task 7: Qualify installed isolation, storage behavior and actual Canvas children

**Files:** Create `Tests/Packaging/test_private_sqlite_helper_distribution.py`; extend appropriate tests in `Tests/Performance/test_app_startup_performance.py`; run the existing import-weight, UI-ready census and screen-preimport budget guards without changing their ceilings; update `Docs/Canvas/V2_VERIFICATION.md`, TASK-31942 implementation notes and the preserved evidence/progress records. Modify `pyproject.toml` only if the wheel test proves a missing helper/leaf file.

**Interfaces:** No new runtime API. Consume the actual launcher, fixed operations and repository lifecycle from Tasks 1–6. Preserve V2 disabled policy while running candidate fixtures.

- [x] Add an installed-wheel test before treating checkout execution as packaging evidence. Build with existing offline packaging fixtures/toolchain into `tmp_path`, install the built artifact with no dependency resolution into an isolated temporary target, and launch the installed absolute entry with `-I -S` from an unrelated hostile cwd. Verify the entry and complete leaf closure are from that wheel, not the checkout/PYTHONPATH/user site. No shared-venv install or network fetch.

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
- [x] Run installed-wheel RED/GREEN tests and verify no imports of app/config/providers/keyring/loguru/Textual startup or user files. Confirm the approved Python3.12 minimum, native policy availability and actual macOS POSIX behavior; report unavailable OS/interpreter coverage explicitly, never as passing Windows/Linux qualification.
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
- [x] Run the exact previously failed actual-child nodes using their existing candidate fixtures:

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

**Task 7 partial checkpoint — 2026-09-08:** Test/documentation implementation
`4af790d46` and import-audit fix `73692f21c` passed task-scoped independent
review and fix-only re-review. The Important audit blind spot is corrected:
complete loaded-module names are checked against stdlib and the exact helper
closure, separately from product-file origins. Twelve test-owned module stubs
prove rejection without importing real credentials/providers. Final packaging
selection: 19 passed, one existing warning, 21.96s. File-access evidence is
bounded to resolved absolute names and hostile-root sentinels, not a complete
kernel-level audit. The UI-loop worker test passed in the earlier focused run;
its assertion-failure teardown Minor remains deferred to whole-correction review.

The exact 27-file command and exact five Canvas nodes did not collect: the
shared environment's editable `tldw_profile_core` points at an absent worktree.
The 26-file continuation returned 1,483 passed, five skipped and 36 failed
(three known unrelated inventory deltas, eleven pre-body SemLock ENOSPC cases,
22 missing-package subprocess cases). Explicit local-source diagnostics passed
26 interop tests and the five actual Canvas workflows; nine nested performance
guards still lacked the package. These diagnostics do not qualify the exact
commands or ADR-097 ceilings. No shared environment or host repair was performed.

Separately isolated five-sample benchmarks against immutable `9bc73ffb3` measured
actual UI-ready medians of 8,157.564ms current / 8,111.325ms baseline, repository
open 142.452ms / 6.428ms, and current proof rechecks 0.239ms. Resource observations
are 0.5ms-sampled high water, not exact kernel peaks; performance gates remain
unqualified. The invalid first benchmark's pre-import isolation incident and
bounded historical audit are preserved in the report and lessons. Local evidence
is macOS arm64 / Python 3.12.11 / SQLite 3.49.1 only. Whole-correction review,
remaining qualification, final ACs and Canvas V2 admission remain pending. Keep
TASK-31942 In Progress; do not change budgets or treat dependency/host/unrelated
repairs as implicitly authorized. See `Docs/Canvas/V2_VERIFICATION.md` and the
preserved Task 7 report/review/re-review for exact accounting.

**Subsequent authorized environment repair — 2026-09-08:** User approved
replacing only the stale `tldw_profile_core` editable install with a locally built
0.1.0 wheel, without dependency resolution or other dependency changes. Archived
source from `033c5a949` was built outside the checkout; fresh isolated import and
byte comparisons confirm the installed copy no longer relies on a worktree.
All 261 distribution names/versions remain unchanged. Repair artifacts and prior
metadata are preserved at `/private/tmp/tldw-core-repair.fNKBoL/`.

Fresh interop/performance selection without overrides: 56 passed, 1 failed,
4 warnings, 33.57s. Missing-package errors are cleared; the reached UI-ready
assertion measures 979/972. The unchanged guard on the immutable original
`9bc73ffb3` archive measures 973/972: a six-module correction increase over an
already one-module-over baseline. No threshold or snapshot change. Exact five
Canvas nodes now pass without overrides (5 passed, 1 warning, 134.49s); the
corresponding checklist item is complete. Whole affected-selection, budget and
final-review gates remain incomplete, with semaphore/inventory failures still
unresolved. No production-source or host cleanup was included in this repair.
See `task-7-environment-repair.md` in the preserved SDD evidence directory.

### Task 7b: Repay the startup import excess through first-use TTS construction

**Authorization and evidence:** User approved addressing the startup-import excess
without raising the limit. Unmodified warm guard measures 979/972; the original
correction baseline measures 973/972. Actual import-edge tracing identifies the
repository implementation as a bounded removable closure; the common DB helper
and identifier-validation leaves have other startup consumers and must remain.
Read `task-7b-diagnosis.md` in this plan's SDD workspace for the edge evidence.

ADR required: no new ADR
ADR path: backlog/decisions/097-boot-budget-ratchets.md; backlog/decisions/028-character-tts-generation-profile-ownership.md; backlog/decisions/125-lock-safe-private-sqlite-validation.md
Reason: defer pure construction through the existing app-owned first-use seam;
preserve one owner, captured path, serialized lifecycle and all SQLite admission
and finalization contracts. The ADR-028 clarification records construction timing.

**Controller ruling:** Replace the test-pinned eager pure repository construction
with eager app-owned lifecycle state and first-use construction. The repository
remains app-scoped, not a singleton, proxy, property or new registry. Capture its
configured path during app construction as before; do not re-read configuration
on first use. Closing an unused app permanently prevents later construction.
The cost if this ruling is wrong is localized lifecycle/test rework; it does not
authorize changing native SQLite policy, Canvas ownership or runtime admission.

**Files:** Modify only `tldw_chatbook/TTS/__init__.py`, `tldw_chatbook/app.py`,
`tldw_chatbook/UI/stts_profile_library.py`,
`Tests/TTS/test_tts_app_ownership.py`, and the actual-app fixture in
`Tests/TTS/test_profile_sqlite_helper_lifecycle.py`. Add
`Tests/Packaging/test_tts_profile_repository_import_closure.py`. Root owns this
plan, ADR clarification, Backlog and verification documentation. Ask the root
before extending production scope. Do not modify low-level repository, schema,
migration, proof/helper modules, budget constants or snapshots.

- [x] Write behavioral RED tests before production edits: ordinary app construction
  neither imports nor constructs the repository implementation; package explicit
  export still resolves to the real class; first ensure constructs one owner at
  the captured path, with concurrent callers sharing one open task.
- [x] Reuse the package's existing PEP 562 export pattern and app's existing
  `_ensure_tts_profile_repository` seam. Construct before the first await so one
  event-loop owner is installed. Preserve existing injected test owners, open
  retry, cancellation shielding, service sharing and bounded source-free errors.
  A close request must latch before awaiting and also when no owner exists, so
  close-before-use and ensure/close races cannot create or reopen an owner.
- [x] Repay the measured screen-preimport shift too: the Personas route reaches
  `UI/stts_profile_library.py`, whose four voice-bundle imports load the repository
  through the existing package deferred exports. Keep annotation-only exports in
  `TYPE_CHECKING` and import the real `TTSVoiceBundleImportChoice` locally in the
  existing `voice_bundle_import_choice` user-decision helper. Preserve the class
  identity, validation and UI behavior; do not change the service or repository
  implementation. Extend the new isolated import-closure test to cover the real
  Personas/profile-library route and first decision-helper use, and run existing
  `Tests/UI/test_stts_profile_library.py -k bundle` coverage. The observed interim
  516/500 preload failure is not acceptable startup repayment.
- [x] Replace the eager-construction assertion with genuine first-use tests,
  including close-before-use, concurrent ensure, canceled waiter, ensure/close
  race and idempotent close. Retain existing real lifecycle assertions. Do not
  add a production API merely to instrument tests.
  If the existing authority-order fixture lacks already-required app shutdown
  hooks, add explicit recording doubles and their exact expected positions in
  that same test. Do not bypass missing hooks or change production shutdown.
- [x] Adapt the actual-app exit child by setting its owned path before app
  construction and instrumenting the real repository factory at its existing
  module seam if pre-open access is required. The app itself must still perform
  first-use construction; do not substitute a fake repository or skip any native
  ordinary/abrupt exit, foreign-cohort preservation or original-data recovery gate.
- [x] Run focused RED/GREEN nodes, then the complete targeted files:
  `Tests/TTS/test_tts_app_ownership.py`,
  `Tests/TTS/test_profile_sqlite_helper_lifecycle.py`,
  `Tests/TTS/test_profile_repository_lifecycle.py`, the new import-closure file,
  `Tests/test_probe_import_provenance.py`,
  `Tests/Performance/test_app_startup_performance.py`,
  `Tests/Performance/test_app_import_weight.py`,
  `Tests/Performance/test_ui_ready_module_census.py`,
  `Tests/Performance/test_screen_preimport_payload_budget.py`, and
  `Tests/Canvas/test_startup_deferral.py`. Keep the unchanged warm limit at 972.
  Every app import/probe must use collected `Tests.conftest` isolation and owned
  HOME/XDG/config/database paths. No full sweep or real user-config imports.
- [x] Run `git diff --check` and Ruff check/format-check on changed Python files;
  distinguish inherited debt from introduced errors without whole-file formatting.
  Report exact commands/results, census values, all warnings/failures, provenance,
  self-review and scope. Commit only named implementation/test files, leaving
  root-owned governance unstaged. Do not repair shared dependencies or host
  semaphores, change unrelated inventory tests, or enable Canvas V2.
- [x] Independent task-scoped spec and quality review.
- [x] Complete the existing whole-correction review against original `9bc73ffb3`.
  This local fix
  does not waive remaining qualification gaps or mark TASK-31942 Done.

**Task7b checkpoint:** Implemented in `bd96a923c4` and `ae629ed080`, with independent
spec compliance and quality approval and no Critical/Important findings. Scoped
groups passed44 ownership,246 native/helper/repository/import,59 bundle UI and50
final performance/provenance/Canvas tests. The annotation-only follow-up passed
its two covering tests. Final import625/660, UI963/972, preload499/500 modules,
364325/378740 LOC and110163/123319 largest-route LOC; thresholds and snapshots
unchanged. Root's fresh committed five-test check independently confirmed the
UI/preload counts and actual idle/partial orderly native exit preservation.
The initial516/500 cost transfer and inherited fixture failures remain documented,
not relabeled as passing evidence. Changed-span lint is clean; aggregate474
inherited findings, four formatter-dirty files and environment warnings remain.
Review cannot-verify items are the existing cross-task/native/platform/Canvas
qualification gates, not waived by this local approval. All seven final task ACs
remain unchecked and V2 stays disabled. See `Docs/Canvas/V2_VERIFICATION.md` and
the preserved Task7b report/review/controller verification.

### Task 8: Apply the single final-review fix wave (SQLite correction)

This is the SQLite correction's final-review wave, not Canvas's separate Task8
admission workflow. BASE is `41f144ab900c9937a25e99717b9fd6c48d4b2942`.
Read the final review's exact I1, M1 and M2 findings in
`.superpowers/sdd/2026-09-07-sqlite-lock-safe-private-validation-implementation/final-review.md`
and the preserved `final-review-probe.py.txt` / `final-review-probe-output.md`.
All three findings belong to one implementation dispatch and one scoped re-review.
Q1 and the unrelated inventory/host/platform gates are not part of this fix wave.

ADR required: no new ADR
ADR path: backlog/decisions/125-lock-safe-private-sqlite-validation.md
Reason: complete the already-approved control-flow and retained-live-owner
contract; no new storage, native policy, authentication or runtime boundary.

**Controller ruling:** I1 is an in-scope incomplete correction even though its
exception-replacement pattern predates this branch. The approved cleanup contract
preserves control-flow precedence and complete ownership; substantial rewriting
of this live opener must satisfy both. Fix it without silently expanding the
migration carrier into live ownership or losing reservation/lease classification.
Cost if wrong: localized live-opener/caller/exception handoff rework. This does not
authorize broader policy changes, forced cleanup, or qualification waivers.

**Permitted production files:** `tldw_chatbook/TTS/profile_schema.py`,
`tldw_chatbook/TTS/profile_repository.py`, and, only if required to reuse the
existing guarded carrier mechanism, `tldw_chatbook/TTS/profile_errors.py`.
**Permitted test files:** `Tests/TTS/test_profile_sqlite_helper_lifecycle.py`,
`Tests/TTS/test_profile_repository_lifecycle.py`,
`Tests/TTS/test_profile_sqlite_policy.py`, and
`Tests/Performance/test_app_startup_performance.py`.
Root owns all plan/ADR/Backlog/evidence documentation. Ask root before extending
these files or changing an established interface; no new production registry,
helper protocol, worker service or public failure code.

Binding constraints for this wave:

- Control-flow body errors remain primary; when there is no earlier control-flow
  signal, retain a cleanup control-flow signal rather than converting it into an
  ordinary failure. Preserve exact object identity and existing earlier-owner
  metadata/hostile attribute-hook protections where a carrier is needed.
  A private live carrier may use guarded exception-local history plus a current
  attempt entry: preserve prior carriers by identity but never adopt stale prior
  ownership into the current repository. The current entry must cover healthy
  cleanup and pre-live/early failure too. Do not close or transfer earlier owners,
  change migration metadata, or introduce a global registry.
- The live opener must settle the transient child and hand off healthy retained
  proof before raising out of its reservation. The repository must adopt the
  complete live wrapper with its SHARED store lease and retained worker. A failed
  native close remains retryable cleanup, not terminal proof loss; a teardown
  retry does not replay validation/initialization or new user work.
- Keep the live handle's native flag enabled through finalization. Healthy
  serialized cleanup rolls back pending work, revalidates each phase, settles
  tombstones and attempts one PASSIVE checkpoint. Valid partial/exact BUSY leaves
  WAL intact; other errors retain the cleanup owner and worker for guarded retry.
  Restore keeps its separate strict checkpoint. Preserve terminal proof-loss
  quarantine and restart-required admission behavior; do not remint proof or
  force-close an unsafe handle.
- Existing fixed helper limits remain: eight helpers (four retained, four
  transient), five-second general waits, 30-second initial TTS validation, one
  second graceful close plus a further two-second terminate/kill/reap bound;
  metadata1M rows,576MiB artifacts,64KiB closed JSON frames. No numeric native
  fallback, raw parent database/sidecar inspection FD, BLOB evidence, or new import
  dependency. Python>=3.12 and the public pre-SQL native admission remain.
- Only collected repository pytest isolation and owned temporary resources. No
  ad hoc app imports before Tests.conftest, real user data/config, shared package
  changes, host cleanup, full suite, PR/push/rebase/merge or V2 enablement.

- [x] Reproduce I1 with committed regression coverage before production edits.
  Use the review probe as evidence, not a test with an unadapted post-fix owner
  lookup. Cover body cancellation/control-flow plus ordinary close failure,
  cleanup-originated control flow and earlier-signal precedence; assert exact
  signal identity, native flag/proof/charged retained ownership, no premature
  helper reap, and safe eventual teardown of the same owner.
- [x] Correct the opener and actual repository adoption together. Add real
  repository-level coverage that reaches the live-opening failure (not a failed
  capability probe), retains its lease/worker/owner, and performs teardown-only
  retry. Preserve ordinary-error, healthy-close, terminal-proof-loss and existing
  cancellation behavior. No production testing hooks or blanket exception skips.
- [x] Address M1 with an observable borrowed-handle close check in both
  missing-method cases. Address M2 by releasing the barrier, joining the open
  task and attempting repository cleanup on assertion/open failure with nested
  cleanup that preserves the primary failure. Demonstrate sensitivity to the
  asserted bad behavior without modifying production policy or adding a broad
  test framework. Keep test-only utilities in tests.
- [x] Run focused RED/GREEN nodes, then the complete four scoped test files above,
  plus `Tests/Packaging/test_private_sqlite_helper_distribution.py`,
  `Tests/Packaging/test_tts_profile_repository_import_closure.py`,
  `Tests/Performance/test_app_import_weight.py`,
  `Tests/Performance/test_ui_ready_module_census.py`,
  `Tests/Performance/test_screen_preimport_payload_budget.py`, and
  `Tests/Canvas/test_startup_deferral.py`. Preserve unchanged budgets
  import660,UI972,preload500 modules,378740 totalLOC and123319 largest-routeLOC.
  If `profile_errors.py` changes, also run its carrier tests in
  `Tests/TTS/test_profile_migration_publication.py` and
  `Tests/TTS/test_profile_migration_recovery.py` to protect the shared metadata
  boundary. Report existing failures explicitly; no unrelated repairs.
- [x] Run diff-check and scoped Ruff/format checks, comparing introduced spans
  against immutable BASE rather than only aggregate counts. Do not reformat
  legacy files wholesale. Self-review, commit only named production/test files,
  and write exact commands, RED/GREEN results, scope, static limits, retained-owner
  accounting and any concerns to this SDD directory's `task-8-report.md`.
- [x] One fix-only re-review against this wave's immutable diff, with a verdict on
  I1/M1/M2 and new breakage in the fix only. Root adjudicates residuals per the
  final-review workflow; no second automatic fix wave. Remaining qualification
  gaps still block TASK-31942 completion and Canvas V2 admission.

**Task8 checkpoint:** Commit `3b5031012c` addresses I1/M1/M2 in six scoped files.
The independent fix-only re-review confirms all three addressed and no new
Critical/Important issue. Covering selection: 343 passed, 5 warnings, 260.93s;
final focused: 22 passed, 1 warning, 5.51s, including the later-added terminal-control
parameter. Controller committed check: 8 passed, 1 warning, 1.98s, covering exact control,
retained-owner retries, both guard sensitivities and terminal classification.
No runtime behavior changed after covering collection; other concurrent edits
were static-only. All six files pass format checks; BASE-mapped lint reports
zero introduced and 128 inherited diagnostics. Startup ceilings and passing
counts 625/660, 963/972, 499/500 remain unchanged. Exact evidence and freshness are
recorded in `Docs/Canvas/V2_VERIFICATION.md` and the preserved SDD reports.

**Controller residual ruling:** The private current-carrier slot guarantees the
approved sequential signal-reuse cases, not simultaneous reuse of one exception
instance across workers. No concrete shared-signal runtime producer was found.
Retain the review's concurrency observation as a documented limitation rather
than inventing a new contract or running another final fix wave. If a producer
later needs concurrent same-object reuse, an attempt-bound carrier and coverage
must precede it. Cost if wrong: localized carrier/adoption rework; a missed
existing producer could associate the wrong retained owner with its repository.

This closes the concrete review wave, not the full correction. Q1 remains:
three unrelated strict-inventory failures, eleven pre-body SemLock ENOSPC cases,
platform/optional qualification and final affected-selection/benchmark evidence.
TASK-31942 stays In Progress with all seven final ACs unchecked, and Canvas V2
remains disabled. No host cleanup, unrelated repair, dependency change, full
suite or external PR action was authorized or performed by this wave.

### Task 9: Reconcile the three authorized inventory failures

This is the user's separately approved Q1 continuation, not another automatic
Task8 final-review fix wave. Immutable BASE is
`50a57004220645665f528edad31f7a31f9a608f2`. The read-only diagnosis reproduced the
three failures (3 failed, 1 warning, 30.35s) and is preserved at this plan's SDD
directory in `task-9-diagnosis.md`. Preserve reviewed Tasks1–8 and their evidence.

ADR required: no new ADR
ADR path: backlog/decisions/029-local-private-data-boundary.md;
backlog/decisions/113-collections-capture-authority-and-legacy-boundary.md;
backlog/decisions/125-lock-safe-private-sqlite-validation.md
Reason: restore the existing module-owned, no-follow, read-only and quiescence
contracts; no new runtime, schema, security or backup authority.

**Permitted implementation files:**

- `tldw_chatbook/DB/private_sqlite.py` (two owner registry entries only).
- `tldw_chatbook/Library/collections_legacy_recovery.py` (path retention,
  checked opening, transaction setup and owned cleanup only).
- `tldw_chatbook/Chat/console_trace_maintenance.py` (owner literal only).
- `Tests/DB/test_private_sqlite_inventory.py`.
- `Tests/DB/test_chachanotes_connection_quiescence.py`.
- `Tests/Library/test_collections_legacy_recovery.py`.
- `backlog/docs/sqlite-private-owner-inventory.md`.

Root owns all other governance/evidence files. Do not modify production
`base_db.py`, the helper protocol, native close policy, prior raw-call exceptions,
or backup semantics. Off-current-history commits `de3cd120e2` and `fd26ea5238`
corroborate the diagnosis but are not current evidence and must not be
cherry-picked wholesale. Implement from the current failing tests and contracts.

Binding interfaces and limits:

- Register `library.legacy_recovery` to its exact module with only
  `_READ_ONLY_URI`, source-mode preservation and no backup authority. Keep the
  original absolute lexical Path so the existing no-follow seam can reject leaf
  and parent aliases. Open with `read_only=True, must_exist=True`; preserve the
  live `LibraryCollectionsDB` branch and schema-independent recovery. Opening
  OSError/ValueError/SQLite failures map to the existing path-free
  `legacy_database_unavailable` error. Setup belongs inside owned cleanup;
  rollback failure must not skip close. No migration or writable fallback.
- Register `chat.trace_maintenance` to its exact module with only `_PRIVATE_FILE`
  and no backup authority. Change only the existing maintenance call's owner ID;
  preserve its separate connection, target, options, PRAGMAs and quiescence flow.
- Append C55 (trace) and C56 (legacy); retain retired C10/C48. There must be 54
  connection rows through C56 excluding those IDs. B rows remain unchanged.
- Qualify backup calls with exact `(module, qualified symbol, receiver)` Counter
  entries and multiplicity: one `private_sqlite._backup_pages` on `source`, and
  one `base_db._QuiescentSQLiteConnection.backup` on zero-argument `super()`.
  This is delegation of the same checked backup, not another backup owner.
  Reject extra, duplicate, moved-symbol, other-module and changed-receiver calls.
  Preserve Task6's helper and raw-connection census protections.
- Use collected pytest isolation and owned temporary resources. No real user
  config/data, shared dependency changes, host cleanup, process termination,
  semaphore unlinking, full suite, external PR action or Canvas V2 admission.
  Do not repeat the eleven blocked spawned cases while the independent stdlib
  allocation control fails. Read-only host diagnosis is recorded separately in
  `Docs/superpowers/reviews/2026-09-08-semaphore-allocation-diagnosis.md`.

- [x] Add regression tests first and record meaningful RED before production
  edits. Existing three RED nodes protect owner/census drift. Cover read-only
  enforcement with query_only disabled, unchanged source bytes/mode, constructor
  leaf/parent aliases, post-construction missing/alias/shared-parent substitution,
  path-free errors and real connection closure after setup failure. Keep the
  existing future-schema recovery/export tests. For the unchanged backup wrapper,
  add real in-memory success/callback-abort reservation coverage and demonstrate
  scanner sensitivity without inventing a production behavior change.
- [x] Apply only the scoped corrections, update the inventory narrative and
  self-review current C48 retirement and cross-module ownership boundaries.
- [x] Run the complete targeted selection: `Tests/DB/test_private_sqlite_inventory.py`,
  `Tests/DB/test_private_sqlite.py`, `Tests/DB/test_core_sqlite_owner_privacy.py`,
  `Tests/DB/test_chachanotes_connection_quiescence.py`,
  `Tests/Library/test_collections_legacy_recovery.py`,
  `Tests/Chat/test_console_trace_compaction.py`, and
  `Tests/Chat/test_console_trace_compaction_admission.py`. Then rerun the exact
  three originally failing nodes. Report failures without unrelated repairs.
- [x] Check scoped format, BASE-mapped introduced Ruff diagnostics and diff
  whitespace. Do not wholesale-format inherited files. Commit only the seven
  named files with serialized index ownership. Record exact commands/results,
  RED/GREEN, behavioral sensitivity, cleanup and residuals in `task-9-report.md`.
- [x] Independent task-scoped spec/quality review of the immutable Task9 diff,
  bounded fixes if required, then root committed smoke of the three original
  failures and critical privacy/backup regressions. Refresh the unchanged startup
  import/UI-ready/preload guards for the registry/import additions. Root records
  inventory evidence and host diagnosis, leaving overall qualification open.

**Host diagnosis checkpoint:** Independent stdlib `spawn.Lock()` fails with
errno28 inside and outside the sandbox, before any Chatbook import. The host's
named-semaphore maximum is 10000; only 48 open semaphore handles were visible,
which does not measure cached names or identify their creators. No host resources
were changed. A clean qualification host or a user-coordinated restart is needed
before rerunning the allocation control and, only if it passes, the eleven cases.
No cleanup or restart is authorized by this diagnosis.

**Task9 checkpoint:** Commit `a6388ed5c4` changes exactly the seven permitted
files. Independent review approves spec compliance and quality, with no
Critical/Important finding. The two review Minors are inherited Requests warning
and static debt; the root accepts that classification from the BASE comparison
and does not broaden this task into dependencies or whole-file formatting.
Original three inventory nodes pass; root committed smoke8passed/1warning40.54s
and unchanged startup3passed/4warnings12.70s. Budgets remain625/660,963/972,
499/500 modules and364325/378740,110163/123319 LOC. Covering selection is explicitly
not all-green:426passed/2skipped/26failed75.11s. Root paused the commit to verify
attribution, then an import-verified immutable BASE archive reproduced all26
assertions (26failed/1warning4.95s). Those22core-owner and4compaction failures
remain qualification gaps, not Task9 regressions or repair authorization.
Current/BASE Ruff both58inherited diagnostics and one formatter-dirty file;
zero introduced diagnostics, whitespace clean. Exact evidence is in Task9's
preserved implementation/review/root reports and `Docs/Canvas/V2_VERIFICATION.md`.
AC8/9 are checked for this scoped continuation; original seven ACs remain
unchecked, Task31942 In Progress and V2 disabled. No full suite, host/dependency
change or external action. The clean-host, platform and final qualification gates
still require further work; do not restart already-reviewed implementation or
the prior review. Resume only the open qualification work after direction.

### Task 10: Repair baseline owner tests and maintenance Canvas validation

**Authorization and scope:** The user approved this bounded repair after the
recorded causal diagnosis. BASE is `e1a2e859127eb83a100b22387e47a8ffbc95d424`.
Work in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/canvas-v1`
on `codex/canvas-v2-mermaid-design`. Preserve reviewed Tasks1–9, the dirty root
governance documents, and all ignored evidence. One implementation agent owns
this combined repair; no parallel implementation or delegated subagents.

ADR required: no
ADR path: existing backlog/decisions/029-local-private-data-boundary.md;
backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md;
backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md;
backlog/decisions/125-lock-safe-private-sqlite-validation.md
Reason: Restore tests to the approved helper boundary and reuse the existing
pure Canvas integrity function during existing same-file maintenance. No new
storage, runtime, authentication or mutation-authority contract is introduced.

**Read first:** TASK-31942 and the preserved `task-10-owner-diagnosis.md` and
`task-10-compaction-diagnosis.md` in this plan's SDD directory. The retained
`task-10-compaction-probe.py` is diagnostic evidence, not a production test.
Relevant lessons cover collected pre-import isolation, selective path guards,
real close versus wrong-thread errors, non-reentrant progress callbacks and
same-file/cursor quiescence. Read applicable repository/skill instructions.

**Only product/test files permitted:**

- `Tests/DB/test_core_sqlite_owner_privacy.py`
- `tldw_chatbook/DB/ChaChaNotes_DB.py`
- `tldw_chatbook/Chat/console_trace_maintenance.py`
- `Tests/Chat/test_console_trace_compaction.py`

**Interfaces and invariants:**

- Owner repair is test-only. Preserve exact kwargs equality, including the
  existing required `base_db._QuiescentSQLiteConnection` factory. Selective
  `Path.resolve` guards reject selected database paths and both backup source
  and target, while delegating unrelated fixed helper-entry resolution to the
  real method. Retain actual SQLite/helper calls, private modes, parent modes,
  exact owner/source/target observations, copied rows and cleanup.
- Relative-path tests observe and delegate the live
  `prepare_in_helper(PrepareRequest(...))` seam. Assert the exact lexical absolute
  path, writable/create-if-missing true and preserve-source-mode false. Forward
  the actual received owner ID unchanged; never repair arguments in a recorder.
  Assert actual owner ID and the raw SQLite lexical target. Do not restore local
  original-inode inspection or weaken registered policies to satisfy old spies.
- Add one narrow shared internal installer for the existing deterministic
  three-argument `canvas_revision_payload_valid` implementation, using it on
  ordinary ChaChaNotes and every dedicated maintenance connection. Do not copy
  validation logic or introduce a generalized initializer/registry. If maintenance
  setup fails after acquiring a native connection, attempt owned close and
  re-raise the original setup failure; cleanup must not mask its control flow.
- Install only the pure validator for maintenance: no semantic mutation guard,
  logical-GC permission, Canvas deletion grant or authorizer expansion. Keep the
  schema/triggers, bounded public errors, maintenance lease, dedicated worker,
  dispatch pause, same-file/cursor quiescence, TRUNCATE checkpoint, preflight,
  cancellation, retry state, reopen and integrity checks unchanged. No payloads
  or raw SQLite exceptions in public diagnostics.

- [x] Establish focused RED before edits. Preserve the recorded 22-case drift
  evidence and show meaningful failures for new compaction regressions before
  production changes. Use collected pytest isolation and owned temporary data.
- [x] Correct the owner tests with behavioral sensitivity checks for missing or
  wrong factory/owner, selected-path resolution, absent/wrong helper request
  path/policy and backup bypass. Prefer scoped negative controls of the actual
  tests over a new test framework; record mutations and restoration in the report.
- [x] Add real populated Canvas revision preservation through same-file VACUUM
  and reopen, including HTML/digest/byte count and lineage. Demonstrate invalid
  UTF-8/digest/size payload rejection through the maintenance connection and
  unchanged refusal of unauthorized mutation/deletion with rows retained. Inject
  setup-registration failure, prove the acquired real handle is closed (not
  merely a wrong-thread ProgrammingError) and maintenance exclusion releases so
  ordinary acquisition can resume. Apply the narrow shared installer/cleanup fix.
- [x] Run the covering targeted selection once: core-owner privacy, private_sqlite,
  private_sqlite_inventory, chachanotes_connection_quiescence, trace_compaction,
  trace_compaction_admission and the existing Canvas persistence/repository test
  file(s) directly covering the validator. Run only focused cases while iterating.
  Explicitly account for all 22 owner parameters and the four diagnosed physical
  compaction/admission nodes. Use `../../.venv/bin/python`; no full repository
  suite, eleven host-blocked spawned-case reruns, dependency or host changes.
- [x] Run changed-file format and BASE-mapped Ruff checks plus `git diff --check`.
  Preserve unrelated inherited static debt; introduce none and do not wholesale
  format legacy modules. Self-review and commit only the four named files using
  serialized index ownership. Leave root governance edits unstaged. Write exact
  commands, RED/GREEN, negative sensitivity controls, cleanup, residuals and commit
  in `task-10-report.md` in the existing plan SDD directory.
- [x] Independent task-scoped spec and quality review of immutable BASE-to-HEAD
  package, bounded fixes if needed, then root committed smoke of the diagnosed
  groups and fresh unchanged startup/import/UI-ready/preload guards. Tests remain
  serialized. Record evidence and AC10/11 only if met; TASK-31942 stays In Progress
  while host/platform/final qualification remains open. No PR/push/rebase/merge,
  user resource cleanup/restart, or Canvas V2 enablement.

**Task10 checkpoint:** `ad5c02e5a8` contains exactly the four permitted files.
Independent spec/quality review approves with no Critical/Important findings.
One new Minor remains deferred: the setup-failure test should put its database
lifetime in `try/finally` so assertion failure also closes its test-owned registry.
The second Minor retains inherited Requests/static noise. Both are recorded for
the remaining qualification handoff; no additional repair wave is implied.

Original owner22 and new compaction6 each demonstrated RED then GREEN; the four
original compaction/admission cases pass. Final implementation115passed/1warning
22.69s; root committed31passed/1warning9.30s. Covering479passed/2skipped/2failed
146.88s remains recorded as non-green: the two gateway cases reproduce sandbox
loopback EPERM and pass with required permission (2passed/1warning1.48s).
Root unchanged startup3passed/4warnings16.22s; counts625/660,963/972,499/500 and
364325/378740,110163/123319 LOC. Aggregate599Ruff/threeformatter-dirty files remain;
no added-range diagnostic identified, with the multiset-comparison limitation
and exact BASE import controls documented. No budget, schema, native/helper,
privacy or mutation-authority change. AC10/11 are checked; original seven remain
open. Memory bypass is confirmed in the unchanged seam and covering memory tests;
Windows skips and host/platform/final gates remain unqualified. Task31942 remains
In Progress, V2 off; no full suite, host/dependency or external action.

### Task 11: Close the deferred compaction-test teardown gap

**Scope:** User's continuation resumes remaining qualification after Task10.
BASE `f2faa5d65cbcb10cabecf7a5bf8daa45da475ae4`; existing worktree
`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/canvas-v1`, branch
`codex/canvas-v2-mermaid-design`. Fix only the recorded Task10 teardown Minor;
do not restart Task10 implementation or whole-correction review.

ADR required: no
ADR path: N/A (test-only teardown; existing ADR-125 runtime contracts unchanged)
Reason: No product code, storage, API, dependency or authority change.

**Only tracked edit/commit allowed:**
`Tests/Chat/test_console_trace_compaction.py::test_maintenance_setup_failure_closes_handle_and_releases_exclusion`.
Read TASK-31942 and `task-10-review.md` in this plan's preserved SDD workspace.
Root owns plan/Backlog/evidence docs; leave them unstaged. No subagents.

- [x] Before the edit, run the existing case with `../../.venv/bin/python -m pytest
  Tests/Chat/test_console_trace_compaction.py::test_maintenance_setup_failure_closes_handle_and_releases_exclusion
  -q --tb=short --show-capture=no`. Preserve its passing functional baseline.
- [x] Prove the review's failure-path cleanup gap using one collected, test-owned
  probe in this SDD directory (load `Tests.conftest` before app imports). Invoke
  the real test with its real temporary database; capture the constructed database
  and original bound `registered_connection_count` method. Force only the late
  count assertion to fail by overriding that instance's count accessor. Catch
  that AssertionError, then use the original method to assert the registry is
  empty. The control fails before the edit and passes afterward. The probe's own
  `finally` must close the captured database even when its assertion fails. No
  permanent meta-test, new product hook, raw user-config import or source-string
  assertion is needed. Preserve the probe as evidence, not collected shipping code.
- [x] Immediately after successful `CharactersRAGDB(...)` construction, wrap all
  remaining existing test setup, monkeypatching, quiescence and assertions in
  `try`. Remove the trailing success-only close and use exactly:

```python
finally:
    database.close_connection()
```

  Preserve every existing assertion, exception matcher, timeout and operation;
  do not change other tests or introduce cleanup helpers/product methods.
- [x] Rerun the failure-path probe and the complete
  `Tests/Chat/test_console_trace_compaction.py` and
  `Tests/Chat/test_console_trace_compaction_admission.py` selection once. Run
  changed-file Ruff/format checks plus `git diff --check`; retain inherited
  formatting debt, do not wholesale-format. Self-review the exact indentation-only
  lifetime change, then commit only the named test file with an empty verified
  index before/after. Preserve all temporary evidence; do not clean archives.
- [x] Write a concise `task-11-report.md` with exact baseline/RED/GREEN commands,
  results, static debt, commit and file scope. Independent task-scoped review
  follows before root qualification proceeds. No full suite, eleven host-blocked
  cases/control repeat without state change, host/dependency action, external
  PR/push/rebase/merge or V2 enablement.

Task11 completed in `55f74aa009` with independent spec/quality approval, no
Critical/Important findings. Meaningful failure-path RED/GREEN and covering
20passes; root committed control and original target each pass independently.
Changed-file Ruff and changed-function format pass; inherited aggregate static
debt and Requests warning remain disclosed. AC12 checked; no new ADR. Evidence:
`Docs/superpowers/reviews/2026-09-08-sqlite-final-local-qualification.md`.
The test-teardown Minor is resolved; resume remaining Task7 qualification only.

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
| Exclusive descriptor finalizers and complete consumer inventory | 6, 9 |
| Real lock oracles, packaging, performance and actual Canvas regressions | 3, 5b, 7 |
| Helper-aware owner tests and pure Canvas integrity support during maintenance | 10 |

Plan self-review checks interfaces across tasks, all approved spec sections, exact existing test names, bounded errors and unchecked work status. Tasks1–4, Task5a, Task5b and Task6 implementation have passed independent review; do not restart them. Task7 test implementation and its import-audit correction passed task-scoped review, but final qualification is incomplete. Runtime admission, live ownership, the macOS actual-app shutdown gates and the exact five Canvas nodes have passing scoped evidence; the full correction remains unqualified. Preserve the historical editable-install failures alongside the explicitly authorized repair. Task7b closes the measured startup-budget breach without changing its ceilings. The whole-correction review and single SQLite Task8 fix-only re-review are complete; I1/M1/M2 are addressed. Separately authorized Task9 closes the three strict-inventory gaps with independent approval and diagnoses the host without mutation. Task10 closes the 26 BASE failures with scoped review and committed verification. Task11 now resolves its test-teardown Minor with independent approval. Fresh affected-selection and checkout benchmarks have completed with the explicit limits in the final-local-qualification report: 1844 affected tests pass, but the five-child Canvas run has one unresolved failure, an exact rerun fails earlier, and aggregate static checks remain nonzero. Eleven semaphore-blocked cases and platform/optional gaps remain. No failed, skipped or deselected gate is passing evidence or permission for host cleanup/unrelated repairs. The user-approved source-free readiness/action spike has now reproduced an empty-card synthetic action before pending UI sync finishes; both temporary test edits were restored exactly. Evidence: Docs/superpowers/reviews/2026-09-08-canvas-card-readiness-spike.md. The separate startup failure remains open and a retained harness readiness correction requires approval; no production fix is underway.
