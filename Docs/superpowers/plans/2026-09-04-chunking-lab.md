# Chunking Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users locally preview, compare, and save full chunking recipes while automatically recovering their single-sample A/B experiment.

**Architecture:** Extend ADR-078's runtime and template service, with a separate Lab capability gate and immutable execution reports. A headless session coordinator owns drafts, one bounded preview process, and a serialized private SQLite recovery writer. A thin Library-owned Textual screen composes editing and inspection regions.

**Tech Stack:** Python ≥3.11, Textual 8.x (≥8.0.0,<9), existing Pydantic and SQLite/private-path helpers, stdlib process/JSON/hash facilities, pytest/Hypothesis/Textual Pilot. No new runtime dependency.

**Spec:** [Reviewed Chunking Lab design](../specs/2026-09-04-chunking-lab-design.md).

ADR required: yes

ADR path: [backlog/decisions/118-chunking-lab-local-execution-and-recovery.md](../../../backlog/decisions/118-chunking-lab-local-execution-and-recovery.md)

Reason: durable recovery, execution and save contracts, privacy, and long-lived UI ownership; extends ADR-003/029/073/078 without replacing canonical template convergence.

## Global Constraints

- v1: one sample with one configuration or lightweight comparison of two.
- v1 execution and template saving are local. Every result records its backend.
- Controls plus JSON; preserve advanced data, raw invalid JSON, and incomplete control strings.
- Use ADR-078's flat body and existing Media DB catalog. No legacy pipeline/inheritance implementation, file store, or Media DB schema migration.
- Keep the server-parity validator and vendored `Chunking/engine/` algorithms unchanged. Strict Lab capability validation is a separate named gate.
- No implicit network requests, model/tokenizer downloads, LLM operations, or source mutation.
- Limits: 2 MiB UTF-8 sample text, 10,000 output chunks, 32 MiB serialized output per result, and 60 seconds per preview.
- Autosave: 300 ms trailing debounce, maximum one-second checkpoint interval during continuous editing under normal storage conditions. Immediate checkpoints for critical transitions.
- Saved locally means the latest revision committed. A sudden crash can lose subsequent uncommitted keystrokes.
- Existing private-profile protections apply; no new encryption or secure-erasure claim.
- Verify at 80x24, 120x40, and 160x50; preserve F6/F1 and text-editor keys. Local r/p/s act only outside editors. No Ctrl-key shadows.
- No v2 scheduler, server adapter, corpus evaluator, named experiment library, or inert Evals action.
- Work that can exceed 100 ms, including preflight, hashing, serialization, and file/DB reads, runs off Textual's event loop. Draft edits reuse immutable blob references, not full-result copies.
- Targeted tests only unless the user explicitly requests a full suite; live verification uses an isolated profile.

## Execution baseline and task discipline

The planning checkout is older than completed chunking convergence. Read-only source
inspection used `origin/dev` at `91757b61e9c7e9f920d80a0ce282261b4161ffff`.
Start execution in an isolated worktree based on current dev, using the worktree
skill. Verify ADR-078 and TASK-19801–19806 are present; do not implement replacement
infrastructure against this checkout's deleted-in-dev files. Revalidate call sites
and line locations on the execution branch. Do not merge/rebase this dirty workspace.

Read each Backlog task and the area lessons before starting it. Set it In Progress
and add its Implementation Plan via CLI before code. Leave the other tasks To Do.
Only after targeted checks, review, and documentation add Implementation Notes,
check ACs, and mark Done. Task files intentionally have no Implementation Plan or
Implementation Notes at this planning stage. Re-sweep task/ADR IDs at integration.

TASK-24404 on dev is an overlapping, unimplemented Settings form. Task 8 must amend
its scope/status with a supersession note referencing ADR-118 and TASK-31428 before
the Lab UI lands; use the CLI archive workflow for superseded work, not fake Done.
Preserve its useful stored-invalid decoration and ingest-cache requirements here.
Do not copy an obsolete task over a changed upstream version.

Use repo-configured lint/format checks where available. No ruff/black configuration
was found in this checkout: confirm tools on the execution branch, then use targeted
`ruff check` and `ruff format --check` with the installed tool; report unavailable
tooling rather than claim a pass or install a runtime dependency. Dev's testing
lessons note that existing files are not globally ruff-format-clean: compare a
flagged modified file with its HEAD baseline, format new files, and do not reformat
unrelated legacy code to make a whole-file gate green. Record inherited failures
separately from changed-code checks. Every task runs
`git diff --check` and inspects its own staged diff before a scoped commit. Do not
commit other tasks' or users' staged files.

## File responsibilities and dependency order

Paths below are exact planned paths, relative to the repository. Existing paths
refer to dev, not necessarily this older checkout. New package initializers belong
to their first task, not separate scaffold tasks.

| Task | Deliverable | Dependencies |
| --- | --- | --- |
| TASK-31421 | Local preflight + faithful execution reports | Existing ADR-078 implementation |
| TASK-31422 | Lossless authoring and candidate state | 31421 |
| TASK-31423 | Durable checkpoints and autosave | 31422 |
| TASK-31424 | Recovery export/restore/undo | 31423 |
| TASK-31425 | Bounded run lifecycle and batch coordinator | 31421–31424 |
| TASK-31426 | Conflict-safe template saves | 31421–31422 |
| TASK-31427 | Comparison model and result widgets | 31421–31422 |
| TASK-31428 | Library screen and complete user flow | 31424–31427 |

`Chunking/lab_models.py` owns serializable domain values; `lab_preflight.py` owns
capabilities; `template_runtime.py` remains the only processor seam. `lab_state.py`
owns pure editing transitions, `lab_autosave.py` writer/status scheduling,
`lab_recovery.py` transfer validation, `lab_runner.py` process lifetime, and
`lab_coordinator.py` their sequencing. `lab_comparison.py` owns measurements/diffs.
`DB/Chunking_Lab_DB.py` owns durable transactions. `RAG_Admin/chunking_lab_service.py`
owns Lab save validation, using `ChunkingInteropService` for writes.
`UI/Chunking_Lab_Modules/` contains presentation regions and dialogs only.

### Task 1 / TASK-31421: Faithful local preview preflight and reports

**Files**

- Create: `tldw_chatbook/Chunking/lab_models.py`, `tldw_chatbook/Chunking/lab_preflight.py`.
- Modify: `tldw_chatbook/Chunking/template_runtime.py` (shared apply/report seam).
- Test: `Tests/Chunking/test_lab_preflight.py`, `Tests/Chunking/test_lab_execution.py`, `Tests/Chunking/test_template_runtime.py`.
- Preserve: `tldw_chatbook/RAG_Admin/template_validation.py`, `tldw_chatbook/Chunking/engine/`.

**Interfaces**

- Consumes existing `template_from_record(record: dict)`, `apply_template(template: dict, text: str, options: dict | None = None) -> list[dict]`, and `validate_template(body: dict) -> dict`.
- Produces frozen Pydantic `RuntimeIdentity(backend: str, engine_version: str, execution_version: str, assets: tuple[dict, ...])`; assets contain kind/name/version/content digest, not secrets.
- Produces `PreparedRecipe(authored_json: str, effective_json: str, runtime: RuntimeIdentity, recipe_hash: str)`. Store canonical JSON strings so a frozen wrapper cannot hide mutable nested inputs.
- Produces `ExecutionReport(chunks: tuple[dict, ...], transformed_text: str, diagnostics: tuple[dict, ...])`; validate and copy nested values at serialization/publication boundaries. Chunk dicts contain text, metadata, provenance, and optional verified spans with coordinate space.
- Produces `prepare_recipe(body: dict, *, runtime: RuntimeIdentity) -> PreparedRecipe`, `current_local_runtime() -> RuntimeIdentity`, and `PreviewUnsupportedError` with `field`/`reason`.
- Produces `execute_prepared(recipe: PreparedRecipe, text: str) -> ExecutionReport` in `template_runtime.py`. Existing `apply_template` retains its signature/return contract and delegates shared execution, not the stricter Lab-only admission gate.

- [ ] Write the initial failing preflight test, plus real execution fixtures for pre/post effects, empty output, metadata preservation, and exact/unavailable mappings.

```python
import pytest
from tldw_chatbook.Chunking.lab_preflight import (
    PreviewUnsupportedError, current_local_runtime, prepare_recipe,
)

def test_unknown_operation_cannot_be_silently_skipped():
    body = {"chunking": {"method": "words", "config": {"max_size": 4}},
            "preprocessing": [{"operation": "unregistered_operation", "config": {}}]}
    with pytest.raises(PreviewUnsupportedError, match="preprocessing"):
        prepare_recipe(body, runtime=current_local_runtime())
```

- [ ] Run `pytest Tests/Chunking/test_lab_preflight.py -q`; confirm a missing-feature failure, not an import/dependency/download failure.
- [ ] Implement the separate admission gate: inspect the live engine/operation registry, define a tested method/operation/parameter capability table outside the vendor tree, distinguish metadata/classifier fields, and capture all used defaults/assets. Refuse unknown executable keys, ignored conditions, legacy shapes, missing assets, network/LLM methods, or lossy combinations. Do not use the parity validator's normalized output as the saved authoring body.

```python
# Core preflight control flow; capability checks raise the field-specific error.
verdict = validate_template(body)
if not verdict["valid"]:
    issue = verdict["errors"][0]
    raise PreviewUnsupportedError(issue["field"], issue["message"])
# After capability/default/asset checks, construct PreparedRecipe from separate
# canonical authored/effective documents; metadata remains in authored_json.
```

- [ ] Extend shared execution reporting without copying splitting/operation algorithms. Capture preprocessing text/metadata once; retain structured chunk fields; namespace authoritative counters/provenance. Only preserve source spans when verified, including unique attribution of repeated text. Reject unsupported combinations during preflight; no stringification of dict chunks or fallback from a legitimate empty final list. Check the concrete pinned processor's behavior before advertising each capability.
- [ ] Run `pytest Tests/Chunking/test_lab_preflight.py Tests/Chunking/test_lab_execution.py Tests/Chunking/test_template_runtime.py Tests/RAG_Admin/test_template_validation.py -q`. Real fixtures must include saved-vs-draft equivalence and socket/download guards. Record skipped optional methods as unavailable, not as passing evidence.
- [ ] Review diff, update task notes/ACs, and commit only task files: `feat(chunking): add faithful local Lab execution reports`.

### Task 2 / TASK-31422: Lossless draft and candidate state

**Files**

- Create: `tldw_chatbook/Chunking/lab_state.py`.
- Modify: `tldw_chatbook/Chunking/lab_models.py`.
- Test: `Tests/Chunking/test_lab_state.py`.

**Interfaces**

- Consumes `PreparedRecipe`, `RuntimeIdentity`, `ExecutionReport`, `prepare_recipe` from task 1.
- Produces `DraftState(raw_json: str, parsed_json: str | None, parse_error: dict | None, pending_controls: dict[str, str], authority: str, record_fields: dict, expected_record: dict | None)`; authority is `json`, `controls`, or `synced`. `parsed_json` retains the last successfully parsed document separately from the current raw text. Invalid raw content stays opaque; a non-null `parse_error` blocks Run/Pin/Save even when a last-valid document exists. Explicit discard restores that last-valid document.
- Produces `SampleSnapshot(sample_hash: str, text: str, source: dict)`, `RunRequest(run_id: str, batch_id: str, candidate_id: str, epoch: str, revision: int, sample: SampleSnapshot, recipe: PreparedRecipe, template_record: dict | None = None)`. The optional detached `template_record` captures loaded ID/UUID/version plus authored name/description/tags; it describes provenance, not equality to the current catalog. Copy it into pinned A and each batch member, never reconstruct it from a later live draft.
- Produces `RunResult(request: RunRequest, status: str, report: ExecutionReport | None, started_at: str, finished_at: str, elapsed_ms: float, error: dict | None)`; status is completed/failed/canceled/interrupted/limited. Only completed results contain comparison output.
- Produces `LabSession(profile_key: str, epoch: str, revision: int, candidates: dict, samples: dict, results: dict, batch: dict | None, view: dict, undo: tuple[dict, ...])`. Candidate entries have stable IDs, role A/B, draft or pinned recipe, and current/previous run IDs. Validate max two candidates, reference integrity, and one editable B.
- Produces pure `new_session(profile_key: str) -> LabSession`, `edit_json(session, candidate_id: str, raw: str) -> LabSession`, `edit_control(session, candidate_id: str, path: str, raw: str) -> LabSession`, `can_execute(session, candidate_id: str) -> bool`, `replace_sample(session, text: str, source: dict) -> LabSession`, `pin_baseline(session, *, replace: bool = False) -> LabSession`, and `undo_edit(session) -> LabSession`; all session parameters/returns are `LabSession`.
- Produces `capture_batch(session: LabSession, candidate_ids: tuple[str, ...]) -> tuple[RunRequest, ...]` and `accept_result(session: LabSession, result: RunResult) -> LabSession` with epoch/batch/run membership checks.

- [ ] Write the failing lossless/invalid-authority test below and property cases for nested extension data, operation ordering, method switches, unknown shapes, pin replacement, source edits, and late results.

```python
from tldw_chatbook.Chunking.lab_state import new_session, edit_json, can_execute

def test_invalid_json_is_the_current_draft():
    session = new_session("test-profile")
    candidate_id = next(iter(session.candidates))
    changed = edit_json(session, candidate_id, '{"chunking":')
    assert changed.candidates[candidate_id]["draft"]["raw_json"] == '{"chunking":'
    assert changed.revision == session.revision + 1
    assert not can_execute(changed, candidate_id)
```

- [ ] Run `pytest Tests/Chunking/test_lab_state.py -q` and confirm the intended failure.
- [ ] Implement pure copy-on-edit transitions and JSON-path patching. Control text remains raw until parsed; pending invalid controls make JSON read-only and block Run/Pin/Save. Explicit discard returns to the base document. Metadata/classifier/unknown fields survive. Never silently remove incompatible options on method change.

```python
# Raw input is authoritative; a parse error neither rolls it back nor
# destroys the separate last-valid base used by explicit discard.
parsed_json = previous_draft.parsed_json
parse_error = None
try:
    parsed = json.loads(raw)
except json.JSONDecodeError as exc:
    parse_error = {"message": exc.msg, "line": exc.lineno, "column": exc.colno}
else:
    parsed_json = json.dumps(parsed, ensure_ascii=False)
draft = DraftState(raw_json=raw, parsed_json=parsed_json,
                   parse_error=parse_error,
                   pending_controls={}, authority="json",
                   record_fields=record_fields, expected_record=expected_record)
```

- [ ] Implement sample hashes over exact UTF-8 text, config/default identities, stable candidate IDs, result/input staleness, and immutable Run both capture. Pin requires a completed current B result; do not infer pin validity from a mutable template name. Retain undo snapshot references; distinguish content mutations from view-only mutations for recovery undo.
- [ ] Run `pytest Tests/Chunking/test_lab_state.py Tests/Chunking/test_lab_preflight.py -q`; include Hypothesis round-trip assertions and same-sample/different-config identities.
- [ ] Review and commit only task files: `feat(chunking): model lossless Lab drafts and candidates`.

### Task 3 / TASK-31423: Durable session checkpoints and autosave

**Files**

- Create: `tldw_chatbook/DB/Chunking_Lab_DB.py`, `tldw_chatbook/Chunking/lab_autosave.py`.
- Modify: `tldw_chatbook/DB/private_sqlite.py` (register `db.chunking_lab`).
- Modify as needed: `tldw_chatbook/Chunking/lab_models.py` (extract shared shallow/reference validation so checkpoint saves do not duplicate invariants or deep-copy retained blobs).
- Docs: `backlog/docs/sqlite-private-owner-inventory.md` (enumerate the new registered connection owner alongside its inventory tests).
- Test: `Tests/DB/test_chunking_lab_db.py`, `Tests/Chunking/test_lab_autosave.py`, `Tests/DB/test_private_sqlite_inventory.py`.

**Interfaces**

- Consumes `LabSession`, `RunResult`, their JSON validation, and existing `connect_private_sqlite`/private directory helpers.
- Produces `CheckpointToken(profile_key: str, epoch: str, revision: int, generation: int)`; generation is durable compare-and-swap state, separate from UI revision.
- Produces `CheckpointStore(path: Path, profile_key: str)` with `load() -> tuple[LabSession, CheckpointToken] | None`, `save(session: LabSession, *, expected: CheckpointToken | None) -> CheckpointToken`, `clear(*, expected: CheckpointToken) -> CheckpointToken`, `close() -> None`; errors `CheckpointConflict`, `RecoverySchemaError`.
- Produces `AutosaveWriter(store: CheckpointStore)` with `submit(session: LabSession, *, immediate: bool = False) -> None`, `async flush() -> CheckpointToken`, `async clear() -> tuple[LabSession, CheckpointToken]`, `async close() -> None`, plus `SaveStatus(state: str, acknowledged: CheckpointToken | None, latest_revision: int, error: str | None)` and `status -> SaveStatus`. Store connections open lazily in the writer thread, never its UI-thread constructor; UI receives copied status snapshots.

- [ ] Write the failing round-trip test and real two-connection conflict/crash-publication tests. Use a subprocess killed before/after COMMIT to prove old-or-new completeness, not mocks of SQLite success.

```python
from tldw_chatbook.Chunking.lab_state import new_session, edit_json
from tldw_chatbook.DB.Chunking_Lab_DB import CheckpointStore

def test_invalid_draft_survives_store_reopen(tmp_path):
    session = new_session("test-profile")
    candidate_id = next(iter(session.candidates))
    session = edit_json(session, candidate_id, '{"chunking":')
    path = tmp_path / "lab.sqlite3"
    store = CheckpointStore(path, "test-profile")
    store.save(session, expected=None)
    store.close()
    reopened = CheckpointStore(path, "test-profile")
    restored, token = reopened.load()
    assert restored.model_dump() == session.model_dump()
    assert token.revision == session.revision
    reopened.close()
```

- [ ] Run `pytest Tests/DB/test_chunking_lab_db.py -q` and verify the missing-store failure.
- [ ] Implement version-1 tables `lab_state` (singleton epoch/generation/current/previous/restore-undo checkpoint IDs), `lab_checkpoints` (revision/document), and `lab_blobs` (content-addressed sample/result payload). Use BEGIN IMMEDIATE, parameterized writes, foreign keys/integrity validation, and a durable commit policy consistent with claimed crash recovery. Save all new blobs and publish their checkpoint in one transaction; CAS includes expected epoch/generation. No corrupt/newer schema reset.

```sql
UPDATE lab_state
SET current_checkpoint = ?, previous_checkpoint = current_checkpoint,
    epoch = ?, generation = generation + 1
WHERE singleton = 1 AND epoch = ? AND generation = ?;
```

- [ ] Implement retention including current/previous/in-session undo/restore-undo references, crash-safe GC after publication, private sidecar policy, and canceled/interrupted recovery normalization. Clear leaves a durable new-epoch tombstone, removes content references/blobs, and invalidates old CAS tokens; load of that tombstone returns a fresh empty session with its current token, not `None`. It is not secure erasure.
- [ ] Implement one serialized asynchronous writer, bounded coalescing, max-wait debounce, immediate critical checkpoints, and revision-aware status. Out-of-order acknowledgments may update durable bookkeeping but cannot mark a newer draft saved or clear a newer error. Retry uses the latest session; cross-instance conflicts stop automatic overwrite and preserve memory.
- [ ] Run `pytest Tests/DB/test_chunking_lab_db.py Tests/Chunking/test_lab_autosave.py Tests/DB/test_private_sqlite.py Tests/DB/test_private_sqlite_inventory.py -q`; test invalid controls, deleted source, disk failure, newer schema, malformed checkpoints, two profiles, old acknowledgments, and late writes after Clear.
- [ ] Review and commit only task files: `feat(chunking): persist recoverable Lab checkpoints`.

### Task 4 / TASK-31424: Recovery transfer and replacement undo

**Files**

- Create: `tldw_chatbook/Chunking/lab_recovery.py`.
- Modify: `tldw_chatbook/DB/Chunking_Lab_DB.py`, `tldw_chatbook/Chunking/lab_autosave.py`.
- Modify: `tldw_chatbook/Chunking/lab_state.py` (bounded content undo and pruning of unused active sample/result entries).
- Modify: `tldw_chatbook/Chunking/lab_models.py` (`LabSession.content_revision: int = 0`, a persisted nonnegative content-mutation counter; older checkpoints default to zero). Optional per-session size-measurement metadata must remain non-serialized, rebuilt on untrusted ingress, and pruned to reachable payload identities.
- Test: `Tests/Chunking/test_lab_recovery.py`, `Tests/DB/test_chunking_lab_db.py`, `Tests/Chunking/test_lab_state.py`.
- Docs: append the evidenced edit/Undo/coalescing trap to `backlog/docs/lessons-testing-evidence.md`.

**Interfaces**

- Consumes `LabSession`, `CheckpointStore`, `CheckpointToken`, `AutosaveWriter`.
- Produces `export_recovery(session: LabSession) -> bytes`, `parse_recovery(payload: bytes) -> LabSession`, `RecoveryImportError`, and store `replace(imported: LabSession, displaced: LabSession, *, expected: CheckpointToken) -> tuple[LabSession, CheckpointToken]` plus `undo_restore(*, expected: CheckpointToken) -> tuple[LabSession, CheckpointToken]`.
- Expose these transactions through `AutosaveWriter.async replace(imported: LabSession, displaced: LabSession) -> tuple[LabSession, CheckpointToken]` and `async undo_restore() -> tuple[LabSession, CheckpointToken]`; callers never write the store concurrently. Drain/invalidate queued old writes before replacement and adopt the new token only on success.
- Replacement normalizes the target profile to the current store, assigns a new epoch, and preserves displaced **in-memory** content in the same transaction. Quiescing workers and old writer requests is the task-5 coordinator's prerequisite, not a UI-only convention.
- Before a new authority epoch is assigned on replacement/Undo restore, materialize unfinished members as Interrupted under their captured epoch, then retire the active manifest (`batch=None`). Keep each existing result request's original epoch/batch ID and full snapshot unchanged. The displaced-original undo checkpoint remains exact; any rebased fallback copy uses the same terminalize-then-retire transition.
- Envelope version 1 is UTF-8 JSON, max 256 MiB total; each raw draft max 2 MiB, each sample max 2 MiB, each result max 32 MiB, at most two candidates, depth 64, at most 16 referenced sample/result blobs combined. A bounded whole current checkpoint is max 8 MiB excluding blobs. Reject over-limit edits/imports explicitly without losing the previous value; no truncation. Apply these limits to exportable active state as well as imports; undo/previous checkpoints are not exported.
- Keep one prior content-action undo in v1 (separate from one-level Undo restore); native editor undo is not reimplemented. Replace, rather than append indefinitely to, the application undo tuple on the next content mutation; view changes preserve it. Prune sample/result map entries unreachable from current/previous candidate results, active sample/batch, and available undo. Repeated editing/reruns must not retain an unbounded history or exhaust the active 16-blob allowance solely because unused results were left in a map. Preserve all still-inspectable and undo-needed content.
- Increment `content_revision` on content transitions (including Undo, sample changes and run-state changes), but preserve it for view-only updates. Persist/export it so edit-then-undo before a coalesced save still expires Undo restore even when final content bytes equal the restored document. Overall revision continues to advance for all recovery-relevant changes.

- [ ] Write the failing pure round-trip test, plus malformed/newer/digest-mismatch/oversized inputs and failed atomic replacement.

```python
from tldw_chatbook.Chunking.lab_state import new_session, edit_json
from tldw_chatbook.Chunking.lab_recovery import export_recovery, parse_recovery

def test_recovery_export_preserves_invalid_authoring_text():
    session = new_session("test-profile")
    candidate_id = next(iter(session.candidates))
    session = edit_json(session, candidate_id, '{"chunking":')
    restored = parse_recovery(export_recovery(session))
    assert restored.candidates == session.candidates
```

- [ ] Run `pytest Tests/Chunking/test_lab_recovery.py -q` and confirm missing-feature failure.
- [ ] Implement structural validation independently of executable recipe validation; reject NaN/Infinity, duplicate JSON keys, dangling references, illegal candidate membership, digest mismatches, and bounded-depth/count violations before state replacement. Raw authoring strings remain opaque. No pickle, archive extraction, embedded path reads, or model reconstruction with side effects.
- [ ] Exercise many draft edits, source replacements, and reruns with bounded active undo/reachability. Confirm current/previous and undo-needed results remain inspectable, view changes preserve undo, and obsolete map entries disappear without rewriting retained blob payloads.

```python
def export_recovery(session: LabSession) -> bytes:
    # Validate referenced content and limits before serialization; exclude undo.
    payload = {"format": "chunking-lab-recovery", "version": 1,
               "session": session.model_dump(mode="json", exclude={"undo"})}
    return json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
```

- [ ] Add transactional replace/undo to the store: retain displaced checkpoint separately from rolling previous, update epoch only on COMMIT, restore writer authority on failure, keep undo through view-only saves, release it on the next content mutation, remove it on Clear. Import inspection stays available if persistence is broken; replacement does not.
- [ ] Run `pytest Tests/Chunking/test_lab_recovery.py Tests/DB/test_chunking_lab_db.py Tests/Chunking/test_lab_autosave.py -q`, including export without DB access and restore into another profile. Verify a later view-only save cannot garbage-collect the displaced checkpoint.
- [ ] Review and commit only task files: `feat(chunking): add safe Lab recovery export and restore`.

### Task 5 / TASK-31425: Bounded process and immutable A/B lifecycle

**Files**

- Create: `tldw_chatbook/Chunking/lab_runner.py`, `tldw_chatbook/Chunking/lab_coordinator.py`.
- Test: `Tests/Chunking/test_lab_runner.py`, `Tests/Chunking/test_lab_coordinator.py`.
- Modify only as needed: `tldw_chatbook/Chunking/lab_preflight.py` (verified resource capability limits).

**Interfaces**

- Consumes `RunRequest`, `RunResult`, `capture_batch`, `accept_result`, `execute_prepared`, `AutosaveWriter`, `parse_recovery`.
- Produces `PreviewLimits(sample_bytes: int = 2097152, chunks: int = 10000, result_bytes: int = 33554432, wall_seconds: float = 60.0)` and `LocalPreviewRunner(limits: PreviewLimits)` with `async run(request: RunRequest) -> RunResult`, `async cancel() -> None`, `async close() -> None`. Cancel returns only after child termination and reaping.
- Produces `LabCoordinator(session: LabSession, writer: AutosaveWriter, runner: LocalPreviewRunner)` with `session -> LabSession`, `async run(candidate_ids: tuple[str, ...]) -> None`, `async cancel() -> None`, `async replace_recovery(payload: bytes) -> None`, `async undo_restore() -> None`, `async clear() -> None`, `async close() -> None`, and `set_session(session: LabSession) -> None` for serialized pure UI transitions. Replacement/clear reject edits until the guarded transition settles.
- The coordinator exposes copied session/status change events to subscribers; UI widgets never receive process or DB handles. One app/profile owns one coordinator, surviving screen unmount.
- A pure undo may remove a newly pinned A and invalidate its installed batch (`batch` becomes `None`). `set_session` must stop that batch's worker and remaining queue; no later member may launch or publish. A new Run remains blocked until the prior worker has stopped.

- [ ] Write the failing limit test below and process-backed timeout/cancel tests with a deliberately non-cooperative local test child. Write coordinator tests using recording runner/writer doubles to inspect immutable requests and publication ordering.

```python
import pytest
from tldw_chatbook.Chunking.lab_state import new_session, replace_sample, capture_batch
from tldw_chatbook.Chunking.lab_runner import LocalPreviewRunner, PreviewLimits

@pytest.mark.asyncio
async def test_sample_limit_reports_failure_without_clipping():
    session = replace_sample(new_session("test-profile"), "one two three", {"kind": "paste"})
    candidate_id = next(iter(session.candidates))
    request, = capture_batch(session, (candidate_id,))
    runner = LocalPreviewRunner(PreviewLimits(sample_bytes=3))
    result = await runner.run(request)
    assert result.status == "limited"
    assert result.request.sample.text == "one two three"
    assert result.report is None
    await runner.close()
```

- [ ] Run `pytest Tests/Chunking/test_lab_runner.py Tests/Chunking/test_lab_coordinator.py -q` and establish the intended red baseline.
- [ ] Implement a spawn-compatible top-level worker and bounded JSON pipe messages. Pass only captured request data, not live DB objects/config clients. Recheck requested runtime/assets in the child; mismatch is failure. Parent supervises monotonic wall time, output size, and child exit; terminate then kill/reap with bounded waits. No second run until the first has stopped.

```python
# Supervisor primitive: cap message allocation before accepting a result.
if connection.poll(0.05):
    payload = connection.recv_bytes(maxlength=limits.result_bytes)
    # Validate JSON and RunResult membership before publishing.
# Polling and joining execute outside Textual's event loop; the async caller
# remains cancelable while the supervisor owns the process through final reaping.
```

- [ ] Enforce input/chunk/serialized-output/time budgets with named limited outcomes. Test intermediate amplification and peak RSS for admitted operations using isolated processes; final JSON size is not a memory cap. Bound configurable expansion before execution and refuse combinations that cannot meet the resource envelope. Record OS-specific process/resource limitations and never describe this worker as a security sandbox. Do not add psutil or fork the engine to get metrics.
- [ ] Implement Run both: capture/validate both, store the batch manifest with queued states, await its committed checkpoint, then execute A and B sequentially from those objects. A failure may continue B; cancellation stops both. Commit each output/reference atomically; retain an unsaved result in memory on persistence failure. Never borrow previous results to satisfy a failed batch. Editing changes only the next run and staleness badges.
- [ ] Implement guarded navigation/restore/clear: stop queue and child, settle writer requests, then replace/clear transactionally; old epoch messages are ignored. Failure keeps the old session and writer retry authority. Reopen marks unfinished runs Interrupted without dispatch. Closing flushes or exposes a recoverable failure rather than silently dropping memory.
- [ ] Run `pytest Tests/Chunking/test_lab_runner.py Tests/Chunking/test_lab_coordinator.py Tests/Chunking/test_lab_recovery.py Tests/Chunking/test_lab_autosave.py -q`. Include mutated catalog while A runs, save failure before launch, A-fail/B-success, stale completion, concurrent Run clicks, restore failure, and quit during a non-cooperative run.
- [ ] Review and commit only task files: `feat(chunking): supervise immutable local Lab comparisons`.

### Task 6 / TASK-31426: Conflict-safe template saving

**Files**

- Create: `tldw_chatbook/RAG_Admin/chunking_lab_service.py`.
- Modify: `tldw_chatbook/Chunking/chunking_interop_library.py` (`update_template` expected-version predicate).
- Test: `Tests/RAG_Admin/test_chunking_lab_service.py`.
- Regression: existing `Tests/RAG_Admin/test_local_rag_admin_service.py`.

**Interfaces**

- Consumes `DraftState`, `PreparedRecipe`, `prepare_recipe`, `ChunkingInteropService` and existing Media DB transaction/live-name/builtin conventions.
- Produces `ExpectedTemplate(id: int, uuid: str, version: int)`, `TemplateSaveConflict`, and `save_lab_template(service: ChunkingInteropService, *, body: dict, name: str, description: str, tags: list[str], expected: ExpectedTemplate | None = None) -> dict`. Return the refreshed canonical record with ID/UUID/version and listing decoration where available.
- Extend `ChunkingInteropService.update_template` with optional keyword-only `expected_uuid: str | None = None` and `expected_version: int | None = None`; require both or neither. Lab always supplies both on update. Existing callers keep their signatures/semantics, but builtin/live protection is rechecked inside the write transaction for everyone.
- Import/export template files use the existing record envelope; preserve tags as the canonical column and the exact authored body semantics. Legacy body import remains a draft, not an implicit migration.

- [ ] Write a failing real SQLite stale-version test using the existing Media DB fixture pattern from the local RAG-admin tests; include concurrent same-name creation and stored-invalid repair.

```python
# With a real MediaDatabase fixture named media_db:
def test_stale_lab_update_cannot_overwrite_newer_record(media_db):
    service = ChunkingInteropService(media_db)
    body = {"chunking": {"method": "words", "config": {"max_size": 4}}}
    record = save_lab_template(service, body=body, name="Test", description="Recipe", tags=[])
    expected = ExpectedTemplate(id=record["id"], uuid=record["uuid"], version=record["version"])
    service.update_template(record["id"], description="Changed elsewhere")
    with pytest.raises(TemplateSaveConflict):
        save_lab_template(service, body=body, name="Test", description="Stale", tags=[], expected=expected)
    assert service.get_template_by_id(record["id"])["description"] == "Changed elsewhere"
```

- [ ] Run `pytest Tests/RAG_Admin/test_chunking_lab_service.py -q` and confirm the expected failing contract.
- [ ] Implement headless Lab Save preflight on the final body/record fields, then existing create/update. Keep global parity validation separate. Refuse invalid pending editor state before calling this service; the service still independently refuses a syntactically valid unsupported body. Normalize documented tags placement, preserve other metadata, and preserve the user's name/description edits on errors.
- [ ] Add atomic expected-version update, with all mutable/protected conditions in SQL and rowcount checked before commit. Map uniqueness/conflict/builtin errors distinctly; no timestamp-only check and no precheck followed by unguarded update.

```sql
UPDATE ChunkingTemplates
SET name = ?, description = ?, template_json = ?, tags = ?, version = version + 1
WHERE id = ? AND uuid = ? AND version = ? AND deleted = 0 AND is_builtin = 0;
```

- [ ] Run `pytest Tests/RAG_Admin/test_chunking_lab_service.py Tests/RAG_Admin/test_local_rag_admin_service.py Tests/RAG_Admin/test_template_validation.py -q`. Prove no source chunk/default mutation; reserved `auto` spelling variants fail; builtins can be copied but not updated; valid advanced metadata/tags round-trip.
- [ ] Review and commit only task files: `feat(chunking): save Lab templates with atomic conflict checks`.

### Task 7 / TASK-31427: Honest comparison and bounded inspection

**Files**

- Create: `tldw_chatbook/Chunking/lab_comparison.py`.
- Create: `tldw_chatbook/UI/Chunking_Lab_Modules/__init__.py`, `tldw_chatbook/UI/Chunking_Lab_Modules/results_region.py`.
- Test: `Tests/Chunking/test_lab_comparison.py`, `Tests/UI/test_chunking_lab_results.py`.

**Interfaces**

- Consumes `RunResult`, `PreparedRecipe`, verified chunk spans and runtime identities.
- Produces `comparison_reason(a: RunResult, b: RunResult) -> str | None` (None means compatible), `summarize_result(result: RunResult, *, token_counts: tuple[int, ...] | None = None, measurement_id: str | None = None) -> dict`, and `diff_configs(a: RunResult, b: RunResult, *, authored: bool = False) -> tuple[dict, ...]` (path/kind/A/B entries).
- Produces `ResultsRegion(Widget)` with `show_results(a: RunResult | None, b: RunResult | None, *, stale_ids: frozenset[str]) -> None`; emits selection and rerun requests, never executes/loads DB records itself.
- Character size uses Python Unicode code-point `len(text)`, words use `len(text.split())`, p95 uses nearest-rank `ceil(.95*n)-1`, empty distributions have unavailable quantiles. Token counts require an explicit available local measurement tokenizer identity; recomputation is separate from execution identity.

- [ ] Write the failing compatibility test below; add snapshot diff, mismatched-unit, repeated-text mapping, zero-output distribution, and 10k-row navigation tests.

```python
def test_identity_difference_is_not_itself_incompatibility():
    from tldw_chatbook.Chunking.lab_state import new_session, replace_sample, capture_batch
    from tldw_chatbook.Chunking.lab_models import RunResult
    from tldw_chatbook.Chunking.template_runtime import execute_prepared
    from tldw_chatbook.Chunking.lab_comparison import comparison_reason
    session = replace_sample(new_session("test"), "one two three", {"kind": "paste"})
    request, = capture_batch(session, tuple(session.candidates))
    report = execute_prepared(request.recipe, request.sample.text)
    a = RunResult(request=request, status="completed", report=report,
                  started_at="2026-09-04T00:00:00Z", finished_at="2026-09-04T00:00:01Z",
                  elapsed_ms=1000, error=None)
    b = a.model_copy(update={"request": request.model_copy(update={"run_id": "another-run"})})
    assert comparison_reason(a, b) is None
```

- [ ] Run `pytest Tests/Chunking/test_lab_comparison.py -q` to establish the red baseline.
- [ ] Implement compatibility from successful status, exact sample hash, backend/engine/execution versions. Show asset/method/tokenizer differences as experimental variables. Compute common counts and expansion ratios; no false token delta across measurement IDs, no subtraction of unlike method units, no runtime ranking/quality score.

```python
# Effective diff compares snapshots, never the currently edited draft.
left = json.loads(a.request.recipe.authored_json if authored else a.request.recipe.effective_json)
right = json.loads(b.request.recipe.authored_json if authored else b.request.recipe.effective_json)
# Walk mappings by key and operation arrays by index, emitting added/removed/changed
# paths with complete values; long values open in the inspector.
```

- [ ] Implement one selected-chunk inspector and paged/virtualized chunk rows (100 visible rows per page, not 10k mounted text widgets). Linked A/B highlights use verified original-source coordinates only. Preprocessed coordinates link to transformed text; absent maps explain why alignment/overlap is unavailable. Persist per-candidate selection and active view in session state.
- [ ] Run `pytest Tests/Chunking/test_lab_comparison.py Tests/UI/test_chunking_lab_results.py -q`. Include genuinely different methods/options and chunking tokenizers, same-old-version results, old/new version mismatch, stale draft badges, and authored metadata/classifier diffs.
- [ ] Review and commit only task files: `feat(chunking): inspect Lab results with honest comparisons`.

### Task 8 / TASK-31428: Library-owned recoverable authoring screen

**Files**

- Create: `tldw_chatbook/UI/Screens/chunking_lab_screen.py`.
- Create: `tldw_chatbook/UI/Chunking_Lab_Modules/editor_region.py`, `sample_region.py`, `dialogs.py` in the same package.
- Modify: `tldw_chatbook/app.py` (lazy app/profile coordinator ownership and orderly close), `tldw_chatbook/UI/Screens/library_screen.py` (entry/handoff only).
- Modify: `tldw_chatbook/UI/Navigation/screen_registry.py`, `shell_destinations.py`, `tldw_chatbook/UI/Workbench/route_inventory.py`, `tldw_chatbook/UI/stable_command_palette.py` (Library-owned tool route, not a global destination).
- Modify: `tldw_chatbook/Widgets/Library/library_ingest_canvas.py` (saved-template refresh signal).
- Test: `Tests/UI/test_chunking_lab_screen.py`, `Tests/UI/test_chunking_lab_recovery_flow.py`, existing `Tests/UI/test_library_ingest_canvas.py`, `test_screen_navigation.py`, `test_command_palette_shell_routes.py`, `test_workbench_route_inventory.py`.
- Docs: create `Docs/Chunking_Lab.md`; reconcile dev's `backlog/tasks/task-24404 - Settings form for creating and editing chunking templates.md` through the CLI workflow.

**Interfaces**

- Consumes `LabCoordinator`, pure editing transitions, `ResultsRegion`, `save_lab_template`, existing Library text read/source-path validation and lazy navigation seams.
- Produces `ChunkingLabScreen(BaseAppScreen)` for route `chunking_lab`, canonical owner `library`; no new shell destination constant. Entry handoff contains `return_route: str` and optional `local_media_id: int`, not raw private text in navigation metadata.
- Produces `EditorRegion`, `SampleRegion`, and task-local save/import/restore dialogs. Region messages request domain transitions; the screen delegates to the app-owned coordinator.
- Add app `get_chunking_lab_coordinator() -> LabCoordinator` lazily for the active profile. On profile change close the old coordinator before opening the new profile store at `get_user_data_dir() / "chunking_lab.sqlite3"` through existing private-path helpers.
- Successful template save emits a local `ChunkingTemplatesChanged` Textual message carrying record ID/version only; the ingest canvas invalidates `_chunk_template_names` and refreshes on next display/currently mounted use.

- [ ] Re-read the current TASK-24404 and ADR-003/078/118. Archive its superseded Settings proposal with a CLI note linking this task (do not mark implementation Done). If someone has implemented it since planning, stop duplication, inspect the landed UI, and amend this integration task's AC/plan before moving it. Keep the newer work intact.
- [ ] Write the failing route/editor Pilot tests and recovery integration test. Use the normal test app fixture pattern and temp profile helpers; do not launch the user's real data.

```python
import pytest
from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_route

def test_lab_is_library_owned_and_lazy():
    route = resolve_screen_route("chunking_lab")
    assert route is not None
    assert route.canonical_tab == "library"
    assert route.module_path == "tldw_chatbook.UI.Screens.chunking_lab_screen"

# In the temp-profile test app, exercise an editor rather than triggering shortcuts:
@pytest.mark.asyncio
async def test_local_shortcuts_do_not_steal_editor_text(lab_app):
    async with lab_app.run_test(size=(80, 24)) as pilot:
        editor = lab_app.screen.query_one("#lab-sample-text")
        editor.focus()
        await pilot.press("r", "p", "s")
        assert editor.text.endswith("rps")
```

- [ ] Run `pytest Tests/UI/test_chunking_lab_screen.py -q`; confirm missing-route/screen failures. Define `lab_app` in the new test module's fixture using the app-owned coordinator, real temp SQLite, and the existing navigation test harness. Do not mock all durable behavior.
- [ ] Wire lazy route/owner/palette/Library-item handoff, opener return, and coordinator lifecycle. Direct handoffs bypass Library starter filtering. Keep local execution visible even when the surrounding runtime is server-scoped. Reject non-local/unextracted sources with a useful explanation; no remote fallback.

```python
"chunking_lab": ScreenRoute(
    "chunking_lab", "library",
    "tldw_chatbook.UI.Screens.chunking_lab_screen", "ChunkingLabScreen",
),
```

- [ ] Compose Sample/Configure/Results regions using theme tokens and focus conventions. At narrow widths show one region; wide results can show A/B equally. Add method-specific fields with correct units, full JSON, explicit pending/discard authority, searchable decorated saved templates, Pin/Replace A, Run/Run both/Cancel, result snapshot diff, and Save A/B. Wire each visible action before advertising it in the footer.
- [ ] Add paste/file/local-Library sample flows with exact copied text, UTF-8 validation, size refusal and explicit excerpt choice; read files off the UI loop through existing path validation. Add JSON template import/export and save conflict/error dialogs. Save refreshes the ingest canvas without source re-chunking or default changes.
- [ ] Wire restore-on-open, Saving/Saved locally/Save failed/Unsaved result labels, Retry, recovery Export/Restore/Undo restore, confirmed Clear, and navigation cancellation. Explain that full sample/results are stored locally. Export/restore file I/O uses explicit user-selected paths and existing overwrite/private-path safeguards; no writes to paths carried in imported data.
- [ ] Run the focused integration set:

```bash
pytest Tests/UI/test_chunking_lab_screen.py Tests/UI/test_chunking_lab_recovery_flow.py Tests/UI/test_chunking_lab_results.py Tests/UI/test_library_ingest_canvas.py Tests/UI/test_screen_navigation.py Tests/UI/test_command_palette_shell_routes.py Tests/UI/test_workbench_route_inventory.py -q
```

- [ ] Perform isolated-profile live verification at all three terminal sizes: paste → preview B → pin A → change method/options → compare → save → select saved template in ingest; close/reopen with invalid JSON and completed results; force-kill the isolated app after a committed checkpoint and recover; simulate failed writes and export/restore into a fresh profile; cancel a long worker and verify no surviving child. Record commands/profile paths, screenshots or terminal evidence, and actual failure limitations in task notes. No full suite without opt-in.
- [ ] Document workflow, sample/privacy limits, supported vs preserved settings, classifier behavior, autosave crash window, conflicts, and recovery transfer in `Docs/Chunking_Lab.md`. Complete targeted lint/format, self-review all task ACs and ADR links, then commit only task files: `feat(ui): add recoverable Library Chunking Lab`.

## Coverage and release gates

| Spec acceptance outcomes | Primary task(s) |
| --- | --- |
| 1–3: navigation, sample sources, core A/B/save flow | 31428, supported by 31422/31425/31426 |
| 4–5: lossless/invalid authoring round trips | 31422, 31423, 31428 |
| 6–9: full execution, refusal, snapshots, backend provenance | 31421, 31422, 31425 |
| 10: staleness and compatibility | 31422, 31427 |
| 11–13: restart, atomic crash recovery, failed/conflicting writes | 31423, 31424, 31428 |
| 14–15: trustworthy maps, limits, usable inspection | 31421, 31425, 31427 |
| 16–17: protection/Clear/keys/layout | 31423, 31425, 31426, 31428 |
| 18–20: canonical flat shape, metadata, atomic saves | 31421, 31422, 31426 |
| 21–23: immutable batches, unlike units, invalid-edit authority | 31425, 31427, 31422 |
| 24–26: restore, snapshot diffs, saved-status ordering/undo | 31424, 31427, 31423 |

- [ ] No requirements rely on a skipped/nonexistent-module test as evidence.
- [ ] No vendored edits, second template store, new Settings editor, or server/Evals stubs.
- [ ] Real full-pipeline equivalence fixtures and crash/concurrency tests pass.
- [ ] Supported capabilities are explicit; missing optional assets cannot download implicitly.
- [ ] In-memory and exported/restored limits agree; no accepted active state is silently truncated on export.
- [ ] Dirty user files/index remain untouched; staged diff contains only the intended task.
- [ ] All eight task notes link ADR-118 and this plan; complete only with actual execution evidence.

Planning verification is document-only: source contracts inspected, dependency IDs
allocated across refs/worktrees, spec coverage reviewed, and links/whitespace checked.
No feature tests or implementation have run during this planning pass.
