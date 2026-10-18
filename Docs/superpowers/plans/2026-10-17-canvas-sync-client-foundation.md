# Canvas Sync Client Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete TASK-35376: preserve offline sibling revisions in local storage, recovery and archives without enabling synchronization.

**Architecture:** Extend the existing immutable Canvas graph contract to allow sibling sequence ties. Keep local authoring serial and preserve its optimistic-parent checks. Ship schema and archive compatibility together so a newly admitted graph is always exportable and recoverable.

**Tech Stack:** Python >=3.12 (verification interpreter 3.12.11), SQLite, existing dataclasses/ZIP Chatbook archive and recovery catalog; no new dependencies.

**Spec:** `Docs/superpowers/specs/2026-10-17-canvas-server-sync-contract-design.md`, especially Revision graphs, Compatibility, acceptance cases 1/4/9/11, and accepted ADR-239.

ADR required: yes; existing ADR path: `backlog/decisions/239-canvas-sync-replica-and-lifecycle-contract.md`.
Reason: direct implementation of the approved revision-order, schema and archive change, not a new decision.

## Global Constraints

- Preserve exact immutable IDs, source UTF-8 bytes, hashes, timestamps, profiles, actor, origin message/turn, parent and stored sequence values.
- Permit positive sibling sequence ties; require one root at sequence 1, strictly lower same-Canvas parent sequences, and contiguous distinct sequence values from 1 through the maximum. Keep all current count/byte/ownership bounds.
- New local appends and staged settlement allocate `max(sequence)+1`; count remains the quota measure. Do not accept stale authored sequence claims merely because storage now permits ties.
- Canonical ordering is `(sequence, revision_id)` within each Canvas. Keep the existing branch resolver `(origin_message_position_on_active_path, sequence, revision_id)` and local historical selection.
- Allocate schema 81→82 on this base and Chatbook 4.0 with Canvas extension 2.0. Existing valid Chatbook 3.0/extension 1.0 remains readable under its original unique-sequence rules. Non-Canvas exports remain 2.0. This is a Chatbook archive version, not the future multi-file Canvas V3 runtime.
- Preserve all existing frozen recovery lineages and security restrictions; add exact installed v82 catalogs and bounded migration support, never trust archive-supplied SQL.
- Canvas synchronization remains disabled. No outbox/inbox, server writes, new runtime capabilities, browser/QuickJS launch, or live user-profile database migration.
- Work only in `/Users/macbook-dev/.codex/worktrees/canvas-followups-20261017/tldw_chatbook`, branch `codex/canvas-followups-20261017`. Preserve unrelated work. No dependency installation, push, merge, full test sweep, timeout/cap relaxation or baseline-debt cleanup.
- Evidence root: `/private/tmp/chatbook-canvas-sync-foundation.KPZ9cL`. Use unique pytest `--basetemp` paths and `-p no:cacheprovider`; retain RED and GREEN logs/JUnit and natural exit codes.

---

### Task 1: Ship the complete local sibling-graph compatibility foundation

This is one review unit: database admission, its archive representation and exact recovery acceptance cannot ship separately. The independent server/client transport work is already in other Backlog tasks and is not part of this plan.

**Files and responsibility:**

- Create `tldw_chatbook/DB/migrations/chachanotes_v81_to_v82_canvas_sibling_sequences.sql`: change only the sequence index and add a unique root index.
- Modify `tldw_chatbook/DB/ChaChaNotes_DB.py`: version 82 and the file-backed, transactional v81→82 step; preserve every shipped predecessor.
- Modify `tldw_chatbook/Canvas/repository.py`: tied complete-graph imports, stable reads/insertion order and max-sequence authored settlement. Keep import and authored-append semantics distinct.
- Inspect `tldw_chatbook/Canvas/service.py`, `Canvas/staging.py` and `Chat/console_canvas_controller.py`; modify only a demonstrated count-as-sequence or unstable-order assumption. The existing resolver itself already has the required UUID tie-break.
- Modify `tldw_chatbook/DB/recovery_core_schema.py`, `DB/recovery_core.py` and `Backup_Recovery/sqlite_validation.py`: immutable v82 catalogs and migration, retain v81 and all older accepted variants, allow the pure Canvas CHECK only for exact installed catalogs.
- Modify `tldw_chatbook/Canvas/archive.py`, `Chatbooks/chatbook_models.py`, `chatbook_creator.py` and `chatbook_importer.py`: new archive version, strict legacy/version pairing and canonical export/idempotent comparison.
- Tests: `Tests/ChaChaNotesDB/test_canvas_migration.py`, `Tests/Canvas/test_repository.py`, `test_service.py`, `Tests/Chatbooks/test_chatbook_canvas_models.py`, `test_chatbook_canvas_round_trip.py`, and `Tests/Backup_Recovery/test_canvas_recovery_validation.py`. Add focused fixtures/tests in these files; use a focused new test module only when isolation makes it smaller.
- Packaging: inspect current wildcard/runtime-derived migration discovery in `pyproject.toml`, `MANIFEST.in`, `Packaging/check_manifest.py` and `Tests/Packaging/test_installed_distribution.py`; change only a registry that actually needs a new entry. Prove the new file is included in real built artifacts.
- Documentation: `Docs/Architecture/canvas.md`, `Docs/Architecture/database-layer.md`, `tldw_chatbook/Chatbooks/CHATBOOKS_GUIDE.md`, the approved sync spec's allocated-version note and a concise tracked qualification report. Root owns task status and final notes.

**Interfaces:**

- Consume existing `CanvasImportBatch`, `CanvasImportDocument`, `CanvasImportRevision`, `CanvasRepository.import_batch` / `import_batch_in_transaction`, `append_batch_in_transaction` and `CanvasService` scopes; do not create a second domain owner or a sync ingress API in this task.
- Produce the same typed repository APIs, now accepting complete valid tied graphs. `append_batch_in_transaction` remains a strictly serial authored-settlement seam.
- Produce `ChatbookVersion.V4 = "4.0"`; existing `select_chatbook_version(has_canvas_records: bool)` selects V4 for Canvas and V2 otherwise.
- Produce current `CANVAS_ARCHIVE_EXTENSION_VERSION = "2.0"` with explicit support for legacy `"1.0"`; only V3/1.0 and V4/2.0 pairings are valid.
- Produce `CHACHANOTES_V82_SCHEMAS` and `CHACHANOTES_CANVAS_V81_TO_V82_SQL` for the existing recovery policy, retaining all historical catalogs.

- [ ] **Step 1: Run a bounded baseline and write native failing controls.**

Use the existing interpreter with `PYTHONPATH=packages/tldw_profile_core/src`. Run the six directly affected modules listed above before changes; record every pre-existing failure rather than adjusting unrelated behavior. Write tests that exercise actual SQLite and archive behavior, not source-string assertions.

Start with a complete imported graph containing literal `(sequence, parent)` values `[(1, None), (2, root_id), (2, root_id)]`, supplied out of order. It must preserve both sibling IDs, values, sources and canonical UUID order. Test both owned and caller-owned import paths. Add a following ordinary append and a staged/turn append; both must allocate sequence 3, not revision count 4. A second concurrent authored claim to sequence 3 must still fail.

For the model test, use existing `_revision` / `_canvas_archive` helpers with two distinct children at sequence 2, one root at 1, and `ChatbookVersion("4.0")`. Assert canonical serialized revision IDs, version `"4.0"`, extension `"2.0"`, and lossless `from_dict` reconstruction. Separately require a 1.0 extension containing those ties to fail. Keep genuine legacy 3.0/1.0 fixtures, not fixtures silently upgraded through the current constant.

Run those new controls against unchanged production source. Expected RED is unique-sequence rejection / old archive-version rejection. Keep existing malformed-root/parent/owner, invalid digest/source, gaps, count/byte quota and rollback controls.

- [ ] **Step 2: Implement schema and repository semantics from the RED controls.**

The migration's DDL is:

```sql
DROP INDEX idx_canvas_revisions_canvas_sequence;
CREATE INDEX idx_canvas_revisions_canvas_sequence
  ON canvas_revisions(canvas_id, sequence, id);
CREATE UNIQUE INDEX uq_canvas_revisions_single_root
  ON canvas_revisions(canvas_id) WHERE parent_revision_id IS NULL;
```

The v81→82 Python step uses `_require_migration_entry_version(conn, 81, "V81→V82")`, reads this SQL file through `_execute_migration_statements` inside `self.transaction()`, and updates the version with a guarded `WHERE version = 81`, following the existing v80→81 step. Do not edit old migration SQL or use `executescript`. Verify the exact predecessor index shape before rewriting it so incompatible/unshipped catalogs are refused without mutation.

For complete-graph import validation, retain per-record validation and quota checks, sort by `(sequence, revision_id)`, and replace the old unique/gapless sequence test with:

```python
distinct_sequences = sorted({revision.sequence for revision in ordered})
if distinct_sequences != list(range(1, distinct_sequences[-1] + 1)):
    raise CanvasValidationError("invalid_revision_sequence")
if sum(revision.parent_revision_id is None for revision in ordered) != 1:
    raise CanvasValidationError("invalid_root_parent")
```

Existing strictly lower same-Canvas parent validation remains. Empty graphs already fail before this block. Retain all per-revision and aggregate quota checks and identity collision refusal.

In authored batch validation, query actual count and max separately in one query:

```sql
SELECT COUNT(*), COALESCE(MAX(sequence), 0)
FROM canvas_revisions WHERE canvas_id = ?
```

Use the count for quota and `int(max_sequence)+1` for the first expected authored sequence, incrementing once per authored revision. Preserve current scope/anchor/currentness validation and immediate write locking. Add revision ID as the final ordering key to all changed graph reads/inserts, not to unrelated queries.

- [ ] **Step 3: Prove real migration, recovery and predecessor compatibility.**

Create a genuine schema-81 file using `chachanotes_db_at_version`, populate Canvas data, close and reopen with the candidate. Compare original immutable rows byte-for-byte, require the new schema/index shape to match fresh v82, then import tied siblings successfully. Inject a failure inside the real step, assert old version/index/rows survive, then retry. A second root, cross-owner/forward/equal-sequence parent, mutable revision update and forged payload still fail.

Extend frozen recovery catalogs from all current v81 variants without normalizing arbitrary SQL. Keep all four v81 variants explicitly accepted once v82 becomes current; the old head/dictionary variants no longer ride current-version entries. Add only exact installed migration SQL and current catalogs to the existing pure-validator trust scope. Run valid/corrupt/custom/rewrite core and hybrid recovery controls against current v82; also prove a genuine v81 candidate upgrades without losing rows or relaxing restricted-connection authority.

- [ ] **Step 4: Implement the archive contract and every real caller.**

In `CanvasArchiveDocument`, accept the same complete tied graph invariant as the repository and sort by `(sequence, revision_id)`. At the containing extension boundary, explicitly accept versions 1.0 and 2.0 but enforce original unique-sequence rules for 1.0. At the Chatbook manifest boundary require V3→1.0 and V4→2.0, rejecting unknown or mismatched combinations before extraction/database access.

```python
class ChatbookVersion(Enum):
    V1 = "1.0"
    V2 = "2.0"
    V3 = "3.0"
    V4 = "4.0"
```

Update creator assignment, deterministic ZIP path, supported importer versions, v2 structured conversation handling and manifest serialization/parsing. New Canvas exports always use V4/2.0; no-Canvas exports stay V2. Same-identity SQL comparison must use `ORDER BY sequence, id` and the matching `(sequence, revision_id)` Python order. No-echo or incremental sync ingress is not added.

- [ ] **Step 5: Qualify archives, real package contents and unchanged boundaries.**

Round-trip a genuine tied graph through `ChatbookCreator`/`ChatbookImporter` into a separate synthetic DB, repeat import idempotently, and export again with deterministic bytes. Verify exact sources/profiles/origins, deleted/history rows and branch projection. Verify valid legacy 3.0 import; malformed 3.0 ties and new-version/extension mismatches refuse atomically, leaving target DB state unchanged. Qualify rejection of a new archive with the actual pre-change reader from recorded base Git bytes in a disposable probe; do not copy a legacy implementation into production or add permanent history-dependent tests.

Run the affected repository/service/staging, migration, Canvas archive and backup controls, `Tests/DB/test_migration_file_sources.py`, and relevant installed-distribution migration checks. Build an sdist and wheel with existing installed tooling (no dependency fetch) into the evidence directory and list/verify the new migration in both. Preserve existing build/runtime input pins.

Run scoped Ruff check/format, whitespace, the index-plan/schema checks and local derived preflight. Attribute inherited static failures against exact base rather than reformatting unrelated large files. No full-suite or fresh hosted/platform claim.

- [ ] **Step 6: Document, self-review and hand off the exact evidence.**

Update the living schema/archive docs, allocated-version note in the approved contract and concise qualification report. Record commands, natural exit codes, RED/GREEN failures and passes, exact source/import roots, package contents and limitations in `task-1-report.md` under this plan's SDD workspace. Root alone closes Backlog acceptance criteria after independent review.

Review every changed caller listed above for count/sequence, version or ordering assumptions. Commit only this task's owned files after `git diff --cached --check` succeeds; do not stage root's unrelated governance work. Return the commit range and brief report contract. A reviewer independently checks spec compliance and code quality before the final whole-foundation review.
