# RAG Citation Foundation Dev Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the completed citation provenance foundation with current `dev`, advance citation storage to schema v27, and publish a merge-ready pull request.

**Architecture:** Merge `origin/dev` into the feature branch once, preserving current `dev` as the owner of v24→v25 message-generation metadata and v25→v26 conversation-summary migrations. Move citation provenance to a new v26→v27 migration, combine both branches' persistence and test-isolation behavior, and regenerate the CSS bundle from merged source modules instead of resolving generated output by hand.

**Tech Stack:** Python 3.12, SQLite migrations, Pydantic, Textual TCSS, pytest, Ruff, Backlog.md, Git/GitHub CLI.

**Backlog:** `TASK-557`

**ADR required:** no new ADR

**ADR path:** `backlog/decisions/024-rag-citation-provenance-and-source-resolution.md`

**Reason:** ADR-024 already governs citation storage and persistence. Selecting the next free migration version and preserving current `dev` behavior are anticipated integration work, not a new architectural decision.

---

### Task 1: Commit the Integration Plan, Merge Current Dev, and Preserve the Conflict Boundary

**Files:**

- Modify through merge: all files changed by `origin/dev`
- Resolve later: `Tests/DB/test_chachanotes_active_leaf_migration.py`
- Resolve later: `Tests/conftest.py`
- Resolve later: `tldw_chatbook/Chat/chat_persistence_service.py`
- Resolve later: `tldw_chatbook/DB/ChaChaNotes_DB.py`
- Regenerate later: `tldw_chatbook/css/tldw_cli_modular.tcss`

- [ ] **Step 1: Stage and commit the approved integration plan**

Run:

```bash
git add \
  Docs/superpowers/plans/2026-07-24-rag-citation-foundation-dev-integration.md \
  "backlog/tasks/task-557 - Integrate-citation-foundation-with-current-dev-schema.md"
git diff --cached --check
git commit -m "docs(rag): plan citation foundation dev integration"
```

Expected: the plan and In Progress Backlog task are committed before the merge begins.

- [ ] **Step 2: Fetch and verify the pre-merge state**

Run:

```bash
git fetch origin dev
git status --short
git rev-list --left-right --count origin/dev...HEAD
git merge-tree --write-tree origin/dev HEAD
```

Expected: clean worktree; the comparison reports five aggregate conflicts in the files listed above.

- [ ] **Step 3: Merge without committing**

Run:

```bash
git merge --no-ff --no-commit origin/dev
```

Expected: merge stops at the five known conflict files. Do not resolve the generated CSS bundle manually.

- [ ] **Step 4: Confirm there are no unexpected conflict sites**

Run:

```bash
git diff --name-only --diff-filter=U
```

Expected: exactly the five known files. If another conflict appears, stop and extend this plan and `TASK-557` before resolving it.

- [ ] **Step 5: Resolve the root test-harness conflict first**

Resolve `Tests/conftest.py` before invoking pytest anywhere else. Keep current `dev`'s call-time `HOME`/XDG isolation and lazy first-run-import pre-arm, plus the citation branch's best-effort shutdown and clearing of already-loaded database/prompt singletons before and after each test. Do not restore the retired `BASE_DATA_DIR_CLI` patch if current `dev` no longer exposes or consumes it.

Run:

```bash
python -m py_compile Tests/conftest.py
rg -n "^(<<<<<<<|=======|>>>>>>>)" Tests/conftest.py
```

Expected: compilation succeeds and the conflict-marker search returns no matches. The other four known merge conflicts remain explicit until their owning tasks below resolve them.

---

### Task 2: Advance Citation Provenance from Schema v26 to v27

**Files:**

- Move: `tldw_chatbook/DB/migrations/chachanotes_v24_to_v25_citation_provenance.sql` → `tldw_chatbook/DB/migrations/chachanotes_v26_to_v27_citation_provenance.sql`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py`
- Modify: `Tests/DB/test_chachanotes_citation_provenance_migration.py`
- Modify: `Tests/DB/test_chachanotes_active_leaf_migration.py`
- Modify if stale citation-version wording is present: `Tests/DB/test_chachanotes_world_book_priority_migration.py`
- Modify if stale citation-version wording is present: `Tests/DB/test_chachanotes_world_book_regex_migration.py`
- Modify as discovered by exact-name search: citation task/plan/benchmark documentation

- [ ] **Step 1: Capture the migration RED boundary and add integrated expectations**

Update migration tests so they require:

```python
assert CharactersRAGDB._CURRENT_SCHEMA_VERSION == 27
assert db.get_schema_version() == 27
```

The v26 upgrade fixture must first contain both current-`dev` migrations, then apply citation provenance. Preserve fail-closed coverage for pre-existing or partial citation tables.

The aggregate merge conflict and old v24→v25 citation migration are the RED state: the database module is intentionally unimportable until the next step establishes one coherent chain. Do not run pytest against a file containing merge markers.

- [ ] **Step 2: Resolve the database migration chain**

In `CharactersRAGDB`:

```python
_CURRENT_SCHEMA_VERSION = 27

migration_functions = {
    # existing migrations unchanged
    24: self._migrate_from_v24_to_v25,
    25: self._migrate_from_v25_to_v26,
    26: self._migrate_from_v26_to_v27,
}
```

Keep current `dev`'s:

- `_migrate_from_v24_to_v25`: message-generation metadata;
- `_migrate_from_v25_to_v26`: conversation context summary and boundary ID.

Rename the citation method to `_migrate_from_v26_to_v27`, load `chachanotes_v26_to_v27_citation_provenance.sql`, and update the schema version from 26 to 27 inside the same transaction only after all citation DDL succeeds.

- [ ] **Step 3: Remove stale version references**

Run:

```bash
rg -n "v24_to_v25_citation|v25_to_v26_citation|v24[[:space:]]*(→|->)[[:space:]]*v25|v25[[:space:]]*(→|->)[[:space:]]*v26|V24.*V25.*citation|V25.*V26.*citation|schema v25|schema v26|version: 25|version: 26" \
  tldw_chatbook Tests Docs backlog
```

Update only citation-foundation references that claim ownership of v25 or v26, including the renamed SQL header and the foundation plan's migration guidance. Do not rewrite historical current-`dev` migration documentation.

- [ ] **Step 4: Run the migration tests in GREEN state**

Run:

```bash
python -m pytest \
  Tests/DB/test_chachanotes_citation_provenance_migration.py \
  Tests/DB/test_chachanotes_active_leaf_migration.py \
  Tests/ChaChaNotesDB/test_chachanotes_db.py -q
```

Expected: all pass, including fresh database creation, sequential v24→v27 upgrade, failure rollback, and newer-schema rejection.

---

### Task 3: Combine Chat Persistence and Test Isolation

**Files:**

- Modify: `tldw_chatbook/Chat/chat_persistence_service.py`
- Modify if required by combined signatures: `Tests/Chat/test_chat_persistence_service.py`
- Modify: `Tests/conftest.py`
- Test: `Tests/test_environment_isolation.py`
- Test: `Tests/UI/test_environment_isolation.py`
- Test: `Tests/RAG/test_first_run_import.py`

- [ ] **Step 1: Resolve chat persistence by behavior**

Preserve current `dev` message creation/update behavior and the citation branch's explicit seams:

```python
sealed_citation: SealedCitationWrite | None
citation_repository: CitationTraceRepository | None
```

Citation preflight must still finish before opening a write transaction. When a sealed citation is supplied, the message, attachments, feedback, and sealed trace must commit in one transaction; uncertain retries must match the exact persisted message and attachments. Non-citation calls must continue through the current `dev` path unchanged.

- [ ] **Step 2: Verify the resolved autouse environment fixture**

Confirm the Task 1 resolution kept current `dev`'s call-time `HOME`/XDG isolation and lazy first-run-import pre-arm, plus the citation branch's best-effort shutdown and clearing of already-loaded database/prompt singletons before and after each test. Do not restore the retired `BASE_DATA_DIR_CLI` patch if current `dev` no longer exposes or consumes it.

- [ ] **Step 3: Run combined persistence tests**

Run:

```bash
python -m pytest \
  Tests/Chat/test_chat_persistence_service.py \
  Tests/Chat/test_citation_trace_repository.py \
  Tests/test_environment_isolation.py \
  Tests/UI/test_environment_isolation.py \
  Tests/RAG/test_first_run_import.py -q
```

Expected: ordinary persistence, citation atomicity/idempotency, and per-test filesystem/database isolation all pass.

---

### Task 4: Regenerate CSS and Resolve Generated Output

**Files:**

- Preserve merged source: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Preserve merged source: `tldw_chatbook/css/features/_scheduling.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`

- [ ] **Step 1: Remove the generated-file conflict**

Take neither side of `tldw_cli_modular.tcss` as authoritative. Leave the merged source TCSS files unchanged unless a source-level conflict is discovered.

- [ ] **Step 2: Rebuild the bundle**

Run:

```bash
python tldw_chatbook/css/build_css.py
```

Expected: successful generation with current `dev` and citation-branch source selectors both present.

- [ ] **Step 3: Verify bundle fidelity**

Run:

```bash
python -m pytest \
  Tests/QA/test_textual_highlight_selectors.py \
  Tests/UI/test_destination_visual_parity_correction.py \
  Tests/UI/test_schedules_workbench.py \
  Tests/UI/test_watchlists_destination_shell.py -q
```

Expected: all pass and no TCSS parse error.

---

### Task 5: Verify, Review, Publish

**Files:**

- Modify: `backlog/tasks/task-557 - Integrate-citation-foundation-with-current-dev-schema.md`
- Modify if version wording changes: `Docs/superpowers/plans/2026-07-23-rag-citation-provenance-foundation.md`
- Modify if version wording changes: `backlog/decisions/024-rag-citation-provenance-and-source-resolution.md`

- [ ] **Step 1: Run the citation foundation gate**

Run the 20-file citation foundation command documented under “Foundation verification gate” in `Docs/superpowers/plans/2026-07-23-rag-citation-provenance-foundation.md`.

Expected: 762 or more tests pass, with no failure.

- [ ] **Step 2: Run integrated database and UI gates**

Run:

```bash
python -m pytest Tests/ChaChaNotesDB/ Tests/DB/ -q
python -m pytest \
  Tests/UI/test_product_maturity_phase6_recovery_docs.py \
  Tests/UI/test_product_maturity_phase6_first_time_release_replay.py \
  Tests/UI/test_unified_shell_phase6_first_time_replay.py \
  Tests/UI/test_destination_visual_parity_correction.py -q
python -m pytest -q
```

Expected: all pass. Run the full suite without another repository-wide pytest process from this worktree; if unrelated worktrees are concurrently saturating shared host resources, wait for them to finish rather than treating a contaminated run as release evidence.

- [ ] **Step 3: Run qualification and static checks**

Run:

```bash
python Helper_Scripts/Benchmarks/rag_citation_provenance_benchmark.py \
  --mode qualification \
  --baseline Docs/Development/RAG/citation-provenance-baseline-v1.json \
  --samples 30 \
  --warmups 5 \
  --output /tmp/rag-citation-foundation-dev-integration.json
python -m ruff check tldw_chatbook/Chat tldw_chatbook/Chatbooks \
  tldw_chatbook/DB Tests/Chat Tests/Chatbooks Tests/DB Tests/Performance
```

Expected: qualification eligible and overall pass; Ruff passes. Git index and diff checks run after explicit staging in Step 5, because resolved merge paths remain classified as unmerged until then.

- [ ] **Step 4: Obtain independent review and prepare Backlog completion**

The reviewer must verify the v24→v27 chain, combined persistence behavior, test-isolation fixture, regenerated CSS, verification evidence, ADR reference, and absence of stale citation-v25 claims. Address all findings, check acceptance criteria 1–4, and add implementation notes. Keep `TASK-557` in progress because criterion 5 requires the pull request to exist.

- [ ] **Step 5: Commit the integration merge**

The merge already stages non-conflicting paths. Explicitly stage every resolved or renamed path, verify the staged tree, and commit the pending merge:

```bash
git add \
  Tests/DB/test_chachanotes_active_leaf_migration.py \
  Tests/DB/test_chachanotes_citation_provenance_migration.py \
  Tests/DB/test_chachanotes_world_book_priority_migration.py \
  Tests/DB/test_chachanotes_world_book_regex_migration.py \
  Tests/Chat/test_chat_persistence_service.py \
  Tests/RAG/test_first_run_import.py \
  Tests/test_environment_isolation.py \
  Tests/UI/test_environment_isolation.py \
  Tests/conftest.py \
  tldw_chatbook/Chat/chat_persistence_service.py \
  tldw_chatbook/DB/ChaChaNotes_DB.py \
  tldw_chatbook/css/tldw_cli_modular.tcss \
  tldw_chatbook/DB/migrations/chachanotes_v24_to_v25_citation_provenance.sql \
  tldw_chatbook/DB/migrations/chachanotes_v26_to_v27_citation_provenance.sql \
  Docs/superpowers/plans/2026-07-23-rag-citation-provenance-foundation.md \
  backlog/decisions/024-rag-citation-provenance-and-source-resolution.md \
  "backlog/tasks/task-557 - Integrate-citation-foundation-with-current-dev-schema.md"
git diff --name-only --diff-filter=U
git diff --check
git diff --cached --check
git diff --exit-code
git status --short
git commit
```

Expected: no unmerged paths, no unstaged reviewed changes, and one merge commit with both parents, preserving the feature branch's existing commit history. If stale-version search changes another tracked citation plan or task, add that exact path to this explicit staging command before committing.

- [ ] **Step 6: Prepare the pull request body**

Use `apply_patch` to create `/tmp/rag-citation-foundation-pr.md` with this structure and the exact final verification counts:

```markdown
## Summary
- Establishes versioned canonical citation trace, identity, locator, and repository contracts.
- Adds governed payload lifecycle, source observations, artifact ownership, and restart-safe legacy migration.
- Integrates with current dev as schema v27 while preserving v25 message metadata and v26 Console summaries.

## Verification
- Citation foundation: <count> passed
- ChaChaNotes/DB: <count> passed
- UI maturity: <count> passed
- Qualification: eligible, overall pass
- Ruff and diff checks: passed

## Limitations
- Canonical producers, server grounding_trace/v1, citation inspector UI, and portable Sync/export remain follow-on work.
- <Any honest remaining full-suite or environment limitation, or "None.">
```

Expected: the file exists, contains no placeholder tokens, and accurately matches the recorded test evidence.

- [ ] **Step 7: Push and create the pull request**

Run:

```bash
git push -u origin codex/rag-citation-provenance-foundation
gh pr create \
  --base dev \
  --head codex/rag-citation-provenance-foundation \
  --title "feat(rag): establish canonical citation provenance foundation" \
  --body-file /tmp/rag-citation-foundation-pr.md
```

Create a ready PR, not a draft.

- [ ] **Step 8: Complete TASK-557 and update the pull request branch**

After the pull request exists, check acceptance criterion 5, append the PR URL and publication result to Implementation Notes, mark `TASK-557` `Done`, commit the Backlog completion, and push it to the same branch:

```bash
git add \
  "backlog/tasks/task-557 - Integrate-citation-foundation-with-current-dev-schema.md"
git diff --cached --check
git commit -m "docs(backlog): complete citation dev integration"
git push
```

Expected: the ready pull request updates automatically and the branch returns clean.
