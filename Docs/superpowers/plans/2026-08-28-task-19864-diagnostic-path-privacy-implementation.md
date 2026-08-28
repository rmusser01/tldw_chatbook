# TASK-19864 Diagnostic Path Privacy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use
> `superpowers:test-driven-development` for every production change and
> `superpowers:verification-before-completion` before any completion claim.

**Goal:** Remove raw filesystem identities from the five TASK-19864 diagnostic
owners, fold TASK-19936 into that repair, and make any new path-shaped diagnostic
an explicit whole-set architecture-gate failure.

**Architecture:** Keep privacy enforcement at logger call sites. Extend the existing
AST diagnostic inventory with one path-candidate projection, preserving its current
owner/sink APIs and report. Represent unresolved legacy candidates explicitly rather
than treating them as safe. Use existing `content_fingerprint` and
`redact_user_paths` seams; add no logging wrapper or dependency.

**Tech stack:** Python 3.11+, stdlib `ast`, Loguru, pytest, Ruff, generated JSON

---

## Baseline and scope

- Worktree: `.worktrees/task-19864-diagnostic-path-privacy`
- Branch: `codex/task-19864-diagnostic-path-privacy`
- Planning base: `origin/dev` `39957143e8`
- Approved design commit after rebase: `25862ee34d`
- Five owners: `file_handlers.py`, `ChaChaNotes_DB.py`,
  `change_review_screen.py`, `console_conversation_inspector.py`, and
  `git_workspace.py`
- Current census: 11 diagnostics in `file_handlers.py`; 94 direct database/backup
  path-expression occurrences in `ChaChaNotes_DB.py`; 11 path/root-bearing Change
  Review diagnostics; 3 Inspector save/export diagnostics; 1 Git detection diagnostic.
- Inherited base failure: the existing inventory checker is already red because PR
  #2168 added two unpinned raw-exception diagnostics in
  `Agents/virtual_cli_provider.py`. The focused architecture baseline is 64 passed / 1
  failed for that row only. Treat it as reviewed base drift, not TASK-19864 output.
- Do not run the full repository suite unless the user explicitly opts in. The final
  evidence set is targeted to modified modules plus architecture/privacy gates.

## File map

### Production and tooling

- Modify `scripts/check_persistent_diagnostic_inventory.py`: path-expression scan,
  schema-v3 projection, complete diff reporting.
- Modify `tldw_chatbook/Utils/file_handlers.py`: remove 11 raw path/basename records.
- Modify `tldw_chatbook/DB/ChaChaNotes_DB.py`: one per-instance database reference,
  backup references, mechanical logger-only substitutions, safe failure metadata.
- Modify `tldw_chatbook/UI/Screens/change_review_screen.py`: root-set fingerprints,
  exception types, no path-bearing tracebacks.
- Modify `tldw_chatbook/Widgets/Console/console_conversation_inspector.py`: safe
  export/save diagnostics without changing notifications.
- Modify `tldw_chatbook/Workspaces/git_workspace.py`: safe detection failure record.
- Modify `tldw_chatbook/Agents/virtual_cli_provider.py`: repair inherited raw-exception
  inventory drift.
- Modify `Docs/security/production-diagnostic-inventory.json`: generated schema-v3
  owner, sink, and path-candidate pin.

### Tests and fixtures

- Create `Tests/Architecture/test_diagnostic_path_privacy.py`: scanner, owned-file,
  mutation, multiplicity, and reporting contracts.
- Create `Tests/Utils/test_file_handler_diagnostic_privacy.py`: representative real
  FileHandlerRegistry log records.
- Create `Tests/DB/test_chachanotes_diagnostic_path_privacy.py`: real initialization
  and path-bearing failure records.
- Modify `Tests/UI/test_change_review_current_mode.py`: raw configured/CWD failure and
  worker-root sentinel coverage.
- Modify `Tests/UI/test_console_conversation_inspector.py`: save/export diagnostic
  privacy without changing visible recovery copy.
- Modify `Tests/Workspaces/test_git_workspace_detection.py`: detection error sentinel.
- Modify `Tests/Agents/test_virtual_cli_provider.py`: inherited drift regression.
- Modify `Tests/test_logs_share_path_privacy.py`: pin the still-accurate Copy visible
  disclosure.
- Modify `Tests/Architecture/test_persistent_diagnostic_inventory.py`: schema-v3 and
  new path-projection integration assertions only; preserve TASK-15103 historical
  ledgers unchanged.
- Modify `Tests/fixtures/summarization_diagnostic_review.json`: restamp the normalized
  inventory projection after the reviewed schema change.

### Governance

- Modify `Docs/superpowers/specs/2026-08-28-diagnostic-path-privacy-and-guard-design.md`:
  committed clarification that path-bearing traceback capture is unsafe.
- Modify `backlog/tasks/task-19864 - Diagnostics interpolate user file paths and workspace roots into log text.md`:
  checked criteria and implementation notes only after verification.
- Modify `backlog/tasks/task-19936 - change-review-debug-line-interpolates-the-raw-console-workspace-root-path.md`:
  mark Done with a note that TASK-19864 folded and verified it; do not duplicate
  implementation notes.

## Plan handoff prerequisite

Commit this plan, the approved-spec clarification, and the task's Implementation Plan
before touching tests or production code. TASK-19864 is five digits, so edit its file
directly; do not use `backlog task edit`.

```bash
git add \
  Docs/superpowers/specs/2026-08-28-diagnostic-path-privacy-and-guard-design.md \
  Docs/superpowers/plans/2026-08-28-task-19864-diagnostic-path-privacy-implementation.md \
  "backlog/tasks/task-19864 - Diagnostics interpolate user file paths and workspace roots into log text.md"
git commit -m "docs: plan diagnostic path privacy implementation"
git status --short --branch
```

Expected: planning commit succeeds and the worktree is clean.

### Task 1: Pin the AST path scanner RED

**Files:**
- Create `Tests/Architecture/test_diagnostic_path_privacy.py`
- Reference `scripts/check_persistent_diagnostic_inventory.py`

- [ ] Add a synthetic-source matrix covering f-strings, Loguru positional and keyword
  arguments, `%` formatting, `.format(...)`, multiline calls, `row.get("root")`, and
  `self.db_path_str`.
- [ ] Add simple assignment-taint cases: `raw = configured or os.getcwd()`, then log
  `raw`; `target = validate_path_simple(raw)`, then log `target`.
- [ ] Add safe-transform cases for `content_fingerprint(path)`,
  `redact_user_paths(path)`, `.suffix`, `len(paths)`, and `type(exc).__name__`.
- [ ] Add negative names whose substrings resemble paths but are not bounded path
  identifiers, such as `root_cause` and `directory_count`.
- [ ] Add a reporting mutation with two files and three findings, including two
  content-identical calls, and assert that every finding and its multiplicity appears.
- [ ] Run only the new tests and confirm RED because the path-scan API does not exist.

```bash
../../.venv/bin/python -B -m pytest \
  Tests/Architecture/test_diagnostic_path_privacy.py -q
```

Expected: collection or assertions fail at the missing scanner seam. Do not write
production code until the failure is observed.

### Task 2: Implement the scanner and schema projection GREEN

**Files:**
- Modify `scripts/check_persistent_diagnostic_inventory.py`
- Modify `Tests/Architecture/test_diagnostic_path_privacy.py`
- Modify `Tests/Architecture/test_persistent_diagnostic_inventory.py`

- [ ] Add a small internal parsed-source scan shared by `scan_source` and a new
  `scan_path_diagnostic_candidates` function. Preserve `scan_source`'s existing
  two-tuple return contract.
- [ ] Split identifiers on snake-case boundaries and classify only bounded terminal
  tokens (`path`, `paths`, `root`, `roots`, `dir`, `directory`, `folder`) plus explicit
  `*_path_str` forms. Treat `.get("root")`-style literal keys as identifiers.
- [ ] Compute assignment taint to a fixed point independently within each lexical
  scope for simple `Assign`/`AnnAssign` values and known path producers (`Path`,
  `os.getcwd`, `Path.home`, `.resolve`, and `validate_path*`). Do not let aliases
  bleed across functions or classes, and do not add cross-module inference.
- [ ] Extract dynamic expressions from all Task-1 formatting forms. A diagnostic is one
  candidate when any dynamic expression remains path-tainted after safe-transform
  recognition. Record method, call digest, scope, path-expression labels, and
  `status="legacy_unreviewed"`; preserve duplicate entries as a list.
- [ ] Advance the generated inventory to schema version 3. Add
  `path_privacy_rules`, `path_privacy_candidates`, and
  `summary.path_privacy_candidate_calls`. Do not mutate TASK-15103's historical ledger
  schema or historical blobs.
- [ ] Add a path-candidate diff section that uses counted identities and emits every
  addition/removal/change. Add the new rules key to metadata reporting and explain that
  baseline candidates are unresolved, not approved.
- [ ] Run formatter, the new scanner suite, and the existing digest/sink unit tests.

```bash
../../.venv/bin/python -B -m ruff format \
  scripts/check_persistent_diagnostic_inventory.py \
  Tests/Architecture/test_diagnostic_path_privacy.py \
  Tests/Architecture/test_persistent_diagnostic_inventory.py
../../.venv/bin/python -B -m pytest \
  Tests/Architecture/test_diagnostic_path_privacy.py \
  Tests/Architecture/test_persistent_diagnostic_inventory.py::test_moving_a_logger_call_within_a_file_does_not_change_the_digest \
  Tests/Architecture/test_persistent_diagnostic_inventory.py::test_editing_a_diagnostic_changes_the_digest \
  Tests/Architecture/test_persistent_diagnostic_inventory.py::test_duplicate_diagnostics_are_digested_with_multiplicity \
  Tests/Architecture/test_persistent_diagnostic_inventory.py::test_sink_entries_are_position_independent_but_still_content_sensitive -q
```

Expected: scanner tests pass; the whole-tree inventory pin remains intentionally red
until Task 7 regenerates it.

### Task 3: Pin the owner violations and runtime leaks RED

**Files:**
- Modify `Tests/Architecture/test_diagnostic_path_privacy.py`
- Create `Tests/Utils/test_file_handler_diagnostic_privacy.py`
- Create `Tests/DB/test_chachanotes_diagnostic_path_privacy.py`
- Modify the three UI/workspace test modules from the file map

- [ ] Add `TASK_19864_OWNER_PATHS` and a test that scans each complete source file,
  reports the full per-file candidate set, and requires zero candidates after repair.
  Against current code it must fail with all five owners named.
- [ ] Capture real Loguru output with a list-appending sink, always removed in `finally`.
  Inspect the complete rendered record, not `capsys`.
- [ ] File handlers: drive one path-bearing exception and the no-handler/not-found path;
  assert the event survives while the absolute path, basename, and raw exception text do
  not.
- [ ] ChaChaNotes: initialize a real temporary database and inject one connection/backup
  failure whose exception repeats the path. Assert stable correlation metadata and no
  path in any emitted record.
- [ ] Change Review: reuse the current-mode harness. Force
  `validate_path_simple` and one worker operation to raise exceptions containing a
  distinctive workspace path. Assert event label, fingerprint, exception type, no path,
  and no traceback-local disclosure.
- [ ] Inspector and Git detection: drive their existing private/public seams with
  path-bearing exceptions and pin both safe log output and unchanged user-facing notify
  behavior.
- [ ] Run the exact new nodes and observe RED against current production code.

### Task 4: Repair file and database diagnostics GREEN

**Files:**
- Modify `tldw_chatbook/Utils/file_handlers.py`
- Modify `tldw_chatbook/DB/ChaChaNotes_DB.py`
- Test the two new privacy modules

- [ ] Import `content_fingerprint` from `Utils.log_sanitizer` in both owners.
- [ ] In file-handler diagnostics, emit `path_sha256`, suffix where useful, handler/type
  metadata, and `type(exc).__name__`. Remove `.name` because a basename remains user data.
  Do not change `ProcessedFile` content or display names.
- [ ] In `CharactersRAGDB.__init__`, compute one
  `self._db_diagnostic_ref`: fixed `memory` for `:memory:`, otherwise
  `content_fingerprint(self.db_path_str)`. This is a per-instance cached value, not a new
  public path abstraction.
- [ ] Replace only logger-call uses of `self.db_path_str` with labelled
  `db_sha256=<ref>`. Keep filesystem, SQLite, exception, backup, and return-value uses of
  `db_path_str` unchanged.
- [ ] Fingerprint `backup_file_path` separately. For changed path-bearing failures,
  replace raw exception interpolation and `opt(exception=True)` with explicit exception
  type so tracebacks cannot recover the path from locals.
- [ ] Use the AST owner gate to prove all 94 current database/backup path expressions
  were handled; do not rely on a text replacement count alone.
- [ ] Run the two new privacy modules plus the existing initialization, backup,
  integrity, and sync-retention nodes they exercise.

### Task 5: Repair Change Review, Inspector, and Git diagnostics GREEN

**Files:**
- Modify `tldw_chatbook/UI/Screens/change_review_screen.py`
- Modify `tldw_chatbook/Widgets/Console/console_conversation_inspector.py`
- Modify `tldw_chatbook/Workspaces/git_workspace.py`
- Test the three existing modules from Task 3

- [ ] Import and inline `content_fingerprint` at diagnostic call sites. For a tuple of
  roots, fingerprint a stable tuple representation rather than joining it into plaintext.
- [ ] Preserve operation identity (`badge`, `status`, `commit`, `push`, `pr`,
  `untracked-writable`) and exception type. Remove raw root/path and
  `opt(exception=True)` from each path-bearing failure.
- [ ] Keep remote refusal behavior and user notifications unchanged; only the diagnostic
  text is private.
- [ ] TASK-19936: replace `raw!r` and `exc` with the fixed disclosure event plus
  `exception_type`; keep the early return and visible banner behavior unchanged.
- [ ] Inspector: use a fingerprint in logs, while visible save-failure notifications
  continue naming the destination selected by the user.
- [ ] Git workspace detection: emit root fingerprint and exception type, never raw root
  or exception text.
- [ ] Run the focused privacy tests and the existing behavior nodes touched by each seam.

### Task 6: Repair and classify inherited virtual-CLI drift

**Files:**
- Modify `Tests/Agents/test_virtual_cli_provider.py`
- Modify `tldw_chatbook/Agents/virtual_cli_provider.py`

- [ ] Add a parameterized born-red test for `_persist` and `_record` callbacks raising
  exceptions whose messages carry a path/secret sentinel.
- [ ] Preserve the two event labels and emit only `type(exc).__name__`; no traceback.
- [ ] Run the focused virtual-CLI provider tests. Record in implementation notes that
  this was red on untouched base and repaired because inventory regeneration cannot
  safely bless raw exception text.

### Task 7: Review and regenerate governed artifacts

**Files:**
- Modify `Docs/security/production-diagnostic-inventory.json`
- Modify `Tests/fixtures/summarization_diagnostic_review.json`
- Modify `Tests/test_logs_share_path_privacy.py`

- [ ] Format all modified Python before regeneration; the diagnostic digest is
  indentation-sensitive.
- [ ] Run `--statements ... --since 5688276e9d` for every changed diagnostic owner.
  Classify the five intended owner deltas and the inherited virtual-CLI row separately.
- [ ] Add a Logs-screen assertion that **Copy visible logs** still warns that file names
  and search terms may remain. Do not weaken **Copy all (redacted)** metadata-only tests.
- [ ] Run the checker without `--write` and review the complete owner, schema, and
  path-candidate report. Confirm no TASK-19864 owner appears in
  `path_privacy_candidates`.
- [ ] Run the checker once with `--write`. Review the JSON diff; schema is exactly 3,
  every baseline row says `legacy_unreviewed`, multiplicity is preserved, and no sink
  topology changed.
- [ ] Recompute the normalized inventory SHA through the existing helper in
  `Tests/LLM_Calls/test_summarization_diagnostic_privacy.py`; patch only the two expected
  fixture hash fields after asserting the old hash occurs exactly twice.
- [ ] Run the checker, full focused architecture file, new path architecture file,
  summarization diagnostic privacy suite, and Logs share-path suite.

### Task 8: Final verification, review, and backlog closeout

**Files:**
- Modify both TASK-19864 and TASK-19936 task files
- Review all modified files

- [ ] Run Ruff check/format-check on every modified Python file and `git diff --check`.
- [ ] Run the focused owner behavior tests, scanner tests, full diagnostic architecture
  file, summarization diagnostic privacy suite, and Logs share-path suite. Do not claim a
  full-suite result.
- [ ] Mutation-check at least these regressions and restore each immediately: add two raw
  path diagnostics (both must be reported); restore one `self.db_path_str`; restore
  TASK-19936's `raw!r`; restore one `opt(exception=True)` path failure. Each relevant test
  must go red.
- [ ] Review `git diff origin/dev...HEAD` for accidental non-logger database changes,
  user-facing copy changes, generated-artifact noise, and raw path/exception remnants in
  the five owner sources.
- [ ] Check every TASK-19864 criterion only after evidence exists. Add concise
  Implementation Notes including the corrected exposure scope, inherited drift, schema
  change, test/mutation results, and modified files.
- [ ] Mark TASK-19936 Done as folded into TASK-19864, with its inventory and architecture
  acceptance evidence linked. Mark TASK-19864 Done only when every DoD item is complete.

## Final focused command set

```bash
../../.venv/bin/python -B -m ruff check \
  scripts/check_persistent_diagnostic_inventory.py \
  tldw_chatbook/Utils/file_handlers.py \
  tldw_chatbook/DB/ChaChaNotes_DB.py \
  tldw_chatbook/UI/Screens/change_review_screen.py \
  tldw_chatbook/Widgets/Console/console_conversation_inspector.py \
  tldw_chatbook/Workspaces/git_workspace.py \
  tldw_chatbook/Agents/virtual_cli_provider.py \
  Tests/Architecture/test_diagnostic_path_privacy.py \
  Tests/Architecture/test_persistent_diagnostic_inventory.py \
  Tests/Utils/test_file_handler_diagnostic_privacy.py \
  Tests/DB/test_chachanotes_diagnostic_path_privacy.py \
  Tests/UI/test_change_review_current_mode.py \
  Tests/UI/test_console_conversation_inspector.py \
  Tests/Workspaces/test_git_workspace_detection.py \
  Tests/Agents/test_virtual_cli_provider.py \
  Tests/test_logs_share_path_privacy.py
../../.venv/bin/python -B -m ruff format --check \
  scripts/check_persistent_diagnostic_inventory.py \
  tldw_chatbook/Utils/file_handlers.py \
  tldw_chatbook/DB/ChaChaNotes_DB.py \
  tldw_chatbook/UI/Screens/change_review_screen.py \
  tldw_chatbook/Widgets/Console/console_conversation_inspector.py \
  tldw_chatbook/Workspaces/git_workspace.py \
  tldw_chatbook/Agents/virtual_cli_provider.py \
  Tests/Architecture/test_diagnostic_path_privacy.py \
  Tests/Architecture/test_persistent_diagnostic_inventory.py \
  Tests/Utils/test_file_handler_diagnostic_privacy.py \
  Tests/DB/test_chachanotes_diagnostic_path_privacy.py \
  Tests/UI/test_change_review_current_mode.py \
  Tests/UI/test_console_conversation_inspector.py \
  Tests/Workspaces/test_git_workspace_detection.py \
  Tests/Agents/test_virtual_cli_provider.py \
  Tests/test_logs_share_path_privacy.py
../../.venv/bin/python -B scripts/check_persistent_diagnostic_inventory.py --diff
../../.venv/bin/python -B -m pytest \
  Tests/Architecture/test_diagnostic_path_privacy.py \
  Tests/Architecture/test_persistent_diagnostic_inventory.py \
  Tests/Utils/test_file_handler_diagnostic_privacy.py \
  Tests/DB/test_chachanotes_diagnostic_path_privacy.py \
  Tests/UI/test_change_review_current_mode.py \
  Tests/UI/test_console_conversation_inspector.py \
  Tests/Workspaces/test_git_workspace_detection.py \
  Tests/Agents/test_virtual_cli_provider.py \
  Tests/test_logs_share_path_privacy.py \
  Tests/LLM_Calls/test_summarization_diagnostic_privacy.py -q
git diff --check
git status --short --branch
```
