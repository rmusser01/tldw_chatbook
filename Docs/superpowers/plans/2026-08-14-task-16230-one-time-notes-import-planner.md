# One-time Database Notes Import Planner and Preview Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task by task.

**Goal:** Build a read-only, immutable import plan that safely previews how individual files or one recursive folder would become Database Notes and manual folder memberships before anything is persisted.

**Architecture:** Add a small domain-model module and a filesystem-facing planner under `tldw_chatbook/Notes/`. The planner reads a bounded selection, parses supported formats without using the legacy path-logging importer, compares private in-memory fingerprints with caller-supplied prior observations, and returns frozen preview values. Pure transformations handle root collisions, uncertain-match confirmation, and per-item overrides; database repositories, receipts, configuration, and UI remain outside this task.

**Tech Stack:** Python 3.11+, frozen dataclasses, enums, `pathlib`/`os.scandir`, standard-library JSON/CSV/hash support, optional existing YAML dependency, pytest.

---

### Task 1: Define the immutable preview vocabulary

**Files:**

- Create: `tldw_chatbook/Notes/note_import_plan_models.py`
- Test: `Tests/Notes/test_note_import_planner.py`

- [ ] Write tests that construct preview payloads, sources, memberships, matches, collision state, and bounds and prove nested values cannot be mutated.
- [ ] Run `../../.venv/bin/python -m pytest -q Tests/Notes/test_note_import_planner.py` and confirm the test fails because the models do not exist.
- [ ] Add enums and frozen dataclasses with tuple-backed collections and validation for classification/action combinations.
- [ ] Re-run the focused test and confirm the model tests pass.

### Task 2: Validate selections and discover sources safely

**Files:**

- Create: `tldw_chatbook/Notes/note_import_planner.py`
- Modify: `Tests/Notes/test_note_import_planner.py`

- [ ] Add failing tests for multiple-file selection, exactly one directory, rejection of mixed inputs, selected and nested links, missing/non-regular inputs, depth/file/byte bounds, deterministic traversal, root inclusion, and no filesystem mutation.
- [ ] Run the focused tests and confirm they fail for missing planner behavior.
- [ ] Implement bounded discovery with non-following directory scans, explicit file checks, deterministic relative paths, and user-safe categorized failures.
- [ ] Re-run the selection/discovery tests and confirm they pass.

### Task 3: Parse sources and preserve hierarchy

**Files:**

- Create: `tldw_chatbook/Notes/note_import_discovery.py`
- Create: `tldw_chatbook/Notes/note_import_parsers.py`
- Modify: `tldw_chatbook/Notes/note_import_planner.py`
- Modify: `Tests/Notes/test_note_import_planner.py`

- [ ] Move the approved descriptor-based discovery implementation behind a dedicated module while retaining the planner's public discovery imports, so later format and matching logic cannot obscure the path-safety boundary.
- [ ] Add failing tests for text/Markdown and structured JSON/YAML/CSV sources, multi-note results, unsupported extensions, malformed inputs, empty branches, and parent-folder membership placement.
- [ ] Run the focused tests and confirm the parsing/hierarchy cases fail.
- [ ] Implement bounded, descriptor-verified in-memory parsers in the dedicated parser module; reopen each leaf through its recorded parent identities, use `O_NOFOLLOW`, enforce the observed size bound while reading, and recheck identity before and after the read.
- [ ] Normalize immutable note payloads without logging content or exception text, and assign every payload from one source to the source parent folder.
- [ ] Build only folders needed by importable items, including the selected directory as the top-level proposed folder.
- [ ] Re-run the parsing/hierarchy tests and confirm they pass.

### Task 4: Classify repeats and enforce safe defaults

**Files:**

- Modify: `tldw_chatbook/Notes/note_import_planner.py`
- Modify: `Tests/Notes/test_note_import_planner.py`

- [ ] Add failing tests for new, unchanged repeat, changed repeat, uncertain match, unsupported, and failed classifications with their required defaults and allowed actions.
- [ ] Add failing tests proving private fingerprints, absolute paths, source content, and raw exception text are absent from public diagnostics and log records.
- [ ] Implement private in-memory comparison against caller-supplied observations: unchanged defaults to Skip, changed defaults to Create new with Update allowed, uncertain defaults to Create new without Update, and unsupported/failed default to Skip.
- [ ] Re-run the classification/privacy tests and confirm they pass.

### Task 5: Resolve collisions and apply immutable overrides

**Files:**

- Modify: `tldw_chatbook/Notes/note_import_planner.py`
- Modify: `Tests/Notes/test_note_import_planner.py`

- [ ] Add failing tests that require explicit use-existing, unique-sibling, or renamed-root resolution when a top-level label collides, and prove no silent merge occurs.
- [ ] Add failing tests for Skip/Create new overrides, Update restrictions, explicit uncertain-match confirmation, and independent content-replacement/folder-membership choices.
- [ ] Implement pure functions that return replaced frozen plans/items while validating collision names and action eligibility.
- [ ] Re-run the collision/override tests and confirm they pass.

### Task 6: Verify the boundary and close out the task

**Files:**

- Modify: `backlog/tasks/task-16230 - Build-one-time-Database-Notes-import-planner-and-preview-model.md`

- [ ] Run `../../.venv/bin/python -m pytest -q Tests/Notes/test_note_import_planner.py`.
- [ ] Run the focused Notes/Library regression gate used at baseline.
- [ ] Run the relevant formatter/linter/type checks available in the repository for the new modules and test.
- [ ] Review the diff for filesystem writes, repository/service dependencies, sensitive logging, symlink traversal, unbounded reads, mutable nested values, and behavior outside the acceptance criteria.
- [ ] Update acceptance criteria, Definition of Done, and concise Implementation Notes with exact verification evidence and the ADR decision.
- [ ] Mark TASK-16230 Done only after all completion checks pass.
