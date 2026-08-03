# Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver TASK-197 as PR 5 of 6: serialize modern Prompt/Recipe records losslessly in local Chatbooks, import them safely, and add complete Prompts-only/Everything/explicit-ID Library export scopes.

**Architecture:** Version each Prompt record independently from artifact schema. Keep canonical System/User/metadata/definition fields plus an exact lossy legacy projection. Validate manifest paths and modern records before writes. Extend Library export scope resolution with uncapped local Prompt IDs while leaving archive serialization in `Chatbooks/` and preserving the server export gate.

**Tech Stack:** Python 3.11+, JSON/ZIP Chatbook service, SQLite, Textual export/import flows, pytest, Backlog.md CLI.

---

## Merge Gate and ADR

- Begin only after TASK-199 is merged into latest `origin/dev`; create a fresh worktree/branch.
- ADR required: yes.
- ADR path: allocate the next free number as `backlog/decisions/NNN-chatbook-prompt-record-contract.md`.
- Reason: this changes the portable Prompt archive record and import/export service contract.

## File Responsibility Map

- Modify `tldw_chatbook/Chatbooks/chatbook_creator.py`: modern record serialization and legacy projection.
- Modify `tldw_chatbook/Chatbooks/chatbook_importer.py`: manifest-item lookup, contained path validation, record dispatch/validation, conflict outcomes.
- Modify `tldw_chatbook/Chatbooks/chatbook_models.py` only for typed record/preview metadata needed by the existing manifest flow.
- Modify `tldw_chatbook/Chatbooks/local_chatbook_service.py`: exact local selection/outcome integration and no-empty-finalization behavior.
- Create a focused pure codec module such as `tldw_chatbook/Chatbooks/prompt_record_codec.py` if needed to keep schema validation and legacy projection independently testable.
- Modify `tldw_chatbook/Library/library_export_scope.py`, `library_export_state.py`, `Widgets/Library/library_export_canvas.py`, and `UI/Screens/library_screen.py`: Prompt scope/count/selection and UI entry.
- Add/modify Chatbook tests and Library export/round-trip/UI tests.
- Modify `tldw_chatbook/Chatbooks/CHATBOOKS_GUIDE.md`, `Docs/User_Guide/library/import-and-export.md`, `Docs/User_Guide/library/prompts.md`, TASK-197, and the ADR.

## Task 1: Refresh Baseline and Record the Portable Contract

- [ ] Confirm TASK-199 merge on `origin/dev`; create `codex/task-197-chatbook-prompt-records`.
- [ ] Allocate/write the ADR with record-version dispatch, exact fields, exact compatibility projection, semantic identity exclusions, modern fail-closed validation, manifest path authority, conflict behavior, local-only producer scope, and atomic/no-empty finalization.
- [ ] Mark TASK-197 In Progress and expand criteria to the approved modern/legacy/complete-scope/import-preview outcomes. Link exact ADR path/reason.
- [ ] Commit ADR + Backlog plan before implementation.

## Task 2: Define a Pure Versioned Prompt Record Codec

- [ ] Add tests for `chatbook_prompt_record_version == 1` with name/author/details, separate lanes, ordered keywords, artifact type, Prompt format/schema/definition, and no local ID/version/sync/usage/collection fields.
- [ ] Add exact compatibility projection tests, including empty lanes, existing trailing newlines, CRLF preserved inside lanes, and inserted LF delimiters:

```python
content = "### SYSTEM ###\n" + system_prompt + "\n### USER ###\n" + user_prompt + "\n"
```

- [ ] Add validation tests for legacy missing-version fallback; modern missing artifact type; unknown record version; type/kind mismatch; structured-v1 preservation; supported-v2 fidelity; compiled/definition mismatch; invalid keywords; size limits; future type/schema.
- [ ] Implement a pure encoder/decoder/validator (new `prompt_record_codec.py` if it improves separation). Modern readers must never parse `content`; legacy readers keep the old flattened fallback.
- [ ] Run codec tests green.

## Task 3: Serialize Prompt/Recipe Records Without Loss

- [ ] Add creator tests for legacy Prompt, structured-v2 Prompt, Recipe, foreign-v1 record, both lanes, ordered keywords, and exact manifest `file_path`.
- [ ] Replace `_collect_prompts` collapsed-record construction with the modern codec. Derive filenames safely inside the exporter; write through the existing work directory/atomic archive flow.
- [ ] Skip concurrently missing/deleted/invalid IDs with bounded metadata-only outcomes; do not log bodies/definitions.
- [ ] Ensure an all-disappeared Prompts-only selection cannot finalize an empty archive, while Everything may continue if other content remains. If every selected type disappears, finalize nothing.

## Task 4: Validate Manifest Mapping Before Import Writes

- [ ] Add hostile-manifest tests for duplicate Prompt IDs, duplicate file paths, type/path mismatch, missing file, absolute/parent traversal/symlink escape, and ID that attempts to influence a filename.
- [ ] Build an index from Prompt `ContentItem`s and resolve each via `ContentItem.file_path` plus `_safe_manifest_relative_path`; never construct `prompt_<manifest-id>.json` for import lookup.
- [ ] Reject affected duplicates/path mismatches before parsing or writing any affected Prompt record. Bound error detail and keep content out of logs.
- [ ] Add modern per-item transaction tests proving invalid schema/keywords/size/name conflict leaves no partial Prompt row or keyword links.

## Task 5: Implement Modern and Legacy Import Outcomes

- [ ] Add round-trip tests into a fresh Prompts DB for legacy Prompt, v2 Prompt, Recipe, separate lanes, metadata, keywords, and complete definition. Assert semantic equality but fresh local identity/current version.
- [ ] Add conflict-policy tests for overwrite/rename/skip and unresolved `ASK`. `ASK` must report the conflict without choosing a mutation in the worker.
- [ ] Implement record-version dispatch in `_import_prompts`; modern records use canonical fields and ordinary validated Prompt create/update seams. Legacy no-version records retain current collapsed-content-to-System behavior.
- [ ] Ensure prefix/rename modifies only the imported name and structured-v1 remains preserved under ADR-040 compatibility rules.
- [ ] Report imported, renamed, skipped, and failed Prompt/Recipe records distinctly.

## Task 6: Extend Library Export Scope Completely

- [ ] Add red `Tests/Library/test_library_export_scope.py` cases for `ExportScope(kind="prompts")`, Everything including Prompts, explicit IDs, invalid kinds/IDs, and exact count/selection resolution.
- [ ] Use a fresh uncapped local DB/service query for Prompt IDs; never use the current browse page. Cover more than 100 rows.
- [ ] Extend scope protocols/count/result maps from media/conversations/notes to prompts while preserving existing callers.
- [ ] Add state/execution tests for deleted-after-resolution skips, Prompts-only all gone, Everything with other survivors, Everything all gone, cancellation, failure, and atomic partial-file replacement.
- [ ] Keep the existing server-mode export gate; do not claim server-created archives use record v1.

## Task 7: Add Prompt Export and Import Preview UI

- [ ] Add UI tests that Prompt list Export opens the existing export canvas pre-scoped to Prompts, Everything includes Prompt/Recipe count, and explicit IDs can be supplied for TASK-203.
- [ ] Add import preview/selection tests for combined Prompt/Recipe count and include/skip content type.
- [ ] Show bounded partial outcomes and no-empty-finalization errors honestly; do not route Chatbooks through the Markdown Prompt importer.
- [ ] Preserve prompt bodies/definitions as literal non-markup text wherever preview content appears.

## Task 8: Documentation and Verification

- [ ] Document the v1 record, exact compatibility projection, semantic round-trip exclusions, legacy fallback, safe manifest lookup, local-only producer scope, conflict behavior, and Prompt export scopes.
- [ ] Run affected suites (adjust exact file names to latest merged dev if tests were reorganized).

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chatbooks Tests/Library/test_library_export_scope.py Tests/Library/test_library_export_state.py Tests/Library/test_library_export_execution.py Tests/UI/test_library_prompts_canvas.py Tests/UI/test_library_shell.py -q
git diff --check
```

- [ ] Run a real round-trip smoke archive into a fresh temporary DB and inspect manifest/Prompt record without exposing body content in review logs.
- [ ] Run the full suite, self-review for traversal/partial writes/lossy canonical reads/empty archives, and request independent review.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest
```

- [ ] Complete TASK-197 criteria/notes with ADR and verification, then mark Done only after DoD.
- [ ] Open one ready PR against `dev`, resolve CI/review, merge, and verify on `origin/dev`. Do not begin TASK-203 implementation before confirmation.
