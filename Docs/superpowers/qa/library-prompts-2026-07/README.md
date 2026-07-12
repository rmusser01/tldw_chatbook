# Library ▸ Prompts — Phase 1 QA evidence (2026-07-12)

Branch: `claude/prompts-library-spec` (worktree at origin/dev @ 5a8720c8).
Spec: `Docs/superpowers/specs/2026-07-12-library-prompts-console-injection-design.md`.
Plan Phase 1 (Tasks 0–8). Captured live from textual-serve (real app CSS, worktree code),
playwright bundled chromium, viewport 2050×1240, isolated profile with 4 real prompts seeded via
`PromptsDatabase.add_prompt` into `…/default_user/tldw_chatbook_prompts.db` (note: NOT the
config-documented default path — the app uses the per-user data dir).

- `rail-row-and-list-2026-07-12.png` — Browse ▸ **Prompts (4)** rail row and the list canvas:
  filter input, single-row `sort · Import… · Export…` toolbar, rows with author · relative age,
  and the escaped bracket title `[draft] Bug triage [wip]` rendered verbatim (markup-escape proof).
- `editor-two-part-multiline-body-2026-07-12.png` — the two-part editor (Name/Author/Details/
  System prompt/User prompt/Keywords) opened on a multi-KB prompt; `Modified 17m · v1` meta line
  (no Created — prompts have no created_at); action row `Save · Use in Console · Export… · Copy ·
  Delete`.
- `save-name-in-use-outcome-2026-07-12.png` — renaming a prompt to an existing name and pressing
  Save shows the exact status copy **"Name already in use — pick another or open the existing
  prompt."** (one of the three distinct save outcomes; soft-deleted-name and optimistic conflict
  are the other two, unit-proven against a real DB).
- `import-outcome-imported-skipped-2026-07-12.png` — folder import: outcome line **"1 imported ·
  1 skipped (duplicate name)"**, count 4→5, the new `Imported greeting prompt` (multi-line SYSTEM
  body) at the top — exercising the round-trip-fixed markdown parser end-to-end.
- `search-source-plural-hit-2026-07-12.png` — Library Search with the fourth source **✓ Prompts
  (4)**: query "feedback loops" matches the "Feedback loop explainer" prompt via the task-185
  plural/singular expansion; Open/Select evidence per-result actions present.
- `prompts-alias-to-library-2026-07-12.png` — the legacy `prompts` route resolves to Library
  ("Switched to Library").

## Live defect found and fixed at the gate (Task 8 fix wave 1, commit 3a2bf5b5)

Opening a prompt row raised `PolicyDeniedError: Unknown runtime-policy action_id:
prompts.detail.local` — the capability registry was missing `prompts.detail.*`, so `get_prompt`
was denied and the editor silently fell back to the list. Every UI test had passed because the
fixtures used a permissive/None policy enforcer, never the real one. Fix: registered the missing
`DETAIL` capability, plus a full audit of every action id the Library Prompts flows emit
(`list`/`detail`/`create`/`update`/`delete`), and a new regression test that drives open-editor
through the REAL default enforcer + registry (RED-verified). After the fix the editor opens (see
`editor-two-part-multiline-body`).

## Verification
- Phase-1 gate suites: `Tests/Library Tests/UI/test_library_shell.py test_library_prompts_canvas
  test_destination_shells test_screen_navigation test_non_obscuring_focus_contract
  Tests/Prompt_Management` = 1062 passed / 1 skipped.
- Per-task reviews all Approved (Tasks 0–7); fix waves on Tasks 3 (CSS parity), 4 (write-time
  conflict banner), 5 (multi-line round-trip parser), and 8 (policy registration).
- Known pre-existing dev failure excluded: `test_unified_shell_phase5_recovery_taxonomy`
  (upstream `thread=True` export worker at 3efb31b3, fails at branch base).

## Residuals for the whole-branch review (not blocking Phase 1)
Snippet only surfaces user_prompt/details on name/system/keyword-only search matches; a dead
module-level `search_prompts` wrapper duplication; empty-Keywords "leave untouched" semantics;
two `_apply_mode("prompts")` direct-call personas tests; `CCPPromptEditorWidget` dead pocket
noted for a future CCP-legacy sweep. `prompt_ingest_events.py` was NOT deleted — the spec's
"confirmed dead" was wrong; it is live legacy Ingest▸Prompts wiring.

## sr-UX/HCI polish wave (Task 8b, commits 3f4b8bb3..21179d75)

A design review of the shipped screens surfaced eight findings (four defects, four upgrades),
all user-approved and implemented before Phase 2. New captures:

- `polish-new-prompt-rail-entry-2026-07-12.png` — **D1**: the Create rail now has **New prompt**
  (the C in CRUD was previously unreachable from the UI — you could only import or save-from-Console).
- `polish-blank-editor-reordered-fields-2026-07-12.png` — the blank create editor: fields reordered
  to Name · **Description** (was "Details", **U4**) · System · User · Keywords · **Author last**
  (**U2**, demoted from 2nd position in a single-author library); meta reads **New prompt**; the
  action row gains **Duplicate** (**U3**). Save creates via the scope service; a create-time race
  conflict is now recoverable (fix wave — was a nav-trap).
- `polish-list-purpose-secondary-2026-07-12.png` — **U1/D2**: list rows now show the prompt's
  PURPOSE ("Friendly release notes", "Explains feedback loop concepts") instead of `author · age`,
  and the filter matches name+description — no longer advertising keyword-matching it structurally
  could not deliver (list rows carry no keywords; a batched-join seam is the backlog path to
  restoring keyword filter + chips).
- `polish-import-browse-button-2026-07-12.png` — **D4**: the import row leads with **Browse…**
  (FileOpen), removing the type-an-absolute-path friction. (Also fixed, not shown: **D3** the
  name-conflict status now backs a real **Open existing** button resolving the captured offending
  name.)

Not folded in (backlogged): U5 (name the skipped files on import), keyword-in-list filtering +
chips (needs the batched seam), and a discard-without-save path for the editor (pre-existing
Task-4 explicit-save veto model).
