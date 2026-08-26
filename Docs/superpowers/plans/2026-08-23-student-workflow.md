# Student Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `library_save_note` — the write tool that lets a Console agent land its per-chapter study notes (folder-grouped, provenance-tagged, visible in the notes screen), plus the documented fan-out pattern and the flashcards ruling.

**Architecture:** One new write descriptor + a `save` operation branch in the note backend dispatch; rows via the legacy `NotesInteropService` (`add_note`/`update_note`), folders via `NotesScopeService` (`get_folder_by_path` ensure + `attach_note_to_folder`, scope pinned `local`); `library.notes/save` policy landed with the handler; the #4 story test upgraded to close the write loop; docs carry the pattern and conventions.

**Tech Stack:** Python ≥3.11, ChaChaNotes DB (notes + v36 folder tables), the #4 tool/policy machinery, `asyncio.run` sync bridge.

**Spec:** `Docs/superpowers/specs/2026-08-23-student-workflow-design.md` — §4 the tool, §4.3 the seam, §4.4 the duplicate ruling, §5 the pattern, §6 flashcards, §8's eleven rulings (all three review passes folded in).

## Global Constraints

- Create-default; update ONLY with `note_id` + `expected_version` (both required together — one without the other → `invalid_argument`).
- `update_note` raises `ConflictError` on version mismatch (verified, ChaChaNotes_DB.py:13657) → handler maps to the named `content_changed`.
- Input bounds are schema-level `maxLength`, harmonized with Task 4's spec-save precedent (name 120 / description 2_000): **title ≤ 512, content ≤ 100_000, folder ≤ 256**; minLength 1 on title/content.
- Folder = ONE segment; ensure via `get_folder_by_path([name])` → `create_note_folder` on miss → re-query on conflict (the race is tolerated, never raises to the agent); placement via `attach_note_to_folder(scope="local", folder_id, note_id)` (re-attach safe — `attach_manual` revives history).
- Scope pinned to `local` (`ScopeType.LOCAL_NOTE`) — the notes UI's scope; any other value makes folders invisible.
- Provenance header (source/revision/chapter/chunks) is a documented convention in the tool description — never enforced code.
- Duplicate-window accepted (no title uniqueness, no title-keyed upsert); the documented re-run convention is **search-based** (`library_search_notes(query=title)`) — the list tool has no folder filter.
- Policy `library.notes`/`save` lands WITH the handler on BOTH Console-direct and MCP mappings; denial before any backend call; RuntimePolicy pins.
- No orchestration code, no prompt preset, no note deletion, no note↔media link table.
- Repo rule: targeted test runs only; venv `.venv/bin/python`.

---

### Task 1: `library_save_note` — descriptor, dispatch, handler, policy

**Files:**
- Modify: `tldw_chatbook/Library/library_tool_contract.py` (the descriptor + `_save_note_schema`), `tldw_chatbook/Library/local_library_tool_service.py` (the `save` operation branch + the handler + the `notes_scope_service` constructor param), `tldw_chatbook/UI/Screens/chat_screen.py` + `tldw_chatbook/MCP/server.py` (both wiring sites gain the scope-service handle, `getattr` degrade pattern), `tldw_chatbook/MCP/local_control_service.py` (the action mapping override), `tldw_chatbook/runtime_policy/registry.py` (the resource)
- Test: `Tests/Library/test_local_library_tool_service.py` (extend), `Tests/RuntimePolicy/` (extend), MCP local-control expectations

**Interfaces:**
- Consumes: `NotesInteropService.add_note(user_id, title, content, note_id?) -> str` / `update_note(user_id, note_id, update_data, expected_version) -> bool` (raises `ConflictError`); `NotesScopeService.get_folder_by_path` — NOTE: this is on the **repository** (`LocalNoteFolderRepository.get_folder_by_path(folder_segments)`); the scope service exposes `list_note_folder_children`/`create_note_folder`/`attach_note_to_folder` — if no scope-level path-getter exists, the handler reaches the repository via the scope service's own access path OR does name-lookup via `list_note_folder_children(parent_id=None)`; pick the seam the scope service actually exposes and document the choice.
- Produces: `library_save_note` in the catalog (24th name) returning `{item: {id, "note", title, folder?}, version, created, notes}`.

- [ ] **Step 1 — failing tests** (per spec §7.1-7.5): schema acceptance/rejection (bounds! required-together rule on note_id+expected_version! unknown keys); create → id+version+created:true; update with matching version → version bumps, created:false; stale version → `content_changed`; unknown note_id → `not_found`; folder create-then-attach (folder appears once — sequential idempotency); concurrent folder ensure (interleaved create → re-query → one folder, no raise); folder-less create touches no folder; policy denial (mutation pin: `add_note` never called) + the `library.notes.save.local` mapping on both MCP seams; the descriptor carries the provenance-header documentation (assert the description mentions source/revision).
- [ ] **Step 2 — implement** the descriptor + schema (bounds per Global Constraints), the `save` dispatch branch, the handler (legacy interop for rows; scope service for folder/placement; `asyncio.run` bridge; ConflictError catch → content_changed; `InputError` → invalid_argument), both wirings, the policy resource + override.
- [ ] **Step 3 — green + pins:** the new tests; catalog pins grow 23→24 across the surface suites; `Tests/RuntimePolicy/` + MCP local-control green.
- [ ] **Step 4 — commit:** `feat(library): library_save_note agent tool (folder-grouped, version-locked) + library.notes/save policy`

### Task 2: The upgraded student story + docs + close-out

**Files:**
- Test: `Tests/Library/test_agent_chunk_student_story.py` (extend — the #4 test is the base)
- Modify: `Docs/User_Guide/library/local-library-tools.md` (the save-note contract + the pattern), `Docs/User_Guide/console/agent-runs-and-tools.md` (the fan-out pattern section), `CHANGELOG.md`
- Board: the flashcards-viewing follow-up + the folder-filtered-listing follow-up (file at close-out, house pattern)

- [ ] **Step 1 — the story upgrade (spec §7.6):** after the #4 read path, `library_save_note` lands the Chapter-7 note (with the provenance header incl. `revision:`), `library_get_note` re-reads it (payload revision matches), and the **re-run leg**: `library_search_notes(query=title)` finds it → update via note_id+version → still one note (no duplicate); the flashcard leg saves a QA-markdown note and re-reads it.
- [ ] **Step 2 — docs:** the save-note contract in the reference (bounds, the together-rule, the search-based re-run convention with its reason); the fan-out pattern section (structure → spawn-per-chunk → fetch → save → re-run); CHANGELOG.
- [ ] **Step 3 — file the two follow-ups** (flashcards viewing/SRS surface; folder-filtered `library_list_notes` candidate) — direct task-file writes, id sweep per house rules.
- [ ] **Step 4 — targeted close-out:** `pytest Tests/Library/ Tests/RuntimePolicy/ Tests/MCP/test_local_control_service.py Tests/Notes/ -q` — zero new failures vs dev baseline.
- [ ] **Step 5 — commit:** `test(library): student story closes the write loop; docs + follow-ups`

## Self-Review (run at save)

1. **Spec coverage:** §4.1→T1 (contract+bounds+together-rule); §4.2→T1 (description) + T2 (story asserts the header); §4.3→T1 (seam+ensure+attach+scope+bridge+wiring); §4.4→T1 (create-default/conflict) + T2 (re-run leg); §5→T2 docs (search-based convention per the third-review correction); §6→T2 flashcard leg + follow-up filing; §7.1-7.5→T1; §7.6→T2; §7.7-7.11→T1 (the resolved hedges). All eleven §8 rulings mapped.
2. **Ordering:** T1 (tool) before T2 (story rides it). Two tasks — the spec is one tool + docs; splitting further is ceremony.
3. **Type consistency:** `library_save_note` input/output shapes identical across T1/T2; the folder-ensure contract (one segment, re-query-on-conflict) stated once in Constraints and consumed by both tasks.
4. **Placeholders:** the scope-level path-getter choice in T1's Interfaces is a genuine verify-at-implementation seam (two verified options named, both on real APIs) — not a dodge; the handler documents which it took.
