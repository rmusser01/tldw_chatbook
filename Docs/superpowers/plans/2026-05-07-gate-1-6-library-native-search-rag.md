# Gate 1.6 Library-Native Search/RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Library Search/RAG a real destination-native retrieval workflow with source scope, query input, retrieval status, evidence/results, citations/snippets, and Console handoff.

**Architecture:** Keep Library as the owner of deliberate retrieval. Add pure display-state and adapter contracts first, then mount a Library-native Search/RAG panel inside the existing three-pane Library shell. Reuse existing Search/RAG result normalization and Console live-work handoff seams; do not embed the full legacy `SearchRAGWindow` inside Library.

**Tech Stack:** Python 3.12, Textual, pytest, Backlog.md, existing `LibraryScreen`, `SearchRAGWindow` helper seams, `ChatHandoffPayload`, `ConsoleLiveWorkLaunch`, `DestinationRecoveryState`, and RAG optional-dependency policy/recovery helpers.

---

## Source Of Truth

- Binding design gate: `Docs/superpowers/specs/2026-05-06-screen-design-adaptation-audit-design.md`
- Binding layout contract: `Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md`
- Current roadmap: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Parent backlog gate: `TASK-10.8`
- Child execution slices: `TASK-10.8.1` through `TASK-10.8.5`

Gate 1.6 is required because Gate 1 only made the `Search/RAG` Library mode selectable. The finished gate must let users query Library-owned sources deliberately, inspect evidence and source authority, and move cited snippets into Console without knowing the legacy `search` route.

## Scope

Included:

- Library-native Search/RAG display-state contracts for source scope, query mode, retrieval status, evidence rows, citations/snippets, blocked states, and Console handoff action state.
- Source scope selectors and recovery states for notes, media, conversations, Workspaces, and Collections so later source adapters can attach without changing the user-facing model.
- A Library-owned Search/RAG panel mounted inside `#library-source-detail` / `#library-source-inspector` when mode is `search`.
- Retrieval adapter seam that can use a fake local service in tests and later wrap current local/server RAG services without UI thread mutation.
- Result/evidence normalization that preserves source IDs, scores, snippets, citations, provenance, runtime backend, and recovery copy.
- Console handoff from Library Search/RAG results with staged evidence and source authority preserved.
- Console-initiated RAG against Library sources as a visible staged retrieval seam, not a hidden prompt-only shortcut.
- QA evidence and roadmap/backlog tracking for Gate 1.6 closeout.

Excluded:

- Replacing or deleting the legacy `SearchRAGWindow`.
- Full embeddings/indexing management redesign.
- Full Workspaces and Collections implementation. Gate 1.6 can expose scope selectors and empty/recovery states, but deeper Workspaces/Collections gates remain separate.
- Server parity beyond the adapter contract and recoverable blocked/server-required states.
- Artifact/Chatbook save changes beyond preserving retrieved evidence in the existing Console handoff path.

## File Structure

### Create

- `tldw_chatbook/Library/__init__.py`
- `tldw_chatbook/Library/library_rag_state.py`
  - Pure dataclasses/builders for Library Search/RAG source scope, query controls, retrieval status, evidence rows, citation/snippet display, and action states.
- `tldw_chatbook/Library/library_rag_service.py`
  - Adapter protocol and normalization seam for local/server retrieval services.
- `tldw_chatbook/Widgets/Library/__init__.py`
- `tldw_chatbook/Widgets/Library/library_search_rag_panel.py`
  - Textual widgets for source scope, query input, results/evidence list, and inspector detail.
- `Tests/Library/test_library_rag_state.py`
- `Tests/Library/test_library_rag_service.py`
- `Tests/UI/test_product_maturity_gate16_library_search_rag.py`
- `Docs/superpowers/qa/product-maturity/phase-3/2026-05-07-gate-1-6-library-native-search-rag.md`

### Modify

- `tldw_chatbook/UI/Screens/library_screen.py`
  - Mount Library-native Search/RAG mode and route query/result/Console actions.
- `tldw_chatbook/UI/Views/RAGSearch/search_handoff.py`
  - Reuse or extract citation/snippet metadata normalization if needed.
- `tldw_chatbook/Chat/console_live_work.py`
  - Add only small payload/action helpers if Library RAG evidence needs clearer Console status text.
- `tldw_chatbook/UI/Screens/chat_screen.py`
  - Add Console-initiated Library RAG staged retrieval only if a focused seam is needed.
- `tldw_chatbook/css/features/_library.tcss` or existing Library TCSS source
  - Add source TCSS for Search/RAG panel regions. Do not edit generated CSS directly.
- `tldw_chatbook/css/tldw_cli_modular.tcss`
  - Regenerate with `tldw_chatbook/css/build_css.py` after source TCSS changes.
- `Docs/superpowers/trackers/product-maturity-roadmap.md`
- `Docs/superpowers/qa/product-maturity/phase-3/README.md`
- `backlog/tasks/task-10 - Product-Maturity-Phase-3-Knowledge-And-Study-Workflows.md`
- `backlog/tasks/task-10.8*.md`

### Read Before Editing

- `tldw_chatbook/UI/Screens/library_screen.py`
- `tldw_chatbook/UI/Views/RAGSearch/search_rag_window.py`
- `tldw_chatbook/UI/Views/RAGSearch/search_handoff.py`
- `tldw_chatbook/UI/Views/RAGSearch/search_result.py`
- `tldw_chatbook/UI/Views/RAGSearch/search_event_handlers.py`
- `tldw_chatbook/Chat/console_live_work.py`
- `tldw_chatbook/Chat/chat_handoff_models.py`
- `tldw_chatbook/UI/Screens/chat_screen.py`
- `Tests/UI/test_product_maturity_phase3_library_contract_layout.py`
- `Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py`
- `Tests/UI/test_search_handoffs.py`
- `Tests/UI/test_ux_audit_smoke.py`

Use:

```bash
PY=.venv/bin/python
```

## Risk Controls

- Do not embed `SearchRAGWindow` inside Library. Gate 1.6 is destination-native.
- Keep route ID `library` stable and preserve existing `#library-mode-search`, `#library-source-browser`, `#library-source-detail`, and `#library-source-inspector` selectors.
- Do not hide optional dependency blockers behind transient notifications; render persistent recovery copy with owner and next action.
- Do not query or mutate Textual UI from worker threads. Worker results must be applied on the message thread.
- Avoid exact paragraph assertions in UI tests. Prefer selectors, disabled/enabled state, payload fields, and durable labels.
- Preserve existing Search/RAG handoff tests and current legacy `SearchRAGWindow` reachability until the native flow proves parity.

---

### Task 1: Gate 1.6.1 Library Search/RAG Display-State Contracts

Backlog: `TASK-10.8.1`

**Files:**
- Create: `tldw_chatbook/Library/__init__.py`
- Create: `tldw_chatbook/Library/library_rag_state.py`
- Create: `Tests/Library/test_library_rag_state.py`
- Test: `Tests/UI/test_product_maturity_phase3_library_contract_layout.py`

- [x] **Step 1: Run current Library baseline**

```bash
$PY -m pytest -q Tests/UI/test_product_maturity_phase3_library_contract_layout.py Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py::test_library_core_loop_modes_are_actionable_without_leaving_library --tb=short
```

Expected: pass with existing warnings only.

- [x] **Step 2: Add pure state red tests**

Create tests for:

- `LibraryRagScopeState.from_source_counts(notes=2, media=1, conversations=0, workspaces=0, collections=0)` exposes `Source Scope: All local sources`, selectable source types, and clear empty state when all counts are zero.
- `LibraryRagQueryState.from_values(query="", mode="rag")` blocks execution with recovery copy.
- `LibraryRagResultRow.from_result(...)` preserves title, snippet, score, source ID, citation labels, and provenance metadata.
- `LibraryRagPanelState.from_values(...)` marks retrieval status as `ready`, `searching`, `blocked`, or `empty` with user-visible next action.

Expected before implementation: import failure for `tldw_chatbook.Library.library_rag_state`.

- [x] **Step 3: Implement minimal dataclasses**

Implement frozen dataclasses:

- `LibraryRagScopeState`
- `LibraryRagQueryState`
- `LibraryRagResultRow`
- `LibraryRagActionState`
- `LibraryRagPanelState`

Keep these pure, non-Textual, and tolerant of loose app/test seam values.

- [x] **Step 4: Verify and commit**

```bash
$PY -m pytest -q Tests/Library/test_library_rag_state.py Tests/UI/test_product_maturity_phase3_library_contract_layout.py --tb=short
git diff --check
git add tldw_chatbook/Library/__init__.py tldw_chatbook/Library/library_rag_state.py Tests/Library/test_library_rag_state.py
git commit -m "Add Library Search RAG display state contracts"
```

### Task 2: Gate 1.6.2 Library-Native Search/RAG Panel

Backlog: `TASK-10.8.2`

**Files:**
- Create: `tldw_chatbook/Widgets/Library/__init__.py`
- Create: `tldw_chatbook/Widgets/Library/library_search_rag_panel.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Create/modify: `Tests/UI/test_product_maturity_gate16_library_search_rag.py`

- [x] **Step 1: Add mounted red tests**

Add tests proving that pressing `#library-mode-search` keeps the user inside Library and mounts:

- `#library-search-rag-panel`
- `#library-rag-source-scope`
- `#library-rag-query-input`
- `#library-rag-run-query`
- `#library-rag-results`
- `#library-rag-inspector`
- `#library-rag-use-in-console`

Also assert legacy `#search-rag-container` is not mounted inside Library.
Keep the existing `Search/RAG` compatibility route action available and tested, but do not make it the primary Library mode experience.

- [x] **Step 2: Implement panel widget**

`LibrarySearchRagPanel` should compose source scope, query input, run button, result/evidence area, and inspector using `LibraryRagPanelState`.

- [x] **Step 3: Wire Library mode**

When `_active_mode == "search"`, `LibraryScreen` should render the native panel in the existing Library detail/inspector shell. Keep `#library-open-search` available as a fallback route button, but the mode itself must be usable without route navigation.

- [x] **Step 4: Verify and commit**

```bash
$PY -m pytest -q Tests/UI/test_product_maturity_gate16_library_search_rag.py Tests/UI/test_product_maturity_phase3_library_contract_layout.py --tb=short
git diff --check
git add tldw_chatbook/Widgets/Library/__init__.py tldw_chatbook/Widgets/Library/library_search_rag_panel.py tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_product_maturity_gate16_library_search_rag.py
git commit -m "Mount Library native Search RAG panel"
```

### Task 3: Gate 1.6.3 Retrieval Adapter And Evidence Results

Backlog: `TASK-10.8.3`

**Files:**
- Create: `tldw_chatbook/Library/library_rag_service.py`
- Create: `Tests/Library/test_library_rag_service.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/UI/test_product_maturity_gate16_library_search_rag.py`

- [ ] **Step 1: Add adapter red tests**

Cover:

- normalizing local service results into `LibraryRagResultRow`
- preserving `citations`, `source_id`, `chunk_id`, `score`, `snippet`, `document_title`
- returning `DestinationRecoveryState` when dependencies are unavailable
- returning policy-blocked recovery when runtime source is incompatible

- [ ] **Step 2: Implement adapter protocol**

Create `LibraryRagSearchService` seam with async `search(query, scope, mode)` and a normalizer. In tests, use `app_instance.library_rag_search_service` fake first. If absent, return a recoverable unavailable state rather than calling the legacy widget directly.

- [ ] **Step 3: Wire query execution**

`LibraryScreen` should run retrieval in a worker, set status to `searching`, then apply results on the Textual message thread. Empty results should show a visible no-results state with recovery suggestions.

- [ ] **Step 4: Verify and commit**

```bash
$PY -m pytest -q Tests/Library/test_library_rag_service.py Tests/UI/test_product_maturity_gate16_library_search_rag.py Tests/UI/test_disabled_action_recovery_tooltips.py --tb=short
git diff --check
git add tldw_chatbook/Library/library_rag_service.py tldw_chatbook/UI/Screens/library_screen.py Tests/Library/test_library_rag_service.py Tests/UI/test_product_maturity_gate16_library_search_rag.py
git commit -m "Add Library RAG retrieval adapter"
```

### Task 4: Gate 1.6.4 Console Handoff And Console-Initiated RAG

Backlog: `TASK-10.8.4`

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `tldw_chatbook/UI/Views/RAGSearch/search_handoff.py`
- Modify: `tldw_chatbook/Chat/console_live_work.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/UI/test_product_maturity_gate16_library_search_rag.py`
- Modify: `Tests/UI/test_console_internals_decomposition.py`
- Test: `Tests/UI/test_search_handoffs.py`

- [ ] **Step 1: Add handoff red tests**

Assert that selecting a Library RAG result and pressing `#library-rag-use-in-console` calls `open_console_for_live_work` with:

- `source="Library Search/RAG"`
- `target_id` tied to the selected result/source
- payload fields for `query`, `source_id`, `chunk_id`, `snippet`, `citations`, `score`, and `runtime_backend`
- recovery copy: `Review citations before sending.`

- [ ] **Step 2: Add Console-invoked RAG red test**

Add a mounted Console test proving a Console RAG action can request Library retrieval against visible Library source scope and shows retrieval state in `#console-run-inspector` or staged context. Keep this as a seam test; do not implement full conversational retrieval rewriting.

- [ ] **Step 3: Implement handoff builders**

Reuse `build_search_chat_handoff_payload` metadata rules where possible. Add a Library-specific builder only if the payload needs Library source scope fields that legacy Search/RAG does not expose.

- [ ] **Step 4: Wire Console action**

Expose a minimal Console RAG invocation seam that stages a Library RAG query/result with visible status. If runtime support is missing, render a blocked state with owner and next action.

- [ ] **Step 5: Verify and commit**

```bash
$PY -m pytest -q Tests/UI/test_product_maturity_gate16_library_search_rag.py Tests/UI/test_console_internals_decomposition.py Tests/UI/test_search_handoffs.py Tests/UI/test_console_live_work_handoffs.py --tb=short
git diff --check
git add tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/UI/Views/RAGSearch/search_handoff.py tldw_chatbook/Chat/console_live_work.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_product_maturity_gate16_library_search_rag.py Tests/UI/test_console_internals_decomposition.py
git commit -m "Stage Library RAG evidence in Console"
```

### Task 5: Gate 1.6.5 QA Closeout And Tracking

Backlog: `TASK-10.8.5`

**Files:**
- Create: `Docs/superpowers/qa/product-maturity/phase-3/2026-05-07-gate-1-6-library-native-search-rag.md`
- Modify: `Docs/superpowers/qa/product-maturity/phase-3/README.md`
- Modify: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Modify: `backlog/tasks/task-10 - Product-Maturity-Phase-3-Knowledge-And-Study-Workflows.md`
- Modify: `backlog/tasks/task-10.8*.md`
- Test: `Tests/UI/test_product_maturity_gate16_library_search_rag.py`

- [ ] **Step 1: Add evidence tracking red test**

Add a test that verifies the Gate 1.6 QA evidence doc exists, includes `## Scope`, `## Walkthrough`, `## Verification`, `## Residual Risk`, and records selectors for `#library-search-rag-panel`, `#library-rag-query-input`, `#library-rag-results`, `#library-rag-use-in-console`, and Console staged evidence.

- [ ] **Step 2: Run focused verification**

```bash
$PY -m pytest -q Tests/Library/test_library_rag_state.py Tests/Library/test_library_rag_service.py Tests/UI/test_product_maturity_gate16_library_search_rag.py Tests/UI/test_product_maturity_phase3_library_contract_layout.py Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py Tests/UI/test_search_handoffs.py Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_live_work_handoffs.py --tb=short
```

Expected: pass with known warnings only.

- [ ] **Step 3: Perform manual QA walkthrough**

Run:

```bash
$PY -m tldw_chatbook.app
```

Walk through:

- Open Library.
- Switch to Search/RAG mode.
- Verify source scope, query input, retrieval status, evidence/results, citations/snippets, and inspector detail.
- Run a query with fake/local indexed data or verify the setup blocker if dependencies/indexes are unavailable.
- Stage a selected result into Console and verify the staged context preserves source authority and citations/snippets.
- Start from Console and invoke Library RAG, verifying retrieval state or a recoverable blocker.

- [ ] **Step 4: Update tracking and commit**

```bash
git add Docs/superpowers/qa/product-maturity/phase-3/2026-05-07-gate-1-6-library-native-search-rag.md Docs/superpowers/qa/product-maturity/phase-3/README.md Docs/superpowers/trackers/product-maturity-roadmap.md "backlog/tasks/task-10 - Product-Maturity-Phase-3-Knowledge-And-Study-Workflows.md" backlog/tasks/task-10.8*
git commit -m "Record Gate 1.6 Library Search RAG verification"
```

### Task 6: Final PR Verification

**Files:**
- Read: all modified files
- Verify: focused suite and diff hygiene

- [ ] **Step 1: Run the full Gate 1.6 focused suite**

```bash
$PY -m pytest -q Tests/Library/test_library_rag_state.py Tests/Library/test_library_rag_service.py Tests/UI/test_product_maturity_gate16_library_search_rag.py Tests/UI/test_product_maturity_phase3_library_contract_layout.py Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py Tests/UI/test_search_handoffs.py Tests/UI/test_ux_audit_smoke.py Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_disabled_action_recovery_tooltips.py --tb=short
```

- [ ] **Step 2: Run diff hygiene**

```bash
git diff --check
git status --short --branch
```

- [ ] **Step 3: Self-review against binding specs**

Confirm:

- Library Search/RAG is usable inside Library without knowing the legacy `search` route.
- Index/source/model blockers are visible before query execution.
- Results expose snippets, citations/provenance, confidence/score where available, and source authority.
- Console handoff preserves staged evidence.
- Console-initiated RAG has visible retrieval state or explicit recovery.
- Workspaces/Collections deeper behavior remains documented residual risk if not implemented in this gate.

- [ ] **Step 4: Prepare PR summary**

```markdown
## Summary
- Added Library-native Search/RAG state, panel, retrieval adapter, and evidence result contracts.
- Preserved citations/snippets and source authority through Library-to-Console handoff.
- Recorded Gate 1.6 QA evidence and roadmap/backlog tracking.

## Verification
- `$PY -m pytest -q Tests/Library/test_library_rag_state.py Tests/Library/test_library_rag_service.py Tests/UI/test_product_maturity_gate16_library_search_rag.py ... --tb=short`
- `git diff --check`
```
