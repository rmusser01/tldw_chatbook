# Library Source Workbench Stage A Shell Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adapt the approved Library content hub into the Stage A Source Workbench shell so users can distinguish source inventory, workspace policy, active work, quick actions, and selected-object inspection without implying unavailable services are wired.

**Architecture:** This is a visual/interaction shell slice on top of existing local Library services. It updates Library screen copy, section hierarchy, mounted regressions, and QA evidence only; it does not introduce tldw_server API calls, new collection item storage, or new Search/RAG/Console handoff behavior.

**Tech Stack:** Python 3.11+, Textual, TCSS, pytest/Textual Pilot, textual-web/CDP screenshots.

---

## Source References

- Spec: `Docs/superpowers/specs/2026-06-23-library-source-workbench-redesign-design.md`
- Current QA baseline: `Docs/superpowers/qa/library-content-hub-closeout/2026-06-22-library-content-hub-actual-use-closeout.md`
- Current screen: `tldw_chatbook/UI/Screens/library_screen.py`
- Current collection state: `tldw_chatbook/Library/library_collections_state.py`
- Current collection service seam: `tldw_chatbook/Library/library_collections_service.py`
- Current mounted tests: `Tests/UI/test_library_content_hub.py`
- TCSS source: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- TCSS variables: `tldw_chatbook/css/core/_variables.tcss`
- Generated TCSS: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Product roadmap: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Backlog task: `backlog/tasks/task-134 - Library-source-workbench-Stage-A-shell-hierarchy.md`

## ADR Check

ADR required: no

ADR path: N/A

Reason: this slice changes Library screen hierarchy, labels, disabled-state copy, tests, and QA evidence only. It does not change storage/schema, sync/conflict policy, data ownership, provider/runtime boundaries, service contracts, security policy, or server integration. A later collection-item adapter, workspace eligibility engine, sync promotion, or tldw_server integration slice must repeat the ADR check.

## File Structure

- Modify `Tests/UI/test_library_content_hub.py`
  - Add mounted regressions for the Stage A shell hierarchy.
  - Update old Collections assertions that still describe collections as reusable source groups.
  - Verify unsupported actions are disabled with visible reasons.
- Modify `tldw_chatbook/UI/Screens/library_screen.py`
  - Update `LIBRARY_COLUMN_TITLES` from `Library Modules / Content Hub / Hub Inspector` toward `Source Map / Active Workbench / Inspector`.
  - Split the left rail into visible `Workspace Context`, `Source Map`, and `Quick Actions` sections.
  - Update Hub and Collections copy to match the accepted Source Workbench IA.
  - Preserve route IDs, mode chip IDs, button IDs, existing service calls, and current disabled-state behavior.
- Modify `tldw_chatbook/Library/library_collections_state.py`
  - Update pure display-state empty copy from grouping language to stored-content review language.
  - Do not add new persistence fields or service calls in this slice.
- Modify `tldw_chatbook/css/components/_agentic_terminal.tcss` only if section grouping needs source stylesheet support.
  - Keep generated CSS untouched until the source TCSS change is done.
  - Use existing design-system variables when possible.
- Modify `tldw_chatbook/css/core/_variables.tcss` only if a reusable Library layout token is required.
- Regenerate `tldw_chatbook/css/tldw_cli_modular.tcss` with `python tldw_chatbook/css/build_css.py` only after source TCSS changes.
- Create `Docs/superpowers/qa/library-source-workbench-stage-a/2026-06-24-library-source-workbench-stage-a.md`
  - Record focused tests, CDP screenshot path, and manual no-server-dependency note.
- Modify `Docs/superpowers/trackers/product-maturity-roadmap.md`
  - Add a concise Stage A Library Source Workbench evidence row/link after QA evidence exists.
- Modify `backlog/tasks/task-134 - Library-source-workbench-Stage-A-shell-hierarchy.md`
  - Add implementation plan before code work begins.
  - Check acceptance criteria and add implementation notes only after verification and screenshot approval.

## Non-Goals

- Do not call tldw_server APIs from Library.
- Do not build full collection item persistence, reading-list import, highlights, tags, generated outputs, digest schedules, or templates in Stage A.
- Do not enable collection-scoped Search/RAG, Study, Console handoff, or server sync promotion unless an existing local capability already backs it.
- Do not move source Import/Export out of Library.
- Do not alter Console, Notes, Media, or Search/RAG ownership.
- Do not edit `tldw_chatbook/css/tldw_cli_modular.tcss` directly.

## Task 1: Add Red Regressions For Stage A Hierarchy

**Files:**
- Modify: `Tests/UI/test_library_content_hub.py`

- [ ] **Step 1: Add a mounted shell hierarchy regression**

Add a test near the existing default-mode tests:

```python
@pytest.mark.asyncio
async def test_library_stage_a_shell_surfaces_source_map_workspace_context_and_quick_actions() -> None:
    app = _build_test_app()
    _seed_library_content(app)
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-content-hub-title")

        visible = _visible_text(screen)

        assert "Source Map" in visible
        assert "Workspace Context" in visible
        assert "Quick Actions" in visible
        assert "Active Workbench" in visible
        assert "Browse: all workspaces" in visible
        assert "Use: current workspace only" in visible
```

- [ ] **Step 2: Add a Collections content-model regression**

Update `test_library_collections_selection_explains_membership_workspace_and_actions` or add a sibling test so it expects stored-content language and explicit disabled reasons:

```python
assert "Collections Reader" in visible
assert "Stored collection content" in visible
assert "Read/review collection items when a local item adapter is available." in visible
assert "Disabled: collection item Search/RAG is not wired yet." in visible
assert "Disabled: collection item Console handoff is not wired yet." in visible
```

- [ ] **Step 3: Add an empty Collections regression**

Update `test_library_collections_empty_state_keeps_global_browse_rule_and_blocks_wip_actions` so empty state copy no longer says `Group saved Library items for Search/RAG, Study, and Console.`

Expected new assertions:

```python
assert "No stored collection items are available locally yet." in visible
assert "Collections are for reading, reviewing, and reusing saved content." in visible
assert "Local collection metadata is available; item reading is pending a local adapter." in visible
```

- [ ] **Step 4: Run the new tests and confirm they fail**

Run:

```bash
python -m pytest -q Tests/UI/test_library_content_hub.py::test_library_stage_a_shell_surfaces_source_map_workspace_context_and_quick_actions --tb=short
python -m pytest -q Tests/UI/test_library_content_hub.py::test_library_collections_selection_explains_membership_workspace_and_actions --tb=short
python -m pytest -q Tests/UI/test_library_content_hub.py::test_library_collections_empty_state_keeps_global_browse_rule_and_blocks_wip_actions --tb=short
```

Expected: FAIL because the current UI still exposes old `Library Modules`, `Content Hub`, and reusable source-group copy.

## Task 2: Update Library Shell Labels And Left-Rail Grouping

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`

- [ ] **Step 1: Update mode column titles**

Change `LIBRARY_COLUMN_TITLES` to Stage A labels:

```python
LIBRARY_COLUMN_TITLES = {
    "sources": ("Source Map", "Active Workbench", "Inspector"),
    "conversations": ("Source Map", "Saved Conversations", "Conversation Inspector"),
    "search": ("Source Map", "Search/RAG Workbench", "Evidence Inspector"),
    "import-export": ("Source Map", "Import/Export Workbench", "Import/Export Inspector"),
    "workspaces": ("Source Map", "Workspace Context", "Handoff Rules"),
    "collections": ("Source Map", "Collections Reader", "Collection Inspector"),
    "study": ("Source Map", "Study Handoff", "Inspector"),
    "flashcards": ("Source Map", "Flashcards Handoff", "Inspector"),
    "quizzes": ("Source Map", "Quizzes Handoff", "Inspector"),
}
```

- [ ] **Step 2: Add explicit left-rail section headers**

In `LibraryScreen.compose`, inside `#library-source-browser`, render these stable sections in order:

```python
yield Static("Workspace Context", id="library-workspace-context-title", classes="destination-section")
yield Static(self._library_workspace_scope_label(workspace_depth_state), id="library-workspace-scope")
yield Static("Browse: all workspaces", id="library-workspace-browse-rule")
yield Static("Use: current workspace only", id="library-workspace-use-rule")
yield Static("Source Map", id="library-source-map-title", classes="destination-section")
yield from self._source_module_action_widgets()
yield Static("Quick Actions", id="library-quick-actions-title", classes="destination-section")
```

If existing route buttons are the only current quick actions, leave them in Source Map and make Quick Actions an honest short guidance block instead of duplicating controls:

```python
yield Static("Open a mode, then use the inspector for selected-item actions.", id="library-quick-actions-guidance")
```

- [ ] **Step 3: Preserve stable IDs and handlers**

Do not rename these existing route/action IDs:

- `library-open-notes`
- `library-open-media`
- `library-open-conversations`
- `library-open-search`
- `library-open-import-export`
- `library-open-collections`
- `library-open-workspaces`
- `library-open-study`
- `library-open-flashcards`
- `library-open-quizzes`
- `library-use-in-console`

- [ ] **Step 4: Run the shell hierarchy test**

Run:

```bash
python -m pytest -q Tests/UI/test_library_content_hub.py::test_library_stage_a_shell_surfaces_source_map_workspace_context_and_quick_actions --tb=short
```

Expected: PASS.

## Task 3: Correct Collections Copy Without Faking Item Services

**Files:**
- Modify: `tldw_chatbook/Library/library_collections_state.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/UI/test_library_content_hub.py`

- [ ] **Step 1: Update pure empty-state copy**

In `library_collections_state.py`, change:

```python
LIBRARY_COLLECTIONS_EMPTY_COPY = "Group saved Library items for Search/RAG, Study, and Console."
```

to:

```python
LIBRARY_COLLECTIONS_EMPTY_COPY = (
    "No stored collection items are available locally yet. Collections are for "
    "reading, reviewing, and reusing saved content."
)
```

- [ ] **Step 2: Update Collections mode description**

In `LIBRARY_MODES["collections"]`, describe the target IA honestly:

```python
"description": (
    "Collections mode: read, review, and reuse stored collection content. "
    "Current local support exposes collection metadata while item reading is staged."
),
"next_action": (
    "Select a local Collection record to inspect item-reader readiness and disabled capabilities."
),
```

- [ ] **Step 3: Update Collections inspector empty copy**

In `_collections_inspector_rows`, for no selected collection, use copy that explains the target and current limitation:

```python
Static("No collection item selected.", id="library-collection-inspector-empty")
Static(
    "Collections are for reading and reviewing saved content; the local item reader is not wired in this slice.",
    id="library-collection-inspector-empty-next-action",
)
Static(
    "Global browsing/search remains available; active staging and manipulation stay workspace-gated.",
    id="library-collection-inspector-global-rule",
)
Static(
    "Local collection metadata is available; item reading is pending a local adapter.",
    id="library-collection-inspector-empty-local-actions",
)
```

- [ ] **Step 4: Update selected Collection inspector copy**

Keep the selected collection metadata visible, but stop presenting it as the end state. Use honest transitional language:

```python
Static("Selected Collection Record", id="library-inspector-title", classes="destination-section")
Static(f"Stored item count: {selected.item_count_label}", id="library-collection-inspector-item-count")
Static(
    "Collection item reader: not wired locally yet.",
    id="library-collection-inspector-reader-state",
)
Static(
    "Disabled: collection item Search/RAG is not wired yet.",
    id="library-collection-inspector-rag-blocked",
    classes="ds-recovery-callout is-blocked",
)
Static(
    "Disabled: collection item Console handoff is not wired yet.",
    id="library-collection-inspector-console-blocked",
    classes="ds-recovery-callout is-blocked",
)
```

Preserve sync dry-run labels already generated from `selected.sync_status_label` and `selected.sync_status_detail`.

- [ ] **Step 5: Update Collections action-region copy**

In `_library_action_widgets`, for `self._active_mode == "collections"`, replace old grouping text with:

```python
Static("Collection item actions", classes="destination-section")
Static(
    "Read/review collection items when a local item adapter is available.",
    id="library-collection-actions-local",
)
Static(
    "Disabled: collection item Search/RAG, Study, Console handoff, and server sync promotion are not wired yet.",
    id="library-collection-actions-wip",
    classes="ds-recovery-callout is-blocked",
)
```

Keep existing buttons disabled unless existing code already enables them with a real capability.

- [ ] **Step 6: Run focused Collections tests**

Run:

```bash
python -m pytest -q Tests/UI/test_library_content_hub.py::test_library_collections_selection_explains_membership_workspace_and_actions Tests/UI/test_library_content_hub.py::test_library_collections_empty_state_keeps_global_browse_rule_and_blocks_wip_actions --tb=short
```

Expected: PASS.

## Task 4: Apply Minimal TCSS Support If Needed

**Files:**
- Modify if needed: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify if needed: `tldw_chatbook/css/core/_variables.tcss`
- Regenerate if needed: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Test: `Tests/UI/test_master_shell_design_system_contract.py`

- [ ] **Step 1: Check whether Python-only hierarchy renders cleanly**

Run the mounted Library tests first. If the section headers and guidance rows are readable without CSS changes, skip this task and document that no TCSS changed.

- [ ] **Step 2: If spacing is needed, edit source TCSS only**

Use existing selectors where possible:

- `.library-region`
- `.library-source-action`
- `.library-source-action-spacer`
- `#library-source-browser`
- `#library-source-detail`
- `#library-source-inspector`

Do not edit `tldw_chatbook/css/tldw_cli_modular.tcss` directly.

- [ ] **Step 3: Regenerate generated TCSS only after source changes**

Run:

```bash
python tldw_chatbook/css/build_css.py
```

Expected: generated CSS changes correspond to source TCSS changes only.

- [ ] **Step 4: Run design-system contract tests**

Run:

```bash
python -m pytest -q Tests/UI/test_master_shell_design_system_contract.py --tb=short
```

Expected: PASS.

## Task 5: Full Focused Verification And QA Evidence

**Files:**
- Create: `Docs/superpowers/qa/library-source-workbench-stage-a/2026-06-24-library-source-workbench-stage-a.md`
- Modify: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Modify: `backlog/tasks/task-134 - Library-source-workbench-Stage-A-shell-hierarchy.md`

- [ ] **Step 1: Run focused automated verification**

Run:

```bash
python -m pytest -q Tests/UI/test_library_content_hub.py --tb=short
python -m pytest -q Tests/UI/test_product_maturity_phase3_library_contract_layout.py Tests/UI/test_post_ux_product_roadmap_handoff.py --tb=short
git diff --check
```

Expected: all tests pass and diff check reports no whitespace errors.

- [ ] **Step 2: Capture actual rendered screenshot via textual-web/CDP**

Use the project CDP workflow, not generated mockups. Capture at least:

- Hub mode with Stage A left rail visible.
- Collections mode showing empty or seeded selected collection copy and disabled reasons.

Save screenshots under:

```text
Docs/superpowers/qa/library-source-workbench-stage-a/
```

Expected: screenshot shows real rendered app UI, top tabs, Source Map/Workspace Context/Quick Actions grouping, and Collections copy that does not claim item services are wired.

- [ ] **Step 3: Create QA evidence note**

Create:

```text
Docs/superpowers/qa/library-source-workbench-stage-a/2026-06-24-library-source-workbench-stage-a.md
```

Include:

- Commands run and pass/fail results.
- Screenshot filenames.
- Manual finding: no tldw_server runtime dependency or API call was introduced.
- Residual risks: collection item reader, local capability flags, collection-scoped RAG/Console, and server parity remain future stages.

- [ ] **Step 4: Update roadmap**

Add a concise entry to `Docs/superpowers/trackers/product-maturity-roadmap.md` near the Library content hub evidence:

```markdown
| Library Source Workbench Stage A shell | `Docs/superpowers/qa/library-source-workbench-stage-a/2026-06-24-library-source-workbench-stage-a.md` | verified; TASK-134 done |
```

- [ ] **Step 5: Update Backlog task hygiene**

In `backlog/tasks/task-134 - Library-source-workbench-Stage-A-shell-hierarchy.md`:

- Check all acceptance criteria.
- Keep the ADR check in the implementation plan.
- Add concise implementation notes with approach, modified files, tests, screenshot evidence, and residual risks.
- Mark the task Done via Backlog tooling only after screenshot approval.

- [ ] **Step 6: Commit the implementation slice**

Run:

```bash
git add Tests/UI/test_library_content_hub.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/Library/library_collections_state.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/core/_variables.tcss tldw_chatbook/css/tldw_cli_modular.tcss Docs/superpowers/qa/library-source-workbench-stage-a/2026-06-24-library-source-workbench-stage-a.md Docs/superpowers/trackers/product-maturity-roadmap.md "backlog/tasks/task-134 - Library-source-workbench-Stage-A-shell-hierarchy.md"
git commit -m "Adapt Library source workbench shell"
```

Only include CSS files in the commit if they changed.

## Plan Review Notes

- Scope is intentionally Stage A only. It does not solve the full Collections reader model.
- The plan avoids server integration because the accepted spec says tldw_server is a product reference, not a runtime dependency for this visual redesign.
- The riskiest implementation detail is wording: the UI must set the right future IA without pretending local collection-item reading exists.
- If implementation pressure pushes toward adding collection item adapters, stop and split a Stage D task with an ADR check.
