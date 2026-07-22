# RAG Scope Narrowing — Phase 3 QA (workspace-level scope + intersection) 2026-07-21

## What Phase 3 adds
- Workspace-level RAG scope: a "RAG Scope" button (`#console-workspace-rag-scope-open`) in the Console left-rail Session/workspace area (`ConsoleWorkspaceContextTray`), enabled only for a real registry workspace (`rag_scope_enabled`); opens the same `ConsoleScopePickerModal` with target "workspace '{name}'", universe=None.
- `conversation ∩ workspace` intersection enforced end-to-end on both retrieval backends; no-overlap → EMPTY(cause="no-workspace-overlap") short-circuit + honest notice.
- Conversation picker inside a scoped workspace: universe restricted to the workspace's in-scope items (D3).
- Header chip / Inspector row reflect the EFFECTIVE (post-intersection) state; chip tooltip shows "conversation A ∩ workspace B → N" (hover-only).

## Captures
- **01-console-session-workspace-area.png** — Console left-rail Session section showing the workspace context area where the "RAG Scope" entry point lives.

## Capture limitation (transparent)
The live "RAG Scope" workspace modal + intersection could NOT be reliably driven through the textual-serve browser rig this session: in the isolated fresh capture profile the built-in "Default" workspace's `rag_scope_enabled` is False (the registry `ensure_default_workspace` row isn't warmed), so the button renders disabled and the modal doesn't open on click. This is a capture-environment artifact, NOT a reachability bug — code inspection confirms the button is rendered by the mounted left-rail tray at method-body level (not gated behind `show_heading`), and it is enabled for any real registry workspace.

The **picker modal UI itself is identical to the workspace target** and is fully captured in the Phase 2 QA set (`Docs/superpowers/qa/rag-scope-2026-07/`): modal chrome, tabs, tag counts, selection, scoped row + chip.

## Test verification (the real proof for Phase 3)
- Intersection enforcement end-to-end: `Tests/RAG/test_scope_pipeline_enforcement.py` — conv∩workspace narrows retrieval to the intersection on both backends; `test_no_workspace_overlap_short_circuits_empty_with_honest_notify` (zero leg calls).
- Entry-point gating (`rag_scope_enabled` True only for real workspaces): `Tests/UI/test_console_scope_row.py`.
- D3 conversation-picker universe = workspace items; chip intersection tooltip; ScopeCache workspace-side invalidation; memory-db guard; workspace scope storage + FK cascade (`Tests/Workspaces/`).
Independent merge-gate review verified hard-filter intersection integrity (narrows, never widens) on both backends.
