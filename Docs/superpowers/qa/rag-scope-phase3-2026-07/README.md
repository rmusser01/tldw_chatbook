# RAG Scope Narrowing — Phase 3 QA (workspace-level scope + intersection) 2026-07-21

Live app via textual-serve, real production CSS bundle, isolated HOME seeded with 5 tagged media items.

## What Phase 3 adds
- Workspace-level RAG scope: a "RAG Scope" button (`#console-workspace-rag-scope-open`) on its OWN row in the Console left-rail Session/workspace area, enabled only for a real registry workspace (`rag_scope_enabled`); opens `ConsoleScopePickerModal` with target "workspace '{name}'", universe=None.
- `conversation ∩ workspace` intersection enforced end-to-end on both retrieval backends; no-overlap → EMPTY(cause="no-workspace-overlap") short-circuit + honest notice.
- Conversation picker inside a scoped workspace: universe restricted to the workspace's in-scope items (D3).
- Header chip / Inspector row reflect the EFFECTIVE (post-intersection) state; chip tooltip shows "conversation A ∩ workspace B → N" (hover-only).

## Captures
- **01-workspace-rag-scope-button-in-rail.png** — the "RAG Scope" button rendered on its own row in the left-rail Session area, fully within the rail (see close-out fix note below).
- **02-workspace-scope-modal.png** — the workspace-target picker open: title "Narrow RAG scope — workspace 'Default'", full library (universe=None), tag chips with counts, item list, footer.
- **03-workspace-modal-selecting.png** — items being selected in the workspace scope picker.

## Close-out fix (task-14): RAG Scope button was clipped off the narrow rail
Final verification caught a real layout defect (systematic debugging): the "RAG Scope" button was the 3rd button in a single Horizontal action row (Switch/New/RAG Scope). Textual `Button` has `min-width: 16`, so three buttons need ~48 cols against the ~38-col left rail — the RAG Scope button overflowed the rail's right clip edge, making it unclickable (`pilot.click` and real users alike; this is why an earlier capture attempt couldn't open the modal). Fixed by moving "RAG Scope" to its own full-width row (`#console-workspace-rag-scope-row`); its region is now fully inside the rail. A regression assertion (`button.region.right <= rail_body.region.right`) now fails loudly on any future overflow. The existing test passes via its real, unmodified `pilot.click`.

## Test/verification proof
- Intersection enforcement (narrows, never widens) on both backends: `Tests/RAG/test_scope_pipeline_enforcement.py`; no-overlap zero-leg-calls test.
- Workspace-scope storage + FK cascade + round-trip: `Tests/Workspaces/`; verified live against the seeded registry DB (set→get round-trip == stored scope, 2 items on `workspace-default`).
- Button reachability, D3 universe, intersection tooltip, ScopeCache workspace-side invalidation, memory-db guard, zero-DB recompose: `Tests/UI/test_console_scope_row.py`.
- Independent merge-gate review verified hard-filter intersection integrity; enforcement never reads the display cache.
