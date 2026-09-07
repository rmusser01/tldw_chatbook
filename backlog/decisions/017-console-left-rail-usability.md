# ADR-017: Console left-rail usability redesign

Status: Accepted
Date: 2026-07-18
Related Task: N/A
Supersedes: N/A

## Decision

Adopt a text-only, bordered-section visual language for the Console screen's left rail, with styled headers and bodies that make the Session, Context, Model, and Details sections scannable without icons or glyphs.

Move local-workspace identity generation from `UI/Screens/library_screen.py` into `Workspaces/registry_service.py` as a shared helper so both the Library screen and the Console screen can create local workspaces consistently.

## Context

The Console screen's left rail currently stacks four collapsible sections with minimal visual hierarchy: one-cell section headers share the same indentation as section bodies, status labels crowd their values, model settings are compressed into two single-line statics that truncate aggressively, and workspace creation is only available from the Library screen. This makes the rail hard to scan and obscures important session state.

The planned redesign (see links) restyles the rail with top-bordered section headers, indented bodies, wider status labels, and clearer workspace/model/source rows, while keeping the change text-only and preserving the existing four-section structure and resize/collapse behavior.

Both the Library screen and the redesigned Console screen need to generate local-only workspace identities (id and display name). Today that logic lives privately in `library_screen.py`. Duplicating it in `chat_screen.py` would invite drift, so the helper is being relocated to the workspace registry service for cross-screen reuse.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Add icons or glyphs to section headers | The project's Console design language is intentionally text-only; icons would introduce a new visual vocabulary and accessibility burden for this pass. |
| Merge, rename, or reorganize the four rail sections | Restructuring navigation would increase scope and risk; the goal is to improve legibility within the existing structure. |
| Keep the two-line model settings summary | Long provider and model names are ellipsized, hiding critical state; splitting into labeled rows with wrapping solves this. |
| Add a Console-native `[Add source]` button | The existing Library-to-Console handoff already handles source staging; a new picker is out of scope. |
| Leave `_next_local_workspace_identity()` in `library_screen.py` and duplicate it in `chat_screen.py` | Duplication would let the two screens diverge on id/name format and collision handling. |
| Move the helper to a new shared utilities module | The helper is tightly coupled to workspace registry semantics, so `Workspaces/registry_service.py` is the natural owner. |

## Consequences

- Future Console rail features must follow the text-only, bordered-section visual language defined here.
- Both `LibraryScreen` and `ChatScreen` depend on `Workspaces/registry_service.py` for local-workspace identity generation; changes to the helper affect both screens.
- CSS tests that guard generated stylesheets should include any new classes introduced by this redesign.
- UI tests for both Library workspace creation and Console `[New]` workspace creation must cover the shared helper path.
- The redesign stays within the existing rail persistence and resize model; no new backend flows or source-staging mechanics are introduced.

## Links

- [Design spec](../../Docs/superpowers/specs/2026-07-18-console-left-sidebar-usability-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-07-18-console-left-sidebar-usability.md)
