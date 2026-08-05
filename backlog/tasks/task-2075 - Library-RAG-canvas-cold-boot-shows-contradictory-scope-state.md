---
id: task-2075
title: Library RAG canvas cold-boot shows contradictory scope state
status: To Do
assignee: []
labels:
  - library
  - rag
  - ux
dependencies: []
priority: medium
---

## Description

Booting straight onto the Library Search/RAG canvas (config `default_tab = "search"`, enabled by RAG UX v2 PR-2) renders one self-contradicting first frame: the Sources toggles already show real counts (`✓ Notes (1)`, `✓ Prompts (1)`) while the Evidence region still shows the empty-library gate copy ("No Library sources yet..."). It clears on any navigate-away-and-back, and the panel is functional throughout — the Run button and toggles are correctly enabled.

This directly contradicts the honesty theme PR-2 shipped: a false claim rendered beside true evidence, on the exact first frame the boot-landing feature exists to serve.

## Acceptance Criteria

- [ ] Booting with `default_tab = "search"` renders a first frame whose Evidence-region gate copy agrees with the Sources toggle counts (no "No Library sources yet" beside populated toggles)
- [ ] Steady-state background snapshots (a background ingest completing while the user is mid-search) still do NOT eject or recompose the canvas — RAG-27's guarantee from PR-2 Task 1 holds
- [ ] A test covers the cold-boot landing path specifically (not only the navigate-back path, which already self-heals)

## Implementation Notes (context from the PR-2 live check, 2026-08-03)

Root cause is an interaction between two PR-2 tasks, both new relative to dev — not an inherited bug:

- Task 1 (RAG-27) made `_apply_local_source_snapshot` take an in-place branch whenever `_library_selected_row_id == LIBRARY_ROW_BROWSE_SEARCH`, calling `_sync_library_rag_scope_toggle_and_run_gate_widgets`, which by design updates only the Run button, toggle `(N)` labels, and scope-summary line — deliberately NOT the recovery/callout block or the `has-recovery` class, because those need remove/mount sequences that were unsafe to interleave with other refresh callers.
- Task 4 made the Search/RAG selection apply *before* the screen's first `compose()`, so the very first paint is guaranteed pre-fetch with no warm snapshot cache (previously this required beating a local DB query with human reaction time on the session's first Library visit — negligible).

Suggested seam: in `_sync_library_rag_scope_toggle_and_run_gate_widgets` (or its caller branch around `library_screen.py:2272`), also toggle `#library-rag-source-scope`'s `has-recovery` class and mount/unmount `library_rag_scope_recovery_children`, mirroring `_refresh_search_rag_panel_state_widgets` (~:16637-16649) — gated to fire only when `library_rag_scope_shows_recovery(...)`'s result actually CHANGES, so steady-state calls keep RAG-27's no-eject guarantee.

Evidence: `/private/tmp/uat-pr2-805d-evidence/07-item-e-default-tab-search-boot.txt` (contradiction) and `08-*post-nav-back*.txt` (self-healed).
