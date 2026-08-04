---
id: TASK-2200
title: Watchlists screen stops full-screen recomposing from background loaders
status: To Do
assignee: []
created_date: '2026-08-04'
labels:
  - watchlists
  - ui
  - tech-debt
dependencies: []
priority: medium
---

## Description (the why)

The Watchlists screen's background loaders request full-screen recomposes:
`_apply_local_wc_snapshot` (watchlists_collections_screen.py:983) and
`_load_tree_data` (:912) call `refresh(recompose=True)` at screen level, tearing
down and rebuilding the entire centre — including panes that are mid-recompose
of their own.

This is the architectural root cause behind TASK-1960's crash class, confirmed
empirically during that investigation: `App._prune` stamps `_pruning` over a
`walk_children` snapshot, `Widget.mount` silently mounts nothing on a pruned
widget, and `MessagePump._pre_process`'s `finally` marks it mounted anyway — so
any widget whose `_on_mount` queries its own children can crash when a screen
recompose lands mid-mount. TASK-1960 shipped `PruneSafeSelect` as the
mechanism-level guard for `Select`, but the guard covers one widget class, not
the hazard: any other composite widget mounted by these rebuilds is exposed the
same way.

A second latent defect is masked by the same behaviour: `SourcesPane`'s
form-close recompose silently mounts *nothing* when the screen prunes it
mid-flight — invisible today only because the screen's own recompose immediately
rebuilds the pane. If a future change stops the screen rebuilding the pane on
that path, form-close visibly breaks.

This is also TASK-1541's standing recommendation (recorded there as "open rec,
documented, not filed"): replace recompose-the-world refresh paths with the
targeted in-place update discipline the screen already uses elsewhere
(`update_item_status_cell`, `refresh_header_content`).

## Acceptance Criteria (the what)

- [ ] Completing a background load (`_load_tree_data`) or applying a local
      watchlists snapshot (`_apply_local_wc_snapshot`) no longer rebuilds the
      whole screen: the affected panes/regions are updated in place, and
      unrelated in-flight pane recomposes (e.g. the Sources create-form
      open/close) are not torn down by it.
- [ ] The rendered result after a background refresh is equivalent to what the
      full recompose produced today, covering the overview/first-run/loading
      states (the TASK-1347-strengthened tests stay green).
- [ ] The TASK-1960 e2e reproduction
      (`test_a_source_can_be_created_end_to_end_through_the_form`) stays green
      10/10 in isolation and in the poisoned order after
      `Tests/UI/test_watchlists_content_pane.py` — now with the destroyer
      removed rather than merely guarded against.
- [ ] The masked SourcesPane defect is addressed or made impossible: closing the
      create form yields a correctly-populated pane without depending on a
      screen-level rebuild to paper over an empty recompose.
- [ ] `PruneSafeSelect` remains in place as defense-in-depth (this task removes
      the known destroyer; it does not un-fix TASK-1960).
