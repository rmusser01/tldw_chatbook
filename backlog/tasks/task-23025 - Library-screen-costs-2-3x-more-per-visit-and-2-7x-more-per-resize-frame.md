---
id: TASK-23025
title: >-
  Library screen costs 2.3x more per visit and 2.7x more per resize frame
status: Done
assignee:
  - '@claude'
created_date: '2026-08-27'
labels:
  - performance
  - ui
  - screens
  - regression
priority: medium
---

## Description

`library_screen.py` grew +6,340 lines in the review window. It did **not** add recompose sites -
those went **down**, 126 -> 96. It made each existing one materially more expensive.

- **Widgets constructed per visit: 67 -> 157 (+134%)**. Library is the only pre-existing screen whose
  cost changed materially; every other destination moved by 1-6 widgets.
- **DOM queries per resize frame: 31.6 -> 84.2** (153 `query` + 689 `query_one` per 10 frames). The
  handler carries a recent optimisation comment (TASK-22228) yet net query volume still grew 2.7x -
  the width-crossing early return sits *after* the query work.
- Focus-change queries per Tab: 22.4 -> 25.8, and that path can reach `refresh(recompose=True)`.

One whole-screen recompose now constructs 103 widgets at ~640-790 ms.

## Acceptance Criteria

- [x] Per-visit widget count and per-resize-frame query count measured before and after, interleaved
- [x] The resize handler returns early **before** doing query work when the layout cannot have changed
- [x] `research_workspace_screen.py:252` -> `_apply_pane_layout` gets the same gate; it is currently ungated
- [x] No behaviour change to layout at any terminal width, including the width-crossing cases the current code handles
- [x] `on_descendant_focus`'s per-Tab query volume is bounded (no all-region scroll-owner probing on routes that cannot mount those regions), measured before and after
- [x] Compose-time blocks invisible on the default route grow on demand where their closed-state is not a queried contract (the model-install progress pair), with an unmount walk for the deferred subtree

## Implementation Plan

1. Measure the base (fix/task-23022, `63464b174b`): widgets constructed per Library visit
   (with per-call-site attribution to find the +90), DOM queries per resize frame, and
   queries per Tab, using the LibraryHarness pilot with instrumented `Widget.__init__`
   and `DOMNode.query`/`query_one`.
2. Resize gate: cache the invariant `#library-shell-grid`/rail/canvas references
   (validated by the cheap `is_mounted` flag, identity-invalidated on recompose) and
   derive a bucket signature (compact/emergency/ingest-collapse buckets + the resolved
   ordinary rail contract + the cheap route/stage flags the legs consume) from attribute
   reads only; return from `on_resize` before any `query`/`query_one` when the signature
   matches the last applied one. Same gate shape for `_apply_pane_layout` in
   `research_workspace_screen.py` (derived `ResearchPaneLayout` + height bucket +
   relocate-focus applicability).
3. Per-visit construction: defer the compose-time blocks that are `display=False` on the
   default route (grow-on-demand, the TASK-23024 house pattern), guided by the census.
4. Focus path: bound `on_descendant_focus` query volume for plain row navigation.
5. Pin width-crossing behaviour with tests that FAIL against a deliberately broken
   implementation (gate moved after queries / deferred block eagerly restored); walk
   unmount/quit for anything deferred; interleave before/after measurements; preflight.

## Evidence

Interleaved A/B, 2 runs per arm, identical every run. Console is worse in absolute terms
(171-185 queries/frame) but **improved** slightly - pre-existing, not this delta.

Source: `Docs/Design/2026-08-27-holistic-perf-review.md`.

## Implementation Notes

Branch stacked on `fix/task-23022` (both tasks touch `library_screen.py`); diff
for review is `fix/task-23022...HEAD`.

**Resize gate (library_screen.py).** `on_resize` now derives a bucket/flag
signature from cheap state -- cached widget references (`region` /
`content_region` attribute reads) plus route/stage flags -- mirroring every
width-derived decision its three legs make: the ingest-collapse bucket (100),
the resolved `OrdinaryRailStyleContract` itself (so the custom-width clamp band
stays exact), the emergency bucket (64), and the compact bucket (120). A frame
whose signature equals the last applied one returns before ANY `query`/
`query_one`. The signature is recorded pre-legs, so a frame whose legs mutate
flags runs once more on the next event and settles; a skip can never hide a
crossing because every compared bucket is part of the signature. The media
reader leg (TASK-22228 item 7) is untouched above the gate.

**Reference caches.** `_library_layout_ref` (positive-only) and
`_library_compose_scoped_ref` / the reader-shell probe (negative-capable, valid
because those ids are constructed exclusively in `compose_content`) are
invalidated at `compose_content` -- the choke point every whole-screen
recompose passes through -- and every hit is validated by
`_library_ref_is_live`: NOT `is_mounted`, which lags detachment while
`App._prune` has already marked the corpse `_pruning` (the pre-existing
notes-list scroll test caught exactly this; see the new lessons-textual entry).

**Focus path.** The scroll-observer installer probes only regions the current
route can mount (7 guaranteed-failing whole-tree walks per Tab removed);
`_library_notes_focus_stage`, `_library_notes_scroll_owner`,
`_library_landing_focus_control_id` and the per-refresh footer-context loop
resolve through the caches.

**Research screen.** `_apply_pane_layout` returns before its ~11 `query_one`
calls when the derived `ResearchPaneLayout`, the height-compact bucket, and the
relocate-focus applicability are all unchanged.

**Per-visit deferral.** The model-install pair (ModelInstallProgress + label)
composes only while an install is retained; the first `InstallProgressed`
event grows it (handlers try the update path first, mount on `NoMatches`,
which also preserves the mocked-`query_one` unit-test contract). The rail
Details body deferral was implemented and REVERTED: its closed-state children
are a queried contract (counts line, `#library-use-in-console` handoff state)
pinned by seven tests across four files -- changing that is its own task; a
new test pins the contract so a future deferral must face it deliberately.

**Measured (interleaved x3 per arm, alternating, landing route,
LibraryHarness, config-isolated under Tests/conftest):**

| metric | base `63464b174b` (x3) | after (x3) |
|---|---|---|
| DOM queries / resize frame | 71.8 / 71.6 / 71.6 | 10.6 / 10.6 / 10.6 |
| DOM queries / Tab | 25.6 / 25.4 / 25.4 | 3.4 / 3.4 / 3.6 |
| widgets constructed / visit | 156 / 156 / 156 | 149 / 149 / 149 |
| research queries / same-band frame | 41.0 / 41.0 / 41.0 | 11.0 / 11.0 / 11.0 |

All remaining after-arm resize/Tab queries attribute to the nav bar's own
handlers (`UI/Navigation/main_navigation.py`); library_screen-attributed
queries on steady frames: 0 (pinned by test). The 110/170-column crossing
probe behaves identically on both arms every round. Crossings (compact both
ways twice, emergency engage/restore, ingest auto-collapse both ways) pinned
by born-red tests in `Tests/UI/test_library_resize_focus_gates_t23025.py`;
nine mutants each killed by a named test (see PR/report).

**Files:** `tldw_chatbook/UI/Screens/library_screen.py`,
`tldw_chatbook/UI/Screens/research_workspace_screen.py`,
`tldw_chatbook/Widgets/Library/library_rail.py` (details-children compose
factored into `_compose_details_body_children`, behaviour unchanged),
`Tests/UI/test_library_resize_focus_gates_t23025.py` (new),
`Docs/User_Guide/library.md` stamp, `backlog/docs/lessons-textual.md`.
