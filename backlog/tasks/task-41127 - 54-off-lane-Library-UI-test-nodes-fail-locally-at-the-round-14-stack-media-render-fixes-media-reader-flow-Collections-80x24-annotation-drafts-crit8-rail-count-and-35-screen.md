---
id: TASK-41127
title: "54 off-lane Library UI test nodes fail locally at the round-14 stack: media render fixes, media reader flow, Collections 80x24 annotation drafts, crit8 rail count and 35 screen-navigation nodes"
status: Done
assignee:
- '@claude'
created_date: 2026-11-15 02:20
updated_date: 2026-11-16 21:38
labels:
- follow-up
- library
- tests
- dev-red
priority: medium
references:
- Tests/UI/test_library_media_render_fixes.py
- Tests/UI/test_library_media_reader_flow.py
- Tests/UI/test_library_collection_reader_journeys.py
- Tests/UI/test_library_crit8_collections_row.py
- Tests/UI/test_screen_navigation.py
- Tests/UI/test_library_prompt_mixins.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** None of these nodes is in the UI census or any workflow, so they can stay red indefinitely; the round-14 Library streams recorded several as red on their base. Under the plain command they hit the local RecoveryRequired trap, so their own failures are only visible with a profile marker.

**Evidence.** The five files at `-n 4` through run.sh with a scratch plugin stamping `bootstrap_profile` on unmarked items (no repository edit; load 44-55): 54 failed, 334 passed. No failing node is a census entry; the 101 census nodes of `test_screen_navigation.py` passed.
- `test_library_media_render_fixes.py` x9: `test_escape_labels_are_one_of_four` (`:1284`, `('armed delete', 'cancel')`), the analyze/generate reason nodes x8 ("OpenAI is not ready (Missing API key)" where "No analysis … Providers & Models." is expected, `:1956`, `:1988`, `:2036`).
- `test_library_media_reader_flow.py` x7: page-three anchor, select-mode settlement, `t` delete confirm, edit-metadata routing (`NoMatches '#library-media-reader-find'` / `'#library-media-reader-more'`), and `test_any_non_markdown_item_with_content_says_why_rendered_is_absent[plaintext|video]` (`:3146`, empty notice). TASK-23145 owns an older scroller-seam failure in this file; these are different.
- `test_library_collection_reader_journeys.py::test_annotation_drafts_survive_disclosures_modes_and_capture_return[textual-dark-size1|textual-light-size1]` (80x24): "Expected readable #library-collections-freeform-note-save".
- `test_library_crit8_collections_row.py::test_rail_count_never_falls_back_to_the_unfiltered_total_after_a_visit`.
- `test_screen_navigation.py` x37: 2 are TASK-23149's fake-row nodes; the other 35 include `WorkspaceProbe` lacking `cancel_path_task` / `cancel_structural_wait`, lambdas taking one argument where `library_screen.py:11078` passes two, `LibraryScreen` lacking `_sync_collections_panel`, and many route tests landing on `ChatScreen`/`Screen` instead of `LibraryScreen`, which may be an artifact of the shared bootstrap profile rather than the code.
- Also off-lane and intermittent: `test_library_prompt_mixins.py::test_literal_history_restore_keeps_kind_definition_and_guards` (red in 1-3 of 3 params on base 26b2e677dd, TASK-40293's notes).
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

1. Re-run the six files plainly through run.sh (`-n 4`) to see which nodes die on `RecoveryRequired` and which fail in their bodies.
2. Run each round-14 red node in a process of its own (fresh bootstrap profile, `bootstrap_profile` stamped by a scratch plugin only where the file has no marker), so no other test shares its profile. A node green alone and red only beside others is a shared-profile artifact.
3. For each node still red alone: find the commit that moved the behaviour (drift: re-pin the test) or prove a product defect (leave red, propose a follow-up). TASK-23149's two fake-row nodes and the Collections/Notes-focus areas owned by fu-r15-libcoll/fu-r15-libfocus are listed, not changed.
4. Make the passing nodes runnable plainly (module/node `bootstrap_profile` or `@private_profile_test`, whichever the config read needs), show each plain run, and append them to the UI census tail with measured serial times; raise `MINIMUM_FILES`.
5. Covering runs, census checker, preflight.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each node is run under its own profile (a private-profile marker where it reads config) and classified as test drift, product defect, or shared-profile artifact, with the evidence
- [x] #2 Drift is re-pinned with the commit that moved the behaviour named; a product defect gets a test red before its fix
- [x] #3 Every node that then passes runs on a PR lane (UI census tail entries with measured times)
<!-- AC:END -->

## Implementation Notes

Classified every round-14 red node plus the nodes the same files showed red here. 36 census entries (42 nodes) now pass under a plain command. The rest are product defects, one product decision, or nodes that only fail under load. Those stay red and are proposed as follow-ups. No product code changed.

**Method.** Base cc6006fcf5, all runs through run.sh.
- Plain `-n 4` on the six files: 194 failed, 222 passed. 121 of the failures were `RecoveryRequired`: `test_library_media_render_fixes.py` and `test_library_crit8_collections_row.py` carry no profile marker.
- With a throwaway stamp plugin (deleted): 53 failed, 363 passed (load 52-80).
- Each of those 53 alone, in its own process with a fresh profile: 47 red, 6 green (load 75-94). The 6 green alone are shared-run or load artifacts.
- None of the reds came from the shared profile. The load was the variable.

**Test drift.** Re-pinned, each citing the commit that moved the behaviour. Each was red alone at base and is green after.
- Media Escape chip set. fb000f77b7 made the chip over an armed delete read "cancel".
- Six no-analysis-provider nodes. 77052f5a85 made `load_settings()` carry `[analysis_defaults]`, so the template's `provider = "OpenAI"` now reaches these tests. They now rebind the app's config snapshot to one with no provider.
- 20 `test_screen_navigation.py` nodes, including the census sibling `test_on_descendant_focus_does_not_disarm_for_the_systems_own_row_refocus`, which was passing vacuously. The cited commits are 7cf89de6c0, 203f4dc633, 5dd1077df6, d00812ac44, 02b0866c1a, 9de08fa193, 0509134982, 1d5297dbbc, 3915a2da2c, 5eca00a482, 7992124e8d, fed0b7257a, 939dee8dc2 and b7dd8e53f2. Each is named in the test it re-pins.

**Races exposed by load.** The tests waited on a state flag, then pressed a widget that is re-mounted after it.
- Find, More, Info and filter-clear.
- Collections scope row, which also read the previous page's total, 3.
- The prompt-history Restore button and `ConfirmationDialog` Confirm.
- The rendered-view note's paint.

The fix waits for the mounted widget (`_wait_for_selector`), or for the page or paint itself. `_load_row_0` now returns once the Reader chrome is back. Its other three callers in `test_library_crit8_polish_media.py` and `test_library_crit9_shell.py` pass. Red before, at base: find_toggles, edit_metadata, info_calls, filter page-three and literal-history 3/3 (alone); rail count alone (`None == 0`, then `3 == 0`). After: rail count 5/5, literal history 9/9, the media nodes 11/11 alone.

**Left red.**
- Product defects:
  - Collections 80x24 annotation drafts (dark and light). After the note saves, the focused Save sits at y=26 on a 24-row terminal and paints nothing.
  - `test_generic_reentry_returns_to_library_landing` and `test_library_screen_round_trip_returns_to_landing_with_rag_draft`. Library reuse (7992124e8d) skips `restore_state`, so re-entry no longer returns to the landing. Turning reuse off made the test pass.
  - `test_boot_with_search_default_tab_lands_on_library_rag_canvas`. `_mirror_library_rag_scope_recovery` mounts into an unmounted `#library-rag-source-scope` and the worker raises `MountError`.
  - `test_overlapping_navigate_requests_complete_in_fifo_order` and `test_rapid_tab_switch_storm_leaves_no_zombie_widgets`. `WorkflowsScreen._show_status` queries controls that are not mounted (NoMatches).
- Product decision: `test_library_screen_bindings_are_all_gated_or_universal`. Shift+F6 is gated off on the landing. The gate came from merge resolution a62d92f590; the test says Shift+F6 is universal.
- Load artifacts: the route and boot nodes with a fixed 3 s wait (`range(150)` x 0.02 s). These are study-escape, boot-prompts, search-route, prompts, skills, media, search round-trip, search-all palette, generic entry, research workspace and prompt-receipt owner. `test_console_staged_live_work_launch_survives_navigate_away_and_back` is re-pinned but failed 1 of 7 runs alone. With a 30 s wait, search-route passed at load 46; with the 3 s wait it passed 0 of 3. Not laned.
- TASK-23149's two `_FakeMediaRowButton` nodes are untouched.

**Lane.** 36 tail entries. `MINIMUM_FILES` is 2181. Profile markers:
- `test_library_media_render_fixes.py` and `test_library_crit8_collections_row.py` are now module `bootstrap_profile`, as TASK-39141 did for `test_library_shell.py`.
- `test_library_media_reader_flow.py` is already bootstrap through `Tests/conftest.py`.
- `test_library_prompt_mixins.py` and `test_screen_navigation.py` are already marked.

One plain serial process with the lane flags (`-p no:xdist --timeout=180 --durations=0`, no plugin): 42 passed in 291.6 s at load 7-35. Each entry in that process, setup+call+teardown in seconds (phases under 1 s are hidden):
- `test_library_media_render_fixes.py`:

  | Entry | s |
  |---|---|
  | escape labels | 9.3 |
  | generate disabled | 8.1 |
  | reader generate reason | 13.3 |
  | select-mode analyze reason | 8.7 |
  | find toggles | 5.9 |
  | read-mode walk | 7.3 |

- `test_library_media_reader_flow.py`:

  | Entry | s |
  |---|---|
  | filter page-three | 8.7 |
  | select-mode settlement | 8.4 |
  | t key | 7.5 |
  | l key | 8.1 |
  | escape from reader | 7.4 |
  | edit metadata | 7.3 |
  | info calls | 7.5 |
  | more strip `[wide]` | 5.6 |
  | non-markdown note | 14.4 |

- `test_library_crit8_collections_row.py` rail count: 14.6. `test_library_prompt_mixins.py` literal history: 89.1.
- `test_screen_navigation.py`, 0-5 s each:

  | Entry | s |
  |---|---|
  | files leave seams, recompose | 3.3 |
  | create route | 2.6 |
  | collections reconcile | 2.2 |
  | flush veto | 2.6 |
  | files back | 2.8 |
  | reload-confirm back | 1.2 |
  | contract | 1.0 |
  | landing help keys | 1.3 |
  | media viewer back | 1.5 |
  | note editor back | 1.6 |
  | prompt editor back | 1.6 |
  | rail focus | 2.5 |
  | compose focus | 2.2 |
  | arm focus | 1.9 |
  | arm twice | 1.4 |
  | descendant focus | 2.8 |
  | fresh instances | 4.8 |
  | routes reach handler | 2.8 |
  | nav-bar overflow | <3 |

Charge per-process times on a loaded host:
- Media, collections and mixins entries, one process each at load 63-105: 31-63 s of pytest time each, mixins 140 s.
- Screen-navigation entries, three processes in parallel at load 108-122: 6-25 s each.

**Files.**
- Tests: `Tests/UI/test_library_media_render_fixes.py`, `test_library_media_reader_flow.py`, `test_library_crit8_collections_row.py`, `test_library_prompt_mixins.py`, `test_screen_navigation.py`.
- Census: `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
- Lessons: one entry in `backlog/docs/lessons-testing-evidence.md`.
