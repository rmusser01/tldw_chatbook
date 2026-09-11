---
id: TASK-32352
title: >-
  Library Collections: the rail row opens 'Quick Capture', its empty state
  blames unset filters and names an absent action, and its pager is not disabled
  at 0 of 0
status: Done
assignee: []
created_date: '2026-09-11 06:16'
updated_date: '2026-09-11 07:43'
labels:
  - library
  - collections
  - critique-10
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The rail row 'Collections (0)' opens a canvas titled Quick Capture whose only message is 'No captures match this scope. Clear filters or save a URL with Quick Capture.' (A caps 16/58); Previous/Next render enabled at '0-0 of 0' while the identical Trash pager renders disabled (B D6 cap 52). Overlaps the open product decision task-32057. Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One name for the feature in the rail and on the canvas
- [x] #2 A true empty state that does not mention filters unless one is set and does not cite an action absent from the screen
- [x] #3 The pager is disabled at 0 of 0 like Trash's
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Give the Collections items canvas a 'Collections' heading; keep 'Quick Capture' as the save-a-URL button label only.
2. Split the empty state: filtered vs never-captured; filtered branch reachable only when a filter field is set.
3. Route the pager through simple_library_pager_display + library_pager_layout so 0 of 0 shows no controls.
4. Tests in Tests/UI/test_library_crit10_pagers.py; update collections.md + stamp.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All three ACs, implementing the user's task-32057 AC#1 decision (recorded under `## Decision` in that task file).

**AC#1 — one name.** The canvas had no heading at all, so its first painted line — the **Quick Capture** button — read as the title while the rail row said "Collections (0)". The pane now leads with a `Collections` heading (`#library-collections-header`, the same `destination-section` class the Skills canvas uses). "Quick Capture" survives only as the label of the button that saves a URL: it is a verb, not a place. The rail row is untouched.

**AC#2 — a true empty state.** One sentence covered both an empty collection and a filtered-to-nothing one, so a profile that had never saved anything was told to clear filters it had never set and pointed at Quick Capture as if it were elsewhere on the screen. Split in two, keyed on the real filter-bearing fields of `CapturePageRequest` (`search`, `tags`, `domain`, `date_from`, `date_to` — the five the Filters form and the search box set, and the five **Clear** clears). `statuses`/`favorite` are deliberately excluded: they carry the rail's scope selection, which Clear does not touch.

- no filter: "No saved captures yet · press Quick Capture above to save a page by URL."
- a filter: "No captures match these filters · clear them to see everything saved."

## Fix round 1 (task review, 2026-09-11)

**P2 — the first cut's empty state was false for a rail scope.** `filtered`
tested only the filter-form fields, but the rail's scope rows set `statuses`
/ `favorite` (`library_collections_controller.py:491-508`), so an empty
**Favorites** beside a rail reading `Collections (57)` claimed nothing had
ever been saved and pointed at an action that does not leave the scope — the
same class of defect AC#2 exists to fix. The empty state is now three
branches, narrowest first, because the two narrowings come from different
controls and have different ways out: a filter is undone by **Clear**, a
scope by choosing **All Captures**. A filter set inside a scope takes the
filter sentence. Pinned red-first by
`test_collections_empty_state_names_an_empty_rail_scope[favorites|archived]`
and `test_collections_empty_state_prefers_the_filter_copy_inside_a_scope`.

**P3 — the Collections/Trash boundary-reason asymmetry is a decision, not an
accident.** Collections gained a `#library-collections-page-reason` line (the
brief's Step 3 asks for it, and it matches Media/Conversations/Prompts);
Trash did not, because its pager is a pinned fixed-height container and the
line would cost a row at exactly the width task-32354's finding is about.
Collections' pager has no fixed height, so it can afford one.

**AC#3 — the pager is disabled at 0 of 0.** Previous/Next were plain `disabled=` Buttons with no `○` marker, so at `0–0 of 0` they read as enabled where the structurally identical Trash pager read as disabled. The canvas now goes through `simple_library_pager_display` + `library_pager_layout` (task-32354), which at one page drops the page copy, the boundary reasons and both controls; on two pages the controls come back with `library_disabled_action_label`'s `○` and the shared reasons.

Live at 235x52, 100x30 and 60x24 on the seeded profile: heading `Collections`, the never-captured sentence, `0–0 of 0` with no controls (`crit10/wave/pagers/caps/02`, `07`, `08`).

## Files

- `tldw_chatbook/Widgets/Library/library_collections_capture_reader.py`.
- `Tests/UI/test_library_crit10_pagers.py` (new).
- `Docs/User_Guide/library/collections.md`.
<!-- SECTION:NOTES:END -->
