---
id: TASK-1346
title: Watchlists tab strip diverges from the approved spec
status: Done
assignee: []
created_date: '2026-07-29 05:30'
labels:
  - watchlists
  - spec-divergence
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The approved design spec specifies five tabs — Read / Sources / Runs / Rules / Artifacts
(`2026-07-25-watchlists-console-rebuild-design.md`, `### Tabs`). The implementation has six
sections (`_SECTION_DETAIL_TITLE:243-250`) with **no Artifacts tab** and two sections the spec does
not mention (Overview, Notifications).

Phase C deliberately kept the pre-existing tab strip rather than adopting the spec's layout. That
decision was never recorded, so the divergence was only discovered in Phase D when a task went
looking for the spec's "Artifacts tab names the next slice" empty state and found no such tab.

The spec's Artifacts section also carries a deep-linking note for spec #2: `NavigateToScreen`
accepts a `screen_context` dict but the Artifacts screen does not read it — it consumes a
chatbook-specific app attribute (`pending_artifacts_chatbook_target_id`), so a watchlist artifact
link needs a parallel pending attribute and consumer.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A decision is recorded on whether the spec adopts the six-section strip or the implementation adopts the spec's five tabs
- [x] #2 Whichever is chosen, the spec and the implementation agree, and the spec's Empty states section no longer references a tab that does not exist
- [x] #3 (moot in its original form — see notes) If Artifacts is added, its empty state names the next slice rather than showing a bare table
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Decision (AC#1, owner ruling 2026-08-03): the spec adopts the shipped strip.** Recorded as a
dated decision block at the top of the spec's `### Tabs` section
(`Docs/superpowers/specs/2026-07-25-watchlists-console-rebuild-design.md`).

Reality had moved past BOTH this task's premises and the spec's: the shipped strip is now **seven**
sections (Overview / Sources / Items / Runs / Rules / Notifications / Artifacts, digits `1`-`7`) —
this task was filed when it was six with no Artifacts, and spec #2's briefings work has since added
a fully built Artifacts section (generation, casts, audio, kept briefings, feed export + serve).

**AC#2:** the spec's Tabs section now describes the shipped seven tabs, records that only
Items/"Read" uses the three-pane split and that the rule is ENFORCED (task-1344's
unmount-plus-refusal), and notes where Overview and Notifications came from. The Empty-states
section no longer references a tab that does not exist — and its Artifacts line now quotes the
shipped actionable empty state. The stale five-tab keybinding line (`1`-`5`) was corrected to
`1`-`7`. The chatbook-Artifacts-screen deep-linking concern is recorded as moot (artifacts render
in-screen; section deep-links go through the shell's validated `screen_context`).

**AC#3 ("If Artifacts is added, its empty state names the next slice"):** moot in its original
form — Artifacts was added AND the "next slice" it was supposed to name has shipped. The section's
empty state is actionable rather than a bare table ("No briefings yet. Press Generate to write
one.", `artifacts_pane.py::_NO_BRIEFINGS`), which satisfies the AC's intent (an empty state that
tells the user what comes next) in the post-ship world; the spec records the supersession.

Docs-only change; no code touched.
<!-- SECTION:NOTES:END -->
