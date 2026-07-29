---
id: TASK-1346
title: Watchlists tab strip diverges from the approved spec
status: To Do
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
- [ ] #1 A decision is recorded on whether the spec adopts the six-section strip or the implementation adopts the spec's five tabs
- [ ] #2 Whichever is chosen, the spec and the implementation agree, and the spec's Empty states section no longer references a tab that does not exist
- [ ] #3 If Artifacts is added, its empty state names the next slice rather than showing a bare table
<!-- AC:END -->
