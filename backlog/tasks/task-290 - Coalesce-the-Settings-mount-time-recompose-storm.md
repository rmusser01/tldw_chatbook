---
id: TASK-290
title: Coalesce the Settings mount-time recompose storm
status: Done
assignee:
  - '@claude'
created_date: '2026-07-17 15:17'
updated_date: '2026-07-17 22:45'
labels:
  - ux
  - performance
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Settings on_mount queues two thread workers whose completions each set recompose=True reactives, causing full-screen recomposes at nondeterministic times shortly after mount (briefly blanking the DOM and forcing footer/context re-seeding). task-264 made tests deterministic with a settle helper, but the product-side storm remains. Coalesce the two refreshes into one recompose (or make the reactives non-recompose with targeted updates).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Opening Settings triggers at most one post-mount recompose
- [x] #2 No visible DOM blank/flicker between mount and the workers landing
- [x] #3 Existing settings hub suite passes without the settle helper needing extra waits
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Coalesced via timeline tracing (three compounding causes): (1) __init__ assigned the recompose reactives -> phantom recompose of the just-composed screen; now set_reactive. (2) Two independent thread workers -> one combined _refresh_sync_rows applying both row sets in a single call_from_thread hop; Textual's flag-based _check_recompose collapses the pair to ONE recompose (own worker group so a solo manual refresh cannot cancel a combined pass). (3) Widget.focus() defers set_focus via call_later, so a storm recompose landing mid navigation-dance destroyed the focus target — pending-intent mechanism (recorded on navigation field focus, cleared only when the intended widget lands, consumed by the post-recompose restore; existing post-recompose focus always wins). AC1 pinned by test_settings_mount_triggers_at_most_one_post_mount_recompose (compose<=2); AC2 evidenced by the single-recompose ceiling + deterministic interleave test (focus survives); AC3: full hub suite 239 passed, settle helper unchanged. Commit c4481e3e.
<!-- SECTION:NOTES:END -->
