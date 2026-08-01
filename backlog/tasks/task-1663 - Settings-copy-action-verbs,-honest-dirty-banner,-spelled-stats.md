---
id: TASK-1663
title: 'Settings copy: action verbs, honest dirty banner, spelled stats'
status: Done
assignee:
  - '@claude'
created_date: '2026-07-31'
labels:
  - settings
  - ux
  - critique-r4
dependencies: []
priority: high
---

## Description (the why)

Critique round 4: the Overview inspector's action slot rendered a bare 'Theme' noun whose verb lived in a mouse-only tooltip; the dirty banner threatened data loss that does not happen (drafts persist per category, source-verified); the footer's 't test category' named an abstraction while buttons say Check/Validate/Test; the P:/C/N:/M: DB stats decoded only via hover tooltip; the full local-first paragraph occupied pinned inspector rows on all 22 categories.

## Acceptance Criteria (the what)

- [x] The Overview Theme action names its verb ('Open Theme editor')
- [x] The dirty banner states the truth: switching categories keeps the draft
- [x] The t hint uses each category's real verb (test provider / check storage / check privacy / validate config / preview appearance / check index)
- [x] DB stats spell their labels (Prompts / Chats/Notes / Media)
- [x] The full reassurance paragraph reads on Overview; one line elsewhere
- [x] Duplicate headings removed (Console Behavior, Provider readiness); stacked colons in the config-path row fixed

## Implementation Notes

TEST_ACTION_LABELS map feeding _footer_shortcut_entries; db_status_manager status string relabeled and the AppFooterStatus tooltip reduced to context; category-conditional reassurance line. The Theme button STAYS in the pinned header (only its label changed) — the 32-row compact contract requires a painted recovery action there (test_compact_overview_keeps_a_painted_recovery_action pins it).
