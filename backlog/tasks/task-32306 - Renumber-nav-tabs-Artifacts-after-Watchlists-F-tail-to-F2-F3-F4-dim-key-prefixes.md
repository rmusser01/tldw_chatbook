---
id: TASK-32306
title: >-
  Renumber nav tabs: Artifacts after Watchlists, F-tail to F2/F3/F4, dim key
  prefixes
status: Done
assignee: []
created_date: '2026-09-11 03:36'
updated_date: '2026-09-11 04:22'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The top nav bar's key prefixes read haphazardly: the digit layer runs 1-9 then 0, then jumps to F7/F8/F9 for Lab/Logs/Settings, so the eye scans 9,0,7,8,9. Rebind the F-tail to the first free F-keys (F2/F3/F4) so the sequence reads as a left-to-right keyboard walk (number row, then F-row from F2; F1=Help and F6=Next Pane reserved per ADR-031). Also move Artifacts after Watchlists in SHELL_DESTINATION_ORDER. Render the key prefix dimmed so it parses as a key hint rather than an ordinal.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Nav bar reads ⌃1 Home ⌃2 Console ⌃3 Library ⌃4 Roleplay ⌃5 Watchlists ⌃6 Artifacts ⌃7 Schedules ⌃8 Workflows ⌃9 MCP ⌃0 ACP F2 Lab F3 Logs F4 Settings
- [x] #2 F2/F3/F4 navigate to Lab/Logs/Settings; f7/f8/f9 no longer bound as destinations
- [x] #3 Key prefix renders dimmed in nav bar and overflow menu; nav_button_label plain-string contract unchanged
- [x] #4 Targeted nav/label tests updated and green
- [x] #5 User guide docs teach the new keys; ADR-031 gains the nav hotkey-layer convention
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. shell_destinations.py: move Artifacts entry after Watchlists in SHELL_DESTINATION_ORDER\n2. main_navigation.py: NAV_FKEY_LABELS -> F2/F3/F4; dim-prefix Text helper used by bar + overflow menu\n3. app.py: SHELL_DESTINATION_FKEYS -> f2/f3/f4\n4. Update tests: golden order list, expected-keys, ghost-clip premise, F-label asserts, splash press, key inventory, latency mapping, press sites\n5. Update User Guide docs; add ADR-031 refinement\n6. Run targeted tests; fix fallout
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Approach: the numbering now reads as one left-to-right keyboard walk —
  ctrl+1..ctrl+9, ctrl+0 across the number row, then F2/F3/F4 continuing the
  F-row (F1=Help and F6=Next Pane are ADR-031 reserved; F5 stays free of the
  app-level layer, leaving the file picker's refresh untouched). Artifacts
  moved behind Watchlists, so Roleplay/Watchlists/Artifacts are now
  ⌃4/⌃5/⌃6. Both bindings and labels derive from zipping
  SHELL_DESTINATION_ORDER with the key lists, so the reorder was one tuple
  edit in shell_destinations.py; the F-tail is two constants
  (NAV_FKEY_LABELS, SHELL_DESTINATION_FKEYS).
- Dim polish: new `nav_button_label_text()` returns the label as Rich `Text`
  with a `dim` span over exactly the key prefix; `nav_button_label()` keeps
  the plain-string contract (str/cell_len identical), so width math, ghost
  clipping, and all label assertions are styling-neutral. Bar and overflow
  menu both render through the new helper; overflow appends "(current)" via
  Text.append.
- Copy fallout (found by the pre-implementation review, missed by the first
  quote-anchored grep): user-facing strings in
  Widgets/Console/console_settings_modal.py and
  Chat/console_provider_endpoints.py taught "F9 Settings"; updated, along
  with the test asserting that scope copy. Recorded as a lesson in
  backlog/docs/lessons-textual.md.
- Tests: golden order lists (test_master_shell_navigation,
  test_screen_navigation), expected-keys inventory, splash regression press
  f9→f4, latency tour ctrl+5→ctrl+4 + f7/f9→f2/f4, css-consolidation tour,
  product-maturity replay label, and new unit test
  test_nav_label_text_dims_only_the_key_prefix. The 80-col ghost-clip pair
  was re-derived per its own instructions: Artifacts ("⌃6 Art" fragment) is
  now the straddler at active=home; the border-route no-op test switched
  active="artifacts"→"home" because the reorder left no straddler under the
  artifacts-centered scroll.
- Known interaction (documented in index.md + roleplay docs + ADR-031
  refinement): Roleplay's own ctrl+4 nav slot sits inside that screen's
  ctrl+1–4 mode-chip range, so ctrl+4 on Roleplay switches modes — harmless
  since you are already there; ctrl+5–ctrl+0 keep navigating away.
- Docs: 15 User Guide files updated (index nav map + shortcut table,
  artifacts/watchlists/lab/logs/settings/rag/home/console/chat-basics/tts,
  three roleplay pages). ADR-031 gained the task-32306 refinement recording
  the hotkey-layer convention. Deliberately left stale: dated qa/ UAT SVG
  captures, the 2026-08-11 latency audit, RAG-Documentation.md's pre-existing
  wrong Ctrl+5 mention, and Docs/User_Guide/images/settings/overview.svg
  (baked screenshot; regeneration left for a docs-image pass).
- Verification: test_master_shell_navigation (41), test_ux_batch3+4 (16),
  screen_nav nav selections (4), css tour, console context controls (10),
  latency guardrails (2), palette tab (1), git-push keyboard-safety (6) —
  all green. Pre-existing WIP failures unrelated to this change (verified by
  stashing only these edits and re-running):
  test_unified_shell_phase6_first_time_replay and
  test_product_maturity_phase6_first_time_release_replay fail on Library
  "Import / Export" copy / startup-settle before any nav assertion. Ruff:
  no new violations on edited lines (pre-existing I001/UP035 debt left
  as-is).
- ADR: no new ADR; ADR-031 refinement added (task-32306).
<!-- SECTION:NOTES:END -->


## Port note (dev PR)

Ported to `dev` in worktree branch `task-32306-nav-renumber`. Dev had evolved
the hotkey layer since the original implementation: shortcuts became a
destination contract (`SHELL_DESTINATION_SHORTCUTS`) and two new destinations
(`research` at f10, `meetings` at f11, with F10/F11 stranded mid-strip) landed.
The approved design was adapted faithfully: the 13 original destinations keep
exactly the approved bar (Artifacts after Watchlists; ⌃4 Roleplay, ⌃5
Watchlists, ⌃6 Artifacts; F2/F3/F4 for Lab/Logs/Settings), and Research and
Meetings are seated in the tail by the same left-to-right walk (F5, then F7 —
F6 stays reserved for Next Pane). Embedded-terminal, file-picker, and
session-switcher F-key shadows are unchanged contextual behavior (they already
swallowed f7–f11).
