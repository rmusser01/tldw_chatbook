---
id: TASK-411
title: Remove redundant dict_panel.display gate in _show_center (personas)
status: Done
assignee:
  - '@zcode'
created_date: '2026-07-21 03:42'
labels:
  - roleplay
  - tech-debt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P2f (#728) moved PersonasCharacterDictionariesWidget into the #personas-character-attachments wrapper, whose display is now gated by _show_center on the character card/editor views. _show_center STILL sets dict_panel.display independently by the identical condition — redundant/dead-ish now that the wrapper controls both panels. Harmless (always consistent) but should be removed so the wrapper is the single source of truth for character-attachment panel visibility.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 _show_center no longer sets PersonasCharacterDictionariesWidget.display separately; the character-attachment panels' visibility is driven solely by the #personas-character-attachments wrapper gate,character-dictionary and world-book screen tests stay green (no visibility regression in card/editor/transcript views)
<!-- AC:END -->

## Implementation Plan

1. Verify the claim on current dev: both gates set display by the identical condition (visible_id in card/editor and runtime_source == "local"), no other site sets PersonasCharacterDictionariesWidget.display, and the child has no own CSS hiding.
2. Delete the dict_panel block from _show_center, updating the wrapper comment that referenced it; the #personas-character-attachments wrapper becomes the single source of truth.
3. Run the visibility-relevant personas suites, separating pre-existing dev failures via a before/after stash diff.

ADR required: no
ADR path: N/A
Reason: Dead-code removal within one method; no boundary or behavior change.

## Implementation Notes

Removed the redundant ``dict_panel.display`` assignment from ``_show_center`` (personas_screen.py) and its P1f-era comment block; rewrote the wrapper comment to name the wrapper as the single source of truth and record the removal. Verified first: the two gates applied byte-identical conditions, no other code path sets the dictionaries widget's display, and the widget carries no ``display: none`` of its own -- child visibility defaults to visible and is fully controlled by the wrapper parent, so behavior is unchanged by construction.

Verification: 130 passed across the six visibility-relevant personas suites (world-books screen gating, character attach, dictionaries, center canvas layout, expression generate, deferred center views). The 10-11 failures in those runs reproduce identically on stashed HEAD (flaky/env set -- before/after stash diff shows zero new failures); the wrapper-gating regression tests specifically all pass. Diff is deletion + comment only.

Modified: tldw_chatbook/UI/Screens/personas_screen.py.
