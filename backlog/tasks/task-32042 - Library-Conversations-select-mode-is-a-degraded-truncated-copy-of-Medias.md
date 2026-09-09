---
id: TASK-32042
title: 'Library: Conversations select mode is a degraded, truncated copy of Media''s'
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 14:36'
updated_date: '2026-09-08 19:54'
labels:
  - library
  - conversations
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #7 P1. Conversations' select-mode toolbar renders 'Selec' (truncated 'Select all'), a bare unlabeled marker, and an awkwardly wrapped '0 selected', and its footer omits the space/s hints Media shows. It is the same feature as Media's clean, labelled select mode but inconsistent and less legible. Share the select-toolbar treatment so the two do not diverge.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Conversations select mode shows full, untruncated action labels at supported widths, matching Media's grammar
- [x] #2 The select toolbar is built from the shared treatment rather than a divergent per-canvas copy
- [x] #3 Painted pins assert the Conversations select actions are legible (no mid-word truncation) at 235x52 and 100x30
- [x] #4 PR #2522 boot CSS census fits the unchanged 804,000-byte cap, all generated sheets reproduce, and full-label toolbar paint is preserved.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
PR #2522 CI follow-up (2026-09-08)
ADR required: no
ADR path: backlog/decisions/097-boot-budget-ratchets.md
Reason: Behavior-preserving comment reduction follows the existing boot-budget ratchet; no new architecture decision.

1. Reproduce the CSS budget failure on PR head and clean exact dev base c0a42150d1785c9ef7394def669a0aeb62578ff4; attribute the source growth.
2. Shorten the TASK-32042 toolbar comment while retaining its task reference and content-width/min-width rationale; regenerate the CSS bundle from source. Preserve every selector/declaration, budget and snapshot.
3. Verify the unchanged budget, CSS build integrity and bundle reproducibility, and original toolbar full-label paint at both supported sizes; record the measured bytes and close the follow-up.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Conversations' select-mode toolbar packed all four actions into one `ds-toolbar` row whose scoped CSS forced each child to `width:1fr`, so at the narrow conversations list pane they split evenly (~9 cells each) and truncated to `Selec`/`Exp` with `0 selected` wrapping. Fix (Media parity, task-30043 treatment): split into a summary row (count + `Select all N shown`) above a bulk-action row (`#library-conversations-select-actions`: Clear + Export selected), and change the scoped `#library-conversations-canvas > .ds-toolbar > .library-canvas-action` width from `1fr` to `auto` (min-width:0 kept) so the buttons take their content width. The selector is scoped to #library-conversations-canvas, so Media (its own already-`auto` rule) and Notes/Prompts (no matching selector; base .library-canvas-action carries no width) are unaffected. Kept the `library_disabled_action_label(align=True)` idiom, 'Done stays pressable in select mode', and the `actions_disabled`/empty gating. New pin `test_conversations_select_labels_paint_in_full_like_media` asserts the full labels + one-line count at 235x52 and 100x30 (red first: 'Select all 2 shown' clipped to 'Selec'). The crit6 column-hold pin was correctly adapted from a first-glyph to a word-column measurement (`_painted_word_column(..., 'Export')`, the Notes/Prompts sibling standard) — it still asserts the label holds its column across the 0->1 selection flip; first-glyph diverged by design once the label un-clipped (the align pad reserves the `○ ` width). Bundle rebuilt (conversations rules are in the boot bundle). Files: library_conversations_canvas.py, _agentic_terminal.tcss (+ tldw_cli_modular.tcss bundle), Tests/UI/test_library_multiselect_conversations.py, Docs/User_Guide.

PR #2522 CI follow-up (2026-09-08): exact dev base c0a42150d1785c9ef7394def669a0aeb62578ff4 reproduced the same 804,241/804,000-byte failure as PR head in a clean git archive with the same interpreter; imported paths confirmed the archive was measured. The introducing commit d0693c9e15 raised the six boot sheets from 803,810 to 804,241 bytes, almost entirely through the toolbar rationale comment. Shortened that source comment from 429 to 128 bytes and regenerated tldw_cli_modular.tcss, saving 301 actual boot-parsed bytes. The full rationale remains in this task; non-comment source and generated CSS are byte-identical to HEAD. Cap, snapshot, selectors and declarations are unchanged. No new ADR: behavior-preserving comment reduction implements backlog/decisions/097-boot-budget-ratchets.md.

Verification: 31 targeted tests passed (boot CSS byte guard, all CSS build integrity tests, original full-label toolbar paint at 235x52 and 100x30); measured 803,940/804,000 bytes, headroom 60. All ten generated sheets reproduce via check_bundle_sync.py, comment-stripped equality holds for both changed CSS files, and git diff --check is clean. No Python source or CSS declarations changed; no additional lint/formatter changes were needed. Existing requests dependency warning and expected budget headroom warning were the only warnings. Test evidence: /private/tmp/pr2522-perf-dev-baseline.txt and /private/tmp/pr2522-perf-fix-tests.txt. No full suite run.
<!-- SECTION:NOTES:END -->
