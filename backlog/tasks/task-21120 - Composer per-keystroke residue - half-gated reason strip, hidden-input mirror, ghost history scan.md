---
id: TASK-21120
title: >-
  Composer per-keystroke residue - half-gated reason strip, hidden-input mirror, ghost history scan
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - console
  - composer
priority: low
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21120).

Per printable key in `console_composer_bar.py`: `_sync_send_disabled_reason` calls
`strip.update(Content(reason))` unconditionally (:1588/:1596) - the computed `reason_changed`
gate covers only the ARIA announcement (the audit's known half-gate pattern); the hidden
compatibility `Input.value` is re-set with the full canonical draft (O(draft), firing a second
Changed handler, :1532-1539); ghost text runs a reverse linear scan of prompt history per draft
render AND per 0.5 s blink tick (:4214-4240).

## Acceptance Criteria

- [ ] The reason strip updates only when reason_changed; the hidden-input mirror skips unchanged text; the ghost-text history scan is capped or cached
- [ ] Composer behavior (send gating, ARIA announcements, ghost suggestions) unchanged - existing composer tests green

## Re-verification against dev 2be18842a (2026-08-23)

An independent read-only pass re-checked all three legs. **Mis-stated on all three, and two of
the three prescribed fixes remove essentially nothing.** The file has moved to
`tldw_chatbook/Widgets/Console/console_composer_bar.py`.

**Leg 1 (reason strip) — drop it.** The mechanism is real (`_sync_send_disabled_reason` at
`:1566-1607` runs unconditionally from `:1717`), but the claim that the `reason_changed` gate
"covers only the ARIA announcement" is false: there is no ARIA announcement there. `reason_changed`
gates a draft re-window at `:1750-1757`. The ungated part costs one `query_one`, one `Content("")`,
six style writes that Textual already no-ops when unchanged, and a `refresh(layout=True)` that
folds into a layout pass `_refresh_visible_draft` has already armed on the same keystroke.
Gating it removes microseconds.

**Leg 2 (hidden-input mirror) — real cost, wrong fix.** "Skip unchanged text" can never fire while
typing, since the value always differs; Textual already skips unchanged reactive sets. The actual
O(draft) cost is `Input._watch_value` recomputing `virtual_size` via `cell_len` over the whole
draft on every key. And the mirror cannot simply be removed: `chat_screen.py:16395` has a live
`@on(Input.Changed)` consumer, and `draft_text()` / `_has_any_draft_content()` /
`_display_draft_text()` still read the Input as the pre-initialisation fallback.

**Leg 3 (ghost scan) — keep.** `_ghost_suffix` (`:4214-4240`) → `PromptHistory.complete`
(`Chat/prompt_history.py:247-265`) is a reverse linear scan over up to 1000 entries, and the
worst case (novel prefix, no match) is the common case: ~1000 `startswith`, order 50-100 us.
Blink interval is 0.53 s, not 0.5, and the timer is paused unless the composer is focused.

**Split out**: the biggest cost found in this file is not in this task at all — the blink tick
arms a full layout pass ~2x/second while merely focused and idle. Filed separately as TASK-21692.

**Action**: rewrite this task down to the ghost-scan cap, or close it and keep only 21692.
