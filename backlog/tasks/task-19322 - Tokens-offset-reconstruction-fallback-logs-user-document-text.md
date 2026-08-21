---
id: TASK-19322
title: Tokens offset-reconstruction fallback logs user document text
status: To Do
assignee: []
created_date: '2026-08-20'
labels:
  - security
  - diagnostics
  - chunking
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from TASK-19191's independent review; re-verified at dev `a542fd463`.
`tldw_chatbook/Chunking/engine/strategies/tokens.py:779-782` — when a token
piece cannot be located during offset reconstruction, the fallback logs
`piece={repr(piece[:50] + '...' if len(piece) > 50 else piece)}` — up to 50
characters of raw user document text — at debug level.

DEBUG is not a defense under the TASK-15103/15600 programme bar (ADR-029,
`backlog/decisions/029-local-private-data-boundary.md`): users run with
debug sinks enabled, and content redaction is level-independent. Repair in
the house idiom — replace the content with what actually diagnoses the
mismatch: piece length, `pos`, and, if correlation across records matters,
a short stable hash of the piece. The diagnostic's job is "the tokenizer
produced a piece the text search could not align, at this position" — none
of that requires the characters themselves.

Owner ruling applies (stability-over-quick-wins, 2026-08-11): redact at
the call site in the established idiom; no formatter/sink cleverness.

Knock-on: `tokens.py` is a TASK-494 owner row in
`Docs/security/production-diagnostic-inventory.json` (call_count 28,
digest `9d6d1bc7cce0c3cc6d4b`) — the repair changes the digest, so
regenerate the inventory with only the reviewed delta in the same PR and
keep `scripts/check_persistent_diagnostic_inventory.py` green (the step
TASK-19042/19043 initially missed; 2026-08-20 lesson in
`backlog/docs/lessons-testing-evidence.md`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The offset-reconstruction fallback diagnostic records no user document text at any log level; it still records enough to diagnose the misalignment (piece length, position, and/or a stable hash)
- [ ] #2 The persistent diagnostic inventory's `tokens.py` owner row is regenerated with only the reviewed delta in the same PR and `scripts/check_persistent_diagnostic_inventory.py` passes
- [ ] #3 Regression coverage pins the repaired shape so document text re-entering this diagnostic turns a test red
- [ ] #4 Offset-reconstruction behavior itself is unchanged — only the log record content changes
<!-- AC:END -->
