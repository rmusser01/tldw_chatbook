---
id: TASK-15451
title: Console cost chip must stop re-tokenizing the transcript on every sync tick
status: To Do
assignee: []
created_date: '2026-08-11 12:05'
labels:
  - perf
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verified first-hand in the audit: `_sync_console_cost_chip` builds cost state unconditionally on every `_sync_console_control_bar` pass — including the 0.2 s tick that runs for the whole duration of any active run — because the equality guard at `chat_screen.py:8266` gates only the repaint, not the build. `build_cost_snapshot` (`Chat/console_cost_tracker.py:485-508`) then calls `_estimate_tokens_locally` for every row lacking `ProviderUsage` (all user/system rows, legacy assistant rows, staged evidence) with no caching. With tiktoken absent from base deps (task-2526) the estimator is a per-character Python loop, so a transcript with ~100 KB of user text costs ~50-100 ms per tick on fast hardware, 5×/s, on the event loop — continuous input lag exactly while the user is typing or watching a run.

Fix direction: cache per-message token estimates keyed by message identity + content (rows are frozen once complete), or gate the snapshot rebuild on the store's payload revision. Stability constraints: preserve chip semantics exactly — the pending/streaming exclusion that freezes the total mid-run, the staged-evidence pseudo-row and `~` estimated prefix, the WARM/EXPIRED TTL behavior (task-2115 history), and the revision-gated fingerprint/projection branches which are already correct. Related: task-2525 (modelling gaps), task-2526 (tiktoken dependency — a faster tokenizer alone would NOT fix the per-tick recompute). Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Cost state build does no repeated tokenization of unchanged rows across ticks (unit or probe evidence on a long transcript)
- [ ] #2 The existing cost-chip test surface passes unchanged (mid-stream freeze, staged evidence, ~ prefix, TTL states)
- [ ] #3 0.2 s tick cost measured before/after on a transcript with substantial usage-less text, and recorded in the task
<!-- AC:END -->
