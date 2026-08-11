---
id: TASK-15451
title: Console cost chip must stop re-tokenizing the transcript on every sync tick
status: Done
assignee:
  - '@claude'
created_date: '2026-08-11 12:05'
updated_date: '2026-08-11 22:37'
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
- [x] #1 Cost state build does no repeated tokenization of unchanged rows across ticks (unit or probe evidence on a long transcript)
- [x] #2 The existing cost-chip test surface passes unchanged (mid-stream freeze, staged evidence, ~ prefix, TTL states)
- [x] #3 0.2 s tick cost measured before/after on a transcript with substantial usage-less text, and recorded in the task
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Baseline: run the whole cost-chip test surface green FIRST (Tests/UI/test_console_cost_chip_screen.py, Tests/Chat/test_console_cost_tracker.py, Tests/UI/test_console_cost_modal.py, Tests/Chat/test_console_status_chips_cost.py) and keep it unmodified.

2. Shape decision -- per-row memo, NOT a revision-gated snapshot. Rejecting the payload-revision gate on evidence: `ConsoleChatStore.set_message_usage` (console_chat_store.py:3214) never calls `_bump_payload_revision` -- usage is not payload-affecting -- so in the documented late-attach ordering (Stop path: terminal mark first, usage attach after) a revision-gated snapshot would keep showing the ESTIMATED total after the real priced usage landed. Also fleet_tokens, provider/model and the staged-evidence pseudo-row all change with no revision movement. A memo keyed on the estimator's own inputs cannot go stale by construction.

3. Add `TranscriptTokenEstimateCache` to Chat/console_cost_tracker.py: a keyed memo whose every HIT is verified by comparing the cached (role, content) rows against the live ones (identity fast path, then ==). Correctness therefore does not depend on the key choice -- the key only determines hit rate. Entries hold the store's own string objects (dataclasses.replace shares the reference), so a hit retains no extra memory; the cache is pruned to the keys seen in each pass so it can never outgrow the transcript.

4. Thread it through as an OPTIONAL keyword: `build_cost_snapshot(..., estimate_cache=None)` defaults to today's uncached behavior byte-for-byte (existing tracker tests untouched). ChatScreen owns one instance via a CLASS-level attribute default (`__new__()` fixtures never run `__init__`), passes it per build, and routes the WARM+break_reason `projected_delta_usd` whole-transcript estimate through the same cache -- still calling the module-global `_estimate_tokens_locally` name on a miss so the existing estimator spy test still intercepts.

5. TDD: new test counts `_estimate_tokens_locally` calls across two identical builds on a long transcript -- fails on the current shape (N calls per tick, every tick), passes when the second tick makes zero. Plus tests for: content edit re-estimates only the edited row; model switch invalidates; staged-evidence pseudo-row (rebuilt as a NEW string each pass) still hits.

6. Measure before/after with an isolated probe (scratch HOME/XDG/TLDW_CONFIG_PATH): per-tick cost of the cost-state build on a transcript with substantial usage-less text. Record honest attribution in the task.

7. Re-run the full cost-chip surface unmodified + ruff; self-review the diff; notes + Done; commit task file with code.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Made the Console cost-chip state build O(changed) instead of O(transcript chars) by memoizing the per-row local token estimates across sync ticks. The chip's semantics are untouched: the existing cost-chip test surface (65 tests across Tests/UI/test_console_cost_chip_screen.py, Tests/Chat/test_console_cost_tracker.py, Tests/UI/test_console_cost_modal.py, Tests/Chat/test_console_status_chips_cost.py) passes UNMODIFIED.

## Approach

`TokenEstimateCache` (new, Chat/console_cost_tracker.py) is a verified memo, not an invalidated cache: every HIT is checked against the estimate's full signature -- `(model, provider, ((role, content), ...))` -- before it is served. So the cache KEY only affects the hit rate, never the answer; there is no invalidation protocol that a future mutation site could forget to call. Tuple comparison short-circuits on per-element identity, so the ordinary case (the store hands back the same `str` objects pass after pass, since `messages_for_session` -> `dataclasses.replace` shares the reference) costs pointer compares, and a rebuilt-but-equal string (the staged-evidence pseudo-row, joined afresh every pass) costs one C-level memcmp. Bounded by an LRU cap (4096 entries) so it can never grow without limit; entries hold the caller's own strings, so a live entry costs a tuple rather than a copy.

`build_cost_snapshot` takes it as an OPTIONAL `estimate_cache=None` keyword -- omitted, it estimates every row on every call exactly as before, so every other caller and the whole existing tracker test surface is byte-identical. `ChatScreen` owns one instance (class-level attribute default, since `__new__()` fixtures never run `__init__`) and passes it from `_build_console_cost_state`; the WARM+break_reason `projected_delta_usd` whole-transcript estimate goes through the same memo under its own per-session key, still calling chat_screen's module-global `_estimate_tokens_locally` on a miss so the existing estimator-spy test keeps intercepting it.

## Why not gate the rebuild on payload_revision

Considered and rejected on evidence, and the reason is now a comment at the call site: usage is not payload-affecting, so `ConsoleChatStore.set_message_usage` never calls `_bump_payload_revision`. A real priced usage landing on an already-terminal row (the documented Stop-path ordering, where the terminal mark precedes the attach) would leave a revision-gated snapshot showing the ESTIMATED total until some unrelated edit moved the counter. Provider/model, fleet tokens and the staged-evidence pseudo-row also all change with no revision movement. Deviation from the plan: the plan said prune-to-last-pass; an LRU cap replaced it because the projection entry (written after the snapshot pass) would have been evicted by the next pass's prune, defeating its own caching. The plan's working name `TranscriptTokenEstimateCache` shortened to `TokenEstimateCache`.

## Measured (AC#3)

Isolated probe on the real mounted screen (pytest-hermetic config), 40 rows / 99,310 chars of usage-less text, anthropic/claude-sonnet-4-6, tiktoken absent so the estimator is the per-character chars-floor loop. Median of 20 `_build_console_cost_state()` calls:

| | per tick | at the 0.2 s tick |
|---|---|---|
| before (cache cold each call) | 27.39 ms | 137.0 ms/s |
| after (warm) | 0.36 ms | 1.8 ms/s |

~76x on this transcript; the residual 0.36 ms is dominated by everything OTHER than tokenization (`messages_for_session` alone is 0.08 ms). Honest scope: this removes the REPEAT cost only -- the first tick after any change still pays the estimate for the changed rows, and `build_cost_snapshot` with no cache passed is unchanged at 27.0 ms. The audit's estimate for this size was 50-100 ms/tick; measured here it is 27 ms, so the win is real but the starting figure was smaller than the audit projected on this hardware. tiktoken (task-2526) and the modelling gaps (task-2525) remain open and untouched.

## Files

- `tldw_chatbook/Chat/console_cost_tracker.py` -- `TokenEstimateCache`, `token_estimate_signature`, `_estimate_row_tokens`, `build_cost_snapshot(estimate_cache=...)`
- `tldw_chatbook/UI/Screens/chat_screen.py` -- per-screen cache + both estimate sites in `_build_console_cost_state`
- `Tests/Chat/test_console_cost_estimate_cache.py` (new, 11 tests) -- hits, misses (edited row, model switch, role switch, forced key collision), LRU bound, cached==uncached totals
- `Tests/UI/test_console_cost_chip_estimate_cache.py` (new, 5 tests) -- screen-level estimator call counts across identical ticks, one-row edit, staged evidence, the projection; plus an anti-staleness control that was green before AND after
<!-- SECTION:NOTES:END -->
