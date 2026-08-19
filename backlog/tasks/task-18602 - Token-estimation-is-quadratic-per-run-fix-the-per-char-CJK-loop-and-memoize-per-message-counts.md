---
id: TASK-18602
title: >-
  Token estimation is quadratic per run: fix the per-char CJK loop and memoize
  per-message counts
status: Done
assignee: []
created_date: '2026-08-18 21:00'
labels:
  - performance
  - agents
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Token estimation re-counts an entire conversation from scratch on every turn, and
the innermost step is a Python-level loop over every character.

`Utils/token_counter._chars_estimate` computes its CJK share with
`sum(1 for ch in text if _is_cjk(ch))` — one Python function call per character.
Measured on this machine: **158.8 ms** for a 640 KB payload, where
`str.isascii()` answers the same question for all-ASCII text in **0.001 ms**.

On top of that, `count_tokens_messages` is called with the whole growing message
list every turn, so the cost is quadratic in turn count. Measured over a
simulated 400-turn agent run: **166 ms at turn 400**, **33.1 s cumulative**.
Counting only each turn's NEW text instead totals **0.16 s**.

This is not agent-only. `Chat/console_history_budget.py` calls
`count_tokens_messages` and is imported by `console_chat_controller`, so it runs
on the normal Console send path; the cost ticker, chat token events, and
world-info processing use the same estimator.

The ASCII guard fixes the no-tokenizer case outright. Memoization is the
structural fix and is the one that also helps installs WITH tiktoken, where the
per-turn re-encode of the whole history is the dominant cost.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Estimating all-ASCII text no longer runs a per-character Python loop.
- [x] #2 Re-estimating an unchanged string is served from a memo rather than recomputed, for every caller of the estimator.
- [x] #3 Estimated token counts are unchanged for ASCII, CJK, mixed, and multimodal part-list content.
- [x] #4 The memo is safe to use from the agent worker thread and the UI thread at once.
- [x] #5 The memo is bounded and holds no strong reference to the estimated text.
- [x] #6 A benchmark records before/after cost for a growing payload so the regression is visible if it returns.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two independent fixes in `Utils/token_counter.py`, each measured.

**1. The CJK count no longer runs a Python loop.** `_count_cjk` replaces
`sum(1 for ch in text if _is_cjk(ch))` with an ASCII short-circuit
(`str.isascii()`) and, for non-ASCII text, a single `re.subn` pass over a char
class built from the same `_CJK_RANGES` tuple `_is_cjk` uses — so the two
encodings of that fact cannot drift. Equivalence was proven exhaustively over
U+0000–U+10FFF, over every range boundary, and over 300 randomized mixed
strings: zero mismatches.

**2. `estimate_tokens` is memoized.** Keyed by
`(model, provider, len(text), hash(text))` rather than by the text, so the cache
holds NO strong reference and memoizing a 600 KB message cannot pin it in memory
(pinned by a weakref test). CPython caches a str's hash on the object, so repeat
lookups of the same message are a dict probe. The read is unlocked (a dict `get`
is atomic under the GIL and a race can only cost one recompute); only the write
takes the lock, so the agent worker thread and the UI thread can estimate
concurrently.

Measured by `Helper_Scripts/Benchmarks/token_estimate_benchmark.py` on a
400-turn conversation, with the two fixes separated because they are not equal
partners:

| | cumulative estimator CPU |
|---|---|
| before (per-char loop, no memo) | 35.02 s |
| + ASCII fast path | 0.113 s — **310x** |
| + memo (shipped) | 0.054 s — 2.1x further, **650x total** |

Single-call CJK scan of a 640 KB payload: 171.9 ms -> 0.007 ms (**25,000x**).

The LIVE user-facing path is `bound_messages_to_window`, which
`console_chat_controller` runs once per Console send to decide what fits in the
context window:

| history | before | after |
|---|---|---|
| 60 turns / 178 KB | 62.5 ms | 0.6 ms |
| 120 turns / 354 KB | 153.4 ms | 1.1 ms |
| 240 turns / 707 KB | 223.8 ms | 1.8 ms |

That is blocking work removed from every send in a long conversation. Stated
separately from the agent-loop figure on purpose: the 400-turn number is a worst
case, this is what a user actually feels.

One measurement was investigated and NOT claimed: `chat_token_events`'s pending-
input path costs ~98 ms per keystroke on a 354 KB history and has no dirty gate,
which would have been a dramatic typing-latency headline -- but task-17653
retired the footer token counter, so that path is not live on Console. Its only
remaining caller is `db_status_manager`, which uses the gated variant.

Read that split honestly: on the shipped default (no tokenizer installed) the
ASCII fast path does nearly all the work and the memo adds 2.1x. The memo earns
its place on installs WITH tiktoken, where the per-turn re-encode of the whole
history is the dominant cost and no fast path can remove it — a configuration
this machine cannot measure, so the benchmark reports which tier it ran.

Two benchmark drafts had to be thrown away before these numbers were
trustworthy, both flattering: one reimplemented the "before" case with a
frozenset lookup instead of calling the real `_is_cjk` generator (understating
the baseline ~25x), and one appended identical content every turn, which let the
memo collapse a whole payload into one entry and made every configuration look
the same. Both traps are called out in the script.

**Trade-off.** A hash collision would serve one text's estimate for another's.
Mitigated by including length and tokenizer identity in the key, and bounded in
consequence by what this function is: an estimate whose own chars tier already
applies approximation headroom. It is never used where an exact count is
required. Documented at the cache.

The regression guard asserts STRUCTURE (estimator invocations across 100 turns),
not wall-clock, so it cannot flake on a loaded machine.

Files: `Utils/token_counter.py`, `Tests/Chat/test_token_estimate_performance.py`
(new, 23), `Helper_Scripts/Benchmarks/token_estimate_benchmark.py` (new).
<!-- SECTION:NOTES:END -->
