---
id: TASK-22204
title: >-
  Resolve the Console expression state once per tick and stop re-copying the transcript
status: Done
updated_date: '2026-08-24'
assignee:
  - '@claude'
created_date: '2026-08-24'
labels:
  - performance
  - console
  - streaming
priority: high
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22204).

New with PR #2020 (streaming emotes), default-ON (`resolve_show_character_avatar` defaults
True, `Chat/console_image_view.py:137-152`). `UI/Console_Modules/character.py:284-295`:
`_current_request` calls `resolve_console_expression_selection` and then — whenever
`selection.source` is `idle`/`operational`, the common case — re-runs
`resolve_console_expression_state`, and `_request_is_current` (`:353-354`) re-enters
`_current_request`. Every resolution funnels into
`store.messages_for_session(...)` (`Chat/console_expression_state.py:71`), which
materializes stream buffers and returns a `dataclasses.replace` copy of every message
(`Chat/console_chat_store.py:5227-5234`). At the pin this was one copy per tick; now a
repainting tick pays 4-6. The 0.2 s tick runs for the whole duration of a run, so this is
10-30 whole-transcript copies per second while streaming. Context: the tick already pays
~3 other full copies (native transcript, cost chip, setup-card guidance) — a shared
per-tick snapshot seam would collapse all of them, but the minimum fix is restoring 1 copy
for the avatar path.

## Acceptance Criteria

- [x] One `messages_for_session` copy at most per avatar refresh per tick (shared snapshot through both resolutions; `_request_is_current` compares against the already-built request) — proven by a call-count probe during a simulated streaming tick (`Tests/UI/test_console_character_avatar_copy_budget.py`)
- [x] Emote/idle/operational selection behavior unchanged (existing expression tests green — 218 passed across the avatar/emote/expression files; one test stub's signature widened to accept the new kwarg, its assertions untouched)
- [x] Stretch — DECLINED for this task via this AC's own split-to-follow-up escape hatch: a shared per-tick snapshot across avatar/guidance/cost-chip crosses three controllers with different staleness contracts and needs its own invalidation design; reason recorded in Implementation Notes, direction stays documented under finding 22204 in `Docs/Design/2026-08-24-holistic-perf-review.md`
- [x] Per-tick copy count during streaming measured before/after (300-message session: repaint tick 8.0 → 1.0 copies, median 7.17 ms → 0.93 ms; steady tick 2.0 → 1.0 copies, 1.80 ms → 0.85 ms)

## Implementation Plan

1. Semantics first: establish why `_current_request` re-runs
   `resolve_console_expression_state` after
   `resolve_console_expression_selection` for idle/operational sources.
   Finding (git `7bd039077`, PR #2020): pre-#2020 the method called only
   `resolve_console_expression_state`; #2020 added the selection call and kept
   the legacy call gated to idle/operational — over one message snapshot the
   two are provably identical for those sources (the explicit params only
   change the streaming+matching branch, which yields source `explicit`, and
   mood-label completes yield `historical`; both are excluded by the gate).
   The retained call is (a) a defensive legacy fallback and (b) the
   monkeypatch seam eight existing avatar tests drive states through — so it
   must SURVIVE, not be deleted.
2. Add a keyword-only `messages=None` pre-fetched-snapshot parameter to both
   resolvers in `Chat/console_expression_state.py`;
   `resolve_console_expression_state` forwards it. `None` keeps today's
   fetch-and-fail-soft behavior byte-for-byte.
3. `_current_request` fetches `store.messages_for_session(...)` exactly once
   (same guards the resolver uses: react on, store and session present;
   exception → empty snapshot, which resolves idle exactly like today's
   except-path) and passes that one snapshot to BOTH resolver calls.
4. Fence: store the last-built request on the controller
   (`_latest_built_request`, written by `_current_request`); change
   `_request_is_current` to compare against it instead of re-entering
   `_current_request`. Freshness is tick-bound (the 0.2 s sync tick rebuilds
   it). The avatar-hidden early return in `_refresh_avatar_request` clears it
   so paints in flight across a visibility toggle still drop; `_is_mounted`
   stays in the fence for teardown.
5. Red-first probe: new `Tests/UI/test_console_character_avatar_copy_budget.py`
   drives the real `ConsoleCharacterController` + real `ConsoleChatStore`
   (streaming assistant message) through one full paint tick with
   `messages_for_session` wrapped by a counter; assert exactly 1 copy per
   tick (red on current code: 10+), plus zero-message and mid-teardown
   failure paths, plus a resolver-snapshot unit test in
   `Tests/Chat/test_console_expression_state.py`.
6. Measure per-tick copies and wall time before/after with a 300-message
   synthetic session (scratch harness, both numbers into Implementation
   Notes).
7. Adapt the one strict-signature test stub
   (`test_decode_completion_live_fences_every_avatar_request_input`) to accept
   the new kwarg; run the avatar/expression/emote-adjacent test files +
   `--collect-only` sweep + `./scripts/preflight.sh`, tee everything.
8. Mutation test: reintroduce an unshared second resolution and confirm the
   probe reds; revert.

## Implementation Notes

Restored the pin's one-transcript-copy-per-tick budget for the Console avatar
path without changing what any tick resolves.

**Why the double call existed (the semantics question).** Pre-#2020,
`_current_request` called only `resolve_console_expression_state`. PR #2020
(`7bd039077`) added `resolve_console_expression_selection` for explicit/
historical emotes and KEPT the legacy call gated to idle/operational sources.
Over one message snapshot the two are provably identical for those sources:
the explicit parameters only alter the streaming-and-matching branch (which
returns source `explicit`), and a mood-labelled complete returns `historical`
— both excluded by the gate; every other path is byte-identical code. The
retained call's real value is (a) a defensive fallback and (b) the seam eight
avatar tests monkeypatch (`character.resolve_console_expression_state`) to
drive operational states without message fixtures. So it was PRESERVED, not
deleted: both resolvers now accept a keyword-only pre-fetched `messages`
snapshot, `_current_request` fetches once and passes it to both calls, and a
monkeypatched state resolver still overrides the state exactly as before.

**Fence.** `_request_is_current` no longer re-enters `_current_request` (2
copies per check); it compares against `_latest_built_request`, which every
build writes. Freshness is tick-bound: the 0.2 s sync tick rebuilds it each
pass, so a world change fences an in-flight paint at the next rebuild — at
most one tick later, and immediately for forced refreshes (the existing
stale-paint race tests, which insert a rebuild between mutation and release,
pass unchanged for every context-change parametrization). The avatar-hidden
early return and `_invalidate_actor` clear the baseline so in-flight paints
drop across a visibility toggle or actor invalidation; `_is_mounted()` stays
in the fence for teardown.

**Persistence-timing check (read-that-writes).** `messages_for_session`
folds stream buffers and may persist a newly materialized pending row.
Verified idempotent: `_fold_stream_buffer_without_persistence` no-ops when
`_stream_materialized_counts` already matches the buffer, and
`_persist_pending_message_if_ready` no-ops once `_persist_new_message`
discards the id from `_pending_persistence_message_ids`. Within one
synchronous tick the 2nd–14th calls were therefore pure copies; dropping them
moves nothing observable — the one remaining call per tick still folds and
persists at the same tick boundary.

**Measured (300-message synthetic session, 200 ticks, real store + real
controller).** Repaint tick: 8.0 → 1.0 copies/tick, median 7.17 ms → 0.93 ms.
Steady (deduped) tick: 2.0 → 1.0 copies, median 1.80 ms → 0.85 ms. Cold
full-paint tick in the probe: 14 copies before, exactly 1 after.

**Mutation-tested.** Reintroducing an unshared second resolution reds the
probe (2 copies, 3 tests fail); making the fence re-entrant again also reds
it (7 copies, 2 tests fail). Restores verified byte-identical to the commit.

**Stretch AC declined** (allowed by its own wording): a shared per-tick
transcript snapshot for avatar + guidance + cost-chip spans three consumers
in different controllers with different staleness contracts (the control-bar
comment at `chat_screen.py` explicitly withdrew one-tuple-per-tick reuse for
the rail-visibility half after PR #660 caught a staleness regression). That
seam needs its own invalidation design and review; bolting it on here
inflates risk for a task whose finding is the avatar path. Direction remains
documented under finding 22204 in the holistic perf review.

**Files.** `tldw_chatbook/Chat/console_expression_state.py` (snapshot
parameter on both resolvers), `tldw_chatbook/UI/Console_Modules/character.py`
(shared snapshot; non-re-entrant fence; baseline clears),
`Tests/UI/test_console_character_avatar_copy_budget.py` (new probes: budget,
steady-state, zero-message, mid-teardown, storeless teardown),
`Tests/Chat/test_console_expression_state.py` (snapshot-path unit tests),
`Tests/UI/test_console_character_avatar.py` (one stub signature widened).

**Not mine, found on the way:**
`Tests/Chat/test_console_chat_controller.py::test_provider_switch_ignores_
unrelated_completed_continuation_history` errors at TEARDOWN on any machine
without a cached tiktoken `o200k_base` encoding — `token_counter.
get_tiktoken_encoding` tries to download it and the network guard blocks the
egress (same class as the CI blocked-HF-egress finding). Reproduced
byte-for-byte on base `983aa5878` with this task's files reverted.
