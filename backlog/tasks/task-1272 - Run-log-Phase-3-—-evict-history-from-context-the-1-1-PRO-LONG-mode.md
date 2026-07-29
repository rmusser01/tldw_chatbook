---
id: TASK-1272
title: 'Run log Phase 3 — evict history from context (the 1:1 PRO-LONG mode)'
status: In Progress
assignee:
  - '@claude'
created_date: '2026-07-28 00:00'
updated_date: '2026-07-29 00:01'
labels:
  - agents
  - run-log
  - llm
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**This is the phase that actually delivers the original goals.** Phase 1 (PR #1066) is
deliberately additive: it writes a lossless log and makes truncated content recoverable, but
`run_agent_loop`'s `messages` list is untouched, so context usage is unchanged. Long-horizon
runs and small-context local models — goals 1 and 2 of the design — land here.

The mechanism: keep recent rounds in context verbatim, replace older ones with a pointer to
the log, and let the agent retrieve on demand via `search_run_log`. The log is already the
authoritative record, and the record format already carries `call=<tool_call_id>` for exactly
this purpose.

**Three traps, all discovered during Phase 1 and recorded so they are not rediscovered:**

1. **Whole-group eviction is mandatory.** The native tool-call protocol pairs an assistant
   `tool_calls` echo with its `role="tool"` replies by `tool_call_id`. Evicting either half
   alone produces a request that strict providers reject. Any policy must operate on entire
   call/result groups.

2. **Reuse `bound_messages_to_window`, do not reimplement it.** TASK-322 shipped
   `Chat/console_history_budget.py` with window lookup, safety margin, reply reservation and
   system-prefix preservation already solved. It also already bounds the history an agent run
   *starts* from, because `console_chat_controller.py` does `agent_messages =
   list(provider_messages)`. This task is the in-run bound; the two are layered, not
   alternatives.

3. **`_group_turns` is wrong for fence-protocol runs.** It splits on `role == "user"`, and its
   docstring notes it never splits a tool_call/tool_result pair *"were tool rows ever present
   in the payload"* — they never are on the Console send path it was built for. In an agent
   run they are, and the two protocols differ: native appends `{"role": "tool", ...}` (grouped
   correctly), but **fence appends `{"role": "user", "content": "Tool result for ..."}`**,
   which reads as a new turn boundary and splits an assistant turn from the result answering
   it. Fence is the protocol local models use — precisely the case this phase targets — so
   reuse is correct for native runs and broken for fence runs until grouping learns the
   convention, or grouping is done on the log's own record structure where `call=` pairs them
   unambiguously.

Also note a Phase 1 failure mode this phase resolves (design spec §10.2): because nothing is
evicted today, a `search_run_log` result enters history like any other tool result, so heavy
log searching currently *increases* context pressure rather than reducing it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An agent run bounded by a small context window completes work that currently fails, with evicted history replaced by a pointer the agent can act on
- [x] #2 Eviction operates on whole call/result groups; no request is ever emitted with an orphaned assistant `tool_calls` echo or an orphaned `role="tool"` reply
- [x] #3 Correct behaviour is proven for BOTH the native and the fence tool-call protocols, with a test that would fail if fence tool-results were treated as turn boundaries
- [x] #4 The trimming primitive from `console_history_budget.py` is reused rather than reimplemented, or a recorded reason explains why it could not be
- [x] #5 The mode is configurable and off by default, so existing runs are unchanged until opted in
- [ ] #6 A live run against a local small-context model demonstrates a task completing that does not complete with eviction disabled
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read Phase 1/2 code (agent_service._make_call_model, agent_runtime._append_tool_result) and the design spec's Phase 3 section (10, 10.1, 10.2, 14.1).
2. Extend console_history_budget.py: add an optional is_turn_boundary predicate to _group_turns/bound_messages_to_window (default unchanged, so Console callers are byte-identical), and a dropped_turns count on BoundResult.
3. Promote the fence tool-result prefix ("Tool result for ") to a shared constant in agent_models.py, used by both agent_runtime._append_tool_result and the new eviction module, so the two can never drift.
4. Add Agents/run_log_eviction.py: a round-boundary predicate (every assistant message starts a new round; a native role="tool" reply or a fence role="user" tool-result row is a continuation) and bound_history_for_send(), which calls bound_messages_to_window and splices in a synthetic note when something was dropped.
5. Wire bound_history_for_send into agent_service._make_call_model's call_model closure, gated on log_active (reused verbatim) AND a new off-by-default [agents] run_log_evict_enabled config flag (via run_log._setting, so it gets the same env-var/TOML/default tiering as the other run-log keys).
6. Tests: unit tests proving the fence/native round-boundary fix against the raw primitive (including the explicit "force naive grouping" experiment), plus integration tests through AgentService.run_turn proving the flag-off/log-unavailable hard gates and the end-to-end payload shape.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented eviction at the SEND seam (agent_service._make_call_model's call_model
closure), never touching run_agent_loop's own messages list. Reused
bound_messages_to_window (console_history_budget.py) rather than reimplementing it,
extending it with an optional is_turn_boundary predicate (default unchanged, so
every existing Console call site stays byte-identical) and a new dropped_turns
count on BoundResult.

Key design call beyond the literal "fix _group_turns for fence" framing: the
reused primitive's "turn" is Console's own -- anchored on the last human message
-- which inside a single agent run (one such message, at the start) would
collapse the run's ENTIRE growth into one undroppable "current turn" and evict
nothing while a run is in progress, defeating goals 1/2 (long single run, small
local model). Agents/run_log_eviction.py therefore uses a finer ROUND boundary:
every assistant-authored message starts a new round; a native role="tool" reply
or a fence role="user" tool-result row is a continuation, never a boundary.
Verified via an explicit experiment (recorded in the task's test file) that the
UNMODIFIED primitive (no is_turn_boundary) orphans a fence tool-result and never
trims a native run's own growth at all -- both failure modes the round boundary
fixes, with no orphaned call/result pair for either protocol at any drop size.

FENCE_TOOL_RESULT_PREFIX ("Tool result for ") promoted to a shared constant in
agent_models.py so agent_runtime._append_tool_result and the eviction module's
protocol check can never drift apart.

Gated on log_active (reused verbatim from _run_one -- the same condition gating
the tool and the prompt section) AND a new [agents] run_log_evict_enabled flag,
off by default, resolved via run_log._setting (same env-var/TOML/default tiering
as the other run-log keys). When something drops, a role="user" synthetic note
(not role="system" -- some local chat templates reject a system row that isn't
first) is spliced in naming a round count and search_run_log; never a specific
record number, since the loop doesn't track which record backs which round.
Eviction never raises: any failure inside bound_history_for_send is caught,
logged at warning, and degrades to sending the full history for that turn.

Files: tldw_chatbook/Agents/run_log_eviction.py (new),
tldw_chatbook/Agents/agent_service.py, tldw_chatbook/Agents/agent_models.py,
tldw_chatbook/Agents/agent_runtime.py, tldw_chatbook/Chat/console_history_budget.py,
Tests/Agents/test_run_log_eviction.py, design spec doc updated (§8, §10).

AC #6 (live run against a local small-context model) was later performed by the
coordinator against llama.cpp gemma-4-26B and confirmed the mechanism works as
designed, but also found the defect recorded below -- see that finding for the
fix. Live verification is otherwise not reproducible inside this sandbox (no
local model available here).

--- FOLLOW-UP (2026-07-28, same day): live-verified task-instruction amnesia ---

The round-boundary fix above is necessary but was not sufficient. Live testing
(coordinator, llama.cpp gemma-4-26B, fence protocol, a read-four-files-then-
report task) found that once eviction actually started dropping rounds, the
agent would either return an empty answer or derail into narrating about its
own log instead of finishing. A flag-on-but-inactive control run (large window,
eviction never fires) was byte-identical to flag-off, isolating the cause to
eviction ACTUALLY DROPPING rounds, not to the flag itself.

Root cause: bound_messages_to_window's contract -- preserve the leading system
prefix and "the current turn" (the LAST role="user" row to the end) -- means
something different for an agent run than for a Console chat, and nothing in
run_log_eviction.py pinned the difference. Console: every new human message
restates intent, so "the last one" is the live context. An agent run: exactly
ONE real role="user" row exists -- the task, as the very first message -- and
every later row bearing that role is a fence tool-result wearing it. So "the
last user message" in an agent run is a tool result, and the task instruction
sits in kept_turns[0], the OLDEST group, evictable like any other round. Once
dropped, the agent no longer knows what it was asked to do.

Fix: extended (not forked) the primitive. bound_messages_to_window gained
pin_first_user: bool = False (default off; no Console call site passes it, so
Console is unaffected -- confirmed by a full Tests/Chat run showing only the
pre-existing 4 failures / 13 errors). When True, the pinned prefix extends
through the first role="user" row found after the leading system rows. This
scan needs no protocol awareness, unlike the backward "last user" search: no
tool result can ever be emitted before the task that triggered it, so the
FIRST role="user" row scanning forward is unambiguously the task, for either
protocol. run_log_eviction.bound_history_for_send now always passes
pin_first_user=True.

Degenerate case, decided deliberately per the coordinator's instruction: if the
pinned prefix (system + task) plus the current turn alone already exceed the
window, the primitive's existing "if nothing fits, drop every middle turn"
fallback still returns them as-is -- an over-budget payload the provider may
reject, rather than ever silently dropping the task instruction. This required
no new code: it is the same fallback that already governs the plain
system-prefix-only case.

Verified with the exact behavioural bar requested: forced pin_first_user=False
at the production call site, confirmed the new regression tests FAIL for both
protocols (fence and native) with the same "task instruction missing from a
payload actually sent" assertion the coordinator's live run exhibited, restored
the fix, confirmed all tests pass again. Full Tests/Agents (553) and Tests/Chat
(2684 passed / 4 known failures / 13 known errors) both re-run clean afterward.

LESSON for future reuse of bound_messages_to_window: "preserve the current
turn" is not a self-evidently-safe invariant to inherit by default -- it is
defined relative to what "a turn" means for the CALLER's payload shape, and an
agent run's shape (one real user row, everything after it is loop-generated)
breaks the assumption Console was built under. Always ask what a caller's
FIRST user-role row means, not just its last.

--- SECOND FOLLOW-UP (2026-07-28, same day): live-verified missing recent-rounds floor ---

The pin fix above was necessary but not sufficient. A further live run
(coordinator, same llama.cpp gemma-4-26B setup, eviction on, window 3000)
produced two runs with byte-identical payload sequences [1402, 1899, 1985,
1985], both status=stuck. The run log showed why: the agent read three
files, then re-read the first two, then the cycle detector fired. Calls 3
and 4 being byte-identical at the same token count is the signature of a
fixed point -- eviction removing exactly as many new rounds as are added,
because "keep whatever fits" under a tight enough window degenerates to
keeping ONLY the current round. The task-1272 backlog's own wording --
"keep recent ROUNDS verbatim" (plural) -- was under-implemented: only ONE
round was actually guaranteed regardless of window size.

Fix: added a floor. bound_messages_to_window gained min_recent_turns: int =
0 (default 0, so Console is unaffected -- 0 and 1 both mean "current turn
only", the original contract). It caps how far the existing oldest-first
binary search is even ALLOWED to drop:
  max_drop = max(0, len(kept_turns) - max(0, min_recent_turns - 1))
(current_turn already counts as 1 of the floor). New [agents]
run_log_evict_min_recent_rounds config key (env-var/TOML/default tiering
via run_log._setting, coerced defensively -- coerce_min_recent_rounds in
run_log_eviction.py, non-negative int, 0 is a valid deliberate opt-out).
Default chosen: 4, because the live reproduction (read four files, one
round each, then answer) needs exactly that many rounds simultaneously
visible to avoid re-reading any of them; smaller risked the same bug one
round later, larger meaningfully narrows what eviction can ever save on a
small-context model, which is this phase's whole point.

Degenerate case (prefix + floor > window), decided the same way as the
pin's: the SAME existing "if nothing fits, drop the most allowed" fallback
already in bound_messages_to_window governs -- an over-budget send rather
than ever shrinking below the floor. No new degenerate-case code was
needed; both the pin and the floor reuse the one fallback the primitive
already had (`best = hi` initialization in the binary search).

Verified with the same bar as both prior fixes: forced min_recent_turns=0
at the production call site (tldw_chatbook/Agents/run_log_eviction.py),
confirmed the two floor-guarantee tests fail -- one showing only a single
round visible instead of >1, the other showing a round that should be
within the floor missing -- restored, confirmed all 27 eviction tests green
again. Full Tests/Agents (560) and Tests/Chat (2684 passed / 4 known
failures / 13 known errors -- unchanged baseline) both re-run clean with
PLAIN pytest output (not -q), per the coordinator's explicit instruction
after -q was found to have hidden real regressions in this repo before.

Files touched by this follow-up: tldw_chatbook/Chat/console_history_budget.py
(min_recent_turns param + binary-search cap), tldw_chatbook/Agents/
run_log_eviction.py (DEFAULT_MIN_RECENT_ROUNDS, RUN_LOG_EVICT_MIN_RECENT_
ROUNDS_KEY, coerce_min_recent_rounds, min_recent_rounds param threaded to
bound_history_for_send), tldw_chatbook/Agents/agent_service.py (resolves
and passes the floor alongside evict_enabled), Tests/Agents/
test_run_log_eviction.py (+14 tests), design spec doc (§10).

LESSON: a token-budget trimmer with no floor on retained UNITS (only a
byte/token ceiling) can starve down to a single unit under a tight enough
window, regardless of how many units exist. For a human conversation that
unit is a whole exchange and starving to one is merely terse; for an
autonomous agent loop the unit is a round of WORK, and starving to one
erases the agent's own short-term memory of what it just did, causing it to
repeat completed steps. Any reuse of a "keep whatever fits" trimmer for an
agentic (not conversational) history needs an explicit minimum-recent-units
floor from the start, not as an afterthought once cycle detection starts
firing.

--- FINAL FINDING (2026-07-28, same day): AC #6 attempted live, NOT met -- Phase 3's value is model-dependent ---

Status deliberately kept at "In Progress", not "Done": AC #6 ("a live run
against a local small-context model demonstrates a task completing that
does not complete with eviction disabled") was attempted live by the
coordinator with both fixes above in place, and the outcome did NOT satisfy
it. This backlog has only To Do / In Progress / Done as statuses -- there is
no dedicated "shipped with a documented limitation" state, so "In Progress"
is being used as the closest honest fit; this note is the explicit record
of that choice, per the coordinator's instruction not to mark Done with an
unmet AC.

The live evidence (coordinator, llama.cpp gemma-4-26B, fence protocol,
10-file sequential task, declared window 6000):
  eviction OFF -> payload 1426..6909, OVERFLOWS the 6000 window, answer CORRECT
  eviction ON  -> payload 1426..4501 then PLATEAUS under the window (the
                  mechanism working exactly as designed), but the run ends
                  status=stuck with an EMPTY answer -- the log shows the
                  agent re-reading files it had already read.
Raising max_tokens 700->2500 did not change the outcome (20 calls, a flat
plateau around 4500 across 13 consecutive turns, still stuck) -- ruling out
a completion-budget artifact. Raising the recent-rounds floor delays where
the wall is hit; it does not remove it.

This is NOT a defect in the implementation -- the payload bound is correct
and provably bounded, exactly per the two fixes already recorded above. The
gap is in the PREMISE: PRO-LONG (and this design) assumes the agent
compensates for evicted history by actively querying search_run_log. A 26B
local model on the fence protocol does not reliably do this -- strip its
recent turns and it re-attempts completed work instead of searching for
what it already learned, until the cycle detector kills the run. The
paper's own headline numbers are measured on frontier models, which are the
ones actually capable of driving that recovery loop.

Conclusion, recorded plainly rather than softened: Phase 3's MECHANISM is
verified (bounded payload, no orphaned pairs, task pinned, floor honored --
all of that holds). Whether the mechanism translates into completing MORE
work than eviction-off depends on the model being strong enough to recover
from the log by searching it, rather than repeating itself. There is a real
irony worth stating directly: small-context local models were one of this
phase's two motivating goals, and they are the class of model LEAST likely
to have the reasoning strength the mechanism assumes. A frontier model
behind an artificially small declared window is the case most likely to
benefit; a weak local model behind a small NATIVE window -- the headline
scenario this phase was framed around -- is the case most likely to
plateau into stuck instead of completing.

No code, test, or default changed for this finding -- documentation only,
per the coordinator's explicit instruction. Recorded in three places so it
cannot be missed by a future reader at any of the natural entry points:
  - Design spec §10 (full narrative + evidence) and §10.1 (goal/phase
    matrix no longer claims Phase 3 delivers goals 1/2 outright -- it now
    reads "Mechanism: yes. Outcome: model-dependent").
  - Agents/run_log_eviction.py's module docstring AND
    RUN_LOG_EVICT_ENABLED_KEY's own comment -- whoever turns the flag on
    reads it right there, not just in a design doc.
  - This task's Implementation Notes (here).

Future idea, NOT implemented, per the coordinator's "say so, don't build
it" instruction: the synthetic eviction note (_synthetic_note) currently
states a round count and names search_run_log once. A more directive
wording -- e.g. explicitly naming the tool(s) already used in the dropped
rounds, or an imperative "before repeating a tool call, search the log for
it first" -- might raise the odds of a weaker model querying the log
instead of repeating itself. This is speculative and untested (recorded
also in the design spec §11 Deferred); it does not close the strength gap
this finding documents, only a possible mitigation worth live-testing
before committing to specific wording.
<!-- SECTION:NOTES:END -->
