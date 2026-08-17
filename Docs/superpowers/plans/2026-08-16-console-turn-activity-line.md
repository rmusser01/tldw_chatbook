# Console turn-activity line

*Branch `feat/console-turn-activity-line`, from dev @ `feea06193`, 2026-08-16.*

## The request

> "add visibility into agent actions in the console under/in the agent's turn
> box, so a user can see what is going on/being done, and it doesn't look like
> it's frozen to the user."

The owner chose the shape, and it is binding:

```
Assistant  ⚙ read_file · 4s

─ after it finishes ─

Tool  ⚙ read_file → def main():\n    …… (+3841 chars)
Assistant  ⚙ web_fetch · 11s
```

A live, ticking activity line **inside the in-flight assistant row**. No new
row types; completed `Tool` rows keep appearing below exactly as before; tool
markers stay append-only and untouched; tool **arguments** are deliberately
not rendered (they are unbounded at the seam — a `write_file` content arg can
be megabytes).

## What "frozen" actually was

The survey that preceded this work said the in-flight row sits on the
`Generating…` placeholder for the whole multi-round turn. **Measured, it is
worse than that.** A probe that ran a real `ConsoleAgentBridge` turn with the
tool held on an Event, then mounted the real transcript over the real store's
messages, read the row back as:

```
BEFORE_ROW_TEXT>>>Assistant<<<
BEFORE_STATUS>>> 'pending'
BEFORE_SNAPSHOT>>> running [('model', '```tool_call…'), ('tool_call', 'calculator', 'primary')]
```

The row is `status='pending'`, `content=''`. `_message_body`'s placeholder
branch is gated on `status == "streaming"`, and the store only reaches
`"streaming"` on the first streamed chunk (`append_stream_chunk`) — which a
tool-calling turn's assistant row never produces, because the fence gate
seals it from the first token. So the user watched the bare word `Assistant`
for the entire turn; not even `Generating…` applied.

The same probe confirmed the good half of the survey: the tool name is
**already at the seam**. `AgentLiveStep('tool_call', 'calculator', 'primary')`
is in `bridge.live_snapshot()` while the tool is held. This was wiring, not
plumbing — except for elapsed.

## The four states

Derived from the primary agent's most recent step
(`console_turn_activity_text`):

| situation | line | source |
|---|---|---|
| a tool is running | `⚙ read_file · 4s` | last primary step is `STEP_TOOL_CALL`; `started_at` is the moment the bridge saw it, one statement before `deps.invoke_tool` |
| between tools / after a result | `Thinking… · 6s` | last primary step is anything else; there is no "model call started" step (`STEP_MODEL` is emitted *after* the round, carrying its text), so this is derived from the negative |
| running, no primary step yet | `Generating…` | pre-first-token. Today's copy, unchanged — and now it actually shows, because the activity path covers `pending` too |
| turn ended | *(nothing)* | snapshot status is not `running`, or the row is no longer in flight/empty |

No elapsed on `Generating…`: there is no per-step base to time from, and
`_format_fleet_elapsed`'s own rule is that claiming a duration without a
usable base is a lie. Same reason `_fleet_row_from_summary` omits it.

## Where each piece lives, and why

**Elapsed → the bridge (`Chat/console_agent_bridge.py`).** `AgentLiveStep`
gains `started_at: float | None`, stamped with `time.monotonic()` in
`on_step`. It cannot come from the runtime: `AgentStep.created_at` is a `str`
that stays empty for the whole life of a live run (`AgentService` stamps the
batch once, at end-of-run persist), and `Agents/agent_runtime.py` /
`agent_models.py` are pure-logic modules whose contract is that they read no
clock. `on_step` is the impure seam and is called *synchronously* inside the
loop, so for a `STEP_TOOL_CALL` the reading is the tool's real start rather
than a poll-quantised approximation. Resume-derived steps keep `None` —
`AgentRunsDB` carries nothing a duration could honestly come from.

**Derivation → `UI/Console_Modules/agent.py`.** `console_turn_activity_text`
is a pure function sitting next to `_format_fleet_elapsed`, which it reuses.
`ConsoleAgentController.console_turn_activity()` is the one impure read.

**Pixels → `Widgets/Console/console_transcript.py`.** The line rides the row's
*existing* header (markdown rows, the default) or body Content (plain rows).
**No new widget is mounted**, so the "a bare `Static` defaults to `1fr` and
pushes its siblings off screen" hazard cannot apply; a geometry test asserts
it anyway.

**Screen → 8 lines, 0 new methods.** `chat_screen.py` is under a one-way size
ratchet and is already ~2,600 lines over it on dev (measured 20,359/672 vs a
budget of 17,727/593 — that test is red at the merge base, not because of
this work). The transcript sync reads the controller's line, hands it to the
transcript, and folds the return into its refresh key.

## The three hazards this programme paid to learn

1. **No idle repaint (task-15664 AC#2).** No new timer, and none is needed:
   during an agent turn the run is in `CONSOLE_ACTIVE_RUN_STATUSES`, which is
   exactly the condition `_start_console_transcript_sync_timer` keeps its
   0.2 s tick alive for. Confirmed by execution, not by reading. Two further
   gates make "nothing live → nothing repaints" true regardless of what the
   bridge last published: the controller refuses to derive a line unless the
   **viewed** session's run is active, and `apply_turn_activity` returns the
   **effective** value (`""` unless a row is genuinely in flight), which is
   what joins the refresh key.
2. **A torn-down screen renders nothing.** The new call sits inside
   `_sync_native_console_transcript`, which the existing
   `_console_screen_is_torn_down` guards in `_sync_native_console_chat_ui`
   already cover. No new sync path was created.
3. **Rendered geometry, not DOM presence.** No widget added; the geometry test
   asserts every row and every child `Static` satisfies
   `region.x + region.width <= screen width`. Markers render markup-off, so
   the line is raw and unescaped — pinned with `fetch [docs]`.

## Sub-agent isolation

`console_turn_activity_text` selects the last step whose `agent_kind` is
`AGENT_KIND_PRIMARY`, skipping the rest. This is not defensive: `on_step`
routes a **sub-agent** step whose run id is empty into the **primary** run's
own live feed (its documented "no run attributed" fallback), so the primary
snapshot's last entry really can belong to a child. A child's work belongs to
the Agent rail's fleet rows, never to the primary assistant row.

## The quiet-tool decision

`_QUIET_STEP_TOOLS` (`find_tools` / `load_tools`) are suppressed from
transcript markers. **They are shown in the activity line.** The quiet rule
exists to keep catalog plumbing out of the *permanent, append-only* markers,
where a discovery round would be lasting clutter. The activity line is
ephemeral and exists precisely so that no working moment looks frozen —
suppressing these would reinstate a silent gap for the whole discovery round,
which is the defect, not the fix. Pinned by test.

## Live-only, and why that does not break the live/resumed parity rule

An activity line has no resumed counterpart and must never grow one: it
reports what an agent is doing *right now*, and a finished turn's transcript
already carries the `Tool` markers that say what it did. This is documented on
the function the same way `format_todo_marker` documents it — the parity rule
that matters (`format_agent_step_marker` must render byte-identical text live
and on resume) is untouched, because that formatter was not modified.

## Mutation testing

Thirteen mutations, ten killed, **three survived — and each survivor was a
real weakness.** A fourth finding came out of *how* one of the killed
mutants died. All four are fixed and pinned.

| # | mutation | outcome |
|---|---|---|
| 1 | `_message_body`'s activity branch disabled | killed (2) — **and diagnosed finding A** |
| 2 | `_assistant_markdown_header`'s activity suffix disabled | killed (7) |
| 3 | `live_activity` dropped from `_message_signature_token` | killed (2) |
| 4 | primary-only step filter widened to include children | killed (2) |
| 5 | every step kind treated as a tool call | killed (2) |
| 6 | `apply_turn_activity`'s eligibility check dropped | killed (2) |
| 7 | in-flight statuses narrowed back to `streaming` only | killed (9) |
| 8 | controller's viewed-run status gate disabled | killed (3) |
| 9 | `started_at` never stamped (`None`) | killed (1) |
| 10 | `_with_turn_activity` stamps every message | **SURVIVED** |
| 11 | `_is_generating_placeholder_body` blind to the line | **SURVIVED** |
| 12 | `apply_turn_activity` skips the empty case | **SURVIVED** |
| 13 | the screen's `apply_turn_activity` call removed | killed (3) |

**A — the signature coupling (from how #1 died).** #1 killed two tests, but
the *failure* was the interesting part: the production-path test's FIRST
paint still showed `⚙ calculator · <1s`, and only the SECOND tick failed to
advance. A markdown row carries the line in its **header**, which
`_message_row_signature` never renders — it renders the *plain* row. So the
elapsed only ticked as a side effect of the plain renderer embedding the same
text; the two renderers were silently coupled, and disabling the plain branch
froze a markdown row's elapsed at its first painted value. `live_activity` is
now named in that signature outright, pinned by a test that paints two ticks
differing in nothing but the elapsed and checks the row updates in place.

**B — blast radius (#10).** Stamping every message rather than the one
in-flight row is invisible to every display assertion (a row with content
never renders the line, and only assistant rows can) — yet it puts
`live_activity` into every row's signature: a whole-transcript re-derive and
re-sync once a second for the entire turn. Pinned by measuring row signatures
and per-message signature compute counts instead of pixels; the target id is
now hoisted out of the row loop.

**C — the doubled/undimmed line (#11).** A **streaming** empty row is
reachable on a fence-gated tool turn (`reset_stream_buffer` discards leaked
prose and leaves the row streaming-and-empty). There, a placeholder predicate
blind to the activity line rendered it as ordinary assistant content with a
`Streaming…` status line stacked under it. Pinned on both the absent status
line and the dim span.

**D — the stale line (#12).** Skipping the empty case left the last line
sitting on a row that stays in flight after its run dies without a terminal
publish — frozen at its last elapsed, which is the exact defect this feature
exists to remove, in a new costume.

## Known limits

- A `pending` assistant row with **no agent run at all** (the direct-provider
  path before its first token) still renders bare. Out of scope here: the
  fix would be to widen the `Generating…` placeholder's own status gate,
  which is a separate behaviour change to a separate path.
- `AgentLiveStep.text` for a tool-call step *is* the tool name, by
  construction (`agent_runtime` adds every `STEP_TOOL_CALL` with `tool_name=`
  and no `summary`/`result`, and `_summarize`'s precedence lands on it). If a
  future change gives those steps a summary, the line's label follows it.
  Documented at the derivation.
