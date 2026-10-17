---
id: TASK-35100
title: Console tokens-per-second readout
status: Done
created_date: 2026-10-16 12:00
labels:
- console
- chat
assignee:
- '@zcode'
updated_date: 2026-10-16 13:30
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Console users have no way to see how fast an endpoint is generating tokens. Streaming shows a "Generating..." placeholder and a cost chip that is intentionally static during a stream, so slow endpoints are invisible until the response finishes. A tokens-per-second readout makes generation speed visible while streaming and settles to an overall average for the finished response. Design approved 2026-10-16 (brainstorming session): live rate estimated from streamed text each 0.2s poll tick, overall rate provider-accurate from attached usage divided by time from first token. ADR not required per policy (small UI feature; no storage/boundary/interface decisions).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 While a Console response is streaming, the live smoothed tokens-per-second rate rides the EXISTING run chip's copy in compact form ("Run: streaming · 42.3 tok/s") -- the status strip gains no slot while streaming (redesigned 2026-10-16 after the first placement pushed stable chips off the row)
- [x] #2 When the response settles and the run goes terminal, a compact badge ("30.9 tok/s avg") appears in the run chip's vacated slot, using the provider-reported output token count when usage is available and the local estimate otherwise; token total and elapsed window live in the tooltip, and the badge never shows while the run chip does (slot discipline)
- [x] #3 The overall rate excludes time-to-first-token (the tracker's clock starts at the first non-empty-content observation, not request start)
- [x] #4 Agent tool-loop streams show the same readout (they land in the store via the same append_stream_chunk path the tracker observes; covered by construction and the shared screen test harness)
- [x] #5 A `[console]` config key (show_tokens_per_second, default on) hides the readout without code changes
- [x] #6 Unit tests cover the rate tracker math (live EMA, final computation, missing usage, missing first token, cancel/partial) and tests assert the live rate rides the run chip during streaming and the settled badge appears at completion
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:IMPLEMENTATION_PLAN:BEGIN -->
1. Write failing unit tests for Chat/console_token_rate.py (TokenRateTracker: live EMA rate, final stats from usage/estimate, edge cases)
2. Implement TokenRateTracker (pure module, no UI imports) until green
3. Wire into chat_screen 0.2s poll tick: feed accumulated message text, push reading to a status-strip chip; settle on usage arrival
4. Add [console] show_tokens_per_second config key (default true)
5. Run targeted tests; update task with implementation notes
<!-- SECTION:IMPLEMENTATION_PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
### Approach

A poll-fed, pure tracker plus a display-only chip, deliberately narrow in each layer:

- `Chat/console_token_rate.py` (new): `TokenRateTracker` -- injected token counter, injected clock, half-life EMA (default 1.0s) over per-tick token deltas, plus `format_rate_chip_copy` (the chip's `(visible, copy)` contract, pinned by tests). No Textual, no store imports. The clock starts at the first non-empty-content observation (TTFT excluded); a terminal observation for a message the tracker never saw streaming is ignored rather than manufacturing an elapsed window; late provider usage upgrades a settled estimate without moving the latched elapsed window; non-monotonic ticks hold; token regressions clamp at zero. No special stall path: the tracker is fed every tick, so a stalled endpoint's rate decays to ~0 through the same EMA.
- `Widgets/Console/console_status_chips.py`: `ConsoleRateChip` + `sync_rate_chip(visible, copy)`, an exact mirror of the run-chip contract (equality-guarded, pure `_rate_chip_render`, hidden by default), composed right after the run chip in the transient cluster.
- `UI/Screens/chat_screen.py`: `_console_rate_tracker` state (count_fn = `Utils/token_counter.estimate_tokens`); `_sync_console_rate_chip(messages)` called from `_sync_native_console_transcript` right after `apply_turn_activity` -- the same 0.2s tick that drives the transcript and run chip. It observes the newest assistant row (role/status/content/`usage.output`) and pushes the chip; config-off resets the tracker and hides.
- `config.py`: `[console] show_tokens_per_second = true`.

### Design deviation (approved design revised before implementation)

The brainstormed design placed the live rate on the turn-activity line. Reading the implementation surface showed the activity line renders ONLY on an in-flight row with no content (`ConsoleTranscript._turn_activity_target_id`: "a row that already has text ... must never be overwritten by a status line") -- it disappears exactly when tokens start flowing, so it cannot host a during-generation rate. The readout moved to the status strip; AC#1 was updated to match before implementation.

### Redesign (2026-10-16, after live demo review: the chip pushed stable chips off the row)

The first implementation added a dedicated rate chip beside the run chip. At 126 columns that pushed Library/Provider/Model toward the strip's scroll edge (streaming carried run 24 + rate 19 = +43 transient columns). Redesigned to occupy NO new slot:

- Live rate rides the run chip's copy in compact form: "Run: streaming · 42.3 tok/s". A live rate exists only mid-stream, so substituting the compact status word for the "Streaming response." sentence is safe and keeps the merged copy inside the chip's 34-cell cap -- the full sentence plus rate ellipsized the rate into "· 30…" unreadability (found in the re-record, fixed before close).
- The settled badge is the same ConsoleRateChip, but shows ONLY the compact "30.9 tok/s avg" (never the token total -- that duplicated the cost chip's domain), with tokens + elapsed in the tooltip, and ONLY once the run has gone terminal and vacated the slot (`_sync_console_rate_chip` gates on `_console_active_run_copy()` emptiness). The two transient readouts are never on the row together.
- Net row cost: idle unchanged; streaming ~+5 cols inside the existing run chip; settled ~+15 cols until the next turn resets the tracker. Screen stash (`_console_rate_reading`) feeds the mode-bar merge one 0.2s tick behind the transcript sync -- invisible on a ~1s-smoothed label.

### Counting semantics

Live counts are estimates (`estimate_tokens`, tiktoken tier with conservative fallback) -- the same "estimate" vocabulary the live usage label and cost tracker already use. The settled figure uses the provider's `ProviderUsage.output` (attached by the controller's `_attach_stream_usage`) divided by the content window, falling back to the final estimate for providers that report nothing. Agent tool-loop streams are covered by construction: every path lands streamed text in the store via `append_stream_chunk`, and the tracker reads store rows, not the transport.

### Tests

- `Tests/Chat/test_console_token_rate.py` (24): live EMA convergence/burst smoothing, settle from provider vs estimate, late-usage upgrade, zero-elapsed, terminal-without-baseline hide, message-id lifecycle, clock edges (non-monotonic, stall decay + recovery, UI-freeze gap, regression clamp), chip copy contract.
- `Tests/UI/test_console_rate_chip.py` (4): default hidden, sync shows copy+tooltip, hides on clear, placement between run chip and library chip. `bootstrap_profile` marker like the cost-chip suite (strip mount imports real config getters).
- `Tests/UI/test_console_rate_chip_screen.py` (2): real mounted ChatScreen + stub gateway streaming two chunks around a mid-stream hold -- live "tok/s" visible during the hold, settled "... tok/s · 200 tok" after usage attaches; config-off hides through a full send.

All green: 30 new + `Tests/test_config_console_defaults.py` (63) = 93 passed in one run, screen tests green separately.

### Verification caveats

- This machine shows pre-existing environmental failures in neighboring UI suites (`Tests/UI/test_console_status_chips.py`, parts of `test_console_cost_chip_screen.py` / `test_console_status_row_collapse.py`) with `RecoveryRequired("raw_source_selection_changed")` at setup. Verified pre-existing: the chips test failed identically before any production file was touched, the stylesheet-contract failure reads only css/ inputs bit-identical to HEAD, and a copy-aside A/B of the three production files ran the full cost-chip suite at HEAD (3 failures) and with the feature (the SAME 3, plus a flaky 2-failure rerun) -- the failure set is environmental, not the feature's.
- Not live-verified in a running TUI against a real endpoint (no endpoint available in this session); the mounted-screen harness drives the real poll path end to end, which is the evidence standard the cost chip shipped with. A recorded demo of the real ChatScreen (sandboxed harness, stub streaming endpoint) lives at /tmp/tokps-demo/console-tokps-demo.{gif,mp4}, captured by /tmp/record_tokps.py.

### Modified/added files

Added: `tldw_chatbook/Chat/console_token_rate.py`, `Tests/Chat/test_console_token_rate.py`, `Tests/UI/test_console_rate_chip.py`, `Tests/UI/test_console_rate_chip_screen.py`.
Modified: `tldw_chatbook/UI/Screens/chat_screen.py`, `tldw_chatbook/Widgets/Console/console_status_chips.py`, `tldw_chatbook/config.py`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
