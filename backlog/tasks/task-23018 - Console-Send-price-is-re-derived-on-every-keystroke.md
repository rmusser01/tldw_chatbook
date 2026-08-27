---
id: TASK-23018
title: Console Send price is re-derived on every keystroke
status: Done
assignee:
  - '@claude'
created_date: '2026-08-27 00:40'
updated_date: '2026-08-27 01:20'
labels:
  - performance
  - console
  - input-latency
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #2114 (`1d59f96def`, "show estimated next-send price on Send button", merged 2026-08-26) put the whole next-request derivation on the printable-keystroke path. Once the Console draft is non-empty, every subsequent key re-projects the entire session's provider history and re-counts its tokens purely to render a hover tooltip nobody is looking at, so per-key input latency grows with conversation length. The estimate itself is worth keeping — a user wants to know what the next send costs — but it must be produced when it can actually be seen, not once per character.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Typing a printable character into a non-empty Console draft performs zero whole-session provider-history projections, whatever the conversation length
- [x] #2 Per-keystroke handler cost no longer scales with the number of messages in the session
- [x] #3 The Send button still shows its "| $" price affordance in exactly the states it did before, and its enabled/disabled state and label are unchanged
- [x] #4 Hovering Send shows a price derived from the live draft and the live session — never a stale number from an earlier draft
- [x] #5 A pricing failure still degrades to "Next request: cost unavailable" and never blocks Send
- [x] #6 The token memo can actually be served from cache on the surviving path, and a cache hit does not pay an O(session) rebuild
- [x] #7 Every new test fails against a deliberately broken implementation
<!-- AC:END -->

## Implementation Plan

1. Re-verify the filing first-hand on the base (`40ba8fe74d`): time `ChatScreen.on_key` from inside the handler, interleaved arms in both orders, plus an A/A control; count the mechanism (session snapshots, token-cache hits/misses) rather than trusting wall clock alone.
2. Split `ConsoleSendPriceController` into a cheap availability half and the expensive presentation half, sharing one decision helper so the two can never disagree.
3. Make the composer ask only the cheap question per keystroke, and derive the rendered tooltip on the hover seam.
4. Fix the memo's hit path; measure whether relocating the memo itself is worth it, and decline it with numbers if it is not.
5. Mutation-test every new gate; walk the unmount and quit paths.

## Implementation Notes

### The finding, re-verified on the base

Reproduced, and worse than filed. Measured inside `ChatScreen.on_key` (never through
`pilot.press()+pause()` wall time), 30 keys per arm, arms interleaved in both orders:

| session | pricer ON (median) | pricer OFF (median) | keys over 3 ms |
|---|---|---|---|
| empty | 0.870 ms | 0.370 ms | 0 / 90 |
| 400 messages | **5.848 ms** | 0.368 ms | **90 / 90** |

A/A control at 400 messages (both arms pricer-ON, same loop shape): 5.735 vs 5.677 ms
median — a ~0.06 ms noise floor on the median, so the 5.85 → 0.37 separation is ~94% of
the handler and two orders of magnitude outside the control.

Mechanism confirmed: **1,200 `dataclasses.replace` message copies per keystroke inside
`on_key`** at 400 messages (three whole-session snapshot passes), and another 1,200
outside it from the screen's own `_sync_native_console_chat_ui` resync — **2,400 copies
per key in total**, both reaching `sync_action_state`. `dataclasses.replace` was 43% of
the profiled keystroke. The token memo measured **0 hits / 30 misses** over 30 keys,
exactly as filed: `token_estimate_signature` ends with the live draft row.

Correction to the brief, from measuring the derivation's internals (400 messages, n=20,
un-profiled): `presentation_for_draft` is 4.28 ms median, of which the history
projection is 2.85 ms (67%), the row/signature rebuilds are 1.30 ms (30%) and the token
counter is **0.138 ms (3%)**. The memo therefore guards the cheapest 3% of the call
while its own draft-inclusive signature is built over the whole session — see the memo
note below.

### What changed

**`UI/Console_Modules/send_price.py`** — the availability decision is split out of the
derivation. `_resolve_context()` settles every "is there a tooltip at all?" branch
(draft validation, store/session presence, pending attachments) without projecting the
transcript or counting a token, and returns either a `_DraftPriceContext` or the final
answer. `availability_for_draft()` and `presentation_for_draft()` both route through it,
so the cheap answer is `presentation_for_draft(...) is not None` by construction. One
deliberate normalisation: a `KeyError` from the history projection now degrades to the
"cost unavailable" copy instead of withdrawing a tooltip the `| $` label has already
promised (previously reachable only when `pending_attachments` succeeded and the
projection did not — i.e. never, synchronously).

**`Widgets/Console/console_composer_bar.py`** — `sync_action_state` now asks only the
cheap availability provider; the rendered tooltip is derived in
`_refresh_send_price_for_pointer()`, called from `on_enter`/`on_leave` and from
`sync_action_state` when the pointer is already on Send. The Send label, width,
enabled/disabled state and every blocked/idle tooltip branch are untouched.

**`UI/Screens/chat_screen.py`** — wires the new `send_price_available_provider` seam.

### Why hover, not a debounce

A debounce would still have to answer "is the number on screen the current one?" with a
timer, and timed/debounced work has broken quit in this repo three times. Hover needs no
timer at all: Textual posts `Enter` the moment the pointer arrives and shows the tooltip
`App.TOOLTIP_DELAY` (0.5 s) later, so there is ~500 ms of headroom for a 4 ms
derivation, and the tooltip is a mouse-only surface — hover is *exactly* the set of
moments the value can be seen. Freshness is structural rather than argued: the value is
derived at display time from the live draft and live session, and while the pointer is
parked on Send every `sync_action_state` re-derives (Textual's `tooltip` setter
re-reads a displayed tooltip), so a draft edit under a stationary pointer repaints the
real number.

Two framework details are load-bearing and are pinned by tests: the pointer test reads
`App.mouse_over` rather than `Button.mouse_hover`, because `Widget.watch_disabled`
clears `mouse_hover` when Send goes disabled under the pointer; and the handlers read
the pointer's *current* position rather than the triggering event's node, because that
same `watch_disabled` queues a synthetic `Leave` that can be delivered after Send is
sendable again.

### The memo

Fixed the part that measurement supports: the O(session) counter-row list is now built
inside the miss callback, so a cache hit no longer pays for rows it never reads — and
because the derivation is now on demand, the memo can be served at all (repeat hover
with an unchanged draft: 1 counter call, not 2; measured 30 misses → 1 miss over a
30-key + hover sequence).

**Declined, with numbers:** relocating the memo to a draft-free signature. The token
count is 0.138 ms of a 4.28 ms derivation (3%) and the estimator already memoises
per-string (`_ESTIMATE_CACHE`, TASK-18602), while splitting it would mean computing the
history and draft halves separately and reimplementing `count_tokens_messages`'s
chat-format framing base outside `token_counter.py` — a second definition of the pricing
arithmetic to keep in sync, for 3% of a call that now happens on hover. The expensive
half is the history projection, which no memo can skip because building any signature
requires it, and whose only cheap staleness key (`ConsoleChatStore.get_payload_revision`)
documents itself as best-effort ("a missed bump means a stale chip … not a wrong send") —
acceptable for a chip, not for a money number shown on demand.

### Result

Same measurement method, same machine, interleaved:

| session | before (median) | after (median) | pricer-OFF baseline |
|---|---|---|---|
| empty | 0.870 ms | 0.377 ms | 0.370 ms |
| 400 messages | 5.848 ms | 0.390 ms | 0.355 ms |
| 2,000 messages | (scales) | 0.379 ms | 0.358 ms |

Keys over 3 ms: 90/90 → 0/90. Whole-session projections per keystroke: 1 → 0. Token-cache
misses over 30 keys: 30 → 0 (1 on the hover). The residual cost of the availability probe
is ~0.03 ms and is flat in conversation length. The hover-derived tooltip is byte-identical
to the pre-fix string.

**Out of scope, unchanged, reported separately:** the same keystroke still costs one full
screen reflow — measured **132 `Widget.arrange` and 1.00 `Screen._refresh_layout` per key**
— from five over-determined layout armers in the same file. Bit-identical before and after
this change (3,960 arrange / 30 reflows over 30 keys in both arms), so it did not
contaminate the A/B. Pre-existing; not touched here.

### Mutation results (11 mutants, 11 killed)

| mutant | killed by |
|---|---|
| M1 keystroke path derives the price again | `..._never_derives_the_send_price_while_typing`, `..._price_affordance_tracks_send_queue_attachment_and_width` |
| M2 no re-derive under a parked pointer | `..._reprices_send_under_a_parked_pointer`, `..._refreshes_send_price_without_changing_cost_chip` |
| M3 `on_enter` no-op | 8 tests |
| M4 `on_leave` no-op | `..._drops_the_derived_price_when_the_pointer_leaves_the_composer` |
| M5 availability always True | `..._availability_agrees_with_presentation`, `..._availability_never_projects_the_session` |
| M6 availability delegates to the expensive path | `..._never_derives_the_send_price_while_typing` |
| M7 counter rows rebuilt eagerly | `..._cache_hit_does_not_rebuild_the_counter_rows` |
| M8 pointer test uses `Button.mouse_hover` | `..._price_affordance_tracks_send_queue_attachment_and_width` |
| M9 hover refresh ignores the blocked/idle gate | 4 tests |
| M10 projection `KeyError` withdraws the tooltip | `..._availability_agrees_with_presentation` |
| M11 price failure raises instead of degrading | `..._price_provider_failure_never_blocks_send` |

M4 and M9 survived the first round. Neither was an assertion-strength problem — both were
*reach*: leaving Send for a sibling posts a bubbling `Enter`, so `on_enter` alone covered
the leave case, and Textual does not deliver `Enter` to a **disabled** widget at all
(verified: `mouse_over` becomes the button, no handler runs), so hovering a blocked Send
is inert by construction. Two tests were added on gestures that do reach — leaving the
composer entirely, and a sibling hover while Send is idle/blocked — and both mutants then
died.

### Files

- `tldw_chatbook/UI/Console_Modules/send_price.py`
- `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- `tldw_chatbook/UI/Screens/chat_screen.py`
- `Tests/UI/test_console_send_price.py`
