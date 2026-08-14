---
id: TASK-15862
title: >-
  Wake-turn UI staleness: transcript, run-state chip, and tab glyph can freeze
  until the session is next viewed
status: Done
assignee: []
created_date: '2026-08-13 21:43'
updated_date: '2026-08-14 01:23'
labels:
  - fleet
  - console
  - ui
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR 3a-2 Task 7's live pass found the wake DELIVERY layer correct and durable in every scenario, but the UI around a wake turn can go stale indefinitely. Three observed shapes, one likely root: (1) a wake turn completing while the user views another Console session leaves the woken session's tab glyph stuck at RUNNING (●) for minutes instead of flipping to the unvisited-outcome glyph — it clears only when the session is viewed; (2) a mount-claim wake delivering into the VIEWED session froze mid-delivery: the assistant reply row stayed empty while the full reply sat in the DB, the status row read 'Run: Agent running.' and the composer read 'Send blocked — finish provider setup to continue' (misleading — provider setup was fine) for 4+ minutes, healing instantly on a session switch; (3) the same freeze recurred on the post-restart poked delivery. Likely the transcript poll / repaint pipeline is armed by user-driven send paths and never armed (or re-armed at the terminal edge) for a wake delivery task — the same self-stopping-poll family as task-15664.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A wake turn's streamed reply and its terminal state paint in the viewed session without requiring a session switch
- [x] #2 A wake turn completing in a non-viewed session flips that session's tab glyph off RUNNING at the terminal edge
- [x] #3 The composer's blocked-state copy during a wake turn names the actual reason (busy with a wake turn), not provider setup
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Diagnose (done): transcript poll armed ONLY in _submit_console_native_draft; wake path (coordinator._deliver -> controller.submit_draft) never arms it -> no repaint during/after a wake turn (viewed freeze AC#1, stuck tab-glyph AC#2). Composer copy: queue presentation 'Preparing...' tooltip (occupies_slot, never accepted_live_turn for a chainless wake) is passed as setup_blocked_reason and build_console_disabled_reason's fallback mislabels ANY unrecognized copy as provider setup (AC#3).\n2. RED tests on a mounted ChatScreen: wake into unviewed session leaves glyph stuck; wake into viewed session never paints the reply; composer disabled reason claims provider setup mid-wake.\n3. Fix: coordinator delivery_started_hook (loop thread) wired by the screen to _start_console_transcript_sync_timer; expose delivering_conversation_id(); thread wake_turn_active into composer sync_action_state + build_console_disabled_reason wake branch.\n4. Mutation-test the new tests; run wake/tick/wiring + Chat/UI gates; live re-verify.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in two layers on fix/fleet-wake-ui-residue (commits ec86f85e0 + ba71f3068).

Mechanism (diagnosed, not assumed): the 0.2s transcript poll is armed only by the
user-send worker (_submit_console_native_draft); a wake turn enters through
ConsoleFleetWakeCoordinator._deliver -> controller.submit_draft and never armed it, so
nothing repainted the wake turn's stream, terminal tab glyph, or composer state. Both
observed shapes (unviewed stuck-tab AND the viewed 4+min freeze) share this one
mechanism. The composer's "finish provider setup" lie: the queue presentation's
not-yet-accepted tooltip (a chainless wake is never queue-accepted) rode the
setup_blocked_reason slot into build_console_disabled_reason's provider-setup fallback.

Fix: coordinator delivery_ui_hook (fired with _delivering already set) + a
delivering_conversation_id() accessor; the screen arms the poll from the hook THROUGH
THE MESSAGE PUMP (call_later) -- live diagnosis proved a timer created straight from the
coordinator's bare call_soon_threadsafe context (copied from the child's thread, no
active_app ContextVar) dies on its first tick (Textual Timer._tick reads active_app; a
task inherits its creation context). A poll stop-guard covers the scheduling gap;
wake_turn_active threads screen -> composer -> display state so the blocked copy/tooltip
name the wake. The poll still self-stops at the wake's terminal edge (15664 AC#2
asserted in-suite and observed live).

Tests: Tests/UI/test_console_fleet_wake_ui_freshness.py (4) -- each reproduced RED
against unmodified production on its own assertion, including the byte-exact live lie;
drains injected from a plain thread after the harness's app-context injection was caught
passing against the broken arming. Mutations: 9 run, 9 killed (M9 = call_later reverted
to the direct call, killed by 3 tests).

Live re-verification (scratch profile, real Anthropic claude-sonnet-5, frames+dbg in
.superpowers/sdd/2026-08-13-supervisor-fleet-pr3a2-autowake/residue-frames/): viewed
session painted the notice + fenced result + reply with zero interaction (9-10 poll
beats per delivery); unviewed session's tab tracked ● live and flipped to ✓ at the
terminal edge on a 1s frame timeline; AC#3's flag path ran every beat (dbg), copy pinned
byte-exact in-suite.
<!-- SECTION:NOTES:END -->
