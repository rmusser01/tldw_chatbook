---
id: TASK-339
title: Fix composer keystroke reordering during the session-switch settle window
status: Done
assignee:
  - '@claude'
created_date: '2026-07-20 14:21'
updated_date: '2026-07-21 02:40'
labels:
  - console
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reproduced twice. Session 1: after selecting a conversation in the Ctrl+K switcher, clicking the composer and typing 'Follow-up: which vector store...' (15ms/char) produced 'Follow-ch vector store should I pick for a 10k doc corpus? up: whi' — the caret was silently relocated mid-typing — and that mangled string was then sent and persisted as the user message (visible in transcript and after app restart). Controlled retrial: typing 'abcdefghij...' at 15ms/char starting ~0.5s after switcher Enter yielded 'abcdefghijklmnrstuvwxyz0123456789▌opq' (chars opq torn out of sequence). At 100ms/char starting later the text stayed intact, so the hazard window is the post-switch settle period (~0.5-2s here; machine load average was 13-15, which widens it, but the caret-steal is app behavior).

**Repro:** Ctrl+K, type a query, Enter to switch. Within ~0.5s click the composer and type a fast burst (15ms/char, i.e. fast typist / key-repeat speed). Composer shows characters out of order; Enter sends the mangled text.

**Verifier note:** Genuine input-integrity defect, nowhere in ledger or backlog. Mangled text is visible persisted in the transcript ('Follow-ch vector store should I pick for a 10k doc corpus?up: whi' in j2-12-reply-later.png), reproduced twice with a controlled retrial. Post-switch activation runs multiple deferred syncs plus draft restore/forced composer focus (_focus_console_composer_if_needed(force=True) after _sync_native_console_chat_ui in chat_screen.py:3342-3344), which can relocate the caret under buffered typing. Load average widened the window but caret-steal is app behavior.

**Source:** Console UX expert review 2026-07-20 (finding j2-post-switch-typing-reordered; P1, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J2 returning power user journey. Evidence: `j2-57-mangle-trialB.png`, `j2-10-just-sent.png`, `j2-12-reply-later.png`, `j2-24-resumed-long.png`, `j2-56-mangle-trialA.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The composer must never reorder buffered keystrokes
- [x] #2 Recompose/focus work after a switch should preserve caret position and pending input, or input should be blocked with a visible busy state until the switch settles
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Composer: user-edit serial (bumped by insert/delete entry points, not by programmatic load/clear/restore)
2. chat_screen: capture (visible session, draft text, edit serial) at switch initiation — _activate_native_console_session before switch_session, _resume_console_workspace_conversation before the conversation-tree await
3. _sync_console_session_draft: when the deferred swap runs after user edits, save the SNAPSHOT text to the old session and carry the typed suffix forward into the new session (append after its restored draft); fall back to today's semantics when no snapshot or non-append edits
4. TDD: deterministic settle-window simulation (block the inline sync, type between activation and swap), assert order + attribution; per-tab draft-restore tests stay green
<!-- SECTION:PLAN:END -->

## Implementation Notes

Root cause: `_sync_console_session_draft`'s deferred swap saved the LIVE
composer text to the old session and reloaded the new session's stored
draft whenever it finally ran — keystrokes typed in the settle window
(coalesced syncs, slow resume loads; the codebase already documented this
clobber class at the `/prompt`-insert call site) were misattributed to the
old session and wiped/resequenced in the composer.

Fix: switch initiation now snapshots (visible session, draft text,
composer edit serial) — captured in `_activate_native_console_session`,
`_resume_console_workspace_conversation` (before the conversation-tree
await), the new-tab path, and the workspace-change switch/create path. The
swap saves the SNAPSHOT text to the old session and carries the
typed-since suffix forward into the new session, in order; non-append
edits fall back to the previous semantics. `ConsoleComposerBar` gains a
user-edit serial (bumped only by typing/delete entry points, never by
programmatic load/clear/restore).

Verified: 2 new UI tests in `Tests/UI/test_console_switch_draft_integrity.py`
(settle window simulated deterministically by holding the sync-coalescing
guard; wipe reproduced RED first), per-tab draft-restore tests stay green
(238 passed across chat-flow/composer suites). Files:
`UI/Screens/chat_screen.py`, `Widgets/Console/console_composer_bar.py`.
