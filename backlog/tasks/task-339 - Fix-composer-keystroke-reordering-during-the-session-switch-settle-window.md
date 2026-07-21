---
id: TASK-339
title: Fix composer keystroke reordering during the session-switch settle window
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux]
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
- [ ] #1 The composer must never reorder buffered keystrokes
- [ ] #2 Recompose/focus work after a switch should preserve caret position and pending input, or input should be blocked with a visible busy state until the switch settles
<!-- AC:END -->
