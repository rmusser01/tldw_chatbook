---
id: TASK-18935
title: 'Verify kitty keyboard protocol support and the Alt+M multiplexer docs quirk'
status: To Do
assignee: []
created_date: '2026-08-19 09:55'
updated_date: '2026-08-19 09:55'
labels:
  - console
  - docs
  - terminal-compat
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from the 2026-08-19 hermes-release review: hermes shipped Kitty keyboard-protocol support to make modifier chords reliable, and chatbook's docs carry a quirk entry — "Alt+M does nothing in terminals/multiplexers that deliver Alt chords as separate Esc + letter" (Docs/User_Guide/console.md, Quirks). Modern Textual (the repo is on Textual 8.x per ADR-022) negotiates the kitty protocol automatically in supporting terminals, so the quirk may already be moot where it matters. Verify observed behavior across a terminal matrix — kitty, Ghostty, WezTerm, Terminal.app, tmux (with and without extended-keys), plain ssh — then either correct the docs to reflect reality or file a characterized follow-up (e.g. an Esc-prefix fallback binding) with evidence. No speculative binding changes without evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A tested matrix of terminal/multiplexer combinations with observed Alt+M (and other Alt-chord) behavior is recorded in this task or the docs
- [ ] #2 Either the docs quirk is corrected to match verified reality, or a follow-up task is filed with the characterized failure modes and reproduction steps
- [ ] #3 No binding or code changes land without a reproduced failure documenting them
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A.
Reason: verification-and-documentation task; any binding change that might follow would be checked against ADR-031's keybinding conventions at that time.

1. Build the terminal matrix and test Alt-chord delivery per combination
2. Record results; correct the console.md quirk text or file the characterized follow-up
<!-- SECTION:PLAN:END -->
