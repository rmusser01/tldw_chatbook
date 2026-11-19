---
id: TASK-18935
title: 'Verify kitty keyboard protocol support and the Alt+M multiplexer docs quirk'
status: Done
assignee: []
created_date: '2026-08-19 09:55'
updated_date: '2026-10-11 00:00'
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
- [x] #1 A tested matrix of terminal/multiplexer combinations with observed Alt+M (and other Alt-chord) behavior is recorded in this task or the docs — matrix recorded in the Implementation Notes below: parser-level probe (all three delivery forms, Textual 8.2.8) + live tmux 3.6a probe (default config and extended-keys on) + driver source inspection for the kitty-protocol negotiation claim; kitty/Ghostty/WezTerm GUI apps and Terminal.app were NOT driven live (synthetic-key injection into GUI apps was out of scope) — their row rests on the kitty-protocol negotiation evidence, honestly labeled
- [x] #2 Either the docs quirk is corrected to match verified reality, or a follow-up task is filed with the characterized failure modes and reproduction steps — the console.md quirk is CORRECTED: the failure is not 'terminals that deliver Alt chords as separate Esc + letter' (those are merged within ~100 ms), it is specifically a >~100 ms gap between the Esc and the letter (slow SSH, or a multiplexer escape delay over that bound; tmux remedy: `set -s escape-time 10`)
- [x] #3 No binding or code changes land without a reproduced failure documenting them — no binding or code change landed; the only file changes are the docs quirk text and this task file
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A.
Reason: verification-and-documentation task; any binding change that might follow would be checked against ADR-031's keybinding conventions at that time.

1. Build the terminal matrix and test Alt-chord delivery per combination
2. Record results; correct the console.md quirk text or file the characterized follow-up
<!-- SECTION:PLAN:END -->

## Implementation Notes

- Approach: docs-verification task; no code or binding changes (AC#3 honored — no reproduced failure warranted any).
- Evidence, gathered 2026-10-11 at dev `9f8533686c`, Textual **8.2.8**, tmux **3.6a**, macOS host:

  **Parser-level probe** (drove `textual._xterm_parser.XTermParser` directly with raw byte sequences):
  | Delivery form | Bytes | Textual 8.2.8 yields |
  |---|---|---|
  | Kitty-protocol disambiguated | `CSI 109;3u` | `alt+m` (one event) |
  | Legacy, one read | `ESC M` together | `alt+m` (merged by the ESC-ambiguity resolver) |
  | Split, gap < 100 ms (`ESCAPE_DELAY = 0.1`) | `ESC` … `M` | `alt+m` (still merged) |
  | Split, gap > 100 ms | `ESC` … `M` | `escape` then `m` — **the only failure form** |

  **Driver source**: `textual/drivers/linux_driver.py` enables the kitty keyboard protocol at startup unconditionally (`CSI > 5u`-family flags: disambiguate + report-all + associated-text; env kill switch `TEXTUAL_DISABLE_KITTY_KEY`) and disables it on exit — so kitty/Ghostty/WezTerm-class terminals deliver Alt+M as a single key event, and the old quirk's premise does not apply in them.

  **Live tmux probe** (raw tty reader inside a real tmux 3.6a session; `tmux send-keys M-M`): the pane receives `b'\x1bM'` — Esc and M **in one write** — under BOTH the default config and `extended-keys on`. Combined with the parser probe, Alt+M works in tmux 3.6a in both modes. (Honest limitation: `send-keys` injects post-outer-terminal, so this verifies tmux's pane-side encoding; the outer-terminal leg for real keypresses is covered by the parser-level split analysis.)

  **Not driven live**: kitty.app / Ghostty.app / iTerm.app GUI terminals (installed on this host but synthetic-key injection into GUI apps was out of scope), Terminal.app, plain ssh. Their rows rest on the kitty-protocol negotiation evidence and the parser analysis, and are labeled as such above.

- Docs change: the console.md "Alt+M does nothing" quirk now states the real failure condition (a >~100 ms Esc/letter gap — slow SSH links or a multiplexer escape delay over that bound), the merge behavior that makes most "separate Esc + letter" setups work anyway (including tmux), and the tmux `escape-time` remedy — while keeping the `/model` and Ctrl+P fallbacks.
- Modified files: `Docs/User_Guide/console.md`, this task file.
