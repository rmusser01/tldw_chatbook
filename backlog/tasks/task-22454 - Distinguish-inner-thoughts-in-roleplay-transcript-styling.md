---
id: TASK-22454
title: Distinguish inner thoughts in roleplay transcript styling
status: Done
assignee:
  - '@codex'
created_date: '2026-08-26 14:31'
labels:
  - roleplay
  - console
  - ux
  - accessibility
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give single-quoted inner thoughts a distinct, accessible presentation in immersive character chats while preserving existing narration, speech, action, emphasis, Markdown, streaming, and export behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Closed straight- and curly-single-quoted inner thoughts render with a theme-aware italic treatment that remains distinct from narration, speech, action, and strong emphasis in supported dark and light themes.
- [x] #2 Thought delimiters remain visible, word-internal apostrophes such as contractions remain intact, and unclosed thought markers stay literal during streaming.
- [x] #3 Full Markdown structure, raw source, exports, links, inline code, append-only streaming, non-roleplay rows, and selected or failed state precedence remain unchanged.
- [x] #4 Focused parser, mounted compositor, streaming, CSS synchronization, lint, and diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing parser and Markdown projection tests for closed thoughts, contractions, curly quotes, protected code and links, and unclosed streaming input.
2. Extend the Console roleplay span projection with one thought semantic component while preserving source text and the existing compatibility fallback.
3. Add accessible dark/light thought tokens and Immersive-RP-only styling, including selected and failed state precedence, then rebuild the generated CSS bundle.
4. Run focused parser, compositor, streaming, transcript, CSS synchronization, lint, and diff verification; record the results and close the task only if every acceptance criterion is met.

ADR required: no new ADR
ADR path: backlog/decisions/046-roleplay-chat-display-identity-and-template-provenance.md
Reason: ADR-046 already governs the shared Console presentation resolver, theme-aware Immersive RP prose accents, non-color identity carriers, and operational-state precedence; this change adds one presentation semantic without changing storage or module ownership.
<!-- SECTION:PLAN:END -->

## Implementation Notes

- Added a boundary-aware straight/curly single-quote thought projection for both the compatibility renderer and full Textual Markdown renderer. Quotes remain visible, word-internal apostrophes remain literal, and code/link spans retain priority.
- Added one italic thought semantic component with accessible dark/light teal tokens, plus selected and failed state overrides. Rebuilt the generated application stylesheet; no storage, export, or provider contract changed.
- Updated the Console user guide and focused coverage for contractions, incomplete pairs, protected Markdown, raw source/export preservation, append-only streaming, dark/light compositor contrast, and operational-state precedence. Repaired three test-only header queries already recorded as stale in the repository UI baseline after grouped assistant turns moved headers outside body rows.
- Qodo review follow-up aligned full Markdown projection with compatibility-parser precedence: a complete outer speech quote now suppresses nested straight or curly thought spans. Added regression coverage and the required Google-style test docstrings.
- Verification: the clean `origin/dev` baseline sweep was stopped at 3% after reproducing four unrelated Agent-runtime failures (`2,351 passed`, `8 skipped`, `4 failed`); after the review follow-up, all `37` focused transcript tests passed. Ruff passed for the changed Python files, the generated CSS exactly matched all 51 declared source modules, and `git diff --check` passed.
- ADR: reused `backlog/decisions/046-roleplay-chat-display-identity-and-template-provenance.md`; no new ADR was required. No new lessons entry was warranted.
