---
id: TASK-32631
title: Convert Console Markdown replies into speakable text
status: Done
assignee:
  - '@codex'
created_date: '2026-09-15 15:45'
updated_date: '2026-09-15 19:31'
labels:
  - console
  - tts
dependencies: []
references:
  - >-
    backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Users hear Markdown delimiters mixed with assistant prose when Console replies are spoken. Produce readable speech from the same validated response while preserving content and announcing omitted code blocks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Manual Speak, automatic Speak replies, retries and global voice fallback synthesize readable content without Markdown delimiters.
- [x] #2 Headings, emphasis, lists, quotes, link labels, inline code and table rows retain their content and meaningful punctuation; each fenced or indented code block announces Code block omitted.
- [x] #3 Original message validation and length limits apply before conversion; empty spoken output finishes without provider work or stuck playback ownership.
- [x] #4 Targeted speech regressions, static checks and Console voice documentation cover the new behavior.
- [x] #5 HTML block and inline formatting preserve visible words and structural separators without speaking comments or script/style bodies; checklist items convey checked state while ordinary bracket text remains literal.
- [x] #6 The UI readiness census passes at the existing module limit after deferring Notes import parsing and selection-value helpers until first use.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md
Reason: Routine text preparation under the existing validated Console speech ownership boundary and first-use import deferral; no schema, dependency, provider contract or UI structure changes.

1. Add failing speech-admission regressions for formatting, omissions, original validation, empty output and fallback.
2. Add a pure Markdown-to-speech helper and wire it into trusted completed-response and fallback preparation.
3. Verify manual and automatic request paths, retry behavior, original snapshot authority and meaningful punctuation with targeted tests.
4. Update Console voice documentation, run scoped lint/format checks and self-review the final diff.
5. Address Qodo HTML and checklist findings with admission and adapter regressions, and complete test docstrings.
6. Defer selection and Notes import parsing helpers until feature use to restore the existing UI readiness limit; verify census and affected behavior.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a pure Markdown-to-speech projection, invoked lazily from trusted Console and global-override speech preparation. It retains prose, link/image labels, inline code and labelled table rows; announces each fenced/indented code block as Code block omitted; and verbalizes checklist state. HTML keeps visible words and structural pauses while suppressing comments and script/style bodies across Markdown blocks. Image captions retain their literal content. Original snapshots remain authoritative; raw and expanded text retain the 5,000-character limit, and empty output settles without provider work or stuck playback ownership.

Addressed all three Qodo findings, including Google-style test docstrings. Admission regressions reproduced 15 additional review failures before their fixes. The six-file targeted speech union passed 256 tests; after the final image-caption fix, admission/autoplay passed 149 tests with only the previously verified bounded-ownership test deselected. Adapter integration exercises manual, automatic, retry and empty paths through the real controller, coordinator and TTS admission. No physical listening test was performed.

Plan deviation: the PR's UI census failure reproduced identically on exact dev base 48d40df8ce (977 modules against 975). Deferred selection-value helpers and Notes import parsing at their four importers, preserving behavior and the existing limit. The census now passes at 975; both helpers are explicitly forbidden at UI readiness. Verification: 4 census tests, 57 affected Settings/Notes tests and 26 remaining performance guards passed. No new dependencies or limit increases.

All seven derived-artifact checks pass. Scoped formatting and whitespace checks pass; touched files introduce zero Ruff diagnostics compared with dev. Existing dependency, pytest temporary-directory cleanup and budget-headroom warnings remain. Self-review and independent reviews completed; concrete findings have regression coverage. Console voice documentation updated. ADR required: no; existing ADR-037 applies to speech authority, and import deferral preserves module contracts. No new generalizable lesson beyond the repository's existing baseline-failure evidence and import-deferral guidance.

Modified speech helper and TTS event preparation, two TTS test files, Console voice guide, four Settings/Notes importers, the UI census test and this task. PR: https://github.com/rmusser01/tldw_chatbook/pull/2696.
<!-- SECTION:NOTES:END -->
