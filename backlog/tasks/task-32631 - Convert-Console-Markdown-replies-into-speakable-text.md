---
id: TASK-32631
title: Convert Console Markdown replies into speakable text
status: Done
assignee:
  - '@codex'
created_date: '2026-09-15 15:45'
updated_date: '2026-09-15 15:55'
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
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md
Reason: Routine text preparation under the existing validated Console speech ownership boundary; no schema, provider contract or UI structure changes.

1. Add failing speech-admission regressions for formatting, omissions, original validation, empty output and fallback.
2. Add a pure Markdown-to-speech helper and wire it into trusted completed-response and fallback preparation.
3. Verify manual and automatic request paths, retry behavior, original snapshot authority and meaningful punctuation with targeted tests.
4. Update Console voice documentation, run scoped lint/format checks and self-review the final diff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a pure Markdown-to-speech projection in Chat/console_speech_text.py, invoked lazily from trusted Console and global-override speech preparation. Headings, emphasis, lists, quotes, link/image labels and inline code retain words; table rows repeat column labels; each fenced/indented code block announces Code block omitted. Original snapshots remain authoritative, raw and expanded text both retain the 5,000-character limit, and formatting-only replies settle successfully without synthesis, cooldown or paused automatic speech.

Verification: baseline admission suite passed 20 tests; new behavior produced 21 expected failures before implementation. Final targeted union passed 239 tests across snapshot admission, Console playback, auto-speak wiring, TTS improvements, format adaptation and hands-free utterance entry. Four integration cases exercise the real Console controller, auto-speak coordinator, consent/destination resolution and TTS service admission, replacing only the TTS adapter and audio-output boundary; manual, automatic, retry and empty cases all pass and all fail with conversion disabled in an isolated mutation test. No physical listening test was performed. Existing requests dependency and unrelated pytest temporary-directory cleanup warnings remain.

Independent read-only review found no actionable issues in the approved scope. The new helper passes Ruff; existing touched files introduce zero diagnostics against HEAD (58/2/1 pre-existing diagnostics in handler/admission tests/autoplay tests). New/helper and changed-range formatting plus diff whitespace checks pass. Updated Docs/User_Guide/console/voice-and-hands-free.md. Existing speech authority ADR-037 applies; no new ADR or schema, dependency, provider or UI structure change.

Modified files: tldw_chatbook/Chat/console_speech_text.py; tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py; Tests/TTS/test_console_speech_snapshot_admission.py; Tests/TTS/test_console_speak_autoplay.py; Docs/User_Guide/console/voice-and-hands-free.md. No implementation-plan deviations. No new generalizable lesson beyond existing test-boundary guidance.


PR preparation: isolated branch codex/console-speech-markdown starts at origin/dev 48d40df8ce. Fresh targeted verification passed 239 tests against this worktree. All seven derived-artifact checks passed; the Mermaid input download required a network-enabled retry. Scoped format and diff checks pass, and touched legacy files introduce zero lint diagnostics against this dev baseline.

<!-- SECTION:NOTES:END -->
