---
id: TASK-31967
title: Document Chatbook Buddy setup controls voice and troubleshooting
status: Done
assignee: []
created_date: '2026-09-07 08:35'
updated_date: '2026-09-07 19:23'
labels:
  - documentation
  - buddy
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_chatbook/pull/2482'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give users a single practical Buddy guide with setup, Migu selection, movement, voice, approvals and honest troubleshooting guidance.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A discoverable user guide covers the current Buddy workflow and links related setup and voice documentation.
- [x] #2 Instructions distinguish verified behavior from outstanding UAT limitations and use current control labels.
- [x] #3 Relative documentation links and navigation resolve; published mirrors match where applicable.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: user documentation of existing behavior, no runtime or contract change.
1. Read current controls, existing guides and UAT records.
2. Write a dedicated guide and link it from the documentation index and relevant feature pages.
3. Validate links/navigation, inspect the rendered Markdown, and record source revision and evidence limitations.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added Docs/User_Guide/buddy.md and discoverability links from the user-guide index, character/Persona guide, and Console voice guide. Covers local Migu selection, movement/resize, explicit visibility controls, Console voice, approvals, visual editing, and troubleshooting. Source checked at dev 307df6c79 and reviewed against recorded UAT; no fresh physical voice claim. Independent source review completed. Validation: 12 case-sensitive relative links resolve, rendered Markdown inspected, git diff --check passes, 3 mounted movement/control tests and 7 Inspector tests passed. Documentation-only change; no runtime, dependency, security boundary, or schema changes, so runtime lint/performance/security suites were not required. ADR required: no; documentation of existing behavior. No new generalizable lesson arose.
User-requested second review: clarify Speak replies timing, how to focus Buddy, first-run speech preparation, and already-open Show Buddy state. Recheck rendered anchors and source references before completing.
Second review resolved: Speak replies must be enabled before sending (console_speech_controls.py tooltip specifies new replies only); click the pet surface to focus keyboard controls, then return focus to composer; document optional speech dependencies, first-run preparation and 60-second dictation cap; explain Show Buddy is disabled when already open. Checked against widget, Inspector, speech controls, dispatch coordinator and existing voice guide. Reparsed final Markdown and checked all 12 rendered relative href/src targets case-sensitively; git diff --check passed. No runtime changes or new physical UAT claimed; previous 10 targeted control tests remain the evidence for unchanged behavior.
Follow-up provider audit clarified that Kokoro is optional in Chatbook and that server Persona/browser voice settings do not configure Chatbook speech. The guide retains historical Kokoro UAT as evidence only. Relative links and whitespace rechecked; runtime behavior unchanged.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

PR #2482 rebased onto dev 2f378c48d. Fresh independent source review found no issues; Qodo reports zero bugs, rule violations, and requirement gaps. Rendered Markdown and all 12 relative link/image targets verified, all three guide entry points resolve, and git diff --check passes. Rebase exposed task ID collision with older Library rail task; moved this record to TASK-31967 with provenance and retained the older task unchanged.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and linked the consolidated Chatbook Buddy user guide. Verified links, rendering, current control labels, and focused mounted controls.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

## Renumbering provenance

Renumbered from TASK-31943 after PR #2482 rebased onto current dev. The Library rail Media-count task was created at 2026-09-07 08:25; this Buddy guide task was created at 08:35. The older task retains the ID under the repository collision rule. A fresh sweep of 420 refs and 72 worktrees found 31967 available. No runtime or guide-content changes were needed.
