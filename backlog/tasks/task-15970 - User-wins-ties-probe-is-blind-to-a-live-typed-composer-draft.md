---
id: TASK-15970
title: User-wins-ties probe is blind to a live-typed composer draft
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-14 01:17'
updated_date: '2026-08-14 02:53'
labels:
  - fleet
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR3a-2 residue-arc live verification (2026-08-13, scratch profile, real Anthropic): a wake fired straight through a held composer draft. The draft 'drafting my next thought' was typed via real terminal keys after a send, was visibly rendered in the composer pane, and was still present after the wake -- yet the instrumented probe logged 'probe: composer=True draft=""' at attempt time and the wake delivered and stamped ~6s after settle (run 2fd18ecf / conv d6dee392, stamp 2026-08-14T01:00:28Z; second repro run identical). ChatScreen._console_wake_user_priority reads ConsoleComposerBar.draft_text(), which returns the canonical SEGMENTS once _segments_initialized -- the live-typed text was not in them. Task 7 (feat/fleet-autowake @e38e62a2f) verified the same flow deferring for a full 50s, so this regressed between that branch's base and current dev (composer draft/segment plumbing changed in the PR #1554 era). The existing wiring test passes because load_draft writes segments directly -- the same harness-shortcut trap as task-15862's context bug. Note the same blindness likely affects every draft_text consumer, not just the wake probe.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The user-wins-ties probe defers a due wake while a draft typed with real keys (not load_draft) is present
- [ ] #2 A test drives the probe through real typed input and fails on the blindness before the fix
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce RED with real key input through the production on_key path (not load_draft)\n2. Diagnose where typed state lives vs where the probe reads\n3. Fix the probe to read the composer the user actually types into\n4. Mutation-test the fix\n5. Live re-verify the deferral with typed text
<!-- SECTION:PLAN:END -->
