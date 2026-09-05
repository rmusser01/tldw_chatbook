---
id: TASK-31585
title: Close remaining Migu Buddy UAT interaction and voice gaps
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 03:32'
updated_date: '2026-09-05 16:13'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Port the remaining reproducible Buddy UAT fixes onto current dev while preserving its newer Buddy, Console, and governance architecture.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Native Buddy move and resize commit final release coordinates when mouse-move events are coalesced.
- [x] #2 Read-back uses trusted Console speech and clears presentation after playback completion or failure.
- [x] #3 Project-instruction recovery refusal releases the run and preserves the unsent draft without replacing newer edits.
- [x] #4 Diagnostic inventory excludes nested virtual environments while retaining application modules.
- [x] #5 Persona Visual import and edit publication succeeds without weakening file identity and containment checks.
- [x] #6 Targeted tests and scoped static checks pass on the actual PR tree; earlier live evidence and remaining OpenAI credential limitation are clearly distinguished.
- [x] #7 Real Manual Speak and readback playback makes Buddy speaking only during playback, releases on terminal acknowledgement after context changes, and preserves another voice owner.
- [x] #8 At 80 columns a busy local transcription keeps the microphone reachable for cancel and preserves attachments; unavailable microphone state is visually distinct and remains clickable to retry.
- [ ] #9 Final native dragging and OpenAI realtime UAT are verified on the PR branch with an application-configured credential.
- [x] #10 Post-acceptance refusal preserves undo and redo for unchanged visible drafts without making successfully sent content undoable.
- [x] #11 Publication roots use centralized path validation with existing canonical-directory and descriptor safeguards preserved, and lazy mount errors retain safe view context only.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Compare older UAT changes with current dev, port only unsuperseded fixes with failing seam regressions, run targeted tests and governance checks, review the final diff, and create a PR against dev. ADR required: no new ADR. Existing ADR-074 Persona Visual/Buddy, ADR-037 trusted speech, ADR-069 project instructions, and ADR-029 private diagnostics govern the repairs; no new ownership boundary or storage schema.

Rebased first onto dev b52080fee0. Investigate and repair the eight baseline verification failures and Library annotation import before final UAT; preserve durable admission and privacy contracts, updating fixtures only where production boundaries demonstrably changed. ADR required: no; existing ADR-037/069/074/029 apply, and test-owner corrections introduce no runtime boundary.

Rebased live Kokoro playback exposed a missing Manual Speak-to-Buddy event. Bind a content-free, request-unique voice lease to existing trusted playback start/terminal callbacks, releasing exact ownership even after session invalidation; verify state transitions and concurrent voice ownership. Existing ADR-074 lifecycle leases and ADR-037 trusted playback govern this repair.

Broader mounted voice checks exposed 80-column clipping after the Send width stabilization and an overridden unavailable-mic CSS rule. Budget the busy chip against current action width, retain the stable Send control, and strengthen the unavailable selector; preserve click-to-reprobe behavior. These are bounded layout repairs under existing Console UX conventions; no new ADR.

Qodo review follow-up: reproduce history loss for keyboard and mouse sends; restore captured history under session/edit ownership guards; extract the existing strict root check into central path validation; add bounded view context without exception capture. Run targeted undo, publication, path and diagnostic tests plus preflight. ADR required: no new ADR; existing ADR-074/069/029 govern unchanged root authority, recovery and privacy contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Rebased first onto dev b52080fee0. Repaired all eight baseline verification failures without disabling privacy checks, and restored the missing Library Iterable import. Real rebased Kokoro UAT exposed a missing Buddy speech lease: actual Manual Speak playback now owns a request-unique lease with exact terminal cleanup, including stale contexts. Real Kokoro drained 128000 bytes with idle/speaking/idle; DeepSeek returned the expected synthetic reply with thinking/speaking/idle. Normal settings unchanged. Fixed actual 80-column busy microphone clipping while retaining stable Send width; compact copy and full tooltip preserve meaning at narrow widths. Investigation found unavailable-mic failures came from bare test harnesses missing the split Console stylesheet; no production CSS change was needed. Final voice UI 103 passed; speech/autoplay/Buddy adapters 102 passed; diagnostic suite 69 passed and 1 historical-object skip; all six preflight guards and scoped fatal lint pass. Independent review has no actionable findings. Existing ADR-074/037/069/029 apply. Evidence: qa/buddy-uat-2026-09-05/README.md and rebased-live-evidence.json. Keep In Progress until physical macOS dragging is confirmed with Terminal foreground and OpenAI realtime UAT has a configured credential. User subsequently authorized review and merge of the verified fixes with these UAT limits recorded.

Qodo review: restore captured undo/redo on post-acceptance refusal only for the unchanged visible draft, preserving successful-send history barriers and newer edits. Centralized the existing canonical-directory root policy in Utils/path_validation.py; publication and cleanup consume its returned paths while retaining descriptor/identity/containment checks. Lazy mount failure records only an allowlisted view key (unknown otherwise), with no exception capture. 174 focused undo/draft/LLM/publication/path tests and two diagnostic privacy guards passed. Diagnostic statement review: exactly one 14-call LLM owner signature changed to add safe_view, with unchanged sink topology. Existing ADR-074/069/029 apply; no new ADR.

Qodo follow-up final checks: all six derived-artifact preflight gates and scoped fatal-rule Ruff checks pass. Physical dragging/OpenAI realtime AC9 remains open as explicitly documented.

Latest-dev CI exposed TASK-31585 collision with PR #2403 Console closeout. Applied the TASK-19601 older-created-date rule: Buddy at 03:32 retains TASK-31585; Console closeout at 03:40 moves to TASK-31591 with provenance and both plan/spec references updated. This is metadata-only; no new ADR. Local backlog guard rerun before publication.

2026-09-05 native follow-up on merged dev f8cb939e2b: foreground Terminal UAT delivered real mouse-down/move/up events. Migu moved from (41,31) to (69,25); physical lower-right resize changed rendered size from 28x15 to 40x21. Graceful exit had no app exception; fresh PID 46565 restored (69,25,40,21). All 22 PTY protocol checks passed separately, including release, viewport bounds, modal/navigation and geometry restore. Background per-PID drag delivered no app mouse events; foreground authorization resolved the native input limitation without another production fix. Evidence under /private/tmp/migu-dragging-uat-20260905 and native screenshots. AC9 remains open only for the separate application-configured OpenAI realtime check. The long-lived harness detected normal config changed since its prior-day baseline, so no normal-config-unchanged claim is made for that interval; fresh restart baseline remained unchanged. Existing ADR-074 applies; no new ADR.

Publish native follow-up receipts and screenshots under qa/buddy-uat-2026-09-05/native-followup with exact tested-commit provenance. This PR adds evidence only; production fixes landed in PR2404. Native acceptance evidence is complete, but combined AC9 remains open for application-configured OpenAI realtime UAT. Existing ADR074 applies; no new ADR.

Native evidence publication PR: https://github.com/rmusser01/tldw_chatbook/pull/2418 against dev93388ba69b. Evidence JSON, local links, all22 recorded terminal checks, backlog ID guard, and diff checks pass. Production fixes are already in PR2404; this follow-up remains evidence-only.
<!-- SECTION:NOTES:END -->
