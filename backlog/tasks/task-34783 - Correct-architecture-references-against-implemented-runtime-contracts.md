---
id: TASK-34783
title: Correct architecture references against implemented runtime contracts
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-10 19:27'
updated_date: '2026-10-11 15:50'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the new engineering reference accurate about live hooks, workflow execution, RAG provider costs, storage and module ownership.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Hooks documentation describes the wired v2 runtime and its permission/checkpoint requirements
- [x] #2 Workflow execution scope and reranking request counts match actual source
- [ ] #3 Schema, module paths and storage descriptions avoid known stale or inaccurate claims
- [x] #4 Console documentation distinguishes streamed transcript polling from synchronous composer, active-tab, title and focus binding during a session switch.
- [x] #5 Dictation documentation describes installed-provider substitution, unavailable resolution when no local provider is installed, and the substitution notice and startup warning.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A; existing ADR-163/138/126 contracts apply. Reason: documentation correction preserves implemented behavior. 1. Verify reviewer findings against source. 2. Correct the affected references and AGENTS drift. 3. Check local links, preflight and independent review.

Stacked PR #3018 source-accuracy follow-up (2026-10-10): verify the Eval default/private override selectors, sparse YAML lifecycle and eval.definitions recovery owner; correct stale configuration, schema and persistence claims plus the database-path override sentence; validate relative links and source references before restoring AC #3 and Done.
ADR required: no
ADR path: backlog/decisions/220-evaluation-defaults-and-private-overrides.md; backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: copy-only alignment with the already implemented and reviewed configuration/recovery contracts.

Final source-accuracy review follow-up (2026-10-10): verify BaseAppScreen.refresh versus recompose and the independent ConsoleTranscript lifecycle; correct app-shell.md's conflation of refresh scheduling, mouse release and keyboard focus restoration. Recheck all 83 relative Architecture links, whitespace and the inherited source/doc correspondence; obtain independent immutable copy review before restoring AC #3 and Done. The existing successful carry preflight remains applicable to this Markdown-only correction.
ADR required: no
ADR path: backlog/decisions/190-console-incremental-display-projections.md
Reason: copy-only description of existing screen/widget presentation lifecycle, with no runtime, storage, security or caller contract change.

Incoming-base source-accuracy follow-up (2026-10-10):
ADR required: no
ADR path: N/A
Reason: routine copy corrections describe existing Console session-binding and dictation-provider behavior; no runtime, storage, security or caller contract changes.

1. Verify the immediate session-switch composer/draft/highlight/title/focus binding and streamed transcript polling against the exact carried source.
2. Verify local dictation provider selection, substitution notice and startup fallback diagnostics against the exact carried source.
3. Correct only console.md and speech.md, preserving their existing streaming, model and privacy descriptions.
4. Check all 83 relative Architecture links, source citations, whitespace and Backlog readability; commit while Task34783 remains In Progress for independent immutable review.

Overnight incoming-source accuracy follow-up (2026-10-11), exact source parent 43475e0273d27afde395a5eb5128fe3737bd1f66:
ADR required: no
ADR path: backlog/decisions/196-dreams-daily-discovery-and-tracking.md; backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: routine copy corrections describe already implemented settings retirement, CSS build inputs and Dreams tracking/scheduler ownership; no runtime, storage, security or caller contract change.

1. Preserve the prior verified Console session-binding, speech substitution and BaseAppScreen/ConsoleTranscript recomposition descriptions while carrying PR3038 onto the exact source parent. Preserve the PR3040 follow-up reference.
2. Correct AGENTS and app-shell.md to describe the removed legacy Tools Settings surface and surviving test-driven backup helpers; keep F4 Settings and the removed enhanced sidebar accurate. Correct both CSS input counts to 64.
3. Add only a bounded Dreams authoritative-file row and active question-track scheduling paragraph, including post-UI-ready binding, retained bare-task shutdown and persisted track-run/last_checked ownership. Qualify reminder-only advancement and timeout claims so they do not describe projected jobs.
4. Verify the affected claims against exact source, all 83 relative Architecture links, whitespace, Backlog uniqueness/readability and the existing 179-node UI census. No pytest or broad preflight for this copy-only correction.
5. Commit the scoped docs and task while AC3 and task status remain pending independent immutable review; do not publish or resolve PR comments.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Corrected live v2 hook ownership/effects/checkpoints, workflow session execution, uncached reranking operation counts and local-provider support; removed volatile schema numbers and inaccurate storage/module claims. Resolved AGENTS conflict with current tool-transcript ownership. Independent source review cleared the scoped changes after one local-provider wording correction; all37 local Markdown links resolve and derived-artifact preflight passes. No runtime behavior changes or new ADR; existingADR163/138/126 contracts remain. Documentation-only verification used source and link checks, without new runtime tests.

Stacked-source accuracy follow-up (2026-10-10, reviewed predecessors through fdb0db32364c778e047acead8cbad90a04458077): corrected the Eval reference to distinguish shipped defaults from sparse eval_overrides.yaml beside the effective config.toml, standalone inputs/full exports, raw YAML persistence and eval.definitions backup ownership. Genuine absence remains unused; invalid/unreadable override paths block complete backup, and retained legacy definitions stay inactive until authorized activation. Replaced stale Eval schema numbers with SCHEMA_VERSION, corrected word-bench HTTP ownership, and clarified explicit database path overrides. Verified against selectors, loader/save/recovery adapter, participant binding, EvalsDB and capture-client source under existing ADR-220/126. All83 relative Markdown links resolve and newly named source files exist; whitespace and Backlog file readability checks pass. Copy-only change; no runtime tests, full suite or new platform qualification. Root will run the final stacked preflight.

Final carry/source review (2026-10-10): rebased the four original docs commits from fdb0db32364c778e047acead8cbad90a04458077..673cc04ca70acdbc9688f51007e65cf7a2336e2e onto exact stack parent 590c18b2b2551f7871684854f982ba79f022bb0a. Pre-copy docs runtime head: 3810f8aab9a8c318003281f51f5e47b8e1b72570. Range-diff reports all four patches unchanged; all 25 owned paths retain their old blobs before this final copy correction. Runtime, tests and CI remain byte-identical to the new parent; inherited lessons-testing-evidence.md exactly matches its parent blob 7730bcd0b7256038f47960cd7075fdba53cf6f67. The preserved PR3040 follow-up branch stays at c08aa049ed2b7d50188e63a105d15e74de60c715.

Independent source review identified one existing app-shell.md sentence conflating refresh scheduling and focus restoration. Corrected it to name screen refresh(recompose=True) mouse release, BaseAppScreen.recompose focus capture/restoration for scheduled and direct awaits, and the independent child-widget lifecycle. Verified actual BaseAppScreen.refresh/recompose plus ConsoleTranscript.recompose/refresh_messages/on_mount; Eval selectors, sparse private override loading and eval.definitions recovery claims remain source-consistent. Existing ADR190 applies; no new ADR or runtime change.

All 83 relative Architecture links resolve before and after the sentence correction. Git whitespace check passes. The normal hash-pinned preflight on 3810f8aab9a8c318003281f51f5e47b8e1b72570 exits zero with all checks passed; its inherited UI census is 175/floor 175. The later Task34785 CI wiring commit b2d24076ca1af97b4a020aa7d90448da9420d547 is not an ancestor of this pinned parent, so no broader claim is made about that separate carry. This Markdown-only correction needs no runtime lint/formatter or test replay; Backlog readability is checked separately after task edits. Receipts: /private/tmp/pr3038-carry-checks.json; pr3038-carry-copy-links.json; pr3038-carry-preflight.log. Final immutable copy review remains pending before restoring AC #3 and Done.

Independent immutable final carry/copy review cleared 998ac7e3ddec5d30c40fdae133361293549ff1b8 (parent 3810f8aab9a8c318003281f51f5e47b8e1b72570, exact stack parent 590c18b2b2551f7871684854f982ba79f022bb0a). Reviewer review_3029 confirmed the screen refresh/recompose sentence matches BaseAppScreen and ConsoleTranscript ownership, all original four patch and 25 pre-copy blob correspondences, exact inherited lesson preservation, no runtime/test/CI delta, all 83 final relative links, whitespace and the explicitly limited pre-copy preflight receipt. The identified copy finding is resolved. All acceptance criteria are verified; restoring AC #3 and Done via CLI. Final closeout changes only this task's metadata and receipt.

Incoming-base source-accuracy copy follow-up (2026-10-10): the six carried documentation patches at 496fb4962133fce8f7abfb9805de9308b2a44654 on exact parent b68b1f8b043cfe6aec52c8d8a05e67494ea3fac1 preserve all 25 prior owned blobs/modes. The bounded compatibility review found two inaccurate claims in console.md and speech.md. Reopened this task and added AC #4/#5 plus the implementation plan through Backlog CLI before editing.

Corrected the Console reference to distinguish streamed transcript/background activity projection (periodic polling and explicit sync calls) from synchronous session-switch draft custody, active-tab highlight and title updates. The shared tab/switcher activation path also restores composer focus before its first await; background binding preserves focus. Corrected dictation resolution to select the configured provider when installed or the first installed local provider in declaration order; only no installed local provider makes resolution unavailable. Replacing a configured provider reports both names in the app-scoped warning toast and the startup WARNING diagnostic (fallback with VAD, degraded-fallback without VAD). Existing model override and provider-scoped model behavior is preserved.

Exact source evidence at 496fb4962133fce8f7abfb9805de9308b2a44654: UI/Console_Modules/session.py:3206-3207 and 5680-5699; Chat/console_voice_input.py:342-367, 413 and 1573-1580; Utils/local_stt_providers.py:120-127; UI/Console_Modules/dictation.py:1535-1559; app.py:3266-3289. All 83 relative Architecture links resolve; source references exist, whitespace passes, and the configured project Python verifies all 5105 Backlog task files readable. Receipt: /private/tmp/pr3038-b68-copy-checks.json. Changes are limited to the two Markdown documents and this task; no runtime, tests, UI, scripts or CI edits, no new ADR, test replay or full preflight. AC #4/#5 and task status remain pending independent immutable review before closeout.

Incoming-base accuracy follow-up: source70b395762b100cd7afb8752e37d5928a78c9bd10 scopes poll-only Console descriptions to streaming/transcript projections and documents immediate draft/highlight/title/focus binding before session activation awaits. Speech now distinguishes missing configured providers (installed declaration-order substitution, warning toast and startup WARNING with both names) from no installed local provider (unavailable). Source-backed independent review2873 clears both corrections;83Architecture relative links, whitespace and task readability5105 pass. Onlyconsole.md/speech.md and this task changed; no runtime/test/CI changes or new ADR. Prior source qualification remains separate and unchanged.

Overnight incoming-source copy correction (2026-10-11): carried the eight docs patches from 58d8439f4ec44aae87d05e0e7a4c8e778e5f8b92..7101faef2ccd1e00fa8713f751a89d438400fd58 onto exact source parent 43475e0273d27afde395a5eb5128fe3737bd1f66. Pre-copy head e53b0d81c56359a47c512e84dc62484600689aa6 preserves 24 non-AGENTS owned blobs/modes; the first AGENTS patch has one context conflict, with incoming merge-queue guidance retained and the reviewed screen list, F4 Settings and removed sidebar copy preserved. The PR3040 follow-up reference remains 0b9f95b4b74173343e8fe40b88fe3c2b08c08e11.

Corrected the removed legacy Tools Settings description and identified its surviving test-driven Backup_Recovery helpers in AGENTS and app-shell; corrected both CSS source-module counts to 64. Added a bounded Dreams row and paragraph for active question-only track projections, page-tracking through watchlists, post-UI-ready handler binding, separately retained/cancelled checks and DreamsDB track-run/last_checked ownership. Qualified reminder advancement and handler-timeout copy so it does not describe detached projected jobs. Existing ADR196/126 contracts apply; no new ADR, feature or runtime change.

Source evidence is pinned to the exact parent: settings_backup_helpers.py:1-14; shell_destinations.py:197-212; build_css.py:195 and 297-299; dreams_projection.py:225-289; dream_track_handler.py:24-79 and 119-139; app_service_wiring.py:3513-3562; app_lifecycle.py:1219-1237; track_service.py:1074-1094 and 1224-1256; scheduler/loop.py:797-853. All 83 relative Architecture links resolve. Seven prior Console/speech/recompose source seams are blob-identical, the console.md/speech.md bytes are preserved and the app-shell recompose distinction remains unchanged. Whitespace, task ID/readability and 179-entry UI census checks pass. Verification is copy/source/guard-only: no pytest, wider preflight, runtime, tests, scripts or CI edits.

Durable receipts are under /Users/macbook-dev/.codex/visualizations/2026/10/10/01a1267e-26ee-7513-9c34-58321c2fef6a/pr-review-evidence/: pr3038-434-carry-correspondence.json, pr3038-434-range-diff.log, pr3038-434-copy-source-links.json and pr3038-434-final-guards.json. Old historical qualifications retain their original limits; purged temporary receipts are not presented as durable evidence. AC3 and task status remain pending independent immutable review before Done or publication.
<!-- SECTION:NOTES:END -->
