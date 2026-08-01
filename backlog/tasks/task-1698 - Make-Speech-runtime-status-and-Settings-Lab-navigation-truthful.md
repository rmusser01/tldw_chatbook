---
id: TASK-1698
title: Make Speech runtime status and Settings Lab navigation truthful
status: Done
assignee: []
created_date: '2026-08-01 06:05'
updated_date: '2026-08-01 19:22'
labels:
  - tts
  - settings
  - status
  - navigation
dependencies:
  - TASK-1692
  - TASK-1695
  - TASK-1696
  - TASK-1697
references:
  - Docs/superpowers/specs/2026-07-31-speech-tts-settings-ownership-design.md
  - backlog/decisions/039-global-and-studio-tts-settings-ownership.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give users one reliable operational picture and recovery path between global Settings and the Speech Lab. Configuration validity, provider readiness, catalog freshness, and unrelated local STT or TTS dependencies must remain independent, revision-aware facts while cross-screen navigation preserves intent without losing drafts or triggering hidden work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Speech surfaces report selected-provider configuration, selected-provider runtime, catalog or voice freshness, and STT or local dependency availability as independent rows using the exact configuration and runtime vocabularies from TASK-1692 (STATE-010 through STATE-012).
- [x] #2 Never checked is never rendered as Ready, stale evidence is rendered Stale, and missing Kokoro, Chatterbox, Higgs, or STT dependencies cannot mark an independently reachable external audio.cpp provider Unavailable; the converse also holds (STATE-011 and STATE-012).
- [x] #3 Every displayed runtime observation is matched by canonical provider ID, saved configuration revision, optional runtime and catalog revisions, model where relevant, observation time, and freshness; an older result cannot overwrite newer Settings, Lab status, or artifacts (STATE-013 and CAT-005).
- [x] #4 A configuration can remain Saved while runtime is Not checked, Stale, Unavailable, or Reconfiguring, and a post-persistence reconfiguration failure keeps the saved values visible with an actionable unavailable state instead of rolling back or selecting another provider (CFG-007, CFG-009, STATE-021, and STATE-022).
- [x] #5 Settings-to-Lab and Lab-to-Settings links preserve only canonical provider and allowed operation intent, return to the applicable exact choices when representable, never carry secrets or synthesis text, and never automatically save, discard, test, refresh, or generate (IA-005 and SEC-001).
- [x] #6 Dirty global and Studio drafts are protected by Save and continue, Discard and continue, and Cancel on every cross-screen, category, provider, and dismissal path; failed Save and Cancel retain the original owner, draft, and focus (CFG-012).
- [x] #7 When a global connection draft is dirty, runtime evidence obtained from the active saved configuration is explicitly attributed to the previously saved revision and cannot be presented as proof that the draft works (CAT-006).
- [x] #8 A completed audio artifact remains playable and exportable after status, configuration, catalog, model, or navigation changes unless the artifact itself is explicitly replaced or cleared (STATE-014).
- [x] #9 Status and recovery diagnostics are bounded and omit credentials, submitted text, raw response bodies, arbitrary exceptions, unsafe URLs, and provider payloads while retaining safe revision, time, category, and recovery metadata (SEC-003 and SEC-004).
- [x] #10 Automated race, Textual, and navigation tests cover independent capability rows, external audio.cpp with missing local dependencies, every configuration/runtime state, revision and model mismatches, out-of-order workers, saved-but-unavailable and reconfiguring outcomes, dirty-config attribution, all deep-link intents, draft protection, zero auto-actions, and artifact independence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/039-global-and-studio-tts-settings-ownership.md
Reason: TASK-1698 implements ADR-039's accepted revisioned status, independent capability-row, artifact-independence, and bounded Settings/Lab navigation contract; it introduces no new storage, provider, or runtime boundary.

1. Add failing pure tests for revision-bound status projection, safe diagnostics, independent provider/catalog/local-dependency rows, and bounded provider/intent navigation context.
2. Implement the minimal shared Speech status/navigation projection over existing provider configuration revisions and accepted capability observations, rejecting stale or mismatched provider, revision, catalog, model, and request results.
3. Replace the global Settings inspector placeholder and Lab aggregate dependency gate with independent configuration, selected-provider runtime, catalog/voice freshness, and local STT/TTS dependency rows; preserve Saved separately from runtime outcomes and attribute evidence to the saved revision while a connection draft is dirty.
4. Route Settings-to-Lab and Lab-to-Settings through the bounded target contract, restore provider intent without automatic work, and add Save/Discard/Cancel protection for dirty global provider/category/deep-link/dismissal paths while retaining Studio guards.
5. Preserve completed Playground artifacts across status, catalog, model, configuration, and navigation changes, while continuing to reject late superseded synthesis results.
6. Run focused race, Textual, navigation, save/reconfiguration, privacy, dependency, and artifact suites; run neighboring regressions and static checks; independently review before completion.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ADR-039's shared revision-bound Speech status projection and app-scoped evidence store; independent configuration, runtime, catalog or voice, and local dependency rows in Settings and Lab; bounded diagnostics; guarded Settings and Lab navigation; and completed-artifact independence. Added saved and applied TTS revisions, accurate local and environment provenance, non-importing Settings dependency detection, stale-result rejection, and race-safe catalog mounting. ADR check: existing backlog/decisions/039-global-and-studio-tts-settings-ownership.md applies; no new ADR was required. Verification: 434 related tests passed; Ruff, format, compile, and diff checks passed; independent review returned READY.
<!-- SECTION:NOTES:END -->
