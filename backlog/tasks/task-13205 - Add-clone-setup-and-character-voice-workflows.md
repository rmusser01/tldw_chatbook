---
id: TASK-13205
title: Add clone setup and character voice workflows
status: Done
assignee:
  - '@codex'
created_date: '2026-08-09 17:39'
updated_date: '2026-08-11 19:43'
labels:
  - tts
  - audio-cpp
  - speech-lab
  - roleplay
  - profiles
dependencies:
  - TASK-13202
  - TASK-13204
references:
  - backlog/decisions/028-character-tts-generation-profile-ownership.md
  - backlog/decisions/040-speech-lab-current-result-and-auto-play.md
  - backlog/decisions/051-private-tts-clone-reference-assets.md
documentation:
  - Docs/superpowers/specs/2026-08-09-audio-cpp-guided-model-setup-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provide transient clone audition, reusable voice-profile save, character assignment, and character-roleplay generation in Speech Lab.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A reference-required guided recipe projects Start & Set Up Voice or Create Voice & Generate only after the matching server/catalog is ready, while an existing compatible Voice Profile remains selectable without confusing User Profiles with character/persona data.
- [x] #2 Speech Lab lets the user choose a bounded WAV, enter or confirm the required bounded transcript, review exact recipe guidance and local-plaintext privacy copy, and correct field-specific validation without losing the rest of the draft.
- [x] #3 Clone audition canonicalizes one private transient reference owned by the current-result workflow, creates no profile before explicit save, and removes the staged artifact when replaced, discarded, or the app closes.
- [x] #4 Create Voice & Generate uses the exact staged artifact and typed clone admission, produces a structurally valid complete WAV, cleans the request materialization, and presents the normal prominent playback/current-result controls with safe reference provenance.
- [x] #5 Save as Voice Profile is offered only after successful generation and persists the exact canonical bytes and transcript captured by that successful result without reopening the source; failed generation preserves recovery but does not present the reference as proven.
- [x] #6 Profile naming and assignment review never silently changes an app default or character assignment; an explicitly assigned character later captures that exact profile revision for Console/Roleplay speech.
- [x] #7 A first lazy character-roleplay Speak request starts or joins the one compatible managed child, generates the response with the assigned clone profile, produces audible playback, and browsing characters/profiles remains passive.
- [x] #8 Provider switching, reference/profile edits, late generation, failed passive observation, retry, and busy states cannot replace a successful playable result, execute a stale operation, or strand controls/focus; the full flow meets keyboard/narrow-layout/live-announcement requirements.
- [x] #9 Hermetic UI/runtime tests plus clean-profile real-process UAT cover transient audition, save, assignment, character roleplay, audible playback, cancellation, cleanup, and privacy without exposing reference audio, transcript, or paths in evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR path: backlog/decisions/028-character-tts-generation-profile-ownership.md; backlog/decisions/040-speech-lab-current-result-and-auto-play.md; backlog/decisions/051-private-tts-clone-reference-assets.md
Reason: the accepted ADRs already define profile/assignment ownership, current-result ownership, private clone references, typed admission, and cleanup; this task implements those boundaries without changing them.

1. Add a typed transient clone-audition snapshot and exact canonical-reference admission path over the existing Guided Managed materializer.
2. Add one atomic profile-reference creation mutation and preserve the exact successful canonical artifact for explicit save.
3. Build the Speech Lab reference-required action/setup flow with bounded validation, privacy copy, immutable busy/stale projections, and current-result preservation.
4. Extend Save as Voice Profile with explicit assignment review/handoff to the existing Roleplay character-assignment owner; never alter defaults or assignments implicitly.
5. Regress the lazy Console/Roleplay Speak path with the assigned exact profile revision and one managed child.
6. Complete accessibility, privacy, cancellation/cleanup, documentation, real-process UAT, review, and Backlog closeout.

Detailed plan: Docs/superpowers/plans/2026-08-11-task-13205-clone-setup-character-voice-workflows.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added path-free transient clone audition and stored-profile preview identities,
  with exact Guided source/capability/configuration/process admission and
  handler-owned successful clone evidence.
- Added atomic profile-plus-reference creation, exact successful-result save,
  non-mutating Roleplay suggestion handoff, explicit character assignment, and
  lazy assigned Console/Roleplay speech without changing global defaults.
- Added Speech Lab setup/recovery/privacy UI, sanitized current-result
  projection, cancellation and teardown ownership, narrow-layout/focus/live
  state coverage, and the production-width geometry regression found by UAT.
- Verification: 1,922 plan-matrix tests passed in the sandbox; the three
  localhost real-child cases blocked by sandbox bind policy passed outside it.
  Exact changed-file Ruff, the planned seven-file mypy gate, CSS bundle sync,
  and diff checks passed. The broad repository Ruff invocation still reports
  104 pre-existing findings outside the branch diff.
- Clean-profile audio.cpp 0.5.1/PocketTTS UAT passed transient audition, exact
  save, explicit assignment, lazy roleplay speech, two human-confirmed audible
  playbacks, structural WAV validation, privacy, and definitive task-owned
  teardown. Sanitized evidence: `Docs/superpowers/qa/audio-cpp-clone-workflow-2026-08-11/live-uat.md`.
- Review of `97a75fb8b..bb334816a` found no remaining Critical, Important, or
  Minor issues. The embedded-pane-width UAT incident is recorded in
  `backlog/docs/lessons-testing-evidence.md`.
- ADR check: no new ADR was required. This implementation follows ADR-028,
  ADR-040, and ADR-051; no ownership, persistence, privacy, or provider-runtime
  boundary was changed beyond those accepted decisions.
<!-- SECTION:NOTES:END -->
