---
id: TASK-13205
title: Add clone setup and character voice workflows
status: To Do
assignee: []
created_date: '2026-08-09 17:39'
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
- [ ] #1 A reference-required guided recipe projects Start & Set Up Voice or Create Voice & Generate only after the matching server/catalog is ready, while an existing compatible Voice Profile remains selectable without confusing User Profiles with character/persona data.
- [ ] #2 Speech Lab lets the user choose a bounded WAV, enter or confirm the required bounded transcript, review exact recipe guidance and local-plaintext privacy copy, and correct field-specific validation without losing the rest of the draft.
- [ ] #3 Clone audition canonicalizes one private transient reference owned by the current-result workflow, creates no profile before explicit save, and removes the staged artifact when replaced, discarded, or the app closes.
- [ ] #4 Create Voice & Generate uses the exact staged artifact and typed clone admission, produces a structurally valid complete WAV, cleans the request materialization, and presents the normal prominent playback/current-result controls with safe reference provenance.
- [ ] #5 Save as Voice Profile is offered only after successful generation and persists the exact canonical bytes and transcript captured by that successful result without reopening the source; failed generation preserves recovery but does not present the reference as proven.
- [ ] #6 Profile naming and assignment review never silently changes an app default or character assignment; an explicitly assigned character later captures that exact profile revision for Console/Roleplay speech.
- [ ] #7 A first lazy character-roleplay Speak request starts or joins the one compatible managed child, generates the response with the assigned clone profile, produces audible playback, and browsing characters/profiles remains passive.
- [ ] #8 Provider switching, reference/profile edits, late generation, failed passive observation, retry, and busy states cannot replace a successful playable result, execute a stale operation, or strand controls/focus; the full flow meets keyboard/narrow-layout/live-announcement requirements.
- [ ] #9 Hermetic UI/runtime tests plus clean-profile real-process UAT cover transient audition, save, assignment, character roleplay, audible playback, cancellation, cleanup, and privacy without exposing reference audio, transcript, or paths in evidence.
<!-- AC:END -->
