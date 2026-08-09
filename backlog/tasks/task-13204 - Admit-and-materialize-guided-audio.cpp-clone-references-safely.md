---
id: TASK-13204
title: Admit and materialize guided audio.cpp clone references safely
status: To Do
assignee: []
created_date: '2026-08-09 17:39'
labels:
  - tts
  - audio-cpp
  - backend
  - privacy
dependencies:
  - TASK-13201
  - TASK-13203
references:
  - backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md
  - backlog/decisions/028-character-tts-generation-profile-ownership.md
  - backlog/decisions/051-private-tts-clone-reference-assets.md
documentation:
  - Docs/superpowers/specs/2026-08-09-audio-cpp-guided-model-setup-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add typed clone capability admission and generation-scoped private reference materialization for compatible guided managed audio.cpp children.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Audio.cpp profile admission supports exact native voice only, clone reference only, or both solely when the accepted recipe defines the combination and precedence; required references and clone-only families are handled without reopening generic options.
- [ ] #2 Admission freezes the exact profile UUID/revision, provider/model/voice selection, recipe identity/revision, reference UUID/digest/transcript, and applied provider/process generation before any asynchronous work begins.
- [ ] #3 Reference-bearing requests are allowed only for a compatible accepted guided recipe and the app-owned managed child; External servers and unclassified user-provided server.json models never receive a client-local reference path.
- [ ] #4 The exact admitted reference is revalidated under repository revision/generation fences and materialized to an opaque owner-private operation directory with typed voice_ref and reference_text request fields, never to server.json, catalog state, profile options, or public provenance.
- [ ] #5 Normal completion, response close, cancellation, timeout, generation replacement, child exit, and app shutdown retain ownership until the adapter can no longer read the file and then definitively remove the exact materialization.
- [ ] #6 Startup cleanup touches only recognized owned directories after proving no live owner holds the lock, follows no symlink/reparse point, and never deletes unknown or merely old paths.
- [ ] #7 Raw request bodies and reference paths remain absent from diagnostics/logs; all validation, capability, generation-loss, transport, and cleanup failures are normalized outside the exception graph with stable safe recovery guidance.
- [ ] #8 Tests cover admission/edit/delete races, incompatible sources/recipes, exact payload shape, lease retention, every terminal cleanup path, stale-directory attacks, and privacy mutation guards.
<!-- AC:END -->
