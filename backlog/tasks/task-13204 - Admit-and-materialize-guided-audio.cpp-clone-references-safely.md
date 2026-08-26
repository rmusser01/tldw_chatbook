---
id: TASK-13204
title: Admit and materialize guided audio.cpp clone references safely
status: Done
assignee: []
created_date: '2026-08-09 17:39'
updated_date: '2026-08-11 05:29'
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
- [x] #1 Audio.cpp profile admission supports exact native voice only, clone reference only, or both solely when the accepted recipe defines the combination and precedence; required references and clone-only families are handled without reopening generic options.
- [x] #2 Admission freezes the exact profile UUID/revision, provider/model/voice selection, recipe identity/revision, reference UUID/digest/transcript, and applied provider/process generation before any asynchronous work begins.
- [x] #3 Reference-bearing requests are allowed only for a compatible accepted guided recipe and the app-owned managed child; External servers and unclassified user-provided server.json models never receive a client-local reference path.
- [x] #4 The exact admitted reference is revalidated under repository revision/generation fences and materialized to an opaque owner-private operation directory with typed voice_ref and reference_text request fields, never to server.json, catalog state, profile options, or public provenance.
- [x] #5 Normal completion, response close, cancellation, timeout, generation replacement, child exit, and app shutdown retain ownership until the adapter can no longer read the file and then definitively remove the exact materialization.
- [x] #6 Startup cleanup touches only recognized owned directories after proving no live owner holds the lock, follows no symlink/reparse point, and never deletes unknown or merely old paths.
- [x] #7 Raw request bodies and reference paths remain absent from diagnostics/logs; all validation, capability, generation-loss, transport, and cleanup failures are normalized outside the exception graph with stable safe recovery guidance.
- [x] #8 Tests cover admission/edit/delete races, incompatible sources/recipes, exact payload shape, lease retention, every terminal cleanup path, stale-directory attacks, and privacy mutation guards.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Define an explicit recipe-level native-voice/reference combination policy and fail closed on undeclared combinations.
2. Extend exact character/default profile resolution to read and freeze the canonical private reference under repository generation and profile revision fences before provider work.
3. Add a lazy POSIX owner-private materializer with opaque operation directories, retained ownership locks, exact cleanup, and proof-based startup orphan cleanup.
4. Add typed native clone request fields and admit them only against the exact compatible Guided Managed app-owned audio.cpp process generation.
5. Bind materialization to the existing admitted-operation and response-held adapter lease so every completion, cancellation, replacement, exit, and shutdown path cleans only after the adapter can no longer read it.
6. Add privacy, source/recipe/process race, terminal cleanup, and mutation-guard tests; update implementation-truth documentation and complete review/verification gates.

ADR required: no new ADR

ADR path: `backlog/decisions/051-private-tts-clone-reference-assets.md`

Reason: ADR-051 already fixes the typed admission, Guided Managed-only local-path authority, private operation materialization, ownership-lock, and definitive-cleanup boundaries implemented by this task. ADR-023 and ADR-028 remain the existing lifecycle and profile-ownership authorities.

Detailed plan: `Docs/superpowers/plans/2026-08-10-task-13204-guided-clone-admission-materialization.md`
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented exact recipe-level voice/reference admission; generation-fenced character and default-profile reference resolution; Guided Managed app-owned-only clone authority; POSIX owner-private materialization with root-scoped cross-instance publication/sweep locking, pre-send identity validation, response-held cleanup, and terminal shutdown joining; exact request/materialization/configuration/process sealing; generation-scoped child diagnostic suppression; and bounded public error graphs. Review findings covering admission forgery, path substitution, cross-instance sweep races, crash residues, shutdown lease ordering, and resolver exception privacy were reproduced and closed with regressions. ADR check: no new ADR required; ADR-051 defines this boundary, with ADR-023 and ADR-028 retained as lifecycle/profile authorities. Verification on the final tree: 1,485 focused unit/integration tests passed with four unrelated dependency/SyntaxWarning warnings; Ruff passed for tldw_chatbook/TTS, the TTS events module, and Tests/TTS; mypy passed for all nine plan-scoped sources; git diff --check passed; independent review reported no remaining Critical, Important, or Minor findings. Documentation now states local-plaintext/POSIX scope and the residual same-UID pathname TOCTOU. No clone setup UI, bundle portability, Windows parity, or live UAT was added.
<!-- SECTION:NOTES:END -->
