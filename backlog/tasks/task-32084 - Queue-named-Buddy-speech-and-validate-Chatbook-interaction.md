---
id: TASK-32084
title: Queue named Buddy speech and validate Chatbook interaction
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 19:33'
updated_date: '2026-09-08 21:06'
labels:
  - buddy
  - console
dependencies:
  - TASK-32083
references:
  - Docs/superpowers/specs/2026-09-08-console-buddy-management-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Speak scoped responses serially and verify the complete Chatbook Buddy journey before a later server port.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Optional spoken responses use one shared queue and start with the conversation name; questions have priority and repetitive progress is consolidated.
- [x] #2 Pause, skip and mute change presentation without cancelling work or answering questions.
- [x] #3 Existing TTS configuration and authority are reused; background listening is never enabled.
- [x] #4 Targeted integration and compact/normal terminal journeys verify management, conversation and workspace behavior.
- [x] #5 User documentation and a server port contract capture shipped behavior and limitations; no server implementation is claimed.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/139-independent-buddy-conversation-and-workspace-bindings.md
Reason: reuses existing TTS privacy and playback authority with one app-owned, explicit-binding Buddy speech queue.

1. Read existing trusted message speech, destination confirmation, playback lifecycle and Buddy scope contracts.
2. Add one serial speech queue with named prefixes, question priority, terminal-response deduplication and pause/skip/mute controls that never alter execution or acknowledgement.
3. Reuse existing configured TTS destination and snapshot admission for explicit bound owners; validate content, profile and owner after asynchronous boundaries and cancel only owned playback.
4. Integrate app-owned lifecycle and optional UI controls with root; verify targeted ordering, privacy, stale-owner/content and failure behavior.
5. Root completes compact/normal integrated journeys, user documentation and server-port contract.

Detailed speech plan: Docs/superpowers/plans/2026-09-08-buddy-named-speech.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the speech portion under ADR-139. One app-owned serial BuddySpeechQueue speaks named terminal responses and current questions for the committed conversation/workspace binding. Questions precede responses; duplicate receipt/message/round keys coalesce and streaming progress is not spoken. Pause restarts the current utterance on Resume, Skip drops speech only, and Mute clears speech without changing run state, decisions or receipt acknowledgement. Explicit microphone capture can hold output and await owned playback cleanup before recording; the queue never starts input.

The coordinator polls only while enabled and survives navigation. It uses existing configured TTS identity/profile resolution, trusted store-issued snapshots with a narrow explicit local owner opt-in, exact destination consent, provider-admission guards and exact playback lifecycle cleanup. Default snapshot callers retain the visible-session requirement. Binding/profile, message edits, Persona revision/retirement and receipt validity are rechecked after asynchronous work. Consent opens only from the reusable speech controls, is scoped to the current binding/profile, and a stale owned prompt closes after rebinding. Root integrated configure, modal controls and app shutdown before shared TTS resources.

Verification: 183 tests passed across existing TTS destination/playback/snapshot admission, trusted store snapshots, queue, native UI controls, workspace ordering and legacy startup tests. A final shutdown-specific run passed 7 tests, including actual app cleanup waiting for Buddy output to stop before closing shared TTS owners. Seven new source/test modules pass Ruff and formatter checks; changed lines in the existing TTS/store/consent modules have no Ruff findings. Scoped diff whitespace checks pass. Tests use the main venv, worktree PYTHONPATH and isolated test profiles; no live paid TTS, microphone device recording or full-suite run. AC4-5 integrated journeys/documentation remain root-owned.

Core files: Persona_Buddy/speech.py, UI/Navigation/buddy_speech.py, Widgets/Persona_Widgets/buddy_speech_controls.py, existing Chat/console_chat_store.py, Event_Handlers/TTS_Events/tts_events.py and Console consent modal. Limitations: unloaded saved result bodies remain in the inbox for explicit review; speech does not restore or activate conversations, and restarting the app resets volatile speech deduplication and Buddy-specific consent.

Final Chatbook integration and review complete. Root gate: 813 passed across Buddy/Persona_Visual, WorkspaceDB/default/provisioning and atomic Persona assignment (162.62s); 285 passed in the 288-case runtime/UI/TTS gate (281.97s). Its new management journey failure was premature Select mutation before nested composition: corrected pilot settling, then all 12 management/entry/workspace journey checks passed. Two other gate failures reproduce identically on pristine c0a42150d1785c9ef7394def669a0aeb62578ff4: test_impersonate_payload_obeys_the_provider_contract returns empty text; test_app_lifecycle_shutdown_drains_all_owners_in_authority_order uses a stale test double lacking _shutdown_collections_capture_runtime. Verified 5,652 baseline source files and all 808 imported production modules; tested methods and test functions are unchanged. Existing legacy builtin architecture boundary failure also reproduced on pristine HEAD; other 7 packaging/architecture checks pass. These unrelated failures were not modified.

Final review fixes: exact saved-binding promotion/restoration, immutable run-owner replay during validating, cold result receipt bootstrap and conversation loading, concurrent-loader retained decisions, and shielded Buddy speech shutdown before shared TTS teardown. Final focused concurrent/cold/retained/bootstrap gate6 and clocks30 pass; cancelled playback teardown13 pass. Additional compact60x20 conversation layout verifies live/unavailable reply and recovery controls (2 pass). All changed Python files parse, zero introduced Ruff diagnostics compared with HEAD, new files pass lint/formatter, and diff whitespace is clean. No full test suite, live paid provider, device microphone or audible playback was claimed.

Docs/User_Guide/buddies.md and console/buddy-conversation.md explain management, exact follow, workspace defaults, dictation, consent, pause/skip/mute and restart limits. Docs/superpowers/specs/2026-09-08-buddy-server-port-contract.md records the later authenticated server contract; no server implementation is part of this change. ADR-139 is the ownership/lifecycle contract and its headless interaction amendment. Testing lessons record source stability, genuine cold startup and modal composition timing. All approved Chatbook tasks are complete; changes remain on the isolated codex/buddy-console-management branch for review.
<!-- SECTION:NOTES:END -->
