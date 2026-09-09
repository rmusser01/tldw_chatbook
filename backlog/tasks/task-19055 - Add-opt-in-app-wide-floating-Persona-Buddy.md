---
id: TASK-19055
title: Add opt-in app-wide floating Persona Buddy
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-20 17:47'
updated_date: '2026-09-05 03:28'
labels: []
dependencies:
  - TASK-19053
references:
  - >-
    Docs/superpowers/specs/2026-08-20-actor-pack-persona-buddy-and-emote-programme-design.md
  - >-
    backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give users an explicitly enabled, app-wide floating visual companion for one selected local Persona, driven only by trusted application lifecycle state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Buddy is default-off and mounts only after the user explicitly selects an eligible local Persona, persisted profile-locally as `(source = local, local_persona_id)`; Workbench highlight, Console actor, and server-source changes never silently retarget it. If the selected Persona is disabled, soft-deleted, missing, or loses its local binding, the view hides with a stable path-free unavailable reason while the enabled preference remains; restoring or explicitly replacing the Persona re-resolves the view, and no selection mounts nothing.
- [x] #2 An app-owned controller survives screen navigation without retaining screen/widget references and resolves the pinned state priority, all nine built-ins, source-scoped leases, safe custom triggers, and exact Persona/binding/version identity.
- [x] #3 A native Textual 8 floating view is bottom-right by default, draggable, resizable, focusable, collapsible, closable, bounded to the viewport, and never steals focus on state changes; it provides keyboard move, resize, reset, collapse, and close actions without shadowing terminal-convention, reserved, or existing global bindings, and collapses to a labelled compact control when its minimum geometry cannot fit.
- [x] #4 Geometry/enabled/open/collapsed preferences persist profile-locally and are never exported; geometry re-clamps after every viewport change, and splash/auth/recovery/modal surfaces safely hide or cover the Buddy so it cannot intercept input behind them.
- [x] #5 `sprite_frames` animation pauses while hidden/collapsed, respects reduced motion, and falls back through state, idle, and portrait without blanking the UI; frame and availability failures report stable path-free categories.
- [x] #6 Same-owner Buddy work is serialized across replacement screens; DB, resolve, decode, and frame-preparation work runs off the event loop, uncancellable work is shielded and drained before releasing serialization, view targets are weak or identity-fenced, and authority is revalidated after every await. Stale work and replaced views cannot repaint or remove the current view.
- [x] #7 No third-party window dependency, taskbar, snapping desktop, maximize system, model-directed state, or default Persona is introduced.
- [ ] #8 Production-shaped Pilot tests cover normal, wide, and 80x24 layouts, compositor output, and zero flow/`fr` budget; isolated real-terminal verification covers mouse drag/resize, keyboard controls, focus, modal hit testing, navigation, viewport resize, and geometry restore. Evidence includes born-RED→GREEN tests at the actual seams, assigned-worktree import provenance, isolated HOME/XDG/config/data roots set before app import, and mutation proof for authority, lease, cancellation, geometry, and modal-input guards; real SQLite repository coverage is required only if persistence storage changes. Impeccable review follows the final visible change; scoped Ruff, format, compile, diff, and static checks plus diagnostic, privacy, architecture, and governance gates pass.
- [x] #9 Console read-that-back uses trusted message speech and releases Buddy speaking on playback completion or failure; microphone and active-run guards remain effective.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implement the existing ADR-074 Buddy design after repairing the UAT startup and publication blockers. Add app-owned explicit local Persona preferences and trusted source-scoped state leases, an identity-fenced serialized renderer, native Textual floating controls, Workbench selection and lifecycle wiring. Verify targeted unit/Pilot, adversarial authority/cancellation/geometry guards, isolated live Migu import and use, and scoped static/UI gates. Detailed plan: Docs/superpowers/plans/2026-09-05-chatbook-buddy-burndown.md. ADR required: no. ADR path: backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md. Reason: direct implementation of the approved local Persona Buddy boundary.

Live voice continuation: reproduce stale readback speech ownership with a mounted regression, delegate readback selection to ConsoleMessageController.request_console_message_speech, retain existing guards, rerun focused dictation/speech/Buddy checks and real local Kokoro sink UAT. ADR required: no new ADR; existing ADR-037 trusted message authority and ADR-074 Buddy leases apply. Fix an existing interface bypass.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-09-04 repair: implemented ADR-074 local Persona Buddy controller, native view, app lifecycle host, and explicit saved-pack Use for Buddy. Preferences are private/profile-local; selection and exact binding authority remain independent of Workbench highlight and Console actor. Trusted Console run, approval, tool, realtime, dictation, and TTS metadata drive source-scoped leases; no model-text control or guessed wake/offline state. Off-thread resolution/decode is serialized and cancellation-drained; views are identity-fenced, viewport-clamped, and hidden behind protected/modal screens. Tests cover finite error leases, FIFO/private-file rejection, negative geometry, reduced motion, same-authority static fallback, replacement identity, and guarded input. Repaired observed startup glyph typo and missing Console rail-height accessor; narrow Console appearance-button geometry repair tracked in UAT report. Main Persona Visual/view/editor tests: 594 passed. Host/controller final integration: 47 passed. Isolated actual TldwCli Migu import/Save/click/paint/keyboard workflow passed twice; live textual-serve confirmed restart, mouse collapse/close, keyboard geometry, primary-screen navigation, modal isolation, and viewport clamp. Physical mouse drag/resize and real provider/microphone/audio behavior remain unverified. Scoped static checks and CSS sync pass; broader diagnostic/ledger and design-governance failures keep AC8 open and task In Progress. No new ADR, schema, or third-party window dependency. Evidence and reproduction: qa/buddy-uat-2026-09-04/repair-report.md and Docs/superpowers/plans/2026-09-05-chatbook-buddy-burndown.md. No full suite or commit.

Final ancillary Console gate: all 70 rail tests passed after restoring the missing height accessor and correcting appearance-button compact padding/specificity; workspace-controller suite also passed all30. The compositor regression was reproduced with the accessor forced to None before the CSS repair, establishing it as an independent pre-existing layout defect. Existing appearance task status was preserved.

Follow-up authorized 2026-09-04: address remaining native dragging, live provider/voice UAT, and governance failures. Plan: verify native terminal gestures in isolated Migu profile; run synthetic-content real provider and available speech paths in a separate isolated profile; repair diagnostic scanner/environment exclusion and reviewed metadata-only inventory deltas; restore approved token catalog and behavior-preserving Console decomposition under existing/new architectural governance as needed. Do not relax guard assertions or ratchet budgets. Preserve all current dirty changes; targeted tests only. ADR074 governs Buddy; ADR029 diagnostics and existing Console extraction/token ADRs govern ancillary work. Keep AC8 open until evidence completes.

Native UAT follow-up found production-seam mismatch: Textual XTermParser creates MouseDown with widget=None, while Pilot supplies a widget; Buddy title/grip handler currently requires event.widget and ignores real terminal drags. Plan: add failing native-parser-to-mounted-screen move/resize regression, resolve missing target from current compositor under existing active-screen guard, retain capture lifecycle, rerun native foreground UAT. ADR required:no; ADR074 applies; routine correctness fix.

2026-09-05 follow-up: repaired native XTerm widget=None target routing and final mouse-release coordinates. Both parser-to-mounted-view failures were born red; final Buddy view/host run 41 passed. Native Terminal foreground UAT now moves (203,3)→(171,11) and resizes 28x19→37x23 with observed compositor and private persisted preferences; test window was reopened after accidental closure. Real isolated DeepSeek visible send succeeded after fixing TASK31553 setup-refusal/draft recovery, with thinking→idle Buddy state. Real tiny.en recognizer handled a synthetic speech fixture and painted listening; Migu was transcribed Mega. Actual mic, live speaker/TTS and OpenAI realtime are not claimed. Governance scanner contamination, unsafe new diagnostic events, extracted Console ownership and missing token catalog repaired; inventory verifies 522 owners and seven unchanged sink files. Historical unavailable-Git-object skip remains. Detailed evidence/limits: qa/buddy-uat-2026-09-04/followup-report.md and diagnostic-governance-review.md. Keep full AC8 sign-off open rather than infer every unobserved path from these targeted runs.

Continuation 2026-09-05: actual default MacBook microphone and speakers were enumerated; announced five-second mounted dictation + production audio-player loopback captured 87040 bytes and eight recognized words, cleanly released the device and returned Buddy to idle. Exact raw substring failed and requires punctuation-normalized scoring before recognition acceptance. Continue with existing local Kokoro ONNX backend for real synthesis/playback; optional packages and model stay in temporary UAT storage, no production dependency/config change or new ADR (existing provider exercised). OpenAI realtime remains without configured credential. Evidence: live_hardware_voice.py and live-hardware-voice.json.

2026-09-05 voice continuation: actual 5.02-second microphone capture recognized the known phrase locally, released the device, and painted listening→idle. Real Kokoro readback exposed a global-event bypass of Manual Speak authority/lifecycle, leaving Buddy speaking after output drained. Delegate readback to existing trusted request_console_message_speech; two stopped/failed mounted regressions were born red then green. Final real Kokoro sink drained 128000 PCM bytes and painted idle→speaking→idle with null speaking owner, no app exception and unchanged normal config. Eight focused readback checks plus 113 speech/Buddy/microphone-guard/ratchet checks pass. Diagnostic inventory unchanged at 522 owners / seven sinks; scoped Ruff, formatting, compile and diff checks pass. ADR-037 and ADR-074 apply; no new architectural decision. Updated followup-report.md, voice JSON/SVG evidence, plan and live-verification lesson. Optional Kokoro installation/model remains temporary; OpenAI realtime needs configured credential. AC9 checked; AC8 and broader task remain In Progress. No full sweep, staging or commit.
<!-- SECTION:NOTES:END -->
