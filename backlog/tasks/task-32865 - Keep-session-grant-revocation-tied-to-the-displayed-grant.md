---
id: TASK-32865
title: Keep session-grant revocation tied to the displayed grant
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 15:29'
updated_date: '2026-09-19 16:29'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent delayed or repeated Revoke presses from removing a different grant after inspector refresh or profile navigation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Obsolete Revoke controls cannot revoke a replacement grant or act under a replacement profile.
- [x] #2 A live Revoke acts once on its displayed grant and profile; fresh controls remain usable and unrelated grants survive.
- [x] #3 Targeted mounted and service checks plus native dark/light evidence preserve the existing session lifetime, layout and permission policy.
- [x] #4 A failed revoke can be retried, and completion of an older revoke never replaces a newer profile's grant listing.
- [x] #5 The native QA entry point rejects malformed arguments and unsafe or reused output roots before side effects, discovers tmux portably and documents its contract.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce queued Revoke delivery across changed/reordered grants, profile changes and clears, plus duplicate delivery.
2. Capture each mounted Revoke control’s exact grant/profile context, invalidate it before replacing its view, and consume it once. Preserve the current service API and policy.
3. Run targeted mounted/service checks and independent review; verify native keyboard revocation in both themes and sizes with disposable in-memory grants. Save a separate draft against dev.
4. Address Qodo runner feedback with bounded CLI/path validation, portable tmux discovery and contract documentation; verify rejection before side effects and replay the native journey.
ADR required: no
ADR path: backlog/decisions/032-local-agent-tool-permission-boundary.md; backlog/decisions/150-design-token-system-and-design-language.md
Reason: routine UI event-targeting correction and QA runner hardening under existing boundaries; no new policy, storage or application structure.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed session Revoke ownership: each mounted button captures its grant and profile context; replacement invalidates controls before pruning, and an admitted button is consumed once. Failure rebuilds usable retry controls. Completion rechecks the captured profile after acquiring the inspector lock, preserving newer-profile listings. Existing layout, session lifetime, permission store and service contract remain unchanged.

Verification: 26 targeted cases pass (10 mounted ownership/retry/profile-completion cases and 16 existing service/runtime-gate cases). Original code reproduced seven stale/duplicate ownership failures and a separate late-profile overwrite. The initial harness attempt failed before behavior due to profile lifetime; the existing private-profile decorator corrected it. All seven derived-artifact guards pass, with no new Ruff diagnostics and new-file formatting clean. Independent review found no concrete blocker. No full suite.

Eight inspected native captures at 120x40 and 170x48 in dark/light use the real app, service and builtin runtime gate. Calculator revocation clears only that grant and matrix suffix; other tool/profile grants survive, the saved Ask policy remains, and a fresh gate check asks again. No tool execution/network attempts. Normal exit, released lock, ten healthy private databases, zero stored conversations/messages, unchanged defaults and nine matching source hashes plus runner hash verified.

Evidence: Docs/superpowers/qa/2026-09-19-mcp-session-revocation/README.md and GALLERY.md. Updated both component review ledgers. The reproduced profile harness trap is already documented in lessons-testing-evidence.md; no new lesson invented. Fresh all-ref/worktree census and sole-owner/history checks validated 32865; the CLI's newly created stale 32829 candidate was corrected before implementation.

ADR required: no new ADR. Existing backlog/decisions/032-local-agent-tool-permission-boundary.md and backlog/decisions/150-design-token-system-and-design-language.md apply. Separate draft against dev; current-head CI/review and owner visual approval remain merge gates. Exact-input rule removal, Re-allow and connected-runtime review remain outside this bounded change.

PR2731 Qodo follow-up: validated exact CLI shape, bounded tmux names, canonical temporary profile containment and unused evidence before startup; discovered tmux via PATH and documented main. Kept existing config/database containment validation and used the supported image warmup. Independent review identified a trailing-newline exception in the reused username validator; both reproductions failed then passed with a local whitespace guard. All 21 targeted runner cases pass. Native replay passed all four cells/eight captures, exit 0 and lifecycle checks with unchanged default settings and application source hashes. Original approved receipts remain historical; fresh evidence and finding-by-finding disposition are in qodo-followup/README.md. Application/UI behavior unchanged. Owner approval received; ready PR still requires current-head CI and accumulated review before merge.

Final follow-up checks: all seven derived-artifact guards pass with pinned public asset downloads available; Ruff and formatting pass for all three QA/test files. Independent re-review confirms the newline correction and no remaining blocker.
<!-- SECTION:NOTES:END -->
