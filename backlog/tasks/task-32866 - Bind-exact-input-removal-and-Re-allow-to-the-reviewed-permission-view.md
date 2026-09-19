---
id: TASK-32866
title: Bind exact-input removal and Re-allow to the reviewed permission view
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 15:49'
updated_date: '2026-09-19 16:10'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep delayed permission controls tied to the displayed rule, profile and tool definition, and prevent completed actions from replacing a newer selection.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Obsolete or repeated Remove and Re-allow presses cannot act on a replacement view; current controls preserve exact rule ownership and profile.
- [x] #2 Re-allow refuses a tool definition changed since rendering and valid re-allow still clears the downgrade.
- [x] #3 Failed actions remain retryable; successful completion never overwrites a newer permission selection.
- [x] #4 Targeted regressions, independent review and disposable native dark/light evidence verify the existing permission model and presentation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce delayed/repeated Remove and Re-allow presses, definition drift and late completion using mounted inspector and private permission-store fixtures.
2. Capture mounted control targets and reviewed definition fingerprints, consume each control once, and guard refresh/retry by the originating view.
3. Run targeted checks and independent review; verify both actions in the native app at both supported review sizes/themes with disposable state, then save a draft stacked on PR2731.
ADR required: no
ADR path: backlog/decisions/032-local-agent-tool-permission-boundary.md; backlog/decisions/150-design-token-system-and-design-language.md
Reason: routine UI ownership repair under existing permission and design boundaries; no new persistence, permission policy or service contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed mounted action ownership for exact-input Remove and Re-allow in the MCP inspector/workbench. Each control captures its target, rule owner and reviewed profile; replacement invalidates it before pruning and one press consumes it. Re-allow checks the rendered definition hash. Cached retry, session refresh and guarded completion retain that hash; a fresh selection captures a new one. Failed writes restore usable controls, and completion is fenced under the inspector lock to preserve newer selections. No permission policy, storage, service API or visual styling change.

Verification: 56 targeted cases pass (27 new mounted regressions, 10 parent Revoke checks, 19 permission-store cases). Original stale/duplicate/completion and definition-drift failures were reproduced; internal review found two cached-refresh variants, both reproduced and repaired. Final review has no remaining blocker. No new Ruff diagnostics; new files formatted. Seven derived guards pass. No full suite.

Twelve inspected native captures cover keyboard Remove and Re-allow at 120x40 and 170x48 in dark/light, using a disposable disconnected discovery fixture and real local/permission stores and service. Selected rule removed, reviewed hash persisted, warning cleared, other profile preserved; zero network attempts/tool executions. Normal exit, released lock, ten healthy private databases, zero conversations/messages, unchanged defaults and matching source/runner hashes. Harness-only startup fixes and initial test mistakes are explicitly recorded.

Evidence: Docs/superpowers/qa/2026-09-19-mcp-rule-actions/README.md and GALLERY.md. Both review ledgers updated; lessons-testing-evidence.md records the reproduced cached-fingerprint trap. Fresh all-ref/worktree census and sole-owner/history checks validate TASK-32866.

ADR required: no new ADR. Existing backlog/decisions/032-local-agent-tool-permission-boundary.md and backlog/decisions/150-design-token-system-and-design-language.md govern the repair. Save a separate draft stacked on PR2731 (6de5273903) because the renderer is shared. Current-head CI/review and owner visual approval remain merge gates. Other permission controls and connected runtime journeys remain outside scope.
<!-- SECTION:NOTES:END -->
