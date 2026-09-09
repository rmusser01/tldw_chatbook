---
id: TASK-32114
title: Prevent MCP Test Tool teardown from rendering detached editors
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 03:06'
updated_date: '2026-09-09 03:27'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
CI reproduced a Textual render race when closing the MCP Test Tool panel: its visible raw editor remains in the compositor after widget teardown clears its component styles. Close the panel safely while preserving immediate preview revocation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Escape closes a visible raw-editor Test Tool panel and revokes its preview nonce without rendering a detached TextArea.
- [x] #2 A deterministic render-at-detach regression reproduces the original failure and passes with the fix.
- [x] #3 Targeted Test Tool lifecycle and preview tests pass, with the failure and validation documented.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: routine widget-teardown bug fix using the existing Textual batching API; no provider, permission, persistence, or UI contract changes.

1. Record the CI stack and reproduce an actual full timer repaint during raw-editor detach.
2. Add a deterministic failing regression at that framework lifecycle boundary.
3. Batch the awaited panel removal so Textual defers repaint until teardown completes, preserving nonce revocation ordering.
4. Run targeted MCP lifecycle/preview tests and lint, review, then include this scoped CI repair in the authorized PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closing a visible raw editor could let Textual repaint a detached TextArea after its component styles were cleared. The awaited Test Tool panel removal now runs inside the public app.batch_update() context; preview clearing and nonce revocation still happen first.

The deterministic regression delegates real widget teardown, clears cached Rich styles with public refresh(), invokes the actual pending Screen timer, and asserts both safe teardown and a normal post-close repaint. It reproduced the exact CI text-area--gutter KeyError before the fix. Final exact Escape tests: 2 passed; targeted preview/lifecycle selection: 26 passed, 308 deselected. Both files pass formatting and add no Ruff findings relative to their 13 existing findings. Diff whitespace checks pass. Independent review found no actionable issue.

Modified mcp_inspector.py and test_mcp_workbench.py. QA records the original CI job and red/green receipts in Docs/QA/tts-runtime-recovery-2026-09-09/ci-followup.json; lessons-testing-evidence.md records the cache masking trap. The regression observes private Textual lifecycle APIs and may need adapting with framework upgrades; production uses the public batching API. No new ADR required: routine lifecycle fix within the existing UI contract. No full test suite run.

Rebased onto dev c4a7b1911f14181faa471eb1ea209cfdeed98226; only the generated diagnostic summary conflicted. Statement review retained dev's redacted Library count diagnostic, then regeneration and the exact guard passed (7667 TASK-494 calls, no new sink). Tested MCP and measured TTS hashes remain unchanged; user configuration hash matches.
<!-- SECTION:NOTES:END -->
