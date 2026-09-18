---
id: TASK-32783
title: Preserve assistant drafts and visible Tool Profile Bind handoffs
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 10:44'
updated_date: '2026-09-18 11:01'
labels:
  - settings
  - ui
  - tool-packs
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users continue from a Tool Profile into a visible workspace assistant control without losing unsaved choices or applying defaults prematurely.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Binding stages only the chosen profile and preserves same-workspace persona and memory drafts; saved defaults remain unchanged until explicit Apply.
- [x] #2 The workspace handoff shows a visible next control and local guidance at compact and wide sizes; missing explicit workspaces offer visible recovery.
- [x] #3 Delayed handoff focus and scrolling never override a newer category, workspace, dialog or user focus.
- [x] #4 Targeted mounted regressions and native dark/light compact/wide journeys pass with private-profile lifecycle evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce hidden Bind continuation and discarded same-workspace drafts using real mounted controls. 2. Preserve same-workspace assistant fields while staging the selected profile and reuse existing assistant result presentation. 3. Complete category handoffs after pane replacement with guarded visible focus; offer a create-workspace continuation when no explicit workspace is active. 4. Verify targeted stale-intent, draft and recovery cases, inspect four native theme/size journeys, review independently and save evidence to draft PR 2707. ADR required: no. ADR path: backlog/decisions/107-portable-tool-use-packs.md; backlog/decisions/079-workspace-assistant-defaults.md; backlog/decisions/150-design-token-system-and-design-language.md. Reason: Bounded navigation and draft-retention repair under existing binding, workspace and design-language rules; no persistence or authority changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Bind now stages only the selected profile, preserving same-workspace Persona/memory drafts and saved defaults until explicit Apply. After pane replacement it reveals Persona or Apply with local guidance; Default receives visible recovery. Revision, screen, workspace and focus guards preserve newer intent. Existing first-bind and memory authority remains unchanged.

91 distinct targeted cases pass: 14 final Bind regressions and 77 existing workflow/assistant/governance cases; the two existing Bind cases were repeated after the final one-line synchronous-focus correction. Independent review reproduced and verified that correction. Four final real native theme/size journeys produced sixteen rendered, inspected captures; exact default persistence, cancellation, clean shutdown, eleven healthy private DBs and unchanged default-profile fingerprints pass.

Scoped Ruff introduces no diagnostics; changed-method/new-file formatting, backlog, diagnostic inventory and diff checks pass. Updated the deferred-focus lesson and component/Tool Profiles ledgers. Native fixture run001 stopped on a duplicate name; run002 used earlier focus ordering; neither is counted as final qualification. Evidence: Docs/superpowers/qa/2026-09-18-tool-profile-bind/README.md. ADR-107, ADR-079 and ADR-150 apply; no new ADR required. The parent PR has a separate prior compact-selector CSS ratchet failure; its repair and the MCP Edit geometry review remain next.
<!-- SECTION:NOTES:END -->
