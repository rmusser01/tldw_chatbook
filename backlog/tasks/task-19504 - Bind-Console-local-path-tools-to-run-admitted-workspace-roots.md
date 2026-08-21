---
id: TASK-19504
title: Bind Console local path tools to run-admitted workspace roots
status: To Do
assignee: []
created_date: '2026-08-21'
labels:
  - console
  - tools
  - security
dependencies:
  - TASK-17067
  - TASK-19637
priority: critical
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove the in-app Console's hidden configured-root and process-CWD fallback so local filesystem and Git tools operate only on workspace bindings admitted for that run, while preserving ADR-069's selected-root behavior and unrelated local tools.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Project-instruction-enabled sessions preserve ADR-069's one selected binding and call-time membership, fingerprint, access, and filesystem-identity checks
- [ ] #2 Disabled named workspaces capture their valid bindings at run admission and select them through stable binding-ID aliases without active-workspace retargeting
- [ ] #3 Multiple admitted roots require an explicit root alias; reads honor ro or rw and mutations require current rw
- [ ] #4 Binding removal, locator retargeting, identity replacement, and rw-to-ro downgrade revoke access during a run
- [ ] #5 Default and binding-less named workspaces remove only local fs and Git schemas while preserving local web, Watchlists, and todo tools and built-in sandbox file access
- [ ] #6 The in-app Console ignores configured workspace_root and process CWD; standalone MCP retains its explicit configured root
- [ ] #7 The schema change invalidates stale persistent approvals through the existing definition-hash guard and the upgrade copy discloses reapproval
<!-- AC:END -->
