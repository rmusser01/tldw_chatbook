---
id: TASK-1231
title: 'File denial teaches recovery; approvals pre-flight the roots check'
status: To Do
assignee: []
created_date: '2026-07-28 09:30'
labels: [console, agents, ux, uat]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expert UAT F3: on an unbound workspace (every fresh install), an approved read_file fails with "outside every allowed root" — truncated in the transcript, no route to the fix (create a workspace, bind a folder in Settings ▸ Workspaces, work in that workspace; Default cannot hold bindings). The model then retries the same path and the user is asked to approve the identical doomed request again until the loop guard kills the run with jargon. First-run users cannot succeed at file access and nothing tells them why or what to do.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The outside-allowed-roots tool error appends the concrete recovery route (Settings ▸ Workspaces folder binding; workspace-scoped sessions).
- [ ] #2 The approval card pre-flights the roots check for file tools and warns when the path will be rejected regardless of approval (never auto-denies; the user can still approve).
- [ ] #3 A retried already-policy-rejected identical request does not generate a fresh approval ask (deny-remembered for the round's path, or surfaced as a single combined ask).
- [ ] #4 Loop-guard termination copy is user-comprehensible.
<!-- AC:END -->
