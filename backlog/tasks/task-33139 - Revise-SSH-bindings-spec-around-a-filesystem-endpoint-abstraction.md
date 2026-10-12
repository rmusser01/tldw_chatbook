---
id: TASK-33139
title: Revise SSH bindings spec around a filesystem endpoint abstraction
status: Done
assignee: []
created_date: '2026-09-27 19:46'
labels: [docs, workspace]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The SSH remote workspace bindings spec and 21-task plan (Docs/superpowers/specs and plans, 2026-09-24) plan to migrate roughly 15 scattered local-filesystem special-case sites into per-site remote branches — for a binding kind enum with exactly one member today. The spec already concedes the abstraction (RunAdmittedWorkspaceRoot.workspace_executor is transport-agnostic; roots become LocalRoot or RemoteRoot descriptors), but stops short of making authority operations endpoint methods. Revising the spec so stat, hash, exclusion evaluation, identity capture, and admission checks are methods of one FilesystemEndpoint interface with local and remote implementations avoids writing the per-site branch pairs entirely. Cheapest concrete piece: the local executor already spawns a one-shot worker per call, so having fs_read return sha256 plus size for both endpoints deletes the laptop-side CAS split and _hash_file outright instead of migrating them four sites at a time. The split-authority security posture (client-side fingerprint check versus server-side pin) is a deliberate property and must be preserved explicitly.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec and plan are revised so authority operations (stat, hash, exclusion, identity capture, admission) are FilesystemEndpoint methods with local and remote implementations.
- [x] #2 Laptop-side CAS and file hashing are replaced by fs_read sha256 plus size for both endpoints.
- [x] #3 The split-authority posture (client fingerprint check versus server pin) is preserved and documented as an endpoint contract.
- [x] #4 The plan's task count and file structure are updated to match the revised design before any phase 3 task executes.
<!-- AC:END -->

## Implementation Notes

OVERTAKEN BY EVENTS — closed with no change. Verified against origin/dev (2026-10-12): the implementation executed past every phase this revision was meant to precede. Dev carries not only the original spec/plan but three further plans already executed (2026-09-27 ssh-session-worker-and-bundle-cache, 2026-09-28 and 2026-09-29 session followups), 17 SSH-related task files, and the planned Tools/remote_workspace_executor.py + remote_workspace_transport.py — all built WITHOUT a FilesystemEndpoint interface. The per-site branch pairs this task would have avoided were shipped as designed; the laptop-side CAS split (10 _hash_file/_record_fs_read_observation sites in Agents/local_tool_provider.py) remains on dev. ACs #1-#4 are withdrawn: they targeted a pre-execution spec that no longer exists. The underlying consolidation (endpoint abstraction + CAS unification) survives as a possible retrospective refactor if the branch-pair debt becomes costly; it would be a fresh filing against the shipped code, not a spec revision.


## Renumbering provenance

Filed 2026-09-27 as task-33094. origin/dev minted its own task-33094 before this branch merged, so per the landed-keeps-id rule this task moved to task-33139; every inbound reference moved with it.
