---
id: TASK-33094
title: Revise SSH bindings spec around a filesystem endpoint abstraction
status: To Do
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
- [ ] #1 Spec and plan are revised so authority operations (stat, hash, exclusion, identity capture, admission) are FilesystemEndpoint methods with local and remote implementations.
- [ ] #2 Laptop-side CAS and file hashing are replaced by fs_read sha256 plus size for both endpoints.
- [ ] #3 The split-authority posture (client fingerprint check versus server pin) is preserved and documented as an endpoint contract.
- [ ] #4 The plan's task count and file structure are updated to match the revised design before any phase 3 task executes.
<!-- AC:END -->
