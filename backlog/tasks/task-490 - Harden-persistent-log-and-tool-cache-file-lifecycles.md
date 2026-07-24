---
id: TASK-490
title: Harden persistent log and tool-cache file lifecycles
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-23 13:55'
updated_date: '2026-07-24 13:43'
labels:
  - security
  - privacy
  - logging
  - tools
dependencies:
  - TASK-488
references:
  - backlog/decisions/022-local-private-data-boundary.md
documentation:
  - Docs/superpowers/specs/2026-07-23-local-privacy-containment-design.md
  - Docs/superpowers/plans/2026-07-24-private-persistent-artifact-lifecycles.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Apply the private-path boundary to rotating application logs, the MCP execution log, and the optional tool-result cache while removing executable cache deserialization.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Configured application log filenames are basename-only and cannot escape the canonical application log directory.
- [ ] #2 The application log directory is `0700`; active, newly rotated, and eligible existing application log generations are `0600` on POSIX.
- [ ] #3 MCP execution JSONL generations use the same private creation, append, rotation, no-follow read, line-count, and existing-file hardening contract.
- [ ] #4 An unsafe persistent-log target disables only that file sink and reports a bounded redacted diagnostic; terminal and UI logging continue.
- [ ] #5 The tool-result cache never loads pickle; a versioned, size-bounded, validated non-executable format is stored atomically as `0600`, while unsupported results remain memory-only.
- [ ] #6 Eligible legacy `tool_results.cache` files are hardened and left inert; they are never deserialized or silently deleted.
- [ ] #7 Behavioral tests cover traversal, absolute log names, rotation, read/count/write symlinks, unsafe parents, target replacement, cache corruption, legacy cache handling, POSIX modes, and Windows posture.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/022-local-private-data-boundary.md
Reason: Implements ADR-022's accepted private persistent-artifact lifecycle without changing it.

1. Add failing basename, mode, no-follow, rotation, cache-format, corruption, replacement, and platform-posture tests.
2. Add descriptor-anchored private append and atomic replacement primitives.
3. Secure the application rotating-file sink and fail closed per sink.
4. Secure MCP log append/read/count/rotation.
5. Replace pickle cache persistence with strict bounded versioned JSON.
6. Run focused and broad verification, sentinel probes, self-review, and task closeout.

Detailed plan: Docs/superpowers/plans/2026-07-24-private-persistent-artifact-lifecycles.md
<!-- SECTION:PLAN:END -->
