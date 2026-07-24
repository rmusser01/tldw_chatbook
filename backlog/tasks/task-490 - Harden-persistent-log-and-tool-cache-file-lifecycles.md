---
id: TASK-490
title: Harden persistent log and tool-cache file lifecycles
status: Done
assignee:
  - '@codex'
created_date: '2026-07-23 13:55'
updated_date: '2026-07-24 14:00'
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
- [x] #1 Configured application log filenames are basename-only and cannot escape the canonical application log directory.
- [x] #2 The application log directory is `0700`; active, newly rotated, and eligible existing application log generations are `0600` on POSIX.
- [x] #3 MCP execution JSONL generations use the same private creation, append, rotation, no-follow read, line-count, and existing-file hardening contract.
- [x] #4 An unsafe persistent-log target disables only that file sink and reports a bounded redacted diagnostic; terminal and UI logging continue.
- [x] #5 The tool-result cache never loads pickle; a versioned, size-bounded, validated non-executable format is stored atomically as `0600`, while unsupported results remain memory-only.
- [x] #6 Eligible legacy `tool_results.cache` files are hardened and left inert; they are never deserialized or silently deleted.
- [x] #7 Behavioral tests cover traversal, absolute log names, rotation, read/count/write symlinks, unsafe parents, target replacement, cache corruption, legacy cache handling, POSIX modes, and Windows posture.
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

## Implementation Notes

Implemented ADR-022's private persistent-artifact lifecycle for rotating
application logs, the MCP execution JSONL store, and the optional tool-result
cache. Added descriptor-anchored private append and atomic replacement
primitives; confined configured log names to basenames; made unsafe file sinks
fail independently with bounded metadata-only diagnostics; and replaced pickle
with a strict 1 MiB versioned JSON envelope. Legacy pickle files are hardened,
never deserialized, and automatic persistence leaves them untouched.

Verification:

- 41 focused persistent-artifact tests passed.
- 96 tests passed with the complete private-path suite.
- 320 MCP tests and 82 adjacent config/tool tests passed.
- Changed-file Ruff, Python compilation, and `git diff --check` passed. The
  repository's existing `config.py` Ruff baseline still contains two unrelated
  unused-local findings.
- A canonical `/private/tmp` sentinel probe verified five artifact generations
  at `0600`, three application-owned directories at `0700`, application/MCP
  rotation, JSON cache round-trip, symlink rejection, and preservation of the
  outside sentinel.

ADR required: yes

ADR path: `backlog/decisions/022-local-private-data-boundary.md`

Reason: This task implements the accepted private persistent-artifact boundary
without changing the architectural decision.
