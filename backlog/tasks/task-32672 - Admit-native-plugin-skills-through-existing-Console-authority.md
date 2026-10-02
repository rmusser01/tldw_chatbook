---
id: TASK-32672
title: Admit native plugin skills through existing Console authority
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 04:19'
updated_date: '2026-10-01 02:20'
labels:
  - plugins
  - implementation
  - foundation
dependencies:
  - TASK-32671
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-foundation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the first usable local plugin path through existing skill, tool and context services with workspace-specific eligibility.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A reviewed native skill can install disabled, enable in a named workspace and run through the existing Console skill/tool path under ordinary permissions.
- [x] #2 Stable workspace and installation generations are rechecked before injection, launch, approval acceptance and invocation; one installed revision applies across scopes.
- [x] #3 Plugin context stays attributed and untrusted with whole-block limits; manual-only and inline/fork metadata, empty tool allowlists and dependencies preserve their meaning.
- [x] #4 Owned skill listings expose package provenance while service-level standalone edit/delete/overwrite paths refuse package mutations; unrelated standalone skills remain usable.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes (existing accepted package ownership, scoped Console admission and schema boundaries). ADR paths: backlog/decisions/162-managed-agent-plugins.md, backlog/decisions/163-expanded-console-hook-runtime.md and backlog/decisions/009-local-skill-trust-boundary.md. 1. Trace the reviewed F5 service/provider/Console flow and current H3/H4 host context, lifecycle and consent callbacks. 2. Reuse reviewed admission/native Console tests and establish missing-feature RED. 3. Integrate immutable scoped admission, owned skill service mutation fences, stable aliases/schema v3 and real provider/run custody, preserving current parent IDs, durable acceptance and hook authority. Reuse the H3 host context checks. 4. Qualify real Console inline/fork/manual/tool/file paths, workspace isolation, stale approval/material and whole-block context; verify actual production authority storage uses the protected canonical accessor. 5. Targeted plugin/Console/standalone neighbors, static baseline checks, self-review/evidence, ACs/notes and exact task commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Integrated reviewed native skills via existing Console skill/tool/approval owners, stable workspace and revision admission, retained actual root/child/provider lifetimes, and whole untrusted context blocks. Kept current hook consent, mixed attribution, builtin gates and service-wiring/lifecycle modules. Added schema v3 aliases preserving pre-alias authenticated projections. Fixed proven protected-store path drift with the canonical accessor, conditional context import shadowing, and native cache leaks through existing ownership helpers. Updated stale test doubles/profile fixtures without changing production admission or leak thresholds. Validation: 193 native/neighbor checks, 51 standalone Skills checks, 51 hook lifecycle/child/post controls and 9 automatic-work lineage checks passed. Owned Python Ruff/format, shared lint baseline comparison, syntax and whitespace passed. ADR required: yes; implemented ADR-162 and ADR-163; notes/spec/review report updated. Platform limits: local macOS/APFS, isolated marker/provider doubles; no live vendor/keychain or full-suite qualification.
<!-- SECTION:NOTES:END -->
