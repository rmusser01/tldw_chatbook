---
id: TASK-22509
title: Read-only virtual CLI with independent command permissions
status: Done
assignee: []
created_date: '2026-08-27 04:52'
updated_date: '2026-08-28 14:48'
labels:
  - console
  - tools
  - security
  - agents
dependencies: []
references:
  - backlog/decisions/094-raw-and-virtual-cli-execution-boundaries.md
  - Docs/superpowers/specs/2026-08-26-raw-and-virtual-cli-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give models a compact shell-like read-only interface over existing workspace and Git capabilities without invoking a host shell. Every virtual command is discoverable by default but remains fail-closed behind its own Allow, Ask, or Off state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One structured virtual_cli model tool accepts a fixed command enum and argv array and never parses or falls back to a shell string
- [x] #2 V1 exposes only read-only ls, cat, grep, find, stat, git_status, git_diff, git_log, git_blame, and git_branches commands
- [x] #3 Each virtual command resolves under local:__virtual_cli__ with its own stable definition hash and Allow, Ask, or Off state; missing state resolves to Ask and one command decision cannot authorize another
- [x] #4 The reserved __virtual_cli__ external profile identity and projected records are rejected or filtered at save, load, and catalog projection seams
- [x] #5 Approved commands dispatch to existing filesystem and read-only Git cores so workspace authority, sensitive-path denial, Git exclusions, and existing scan and result caps remain authoritative
- [x] #6 The virtual tool is model-only, available by default when local tools are enabled, blocked by the global tool kill switch, and never treats catalog discoverability as execution authorization
- [x] #7 Approval rows and stored decisions identify the selected virtual command, including batches with multiple virtual_cli calls, without permission-key collisions
- [x] #8 The canonical Tools destination displays virtual commands as a distinct group and explains that their permissions are independent from equivalent fs and Git tools
- [x] #9 Focused tests cover schema and argv validation, unknown flags and commands, per-command permission isolation, reservation filtering, path confinement, sensitive paths, output sanitization, and mounted Tools behavior
- [x] #10 The Console tools guide documents the virtual command set, read-only boundary, independent permissions, and absence of shell syntax
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow Docs/superpowers/plans/2026-08-26-read-only-virtual-cli.md using focused red-green cycles: 1. Add the fixed argv parser and direct read-only dispatch registry. 2. Reserve __virtual_cli__ at external-profile seams. 3. Add one model schema with per-command HubTool permissions and per-call identity. 4. Register the provider under existing local-tool/global gates. 5. Expose independent command rows in the canonical Tools UI. 6. Document and verify the boundary. ADR required: yes. ADR path: backlog/decisions/094-raw-and-virtual-cli-execution-boundaries.md. Reason: ADR-094 defines the synthetic principal, no-shell boundary, default Ask state, and independent command authority.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented one structured, model-only `virtual_cli` tool backed by a strict fixed-command argv parser and direct reuse of the existing confined filesystem and read-only Git cores. Added the independently reserved `local:__virtual_cli__` permission principal, ten stable command-level Hub rows, fresh per-command Allow/Ask/Off resolution, call-scoped approval stamps, global/root gate rechecks, and bounded sanitized output.

Wired the provider into both disposable preview and live Console registries without changing the established provider-composer return contract. Native call IDs now survive the approval-card round trip, so multiple `virtual_cli` calls in one batch can receive different decisions. MCP ▸ Tools shows a non-executable **Virtual CLI (read-only)** group whose copy explicitly distinguishes these permissions from equivalent filesystem and Git tools; the Console and MCP user guides document the no-shell boundary.

Verification on the rebased `dev` tree: 572 focused tests passed across command parsing/dispatch, provider gates, permission reservation, approval routing, Console preview/live composition, mounted Tools UI, and documentation contracts. Ruff passed for all changed production and new test code; the touched legacy local-tool test file passed with its pre-existing E401/E702 exclusions. `git diff --check` passed. The full suite was not run, following the repository's targeted-test policy. ADR: [ADR-094](../decisions/094-raw-and-virtual-cli-execution-boundaries.md); no new ADR was required.
<!-- SECTION:NOTES:END -->
