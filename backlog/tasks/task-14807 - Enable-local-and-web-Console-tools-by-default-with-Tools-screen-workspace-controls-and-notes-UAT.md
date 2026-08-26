---
id: TASK-14807
title: >-
  Enable local and web Console tools by default with Tools-screen workspace
  controls and notes UAT
status: Done
assignee: []
created_date: '2026-08-10 06:04'
updated_date: '2026-08-10 07:14'
labels:
  - console
  - tools
  - uat
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the Console's standard web and workspace tool provider available on fresh and existing profiles by default, expose its master switch and confinement root directly in MCP Hub Tools mode, and prove the real Console agent can read and update a note in an isolated configured workspace.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fresh and key-missing profiles register standard web and workspace tools by default while permission prompts, kill switch, and workspace confinement remain enforced
- [x] #2 MCP Hub Tools mode exposes the local/web master switch and editable workspace confinement root even when the local catalog is disabled
- [x] #3 Tools-mode changes round-trip through config.toml, clearly state next-agent-run semantics, reject invalid workspace directories, and refresh to persisted truth after save failures
- [x] #4 Disabling the master switch removes the local/web provider on the next Console agent run, and enabling it restores the catalog without granting any tool an automatic Allow verdict
- [x] #5 Automated tests directly cover template defaults, missing-key read-site defaults, Tools-mode controls, configuration persistence, and permission/confinement invariants
- [x] #6 An isolated real-Console UAT configures a notes workspace, has a Console agent read an existing note, then writes the exact message `Hi from tldw_Chatbook!` back into it through the real local tool provider
- [x] #7 UAT evidence records the configured root, agent/tool transcript, before/after note content, and verifies the scratch config remains valid and the real user profile is untouched
- [x] #8 User documentation explains default availability, Tools-mode configuration, permission behavior, and restart requirements
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/032-local-agent-tool-permission-boundary.md
Reason: changing the registration default for network and workspace-mutating tools alters the security posture at the existing local-tool runtime boundary; ADR-032 is the canonical boundary record and will be amended rather than duplicated.

1. Amend ADR-032 to distinguish default catalog availability from per-call authorization and pin the default-on, Ask-preserving policy.
2. Change the config template and every direct read-site fallback so fresh and key-missing profiles enable the provider consistently; preserve explicit false values.
3. Add always-visible Local workspace + web controls to MCP Hub Tools mode for the master switch and workspace root, with validated persistence, next-agent-run copy, and truth-refresh behavior.
4. Add focused config, catalog, controller, and Textual/Pilot tests, including mutation-sensitive checks for false opt-out and no implicit Allow.
5. Run an isolated app-level Console UAT with a scratch profile and notes workspace, a deterministic scripted model, the real agent loop, permission store, local provider, and file tools; retain a concise evidence artifact.
6. Update user documentation, run affected inventory/regression suites and static checks, self-review the diff, then complete the task notes and checklist.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Amended ADR-032 so catalog availability defaults on while fresh permission state remains Ask; explicit false, the global kill switch, definition-hash checks, mutating-tool risk floors, and workspace confinement remain authoritative.
- Added one shared production default and applied it to the config template, loader normalization, Console provider composition, gate inventory/counting, and MCP Tools catalog reads so fresh and key-missing profiles cannot disagree.
- Added direct Tools-mode controls for the local/web master switch and workspace root, including existing-directory validation, next-agent-run copy, failed-save truth restoration, and local-source-only visibility. A bundled-CSS regression test exposed the app-wide Checkbox rule; the bundle now contains the scoped compact-toggle override required for the real 100x30 layout.
- Added focused default/controller/Textual tests and a joined Console UAT using a scratch TOML profile, real permission store, real bridge/agent loop, real local provider, and real `fs_read`/`fs_edit` persistence. Deterministic model planning reads the seeded note and writes exactly `Hi from tldw_Chatbook!`; the evidence artifact records the transcript, exact before/after file, valid scratch config, and unchanged real-profile hash.
- Verification before latest-dev rebase: defaults/controller/Tools/UAT group `115 passed`; MCP Servers + Permissions companions `114 passed`; local provider/integration group `115 passed`; strengthened Workbench/UAT delta `8 passed`; `git diff --check` passed. Ruff's changed-file scan reported five findings, all reproduced at the same locations in the unmodified base (`E721` in controller/config and `F821` in Workbench), so no new Ruff finding was introduced; whole-file formatting drift is likewise pre-existing. The user explicitly requested that CI be ignored.
- Rebasing onto `origin/dev` at `1889f4ed9` completed without conflicts. The post-rebase high-signal suite passed `126` tests across defaults, controller composition, bundled Tools UI, every changed Workbench control, and the joined notes UAT.
- Qodo review on PR #1474 produced three findings, all addressed: workspace-root input now routes through `Utils.path_validation.validate_path()` and persists its normalized result; UAT helper classes use strict PascalCase; and the new public Tools-mode config methods use Google-style `Args:` documentation. A focused validator-routing regression was added, and the review-fix suite passed `33` tests.
- Updated the Console tools guide, MCP guide, example-skill requirements, ADR-032, and a durable UAT evidence document. No new lesson entry was added because the bundled-CSS specificity trap is already captured in the repository's UI guidance and the existing testing lessons.
<!-- SECTION:NOTES:END -->
