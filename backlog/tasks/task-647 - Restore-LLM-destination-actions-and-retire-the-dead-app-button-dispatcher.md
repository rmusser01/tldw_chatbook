---
id: TASK-647
title: Restore LLM destination actions and retire the dead app button dispatcher
status: Done
assignee:
  - '@codex'
created_date: '2026-07-26 23:48'
updated_date: '2026-07-27 06:04'
labels:
  - architecture
  - state
  - llm
  - reliability
dependencies: []
references:
  - backlog/decisions/026-application-session-state-ownership.md
  - backlog/decisions/011-chatbook-workbench-ui-system.md
  - >-
    Docs/superpowers/specs/2026-07-26-tldwcli-reactive-state-decomposition-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore truthful production LLM controls and remove the unused root dispatcher so destination view state and actions have one live owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every visible actionable control in the production LLM window is either handled exactly once by the destination or removed when no runtime contract exists.
- [x] #2 The unsupported custom Transformers server-launch block is absent without adding a new process lifecycle.
- [x] #3 LLM navigation remains owned by LLMManagementWindow while TldwCli.llm_active_view, its watcher, llm_nav_events root routing, button_handler_map, and _build_handler_map are removed.
- [x] #4 Normal production TldwCli tests cover destination navigation and a fault-injected safe action without test or simplified application classes.
- [x] #5 Focused static, formatting, compile, and authorized integration checks pass.
- [x] #6 Production configuration default resolution probes optional macOS speech providers without importing native runtimes, so a headless full TldwCli can be imported without a process abort
- [x] #7 Process controls derive truthful state from app-owned lifecycle claims on every production destination mount; duplicate or stale launch generations cannot overwrite or clear a newer process
- [x] #8 Every live provider Stop action performs bounded non-blocking terminate/kill/reap handling and retains a still-live process handle with Stop enabled
- [x] #9 Every live provider diagnostic path keeps commands, credentials, raw subprocess or API output, and exception payloads out of persistent logs and user-visible failure messages; only bounded generic metadata is emitted for failures while explicit successful domain results remain available
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/026-application-session-state-ownership.md; backlog/decisions/011-chatbook-workbench-ui-system.md
Reason: Existing ADRs assign view state and actions to the mounted destination and process lifecycle to the application; the prerequisite native-package probe correction preserves the existing provider/default boundary and needs no new ADR.
Full plan: Docs/superpowers/plans/2026-07-26-task-647-llm-destination-actions.md

1. Replace import-time native STT default detection with a side-effect-free installed-package probe and lock it with a direct configuration regression test.
2. Freeze the visible Models button census through the normal production TldwCli.
3. Register every supported action on LLMManagementWindow with the normalized (window, app, event) contract.
4. Remove the unsupported Transformers launch block and retain its log as model-operation output.
5. Delete the root dispatcher, duplicate LLM navigation state, and obsolete root routing module.
6. Add production-app regressions for destination navigation, safe fault recovery, sensitive-data containment, generation-safe process ownership, truthful remount controls, and bounded non-blocking Stop/reap behavior.
7. Remove conflicting root worker presentation for destination-owned LLM actions and make process publication/clearing conditional on the app-owned lifecycle claim.
8. Run production-app, structural, formatting, compile, and authorized integration gates; never collect surrogate application tests.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented destination-owned Models actions and removed the dead TldwCli dispatcher, root LLM navigation state, obsolete routing module, duplicate server worker owner, and unsupported Transformers server controls. Added app-owned identity claims for all six local providers with atomic generation-safe publication/clearing, truthful remount and modal-aware controls, bounded off-loop Stop terminate/kill/reap handling, and live-handle retention only while a process remains alive. Added destination-local async presentation generations and exact mounted-owner checks for Transformers scans, Ollama API work, and file-picker completion; discarded Transformers CLI output with bounded cleanup. Hardened Ollama diagnostics so commands, credentials, raw API/subprocess output, exception payloads, and unvalidated metric values do not persist while bounded sanitized successful results remain visible. Replaced native macOS STT imports with side-effect-free installed-package probes. Deleted surrogate LLM tests and added production TldwCli coverage without test/simplified applications.

ADR required: yes. Existing ADR-026 and ADR-011 apply; no new ADR was needed.

Plan deviations and review corrections: expanded lifecycle coverage for retained-dead cleanup, exact current-destination settlement across recomposition and production modals, interleaved output generations, stale Stop presentation, hostile Ollama metric payloads, and bounded Transformers downloader output/cleanup.

Verification: authorized suite 45 passed with 2 dependency warnings; scoped Ruff lint passed; 22 files format-clean; compileall and git diff --check passed; diagnostic inventory independently verified at 424 owners, 1010 TASK-492 calls, 7018 TASK-494 calls, and 5 sink files; fresh specification and quality reviews approved with no remaining blockers.
<!-- SECTION:NOTES:END -->
