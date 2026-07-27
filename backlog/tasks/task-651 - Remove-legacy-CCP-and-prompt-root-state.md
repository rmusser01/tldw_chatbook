---
id: TASK-651
title: Remove legacy CCP and prompt root state
status: Done
assignee: []
created_date: '2026-07-26 23:50'
updated_date: '2026-07-27 17:44'
labels:
  - architecture
  - state
  - personas
  - prompts
dependencies:
  - TASK-647
references:
  - backlog/decisions/011-chatbook-workbench-ui-system.md
  - backlog/decisions/033-application-session-state-ownership.md
  - >-
    Docs/superpowers/specs/2026-07-26-tldwcli-reactive-state-decomposition-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove root state and stale callbacks for the retired CCP and prompt editor so Personas and Library remain the only production owners.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ccp_active_view, CCP provider state, editing and current CCP or conversation identifiers, current prompt state, and their root watchers and handlers are removed.
- [x] #2 Companion character-image, search-timer, generation, and dead-initializer state is removed.
- [x] #3 Production import completion refreshes the mounted real owner or defers to a fresh owner load without old widget identifiers or a root cache.
- [x] #4 Canonical Personas and Library prompt flows pass in the normal production TldwCli.
- [x] #5 Focused ownership, privacy, static, formatting, compile, and authorized integration checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/033-application-session-state-ownership.md; backlog/decisions/011-chatbook-workbench-ui-system.md
Reason: Existing ADRs assign character/persona and prompt state to Personas and Library; no new ADR is required.

1. Guard the exact removed root set.
2. Delete legacy CCP/prompt app paths.
3. Remove stale import refresh callbacks.
4. Verify canonical production imports and privacy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed the exact retired CCP/prompt root reactive, initializer, watcher, event-dispatch, import-staging, timer/generation, character-image, and AI-generation handler state. Deleted the unreachable conv_char_events, character_ingest_events, prompt_ingest_events, and ai_generation_handler modules after production reachability scans; kept the ccp route alias to Personas and preserved unrelated Notes compatibility exports.

Personas and Library now own import completion. Durable imports settle first; presentation refresh/selection occurs only for the exact mounted current owner and otherwise defers to a fresh owner load. Failure diagnostics contain bounded file-type/category metadata only. A full production TldwCli test imports valid character and prompt JSON, navigates away during a delayed real character import, verifies the fresh Personas owner reloads it, observes the bounded Loguru RuntimeError diagnostic, and proves private system/user sentinels are absent.

Verification: 41 passed, 1 dependency warning in 201.72s for the authorized ProductionApp plus ownership gate; 16 passed, 1 dependency warning in 2.08s for retained-adapter/dead-entrypoint static checks; 2 persistent-diagnostic inventory tests passed in 10.82s. Ruff check passed for every surviving changed Python file; Ruff format passed for the eight scoped changed/new files; compileall, git diff --check, and the diagnostic inventory sentinel passed (399 owners, 971 TASK-492 calls, 5698 TASK-494 calls, 4 sink files). Whole-file Ruff formatting remains pre-existing origin/dev drift in app.py, personas_screen.py, and library_screen.py, so those files were not mass-formatted. The diagnostic inventory also reconciles pre-existing pristine-dev digest drift plus this task removal/addition delta.

Architecture: existing ADR-033 and ADR-011 apply; no new ADR was needed. Main production changes are in app.py, personas_screen.py, library_screen.py, ingest_events.py, ingest_utils.py, and worker_handlers/__init__.py; ownership/production tests and the diagnostic inventory were updated, and obsolete handler tests were removed.

Plan deviations: current dev already had no live CCPTabInitializer/sidebar initializer files to edit. Self-review corrected a vacuous caplog privacy assertion to a post-mount Loguru sink and restored unrelated Notes re-exports before closure.
<!-- SECTION:NOTES:END -->
