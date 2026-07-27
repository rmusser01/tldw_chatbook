---
id: TASK-651
title: Remove legacy CCP and prompt root state
status: Done
assignee: []
created_date: '2026-07-26 23:50'
updated_date: '2026-07-27 18:22'
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

Personas and Library own import behavior and presentation. Review proved that the original screen-owned Textual workers were cancelled by `Screen.remove()` during navigation, even though their `asyncio.to_thread` calls could still commit afterward. Character and prompt imports therefore use one app-owned Textual worker slot per operation type. The application owns only the generic worker lifetime: paths remain inside the coroutine, no prompt/character payload or result is mirrored onto `TldwCli`, repeated starts preserve single-slot semantics, and UI completion still requires the exact current screen owner. A stale owner performs no refresh; a fresh owner reloads durable records.

The production test uses only the normal `TldwCli` and real registered screens. It launches both real production import paths, waits for completed route teardown, proves the old message pump is closed/detached while its app-owned worker remains live, completes a two-prompt batch after Library unmount, verifies a fresh Library reloads both records, and verifies a fresh Personas owner reloads a delayed character import. The bounded Loguru failure assertion proves private system/user sentinels are absent. No test or simplified application is used.

Verification after rebasing onto `origin/dev` at `62ffa5272`: the authorized ProductionApp plus ownership gate passed 42 tests with two dependency/deprecation warnings in 245.24s; the real import-race file passed 2 tests with one dependency warning in 15.27s; and the retained-adapter, dead-entrypoint, and persistent-diagnostic inventory gate passed 18 tests with one dependency warning in 15.06s. Ruff lint passed every surviving changed Python file. Ruff format passed the nine format-clean scoped files; `Tests/UI/test_screen_navigation.py`, `app.py`, `personas_screen.py`, and `library_screen.py` retain verified pre-existing whole-file format drift and were not mass-formatted. Compileall, `git diff --check`, and the diagnostic sentinel passed.

The regenerated branch inventory contains 401 owners, 971 TASK-492 calls, 5,706 TASK-494 calls, and four sink files. A regenerated archive of exact `origin/dev` contains 405 owners, 971 TASK-492 calls, 6,025 TASK-494 calls, and four sink files. The reviewed delta is exactly the four retired diagnostic owners, their calls, and the intended App/Personas/Library line/digest changes; new citation and File Notes diagnostics from current `dev` are preserved.

Architecture: existing ADR-033 and ADR-011 apply; no new ADR was needed. Main production changes are in app.py, personas_screen.py, library_screen.py, ingest_events.py, ingest_utils.py, and worker_handlers/__init__.py; ownership/production tests and the diagnostic inventory were updated, and obsolete handler tests were removed.

Plan deviations: current dev already had no live CCPTabInitializer/sidebar initializer files to edit. Self-review corrected a vacuous caplog privacy assertion to a post-mount Loguru sink and restored unrelated Notes re-exports. The implementation plan originally described screen-owned workers; verified Textual unmount cancellation required the generic worker lifetime to move to `TldwCli` while all domain data, results, and presentation remain destination-owned as required by ADR-033 and ADR-011.
<!-- SECTION:NOTES:END -->
