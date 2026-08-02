---
id: TASK-865
title: >-
  Sweep hardcoded ~/.config/tldw_cli and ~/.local/share/tldw_cli call sites onto
  the real accessors
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 04:35'
updated_date: '2026-08-02 02:24'
labels:
  - security
  - config
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Around 30 call sites build config/state paths from Path.home()/".config"/"tldw_cli"/... or Path.home()/".local"/"share"/"tldw_cli"/... directly instead of using _get_effective_config_path().parent or get_user_data_dir(). The config-dir group (UI/Screens/chat_screen.py:15394,15425 for ui_state.toml; Event_Handlers/notes_events.py:144 and note_ingest_events.py:350 for note_templates.json; Subscriptions/website_monitor.py:72 for feed_cache/, plus ~25 lower-value sites) silently ignores TLDW_CONFIG_PATH -- these files land in the real ~/.config/tldw_cli regardless of which profile is active.

The data-dir group (~18 sites, including Chatbooks/chatbook_importer.py:77-79, Chatbooks/local_chatbook_service.py:102-107, Character_Chat/Character_Chat_Lib.py:1274,2790,3856, Event_Handlers/conv_char_events.py:4152,4213,4264) additionally omits the <user_folder> segment that get_user_data_dir() appends, so multi-user profiles collide into the same directory. The chatbook importer is the highest-value fix in this group because the drift is already live in production, not latent: a reproduction showed chatbook_importer.py:77's literal ~/.local/share/tldw_cli/temp/imports already exists on disk, while the derived, correct .../default_user/temp/imports (matching chatbook_creator.py:97's sibling path) does not -- meaning imports have been extracting to a path outside the per-user tree that any other local user account, or a future multi-profile user, would also read and write.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every swept hardcoded profile-owned config-dir occurrence in ADR-040's normative inventory derives its parent directory from get_cli_config_path().parent instead of a Path.home()/'.config'/'tldw_cli' literal
- [x] #2 Every swept hardcoded active-user data-dir occurrence in ADR-040's normative inventory derives its path from get_user_data_dir() instead of a Path.home()/'.local'/'share'/'tldw_cli' literal that omits the user folder segment
- [x] #3 The chatbook importer's extraction root matches the chatbook creator's temp root (both under get_user_data_dir()/temp/...) with a test asserting the two derive to the same parent
- [x] #4 A test with TLDW_CONFIG_PATH pointed at a profile confirms at least one swept config-dir site (e.g. ui_state.toml) writes under that profile's directory, not the real ~/.config/tldw_cli
- [x] #5 Every remaining executable literal is classified by an exact sentinel exception as inert configuration data, a canonical resolver/default seed, a compatibility constant, a shared artifact, or a read-only legacy probe
- [x] #6 The rejected transcription-history store/viewer, unmounted legacy Dictation window, and unused legacy user-database path helper are retired rather than allowlisted
- [x] #7 No existing global file is copied, moved, imported, or deleted by the completion tranche
- [x] #8 Regression tests use production functions or the full TldwCli application; no reduced test application is introduced
- [x] #9 Swept profile-owned state writers use ADR-029 private atomic replacement and preserve the previous file when serialization or replacement fails
- [x] #10 Generated diagnostics/SQLite inventories, affected legacy-window tests/source censuses, stale feature documentation, release notes, and installed-wheel coverage agree with the retired modules and symbols
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Use ADR-040 to classify every remaining executable literal as profile-owned config state, active-user data, shared artifact, inert default, read-only legacy probe, or unreachable code.
2. Write failing function/full-application path-isolation tests and an executable-token/source sentinel that detects embedded, multiline, indirect-join, duplicate, and stale cases with exact counted exceptions.
3. Retire every rejected transcript-history implementation, including the unmounted legacy Dictation window, plus the unused legacy user-database path helper; reconcile every importing test/source census, generated/curated inventory, compatibility comment, current architecture document, and release note without rewriting historical Backlog records.
4. Apply the design's normative disposition inventory: move each swept profile-owned config/data occurrence onto get_cli_config_path().parent or get_user_data_dir() at the call boundary without migrating existing files, preserve classified exceptions, and route swept private state writers through ADR-029 atomic replacement.
5. Run targeted ownership/privacy/inventory/installed-wheel suites, the full suite, and static checks; then reconcile the acceptance criteria, implementation notes, and task status.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the complete [ADR-040](../decisions/040-profile-owned-state-and-shared-asset-paths.md) ownership inventory under the [TASK-865 completion design](../../Docs/superpowers/specs/2026-08-01-profile-owned-path-completion-design.md). Swept effective-config state now resolves lazily through `get_cli_config_path().parent`; active-user state and Dictation exports resolve through `get_user_data_dir()`. Chatbook import and creation share the active user's `temp` parent. The count-sensitive executable-source sentinel classifies every retained literal as a persisted default, resolver seed, compatibility constant, shared artifact, or read-only legacy probe.

Retired the rejected transcription-history store/viewer, unmounted legacy Dictation window, and unused legacy user-database helper/constants without touching existing user files. Current TTS documentation, release notes, source censuses, the SQLite owner ledger, generated diagnostic inventory, and sdist/wheel absence checks agree with those removals. No Notes/Sync path changed and the locked-base audit found no user-data migration, fallback import, copy, move, or delete behavior.

Kokoro UI blends and backend blend state follow the effective profile and use ADR-029 private atomic replacement; serialization or replacement failures preserve both the prior file and the in-memory mapping. Explicit backend directories remain explicit. Backend managers intentionally belong to one immutable application/config session, while reusable models, voices, and tokenizer artifacts remain shared to avoid duplicate downloads. Existing global state is not silently adopted by a newly selected profile.

Regression coverage calls real production functions/methods or mounts the full installed `TldwCli`; no reduced Textual application was introduced. The focused help regression directly invokes the real `ImprovedDictationWindow.action_show_help()` on a real Widget with only its one-call notification collaborator supplied, which is within the production-method contract. Final review found no Critical issues; the production help wording, scanner assertions, dictionary documentation, unused import, regression coverage, and resulting diagnostic digest were addressed in rebased commit `e2eba8c6a`.

Final post-rebase verification: the exact 16-path Task 8 matrix, including full installed-distribution coverage, passed **210 tests with 7 known warnings in 170.77s**. Both ownership inventories pass at **432 owners / 1073 TASK-492 calls / 6677 TASK-494 calls / 4 sink files**; focused Ruff on all final-fix Python files, `compileall`, and `git diff --check` pass. The earlier repository-wide run completed with **26045 passed, 217 skipped, 65 failed, 2 errors, and 120 warnings**; representative failures reproduced on clean current development as sandbox loopback denial and repository UI/static baselines, not TASK-865 regressions. Branch-wide Ruff/format remain documented upstream baselines, while TASK-owned/final-fix files are no worse than their locked base or green. These baseline-attributed gates are the only verification exceptions permitted by the plan.
<!-- SECTION:NOTES:END -->

## ADR Check

ADR required: yes

ADR path: [ADR-040: Profile-Owned State and Shared Asset Paths](../decisions/040-profile-owned-state-and-shared-asset-paths.md)

Reason: The completion tranche classifies persistent data ownership, profile
isolation, shared artifacts, legacy probes, and migration behavior across
multiple modules.

Design: [TASK-865 Profile-Owned Path Completion Design](../../Docs/superpowers/specs/2026-08-01-profile-owned-path-completion-design.md)

Completion-scope note: existing consumers of `_get_effective_config_path()`
already resolve the active config correctly and are not part of this
hardcoded-literal sweep. New or changed consumers use the public
`get_cli_config_path()` wrapper. This tranche does not modify Notes Sync.
