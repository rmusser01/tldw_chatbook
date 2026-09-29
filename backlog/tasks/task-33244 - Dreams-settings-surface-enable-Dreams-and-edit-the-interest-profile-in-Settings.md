---
id: TASK-33244
title: >-
  Dreams settings surface - enable Dreams and edit the interest profile in
  Settings
status: To Do
assignee: []
created_date: '2026-09-29 03:18'
labels:
  - dreams
  - settings
  - ui
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Re-filed (the original task-32903 was lost when dev reassigned the number; original filed 2026-09-23 from Phase 1 final review ruling R20). Dreams currently requires hand-editing config.toml to enable, and the interest profile (topics/region) has no editing UI - the profile only populates from notes/media/Personal Context signals. Add a Settings screen section: enable toggle, provider/model pickers, region field, topic list editor (user-seeded topics addable/removable with weight), goals are edited via the existing DreamsGoalsModal (g from the story modal) - cross-link rather than duplicate. Fold the queued stack polish: rename the stale footer-hints test (says ten actions, modal now has eleven).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Dreams can be enabled and disabled from the Settings screen without editing config.toml by hand,Interest profile topics and region are viewable and editable; user topics addable and removable with weight,Section follows ADR-150 tokens and lives in the canonical settings_screen.py (no legacy settings windows),Stale footer-hints test name corrected to the current action count,dreams.md enable-path section points at the Settings screen first, config.toml as the escape hatch
<!-- AC:END -->

## Implementation Plan

1. Study settings_screen.py precedents: Schedules gate (immediate-apply toggle + `apply_settings_mutation_to_cli_config`), Workspaces folder bindings (list editor with attr-stashed per-row buttons), Library/RAG (Select/Input rows); register the new category through every domain-category hook (enum, contract, summary, group, ownership record, inspector guidance, state badge/scope, guided-action message).
2. Build the Dreams section: enable toggle, provider Select with a real "Follow chat defaults" option, model/region Inputs (Enter-to-save, empty = key deleted), read-only rows for the 14 remaining `[dreams]` keys, topic editor writing `source='user'` rows, goals cross-link only, graceful notice when `app.dreams_db` is None.
3. TDD: write Tests/UI/test_settings_dreams.py first (red), implement to green.
4. Fold the queued polish: rename the stale footer-hints test; rewrite dreams.md's enable path to lead with Settings.
5. Gate ×2: Tests/Dreams/ + Dreams UI trio + the new settings test file.

ADR required: no
ADR path: N/A
Reason: UI surface within ADR-196's Dreams feature; settings_screen.py is the canonical settings surface per repo docs (AGENTS.md); no new boundary, storage, or contract decision.

## Implementation Notes

- One new Settings category, **Dreams**, in the Domain Defaults rail group, registered through every domain-category hook in `tldw_chatbook/UI/Screens/settings_screen.py` (+ `DREAMS` in `settings_config_models.SettingsCategoryId`). Immediate-apply model like the Schedules gate: no draft, no screen-wide Save; the s/r footer hints correctly stay hidden for it.
- Enable toggle persists `[dreams] enabled` via `apply_settings_mutation_to_cli_config` in an exclusive worker; a successful first enable builds storage through the app-owned `app.get_dreams_db()` so the topic editor activates without a restart. Reads use the raw `getattr(app, "dreams_db", None)` handle (ruling R2) with a "Dreams storage unavailable" notice when None.
- Provider picker: first option is a real "Follow chat defaults" (empty value; picking it deletes `[dreams] provider`); catalog options reuse `_provider_select_options()` minus Manual (no manual-key input here); a hand-set non-catalog provider stays visible as "(from config.toml)". Model and region are free-text Inputs persisted on Enter (empty submit deletes the key) — per-key writes only, never whole-section.
- Budget/cadence keys (`stories_per_cycle` … `track_quiet_retire_count`) render as read-only detail rows labelled by their TOML key (v1: config.toml-only edits, per controller ruling R1).
- Topic editor: facet='topic' rows with weight + source badge ("yours" for source='user'); add row = text Input + weight Select stepper (0.1–1.0, default 0.5) writing `upsert_profile_entry(..., source='user')`; per-row Remove via id-prefix dispatch with the DB id stashed on the button (workspace-exclusions idiom). Goals are cross-linked ("Goals: press g on any Dreams story"), never editable here.
- Protected semantics pinned by test: the cycle's `_upsert_profile_signals` refresh leaves `source='user'` rows untouched while derived rows take merged weights.
- Tests: new `Tests/UI/test_settings_dreams.py` (8 tests, TDD red→green). Mounted-app cases are `@private_profile_test` children — the sanctioned seam for real config-file round-trips (plain in-process factory boots trip the pre-existing `raw_source_selection_changed` admission; seeding writes the pinned `TLDW_CONFIG_PATH` file + `load_settings(force_reload=True)`).
- Polish folded: footer-hints test renamed `..._exactly_the_eleven_actions` (assertions untouched); `Docs/User_Guide/dreams.md` "Enabling Dreams" now leads with the Settings screen, config.toml reframed as the escape hatch.
- ADR-150: only pre-existing token-backed classes composed; no CSS changes.
- Verification: gate green twice (238 passed / 0 failed each). `test_settings_configuration_hub.py` mass-fails identically with and without these changes on this machine (299/147/1445 admission hits, pre-existing environment quirk — matched counts on a stashed pristine run); the Dreams gate itself is unaffected.
- Modified/added files: `tldw_chatbook/UI/Screens/settings_screen.py`, `tldw_chatbook/UI/Screens/settings_config_models.py`, `Tests/UI/test_settings_dreams.py` (new), `Tests/UI/test_artifacts_dreams_modal.py` (rename only), `Docs/User_Guide/dreams.md`.
