---
id: TASK-34792
title: >-
  Dreams settings surface - enable Dreams and edit the interest profile in
  Settings
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-29 03:18'
updated_date: '2026-10-12 20:23'
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
- [ ] #1 Dreams can be enabled and disabled from the Settings screen without editing config.toml by hand,Interest profile topics and region are viewable and editable; user topics addable and removable with weight,Section follows ADR-150 tokens and lives in the canonical settings_screen.py (no legacy settings windows),Stale footer-hints test name corrected to the current action count,dreams.md enable-path section points at the Settings screen first, config.toml as the escape hatch
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Study settings_screen.py precedents: Schedules gate (immediate-apply toggle + `apply_settings_mutation_to_cli_config`), Workspaces folder bindings (list editor with attr-stashed per-row buttons), Library/RAG (Select/Input rows); register the new category through every domain-category hook (enum, contract, summary, group, ownership record, inspector guidance, state badge/scope, guided-action message).
2. Build the Dreams section: enable toggle, provider Select with a real "Follow chat defaults" option, model/region Inputs (Enter-to-save, empty = key deleted), read-only rows for the 14 remaining `[dreams]` keys, topic editor writing `source='user'` rows, goals cross-link only, graceful notice when `app.dreams_db` is None.
3. TDD: write Tests/UI/test_settings_dreams.py first (red), implement to green.
4. Fold the queued polish: rename the stale footer-hints test; rewrite dreams.md's enable path to lead with Settings.
5. Gate ×2: Tests/Dreams/ + Dreams UI trio + the new settings test file.

ADR required: no
ADR path: N/A
Reason: UI surface within ADR-196's Dreams feature; settings_screen.py is the canonical settings surface per repo docs (AGENTS.md); no new boundary, storage, or contract decision.

Current-dev integration (2026-10-11): follow Docs/superpowers/plans/2026-10-11-pr2921-current-dev-integration.md and the pinned audit. Reconcile exact feature range with actual sequential dev; fix reviewed consent, URL provenance, captured-chat notice, shared budget, query-goal attribution, disable, native lifetime, recovery and UI contracts in existing owners. Qualify actual integrated source before review/publication. ADR required: existing196/204/207/126 amendments as applicable; no competing new decision.

Updated-source integration: frozen6c074d6eed feature range7b7193d647..head84commits/78paths, Dreamsv6; preserve addressed atomic reservation/generating-goal fixes and fix remaining actual privacy/dispatch/recovery/Guardian/production-Library boundaries plus admitted34801–4. Actual post-PC/Buddy dev and fresh author lease before code/publication; final qualification follows all product changes. Existing ADR196/204/207/126 amendments as needed, no duplicate IDs.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
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

The earlier implementation notes below describe frozen source7f18b6ec1063009b916dad51d468629f20687917 and historical source qualification. Current integration is In Progress; reopened acceptance outcomes require the actual fixed current-dev candidate. No integrated test, publication or merged claim is made by this preparation.

Scoped updated-plan preflight found final qualification preceded the admitted follow-up changes and required-CI coverage was absent from task acceptance. Plan order now places all changes before final source/node freeze and qualification; Guardian integration acceptance gains the bounded required-CI outcome. Collections predecessor/rollback/reopen/future-refusal and installed recovery obligations are explicit in extracted Task1. No product or completion claim.

Current source update: author0c941ba2ec984d960ba83376aeace0e830452bbc, 113 nonmerge commits/88 feature paths after actual merged Goal9595facbbb. Independent source review preserves the real shared Toast fix and stronger reconcile/suspension work; remaining integrated proof and actual post-Buddy/latest-dev publication gate are pending. Prior source/plan/qualification claims remain historical. No product change or completion is claimed here.

PR2921 current private integration qualification (2026-10-12):
Canonical F4 Dreams/Guardian Settings, production CSS tokens, modal focus and narrow geometry are qualified. Final Dreams Settings 19 pass, remaining Guardian Settings 6 pass; six mounted 80x24/120x40 layouts retain native SVG evidence. Original Settings119 diagnostics are source-bound, while feature additions are fixed. ADR-150/161 and existing feature ADRs apply.
Full source-bound outcomes, original failures, native owners and tradeoffs: .superpowers/sdd/2026-10-11-pr2921-current-dev-integration/task-1-report.md; external pr-review-wave2/pr2921-current-integration. Local implementation is ready for independent review. Actual-base restack, complete independent review and hosted required gates remain pending; preserve In Progress and the existing acceptance checklist until the full DoD is met.

Final independent review fix plan (before code):
1. Add native RED controls for file-backed Guardian inbox contention/currentness and actual Settings first-enable/pre-disable getter construction, borrowed handles and cancellation settlement.
2. Enclose notification callbacks and all three Settings getter hops in existing finite local-worker ownership; retain daily physical tasks through shutdown, memory/unknown affinity and exact toggle/reconciliation ordering.
3. Correct the stale ingest_action provenance explanation only.
4. Qualify amended native controls, affected existing cases and recurring CI membership; preserve sealed evidence and write a separate post-review report. ADR required: no; existing ADR126/196/204 govern finite-worker and notification-affinity contracts. Status/AC stay unchanged pending independent review and publication.

Final review fix implementation: file-backed Guardian delivery and all three Settings constructors now use existing finite local-worker ownership. Native controls exposed and then qualified the connected Dreams reconciliation worker caches in the same fix. Final native daily16/Settings9, connected ownership and membership47, existing mounted toggle/import6 pass. Borrowed handles, cancellation settlement, installed recovery drain and exact ordering remain covered; no public-close change. Full separate source-bound report: pr-review-wave2/pr2921-current-integration/post-review-fix-report.md. Existing ADR126/196/204 apply. Keep In Progress/AC/history; scoped independent re-review and hosted publication remain pending.
<!-- SECTION:NOTES:END -->
