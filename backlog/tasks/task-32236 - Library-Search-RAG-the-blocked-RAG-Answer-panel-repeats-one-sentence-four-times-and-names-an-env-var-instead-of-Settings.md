---
id: TASK-32236
title: >-
  Library Search/RAG: the blocked RAG Answer panel repeats one sentence four
  times and names an env var instead of Settings
status: Done
assignee: []
created_date: '2026-09-10 14:52'
updated_date: '2026-09-10 19:02'
labels:
  - library
  - search-rag
  - copy
  - critique-9
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With no provider the panel prints `Blocked.` / `Unavailable: …` / `Why: The configured provider has no usable API key. Set OPENAI_API_KEY or add api_key under [api_settings.openai].` / `Next: …` / `Recovery: <same sentence>` / `Owner: LLM provider credential.` The Media reader's identical condition says 'Set one in Settings ▸ Providers & Models.' One missing key, two remedies, one of them TOML; 'Owner' is an internal concept. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 5.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The blocked panel is one line in the Media grammar (`No analysis provider is configured · Set one in Settings ▸ Providers & Models.`) plus an 'Open Settings ▸ Providers' action
- [x] #2 Why/Recovery/Owner are no longer painted; the structured record stays in the log
- [x] #3 Every provider gate in Library resolves reason and remedy from one place
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Credential branch of library_rag_state resolves reason+remedy from ingest_analysis's shared constants
2. library_rag_query_status_children paints one bare-reason callout, drops the recovery Static, adds an Open Settings action
3. The structured record goes to the log instead of the screen
4. Docs
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The credential branch of `LibraryRagQueryState.from_values` now resolves
its reason from `ingest_analysis.NO_ANALYSIS_PROVIDER_REASON` +
`NO_ANALYSIS_PROVIDER_NEXT_STEP` (exported as
`LIBRARY_RAG_NO_PROVIDER_BLOCKED_REASON`), so the Media reader's gate and
this one cannot grow two remedies for one missing key. The panel paints
that sentence bare -- the `Blocked | ` prefix restated what the callout's
own styling says -- and the `#library-rag-query-recovery` Static (the
Why / Next / Recovery / Owner dump) is gone. An "Open Settings ▸
Providers" Button appears beside it for that blocker only, posting the
same `NavigateToScreen("settings", {"category": PROVIDERS_MODELS})`
deep-link the Personas readout gate uses.

`recovery_copy` is unchanged and still carries the env var, the
`[api_settings.<provider>]` table and the owner; it is now logged once per
distinct blocker (`logger.info`, deduped because the builder runs on every
keystroke) instead of painted. Confirmed by grep that no other Library
surface renders it.

AC#3 caveat: the OTHER provider branch ("Select a provider/model before
asking for a RAG answer.", fired when no provider is named AND the
readiness object offers no credential remedy) keeps its own sentence. It
is pinned in 6 places across 3 files and reversing it is outside this
task; it is a different condition (nothing selected, versus selected but
unusable) and it no longer renders a six-line block either.

One line outside this branch's file set: the query-status teardown tuple
in `UI/Library_Modules/library_rag_search_controller.py` must list every
id the builder can mount, or the next refresh raises `DuplicateIds`.

Pins updated (they asserted the six-line block):
`Tests/Library/test_library_rag_state.py` --
`test_named_but_uncredentialed_provider_shows_the_real_remedy`,
`test_credential_remedy_is_markup_escaped_for_its_rendering_sinks` (the
escaping is now pinned on `recovery_copy`, its remaining sink), and
`test_panel_state_threads_the_credential_remedy_into_query_state`. The
four existing `assert not screen.query("#library-rag-query-recovery")`
pins stay green by construction.

Live-verified on a profile with no provider at 235x52 and 100x30: one
line, the button, and the button lands on Settings ▸ Providers & Models
(`caps/rag-blocked-235x52.txt`, `caps/rag-sources-100x30.txt`).
<!-- SECTION:NOTES:END -->
