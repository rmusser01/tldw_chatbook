---
id: TASK-32236
title: >-
  Library Search/RAG: the blocked RAG Answer panel repeats one sentence four
  times and names an env var instead of Settings
status: In Progress
assignee: []
created_date: '2026-09-10 14:52'
updated_date: '2026-09-10 19:34'
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
- [ ] #3 Every provider gate in Library resolves reason and remedy from one place
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

AC#3 IS LEFT UNTICKED (review round 1, finding 4). The OTHER provider
branch ("Select a provider/model before asking for a RAG answer.", fired
when no provider is named AND the readiness object offers no credential
remedy) still owns its literal reason, owner and recovery pointer
("Console controls") at `library_rag_state.py:1350-1355`, so "every
provider gate ... one place" is not literally true. The reviewer agreed
the separate SENTENCE is justified -- the condition really is different
(nothing configured, versus one configured that cannot authenticate) and
its remedy lives somewhere else -- and that its six pins across three
files should be left alone, so this is the "implement the rest, leave the
AC unticked" case the brief describes rather than a defect to fix here.
Rider-worthy follow-up for the controller: give that branch a shared
constant and an action of its own (it is the blocker a brand-new user is
most likely to hit, and its callout is currently a bare imperative with
no "where").

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
**Fix round 1.** The action now asks the state
(`LibraryRagQueryState.blocked_is_no_provider`, beside its two sibling
predicates) instead of comparing the rendered callout copy; the logged
record is un-escaped first, since the log is not a Rich-markup sink
(`\[api_settings.openai]` was reaching the diagnostic with a backslash);
and AC#2 gains the painted assertion the brief asked for --
`test_a_missing_provider_key_paints_one_line_and_no_owner_block` opens the
canvas with a stubbed provider gate at 235x52 and scans the painted panel
region for `Owner:` / `Recovery:` / `Why:` / `OPENAI_API_KEY` /
`api_settings`. The builder-level test could not have seen the panel's
other `recovery_copy` Static (`library_search_rag_panel.py:1309`, gated on
`recovery_selector`, which the query gate leaves empty).
<!-- SECTION:NOTES:END -->
