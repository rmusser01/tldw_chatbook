---
id: TASK-35012
title: Switch model lists each provider's saved default among its first rows
status: In Progress
assignee:
- '@claude'
created_date: 2026-10-14 17:10
updated_date: 2026-10-20 11:13
labels:
- console
- model-switcher
- providers
- model-config
- ux-review-2026-10-14
dependencies:
- TASK-35010
- TASK-33008
- TASK-33641
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verified finding C-M1: a model saved to a provider's list is appended at the end. With no Find text, Switch model (Alt+M) shows the first three models of each ready provider in list order, and lifts only the open chat's current model. So for a provider that already holds three or more models, the model the user just saved sits under '… N more' and looks unsaved.

After TASK-35011, the model a user picks in Settings becomes that provider's saved default, so the default is the model they are looking for. The verifier's constraints hold: keep the three-row cap, and do not reorder [providers] on disk, because other readers use that order. Rows are therefore ordered at display time.

Phase 8 (TASK-33008) already adds the saved model as a row when the list reopens after a return from Settings. Phase 9's TASK-33005.12 decides which models a server listing still offers as rows. This task only orders the rows those rules produce.

Plan: `backlog/docs/plan-2026-10-14-s1-discovered-model-default-saved-lists.md`. Findings closed: C-M1 (Settings UX review, 2026-10-14).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With no Find text, each ready provider's group in Switch model and in Chat settings' model picker shows that provider's saved default for new chats first, or second when the surface's current model of that provider leads the group (the open chat's model in Switch model, Chat settings' draft pair in its picker), when the default is one of that group's rows, even when the provider has more than three models
- [x] #2 Each group lists the same models as before, only reordered, and the '… N more' row counts the remaining ones
- [x] #3 A shipped sample model id is never moved forward as a provider's default
- [x] #4 Find results and their ranking are unchanged
- [x] #5 The Console User Guide says each ready provider's saved default shows first, after the open chat's current model, when it is one of that provider's rows, in both Switch model and Chat settings' model picker
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follows the plan's TASK-35012 mechanism (backlog/docs/plan-2026-10-14-s1-discovered-model-default-saved-lists.md), on T1 (TASK-35010) and T2 (TASK-35011).

1. Tests first, in a new Tests/UI file on the switcher harness (the real popover, the production stylesheets): each new test fails against the branch head on an assertion; invariants the head already keeps get a temporary source edit as their negative control.
   - The position rule: a default saved last of four or more leads its group; second behind the surface's current model; nothing moves when it is the current model; a pair under PREVIOUS/RECENT is not repeated.
   - AC#2: the same models, only reordered, and the '… N more' count. AC#3: a shipped sample id is never lifted (seeded in providers_models and served_models). AC#4: typed Find (exact-name 'Anthropic ', prefix tie 'claude') ranks as before.
   - The probe rule with the real prober: before a local probe answers the default is lifted; after it answers, an unlisted default is absent and a listed one is lifted within the listing.
   - NO-3: Chat settings' pick mode lists the same order. D-c: with no PREVIOUS and no CURRENT row, empty Find highlights the lifted default.
2. `ConsoleModelPopover`: cache each provider's saved default per open beside `_saved` (`resolve_effective_chat_configuration(app_config, provider=key).model`, skipping `SHIPPED_SAMPLE_MODEL_IDS`), and in `_build_rows`' untyped ready branch move it within `visible` only: first, or second when the current model's row leads. `models`, `_models_for`, `_representative_model`, the typed branch and `_capped` are untouched.
3. Docs/User_Guide/console.md: the READY PROVIDERS bullet, and the listing paragraph, name the saved default's place in Switch model and in Chat settings' model picker.
4. Lanes: one census node per new test function, appended after the S1 block, ordered over the shard groups by measured time; MINIMUM_FILES raised with a dated comment; Tests/CI green. No workflow edit (NO-2 (b)).
5. Evidence: the pinned tests the plan names for T3, a covering run at the head and at a base worktree compared by failure name, the size ratchets serially, check_markup_interpolation, ruff, preflight under a scratch HOME.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
With Find empty, Switch model (Alt+M) and Chat settings' model picker (its pick mode) list each ready provider's saved default for new chats first in its group, or second when the surface's current model of that provider leads, when the default is one of the group's rows. Nothing is added, the saved list keeps its order, and typed Find ranks as before.

**Approach** (the plan's T3 mechanism, on T1 and T2). `ConsoleModelPopover._saved_default(key)` resolves the default once per open into `_saved_defaults`, beside `_saved`: P8's `resolve_effective_chat_configuration(app_config, provider=key).model` (`chat_defaults.model` on its own provider, else the provider's table), normalized, and None for a `SHIPPED_SAMPLE_MODEL_IDS` id. In `_build_rows`' untyped ready branch the default moves within `visible` only, after the PREVIOUS/RECENT pairs are taken out: first, or second when `visible[0]` is the current pair (ADR-020 already put it first). A default that is the current model stays where it is, and one already shown under PREVIOUS/RECENT is not in `visible`, so it is not repeated. `models`, `_models_for`, `_representative_model`, the typed branch and `_capped` are untouched; the untyped branch never goes through `_capped`. No `_pick_only` gate (NO-3): in pick mode the current pair is Chat settings' draft pair. No new label path, so no markup change.

**D-b.** The lifted default is the configured connection's saved default, matched by id only within the rows already shown. On a session endpoint (TASK-35190) the rows are that endpoint's, and the default is lifted only if that endpoint's rows hold the same id; nothing gates on `pick_base_url`.

**Evidence** (Tests/UI/test_console_switcher_saved_default.py, new: 9 functions, 14 cases, every one `@private_profile_test`, on the switcher harness with the production stylesheets; the local-server cases run the real prober over a stubbed network, the `servers` fixture of test_console_switcher_local_probe.py). Against the pre-fix popover (the branch head 1ca320e4bb, the test file copied into a base worktree) 12 of the 14 fail, each on an assertion; the two that pass there are invariants the base keeps, and the temporary source edits below turn them red.
- AC#1: `test_the_saved_default_leads_its_group_though_saved_last[provider-table|new-chat-default]` (saved last of five, it leads; `chat_defaults` outranks the table), `test_the_saved_default_follows_this_chats_model_or_is_it[another-model|the-default]`, `test_chat_settings_model_picker_lists_the_same_order` (NO-3: the same row keys as Alt+M, the default second behind the draft's model), `test_empty_find_with_no_previous_or_current_row_highlights_the_lifted_default` (D-c: highlighted, painted with the glyph, Enter applies it).
- The probe rule: `test_a_local_servers_listing_orders_the_default_and_never_adds_it[lists-the-default-4th|leaves-it-out]`: while the probe is held, Ollama's saved rows lead with the default; after the listing answers, a default listed 4th leads within the listing, and an unlisted one is in no row and not in `_models_for`.
- AC#2: `test_a_lifted_default_lists_the_same_models_and_the_more_count` (three rows plus the hidden ones are the saved list, the painted "… 2 more Anthropic models", `_models_for` in saved order) and `test_a_saved_default_listed_under_recent_is_not_listed_again`.
- AC#3: `test_a_shipped_sample_id_default_is_never_lifted[saved-list|served]` (TabbyAPI's template default `tabby-model`, seeded past the loader in its list, or listed; Anthropic's real default lifted in the same open).
- AC#4: `test_find_ranks_its_matches_as_before[exact-name|prefix-tie]` ('anthropic ' and 'claude': the lift runs untyped, then the typed rows and the highlight are the saved order).
- AC#5: the docs commit.

**Negative controls** (temporary edits of console_model_popover.py, the code the private-profile child imports, each restored with `git checkout` and checked clean; every red an assertion): the current provider always puts the default second (`[the-default]` and the D-c case red); lifting from `models` instead of `visible` (the RECENT case red); reordering `models` before the typed branch (both Find cases red); reordering `_models_for` (the AC#2 case, both Find cases and the listed probe case red); no sample-id skip (both AC#3 cases red); adding a default the rows lack (the RECENT case and the unlisted probe case red); gating the reorder on Alt+M only (the pick-mode case red).

**Pinned tests (the plan's T3 list): none rewritten.** test_console_model_switcher.py `test_switcher_lists_four_groups_of_one_line_pairs`, `test_typed_rows_are_headed_as_matches_not_top_3_each`, `test_a_ready_returned_provider_lists_the_model_just_saved` (the Settings-return case: `_capped`'s first-row keep and `_returned_saved_model`, typed path, untouched); test_console_provider_return.py `test_a_model_saved_in_the_default_model_field_is_the_reopened_highlight`; test_console_switcher_local_probe.py (`SHIPPED_MODELS`, and `test_a_configured_default_that_refuses_still_reads_refused` with `tabby-model` in `chat_defaults`, a NEEDS SETUP row); Tests/Chat/test_model_switcher_recents.py. They pass unchanged with no plugin: in the covering run below (xdist files mode) all 230 tests of these four files passed at the head and at 1ca320e4bb, each named node among them. Only a lone node run, with no earlier test in its process, failed at setup here with RecoveryRequired (raw_source_selection_changed); a run with a diagnosis-only plugin that imports the app at collection gave the same 230 at both, so it was not needed. In CI test_console_model_switcher.py runs whole in the census.

**Deviation from the plan's test placement.** The plan put the new nodes in test_console_model_switcher.py. That file is a whole-file census entry, and the checker refuses a node id beside its own file, so one node per function over the shards needs a file of its own. Review round 1 corrected that one fact in the plan (its New tests entry for TASK-35012 now names test_console_switcher_saved_default.py); nothing else in the plan changed. Plan item 8 (a Settings return) is the unchanged P8 node above, not a new one.

**Lanes.** The nine functions are census node ids after T2's (no `bootstrap_profile`, so the UI census; ledger R11 holds), `MINIMUM_FILES` 387 -> 396 with a dated comment, and the S1 comment restates the invariant (tail positions i, i+10, i+20 and i+30 share a shard) and the groups' measured times. Each function 3 of 3 serial in the lane's install set, junit seconds: leads 4.8-4.9, follows 4.8-4.9, RECENT 2.4-2.5, AC#2 2.4-2.5, Find 6.6-6.7, samples 4.9, pick mode 2.7-2.8, D-c 2.6, probe 4.9 (36.9 s together). The five-case position-rule function first measured 11.9-12.1 s, so it became three functions. Per shard at today's tail (1-10): +4.9, +4.9, +2.5, +2.8, +4.9, +2.5, 0, +4.9, +2.6, +6.7 s, the heaviest on the lightest group the nine tail positions reach. Each of the ten S1 groups re-timed as its own serial invocation, 3 of 3, lane install set: shards 1-10 junit 27.3-28.6, 27.1-28.8, 26.3-27.3, 26.0-26.9, 27.2-28.0, 25.7-26.0, 21.9-24.3 (unchanged), 27.2-30.8, 25.3-28.7, 29.0-30.0 s (pytest session 23.6-32.8 s), about 1.1-1.3 min each on CI at 2.6x. **NO-2 (b):** draft PR #3036 touches .github/workflows/derived-artifacts.yml, so no workflow file and no `FAST_LANE_TARGETS` edit. Tests/CI: 1015 passed, 1 skipped.

**Covering run** (45 files: the plan's T3 files and every test file naming console_model_popover, ConsoleModelPopover, `_build_rows`, `SHIPPED_SAMPLE_MODEL_IDS` or console.md, untruncated grep; xdist files mode) at the head and at a base worktree of 1ca320e4bb (44 files, without the new one), compared by failure name: 319 vs 318, the known local RecoveryRequired wall on both sides, and no base-only name. Head-only: `Tests/Chat/test_console_session_settings.py::test_settings_active_compaction_close_anyway_keeps_provider_work_running_and_reopens_fresh` ("Summary for original-memory." where fresh-memory was expected after reopening Console settings); re-run serially, 30 runs each side, it failed 6 of 30 at the head and 3 of 30 at base with the same assertion, a pre-existing race, and it never opens Switch model (only model_switcher.py builds the popover). One shared failure's message differed: `Tests/UI/test_console_endpoint_discovery.py::test_shared_evidence_change_refreshes_the_console_once` fails 5 of 5 serially at both trees, each tree showing both of its two assertions (lines 1008 and 1052), an idle-poll test that opens no switcher. The ProductionApp file named by the grep: 3 passed at both. Tests/Architecture size ratchets and the guarded-modal census serially: the same 8 failures at both (the 7 base-red module rows, and the census's three Console modals with the same message).

**Size rows.** No row changed. settings_screen.py is untouched (35,746, on its row); console_model_popover.py 2,497 -> 2,519 lines and has no row, and no architecture test governs its size. Checks: preflight passes under a scratch HOME; check_markup_interpolation OK (541 pinned); ruff adds no finding on the touched .py files (the popover's 3 and the census checker's 2 are the base's), and ruff format is clean on the new file; Tests/Performance boot census and pre-import payload: 5 passed.

**Files.** tldw_chatbook/Widgets/Console/console_model_popover.py; Tests/UI/test_console_switcher_saved_default.py (new); scripts/ui_pr_gate_census.txt, scripts/check_ui_pr_gate_census.py; Docs/User_Guide/console.md (the READY PROVIDERS bullet names both surfaces, and the listing paragraph says an answered server keeps the order when it listed the default); the S1 plan (one fact, review round 1).

**Review round 1** (range 1ca320e4bb..3308147e9a, "With fixes": the code is correct, three documentation fixes). (1) console.md said READY PROVIDERS lists this chat's model first; when RECENT already lists this chat's pair (usual for a used chat; PREVIOUS never lists it) the pair is left out of READY (`shown`), and the saved default leads. The bullet now says the group never repeats a pair PREVIOUS or RECENT lists, and this chat's model leads only when RECENT does not list it; the listing paragraph's sentence says the same of an answered server's rows. P9's older "this chat's own model stays listed first" (the listing-bound sentence) is left as it was. (2) The pinned-tests paragraph above said their files fail at setup here; the covering run passed all 230 of their tests at both trees, and the paragraph now says so. (3) The plan names the new test file (above).
<!-- SECTION:NOTES:END -->
