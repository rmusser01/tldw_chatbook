# Pending draft follow-ups: TASK-36454 through TASK-36458

Status: all five implementation tasks complete with targeted acceptance evidence and independent Superpowers approval. Final hosted delivery results are recorded in [PR #3228](https://github.com/rmusser01/tldw_chatbook/pull/3228), whose live check rollup identifies the exact tested head.

## Scope and ownership

| Task | Repair | Existing decisions |
| --- | --- | --- |
| TASK-36454 | Prompt Save adopts committed identity/version, keeps newer raw metadata and Basic/Advanced blocks dirty, and keeps the next Save on that record | ADR-033 application state, ADR-040/049/076/086 |
| TASK-36455 | Dictionary Settings acknowledges submitted raw controls without reloading over a newer draft | ADR-033 application state, ADR-046/079/115 |
| TASK-36456 | First Basic block mounts re-read both live lanes after structural awaits, including clear and cross-lane input | ADR-033 application state, ADR-040/076 |
| TASK-36457 | Theme Save retains later raw palette/name/dark edits and protects the Saved-message handoff; actual saved-file identity remains separate | ADR-011, both applicable ADR-033 records, ADR-150/161 |
| TASK-36458 | Parakeet completion may finish consented activation/preference but selects the batch only when its original form and live controls still match | ADR-014/025/050/033 application state/011 |

The [audit](2026-10-23-textual-book-after-pr3219.md) and each task retain the existing-task/ADR links and source-bound findings. No new ADR: these repairs preserve the existing draft, persistence, artifact, consent and navigation boundaries. The [implementation plan](../plans/2026-10-23-textual-pending-draft-followups.md) records scope and review focus.

`prompt_save_ack.py` and `parakeet_install_selection.py` are narrow presentation helpers for existing owners. The former extracts the old post-save completion; the latter extracts the old install-result callback while preserving the original UI-thread delegate. Dictionary completion joins the existing Persona completion helper. The three affected size ceilings decrease to their actual measurements: Prompt controller 5,100, Library screen 37,493 after the current-dev integration below (previous checkpoint 37,525), Personas screen 16,554. No unrelated ceiling was raised.

## Regression evidence

Tests use enabled mounted child controls and explicitly witness accepted input before releasing held work. Prompt and Dictionary writes reach real SQLite services; Theme writes real TOML. Parakeet retains the actual install button, consent modal, thread worker and original UI publication, with dependency availability and external acquisition/preference seams qualified. It proves UI selection ownership, not downloads, package installation or inference.

| New regression module | Exact cases in required PR census |
| --- | ---: |
| `Tests/UI/test_library_prompt_pending_drafts.py` | 22 |
| `Tests/UI/test_dictionary_settings_pending_save_drafts.py` | 13 |
| `Tests/UI/test_theme_pending_save_drafts.py` | 13 |
| `Tests/UI/test_parakeet_install_selection_retention.py` | 10 |
| Total | 58 |

The original 56 have passing local coverage across the relevant selections, with iterations qualified below; the two aggregate follow-up controls are recorded separately at the end. They are appended as exact parameterized node IDs: all 664 inherited selectors remain, floor 722, now 17 shards, and unchanged time limits. All 56 new positions remain fixed through both measured CI follow-ups below; the capacity layout changes thirty inherited positions. The two aggregate follow-up controls append after all 720 positions, leaving their order unchanged. The required CI selection includes the full new set on the PR head; collection alone is not counted as execution.

Qualified RED controls reproduce later-key/clear/cross-lane loss in both Basic lanes; create/update false-clean acknowledgement and Advanced clobber; held older semantic publication losing actual `xy`; Dictionary name/description/raw budget/strategy/enabled clobber; Theme palette/name/dark loss and its delayed Saved handoff; Parakeet provider, precision and folder replacement. Matched ordinary and definite-failure controls distinguish normal successful completion and retry.

Prompt local evidence: first-mount final selection 8/8; corrected queued-body selection 2/2; create final selection passes ordinary/body/metadata, with its older queued-body fixture result superseded by the corrected 2/2 run; other update, failure and held-semantic cases pass. The first acknowledgement extraction omitted the existing action-capability refresh after creation: the second-Save regression caught it, the existing refresh was restored, and the affected create controls passed. An earlier queued injector incorrectly added another edit during the second Save; limiting it to the intended first attempt fixed that test without a production change. Initial missing-request/wrong-import/wrong-private-seam failures are fixture setup failures, not product evidence.

Dictionary: 10/10 initial GREEN plus 6/6 supplemental final-source controls cover the 12 distinct cases, including a reverted edit, raw invalid budget, queued notification and actual guard Stay actions. Theme: all 13 final cases pass, including Save-as, next Save, invalid raw palette/name and delayed Saved publication. Parakeet: all 9 new cases and 25 existing contracts pass, including queued folder input, changed batch, ABA choice and ordinary override.

## Compatibility and static checks

- Prompt block-editor controls: 29 pass. Five existing Prompt service/reader journeys and six recompose guards pass with the repository-supported bootstrap-profile qualification.
- Dictionary: eight existing widget controls and 17 existing screen contracts pass; the latter use the same supported bootstrap-profile qualification.
- Theme: 17 of 18 existing controls and five Reset controls pass directly. The remaining Save-as test assumes catalog completion after a fixed pause; it also fails on pristine HEAD before the pending scan lands. A bounded final-source scan witness preserves every original assertion and passes, with the new file selected after the real scan. The existing test is unchanged.
- Gate contracts: 178 pass. Affected owner ratchets and Prompt wiring: 19 pass. Task ID/readability checks pass across 5,697 files. New helpers/tests pass Ruff and formatting; nine existing production owners have no new Ruff diagnostics compared with dev (existing findings are retained).
- The diagnostic inventory change is one existing sanitized warning moved from LibraryScreen to its helper, with the same text, error-type metadata and `module=LibraryScreen` binding. No draft content, path, secret or persistent sink is added.

Unqualified standalone legacy UI selections hit `RecoveryRequired: raw_source_selection_changed` before their behavior. The supported `bootstrap_profile` marker (private compatibility plugin) or normal collection import context qualifies them without changing production admission guards. Dictionary's representative failure is reproduced with pristine Git HEAD module bytes. Parakeet's final 25-contract confirmation passes with normal collection context. These setup failures remain in the raw receipts.

A wider size-only check found nine unrelated baseline overages in the ingest controller, MCP workbench, change-review/LLM screens, Console transcript, app command providers/ingest queue/lifecycle, and API client. Each source file is byte-identical to dev; they were not repaired or repinned here. The affected Prompt-controller slack is repaired by lowering its own pin. The initial broad UI recompose setup errors are superseded by the six qualified passing guards above.

## Reproduction and delivery

The four new modules are independently runnable via targeted pytest; disposable private-profile wrappers are part of the repository test harness. Required PR selection is generated by `scripts/check_ui_pr_gate_census.py --shard <zero-based-index> 17`. No full local test sweep was run.

Original native commands, source snapshots, full logs and JUnit receipts are preserved under `.test-tmp/task-36360-36363/` and `.test-tmp/task-36454-36458/`; they are private working evidence and are not committed. The public tests, task notes, required CI logs and this report are the durable review surface.

Superpowers review of `7103ed34` found one Important/P1 issue: a two-lane Basic capture refreshed only the triggering Advanced card, leaving an already populated other lane stale. Switching to Advanced could then save its older text. The analogous Save-acknowledgement convergence used a blank block ID and refreshed no card. Both convergence loops now refresh every captured block, then recapture after structural awaits.

Three additional mounted controls reproduce the actual stale Advanced TextArea before the repair and pass afterward (3/3, 104.872 seconds), including both lane directions, stable child/block identity, mode-switch capture and real SQLite Save/version checks. The acknowledgement control holds actual Changed notifications at child publication while allowing DOM lifecycle messages; it does not claim a naturally occurring race frequency. Initial fixture attempts paused empty lanes, waited on a two-key Pilot sequence behind a held parent, or blocked lifecycle work with the canvas pump; those setup/timeout failures are excluded. Their receipts remain private. The original 717 census slots remain unchanged, with these three appended.

The focused Superpowers follow-up approved `8a6ca765` with no remaining Critical, Important or Minor findings and no behaviors excluded. The 11 affected neighboring Prompt controls pass on the repaired convergence code (333.007 seconds), including both queued-body Save paths and the held older semantic message. The 178 gate contracts pass again with the 720-entry census.

The first hosted Perf Guard exposed an eager import introduced by the Parakeet helper extraction: screen preloading added 559 modules against the unchanged 558 limit. The existing guard reproduces locally (1 exact failure, 17.803 seconds). The helper now loads through scoped request/completion imports; its internal capture function is named `capture_selection` within that module. Both preimport/warmup controls pass afterward (2/2, 18.818 seconds): 558/558 preimport modules and 94/94 warmup modules. At that checkpoint the LibraryScreen ceiling and actual size remained 37,525; no size or performance limit rises. ADR-097 governs this import timing repair. An explicit F401-only lint selection surfaced legacy unused re-exports; the final normal-rule comparison against dev still finds no new diagnostics in the nine affected existing owners.

After import deferral, the mounted ordinary/provider-change Parakeet controls and both exact screen-size guards pass (4/4, 66.956 seconds). The scoped Superpowers review approved the import-timing follow-up. After rebasing onto dev `5b5920b8`, the reviewer verified that all 11 changed production files were byte-identical to the reviewed implementation; upstream test/lesson updates were preserved. Native preimport/warmup confirmation passed again (2/2, 18.841 seconds, 558/558 and 93/94).

## Hosted CI follow-up

[Perf Guard](https://github.com/rmusser01/tldw_chatbook/actions/runs/40020015425) passed on rebased implementation head `6c0d5255cd`. The corresponding [Derived Artifacts run](https://github.com/rmusser01/tldw_chatbook/actions/runs/40020015317) found one Dictionary fixture readiness race in UI shard 3 (168 passed, one failed): the fixture selected an old Character row before the asynchronous Dictionary list replacement finished. It failed before Save or pending-draft input began. The selected numeric ID alone could not distinguish Character 1 from Dictionary 1.

The fixture now waits for the exact kind-qualified Dictionary row, highlights its existing kind/identity, and requires selected kind, ID and mounted detail. All 12 case bodies and parameter IDs remain unchanged; no timeout, census entry or production contract changes. The formerly failing reverted-name case and ordinary, queued-name and raw-name/Stay guard controls pass (4/4, 123.997 seconds). Scoped Ruff, formatting and diff checks pass. Independent Superpowers review approved this fixture-only repair with no findings; it preserves actual selection dispatch, test bodies and guard assertions. Required current-head CI remains the final delivery check.

Task statuses were set to Done through the Backlog CLI after acceptance, targeted verification, documentation and independent review completed. Final task/documentation metadata is bundled before the delivery CI run, so its result covers the complete submission. Current-head hosted checks remain the delivery gate; their final run links and review disposition are recorded in the PR description rather than creating another metadata-only CI cycle. The final backlog hygiene checks cover 5,700 readable task files with no duplicate IDs or incompatible paths.


### Historical two-slot UI census rebalance (failed hosted qualification)

The complete `a7129eabea` [Derived Artifacts attempt 1](https://github.com/rmusser01/tldw_chatbook/actions/runs/40029392931/attempts/1) passed eighteen prerequisite jobs; UI shard 3 passed all 169 tests in 1150.31s and was cancelled at the unchanged 20-minute job cap during shutdown. One bounded same-head retry reached 98% and hit that cap again. Neither cancellation is a hosted pass. Perf Guard passed separately. No further unchanged retry was attempted.

The workflow explicitly calls for timing replay and census reordering when a shard nears its cap. Existing [ADR-103](../../../backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md), [TASK-35834](../../../backlog/tasks/task-35834%20-%20UI-Fast-Lane-job-9-runs-within-a-minute-of-its-20-minute-cap-on-dev.md) and [TASK-33664.4](../../../backlog/tasks/task-33664.4%20-%20Keep-the-gated-UI-census-within-its-existing-job-cap.md) govern capacity tuning; this follow-up adds no new gate policy. Read-only native MCP probes passed two unchanged small/large cases. Tool-name typing took only 0.99/1.39s across both passes; duplicate preview waits cost 3.41s in the small case and zero in the large. No robust test-body saving was established, so all six MCP parameter cells, keys and assertions are preserved.

Replay uses all thirteen complete attempt-1 shard logs and the existing `scripts/ui_lane_shard_replay.py` file/node attribution. Only inherited slots 77 and 432 exchange their whole-file selectors: `Tests/UI/test_destination_rail_row.py` moves from shard 12 to 3 (94.06s measured), and `Tests/UI/test_mcp_permissions_handoff.py` moves from 3 to 12 (221.69s). All 720 unique selectors remain, 662 of 664 inherited positions and all 56 appended positions stay fixed. No source test, dependency, workflow, shard count or timeout changes.

| Timing basis | Shard 3 before | Shard 3 after estimate | Shard 12 before | Shard 12 after estimate |
| --- | ---: | ---: | ---: | ---: |
| Replay attributed seconds | 1107.73 | 980.10 | 832.38 | 960.00 |
| Pytest seconds, calibrated to actual prior summaries | 1150.31 | 1022.69 | 866.93 | 994.55 |

The replay's largest attributed load falls from 1107.73s to 1024.69s (unchanged shard 5). Calibration retains each shard's observed collection/reporting overhead; runner setup and shutdown still consume job time. These are single-run estimates, not a variance guarantee or hosted qualification. Fresh exact-head CI remains required. The 20-minute job cap, 180-second test timeout, thirteen serial shards, minimal dependency set and fail-closed aggregate remain unchanged.

The four targeted census/replay/shard-count/workflow modules pass (136 tests, zero failures/errors/skips, 32.287s JUnit). All thirteen real checker subprocesses produce exactly their ordered, disjoint census subsequences; the combined multiset is the unchanged 720-entry set. The normal census validator, scoped checker Ruff and whitespace checks pass. Product/test bodies are unchanged by this repair. Scoped Superpowers review and fresh hosted CI qualify the resulting commit through the PR delivery record.


### Final capacity layout and current-dev integration

The two-slot attempt on `d4415b7c40` did not qualify in [Derived Artifacts](https://github.com/rmusser01/tldw_chatbook/actions/runs/40046371047): UI shard 3 reached 99% without a recorded assertion failure, then hit the unchanged 20-minute job cap. The other eighteen prerequisite jobs passed, including all twelve other UI shards; [Perf Guard](https://github.com/rmusser01/tldw_chatbook/actions/runs/40046371066) passed. The destination-rail file took 169.42s versus the earlier 94.06s, and Settings Connect rose from about 312s to 345.74s. The receiver shard passed 177 tests in 873.58s. The existing replay agrees with reported duration rows; no replay implementation bug or safe test-body shortcut was established. No further unchanged retry was attempted.

The final replay combines all thirteen original complete logs with all thirteen newer logs (twelve complete and the canceled partial shard), taking each file/node at its worst observed attributed cost. Unfinished work retains prior measured cost. Total modeled load is 12,997.67s: even ideal 16-way balance exceeds the established 1020s target after adding 15% variance and 110s setup. Seventeen is the smallest feasible count under that model. At seventeen the original ordering has a 1051.19s maximum; sixteen paired swaps reduce the maximum to 773.77s, projecting `773.77 * 1.15 + 110 = 999.83s`. This estimate is not a guarantee or a hosted pass. Full raw logs, exact durations, hashes and selected model are retained privately.

All 720 unique selectors remain. The first 181 positions (including the existing P0 block) and last 104 positions (including all 56 new cases) are fixed. Thirty distinct inherited positions change; the following final mapping uses one-based positions from the original layout. Comments travel with selectors and historical block-grouping prose is marked as insertion history.

| Original position | Final position | Selector |
| ---: | ---: | --- |
| 185 | 549 | `Tests/UI/test_settings_personal_context.py::test_production_settings_inspects_private_and_deleted_metadata[size0]` |
| 187 | 268 | `Tests/UI/test_settings_personal_context.py::test_export_and_deleted_labels_keep_literal_metadata` |
| 197 | 432 | `Tests/UI/test_settings_default_model_picker.py` |
| 198 | 221 | `Tests/UI/test_settings_connect_rows.py` |
| 199 | 200 | `Tests/UI/test_settings_connect_key_rows.py` |
| 200 | 199 | `Tests/UI/test_settings_context_window_staged_reset.py` |
| 201 | 471 | `Tests/UI/test_console_settings_disclosures.py` |
| 202 | 431 | `Tests/UI/test_settings_console_fallback_rows.py` |
| 203 | 202 | `Tests/UI/test_settings_advanced_disclosures.py` |
| 211 | 311 | `Tests/UI/test_settings_provider_save_scope.py` |
| 214 | 294 | `Tests/UI/test_settings_openai_reconnect_reread.py` |
| 221 | 198 | `Tests/UI/test_settings_blank_model_card_rebuild.py` |
| 259 | 354 | `Tests/UI/test_settings_save_commit_models.py::test_model_catalog_controls_are_labeled_and_visually_separated` |
| 268 | 187 | `Tests/UI/test_settings_theme_picker_screen.py::test_quit_with_staged_changes_never_stacks_on_the_theme_prompt` |
| 294 | 214 | `Tests/UI/test_console_provider_return.py::test_return_to_a_closed_chat_stays_put_with_a_notice` |
| 311 | 211 | `Tests/UI/test_console_local_server_discovery_card.py::test_setup_modal_first_focus_is_the_detected_action_when_one_is_offered` |
| 353 | 538 | `Tests/UI/test_settings_image_gen_panel.py::test_backend_row_controls_visible_and_clickable_under_real_css` |
| 354 | 259 | `Tests/UI/test_console_settings_unsaved_guard.py::test_credential_round_trip_keeps_the_restored_edit_unsaved` |
| 361 | 364 | `Tests/UI/test_console_transcript_poll_stop_removal.py` |
| 364 | 361 | `Tests/UI/test_settings_saved_lists_follow_file.py::test_config_reload_reads_an_in_place_edit_into_the_saved_lists` |
| 365 | 443 | `Tests/UI/test_settings_pick_joins_saved_list.py::test_s_saves_the_default_and_says_why_a_pick_did_not_simply_join` |
| 431 | 203 | `Tests/UI/test_settings_tools_hub.py` |
| 432 | 197 | `Tests/UI/test_mcp_permissions_handoff.py` |
| 443 | 365 | `Tests/UI/test_screen_footer_hints.py::test_built_bundle_carries_the_footer_rules` |
| 458 | 608 | `Tests/UI/test_screen_navigation.py::test_prompts_route_resolves_to_library_screen` |
| 471 | 605 | `Tests/UI/test_screen_navigation.py::test_ctrl9_is_tools_compatibility_binding` |
| 538 | 353 | `Tests/UI/test_screen_navigation.py::test_screen_load_error_handles_unknown_route` |
| 549 | 185 | `Tests/UI/test_schedules_bulk_admission.py::test_pending_batch_refuses_mark_and_clear_gestures` |
| 605 | 201 | `Tests/UI/test_console_environment_controller.py` |
| 608 | 458 | `Tests/UI/test_library_media_editor_dirty_escape.py` |

Seventeen serial UI shards request four more jobs; together with the six other prerequisites, 23 test jobs are eligible, or 24 including Perf Guard. This does not guarantee concurrent runner availability. Minimal dependencies, round-robin selection, all test bodies/assertions, the 20-minute job cap, 180-second test timeout and fail-closed aggregation are unchanged. Existing ADR-103 and TASK-35834/33664.4 govern the adjustment. Numeric count governance now handles the resulting totals above twenty; its RED control failed four cases (three eager word-table IndexErrors and the old explicit matrix size), then the twelve affected checks passed after the numeric guard/matrix update.

PR #3221 merged related Auto transcription defaults while CI queued. Rebase onto dev `d0d926e11b` retained its displayed-model/submission binding, preference snapshot and external validation. The combined screen grew to 37,551 lines against the unchanged 37,525 ceiling; the existing size guard reproduced that failure. Moving the complete existing pure snapshot projection into the already imported `Library/library_ingest_state.py` retains the screen delegate and all Auto/Parakeet behavior. Whole-screen AST changes are limited to the import/delegate; prior state nodes are unchanged; the moved body is identical modulo the form argument and local binding rename needed to avoid shadowing `dataclasses.field`. No dependency, consent, persistence, service or runtime boundary changes. Existing ADR-025/033/097 apply.

Final native integration selection passes 80 cases (zero failures/errors/skips; JUnit 101.251s): exact size pins, existing snapshot/submission contracts, 66 Auto/readiness controls, ordinary/provider-change mounted Parakeet controls and both boot guards. LibraryScreen measures 37,493 lines/1,354 methods; its line ceiling is lowered to 37,493. Preimport is 558/558 modules and warmup 93/94; no budget rises. Normal-rule Ruff comparison retains the same 63 inherited diagnostics with zero additions/removals, and the same two inherited unformatted owners remain; the ratchet file is formatted. No full local suite was run. Final scoped review, gate-contract results and exact-head hosted run links are recorded in the PR delivery record.

Before the final comment corrections, the four targeted census/replay/count/workflow modules passed 140 cases (zero failures/errors/skips; JUnit 39.004s). All seventeen real checker subprocesses produce their nonempty, ordered, disjoint subsequences, covering the 720 selectors exactly once. Those corrections changed no selector or executable AST, but the independent review subsequently found the parsed-header regression below; that earlier pass does not qualify the later header. Independent production AST/diagnostic proof still passes.

Final count/prose/task-ID controls pass 14 cases (JUnit 2.770s); the normal census validator passes. Scoped normal-rule Ruff passes for the three changed CI Python files. The count-governance file is formatted; the workflow-contract file retains the same pre-existing formatter failure as its untouched HEAD copy. Whitespace checks pass.

Scoped Superpowers review of `00a7da51` reproduced one Important issue: the final historical-comment edit removed the `the 92` suffix parsed by the existing replay-boundary test. The native exact control reproduced the failure (1 failure, JUnit 1.128s). The header now retains its machine-readable suffix and explains the historical grouping separately. All selectors, model values, tests and assertions remain unchanged. The complete affected CI selections are rerun after this repair; the fresh follow-up review and exact-head CI results are recorded in the PR.

After restoring the parsed historical header, the complete four-module selection passes again: 140 passed, zero failures/errors/skips, JUnit 39.447s. This receipt qualifies the final census text and all seventeen real checker outputs; no later census, selector or executable-source edit is included in this documentation update.


### Required-aggregate completion-safety repairs

On `30a9aa0c`, all 23 prerequisites and Perf Guard passed; all 17 UI logs prove the original 56 new cases executed exactly once with positive call durations. UI jobs ranged from 696s to 945s (11m36s to 15m45s), below the unchanged 20-minute cap and the 1020s qualification target. These passing components do not qualify the complete submission: the required aggregate found two deterministic owned failures, a new W002 post-await Library-pane lookup in `finish_dictionary_settings_save` and unsafe runtime interpolation in the relocated Parakeet install-error notification. Both exact default native guards reproduce exit1; raw aggregate and native logs are retained.

The Dictionary completion now queries the current pane without raising and ends presentation refresh if the pane disappeared. The real SQLite Save and its identity/version/raw-draft acknowledgement are unchanged. The install-error notification now renders text literally, retaining the original sanitized message, severity, consent and artifact/preference paths. Only the obsolete old-wrapper W002 pin (two lookups) and markup pin (one interpolation) are removed; no new unsafe site is allowed or globally re-pinned.

Two added controls witness real behavior. The Dictionary test presses the actual Save button, observes committed version2 while the real row renderer is held, removes the Library pane, then checks that completion and later typing remain usable without changing persisted version2. The notification test uses actual App.notify and Toast rendering for markup-shaped error text; it claims the publication boundary, not a download or install. Matched native RED has two expected failures (NoMatches and MarkupError; JUnit38.148s); both pass after the narrow repairs (JUnit34.707s). The earlier fixture attempt used is_mounted as an attachment check and left headless notifications disabled; those setup failures are retained and excluded. The qualified test instead witnesses the actual current query result and enables real notifications.

The two controls append after the complete 720-selector census, yielding 722 selectors and 58 new regression cases. Every prior position, the seventeen shards, dependencies and all caps remain. They land on jobs7 and8, whose observed full-job times were909s and913s; conservatively adding 2.6 times their native call durations (22.26s/7.97s, including private-profile child startup) projects966.88s/933.72s, below1020s. This is an addition estimate requiring fresh final-head hosted qualification, not a guarantee. Existing ADR-033/011/025/097 govern this bounded repair. Targeted compatibility, all required-aggregate commands and fresh scoped Superpowers review qualify the next publication; final results are recorded in the PR delivery description.

Final targeted selection after these fixes passes213 cases (zero failures/errors/skips; JUnit220.514s): worker/markup negative controls, all four CI-contract modules, seven mounted completion/compatibility cells, boot guards and exact Library/Persona size ratchets. The two original UI modules retain every existing function body and parameter ID byte-for-byte at AST level; only the two new functions append. All original720 census positions remain identical. No full local suite was run.

The exact required-aggregate command batch passes eleven checks on frozen owned source, including both originally failing guards, diagnostic inventory, all path/schema/index/timestamp/backlog checks, stylesheets and the722-selector census. Canvas reproduction alone refuses the local3.12.10 interpreter because its existing manifest pins3.12.11; the same unchanged builder/assets passed on the earlier30a9 CI run with3.12.11 and remain a required fresh-head hosted check. No pin, source guard, artifact or dependency is relaxed to bypass that environment precondition. All command outputs and before/after source hashes are retained.

Normal-rule Ruff passes the other three changed helper/test files and the census checker. The existing Persona helper retains its one E721 diagnostic in the untouched profile finisher (exact integer-type check); baseline/current diagnostics are identical and the new completion code introduces none. All four helper/test files pass formatting and the complete diff passes whitespace checks. Fresh scoped review and the final-head delivery record follow the rebase; existing ADR links remain.

Publication checkpoint after aggregate repairs: clean rebase onto dev `1381e70735` preserves production and script bytes from the qualified repair. The five affected CI-contract modules, including incoming queue-pressure governance, pass224 cases (zero failures/errors/skips; JUnit37.453s). Fresh scoped Superpowers review of `e9692090` found no Critical, Important or Minor findings and approved publication for fresh CI, with hosted Canvas and all17 UI/aggregate results explicitly outstanding. TASK-36455 and TASK-36458 are closed through the CLI after that review; all five acceptance sets are complete. The final closure commit changes documentation/task metadata only, before the next normal pull-request CI cycle. Existing ADR links and all guard policies remain. Final exact-head CI and feedback results are recorded in PR #3228 without a post-CI metadata commit.

### Live-dev B2a integration qualification

The completed c3d727a95e run passed all25 main job results and Perf Guard, with58 new cases verified in the accepted17 passing UI logs (maximum943s). That evidence is preserved for its exact head. The live dev ref then showed PR #3229/TASK-33910.3 at ae85e59e8f,25 commits ahead of the recorded PR base SHA. Rebase preserves B2a's list/window/mark/visit ownership, workflow target, diagnostic inventory, CSS and its same-position census replacement of the removed paging module. The only conflict was the Personas line ratchet: preserve the incoming16885 decision/comment and lower it to the combined measured16878, the same seven-line completion extraction.

The old Dictionary fixture's ordinary control fails before migration because B2a no longer mounts its ListView row widget (TimeoutError; JUnit39.596s). The fixture now waits for kind-qualified `dictionary:<id>` membership and selects it through the existing B2a harness helper's real OptionList action. All original test bodies, decorators, parameter IDs, child witnesses and persistence/guard assertions are AST-identical; only `_dictionary_app` and imports change. The Dictionary handler/completion are AST-identical to the previously reviewed head; nine other owned production files are byte-identical. All58 regression positions remain; the722-selector census has exactly one inherited replacement at index124, from the retired paging module to B2a's pure list tests.

Qualified native integration selection:387 passed, zero failures/errors/skips, JUnit125.735s (two mounted Dictionary controls,155 current list cases,224 CI/census/replay/queue contracts, both boot guards and four exact screen-size controls). The other11 Dictionary cases then pass with zero failures/errors/skips, JUnit254.730s; the two previously qualified cases were explicitly deselected. Together these receipts qualify all13 Dictionary cases without altering their acceptance assertions. The first integration command aborted collection at zero cases because a module-ratchet selector was misspelled; that receipt is excluded, and the corrected complete387-case command supplies the evidence. No full local suite was run.

All exact native aggregate commands ran on frozen combined owned source. Eleven pass, including CSS reproduction, diagnostic/path/schema/index/timestamp/backlog, both safety guards and the current census. Canvas alone rejects local Python3.12.10 against its unchanged3.12.11 manifest pin; no builder, artifact or pin is changed, and fresh hosted qualification remains mandatory. The changed fixture and ratchet pass normal-rule Ruff and formatting; whitespace checks pass. Fresh scoped Superpowers review and final-head hosted CI remain the next publication gates; previous c3d results do not qualify the new combined source.

B2a publication checkpoint: fresh scoped Superpowers review of8e06b3bb found no Critical, Important or Minor issues and approved publication for fresh CI. The read-only reviewer independently verified the rebase/range-diff, all13 Dictionary receipts, unchanged acceptance ASTs/58 positions, lowered16878 ceiling and11/12 native guard limitation. TASK-36455 is closed through the CLI after review. This closure changes metadata only; new-head hosted qualification remains mandatory and will be recorded in PR #3228.


### PR #3231 census integration (2026-10-24)

ADR required: no
ADR path: backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md
Reason: reconcile the existing 17-shard capacity qualification with TASK-36512's incoming split-file coverage, keeping every gate and timeout.

Live dev `f5585b020d` includes PR #3231/TASK-36512: four heavy whole-file entries are split by function/case, with completeness guards. All 704 incoming selectors and all 58 PR regressions are retained, exactly once (762 selectors). The two existing timing blocks, Phase 7 (zero-based195,92 entries) and the tail (zero-based664,98 entries), are re-dealt for17 shards; every one of the572 outside positions stays fixed. No test, assertion, dependency, job cap or per-test timeout is removed or relaxed. Production source is unchanged from reviewed `cd76774780`.

A literal append to dev's13-shard ordering predicted an over-cap17-shard job. The existing replay's file/node attribution and deterministic pair swaps re-deal only those blocks. Costs take the worst per-selector observation from the complete qualified17-shard runs40072763102 and40086080161, plus the measured new upstream module costs from successful run40095575123 (29.20s Import,7.26s notes skips,27.60s focus). The29 swaps give a789.77s maximum pytest-work estimate;110s setup and15% variance give1018.24s, below the1020s planning target and unchanged1200s cap. This is a model, not a hosted pass. Fresh current-head CI remains required.

TASK-36512's incoming completeness guards remain intact; its three-run observation criterion is not claimed complete. TASK-36514 separately proposes15 shards and a committed full-deal tool; this integration retains the already qualified17-shard count and does not mark that task complete. Reassess its count-specific proposal against the final merged matrix rather than duplicating this coverage work.

Targeted integration verification: all228 CI/census/replay/workflow/queue checks pass on the final combined census. Three intermediate shard-description assertions were red until the17-shard metadata was restored; no tests were changed to obtain the green. Checker Ruff and whitespace checks pass.
