# Task16 source-specific integration report

**Status: NEEDS_CONTEXT — integration and selected checks are frozen; two inherited reader failures require root scope amendment before repair.**

Actual HEAD: `749e7540affab2b543f472fcf6fae6bff34d8122`. Starting metadata BASE: `6cd44c72ffef6d19fca21eab3a05bdcade7c1811`. Qualified feature source: `eb7cb871390616de22d28c7a6679bbc9202841d7`. Selected latest dev: `a78a9a900b4901e33c031f830dd2d80224d5147d`; previous integrated dev: `49206beea90d35ea9e6842ffa44e4b274db29d8a`.

ADR required: no new ADR. Existing ADRs 218 (merge queue), 103 (required aggregation) and 219 (Console bounded starts) apply. No new runtime, storage, authority, schema or UX boundary was selected.

## Integration and preservation

Pinned recovery ref `refs/recovery/pr2995-task16-6cd44c72` and verified bundle `/private/tmp/pr2995-task16-recovery-6cd44c72.bundle` before the sole rebase. The rebase replayed 87 feature/metadata commits automatically without conflicts or manual overlays. No empty source commit was created. Before the immutable comparison, current HEAD was additionally pinned at `refs/recovery/pr2995-task16-rebased-749e7540`.

All 25 incoming commits are ancestors. All 49 incoming paths match the exact selected hashes: 47 byte-identical selected-dev files and these two approved compositions:

- `Tests/Architecture/test_module_size_ratchet.py`: `3785fd302268080fb2f7e1e402140cf6f122c4b412646d7357e30d358bbd6041`.
- `backlog/docs/lessons-backlog-hygiene.md`: `d08faa68d7c4e790cbbb563acfdb6a5faee3dc6ad447dbe447a03c6f9662ddf9`.

All 132 protected feature source/test/script paths are exact. The before snapshot covered 29,987 tracked files; all 29,954 existing paths outside incoming ownership remain exact. Final union contains 30,003 files. The 1,983 historical feature QA paths retain their path/blob/SHA digest `ad92f9666141271109dde626bd96522d969f0dd366da9a0db931e492b3940d04`. No original assertions, markers, authority/native/Close/Session/physical-drain/strict-abort/recovery/import/worker/route guards were changed.

The original qualification ZIP retains 63,166,118 bytes and SHA256 `acecbdfb6ddbafe6df679c39f9f137f2bfb49a1f87b62556e1345021f8384c84`, carrying all 2,118 entries without decompression/export/replay. Task14/15/R1 manifests verify 69/119/46 rows respectively; deduplicated artifact paths plus exact manifests, original snapshots and root receipts form 193 frozen private pins. Root mutable coordination/progress/exporters were excluded. Original Task15 61-case, R1 27-case, original source revisions and all archival receipts remain unchanged.

Exact path/hash/byte union: `task-16-safe-evidence/source-union-and-carry.json`; before snapshots: `before-tracked.json`, `before-private-carry.json`, `before-qa-carry.json`, `before-original-zip.json`; post Git-object QA check: `after-qa-carry.json`.

## Actual selected test results

All commands used shared Python 3.12 (`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`), this managed worktree as cwd and PYTHONPATH, canonical private fixture/profile behavior, separate fresh pytest processes/basetemps, `-q -p no:randomly`, explicit selected IDs and unchanged configured 300s timeout. No broad cohort/full suite, install, blanket plugin, new skip/XFAIL, warning suppression or timeout/budget change occurred. Exact argv/source hashes/stdout/stderr/log/XML/exit/elapsed and actual IDs are frozen under each group prefix.

| Group | Source inference | Actual result | Exit |
|---|---:|---|---:|
| `runtime_intersection` | 25 | 25 cases: {'failure': 2, 'passed': 23} | 1 |
| `incoming_ci_contracts` | 21 | 24 cases: {'passed': 24} | 0 |
| `exact_module_ratchets` | 8 | 8 cases: {'passed': 8} | 0 |
| `changed_preimport_loc_budget` | 1 | 1 cases: {'passed': 1} | 0 |

**58 actual selected cases: 56 passed and two failed.** CI has 24 actual cases rather than the proposed 21: `test_required_aggregation_contract_rejects_continue_on_error` parametrizes four cases. Exact selectors stayed unchanged; no inferred 55-case completion claim is made. No 23-passing-runtime replay occurred.

The one loading case measured **557 modules, 415370 total LOC and 127527 LOC for fattest route library**, within exact limits 557/425347/135111. Its `UserWarning` reports zero module headroom, 9977 total LOC headroom and 7584 fattest-route headroom and is retained. Runtime emits one retained `PytestUnhandledThreadExceptionWarning` from the typed-answer worker. Optional PyAudio diagnostic logging is retained. CI/module results have no pytest warning summary.

Initial harness setup attempt: all 25 runtime cases errored before bodies because `/private/tmp/pr2995-task16` parent was absent (`Path.mkdir(parents=False)`). Under root Ruling69, original argv/source/log/XML/result were copied byte-identically to `runtime_intersection-setup-error*`; only the missing private parent was created before exact rerun. No source or fixture changed.

TDD: not required for automatic upstream integration. No new production logic or tests were authored. The actual failures and immutable-base comparison provide RED evidence for the proposed canonical reader-profile repair; no repair/GREEN claim exists.

## Frozen failure attribution and smallest proposal

Under Ruling70, only these two failed nodes ran on immutable pre-rebase BASE6cd44, using the same cwd/environment/canonical fixtures/300s timeout and a fresh basetemp:

- `Tests/UI/test_console_ask_user_typed_answers.py::test_typed_answer_resolves_a_real_round_through_the_real_card_and_composer`: same `textual.css.query.NoMatches` for `#chat-question-card`. The worker traceback in both logs ends in `RecoveryRequired: raw_source_selection_changed` through `get_cli_setting` → config bootstrap → raw-participant selected-source admission before card publication.
- `Tests/UI/test_personas_workbench.py::TestPreviewIntegration::test_open_in_console_stages_preview_transcript`: same direct `RecoveryRequired: raw_source_selection_changed` and pre-mount library absence.

Both baseline nodes failed. Exact JUnit messages, direct exception class text/fixed reason and relevant test/shared-fixture/config/raw-participant/interrupt sources match; see `two-node-immutable-base-attribution.json`. These are inherited admission failures, not demonstrated regressions from Roleplay literal rendering. The temporary detached checkout was guarded with try/finally and restored to exact branch749e clean; full source/QA/ZIP/private preservation reverified afterward.

**Proposal only; not implemented:** add one `@pytest.mark.bootstrap_profile` decorator to each exact failed function. `Tests/conftest.py`1183–1188 already checks the closest per-node marker to retain the source-bound canonical collection profile. There is no fixture named bootstrap_profile. Existing Console composer reader tests use the same marker. Do not add a signature/fixture argument, modify shared fixtures, weaken admission or change production. Preserve each function body, signature, waits, assertions and every original marker byte/AST; preserve all unrelated functions.

The simulated proposal proves entire owner AST equality after removing only the added marker. Exact reversible two-line patch, current/proposed hashes, per-function body AST hashes, existing owner-format debt and minimal rerun selection are frozen in `proposed-exact-reader-bootstrap-marker.patch` and its JSON. Root must amend plan/Backlog scope before implementation; rerun only the two failed nodes, preserving current 23-pass runtime and CI24/module8/payload1 receipts.

## Incoming method/import authority and runtime carry

Every nonoverlap incoming file remains exact selected-dev bytes. Full before/after method AST maps and imports for the 18 incoming production owners are in `method-import-AST-carry.json`. Actual changed methods equal exactly the selected exception arrays. Incoming method/import exceptions are:

| Owner | Changed methods | Import change |
|---|---|---|
| `tldw_chatbook/UI/CCP_Modules/ccp_character_handler.py` | `CCPCharacterHandler._notify` | unchanged |
| `tldw_chatbook/UI/CCP_Modules/ccp_loading_indicators.py` | `LoadingManager.start_loading`, `with_loading` | unchanged |
| `tldw_chatbook/UI/CCP_Modules/ccp_persona_handler.py` | `CCPPersonaHandler._notify` | unchanged |
| `tldw_chatbook/UI/CCP_Modules/ccp_validation_decorators.py` | `validate_file_import`, `validate_input` | unchanged |
| `tldw_chatbook/UI/Navigation/buddy_management.py` | `BuddyManagementCoordinator._apply_and_report` | unchanged |
| `tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py` | `PersonasPreviewController._provider_label` | remove `from ...Utils.input_validation import escape_markup` |
| `tldw_chatbook/UI/Screens/personas_screen.py` | `PersonasScreen._expression_upload_dialog_worker`, `PersonasScreen._header_subtitle_text`, `PersonasScreen._notify`, `PersonasScreen._persona_visual_replace_dialog`, `PersonasScreen._visual_identity_replace_dialog` | add `from ...Utils.input_validation import escape_markup` |
| `tldw_chatbook/Widgets/Chat_Widgets/chat_question_card.py` | `ChatQuestionCard._option_prompt` | add `from textual.content import Content` |
| `tldw_chatbook/Widgets/Persona_Widgets/buddy_character_review.py` | `BuddyCharacterReviewDialog.compose` | add `from textual.content import Content` |
| `tldw_chatbook/Widgets/Persona_Widgets/buddy_workspace_modal.py` | `BuddyWorkspaceModal._mark_seen` | unchanged |
| `tldw_chatbook/Widgets/Persona_Widgets/persona_profile_editor_widget.py` | `PersonaProfileEditorWidget.begin_actor_pack_creation` | add `from textual.content import Content` |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py` | `PersonasCharacterEditorWidget._render_greetings_table`, `PersonasCharacterEditorWidget.compose` | add `from rich.text import Text` |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_character_tts_widget.py` | `PersonasCharacterTTSWidget.apply_state` | add `from textual.content import Content` |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py` | `PersonasInspectorPane._apply_action_state`, `PersonasInspectorPane.compose` | add `from textual.content import Content` |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_library_pane.py` | `PersonasLibraryPane.set_tag_label` | add `from textual.content import Content` |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_preview_pane.py` | `PersonasPreviewPane._styled_line`, `PersonasPreviewPane.compose`, `PersonasPreviewPane.set_greetings` | add `from textual.content import Content`; remove `from tldw_chatbook.Utils.input_validation import escape_markup as escape` |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_visual_identity_pack_widget.py` | `PersonasVisualIdentityPackWidget.apply_filter`, `PersonasVisualIdentityPackWidget.compose` | add `from rich.text import Text` |
| `tldw_chatbook/Widgets/Persona_Widgets/petdex_import_review.py` | `PetdexImportReviewDialog._set_state_options` | add `from textual.content import Content` |

Named unchanged methods: preview controller `open_in_console`, `provider_readout`, `ensure_gateway`, `_selection_from_defaults`; question card `_build_section`, `collect_answers`, `action_submit_answers`, `_submit`. Corrected attribution: the original preflight prose names `PersonasScreen.compose`, but the actual owner defines **`PersonasScreen.compose_content` and inherits `BaseAppScreen.compose`**. Both actual method AST hashes are unchanged and pinned. The original preflight proposal remains byte-preserved.

Feature ChatScreen handoff/start admission/runtime owners remain exact. Imported hostile Roleplay provenance (103 hostile cases/36 mutations), TASK34401 file-picker scope and unswept Console handoff strip remain upstream historical provenance/limitations, not fresh Task16 claims.

## Loading, limits and formatting carry

Historical 35-loading qualification remains at its original a688 source and receipts. No fresh35/census claim or replay. Named boot closure/UI-ready/runtime construction/timer controls carry by exact unchanged source ownership. Input_validation remains resident in both boot and ready snapshots; added Content/Text imports are third-party literal text. Existing timer-overlap strict XFAIL and reason remain exact; no new exclusion.

Limits preserved: controller29301/store22344/interrupt6479/compaction4185/ChatScreen25218lines759methods, UI-ready1033, preimport557/425347/135111. Actual upstream Personas row16528 and its dated2026-10-04 owner decision remain exact. Full source hashes/named carries: `loading-and-runtime-named-carry.json`.

Fatal Ruff (`E9,F63,F7,F82`) passes before and on all actual incoming Python paths after integration. Whitespace (`git diff --check BASE HEAD`) passes. Whole-file formatter checks remain red with inherited debt: every incoming/protected owner and its formatter output equal the respective selected-dev/prior-feature source. No reflow or normalization was performed. Combined Python ratchet composition is formatter-exact; its actual incoming hunk is the selected upstream Personas row/comment. Hygiene composition is exact additive prose. Actual formatter attribution overrides historical Task1 inference; current red paths:

- `Tests/CI/test_ci_queue_pressure_contract.py`.
- `Tests/CI/test_derived_artifacts_workflow.py`.
- `Tests/CI/test_measure_required_runs_per_merge.py`.
- `Tests/CI/test_merge_queue_actions.py`.
- `Tests/CI/test_merge_queue_rules.py`.
- `Tests/CI/test_merge_queue_workflow.py`.
- `Tests/CI/test_pr_workflows_dispatch_safe.py`.
- `Tests/UI/test_console_composer_cursor.py`.
- `Tests/UI/test_console_environment_controller.py`.
- `Tests/UI/test_console_runtime_ownership.py`.
- `scripts/measure_required_runs_per_merge.py`.
- `scripts/merge_queue.py`.
- `tldw_chatbook/Chat/console_chat_controller.py`.
- `tldw_chatbook/Widgets/Chat_Widgets/chat_question_card.py`.

## Ownership handoff and concerns

Tracked/index/HEAD are clean at749e7540. All owned pytest/static/Git subprocesses are closed; no running app/worker/process remains owned. No shared checkout, user profile, external PR edit/reply/arming/dispatch/push/merge, GC log removal or prune occurred. Root retains external gates, must reread actual MERGE_QUEUE before final merge, and performs the independent scoped integration review after qualified source. Current externally verified missing variable meant off at preflight time only.

No new source overlay commits. Rebase rewrote existing history only; current HEAD is `749e7540af docs: scope latest-dev roleplay and merge queue integration`. New Task16 evidence is private/additive for root publication. Root now receives source/index/HEAD ownership to amend scope; the same implementer can resume only the exact selected repair.

Open concern: selected runtime qualification remains23/25 passing until root selects and verifies the two-node canonical reader-profile marker repair. No claim of all-green integration or merge readiness. Safe export excludes profiles/config bodies/databases/caches/private probes and recovery bundle contents.

Report: `task-16-report.md`; safe manifest: `task-16-safe-evidence-manifest.json`; evidence: `task-16-safe-evidence/`.


# Task16 extension Ruling71 — selected canonical reader profile markers

**Status: NEEDS_CONTEXT.** Root metadata BASE is `b084d720e98e4c2a7aaaf17ab5c8386c84495aa9`; its two changed paths are plan/Backlog metadata only after frozen749e. This extension was performed by the same implementer with sole tracked/index/HEAD ownership. No new source commit has been made because a deeper proven baseline harness failure requires another scoped selection.

Applied exactly the two proposed `@pytest.mark.bootstrap_profile` lines, one to the real typed-answer integration function and one to the staged Roleplay handoff function. Original imports, all signatures/bodies/decorators/assertions/loops/waits/deadlines and every other definition reverse byte/AST-exact by removing those two lines. No production/config/admission/shared helper/fixture/plugin change occurred. The two owners equal the preselected proposed hashes:

- `Tests/UI/test_console_ask_user_typed_answers.py`: `5a03c3b5a041c5d0aa5a0e355060448860b468249c31a831765d21239175230e`.
- `Tests/UI/test_personas_workbench.py`: `a8378c3d429769052198a44c6f9615ac2aeeb9a414deea7ce1013402334975ce`.

The root-created initial report, 140-artifact manifest and brief snapshots are exact and preserved at their before-fixture-fix paths. The original Task16 report is this report's exact byte prefix. Original setup25 errors, immutable2-node admission failures, actual56 passes and all original source phases remain immutable.

## Actual two-node extension result and reached assertions

Only the two original failing nodes ran on stable amended source, using shared Python3.12, worktree PYTHONPATH/cwd, canonical private profiles, a fresh basetemp and unchanged300s timeout. Exact argv, complete owner/runtime/fixture execution hashes, stdout/stderr/log/XML/result and actual IDs are under `task-16-fixture-fix-safe-evidence/two-reader-nodes*`.

Result: **one pass, one failure**. The staged Roleplay handoff passed its unchanged original transcript/source/item/title/suggested-prompt assertions. The typed-answer integration passed real question publication and card display, then failed at its original visible-send assertion: `AttributeError: 'types.SimpleNamespace' object has no attribute '_ui_responsiveness_monitor'`, from the unchanged real ChatScreen wrapper at18832. Config-participant admission now succeeds; the prior raw-source failure is absent in this phase. No assertion or wait was modified and no failed-node retry occurred.

Two `RuntimeWarning`s for unawaited `App.call_from_thread.<locals>.run_callback` were retained. Pytest labels them with the following Persona case, but they originate in the failing typed test's abort teardown at interrupt rounds3237/1373. No warning suppression or production teardown repair was attempted.

TDD evidence: initial actual749e/immutable6cd raw-source failures remain the profile repair RED. With selected markers, original Persona assertions are GREEN; the typed node reveals a deeper RED, so no complete GREEN claim exists.

## Narrow immutable-base attribution and proposal only

Root selected only the typed node for a new immutable6cd comparison with the exact authorized typed marker overlay. The passed Persona node was not rerun. The baseline tree carried exactly one added typed marker; the second marker was omitted there. Actual baseline source/overlay hashes and diff are frozen under `deeper-immutable-*`.

The exact same typed node fails with the exact same `AttributeError` message on immutable6cd plus that marker. Its failed teardown also reproduces closed-loop/unawaited-coroutine and Loguru closed-sink diagnostics; complete stdout/stderr are preserved. The real ChatScreen send wrapper/observed method and send-diagnostic module remain exact to pre-rebase source. This is a demonstrated inherited stale stand-in contract, not a new Roleplay render regression.

The comparison preserved the current two-line patch first and restored exact metadata HEADb084 plus both selected markers in try/finally. Full carry was reverified afterward. No source branch/rebase/history/QA retargeting or source commit occurred.

Read-only next proposal: add five keyword values only to the typed function's existing `SimpleNamespace`: `_console_pending_send=None`, `_console_visible_draft_session_id=session.id`, `_console_visible_send_session_id=lambda: session.id`, `_ui_responsiveness_monitor=lambda: None`, and a delegate that calls real `ChatScreen._send_console_message_from_visible_action_observed(screen, **kwargs)`. This covers the real wrapper and pre-answer observed send's existing session ownership contract; returning None to the actual diagnostic scope lets its documented owner create/drain its temporary disabled monitor. Original real wrapper, observed routing, card, composer, validation and round remain live, with original assertions intact. Removing only those five new keyword AST nodes reverses the entire owner AST exactly. See `proposed-typed-harness-current-send-seams.patch` and `frozen-deeper-baseline-harness-failure-and-proposal.json`. **The five-field proposal has not been implemented. Root must amend plan/Backlog scope before the same implementer proceeds.**

## Full preservation and static checks

All30,001 unowned tracked paths equal metadata BASE. Both incoming compositions and47 exact selected-dev files remain exact; all132 selected protected feature paths remain exact (neither newly marked test belongs to that132-path array; the two markers are explicit overrides in the complete tracked union). The original1,983 QA paths/ZIP2118-entry container/Task14,15,R1 manifests/snapshots/root proofs and original Task16 140-artifact evidence are exact. All production import/AST/runtime/Close/authority/worker/route guards and all stated limits remain unchanged. Earlier payload557/415370/127527 against557/425347/135111 and its headroom warning remain original source-phase evidence; no payload/census/loading replay.

Fatal Ruff on both owners passes. Whitespace passes. Added decorator hunks are formatting-exact: formatter output differs only by the selected marker lines. Typed owner's existing95 formatter debt units remain95; Persona owner remains0. No whole-owner reflow was applied. The whole-owner formatter command retains its expected inherited red exit1.

Current distinct-case union is **57 qualified passes plus one still-failing typed case**, with phase-specific attribution: original749e runtime23 + CI24 + rows8 + payload1 =56 passes, and the new marked Persona pass =1. The successful56 were not rerun or relabeled to the marked source; no56/61/27/35 replay, install, full suite, newskip/XFAIL, fixture bypass, budget/timeout change or warning suppression occurred.

## Ownership handoff

All owned subprocesses are closed. Index is clean; HEAD remainsb084. The working tree intentionally retains exactly the two authorized uncommitted decorator additions for root's scope amendment, with no unowned changes. The proposed five-field body edit remains unapplied. No external PR/reply/push/arming/merge or shared-checkout/user-profile/GC cleanup occurred. Source/index/HEAD ownership now returns to root; independent Task16 review follows all qualified phases.

Separate extension evidence: `task-16-fixture-fix-safe-evidence/`; separate manifest: `task-16-fixture-fix-safe-evidence-manifest.json`. Current publication manifest remaps the original report row to its exact before-fixture-fix snapshot and adds current report/new phase, preserving historical receipts without overwriting their maps. Safe artifacts exclude profiles/config bodies/DB/cache/private probes.


# Task16 final typed-adapter correction — completed source handoff

**Status: DONE.** Source commit: `81484b688a03c0f924a0ea6a6474b5e2d4b355e8` (`test(ui): restore canonical profiles and typed send adapter`), parent root metadata BASE`f817cbefabdb57896b40371bcdc9346f5bb0659b`. This append preserves the entire interim report prefix byte for byte. All original/interim concerns and failed receipts remain historical evidence; the same implementer completed only the newly selected five-field correction.

## Final implementation and exact ownership

The source commit changes exactly two test owners with11 additive text lines: both selected `@pytest.mark.bootstrap_profile` decorators and five new keywords in only the typed-answer function's existing `SimpleNamespace`. The exact keywords are `_console_pending_send`, `_console_visible_draft_session_id`, `_console_visible_send_session_id`, `_ui_responsiveness_monitor`, and `_send_console_message_from_visible_action_observed`. Optional monitor None uses the real diagnostic scope's owned disabled monitor and close; exact session fields bind the real visible composer; the observed action delegates to the real ChatScreen method. Real wrapper/observed/answer/card/composer/controller/native-round/worker paths remain live. No fake answer/direct alternate action/proxy/shared helper/config/admission/production change occurred.

Removing the five inserted keyword lines and the two selected markers recovers both entire original owner texts and ASTs exactly. Every original namespace key, function signature, statement, decorator, import, assertion, wait, loop, deadline and unrelated definition remains exact. Formatting was limited to the inserted block. Final committed source hashes:

- `Tests/UI/test_console_ask_user_typed_answers.py`: `b9c3ea4b8b0c35bb4501d36285b03b7b374f7a02466939451fb30fc7d93497e9`.
- `Tests/UI/test_personas_workbench.py`: `a8378c3d429769052198a44c6f9615ac2aeeb9a414deea7ce1013402334975ce`.

`source-AST-body-marker-keyword-reversal.json` pins exact reversal and inherited formatter attribution. `source-commit-identity.json` proves the exact two committed paths and execution source bytes equal the committed blobs.

## Actual final typed-only result and source-phase union

Ran ONLY the still-failed typed node once on stable corrected source. Exact argv:

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q -p no:randomly --basetemp=/private/tmp/pr2995-task16/final-typed-adapter-only Tests/UI/test_console_ask_user_typed_answers.py::test_typed_answer_resolves_a_real_round_through_the_real_card_and_composer --junitxml=/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge/task-16-typed-adapter-safe-evidence/typed-node.xml
```

Environment override: worktree PYTHONPATH. Shared Python3.12, canonical private profiles and unchanged300s timeout retained. Result **1 passed, exit0**, 3.881s owning-process elapsed; pytest body/output reports1pass in2.03s. No pytest warning summary, teardown warning, logging error, skip or XFAIL in this final run. Complete stdout/stderr/log/XML/result/source hashes and actual node ID are frozen under `task-16-typed-adapter-safe-evidence/typed-node*`.

Passing reaches all original assertions: real admitted worker publishes the real card; composer typed answer routes through real visible wrapper/observed/answer methods; draft is consumed as an answer with no turn dispatch; real native round resolves both unanswered fields; worker joins while the loop remains available; composer is empty; card hides; pending request IDs drain. Original `controller.begin_shutdown()` remains unchanged. This completes the deeper typed harness RED→GREEN using the frozen immutable6cd marker-overlay failure as RED; no RED or passing cohort replay was performed.

**58 distinct selected cases qualified through preserved source-specific phases:** original749e runtime23 + CI24 + module rows8 + payload1 =56 passes; marker phase staged Persona handoff =1 pass; final typed-adapter node =1 pass. The earlier source-inferred55-case estimate is not repeated as actual. All prior57 passing receipts retain their original source revisions and were not rerun or relabeled to this source commit. Original setup errors, admission failures, deeper harness failure, immutable comparisons and teardown warning/sink-error attribution remain exact historical bytes.

## Verification, preservation and self-review

Fatal Ruff on the two test owners passes. Added keyword/decorator hunks are formatting-exact and all original inherited typed-owner95 formatter debt units remain95; Persona0 remains0. Whole-owner formatter retains the expected inherited red exit1. Whitespace passes for both staged and actual committed overlay. Self-review confirms only selected additions, no assertion or boundary changes, correct real delegation/session authority and no production impact.

Complete current map verifies30,001 unowned tracked paths, all132 protected feature paths,47 exact selected-dev files and both exact compositions. All production methods/imports/guards/runtime/Close/Session/authority/native/physical-drain/strict-abort/recovery/worker/route sources and all stated limits remain exact. Original1,983 QA paths and the63,166,118-byte2118-entry ZIP/SHA carry are exact. All Task14/15/R1 frozen manifests/snapshots/root proofs, initial140 artifacts and interim81 fixture-phase artifacts retain exact bytes, resolving their report rows to the proper frozen phase snapshots.420 unique frozen prior artifact/snapshot/manifest paths were verified before and after correction. No old manifest was refreshed or historical source retargeted.

Earlier loading evidence remains original: actual557/415370/127527 against fixed557/425347/135111, UI-ready1033 unchanged, historical35 at originala688 source. Controller29301/store22344/interrupt6479/compaction4185/ChatScreen25218/759 and Personas16528 with its dated owner decision are unchanged. No loading/census/cap/profile/native/schema test cohort replay, full sweep, install, newskip/XFAIL, warning suppression or timeout/budget increase occurred.

## Final evidence and ownership handoff

Separate final phase: `task-16-typed-adapter-safe-evidence/` and `task-16-typed-adapter-safe-evidence-manifest.json`. The current publication manifest preserves the initial report at before-fixture-fix, interim report/publication/fixture manifests/brief at before-typed-adapter, and adds current report/final typed phase transparently. Historical map bodies remain exact; report-row resolution is explicit. Profiles/config bodies/databases/caches/private probes are excluded.

HEAD/source/index are clean at`81484b688a03c0f924a0ea6a6474b5e2d4b355e8`. All owned subprocesses are closed; no owned app/worker remains. Recovery refs/bundle remain intact; no GC log removal/prune or cleanup occurred. The pinned selected dev remains an ancestor. No external PR/reply/push/arming/dispatch/merge occurred. Source/index/HEAD ownership returns to root. Root's first independent scoped integration review and publication/current Qodo/CI/PerfGuard/fresh ancestry/normal merge gates remain next; no claim of merged state or external gate completion.

No remaining implementer correctness concern; inherited whole-owner formatter debt and preserved source-phase warnings are disclosed above.
