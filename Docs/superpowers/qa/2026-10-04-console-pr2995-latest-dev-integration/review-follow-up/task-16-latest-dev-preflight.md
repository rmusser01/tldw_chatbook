# Task 16 latest-dev preflight — bounded proposal

Status: read-only proposal, awaiting root selection. No rebase, source edit, suite, PR mutation, push or dispatch performed.

Feature source HEAD: `eb7cb871390616de22d28c7a6679bbc9202841d7`. Root bookkeeping HEAD is now `d9d975eff5c93da1645910f4e2eb938c03bbb40e`; actual Git diff contains only this plan and TASK34215.2 metadata, so the protected source pins apply unchanged. Previously integrated dev: `49206beea90d35ea9e6842ffa44e4b274db29d8a`. Selected latest dev: `a78a9a900b4901e33c031f830dd2d80224d5147d`. The incoming delta is 25 commits, 49 paths, from PR3015 (TASK34400 literal Roleplay text) and PR2996 (TASK34211 merge queue).

ADR required: no new ADR.
ADR paths: `backlog/decisions/218-in-repo-merge-queue.md`, `backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md`.
Reason: integrate landed policy and implementation without introducing a boundary.

## Resolution and source preservation

The feature delta since integrated dev has 2134 paths, overwhelmingly preserved QA. The only two incoming path overlaps are:

- `Tests/Architecture/test_module_size_ratchet.py`: keep feature controller **29301**, interrupt rounds **6479**, compaction **4185** rows; retain incoming owner-approved Personas **16525→16528**, including its dated comment. These are separate hunks. Neither changing a feature budget nor raising an unrelated limit is authorized.
- `backlog/docs/lessons-backlog-hygiene.md`: preserve the feature's QA-in-Git incident and upstream's one-line merge-queue supersession notice. Separate hunks; no prose rewrite.

Three-way `git merge-file -p HEAD BASE DEV` probes returned 0 without markers for both. Their expected combined hashes are pinned in the JSON. **No final-tree conflict is predicted.** A full historical rebase was not attempted; this does not promise every replayed commit is conflict-free. If a rebase stops elsewhere, the implementer must preserve it and report the exact path/hunk to root before resolving it.

The other **47** incoming paths must equal the selected-dev blobs exactly, including all workflows, scripts, rules, tests, task documents and lessons. All **132** feature source/test/script paths outside the two overlaps must retain exact preflight HEAD bytes. The selection JSON contains every incoming exception and every protected feature source hash, plus import/method AST rows for every changed production module. No broad 'upstream changed' exception is authorized.

Root supplied the actual carry receipt `task-15-fix1-root-handoff-verification.json`: 8124 current source pins (8119 exact + five Task15 owned overrides), all 69 Task14, 119 Task15 and 46 R1 manifest pins, and the full original ZIP hash verified at the source HEAD. This preflight does not rerun that qualification.

The 1983 feature QA paths have **zero** incoming intersections. Preserve their original bytes and original source revisions. The JSON pins a reproducible ordered path/blob/hash digest; verify against Git objects before and after rebase. Any new Task16 receipt is additive. Do not retarget historical receipts to the rebased SHA, regenerate old archives/manifests or refresh failing snapshots.

## Actual runtime intersection

**Shared questions:** `ChatQuestionCard._option_prompt` now returns literal `Content`; labels and descriptions are model-authored. `_build_section` and submission logic retain their AST. `Content` changes rendering, while answer values still come from original option dictionaries. Qualify the incoming seven hostile strings (single and multi-select painted in each case), the original mixed radio/multi-select/Other answer assertion, and the real controller→card→composer→round result case. No card assertions, limits or deadline behavior may be rewritten.

**Roleplay preview→Console:** `PersonasPreviewController._provider_label` now returns plain provider display text to literal readout/status widgets. `PersonasPreviewPane._styled_line` constructs `Text` from literal segments with italic spans; it no longer feeds a Textual escaper to Rich's parser. Greeting prompts become literal `Content`. Qualify the seven incoming provider-label cases, seven painted preview/status/greeting cases (including `a\[/]b` and LaTeX), original italic-span assertion, and exact existing staged-preview handoff assertion. `open_in_console`, `provider_readout`, `ensure_gateway`, `_selection_from_defaults` and `PersonasScreen.compose` retain their AST. Feature ChatScreen handoff/start admission and runtime owners are byte-unchanged.

**Other landed Roleplay surfaces:** notifications, inspector, labels, cells, titles and Buddy/Petdex prompts are imported as exact selected-dev files. Their tests are preserved. No feature source intersects those changes. The incoming record reports 103 hostile cases and 36 mutations on its own source; carry that provenance without relabeling it as fresh Task16 verification. TASK34401's app-wide file-picker sinks and the unswept Console handoff strip remain their original scope limitations.

| Incoming production path | Changed method AST exception | Required preservation |
|---|---|---|
| `tldw_chatbook/UI/CCP_Modules/ccp_character_handler.py` | `CCPCharacterHandler._notify` | Exact selected-dev bytes/AST; incoming literal rendering retained. |
| `tldw_chatbook/UI/CCP_Modules/ccp_loading_indicators.py` | `LoadingManager.start_loading`, `with_loading` | Exact selected-dev bytes/AST; incoming literal rendering retained. |
| `tldw_chatbook/UI/CCP_Modules/ccp_persona_handler.py` | `CCPPersonaHandler._notify` | Exact selected-dev bytes/AST; incoming literal rendering retained. |
| `tldw_chatbook/UI/CCP_Modules/ccp_validation_decorators.py` | `validate_file_import`, `validate_input` | Exact selected-dev bytes/AST; incoming literal rendering retained. |
| `tldw_chatbook/UI/Navigation/buddy_management.py` | `BuddyManagementCoordinator._apply_and_report` | Exact selected-dev bytes/AST; incoming literal rendering retained. |
| `tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py` | `PersonasPreviewController._provider_label` | Exact selected-dev bytes/AST; incoming literal rendering retained. |
| `tldw_chatbook/UI/Screens/personas_screen.py` | `PersonasScreen._expression_upload_dialog_worker`, `PersonasScreen._header_subtitle_text`, `PersonasScreen._notify`, `PersonasScreen._persona_visual_replace_dialog`, `PersonasScreen._visual_identity_replace_dialog` | Exact selected-dev bytes/AST; incoming literal rendering retained. |
| `tldw_chatbook/Widgets/Chat_Widgets/chat_question_card.py` | `ChatQuestionCard._option_prompt` | Exact selected-dev bytes/AST; incoming literal rendering retained. |
| `tldw_chatbook/Widgets/Persona_Widgets/buddy_character_review.py` | `BuddyCharacterReviewDialog.compose` | Exact selected-dev bytes/AST; incoming literal rendering retained. |
| `tldw_chatbook/Widgets/Persona_Widgets/buddy_workspace_modal.py` | `BuddyWorkspaceModal._mark_seen` | Exact selected-dev bytes/AST; incoming literal rendering retained. |
| `tldw_chatbook/Widgets/Persona_Widgets/persona_profile_editor_widget.py` | `PersonaProfileEditorWidget.begin_actor_pack_creation` | Exact selected-dev bytes/AST; incoming literal rendering retained. |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py` | `PersonasCharacterEditorWidget._render_greetings_table`, `PersonasCharacterEditorWidget.compose` | Exact selected-dev bytes/AST; incoming literal rendering retained. |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_character_tts_widget.py` | `PersonasCharacterTTSWidget.apply_state` | Exact selected-dev bytes/AST; incoming literal rendering retained. |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py` | `PersonasInspectorPane._apply_action_state`, `PersonasInspectorPane.compose` | Exact selected-dev bytes/AST; incoming literal rendering retained. |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_library_pane.py` | `PersonasLibraryPane.set_tag_label` | Exact selected-dev bytes/AST; incoming literal rendering retained. |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_preview_pane.py` | `PersonasPreviewPane._styled_line`, `PersonasPreviewPane.compose`, `PersonasPreviewPane.set_greetings` | Exact selected-dev bytes/AST; incoming literal rendering retained. |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_visual_identity_pack_widget.py` | `PersonasVisualIdentityPackWidget.apply_filter`, `PersonasVisualIdentityPackWidget.compose` | Exact selected-dev bytes/AST; incoming literal rendering retained. |
| `tldw_chatbook/Widgets/Persona_Widgets/petdex_import_review.py` | `PetdexImportReviewDialog._set_state_options` | Exact selected-dev bytes/AST; incoming literal rendering retained. |

## Narrow qualification proposal

Use the exact node arrays in `task-16-latest-dev-preflight-selection.json` in separate fresh pytest processes. Proposed selection: **25 runtime cases**, **21 CI contract cases**, **8 exact ratchet rows**, and **1 preimport payload case**. These counts are inferred from source parametrization, not a test run or collection receipt. No whole-file Console/loading cohort is selected.

The CI nodes pin both required lane conditions, fail-closed aggregation, unchanged shard/dependency/check-name contracts, `pr` dispatch input, non-required queue tick, no-PR dispatch off dev, PR-workflow dispatch guards and checkout fallback, queue dev-only triggers/permissions, dry-mode non-mutation and never-arm/merge/push. Preserve all additional incoming queue tests and the measure script as exact upstream bytes; Task16 does not revalidate or rewrite the whole queue algorithm.

**One necessary loading case:** source-line payload changed in preimported Roleplay modules, so run only `Tests/Performance/test_screen_preimport_payload_budget.py::test_preimport_pass_payload_stays_within_budget`. Preserve its actual current constants: **557** modules, **425347** total LOC, **135111** single-route LOC. The older QA summary's 556 measurement is historical and must remain historical. No budget increase or refreshed failing snapshot is proposed.

**Named carry:** preserve `Tests/Packaging/test_console_interaction_boot_closure.py::test_console_defers_setup_and_environment_io_until_requested` and the UI-ready census at **1033**. Their source and Console startup/environment seams are unchanged. The added local import is the already resident `Utils.input_validation` (`boot_import_modules.txt` and `ui_ready_modules.txt` both name it); the other imports are third-party literal text classes. Carry the explicit runtime-construction ownership pin and the exact existing strict-XFAIL timer-overlap test; no whole runtime ownership rerun or new skip/XFAIL.

Run commands use the verified Python3.12 interpreter `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`, the PR worktree cwd, explicit selectors, fresh isolated `--basetemp` directories and existing test configuration. Do not blanket-add `bootstrap_profile` using a plugin. The incoming real Roleplay flow module has its own marker; other tests retain their own fixture contracts. A setup/admission failure is evidence, not permission to weaken tests. Preserve stdout, stderr, exit code, JUnit and exact source hashes. On a failure compare only the same node with selected dev or pre-rebase HEAD before asking root to select a concrete repair.

## Formatting and limits

Capture pre-rebase current-source Ruff/formatter detail without edits. The old Task1 format reference records inherited debt in controller, Session and five UI test files (exact paths/counts in the JSON); it is historical, not today's computed baseline. Protected feature bytes must stay exact. Incoming modules and CI tests must stay exact selected-dev bytes, including inherited formatting (the new derived-artifacts assertions are visibly not uniformly reflowed). Do not normalize whole files to make a whole-file formatter check pass. Both overlapping files are additive text composition only; verify no unapproved AST/assertion/cap change. All imported/source-inspection checks execute only after source is stable.

## Eventual merge rules

Worker's initial `gh variable get MERGE_QUEUE` failed to connect, so it established no mode. Separately, root supplied the authorized read-only result of `gh variable get MERGE_QUEUE --repo rmusser01/tldw_chatbook`, exit1: **variable MERGE_QUEUE was not found**, approximately 22:27 on 2026-10-04. Under the landed AGENTS rule that means **off**.

For off/dry, re-sync only the PR about to merge by **rebase** (local `git rebase origin/dev` or `gh pr update-branch --rebase 2995`); use `--force-with-lease` for a local rebase push. Arm only when Qodo has reviewed the current head and every current-head thread is resolved. Disable auto-merge before pushing more work. Never merge dev into the branch. Strict protection still requires current-dev head, green required check and resolved threads, including admins.

Re-read mode before the final action because it may change. If on, arm only under the same review/thread rule, then let the queue rebase/dispatch; do not hand-sync or merge an armed PR. Before modifying an armed branch disable-auto, pull-rebase the possibly queue-rebased branch, then force-with-lease. Never approve the token-rebase's duplicate action_required runs. Queue eviction turns auto-merge off; repair the reported cause before re-arming. No merge action is part of this preflight.

## Recommendation

Select a concrete Task16 that rebases the pinned head onto the pinned dev, preserves the exact source exceptions and all original limits/assertions/QA, runs only the named cases, and records additive evidence. The two final-tree overlaps compose automatically; runtime source changes are wholly incoming. A fresh implementer should stop on any additional conflict or failure and return exact evidence for root's next selection.
