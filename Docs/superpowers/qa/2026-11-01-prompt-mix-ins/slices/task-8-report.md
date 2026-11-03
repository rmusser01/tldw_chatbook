# Task 8 implementation report — TASK-38543

Status: DONE_WITH_CONCERNS (implemented; independent review pending; Backlog remains In Progress).

BASE: `a1fed4545bfc3b438d356b27503afdd6f5f611bd`.
ADR required: no new ADR. This directly implements `backlog/decisions/286-console-prompt-mix-ins.md`; the controller approved the transient, display-only usage-count projection before it was added.

## Implemented behavior

- Preserved `ACTION_PROMPTS` and its Browse landing, per the controller ruling. Added the enabled count to its existing menu entry and a Mix-ins section in the same Workbench. Ordinary Browse/Edit/Improve/Recipe navigation stays available; no standalone manager or permanent composer row was introduced.
- Added a renderer with body-free projections and typed intents carrying exact session/incarnation/preference revision/source references. The opening adapter additionally fences the real store/session object, conversation binding, scope wrapper/backend/DB objects and active chat. Add delegates to `enable_reference`; toggle/reorder/remove use existing runtime CAS. Disabled entries preserve their position. Source-sorted candidate rows do not determine selection order.
- Reconciles available Library placement/name observations in one exact-origin CAS after all reads, placing moved entries at their new group end while preserving enabled flags and last-used-version facts. Unchanged refreshes do not write. Deleted disabled entries remain visible and do not block read-only capture; unavailable enabled selections stay included in the enabled count.
- Reused `PromptMixinEditor` for explicit inspect/edit, with separate Library saves, raw dirty-draft preservation, validation/conflict recovery, and the existing modal discard guard. Ordinary Library Mix-in browsing reaches that same editor. Choice changes remain live across close/reopen; Library edits require Save or explicit discard.
- Reused the existing settings-component save failure/retry owner; there is no new timer or preference store. Retry saving remains a failure-only row.
- Workbench preview consumes the existing exact Next send builder once. That owner captures with `publish_facts=False`; the Workbench does not independently recapture or publish usage/save facts. Context Next send adds body-free used-name/placement/version facts and guards its selection diagnostics against source/revision changes.
- Selected-turn inspection lazily reads original operational snapshots through the existing service, independently of diagnostic traces and current Library bodies. Empty originals remain distinct from unavailable originals.
- Added repr-hidden `ConsoleChatMessage.mixin_usage_count` as a derived display field, appended to preserve positional compatibility. Publication derives it from the confirmed attempt; off-loop hydration and the existing synchronous restore fallback derive it through the raw-marker/anchor/item-validating operational reader. Known legacy/recorded-empty produces zero; unavailable/unqualified produces None. No schema/generic metadata/export authority was added. Dataclass snapshot copies preserve it; restoration requalifies it, and exact Retry never trusts it.
- Nonempty original attempts display `Retry with original mix-ins`; known empty attempts keep `Retry`. Missing originals expose the existing `Regenerate` action with current-choice copy. Failed exact lookup clears only the matching session/incarnation/binding/assistant/metadata projection; transport and NEW-attempt write failures retain valid original usage. Current-B regeneration is exercised after original-record loss.
- Added token-backed dense rows/narrow grids in source TCSS and rebuilt the bundle. Updated the Console guide.

## TDD and verification evidence

All commands used `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python` from this worktree, private `/private/tmp/task8-*` basetemps, and `-p no:cacheprovider`. No full suite was run.

RED:

```text
python -m pytest Tests/UI/test_console_prompt_mixins.py -q --basetemp=/private/tmp/task8-red -p no:cacheprovider
```

The new interaction module failed collection because `_ConsolePromptMixinsFlow` did not exist. This was the expected missing Workbench/runtime integration seam. The actual mounted sequence was then exercised after implementation (rather than treating collection failure as proof of mounted behavior). See `task-8-evidence/red.txt`.

Final risk-focused GREEN:

```text
python -m pytest Tests/Chat/test_console_prompt_mixin_retry.py Tests/UI/test_console_prompt_mixins.py Tests/UI/test_console_conversation_inspector.py::test_explicit_turn_inspection_uses_operational_mixins_without_trace Tests/UI/test_design_token_governance.py Tests/UI/test_css_bundle_sync_guard.py -q --basetemp=/private/tmp/task8-final-risk -p no:cacheprovider
94 passed in 127.00s
```

The final narrow recovery extension, after adding exact failure-category/incarnation/binding guards:

```text
python -m pytest Tests/Chat/test_console_prompt_mixin_retry.py::test_regenerate_current_choices_recovers_unavailable_original Tests/Chat/test_console_prompt_mixin_retry.py::test_valid_original_failure_retains_retry_usage_and_exact_snapshot Tests/Chat/test_console_prompt_mixin_retry.py::test_operational_inspection_and_display_count_never_authorize_retry -q --basetemp=/private/tmp/task8-recovery-final -p no:cacheprovider
9 passed in 11.49s
```

Native/factory GREEN after fixing the deferred-focus race and single-builder preview:

```text
python -m pytest Tests/UI/test_console_prompt_mixins.py Tests/UI/test_chat_screen_console_inspector_loader.py::test_inspector_push_captures_immutable_revision_target Tests/UI/test_console_personal_context_selection_factory.py -q --basetemp=/private/tmp/task8-focus-preview -p no:cacheprovider
25 passed in 27.82s
```

The main native test constructs the real application/runtime, source service, store and controller; opens actual `ACTION_PROMPTS`; uses mouse access plus Tab/Space/Enter and Move controls; asserts real service order/enabled flags; verifies original draft/System preservation, one exact read-only preview, shared editor opening and opener-focus restoration. Other mounted tests cover close/reopen, stale chat/revision/source refusal, deleted records/counts, dirty close/invalid save, live-unsaved retry, external disabled placement/group-end updates and a held refresh losing its CAS to a newer toggle. Worker completion and bounded state predicates are used.

Receipt file verification also included six capture-off prepared/synchronous-reopen cases (nonempty/empty × valid/missing/raw-corrupt), and a forged positive display count that cannot authorize Retry. The prior complete receipt file run passed 68 cases; final risk run includes those cases. Explicit historical renderer cases pass for used/empty/unavailable originals with no diagnostic captures.

## Required files and inherited failures

The required three UI files were run in full:

```text
python -m pytest Tests/UI/test_console_prompt_mixins.py Tests/UI/test_console_prompts_modal.py Tests/UI/test_console_composer_menu.py -q --basetemp=/private/tmp/task8-ui1 -p no:cacheprovider
158 passed, 6 failed in 97.55s
```

The four modal failures match exact nodes already qualified by `task-1-report.md` against its earlier base:

- `test_no_matches_retry_and_source_unavailable_are_explicit`
- `test_source_switch_clears_foreign_rows_before_unavailable_result`
- `test_real_recipe_fill_with_additional_context_reaches_block_review`
- `test_mapped_context_blocks_recipe_save_until_deleted`

Two menu failures were rerun against an isolated archive of this exact Task 8 BASE, without stash/reset or modifying the managed worktree:

```text
git archive a1fed4545bfc3b438d356b27503afdd6f5f611bd tldw_chatbook Tests pyproject.toml | tar -x -C /private/tmp/task8-base
# From /private/tmp/task8-base:
python -m pytest Tests/UI/test_console_composer_menu.py::test_impersonate_payload_obeys_the_provider_contract Tests/UI/test_console_composer_menu.py::test_activate_console_session_for_workspace_keeps_temporary_chip_honest -q --basetemp=/private/tmp/task8-base-nodes -p no:cacheprovider
2 failed, 1 warning in 7.89s
```

The last broader affected UI run was 200 passed/6 failed: five of the above existing failures plus a new Tab focus race. That race was fixed by focusing the mounted replacement synchronously, then covered by the clean 25/94/final-native runs. The workspace-session baseline failure did not reproduce in that broader run; it is not claimed deterministic.

An affected-owner run was 205 passed/11 failed. The Inspector fake needed its real owner's incarnation field added; its original assertions now pass. The other ten Canvas/tool-label nodes all failed on the same isolated Task 8 BASE in 0.77s (`exact-base-actions.txt`): six Canvas/Mermaid cases and four parameterized refused-tool-marker label cases. Their origin before Task 8 BASE is NOT attributed; they may require the controller's planned earlier-feature/original-baseline audit. No unrelated Canvas/tool behavior was changed.

## Bounded visual verification and artifacts

One initial batched inspection at 160×45, 100×30 and 80×24 found excessive generic section/row spacing. One token-backed compact fix batch was applied. The confirmation's first capture attempt exposed test synchronization/focus problems; those were corrected and the final artifact export was source-qualified after the necessary functional fixes. No further visual polish loop was performed.

```text
TASK8_CAPTURE_DIR=/private/tmp/task8-captures-source-qualified python -m pytest Tests/UI/test_console_prompt_mixins.py::test_real_console_prompts_action_retains_browse_and_controls_runtime -q --basetemp=/private/tmp/task8-source-qualified-capture -p no:cacheprovider
3 passed in 16.23s
```

`task-8-evidence/captures-final/` contains Workbench and shared-editor SVGs at all three sizes, plus raster previews/contact sheet for review. These use production styles and the actual Console host, not a replacement visual shell. Final app source hashes match `source-hashes-at-final-capture.json` exactly. `source-hashes-final.json` separately records the final review bytes, including later Backlog notes; `capture-hashes.json` identifies the SVG artifacts. Initial captures are retained separately.

## Static checks and self-review

- Source TCSS rebuilt through `python tldw_chatbook/css/build_css.py`; token and bundle guards pass.
- Full-file Ruff current-versus-exact-BASE diagnostic code/message multiplicities compared per path, including the large controller/store files: `lint-delta.json` is `[]`. This is NOT changed-line-only lint. Existing whole-file lint debt remains recorded in both JSON outputs.
- New view/test files pass Ruff and formatter checks; modified logical ranges were formatted without sweeping unrelated formatter debt.
- Targeted compileall and `git diff --check` pass.
- Self-review found and fixed: empty editor query crashes; worker-local SQLite in-memory fixture misuse (replaced with a real file-backed source); deferred focus stealing the next keyboard action; a double capture in preview; missing current-choice recovery after a stale positive usage count; and stale-refresh overwrite risk. No bodies/raw errors were added to generic metadata or logs.
- Read the full slice diff, inspected copied/restored/promotion/fork projection behavior and verified source/session/revision fencing. Existing operational service remains the sole Retry proof owner. Added no dependency, provider operation, permission, schema or unrelated cleanup.

## Remaining limits

Required full UI files and broader owners retain the baseline failures detailed above. No full suite, live external provider call, server snapshot endpoint, push or merge was attempted. Server mix-in capture remains explicitly unsupported. Independent source/UI review is still required; Task 8 remains In Progress for it.

## Local commit and changed files

`3c77bbbdd701052bb778eaf1bc09fea7b838c8ae` — `feat: manage chat mix-ins in the Prompt Workbench`.

- `Docs/User_Guide/console/chat-basics.md`
- `Tests/Chat/test_console_prompt_mixin_retry.py`
- `Tests/UI/test_chat_screen_console_inspector_loader.py`
- `Tests/UI/test_console_composer_menu.py`
- `Tests/UI/test_console_conversation_inspector.py`
- `Tests/UI/test_console_personal_context_selection_factory.py`
- `Tests/UI/test_console_prompt_mixins.py`
- `Tests/UI/test_console_prompts_modal.py`
- `backlog/tasks/task-38543 - Manage-chat-mix-ins-through-the-existing-composer-Prompts-action.md`
- `tldw_chatbook/Chat/console_chat_controller.py`
- `tldw_chatbook/Chat/console_chat_models.py`
- `tldw_chatbook/Chat/console_chat_store.py`
- `tldw_chatbook/Chat/console_conversation_hydration.py`
- `tldw_chatbook/Chat/console_message_actions.py`
- `tldw_chatbook/Chat/console_turn_resend.py`
- `tldw_chatbook/UI/Console_Modules/composer_run_controls.py`
- `tldw_chatbook/UI/Console_Modules/inspector_data.py`
- `tldw_chatbook/UI/Console_Modules/prompts.py`
- `tldw_chatbook/UI/Screens/chat_screen.py`
- `tldw_chatbook/Widgets/Console/console_composer_menu_modal.py`
- `tldw_chatbook/Widgets/Console/console_conversation_inspector.py`
- `tldw_chatbook/Widgets/Console/console_prompt_mixins_view.py`
- `tldw_chatbook/Widgets/Console/console_prompts_browse.py`
- `tldw_chatbook/Widgets/Console/console_prompts_modal.py`
- `tldw_chatbook/Widgets/Console/console_prompts_state.py`
- `tldw_chatbook/css/features/_console_panels.tcss`
- `tldw_chatbook/css/tldw_cli_modular.tcss`

The local commit succeeded. Git emitted existing automatic-GC/loose-object warnings; no cleanup was attempted.


## Fix round 1 — review I1/I2

FIX_BASE: `3c77bbbdd701052bb778eaf1bc09fea7b838c8ae`.

I1: Added a private Workbench generation, invalidated by every mode mount (including Back) and unmount. Projection replacement checks generation, rendered view identity and current editor/raw-draft identity before success/error publication, and rechecks after awaited removals/mounts. Action, inspect, save and preview completion paths likewise retain their captured view generation/identity. The shared editor's awaited mounting/focus path accepts the same currentness guard. Stale workers cannot overwrite Browse, error into a replacement view, or restore stale focus. No finally publication/reset was added.

I2: Save captures the submitted raw draft and editor identity. The existing source adapter returns the exact saved record/version. If later typing is present when that write finishes, the original editor remains mounted, its saved baseline advances to the submitted draft, the newer raw text remains dirty, and its editor intent uses the current preference revision for a subsequent conditional Save. Rendered selection snapshots are not rewritten behind existing row controls. A normal unchanged Save still refreshes the section. The regression includes an initial name/placement change (which advances preference revision), newer Instructions typed while held, verified saved version 2, and a successful second Save of the newer text at version 3.

Deterministic RED on the two reported defects:

```text
python -m pytest Tests/UI/test_console_prompt_mixins.py -k 'held_projection_cannot or held_save_preserves' -q --basetemp=/private/tmp/task8-fix1-red -p no:cacheprovider
3 failed, 10 deselected in 3.66s
```

The two projection cases replace Browse on both success and error; the Save case loses the newer editor. After the fix, the same initial nodes passed (`3 passed, 10 deselected in 3.82s`).

Covering verification (same private-profile runtime, real source/store and production styles):

```text
python -m pytest Tests/UI/test_console_prompt_mixins.py -q --basetemp=/private/tmp/task8-fix1-final -p no:cacheprovider
19 passed in 27.50s
```

The already-added controls cover inspect/save/preview success and error completions after navigation. Their first held-save setup attempted direct Save before the actual inspect worker completed child initialization; two bounded 5-second barriers timed out. Waiting for that real worker completion fixed setup (`2 passed, 17 deselected in 3.45s`), without sleeps or weakening product checks.

After the final adjustment keeping rendered selection snapshots immutable, only the already-added held completion cases were repeated:

```text
python -m pytest Tests/UI/test_console_prompt_mixins.py -k 'held_projection_cannot or held_save_preserves or held_completion' -q --basetemp=/private/tmp/task8-fix1-completions-final -p no:cacheprovider
9 passed, 10 deselected in 6.85s
```

Static checks: full-file per-path Ruff code/message multiplicities against exact FIX_BASE bytes are unchanged (`42 BASE; 42 current`, `fix-round-1/lint-delta.json` is `[]`). Modified logical ranges were formatted. Formatter check on the view/test files reports `2 files already formatted`; `git diff --check` and targeted compileall with `PYTHONPYCACHEPREFIX=/private/tmp/task8-fix1-pycache` pass. An initial unprivileged compileall could not write bytecode into the managed worktree; the explicit private cache avoids that filesystem restriction.

No style/layout values changed, so screenshots were not refreshed. Prior captures remain evidence for their recorded source packet; `fix-round-1/source-hashes.json` identifies the changed fix bytes. No full suite, live provider, unrelated baseline qualification, helper agent or variant-count follow-up was run. The reviewer's minor generation-variant count-copy seam remains the named Task9 audit. Task38543 remains In Progress for scoped re-review. Self-review confirmed all awaited UI success/error/focus publications are guarded and newer raw drafts survive against the saved version.

Fix round 1 commit: `0dcf71c2e5a3e93aecf1618a62f3c423a7b35f20` — `fix: fence prompt mix-in workbench completions`. Working tree clean; fix source hashes verified after commit.
