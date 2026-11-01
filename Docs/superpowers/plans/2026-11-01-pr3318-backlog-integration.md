# PR3318 backlog integration plan

Use superpowers:executing-plans inline and requesting-code-review for independent review.
Goal: integrate the approved Temporary Capture Off behavior while preserving newer authored drafts and current native/trace recovery ownership.
Architecture: use the existing received-input revision/composer snapshot and preparation owner. No text-equality substitute for ownership and no new service or recovery system.
Tech stack: Python, Textual, real SQLite/private-profile test harnesses.
Spec: TASK-33621.17 and TASK-33621.18 acceptance criteria; ADR-097 temporary capture policy; ADR-230 original-input ownership.
ADR required: no new ADR. Existing ADR-097 source amendments and ADR-230 govern these corrections.

## Rebase and preservation
- Preserve source00272b4d525fd14d77326ace78d20bd2be7a02d8 and its29 commits; true base4d6764b132edde35c8a32b5d2c92ab189416dd6d. Rebase onto current dev in the freed isolated PR3315 worktree, retaining its completed branch. Refresh final dev again after older PR3317 merges.
- Resolve overlaps by keeping current-dev physical-call refusal evidence, native write retirement, original input snapshots, trace Cancel restoration and stronger assertions. Merge census additions with exact current prefix. Do not overwrite whole modules from the older source.
- Inspect current console_unsent_turn.py, UI/Console_Modules/session.py card dispatch, console_runtime.py retained recoveries, console_chat_store.py cancellation and existing input revision owners before finalizing the repair interface.

## Authored-draft ownership repair
- Amend TASK-33621.18 AC6/7 to explicitly preserve later authored identical or empty drafts and keep original attachments recoverable. Put integration task in progress and append this plan before code repairs.
- Add real-runtime mounted regressions in Tests/UI/test_console_temporary_capture_flow.py: hold card-sent provider dispatch; edit away/retype identical prompt before completion; preserve newer composer/store revisions. Cancel after an authored empty draft with an attachment; preserve empty draft and recoverable original attachment. Keep unchanged-original controls.
- Run new regressions RED on the integrated source before repair.
- Repair existing clear_answered_draft/card settlement using the original preparation input revision and captured composer ownership, never a later completion snapshot. Refuse automatic restore/clear when the original owner was superseded. Keep explicit recovery accessible. Preserve current dev trace-card Cancel for already accepted turns without inventing a preacceptance entry.
- Run the new regressions GREEN plus only reached input/capture/promotion/Cancel/native-retirement controls recommended in the independent source review. Use the private-profile runner and serial mounted UI runs; never a full sweep.
- Record exact source-stable evidence, relevant ADR/task notes, scoped static/census checks, and final independent review before exact-lease publication.

Scope: only non-draft PRs, oldest publication first. PR3267/3297 excluded. Hosted CI and size limits waived; production/security/ownership assertions remain. No release or main promotion.

## Rebase result

All29 source commits are integrated on confirmed dev945d89184bd0824e97a79c7243fa7a07c0fdc8c6. Current integration head7662706edf687fb7b1524427caa8986a748a7ab8. Conflicts preserved dev trace_card_returns_draft and its accepted-trace Cancel restoration alongside source retained preacceptance recoveries. Census preserves all1276 dev entries plus6 source selections (floor1282). Unrelated module/screen rows stay at dev values; source-owned rows are measured under the user waiver. Draft-revision repair and regression execution remain pending.

## Completed integration

Original-revision settlement, exact Cancel restoration receipt and synchronous accepted-result retention were implemented and independently approved. Focused RED/GREEN receipts, fixture corrections, current-base preservation and qualification limits are recorded in ../qa/2026-11-01-pr3318-backlog-integration.md. Census now1285 with the exact1276-entry dev prefix. No new ADR: existing ADR097/230 apply.
