# PR3346 UI12/UI1 stale oracle correction

TASK38544 AC16 covers exact shared UI CI diagnosis and preserves existing assertions, authority, cleanup, dispatch, design tokens and budgets. Root requested these two exact findings. No new ADR: test-only alignment to existing destination and bounded-copy decisions.

## Cause

UI12: the strict destination-to-class map omitted two intentional dev shell destinations. Plugins was added by 177a2936c7 and Kanban by 04233174e6. The original exact node was reproduced RED (1 failed in 6.33s; private child 1 failed in 2.93s), showing only the two extra actual entries. The fix adds explicit plugins=PluginsScreen and kanban=KanbanBoardScreen expectations; exact dictionary equality remains unchanged.

UI1: dev34af426e7d deliberately changed the coded image Send strip to “this model can't accept images” to preserve the limitation when the model id is long. TASK38424 and the user guide explicitly document that policy, with full model-specific detail retained in the attachment row and Send tooltip. The older attachment test's STRIP_REASON constant still named local-model. Parent's exact RED capture and the hosted log show that mismatch at the final setup-to-model transition. Correct the shared STRIP_REASON expected text and remove the long-model case's obsolete explicit model-specific strip override, retaining NOTICE, full model-specific tooltip/row checks, setup priority, hidden/disabled row assertions, attachment retention and all action/focus/painted-output oracles.

All five relevant files were byte-identical between original rebased source 79e659dcde6f5360edb0070f9d184713bd8584d1 and pinned dev62a169f906508ed02bd7546ab278d916638d1607 before these edits (source-proof.json). Both are shared baseline expectation drift, not a mix-in-only production behavior change. Intent source diffs are preserved as vision-copy-history.txt, plugins-history.txt and kanban-history.txt.

## Verification

Initial affected verification: **13 passed, 1 failed in 57.36s**. Both reported CI nodes passed. The remaining long-model case failed because it explicitly overrode the corrected shared strip expectation with the old model-specific text. After removing that obsolete override, its unchanged rendered ellipsis/full-tooltip/button assertions passed: **1 passed in 7.03s**, exit0. No same-byte passing tests were repeated; this is combined covering evidence, not a claim of one all-green14 run. Exact argv/cwd/results are green.json and long-model-green.json; logs and junit files share those prefixes. The command contains the exact navigation test, the affected attachment vision module because its shared constant drives helper waits, and the existing long-model strip control at both parametrized widths. Original navigation repro is nav-red.json/log/xml; parent's separately owned vision RED is copied as root-red.json; hosted evidence is ui12-ci.log and ui1-ci.log.

Per-file Ruff against FIX_BASE79e659: navigation has 15 inherited diagnostics, vision module 2 inherited diagnostics; code lists are identical before/after. Navigation formatter changes are identical; vision formatter changes introduce no new edit and remove the existing formatting issue at the deleted obsolete long-model argument. No blanket cleanup. git diff --check is exit0. Evidence in lint-comparison.json and diff-check.json.

Only Tests/UI/test_screen_navigation.py and Tests/UI/test_console_attachment_vision_block.py changed in this correction; change.patch preserves exact diff. No production gate, route, copy, authority, CSS, time budget or assertion strength changed. No full suite, helpers, commits, pushes or repository evidence-manifest edits. Private tests use existing bootstrap/private-profile ownership and real Textual controls; no cloud-provider behavior is claimed. Root concurrently owns other CI corrections.

Exact selected evidence paths, sizes and SHA256 hashes are recorded in report-files.json. Current tested source hashes are source-proof.json. All evidence paths belong to this uniquely created private directory; no shared temporary task globs were collected.
