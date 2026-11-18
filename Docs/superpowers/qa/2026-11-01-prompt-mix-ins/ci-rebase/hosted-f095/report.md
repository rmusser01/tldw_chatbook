# f095 hosted Perf Guard contract repair

Starting HEAD f09597cdd5a2b5af1b07dea7eabe2eb6d33d9c80. Only three test/harness files changed. Task38544 was reopened/plan updated by parent before work. No product, threshold, schema/index, Qodo, commit or push changes.

## Causes and fixes

- Compiled-source cold FULL fixture omitted upstream TASK37840's temporary-chip callback (also present on origin/dev). Added an explicit recording callback asserting publication completed before the one chip refresh. Existing source compilation/SHA, no-native-import oracle, deferred projection and replay checks stay intact.
- Reaching that path exposed another stale fixture: `lambda: object()` cannot satisfy the current real post-await attachment-visit check. Use a stable actual pure `ConsoleAttachVisit` value; no predicate/fence is mocked or weakened.
- Inventory increased28 to30 solely through console_prompt_mixin_attempts.user_message_id and assistant_message_id, already classified as forbidden by production. Pin their exact names and30 total in both old count oracles; assert26 forbidden distribution entries and100× scaling.
- The old distribution now summed1082, so the existing strict-xfail was masking setup drift. Allocate2checkpoint columns×44 and24others×38=1000, preserving100000 large observations. Same-table columns keep equal row counts. All trial counts, timings, ABBA order, thresholds and strict-xfail remain unchanged.

## Evidence

Exact hosted27-contract cohort:26 passed,1 expectedxfail in3.64s (`final.log/xml`). `run_observed.py` calls original tests/assertions and only records the history helper's unmodified return. Initial bounded run:25passed/1failed/1xfail; its failure revealed the stale attachment fixture (`contracts.log/xml`).

`history-report.json` proves full history construction and40trials/10warmups/ABBA completed, matching pragmas,1000/100000 locator observations and842/84200 physical unrelated rows (multiple columns share rows). The original assertions preceding scan_count all pass against that return; its14scans are the first failing assertion. `history-assertion-proof.json` records this inference explicitly because JUnit emits only the marker message, not the xfail traceback. No new run needed to infer the first failing assertion from the actual returned data.

There are7actual scan plans per corpus (14total), despite old marker prose saying8 unindexed columns; marker left unchanged. Both new Mix-in columns use actual covering indexes idx_console_mixin_attempt_user/assistant in both baseline and large corpora, no scans. No INDEXEDBY, fake plans, index edits or threshold changes.

Scoped default Ruff3->3 versus f095, zero new diagnostics (`lint.json`). Both history/inventory files format-clean; scoped diff whitespace check passes. `source-proof.json` records hashes and confirms all module constants unchanged; strict-xfail decorator AST unchanged. No full suite/budget rerun. All owned processes stopped.

## Owned files

- Packaging/speculative_voice_history_gate.py
- Tests/Performance/test_console_tab_sync_source_ownership.py
- Tests/Performance/test_speculative_voice_latency.py

Independent review is assigned by parent. `artifact-files.json` enumerates exact reports/logs/proofs and hosted input with hashes. Scratch profile databases excluded.
