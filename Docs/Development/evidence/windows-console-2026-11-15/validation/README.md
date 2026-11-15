# Current Console validation — 2026-11-15

Source base: `cc6006fcf59328353004206bba8724c399216749`. Each JSON receipt records its exact command, head, source-diff SHA256, exit and private fixture root; its neighboring patch reconstructs tested changes. These are targeted functional/static results, not performance measurements or a full-suite claim.

This directory relocates only the named current-task receipts, patches and raw logs from `pr3166-final-integration-2026-10-31`. Historical records there are unchanged. Receipt commands and source patches retain their original paths verbatim; relocation changes evidence location only. Raw logs remain unedited and may require explicit force-add because of ignore rules.

| Receipt label | Result | What it establishes / remaining limit |
| --- | --- | --- |
| current-after-close | 1 failed, 92.36s | Initial `_show_tabs` failed before the gate; composer existence was not startup publication completion. |
| current-after-close-observed | 1 failed, 14.71s | Retained initial observer attempt; not a passing functional result. |
| current-after-close-observed-r2 | 1 failed, 46.59s | Retained revised startup observation; not a passing functional result. |
| current-after-close-fixed | 1 failed, 102.55s | Startup prerequisite moved earlier; actual Close later logged RuntimeError originating in finalization. Exact raised guard was not captured, so no causal attribution is made. |
| current-after-close-close-trace | 1 passed, 33.05s | Instrumented repeat passed; it does not explain the preceding intermittent failure. |
| context-close-red | 2 failed, 9.41s | Causal original context-read controls exposed close outrunning native retirement and new context-read admission after fencing. |
| context-close-green | 2 passed, 7.04s | Existing physical retirement and close-admission controls pass after the narrow lifetime repair; not proof of the earlier mounted failure's cause. |
| tabs-current-after_close | 1 passed, 39.20s | Real Close and keeper-only tab publication before unrelated held-reader release. |
| tabs-current-coalesced | 1 passed, 57.17s | Coalesced real publication while unrelated reader remains held. |
| tabs-current-same_session_drain | 1 failed, 59.63s | Retained original owner-observer failure after read-drain ownership moved. |
| tabs-current-same-session-owner-fixed | 1 passed, 31.95s | Correct defining drain owner observed; same-session non-cancelled physical retirement and real tab publication pass. |
| current-ci-static | 123 passed, 1 failed, 383.82s | New PluginRestoreAdoptionModal lacked wide-tier classification; remaining classifications/checks passed in this selection. |
| current-ci-boot-runtime | 9 passed, 1 failed, 470.49s | Ready census 1041 exceeded pinned 1040; reported new setup-facts/native modules. No ratchet increase accepted. |
| endpoint-import-red | 1 failed, 2.96s | New endpoint-routing import-closure regression discriminates eager setup-facts loading. |
| context-endpoint-connected | 89 passed, 3 failed, 154.60s | Cold-reader test invalidated stock identity with a fail monkeypatch; both compaction cases expected version reads without seeding possible memory. They fail before resume. |
| context-fixtures-fixed | 6 passed, 49.94s | Stock-reader observation and genuine memory-presence fixture preserve original qualification and compaction-resume contracts. |
| current-ci-final-contracts | 11 passed, 1 failed, 367.53s | Remaining `test_the_workbench_send_action_agrees_with_the_composer_send` fails its existing control-projection deadline (`Console control projection did not complete`). Not a passing workbench result. |

The original reader/native/source checks, real Close/DOM assertions and physical retirement requirements remain required. Author results above do not justify a broader stability or speedup claim. Results from later profile/review work belong in their own receipts.

## Cleanup custody

`receipt-roots.json` enumerates all 17 exact receipt-owned roots for future root-agent cleanup. Their existence and process ownership have **not** been checked by this documentation pass; none was deleted here. Confirm no owned process remains before removing any exact root. The initial failed ambient-TEMP launch remains separately recorded in worktree `.tmp-validation-current.log` (exit4 before collection), outside these 17 receipts.

ADR required: no new decision for evidence relocation. Existing ADR-126/230 govern finite preparation ownership; source changes and validation are documented by the task's implementation plan.
