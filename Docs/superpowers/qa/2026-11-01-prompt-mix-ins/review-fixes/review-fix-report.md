# PR3346 review fixes

Fix base: `963fe12e87ae4d2fb415bff3814943ca448cb0e0`. Target dev remains `2acbcd42f3e7b07fd122f5f6dc6b22e7385f10c5`. Branch `codex/prompt-mix-ins`. TASK-38544 remains In Progress for controller scoped re-review. No push, merge, full suite, helper agent or Qodo request.

## Findings fixed

**I1 — truthful checkbox state.** The control proposes an intent while retaining the projected enabled value. Silent restoration cannot emit another toggle. Accepted actions also reconcile rows and enabled count before a dirty-preserving refresh; a separate rendered-value map preserves truthful repeated clicks without advancing the revision-pinned intent. Dirty refusal and failed/stale actions reconcile surviving controls against their exact owning session without replacing the editor or refreshing a stale intent. Mounted regressions cover Keep editing, Save, Discard, source failure and changed-revision refusal; draft and preference values are both asserted.

**M1 — accepted-only usage.** Capture is now always provisional, including Preview. Only last-used revision facts publish at durable/ephemeral acceptance; queued usage publishes when custody admission succeeds, including supplied received-input configurations. Current-choice Regenerate/Edit/Continue publish after operational receipt acceptance. Original Retry and queued drain never refresh current selections. Publication reads no source, skips unavailable preferences and never re-adds removed selections. It leaves names/placement/order and live intent revision unchanged. Workbench project/save observe layout changes with their existing revisioned CAS, so old layout intents are rejected while accepted source witnesses stay current. A revision already used is not shown as updated since last send merely because its placement is first observed. Optional save scheduling failure uses existing unsaved-settings recovery rather than retracting acceptance. ADR-286 records timing.

Native regressions cover hook refusal, unready provider, real SQLite commit abort followed by Retry preparation, eventual version publication and correct wire placement, fresh durable and temporary chats, failed/successful queue admission with direct and supplied snapshots, original Retry with unsupported current preferences, and frozen queue drain after removal/unavailability and source deletion.

## Hosted preflight repairs

- Register the four installed mix-in tables in the SQL allowlist.
- Pin all three actual mix-in indexes against a real 200-revision/attempt SQLite corpus with `sqlite_stat1` absent. Exact digest lookup and the real message DELETE's foreign-key SET NULL searches choose their indexes without `INDEXED BY`. The fourth reported name (`idx_prompts_`) was a literal prefix in recovery selection code, not a created index. Recovery now identifies index statements structurally; existing migration/refusal/rollback checks remain green.
- Replace new post-await `query_one` calls with bounded query/first access and keep currentness guards. No worker census repin.
- Run ArtifactSelected processing in a worker. Mounted event-route tests drive the real capture policy dialogs through conversation/global confirmation and Cancel, proving the message pump remains responsive. These use the existing detail callback seam to exercise the actual nested wait, not a direct open_artifact test worker.
- Render the Mix-ins count label as literal Rich Text; no markup census repin.
- Review every changed diagnostic statement against pinned dev before regenerating its inventory. Controller/import mix-in outcomes are constant strings; keyword changes remove names/IDs/exception text. Remaining category arguments are fixed entity labels and exception class names. Three ordinary-prompt path diagnostics only re-indent and remain explicitly legacy-unreviewed. No new sink topology. Exact statement diff and review are retained.

## Verification

Exact current covering argv and exit records are in `commands.json` and `*-result.json`; logs are retained, including red and test-setup iterations. The red transcripts with trailing whitespace are losslessly stored as `.log.json` (`raw_text` plus original SHA-256); original raw logs remain in the private task directory.

| Check | Result |
| --- | --- |
| Native mix-in Retry/admission/journey modules | 136 passed; one new native-commit control initially tried a fresh Send over a paused preparation. Corrected to real SQLite abort and the existing Retry preparation action: that exact node passed (1). No product change was needed for this test correction. |
| Mounted Console/Library mix-ins and capture policy dialog modules | 74 passed |
| Final acceptance/layout/persistence controls | 23 passed (overlap with module checks); real same-intent usage flush/reopen and held-save/new-selection race use separate durable SQL revisions and converge to the newer selection |
| Final I1 success/refusal/error/repeated-click controls | 6 passed after rendered-value/count reconciliation; overlaps the 23-node selection |
| Operational DB, mix-in recovery, installed schema policy | 44 passed |
| Remaining prior final covering nodes: fork module, archive export privacy, public fork fields, CSS bundle guards | 35 passed |
| Prompt recovery/migration/refusal/rollback subset | 8 passed, 74 deselected |
| SQL allowlist, index plan census, worker contract, markup census, diagnostic inventory | All five passed; exact logs retained |
| FIX_BASE per-file Ruff | 236 baseline / 236 current, zero introduced diagnostics |
| AST duplicate class methods | No new shadows in 14 changed Python files; property setter/deleter pairs excluded |
| Formatter | No newly failing files versus FIX_BASE; three formerly clean files formatted; new test blocks formatted separately to preserve inherited debt |
| Git whitespace | `git diff --check` passed |

`covering-map.json` maps every node selector from the prior 57-case final covering command to these completed module runs or the remaining covering command. Already-passing unchanged nodes were not rerun merely to recreate an aggregate number. This pass introduces 21 behavioral/plan cases; the fixture/source/API setup corrections in early logs are not product failures. The initial I1 controls and hook/provider M1 controls failed against unchanged product before fixes. The earliest provider test had an incorrect fixture attribute, corrected before its red verification. Early queue-drain control incorrectly started a chain without an existing terminal turn; the native coordinator correctly refused it. The final control starts behind an actual completed send. The first AST audit used system Python3.9, then ran successfully with the project's Python. None of those initial setup failures is represented as passing evidence.

## Provenance and limits

Current source/evidence manifests describe this review-fix candidate. `integration/live/final-source-sha256.json` belongs to focused port commit `5dac2bb0207285eca67d571565a3ee88113e4a84`; the verifier now checks it against that captured commit, never against changed current files. Earlier whole-journey `247c9eb` and Preview `d7744a78` captures remain historical mixed-ancestry evidence. Missing historical commit objects are explicitly UNAVAILABLE, while current filesystem/Git bytes remain strict. A depth-one `--no-local` clone at initial fix commit `d981abb9aa37d4c4156f65effc29a482c3d315fe` verified all 720 then-current records with all three historical commit objects absent and no object alternates; exact proof is retained. That proof tests the unchanged verifier, not the later follow-up source behavior, which is qualified by the 23/6 controls and current manifests. Historical raw captures/hashes were not rewritten.

The current pass qualifies mounted UI, native adapter/SQLite and targeted guards. No new actual-terminal whole journey, external provider/cache savings, OS font coverage or broad FD qualification is claimed. Earlier target-dev baseline failures and aggregate FD warning remain as documented in integration/report.md. No new schema allocation: ChaChaNotes86 and Prompts6 remain unchanged. Controller performs the scoped re-review and publication.
