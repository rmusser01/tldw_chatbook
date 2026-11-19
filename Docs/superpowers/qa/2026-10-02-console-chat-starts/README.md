# Console chat destinations and starts — verification record

Implementation is complete on `codex/console-chat-starts`. Both task reviews and the final whole-branch review/fix gate are approved: the final scoped review closed all four findings and found no new breakage. The user chose push and PR; its base choice remains pending because the original local base is absent from GitHub.

`new_chat` supports `destination=same_workspace|casual` and `mode=draft|start`, defaulting to same_workspace/draft. Start requests one immediate bounded background turn; source workspace/chat/composer remain in place. Destination defaults, fresh private state, exact grants, revision-fenced drafts and conservative recovery follow ADR-211.

Final feature checks: 140 amended-owner cases passed before the last capture refinement, followed by 14 passing capture/admission cases on that refinement; postcommit lint, new-file formatting and five ratchets passed. The historical broader group remains recorded as 905 passed / 4 failed. The subsequent [baseline remediation](../2026-10-03-console-baseline-remediation/README.md) verifies all 21 previously failing nodes and all seven complete affected modules: 443 passed, with seven warnings. The remaining fork and reasoning-replay test doubles are repaired. FD-growth warnings, guide conflict markers and streaming usage uncertainty remain qualified. No full suite, general resource cleanup or OS power-loss verification is claimed.

- [Live Console qualification](live-qualification.md): real app/local-provider behavior, defects and limitations.
- [Full terminal frames](terminal-captures.txt): selected unmodified 235×52 PTY captures.
- [Curated database receipts](qualification-receipts.json): disposable test targets, native attempts, direct allowance membership and reservations.
- [Foundation report](task-1-report.md) and [foundation review](task-1-review.md).
- [Integration report](task-2-report.md) and [initial integration review](task-2-initial-review.md): implementation, fixes and verification.
- [Round 1 scoped review](task-2-fix1-review.md): initial findings resolved; manual recovery activity finding carried into round 2.
- [Round 2 scoped review](task-2-fix2-review.md): manual recovery attention resolved; task gate approved.
- [Final whole-branch review](final-branch-review-full.md): complete four-finding batch, exact probes and deferred dispositions; [fix brief](final-fix-brief.md).
- [Final fix report](final-fix-report.md) and [single scoped re-review](final-fix-scoped-review.md): all four findings closed; exact evidence and qualifications.
- [Execution ledger](execution-ledger.md), [global constraints](global-constraints.md), and original task briefs [1](task-1-brief.md)/[2](task-2-brief.md).
- [Baseline extraction/probe](final-fix-baseline-probe.py.txt), [whole-function origin](final-fix-overall-origin.txt), and exact compressed baseline manifest under `source-archives/`; [source manifest](artifact-source-manifest.json) carries hashes.
- [Initial review probes](initial-review-probes.py.txt) and [manual-recovery probe](fix1-recovery-probe.py.txt): exact reproduction sources.
- [Post-review refusal receipt](fix-refusal-receipt.json): durable outcome and no replay.
- [Manual-recovery receipt](fix2-recovery-receipt.json): actual human Send completes and clears live blocked activity while preserving launch history.
- [Final approval receipt](final-fix-approval-receipt.json): real disclosed session grant and explicit override create a casual draft.
- [Automatic approval review record](fix-approval-review.md): broad replacement rejected; scoped edits accepted.
- [Verification output excerpts](verification-output.md), [source-log manifest](verification-log-manifest.json), and full logs under `verification-logs/`. Readable copies omit trailing whitespace; byte-exact `.gz` copies reproduce the original source hashes.
- [Formatter baseline manifest](format-baseline-manifest.json): exact old-file debt snapshots used by scoped ratchets.
- [Execution rulings](rulings.md): decisions, evidence and costs.

All implementation and review gates are closed. Historical commands retain their original scratch paths; readable artifact links use this maintained directory, and byte-exact sources are preserved under `source-archives/`. These artifacts contain only the disposable verification profile; no user database or credentials were copied.
