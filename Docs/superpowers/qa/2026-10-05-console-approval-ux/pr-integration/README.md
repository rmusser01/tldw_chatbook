# Approval PR integration onto dev

The candidate starts from dev `6feb84c1d2203bc3c6d0eecd2823f4e2409b4b4e` and carries only the reviewed Console approval patch from `0f8e97fe16..686cf8d84d`. The unrelated predecessor Eval/recovery change is excluded. Original implementation/component receipts and source hashes remain historical; current-dev integration evidence is kept here. The original reviewed and comparison worktrees are retained.

Current dev extracted approval request, resolution and definitive completion into `InterruptRoundHost`. The integration preserves that ownership and forwards the reviewed capture, receipt, settlement and finishing display hooks to the actual owner. Independent automatic-integration and controller/host reviews found no material drift. The source/evidence hashes and 84 admitted checks are recorded in [controller-host-report.md](controller-host-report.md).

Shared ADR-index and testing-lesson edits retain both histories. Approval tasks moved from 34411â€“34416 to 34564â€“34569 because older dev tasks already own 34411/34412. Existing dev tasks retain their IDs; approval dependencies preserve order. Full reachable-object and 44 live-worktree scans found maximum 34563, and remote content checks found the new IDs unused before renumbering. See [task-renumbering.json](task-renumbering.json).

## Verification before the final rebase

Focused private-profile checks passed: host/controller 84, MCP permissions 90, interaction 25, mounted UI journeys 4 and recorder contracts 16. Backlog IDs/files, UI gate census, rebuilt CSS sync and the refreshed diagnostic inventory passed. The diagnostic inventory update accounts for three moved or re-indented warning statements with no text, count or sink changes.

The Details file initially failed after a fixed polling loop expired before a requested worker started. The bounded harness repair waits for that existing worker, then strictly checks the painted page and displayed text. Focused 1 and full affected-file 19 checks passed; Ruff and formatting checks passed. The independent repair review traced Textual worker completion and UI delivery and found no assertion weakening, retry, new request, product change or guard change at repaired test SHA256 `a40a19fdfbc8f46a5571defe95e562e252530a502116dd51b86119185bf4cbb5`. See [details-harness-report.md](details-harness-report.md) and its preserved green receipts.

Dev subsequently advanced to `cddc89d3e780b27392549b58ff9134b64e7d5907`. Its only overlaps with approval work are the appended testing lesson and a disjoint established-session registry-read guard in `chat_screen.py`; both are retained by the final rebase. Post-rebase evidence is recorded below after verification.

## Qualification limits

Native/browser presentation, calibrated end-to-end timing, the full geometry/theme/Inspect matrix, input-queueing measurements and actual Windows dispatch remain unqualified. The 100 ms feedback and 200 ms actionable-card p95 targets are unverified; no speed or freeze claim is made. Historical Windows root-pin refusals and existing governance/static/startup findings remain attributed to their baselines. The controller's 61 inherited Ruff diagnostics match the original dev baseline and are not presented as clean.

No full test sweep is authorized or run. This PR is a draft, and all six tasks remain In Progress. ADR-221 and the approved spec/plan govern the interaction contract.

## Post-rebase verification

The single approval commit was rebased onto dev cddc89d3e7. The only conflict was the appended testing lesson; both sections were retained. The automatically merged ChatScreen preserves dev's registry-read guard and the approval hooks. Own-code comparison against the pre-rebase commit changes only that disjoint five-line dev guard.

Fresh protected private-profile mounted journeys passed 4 tests in 24.52 s and action ownership passed 25 in 12.69 s. CSS bundle reproduction, task IDs/files (5049 records), UI gate census (152 files) and diagnostic inventory (655 owners; 1445/56/7626 calls; 16 sink files) all passed. Seven preserved post-rebase logs and their current-worktree/canonical Git hashes are recorded in receipts.json.

The earlier 84 host/controller checks remain pre-rebase evidence. All five owner/test Git blobs are exactly unchanged; Git's Windows checkout converted their LF bytes to CRLF, and LF-normalized SHA256 still equals each earlier source hash. All eight associated private evidence files remain raw-byte identical. Current raw and canonical Git source hashes are in [post-rebase-source-hashes.json](post-rebase-source-hashes.json). The Details repair remains at its independently reviewed hash. The final metadata amend changes only QA records and receipts; no source is changed or tests replayed.

Historical implementation hashes and receipts are not relabeled as current platform or timing qualification.
