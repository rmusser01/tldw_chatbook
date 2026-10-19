# Console browser correctness at approved Quit

Status: implemented and verified locally; standalone correctness PR and merge after CI authorized by the user on 2026-10-18. TASK-34406.28 remains In Progress pending integration. No new latency cohort or speed gain is claimed.

## Problem and repair

Current-dev causal controls reproduced two ownership gaps. First, approved Quit disposes the runtime before awaited audio/config cleanup, while an original cache or flat-search browser worker has not yet received cancellation. Its unchanged loop-owner guard correctly refuses the disposed publisher and the UI logs a captured-source error. Second, actual runtime drain detaches the owner from the App while the screen can remain running; checking only `runtime._disposed` then loses the terminal signal. Both detached-route controls executed two original queries after teardown.

Approved Quit now synchronously cancels only the original cache and flat-search worker groups after the successful terminal fence and before cleanup yields. The existing manager retains exact issued Tasks and native callbacks for the later physical drain. Flat search is included in that drain and supplies a callable, avoiding an abandoned precreated coroutine. Late read admission retains the runtime-disposed check and also consults the same App's existing permanent `_shutting_down` flag, including after detachment. An absent flag defaults false, preserving ordinary no-runtime/custom dispatch.

`tldw_chatbook/UI/Console_Modules/console_browser_read.py` is byte-identical to dev. Native source/freshness refusal, `_BrowserSourceChanged`, logging, owned-handle retirement and borrower preservation remain unchanged. No logger suppression, cancellation/source-error reclassification, permission cache, new authority/owner, timer or deadline relaxation.

The repair transparently reuses the useful correctness subset of the earlier unmerged current25 experiment, then closes its newly reproduced post-detachment hole. Its old branch and all other losing candidates remain untouched. This is separate correctness work after the user's explicit priority decision, not a renamed latency win.

## Verification

Base: dev `69771c1811e734421f0bdf31ed90969f8c1f0c63`. Verified product head: `709522c78070c74c0bd5bf8a29118ec74c0b89fd`.

- Initial causal RED on unchanged dev: six failures across cache/flat queued/native/late cases, two Stay passes. The original loop-check witness observes disposed runtime without issuer cancellation; captured function/App/local-service identities remain current. This establishes the controlled ordering failure, not the exact predicate history of the earlier rejected live packet.
- A test-observer frame error contaminated an earlier run; that run remains rejected. The corrected witness and original assertions/deadlines are preserved separately.
- Same-tree baseline proves the flat worker group was omitted from finite drain and four existing search fixtures failed raw-source selection. Their private-profile/valid WorkType and real-manager fixture migrations keep all original assertions.
- Post-detachment RED: both routes make two original queries after actual runtime drain/detachment. This was outside the earlier74-case coverage. The unused live-r1 chain was retired before any App or API run.
- Final targeted run:76 passed, zero failures/errors/skips in272.02s. Covers ten Quit scenarios plus actual native created/borrowed handles, source drift, custom/memory/error ABI, worker draining, App Quit and connected searches. All787 protected-run source entries remain equal; home/network guards record no violations.
- No new Ruff findings versus150 inherited findings. New controls are lint/format clean; seven modified Python files parse. Legacy whole-file formatting is not claimed.
- Independent source and complete live-chain reviews clear the final scope.
- One new unprofiled actual Enter→DeepSeek Flash exact reply→Control+q cycle passes all original settings/expanded config/ordered27schemas/headers/trace/zero-checkpoint/source/strict-log/Quit/unmount/exit/PID-port checks. There are no extra shutdown errors; the single inherited app-log-display setup error remains. All8618 tracked Python files and frozen helpers remain unchanged. Driver/server exit zero, observed owned processes absent and port9331 closed. Profiling and the acceptance timing cohort are disabled.

No timings or I/O counts from either rejected current27 diagnostic are attributed or reused. The final live correctness check is a new-version validation following causal code changes, not a replacement or retrospective qualification. The earlier894ms figure remains an earlier qualified baseline, not a newly measured current-dev result.

## Structural efficiency assessment

Some earlier changes remove real work even without a consistent Send win:

- Already in dev, optional unpublished raw-pin evidence changes isolated snapshots from1080→1048 open attempts,1758→1678 info entries and6→4 metadata snapshots. This is a scoped3% open-attempt reduction; actual pin/source/parent/pause and retirement checks remain. See TASK-34563.37. The separate36-attempt causal fixture is not additive to that32-attempt delta.
- Conditional imports for absent providers and eliminating an intermediate controls rebuild remove unnecessary work. Their evidence does not establish a consistent Enter-to-HTTP gain.
- Permission entry reduces acquisitions12→4, with a provenance-complexity tradeoff; fewer acquisitions alone do not make the whole design simpler.
- Native-read scheduling redistributes work rather than proving less I/O. Current26 repeats an older Summary-refresh seam and lost its Enter-to-HTTP comparison.

The old1064-call figure is metadata work (stat/lstat/statvfs), not1064 opens. Open attempts are not unique files, successful opens, bytes or a complete kernel/SQLite/child-process census. The benchmark's whole-source census also reads thousands of Python files before the App starts and after Quit; those source checks are outside the measured Send window. The evidence does not justify claiming thousands of redundant Send opens have been removed. Optional unpublished evidence and duplicate pure work within one finite checked operation remain better targets than a new global freshness cache.

This Quit repair prevents invalid late reads irrespective of Send timing. It does not establish a first-send speed gain. On 2026-10-18 the user authorized a separate correctness PR and merge after review and required CI. That exception applies only to this Quit repair; earlier latency candidates remain unmerged under the original consistency gate. No speed gain is claimed.

## Evidence and remaining work

External evidence in `pr-review-wave2`:

- `quit-correctness-current28-causal-red-r2.json`, `quit-correctness-current28-connected-baseline-receipt-r1.json`, `quit-correctness-current28-detached-red-qualified-r1.json`.
- `quit-correctness-current28-green-qualified-r2.json`, `quit-correctness-current28-statics-r2.json`, `quit-correctness-current28-source-review-r2.md`.
- `quit-correctness-current28-live-frozen-chain-r2.json`, protocol/review/seal, `quit-correctness-current28-live-qualified-r2.json`.
- `postsave-current27-prior-efficiency-review-r1.md` and both current27 rejected receipts.

Integration: rebased cleanly onto dev `768d58b7564bb2ea6e55a7e0e1f40dbc4016d8fa` (an unrelated notifications migration); all three repaired product files and the unchanged native source-guard module remain byte-identical to the live-verified product head above. The ten-case Quit module and exact flat native-drain case join the required UI gate, with two census entries appended and its floor raised to326. Rebased integration head `779c5eb28957a75b0a1e03d32a09ca081e33b937` passes182 targeted checks (76 connected,67 census checker,39 workflow) in179.63s, with zero failures/errors/skips, all789 protected source entries unchanged and no home/network guard violations. Required CI and review threads remain prerequisites to merge; this is no additional live or latency result. Current-dev latency profiling remains open. No full suite, Windows/native Terminal, populationp95, OS-cold-cache, provider breadth, separate real-home census or universal native-worker-retirement claim.

ADR required:no new; existing ADR126/230 ownership and retirement. General ADR231 polling remains Proposed.
