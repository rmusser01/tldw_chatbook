# Console configuration worker verification

Task: TASK-34563.14. Base: a7f93c6ff41dfd44fb45ee4918a99225b87b8ce9.
ADR: [ADR-225](../../backlog/decisions/225-console-send-preparation-and-io-ownership.md).
Plan: [configuration capture worker](../superpowers/plans/2026-10-07-console-configuration-worker.md).

## Result

Eligible stock Console configuration now runs native reads off the input loop through one retained Chat-owned operation. Mounted view values are selected once; the existing complete snapshot producer remains shared with screen-free runtime capture. Controller and runtime observe the same physical read records, including the extracted hook lifetime mechanism. Original connections retire before cancellation, Close or disposal completes, and changed sources cannot publish a stale result.

Send reads already-published plugin metadata through the exact existing owner instead of constructing an empty cold facade. A private supplied-owner input preserves the original synchronous/custom getter contract. Replaced getters and factories decline the optimization. No LocalSkillsService API, construction lock, scheduler, persistence mode or permission cache was added. Cold scope/trust and custom sources retain their prior affinity; cold-start responsiveness is not fully qualified.

Saved-turn acceptance failure still stops dispatch and keeps the draft. Temporary chats, WAL/NORMAL, required consent/checkpoint/effect order and awaited history remain unchanged.

## Final source-current checks

All runs were sequential, after both implementation lanes froze, using private profiles and the original bounded native runner. No full sweep or external provider call was performed.

| Run label | Result | Scope |
| --- | --- | --- |
| configuration-worker-domain-final | 31 passed | Selected-value detachment, original Workspace and skill reads, source replacement, cancellation, shared capture parity and custom affinity. |
| configuration-worker-adapter-final | 24 passed | MCP source/loop phases, stale-owner refusal, permission semantics, mounted wiring and exact draft-change protection. |
| configuration-worker-lifecycle-final-clean | 35 passed | Generic/hook lifetime, Stop/Close/disposal, actual Enter/button input and natural frames, saved-turn refusal and native commit ownership. |
| configuration-worker-plugin-final | 1 passed; 1 failed; 1 setup error | Built-in skill capture passed. Two native plugin cases stopped at the existing host qualification gate before the tested behavior. |

Thus 90 core controls and one additional built-in control passed, with no skips. Native plugin payload and later-enablement integration are **not qualified on Windows**: unchanged Plugins/runtime_owner.py requires 64-bit macOS and raises `plugin runtime platform is unqualified`. Its normalized source hash matches the base: 70d8bb629dd8bfcc98f399e18984a62373179e317c78b9f44fa63ee51e731817. That gate was not bypassed.

### Mounted feedback

The actual eager navigation/Send route used the canonical private profile and production bundled stylesheet at 120x40. An independent helper held the original admitted Workspace SQL reader while delivering a driver key. The observer inspected actual input mutation and supplied natural compositor updates; it never forced a render.

| Route | Input mutation | Natural frame containing changed input |
| --- | ---: | ---: |
| Enter | 1.197 ms | 19.719 ms |
| Send button | 1.235 ms | 5.173 ms |

Both inputs and frames occurred while the native reader was held. The newer draft remained intact, and its actual native connection/lease retired before fixture teardown. These are two observed headless liveness samples under a bounded 500 ms hold, not physical terminal write/flush or whole-Send timing qualification. The overall 100 ms feedback and under-one-second action-to-adapter goals remain unqualified; no total-latency reduction is claimed from this worker move. Historical .7 samples are not a matched baseline for this change.

## Review and intermediate findings

The unchanged base first failed the original Workspace worker-affinity assertion. Integration then found an empty skill fixture and a changed refusal reason code; the fixture now imports a real skill and the existing refusal code is restored. Source review replaced module-name callback heuristics with exact stock provenance and qualified the directly bypassed plugin getter.

Mounted diagnostics separated two fixture issues from one product issue: a noncanonical MCP store was correctly rejected by native binding; the stock cold plugin factory then kept capture synchronous; finally an omitted app stylesheet placed the text below the viewport. The test now uses the configured profile, the real stylesheet and a visible-region precondition. Route-specific workspace names avoid shared-profile collisions. All temporary diagnostics were removed before the final lifecycle run. Tests retain their original deadlines and behavioral assertions. The native fixture traps are recorded in lessons-testing-evidence.md.

Final independent source review found no remaining actionable correctness issue in this scope. Native process trees and diagnostic tasks retired normally in all four final runs, with no forced retirement or identity overflow; private profiles were removed. The final lifecycle run recorded one diagnostic process-lookup race; the other three runs recorded zero. Separate in-test assertions establish actual original reader resource retirement. Final logs have no unawaited-coroutine, destroyed-task, ignored-exception or unconsumed-Future signature. Warnings are existing dependency/async-fixture deprecations and pytest's xunit2 record_property warning; XML properties were read directly.

## Static checks and remaining work

AST and diff checks pass. There are no introduced Ruff findings or formatter transformations. Existing Ruff findings remain: controller 60, session 10, app wiring 1. Existing formatter differences remain unchanged.

The module-size ratchet is still red: controller grows from 30,465 to 30,576 lines against 29,299; unchanged store is 22,814 against 22,344 and interrupts 6,520 against 6,471. Caps were not raised. TASK-34563.14 remains In Progress because full repository DoD/static hygiene and the two native plugin integrations are not established.

Early unconfigured UI receipt, exact draft/staging revision transfer, queue admission/continuation simplification and clean whole-Send timing remain in the larger approved architecture work. This step establishes responsive eligible configuration capture and retained native ownership; it does not complete that larger goal. No primary checkout edits, push or merge.

Source hashes, selected nodes and receipts are retained locally under `.superpowers/sdd/2026-10-07-console-configuration-worker/` (final-audit.json, final-static.json, final-selection.json and per-run source records). Runner logs/XML/results/custody are in the shared preparation checks directory under the labels above.
