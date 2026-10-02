# Agent orchestration burn-down review — 2026-09-29

Status: all 13 tasks completed, independently reviewed and rebased; ready for PR review.
Implementation base: dev `64579cce2c8dc64053fb50c00eb4f59b56716b01`.
Integration base: latest fetched dev `31d4f9b76492120706ba8e3ad7d355f1aa4e0273`.
Delivery: PR #2918 against dev; no merge is claimed.
Branch: `codex/agent-orchestration-burndown`.
Scope: all 13 tasks authorized by the user's request to burn down the remaining work.

## Dispositions

| Task | Result |
| --- | --- |
| TASK-32929 | Done: raw custom-endpoint identity and owned primary URL survive the Console/service boundary. |
| TASK-33001.9 | Done: agent sampling uses the shared one-to-many request-key table, including both top_p spellings. |
| TASK-32499 | Done: refused spawns preserve allowance; routing diagnostics include the failing level; both editors share the parser. |
| TASK-32517 / TASK-32639 | Done: absent progress labels skip only the label write; counts/navigation continue and unmount stops the timer. |
| TASK-32497 | Done: live, finished, resumed and historical target displays follow the active frozen target, with honest legacy fallback. |
| TASK-32508 | Done: explicitly authored bounded pre-tool fallback candidates freeze their own URL, model, execution family and parameters. |
| TASK-32477 | Done: reconcile the already merged routing PR's stale publication hold. |
| TASK-32520 | Done: the user-guide conflict-marker regression guard is installed. |
| TASK-33430 | Done: direct sibling discovery/delivery stays within exact live parent/coordinator/chain ownership and existing limits. |
| TASK-22061 | Done: actual mounted navigation-away wake retains the saved transcript and ledger settlement. |
| TASK-33431 | Done: durable saved progress, atomic Save, responsive observation, exact revocation and owned SQL worker cleanup are independently qualified. |
| TASK-33432 | Done: shared progress wake scheduling, exact source intake, native alias admission and Canvas authority are independently re-reviewed. |

## Review findings and corrections

- Deleted registry targets previously fell through to provider-family credentials. Resolution now refuses before credentials or probes when the raw entry is missing.
- Built-in fallback candidates previously stored no effective URL and could retarget after configuration edits. Candidate and primary URL snapshots now use the owned selected or configured/default endpoint; retained continuation uses that frozen target.
- Resumed handles previously displayed original audit identity while dispatching the active fallback. Display now follows the active configuration; original audit columns remain unchanged.
- Active fallback display and duplicate reserved context capture were reproduced through actual inline dispatch and saved trace records, then corrected.
- Loading more saved queues than the runtime cap previously aborted registration and hid pending counts. Deferred queues retain count-only metadata without eviction; unrelated sessions remain usable and explicit inspection retries loading after capacity frees.
- Loaded-report callbacks previously ran under bridge initialization ownership. Metadata publishes after that lock exits; a reentrant consumer regression passes.
- Deferred inspection previously claimed an empty queue. It now shows a typed capacity refusal, performs one worker load per opening and polls memory afterward.
- Closing/disposal during a committed claim previously abandoned the SQL worker and left a prepared generation reserved. The shared claim boundary shields and settles that finite worker before refunding a confirmed preacceptance claim.
- Completed temporary wake claims are deliberately process-local. After Save and restart, lost claim knowledge now fences the old chain for review, preserving manual report reads without automatic replay.
- Saved-report discard previously stalled UI/count polling while waiting for a concurrent SQLite writer. The existing Textual worker now owns SQL; immutable committed views keep observation responsive without weakening mutation authority. Both real contention cases and 56 affected regressions pass; independent re-review passes 43 checks with no findings. Actual native session closure during the same SQL wait then reproduced a second UI stall at synchronous inbox release. Exact-generation revocation and the existing bounded close drain pass mounted checks; independent review subsequently found the same shared-writer wait under replacement initialization, first-fleet preparation, bound-sender cancellation and peer custody. Those custody paths pass 58 independent affected checks. The caller audit then found saved-chat hydration/rollback on the UI loop and a terminal-cancellation spin in the new preparation helper. Preparation now uses the existing finite Chat worker receipt; rollback and both cancellation boundaries pass the final focused checks. Independent lifecycle review approves the correction with no remaining findings: 12 lifecycle/alias checks, 58 capability/lock/peer checks and five final hydration/cancellation checks passed, with separate runner-teardown probes. The five final harness failures are corrected with their original outcome assertions preserved; their focused selection passes 40 checks in 21.66s. The final clean combined affected selection passes all 247 checks in 340.36s; its aggregate descriptor sentinel prompted the resource audit below before closure.
- Four new worker entries retained newly acquired Chat DB connection caches: native preparation, modal preparation, modal discard and child reporting. Real file-backed and actual threaded/mounted probes proved that pool shutdown plus closing the UI handle did not permit recovery participant drain. All four callbacks now reuse the existing finite-worker cleanup helper, preserving borrowed handles, transactions, authority and physical cancellation receipts. Fresh/borrowed and actual participant-drain checks pass; the complete load/append/remove caller audit found no further shipping SQL entry without ownership.
- A new saved-chat source chain previously reused the temporary source bucket after promotion. Source intake now resolves exact ledger lineage and rechecks native owner, pending identity and fences after the await. A live accepted exact-session/chain wake token preserves the old causal source; manual and foreign tokens cannot borrow it.
- Removing a stale completion previously stranded pending progress. Shared membership remains until both sources are empty; preparation admission also excludes concurrent old/new source aliases in one native session.
- An initial source alias repair also changed Canvas chat-data scope. The real NativeConsoleCanvasAuthority reproduced the refusal after Save; the final repair separates work-source metadata from current native Canvas/session authority.

## Independent review and evidence

Routing identity, sampling, timer, refusal diagnostics, parser, documentation guard and scoped peer authority/privacy received an independent read-only review with no actionable findings. Fallback/target review found three issues above; the independent re-review approved their corrections and ran **50 tests in 19.90s**.

The durable reviewer ran **20 tests in 124.58s**, found no data-loss/ownership issue and identified the UI contention defect subsequently corrected and re-reviewed. Final progress qualification passed **72 affected mixed-source/completion/schema checks in 142.45s**; independent final re-review passed **eight cases in 41.94s** with no remaining findings. Earlier claim-custody and restart regressions remain part of that evidence. These are separate selections and are not summed with overlapping earlier runs.

A broader final repair selection passed 112 checks and exposed four missed Settings Save clicks in an unstyled harness; production CSS plus explicit scroll/click qualification now passes all nine routing/Settings cases, including ordered fallback-list persistence. The fresh combined routing, sampling, Settings, fallback, mounted target/timer and guide/token selection then passes **116 checks in 152.91s** (`/private/tmp/agent-burndown-repairs-green.log`). A broader schema/ledger selection passed 135 checks and exposed two stale recovery version assertions; a fresh combined schema, recovery and wake-ledger run passes **137 checks in 70.92s** after using the current version and its genuinely future successor (`/private/tmp/agent-burndown-schema-ledger-green.log`). These fixture corrections preserve production gates and original outcome assertions.

The fresh actual navigation-away wake regression passed **one case in 32.02s** after scheduler integration. Root primary identity/URL/child/custom-hosted qualification passed **nine checks in 18.45s**. Messaging receipts and the two previously externally marked gateway/bridge nodes passed **22 checks in 14.67s** using committed bootstrap-profile markers. The markers preserve the real selected config source; production recovery gates and original assertions remain intact.

The final durable affected selection passes **247 checks in 340.36s** (`/tmp/task33431_final-affected-green.log`). Following the confirmed worker-cache fixes, the directly affected native/modal/hydration selection passes **82 checks in 173.36s** (`/tmp/task33431_final-worker-lifetimes-green.log`) and the final durable/threaded-report/tool/queue selection passes **74 checks in 24.52s** (`/tmp/task33431_final-child-report-green.log`). Final independent cache review approves all four entries with **13 checks**, actual mounted/threaded physical-drain probes and the complete shipping sink audit. These selections overlap and are not summed.

## Latest-dev integration

Rebased cleanly onto `6423c4fbd1460952dd8cb5df04e51c4e0e7e0d0b`. Comparing the frozen pre-rebase source manifest shows incoming changes in only two patch files: `Chat_Functions.py` (new provider dispatch/preset entries) and `ChaChaNotes_DB.py` (uncorrelated conversation-search FTS). The durable schema and reviewed worker/capability/queue/wake code retain their reviewed changes.

A fresh combined post-rebase selection passes **217 checks in 87.14s**: incoming gateway/host presets and search FTS, sampling, the nine actual primary/child/custom-endpoint gateway checks, and all four startup census checks (`/private/tmp/agent-burndown-post-rebase-green.log`). Census is 1031/1033 with headroom two and the existing +1/-0 snapshot warning. The final base-relative static scan remains clean across 72 changed/new Python files and ten new Python files; whitespace passes. No further source changes were required.

## Contracts and qualification limits

[ADR-199](../../../backlog/decisions/199-scoped-peers-durable-progress-and-wakes.md) governs scoped peers, ChaChaNotes schema 74 report ownership, atomic Save and progress wakes. [ADR-200](../../../backlog/decisions/200-preset-pre-tool-fallback-targets.md) governs AgentRuns schema 22 fallback snapshots; progress claims extend it to schema 23. Independently frozen recovery catalogs retain historical versions and the linear migrations.

No second scheduler, communication bus, credential snapshots or new dependency was added. Reports remain untrusted context; saved data never restores sender or execution authority. Existing queue, sender, run and automatic-work limits remain in force. The original September 7 audit and earlier reliability waves remain completed; this report records only the newly authorized remaining scope.

Verification is targeted. No full suite, live provider, Windows or native terminal UI qualification is claimed. Inherited whole-file lint/format debt is preserved; changed-code checks and derived UI token/CSS checks govern this patch. Existing dependency, temporary-directory cleanup and module-census drift warnings are recorded separately from correctness results. The earlier aggregate 282-descriptor sentinel is not classified as wholly inherited or wholly explained by the four confirmed leaks; the final 82- and 74-case selections emit no FD-growth sentinel. Stale pytest garbage-directory warnings remain; no broad aggregate resource sweep is claimed.

Final frozen-code static qualification: 72 changed/new Python files pass fatal checks; all ten new Python files pass full Ruff and formatting; the full changed-file scan reports zero added-line/new-file findings. Whitespace checks pass. Inherited whole-file lint and formatting findings remain outside those claims.

Size-ratchet measurements are not claimed green. The initial dev baseline already exceeds all three affected pins; this patch adds required native ownership and UI boundary code without raising those pins:

| Module | Pin | Initial dev | Final code | Patch delta |
| --- | ---: | ---: | ---: | ---: |
| `tldw_chatbook/Chat/console_chat_store.py` | 22344 | 22545 | 22771 | +226 |
| `tldw_chatbook/Chat/console_chat_controller.py` | 29367 | 30559 | 30613 | +54 |
| `tldw_chatbook/UI/Screens/chat_screen.py` | 25218 | 25435 | 25450 | +15 |

## October 1 PR CI and latest-dev integration

PR #2918's first CI run found an exact runtime inventory missing the two peer tools, two unregistered indexes and a noncanonical progress timestamp writer. Targeted RED checks reproduced those defects. The writer now reuses `utc_now_iso()` under ADR-173; the exact inventory includes both existing peer tools. Existing populated repository tests now trace the actual FIFO load and mixed completion/progress abort SQL, assert indexed search without statistics and preserve negative controls. Both indexes are registered in the census. The expanded derived sweep also found the new report table missing from Chat's SQL allowlist; the existing allowlist now covers it.

The MCP renderer CI failure was an unchecked missed Run click. Upstream PR #2910 already fixes it; the rebase includes that correction. Four focused renderer cases pass in 8.52s. No production renderer change was needed.

Rebased the two delivery commits onto dev `31d4f9b76492120706ba8e3ad7d355f1aa4e0273`; lesson conflicts preserve both sets of evidence. Independent review of the overlapping Console/runtime/provider/routing and frozen recovery boundaries found one additional P2: the upstream hook-admission early return bypassed proven-preacceptance cleanup, leaving an automatic wake prepared with a reserved generation. The minimal seven-line repair marks only an exact live authorized AGENT_WAKE token whose acceptance has not started. Existing cleanup aborts/refunds it. Manual and copied tokens cannot borrow that authority; uncertain or accepted attempts keep their existing fences. The real plain and agent gateway regression proves zero provider calls before refusal, no reserved generation, one retry after the hook clears and no duplicate delivery.

Fresh evidence:

- Exact timestamp, runtime inventory and populated claim-plan regression: **3 passed in 5.73s** (`/private/tmp/pr2918-ci-corrections-green.log`).
- Affected messaging, peers, installer, budget and recovery/schema modules: **145 passed in 58.50s** (`/private/tmp/pr2918-ci-affected-green.log`).
- Fallback, sampling, saved close/hydration, mounted durable progress, design tokens and guide guard: **147 passed in 271.09s** (`/private/tmp/pr2918-rebase-lifecycle-green.log`).
- Hook repair and manual/foreign authority: **6 passed in 43.72s**; existing readiness, acceptance-write failure and completion nonreplay guards: **3 passed in 11.51s** (`/private/tmp/pr2918-hook-wake-refund-targeted-green.log`, `/private/tmp/pr2918-hook-wake-budget-guards-green.log`).
- Independent integration review approves the hook repair and scoped overlap review: 37 boundary/recovery cases, a separate seven-case lookup/cache selection, both new real-gateway cases and three existing wake guards passed. These selections overlap and are not summed.
- Frozen-code static scan: **75 changed Python files fatal-clean, ten new Python files full Ruff/format-clean, zero added-line/new-file findings and clean whitespace** (`/private/tmp/pr2918-final-static.log`).

The diagnostic inventory was refreshed only after reviewing the three exception-type-only warnings and the controller's unchanged constant warning. No communication body, URL, path, raw exception or sink-topology change was admitted by that refresh. Derived CSS, bootstrap-profile, task IDs/readability, table allowlist, index census, workers, timestamps and UI census checks pass with Python 3.12. GitHub must rerun its checks on the published head; this record does not claim a successful remote rerun.

Current size-ratchet measurements retain the pins and disclose inherited overruns:

| Module | Pin | Current dev | Final code | Patch delta |
| --- | ---: | ---: | ---: | ---: |
| `tldw_chatbook/Chat/console_chat_store.py` | 22344 | 22534 | 22760 | +226 |
| `tldw_chatbook/Chat/console_chat_controller.py` | 29367 | 30660 | 30721 | +61 |
| `tldw_chatbook/UI/Screens/chat_screen.py` | 25218 | 25311 | 25326 | +15 |

The earlier qualification limits remain: targeted checks only, no full suite/live provider/Windows claim, inherited whole-file static debt, stale pytest garbage-directory warnings and no broad aggregate resource sweep.
