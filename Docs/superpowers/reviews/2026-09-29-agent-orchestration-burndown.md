# Agent orchestration burn-down review — 2026-09-29

Status: all 13 tasks completed, independently reviewed and rebased; ready for PR review.
Implementation base: dev `64579cce2c8dc64053fb50c00eb4f59b56716b01`.
Integration base: latest fetched dev `84247cb8435fcf59b6d8e2d97c6b2f0913934dd0`.
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

## October 1 Qodo review and boot CSS correction

The published `02e581135a` head passed GitHub's PR and UI fast lanes. Its latency job failed the aggregate CSS byte ratchet: 608118/608090 B. Current dev already measured 608088 B, and the new fallback selector added 30 B. No ratchet was raised. A shared scoped `.agents-area` class now applies the same sizing tokens only to the instructions and fallback fields; parameters remain unclassed. Source regeneration removes exactly 30 parsed bytes. Both existing production-CSS mounted widths now assert real fallback text painting. The exact budget/paint/authoring/token/bundle selection passes **17 cases in 57.55s** (`/private/tmp/pr2918-css-budget-green.log`); independent budget plus both mounted widths pass **3 cases in 13.18s**.

Qodo posted four findings on that head, all addressed:

1. Public peer methods now describe arguments, generated queued receipts and expected MessageError refusal conditions.
2. The model-unavailable exception constructor and classifier have complete public type annotations and meaningful Google-style documentation, without altering provider classification.
3. Global close uses existing immutable published membership after setting the store-wide revocation Event. This neither waits for the SQL writer nor grants authority; positive APIs consult the global fence, including unpublished inboxes. Physical close still owns and drains the authoritative dictionary under its existing lock. A forced yielding-values schedule reproduces mutable-iteration failure during actual deferred removal. **This is controlled snapshot robustness evidence, not a naturally reproduced CPython3.12/GIL race.**
4. Peer tool arguments pass through a strict private Pydantic model that reuses `_validate_text`. Exact shape, no coercion, control/Unicode/length rules and original fixed refusal codes are preserved. Capability and atomic allowance admission stay with the existing messenger; invalid tool payloads are refused before invoking it. The strengthened existing boundary selection was RED: 6 failed / 2 passed before this repair.

Fresh qualification: **88 peer/inventory/fallback cases in 49.27s** (`/private/tmp/pr2918-qodo-peer-api-green.log`); **2 startup guards in 25.12s** (`/private/tmp/pr2918-qodo-startup-green.log`), with imports 679/686 and UI-ready 1031/1033; **44 queue cases** and **4 actual mounted disposal/cancelled-disposal/replacement-disposal/durable-reopen cases in 31.68s** (`/private/tmp/pr2918-close-membership-green.log`, `/private/tmp/pr2918-close-membership-lifecycle-green.log`). Independent final re-review approves **22 focused checks**, covering malformed peer boundary, real delivery, private trace/log output, shared allowance, controlled close/replacement and actual HTTP machine-code classification. Selections overlap and are not summed.

Final changed-code verification covers **76 Python files**, with fatal checks clean, all ten new files fully Ruff/format-clean, zero added-line/new-file findings and clean whitespace (`/private/tmp/pr2918-qodo-final-static.log`). The unchanged audited diagnostic inventory passes again (`/private/tmp/pr2918-qodo-final-diagnostics.log`). Existing inherited static/size/resource qualification limits remain. CodeRabbit automatically skipped because `dev` is not the configured default base; its success status is not a review approval. Qodo's actual findings and independent scoped review are recorded above. The follow-up must pass fresh GitHub checks after publication; no merge is claimed.


## Final rebase onto the active-run readiness repair

Rebased cleanly onto dev `84247cb8435fcf59b6d8e2d97c6b2f0913934dd0` (PR #2948). The only reviewed patch Python file changed by this rebase is `UI/Screens/chat_screen.py`, where incoming readiness display stops treating an active run as provider setup failure. Controller submission, slot ownership, native wake priority, caps and acceptance authority retain their gates.

Independent read-only review approves this overlap. Four focused chainless/manual preparing, actual wake acceptance, mounted wake-copy/keypress and held healthy-run readiness checks pass in 100.19s; nine structured active/nonactive rail badge cases pass in 1.94s. The base-relative static scan remains clean across 76 changed Python files and ten new files (`/private/tmp/pr2918-latest-dev-static.log`). These focused selections are separate from the integration qualification below and are not summed.


The combined integration run was stopped for diagnosis after **138 passed / 7 failed in 1147.92s** (`/private/tmp/pr2918-final-rebase-green.log`); it is not claimed green. Five failures expired the held fixture's first-chunk precondition before valid dispatch. Independent observation recorded dispatch at 9.578s and first yield 4.6ms later, normal commit/cleanup and a controller/store/gateway path unchanged from the previously reviewed head. Only preparation now waits up to 15s; all control-action waits retain 5s. Two wake tests use the same preparation bound while paint, ledger and timer settles retain 8s.

The off-viewed wake assertion also assumed an awaited sync acknowledged paint and a fixed 1.2s sleep acknowledged completion. A sync may instead coalesce behind an existing worker. The regression now observes actual Running and completed glyph publication through its existing bounded helper, preserving no interaction and both idle-poll self-stop assertions. Production behavior, deadlines, caps and authority remain unchanged.

Fresh final qualification passes **13 targeted checks in 177.22s** (`/private/tmp/pr2918-final-rebase-affected-green.log`): all four mounted wake cases, both real hook-refund gateways, boot CSS/bundle guards and UI-ready census 1031/1033. Independent review approves both test corrections and passes **all five previously failing controls in 209.78s** (`/private/tmp/review-pr2918-final-controls-green.log`). A private frozen-terminal-publication control reaches real streaming and ledger completion, then fails the exact settled-glyph assertion; a private never-first-chunk control fails its bounded 15s precondition. Disabling only the delivery hook passed because a coalesced tail could still repaint, so that experiment is not claimed as necessary-hook evidence. The testing lesson records this distinction.

Final static verification covers **78 changed Python files**, with fatal and added-line/new-file checks clean, all ten new files fully Ruff/format-clean, owned test-range formatting clean and whitespace clean (`/private/tmp/pr2918-final-rebase-static.log`). The final predicate formatting preserves the tested AST. TASK-33432's rebase qualification is closed through the Backlog CLI with all five criteria checked; all 13 original tasks are Done. The previous published head's GitHub latency job passes; fresh checks must run on this rebased follow-up. No full suite or merge is claimed.


## Incoming compaction schema and recovery qualification

Dev `27e718f01d` ships Chat version 74 and a 73→74 auxiliary failure-reason migration. The preserving rebase keeps that method and SQL byte-for-byte and moves unmerged durable progress to version 75 through a separate 74→75 step. The fleet DDL is unchanged. Exact Chat and shared Subscriptions catalogs include both features; their current schema gates use 75. Frozen AgentRuns histories and current-only Chat recovery policy are unchanged. ADR-199 records this integration under the existing ADR-052 boundary.

The initial affected selection recorded **29 passed / 2 failed in 71.29s** (`/private/tmp/pr2918-compaction-schema-green.log`) and is not claimed green. Both new failures belonged to qualification setup: the primary SQLite owner refuses a read-only URI, and immutable candidate validation of a live WAL database read the old main-file catalog. The rollback test now uses the installed read-only recovery owner after the failed constructor has settled. Shared recovery checks use separate candidates staged by `backup_database`; the core wrong-stamp check runs before subscription tables are added. No production validator changed. Independent observation proves the staged shared catalog's 584 SQL objects, all 715 catalog rows and 523 metadata entries match the installed reference exactly; another staged copy with stamp 74 refuses `unsupported_schema_version` (`/private/tmp/pr2918-review-candidate-differences.log`).

Final affected selection: **31 passed in 82.20s** (`/private/tmp/pr2918-compaction-schema-final-green.log`), including genuine 73/74 upgrades, retained failure reasons, injected partial-DDL rollback/retry, exact catalogs/stamps, durable reopen, atomic Save, FIFO/privacy/capacity and worker cache ownership. Mounted reopen/count/read/discard and promoted-alias native close pass **2 in 18.15s** (`/private/tmp/pr2918-compaction-saved-ui-green.log`). Independent complementary recovery passes **30 in 32.99s**, including all eight dictionary-trigger variants and 17 AgentRuns cases (`/private/tmp/pr2918-review-composed-schema.log`); frozen 18/21/22 progress migration/target checks pass **5 in 1.67s** (`/private/tmp/pr2918-review-frozen-agent-history.log`).

Independent source review finds no actionable compaction/wake overlap issues; **10 cases pass in 82.63s** (`/private/tmp/review-pr2918-compaction-wake-rebase.log`). Acceptance precedes compaction, so accepted failures retain their generation/nonreplay fence; exact recovery-copy ownership, durable-parent lineage and both hook-refund provider paths remain qualified. Startup/CSS/guide guards pass **4 in 51.29s**, imports **679/686** and UI-ready **1032/1033** (`/private/tmp/pr2918-compaction-startup-green.log`). Original pins remain unchanged. Selections overlap and are not summed; inherited snapshot drift and pytest garbage-directory warnings remain disclosed. The unchanged audited diagnostic guard verifies 629 owners, 1421 TASK-492 calls, 56 TASK-31551 calls, 7589 TASK-494 calls and 15 sink files (`/private/tmp/pr2918-compaction-diagnostics.log`).

Dev then advanced to `922440b93e` through docs-only ADR-210 acceptance. No runtime file overlaps; its migration plan retains current Agent rail sections until the planned replacement. Preserve that canonical documentation and verify runtime bytes after rebase. Fresh PR-head CI is still required; no full suite, live-provider, Windows, aggregate resource qualification or merge is claimed.

Final independent schema approval has **39 passing checks** across the preceding selections, including three genuine linear upgrade/rollback cases in 12.32s (`/private/tmp/pr2918-review-chat-linear-migration.log`). No remaining actionable findings. Final changed-code verification covers **79 Python files**, with fatal/added-line/new-file checks clean, all ten new files Ruff/format-clean, owned migration/test formatting clean and whitespace clean (`/private/tmp/pr2918-compaction-final-static.log`). TASK-33431/33432 are closed through Backlog CLI with all criteria checked; all 13 original tasks are Done.
