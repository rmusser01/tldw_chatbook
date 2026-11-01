# PR3166 final integrated tab qualification — 2026-10-31

Tested product head: `0c39d774c13e0860a0afeb6ed174991dc1afd4cc`, on confirmed PR3162 merge `a0dbc1174c3dbd4d2732b0212786ec9e08bfc3b5`. The 37 reviewed commits plus the preview-evidence commit were rebased without conflicts; `final-rebase-map.json` and `final-range-diff.log` retain all 38 correspondences. `actual-base-source-preservation.json` verifies 52 critical/new blobs against the actual merge, including its final archive runtime/card fixes. This handoff adds only a test-fixture correction and evidence/documentation; production bytes are unchanged from the tested head. No remote action occurred.

ADR required: no. Existing ADR-126/230 finite preparation ownership and ADR-166 archive ownership apply. The preview conflict ledger, exact size pins, reviewed diagnostic statements/inventory delta and 1,175-entry census-prefix proof remain in `../pr3166-candidate-integration-2026-10-31/`.

## Passing functional contracts

Each phase ran serially in a fresh genuine native-private Windows profile, using the supported private-child node entry. The runner records its exact command, parent head and raw product/test diff SHA256; every `.patch` is retained. No full suite, performance sweep or timing-bound increase was used.

| Phase | Passing receipt | Pytest result | Contract retained |
| --- | --- | --- | --- |
| after_close | `after-close-event-control.json` | 1 passed in 28.04s | Real Close removes the doomed session; keeper-only laid-out tab DOM is published before the unrelated original post-Close reader is released. |
| coalesced | `coalesced-startup-control.json` | 1 passed in 31.40s | An earlier original FULL and Close's actual tab publication overlap; the real keeper-only DOM publishes while the unrelated reader remains held. |
| same_session_drain | `same-session-drain-retirement-control.json` | 1 passed in 34.27s | Real Close retains its session while its exact native reads remain unretired; after release, those reads physically retire without cancellation, then the session disappears and keeper-only DOM publishes. |

All passing paths retain the original installed reader, original native calls, checked source proof, source/function identity checks, real Pilot click and DOM assertions, and clean held-task/view/runtime teardown. The native gate still holds for at most 30s; logical waits remain 10s, held-sync/warming cleanup 8s, and the test timeout 180s. `test-summary.json` summarizes retained receipts; `final-static-checks.json` records passing Ruff and the 1,177-entry UI gate census.

The passing sources differ only as diagnosis narrowed the affected fixture paths. after_close's exercised setup remained unchanged after its pass. The startup prerequisite was added only to the two Surface-lock phases; the last correction observes Close return only in same_session_drain. Coalesced's exercised code remained unchanged after its pass. This is targeted functional evidence, not an overall speedup or stability estimate.

## Retained failures and fixture correction

1. `after-close-original` failed its original readiness wait. The observer then raised AttributeError for an absent optional replay flag, obscuring the ordinary assertion; that diagnostic defect is acknowledged. The original native reader later returned without a proof after 14.562s. Readiness now awaits existing exact-owner worker tasks instead of repeated Pilot pauses, retaining the original pending/sync/replay predicate and deadline. Native entry/return and existing Surface/drain events have asyncio mirrors; production tasks are never cancelled by these waits.
2. `coalesced-event-control` and `same-session-drain-event-control` failed before real mouse delivery. The latter records the exact original task chain: the screen pump's deferred startup `call_after_refresh(FULL)` waits on the screen tab lock; the earlier held FULL owns that lock while waiting on the deliberately held Surface lock. Pilot's global message-queue settle cannot finish. `startup-lock-owner-excerpt.json` is a selected copy of the raw task-chain evidence, not a replacement for it. The two lock-holding phases now explicitly finish the existing attach owner and its queued startup publication before taking the downstream fixture lock. This prerequisite and readiness share one original ten-second deadline. No active owner is suppressed, cancelled or classified as harmless.
3. `same-session-drain-startup-control` reached the actual Close drain but timed out at the remaining post-release Pilot poll. The observed Proactor loop resolution was 0.015625s; the exact original reader remained in Windows `_Native.info` through startup_permission/config admission, while its producer and retirement were pending. It subsequently returned a checked proof at 10.547s and cleanup succeeded. Only this phase now observes the original runtime.close_session return through sys.monitoring (exact runtime/session/source), at the same ten-second bound, then asserts store removal and all original non-cancelled retirement/DOM conditions. Its controlled run passed. This supports the narrow fixture correction; it does not establish the cause of every historical delay.

The original failing logs/patches are retained. `tab_setup_trace_red.py` preserves the full task observer used for the same-session RED; the current observer filters irrelevant idle pumps. Earlier after-close/coalesced observers had only reader/setup diagnostics. Observer hashes are recorded separately; the runner's source-diff hash covers product/tests, not evidence-plugin files. Cleanup records only the seven receipt-owned private roots.

## Limits retained for independent review

The slow-tokenizer freshness repair and incoming TASK38237/38267 identities remain as previously reviewed. The historical cadence 6-versus-5 failure remains unattributed; the later config-generation refusal is not an explanation for it. Current traced six-case cadence and deterministic late-publication/refusal evidence remain in `../pr3166-review-2026-10-31/`; they were not repeated here.

The preview static timer-census batch still has two explicitly recorded failures (ProjectIndexBindingCard and PluginsScreen scope sites). Exact-b1ca control reproduced the former; Plugins source preservation is established, but no exact-candidate full-census control was run. No blanket baseline or allowlist was added. These limits are not hidden by the passing UI gate census.

TASK35679 AC2 remains open and status In Progress pending root's fresh independent review of the final integrated fixture/evidence. The three connected functional paths now have qualified author-run results; no claim is made that every historical performance limit or broader test suite is resolved. Root retains publication/merge ownership and must recheck allocated task-ID collisions before any push.
