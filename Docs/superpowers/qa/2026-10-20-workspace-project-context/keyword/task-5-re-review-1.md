# K5 fix round 1 scoped re-review

Exact range: `0295b820b5687c26020561a5a83fa4d743d877b8..3820fe11572696f62d27291971f5c8d1dba0faec`. Reviewed the complete 8461-byte commit/stat/contextual package and its three authorized paths under `task-5-re-review-1-brief.md`. This re-review covers I1, I2 and new breakage in the fix; it does not repeat the broad K5 source review.

## Source disposition

**I1: ADDRESSED.** `ProjectIndexBindingCard._pressed` starts with saved immutable `self.policy` and reads/validates input fields only for Enable/Apply. Clear, Disable, Pause, Resume and Refresh no longer depend on malformed draft fields. The existing action construction still captures the same status scope, job ID and revision. No policy validator or service CAS was weakened. New mounted regressions execute real Clear/Disable through the service with invalid suffix/prefix drafts, assert exact saved policy and CAS metadata, and separately prove invalid Enable/Apply still refuse without dispatch or scope change.

**I2: ADDRESSED.** The change is confined to the injected app-shutdown fixture. Its local callable Factory delegates to `rig.store`, then `assert_retired` checks closed service, actual scanner retirement and pipe closure, nonempty run history, every DB/registry/native retirement flag and stopped writer/supervisor threads. The proof event must remain unset while the real writer is held and be set after complete actual app shutdown. The original no-Console and physical retirement assertions remain. This is an honest post-drain fixture adaptation, not a no-op method or a bypass of the runtime shutdown path. Production runtime and mandatory `_AdmittedStoreFactory.assert_retired` are unchanged; production failed-migration ownership refusal remains covered separately.

**New fix-diff findings: 0 Critical, 0 Important, 0 Minor.** No new breakage identified in the three-file diff. Original Minor M1 (scanner public types/docs) remains explicitly deferred by root to whole-branch triage and is not silently closed.

## Verification assessed

No tests, benchmarks, provider calls or source mutations were performed by this reviewer. Read-only source and retained receipts were sufficient; no unresolved doubt required a new check.

- `k5-fix1-ui-green.txt`: complete post-fix Settings card file, **23 passed, 1 warning, 576.91 seconds**. It supersedes the original covering run's pre-fix card cases. The invalid Clear/Disable and retained Enable/Apply cases pass; existing mounted state, stale action, policy fidelity, default-file, unavailable/busy and legacy-pressure Clear checks remain included.
- `k5-fix1-close-green.txt`: corrected exact app-shutdown node plus four affected cancellation/native/migration proof checks, **5 passed, 1 warning, 131.40 seconds**. The app node takes 76.84 seconds and reaches the full application shutdown path. This closes the concrete I2 failure rather than suppressing it.
- `k5-final-index-covering.txt`: original required eight-file run is retained accurately as **1 failed, 181 passed, 3 skipped, 1 deselected, 1 warning, 2946.52 seconds**. Its sole failure was I2, now superseded by the exact corrected node. The three scanner skips are Windows filename/permission-bit/rename constraints; Windows ACL-denial behavior remains unqualified. The omitted native junction node separately passed on final backend source. These receipts are not represented as one all-green final-tree sweep.
- `k5-fix1-source-equivalence.json`: only the authorized card and two test files differ from source 0295; the other 26 K5-owned paths are byte-identical. Thus runtime, service, scanner, store, protocol, provider and harness source for the previously assessed scale/recovery/backend results remains applicable. The earlier formatter-only AST qualification remains distinct from this byte-identical backend comparison.
- Scoped Ruff/formatter/diff checks are reported clean for the three changed files. No CSS or geometry changed in the fix.
- `k5-fix1-mounted-render.json` retains seven SVG/PNG pairs with per-image source qualification. Console pairs derive from source 0295 with unchanged Console source; Settings pairs derive from the complete post-fix card run. Historical requested-light/startup-overridden-dark is explicitly adverse setup evidence. The prior source-0295 images remain separately retained. Clear frames are focus/completion evidence, not a claim that Off controls have repainted. This review does not relabel the initial renderer timeout as a UI failure or successful pixel receipt.
- The final actual staged recovery result and retained native-retirement artifacts are now reported complete; the approved staging limitations from the original review continue to apply.

## Current spec and quality verdicts

**Spec: source fixes accepted; final release gate PENDING.** I1 and I2 are resolved, and the completed targeted covering evidence is reconciled with exact post-fix superseding runs. The final idle path-plus-text candidate p50/p95, fresh Capture and whole-tool measurements have not yet been supplied for assessment. The <=500 ms candidate p95 target is therefore not yet passed here.

**Quality: source fixes accepted; final release gate PENDING.** There are no remaining Critical/Important source findings from the reviewed K5 scope or this fix. Minor M1 remains deferred. Pending original latency evidence and the original explicit platform/governance/source-timing limitations prevent an unconditional final PASS or task-Done authorization.

Root will supply the final same-cache, exact-source idle receipts, including load conditions and native retirement. Evidence closure may be appended here without another broad source review or duplicate suites. All original Declined to judge dispositions remain visible in `task-5-review.md`; no wider work was inferred from this fix.