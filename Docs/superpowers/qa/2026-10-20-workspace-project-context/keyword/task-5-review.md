# K5 independent spec and quality review

Reviewed range: `7d79700e025310e43ab9537ee65bb1792cfe2ba1..0295b820b5687c26020561a5a83fa4d743d877b8` on `codex/workspace-project-keyword-index`. The exact 29-path committed range was reviewed, including source, tests, performance harness and documentation; no `HEAD~1` substitution or unstaged root metadata was treated as implementation. Requirements are the current spec, ADR-240, keyword plan and full chronological `task-5-brief.md` amendments, under `task-5-review-brief.md`.

## Verdicts

**Spec: NEEDS FIXES.** The principal amended contracts are represented in source, but ordinary malformed draft input prevents the explicit Clear/Disable controls from reaching their service actions (I1). Final release qualification also remains open: the final eight-file covering result and final idle path-plus-text candidate/live-validation timing receipts were still pending when this report was written. A planned run is not passing evidence.

**Quality: NEEDS FIXES.** Two Important issues (I1 below and qualification addendum I2) and one Minor issue. No Critical issue identified. The whole K5 source/docs/test review is complete; the complete scoped fix list is I1, I2 and M1, including the subsequently received qualification failure recorded below. This report does not close the release gate or authorize Done/merge. Final evidence assessment and any exact scoped fix diff remain necessary.

## Strengths

- Settings supplies exact workspace/binding/root input; constructors and compose contain no added index I/O. Runtime/key/status work is threaded. Local request/visit/binding fencing supplements durable scope/job CAS. Console reuses its operation lane and waits for dismissal before category navigation.
- The default all/all processing policy and partially empty saved choices retain their intended meaning. Scanner empty-prefix/suffix behavior now agrees with SQL eligibility. Legacy management uses a validated nonmigrating open, preserving exact-scope Clear under migration pressure and avoiding a false Ready display.
- The index-only provider wrapper begins before guard/exclusion callbacks for both public invocation entries. Controller preparation and direct key/service calls receive finite registry ownership. The strict adapter checks the exact captured native descriptor and registered retirement after the existing helper; borrowed handles are preserved. Writer and supervisor failure paths retain lifetime/read admission when closure is unproven.
- The production scan path queries its bounded durable frontier after publication. Entries precede independent completion receipts, identity-compatible completed children survive parent insertion, seen membership precedes pruning, incomplete receipts do not prune, and the whole prepared unit has an atomic replay digest. Durable attempt revisions replace tree-sized attempted state without losing newly inserted children behind an old lexical cursor.
- Migration is file-backed and transactionally rechecked after migration-only native admission. Whole-binding quota accounting and rollback headroom precede DDL. Validated v2 cleanup does not allocate the v3 index. Failed migration/native close retains a counted store and binding capability in the factory and refuses repeated affected construction.
- Path-plus-text FTS, zero-length chunks for eligible empty/whitespace captures, current exclusion filtering before rank, and live Capture identity/hash verification preserve the search evidence boundary. No Library policy, ordinary file-tool wire, semantic stack or generic recovery ownership change was introduced.

## Complete findings

### Critical

None identified in the reviewed range.

### Important

**I1 - Clear/Disable are blocked by unrelated invalid policy drafts.**

Location: `tldw_chatbook/Widgets/Settings_Widgets/project_index_binding_card.py:213` through the validation/early return at line 232; action dispatch follows at line 233.

Trigger: Open an enabled Ready/Paused/Partial index, enter an invalid unsaved suffix such as `py` (or an invalid relative prefix such as `../outside`), then press Disable or Clear. `_pressed` constructs `ProjectIndexPolicy` for every action. Validation fails, displays the folder/suffix error and returns before `ActionRequested` is posted. Refresh/Resume are similarly coupled to draft validation even though `_execute` does not apply their `policy` argument. Pause also traverses this path, although normal Building inputs are disabled.

Impact: The user's explicit recovery/revocation action is silently converted into an unrelated form-validation refusal; the enabled cache remains enabled and Clear never starts. This is ordinary editing/canceling behavior and conflicts with usable explicit management, even though the spec does not enumerate malformed-field clicks. Existing stale-action and mounted happy-path tests do not exercise this trigger.

Smallest correction: Parse and validate input values only for Enable/Apply choices. Other actions should use the already admitted saved `self.policy` (or otherwise avoid reading draft policy) while preserving the captured expected scope/job/revision. Add a focused mounted regression proving invalid drafts cannot prevent Clear/Disable and are still rejected by Enable/Apply. Do not weaken `ProjectIndexPolicy` validation or management CAS.

Evidence: Deterministic source control-flow finding; no additional test run was needed or launched.

### Minor

**M1 - The public scanner signature still describes only the superseded production input/output shape.**

Location: `tldw_chatbook/Workspaces/project_index/scanner.py:294`, line 295 and the argument documentation at lines 307-309; callable dispatch is now at line 318 and `_scan_units` begins at line 410.

Trigger: A caller or maintainer follows the public `scan` annotations/docstring, which advertise `frontier: Iterator[str]` and `emit: Callable[[ScanPage], bool]`. The production service now passes a zero-argument frontier callback and receives `ScanUnit` values. Both modes intentionally remain supported, but the public contract omits the newly authoritative one and still describes fresh admission as per-page.

Impact: Static consumers and future callers see an inaccurate interface and can write a ScanPage-only callback for the production unit path. This does not demonstrate a current runtime failure.

Smallest correction: Describe/type both supported shapes accurately, using concise union/overload annotations or an equivalent local typing solution; update the argument and admission wording to distinguish singleton compatibility from bounded-unit production. No new runtime abstraction or broad refactor is needed.

Evidence: Source signature versus the new service call and callable dispatch; no additional suite warranted.

## Evidence assessment and release qualifications

The review consumed the implementer report and relevant raw receipts rather than equating test volume with correctness. No suites were rerun by this reviewer and no code, index, HEAD, source corpus, original checkout or user profile was changed.

- Both literal full-pipeline builds are supported by the retained `nested-pipeline.json` / `wide-pipeline.json` and corresponding `*-service-retired.json` files under `.superpowers/k5-scale`. Nested: 20,000 files across 20,200 folders, Ready with 40,200 discovered and 20,000 indexed, 705.188 seconds, 329 manifests. Wide: 20,000 files in one directory, Ready with 20,000 indexed, 448.312 seconds, 79 manifests. Both normal tool payloads contain the final marker and report actual worker/pipes/cache/registry/pin retirement; final service receipts show zero active searches.
- Peak extra combined RSS is **sampled** at 36,970,496 / 34,652,160 bytes; whole-binding peak/final bytes are 19,853,312 / 11,186,176. Frames and retained/source counters are below their contractual ceilings in these runs, with one observed active read. Sampling is not proof of exact native allocation maxima. These builds ran under recorded competing work and are not idle latency measurements.
- The final literal builds were new stable-profile builds, not Resume. The original approximately 1793-second faulty-retention partial run remains adverse evidence. The matched 12-file baseline and batched timings support the documented end-to-end improvement, but inclusive phases overlap and are not additive; multiple source changes prevent attributing all improvement to one line.
- `k5-format-source-equivalence.json` records both changed byte hashes and equal parsed ASTs for the worker and recovery fixture. The full builds qualify the mechanically formatted behavior under the root's explicit ruling; they are not claimed to have run on byte-identical final files.
- The omitted final native junction node now has `k5-final-junction.txt`: 1 passed, 1 warning, 2.18 seconds. Root confirmed actual Windows `mklink /J` succeeded, and the scanner/direct Capture refused the escape. This is distinct from the earlier inherited K4 privilege-failing fixture; the corrected report must retain that distinction.
- Real-provider outer-close tests use actual guard/exclusion/native ownership and no test-only wrapper around query entry. Recovery is correctly staged: hold actual Capture/pin and prove drain blocked, retire full query invocation/native owners, exit the query child, perform actual encrypted rollback in the parent, then have a fresh admitted child refuse the serialized old key/scope. This does not prove simultaneous same-live-service restore.
- Mounted real-app coverage exercises Enable/build/query/Clear, focus/painted bounds, stable Ready polling, saved Library configuration and dark 80x24/light 170x48. The final dark/light PNGs were inspected during this review; the light palette is visibly light and the compact actions are readable. The source asserts selected theme after startup and before capture. Clear-completion pixels alone are not proof that the Off action repaint has occurred; mounted state assertions are the separate evidence.
- Missing optional embeddings were simulated with unavailable imports, and the network guard was blocked with no attempts; this is keyword independence evidence, not an uninstall or real-provider claim.
- Scoped Ruff comparison reports 99 existing shared-file findings and zero introduced code/message/source-line diagnostics per file. The final formatter receipt and diff check are reported clean. The governance run was **not green**: 14 failed, 12 passed, 18 deselected. The retained comparator shows unchanged relevant CSS/governance inputs and attributes Windows separator/encoding/import and inherited style-size failures; this is source attribution, not an exact-base runtime reproduction. No stylesheet changed, so rebuilding or running bundle-sync for a nonexistent CSS edit was unnecessary.
- The required final eight-file covering run and final idle candidate p50/p95, live-validation and whole-tool receipts are **pending here**. In particular, the spec's cached lexical candidate p95 <=500 ms target is not yet judged for final path-plus-text source. Previously reported K4 text-only or concurrent-load timings cannot close it. The exact controller preparation node and omitted native junction receipt are separate focused qualifications, not substitutes for the pending covering result.

## Declined to judge

These are explicit boundaries for root disposition, not implicit passes or additional hidden findings:

1. Semantic/vector indexing, embedding destination consent, SSH indexing, watcher/automatic refresh, and binary extraction: separately deferred by the accepted delivery scope.
2. Full-repository regressions, unrelated Library suites and generic recovery/storage API redesign: prohibited full sweep or outside this exact K5 range; preserved boundaries were reviewed, but unrelated systems were not requalified wholesale.
3. POSIX/macOS native handle inheritance, real interactive terminal/human approval UI, and live provider behavior: Windows/headless/blocked-network evidence cannot establish them; no contrary claim is made.
4. Exact per-binding Settings deep-linking: explicitly replaced by the existing Workspaces category navigation ruling, which requires the user to select the row.
5. Existing completed `_Run` history/exception-frame retention (earlier K3 M1): expressly deferred to whole-branch triage. K5's finite registry retirement additions were reviewed, but this review does not reopen the earlier broad retention policy.
6. New UX for dirty-draft preservation across Check status, stricter action debouncing, or a broader general management-operation shutdown tracker: considered while reading the new UI/thread path, but no separate source-qualified regression was established here. They are not requirements to expand the scoped fix or substitutes for I1; root may decide whether any warrant later work.
7. Simultaneous encrypted replacement while the same live service remains running: the approved staged recovery evidence is narrower, as explicitly authorized. No same-live-service claim is accepted.
8. Pending final covering/idle success: intentionally unjudged, not out of scope. Root must supply and assess exact-source receipts before the final spec/quality gate can close.
## Qualification addendum: held-writer app shutdown fixture

Same reviewed source `0295b820b5687c26020561a5a83fa4d743d877b8`; no implementation change or reviewer rerun. Root supplied the final covering child's concrete failure before dispatching the fix loop. `k5-final-shutdown-factory-failure.xml` contains one failing test (46.419 seconds; child suite 47.050 seconds), with an AttributeError at runtime.py:313 after `await service.aclose()` completed.

### Important I2 - Adapt the injected lifecycle fixture to the runtime retirement-proof contract

Location: `Tests/Workspaces/test_project_index_service.py:461` injects `rig.service` directly into `app._project_index_service`; the newly required proof is invoked at `tldw_chatbook/Workspaces/project_index/runtime.py:313`. The assertion failure surfaces at test line 474. The constructor at test lines 24-42 passes `Rig.store`, a callable-only factory, to the standalone service.

Trigger: Run `test_real_app_shutdown_without_console_drains_index_writer` on the reviewed source. The actual app shutdown waits for the held scanner/writer to retire, then runtime close tries `service._factory.assert_retired()`. The injected standalone factory has no such method. The private-profile wrapper reports it as a function object in the retained child traceback.

Disposition: **Fixture-adapter obligation and required-qualification failure, not a demonstrated production compatibility defect.** A callable-only factory remains supported by `ProjectIndexService` itself: its retention and cleanup capabilities are deliberately optional, and direct `service.aclose()` works with the Rig. The application runtime composition is narrower. The only production assignment to `app._project_index_service` is `get_project_index_service`, which constructs `_AdmittedStoreFactory`; runtime key construction already requires that factory's `retain`/identity semantics. Direct test mutation of the private app field does not establish a public runtime composition contract for arbitrary factories. The new runtime-close cancellation test at lines 1550-1558 already explicitly adapts its standalone service to the factory proof seam.

Impact: Required targeted qualification is failing on new K5 assumptions, so it cannot be marked green or attributed to an inherited baseline failure. The AttributeError also prevents the remainder of that app-shutdown test from running. The retained stack shows the service drain returned first; it is not evidence that native retirement was skipped or that the production factory lacks its proof method.

Smallest correction: Adapt this lifecycle fixture locally with a callable factory that delegates to `Rig.store` and provides an honest `assert_retired` check of its owned work after service drain (the fixture owns no failed-migration retention map), or use the real runtime-composed factory for this test. Keep the held-writer barrier, no-Console assertion, physical scanner/pipes/thread retirement and successful complete app shutdown checks. Add/retain an assertion that the factory proof is reached only after retirement. A local adapter is sufficient; do not redesign the public factory protocol or broadly rewrite Rig fixtures.

**Do not make production proof optional with a generic `getattr(..., no_op)` fallback.** The approved failed-migration map is a production owner and must still be checked; its unresolved retirement must continue to fail shutdown. Do not patch out `close_project_index_service` or substitute a synthetic early drain.

Verification after correction: Run the exact failed lifecycle node and the directly affected retirement-proof/cancellation checks as the coordinator's scoped verification requires. No duplicate suite was run for this review; source and retained XML fully resolve the classification.

Updated complete list: **0 Critical, 2 Important (I1 product UI, I2 fixture/qualification), 1 Minor (M1 public scanner types/docs).** I1's malformed-draft Clear/Disable finding remains unchanged and must be sent with this addendum in the same fix loop. Both verdicts remain NEEDS FIXES; the aggregate covering result and idle timing evidence remain open.