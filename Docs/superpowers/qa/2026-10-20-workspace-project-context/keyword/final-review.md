# Final whole-delivery review

Reviewed BASE `43054ffa6f02ef4ffc57c0e49ce5b169a5dd5e23` through HEAD `2210c0b5f5b8f48ffaa02cfbba66c4db5b0ddba6` in the managed keyword worktree. HEAD and the clean tracked working tree were independently checked. The supplied final brief was read first. This is the whole integration review, including the approved spec, ADR-240 and its chronological amendments, K1-K5 plan and reviews, current source, owner integration, and qualified release receipts. QA JSON/SVG was inspected selectively; no suite, provider, native probe, dependency installation or source edit was performed.

**Spec: NEEDS FIXES. Quality: NEEDS FIXES. Ready to merge: NO.** Findings: **0 Critical, 4 Important, 3 Minor.** The four Important findings are the complete consolidated blocking list. Earlier K5 I1/I2 are addressed; the new I1 below is a different shutdown interleaving. The admitted-latency deviation is explicitly accepted by this review as a nonblocking limitation, not relabeled a passing target.

## Strengths

- The index remains opt-in and binding-local, with no Library records, Library scope changes, network model requirement or automatic folder scan per send. The existing local provider, permission floor, project-instruction path semantics and output budgets are reused.
- Registry revisions/tombstones and the registry-then-cache fence establish cross-process admission rather than relying on callbacks. Keys include data generation and full pinned root identity. Current plus requesting-run exclusions precede ranking, and returned text comes from fresh native Capture with identity/hash checks. No stale-text fallback was found.
- Storage enrollment precedes production construction; the cache is excluded from ordinary backup/export/sync. Failed migration/native retirement is not silently converted into success. The migration-only capability, whole-binding quota/journal reserve, v2 nonmigrating management/Clear, and strict native-plus-registered-core connection retirement address real ownership boundaries.
- The multi-directory unit remains bounded across entries, attempted directories, raw inspections, pending paths, text and wire frames. Preparation occurs outside publication fences. Entries precede completion/pruning receipts, incomplete traversal does not establish deletion, replay is atomic, and durable attempt revisions replace the tree-sized attempted set without skipping newly discovered children behind a cursor.
- Actual full-pipeline 20k nested and wide builds, actual native child/pipe/connection retirement, and final common/rare/miss timings are retained with their source and load limitations. The final report preserves adverse runs and corrections instead of presenting one fictitious all-green run.
- Settings captures expected scope/job before dispatch, keeps storage work off-loop, preserves blank/partly empty policies, and avoids implicit legacy migration. Console status follows the captured selected binding and drains its lane before navigation. Final light pixels are actually light; compact dark controls remain readable with scrolling.

## Critical findings

None identified.

## Important findings

### I1 — Shutdown does not drain complete already-issued operations

**Locations:** `tldw_chatbook/Workspaces/project_index/service.py:446`, `:463`, `:892-913`; `tldw_chatbook/Widgets/Settings_Widgets/project_index_binding_card.py:299`; `tldw_chatbook/Workspaces/project_index/runtime.py:306`; `tldw_chatbook/app_lifecycle.py:698`.

**Trigger:** Start Clear from Settings and let it pass the short `_lock`/registry revocation section. Hold it inside `_clear_binding` during namespace cleanup or reclamation, then quit. Clear's off-thread call is neither a `_Run` nor a `_SearchRun`. `aclose()` obtains `_lock`, sets `_closed`, snapshots only those two collections and can complete while Clear still owns its admitted pin, registry/cache handles and lifetime capabilities. Runtime then checks only failed-constructor retention and returns. The card awaits `asyncio.to_thread`; cancellation/unmount does not cancel or join the physical thread. The app-owned lifecycle drain runs before Textual tears down the card and proceeds to unrelated shutdown.

**Outer provider owner:** The same finite-operation gap extends beyond management. `service.py:273` removes the search record and signals its done event after the inner cache/pin retires, before `search()` exits its registry scope (`:290`) and before `LocalToolProvider.invoke_detailed()` exits the actual outer registry scope (`local_tool_provider.py:2349`). The existing `Tests/Agents/test_project_index_tool.py:852` test, `test_real_provider_registry_owner_spans_outer_close`, explicitly holds that real outer native close and asserts all query pins closed, `_searches` empty, invocation not returned and independent maintenance still blocked. Thus `aclose()` can snapshot no search and return during this proven public-invocation interval. If that outer native close fails, the strict adapter correctly withholds evidence and keeps maintenance blocked, but there is no remaining service operation record through which close reports that unresolved ownership. The test releases/joins the provider before calling `aclose()`, so it does not cover shutdown during the barrier. This is a lifecycle completion defect; access and maintenance revocation remain fail-closed.

**Impact:** The declared shutdown contract (spec lines 335-337: drain issued writes and retire owned handles) is not met. Native Clear work can continue after index close reports completion and overlap subsequent owner/UI teardown; the final management snapshot may then fail because `_closed` was set. The existing held-writer, held-migration-under-start-lock, query and composition-lock tests do not cover this unlocked management interval. This is not a demand for generic UX debounce or a general application operation framework.

**Smallest correction:** Add an index-owned admission/retirement boundary for complete finite operations, registered atomically against the same close fence before their first owned work/effects. Cover the actual outer public provider/service, key-retention, UI and controller preparation boundaries, including their final strict registry proof; nesting must preserve the existing borrowed-handle ownership rule. Extending only the inner `_search_store` lifetime does not cover the outer provider owner. Track the actual operation through its final cache and registry native retirement, not merely the awaiting coroutine. Include those issued operations in the existing shielded off-loop close drain, and refuse later admission. Reuse `_store`/`registry_connection_scope`, the existing pins and strict native proof; do not hold `_lock` across joins or invent another storage authority. The production outer UI registry scope must either be part of that complete owner or be explicitly joined before shutdown claims retirement. Preserve failed-retirement reporting rather than discarding an operation record on unknown closure.

**Verification:** Hold an actual Clear after revocation and before physical cleanup/close; start app close, cancel its observer repeatedly, assert the UI loop stays responsive and close remains pending, then release and require complete owner-thread/native retirement before close finishes. Also hold the actual provider outer native-close barrier while closing: close must remain pending, and a failed native close must remain explicitly unresolved. Exercise a queued management action after admission closes. Source control flow and the existing held-owner test establish these defects; no new execution was needed for this review.

### I2 — Detaching a binding strands its derived cache without a cleanup/retry route

**Locations:** `tldw_chatbook/Workspaces/project_index/service.py:451`, `:467-547`, `:507`, `:549-560`, `:886-890`; `tldw_chatbook/Workspaces/registry_service.py:3179`; `tldw_chatbook/Widgets/Settings_Widgets/project_index_binding_card.py:158-166`.

**Trigger:** Build an index, then remove the folder binding (or remove the workspace containing it). The durable registry fence correctly stops later read/publication admission, and the callback only signals local runs. Nothing invokes physical cleanup. A saved exact old scope cannot use public Clear after detachment: `_revoke` refuses unavailable authority, and `_clear_binding` independently requires a live authority. The Settings row has disappeared. Reattaching the same path with a new binding ID has a different accounting namespace and cannot reclaim the previous one.

**Impact:** Cached source bodies and disk allocation remain indefinitely with no supported retry mechanism. Immediate access revocation is effective; this finding is about physical reclamation and retry, not stale disclosure. It falls short of spec lines 333-337 and ADR-240's retryable cleanup intent. The existing binding-wide Clear tests cover obsolete roots/data generations under a still-authorized binding, not a detached binding. Merely counting/excluding the files in recovery inventory does not remove them.

**Smallest correction:** Preserve narrowly scoped cleanup ownership when the binding is removed and provide a durable retry path independent of renewed source-read authority. Reuse the current stable `binding_namespace` accounting group, admitted data pin, native lifetime acquisition, `cleanup_eligible`, `clear_revoked`/`clear_owned_namespace`, and `reclaim_free_pages`. A verified detached/tombstoned expected ownership state may authorize deletion of already-owned derived namespaces; it must never authorize querying, source scanning, or reconstructing a key from a matching pathname. Capture sufficient exact ownership before detachment to retry after restart; the current opaque namespace/control hash and binding-only revision tombstone are not by themselves a UI retry route. Keep namespace preflight, all native retirement proofs, current-scope/CAS safety and quota headroom. This needs a narrow index cleanup-contract clarification, not a generic background collector or relaxed read fence.

**Disable distinction:** The final user guide explicitly says Disable revokes use and Clear removes cached content. Retention after Disable is understandable and has a usable Clear action while the binding remains available. I do not require Disable to become an automatic eraser. Reconcile that deliberate retained-cache distinction with the broader spec/ADR cleanup wording. It does not justify stranding detached data.

**Verification:** Build, detach with work in flight and idle, verify immediate no-disclosure, force cleanup failure, restart and retry using recorded cleanup ownership, and prove physical reclamation without reattaching/readmitting the source. Include a replacement binding at the same path to prove the cleanup cannot touch its new generation.

### I3 — Included-folder matching disagrees between scanning and search

**Locations:** `tldw_chatbook/Tools/project_index_scan_worker.py:90-99`; `tldw_chatbook/DB/Workspace_Project_Index_DB.py:954-956`, `:997-1003`.

**Trigger:** On the supported Windows filesystem, a project contains `Src/example.py` and the user enters `src` in Included folders. The worker casefolds both paths and prefixes, so it scans the file and can finish Ready with it counted as indexed. The SQL eligibility callback retains the original prefix spelling and tests `path not in prefixes` / `not path.startswith(prefix_heads)` case-sensitively. Every candidate under `Src/` is rejected before ranking. The reverse spelling and Unicode case-folding variants have the same mismatch.

**Impact:** A valid narrowed build reports indexed coverage but returns no matching evidence, even though default-all works. The DB suffix/policy oracle test repeats the DB's own case-sensitive prefix expectation using lowercase fixture prefixes, so it does not establish scanner/search agreement.

**Smallest correction:** Settle one processing-prefix comparison rule and apply it at both boundaries. The current accepted scanner behavior is casefolded path/prefix matching; preserving that behavior in SQL is the smallest compatibility correction. If platform-sensitive exact-case semantics are deliberately chosen instead, change and document both sides consistently, including the Windows case-insensitive case. Do not change mandatory/current/requesting-run exclusion floors, containment, permission rules, query normalization or rank/LIMIT order as part of this policy fix.

**Verification:** A real scan-to-public-search test with differently cased path and configured prefix must produce consistent indexed/searchable coverage; include a peer prefix (`src-peer`), nonmatching directory and excluded file to preserve boundaries. No source-execution probe was needed to establish the opposing predicates.

### I4 — Committed compatible entries cannot be searched while a build is running

**Locations:** `tldw_chatbook/Chat/console_chat_controller.py:15690`; `tldw_chatbook/Workspaces/project_index/service.py:311-314`, `:334`, `:368`, `:393-397`.

**Trigger:** Refresh a Ready index without changing authority/policy, or let an initial build commit its first unit and hold it between publications. The cache is Building and contains committed compatible rows. New Console preparation only supplies index scopes for Ready/Partial, so the tool is omitted. An already-composed provider is also rejected by the service's Ready/Partial-only entry, candidate and final state gates.

**Impact:** Explicit Refresh removes keyword discovery for the entire build, including previously indexed unchanged files. This can last the measured several minutes at the requested scale. Spec lines 299-300 explicitly allow committed individually compatible entries to be searchable during a partial build; the later migration/reader amendments preserve search during active builds and reserve a concurrent read slot. No reviewed ruling explicitly removes that behavior. Preserving rows in the DB is not the same as making them available through the production tool.

**Smallest correction:** Make committed compatible Building coverage eligible in both scope preparation and service query admission, with a truthful partial result rather than ready. Preserve exact scope/root/policy/exclusion checks, manifest fencing, fresh Capture and final admission; a build with no committed compatible rows still must not advertise useful coverage. Do not add queries against uncommitted data or silently adopt a newer scope. Updating only the initial state set is insufficient because later state checks and ready/partial serialization also reject or mislabel it.

**Verification:** Hold the real scanner after a committed unit, exercise actual Console scope preparation and public provider search for its marker, and require partial evidence before the build is released. Repeat with a Refresh of previously Ready content, plus revocation/stale-source controls. The existing restart test at `Tests/Workspaces/test_project_index_service.py:592` proves Building with indexed rows is a reachable state but never searches it.

## Minor findings and deferred-minor triage

### M1 — Completed run/exception history remains unbounded

`service.py:107`, `:708`, `:749`, `:832-880`. Every Refresh/Resume leaves its `_Run` in `_runs` after positive retirement. Starts, stops and close traverse historical runs; error objects retain traceback frames and potentially captured page/store references. The earlier K3 minor is still present. Remove fully retired records after waiters have captured references, or retain a small bounded diagnostic history with cleared tracebacks. Never remove unresolved native owners. This remains nonblocking relative to the current correctness findings, but it should not be described as resolved by the durable frontier change.

### M2 — Scanner public annotations/documentation describe only the compatibility mode

`scanner.py:294-309`, callable dispatch at `:318`, production `_scan_units` at `:410`. `frontier: Iterator[str]` and `emit: Callable[[ScanPage], bool]` omit the production callback returning a bounded seed tuple and the `ScanUnit` handoff. Describe/type both supported shapes with a concise union/overload and distinguish singleton compatibility from production units. This is the earlier K5 M1, still present; no new abstraction is required.

### M3 — Check status silently replaces unsaved processing choices

`project_index_binding_card.py:314-322`. With Ready/Off/Paused status, edit either input and press Check status. The successful reload overwrites both input values with saved policy because `action is None` but the state is not Building. The same can occur if the initial status request finishes after the user starts typing. This is a concrete unsaved-edit loss, not hypothetical debounce expansion. Populate fields on initial untouched load and successful Enable/Apply; refresh status without replacing dirty local inputs (or make an explicit discard choice). A small mounted regression is sufficient. No new general draft framework is required. Existing Settings inspector copy also says workspace actions have no draft; avoid implying that these new Apply choices save as typed.

Other carried items:

- **K1 public adapter return annotation: resolved.** Final `registry_service.py:2668-2670` declares `tuple[IndexAuthorityReader, IndexAuthorityFence]` using the type-checking imports. Do not reopen the obsolete missing-annotation finding.
- **Old tree-sized attempted-directory set: resolved.** Final production frontier is `eligible_directories(claim)` with indexed `(complete,attempt_revision,path)` eligibility, bounded LIMIT 256, attempt revisions advanced by explicit Resume/new run, and terminal-vs-continuation receipts. The remaining worker attempted set is per bounded unit, not whole-tree history.
- **K2/K3/K4 warning/lint observations: evidence limits, not product findings.** Retain inherited pytest-asyncio/Pydantic warnings and baseline diagnostics; no suppression or unrelated lint cleanup is requested.
- **K5 original malformed-draft Clear/Disable and fixture proof failures: resolved by the final source and superseding affected receipts.** The card now validates drafts only for Enable/Apply. Production factory retirement proof remains unconditional.
- **Action debounce:** no separate finding. Repeated actions can issue overlapping requests, but expected scope/job CAS and lifetime/read admission prevent a demonstrated second authoritative stale effect. I1 concerns physical lifetime, not a request to redesign all action UX.

## Evidence, performance and limits

### Independent performance judgment

I read the metric-alignment review, fix2 brief, final report and raw `fix2-idle-latency.json`/source manifest. I recalculated each canonical p95 as sorted sample 47, confirmed 50 measured samples after 5 warmups per workload/shape and expected counts 100/1/0, and checked the public ownership booleans and 42 native receipts. No benchmark was rerun.

| Shape/workload | Canonical lexical p95 ms | Admitted candidate p95 ms |
| --- | ---: | ---: |
| Nested common | 222.326 | 534.466 |
| Nested rare | 172.163 | 469.083 |
| Nested miss | 181.923 | 493.769 |
| Wide common | 232.795 | 561.909 |
| Wide rare | 177.233 | 479.246 |
| Wide miss | 168.199 | 467.353 |

All six canonical decorated lexical measurements meet the original 500 ms target. Both common admitted measurements still fail the later stronger 500 ms target. The retained earlier run failed all six admitted cells and nested/common canonical 577.305 ms. Those were real production costs, not fixture-only status work.

**I accept the explicit admitted-target deviation as nonblocking for this delivery.** It was recorded before the bounded fix, preserves every executed authority/activation/status/floor check, and does not disguise a failed broader metric as success. The optional discovery feature has no separate 500 ms whole-invocation requirement. Public marker calls remain materially slower: 20 per shape, instrumented p50/p95 1994.482/2703.472 ms nested and 1909.283/1957.342 ms wide. Native Capture p95 is 281.957/257.156 ms. Whole-tool clocks include probe retirement bookkeeping and are not pure uninstrumented user latency; neither are they evidence of instant interaction. The 40 public calls entered with no borrowed registry handle and returned with it retired; 42 Capture children were gone, retired and pipes closed, and final service searches were zero. This is a measured limitation worth keeping visible, not a reason to remove security checks. It does not excuse I4's build-wide unavailability.

### Other qualification

- Raw full-pipeline receipts establish 20,000 nested text files across 20,200 folders: Ready, 40,200 entries/20,000 files, 705.188 s; and 20,000 files in one directory: Ready, 448.312 s. Final marker evidence crosses runtime/service/scanner/store/provider/Capture. Sampled extra combined RSS is 36,970,496/34,652,160 bytes; sampled whole-binding/final storage 19,853,312/11,186,176 bytes. Sampling is not exact highwater proof. Builds had recorded competing work; final query runs used the same retained caches with no competing K5 test/build/render and uncontrolled OS background.
- Both build retirement receipts say actual runs retired and active searches zero. Production read budgets and worker/source/frame bounds are supported by source plus earlier targeted native/resource evidence. These receipts do not establish every future workload or simultaneous reads across independent applications.
- Mechanical formatting qualification was checked: worker bytes changed `ed2261a7...` to `1cc7f40a...`; recovery fixture `802e39b4...` to `2546f55f...`; the retained receipt records equal parsed ASTs for each after both builds retired. Current hashes match the after hashes. Fix2 query source hash `acf1b1fa...` matches its final source manifest; its source-scope receipt preserves all other product AST. Build receipts remain historical-byte/AST-qualified, not falsely exact-final-byte runs.
- The original final covering receipt is **1 failed, 181 passed, 3 skipped, 1 deselected**, not all green. Its known fixture failure is superseded by the exact shutdown/related five-pass receipt; final card behavior by the post-fix 23-pass file; query changes by six scoped nodes and the final timing pass. No repeat unchanged suite is required merely to combine these into one number. These passes do not test I1's Clear interval, I3's mixed-case prefix integration or I4's Building query.
- Scaled migration evidence uses real SQLite, native SQLITE_FULL rollback, a 5,000-directory legacy fixture and real binding locks with scaled quotas. The failed-close test retains the exact live registered connection and capability and refuses another affected construction. This is stronger than synthetic flags, but is not a literal 1 GiB physical exhaustion experiment on every filesystem. Public legacy management/Clear pressure is separately mounted.
- Actual staged recovery holds real Capture/admission, proves maintenance blocked, retires query/native owners and the query process, then performs encrypted rollback and fresh-child old-key refusal. It is not simultaneous replacement against the same still-running service. Recovery declaration/owner catalog changes preserve the excluded cache disposition.
- Mounted Settings/Console evidence is Windows/headless Textual with retained native exports. I independently inspected final dark80 and light170 Settings PNGs; selected theme is asserted after startup in the source. Clear images are focus/completion frames, while settled Off controls come from assertions. The startup-overridden dark image remains adverse light-setup evidence. No real human approval interaction is established.
- Optional embeddings unavailability is simulated, with network guarded and no attempts; no actual uninstall or real-model result is claimed. The admitted provider roots/guard in performance fixtures are test-seeded; real registry/runtime/native Capture and full provider return ownership run beneath them.
- POSIX/macOS and other Python runtimes remain source-only. Windows newline, POSIX mode-bit and retained-no-delete-share rename skips remain qualified. The earlier WinError1314 symlink setup failure is not a security pass; the later actual Windows junction test is separately passing. K5's permission-bit skip is not a new ACL-denial qualification.
- Two unchanged private-SQLite census failures were reproduced on exact earlier baseline. Shared-file lint retains 99 attributed existing findings and zero introduced diagnostics. Governance retains 14 failed/12 passed/18 deselected with unchanged-input source attribution, not exact-base runtime reproduction or an all-green gate. No stylesheet was changed. No full-suite, hosted CI, clean-install, original-profile or real-user-folder claim is made.

## Declined to judge

These are explicit remaining scope boundaries requiring root disposition, not implicit passes:

1. Semantic/vector indexing, embedding endpoint consent, SSH indexing, filesystem watchers, automatic refresh and binary/OCR extraction: deliberately separate deliveries. Keyword ownership seams were reviewed; those features were not implemented or certified here.
2. General recovery/storage/config API redesign, machine-wide concurrency limits, and cross-query authority caching as performance remedies: outside this plan and unnecessary for the accepted latency deviation. No such change is requested by these findings.
3. Unrelated whole-repository regressions and full/hosted suites: targeted-only instructions prohibit a sweep. Existing failures are qualified at the evidence levels above.
4. POSIX/macOS native execution, different interpreters, real terminal/human approval UI and real-model natural-language project answers: not established by Windows/headless/blocked-network receipts. Guidance was a prior delivery; this review does not convert its prior or unexecuted real-model obligation into new keyword proof.
5. Simultaneous encrypted replacement while the same live service remains active: source fencing was examined, but the accepted actual recovery sequence is staged. No broader successful experiment is claimed.
6. Exact per-binding Settings deep-linking: explicitly replaced by supported category navigation. Manual row selection remains an accepted usability cost.
7. Choosing a new universal numeric whole-tool latency budget or guaranteeing performance on arbitrary user corpora/hardware: the measured reference results and original canonical target are sufficient to judge the current amendment. The admitted target remains unmet; no new budget is invented.
8. A general debounce framework, arbitrary permanent UI draft persistence across navigation, or new detached-cache collector architecture: not authorized or necessary merely for polish. The concrete dirty-field loss is M3, the finite management drain is I1, and required detached cleanup ownership/retry is I2; their smallest remedies should stay index-specific.

All earlier deferred lifecycle questions were judged explicitly above rather than left under these exclusions. No pending release receipt is treated as a pass.

## Final assessment

The implementation has strong authority, native-retirement and bounded-scan foundations, and the performance qualification is now candid and meaningful. It is not ready to merge because management shutdown, detached cleanup, scan/search policy agreement and committed Building coverage remain incomplete. Address I1-I4 in one consolidated wave with focused regression evidence and one scoped rereview. M1-M3 may be fixed narrowly or explicitly retained as nonblocking debt. Preserve the accepted performance deviation, inherited-output qualifications and source timing distinctions in the final release notes.
