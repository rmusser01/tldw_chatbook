# Bounded independent landing-callback review

> Historical pre-fix assessment. Final frozen source and proof are CLEAR; see [candidate-review.md](candidate-review.md), including its Final scoped verification assessment.

Reviewed published head f70eb635f5991d25b7cd5d32d789406d837be32b context, with source-byte inventory in reviewed-sources-before.json. Read-only review: no app launches, tests, source, index, branch, or Git mutations. Native fault scheduling and exact UI19 execution belong exclusively to library_readiness_diag/root.

## Requirements and evidence

User objective is merge PR3346 into dev only after all demonstrated issues are fixed/reviewed, with explicit Qodo bypass; no increased size/time caps or full-suite sweep. Existing latest feature, focus, session and UAT reviews remain outside this bounded new-defect audit.

The private fault schedule at ../mixins-f70-ui19-diagnosis/owner_probe.py wraps native LibraryLandingCanvas.mount_all entry and invokes the original lifecycle presentation method once. It does not replace sync_state, callback, recompose, state builders or normal UI journey. Before invocation: parent attached/mounted/queryable, zero native children, callback a real partial(_finish_library_entry_canvas_sync), current generation1/route, rendered0, dirtytrue, pending matches. Immediately afterward: callback None, same current ownership, no children, no recompose request. Original UI19 journey then fails its unchanged 30s healthy-DOM predicate, with loadedtrue/lookup_errorfalse/rendered0/state1/dirtytrue and active Library. This proves a supported explicitly scheduled event loses the current owner; it does not establish the precise hosted event timestamp.

## Strengths

The original readiness predicate correctly refuses loaded-but-unrendered state and retains a concrete diagnosis. Callback composition already distinguishes defaults from explicit action supersession. Generation/route completion checks refuse stale publication, and PostRecomposeCallback already carries a pending callback across a newly requested recompose.

## Issues

### Critical

None beyond the bounded Important correctness defect below; no data loss/security claim is supported.

### Important

1. LibraryLandingCanvas.sync_state (library_entry_canvases.py:429) assumes the current stored widget-set key describes mounted children. Its first structural sync adopts state before native refresh/recompose. A second same-key lifecycle sync can therefore query old children while recompose is pending, or absent children while Widget.recompose holds self.batch/lock across remove/mount. NoMatches is caught by _sync_library_canvas strict fallback (canvas_sync.py:1309), whose unconditional callback retirement deletes the foreign current _finish_library_entry_canvas_sync queued by successful entry reconciliation. The original successful call has no reason to schedule another reconcile, leaving state1/rendered0 dirty indefinitely.

Preferred minimum repair: in the retained landing owner, adopt latest state and serial as before, then request its existing recompose and return while `_recompose_required` OR `self.lock.is_locked` OR the existing widget-set change condition holds. This avoids querying a tree not represented by stored state and lets existing PostRecomposeCallback recompose handoff preserve/run the owner once. Pending as well as active coverage is required: active-lock-only leaves the before-start sibling window. No new wrapper, duplicate callback, retry loop, catch broadening or timeout change is needed. Generic exception callback-retirement changes affect every canvas and are unnecessary for this demonstrated root cause.

### Minor

None proposed; avoid unrelated broad feature or callback refactors.

## Reach and verification inventory

- Production shared dispatcher: _sync_library_canvas landing arm, deferred-guard setup, existing callback queue/default ordering, failure cleanup. Candidate should leave this owner byte-identical and retain 1314-line cap.
- Native landing owner: LibraryLandingCanvas.sync_state and compose/widget-set contract; _replace_recent_rows serial/currentness must continue rejecting obsolete deferred work.
- Callback lifecycle: PostRecomposeCallback.recompose checks attached/pruning, clears consumed slots, preserves callback through `_recompose_required`, then runs once. _RetainedSyncCallback._complete_targeted_sync handles normal patch completion. Candidate should leave both unchanged.
- Screen callers: _reconcile_library_entry_state queues exact partial finish plus generation/route guard; _sync_library_entry_owner_from_current_state; _sync_library_landing_lifecycle_presentation and onboarding/ingest callers. Current completion is generation/route fenced. No signature, export, config or data-model changes needed. Screen should stay byte-identical at its exact current cap.
- Existing public exports in Widgets/Library/__init__.py unchanged. LibraryStudyHandoffCanvas is not a demonstrated owner of this landing race and does not need a speculative patch.
- Permanent narrow regression should schedule the real native empty-child mount window, retain original callback and actual current generation/route, release original mount, assert latest DOM plus exact rendered completion and clean pending/dirty state. Show same-byte old RED and repaired GREEN.
- Add a direct pending-before-native-start owner case: first structural state change followed synchronously by same-new-key/latest-copy sync. It must never query incompatible old children and must publish the latest state/callback exactly once.
- Keep current normal in-place identity/focus/recents, latest-recents convergence, stale deferred route refusal, default/explicit callback ordering, detached/pruning refusal, and genuine failed-sync retirement controls selected where affected. No blanket preservation on unrelated exceptions.
- Run original exact UI19 node separately without fault scheduling and preserve its healthy-DOM deadline/assertions. This is local qualification; current-head protected CI remains a separate publication requirement.
- Architecture note can document stored state versus mounted tree during deferred recompose; this is a routine existing-owner bugfix, no new ADR. Existing Library decomposition/callback ownership and ADR286 task governance apply.

## Assessment

Ready to merge: NO on current reviewed pre-fix bytes. Causal defect confirmed; preferred owner repair is cohesive and minimal in outline. Actual candidate bytes and permanent RED/GREEN proof are still required before a post-fix verdict. No hosted-green or native exact-journey success is claimed by this reviewer.

Skipped app/test/native reruns and broad audit as requested; reviewed source and the single delegated fault-scheduled RED evidence only.
