---
id: TASK-1697
title: Adopt marker-based ownership for download-staging recovery
status: Done
assignee:
  - '@claude'
created_date: '2026-08-01 07:02'
updated_date: '2026-08-01 09:41'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reconciliation item 4: replace shape-sniffing of the fetch-state sidecar in reconcile()'s GC with the parallel branch's marker-based ownership proof (schema, operation kind, artifact reference, descriptor fingerprint), so recovery refuses and reports when containment or ownership cannot be proven rather than guessing. Do after the finalization-seam port, which makes marked stages natural.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Staging entries are classified by an owned marker, not by sidecar shape
- [x] #2 Recovery refuses to delete when ownership or containment cannot be proven
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extend reconcile()'s staging GC to recognize service-owned download-<fingerprint>/ operations (previously an unrecognized top-level name, left alone forever). 2. Add marker-based ownership proof (schema + fingerprint self-consistency against the operation directory's own name) plus containment verification (exact expected entries, no symlinks anywhere in payload/ or state/), gated behind the same non-blocking ACQUISITION_SESSION_LEASE_KEY the legacy managed/ GC uses. 3. Reclaim only when ownership is proven AND the reference is already installed (the one on-disk-provable 'will never be resumed' signal without a catalog) -- never a not-yet-installed, still-resumable stage. 4. Fix _remove_incomplete_download_stage so a post-marker os.rename failure during _download_stage_for's temp-to-canonical publish also gets cleaned up, not just a pre-marker one.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
service.py: added _DOWNLOAD_STAGE_PREFIX, extended _gc_staging to collect download-<fingerprint>/ top-level entries and hand them to the new _gc_download_staging, which (like _gc_managed_staging) requires ACQUISITION_SESSION_LEASE_KEY non-blocking for the whole batch. Added _download_stage_ownership: parses the marker, requires exact schema (_DOWNLOAD_STAGE_KEYS/_DOWNLOAD_STAGE_SCHEMA_VERSION), and requires the marker's OWN descriptor_fingerprint to match the fingerprint encoded in the operation directory's name -- self-consistency is the ownership proof reconcile() can check without a catalog (unlike _read_download_stage_marker, which always compares against a caller-supplied descriptor). Containment is checked alongside: exact expected top-level entries, no symlinks at any level (_download_stage_node_identity per node, _validate_download_stage_state for state/, new _assert_no_symlinks for payload/, which holds arbitrary bytes, not JSON so state/'s JSON-parse check doesn't apply). DESIGN CHOICE for AC #2's 'ownership or containment cannot be proven': an unparseable/self-inconsistent marker returns None from _download_stage_ownership and the entry is left COMPLETELY ALONE, never removed -- more conservative than the legacy managed/ GC (which reclaims a structurally-invalid entry outright), because this layout's payload can hold a large, real, in-progress download that a transient/corrupt marker read must never cost. An owned-and-contained stage is reclaimed only when its reference is ALREADY installed (core.artifact_path exists) -- the only on-disk-provable 'this exact stage will never be resumed again' signal without a catalog; a not-yet-installed reference is exactly the 'valid and resumable' case and survives regardless of lease state, confirmed against the existing crash-recovery suite (test_provision_crash_recovery.py), which already asserts a not-yet-installed stage survives reconcile(). Separately fixed _remove_incomplete_download_stage (part ii of the P2 finding): it used to no-op unconditionally once the marker existed, leaking the whole temp .download-<random>/ directory forever on any failure AFTER marker creation (e.g. the final os.rename into the canonical path). Since operation_identity already proves this call exclusively owns that ephemeral temp directory (tempfile.mkdtemp names are unique, and nothing ever looks up a stage by its temp name), the marker-exists early return was removed and the marker is now unlinked before the empty-dir rmdir cleanup. Added 4 regression tests to test_reconcile_staging_gc.py covering: an already-installed stage is reclaimed and reported; a not-yet-installed one survives; an unparseable marker is left alone even when the reference is installed; a post-marker os.rename failure leaves no orphan temp dir. Verified the first and last of these fail against the pre-fix code (git stash) for the expected reasons (the other two pass either way, since pre-fix nothing recognized the layout at all -- they guard the safety property going forward). Fixed this task's own bookkeeping: ACs were already ticked while status was still To Do; both are now genuinely verified by the tests above. Full suite green: Tests/Model_Artifacts/ + Tests/STT/test_boundaries.py, 432 passed.
<!-- SECTION:NOTES:END -->
