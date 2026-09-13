# Cross-platform backup correction plan

TASK-32496. Spec: ../specs/2026-09-12-cross-platform-backup-correction.md.
ADR required: existing ADR amendment.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md.
Reason: correct platform operation contracts without changing recovery behavior.

Use subagent-driven development for the Windows native boundary and independent
review. Root owns Linux implementation, actual product testing, task tracking and
PR publication. Keep file ownership disjoint until reviewed integration.

## Stage 1 — Reproduce and map
Status: Complete.
Record real Linux product failure and native Windows runner failure. Map calls to
private paths, locking, identity, flush and exclusive publication.

## Stage 2 — Platform operations
Status: Complete. Native APFS, ext4 and local NTFS operations are verified, including private ownership, nested directory publication, metadata, locks and persistence barriers. Actual installed backup, restore, replacement and retained-copy rollback have passed on all three platforms.
Windows implementer owns new platform adapter modules and Windows primitive tests.
Root owns Linux native operations, integration imports and release contracts.
Preserve containment, permissions, exclusive rename and failure propagation.

## Stage 3 — Actual product verification
Status: Complete. Revision e733 passes all six actual Windows product-flow groups, including later rollback; macOS and the supplied Linux host also pass. Final revision a5a536565 passes all 117 Windows support tests and 42 native cases, including canonical TTS reference seed, capture, restore and fresh read. All 31 final artifact hashes verify. Final Linux checks pass 19/19; local macOS regressions pass. Exact receipts and pre-existing broader failures are recorded in the verification document.
Use installed package, private profile and synthetic data on supplied Linux host,
GitHub Actions Windows runner and local macOS. Create and verify backups, restore
and open them, then exercise replacement and retained-copy rollback. Fix failures.

## Stage 4 — Review and PR evidence
Status: Complete for the published pre-merge revision. Native adapter, integration, SQLite snapshot and final TTS/file-inventory reviews are accepted. Scoped security/static checks introduce no findings. PR2642 records those exact product and support results. The requester has now supplied the required human-written Change summary verbatim; dev integration is tracked separately below.
Independent review of changed boundaries and evidence; targeted static/Bandit
checks; update existing PR2642 with exact results and platform support.

## Stage 5 — Resolve current dev conflicts
**Goal**: Integrate current dev into the existing backup PR without adding feature scope. The initial merge parent is a3142cb3569370bfb201d09fef26f288dd5d657c; dev advanced to b2e31e4b14fa2ede7e840f44541fcc0910e4f4d7 during verification.
**Success Criteria**: Resolve the 99 conflicted paths, preserve current application services and Python platform support, verify the merged backup workflows, and push the merge commit to PR2642 against dev. Do not merge the PR into dev.
**Tests**: Current constructor/schema qualification; package contents; CI structure; producer pause/drain/resume and native borrower retention; actual backup, isolated restore, replacement and rollback on macOS, the supplied Linux host and the authorized Windows Actions runner; scoped lint and Bandit.
**Status**: In Progress.

Conflict markers are resolved. Integration checks found current schemas, lazy service construction, device-local Notes state exclusion, a shared profile runtime packaging dependency, repeated TTS admission work, and Canvas payload SQL validation. Each is being reconciled with targeted regressions and real product runs. The empty Research paste staging directory created by dev is recognized; nonempty staging remains explicitly unsupported pending separately approved payload coverage. Previous platform results above do not qualify the new merge revision.

The actual installed two-profile restore/open passes on macOS. Core owners pass62,
operational owners pass113, and the workflow structure passes19 tests. Six lazy
default SQLite owners reuse the existing observable-absence policy (60 new and4
existing checks pass). Child launch now retains an exact parent AgentRuns
operation until independent child admission or refused-launch cleanup completes;
five native tests cover held-pause terminal persistence, worktree discard,
reservation refund, thread-start refusal and accepted-child native retirement.

Mounted replacement exposed additional upstream startup-only scaffolding and
runtime records; qualify only their verified empty/protocol forms and existing
runtime exclusions. Research's canonical zero-operation index is recognized with
a bounded checked read; nine discovery cases pass, including refusal of pending
payloads and malformed indexes. The newer dev delta changes Console persona and
thinking rendering, with no further database schema changes. Preserve those
semantics and give its new resumed-character DB offload the existing native
operation-owned connection lifetime.

All 99 original conflicts are resolved in the index. Installed macOS F9 replacement
passes in 113.17s; combined replacement/later rollback passes in 239.77s with stale
preview refusal, explicit credential review, unchanged abort and validated
restoration. Legacy persona safety dependencies retain strict whole-tree, native
identity and encrypted readback validation (31 focused checks). Live preview
shares one verified private database image across owner reads without rejecting
ordinary later writes; initial-copy races, source replacement and final maintained
scope changes remain refused (15 capture/preview checks pass). Eight exact startup
worker caches now retire on their native worker, preserving callers; monitor
settlement precedes final Notes shutdown. Startup/Home and shutdown checks pass62.
The two observed empty scaffold roots receive native-held read scope, including
explicit final rediscovery nested inside capture; payload authority stays unchanged
(62 discovery/admission checks pass). Actual installed F9 backup→restore→reopen
passes plaintext73.29s, encrypted75.27s and encrypted-with-credentials91.37s, after
mapping the existing legacy persona-assets destination in the test. The newer dev
Console delta and exact published Linux/Windows verification remain.

The first conflict resolution is committed as `d914a76a2`. The subsequent merge
incorporates dev `5fd502dba`, preserving its Console and Notes behavior. The new
resumed-character worker uses the existing native ownership context; four real
SQLite lifetime regressions pass after reproducing three leaks. Diagnostic
inventory changes were reviewed statement by statement: six upstream diagnostic
calls, no new persistent destination. Final published platform runs are pending.
