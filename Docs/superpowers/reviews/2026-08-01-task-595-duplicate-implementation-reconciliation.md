# TASK-595 implemented twice — reconciliation

**Date:** 2026-08-01
**Status:** decision needed
**Author:** written after discovering a second, independently-specced implementation of TASK-595 during TASK-596 design.

## What happened

TASK-595 ("Add verified managed model downloads and recovery") was implemented
**twice, in parallel, in two working copies of the same GitHub repo**
(`rmusser01/tldw_chatbook`), by two different agents, against two different
approved specs. Neither knew about the other.

| | Implementation A (merged) | Implementation B (unmerged) |
|---|---|---|
| Clone | `Documents/GitHub/ppqq/tldw_chatbook` | `Documents/GitHub/tldw_chatbook` |
| Branch | `feat/managed-model-acquisition` → **merged to `dev` as PR #1157** | `codex/task-595-managed-downloads-v2` — **not on `dev`** |
| Spec | `Docs/superpowers/specs/2026-07-30-managed-model-acquisition-design.md` | `Docs/superpowers/specs/2026-07-31-verified-managed-model-downloads-design.md` ("approved by the user on 2026-07-31") |
| Plan | `2026-07-31-managed-model-acquisition.md` (10 tasks) | `2026-07-31-verified-managed-model-downloads.md` (853 lines) |
| Size | ~5k lines across 24 commits; 379 tests in `Tests/Model_Artifacts/` | 2,518 insertions across 11 commits; +528 lines of `test_service.py` |

Both were built to ADR-025 and both are coherent. They are **not**
interchangeable: they disagree about where staging lives and how bytes reach
the immutable destination.

## Completeness — the decisive asymmetry

**A is end-to-end; B is a foundation.**

A delivers the whole path: `preflight()` → `grant()` → `provision()` with
durable resumable fetch (`fetch.py`), pre-verify with one refetch,
`consume_source` install, activate-last, credential resolution, session-lease
serialization, and real-subprocess `kill -9` crash recovery.

B delivers **only the service-side seam**: `service.py` grows a download-stage
API (`_download_stage_for`, `_finalize_download_stage`, `_open_download_stage`,
containment/marker validation, retirement) plus tests. There is **no
`downloads.py`** on the branch — `ManagedArtifactDownloader`, the HTTP/resume
layer, the Parakeet adapter, and the Library wiring described in its spec are
**unwritten**. Its own spec's flow (steps 1–12) is therefore not yet runnable.

So this is not "two finished features, pick one." It is one finished feature
(A) and one carefully-built foundation with a better boundary (B).

## Where they genuinely disagree

### 1. How verified bytes reach the immutable destination

- **A:** downloads into `staging/managed/<id>/<rev>/<variant>/`, then calls
  `ModelArtifactService.install(..., consume_source=True)`, which moves files
  per-file (`os.replace`) into the core's own install staging and promotes.
- **B:** the service creates a *marked, contained stage* with a `payload/`
  subtree; on finalization the service verifies and **renames the payload
  subtree itself** into the destination — no second staging hop at all.

B's spec rejects A's approach explicitly: *"Copy completed downloads through
`ModelArtifactService.install` … Rejected for remote acquisition because it
temporarily needs a second full copy and undermines persistent
same-filesystem resume."*

**Assessment:** B is right about the shape, and A's own review history proves
it — A had to add `consume_source=True` precisely to stop a double copy, and
still hit a related defect (a *retryable* install failure destroyed the
resumable download, fixed by relocating the sidecar out of the payload
directory). B's `payload/`-subtree design makes that class of bug structurally
impossible: resume metadata can never be inside what gets promoted. A's
current behavior is correct but arrived at by patching; B's is correct by
construction.

### 2. Consumer migration

- **A:** none. Defers the descriptor question to TASK-596/1301 and *refuses
  multi-file artifacts* (`CatalogError`) rather than guessing per-file URLs.
- **B:** in scope — the Parakeet v2 module becomes a thin adapter supplying the
  first exact descriptor (pinned repo/revision/license/files/sizes/digests) and
  a credential-free **source map**, plus a managed-first model-directory
  resolver with legacy `.tldw-verified.json` fallback, and the Library modal
  rendering the plan.

**Assessment:** B's scope is more useful. A shipped a downloader with **zero
production consumers** — nothing in the app can actually download anything yet,
because no descriptor exists. That is the real gap in A.

### 3. Per-file source URLs

- **A:** unresolved; deferred to 596/1301 (this is what blocked TASK-596's
  design and prompted TASK-1693).
- **B:** solved — the downloader takes "exact descriptor **and source
  mappings**", so URLs never need to live in the frozen descriptor schema.

**Assessment:** B's answer is better and cheaper. It makes TASK-1693 (descriptor
schema v2) unnecessary.

### 4. Where the HTTP lives

- **A:** new `fetch.py` — an async `httpx` streaming fetch forked from the
  guarded egress hop loop; the package uses PEP-562 lazy exports to keep
  `httpx` out of the STT worker import path.
- **B:** *"Use the existing guarded synchronous HTTP egress boundary from a
  background worker. Add only the response-before-body seam needed to validate
  resume headers."* No new HTTP dependency, no async layer.

**Assessment:** genuine trade-off. A's async surface fits Textual and is done
and tested (including cross-origin credential stripping and a mutation-proved
secret-hygiene test). B's is less code and reuses the audited egress path, but
is unwritten. A is ahead here.

### 5. Recovery ownership

- **A:** `reconcile()` classifies managed staging (orphans-only GC; valid
  sidecar survives).
- **B:** *"Normal artifact reconciliation may report staging but does not guess
  which entries are safe to delete"* — the downloader owns a focused recovery
  pass over its own marked stages.

**Assessment:** B's is the safer boundary (markers + ownership proof rather
than shape-sniffing a sidecar). A's GC has already been tightened once for
exactly this reason (garbage JSON used to keep dead staging alive).

## Recommendation

**Keep A on `dev`; port B's boundary and its consumer migration onto it.** Not
because A is better designed — on the two architectural questions that matter,
B is — but because A is merged, tested end-to-end, and reverting ~5k reviewed
lines to finish an unwritten downloader is the more expensive path to the same
place.

Concretely, in priority order:

1. **Adopt B's `payload/`-subtree finalization seam** (port the `service.py`
   stage API and its 528 lines of tests; retarget A's `_install_artifact` at
   `_finalize_download_stage` instead of `install(consume_source=True)`). This
   removes A's sibling-sidecar workaround and the whole "retryable failure
   destroys resumable bytes" class. **This is the one change worth doing before
   TASK-596 builds on the layer.**
2. **Adopt B's source-map contract** for per-file URLs; close TASK-1693 as
   superseded and unblock multi-file artifacts (currently a hard `CatalogError`).
3. **Port B's Parakeet v2 adapter and resolver** — the descriptor, the
   managed-first/legacy-fallback resolution, and the Library modal rendering
   the plan. This gives A its first real consumer and is most of what TASK-1301
   needs anyway.
4. **Adopt B's marker-based recovery ownership** if (1) lands, since marked
   stages make it natural.
5. Leave A's async `fetch.py`, credential resolver, session lease, and crash
   tests as-is; B has no written equivalent and A's are proven.

If instead **B is canonical**, the honest cost is: revert PR #1157, write
`downloads.py` + HTTP/resume + Parakeet adapter from its plan (the largest
remaining chunk), and re-derive A's credential/redirect/secret-hygiene and
crash-recovery tests, which B's spec calls for but has not yet built.

## Process failure worth fixing

Two agents implemented the same backlog task, to two separately-approved specs,
in two clones of one repo, over the same days, and neither surfaced the
collision. The backlog task itself was the shared coordination point and it
recorded nothing until one side closed it.

Cheap guards:

- **Claim the task before designing:** set `status: In Progress` with an
  assignee/branch note as step one of brainstorming, not at close-out.
- **Spec filename includes the task id** (`task-595-*`) so a `find` for the id
  surfaces every spec regardless of wording.
- **Check sibling clones and their `.worktrees/`** before starting: the
  duplicate here was invisible to searches scoped to one checkout — a
  repo-wide `find / -name "*<task-id>*"` would have caught it on day one.

## Outcome — branch B re-inspected after the ports landed

All four ports are merged: TASK-1694 and TASK-1695 in PR #1165, TASK-1696 in
PR #1167, TASK-1697 closed during #1165's fix wave.

B kept building for roughly six hours *after* this document was written,
finishing its own `downloads.py` (2,273 lines), a 3,337-line test file, and —
between 03:28 and 05:11 on 2026-08-01 — the same Parakeet adapter, Library
wiring, and Console resolver that TASK-1696 was porting at the same time. The
collision recurred because nothing in the shared repo ever told B to stop:
writing the decision down is not the same as making it visible where the other
agent looks. B's last activity was an uncommitted formatting/typing pass at
05:26; its branch was never pushed and has no PR.

Re-reading B against merged `dev` at that final state, its remaining unique
value is two egress changes, both now filed:

- **TASK-1722** (high) — `_log_origin()`: `egress._blocked()` logs the full URL,
  so a query-string token in a presigned source-map URL lands in the log file.
  Affects every egress caller, not just downloads. B has the fix and the test.
- **TASK-1723** (low) — `trusted_private_origins`: exact scheme/host/port trust
  enforced per redirect hop. Our `trusted_origins` is hostname-only and
  fixture-only today; adopt when a real private or LAN source lands.

Everything else in B is superseded: its `downloads.py` is the synchronous-egress
alternative to a merged, tested `fetch.py`, and its STT layer duplicates
TASK-1696. **Retire the branch** — nothing further to port.
