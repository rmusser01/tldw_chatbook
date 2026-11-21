# TASK-40608 Console redesign: check storage once, then simplify the pipeline

> Status: **proposal, not approved** (2026-11-20). Four decisions below need the owner before any implementation starts.

**Goal:** Both original Windows journeys (README, approved edit) under 5 s of application time in all three sequential pairs (TASK-40608 AC5), without weakening a user-facing safety guarantee.
**Architecture:** Verify storage when a profile session opens, hold its connections, and revalidate on events instead of on every call. The Console runtime stays in its own process (your 2026-11-18 direction), but its display protocol drops per-read retirement proofs.
**Tech stack:** Python 3.12, Textual 8, SQLite (WAL), native Windows NTFS handles.
**Evidence branch:** draft PR #3575 at `0f6da22775`. The spike branch `spike/task-40608-storage-floor` is local only.

ADR required: yes. `backlog/decisions/126-complete-local-backup-and-recovery.md` (replace the PERF-07/PERF-08 per-call stamp rule on Windows, and the TASK-40608 runtime/profile-writer section). ADR-356 rule 4 applies as already written.

## Why the current approach cannot reach 5 s

Measured on Windows 10 with the original journey and its scripted loopback provider. The provider answers each request in about 0.01 s, so all of the time is the app's own.

| Turn | Now (PR head) | Estimated without storage checks |
|---|---:|---:|
| README | 21.9 s | ~7.6 s |
| Approved edit | 22.3 s | ~7.4 s |

- **The Console runtime process is the critical path.** In every phase its CPU time is close to the wall time. README's 9.7 s before the first request contains 8.6 CPU-s in the runtime. The UI process is mostly idle, waiting on its event loop.
- **About 78% of the runtime's turn CPU is storage verification.** The shares overlap and come from inclusive stack samples: admission acquire 33%, folder-chain metadata walks 24%, opening a SQLite connection for each owned call with its private-path checks 23%, config-file scopes 20%, trusted-folder checks 16% and pinned folders 11%. At the bottom, the time goes to `windows_files._Native.open_handle`, `info` and `security`.
- **The cause is the design, not untuned code.** From 2026-11-12 to 2026-11-19, Codex reduced work inside the rule "keep every per-operation check". That cut turns from 4–7 minutes to about 20 s, then stopped improving. The PERF-07/PERF-08 amendment requires each call to re-stamp every folder from the drive root. On Windows each folder check is a handle open, so one check costs about 30 handle opens and 15 security-descriptor reads in the test profile.
- **Spike results:**
  - Checking the folder chain once per process: README 24.5 → 17.6 s, edit 25.2 → 17.6 s, and the edit's "sending" paint 5.3 → 0.24 s.
  - Also caching each file's metadata: no further gain.
  - Keeping connections open across calls: broke the Chat screen at startup. The display protocol requires each read's connection to be closed and proved closed.
- **Estimate method.** The "without storage checks" column subtracts the runtime's storage-check CPU from each phase's wall time (run `fixedc`, 10 ms all-thread CPU sampler, aligned to the journey's own clock). It is an estimate. It assumes the UI and the writer do not become the new critical path; both are mostly idle in these traces.

What remains in the estimate is pipeline work:

| Phase | Now | Without storage checks |
|---|---:|---:|
| README Send → first request | 9.7 s | ~3.1 s |
| Plain model round | 1.6–1.9 s | ~0.5–0.7 s |
| Round with an approval (human wait excluded) | 3.1–3.6 s | ~1.2–1.4 s |
| Last response → reply shown | 2.5–2.7 s | ~0.8–0.9 s |

## Windows facts this design relies on (measured on this machine, 2026-11-20)

While a SQLite connection is open (SQLite opens files without `FILE_SHARE_DELETE`):

| Attempt | Result |
|---|---|
| Rename, replace or delete the open database file | refused (winerror 32 / 5) |
| Rename the parent or grandparent folder | refused (winerror 5) |
| Replace the parent folder with another folder | refused (winerror 5) |
| Delete the parent folder | refused (winerror 145) |
| Same rename after the connection closes | allowed |

The PERF-07/PERF-08 amendment rejected held handles because on POSIX `fstat` on a held descriptor cannot see a folder renamed away and replaced at the same path. On Windows the OS refuses that rename while the handle is held, so the reason does not apply there.

## Guarantees that stay

- **Authority checks:** Once/Always approvals, the kill switch, workspace binding access (read-only/read-write), sensitive-path protection and before/after file proofs for edits keep running where they run today. They are policy checks on user-supplied paths, not storage re-verification.
- **Backup and restore:** a backup or restore can still drain every writer before capture or replacement. Per ADR-126 (PERF-07/PERF-08 context), the live hold's shared lease lock already blocks a cooperating backup or restore, and the 10 Hz monitor and `pause_requested` observe pauses. Per-call re-derivation was never what detected them.
- **Durable turns:** a step's commit still lands before its UI or tool callback.
- **Profile privacy:** owner-only ACLs on the profile chain are verified at session open, and again after any revalidation event.

## Proposed design

### D1. Storage: check at open, hold, revalidate on events (Windows)

- **One storage session per profile per process.** On first use it runs today's full derivation once: registry, selector, control records and the posture of each folder chain. It then opens and keeps the connections it needs (one per database, per thread where SQLite needs it).
- **No per-call path checks while held.** While its connections are open, Windows pins the files and their parent folders, so per-call path and posture stamps are dropped for held handles. In-memory gates (local pause, participant closure, fork, provenance) stay on every call; they are cheap.
- **Cheap content checks.** Control records, the registry and the config selector are small files. A cheap single-file stamp check, or a directory change notification with a periodic fallback, triggers a full re-derivation and reopen when one changes.
- **ACL and owner changes off the critical path.** These are checked at open and by a background posture check (for example every 1–2 s). A violation refuses new operations and surfaces the existing recovery state. This trades per-call detection for a bounded window, which is owner decision O3.
- **Backup and restore drain the sessions.** A pause request makes every session close its held connections and acknowledge, which reuses the existing lease and `pause_requested` path. Resuming reopens with a full derivation. Draining on pause replaces per-read physical retirement as the way backup gets exclusive access.
- **POSIX keeps today's stamp rule.** `lstat` is cheap there and held handles do not pin. The slow path being fixed is Windows-specific.

### D2. Runtime/UI protocol: adopt by revision, push changes

- The runtime answers display reads from its own state and held connections. The UI adopts a publication by revision and source identity alone, with no per-read retirement proof.
- The runtime pushes change notices (or answers a long-poll) instead of the 200 ms poll and checkpoint handshake. Approvals reach the UI as events.
- Targets: a plain model round ≤ 0.3 s and an approval round ≤ 0.5 s beyond the human wait.

### D3. Writes: SQLite locking plus one maintenance lock, instead of the elected writer

- **Change:** with D1, each process holds its own verified connections, and SQLite in WAL mode already serializes writers across processes. Settings, imports and other app instances would write through their own sessions. The elected writer process, its named-pipe protocol, grants, guard pipes and retirement handshakes would be removed.
- **Why:** the writer was introduced to give one owner the persistence work (removing repeated per-write admission) and to give backup one place to drain writes. D1 removes the repeated admission, and the existing maintenance lease already drains writers.
- **Evidence of the cost:** in `20a11155c3` a grant that could not prove retirement kept its file guard for the rest of the process, and every later write to that database was refused (fixed in `49158d5407`).
- **This reverses your 2026-11-18 single-writer direction and changes AC6.** Keep the writer only if a cross-process invariant needs it. None has been found yet; checking is part of Phase 1.

### D4. Remove the code-identity checks instead of converting them

- The branch adds about 577 reads of `__code__`, `__func__`, `__closure__`, `__wrapped__`, `_getframe`, `f_locals` and similar to establish "original source" authority. Over the same period, `dev` removed 207 such reads.
- ADR-356 rule 4 forbids this pattern, and `Tests/Architecture/test_code_identity_ratchet.py` enforces it.
- These checks guard nothing real (any code in the process can bypass them), and they run on hot paths. Delete them. Where a real ownership question exists, use a runtime registration, as ADR-356 asks.

## Owner decisions needed first

| # | Decision | Recommendation |
|---|---|---|
| O1 | Amend ADR-126 so that, on Windows, held handles plus event revalidation replace per-call stamps (D1) | Approve |
| O2 | Drop per-read retirement proofs from the runtime/UI display protocol (D2) | Approve |
| O3 | Accept a bounded window (for example 2 s) for detecting an ACL or owner change on the profile chain, instead of per-call detection | Approve, choosing the window |
| O4 | Replace the elected profile writer with SQLite locking plus the maintenance lock, and rewrite AC6 (D3) | Approve, unless Phase 1 finds an invariant that needs it |

## Delivery plan

Branch strategy: build on current `dev`. Do not squash #3575's 695 files and 239 commits onto it. #3575 remains the evidence and prototype branch, and its parts are ported when a phase needs them.

### Phase 0: independent fixes (no decision needed)

1. Send the workspace-dialog activation fix (`0f6da22775`: activate before dismiss) to `dev` as its own small PR. On `dev`, the dialog's callers also activate only in the result callback. Run the new regression test there first to confirm.
2. Separately, fix or quarantine the 7 tests in `Tests/UI/test_console_new_workspace.py` and `test_workspace_cold_provisioning.py` that fail on the PR head. They are outside the CI mount list, so CI never ran them.

### Phase 1: prove the floor before building (after O1–O2)

1. On a spike branch from `dev`, give the in-process ChaChaNotes and AgentRuns stores a minimal D1 session: held connections, draining on `pause_requested`, no per-call stamps. Also remove the per-read retirement wait from display reads.
2. Run the original journey twice. Gate: plain rounds ≤ 0.8 s and turns ≤ ~9 s. That is consistent with the ~7.5 s estimate; anything worse means the estimate missed a cost.
3. Inventory cross-process write invariants, for O4.

### Phase 2: storage sessions (D1)

1. Add a profile storage session that owns the full derivation, the held connections, content-change revalidation, background posture checks and drain/reopen on pause. Migrate the Console databases first (ChaChaNotes, AgentRuns, Workspace), then config reads (`get_user_data_dir` and the config scopes), then the remaining participants.
2. Tests:
   - A backup and a restore started during an active turn drain the sessions and reopen afterwards.
   - Renaming or replacing a held database or its folders is refused (Windows).
   - A change to the registry, selector or a control record triggers re-derivation.
   - An ACL change on a chain folder is refused within the agreed window.
   - Process death releases everything.
3. Amend ADR-126 (O1, O3) in the same PR.

### Phase 3: protocol (D2)

Adopt by revision without retirement proofs, push change notices, and deliver approvals as events. Gate: plain round ≤ 0.3 s and approval round ≤ 0.5 s, measured with the phase-aligned sampler.

### Phase 4: writes and code identity (D3, D4)

Remove the elected writer and grants (if O4 is approved) and the code-identity checks. Gate: `test_code_identity_ratchet.py` at `dev`'s pins, plus multi-instance write tests (two app instances, Settings and an import writing during a turn).

### Phase 5: qualification

Three sequential baseline/candidate pairs with the original journey and its original endpoints. Gate: every candidate turn under 5 s (AC5). Re-check AC6 against the wording agreed under O4.

## Risks and open questions

- **The estimate is a model.** Phase 1 exists to test it cheaply. If turns stay well above ~9 s, stop and re-profile before Phase 2.
- **Other share modes.** Any handle in the chain opened with `FILE_SHARE_DELETE` does not pin it. The pinning guarantee depends on SQLite's share mode and on the share modes of our own folder handles.
- **Sync and backup tools.** While the app holds connections, tools that move the profile folder (sync clients, third-party backup) are blocked. SQLite already blocks them today whenever a connection is open, so this is not new.
- **Multi-database imports.** An import that writes ChaChaNotes, Media and Prompts needs its cross-database ordering re-checked once the writer is gone.
- **The journey test measures itself.** It queues one callback per mounted widget on every pilot pause (over 100,000 per turn). Also time the real app.
- **The suspend/resume race class.** The workspace-setup race came from a modal's result callback waiting behind the opener's resume work. Phase 3 should give push-delivered state a single owner, so that UI callbacks never wait behind display reads.

## Evidence index

- Phase-aligned profile and estimate: runs `fixedc` (uncached) and `cache2` (folder cache), 2026-11-20. The scripts were in Claude's scratchpad and are not committed.
- Workspace race root cause and fix: `0f6da22775`, plus the 2026-11-20 entry in `backlog/docs/lessons-testing-evidence.md`.
- Writer grant retirement fix: `49158d5407`, plus `Docs/Architecture/agent-runtime.md` (Windows AgentRuns writers).
- PERF-07/PERF-08 rule: ADR-126, "PERF-07/PERF-08 amendment".
- Code-identity rule: ADR-356 rule 4 and `Tests/Architecture/test_code_identity_ratchet.py` on `dev`.
