# PR #1157 fix-wave report (TASK-595 managed model acquisition)

Branch: `feat/managed-model-acquisition`. All fixes are TDD: each new/changed
test was confirmed to fail against the pre-fix code (via `git stash` of the
production file) before the fix was applied, then confirmed green after.

Full gate: `PYTHONPATH=<worktree> <worktree>/.venv/bin/pytest
Tests/Model_Artifacts/ Tests/STT/test_boundaries.py -q` → **379 passed**
(baseline was 366; 13 new tests added, 0 removed). The sealed TASK-594 suite
(`Tests/Model_Artifacts/test_service.py`, `test_operation_leases.py`,
`test_operation_leases_process.py`, 280 tests) has **zero diff** and stays
green. `Tests/Wizards/` (179 tests, unrelated to this PR) also re-verified
green as a sanity check (366+179=545, matching the baseline figure given).

---

## P1 — SECURITY: client-level Authorization leaks across cross-origin redirect

**Root cause.** `fetch.py`'s `stream_fetch` built its per-hop header dict
(`send_headers`) and stripped `_STRIP_HEADERS` from THAT dict only, then
passed it straight to `client.stream(...)`. httpx merges a client's own
default headers (e.g. `httpx.AsyncClient(headers={"Authorization": ...})`)
onto the request during `send()`, AFTER `send_headers` was already built —
so a client-level credential was invisible to the strip and reached a
cross-origin redirect target verbatim. The existing cross-origin test only
ever exercised the PER-CALL `headers=` argument, which was already handled
correctly.

**Change.** `stream_fetch` now builds the request explicitly via
`client.build_request(...)` and, on a cross-origin hop, pops
`_STRIP_HEADERS` off the BUILT request's merged headers (mirroring
`Utils/egress.py`'s `guarded_fetch_httpx_async`, which already does this).
`tldw_chatbook/Model_Artifacts/fetch.py`.

**Test.** `Tests/Model_Artifacts/test_credentials_and_boundaries.py::test_cross_origin_redirect_strips_client_level_default_authorization`
— two `FixtureArtifactServer`s, client constructed with a client-level
`Authorization` header, origin A redirects to origin B; asserts A saw the
header and B did not, and the body still downloaded.
Pre-fix: `AssertionError: the client-level credential must NOT have crossed
the origin boundary`. Post-fix: passes.

---

## P1 — RACE: install() staging-dir creation vs. reconcile()'s orphan lease

**Root cause.** `service.py`'s `install()` called `tempfile.mkdtemp(...)`
(which creates the directory as part of name generation) and only acquired
the per-directory `_install_staging_lease_key` lease AFTERWARD. Between
those two steps, the directory existed on disk with no lease protecting
it; a concurrent `reconcile()` pass could see it, acquire the (still-free)
lease itself, and delete a staging dir `install()` was about to write into.

**Change.** `install()` now generates the staging directory's NAME first
(`uuid.uuid4().hex`, prefixed `install-`), acquires the lease keyed on that
name, and only THEN calls `os.mkdir()`. Since `reconcile()`'s staging scan
only ever sees names that already exist on disk, nothing is visible to GC
until the lease is already held. `tldw_chatbook/Model_Artifacts/service.py`
(`ModelArtifactService.install`).

**Test.** `Tests/Model_Artifacts/test_reconcile_staging_gc.py::test_install_staging_directory_creation_is_atomic_with_lease_acquisition`
— monkeypatches `os.mkdir` (which BOTH `tempfile.mkdtemp` internally and the
new direct call go through, so the same probe works against either
implementation) to independently attempt a non-blocking acquire of the
directory's orphan-detection lease at the exact moment it's about to be
created. Pre-fix: the first (real creation) call finds the lease free
(`probe_acquired == [True, False, False]`, the later `False`s being
redundant `Path.mkdir(exist_ok=True)` calls from the copy phase that run
after pre-fix's own lease acquisition). Post-fix: `not any(probe_acquired)`
holds — the lease is always already held.

The sealed TASK-594 suite's own race-adjacent tests
(`test_reconcile_reports_live_pre_lifecycle_install_staging_entry`,
`test_install_acquires_exact_writer_leases_in_fixed_order`, etc.) all still
pass unmodified — they probe a LATER point in `install()` (after
`_copy_payload`) where the lease was already held either way.

---

## P2 — RETRYABLE LIE: install failure destroys the resumable download

**Root cause.** The fetch-state sidecar lived INSIDE the payload directory
(`staging_dir/fetch-state.json`). `core.install`'s payload-tree validation
rejects any file it doesn't declare, so `_install_artifact` deleted the
sidecar UNCONDITIONALLY before every `core.install` call — regardless of
whether install then succeeded, failed retryably, or failed for a reason
that never even touched the payload files. On the NEXT `provision()`
attempt, `_fetch_one_file` would see no sidecar (`recorded_done=0`) while
the actual file was fully present and correct, and
`_reconcile_durable_bytes` would then `truncate(0)` the good file, forcing
a full re-download — even though nothing was ever actually wrong with the
staged bytes.

**Change (the "preferred approach" from the finding).** The sidecar now
lives as a SIBLING file, `staging/managed/<id>/<rev>/<variant>.fetch-state.json`,
never a child of the payload directory `core.install` validates. Added
`_fetch_sidecar_path()` in `acquisition.py` and updated every reader/writer
to use it (`_fetch_artifact`, `_fetch_one_file`'s sidecar_path param,
`_preverify_one_file`, `_staged_bytes_for`). `_install_artifact` no longer
deletes the sidecar before calling `core.install`; it is now left
completely untouched until AFTER a successful install, then cleaned up
together with `staging_dir`. Updated the Task 2 GC classifier in
`service.py`: `_is_valid_managed_staging_entry` now handles either the
payload directory OR its sibling sidecar file as the scanned candidate,
resolving the counterpart and requiring BOTH to be real for the entry to
survive GC. Added a mirrored `_MANAGED_FETCH_SIDECAR_SUFFIX` constant in
`service.py` plus a drift-guard test asserting the two constants match
(mirrors the existing `fetch.py`/`egress.py` `_STRIP_HEADERS` pattern).

I judged the "preferred" relocation sufficient and did NOT reorder the
sealed core's `install()` internals (lease-acquire-then-copy vs.
copy-then-lease-acquire) — that would be a much larger, riskier change to
`ModelArtifactService.install()` (TASK-594, protected by the sealed-suite
constraint) for a narrower residual: a LATE lease-contention failure inside
`core.install`, occurring AFTER `consume_source` has already moved files
into the core's own ephemeral staging, still destroys those specific bytes
via that ephemeral staging's own failure-path `rmtree` — a genuine, but
separate, core-level design property, not something `_install_artifact`'s
own logic controls. The fix here closes the concrete, demonstrated
own-goal: acquisition.py destroying its own crumb trail before even
calling `core.install`, for ANY reason install might fail, including one
that never touches the payload at all.

**Tests (both required by the finding):**

(a) `Tests/Model_Artifacts/test_provision_install.py::test_retryable_install_failure_leaves_staged_bytes_resumable_via_range`
— seeds a partial file + sidecar, monkeypatches `core.install` to raise
`ArtifactStateError` directly (a retryable failure, independent of the
sealed core's internal mechanics), asserts the partial bytes and sidecar
survive untouched, then calls `_fetch_artifact` again and asserts the
request the fixture server received included a `Range` header (a resume,
not a full re-download) and the file completes correctly.
Pre-fix: `assert False` (no `Range` in the second request — the sidecar was
gone, forcing a from-scratch fetch). Post-fix: passes.

(b) GC classifies the new layout correctly:
`test_managed_entry_with_valid_sidecar_survives` (sibling sidecar + payload
dir → survives), `test_managed_entry_with_sidecar_inside_payload_dir_is_ignored_and_removed`
(sidecar in the OLD in-tree location no longer counts → orphan, removed),
`test_managed_entry_sidecar_without_payload_dir_is_removed` (stray sidecar
with no matching payload dir → removed), all in
`Tests/Model_Artifacts/test_reconcile_staging_gc.py`. Plus
`test_fetch_sidecar_suffix_mirror_matches_acquisition` (drift guard).

**Test churn (mechanical, not new coverage).** Every existing TASK-595 test
that hand-constructed a sidecar at the old `staging_dir / "fetch-state.json"`
path was updated to the sibling convention (`staging_dir.parent /
f"{staging_dir.name}.fetch-state.json"`): `test_preflight.py`,
`test_provision_fetch.py`, `test_provision_install.py`,
`test_provision_crash_recovery.py`, `test_credentials_and_boundaries.py`.
Two tests in `test_provision_install.py` were also renamed/re-asserted to
reflect the new "sidecar survives a failed install" behavior
(`test_install_failure_leaves_staging_dir_and_sidecar_intact_for_resume`,
formerly `test_install_failure_does_not_install_and_staging_dir_survives`,
which used to assert the OPPOSITE — that the sidecar was gone).

---

## P2 — ArtifactPathError escapes `_run_core_call` (closes TASK-1566)

**Root cause.** `_run_core_call` caught `ArtifactIntegrityError`,
`ArtifactConflictError`, and `ArtifactStateError` from `core.install`/
`core.activate`, but not `ArtifactPathError` (also a documented
`core.install` failure mode) — it escaped `provision()` raw, breaking the
spec's never-trap rule.

**Change.** Added `ArtifactPathError` to the non-retryable except tuple
(sibling of `ArtifactIntegrityError`/`ArtifactConflictError` under
`ArtifactError`, so exception ordering is unaffected) and imported it from
`.service`. `tldw_chatbook/Model_Artifacts/acquisition.py`.

**Test.** `Tests/Model_Artifacts/test_provision_install.py::test_install_artifact_wraps_core_path_error_as_non_retryable`
— monkeypatches `core.install` to raise `ArtifactPathError`, asserts
`TransferError(retryable=False)` with the cause chained. Pre-fix: the raw
`ArtifactPathError` propagates unwrapped. Post-fix: passes.

**Backlog.** TASK-1566 ("Wrap ArtifactPathError from core.install in the
acquisition never-trap taxonomy") set to Done with implementation notes, in
this commit. Note: this worktree's `backlog/tasks/` has a genuine TASK-1566
ID COLLISION with an unrelated task from another branch/worktree
("Wizard step compose() crash policy…" — filed on `wizard-loose-ends`,
unrelated to this PR). `backlog task edit 1566` resolves ambiguously
between the two same-numbered files, so I edited the correct file
(`task-1566 - Wrap-ArtifactPathError-...md`) directly rather than via the
CLI, to avoid silently corrupting the other task. The collision itself is
out of scope for this fix-wave (a backlog-hygiene issue, not a PR #1157
finding) and is left for whoever reconciles the two branches' task numbers.

---

## P2 — STALE-CHECKPOINT FAILURE: over-large checkpoint should restart, not fail

**Root cause.** `_reconcile_durable_bytes` only cross-checks a sidecar's
`bytes_done` against the file's ACTUAL on-disk size, never against the
file's CURRENT declared `size_bytes`. If a catalog's declared size for an
artifact/revision/variant shrank between provision() runs (a corrected or
re-cut upstream entry) while a fully-consistent (actual bytes == recorded
bytes) but now-oversized checkpoint survived, `resume_from` would end up
`>= max_bytes` inside `stream_fetch`, raising `FetchTooLargeError` — wrapped
by `_fetch_one_file` as a NON-retryable "upstream body exceeds declared
size" `TransferError`, when a clean restart-from-zero was really called for.

**Change.** After `_reconcile_durable_bytes` returns, normalize
`recorded_done` to 0 if it exceeds `file.size_bytes`, before deriving
`resume_from`. `stream_fetch`'s `mode="wb"` path (used when `resume_from ==
0`) naturally truncates whatever stale bytes remain on disk.
`tldw_chatbook/Model_Artifacts/acquisition.py` (`_fetch_one_file`).

**Test.** `Tests/Model_Artifacts/test_provision_fetch.py::test_fetch_over_large_checkpoint_restarts_cleanly`
— seeds a 4000-byte on-disk file + matching sidecar claiming 4000 bytes,
while the descriptor's CURRENT declared size is only 1000 bytes; asserts
the fetch restarts cleanly (full GET, no `Range` header) and completes with
the correct 1000-byte body. Pre-fix: raises `TransferError` ("upstream body
exceeds declared size... resume offset already at or past the bound").
Post-fix: passes.

---

## P2 — PREFLIGHT CREDIT: staged credit not capped by actual on-disk size

**Root cause.** `_staged_bytes_for` summed a sidecar's `bytes_done` per
file, capped only by the ENTRY's aggregate declared total (`entry.total_bytes`)
— never by the file's actual on-disk size, and without checking that the
sidecar's file-path key was even one of the descriptor's declared files. A
stale sidecar claiming bytes for a file that doesn't exist (or is smaller
than claimed) inflated `already_staged_bytes`, letting preflight's space
math approve an acquisition that could then run out of space partway
through the real download.

**Change.** `_staged_bytes_for` now takes the full `descriptor` (not just
the `ArtifactRef`) and caps each file's credit by
`min(recorded bytes_done, actual on-disk file size, declared file size)`;
sidecar entries naming a file the descriptor doesn't declare are ignored
outright. `tldw_chatbook/Model_Artifacts/acquisition.py`
(`_staged_bytes_for`, `_aggregate_closure`'s call site).

**Test.** `Tests/Model_Artifacts/test_preflight.py::test_preflight_stale_sidecar_credit_capped_by_actual_file_size`
— sidecar claims 5000 bytes for a 2048-byte declared file whose staged file
on disk is only 100 bytes; asserts `already_staged_bytes == 100` (not 2048
or 5000). Pre-fix: `assert 2048 == 100` (capped only by declared total).
Post-fix: passes. Two pre-existing tests
(`test_preflight_counts_staged_credit`,
`test_preflight_clamps_oversized_staged_credit_to_entry_total`) were
updated to also write a real on-disk file matching their sidecar's claim
(previously they asserted credit for a file that was never actually
staged, tolerated only because the old code never checked). Both also had
a latent filename bug (sidecar key `"m.onnx"` vs. the descriptor's real
declared file `"model.onnx"` from `make_descriptor`) that the new
"ignore undeclared files" check surfaced and required fixing.

---

## P2 — MISSING ETag / P2 — UNVALIDATED Content-Range on resume (fetch.py)

Both findings live in the same `stream_fetch` hop-response-handling block
and were fixed together.

**Root cause (missing ETag).** `stream_fetch` only rejected a resumed
response as a validator mismatch when the server's replied `ETag` was
present AND differed from the saved one (`if resume_from and validators
and validators.etag and got.etag: if got.etag != validators.etag: raise`).
A 206 that omitted `ETag` entirely (`got.etag` falsy) skipped the check
silently, treating "no information" as "matches".

**Root cause (Content-Range).** A 206 status code alone does not prove the
response body starts at the requested `Range` offset — only `Content-Range`
does. `stream_fetch` never parsed or checked it at all, so a server (or a
buggy proxy) answering 206 with a body that doesn't actually start where
requested would have its bytes silently appended to stale on-disk data.

**Change.** In `tldw_chatbook/Model_Artifacts/fetch.py`: (1) the ETag check
now fires whenever `validators.etag` was set and the resumed response's
`got.etag` is EITHER missing or different (`if not got.etag or got.etag !=
validators.etag: raise FetchRestartRequired`). (2) Added
`_parse_content_range_start()` (a small regex-based parser for `Content-
Range: bytes <start>-<end>/<total>`) and, whenever `resume_from` is set and
the response is 206, require the parsed start to equal `resume_from` — a
missing or unparseable header, or a mismatched start, raises
`FetchRestartRequired` BEFORE the destination is ever opened for append.

Both checks required updating `Tests/Model_Artifacts/fixture_http.py`'s
`FixtureArtifactServer` to actually send `Content-Range` on 206 responses
(it previously sent none at all — every existing resume test only ever
passed because nothing validated it); added `omit_content_range` and
`bad_range_start` route options to construct the adversarial cases without
breaking any legitimate resume test (the fixture defaults to reporting the
REAL slice offset it just served, so every pre-existing resume test's
`Content-Range` now correctly matches `resume_from` with no test changes
needed beyond the fixture itself).

**Tests**, all in `Tests/Model_Artifacts/test_stream_fetch.py`:
- `test_missing_etag_on_resume_raises_restart_without_append` — route
  serves 206 with no ETag (`ignore_if_range=True` forces 206 despite the
  If-Range mismatch this produces); asserts `FetchRestartRequired` and no
  bytes appended. Pre-fix: `DID NOT RAISE FetchRestartRequired`.
- `test_content_range_start_mismatch_raises_restart_without_append` — route
  reports `bad_range_start=999` while resuming from 100; asserts
  `FetchRestartRequired`, no append. Pre-fix: did not raise.
- `test_missing_content_range_on_resume_raises_restart_without_append` —
  route omits `Content-Range` entirely on a real 206; asserts
  `FetchRestartRequired`, no append. Pre-fix: did not raise.

All three pass post-fix; all three confirmed to fail against pre-fix
`fetch.py` (via `git stash`).

---

## BUG — BLOCKING HASH: `_preverify_one_file` hashes synchronously

**Root cause.** `_preverify_one_file` is `async def`, but called
`self._hash_staged_file(...)` (a plain synchronous method) directly, with
no `await` anywhere in the loop. Hashing a multi-gigabyte staged file this
way blocks the ENTIRE event loop — every other coroutine, including any
other artifact's fetch/pre-verify progress and the app's own UI loop if
this service shares one — for as long as the hash takes.

**Change.** `_hash_staged_file` now runs via `loop.run_in_executor(None,
...)`, called from `_preverify_one_file`. Progress-callback invocation
inside `_hash_staged_file` is marshalled back onto the event loop via
`loop.call_soon_threadsafe(progress_state.callback, event)` rather than
called directly from the executor thread — justified because
`progress_state.callback` is caller-supplied and, per this codebase's
"Background Work" threading convention (CLAUDE.md: workers call back via
`call_from_thread`, never directly), may touch UI state that must only be
touched from the event-loop thread. Byte-counter mutation
(`progress_state.preverify_bytes_done += ...`) stays directly on the
executor thread — safe because `_preverify_artifact` processes one file at
a time, so only one executor call is ever hashing at once; verified this
doesn't reorder or drop progress events because `run_in_executor`'s own
"future done" notification is scheduled via the SAME `call_soon_threadsafe`
mechanism, strictly AFTER every per-chunk callback scheduled while the
executor function was still running (asyncio's ready queue is FIFO), so by
the time the awaiting coroutine resumes, all progress events for that hash
have already fired in order.
`tldw_chatbook/Model_Artifacts/acquisition.py` (`_preverify_one_file`,
`_hash_staged_file`).

**Test.** `Tests/Model_Artifacts/test_provision_install.py::test_preverify_hashing_does_not_block_event_loop`
— monkeypatches `hashlib.sha256` (as imported into `acquisition.py`) to a
wrapper that sleeps 0.05s per `update()` call (8 chunks ≈ 0.4s total),
races a concurrent "heartbeat" coroutine ticking every ~0.01s alongside the
hash, and asserts the heartbeat ticks at least 3 times during the hash.
Pre-fix: `assert 0 >= 3` (the synchronous call never yields control, so the
heartbeat coroutine never gets scheduled at all during the ~0.4s hash).
Post-fix: passes.

---

## Files changed

- `tldw_chatbook/Model_Artifacts/fetch.py` — client-default header strip,
  Content-Range validation, missing-ETag rejection.
- `tldw_chatbook/Model_Artifacts/service.py` — atomic staging-dir creation
  vs. lease; sidecar-sibling-aware GC classifier.
- `tldw_chatbook/Model_Artifacts/acquisition.py` — sidecar relocation
  (`_fetch_sidecar_path`), `ArtifactPathError` wrapping, stale-checkpoint
  normalization, preflight staged-credit capping, off-loop hashing.
- `Tests/Model_Artifacts/fixture_http.py` — `Content-Range` on 206
  responses; `omit_content_range`/`bad_range_start` route options.
- `Tests/Model_Artifacts/test_stream_fetch.py`,
  `test_credentials_and_boundaries.py`, `test_preflight.py`,
  `test_provision_fetch.py`, `test_provision_install.py`,
  `test_provision_crash_recovery.py`, `test_reconcile_staging_gc.py` — new
  regression tests + sidecar-path/layout updates for existing tests.
- `backlog/tasks/task-1566 - Wrap-ArtifactPathError-....md` — closed.
