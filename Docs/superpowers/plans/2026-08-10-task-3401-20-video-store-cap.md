# TASK-3401.20 Generated Video Store Capacity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce the configured generated-video capacity after every save while safely resolving oversized or failed managed saves through explicit Keep/Retry, external-save, or discard choices.

**Architecture:** `VideoStore` owns one non-following, root-serialized capacity transaction using an instance `RLock` plus the existing `portalocker` dependency. `run_video_generation()` converts capacity and managed-save failures into a temporary-file-backed `PendingVideoArtifact`; one `ChatScreen` resolver owns that artifact across a three-choice modal, file picker, retry/adoption, external no-clobber copy, and teardown. Existing provider, message, and startup-retention boundaries remain unchanged.

**Tech Stack:** Python 3.11, Textual 8, pytest, Loguru, `portalocker`, standard-library `tempfile`, `threading`, `os`, `shutil`, and `subprocess`.

---

## File map

- Modify `tldw_chatbook/Video_Generation/video_store.py`: safe inventory, typed store outcomes/errors, bounded root-scoped locking, ordinary cap transaction, sole-oversized adoption, and atomic stream publication.
- Modify `tldw_chatbook/Chat/console_generate_video.py`: `PendingVideoArtifact`, metadata-before-storage, and recovery of capacity/store failures without provider changes.
- Create `tldw_chatbook/Widgets/Console/console_video_capacity_modal.py`: focused three-choice modal returning one typed action.
- Modify `tldw_chatbook/UI/Screens/chat_screen.py`: one initial/Regenerate result resolver, pending-artifact registry, retry/adopt, picker/overwrite flow, atomic external copy, OS open, and unmount cleanup.
- Modify `Tests/Video_Generation/test_video_store.py`: real file, security, ordering, failure, lock-timeout, thread, and process evidence.
- Modify `Tests/Chat/test_console_generate_video.py`: normal tuple and both pending-artifact reasons/ownership.
- Create `Tests/Chat/test_console_video_capacity.py`: modal and mounted Console outcome flows, overwrite races, parity, and teardown.
- Re-run only the existing generated-video startup-containment case in `Tests/ProductionApp/test_chat_composition_retirement.py`; modify it only if the new typed busy error requires a missing assertion.
- Modify `backlog/tasks/task-3401.20 - Enforce-generated-video-store-size-cap-after-every-save.md`: keep the plan, later check ACs, and record exact evidence/commits.
- Keep provider adapters, workflow JSON, message schema, config schema, and `tldw_chatbook/app.py` unchanged.

## Test-scope constraint

The user explicitly authorized only tests related to touched files. Do not run the full suite, broad `Tests/Chat`, broad `Tests/Video_Generation`, broad `Tests/ProductionApp`, RuntimePolicy, or live generation UAT. Use the repository-root virtual environment if this worktree has no `.venv`.

Final automated gate:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest \
  Tests/Video_Generation/test_video_store.py \
  Tests/Chat/test_console_generate_video.py \
  Tests/Chat/test_console_video_capacity.py \
  Tests/ProductionApp/test_chat_composition_retirement.py::test_video_retention_startup_failure_is_bounded \
  -q
```

Record the expanded test node list/count in the task report; do not silently run unrelated tests from the ProductionApp file.

Static gate:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  tldw_chatbook/Video_Generation/video_store.py \
  tldw_chatbook/Chat/console_generate_video.py \
  tldw_chatbook/Widgets/Console/console_video_capacity_modal.py \
  Tests/Video_Generation/test_video_store.py \
  Tests/Chat/test_console_generate_video.py \
  Tests/Chat/test_console_video_capacity.py
```

Run `ruff E9,F63,F7,F82` plus `py_compile` on `chat_screen.py` if whole-file Ruff has unrelated baseline findings; document the exact scope rather than claiming full-file Ruff.

### Task 1: Secure and serialize `VideoStore` capacity transactions

**Files:**
- Modify: `Tests/Video_Generation/test_video_store.py`
- Modify: `tldw_chatbook/Video_Generation/video_store.py`

- [x] **Step 1: Add named RED tests for ordinary saves and startup separation**

Add tests using the existing `_config(max_store_mb=1)` and real `tmp_path` files:

```python
def test_save_enforces_cap_oldest_first_without_startup_cleanup(tmp_path):
    store = VideoStore(root=tmp_path / "gv", config=_config(retention="session", max_store_mb=1))
    oldest = store.save("old", "clip", b"a" * 600_000)
    os.utime(oldest, (1, 1))
    survivor = store.save("new", "clip", b"b" * 300_000)
    newest = store.save("latest", "clip", b"c" * 600_000)

    assert not oldest.exists()
    assert survivor.exists()
    assert newest.exists()
    assert sum(item.size_bytes for item in store.iter_stored()) <= 1024 * 1024
```

Use payload sizes/order that require exactly the oldest victim and prove an unrelated current-run survivor remains. Assert `resolve(old_message_id, old_slug) is None` while newer pairs still resolve. Add equal-mtime candidates whose safe paths sort differently and prove the documented path tie-breaker makes the same victim deterministic. Parameterize the startup-policy-separation case over both `session` and `ttl`; spy on `enforce_retention()` and make any save-time call fail, proving cap-only enforcement never reuses either startup policy.

- [x] **Step 2: Add RED overflow and sole-exception tests**

Expect a frozen `VideoCapacityExceeded(size_bytes, max_bytes)` from `save()` for a `1 MiB + 1` payload, with no managed file. Add `adopt_oversized(message_id, slug, stream, size_bytes)` expectations proving:

- old managed files are removed only after the candidate is complete;
- exactly one oversized file remains;
- fresh TTL startup keeps that sole exception;
- session startup removes it; and
- a later ordinary save removes it before reporting success.

- [x] **Step 3: Add RED non-following deletion tests**

On POSIX, create an external directory containing `PRIVATE-SENTINEL`, symlink `store.root / "linked-message"` to it, and force ordinary capacity, startup capacity, and `adopt_oversized()` scans. Assert the external file and bytes are unchanged and the symlink is not traversed. Add a direct file symlink case. On Windows, create a junction/reparse fixture where supported and skip only when construction is unavailable.

Mutation requirement: temporarily change the inventory directory check to follow links; the POSIX sentinel test must fail.

- [x] **Step 4: Add RED bounded locking and concurrency tests**

Add deterministic tests for:

1. two threads using one `VideoStore` cannot overlap the internal transaction;
2. two independent `VideoStore` objects with the same root serialize through the root lock;
3. a spawned process holding the stable lock makes another save raise `VideoStoreBusyError` within a monkeypatched short timeout; and
4. two spawned processes saving into a 1 MiB root finish with an actual total at or below 1 MiB.

For the thread case, patch only the root-lease helper to `nullcontext`, block the first thread inside atomic publication with Events, and prove the second thread cannot enter publication until release; removing `RLock` must set the forbidden second-entered Event. For process cases, use a top-level spawn-safe worker and `multiprocessing.get_context("spawn")`; do not reload the module that defines result/error types. Capture child outcomes through a queue and enforce bounded joins.

- [x] **Step 5: Add RED publication and victim-deletion failure tests**

Plant byte-distinct old files that are within the cap. For publication failure, patch only the private atomic-commit seam to raise `OSError` after the complete sibling exists but before target replacement; assert `VideoStoreSaveError`, every old path/byte remains exact, the new target is absent, and no sibling remains.

For required-victim deletion failure, allow new candidate publication, patch the checked-unlink seam to fail on the first required old victim, and assert `VideoStoreSaveError`, the new managed target is withdrawn, the failing old path/bytes remain, actual managed bytes do not falsely report success, and no partial sibling remains. Add a second case that permits one earlier oldest victim to be deleted before a later victim fails, proving the resulting surviving store remains bounded and the new target is still withdrawn.

- [x] **Step 6: Run the storage tests to verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest Tests/Video_Generation/test_video_store.py -q
```

Expected: failures for missing typed outcomes, post-save enforcement, safe inventory, lock timeout, process serialization, and oversized adoption. Record the exact failure set.

- [x] **Step 7: Implement minimal typed contracts and lock boundary**

In `video_store.py`, add:

```python
@dataclass(frozen=True)
class VideoCapacityExceeded:
    size_bytes: int
    max_bytes: int


class VideoStoreSaveError(RuntimeError):
    """Managed publication or capacity enforcement failed."""


class VideoStoreBusyError(VideoStoreSaveError):
    """The root-scoped capacity lease was not acquired in time."""
```

Add a read-only `capacity_bytes` property that applies the existing integer-MiB normalization once and is the public source for storage/generation outcome metadata. Initialize `threading.RLock()` per store. Add a private context manager that opens the stable sibling lock file, attempts `portalocker.LockFlags.EXCLUSIVE | NON_BLOCKING` until `time.monotonic()` reaches the five-second module constant, and always unlocks/closes in `finally`. Acquire `RLock` before the process lease everywhere; private helpers assume the transaction is already held and never reacquire it.

Do not reuse the app's profile-instance warning lock: it is detection-only and concurrent app instances remain allowed.

- [x] **Step 8: Implement non-following inventory and deletion**

Replace follow-link `Path.is_dir()/is_file()/stat()` inventory with an internal snapshot based on `os.scandir` and `entry.stat(follow_symlinks=False)`. Add one helper that rejects:

- unsafe message components;
- non-real directories;
- symlink/reparse entries (`st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT` when available);
- nested entries; and
- parents resolving outside the resolved root.

Repeat those checks immediately before unlink. Return immutable `StoredVideo` snapshots; never log rejected paths or external targets. Public `iter_stored()` returns an iterator over a completed safe snapshot so no lock is held across caller iteration.

- [x] **Step 9: Implement atomic publish, ordinary eviction, and oversized adoption**

Create a complete private sibling with `tempfile.NamedTemporaryFile(delete=False, dir=target.parent)`, flush it, then `os.replace()` it into the target; unlink any unpublished sibling in `finally`. For ordinary saves:

- return `VideoCapacityExceeded` before any managed write when the payload alone is over cap;
- publish the complete target under both locks while prior files remain;
- evict oldest existing candidates, never the new target;
- if eviction/check fails, remove the new target and raise `VideoStoreSaveError`;
- re-snapshot actual bytes before returning the `Path`.

For `adopt_oversized()`, rewind/copy the caller-owned stream without closing it, publish a candidate, remove every prior managed file, roll back the candidate on failure, verify it is the sole file, then return the path. Modify startup cap enforcement so one sole oversized survivor is retained after TTL filtering, but multiple/ordinary over-cap survivors are reduced oldest-first. Keep startup per-file removal best-effort while save/adopt operations fail closed.

- [x] **Step 10: Run storage tests GREEN and perform mutations**

Run the Task 1 command. Expected: all tests pass. Then prove named failures by separately removing:

- save-time cap enforcement;
- oldest-first sort;
- `RLock`;
- portalocker acquisition;
- non-following directory check; and
- sole-oversized startup exception.

Also mutate the atomic-commit and checked-unlink failure rollbacks so the new candidate remains; the new named failure tests must detect both false-success/data-loss variants.

Restore exactly after each mutation and rerun GREEN.

- [x] **Step 11: Commit the storage boundary**

```bash
git add tldw_chatbook/Video_Generation/video_store.py Tests/Video_Generation/test_video_store.py
git diff --cached --check
git commit -m "fix: enforce generated video store capacity"
```

### Task 2: Preserve unstored generation results as pending artifacts

**Files:**
- Modify: `Tests/Chat/test_console_generate_video.py`
- Modify: `tldw_chatbook/Chat/console_generate_video.py`

- [x] **Step 1: Write RED tests for both pending reasons**

Extend the existing fake adapter path with payloads larger than a 1 MiB store. Assert `run_video_generation()` returns `PendingVideoArtifact(reason="over_capacity")`, no managed file, exact metadata/size/cap/message id, and a rewound temporary handle containing exact bytes.

Inject `VideoStoreSaveError("PRIVATE-PATH")` from a store after generation and assert a `reason="store_failure"` artifact holds exact bytes, obtains `max_bytes` from the store's public `capacity_bytes`, and exposes only `error_type == "VideoStoreSaveError"`, never exception text. Hold the real root lease through a second `VideoStore`, shorten the timeout, and prove real contention returns the same readable `store_failure` artifact instead of hanging. Verify `close()` is idempotent and makes later reads fail.

- [x] **Step 2: Run focused generation tests RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest Tests/Chat/test_console_generate_video.py -q -k 'pending_video or run_video_generation_saves'
```

Expected: missing artifact/result union and current exception propagation.

- [x] **Step 3: Implement the minimal artifact contract**

Add:

```python
PendingReason = Literal["over_capacity", "store_failure"]

@dataclass
class PendingVideoArtifact:
    metadata: VideoGenerationMetadata
    message_id: str
    slug: str
    extension: str
    size_bytes: int
    max_bytes: int
    reason: PendingReason
    stream: BinaryIO = field(repr=False)
    error_type: str | None = None

    def rewind(self) -> None: ...
    def close(self) -> None: ...  # idempotent
```

Build metadata immediately after the adapter result. Call `VideoStore.save()` while the adapter bytes remain owned. On `VideoCapacityExceeded` or `VideoStoreSaveError`, write those bytes once to `tempfile.TemporaryFile(mode="w+b")`, rewind, and return the artifact. If staging itself fails, close the temporary handle and re-raise; do not create a half-valid artifact.

- [x] **Step 4: Run generation tests GREEN and mutate ownership**

Run the Task 2 command and then the complete file. Remove the rewind and idempotent close guards separately; the exact artifact tests must fail. Restore and rerun GREEN.

- [x] **Step 5: Commit the generation contract**

```bash
git add tldw_chatbook/Chat/console_generate_video.py Tests/Chat/test_console_generate_video.py
git diff --cached --check
git commit -m "fix: preserve unstored generated videos"
```

### Task 3: Add the focused capacity-choice modal

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_video_capacity_modal.py`
- Create: `Tests/Chat/test_console_video_capacity.py`

- [x] **Step 1: Write modal RED tests**

Create a minimal Textual test app and assert the mounted modal:

- displays generated size and configured cap without a path/prompt;
- for `over_capacity`, offers `Keep here — remove other videos`, `Save to disk`, and `Discard`;
- for `store_failure`, offers `Retry here`, `Save to disk`, and `Discard`;
- dismisses typed literals `"keep"`, `"save_external"`, or `"discard"`; and
- maps Escape to `"discard"`.

Add geometry assertions proving all three buttons remain inside a 90-column screen.

- [x] **Step 2: Run modal tests RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest Tests/Chat/test_console_video_capacity.py -q -k 'modal'
```

Expected: import failure because the modal does not exist.

- [x] **Step 3: Implement one narrow modal**

Create `ConsoleVideoCapacityModal(ModalScreen[CapacityAction])` with a frozen input model or explicit constructor arguments for reason/size/cap. Use three Buttons in one responsive container, `markup=False` for dynamic copy, and no storage/Console imports. Button handlers only stop the event and dismiss the typed result. Escape calls the discard action.

- [x] **Step 4: Run modal tests GREEN and mutate action routing**

Run the Task 3 command. Temporarily map Keep and Escape to the wrong results; named tests must fail. Restore and rerun GREEN.

- [x] **Step 5: Commit the modal**

```bash
git add tldw_chatbook/Widgets/Console/console_video_capacity_modal.py Tests/Chat/test_console_video_capacity.py
git diff --cached --check
git commit -m "feat: add generated video capacity choices"
```

### Task 4: Resolve pending video outcomes through the real Console paths

**Files:**
- Modify: `Tests/Chat/test_console_video_capacity.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/Video_Generation/test_video_store.py`
- Modify: `tldw_chatbook/Video_Generation/video_store.py`
- Modify: `Tests/Chat/test_console_generate_video.py`
- Modify: `tldw_chatbook/Chat/console_generate_video.py`

> **Review amendment:** mounted teardown must be linearizable with final
> managed-store publication. A screen-owned cancellation flag alone has a
> check/commit race, while private store locks or an unlocked compensating
> unlink violate the storage boundary. The store therefore accepts a narrow
> optional publication gate whose lock spans the final active check and
> commit. Existing callers retain the unchanged default path.

- [x] **Step 1: Write RED tests for shared initial/Regenerate dispatch**

Drive `_console_command_generate_video()` and `_regenerate_console_video_message()` with `asyncio.to_thread` returning real temporary-file-backed artifacts. Assert both call one `_resolve_generated_video_outcome(...)` seam. For normal tuple results, preserve current append/sync behavior. For Discard and picker cancellation, assert no `append_video_message`, stage closed, and in-flight/cancel bookkeeping cleared.

- [x] **Step 2: Write RED Keep/Retry tests with real `VideoStore` files**

For oversized Keep, plant two managed videos and associated resolvable specs, choose Keep, and prove both old paths disappear, the oversized target is the sole file, and exactly one new card is appended only after `resolve()` succeeds. Inject an adoption unlink failure and prove no card, readable stage, sanitized notification, and the choice is re-offered.

For `store_failure`, choose Retry and prove the normal capped save path is called without `adopt_oversized()` or evict-all copy. A repeated busy/failure outcome must re-offer choices without recursion or artifact loss.

- [x] **Step 3: Write RED external-save race tests**

Patch `EnhancedFileSave` only at the modal-result boundary while using real files for copy behavior. Assert:

- default filename is `<slug>.mp4`;
- picker cancellation closes stage and appends no card;
- a missing destination created by another actor before commit is not overwritten and returns to confirmation;
- an existing destination requires confirmation;
- declining preserves its bytes and returns to picker;
- changing `(lstat dev, ino, size, mtime_ns, mode)` after confirmation triggers fresh confirmation;
- confirmed unchanged replacement receives exact bytes atomically;
- success opens the final path and appends no card; and
- OS-open failure keeps the saved file and reports “saved, could not open”.

Inject an `OSError` during sibling copy and separately during commit; both must leave the original destination unchanged, remove the incomplete sibling, keep the pending artifact readable, append no card, and re-offer choices.

Use a complete sibling plus an atomic hard-link/no-clobber commit for new targets. For confirmed replacement, revalidate non-following identity immediately before `os.replace`. Always remove the temporary sibling.

- [x] **Step 4: Write RED mounted teardown tests**

Run a mounted production `ChatScreen` with a resolver waiting first on `ConsoleVideoCapacityModal`, then separately on `EnhancedFileSave`. Navigate away/unmount through the real screen lifecycle. Assert `_pending_video_artifacts` is drained, every stream is closed once, the modal/picker waiter may finish late without error, and no card/path is created afterward.

Also pause ordinary retry and oversized adoption immediately before final
publication. Let unmount win the shared publication gate and prove the store
aborts without a target or partial stage. If publication wins first, it is
linearized before teardown; teardown must still prevent every later card or UI
continuation. Do not call private store transaction helpers or perform an
unlocked compensating unlink from `ChatScreen`.

Create that same per-message gate before initial or Regenerate generation
enters `run_video_generation()` and pass it through the normal `VideoStore.save`
boundary. A cancel-winning generation leaves no path. A commit-winning normal
or pending result persists its durable message metadata even if teardown wins
screen ownership immediately afterward, but skips stale-screen UI work. This
prevents a committed managed file from becoming an orphan.

- [x] **Step 5: Run Console capacity tests RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest Tests/Chat/test_console_video_capacity.py -q -k 'not modal'
```

Expected: failures for the missing resolver, registry, external-copy helper, and shared dispatch wiring.

- [x] **Step 6: Implement screen ownership and result resolution**

In `ChatScreen`, lazily own `dict[str, PendingVideoArtifact]`. Add one resolver that:

1. directly appends/syncs a normal tuple;
2. registers a pending artifact before pushing any modal;
3. loops on the accurate modal result;
4. runs store retry/adoption and external copies in `asyncio.to_thread`;
5. appends a card only after managed resolution succeeds;
6. reports bounded action errors and re-offers choices; and
7. unregisters/idempotently closes in `finally`.

All modal and picker waits use a small screen helper that starts `app.push_screen_wait(...)` through `run_worker(exclusive=False, exit_on_error=False)` and awaits `worker.wait()`. Do not bare-await `push_screen_wait` from the slash-command event stack. Before every append, external open, or retry continuation, require `_pending_video_artifacts.get(message_id) is artifact`; unmount drains the mapping, so a late modal/picker callback becomes a no-op rather than appending after navigation.

Use one small thread-safe publication gate for active store work. `VideoStore`
holds that gate across its final active check and candidate commit; teardown
cancels under the same gate and defers stream closure while a worker still
owns it. This makes pre-publication cancellation atomic without exposing
private store locks or deleting a committed path outside the store.

External saves require pinned directory-relative staging, commit, and cleanup.
Select that capability before allocating the sibling. If the platform cannot
provide the required non-following directory operations, fail closed and
retain the pending artifact for another choice; do not fall back to a
path-re-resolving commit that can be redirected by a parent swap.

Replace both tuple-unpack call sites with this resolver. Add one line in `on_unmount()` calling a small drain helper; do not inline artifact cleanup into the already-large teardown method.

- [x] **Step 7: Implement external copy and OS-open helpers**

Keep helpers private to `chat_screen.py` unless tests prove a second production owner. Reuse `EnhancedFileSave` and `ConfirmationDialog`; do not modify the shared picker. Extract the existing platform opener from `_play_console_video()` so managed playback fallback and newly external-saved playback share one path. Pass argv lists, never `shell=True`.

External destination errors may include the chosen path in user-facing escaped notifications, but diagnostic logs contain only operation/error type. Never log prompt, staged filename, bytes, or exception text.

- [x] **Step 8: Run Console tests GREEN and mutate load-bearing seams**

Run `Tests/Chat/test_console_video_capacity.py` and the touched `Tests/Chat/test_console_generate_video.py`. Prove named failures after separately removing:

- initial or Regenerate resolver call;
- pending registration or unmount drain;
- run+message ownership check before a late callback;
- no-clobber commit;
- identity revalidation;
- finally-close; and
- append-after-resolve ordering.

Restore after each mutation and rerun GREEN.

- [x] **Step 9: Commit the Console integration**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/Chat/test_console_video_capacity.py
git diff --cached --check
git commit -m "feat: resolve generated video capacity outcomes"
```

### Task 5: Focused verification, review, and backlog closeout

**Files:**
- Modify: `backlog/tasks/task-3401.20 - Enforce-generated-video-store-size-cap-after-every-save.md`
- Modify only if required: `Tests/ProductionApp/test_chat_composition_retirement.py`
- Modify: `backlog/docs/lessons-testing-evidence.md`

- [x] **Step 1: Run the exact related-file gate**

Run all tests in the three touched/new focused files plus only the exact existing generated-video startup-containment node. Record command, count, warnings, duration, and any justified deviation. Do not run broad/full collections.

- [x] **Step 2: Run static and syntax checks**

Run full Ruff on the six focused production/test files listed in the static gate. Run targeted fatal-rule Ruff on `chat_screen.py` if it retains unrelated baseline findings. Compile all four touched production modules to a `TemporaryDirectory` destination so no `__pycache__` artifacts enter the worktree. Run `git diff --check`.

- [x] **Step 3: Audit security, privacy, scope, and artifacts**

Verify:

- the linked external sentinel tests are non-vacuous;
- no log statement includes pending/external/private paths, prompt text, bytes, or raw exception text;
- no provider, workflow JSON, schema, config schema, or unrelated UI file changed;
- no media, lock, build, temporary sibling, `.pyc`, or cache artifact is tracked/untracked; and
- the diff from `076c39219` contains only planned production/tests/task/plan paths.

- [x] **Step 4: Request code review and fix findings separately**

Use `superpowers:requesting-code-review` against the implementation range. Any substantive fix gets its own RED/GREEN evidence and separate commit; do not amend reviewed commits. Re-run only the affected focused tests plus the final exact gate.

- [x] **Step 5: Complete the task through Backlog CLI**

After all evidence is fresh:

1. check AC #1–#8;
2. replace the task's Implementation Notes with the concise approach, files, ADR-044 revision, exact test/static/mutation evidence, commits, no-live-UAT reason, and no-new-lesson or incident-backed lesson decision; and
3. run `backlog task edit 3401.20 -s Done`.

Re-read with `backlog task 3401.20 --plain` and confirm status, checked ACs, plan, notes, and ADR hygiene survived the CLI edit.

- [x] **Step 6: Commit closeout documentation**

```bash
git add \
  "backlog/tasks/task-3401.20 - Enforce-generated-video-store-size-cap-after-every-save.md" \
  Docs/superpowers/plans/2026-08-10-task-3401-20-video-store-cap.md
git diff --cached --check
git commit -m "docs: complete generated video capacity task"
```

- [x] **Step 7: Push and update the existing draft PR**

Push `codex/task-3401-14-h3-generation-uat`, update draft PR #1460 with TASK-3401.20 behavior/evidence/commits, and leave it draft if later work-stream tasks remain. Verify remote HEAD matches local HEAD and the worktree is clean.
