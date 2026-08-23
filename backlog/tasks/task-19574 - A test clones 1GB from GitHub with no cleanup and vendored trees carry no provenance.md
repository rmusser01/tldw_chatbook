---
id: TASK-19574
title: >-
  A test clones 1 GiB from GitHub with no cleanup, vendored trees carry no
  provenance, and the repo is 47 GiB with gc suppressed
status: In Progress
assignee: []
created_date: '2026-08-21 20:27'
labels:
  - testing
  - infrastructure
  - vendoring
priority: medium
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 7 (process, tooling & repo health) —
its **F4**, **F5** and **F6**. Grouped as repo custody. All re-measured at this
branch base.

**A — the vendoring sync test does not skip; it clones 1 GiB inside the test.**
`Tests/Chunking/test_sync_script.py:15-22`:

```python
SOURCE = os.environ.get("TLDW_SERVER_SYNC_SOURCE", "/tmp/tldw_server_sync")

def _run_sync() -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(SYNC)]
    if Path(SOURCE).exists():
        cmd += ["--source", SOURCE]
    return subprocess.run(cmd, ...)
```

There is **no `pytest.skip`**. An absent source silently falls through to the
no-arg path, and `Helper_Scripts/sync_chunking_engine.py:646-649` then does
`tempfile.mkdtemp(...)` + `git clone --no-checkout` from GitHub — and **never
removes `tmp`**. `test_sync_idempotent_and_rejects_local_edits` calls
`_run_sync()` **three times**, so three clones per run.

**Four leaked clones found on this machine, 3.9 GiB, nothing deleted:**
`tldw_server_sync_96ul07_k`, `_qx21_8na`, `_epobbh2a`, `_ehglb1h1`, all dated
Aug 20, under the macOS `TMPDIR`
(`/var/folders/p_/.../T/`) — **not** `/private/tmp` as the review assumed,
because that is where `mkdtemp` lands. Each is **1.0 GiB**, not the
"~532 MiB" the test's own comment claims.

The repo's network guard **cannot** stop this — it patches sockets, and this is
a `git` subprocess. The test also rewrites all 35 vendored files and 40 ported
tests on every run, so a crash mid-test leaves a corrupted tree.

Note `/private/tmp/tldw_server_sync` currently exists on this machine (1.0 GiB,
HEAD at the pin), so the test takes the `--source` path **here**. The
fallthrough is latent locally and live on any other machine or in CI. Also note
`/private/tmp` is where the standing rule says never to keep work — the macOS
cleaner has destroyed a worktree there three times.

`verify_clean()` overstates its guarantee (no `--porcelain` check), but the
hazard is **mooted** because every read is `git show PIN:path` from the object
store — do not spend time on it.

**The pin is duplicated in six authoritative places with nothing cross-checking
them**, and there is no documented upgrade path:
`Chunking/engine/VENDOR_MANIFEST.toml:6`,
`Helper_Scripts/sync_chunking_engine.py:12`,
`Tests/Chunking/test_sync_script.py:9`,
`Character_Chat/visual_identity.py:210`,
`assets/characters/samira/visual_identity_pack.json:751,758`,
`Tests/Character_Chat/test_visual_identity_contract.py:328`
(plus 13 further copies across `backlog/` and `Docs/superpowers/`).
A pin bump does fail loudly, which is good.

**B — `Third_Party/` is vendored with zero provenance machinery.**
`tldw_chatbook/Third_Party/` — 25 source files: `aider/` (6) and
`textual_fspicker/` (18) plus an `__init__.py`. **No pin, no manifest, no drift
test.** aider's only provenance line is wrong twice over —
`Third_Party/aider/repomap.py:1`:

```
# repomap.py - taken from https://github.com/Aider-AI/aider/blob/main/aider/waiting.py
```

…naming the **wrong upstream file** (`waiting.py` for `repomap.py`) and
`blob/main` rather than a SHA. The other four aider files have **zero**
provenance. textual_fspicker carries the upstream's own `pyproject.toml`
(`version = "0.4.1"`) — an incidental artifact of the copy, not a maintained
pin, and no SHA; `base_dialog.py:3` acknowledges it is a "vendored fork" that
"requires a small patch when syncing" but names no source revision.
Licences do ship (`aider/LICENSE.txt`, `textual_fspicker/LICENSE`).

**C — repo hygiene.** Measured now: `.git` **1.26 GiB** (`rr-cache` 475 MiB —
the largest reclaimable item; `objects` 637 MiB); `.worktrees` **46.54 GiB**;
**158** registered worktrees, **0 prunable**; **74** have a HEAD already an
ancestor of `origin/dev`, totalling **23.2 GiB**, of which **12 have
uncommitted changes**. Auto-gc is **suppressed** by an 87-byte `.git/gc.log`
dated Aug 21: *"There are too many unreachable loose objects; run 'git prune'
to remove them."*

> ⚠️ **Do NOT follow `gc.log`'s advice literally. Do NOT run bare `git prune`.**
> It expires all unreachable objects immediately, and with 158 live worktrees
> that risk is not worth taking. The 12 worktrees with uncommitted changes must
> be **inspected individually**, never bulk-removed — earlier incidents in this
> repo destroyed reviewed work exactly this way. Remediation here is
> non-destructive and opt-in only.

## Acceptance Criteria

- [x] `Tests/Chunking/test_sync_script.py` **skips** when its source clone is
      absent; it never initiates a network clone from inside a test
- [x] Any temporary clone the sync script creates is removed on both success
      and failure paths
- [ ] The test cannot leave a corrupted vendored tree if it dies mid-run
      (partially addressed — see Implementation Notes for the residual risk)
- [ ] The 3.9 GiB of leaked clones under `TMPDIR` are removed **after** being
      confirmed as leaked temp clones (inspect, then delete — do not glob-delete)
      (inspected and reported, NOT deleted — see Implementation Notes)
- [x] The default source path moves off `/private/tmp`, per the standing rule
- [x] The six copies of the vendor pin are reduced to one source of truth, or a
      test cross-checks them; a documented upgrade path exists for bumping it
- [ ] `Third_Party/` gains provenance: upstream URL and a **commit SHA** per
      vendored tree, and a drift test that fails when a vendored file diverges
      from its recorded source
- [ ] The wrong provenance comment at `Third_Party/aider/repomap.py:1` is
      corrected
- [ ] Repo hygiene is reduced by **non-destructive** means only, with each
      command recorded: expire `rr-cache` by age, remove **only** the worktrees
      confirmed merged **and** clean, and clear `gc.log` so normal auto-gc can
      resume
- [ ] The 12 dirty-but-merged worktrees are individually inspected and their
      owners consulted before anything is removed
- [ ] Before/after sizes are recorded

## Implementation Plan

This pass scopes to the description's **A** section (the leaking test/clone)
plus the six-copy vendor-pin duplication called out at the end of A. The
Third_Party/ provenance work (description's B) and repo-hygiene work
(description's C — `rr-cache`, worktree pruning, `gc.log`) are explicitly
out of scope for this pass; task stays **In Progress** rather than Done.

1. Reproduce born-red: with `subprocess.run` monkeypatched (no real clone),
   confirm the CURRENT `test_sync_script.py::_run_sync()` builds a command
   with no `--source` when the source path is absent — i.e. it would fall
   through to the script's no-arg network-clone path.
2. Fix `Tests/Chunking/test_sync_script.py`: remove the `/tmp/tldw_server_sync`
   default (moves the default off `/private/tmp` entirely — no default at
   all), and make `test_sync_idempotent_and_rejects_local_edits` `pytest.skip`
   explicitly when `TLDW_SERVER_SYNC_SOURCE` is unset or the path doesn't
   exist, instead of silently falling through.
3. Fix `Helper_Scripts/sync_chunking_engine.py`: wrap the no-arg temp-clone
   path in `try`/`finally` so `shutil.rmtree` always removes it (success,
   loud `sys.exit`, or an uncaught exception). Split the vendored-file and
   ported-test write loops into validate-then-write passes (mirroring the
   existing step-1 local-modification check) so a FATAL patch-anchor failure
   can't happen mid-write.
4. Prove cleanup on both the success path and a forced mid-sync failure path,
   with everything network-related mocked and all writes redirected to an
   isolated sandbox (never touching the real vendored tree).
5. Run the real end-to-end sync against a local `tldw_server` worktree at the
   pin (no network — the commit is already present in a local `tldw_server2`
   checkout) to confirm the refactor doesn't change output.
6. Investigate the "six authoritative pin copies" claim: confirm they are
   actually TWO independent pins (Chunking vendoring vs. Samira visual-identity
   compatibility) that coincidentally share one value today. Add a guard test
   cross-checking each cluster against its own designated source of truth,
   plus a documented upgrade path for bumping either one.
7. Bite-proof the guard: mutate one copy in each cluster, confirm the
   corresponding (and only that) test reds, Edit-restore, confirm clean diff.
8. Inspect and report leaked clones under `$TMPDIR` — do not delete.
9. Run `Tests/Chunking/`, the new guard test, and a repo-wide
   `--collect-only` sweep; hand-edit this task file (no `backlog` CLI on a
   five-digit id).

## Implementation Notes

**Scope.** This pass covers the leaking-test/clone problem and the
vendor-pin-duplication guard only. `Third_Party/` provenance and repo-hygiene
(rr-cache/worktree pruning/gc.log) are untouched — separate work, left To Do
on this same task; status stays In Progress rather than Done since not all
ACs are met.

**A — test skip + script cleanup.**
- `Tests/Chunking/test_sync_script.py`: `SOURCE` no longer defaults to
  `/tmp/tldw_server_sync` (a `/private/tmp` path — the standing rule is never
  to keep work there). It now reads only `TLDW_SERVER_SYNC_SOURCE`.
  `test_sync_idempotent_and_rejects_local_edits` (the only caller of
  `_run_sync()`) `pytest.skip`s with an actionable message when that path is
  unset or absent; `_run_sync()` itself now asserts the source exists rather
  than silently omitting `--source`, so this module can never trigger the
  no-arg network-clone path from any test.
  Born-red proof (against the unedited file, `subprocess.run` monkeypatched
  so nothing real ran): absent source → `_run_sync()` built
  `[python, sync_chunking_engine.py]` with no `--source` — confirmed the
  vulnerable fallthrough existed, including on THIS machine right now (the
  local reference worktree the review's snapshot referenced,
  `/private/tmp/tldw_server_sync`, is gone as of this session — probably
  reaped by the same tmp cleaner the standing rule warns about).
- `Helper_Scripts/sync_chunking_engine.py`: chose to make the script clean up
  after itself (a `finally: shutil.rmtree(tmp, ignore_errors=True)` around the
  whole sync flow) rather than relying solely on the test-level fix, since the
  script is invoked outside tests too and every other caller was equally
  exposed to the leak. `SystemExit` (the script's own `sys.exit(...)` FATAL
  path) still runs `finally`, so both the happy path and a loud failure clean
  up. Also split the two write loops (vendored-file copy, step 2; ported-test
  copy, step 4) into validate-then-write passes, matching the pattern the
  existing step 1 (local-modification check) already used for the vendored
  files: every output is computed first (a missing patch anchor can still
  FATAL here, e.g. on a first-time sync of a newly-vendored file with no
  local copy for step 1 to diff against), and only written once the full set
  succeeds. This closes the most likely real trigger for "crash mid-test
  corrupts the tree" (a `_fail()`/`sys.exit` firing partway through a write
  loop) but does **not** make the write itself atomic — a true `kill -9` or
  power loss mid-write-loop can still leave a partial tree; a full
  transactional swap (write to staging dir, then rename into place) would be
  needed to close that residual gap and was judged out of proportion for this
  pass, hence that AC is left unticked with this caveat rather than claimed.
  Cleanup proof: a sandboxed harness (writes redirected via `TARGET_ROOT`/
  `TARGET_TESTS_ROOT` monkeypatch, `subprocess.run` replaced with an
  in-process fake — no real clone/network at any point) exercised both the
  success path and a forced mid-sync FATAL (missing upstream file), and
  confirmed no `tldw_server_sync_*` temp dir survived either scenario.
  A real end-to-end run against a local `tldw_server` worktree at the pin
  (created from the pre-existing local `~/Documents/GitHub/tldw_server2`
  checkout, which already had the pin commit — no network touched) confirmed
  the refactor is byte-identical: `Tests/Chunking/test_sync_script.py` 10/10
  passed, `git status` clean afterward (idempotent re-sync).
- **Leaked clones found, NOT deleted (owner's call):** 17 directories under
  `$TMPDIR` (`/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/`), each 1.0
  GiB, **17 GiB total** — worse than the review's 4/3.9 GiB snapshot, and all
  dated *today* (18:47–21:06), confirming other concurrent sessions on this
  machine kept hitting the bug throughout the day:
  `tldw_server_sync_{te2lh67f,0bcgwtmx,lj_0wrm8,9uskt8c1,5ty_150w,hwyawdaj,
  mp3v13pz,hddj3sqk,uyp403d9,34uk7snw,p6g2i9cn,h3wg9z7c,vsuoltrq,8pi4b5ux,
  vd4394v7,v9xrvg2o,eel_6wvp}`.

**B (pin duplication) — cross-check guard.** Investigated the "six
authoritative places" claim: they are actually **two independent pins**
that happen to share one commit SHA today — the Chunking-engine vendoring
pin (code copied from `app/core/Chunking`) and the Samira visual-identity
compatibility pin (`app/core/Visual_Identities/expression_slots.py`
normalization contract + bundled asset pack). Nothing requires them to move
together, so the new guard (`Tests/Architecture/test_vendor_pin_consistency.py`)
does **not** assert cross-cluster equality — only that every hand-maintained
copy in each cluster agrees with that cluster's own source of truth:
`Helper_Scripts/sync_chunking_engine.py`'s `PIN` for Chunking (checked
against `VENDOR_MANIFEST.toml` and `test_sync_script.py`'s own `PIN`), and
`Character_Chat/visual_identity.py`'s `SAMIRA_SERVER_COMMIT` for Samira
(checked against its own docstring comment, `visual_identity_pack.json`'s
two fields, and `test_visual_identity_contract.py`'s hardcoded literal). The
module docstring documents the upgrade path for bumping either pin. The
further ~19 copies scattered across `backlog/` decision records and
`Docs/superpowers/` planning docs are deliberately excluded — they are
point-in-time historical snapshots, not live configuration; rewriting them on
every future pin bump would misrepresent history.
Bite-proof: mutated `VENDOR_MANIFEST.toml`'s `commit` (Chunking cluster) —
exactly `test_vendor_manifest_matches_sync_script_pin` reddened, the other 6
guard tests stayed green; Edit-restored, `git diff` clean. Repeated for
`visual_identity_pack.json`'s `source_server_commit` (Samira cluster) —
exactly `test_samira_pack_matches_visual_identity_pin` reddened; Edit-restored,
`git diff` clean.

**Verification.** `Tests/Chunking/` (no `TLDW_SERVER_SYNC_SOURCE`): 600
passed, 26 skipped (incl. the new explicit skip), 1 xfailed, **2 pre-existing
failures unrelated to this change** (`test_process_text_tokenizer_override`,
`test_golden_parity[tokens-cjk]` — both fail on this machine's stale/missing
real gpt2 tokenizer cache reaching out to huggingface.co; `git diff origin/dev`
on the touched files is empty, confirming they're environmental, not caused
by this branch). `Tests/Architecture/test_vendor_pin_consistency.py`: 7/7
passed. Repo-wide `--collect-only -q`: 56862 tests collected, 1 collection
error — `Tests/UI/test_library_file_notes_workspace.py` (TASK-20972, a
pre-existing dev red, not from this branch).

**Files changed:**
- `Tests/Chunking/test_sync_script.py`
- `Helper_Scripts/sync_chunking_engine.py`
- `Tests/Architecture/test_vendor_pin_consistency.py` (new)

**Follow-up.** `Third_Party/` provenance (upstream URL + commit SHA + drift
test for `aider`/`textual_fspicker`, and the wrong `repomap.py:1` comment) and
repo hygiene (rr-cache expiry, merged-worktree pruning, clearing `gc.log`)
remain — same task, To Do. The 17 GiB of leaked clones listed above are
reported, not deleted, per instruction (owner's call).
