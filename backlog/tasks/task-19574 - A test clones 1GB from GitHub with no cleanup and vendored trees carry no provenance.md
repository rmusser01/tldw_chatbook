---
id: TASK-19574
title: >-
  A test clones 1 GiB from GitHub with no cleanup, vendored trees carry no
  provenance, and the repo is 47 GiB with gc suppressed
status: To Do
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

- [ ] `Tests/Chunking/test_sync_script.py` **skips** when its source clone is
      absent; it never initiates a network clone from inside a test
- [ ] Any temporary clone the sync script creates is removed on both success
      and failure paths
- [ ] The test cannot leave a corrupted vendored tree if it dies mid-run
- [ ] The 3.9 GiB of leaked clones under `TMPDIR` are removed **after** being
      confirmed as leaked temp clones (inspect, then delete — do not glob-delete)
- [ ] The default source path moves off `/private/tmp`, per the standing rule
- [ ] The six copies of the vendor pin are reduced to one source of truth, or a
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
