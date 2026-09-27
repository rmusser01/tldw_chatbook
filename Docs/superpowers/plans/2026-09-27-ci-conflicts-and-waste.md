# CI Conflicts and Waste Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop artificial merge conflicts and wasted runner work in `tldw_chatbook` CI, and fix the
fast-lane `text-area--gutter` flake at its root, following the approved spec.

**Architecture:** The work ships as four sequential PRs to `dev`, in the spec's rollout order
(step 1, PR #2848, is the spec itself):

- **PR-A** removes the committed `summary` totals from the diagnostic inventory. Totals become
  derived by a new `inventory_summary()`.
- **PR-C** deletes two duplicated guard workflows, narrows the GGUF evidence triggers, adds the
  GGUF UI tests to the UI census, and amends ADR-103.
- **PR-D** adds a detach-safe `TextArea` subclass, used by every MCP-module TextArea.
- **PR-B** changes the `CLAUDE.md` User Guide stamp rule.

Each PR merges before the next is branched.

**Tech Stack:** Python ≥3.12, Textual 8.2.8, pytest (+pytest-asyncio, pytest-timeout), PyYAML,
GitHub Actions YAML, `gh` CLI.

**Spec:** `Docs/superpowers/specs/2026-09-27-ci-conflicts-and-waste-design.md` (revision 2).

## Global Constraints

- **Python:** only `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`, called
  `$PY` below. System `python3` is 3.9 and fails.
- **Worktrees:** each PR gets its own worktree off the **latest** `origin/dev`. Prefix every Bash
  call with `cd <worktree> &&`, because cwd resets between calls.
- **Git and tooling:**
  - Never `git stash`: `refs/stash` is repo-global.
  - Never run the app: it regenerates `css/tldw_cli_modular.tcss`.
  - `timeout(1)` does not exist.
  - Never verify with `pytest -k`.
- **Exit codes:** `cmd | tail` reports tail's exit code. Use `out=$(cmd 2>&1); rc=$?`.
- **zsh:** it does not word-split `$VAR`. Use `${=VAR}` or a bash script for file lists.
- **Required check:** its name stays exactly `Derived artifacts reproduce from their sources`.
  Its `needs` stay `[pr-fast-lane, ui-fast-lane]`. The required workflow is never path-filtered.
- **No selection-suppressing flags** (`--deselect`, `-k`, `--ignore`) are added to the fast lane.
  `Tests/CI/test_ci_queue_pressure_contract.py` forbids them.
- **The summarization review fixture is not regenerated.** Do not change the SHAs in
  `Tests/fixtures/summarization_diagnostic_review.json`. It is a review record; regenerating it
  would certify the whole current inventory as reviewed (lesson TASK-14651).
- **Baseline red sets on `origin/dev` (2026-09-27).** A task must leave these identical, compared
  by node-id set and not by count. The set includes `ERROR` lines as well as `FAILED` ones, and
  the pytest summary line must show no `error`: a collection or setup error can otherwise hide
  behind an unchanged `FAILED` set (Qodo on #2848):
  - `Tests/LLM_Calls/test_summarization_diagnostic_privacy.py`: 3 failures —
    `test_manifest_boundary_changes_only_summarization_owner_diagnostics`,
    `test_manifest_boundary_rejects_owned_digest_schema_changes`,
    `test_manifest_boundary_rejects_unreconciled_owned_digest`.
  - `Tests/Architecture/test_persistent_diagnostic_inventory.py`: 2 failures —
    `test_reviewed_diagnostic_changes_are_metadata_only`,
    `test_task_15743_final_rebase_diagnostics_are_metadata_only`.
  - `Tests/Architecture/test_derived_artifact_checkers.py` and
    `Tests/Architecture/test_diagnostic_path_privacy.py`: 0 failures.
- **Merging** (dev protection: strict, conversation resolution, required check). Re-sync with
  `gh pr update-branch <n>` when behind. Merge the moment the required check is green and every
  Qodo thread is resolved. Use `--auto` only after Qodo has posted on the current head and all
  threads are resolved.
- **Preflight:** `PYTHON=$PY ./scripts/preflight.sh` must be rc=0 before every push.
- **Commit trailer:** end every commit message with
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- **Backlog task:** one per PR, created with the `backlog` CLI at the start of the PR, with an id
  above the highest id on `origin/dev`
  (`git ls-tree -r --name-only origin/dev backlog | grep -oE 'task-[0-9]+' | sort -t- -k2 -n | tail -1`).
  Mark it Done with Implementation Notes when the PR merges.

## Review Focus

These are conditions no task's main test exercises by default. Each has a test in the task named.

1. **An open PR still carrying a schema-3 inventory (with `summary`) after PR-A lands.** The check
   must report `~ schema_version` and the `--write` instruction, not crash. *(Task 1,
   `test_old_schema_committed_file_reports_the_schema_change`.)*
2. **A committed inventory row missing `call_count`** (hand-edited or corrupted).
   `inventory_summary()` must count it as 0 rather than raise, so the drift report still renders.
   *(Task 1, `test_inventory_summary_tolerates_rows_missing_counts`.)*
3. **A detach-safe TextArea while attached.** It must render exactly what a stock `TextArea`
   renders; the guard is only for the detached state. *(Task 6,
   `test_attached_detach_safe_text_area_renders_like_stock`.)*
4. **Pushes to `dev`/`main` after the guards are deleted.** The bundle and backlog-id checks must
   still run on push events, not only on PRs. *(Task 3,
   `test_bundle_and_backlog_checks_run_on_push_events`.)*
5. **The GGUF evidence workflows after narrowing.** A PR touching only `app.py`, `css/**` or
   `Tests/conftest.py` must not trigger them. A PR touching `Model_Artifacts/**` must. *(Task 4,
   pinned by the path tuples plus the trigger table in the PR notes.)*

---

## PR-A (rollout step 2): stop committing the inventory `summary`

Worktree setup, done once for Tasks 1–2:

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook && git fetch -q origin && \
  git worktree add -b ci/inventory-derived-totals /private/tmp/ci-a origin/dev
```

### Task 1: Derive the inventory totals from rows (checker + every consumer)

**Files:**
- Modify: `scripts/check_persistent_diagnostic_inventory.py`:
  - `build_inventory` (the tail around lines 1081-1134);
  - `_summary_lines` (1241-1247);
  - `main` (lines 1773-1774 and 1788-1796).
- Test (modify): `Tests/Architecture/test_derived_artifact_checkers.py`:
  - the fixture `_inventory` (lines 22-60);
  - `test_added_diagnostic_names_the_file_and_the_delta` (line 77);
  - two new tests.
- Test (modify): `Tests/Architecture/test_diagnostic_path_privacy.py`:
  - `_inventory_with_path_candidates` (995-1008) and its call sites (around 1465-1515).
- Test (modify): `Tests/Architecture/test_persistent_diagnostic_inventory.py`: lines 1147-1149
  and 3455-3462.
- Test (modify): `Tests/LLM_Calls/test_summarization_diagnostic_privacy.py`:
  - `_normalized_inventory_projection` (1067-1081);
  - `_assert_task_492_summary` (1084-1103) and its call (2291);
  - `test_manifest_boundary_rejects_new_generated_origin_dev_drift`;
  - `test_manifest_boundary_rejects_forged_task_492_summary`.

**Interfaces:**
- Produces: `inventory_summary(inventory: dict[str, Any]) -> dict[str, int]` in
  `scripts/check_persistent_diagnostic_inventory.py`. It returns the keys `owner_files`,
  `task_492_calls`, `task_31551_calls`, `task_494_calls`, `persistent_sink_files` and
  `path_privacy_candidate_calls`.
- Produces: `build_inventory()` output with `schema_version == 4` and no `"summary"` key.

- [ ] **Step 1: Record the baseline red set for the four files**

```bash
cd /private/tmp/ci-a && PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python && \
  $PY -m pytest Tests/LLM_Calls/test_summarization_diagnostic_privacy.py \
    Tests/Architecture/test_persistent_diagnostic_inventory.py \
    Tests/Architecture/test_derived_artifact_checkers.py \
    Tests/Architecture/test_diagnostic_path_privacy.py \
    -q -p no:randomly -p no:cacheprovider --tb=no 2>&1 | grep -E '^(FAILED|ERROR)' | sed 's/ - .*//' | sort > /tmp/ci-a-baseline.txt; \
  cat /tmp/ci-a-baseline.txt
```

Expected: exactly the 5 node ids listed under Global Constraints. If the set differs, stop and
report: `dev` has moved.

- [ ] **Step 2: Write the failing tests**

Edit the fixture in `Tests/Architecture/test_derived_artifact_checkers.py`. Delete the whole
`"summary": {...}` entry from the `base` dict in `_inventory()` (lines 28-33 on `origin/dev`).
Leave everything else as it is.

In `test_added_diagnostic_names_the_file_and_the_delta`, delete the line
`rebuilt["summary"]["task_494_calls"] = 6`. Keep the assertion
`assert "task_494_calls: 4 -> 6" in report`. It now proves the total is derived from the edited
row.

Append these tests to the end of `Tests/Architecture/test_derived_artifact_checkers.py`:

```python
def test_inventory_summary_derives_every_total_from_rows():
    assert inventory.inventory_summary(_inventory()) == {
        "owner_files": 2,
        "task_492_calls": 3,
        "task_31551_calls": 0,
        "task_494_calls": 4,
        "persistent_sink_files": 1,
        "path_privacy_candidate_calls": 0,
    }


def test_inventory_summary_tolerates_rows_missing_counts():
    """A hand-edited or corrupted committed row must not crash the report."""
    broken = _inventory()
    del broken["owners"][1]["call_count"]

    assert inventory.inventory_summary(broken)["task_494_calls"] == 0


def test_old_schema_committed_file_reports_the_schema_change():
    """An open PR still carrying a schema-3 file (with stored totals) must get
    a reviewable drift report naming the schema change, not a crash."""
    committed = _inventory(schema_version=3, summary={"owner_files": 2})
    rebuilt = _inventory(schema_version=4)

    report = _diff(committed, rebuilt)

    assert "~ schema_version:" in report
    assert "--write" in report
```

Edit `Tests/Architecture/test_persistent_diagnostic_inventory.py` lines 1147-1149. Replace:

```python
    assert inventory["schema_version"] == 3
    assert inventory["path_privacy_rules"]["candidate_status"] == ("legacy_unreviewed")
    assert inventory["summary"]["path_privacy_candidate_calls"] == 1
```

with:

```python
    assert inventory["schema_version"] == 4
    assert "summary" not in inventory
    assert inventory["path_privacy_rules"]["candidate_status"] == ("legacy_unreviewed")
    assert (
        diagnostic_inventory.inventory_summary(inventory)["path_privacy_candidate_calls"]
        == 1
    )
```

In the same file at lines 3455-3462, replace `assert inventory["summary"] == {` with the
following, keeping the dict body and its closing `}` unchanged:

```python
    assert "summary" not in inventory
    assert diagnostic_inventory.inventory_summary(inventory) == {
```

- [ ] **Step 3: Run the new tests to verify they fail**

```bash
cd /private/tmp/ci-a && PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python && \
  $PY -m pytest Tests/Architecture/test_derived_artifact_checkers.py -q -p no:randomly -p no:cacheprovider --tb=line 2>&1 | tail -8
```

Expected: failures including `AttributeError: module ... has no attribute 'inventory_summary'`.
`test_added_diagnostic_names_the_file_and_the_delta` also fails, because the stored summary no
longer shows `4 -> 6`.

- [ ] **Step 4: Implement `inventory_summary` and drop the stored totals**

In `scripts/check_persistent_diagnostic_inventory.py`, add this function immediately above
`def _encoded(`:

```python
def inventory_summary(inventory: dict[str, Any]) -> dict[str, int]:
    """Derive the headline totals from an inventory's rows.

    Since schema 4 the committed file stores no totals: any two pull requests
    that each touch any diagnostic would otherwise both edit the same total
    lines and conflict on every re-sync (102 of 127 real two-sided sync merges
    did, 2026-09-19..26). The rows are the reviewed record; totals are
    arithmetic over them.

    Args:
        inventory: A parsed inventory, committed or rebuilt. A row missing a
            field counts as zero instead of raising, so a malformed committed
            file still produces a drift report.

    Returns:
        dict[str, int]: ``owner_files``, ``task_492_calls``,
            ``task_31551_calls``, ``task_494_calls``,
            ``persistent_sink_files`` and ``path_privacy_candidate_calls``.
    """
    owners = [row for row in inventory.get("owners", []) if isinstance(row, dict)]

    def calls(owner: str) -> int:
        return sum(
            row["call_count"]
            for row in owners
            if row.get("owner") == owner and isinstance(row.get("call_count"), int)
        )

    return {
        "owner_files": len(owners),
        "task_492_calls": calls("TASK-492"),
        "task_31551_calls": calls("TASK-31551"),
        "task_494_calls": calls("TASK-494"),
        "persistent_sink_files": len(inventory.get("persistent_sink_topology", [])),
        "path_privacy_candidate_calls": sum(
            len(row.get("candidates", []))
            for row in inventory.get("path_privacy_candidates", [])
            if isinstance(row, dict)
        ),
    }
```

In `build_inventory()`, make three edits:

- Delete the three assignments `task_492_calls = sum(...)`, `task_31551_calls = sum(...)` and
  `task_494_calls = sum(...)` just above `return {`.
- Delete the whole `"summary": {...},` entry from the returned dict.
- Replace the `schema_version` lines:

  ```python
          # 3: adds the unresolved path-privacy candidate projection. Existing
          # owner digests and sink identities retain their schema-v2 meaning.
          "schema_version": 3,
  ```

  with:

  ```python
          # 4: drops the stored `summary` totals; `inventory_summary()` derives
          # them from the rows (they caused a conflict on nearly every re-sync).
          # 3 added the unresolved path-privacy candidate projection; owner
          # digests and sink identities keep their schema-v2 meaning.
          "schema_version": 4,
  ```

Replace the first line of the `_summary_lines` body:

```python
    old, new = committed.get("summary", {}), rebuilt.get("summary", {})
```

with:

```python
    old, new = inventory_summary(committed), inventory_summary(rebuilt)
```

In `main()`, replace the missing-file message fragment:

```python
            f"{inventory['summary']['owner_files']} owner files and "
            f"{inventory['summary']['persistent_sink_files']} sink files.\n"
```

with:

```python
            f"{inventory_summary(inventory)['owner_files']} owner files and "
            f"{inventory_summary(inventory)['persistent_sink_files']} sink files.\n"
```

Replace `summary = inventory["summary"]` near the end of `main()` with
`summary = inventory_summary(inventory)`.

- [ ] **Step 5: Update the path-privacy fixture**

In `Tests/Architecture/test_diagnostic_path_privacy.py`, `_inventory_with_path_candidates`, make
two edits:

- Change the signature to `def _inventory_with_path_candidates(rows: list[dict[str, object]]) -> dict[str, object]:`.
- Delete the line `"summary": {"path_privacy_candidate_calls": candidate_count},`.

Then remove the `candidate_count=` argument from every call site:

```bash
cd /private/tmp/ci-a && grep -n "candidate_count" Tests/Architecture/test_diagnostic_path_privacy.py
```

For each hit, delete the `candidate_count=N` argument. When it is on its own line, delete the
line. When it is inline, as in `_inventory_with_path_candidates([], candidate_count=0)`, change
the call to `_inventory_with_path_candidates([])`. Re-run the grep: expected no output.

- [ ] **Step 6: Update the summarization review helpers (preserving the baseline red set)**

In `Tests/LLM_Calls/test_summarization_diagnostic_privacy.py`:

- In `_normalized_inventory_projection`, delete the two lines `summary = normalized["summary"]`
  and `assert isinstance(summary, dict)`, and the line
  `summary["task_492_calls"] = "<derived-task-492-call-count>"`.
- Delete the whole function `_assert_task_492_summary`. Delete its one call,
  `_assert_task_492_summary(inventory, inventory_name=name)`, inside
  `test_manifest_boundary_changes_only_summarization_owner_diagnostics`. Its invariant ("the
  stored TASK-492 total equals the rows") is now structural, because nothing is stored.
- Delete `test_manifest_boundary_rejects_forged_task_492_summary` entirely. There is no stored
  total left to forge.
- In `test_manifest_boundary_rejects_new_generated_origin_dev_drift`, replace
  `generated_inventory["summary"]["owner_files"] += 1` with:

```python
    generated_inventory["owners"].append(
        {
            "path": "tldw_chatbook/zz_unreviewed_drift.py",
            "owner": "TASK-494",
            "reason": "remaining Chatbook production diagnostic owner",
            "call_count": 1,
            "diagnostic_digest": "0" * 20,
        }
    )
```

- [ ] **Step 7: Run the four files and compare against the baseline**

```bash
cd /private/tmp/ci-a && PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python && \
  $PY -m pytest Tests/LLM_Calls/test_summarization_diagnostic_privacy.py \
    Tests/Architecture/test_persistent_diagnostic_inventory.py \
    Tests/Architecture/test_derived_artifact_checkers.py \
    Tests/Architecture/test_diagnostic_path_privacy.py \
    -q -p no:randomly -p no:cacheprovider --tb=short 2>&1 | grep -E '^(FAILED|ERROR)' | sed 's/ - .*//' | sort > /tmp/ci-a-after.txt; \
  echo "new failures:"; comm -13 /tmp/ci-a-baseline.txt /tmp/ci-a-after.txt; \
  echo "fixed:"; comm -23 /tmp/ci-a-baseline.txt /tmp/ci-a-after.txt
```

Expected: `new failures:` is empty and `fixed:` is empty (the baseline set is unchanged). The
three new checker tests pass.

- [ ] **Step 8: Commit**

```bash
cd /private/tmp/ci-a && git add scripts/check_persistent_diagnostic_inventory.py \
  Tests/Architecture/test_derived_artifact_checkers.py Tests/Architecture/test_diagnostic_path_privacy.py \
  Tests/Architecture/test_persistent_diagnostic_inventory.py Tests/LLM_Calls/test_summarization_diagnostic_privacy.py && \
  git commit -m "feat(inventory): derive diagnostic totals from rows; stop committing summary (schema 4)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 2: Regenerate the inventory, amend ADR-029, commit the conflict-rate measurement, open PR-A

**Files:**
- Modify (generated): `Docs/security/production-diagnostic-inventory.json`
- Modify: `backlog/decisions/029-local-private-data-boundary.md` (append an amendment)
- Create: `scripts/measure_sync_merge_conflicts.py`
- Create: the backlog task for PR-A

**Interfaces:**
- Consumes: `build_inventory()` (schema 4) and `inventory_summary()` from Task 1.
- Produces: `scripts/measure_sync_merge_conflicts.py [--merges N]`. It prints `sync merges: <n>`,
  `conflicted: <c> (<pct>%)` and a per-file conflict count. The 2-week review uses it.

- [ ] **Step 1: Regenerate and prove the diff is only the schema change**

```bash
cd /private/tmp/ci-a && PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python && \
  $PY scripts/check_persistent_diagnostic_inventory.py --write && \
  git diff --stat -- Docs/security/production-diagnostic-inventory.json && \
  git diff -U0 -- Docs/security/production-diagnostic-inventory.json | grep -E '^[-+] ' 
```

Expected: `schema_version` 3 → 4, and the whole `summary` block removed (9 lines: the key, six
totals, the closing brace, and one comma change on the preceding line). No `owners`,
`path_privacy_candidates` or `persistent_sink_topology` line changes. If any row changes, stop:
`dev` has un-regenerated drift, which must be reviewed per `NEXT_STEPS`, not blessed here.

- [ ] **Step 2: Round-trip check**

```bash
cd /private/tmp/ci-a && PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python && \
  out=$($PY scripts/check_persistent_diagnostic_inventory.py 2>&1); rc=$?; echo "rc=$rc"; echo "$out" | tail -2
```

Expected: `rc=0` and `diagnostic inventory verified: <N> owners, ... sink files`.

- [ ] **Step 3: Write the measurement script**

Create `scripts/measure_sync_merge_conflicts.py`:

```python
#!/usr/bin/env python3
"""Measure how often real dev->PR-branch sync merges conflict, and on which files.

Replays every sync merge (a merge commit inside a PR branch whose second parent
brought `dev` in) found in the last N first-parent merges of `origin/dev`, with
`git merge-tree`, and tallies conflicting files. This is the method behind the
2026-09-27 CI spec's baseline (547 syncs, 48% conflicted).

Caveat: `git merge-tree` honours the LOCAL `.gitattributes` merge drivers, while
GitHub's server-side merge does not, so a local `merge=` attribute would make the
rate look better than GitHub sees it.
"""

from __future__ import annotations

import argparse
import subprocess
from collections import Counter


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False)


def _sync_merges(first_parent_merges: int) -> list[str]:
    merges = _git(
        "log", "origin/dev", "--first-parent", "--merges", "--format=%H",
        "-n", str(first_parent_merges),
    ).stdout.split()
    syncs: set[str] = set()
    for merge in merges:
        syncs.update(
            _git("log", "--merges", "--format=%H", f"{merge}^1..{merge}^2").stdout.split()
        )
    return sorted(syncs)


def _conflicted_files(sync: str) -> list[str] | None:
    result = _git("merge-tree", "--write-tree", "--name-only", f"{sync}^1", f"{sync}^2")
    if result.returncode != 1:
        return None
    lines = result.stdout.splitlines()[1:]
    return [
        line for line in lines
        if line and not line.startswith(("Auto-merging", "CONFLICT"))
    ]


def main() -> int:
    """Print the sync-merge conflict rate and the files most often in conflict."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--merges", type=int, default=300,
                        help="first-parent merges of origin/dev to scan (default 300)")
    args = parser.parse_args()
    syncs = _sync_merges(args.merges)
    per_file: Counter[str] = Counter()
    conflicted = 0
    for sync in syncs:
        files = _conflicted_files(sync)
        if files is None:
            continue
        conflicted += 1
        per_file.update(set(files))
    pct = (100 * conflicted // len(syncs)) if syncs else 0
    print(f"sync merges: {len(syncs)}")
    print(f"conflicted: {conflicted} ({pct}%)")
    for path, count in per_file.most_common(15):
        print(f"  {count:5d}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run it (the baseline under the old format)**

```bash
cd /private/tmp/ci-a && git fetch -q origin && PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python && \
  $PY scripts/measure_sync_merge_conflicts.py --merges 300 | head -8
```

Expected: roughly `sync merges: ~550`, `conflicted: ~48%`, with the inventory at the top. Record
the output in the task notes as the pre-change baseline. The post-change number is only
measurable once new syncs exist after merge; that is the 2-week review.

- [ ] **Step 5: Append the ADR-029 amendment**

Append to `backlog/decisions/029-local-private-data-boundary.md`:

```markdown

## Amendment (2026-09-27): the diagnostic inventory stores rows only

`Docs/security/production-diagnostic-inventory.json` (schema 4) no longer stores the
`summary` totals. `scripts/check_persistent_diagnostic_inventory.py` derives them from
the rows (`inventory_summary()`) for its reports.

The review guarantee is unchanged. Every added, removed, reworded or re-levelled
diagnostic is still a per-file row that changes in the PR diff, and the required check
still fails on any drift.

Why: the six totals changed with any logger edit, so any two PRs touching diagnostics
edited the same lines. 102 of 127 real two-sided sync merges (2026-09-19..26)
conflicted. Replayed without the totals, 17 did, and those 17 are genuine same-file
overlaps. See `Docs/superpowers/specs/2026-09-27-ci-conflicts-and-waste-design.md`.
```

- [ ] **Step 6: Preflight, then commit**

```bash
cd /private/tmp/ci-a && PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python && \
  out=$(PYTHON=$PY ./scripts/preflight.sh 2>&1); rc=$?; echo "PREFLIGHT_RC=$rc"; [ $rc -ne 0 ] && echo "$out" | tail -20
```

Expected: `PREFLIGHT_RC=0`. Then:

```bash
cd /private/tmp/ci-a && git add Docs/security/production-diagnostic-inventory.json \
  backlog/decisions/029-local-private-data-boundary.md scripts/measure_sync_merge_conflicts.py && \
  git commit -m "chore(inventory): regenerate as schema 4; ADR-029 amendment; sync-merge conflict meter

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 7: Create the backlog task, push, and open PR-A**

```bash
cd /private/tmp/ci-a && backlog task create "CI: derive diagnostic-inventory totals from rows (schema 4)" \
  -d "Stop committing the inventory summary totals that conflicted on 102 of 127 two-sided sync merges; spec 2026-09-27-ci-conflicts-and-waste-design.md part A." \
  --ac "build_inventory emits schema 4 with no summary; totals derived by inventory_summary()" \
  --ac "All five consumer test files keep the dev baseline red set exactly" \
  --ac "Committed inventory regenerated; only schema_version and summary lines changed" \
  --ac "ADR-029 amendment recorded" --plain
```

Then add and commit the task file. Push with `git push -u origin HEAD`. Open the PR:
`gh pr create --base dev --title "CI: derive diagnostic-inventory totals from rows (schema 4)"`.
The body covers:

- **Evidence:** the 102 → 17 replay.
- **Baseline red set:** unchanged, with the list.
- **Inert negative control, not fixed here:** the mutant tests in
  `test_summarization_diagnostic_privacy.py` pass only because their base test is already red on
  the stale review hash.
- **Heads-up for open PRs:** the 7 open PRs that touch the inventory need one `--write` after
  this lands.

Merge it under the Global Constraints merge rule. Then mark the backlog task Done with
Implementation Notes.

---

## PR-C (rollout step 3): delete duplicated guards, narrow GGUF evidence, amend ADR-103

Worktree, after PR-A is merged:

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook && git fetch -q origin && \
  git worktree add -b ci/guards-and-gguf /private/tmp/ci-c origin/dev
```

### Task 3: Delete the two duplicated guard workflows

**Files:**
- Delete: `.github/workflows/css-bundle-guard.yml` and `.github/workflows/backlog-guard.yml`
- Modify: `Tests/CI/test_ci_queue_pressure_contract.py`:
  - `STANDALONE_WORKFLOWS` (lines 47-52);
  - a new test.
- Modify: `Tests/CI/test_derived_artifacts_workflow.py`: delete
  `test_backlog_guard_delegates_to_the_shared_script` (lines 219-226).
- Modify: `Tests/Packaging/test_python_runtime_floor.py:94-95`
- Modify (comments/docs only):
  - `Tests/CI/test_backlog_task_id_uniqueness.py:8`;
  - `Tests/README.md:363-368`;
  - `.github/workflows/test.yml` (comments near 31, 56, 69);
  - `.github/workflows/derived-artifacts.yml` (comments near 26, 128, 255);
  - `.github/workflows/perf-guard.yml` (comments near 9, 43);
  - `scripts/check_backlog_task_ids.py` (near 20, 209);
  - `scripts/check_bundle_sync.py` (near 15).

**Interfaces:**
- Produces: `STANDALONE_WORKFLOWS == ("derived-artifacts.yml", "perf-guard.yml")`.

- [ ] **Step 1: Write the failing test (push events still run both checks)**

Append to `Tests/CI/test_ci_queue_pressure_contract.py`:

```python
def test_bundle_and_backlog_checks_run_on_push_events() -> None:
    """With css-bundle-guard/backlog-guard deleted, the required workflow is the
    only place these checks run -- so they must run on dev/main pushes too, not
    only inside the pull-request-only fast lanes."""
    workflow = _workflow("derived-artifacts.yml")
    assert {"dev", "main"} <= set(_triggers(workflow)["push"]["branches"])
    steps = workflow["jobs"]["derived-artifacts"]["steps"]
    for script in ("scripts/check_bundle_sync.py", "scripts/check_backlog_task_ids.py"):
        matching = [step for step in steps if script in str(step.get("run", ""))]
        assert matching, f"{script} is not run by the required job"
        for step in matching:
            assert "pull_request" not in str(step.get("if", "")), (
                f"{script} must not be pull-request-only"
            )
    assert not (PROJECT_ROOT / ".github/workflows/css-bundle-guard.yml").exists()
    assert not (PROJECT_ROOT / ".github/workflows/backlog-guard.yml").exists()
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cd /private/tmp/ci-c && PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python && \
  $PY -m pytest "Tests/CI/test_ci_queue_pressure_contract.py::test_bundle_and_backlog_checks_run_on_push_events" -q -p no:cacheprovider 2>&1 | tail -3
```

Expected: FAIL on `assert not (... css-bundle-guard.yml).exists()`. The earlier assertions pass,
which proves the checks already run on push.

- [ ] **Step 3: Delete the workflows and update the pins**

```bash
cd /private/tmp/ci-c && git rm -q .github/workflows/css-bundle-guard.yml .github/workflows/backlog-guard.yml
```

In `Tests/CI/test_ci_queue_pressure_contract.py`, replace the `STANDALONE_WORKFLOWS` tuple with:

```python
STANDALONE_WORKFLOWS = (
    "derived-artifacts.yml",
    "perf-guard.yml",
)
```

In `Tests/CI/test_derived_artifacts_workflow.py`, delete the whole function
`test_backlog_guard_delegates_to_the_shared_script`.

In `Tests/Packaging/test_python_runtime_floor.py`, replace:

```python
    css_guard = _text(".github/workflows/css-bundle-guard.yml")
    assert "python-version: '3.12'" in _job_block(css_guard, "css-bundle-reproducible")
```

with:

```python
    derived = _text(".github/workflows/derived-artifacts.yml")
    assert "python-version: '3.12'" in _job_block(derived, "pr-fast-lane")
```

- [ ] **Step 4: Update comments and docs that name the deleted guards**

```bash
cd /private/tmp/ci-c && git grep -n -E "css-bundle-guard|backlog-guard|CSS Bundle Guard|Backlog Guard" -- \
  ':!backlog/**' ':!Docs/Development/**' ':!Docs/superpowers/**' ':!.superpowers/**'
```

At each hit, rewrite the sentence so it says the check runs in "the Derived Artifacts required
job (`derived-artifacts.yml`)". Where a comment explains that a workflow is standalone "like
css-bundle-guard/backlog-guard", keep the reason and drop the comparison.

Re-run the grep. Expected: no output, apart from the deliberate historical reference that
`Tests/CI/test_backlog_task_id_uniqueness.py:8` keeps, reworded as "formerly also enforced by
backlog-guard.yml (deleted 2026-09-27)".

- [ ] **Step 5: Run `Tests/CI` and the runtime-floor test**

```bash
cd /private/tmp/ci-c && PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python && \
  out=$($PY -m pytest Tests/CI Tests/Packaging/test_python_runtime_floor.py -q -p no:randomly -p no:cacheprovider --tb=short 2>&1); \
  echo "rc=$?"; echo "$out" | grep -E '^(FAILED|ERROR)|[0-9]+ (passed|failed|error)' | tail -8
```

Expected: rc=0 and 0 failed. If a failure also fails on `origin/dev`, record it as baseline and
compare node-id sets as in Task 1 Step 7.

- [ ] **Step 6: Commit**

```bash
cd /private/tmp/ci-c && git add -A .github/workflows Tests/CI Tests/Packaging/test_python_runtime_floor.py \
  Tests/README.md scripts/check_backlog_task_ids.py scripts/check_bundle_sync.py && \
  git commit -m "ci: delete css-bundle-guard and backlog-guard (duplicated by the required job)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 4: Narrow the GGUF evidence triggers and add their UI tests to the census

**Files:**
- Modify: `.github/workflows/task-2062-1-gguf-import-evidence.yml` (`on.pull_request.paths`)
- Modify: `.github/workflows/task-2062-2-gguf-source-evidence.yml` (`on.pull_request.paths`)
- Modify: `Tests/CI/test_task2062_1_gguf_import_evidence.py` (`EXPECTED_PULL_REQUEST_PATHS`)
- Modify: `Tests/CI/test_task2062_2_gguf_source_evidence.py` (`EXPECTED_PULL_REQUEST_PATHS`)
- Modify: `scripts/ui_pr_gate_census.txt` (add 2 lines, conditionally)

- [ ] **Step 1: Pin the new paths in the tests first (they fail against the current workflows)**

In `Tests/CI/test_task2062_1_gguf_import_evidence.py`, replace `EXPECTED_PULL_REQUEST_PATHS` with:

```python
EXPECTED_PULL_REQUEST_PATHS = (
    ".github/workflows/task-2062-1-gguf-import-evidence.yml",
    "tldw_chatbook/Model_Artifacts/**",
    "tldw_chatbook/UI/Screens/model_installed_view.py",
    "Tests/Model_Artifacts/**",
    "Tests/UI/test_model_installed_view.py",
)
```

In `Tests/CI/test_task2062_2_gguf_source_evidence.py`, replace `EXPECTED_PULL_REQUEST_PATHS` with:

```python
EXPECTED_PULL_REQUEST_PATHS = (
    ".github/workflows/task-2062-2-gguf-source-evidence.yml",
    "tldw_chatbook/Event_Handlers/LLM_Management_Events/**",
    "tldw_chatbook/Model_Artifacts/**",
    "tldw_chatbook/UI/LLM_Management_Window.py",
    "tldw_chatbook/UI/Screens/llm_screen.py",
    "Tests/LLM_Management/**",
    "Tests/Model_Artifacts/**",
    "Tests/UI/test_llm_gguf_source_modes.py",
)
```

```bash
cd /private/tmp/ci-c && PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python && \
  $PY -m pytest Tests/CI/test_task2062_1_gguf_import_evidence.py Tests/CI/test_task2062_2_gguf_source_evidence.py -q -p no:cacheprovider --tb=line 2>&1 | tail -4
```

Expected: FAIL on the paths assertion in each file.

- [ ] **Step 2: Narrow the workflow `paths` to match**

In each workflow, replace the `on.pull_request.paths:` list with exactly the entries of the
matching tuple above, in the same order, one `- '<path>'` per line. Leave `branches: [dev]` and
`workflow_dispatch:` unchanged. Re-run the Step 1 command. Expected: PASS.

- [ ] **Step 3: Check whether the two GGUF UI test files are green on the fast lane's minimal dependency set**

```bash
cd /private/tmp/ci-c && uv venv -q -p 3.12 /private/tmp/ci-c-minvenv && \
  VIRTUAL_ENV=/private/tmp/ci-c-minvenv uv pip install -q -e . pytest pytest-asyncio pytest-timeout packaging && \
  out=$(/private/tmp/ci-c-minvenv/bin/python -m pytest Tests/UI/test_model_installed_view.py Tests/UI/test_llm_gguf_source_modes.py \
    --timeout=180 -q -p no:randomly -p no:cacheprovider --tb=short 2>&1); echo "rc=$?"; echo "$out" | grep -E '^(FAILED|ERROR)|[0-9]+ (passed|failed|error)' | tail -6
```

Expected: rc=0. If a file is not fully green, do **not** add it. Record its failing node ids in
the task notes, add only the green file, and continue.

- [ ] **Step 4: Add the green file(s) to the census**

Append each green file's path, one per line, to the end of `scripts/ui_pr_gate_census.txt`:

```
Tests/UI/test_model_installed_view.py
Tests/UI/test_llm_gguf_source_modes.py
```

```bash
cd /private/tmp/ci-c && PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python && \
  $PY scripts/check_ui_pr_gate_census.py; echo "rc=$?"
```

Expected: rc=0.

- [ ] **Step 5: Trigger table for the PR notes (Review Focus #5)**

Record this table in the PR body. It is derived from the new `paths` under GitHub's rule that
any changed file matching any pattern triggers the workflow:

| Changed files | 2062.1 import | 2062.2 source |
|---|---|---|
| `tldw_chatbook/app.py` only | no | no |
| `tldw_chatbook/css/x.tcss` only | no | no |
| `Tests/conftest.py` only | no | no |
| `pyproject.toml` only | no | no |
| `tldw_chatbook/Model_Artifacts/x.py` | yes | yes |
| `tldw_chatbook/UI/Screens/llm_screen.py` | no | yes |

- [ ] **Step 6: Commit**

```bash
cd /private/tmp/ci-c && git add .github/workflows/task-2062-1-gguf-import-evidence.yml \
  .github/workflows/task-2062-2-gguf-source-evidence.yml Tests/CI/test_task2062_1_gguf_import_evidence.py \
  Tests/CI/test_task2062_2_gguf_source_evidence.py scripts/ui_pr_gate_census.txt && \
  git commit -m "ci: narrow GGUF evidence triggers to GGUF code; gate its UI tests in the census

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 5: ADR-103 amendment and open PR-C

**Files:**
- Modify: `backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md`: append an
  amendment, and edit the Consequences bullet at lines 104-108.
- Create: the backlog task for PR-C

- [ ] **Step 1: Update the Consequences runner-count bullet**

Replace the sentence fragment at lines 104-107 of ADR-103 that reads "... three routine
path-scoped guards, an ordinary unlabeled, non-GGUF PR has a peak of at most four runners ..."
with the same sentence reading "... one routine path-scoped guard (perf-guard), an ordinary
unlabeled, non-GGUF PR has a peak of at most two runners ...". Keep the rest of the bullet,
including the GGUF caveat that follows it.

- [ ] **Step 2: Append the amendment**

```markdown

## Amendment (2026-09-27): nightly disabled; duplicated guards removed

- **Full-tree cadence change.** `nightly-deep.yml` was disabled with
  `gh workflow disable` on 2026-09-27 (owner decision). It had produced 0 complete
  runs in 8 nights while using about 22% of the account's runner-minutes and 64% of
  its macOS minutes. Until CI throughput sub-project 3 restores it (`gh workflow
  enable`, once a run can finish and report), full-tree coverage comes only from
  `main` pushes and manual dispatch. The last `main` push was 2026-09-14. This
  records the loss rather than hiding it; the nightly had not produced a complete
  verdict before the change either.
- **Guards.** `css-bundle-guard.yml` and `backlog-guard.yml` were deleted. Their
  checks already ran inside the required job on every pull request and on every
  `dev`/`main` push. The Consequences runner count above is updated to match.
- **Strict.** Restored on 2026-09-27 (see the note under the 2026-08-30 amendment).
- Spec: `Docs/superpowers/specs/2026-09-27-ci-conflicts-and-waste-design.md`.
```

- [ ] **Step 3: Preflight, backlog task, push, PR-C**

```bash
cd /private/tmp/ci-c && PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python && \
  out=$(PYTHON=$PY ./scripts/preflight.sh 2>&1); rc=$?; echo "PREFLIGHT_RC=$rc"
```

Expected: `PREFLIGHT_RC=0`. Then:

1. Create the task:

   ```bash
   cd /private/tmp/ci-c && backlog task create "CI: delete duplicated guards, narrow GGUF evidence, ADR-103 amendment" \
     -d "Spec 2026-09-27-ci-conflicts-and-waste-design.md parts C1, C2 and E." \
     --ac "css-bundle-guard and backlog-guard deleted; bundle and backlog-id checks pinned to run on dev/main pushes" \
     --ac "GGUF evidence paths narrowed to GGUF code and pinned in Tests/CI" \
     --ac "GGUF UI test files added to the UI census only if green on the minimal dependency set" \
     --ac "ADR-103 amended for the nightly cadence change and the removed guards" --plain
   ```
2. Commit the ADR and the task.
3. `git push -u origin HEAD`.
4. `gh pr create --base dev ...`. The body includes the trigger table from Task 4 Step 5 and a
   note that open PR #2026 edits `backlog-guard.yml` and will need to drop that change.
5. Merge under the Global Constraints rule, then mark the task Done.

---

## PR-D (rollout step 4): fix the `text-area--gutter` flake at its root

**Mechanism (verified 2026-09-27 against Textual 8.2.8):**

1. When a widget detaches, `Widget._message_loop_exit` calls `self._detach()` and then
   `self._component_styles.clear()`.
2. If a screen repaint is already queued, the compositor calls `TextArea.render_lines`, which runs
   `theme.apply_css(self)`, which calls `get_component_styles("text-area--gutter")`, which raises
   `KeyError: "No 'text-area--gutter' key in COMPONENT_CLASSES"`.
3. `_detach()` runs first, so `is_attached` is already `False` whenever the styles are gone. A
   detached widget is never visible, so rendering blank lines is correct.

Worktree, after PR-C is merged:

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook && git fetch -q origin && \
  git worktree add -b fix/detach-safe-text-area /private/tmp/ci-d origin/dev
```

### Task 6: `DetachSafeTextArea`, used by every MCP-module TextArea

**Files:**
- Create: `tldw_chatbook/Widgets/detach_safe_text_area.py`
- Create: `Tests/Widgets/test_detach_safe_text_area.py`
- Modify (swap `TextArea(` for `DetachSafeTextArea(`):
  - `tldw_chatbook/UI/MCP_Modules/mcp_schema_form.py:244`;
  - `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py:1614`;
  - `tldw_chatbook/UI/MCP_Modules/mcp_profile_form.py:107`, `:132`, `:432`.
- Modify: TASK-32049, extending it to every test showing this error; implementation notes.

**Interfaces:**
- Produces: `class DetachSafeTextArea(TextArea)`. It has the same constructor as `TextArea` and
  overrides only `render_lines(self, crop: Region) -> list[Strip]`.
- Keeps working: `query_one("#mcp-schema-raw", TextArea)` and `@on(TextArea.Changed, ...)`,
  because the class subclasses `TextArea` and posts the same messages.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Widgets/test_detach_safe_text_area.py`:

```python
"""TASK-32049: a TextArea repainted after detaching must not raise.

Textual clears a widget's component styles when it detaches; a screen repaint
already queued can still reach `TextArea.render_lines`, whose theme step asks
for `text-area--gutter` and raises KeyError. That is the fast-lane flake seen
across `Tests/UI/test_mcp_workbench.py`, reproduced here deterministically.
"""

import pytest
from textual.app import App, ComposeResult
from textual.geometry import Region
from textual.widgets import TextArea

from tldw_chatbook.Widgets.detach_safe_text_area import DetachSafeTextArea


class _Host(App):
    def __init__(self, widget_type: type[TextArea]) -> None:
        super().__init__()
        self._widget_type = widget_type

    def compose(self) -> ComposeResult:
        yield self._widget_type('{"a": 1}', id="editor")


async def _render_after_removal(widget_type: type[TextArea]) -> list:
    app = _Host(widget_type)
    async with app.run_test(size=(60, 10)) as pilot:
        editor = app.query_one("#editor", TextArea)
        await pilot.pause()
        await editor.remove()
        return editor.render_lines(Region(0, 0, 20, 3))


@pytest.mark.asyncio
async def test_stock_text_area_raises_after_detach():
    """Negative control: proves the race is real on the pinned Textual."""
    with pytest.raises(KeyError, match="text-area--gutter"):
        await _render_after_removal(TextArea)


@pytest.mark.asyncio
async def test_detach_safe_text_area_renders_blank_after_detach():
    lines = await _render_after_removal(DetachSafeTextArea)

    assert len(lines) == 3
    assert all(line.cell_length == 20 for line in lines)
    assert all(not line.text.strip() for line in lines)


@pytest.mark.asyncio
async def test_attached_detach_safe_text_area_renders_like_stock():
    """The guard must not change rendering while attached."""
    rendered = {}
    for widget_type in (TextArea, DetachSafeTextArea):
        app = _Host(widget_type)
        async with app.run_test(size=(60, 10)) as pilot:
            await pilot.pause()
            editor = app.query_one("#editor", TextArea)
            rendered[widget_type] = [
                line.text for line in editor.render_lines(Region(0, 0, 20, 3))
            ]
    assert rendered[DetachSafeTextArea] == rendered[TextArea]
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /private/tmp/ci-d && PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python && \
  $PY -m pytest Tests/Widgets/test_detach_safe_text_area.py -q -p no:cacheprovider --tb=line 2>&1 | tail -4
```

Expected: collection error `ModuleNotFoundError: tldw_chatbook.Widgets.detach_safe_text_area`.

- [ ] **Step 3: Implement**

Create `tldw_chatbook/Widgets/detach_safe_text_area.py`:

```python
"""A TextArea that renders blank once detached (TASK-32049)."""

from __future__ import annotations

from textual.geometry import Region
from textual.strip import Strip
from textual.widgets import TextArea


class DetachSafeTextArea(TextArea):
    """TextArea whose queued repaint after removal renders blank instead of raising.

    Textual's ``Widget._message_loop_exit`` detaches the widget and then clears
    its component styles. A screen repaint already queued can still call
    ``render_lines``, where the theme step looks up ``text-area--gutter`` and
    raises ``KeyError``. ``_detach()`` runs first, so ``is_attached`` is already
    False whenever the styles are gone -- and a detached widget is never visible,
    so blank lines are the correct render.
    """

    def render_lines(self, crop: Region) -> list[Strip]:
        """Render normally while attached; blank lines once detached.

        Args:
            crop: Region of the widget to render.

        Returns:
            list[Strip]: One strip per row of ``crop``.
        """
        if not self.is_attached:
            return [Strip.blank(crop.width) for _ in range(crop.height)]
        return super().render_lines(crop)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run the Step 2 command. Expected: `3 passed`, including the stock-`TextArea` negative control.

- [ ] **Step 5: Use it in every MCP-module TextArea**

In each of the three files, add
`from tldw_chatbook.Widgets.detach_safe_text_area import DetachSafeTextArea` to the local-imports
group, and change the listed `TextArea(` construction calls to `DetachSafeTextArea(`. Do not
change `query_one(..., TextArea)` calls or `@on(TextArea.Changed, ...)` decorators.

```bash
cd /private/tmp/ci-d && git grep -n -E "(^|[^A-Za-z])TextArea\(" -- tldw_chatbook/UI/MCP_Modules/
```

Expected after the edit: no output. Every construction is `DetachSafeTextArea(`.

- [ ] **Step 6: Before/after evidence on the flaky file**

Run the file 10 times on `origin/dev` (before) and 10 times on the branch (after), serially, on
the fast lane's minimal dependency set:

```bash
cd /private/tmp/ci-d && cat > /tmp/ci-d-loop.sh <<'EOF'
#!/bin/bash
# usage: ci-d-loop.sh <worktree> <label>
cd "$1" || exit 1
for i in $(seq 1 10); do
  /private/tmp/ci-c-minvenv/bin/python -m pytest Tests/UI/test_mcp_workbench.py --timeout=180 \
    -q -p no:randomly -p no:cacheprovider --tb=line 2>&1 | grep -c "text-area--gutter"
done | awk -v l="$2" '{s+=$1} END {print l": text-area--gutter failures in 10 runs =", s}'
EOF
chmod +x /tmp/ci-d-loop.sh
```

If `/private/tmp/ci-c-minvenv` is gone, recreate it with Task 4 Step 3's first line. Create a
detached `origin/dev` worktree at `/private/tmp/ci-d-base`. Run
`/tmp/ci-d-loop.sh /private/tmp/ci-d-base before`, then `/tmp/ci-d-loop.sh /private/tmp/ci-d after`,
in the background; each takes several minutes.

Expected: `after ... = 0`. If `before ... = 0` too, the loop did not reproduce the race. Say so
in the PR. The deterministic test in Step 1 is then the evidence, and the loop is supporting
only.

- [ ] **Step 7: Preflight, commit, backlog, PR-D**

```bash
cd /private/tmp/ci-d && PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python && \
  out=$(PYTHON=$PY ./scripts/preflight.sh 2>&1); rc=$?; echo "PREFLIGHT_RC=$rc"
```

Expected: `PREFLIGHT_RC=0`. Then commit:

```bash
git add tldw_chatbook/Widgets/detach_safe_text_area.py Tests/Widgets/test_detach_safe_text_area.py tldw_chatbook/UI/MCP_Modules/ && \
git commit -m "fix(mcp): detach-safe TextArea stops the text-area--gutter repaint race (TASK-32049)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

Then:

1. Update TASK-32049: widen the scope to every test with this error. Add Implementation Notes
   covering the mechanism and the evidence. Set its status to Done after merge.
2. Push, then `gh pr create --base dev`. The body covers the mechanism, the deterministic
   negative control, and the before/after counts.
3. Merge under the Global Constraints rule.

---

## PR-B (rollout step 5): User Guide stamp rule

### Task 7: Replace the `CLAUDE.md` "UI changes" rule and note the lesson

**Files:**
- Modify: `CLAUDE.md` (the `**UI changes:**` paragraph)
- Modify: `backlog/docs/lessons-live-verification.md` (a dated note near lines 3124-3180)

Worktree, after PR-D is merged:

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook && git fetch -q origin && \
  git worktree add -b docs/ug-stamp-rule /private/tmp/ci-b origin/dev
```

- [ ] **Step 1: Replace the rule**

In `CLAUDE.md`, replace:

```markdown
**UI changes:** PRs that change a screen's UI should update the matching
`Docs/User_Guide/` page (or at least its "Verified against" stamp).
```

with:

```markdown
**UI changes:** PRs that change a screen's UI update the matching `Docs/User_Guide/`
page's content where behaviour changed. Record what was verified -- and against which
branch and date -- in the task's Implementation Notes, not in the User Guide page. Do
not add "Verified against" paragraphs to User Guide pages: parallel PRs appending them
at the same spot caused 121 sync-merge conflicts in one week (2026-09-27 CI spec).
```

- [ ] **Step 2: Add the lesson note**

Directly under the heading of the lesson that contains "run it before you stamp anything"
(around line 3124 of `backlog/docs/lessons-live-verification.md`), insert:

```markdown
> **2026-09-27:** verification is now recorded in the task's Implementation Notes, not
> as a "Verified against" paragraph on the User Guide page (CLAUDE.md "UI changes").
> The discipline below -- run it before you claim it -- is unchanged; only where the
> claim is written moved.
```

- [ ] **Step 3: Verify, commit, PR-B**

```bash
cd /private/tmp/ci-b && grep -n "Verified against" CLAUDE.md
```

Expected: only the new rule's "Do not add" sentence. Then:

1. Preflight, expecting `PREFLIGHT_RC=0`.
2. Commit with `docs: record UI verification in task notes, not User Guide stamps`, plus the
   trailer.
3. Create the backlog task and open the PR.
4. Merge under the Global Constraints rule.

---

## After all four PRs: the 2-week review

About 2 weeks after PR-B merges, measure:

- `scripts/measure_sync_merge_conflicts.py --merges 300`: the target is ≤ 25% conflicted.
- Runner-minutes for the changed workflows versus the baseline of about 1,100/day: the target is
  at least 80% lower.
- Fast-lane failures carrying `text-area--gutter`: the target is 0.
- The throughput measures in the spec: merges/day, ready-to-merged time, push-to-verdict p50/p90
  split into queue and run time, re-syncs per merged PR, and the DIRTY share.

Record the results in the program's backlog task and in memory.
