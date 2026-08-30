# TASK-26000 Ruff Formatter Debt Characterization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a pinned, mechanically checked current-development Ruff formatter
census and create conflict-safe cleanup records that own every current failure once.

**Architecture:** Four temporary standard-library Python tools census, validate,
allocate, and render one committed JSON evidence artifact plus cleanup records.
Revision-local censuses remain path-based;
explicit lineage records project stable identities across TASK-22514's divergent
branch, its common ancestor with current `dev`, and the current pin before any set
operation. The task then creates owner-aligned Backlog cleanup records, with the
last-created record owning the eventual repository-wide zero-exit gate.

**Tech Stack:** Git, Python 3.12.11 standard library, Ruff 0.15.22, Backlog Markdown,
pytest task-ID guard.

**Spec:**
`Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md`

## Global Constraints

- TASK-26000 changes no tracked Python source or test file.
- Use an explicitly supplied Python 3.12.11 interpreter whose
  `python -m ruff --version` output is exactly `ruff 0.15.22`; record the resolved
  executable but do not make its machine-specific absolute path normative.
- Before any command below, establish the one explicitly supplied interpreter input:

  ```bash
  : "${TASK26000_PYTHON:?Set TASK26000_PYTHON to an absolute Python 3.12.11 executable}"
  case "${TASK26000_PYTHON}" in
    /*) ;;
    *) echo 'E_TASK26000_PYTHON: executable must be absolute' >&2; exit 2 ;;
  esac
  test -x "${TASK26000_PYTHON}" || { echo 'E_TASK26000_PYTHON: executable is not executable' >&2; exit 2; }
  task26000_python="${TASK26000_PYTHON}"
  task26000_resolved_python="$("${task26000_python}" -c 'import os, sys; print(os.path.realpath(sys.executable))')"
  case "${task26000_resolved_python}" in
    /*) ;;
    *) echo 'E_TASK26000_PYTHON: resolved executable must be absolute' >&2; exit 2 ;;
  esac
  test "$("${task26000_python}" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')" = '3.12.11' || { echo 'E_TASK26000_PYTHON: expected Python 3.12.11' >&2; exit 2; }
  test "$("${task26000_python}" -m ruff --version)" = 'ruff 0.15.22' || { echo 'E_TASK26000_PYTHON: expected ruff 0.15.22' >&2; exit 2; }
  export task26000_python task26000_resolved_python
  ```

- Initial current pin is `ceac56e06eda4d3d2995a2f5ac8010a7a1821ed2`.
  Rebase, repin, and rerun the current census if `origin/dev` advances before the
  characterization records are committed.
- TASK-22514 evidence commit is
  `642b1c782fe6c066a781314dae669a55b05b62ad`, implementation base is
  `31ed49bb368f54211d6482599e00a5c1340f80b2`, and pre-closeout census is
  `1f4f72ac5ff02f5237a4946745e82e8932cd41cf`.
- Every census runs in a clean detached worktree at the exact revision and uses that
  revision's Ruff configuration.
- Enumerate Git paths with NUL delimiters, prefix Ruff path arguments with `./`, use
  `ruff format --check --force-exclude`, and derive failure membership only from
  exit codes: zero is non-failing, one would reformat, every other code is a blocker.
- Non-UTF-8 Git paths are losslessly recorded as base64 blockers. A blocker prevents
  manifest acceptance and cleanup-record creation.
- No current formatter failure may be omitted, duplicated, silently reclassified,
  or absorbed by the final cleanup record.
- Parent plans and task records use stable batch labels, not future task IDs. Cleanup
  records may reference TASK-26000 because it has a lower ID. The final cleanup
  record is created last and depends only on already-created lower IDs.
- Cleanup-record contracts require Ruff-only edits, AST equivalence with only
  `TypeIgnore.lineno` normalized, exact comment text/order, directive attachment and
  format-range preservation, focused-test rationale, scoped lint/format checks,
  `git diff --check`, and the task-ID guard.
- Use direct Markdown edits for five-digit Backlog IDs; the Backlog CLI's individual
  edit path is known to corrupt five-digit task IDs.
- Do not run the full test suite. This task is documentation/evidence-only; run the
  task-ID guard and exact mechanical evidence checks.
- ADR required: no. ADR path: N/A. This work records formatter evidence and future
  mechanical cleanup boundaries without changing runtime architecture.

---

## File Map

- Modify:
  `Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md`
  only if the current pin or derived common ancestor changes.
- Modify:
  `backlog/tasks/task-26000 - Characterize-current-dev-inherited-Ruff-formatter-drift.md`
  for status, the concise implementation plan, checked acceptance criteria, and
  implementation notes.
- Conditionally out of the approved boundary: a relevant
  `backlog/docs/lessons-*.md`. If Task 7 finds a genuinely new incident-backed
  lesson, stop before editing it, amend the design's explicit modification boundary,
  obtain owner reapproval, then include that one exact path in the stage set and
  no-Python scope check. Without that reapproval, `task26000_lesson_path` stays empty.
- Create:
  `Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json` as the
  single durable point-in-time census, lineage, comparison, and batch manifest.
- Create: one collision-safely allocated Markdown file under `backlog/tasks/` per
  stable batch label. Exact filenames are outputs of the ID-allocation step and are
  recorded in the JSON manifest; no higher task ID is written into TASK-26000 itself.
- Create temporarily, outside the repository:
  `task26000_tmp_root/task26000_ruff_census.py` for revision-local path/result
  capture, where `task26000_tmp_root` is one validated `mktemp -d` result.
- Create temporarily, outside the repository:
  `task26000_tmp_root/task26000_manifest_check.py` for schema, arithmetic, lineage,
  partition, batch, and cleanup-record validation.
- Create temporarily, outside the repository:
  `task26000_tmp_root/task26000_allocate_ids.py` for remote, open-PR, and worktree
  task-ID claims plus race-detecting allocation.
- Create temporarily, outside the repository:
  `task26000_tmp_root/task26000_render_cleanup.py` for deterministic, exclusive-create
  rendering of the allocated five-digit cleanup records.

## Interfaces

The temporary census tool consumes:

```text
"${task26000_python}" "${task26000_tmp_root}/task26000_ruff_census.py" \
  --checkout "${task26000_tmp_root}/checkouts/current" \
  --revision ceac56e06eda4d3d2995a2f5ac8010a7a1821ed2 \
  --label current \
  --output "${task26000_tmp_root}/raw/current.json"
```

An optional scoped call passes `--paths0 PATH`, where `PATH` is a NUL-terminated
byte stream of Git paths. TASK-26000 uses unscoped whole-tree snapshots for all five
revisions and projects `M` into the base/pre-closeout snapshots afterward. The
example current SHA is replaced by Task 1's recorded pin when `origin/dev` advances.

It produces this revision-local typed shape:

```python
class TrackedPythonRecord(TypedDict):
    path: str
    blob_id: str
    mode: str
    result: Literal["not_failing", "would_reformat", "blocked"]
    exit_code: int
    command_path: str


class RevisionSnapshot(TypedDict):
    schema_version: Literal[1]
    label: Literal["base", "pre_closeout", "closeout", "common", "current"]
    revision: str
    tree_oid: str
    toolchain: dict[str, str]
    scope: Literal["all_tracked_dot_py"]
    command_template: list[str]
    aggregate_command: list[str]
    configuration_inputs: list[dict[str, str]]
    entries: list[TrackedPythonRecord]
    blockers: list[dict[str, object]]
    aggregate: dict[str, object]
```

`result` is exactly `not_failing` or `would_reformat` for accepted entries;
`blocked` may appear only together with a blocker that makes the snapshot invalid.
Failures to decode, execute, parse configuration, obtain a blob ID, or reconcile the
aggregate control are blocker records, never formatter-debt membership.
Selected-scope snapshots exist only inside Appendix A self-tests/integrity probes and
are never promoted into the durable five-census manifest.

The temporary manifest checker consumes the final committed-shape JSON and repository
root. It exits nonzero unless all historical cardinalities and projected invariants
match, all current classifications are exhaustive/disjoint, every moved path has an
explicit lineage record, every current failure occurs in exactly one batch, blockers
occur in no batch, and every batch label resolves to exactly one cleanup record with
the required acceptance-criteria contract.

---

### Task 1: Refresh authority, pin the branch, and record planning state

**Files:**

- Modify:
  `backlog/tasks/task-26000 - Characterize-current-dev-inherited-Ruff-formatter-drift.md`
- Conditionally modify:
  `Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md`

**Interfaces:**

- Consumes: approved spec, current branch, `origin/dev`, TASK-22514 local evidence
  objects.
- Produces: one recorded task base, one full current pin, and one derived common
  ancestor used by every later task.

- [x] **Step 1: Recheck duplicate work and historical-object availability**

  Read the two mandatory area lessons before executing any plan step:

  ```bash
  sed -n '1,$p' backlog/docs/lessons-testing-evidence.md
  sed -n '1,$p' backlog/docs/lessons-backlog-hygiene.md
  ```

  Run:

  ```bash
  gh pr list --state all --search "26000" --json number,title,state,headRefName,url
  git branch -a --list '*26000*'
  git rev-parse 642b1c782fe6c066a781314dae669a55b05b62ad^{commit}
  git rev-parse 31ed49bb368f54211d6482599e00a5c1340f80b2^{commit}
  git rev-parse 1f4f72ac5ff02f5237a4946745e82e8932cd41cf^{commit}
  ```

  Expected: no competing PR/branch; all three historical objects resolve. A missing
  object blocks the task because the durable reconstruction has not yet been made.

- [x] **Step 2: Refresh `origin/dev` and inspect the exact feature range**

  Run:

  ```bash
  git fetch origin dev
  git log --oneline origin/dev..HEAD
  git merge-base origin/dev HEAD
  git rev-parse origin/dev
  ```

  Expected: the range contains only TASK-26000 documentation commits. Read the
  previously recorded `task_base` from the task plan (currently
  `ceac56e06eda4d3d2995a2f5ac8010a7a1821ed2`) into `task26000_previous_base`, and
  read refreshed `origin/dev` into `task26000_new_origin`. Verify the replay range:

  ```bash
  git diff --name-only "${task26000_previous_base}..HEAD"
  git log --oneline "${task26000_previous_base}..HEAD"
  ```

  Every path must be in the approved TASK-26000 documentation/evidence/Backlog
  boundary. Then replay only that range:

  ```bash
  git rebase --onto "${task26000_new_origin}" "${task26000_previous_base}"
  ```

  Never hardcode the initial base after the first repin and never replay imported dev
  commits.

- [x] **Step 3: Rebase and repin before any evidence run**

  Derive and record:

  ```bash
  task26000_current_pin="$(git rev-parse origin/dev)"
  task26000_task_base="${task26000_current_pin}"
  task26000_common_ancestor="$(git merge-base \
    642b1c782fe6c066a781314dae669a55b05b62ad \
    "${task26000_current_pin}")"
  ```

  ```text
  task_base = full origin/dev SHA after rebase
  current_pin = task_base
  common_ancestor = derived full SHA
  ```

  Update the design's initial pin if it changed. Update the task's Implementation
  Plan with the full three values; do not abbreviate evidence SHAs.

- [x] **Step 4: Verify the task boundary before continuing**

  Run:

  ```bash
  git merge-base --is-ancestor origin/dev HEAD
  git diff --name-only origin/dev...HEAD
  git diff --check origin/dev...HEAD
  ```

  Expected: only TASK-26000 task/spec/plan documentation exists so far and the diff
  check passes.

- [x] **Step 5: Commit the refreshed planning pin if it changed**

  ```bash
  git add Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md \
    Docs/superpowers/plans/2026-08-30-task-26000-ruff-formatter-debt.md \
    'backlog/tasks/task-26000 - Characterize-current-dev-inherited-Ruff-formatter-drift.md'
  git commit -m "docs: pin TASK-26000 formatter census"
  ```

  Skip the commit only when none of those files changed.

#### Task 1 execution record

Historical TASK-22514 objects resolved; no competing formatter branch or PR was
found. The formatter task was collision-renumbered to TASK-26000 under TASK-19601.
After origin/dev advanced, an initial safety abort exposed that obsolete historical
base `0ec518610cb50c4fa749bc97bc32761d4754cb81` would replay unrelated TASK-2803
commits. The corrected, explicitly historical prior pin
`53403791ca6b0faed8acd1ca649aa8cfc65a0043` replayed exactly the eight formatter
task/spec/plan commits onto current task base and pin
`c2f64f690bf4a712b604a1a1db348398df932f36`; common ancestor remains
`f0e8961222fe1a7a3ac7566f7f78142e717358f3`. `git merge-base --is-ancestor
origin/dev HEAD`, `git diff --check origin/dev...HEAD`, the no-Python-diff check,
and `Tests/CI/test_backlog_task_id_uniqueness.py -q` passed (3 passed). Appendices
A/B/C/D compiled and their self-tests passed; this review recorded the explicit
interpreter contract, scanner self-test, provenance narrowing, and completion boxes.

Safe recorded-base repin (2026-08-30): Task 2 changes were stashed only at
`stash@{0}` for the plan and TASK-26000 record. With a clean index, the verified
eleven-commit `c2f64f690bf4a712b604a1a1db348398df932f36..HEAD` slice touched only
the TASK-26000 task/spec/plan boundary. The fresh upstream delta to
`ceac56e06eda4d3d2995a2f5ac8010a7a1821ed2` touched README/screenshot/TASK-2803
paths only and had no overlap or TASK-26000 claim. That exact slice was rebased onto
the fresh pin; `task_base` and `current_pin` now equal the fresh pin, while the
derived common ancestor remains `f0e8961222fe1a7a3ac7566f7f78142e717358f3`.

---

### Task 2: Build and prove the temporary revision-local census tool

**Files:**

- Create temporarily: `task26000_tmp_root/task26000_ruff_census.py`
- Create temporarily: `task26000_tmp_root/census-selftest/`

**Interfaces:**

- Consumes: a clean checkout, exact revision, optional UTF-8 path array, and the
  pinned interpreter.
- Produces: one deterministic revision-local JSON snapshot matching the interface
  above; it never edits the checkout.

- [x] **Step 1: Create the temporary tool with fail-closed primitives**

  Create and validate one collision-resistant temporary root:

  ```bash
  task26000_tmp_root="$(mktemp -d /tmp/task26000.XXXXXX)"
  test -d "${task26000_tmp_root}"
  case "${task26000_tmp_root}" in
    /tmp/task26000.*) ;;
    *) exit 2 ;;
  esac
  mkdir "${task26000_tmp_root}/raw" "${task26000_tmp_root}/checkouts"
  ```

  Record the exact root in the task's execution notes so later commands reuse it;
  never fall back to a predictable `/tmp/task26000*` directory.

  Materialize Appendix A exactly. Its `build_snapshot(repo_value: str,
  expected_revision: str, label: str, selected: list[bytes] | None = None, ...)`
  loads one NUL-safe `git ls-tree -rz --full-tree` inventory, rejects checkout or
  ignore residue, inventories every tracked `pyproject.toml`, `ruff.toml`,
  `.ruff.toml`, `.gitignore`, and `.ignore` blob, and records the exact schema in the
  top-level Interfaces section. A selected path absent from the revision tree is a
  blocker. TASK-26000's real five-revision run leaves `--paths0` unset, so the
  aggregate control invokes Ruff on `.` and must agree with per-path failure
  existence.

  Each per-path invocation is exactly:

  ```python
  [
      sys.executable,
      "-m",
      "ruff",
      "format",
      "--check",
      "--force-exclude",
      "--no-cache",
      f"./{path}",
  ]
  ```

  It must not infer membership from stdout or stderr.

- [x] **Step 2: Add an internal self-test mode before using real revisions**

  `--self-test` creates a temporary Git repository containing a clean Python file,
  a formatter-red file, an excluded Python file, and tracked Python paths beginning
  with a dash and containing a space and newline. Use `x = 1\n` for every clean
  fixture and `x=[1,2,3]\n` for `fail.py` and `excluded.py`; configure
  `excluded.py` in a minimal `pyproject.toml` `[tool.ruff] exclude` list. Commit the
  fixtures before invoking the census so blob-ID capture exercises the production
  path.
  It asserts:

  ```python
  assert by_path["clean.py"]["result"] == "not_failing"
  assert by_path["fail.py"]["result"] == "would_reformat"
  assert by_path["excluded.py"]["result"] == "not_failing"
  assert by_path["-lead.py"]["result"] == "not_failing"
  assert by_path["space name.py"]["result"] == "not_failing"
  assert by_path["line\nbreak.py"]["result"] == "not_failing"
  assert snapshot["aggregate"]["exit_code"] == 1
  assert snapshot["blockers"] == []
  ```

  The test also creates a raw non-UTF-8 Git path, selects an absent path, supplies a
  malformed Ruff configuration, injects a nonzero Ruff execution error, and forces
  aggregate/per-path disagreement. Each case must have one exact expected blocker
  code and make the process exit nonzero; no test accepts multiple statuses.

- [x] **Step 3: Run the self-test and version gates**

  Run:

  ```bash
  "${task26000_python}" --version
  "${task26000_python}" -m ruff --version
  "${task26000_python}" \
    "${task26000_tmp_root}/task26000_ruff_census.py" --self-test
  ```

  Expected: Python 3.12.11, Ruff 0.15.22, and self-test exit zero. Record the
  resolved executable in evidence; do not copy its absolute path into child-task
  requirements.

---

### Task 3: Reconstruct historical scope and run all pinned censuses

**Files:**

- Create temporarily: `task26000_tmp_root/checkouts/`
- Create temporarily: `task26000_tmp_root/raw/`
- Create temporarily: `task26000_tmp_root/m-identities.json`

**Interfaces:**

- Consumes: Task 1's five revisions and Task 2's census tool.
- Produces: raw snapshots for `base`, `pre_closeout`, `closeout`, `common`, and
  `current`, plus revision-path identities for the original changed manifest.

- [ ] **Step 1: Create clean detached worktrees at every exact revision**

  Reuse Task 2's validated temporary root and run:

  ```bash
  git worktree add --detach "${task26000_tmp_root}/checkouts/base" \
    31ed49bb368f54211d6482599e00a5c1340f80b2
  git worktree add --detach "${task26000_tmp_root}/checkouts/pre_closeout" \
    1f4f72ac5ff02f5237a4946745e82e8932cd41cf
  git worktree add --detach "${task26000_tmp_root}/checkouts/closeout" \
    642b1c782fe6c066a781314dae669a55b05b62ad
  git worktree add --detach "${task26000_tmp_root}/checkouts/common" \
    "${task26000_common_ancestor}"
  git worktree add --detach "${task26000_tmp_root}/checkouts/current" \
    "${task26000_current_pin}"
  ```

  The five paths correspond to:

  ```text
  base          31ed49bb368f54211d6482599e00a5c1340f80b2
  pre_closeout  1f4f72ac5ff02f5237a4946745e82e8932cd41cf
  closeout      642b1c782fe6c066a781314dae669a55b05b62ad
  common        Task 1 common_ancestor
  current       Task 1 current_pin
  ```

  For every worktree, assert full `HEAD`, empty `git status --porcelain`, and absence
  of untracked Python files before running Ruff.

- [ ] **Step 2: Reconstruct `M` as stable identities**

  Parse this command as NUL-delimited bytes:

  ```bash
  git diff --name-status -z -M \
    31ed49bb368f54211d6482599e00a5c1340f80b2..1f4f72ac5ff02f5237a4946745e82e8932cd41cf \
    -- '*.py'
  ```

  Emit identity records with stable IDs beginning at `I-0000` and increasing
  monotonically in tuple sort order `(base_path or "", pre_closeout_path or "")`.
  Modified paths project to the same
  name at both revisions, adds have `base_path: null`, deletes have
  `pre_closeout_path: null`, and renames retain both names and Git's similarity
  score. Assert exactly 99 identity records before continuing.

- [ ] **Step 3: Run full base and pre-closeout censuses, then project `M`**

  Run the census tool over the full tracked-Python universe at both revisions. Derive
  `B` from non-null `base_path` projections into the base failure set and `C` from
  non-null `pre_closeout_path` projections into the pre-closeout failure set. Assert:

  ```text
  |M| = 99
  |B| = 64
  |C| = 77
  |C - B| = 16
  |B - C| = 3
  |H = B & C| = 61
  ```

  Identity membership, not a shared path string, defines `B`, `C`, and `H`.

- [ ] **Step 4: Run whole-repository closeout, common, and current censuses**

  Run the census tool without `--paths0` at all three revisions. Keep each run
  in its own raw JSON file. Any blocker or disagreement between aggregate and
  per-path failure existence stops the task.

- [ ] **Step 5: Resolve the complete lineage graph needed by every classification**

  Record optional `base`, `pre_closeout`, `closeout`, `common`, and `current` paths
  for every `M` identity. Separately create an identity for every `F_common` failure
  and project it from common ancestor to current, including non-`H` debt. Use Git
  rename evidence (`git diff --find-renames --name-status`) as a candidate, then
  inspect each moved/deleted/copied identity with `git log --follow` and blob IDs. A
  path string match across divergent branches is insufficient by itself. Record
  explicit `rename`, `delete`, `add`, `copy`, `unchanged`, or `ambiguous` lineage
  entries with source/target revisions and paths. Any ambiguity is a blocker rather
  than a classification guess.

- [ ] **Step 6: Prove TASK-22514's final projected invariant**

  Assert:

  ```text
  F_closeout & project(M, closeout) == project(H, closeout)
  ```

  Expected: exactly the projected 61 identities. A mismatch blocks the task and is
  reported as a TASK-22514 evidence inconsistency; do not change the expected set to
  make the assertion pass.

---

### Task 4: Build the durable manifest and prove its negative cases

**Files:**

- Create:
  `Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json`
- Create temporarily: `task26000_tmp_root/task26000_manifest_check.py`

**Interfaces:**

- Consumes: all Task 3 raw snapshots and identity/lineage records.
- Produces: the single durable manifest and a validator that exits zero only for a
  complete, consistent artifact.

- [ ] **Step 1: Assemble the manifest with canonical JSON serialization**

  Use UTF-8, `sort_keys=True`, `indent=2`, and a final newline. Include:

  ```text
  schema_version = 1
  generated_at_utc
  tools.{python_version,ruff_version,resolved_python}
  revisions.{task_base,base,pre_closeout,closeout,common,current}
  commands.{common_ancestor,historical_diff,censuses}
  source_reachability.{base,pre_closeout,closeout,common,current}
  censuses.{base,pre_closeout,closeout,common,current}
  identities
  historical_diff
  historical_sets.{M,B,C,H}
  classifications.{historical_still_current,historical_no_longer_current,
                    shared_ancestor_debt,current_line_drift}
  copy_splits
  blockers
  batches
  final_batch_label
  cleanup_records
  ```

  This list is the exact top-level schema; extra or missing keys fail validation.
  Preserve each complete Appendix A snapshot, including revision/tree OIDs,
  toolchain, exact commands, configuration/ignore blob inventory, scope, aggregate
  control, entries, and blockers. Compare retained JSON only after
  JSON-normalizing fresh producer output so tuple/list representation cannot create
  a false mismatch. The `identities` universe is the union of every `M` identity,
  every `F_common` failure identity, and an explicit identity for each remaining
  current-only failure; therefore every common/current failure is projectable before
  classification.

  Each `source_reachability` value is exactly
  `{object_present: true, remote_tracking_refs: [...]}`, with the refs sorted and
  unique. The three historical raw commits currently have empty ref arrays;
  `current` must include `refs/remotes/origin/dev`. The committed appendices are the durable tool source;
  `tools`, `commands`, and every complete census snapshot record the exact runtime
  and invocation provenance without a pre-records/final schema transition.

- [ ] **Step 2: Classify current failures exhaustively**

  Apply lineage projections first, then assign each current failure to exactly one:

  ```text
  historical_still_current
  shared_ancestor_debt
  current_line_drift
  ```

  Separately record every `H` identity with no current failing projection under
  `historical_no_longer_current`, with formatting, deletion, rename, or configuration
  lineage. For each `current_line_drift` item, identify the first relevant
  current-line commit or
  addition, rename, or Ruff-configuration interval from the common ancestor; do not
  label a divergent closeout-branch state as its temporal ancestor.

  The validator derives, rather than trusts, the stored categories using these exact
  formulas over identity projections:

  ```text
  historical_still_current =
      current_failures & project(H, current)
  historical_no_longer_current =
      H identities whose current projection is absent or not in current_failures
  shared_ancestor_debt =
      (current_failures - historical_still_current) &
      project(F_common identities, current)
  current_line_drift =
      current_failures - historical_still_current - shared_ancestor_debt
  ```

  `project(F_common identities, current)` comes from Task 3's complete
  common-to-current lineage graph. A duplicate current projection, unexplained copy,
  or ambiguous mapping is a blocker.

- [ ] **Step 3: Define owner-aligned stable batches**

  Capture active ownership evidence first:

  ```bash
  gh pr list --state open --json number,headRefName,title
  git worktree list --porcelain
  git log --since='14 days ago' --name-only --format='commit %H' origin/dev -- '*.py'
  ```

  Inspect each open PR that overlaps a current failure with
  `gh pr view` and its `files` JSON field, then record the PR number and path overlap
  in `conflict_basis`.

  Use stable labels with prefix `ruff-` followed by lowercase owner and surface
  slugs. Group production modules with their directly corresponding tests and shared
  focused-test surface. Separate
  root/CI/architecture paths, unusually large modules, and paths modified on active
  branches. Record `owner_basis`, `test_surface`, and `conflict_basis` for every
  batch. Each test surface and conflict basis is nonempty; when no overlap exists,
  record one `{source: "none", reference: "none-at-" + current_pin, paths: []}`
  entry rather than an ambiguous empty list. Split a broad subsystem only at a
  reviewable ownership/test boundary; do not use arbitrary fixed-size chunks.

- [ ] **Step 4: Implement the manifest checker**

  Materialize Appendix B verbatim. The checker owns the exact 16-key schema shown in
  Step 1; every nested object is closed to extra keys, every array has a declared
  order/uniqueness rule, every revision, blob, command, source-reachability claim,
  census control, lineage transition, provenance record, stored set, classification,
  blocker, batch, and cleanup-record binding is checked recursively. Do not replace
  the oracle with a JSON Schema or a looser producer-local assertion set.

  Its core assertions must be explicit:

  ```python
  assert len(M) == 99
  assert len(B) == 64
  assert len(C) == 77
  assert len(C - B) == 16
  assert len(B - C) == 3
  assert len(H) == 61
  assert H == B & C
  assert closeout_failures & projected_M_closeout == projected_H_closeout
  derived_historical = current_failures & projected_H_current
  derived_historical_gone = H - H_with_failing_current_projection
  derived_common = (current_failures - derived_historical) & projected_F_common_current
  derived_current_line = current_failures - derived_historical - derived_common
  assert stored_historical == derived_historical
  assert stored_historical_gone == derived_historical_gone
  assert stored_common_debt == derived_common
  assert stored_current_line_drift == derived_current_line
  assert current_failures == derived_historical | derived_common | derived_current_line
  assert not derived_historical & derived_common
  assert not derived_historical & derived_current_line
  assert not derived_common & derived_current_line
  assert current_failures == set().union(*(batch.paths for batch in batches))
  assert sum(len(batch.paths) for batch in batches) == len(current_failures)
  assert not blocker_paths & current_failures
  ```

  Also require full 40-character revisions, sorted unique arrays, blob IDs for every
  present path, a lineage record for every moved path, one cleanup-record path per
  batch label, and the required child-task acceptance-criteria phrases.

- [ ] **Step 5: Prove the checker fails for corrupt manifests**

  Generate JSON-normalized temporary mutations and require nonzero exits for:

  ```text
  missing-current-failure         E_BATCH_UNION
  duplicate-batch-path            E_BATCH_OVERLAP
  blocker-in-batch                E_BLOCKER_BATCH
  missing-rename-lineage          E_LINEAGE_KEYS
  category-swap                   E_CLASS_SHARED
  omitted-resolved                E_CLASS_RESOLVED
  duplicate-current-owner         E_CURRENT_OWNER
  wrong-closeout-projection       E_CLOSEOUT_INVARIANT
  wrong-historical-cardinality    E_HISTORICAL_COUNTS
  altered-historical-row          E_HISTORICAL_DIFF
  overlapping-comparison          E_CLASS_OVERLAP
  absent-cleanup-record           E_RECORD_COUNT
  missing-behavior-contract       E_RECORD_CONTRACT
  missing-final-gate              E_FINAL_GATE
  ```

  Restore the unmodified manifest after every mutation. Run the Appendix B built-in
  fixtures first. The arithmetic fixture contains exactly 99 `M` identities plus one
  shared-ancestor and one current-only identity, passes both positive phases, and
  applies the original 13 mutations independently. The authentic Git fixture covers
  `A`, `D`, `M`, and `R100`, then changes one stored historical row without changing
  the authentic stdout digest for the 14th mutation. Each mutation must produce the
  exact first code recorded by Appendix B.

  ```bash
  "${task26000_python}" \
    "${task26000_tmp_root}/task26000_manifest_check.py" --self-test
  ```

  Expected stdout is exactly:

  ```text
  manifest self-tests: 2 positive phases and 14 deterministic mutations passed
  ```

  The positive check remains red until cleanup records are created in Task 5; all
  earlier structural and arithmetic assertions must already pass.

  Run the pre-records phase explicitly:

  ```bash
  "${task26000_python}" \
    "${task26000_tmp_root}/task26000_manifest_check.py" \
    --phase pre-records \
    --manifest Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
    --repo "$PWD"
  ```

- [ ] **Step 6: Record derived counts and stable labels in both plans**

  After all Task 4 arithmetic checks pass, append an `Execution Record` section to
  this detailed plan containing the exact pins, `M/B/C/H/F_closeout/F_common/current`
  counts, the four comparison counts, blocker count, and sorted stable batch labels.
  Append the same counts and labels—never higher task IDs—to TASK-26000's concise
  Implementation Plan. Regenerate canonical JSON after any resulting label change.

---

### Task 5: Allocate and create every cleanup Backlog record

**Files:**

- Create: one direct file under `backlog/tasks/` per manifest batch.
- Modify:
  `Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json`
- Create temporarily: `task26000_tmp_root/task26000_allocate_ids.py`
- Create temporarily: `task26000_tmp_root/task26000_render_cleanup.py`
- Create temporarily: `task26000_tmp_root/active-cleanup-state.json`
- Create temporarily: `task26000_tmp_root/cleanup-render-transaction.json`

**Interfaces:**

- Consumes: stable batches whose union exactly equals current failures.
- Produces: one atomic cleanup record per batch and a final record whose dependencies
  name only earlier-created lower IDs.

- [ ] **Step 1: Allocate IDs against the live repository and in-flight work**

  Materialize Appendix C, then run:

  ```bash
  "${task26000_python}" \
    "${task26000_tmp_root}/task26000_allocate_ids.py" --self-test
  "${task26000_python}" \
    "${task26000_tmp_root}/task26000_allocate_ids.py" \
    --repo "$PWD" \
    --manifest Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
    --output "${task26000_tmp_root}/raw/allocation.json"
  "${task26000_python}" -m pytest \
    Tests/CI/test_backlog_task_id_uniqueness.py -q
  ```

  Expected scanner self-test stdout is exactly:

  ```text
  allocation scanner self-tests: 8 cases passed
  ```

  Inspect the scanner's claims for title-fragment renumbered twins before accepting
  its leapfrog allocation. Do not assume the local maximum is authoritative; the
  scanner's refreshed origin branches, paginated PR-head snapshots, and all local
  worktrees are mandatory inputs.

- [ ] **Step 2: Create non-final cleanup records first**

  Materialize Appendix D verbatim. If the create journal exists, recover it before
  the precreate scanner: recovery removes a partial uncommitted generation or
  completes a generation whose manifest commit landed. Then repeat Appendix C with
  the initial audit as `--expect-map`; an external claim on any reserved ID blocks
  rendering. Then invoke the renderer once:

  ```bash
  "${task26000_python}" \
    "${task26000_tmp_root}/task26000_render_cleanup.py" --self-test
  if test -e "${task26000_tmp_root}/cleanup-render-transaction.json"; then
    "${task26000_python}" \
      "${task26000_tmp_root}/task26000_render_cleanup.py" \
      --mode recover \
      --repo "$PWD" \
      --journal "${task26000_tmp_root}/cleanup-render-transaction.json"
  fi
  test ! -e "${task26000_tmp_root}/cleanup-render-transaction.json"
  "${task26000_python}" \
    "${task26000_tmp_root}/task26000_allocate_ids.py" \
    --repo "$PWD" \
    --manifest Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
    --output "${task26000_tmp_root}/raw/allocation-precreate.json" \
    --expect-map "${task26000_tmp_root}/raw/allocation.json"
  "${task26000_python}" \
    "${task26000_tmp_root}/task26000_render_cleanup.py" \
    --mode create \
    --repo "$PWD" \
    --manifest Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
    --allocation "${task26000_tmp_root}/raw/allocation.json" \
    --paths0-output "${task26000_tmp_root}/raw/new-task-paths0" \
    --active-state-output "${task26000_tmp_root}/active-cleanup-state.json" \
    --journal "${task26000_tmp_root}/cleanup-render-transaction.json"
  ```

  The renderer validates every batch/title/allocation before writing, creates every
  non-final file first with dependency `TASK-26000`, creates the final/highest-ID
  file last with all earlier cleanup IDs as dependencies, atomically writes the
  exact top-level `cleanup_records`, and emits the NUL-delimited task path list.
  An initial create requires every target and handoff to be absent; after the manifest
  is bound, an identical rerun accepts only exact rendered task bytes and exact-or-
  absent handoffs. Nothing is silently overwritten. `--active-state-output`
  atomically publishes the outside-repository handoff that later tasks must use
  instead of hardcoding `allocation.json` or `new-task-paths0`. The renderer self-test must print exactly
  `cleanup renderer self-tests: 5 cases passed` before any real create, refresh, or
  reallocate invocation.

- [ ] **Step 3: Create the final cleanup record last**

  Give it the same per-batch contract and dependencies on every earlier cleanup
  record. Add an acceptance criterion requiring an explicit Git-tracked,
  repository-wide `ruff format --check --force-exclude` zero exit after dependencies
  merge. Require any new unassigned failure to block the gate or receive an
  independently created correction record; the final task must not absorb it.
  The manifest checker parses every final dependency and asserts its numeric task ID
  is lower than the final record's ID. The final command is exactly the pinned
  interpreter's `-m ruff format --check --force-exclude .` from a clean Git-tracked
  repository checkout after all dependencies merge.

- [ ] **Step 4: Bind batch labels to records and run the positive checker**

  Inspect the renderer's canonical record summary. Do not duplicate task paths inside
  batch objects: top-level `cleanup_records` is the only label-to-record binding.
  Run:

  ```bash
  "${task26000_python}" \
    "${task26000_tmp_root}/task26000_manifest_check.py" \
    --phase final \
    --manifest Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
    --repo "$PWD"
  ```

  Expected: exit zero with counts for historical sets, current classifications,
  batches, cleanup records, and zero blockers.

- [ ] **Step 5: Run Backlog identity and platform-name guards**

  Run:

  ```bash
  "${task26000_python}" -m pytest \
    Tests/CI/test_backlog_task_id_uniqueness.py -q
  git diff --check
  ```

  Expected: three task-ID tests pass, every new basename is checkout-safe, and diff
  check is clean. Existing pytest temporary-directory cleanup warnings are recorded
  separately and do not change the test result.

- [ ] **Step 6: Commit the evidence and cleanup records**

  Immediately repeat the remote-ref, open-PR-head, candidate-ID, filename/frontmatter,
  and uniqueness scans from Step 1. The normal no-collision path is:

  ```bash
  "${task26000_python}" \
    "${task26000_tmp_root}/task26000_allocate_ids.py" \
    --repo "$PWD" \
    --manifest Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
    --output "${task26000_tmp_root}/raw/allocation-rescan.json" \
    --expect-map "${task26000_tmp_root}/raw/allocation.json"
  ```

  `E_ORIGIN_DEV` is not an allocation collision. It means the fetch advanced
  `origin/dev` beyond the manifest pin: stop, run the full Task 7 Step 1 repin cycle
  (including census, lineage, renderer, mutations, and review), and return here only
  with a manifest pinned to the refreshed remote-tracking commit.

  An `E_ID_COLLISION` after rendering, or a reviewed change that adds, removes, or
  renames any batch label, must not be repaired with manual renames or a sequence of
  `refresh` calls. Preserve the old manifest `cleanup_records` and their task files,
  update only the reviewed `batches` / `final_batch_label` structure when applicable,
  and run this exact recovery once. The allocator intentionally runs without
  `--expect-map`: its fresh audit must observe every old generated ID as occupied,
  and Appendix D rejects an audit that does not contain all of those old IDs.

  ```bash
  "${task26000_python}" \
    "${task26000_tmp_root}/task26000_allocate_ids.py" \
    --repo "$PWD" \
    --manifest Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
    --output "${task26000_tmp_root}/raw/allocation-recovery.json"
  "${task26000_python}" \
    "${task26000_tmp_root}/task26000_render_cleanup.py" \
    --mode reallocate \
    --repo "$PWD" \
    --manifest Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
    --allocation "${task26000_tmp_root}/raw/allocation-recovery.json" \
    --paths0-output "${task26000_tmp_root}/raw/reallocated-task-paths0" \
    --active-state-output "${task26000_tmp_root}/active-cleanup-state.json" \
    --journal "${task26000_tmp_root}/cleanup-render-transaction.json"
  ```

  `reallocate` is journaled and no-overwrite: before any retirement it verifies every
  old task path and `task_sha256`, pre-renders the complete replacement set, refuses
  any occupied new path, and durably records exact old/new bytes for tasks, manifest,
  NUL path list, and active handoff. The manifest is the commit oracle. Recovery
  restores the old generation when its bytes remain, or completes and verifies the
  new generation when the manifest commit landed. `E_REALLOCATE_DIRTY`,
  `E_REALLOCATE_TARGET_EXISTS`, or `E_TRANSACTION_RECOVERY` is a hard stop for
  inspection. The recovery path list is
  the NUL-delimited union of deleted/renamed old paths and all new paths. Use
  `update-index --add --remove` so both a tracked deletion and a retired, never-tracked
  path are handled without broadening the stage set.

  A normal refresh/reallocation removes its journal only after the selected
  generation is fully verified. After an interruption, or whenever the journal
  remains, run this exact idempotent recovery command before any renderer, scanner,
  staging, or commit command; do not delete or edit the journal manually:

  ```bash
  "${task26000_python}" \
    "${task26000_tmp_root}/task26000_render_cleanup.py" \
    --mode recover \
    --repo "$PWD" \
    --journal "${task26000_tmp_root}/cleanup-render-transaction.json"
  test ! -e "${task26000_tmp_root}/cleanup-render-transaction.json"
  ```

  Expected output is exactly `cleanup renderer recovery: rolled-back` when the old
  manifest remained or `cleanup renderer recovery: committed` when the manifest swap
  landed. Recovery changes or removes only bytes recorded in the journal; any third
  generation, symlink, schema mismatch, or changed journal fails closed and retains
  the journal for inspection.

  Both successful branches leave exactly one durable selection handoff at
  `${task26000_tmp_root}/active-cleanup-state.json`. Its closed object contains
  `schema_version`, `mode`, the active label-to-ID `allocation`, the absolute active
  `paths0_output`, and `record_set_sha256`. The digest covers the sorted closed
  identity projection (`label`, `path`, `task_id`) of the current manifest records,
  so an ordinary repin refresh does not invalidate the allocation handoff. Before
  `paths0_sha256` separately binds the exact NUL path-list bytes, including the
  old-plus-new union required after reallocation. Before any later task consumes the
  allocation or path file, run this exact resolver; a
  schema, digest, allocation, path, or NUL-list mismatch is
  `E_ACTIVE_ALLOCATION` and blocks all staging or scans:

  ```bash
  task26000_active_state="${task26000_tmp_root}/active-cleanup-state.json"
  task26000_active_exports="$(
  env -u PYTHONOPTIMIZE "${task26000_python}" -c '
  import hashlib, json, os, shlex, sys
  from pathlib import Path, PurePosixPath
  def need(condition):
      if not condition:
          raise SystemExit("E_ACTIVE_ALLOCATION")
  state_path, manifest_path, tmp_path, repo_path = map(lambda value: Path(value).resolve(), sys.argv[1:])
  def valid_task_path(value):
      relative = PurePosixPath(value)
      return (
          not relative.is_absolute()
          and len(relative.parts) == 3
          and relative.parts[:2] == ("backlog", "tasks")
          and relative.parts[2] not in {".", ".."}
          and "\n" not in value
      )
  state = json.loads(state_path.read_text(encoding="utf-8"))
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  records = manifest["cleanup_records"]
  identities = sorted(
      ({"label": row["label"], "path": row["path"], "task_id": row["task_id"]} for row in records),
      key=lambda row: row["label"],
  )
  canonical = (json.dumps(identities, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
  expected_allocation = {row["label"]: row["task_id"] for row in identities}
  need(set(state) == {"schema_version", "mode", "allocation", "paths0_output", "paths0_sha256", "record_set_sha256"})
  need(state["schema_version"] == 1 and state["mode"] in {"create", "reallocate"})
  need(state["allocation"] == expected_allocation)
  need(state["record_set_sha256"] == hashlib.sha256(canonical).hexdigest())
  paths0 = Path(state["paths0_output"]).resolve()
  need(paths0.parent == tmp_path / "raw" and paths0.is_file() and not paths0.is_symlink())
  expected_name = "new-task-paths0" if state["mode"] == "create" else "reallocated-task-paths0"
  need(paths0.name == expected_name)
  paths_raw = paths0.read_bytes()
  need(state["paths0_sha256"] == hashlib.sha256(paths_raw).hexdigest())
  fields = paths_raw.split(b"\0")
  need(bool(fields) and fields[-1] == b"" and all(fields[:-1]))
  decoded = [field.decode("utf-8") for field in fields[:-1]]
  need(decoded == sorted(set(decoded)) and all(valid_task_path(value) for value in decoded))
  current_paths = {row["path"] for row in records}
  need(current_paths <= set(decoded))
  retired_paths = set(decoded) - current_paths
  need(state["mode"] == "reallocate" or not retired_paths)
  need(not any(os.path.lexists(repo_path.joinpath(*PurePosixPath(value).parts)) for value in retired_paths))
  stage_mode = "add" if state["mode"] == "create" else "update-index"
  print("task26000_active_allocation=" + shlex.quote(str(state_path)))
  print("task26000_active_paths0=" + shlex.quote(str(paths0)))
  print("task26000_active_stage_mode=" + shlex.quote(stage_mode))
  ' "${task26000_active_state}" \
    Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
    "${task26000_tmp_root}" \
    "$PWD"
  )" || { echo 'E_ACTIVE_ALLOCATION' >&2; exit 2; }
  eval "${task26000_active_exports}"
  ```

  A mismatch is a hard stop, so a process interruption before the manifest commit
  cannot activate a prematurely published recovery handoff. Create mode is
  byte-idempotent: if only handoff
  publication fails after a successful create, rerun the exact Step 2 renderer
  command to publish it without changing generated records. Reallocate publishes the
  new handoff before mutation and restores the exact prior handoff on every caught
  rollback; it performs no fallible cleanup or handoff write after manifest commit.

  After either the normal rescan or the recovery command, rerun the positive checker,
  identity guard, and diff guard unconditionally:

  ```bash
  "${task26000_python}" \
    "${task26000_tmp_root}/task26000_manifest_check.py" \
    --phase final \
    --manifest Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
    --repo "$PWD"
  "${task26000_python}" -m pytest \
    Tests/CI/test_backlog_task_id_uniqueness.py -q
  git diff --check
  ```

  If the normal rescan succeeded, stage and commit with the original create-mode
  path list:

  ```bash
  git add Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json
  git add Docs/superpowers/plans/2026-08-30-task-26000-ruff-formatter-debt.md \
    'backlog/tasks/task-26000 - Characterize-current-dev-inherited-Ruff-formatter-drift.md'
  git add --pathspec-from-file="${task26000_tmp_root}/raw/new-task-paths0" \
    --pathspec-file-nul
  git commit -m "chore(backlog): partition current Ruff formatter debt"
  ```

  If recovery ran, stage the same durable inputs plus the exact removal/addition
  union, inspect its name-status, and commit instead with:

  ```bash
  git add Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json
  git add Docs/superpowers/plans/2026-08-30-task-26000-ruff-formatter-debt.md \
    'backlog/tasks/task-26000 - Characterize-current-dev-inherited-Ruff-formatter-drift.md'
  git update-index --add --remove -z --stdin \
    < "${task26000_tmp_root}/raw/reallocated-task-paths0"
  git diff --cached --name-status -- backlog/tasks
  git commit -m "chore(backlog): partition current Ruff formatter debt"
  ```

---

### Task 6: Independent evidence and contract review

**Files:**

- Modify as findings require:
  `Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json`
- Modify as findings require: newly created cleanup Backlog records.

**Interfaces:**

- Consumes: committed manifest, raw temporary snapshots, validator results, and
  cleanup records.
- Produces: an independent approval or concrete findings resolved before closeout.

- [ ] **Step 1: Request a subagent evidence review**

  Provide the reviewer with all pinned revisions, the approved spec/plan, manifest,
  validator, negative-mutation results, and task-record diff. Ask it to verify:

  ```text
  M/B/C/H reconstruction and 99/64/77/16/3/61 arithmetic
  divergent-branch common-ancestor choice
  every revision-path projection and moved-path lineage
  final-closeout projected invariant
  exhaustive/disjoint current classifications
  batch union/disjointness and conflict boundaries
  one cleanup record per label
  lower-ID-only final dependencies
  behavior-preservation and final-gate acceptance criteria
  no Python changes and no hidden blocker
  ```

- [ ] **Step 2: Verify every finding before editing**

  Reproduce factual findings against the raw snapshots and Git history. Apply only
  verified corrections, regenerate canonical JSON, rerun all negative mutations and
  the positive checker, then request re-review. Repeat until the reviewer returns
  APPROVED.

- [ ] **Step 3: Commit reviewed corrections**

  Rerun Task 5's exact active-state resolver against the committed manifest rather
  than assuming collision recovery did not run. A valid create handoff selects
  `new-task-paths0` with `git add`; a valid reallocation handoff selects
  `reallocated-task-paths0` with `git update-index --add --remove`. Any mismatch is
  `E_ACTIVE_ALLOCATION` and blocks the commit. Stage only manifest/task-record
  changes with the selected exact path set and commit:

  ```bash
  git add Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json
  git add Docs/superpowers/plans/2026-08-30-task-26000-ruff-formatter-debt.md \
    'backlog/tasks/task-26000 - Characterize-current-dev-inherited-Ruff-formatter-drift.md'
  if test "${task26000_active_stage_mode}" = add; then
    git add --pathspec-from-file="${task26000_active_paths0}" --pathspec-file-nul
  elif test "${task26000_active_stage_mode}" = update-index; then
    git update-index --add --remove -z --stdin < "${task26000_active_paths0}"
  else
    echo 'E_ACTIVE_ALLOCATION: invalid stage mode' >&2; exit 2
  fi
  git commit -m "docs: harden TASK-26000 formatter debt records"
  ```

  Skip only if the reviewer approved without changes.

---

### Task 7: Close TASK-26000 without executing cleanup work

**Files:**

- Modify:
  `backlog/tasks/task-26000 - Characterize-current-dev-inherited-Ruff-formatter-drift.md`

**Interfaces:**

- Consumes: approved manifest and cleanup records.
- Produces: a Done characterization task; future formatter cleanup remains To Do.

- [ ] **Step 1: Recheck `origin/dev` before closeout**

  Fetch and compare `origin/dev` with the recorded current pin:

  ```bash
  task26000_old_current_pin="${task26000_current_pin}"
  task26000_old_common_ancestor="${task26000_common_ancestor}"
  git fetch origin dev
  task26000_new_current_pin="$(git rev-parse refs/remotes/origin/dev)"
  if test "${task26000_new_current_pin}" != "${task26000_old_current_pin}"; then
    task26000_status="$(git -C "${task26000_tmp_root}/checkouts/current" status --porcelain=v1 --untracked-files=all)" || exit 2
    task26000_ignored="$(git -C "${task26000_tmp_root}/checkouts/current" ls-files --others --ignored --exclude-standard)" || exit 2
    test -z "${task26000_status}"
    test -z "${task26000_ignored}"
    git -C "${task26000_tmp_root}/checkouts/current" diff --quiet
    git -C "${task26000_tmp_root}/checkouts/current" diff --cached --quiet
    git worktree remove "${task26000_tmp_root}/checkouts/current"
    git worktree add --detach "${task26000_tmp_root}/checkouts/current" "${task26000_new_current_pin}"

    task26000_new_common_ancestor="$(git merge-base 642b1c782fe6c066a781314dae669a55b05b62ad "${task26000_new_current_pin}")"
    if test "${task26000_new_common_ancestor}" != "${task26000_old_common_ancestor}"; then
      task26000_common_status="$(git -C "${task26000_tmp_root}/checkouts/common" status --porcelain=v1 --untracked-files=all)" || exit 2
      task26000_common_ignored="$(git -C "${task26000_tmp_root}/checkouts/common" ls-files --others --ignored --exclude-standard)" || exit 2
      test -z "${task26000_common_status}"
      test -z "${task26000_common_ignored}"
      git worktree remove "${task26000_tmp_root}/checkouts/common"
      git worktree add --detach "${task26000_tmp_root}/checkouts/common" "${task26000_new_common_ancestor}"
    fi
  fi
  ```

  If the pin changed, rebase only the TASK-26000 range using the previously recorded
  task base, then update the full `task_base`, `current`, and `common` values in the
  design, task, plan, and manifest. Recreate the detached checkout before overwriting
  `${task26000_tmp_root}/raw/current.json`; always rerun the current census, and rerun
  the common census when its pin changed. Assert full HEAD, ordinary clean status,
  and no ignored residue in each recreated checkout. Rebuild common-to-current
  lineage, classifications, batches, record digests, and allocator evidence. A
  renderer refresh may replace only a task file whose current bytes match its old
  manifest SHA-256. If labels or IDs change, use Appendix D's hash-guarded
  structural-regeneration mode and its old-plus-new NUL pathspec output. Resolve any
  ID collision, rerun the census/manifest self-tests, the positive checker, the
  task-ID guard, and independent review. Never reset or reuse the stale detached
  checkout or raw current snapshot.

  A changed pin does not flow directly into Done closeout. After the corrected
  manifest/records receive independent approval, stage and commit that refresh as a
  separate cycle:

  ```bash
  git add -- \
    Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md \
    Docs/superpowers/plans/2026-08-30-task-26000-ruff-formatter-debt.md \
    Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
    'backlog/tasks/task-26000 - Characterize-current-dev-inherited-Ruff-formatter-drift.md'
  # First rerun Task 5's exact active-allocation resolver. It must bind
  # task26000_active_allocation, task26000_active_paths0, and
  # task26000_active_stage_mode to the one audit matching cleanup_records.
  if test "${task26000_active_stage_mode}" = add; then
    git add --pathspec-from-file="${task26000_active_paths0}" --pathspec-file-nul
  elif test "${task26000_active_stage_mode}" = update-index; then
    git update-index --add --remove -z --stdin < "${task26000_active_paths0}"
  else
    echo 'E_ACTIVE_ALLOCATION: invalid stage mode' >&2; exit 2
  fi
  "${task26000_python}" \
    "${task26000_tmp_root}/task26000_manifest_check.py" \
    --phase final \
    --manifest Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
    --repo "$PWD"
  "${task26000_python}" -m pytest \
    Tests/CI/test_backlog_task_id_uniqueness.py -q
  task26000_task_base=$("${task26000_python}" \
    -c 'import json; print(json.load(open("Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json"))["revisions"]["task_base"])')
  git diff --check "${task26000_task_base}"
  test -z "$(git diff --name-only "${task26000_task_base}" -- '*.py')"
  test -z "$(git ls-files --others --exclude-standard)"
  git commit -m "docs: refresh TASK-26000 formatter debt pin"
  test -z "$(git status --short)"
  ```

  Restart Task 7 Step 1 after this commit; proceed to Step 2 only when a fresh fetch
  leaves `origin/dev` equal to the committed manifest pin.

- [ ] **Step 2: Unconditionally rescan live task-ID claims**

  Rerun Task 5's exact active-allocation resolver, then execute Appendix C against
  that selected audit even when Step 1 found no `origin/dev` change:

  ```bash
  "${task26000_python}" \
    "${task26000_tmp_root}/task26000_allocate_ids.py" \
    --repo "$PWD" \
    --manifest Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
    --output "${task26000_tmp_root}/raw/allocation-closeout-rescan.json" \
    --expect-map "${task26000_active_allocation}"
  "${task26000_python}" \
    "${task26000_tmp_root}/task26000_manifest_check.py" \
    --phase final \
    --manifest Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
    --repo "$PWD"
  "${task26000_python}" -m pytest \
    Tests/CI/test_backlog_task_id_uniqueness.py -q
  ```

  Appendix C fetches every mandatory remote/PR/worktree source before comparing the
  full content-bound claim identities. `E_ORIGIN_DEV` returns to Step 1's repin
  cycle. `E_ID_COLLISION` runs Task 5's fresh no-`--expect-map` allocation and
  rollback-backed `reallocate`, reruns independent review, commits that recovery as
  its own cycle, and restarts Step 1. The final manifest oracle and task-ID guard
  must pass after the successful rescan; a recovery never permits closeout without
  another fresh `--expect-map` scan.

- [ ] **Step 3: Complete task hygiene before the final gate**

  Check all four TASK-26000 acceptance criteria, add concise Implementation Notes
  naming the pins, historical/current counts, comparison categories, batch labels,
  created record count, validator/mutation/review evidence, targeted test result,
  no-Python diff, and `ADR required: no`. Set status to Done only after every item is
  true. Do not mark any cleanup record Done.

- [ ] **Step 4: Decide whether a lessons entry is warranted**

  Add a concise incident-backed entry only if execution exposes a reusable trap not
  already captured by the historical-scope, divergent-branch, JSON-normalization,
  task-ID, or stale-dev lessons. Do not invent a lesson merely to fill the field. Set
  `task26000_lesson_path` to the empty string by default. Before editing a lesson,
  amend the approved design's modification boundary and obtain owner reapproval;
  only then set it to the one exact modified `backlog/docs/lessons-*.md` path.

- [ ] **Step 5: Stage closeout-only files and run final mechanical verification**

  Stage the parent task and the optional, validated lesson path before checking the
  task boundary, so the checks include both index and working-tree closeout edits:

  ```bash
  git add -- 'backlog/tasks/task-26000 - Characterize-current-dev-inherited-Ruff-formatter-drift.md'
  if test -n "${task26000_lesson_path}"; then
    case "${task26000_lesson_path}" in
      backlog/docs/lessons-*.md) git add -- "${task26000_lesson_path}" ;;
      *) echo 'invalid TASK-26000 lesson path' >&2; exit 2 ;;
    esac
  fi
  "${task26000_python}" \
    "${task26000_tmp_root}/task26000_manifest_check.py" \
    --phase final \
    --manifest Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
    --repo "$PWD"
  "${task26000_python}" -m pytest \
    Tests/CI/test_backlog_task_id_uniqueness.py -q
  task26000_task_base=$("${task26000_python}" \
    -c 'import json; print(json.load(open("Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json"))["revisions"]["task_base"])')
  git diff --check "${task26000_task_base}"
  test -z "$(git diff --name-only "${task26000_task_base}" -- '*.py')"
  test -z "$(git ls-files --others --exclude-standard)"
  ```

  Expected: manifest checker zero; three task-ID tests pass; diff check clean; the
  Python-path diff is empty. These commands do not claim the repository-wide Ruff
  gate is green—the final cleanup record owns that future outcome.

- [ ] **Step 6: Commit the characterization closeout**

  ```bash
  git commit -m "docs: close TASK-26000 formatter debt characterization"
  git status --short
  ```

  Expected: clean worktree. Preserve the temporary raw snapshots until the branch is
  reviewed/integrated; the committed JSON remains the durable evidence afterward.
  Remove every clean detached worktree with:

  ```bash
  git worktree remove "${task26000_tmp_root}/checkouts/base"
  git worktree remove "${task26000_tmp_root}/checkouts/pre_closeout"
  git worktree remove "${task26000_tmp_root}/checkouts/closeout"
  git worktree remove "${task26000_tmp_root}/checkouts/common"
  git worktree remove "${task26000_tmp_root}/checkouts/current"
  ```

  Keep tools/raw files under the validated root until integration; do not use a
  recursive deletion command.

---

## Appendix A: Exact Temporary Census Tool

Task 2 materializes the following standard-library implementation under the validated
temporary root. The CLI wrapper adds `--checkout`, `--revision`, `--label`, optional
`--paths0`, `--output`, and `--self-test`; it writes canonical JSON outside the
checkout and exits 2 when `blockers` is nonempty.

```python
from __future__ import annotations

import argparse
import base64
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

EXPECTED_PYTHON = (3, 12, 11)
EXPECTED_RUFF = "ruff 0.15.22"
RUFF = ("-m", "ruff", "format", "--check", "--force-exclude", "--no-cache")


class EvidenceError(RuntimeError):
    pass


def run(argv: tuple[str, ...], cwd: Path) -> subprocess.CompletedProcess[bytes]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("RUFF_")}
    env.update(
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_SYSTEM=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
        RUFF_NO_CACHE="1",
    )
    return subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise EvidenceError(f"{code}: {detail}")


def nul_records(raw: bytes, label: str) -> list[bytes]:
    require(not raw or raw.endswith(b"\0"), "E_GIT_NUL", f"{label} is unterminated")
    records = raw[:-1].split(b"\0") if raw else []
    require(all(records), "E_GIT_NUL", f"{label} contains an empty record")
    return records


def path_record(raw: bytes) -> dict[str, str]:
    try:
        return {"path": raw.decode("utf-8")}
    except UnicodeDecodeError:
        return {"path_b64": base64.b64encode(raw).decode("ascii")}


def load_tree(repo: Path, revision: str) -> dict[bytes, dict[str, str]]:
    cp = run(("git", "ls-tree", "-rz", "--full-tree", revision), repo)
    require(cp.returncode == 0, "E_GIT_TREE", f"exit {cp.returncode}")
    result: dict[bytes, dict[str, str]] = {}
    for row in nul_records(cp.stdout, "git ls-tree"):
        meta, raw_path = row.split(b"\t", 1)
        mode, kind, blob = meta.split(b" ", 2)
        if kind == b"blob":
            result[raw_path] = {
                "mode": mode.decode("ascii"),
                "blob_id": blob.decode("ascii"),
            }
    return result


def require_toolchain(
    version: tuple[int, int, int],
    ruff_result: subprocess.CompletedProcess[bytes],
) -> str:
    require(
        version == EXPECTED_PYTHON,
        "E_PYTHON_VERSION",
        f"expected 3.12.11; got {'.'.join(map(str, version))}",
    )
    got = ruff_result.stdout.decode("utf-8", "backslashreplace").strip()
    require(
        ruff_result.returncode == 0 and got == EXPECTED_RUFF,
        "E_RUFF_VERSION",
        f"expected {EXPECTED_RUFF!r}; got {got!r}; exit {ruff_result.returncode}",
    )
    return got


def require_clean_checkout(repo: Path) -> None:
    status = run(("git", "status", "--porcelain=v1", "-z", "--untracked-files=all"), repo)
    require(status.returncode == 0 and not status.stdout, "E_DIRTY_CHECKOUT", "status is nonempty")
    ignored = run(("git", "ls-files", "--others", "--ignored", "--exclude-standard", "-z"), repo)
    require(ignored.returncode == 0 and not ignored.stdout, "E_IGNORED_RESIDUE", "ignored files exist")
    local_excludes = run(("git", "config", "--local", "--get", "core.excludesFile"), repo)
    require(
        local_excludes.returncode != 0 or not local_excludes.stdout.strip(),
        "E_EXTERNAL_EXCLUDES",
        "local core.excludesFile is set",
    )
    info = run(("git", "rev-parse", "--git-path", "info/exclude"), repo)
    require(info.returncode == 0, "E_EXTERNAL_EXCLUDES", "cannot resolve info/exclude")
    info_path = Path(info.stdout.decode("utf-8").strip())
    if not info_path.is_absolute():
        info_path = repo / info_path
    if info_path.exists():
        active = [
            line
            for line in info_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        require(not active, "E_EXTERNAL_EXCLUDES", ".git/info/exclude has active patterns")


def aggregate_blocker(
    entries: list[dict[str, object]],
    aggregate_exit: int,
    blockers: list[dict[str, object]],
) -> dict[str, object]:
    aggregate = {"exit_code": aggregate_exit}
    if aggregate_exit not in (0, 1):
        blockers.append({"category": "aggregate_nonformatter_exit", "exit_code": aggregate_exit})
    elif not blockers:
        per_path_red = any(row["result"] == "would_reformat" for row in entries)
        if (aggregate_exit == 1) != per_path_red:
            blockers.append(
                {
                    "category": "aggregate_mismatch",
                    "aggregate_exit": aggregate_exit,
                    "per_path_failure_count": sum(
                        row["result"] == "would_reformat" for row in entries
                    ),
                }
            )
    return aggregate


def build_snapshot(
    repo_value: str,
    expected_revision: str,
    label: str,
    selected: list[bytes] | None = None,
    runner: Callable[[tuple[str, ...], Path], subprocess.CompletedProcess[bytes]] = run,
    tree_loader: Callable[[Path, str], dict[bytes, dict[str, str]]] = load_tree,
) -> dict[str, object]:
    repo = Path(repo_value).resolve()
    require(re.fullmatch(r"[0-9a-f]{40}", expected_revision) is not None, "E_REVISION", "full lowercase SHA required")
    head_result = runner(("git", "rev-parse", "HEAD^{commit}"), repo)
    require(head_result.returncode == 0, "E_REVISION", "cannot resolve checkout HEAD")
    head = head_result.stdout.decode("ascii").strip()
    require(head == expected_revision, "E_REVISION", f"expected {expected_revision}; got {head}")
    require_clean_checkout(repo)
    ruff_version = require_toolchain(
        sys.version_info[:3],
        runner((sys.executable, "-m", "ruff", "--version"), repo),
    )
    tree = tree_loader(repo, head)
    universe = sorted(raw for raw in tree if raw.endswith(b".py"))
    chosen = universe if selected is None else selected
    require(len(chosen) == len(set(chosen)), "E_SELECTION_DUPLICATE", "selected paths repeat")
    blockers: list[dict[str, object]] = []
    entries: list[dict[str, object]] = []
    for raw_path in chosen:
        record: dict[str, object] = path_record(raw_path)
        if raw_path not in tree:
            blockers.append({**record, "category": "selected_path_absent"})
            continue
        if not raw_path.endswith(b".py"):
            blockers.append({**record, "category": "selected_path_not_python"})
            continue
        record.update(tree[raw_path])
        try:
            path = raw_path.decode("utf-8")
        except UnicodeDecodeError:
            record["result"] = "blocked"
            entries.append(record)
            blockers.append({**path_record(raw_path), "category": "non_utf8_path"})
            continue
        argv = (sys.executable, *RUFF, f"./{path}")
        cp = runner(argv, repo)
        result = "not_failing" if cp.returncode == 0 else "would_reformat" if cp.returncode == 1 else "blocked"
        record.update(result=result, exit_code=cp.returncode, command_path=f"./{path}")
        if cp.returncode not in (0, 1):
            blockers.append(
                {
                    "path": path,
                    "category": "ruff_nonformatter_exit",
                    "exit_code": cp.returncode,
                    "stdout": cp.stdout.decode("utf-8", "backslashreplace"),
                    "stderr": cp.stderr.decode("utf-8", "backslashreplace"),
                }
            )
        entries.append(record)
    aggregate: dict[str, object] = {"status": "not_run_selected_scope"}
    if selected is None:
        cp = runner((sys.executable, *RUFF, "."), repo)
        aggregate = aggregate_blocker(entries, cp.returncode, blockers)
        aggregate.update(
            stdout=cp.stdout.decode("utf-8", "backslashreplace"),
            stderr=cp.stderr.decode("utf-8", "backslashreplace"),
        )
    after = runner(("git", "status", "--porcelain=v1", "-z", "--untracked-files=all"), repo)
    require(after.returncode == 0, "E_CHECKOUT_STATUS", "cannot inspect checkout after Ruff")
    ignored_after = runner(
        ("git", "ls-files", "--others", "--ignored", "--exclude-standard", "-z"),
        repo,
    )
    require(ignored_after.returncode == 0, "E_CHECKOUT_STATUS", "cannot inspect ignored residue after Ruff")
    if after.stdout or ignored_after.stdout:
        blockers.append(
            {
                "category": "checkout_mutated",
                "status_b64": base64.b64encode(after.stdout).decode("ascii"),
                "ignored_b64": base64.b64encode(ignored_after.stdout).decode("ascii"),
            }
        )
    config_names = {b"pyproject.toml", b"ruff.toml", b".ruff.toml", b".gitignore", b".ignore"}
    config = [
        {**path_record(raw), **tree[raw]}
        for raw in sorted(tree)
        if raw.rsplit(b"/", 1)[-1] in config_names
    ]
    tree_result = runner(("git", "rev-parse", "HEAD^{tree}"), repo)
    require(tree_result.returncode == 0, "E_GIT_TREE", "cannot resolve checkout tree")
    tree_oid = tree_result.stdout.decode("ascii").strip()
    return {
        "schema_version": 1,
        "label": label,
        "revision": head,
        "tree_oid": tree_oid,
        "toolchain": {
            "python": platform.python_version(),
            "resolved_python": str(Path(sys.executable).resolve()),
            "ruff": ruff_version,
        },
        "scope": "all_tracked_dot_py" if selected is None else "selected",
        "command_template": [sys.executable, *RUFF, "./PATH_FROM_GIT"],
        "aggregate_command": [sys.executable, *RUFF, "."] if selected is None else None,
        "configuration_inputs": config,
        "entries": entries,
        "blockers": blockers,
        "aggregate": aggregate,
    }


def read_paths0(path: Path) -> list[bytes]:
    raw = path.read_bytes()
    records = nul_records(raw, "--paths0")
    require(bool(records), "E_SELECTION_EMPTY", "--paths0 is empty")
    return records


def make_repo(root: Path, files: dict[bytes, bytes], config: bytes) -> tuple[Path, str]:
    root.mkdir()
    require(run(("git", "init"), root).returncode == 0, "E_SELFTEST", "git init failed")
    for raw_path, content in files.items():
        full = os.path.join(os.fsencode(root), raw_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as handle:
            handle.write(content)
    (root / "pyproject.toml").write_bytes(config)
    require(run(("git", "add", "--all"), root).returncode == 0, "E_SELFTEST", "git add failed")
    commit = run(
        (
            "git",
            "-c",
            "user.name=TASK-26000",
            "-c",
            "user.email=task26000@example.invalid",
            "commit",
            "-m",
            "fixture",
        ),
        root,
    )
    require(commit.returncode == 0, "E_SELFTEST", "git commit failed")
    head = run(("git", "rev-parse", "HEAD^{commit}"), root).stdout.decode("ascii").strip()
    return root, head


def expect_error(code: str, callback: Callable[[], object]) -> None:
    try:
        callback()
    except EvidenceError as exc:
        require(str(exc).startswith(f"{code}:"), "E_SELFTEST", f"expected {code}; got {exc}")
        return
    raise EvidenceError(f"E_SELFTEST: expected {code}")


def run_self_tests() -> None:
    with tempfile.TemporaryDirectory(prefix="task26000-census-") as temp_value:
        temp = Path(temp_value)
        basic, basic_head = make_repo(
            temp / "basic",
            {
                b"clean.py": b"x = 1\n",
                b"fail.py": b"x=[1,2,3]\n",
                b"excluded.py": b"x=[1,2,3]\n",
                b"-lead.py": b"x = 1\n",
                b"space name.py": b"x = 1\n",
                b"line\nbreak.py": b"x = 1\n",
            },
            b'[tool.ruff]\nexclude = ["excluded.py"]\n',
        )
        snapshot = build_snapshot(str(basic), basic_head, "selftest")
        by_path = {
            row["path"]: row
            for row in snapshot["entries"]
            if "path" in row
        }
        require(by_path["clean.py"]["result"] == "not_failing", "E_SELFTEST", "clean status")
        require(by_path["fail.py"]["result"] == "would_reformat", "E_SELFTEST", "fail status")
        require(by_path["excluded.py"]["result"] == "not_failing", "E_SELFTEST", "exclude status")
        require(by_path["-lead.py"]["result"] == "not_failing", "E_SELFTEST", "dash status")
        require(by_path["space name.py"]["result"] == "not_failing", "E_SELFTEST", "space status")
        require(by_path["line\nbreak.py"]["result"] == "not_failing", "E_SELFTEST", "newline status")
        require(snapshot["aggregate"]["exit_code"] == 1, "E_SELFTEST", "aggregate status")
        require(not snapshot["blockers"], "E_SELFTEST", "unexpected basic blocker")

        absent = build_snapshot(str(basic), basic_head, "selftest", [b"missing.py"])
        require(
            [row["category"] for row in absent["blockers"]] == ["selected_path_absent"],
            "E_SELFTEST",
            "absent selection blocker",
        )

        synthetic = {b"bad-\xff.py": {"mode": "100644", "blob_id": "0" * 40}}
        non_utf8 = build_snapshot(
            str(basic),
            basic_head,
            "selftest",
            [b"bad-\xff.py"],
            tree_loader=lambda _repo, _revision: synthetic,
        )
        require(
            non_utf8["blockers"] == [
                {"path_b64": base64.b64encode(b"bad-\xff.py").decode("ascii"), "category": "non_utf8_path"}
            ],
            "E_SELFTEST",
            "non-UTF-8 blocker",
        )

        malformed, malformed_head = make_repo(
            temp / "malformed",
            {b"clean.py": b"x = 1\n"},
            b"[tool.ruff\n",
        )
        malformed_snapshot = build_snapshot(
            str(malformed), malformed_head, "selftest", [b"clean.py"]
        )
        require(
            [row["category"] for row in malformed_snapshot["blockers"]]
            == ["ruff_nonformatter_exit"],
            "E_SELFTEST",
            "malformed-config blocker",
        )

        def nonformatter_runner(
            argv: tuple[str, ...],
            cwd: Path,
        ) -> subprocess.CompletedProcess[bytes]:
            if "format" in argv and argv[-1] == "./clean.py":
                return subprocess.CompletedProcess(argv, 2, b"", b"injected Ruff failure")
            return run(argv, cwd)

        injected = build_snapshot(
            str(basic),
            basic_head,
            "selftest",
            [b"clean.py"],
            runner=nonformatter_runner,
        )
        require(
            [row["category"] for row in injected["blockers"]]
            == ["ruff_nonformatter_exit"],
            "E_SELFTEST",
            "injected nonformatter blocker",
        )

        fake_ok = subprocess.CompletedProcess(("ruff",), 0, b"ruff 0.15.22\n", b"")
        fake_bad = subprocess.CompletedProcess(("ruff",), 0, b"ruff 0.15.21\n", b"")
        expect_error("E_PYTHON_VERSION", lambda: require_toolchain((3, 12, 10), fake_ok))
        expect_error("E_RUFF_VERSION", lambda: require_toolchain(EXPECTED_PYTHON, fake_bad))
        expect_error(
            "E_REVISION",
            lambda: build_snapshot(str(basic), "not-a-full-sha", "selftest"),
        )

        mismatch_blockers: list[dict[str, object]] = []
        aggregate_blocker(
            [{"result": "would_reformat"}],
            0,
            mismatch_blockers,
        )
        require(
            [row["category"] for row in mismatch_blockers] == ["aggregate_mismatch"],
            "E_SELFTEST",
            "aggregate mismatch blocker",
        )
    print("census self-tests: 10 cases passed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout")
    parser.add_argument("--revision")
    parser.add_argument("--label")
    parser.add_argument("--paths0")
    parser.add_argument("--output")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_tests()
        return 0
    require(all((args.checkout, args.revision, args.label, args.output)), "E_ARGS", "missing required argument")
    checkout = Path(args.checkout).resolve()
    output = Path(args.output).resolve()
    require(not output.is_relative_to(checkout), "E_OUTPUT_SCOPE", "output must be outside checkout")
    selected = read_paths0(Path(args.paths0)) if args.paths0 else None
    snapshot = build_snapshot(args.checkout, args.revision, args.label, selected)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 2 if snapshot["blockers"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvidenceError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"E_CENSUS_IO: {type(exc).__name__}", file=sys.stderr)
        raise SystemExit(2)
```

`run_self_tests()` is part of the same temporary file. It creates committed temporary
repositories with the exact fixture bytes from Task 2, invokes `build_snapshot`, and
asserts the exact statuses and blocker codes listed there. The non-UTF-8 case injects
`{b"bad-\xff.py": {"mode": "100644", "blob_id": "0" * 40}}` through `tree_loader`
instead of asking APFS to create an invalid filename. An injected runner returns
exit 2 for one otherwise-valid Ruff path invocation to prove nonformatter errors are
blocked independently of malformed configuration. Toolchain failures call
`require_toolchain` directly. Aggregate mismatch calls `aggregate_blocker` with one
`would_reformat` entry and aggregate exit zero. The self-test prints exactly
`census self-tests: 10 cases passed` only after clean/fail/excluded, dash/space/newline,
non-UTF-8, absent-selection, malformed-config/nonformatter, tool-version, and
aggregate-mismatch assertions all pass.

---

## Appendix B: Exact Manifest Derivation and Validation Oracle

Task 4 materializes this standard-library CLI verbatim. Its durable manifest has
exactly these 16 top-level keys and no others:

```text
schema_version
generated_at_utc
tools
revisions
commands
source_reachability
censuses
identities
historical_diff
historical_sets
copy_splits
classifications
blockers
batches
final_batch_label
cleanup_records
```

The validator closes every nested object, validates every ordered set and provenance
record, derives all stored historical/comparison sets, and checks census command
digests and aggregate controls before accepting either phase. `--self-test` builds
exactly 99 historical identities plus two added identities, passes both positive
phases, and applies the original 13 Task 4 mutations to fresh JSON-normalized
fixtures plus one authentic-repository historical-row mutation.

```python
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

LABELS = ("base", "pre_closeout", "closeout", "common", "current")
TOP_KEYS = {
    "schema_version", "generated_at_utc", "tools", "revisions", "commands",
    "source_reachability", "censuses", "identities", "historical_diff",
    "historical_sets", "copy_splits", "classifications", "blockers", "batches",
    "final_batch_label", "cleanup_records",
}
SHA = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")
IDENTITY = re.compile(r"I-[0-9]{4}")


class ManifestError(RuntimeError):
    pass


def need(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise ManifestError(f"{code}: {detail}")


def exact(value: Any, keys: set[str], code: str, detail: str) -> dict[str, Any]:
    need(isinstance(value, dict), code, f"{detail} is not an object")
    need(set(value) == keys, code, f"{detail} keys={sorted(value)}")
    return value


def text(value: Any, code: str, detail: str, *, nonempty: bool = True) -> str:
    need(isinstance(value, str) and (bool(value) or not nonempty), code, detail)
    return value


def full_sha(value: Any, code: str, detail: str) -> str:
    value = text(value, code, detail)
    need(SHA.fullmatch(value) is not None, code, detail)
    return value


def sha256_value(value: Any, code: str, detail: str) -> str:
    value = text(value, code, detail)
    need(SHA256.fullmatch(value) is not None, code, detail)
    return value


def sorted_unique(values: Any, code: str, detail: str) -> list[str]:
    need(isinstance(values, list) and all(isinstance(v, str) for v in values), code, detail)
    need(values == sorted(set(values)), code, detail)
    return values


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def paths_digest(paths: list[str]) -> str:
    raw = json.dumps(sorted(paths), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return digest(raw)


def validate_evidence(value: Any, code: str, detail: str) -> None:
    row = exact(value, {"commits", "summary"}, code, detail)
    commits = sorted_unique(row["commits"], code, f"{detail}.commits")
    for commit in commits:
        full_sha(commit, code, f"{detail}.commits")
    text(row["summary"], code, f"{detail}.summary")


def validate_tools(data: dict[str, Any]) -> None:
    tools = exact(data["tools"], {"python_version", "ruff_version", "resolved_python"}, "E_TOOLS", "tools")
    need(tools["python_version"] == "3.12.11", "E_TOOLCHAIN", "python")
    need(tools["ruff_version"] == "ruff 0.15.22", "E_TOOLCHAIN", "ruff")
    python = Path(text(tools["resolved_python"], "E_TOOLS", "resolved_python"))
    need(python.is_absolute(), "E_TOOLS", "resolved_python must be absolute")


def validate_revisions(data: dict[str, Any]) -> None:
    revisions = exact(data["revisions"], set(LABELS) | {"task_base"}, "E_REVISIONS", "revisions")
    for label, revision in revisions.items():
        full_sha(revision, "E_REVISIONS", label)


def validate_commands(data: dict[str, Any]) -> None:
    commands = exact(data["commands"], {"common_ancestor", "historical_diff", "censuses"}, "E_COMMANDS", "commands")
    common = exact(commands["common_ancestor"], {"argv", "cwd", "exit_code", "stdout"}, "E_COMMANDS", "common_ancestor")
    need(common["argv"] == ["git", "merge-base", data["revisions"]["closeout"], data["revisions"]["current"]], "E_COMMANDS", "common argv")
    need(common["cwd"] == "." and common["exit_code"] == 0, "E_COMMANDS", "common control")
    need(common["stdout"] == data["revisions"]["common"] + "\n", "E_COMMON_ANCESTOR", "stored stdout")
    historical = exact(commands["historical_diff"], {"argv", "cwd", "exit_code", "stdout_sha256"}, "E_COMMANDS", "historical_diff")
    need(historical["argv"] == ["git", "diff", "--name-status", "-z", "-M", f"{data['revisions']['base']}..{data['revisions']['pre_closeout']}", "--", "*.py"], "E_COMMANDS", "historical argv")
    need(historical["cwd"] == "." and historical["exit_code"] == 0, "E_COMMANDS", "historical control")
    sha256_value(historical["stdout_sha256"], "E_COMMANDS", "historical stdout")
    census_commands = exact(commands["censuses"], set(LABELS), "E_COMMANDS", "census commands")
    for label in LABELS:
        row = exact(census_commands[label], {"argv", "cwd", "exit_code", "output_sha256"}, "E_COMMANDS", f"census command {label}")
        need(isinstance(row["argv"], list) and all(isinstance(v, str) and v for v in row["argv"]), "E_COMMANDS", f"{label} argv")
        argv = row["argv"]
        need(len(argv) == 10, "E_COMMANDS", f"{label} argv length")
        need(argv[0] == data["tools"]["resolved_python"] and Path(argv[1]).name == "task26000_ruff_census.py", "E_COMMANDS", f"{label} executable")
        need(argv[2:3] == ["--checkout"] and Path(argv[3]).parts[-2:] == ("checkouts", label), "E_COMMANDS", f"{label} checkout")
        need(argv[4:6] == ["--revision", data["revisions"][label]], "E_COMMANDS", f"{label} revision")
        need(argv[6:8] == ["--label", label], "E_COMMANDS", f"{label} label")
        need(argv[8:9] == ["--output"] and Path(argv[9]).parts[-2:] == ("raw", f"{label}.json"), "E_COMMANDS", f"{label} output")
        need(row["cwd"] == "." and row["exit_code"] == 0, "E_COMMANDS", f"{label} control")
        need(row["output_sha256"] == digest(canonical_bytes(data["censuses"][label])), "E_COMMANDS", f"{label} snapshot digest")


def validate_reachability(data: dict[str, Any]) -> None:
    reachability = exact(data["source_reachability"], set(LABELS), "E_REACHABILITY", "source_reachability")
    for label in LABELS:
        row = exact(reachability[label], {"object_present", "remote_tracking_refs"}, "E_REACHABILITY", label)
        need(row["object_present"] is True, "E_REACHABILITY", f"{label} object")
        refs = sorted_unique(row["remote_tracking_refs"], "E_REACHABILITY", f"{label} refs")
        need(all(ref.startswith("refs/remotes/") for ref in refs), "E_REACHABILITY", f"{label} refs")
        if label in {"base", "pre_closeout", "closeout"}:
            need(refs == [], "E_REACHABILITY", f"{label} refs must be empty")
    need("refs/remotes/origin/dev" in reachability["current"]["remote_tracking_refs"], "E_REACHABILITY", "current lacks origin/dev")


def git_output(repo: Path, argv: list[str], code: str) -> bytes:
    cp = subprocess.run(tuple(argv), cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    need(cp.returncode == 0, code, cp.stderr.decode("utf-8", "backslashreplace").strip())
    return cp.stdout


def validate_repo_provenance(data: dict[str, Any], repo: Path) -> None:
    origin_dev = git_output(repo, ["git", "rev-parse", "refs/remotes/origin/dev^{commit}"], "E_REACHABILITY").decode("ascii").strip()
    need(origin_dev == data["revisions"]["current"], "E_REACHABILITY", "origin/dev moved")
    trees: dict[str, dict[bytes, tuple[str, str]]] = {}
    for label in LABELS:
        revision = data["revisions"][label]
        tree_oid = git_output(repo, ["git", "rev-parse", f"{revision}^{{tree}}"], "E_CENSUS_TREE").decode("ascii").strip()
        need(tree_oid == data["censuses"][label]["tree_oid"], "E_CENSUS_TREE", label)
        raw_tree = git_output(repo, ["git", "ls-tree", "-rz", "--full-tree", revision], "E_CENSUS_TREE")
        need(not raw_tree or raw_tree.endswith(b"\0"), "E_CENSUS_TREE", f"{label}: unterminated tree")
        tree: dict[bytes, tuple[str, str]] = {}
        for raw_row in raw_tree[:-1].split(b"\0") if raw_tree else []:
            meta, raw_path = raw_row.split(b"\t", 1)
            mode, kind, blob = meta.split(b" ", 2)
            if kind == b"blob":
                tree[raw_path] = (mode.decode("ascii"), blob.decode("ascii"))
        trees[label] = tree
        expected_python: set[str] = set()
        for raw_path in tree:
            if not raw_path.endswith(b".py"):
                continue
            try:
                expected_python.add(raw_path.decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise ManifestError(f"E_CENSUS_PATH: {label}: non-UTF-8 tracked Python path") from exc
        recorded_python = {row["path"] for row in data["censuses"][label]["entries"]}
        need(recorded_python == expected_python, "E_CENSUS_UNIVERSE", f"{label}: tracked Python set differs")
        for row in data["censuses"][label]["entries"] + data["censuses"][label]["configuration_inputs"]:
            raw_path = row["path"].encode("utf-8")
            need(tree.get(raw_path) == (row["mode"], row["blob_id"]), "E_CENSUS_BLOB", f"{label}:{row['path']}")
        refs_raw = git_output(
            repo,
            ["git", "for-each-ref", "--format=%(refname)", "--contains", revision, "refs/remotes/"],
            "E_REACHABILITY",
        )
        refs = sorted(set(refs_raw.decode("utf-8").splitlines()))
        need(refs == data["source_reachability"][label]["remote_tracking_refs"], "E_REACHABILITY", label)
    historical = data["commands"]["historical_diff"]
    historical_raw = git_output(repo, historical["argv"], "E_HISTORICAL_DIFF")
    need(digest(historical_raw) == historical["stdout_sha256"], "E_HISTORICAL_DIFF", "stdout digest")
    need(not historical_raw or historical_raw.endswith(b"\0"), "E_HISTORICAL_DIFF", "unterminated name-status output")
    tokens = historical_raw[:-1].split(b"\0") if historical_raw else []
    parsed: list[tuple[str, list[str]]] = []
    index = 0
    while index < len(tokens):
        try:
            status = tokens[index].decode("ascii")
        except UnicodeDecodeError as exc:
            raise ManifestError("E_HISTORICAL_DIFF: non-ASCII status") from exc
        index += 1
        if status in {"A", "D", "M"}:
            path_count = 1
        elif re.fullmatch(r"R[0-9]{3}", status) is not None and int(status[1:]) <= 100:
            path_count = 2
        else:
            raise ManifestError(f"E_HISTORICAL_DIFF: unsupported status {status!r}")
        need(index + path_count <= len(tokens), "E_HISTORICAL_DIFF", f"truncated {status} record")
        paths: list[str] = []
        for raw_path in tokens[index:index + path_count]:
            try:
                path = raw_path.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ManifestError(f"E_HISTORICAL_DIFF: {status}: non-UTF-8 path") from exc
            need(bool(path), "E_HISTORICAL_DIFF", f"{status}: empty path")
            paths.append(path)
        parsed.append((status, paths))
        index += path_count

    normalized: list[dict[str, Any]] = []
    for status, paths in parsed:
        if status == "A":
            change, base_path, pre_path = "add", None, paths[0]
        elif status == "D":
            change, base_path, pre_path = "delete", paths[0], None
        elif status == "M":
            change, base_path, pre_path = "modify", paths[0], paths[0]
        else:
            change, base_path, pre_path = "rename", paths[0], paths[1]
        base_blob = None
        if base_path is not None:
            raw_base_path = base_path.encode("utf-8")
            need(raw_base_path in trees["base"], "E_HISTORICAL_DIFF", f"base path absent: {base_path!r}")
            base_blob = trees["base"][raw_base_path][1]
        pre_blob = None
        if pre_path is not None:
            raw_pre_path = pre_path.encode("utf-8")
            need(raw_pre_path in trees["pre_closeout"], "E_HISTORICAL_DIFF", f"pre-closeout path absent: {pre_path!r}")
            pre_blob = trees["pre_closeout"][raw_pre_path][1]
        matches = [
            identity["id"]
            for identity in data["identities"]
            if (
                identity["paths"]["base"],
                identity["paths"]["pre_closeout"],
                identity["blobs"]["base"],
                identity["blobs"]["pre_closeout"],
            ) == (base_path, pre_path, base_blob, pre_blob)
        ]
        need(len(matches) == 1, "E_HISTORICAL_DIFF", f"{status}:{paths!r}: identity matches={matches!r}")
        normalized.append({
            "identity": matches[0],
            "change": change,
            "status": status,
            "base_path": base_path,
            "pre_closeout_path": pre_path,
            "base_blob": base_blob,
            "pre_closeout_blob": pre_blob,
        })
    normalized.sort(key=lambda row: (row["base_path"] or "", row["pre_closeout_path"] or ""))
    need(data["historical_diff"] == normalized, "E_HISTORICAL_DIFF", "stored rows differ from authentic diff")


def validate_census(data: dict[str, Any], label: str) -> tuple[dict[str, dict[str, Any]], set[str]]:
    census = exact(data["censuses"][label], {"schema_version", "label", "revision", "tree_oid", "toolchain", "scope", "command_template", "aggregate_command", "configuration_inputs", "entries", "blockers", "aggregate"}, "E_CENSUS_SCHEMA", label)
    need(census["schema_version"] == 1 and census["label"] == label, "E_CENSUS_SCHEMA", label)
    need(census["revision"] == data["revisions"][label], "E_CENSUS_REVISION", label)
    full_sha(census["tree_oid"], "E_CENSUS_TREE", label)
    toolchain = exact(census["toolchain"], {"python", "resolved_python", "ruff"}, "E_CENSUS_SCHEMA", f"{label}.toolchain")
    need(toolchain == {"python": data["tools"]["python_version"], "resolved_python": data["tools"]["resolved_python"], "ruff": data["tools"]["ruff_version"]}, "E_TOOLCHAIN", label)
    expected_scope = "all_tracked_dot_py"
    need(census["scope"] == expected_scope, "E_CENSUS_SCOPE", label)
    template = [data["tools"]["resolved_python"], "-m", "ruff", "format", "--check", "--force-exclude", "--no-cache", "./PATH_FROM_GIT"]
    need(census["command_template"] == template, "E_CENSUS_COMMAND", label)
    expected_aggregate = template[:-1] + ["."]
    need(census["aggregate_command"] == expected_aggregate, "E_CENSUS_COMMAND", f"{label}.aggregate")
    configs = census["configuration_inputs"]
    need(isinstance(configs, list), "E_CENSUS_SCHEMA", f"{label}.configuration_inputs")
    config_paths: list[str] = []
    for row in configs:
        row = exact(row, {"path", "mode", "blob_id"}, "E_CENSUS_SCHEMA", f"{label}.config")
        config_paths.append(text(row["path"], "E_CENSUS_PATH", label))
        need(re.fullmatch(r"[0-7]{6}", text(row["mode"], "E_CENSUS_SCHEMA", label)) is not None, "E_CENSUS_SCHEMA", label)
        full_sha(row["blob_id"], "E_CENSUS_BLOB", f"{label}:{row['path']}")
    need(config_paths == sorted(set(config_paths)), "E_CENSUS_ORDER", f"{label}.configuration_inputs")
    need(census["blockers"] == [], "E_CENSUS_BLOCKER", label)
    entries = census["entries"]
    need(isinstance(entries, list), "E_CENSUS_SCHEMA", f"{label}.entries")
    index: dict[str, dict[str, Any]] = {}
    failures: set[str] = set()
    for row in entries:
        row = exact(row, {"path", "mode", "blob_id", "result", "exit_code", "command_path"}, "E_CENSUS_SCHEMA", f"{label}.entry")
        path = text(row["path"], "E_CENSUS_PATH", label)
        need(path.endswith(".py") and "\x00" not in path, "E_CENSUS_PATH", f"{label}:{path!r}")
        need(path not in index, "E_CENSUS_DUPLICATE", f"{label}:{path!r}")
        need(re.fullmatch(r"[0-7]{6}", text(row["mode"], "E_CENSUS_SCHEMA", path)) is not None, "E_CENSUS_SCHEMA", path)
        full_sha(row["blob_id"], "E_CENSUS_BLOB", f"{label}:{path}")
        need((row["result"], row["exit_code"]) in {("not_failing", 0), ("would_reformat", 1)}, "E_CENSUS_RESULT", f"{label}:{path}")
        need(row["command_path"] == f"./{path}", "E_CENSUS_COMMAND", f"{label}:{path}")
        index[path] = row
        if row["result"] == "would_reformat":
            failures.add(path)
    need(list(index) == sorted(index), "E_CENSUS_ORDER", f"{label}.entries")
    aggregate = census["aggregate"]
    aggregate = exact(aggregate, {"exit_code", "stdout", "stderr"}, "E_CENSUS_CONTROL", label)
    need(aggregate["exit_code"] == (1 if failures else 0), "E_CENSUS_CONTROL", label)
    text(aggregate["stdout"], "E_CENSUS_CONTROL", label, nonempty=False)
    text(aggregate["stderr"], "E_CENSUS_CONTROL", label, nonempty=False)
    return index, failures


def transition_kind(kind: str, before: str | None, after: str | None, before_blob: str | None, after_blob: str | None) -> bool:
    if kind == "unchanged":
        return before is not None and before == after and before_blob == after_blob
    if kind == "modify":
        return before is not None and before == after and before_blob != after_blob
    if kind in {"rename", "copy"}:
        return before is not None and after is not None and before != after
    if kind == "add":
        return before is None and after is not None
    if kind == "delete":
        return before is not None and after is None
    if kind == "absent":
        return before is None and after is None
    return False


def validate_identity(identity: dict[str, Any], data: dict[str, Any], entries: dict[str, dict[str, dict[str, Any]]]) -> None:
    identity = exact(identity, {"id", "paths", "blobs", "lineage"}, "E_IDENTITY_SCHEMA", "identity")
    identity_id = text(identity["id"], "E_IDENTITY_SCHEMA", "identity.id")
    need(IDENTITY.fullmatch(identity_id) is not None, "E_IDENTITY_SCHEMA", identity_id)
    paths = exact(identity["paths"], set(LABELS), "E_PROJECTION_PATH", identity_id)
    blobs = exact(identity["blobs"], set(LABELS), "E_PROJECTION_BLOB", identity_id)
    for label in LABELS:
        path = paths[label]
        blob = blobs[label]
        need(path is None or isinstance(path, str), "E_PROJECTION_PATH", f"{identity_id}:{label}")
        if path is None:
            need(blob is None, "E_PROJECTION_BLOB", f"{identity_id}:{label}")
        else:
            need(path in entries[label], "E_PROJECTION_PATH", f"{identity_id}:{label}:{path}")
            need(blob == entries[label][path]["blob_id"], "E_PROJECTION_BLOB", f"{identity_id}:{label}:{path}")
    transitions = {
        "base_to_pre": ("base", "pre_closeout"),
        "pre_to_closeout": ("pre_closeout", "closeout"),
        "common_to_current": ("common", "current"),
    }
    lineage = exact(identity["lineage"], set(transitions), "E_LINEAGE_KEYS", identity_id)
    for name, (source_label, target_label) in transitions.items():
        row = exact(lineage[name], {"kind", "source_revision", "target_revision", "source_path", "target_path", "source_blob", "target_blob", "git_status", "follow_commits", "rationale"}, "E_LINEAGE_KEYS", f"{identity_id}:{name}")
        need(row["source_revision"] == data["revisions"][source_label] and row["target_revision"] == data["revisions"][target_label], "E_LINEAGE_REVISION", f"{identity_id}:{name}")
        need(row["source_path"] == paths[source_label] and row["target_path"] == paths[target_label], "E_LINEAGE_PATH", f"{identity_id}:{name}")
        need(row["source_blob"] == blobs[source_label] and row["target_blob"] == blobs[target_label], "E_LINEAGE_BLOB", f"{identity_id}:{name}")
        kind = text(row["kind"], "E_LINEAGE_KIND", f"{identity_id}:{name}")
        need(kind != "ambiguous" and transition_kind(kind, row["source_path"], row["target_path"], row["source_blob"], row["target_blob"]), "E_LINEAGE_KIND", f"{identity_id}:{name}:{kind}")
        text(row["git_status"], "E_LINEAGE_PROVENANCE", f"{identity_id}:{name}.git_status")
        commits = sorted_unique(row["follow_commits"], "E_LINEAGE_PROVENANCE", f"{identity_id}:{name}.follow_commits")
        for commit in commits:
            full_sha(commit, "E_LINEAGE_PROVENANCE", f"{identity_id}:{name}")
        text(row["rationale"], "E_LINEAGE_PROVENANCE", f"{identity_id}:{name}.rationale")


def validate_copy_splits(data: dict[str, Any], identities: dict[str, dict[str, Any]]) -> set[tuple[str, str, tuple[str, ...]]]:
    rows = data["copy_splits"]
    need(isinstance(rows, list), "E_COPY_SPLIT", "copy_splits")
    keys: list[tuple[str, str, tuple[str, ...]]] = []
    for row in rows:
        row = exact(row, {"source_label", "source_revision", "source_path", "source_blob", "target_label", "target_revision", "target_paths", "identities", "evidence"}, "E_COPY_SPLIT", "copy split")
        source_label = text(row["source_label"], "E_COPY_SPLIT", "source_label")
        target_label = text(row["target_label"], "E_COPY_SPLIT", "target_label")
        need(source_label in LABELS and target_label in LABELS, "E_COPY_SPLIT", "labels")
        need(row["source_revision"] == data["revisions"][source_label] and row["target_revision"] == data["revisions"][target_label], "E_COPY_SPLIT", "revisions")
        source_path = text(row["source_path"], "E_COPY_SPLIT", "source_path")
        full_sha(row["source_blob"], "E_COPY_SPLIT", "source_blob")
        target_paths = sorted_unique(row["target_paths"], "E_COPY_SPLIT", "target_paths")
        identity_ids = sorted_unique(row["identities"], "E_COPY_SPLIT", "identities")
        need(len(target_paths) == len(identity_ids) and len(identity_ids) > 1, "E_COPY_SPLIT", "cardinality")
        for identity_id, target_path in zip(identity_ids, target_paths):
            need(identity_id in identities, "E_COPY_SPLIT", identity_id)
            need(identities[identity_id]["paths"][source_label] == source_path, "E_COPY_SPLIT", identity_id)
            need(identities[identity_id]["paths"][target_label] == target_path, "E_COPY_SPLIT", identity_id)
        validate_evidence(row["evidence"], "E_COPY_SPLIT", source_path)
        keys.append((source_label, source_path, tuple(identity_ids)))
    need(keys == sorted(set(keys)), "E_COPY_SPLIT", "order/uniqueness")
    return set(keys)


def projection_owners(data: dict[str, Any], identities: dict[str, dict[str, Any]], allowed_splits: set[tuple[str, str, tuple[str, ...]]]) -> dict[str, dict[str, str]]:
    grouped: dict[str, dict[str, list[str]]] = {label: {} for label in LABELS}
    for identity_id, identity in identities.items():
        for label, path in identity["paths"].items():
            if path is not None:
                grouped[label].setdefault(path, []).append(identity_id)
    owners: dict[str, dict[str, str]] = {label: {} for label in LABELS}
    for label, by_path in grouped.items():
        for path, identity_ids in by_path.items():
            ids = tuple(sorted(identity_ids))
            if len(ids) > 1:
                need(label != "current", "E_CURRENT_OWNER", f"{path}:{ids}")
                need((label, path, ids) in allowed_splits, "E_PROJECTION_COLLISION", f"{label}:{path}:{ids}")
            owners[label][path] = ids[0]
    return owners


def validate_blockers(data: dict[str, Any]) -> set[str]:
    rows = data["blockers"]
    need(isinstance(rows, list), "E_BLOCKERS", "blockers")
    order: list[tuple[str, str, str]] = []
    paths: set[str] = set()
    for row in rows:
        row = exact(row, {"code", "path", "detail"}, "E_BLOCKERS", "blocker")
        code = text(row["code"], "E_BLOCKERS", "blocker.code")
        path = text(row["path"], "E_BLOCKERS", "blocker.path")
        detail = text(row["detail"], "E_BLOCKERS", "blocker.detail")
        need(re.fullmatch(r"E_[A-Z0-9_]+", code) is not None, "E_BLOCKERS", code)
        order.append((code, path, detail))
        paths.add(path)
    need(order == sorted(set(order)), "E_BLOCKERS", "order/uniqueness")
    return paths


def validate_batches(data: dict[str, Any], current_failures: set[str], blocker_paths: set[str]) -> dict[str, set[str]]:
    rows = data["batches"]
    need(isinstance(rows, list) and rows, "E_BATCH_SCHEMA", "batches")
    labels: list[str] = []
    seen: dict[str, str] = {}
    batches: dict[str, set[str]] = {}
    for row in rows:
        row = exact(row, {"label", "paths", "owner_basis", "test_surface", "conflict_basis"}, "E_BATCH_SCHEMA", "batch")
        label = text(row["label"], "E_BATCH_LABEL", "batch.label")
        need(re.fullmatch(r"ruff-[a-z0-9]+(?:-[a-z0-9]+)*", label) is not None, "E_BATCH_LABEL", label)
        paths = sorted_unique(row["paths"], "E_BATCH_ORDER", label)
        need(paths, "E_BATCH_SCHEMA", f"{label} empty")
        text(row["owner_basis"], "E_BATCH_SCHEMA", f"{label}.owner_basis")
        tests = sorted_unique(row["test_surface"], "E_BATCH_SCHEMA", f"{label}.test_surface")
        need(bool(tests), "E_BATCH_SCHEMA", f"{label}.test_surface is empty")
        conflicts = row["conflict_basis"]
        need(isinstance(conflicts, list) and conflicts, "E_BATCH_SCHEMA", f"{label}.conflict_basis")
        conflict_order: list[tuple[str, str, tuple[str, ...]]] = []
        for conflict in conflicts:
            conflict = exact(conflict, {"source", "reference", "paths"}, "E_BATCH_SCHEMA", f"{label}.conflict")
            source = text(conflict["source"], "E_BATCH_SCHEMA", f"{label}.conflict.source")
            need(source in {"open_pr", "worktree", "recent_history", "none"}, "E_BATCH_SCHEMA", source)
            reference = text(conflict["reference"], "E_BATCH_SCHEMA", f"{label}.conflict.reference")
            overlap = sorted_unique(conflict["paths"], "E_BATCH_SCHEMA", f"{label}.conflict.paths")
            need(set(overlap) <= set(paths), "E_BATCH_SCHEMA", f"{label}.conflict.paths")
            if source == "none":
                need(not overlap and reference.startswith("none-at-"), "E_BATCH_SCHEMA", f"{label}.conflict.none")
            conflict_order.append((source, reference, tuple(overlap)))
        need(conflict_order == sorted(set(conflict_order)), "E_BATCH_SCHEMA", f"{label}.conflicts")
        labels.append(label)
        batches[label] = set(paths)
        for path in paths:
            need(path not in seen, "E_BATCH_OVERLAP", f"{path}:{seen.get(path)}:{label}")
            seen[path] = label
    need(labels == sorted(set(labels)), "E_BATCH_ORDER", "batch labels")
    need(not blocker_paths & set(seen), "E_BLOCKER_BATCH", repr(sorted(blocker_paths & set(seen))))
    need(set(seen) == current_failures, "E_BATCH_UNION", f"missing={sorted(current_failures-set(seen))!r};extra={sorted(set(seen)-current_failures)!r}")
    final_label = text(data["final_batch_label"], "E_FINAL_LABEL", "final_batch_label")
    need(final_label in batches, "E_FINAL_LABEL", final_label)
    need(not data["blockers"], "E_BLOCKERS", "manifest has blockers")
    return batches


def parse_task(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        body = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestError(f"E_RECORD_UTF8: {path}") from exc
    need(body.startswith("---\n") and "\n---\n" in body[4:], "E_RECORD_ID", str(path))
    front = body.split("---\n", 2)[1]
    task_match = re.search(r"(?m)^id: TASK-([0-9]+)$", front)
    file_match = re.fullmatch(r"task-([0-9]+) - [^/\n]+\.md", path.name)
    need(task_match is not None and file_match is not None and task_match.group(1) == file_match.group(1), "E_RECORD_ID", str(path))
    need(re.search(r"(?m)^status: To Do$", front) is not None, "E_RECORD_FRONTMATTER", str(path))
    need("labels:\n  - maintenance\n  - formatting\n  - quality\n" in front, "E_RECORD_FRONTMATTER", f"{path}:labels")
    need(
        "references:\n"
        "  - Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md\n"
        "  - Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json\n"
        in front,
        "E_RECORD_FRONTMATTER",
        f"{path}:references",
    )
    deps = {int(v) for v in re.findall(r"(?m)^  - TASK-([0-9]+)$", front)}
    ac_match = re.search(r"(?s)<!-- AC:BEGIN -->(.*?)<!-- AC:END -->", body)
    need(ac_match is not None, "E_RECORD_CONTRACT", str(path))
    markers = set(re.findall(r"<!-- TASK-26000-CONTRACT: ([a-z-]+) -->", ac_match.group(1)))
    return {"raw": raw, "text": body, "id": int(task_match.group(1)), "dependencies": deps, "markers": markers}


def validate_records(data: dict[str, Any], repo: Path, batches: dict[str, set[str]]) -> None:
    records = data["cleanup_records"]
    need(isinstance(records, list), "E_RECORD_SCHEMA", "cleanup_records")
    labels: list[str] = []
    by_label: dict[str, dict[str, Any]] = {}
    parsed: dict[str, dict[str, Any]] = {}
    required = {"rebase-reconcile", "drift-reconciliation", "assigned-paths-only", "ast-type-comments", "comment-directives", "ruff-checks", "focused-tests", "governance", "no-handwritten-behavior"}
    for record in records:
        record = exact(record, {"label", "path", "task_id", "final", "dependencies", "paths_sha256", "task_sha256", "created_at", "updated_at"}, "E_RECORD_SCHEMA", "cleanup record")
        label = text(record["label"], "E_RECORD_SCHEMA", "record.label")
        need(label in batches, "E_RECORD_COUNT", label)
        path = text(record["path"], "E_RECORD_PATH", label)
        need(path.startswith("backlog/tasks/") and "\n" not in path and "\x00" not in path, "E_RECORD_PATH", path)
        need(isinstance(record["task_id"], int) and record["task_id"] > 26000, "E_RECORD_ID", label)
        need(record["final"] is (label == data["final_batch_label"]), "E_RECORD_SCHEMA", f"{label}.final")
        need(isinstance(record["dependencies"], list) and record["dependencies"] == sorted(set(record["dependencies"])) and all(isinstance(v, int) for v in record["dependencies"]), "E_RECORD_DEPENDENCIES", label)
        need(record["paths_sha256"] == paths_digest(sorted(batches[label])), "E_RECORD_DIGEST", label)
        sha256_value(record["task_sha256"], "E_RECORD_DIGEST", f"{label}.task")
        for key in ("created_at", "updated_at"):
            need(re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}", text(record[key], "E_RECORD_SCHEMA", f"{label}.{key}")) is not None, "E_RECORD_SCHEMA", f"{label}.{key}")
        task = parse_task(repo / path)
        need(task["id"] == record["task_id"], "E_RECORD_ID", label)
        need(digest(task["raw"]) == record["task_sha256"], "E_RECORD_DIGEST", label)
        body = task["text"]
        need(f"<!-- TASK-26000-BATCH: {label} -->" in body and f"<!-- TASK-26000-PATHS-SHA256: {record['paths_sha256']} -->" in body, "E_RECORD_MARKER", label)
        need(f"<!-- TASK-26000-FINAL: {'true' if record['final'] else 'false'} -->" in body, "E_RECORD_MARKER", f"{label}.final")
        need(f"created_date: '{record['created_at']}'" in body and f"updated_date: '{record['updated_at']}'" in body, "E_RECORD_FRONTMATTER", f"{label}.timestamps")
        need(required <= task["markers"], "E_RECORD_CONTRACT", f"{label}:{sorted(required-task['markers'])}")
        literals = ("type_comments=True", "TypeIgnore.lineno", "include_attributes=False", "# noqa", "# type: ignore", "# fmt: off", "# fmt: on", "deepest AST-node path", "significant-token position", "adjacent statement paths")
        need(all(value in body for value in literals), "E_RECORD_CONTRACT", label)
        labels.append(label)
        by_label[label] = record
        parsed[label] = task
    need(labels == sorted(set(labels)), "E_RECORD_COUNT", "record order/uniqueness")
    need(set(by_label) == set(batches), "E_RECORD_COUNT", "record labels")
    final_label = data["final_batch_label"]
    final_id = by_label[final_label]["task_id"]
    non_final_ids = {row["task_id"] for label, row in by_label.items() if label != final_label}
    need(final_id == max(row["task_id"] for row in by_label.values()), "E_FINAL_DEPENDENCIES", final_label)
    for label, record in by_label.items():
        expected = sorted({26000} | (non_final_ids if label == final_label else set()))
        need(record["dependencies"] == expected and parsed[label]["dependencies"] == set(expected), "E_FINAL_DEPENDENCIES" if label == final_label else "E_RECORD_DEPENDENCIES", label)
        need(all(dep < record["task_id"] for dep in expected), "E_RECORD_DEPENDENCIES", label)
    final = parsed[final_label]
    need("repository-zero-gate" in final["markers"] and "ruff format --check --force-exclude ." in final["text"], "E_FINAL_GATE", final_label)


def validate(data: dict[str, Any], phase: str, repo: Path | None) -> dict[str, int]:
    exact(data, TOP_KEYS, "E_TOP_LEVEL_SCHEMA", "manifest")
    need(data["schema_version"] == 1, "E_SCHEMA_VERSION", repr(data["schema_version"]))
    stamp = text(data["generated_at_utc"], "E_GENERATED_AT", "generated_at_utc")
    need(re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", stamp) is not None, "E_GENERATED_AT", stamp)
    try:
        dt.datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ManifestError(f"E_GENERATED_AT: {stamp}") from exc
    validate_tools(data)
    validate_revisions(data)
    need(exact(data["censuses"], set(LABELS), "E_CENSUS_SCHEMA", "censuses") is not None, "E_CENSUS_SCHEMA", "censuses")
    validate_reachability(data)
    if repo is not None and (repo / ".git").exists():
        cp = subprocess.run(("git", "merge-base", data["revisions"]["closeout"], data["revisions"]["current"]), cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        need(cp.returncode == 0 and cp.stdout.strip() == data["revisions"]["common"], "E_COMMON_ANCESTOR", cp.stderr.strip() or cp.stdout.strip())
    entries: dict[str, dict[str, dict[str, Any]]] = {}
    failures: dict[str, set[str]] = {}
    for label in LABELS:
        entries[label], failures[label] = validate_census(data, label)
    validate_commands(data)
    need(isinstance(data["identities"], list), "E_IDENTITY_SCHEMA", "identities")
    identities: dict[str, dict[str, Any]] = {}
    for identity in data["identities"]:
        validate_identity(identity, data, entries)
        need(identity["id"] not in identities, "E_IDENTITY_DUPLICATE", identity["id"])
        identities[identity["id"]] = identity
    need(list(identities) == sorted(identities), "E_IDENTITY_ORDER", "identities")
    if repo is not None and (repo / ".git").exists():
        validate_repo_provenance(data, repo)
    allowed_splits = validate_copy_splits(data, identities)
    owners = projection_owners(data, identities, allowed_splits)
    need(failures["common"] <= set(owners["common"]), "E_COMMON_UNMAPPED", repr(sorted(failures["common"] - set(owners["common"]))))
    need(failures["current"] <= set(owners["current"]), "E_CURRENT_UNMAPPED", repr(sorted(failures["current"] - set(owners["current"]))))
    rows = data["historical_diff"]
    need(isinstance(rows, list), "E_HISTORICAL_DIFF", "historical_diff")
    historical_ids: list[str] = []
    for row in rows:
        row = exact(row, {"identity", "change", "status", "base_path", "pre_closeout_path", "base_blob", "pre_closeout_blob"}, "E_HISTORICAL_DIFF", "historical row")
        identity_id = text(row["identity"], "E_HISTORICAL_DIFF", "identity")
        need(identity_id in identities, "E_HISTORICAL_DIFF", identity_id)
        transition = identities[identity_id]["lineage"]["base_to_pre"]
        need(row["change"] == transition["kind"] and row["status"] == transition["git_status"], "E_HISTORICAL_DIFF", identity_id)
        need((row["base_path"], row["pre_closeout_path"], row["base_blob"], row["pre_closeout_blob"]) == (transition["source_path"], transition["target_path"], transition["source_blob"], transition["target_blob"]), "E_HISTORICAL_DIFF", identity_id)
        historical_ids.append(identity_id)
    need(historical_ids == sorted(set(historical_ids)), "E_HISTORICAL_DIFF", "order/uniqueness")
    M = set(historical_ids)

    def project(identity_id: str, label: str) -> str | None:
        return identities[identity_id]["paths"][label]

    B = {identity_id for identity_id in M if project(identity_id, "base") in failures["base"]}
    C = {identity_id for identity_id in M if project(identity_id, "pre_closeout") in failures["pre_closeout"]}
    H = B & C
    stored_sets = exact(data["historical_sets"], {"M", "B", "C", "H"}, "E_HISTORICAL_SETS", "historical_sets")
    for name, derived in (("M", M), ("B", B), ("C", C), ("H", H)):
        values = sorted_unique(stored_sets[name], "E_HISTORICAL_SETS", name)
        need(set(values) == derived, "E_HISTORICAL_SETS", name)
    counts = (len(M), len(B), len(C), len(C - B), len(B - C), len(H))
    need(counts == (99, 64, 77, 16, 3, 61), "E_HISTORICAL_COUNTS", repr(counts))
    projected_M_closeout = {project(identity_id, "closeout") for identity_id in M} - {None}
    projected_H_closeout = {project(identity_id, "closeout") for identity_id in H} - {None}
    need(failures["closeout"] & projected_M_closeout == projected_H_closeout, "E_CLOSEOUT_INVARIANT", "projected sets differ")
    still = {project(identity_id, "current") for identity_id in H} & failures["current"]
    H_with_failing_current = {owners["current"][path] for path in still}
    resolved = H - H_with_failing_current
    projected_common = {project(identity_id, "current") for identity_id in identities if project(identity_id, "common") in failures["common"]} - {None}
    shared = (failures["current"] - still) & projected_common
    drift = failures["current"] - still - shared
    classifications = exact(data["classifications"], {"historical_still_current", "historical_no_longer_current", "shared_ancestor_debt", "current_line_drift"}, "E_CLASS_SCHEMA", "classifications")
    stored_still = set(sorted_unique(classifications["historical_still_current"], "E_CLASS_SCHEMA", "historical_still_current"))
    stored_shared = set(sorted_unique(classifications["shared_ancestor_debt"], "E_CLASS_SCHEMA", "shared_ancestor_debt"))
    drift_rows = classifications["current_line_drift"]
    need(isinstance(drift_rows, list), "E_CLASS_SCHEMA", "current_line_drift")
    drift_paths: list[str] = []
    for row in drift_rows:
        row = exact(row, {"path", "identity", "reason", "first_current_commit", "lineage_evidence"}, "E_CLASS_SCHEMA", "current_line_drift row")
        path = text(row["path"], "E_CLASS_DRIFT", "drift.path")
        identity_id = text(row["identity"], "E_CLASS_DRIFT", path)
        need(identity_id in identities and project(identity_id, "current") == path, "E_CLASS_DRIFT", path)
        common_path = project(identity_id, "common")
        expected_reason = "added_on_current" if common_path is None else "renamed_and_introduced_on_current" if common_path != path else "introduced_on_current"
        need(row["reason"] == expected_reason, "E_CLASS_DRIFT", path)
        full_sha(row["first_current_commit"], "E_CLASS_DRIFT", path)
        validate_evidence(row["lineage_evidence"], "E_CLASS_DRIFT", path)
        drift_paths.append(path)
    need(drift_paths == sorted(set(drift_paths)), "E_CLASS_DRIFT", "order/uniqueness")
    stored_drift = set(drift_paths)
    need(not stored_still & stored_shared and not stored_still & stored_drift and not stored_shared & stored_drift, "E_CLASS_OVERLAP", "stored categories overlap")
    need(stored_still == still, "E_CLASS_STILL", repr(sorted(still)))
    need(stored_shared == shared and stored_drift == drift, "E_CLASS_SHARED", f"shared={sorted(shared)!r};drift={sorted(drift)!r}")
    resolved_rows = classifications["historical_no_longer_current"]
    need(isinstance(resolved_rows, list), "E_CLASS_RESOLVED", "historical_no_longer_current")
    resolved_ids: list[str] = []
    for row in resolved_rows:
        row = exact(row, {"identity", "current_path", "reason", "lineage_evidence"}, "E_CLASS_RESOLVED", "resolved row")
        identity_id = text(row["identity"], "E_CLASS_RESOLVED", "resolved.identity")
        need(identity_id in identities and row["current_path"] == project(identity_id, "current"), "E_CLASS_RESOLVED", identity_id)
        need(row["reason"] in {"deleted", "formatted", "renamed", "configuration_changed"}, "E_CLASS_RESOLVED", identity_id)
        validate_evidence(row["lineage_evidence"], "E_CLASS_RESOLVED", identity_id)
        resolved_ids.append(identity_id)
    need(resolved_ids == sorted(set(resolved_ids)) and set(resolved_ids) == resolved, "E_CLASS_RESOLVED", repr(sorted(resolved)))
    blocker_paths = validate_blockers(data)
    batches = validate_batches(data, failures["current"], blocker_paths)
    expected_identity_universe = M | {owners["common"][path] for path in failures["common"]} | {owners["current"][path] for path in failures["current"]}
    need(set(identities) == expected_identity_universe, "E_IDENTITY_UNIVERSE", repr(sorted(set(identities) ^ expected_identity_universe)))
    if phase == "pre-records":
        need(data["cleanup_records"] == [], "E_RECORD_COUNT", "pre-records must be empty")
    else:
        need(phase == "final" and repo is not None, "E_ARGS", "final requires --repo")
        validate_records(data, repo, batches)
    return {"M": len(M), "B": len(B), "C": len(C), "H": len(H), "current_failures": len(failures["current"]), "batches": len(batches), "cleanup_records": len(data["cleanup_records"]), "blockers": len(data["blockers"])}


def fake_sha(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def make_entry(label: str, path: str, failing: bool) -> dict[str, Any]:
    return {"path": path, "mode": "100644", "blob_id": fake_sha(f"{label}:{path}"), "result": "would_reformat" if failing else "not_failing", "exit_code": 1 if failing else 0, "command_path": f"./{path}"}


def make_census(label: str, revision: str, entries: list[dict[str, Any]], tools: dict[str, str]) -> dict[str, Any]:
    template = [tools["resolved_python"], "-m", "ruff", "format", "--check", "--force-exclude", "--no-cache", "./PATH_FROM_GIT"]
    failures = [row for row in entries if row["result"] == "would_reformat"]
    return {"schema_version": 1, "label": label, "revision": revision, "tree_oid": fake_sha(f"tree:{label}"), "toolchain": {"python": tools["python_version"], "resolved_python": tools["resolved_python"], "ruff": tools["ruff_version"]}, "scope": "all_tracked_dot_py", "command_template": template, "aggregate_command": template[:-1] + ["."], "configuration_inputs": [{"path": "pyproject.toml", "mode": "100644", "blob_id": fake_sha(f"config:{label}")}], "entries": sorted(entries, key=lambda row: row["path"]), "blockers": [], "aggregate": {"exit_code": 1 if failures else 0, "stdout": "", "stderr": ""}}


def lineage_row(data: dict[str, Any], source: str, target: str, source_path: str | None, target_path: str | None, source_blob: str | None, target_blob: str | None) -> dict[str, Any]:
    if source_path is None and target_path is None:
        kind, status = "absent", "absent"
    elif source_path is None:
        kind, status = "add", "A"
    elif target_path is None:
        kind, status = "delete", "D"
    elif source_path != target_path:
        kind, status = "rename", "R100"
    elif source_blob == target_blob:
        kind, status = "unchanged", "same"
    else:
        kind, status = "modify", "M"
    return {"kind": kind, "source_revision": data["revisions"][source], "target_revision": data["revisions"][target], "source_path": source_path, "target_path": target_path, "source_blob": source_blob, "target_blob": target_blob, "git_status": status, "follow_commits": [data["revisions"][target]], "rationale": f"fixture {source} to {target}"}


AC_LINES = [
    "- [ ] After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->",
    "- [ ] Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->",
    "- [ ] Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->",
    "- [ ] Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and significant-token position, preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->",
    "- [ ] Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->",
    "- [ ] Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->",
    "- [ ] `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->",
    "- [ ] The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->",
]
FINAL_AC = "- [ ] After all lower-ID cleanup dependencies pass, the explicit Git-tracked repository-wide command exits zero under the recorded Python 3.12.11 interpreter: `python -m ruff format --check --force-exclude .`; any new unassigned failure blocks this gate and is not absorbed. <!-- TASK-26000-CONTRACT: repository-zero-gate -->"


def task_bytes(task_id: int, label: str, paths: list[str], dependencies: list[int], final: bool, *, drop_behavior: bool = False, drop_gate: bool = False) -> bytes:
    lines = ["---", f"id: TASK-{task_id}", f"title: Clean Ruff formatter debt for {label}", "status: To Do", "created_date: '2026-08-30 20:00'", "updated_date: '2026-08-30 20:00'", "labels:", "  - maintenance", "  - formatting", "  - quality", "dependencies:"]
    lines.extend(f"  - TASK-{value}" for value in dependencies)
    lines.extend(["references:", "  - Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md", "  - Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json", "---", "", f"<!-- TASK-26000-BATCH: {label} -->", f"<!-- TASK-26000-PATHS-SHA256: {paths_digest(paths)} -->", f"<!-- TASK-26000-FINAL: {'true' if final else 'false'} -->", "", "## Acceptance Criteria", "<!-- AC:BEGIN -->"])
    ac = [line for line in AC_LINES if not (drop_behavior and "no-handwritten-behavior" in line)]
    lines.extend(ac)
    if final and not drop_gate:
        lines.append(FINAL_AC)
    lines.extend(["<!-- AC:END -->", ""])
    return "\n".join(lines).encode("utf-8")


def fixture(repo: Path, with_records: bool) -> dict[str, Any]:
    tools = {"python_version": "3.12.11", "ruff_version": "ruff 0.15.22", "resolved_python": "/opt/python/3.12.11/bin/python"}
    revisions = {name: fake_sha(name) for name in (*LABELS, "task_base")}
    data: dict[str, Any] = {"schema_version": 1, "generated_at_utc": "2026-08-30T20:00:00Z", "tools": tools, "revisions": revisions, "commands": {}, "source_reachability": {label: {"object_present": True, "remote_tracking_refs": ["refs/remotes/origin/dev"] if label == "current" else []} for label in LABELS}, "censuses": {}, "identities": [], "historical_diff": [], "historical_sets": {}, "copy_splits": [], "classifications": {}, "blockers": [], "batches": [], "final_batch_label": "ruff-final-gate", "cleanup_records": []}
    entries: dict[str, list[dict[str, Any]]] = {label: [] for label in LABELS}
    for number in range(99):
        identity_id = f"I-{number:04d}"
        base_path = f"base_{number:03d}.py"
        pre_path = f"pre_{number:03d}.py" if number == 98 else base_path
        paths = {"base": base_path, "pre_closeout": pre_path, "closeout": pre_path, "common": f"common_{number:03d}.py", "current": f"current_{number:03d}.py"}
        failing = {"base": number < 64, "pre_closeout": number < 61 or 64 <= number < 80, "closeout": number < 61, "common": False, "current": number == 0}
        blobs: dict[str, str] = {}
        for label in LABELS:
            row = make_entry(label, paths[label], failing[label])
            entries[label].append(row)
            blobs[label] = row["blob_id"]
        identity = {"id": identity_id, "paths": paths, "blobs": blobs, "lineage": {}}
        data["identities"].append(identity)
    for number, path, common_fail, current_fail in ((99, "shared.py", True, True), (100, "current_only.py", False, True)):
        identity_id = f"I-{number:04d}"
        paths = {"base": None, "pre_closeout": None, "closeout": None, "common": path if number == 99 else None, "current": path}
        blobs: dict[str, str | None] = {label: None for label in LABELS}
        if paths["common"] is not None:
            row = make_entry("common", path, common_fail)
            entries["common"].append(row)
            blobs["common"] = row["blob_id"]
        row = make_entry("current", path, current_fail)
        entries["current"].append(row)
        blobs["current"] = row["blob_id"]
        data["identities"].append({"id": identity_id, "paths": paths, "blobs": blobs, "lineage": {}})
    for label in LABELS:
        data["censuses"][label] = make_census(label, revisions[label], entries[label], tools)
    for identity in data["identities"]:
        p, b = identity["paths"], identity["blobs"]
        identity["lineage"] = {
            "base_to_pre": lineage_row(data, "base", "pre_closeout", p["base"], p["pre_closeout"], b["base"], b["pre_closeout"]),
            "pre_to_closeout": lineage_row(data, "pre_closeout", "closeout", p["pre_closeout"], p["closeout"], b["pre_closeout"], b["closeout"]),
            "common_to_current": lineage_row(data, "common", "current", p["common"], p["current"], b["common"], b["current"]),
        }
    for identity in data["identities"][:99]:
        row = identity["lineage"]["base_to_pre"]
        data["historical_diff"].append({"identity": identity["id"], "change": row["kind"], "status": row["git_status"], "base_path": row["source_path"], "pre_closeout_path": row["target_path"], "base_blob": row["source_blob"], "pre_closeout_blob": row["target_blob"]})
    M = [f"I-{number:04d}" for number in range(99)]
    B = [f"I-{number:04d}" for number in range(64)]
    C = [f"I-{number:04d}" for number in list(range(61)) + list(range(64, 80))]
    H = [f"I-{number:04d}" for number in range(61)]
    data["historical_sets"] = {"M": M, "B": B, "C": C, "H": H}
    def evidence(summary: str) -> dict[str, Any]:
        return {"commits": [revisions["current"]], "summary": summary}
    data["classifications"] = {
        "historical_still_current": ["current_000.py"],
        "historical_no_longer_current": [{"identity": f"I-{number:04d}", "current_path": f"current_{number:03d}.py", "reason": "formatted", "lineage_evidence": evidence("fixture resolution")} for number in range(1, 61)],
        "shared_ancestor_debt": ["shared.py"],
        "current_line_drift": [{"path": "current_only.py", "identity": "I-0100", "reason": "added_on_current", "first_current_commit": revisions["current"], "lineage_evidence": evidence("fixture addition")}],
    }
    data["batches"] = [
        {"label": "ruff-core", "paths": ["current_000.py", "shared.py"], "owner_basis": "fixture core owner", "test_surface": ["Tests/core"], "conflict_basis": [{"source": "none", "reference": f"none-at-{revisions['current']}", "paths": []}]},
        {"label": "ruff-final-gate", "paths": ["current_only.py"], "owner_basis": "fixture final owner", "test_surface": ["Tests/final"], "conflict_basis": [{"source": "none", "reference": f"none-at-{revisions['current']}", "paths": []}]},
    ]
    census_commands: dict[str, Any] = {}
    for label in LABELS:
        argv = [tools["resolved_python"], "/tmp/task26000_ruff_census.py", "--checkout", f"/tmp/checkouts/{label}", "--revision", revisions[label], "--label", label, "--output", f"/tmp/raw/{label}.json"]
        census_commands[label] = {"argv": argv, "cwd": ".", "exit_code": 0, "output_sha256": digest(canonical_bytes(data["censuses"][label]))}
    data["commands"] = {
        "common_ancestor": {"argv": ["git", "merge-base", revisions["closeout"], revisions["current"]], "cwd": ".", "exit_code": 0, "stdout": revisions["common"] + "\n"},
        "historical_diff": {"argv": ["git", "diff", "--name-status", "-z", "-M", f"{revisions['base']}..{revisions['pre_closeout']}", "--", "*.py"], "cwd": ".", "exit_code": 0, "stdout_sha256": digest(b"fixture diff")},
        "censuses": census_commands,
    }
    if with_records:
        ids = {"ruff-core": 30000, "ruff-final-gate": 30001}
        for batch in data["batches"]:
            label = batch["label"]
            final = label == data["final_batch_label"]
            dependencies = [26000, 30000] if final else [26000]
            path = f"backlog/tasks/task-{ids[label]} - Clean Ruff formatter debt for {label}.md"
            raw = task_bytes(ids[label], label, batch["paths"], dependencies, final)
            target = repo / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            data["cleanup_records"].append({"label": label, "path": path, "task_id": ids[label], "final": final, "dependencies": dependencies, "paths_sha256": paths_digest(batch["paths"]), "task_sha256": digest(raw), "created_at": "2026-08-30 20:00", "updated_at": "2026-08-30 20:00"})
    return data


def git_fixture_output(repo: Path, *argv: str) -> bytes:
    cp = subprocess.run(("git", *argv), cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    need(cp.returncode == 0, "E_SELFTEST", cp.stderr.decode("utf-8", "backslashreplace").strip())
    return cp.stdout


def authentic_provenance_fixture(repo: Path) -> dict[str, Any]:
    repo.mkdir(parents=True)
    git_fixture_output(repo, "init", "-q")
    (repo / "pyproject.toml").write_text("[tool.ruff]\n", encoding="utf-8")
    (repo / "a_delete.py").write_text("DELETE = True\n", encoding="utf-8")
    (repo / "keep.py").write_text("KEEP = 1\n", encoding="utf-8")
    (repo / "rename_old.py").write_text("RENAMED = True\n", encoding="utf-8")
    git_fixture_output(repo, "add", "--", ".")
    git_fixture_output(repo, "-c", "user.name=Task 26000", "-c", "user.email=task26000@example.invalid", "commit", "-qm", "base")
    base = git_fixture_output(repo, "rev-parse", "HEAD^{commit}").decode("ascii").strip()
    (repo / "z_add.py").write_text("ADDED = True\n", encoding="utf-8")
    (repo / "a_delete.py").unlink()
    (repo / "keep.py").write_text("KEEP = 2\n", encoding="utf-8")
    (repo / "rename_old.py").rename(repo / "rename_new.py")
    git_fixture_output(repo, "add", "-A")
    git_fixture_output(repo, "-c", "user.name=Task 26000", "-c", "user.email=task26000@example.invalid", "commit", "-qm", "pre-closeout")
    pre = git_fixture_output(repo, "rev-parse", "HEAD^{commit}").decode("ascii").strip()
    git_fixture_output(repo, "update-ref", "refs/remotes/origin/dev", pre)
    revisions = {"task_base": base, "base": base, "pre_closeout": pre, "closeout": pre, "common": pre, "current": pre}
    tools = {"python_version": "3.12.11", "ruff_version": "ruff 0.15.22", "resolved_python": "/opt/python/3.12.11/bin/python"}

    def census(label: str) -> tuple[dict[str, Any], dict[str, tuple[str, str]]]:
        revision = revisions[label]
        raw = git_fixture_output(repo, "ls-tree", "-rz", "--full-tree", revision)
        tree: dict[str, tuple[str, str]] = {}
        for raw_row in raw[:-1].split(b"\0"):
            meta, raw_path = raw_row.split(b"\t", 1)
            mode, kind, blob = meta.split(b" ", 2)
            if kind == b"blob":
                tree[raw_path.decode("utf-8")] = (mode.decode("ascii"), blob.decode("ascii"))
        entries = [{"path": path, "mode": mode, "blob_id": blob} for path, (mode, blob) in sorted(tree.items()) if path.endswith(".py")]
        config_mode, config_blob = tree["pyproject.toml"]
        snapshot = {
            "tree_oid": git_fixture_output(repo, "rev-parse", f"{revision}^{{tree}}").decode("ascii").strip(),
            "entries": entries,
            "configuration_inputs": [{"path": "pyproject.toml", "mode": config_mode, "blob_id": config_blob}],
            "toolchain": {"python": tools["python_version"], "resolved_python": tools["resolved_python"], "ruff": tools["ruff_version"]},
        }
        return snapshot, tree

    censuses: dict[str, Any] = {}
    trees: dict[str, dict[str, tuple[str, str]]] = {}
    for label in LABELS:
        censuses[label], trees[label] = census(label)
    specs = [
        ("I-0000", "add", "A", None, "z_add.py"),
        ("I-0001", "delete", "D", "a_delete.py", None),
        ("I-0002", "modify", "M", "keep.py", "keep.py"),
        ("I-0003", "rename", "R100", "rename_old.py", "rename_new.py"),
    ]
    identities: list[dict[str, Any]] = []
    historical_rows: list[dict[str, Any]] = []
    for identity_id, change, status, base_path, pre_path in specs:
        base_blob = trees["base"][base_path][1] if base_path is not None else None
        pre_blob = trees["pre_closeout"][pre_path][1] if pre_path is not None else None
        identities.append({"id": identity_id, "paths": {"base": base_path, "pre_closeout": pre_path}, "blobs": {"base": base_blob, "pre_closeout": pre_blob}})
        historical_rows.append({"identity": identity_id, "change": change, "status": status, "base_path": base_path, "pre_closeout_path": pre_path, "base_blob": base_blob, "pre_closeout_blob": pre_blob})
    historical_argv = ["git", "diff", "--name-status", "-z", "-M", f"{base}..{pre}", "--", "*.py"]
    historical_raw = git_fixture_output(repo, *historical_argv[1:])
    return {
        "tools": tools,
        "revisions": revisions,
        "censuses": censuses,
        "source_reachability": {label: {"remote_tracking_refs": ["refs/remotes/origin/dev"]} for label in LABELS},
        "commands": {"historical_diff": {"argv": historical_argv, "stdout_sha256": digest(historical_raw)}},
        "identities": identities,
        "historical_diff": historical_rows,
    }


def refresh_census_digest(data: dict[str, Any], label: str) -> None:
    data["commands"]["censuses"][label]["output_sha256"] = digest(canonical_bytes(data["censuses"][label]))


def expect(code: str, callback: Callable[[], Any]) -> None:
    try:
        callback()
    except ManifestError as exc:
        need(str(exc).startswith(f"{code}:"), "E_SELFTEST", f"expected {code}; got {exc}")
        return
    raise ManifestError(f"E_SELFTEST: expected {code}")


def run_self_tests() -> None:
    with tempfile.TemporaryDirectory(prefix="task26000-manifest-") as temp_value:
        root = Path(temp_value)
        pre = fixture(root / "pre", False)
        final_root = root / "final"
        final = fixture(final_root, True)
        validate(copy.deepcopy(pre), "pre-records", None)
        validate(copy.deepcopy(final), "final", final_root)
        provenance_root = root / "provenance"
        provenance = authentic_provenance_fixture(provenance_root)
        validate_repo_provenance(copy.deepcopy(provenance), provenance_root)

        def altered_historical_row(data: dict[str, Any]) -> None:
            data["historical_diff"][2]["pre_closeout_blob"] = data["historical_diff"][2]["base_blob"]

        corrupt_provenance = copy.deepcopy(provenance)
        altered_historical_row(corrupt_provenance)
        expect("E_HISTORICAL_DIFF", lambda: validate_repo_provenance(corrupt_provenance, provenance_root))

        mutations: list[tuple[str, str, Callable[[dict[str, Any], Path], None], str]] = []

        def missing_current(data: dict[str, Any], _repo: Path) -> None:
            row = next(row for row in data["censuses"]["current"]["entries"] if row["path"] == "current_only.py")
            row.update(result="not_failing", exit_code=0)
            data["classifications"]["current_line_drift"] = []
            refresh_census_digest(data, "current")

        def duplicate_batch(data: dict[str, Any], _repo: Path) -> None:
            data["batches"][1]["paths"].append("shared.py")
            data["batches"][1]["paths"].sort()

        def blocker_batch(data: dict[str, Any], _repo: Path) -> None:
            data["blockers"] = [{"code": "E_FIXTURE", "path": "shared.py", "detail": "fixture blocker"}]

        def missing_lineage(data: dict[str, Any], _repo: Path) -> None:
            del data["identities"][98]["lineage"]["base_to_pre"]

        def category_swap(data: dict[str, Any], _repo: Path) -> None:
            data["classifications"]["shared_ancestor_debt"] = ["current_only.py"]
            data["classifications"]["current_line_drift"] = [{"path": "shared.py", "identity": "I-0099", "reason": "introduced_on_current", "first_current_commit": data["revisions"]["current"], "lineage_evidence": {"commits": [data["revisions"]["current"]], "summary": "swapped"}}]

        def omitted_resolved(data: dict[str, Any], _repo: Path) -> None:
            data["classifications"]["historical_no_longer_current"].pop()

        def duplicate_owner(data: dict[str, Any], _repo: Path) -> None:
            data["identities"][100]["paths"]["current"] = "shared.py"
            data["identities"][100]["blobs"]["current"] = next(row["blob_id"] for row in data["censuses"]["current"]["entries"] if row["path"] == "shared.py")
            row = data["identities"][100]["lineage"]["common_to_current"]
            row["target_path"] = "shared.py"
            row["target_blob"] = data["identities"][100]["blobs"]["current"]

        def wrong_closeout(data: dict[str, Any], _repo: Path) -> None:
            row = next(row for row in data["censuses"]["closeout"]["entries"] if row["path"] == "base_061.py")
            row.update(result="would_reformat", exit_code=1)
            refresh_census_digest(data, "closeout")

        def wrong_cardinality(data: dict[str, Any], _repo: Path) -> None:
            row = next(row for row in data["censuses"]["base"]["entries"] if row["path"] == "base_063.py")
            row.update(result="not_failing", exit_code=0)
            data["historical_sets"]["B"].remove("I-0063")
            refresh_census_digest(data, "base")

        def overlap(data: dict[str, Any], _repo: Path) -> None:
            data["classifications"]["shared_ancestor_debt"].append("current_000.py")
            data["classifications"]["shared_ancestor_debt"].sort()

        def absent_record(data: dict[str, Any], _repo: Path) -> None:
            data["cleanup_records"].pop(0)

        def missing_behavior(data: dict[str, Any], repo: Path) -> None:
            record = data["cleanup_records"][0]
            raw = task_bytes(record["task_id"], record["label"], data["batches"][0]["paths"], record["dependencies"], False, drop_behavior=True)
            (repo / record["path"]).write_bytes(raw)
            record["task_sha256"] = digest(raw)

        def missing_gate(data: dict[str, Any], repo: Path) -> None:
            record = data["cleanup_records"][1]
            raw = task_bytes(record["task_id"], record["label"], data["batches"][1]["paths"], record["dependencies"], True, drop_gate=True)
            (repo / record["path"]).write_bytes(raw)
            record["task_sha256"] = digest(raw)

        mutations.extend([
            ("missing-current-failure", "E_BATCH_UNION", missing_current, "pre-records"),
            ("duplicate-batch-path", "E_BATCH_OVERLAP", duplicate_batch, "pre-records"),
            ("blocker-in-batch", "E_BLOCKER_BATCH", blocker_batch, "pre-records"),
            ("missing-rename-lineage", "E_LINEAGE_KEYS", missing_lineage, "pre-records"),
            ("category-swap", "E_CLASS_SHARED", category_swap, "pre-records"),
            ("omitted-resolved", "E_CLASS_RESOLVED", omitted_resolved, "pre-records"),
            ("duplicate-current-owner", "E_CURRENT_OWNER", duplicate_owner, "pre-records"),
            ("wrong-closeout-projection", "E_CLOSEOUT_INVARIANT", wrong_closeout, "pre-records"),
            ("wrong-historical-cardinality", "E_HISTORICAL_COUNTS", wrong_cardinality, "pre-records"),
            ("overlapping-comparison", "E_CLASS_OVERLAP", overlap, "pre-records"),
            ("absent-cleanup-record", "E_RECORD_COUNT", absent_record, "final"),
            ("missing-behavior-contract", "E_RECORD_CONTRACT", missing_behavior, "final"),
            ("missing-final-gate", "E_FINAL_GATE", missing_gate, "final"),
        ])
        for name, code, mutate, phase in mutations:
            case_root = root / name
            case = fixture(case_root, phase == "final")
            mutate(case, case_root)

            def validate_case(
                case: dict[str, Any] = case,
                phase: str = phase,
                case_root: Path = case_root,
            ) -> None:
                validate(case, phase, case_root if phase == "final" else None)

            expect(code, validate_case)
    print("manifest self-tests: 2 positive phases and 14 deterministic mutations passed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pre-records", "final"))
    parser.add_argument("--manifest")
    parser.add_argument("--repo")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        need(args.phase is None and args.manifest is None and args.repo is None, "E_ARGS", "--self-test is exclusive")
        run_self_tests()
        return 0
    need(args.phase is not None and args.manifest is not None, "E_ARGS", "--phase and --manifest are required")
    need(args.phase != "final" or args.repo is not None, "E_ARGS", "final requires --repo")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    counts = validate(manifest, args.phase, Path(args.repo).resolve() if args.repo else None)
    print(json.dumps(counts, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ManifestError, KeyError, IndexError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
```

The self-test's deterministic first failures are:

```text
missing-current-failure         E_BATCH_UNION
duplicate-batch-path            E_BATCH_OVERLAP
blocker-in-batch                E_BLOCKER_BATCH
missing-rename-lineage          E_LINEAGE_KEYS
category-swap                   E_CLASS_SHARED
omitted-resolved                E_CLASS_RESOLVED
duplicate-current-owner         E_CURRENT_OWNER
wrong-closeout-projection       E_CLOSEOUT_INVARIANT
wrong-historical-cardinality    E_HISTORICAL_COUNTS
altered-historical-row          E_HISTORICAL_DIFF
overlapping-comparison          E_CLASS_OVERLAP
absent-cleanup-record           E_RECORD_COUNT
missing-behavior-contract       E_RECORD_CONTRACT
missing-final-gate              E_FINAL_GATE
```

It prints exactly
`manifest self-tests: 2 positive phases and 14 deterministic mutations passed`
only after both positive phases and all mutations pass.

---

## Appendix C: Exact Collision-Safe Task-ID Scanner

Task 5 materializes this temporary scanner. It accepts `--manifest`, `--output`,
optional `--expect-map`, and fixture-only `--self-test`; it reads batch labels plus
`final_batch_label` from the
manifest, writes canonical audit JSON, and exits 2 on a moved PR head, malformed task
identity, inaccessible checkout/ref, self-claim mismatch, external ID collision, or
changed allocation.

```python
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

FILE_ID = re.compile(r"^task-(\d+)(?:\.\d+)* - .+\.md$", re.IGNORECASE)
FRONT_ID = re.compile(r"(?m)^id:[ \t]*TASK-(\d+)(?:\.\d+)*[ \t]*$")
BUCKETS = (
    "backlog/tasks/",
    "backlog/completed/",
    "backlog/archive/tasks/",
    "backlog/drafts/",
)


class AllocationError(RuntimeError):
    pass


@dataclass(frozen=True, order=True)
class ClaimIdentity:
    path: str
    batch_label: str | None
    content_sha256: str


@dataclass(frozen=True)
class ParsedClaim:
    task_id: int
    identity: ClaimIdentity


def fail(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise AllocationError(f"{code}: {detail}")


def execute(argv: tuple[str, ...], repo: Path, code: str) -> bytes:
    try:
        cp = subprocess.run(
            argv,
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise AllocationError(f"{code}: exec {argv[0]!r}: {type(exc).__name__}") from exc
    if cp.returncode:
        diagnostic = cp.stderr.decode("utf-8", "backslashreplace").strip()
        raise AllocationError(f"{code}: exit {cp.returncode}: {diagnostic}")
    return cp.stdout


def read_json(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AllocationError(f"{code}: {path}: {type(exc).__name__}") from exc
    fail(isinstance(value, dict), code, "root is not an object")
    return value


def parse_claim(path: str, raw: bytes, source: str) -> ParsedClaim | None:
    filename = FILE_ID.fullmatch(PurePosixPath(path).name)
    text = raw.decode("utf-8")
    front = ""
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        fail(end >= 0, "E_TASK_IDENTITY", f"{source}:{path}: unterminated frontmatter")
        front = text[4:end]
    front_ids = FRONT_ID.findall(front)
    if filename is None and not front_ids:
        return None
    fail(
        filename is not None and len(front_ids) == 1 and filename.group(1) == front_ids[0],
        "E_TASK_IDENTITY",
        f"{source}:{path}",
    )
    batch_markers = re.findall(
        r"(?m)^<!-- TASK-26000-BATCH: ([a-z0-9]+(?:-[a-z0-9]+)*) -->$",
        text,
    )
    fail(len(batch_markers) <= 1, "E_TASK_IDENTITY", f"{source}:{path}: batch markers")
    return ParsedClaim(
        task_id=int(front_ids[0]),
        identity=ClaimIdentity(
            path=path,
            batch_label=batch_markers[0] if batch_markers else None,
            content_sha256=hashlib.sha256(raw).hexdigest(),
        ),
    )


def claim(
    path: str,
    raw: bytes,
    source: str,
    claims: dict[int, dict[ClaimIdentity, set[str]]],
) -> None:
    parsed = parse_claim(path, raw, source)
    if parsed is None:
        return
    claims.setdefault(parsed.task_id, {}).setdefault(parsed.identity, set()).add(source)


def scan_archive(
    repo: Path,
    revision: str,
    source: str,
    claims: dict[int, dict[ClaimIdentity, set[str]]],
) -> None:
    raw_archive = execute(
        ("git", "archive", "--format=tar", revision, "--", "backlog"),
        repo,
        "E_ARCHIVE",
    )
    with tarfile.open(fileobj=io.BytesIO(raw_archive), mode="r:") as archive:
        for member in archive:
            if not member.isfile() or not member.name.startswith(BUCKETS):
                continue
            handle = archive.extractfile(member)
            if handle is not None:
                claim(member.name, handle.read(), source, claims)


def refs(repo: Path) -> list[tuple[str, str]]:
    raw = execute(
        (
            "git",
            "for-each-ref",
            "--format=%(refname) %(objectname)",
            "refs/heads/",
            "refs/remotes/origin/",
        ),
        repo,
        "E_REFS",
    )
    result = [tuple(line.decode("ascii").split(" ", 1)) for line in raw.splitlines()]
    fail(all(len(row) == 2 and re.fullmatch(r"[0-9a-f]{40}", row[1]) for row in result), "E_REFS", "malformed ref row")
    return result


def open_prs(repo: Path) -> list[dict[str, Any]]:
    raw = execute(
        (
            "gh",
            "api",
            "--paginate",
            "--slurp",
            "repos/{owner}/{repo}/pulls?state=open&per_page=100",
        ),
        repo,
        "E_PR_LIST",
    )
    pages = json.loads(raw)
    fail(isinstance(pages, list) and all(isinstance(page, list) for page in pages), "E_PR_LIST", "pagination shape")
    result: list[dict[str, Any]] = []
    for page in pages:
        for row in page:
            number = row["number"]
            head_oid = row["head"]["sha"]
            fail(isinstance(number, int) and number > 0, "E_PR_LIST", "invalid PR number")
            fail(isinstance(head_oid, str) and re.fullmatch(r"[0-9a-f]{40}", head_oid) is not None, "E_PR_LIST", f"PR {number} head")
            result.append({"number": number, "head_oid": head_oid})
    return sorted(result, key=lambda row: row["number"])


def worktree_paths(repo: Path) -> list[bytes]:
    raw = execute(("git", "worktree", "list", "--porcelain", "-z"), repo, "E_WORKTREES")
    return [field[len(b"worktree ") :] for field in raw.split(b"\0") if field.startswith(b"worktree ")]


def scan_worktree_files(
    root: Path,
    source: str,
    claims: dict[int, dict[ClaimIdentity, set[str]]],
) -> None:
    for bucket in BUCKETS:
        directory = root / bucket
        if not directory.exists():
            continue
        walk_error: list[OSError] = []
        for current, _directories, files in os.walk(
            directory,
            onerror=walk_error.append,
        ):
            for filename in files:
                path = Path(current) / filename
                try:
                    raw = path.read_bytes()
                except OSError as exc:
                    raise AllocationError(
                        f"E_WORKTREE_READ: {root}:{path}: {type(exc).__name__}"
                    ) from exc
                claim(path.relative_to(root).as_posix(), raw, source, claims)
        fail(
            not walk_error,
            "E_WORKTREE_READ",
            f"{root}:{type(walk_error[0]).__name__}" if walk_error else "",
        )


def scan_worktree(
    raw_root: bytes,
    claims: dict[int, dict[ClaimIdentity, set[str]]],
    audit: list[dict[str, Any]],
) -> None:
    root = Path(os.fsdecode(raw_root))
    head = execute(("git", "rev-parse", "HEAD^{commit}"), root, "E_WORKTREE_HEAD").decode("ascii").strip()
    status = execute(
        ("git", "status", "--porcelain=v1", "-z", "--untracked-files=all"),
        root,
        "E_WORKTREE_STATUS",
    )
    audit.append(
        {
            "path_b64": base64.b64encode(raw_root).decode("ascii"),
            "head": head,
            "dirty": bool(status),
        }
    )
    scan_worktree_files(root, f"worktree:{root}", claims)


def planned_self_claims(
    manifest: dict[str, Any],
    repo: Path,
    expected_map: dict[str, int],
) -> dict[int, ClaimIdentity]:
    records = manifest["cleanup_records"]
    fail(
        len(records) in {0, len(expected_map)},
        "E_SELF_CLAIMS",
        "cleanup records must be empty or complete",
    )
    if not records:
        return {}
    result: dict[int, ClaimIdentity] = {}
    for row in records:
        fail(
            set(row) == {
                "label",
                "path",
                "task_id",
                "final",
                "dependencies",
                "paths_sha256",
                "task_sha256",
                "created_at",
                "updated_at",
            },
            "E_SELF_CLAIMS",
            repr(row),
        )
        label = row["label"]
        task_id = row["task_id"]
        fail(expected_map.get(label) == task_id, "E_SELF_CLAIMS", label)
        rel = PurePosixPath(row["path"])
        fail(rel.parts[:2] == ("backlog", "tasks") and len(rel.parts) == 3, "E_SELF_CLAIMS", row["path"])
        try:
            raw = (repo / Path(*rel.parts)).read_bytes()
        except OSError as exc:
            raise AllocationError(f"E_SELF_CLAIMS: {row['path']}: {type(exc).__name__}") from exc
        fail(
            re.fullmatch(r"[0-9a-f]{64}", row["task_sha256"]) is not None
            and hashlib.sha256(raw).hexdigest() == row["task_sha256"],
            "E_SELF_CLAIMS",
            row["path"],
        )
        parsed = parse_claim(row["path"], raw, "self")
        fail(
            parsed is not None
            and parsed.task_id == task_id
            and parsed.identity.batch_label == label,
            "E_SELF_CLAIMS",
            row["path"],
        )
        fail(task_id not in result, "E_SELF_CLAIMS", f"duplicate TASK-{task_id}")
        result[task_id] = parsed.identity
    fail(set(expected_map.values()) == set(result), "E_SELF_CLAIMS", "allocated ID set differs")
    return result


def classify_claims(
    claims: dict[int, dict[ClaimIdentity, set[str]]],
    self_claims: dict[int, ClaimIdentity],
    expected_map: dict[str, int] | None,
) -> tuple[set[int], dict[str, list[dict[str, Any]]]]:
    external_ids: set[int] = set()
    claim_audit: dict[str, list[dict[str, Any]]] = {}
    expected_ids = set(expected_map.values()) if expected_map is not None else set()
    for task_id, identities in sorted(claims.items()):
        allowed = self_claims.get(task_id)
        unexpected = set(identities) - ({allowed} if allowed is not None else set())
        if task_id in expected_ids:
            fail(not unexpected, "E_ID_COLLISION", f"TASK-{task_id}:{sorted(unexpected)!r}")
        if unexpected or allowed is None:
            external_ids.add(task_id)
        claim_audit[str(task_id)] = [
            {
                "path": identity.path,
                "batch_label": identity.batch_label,
                "content_sha256": identity.content_sha256,
                "sources": sorted(identities[identity]),
                "accepted_self": identity == allowed,
            }
            for identity in sorted(identities)
        ]
    return external_ids, claim_audit


def allocate_ids(
    labels: list[str], final_label: str, external_ids: set[int]
) -> dict[str, int]:
    start = max(external_ids, default=0) + 100
    non_final = sorted(label for label in labels if label != final_label)
    return {
        label: task_id
        for label, task_id in zip(non_final + [final_label], range(start, start + len(labels)))
    }


def verify_pr_head(number: int, expected: str, actual: str) -> None:
    if actual != expected:
        raise AllocationError(
            f"E_PR_HEAD_MOVED: PR {number} expected {expected}; fetched {actual}"
        )


def write_audit(output: Path, audit: dict[str, Any]) -> None:
    fail(output.parent.is_dir(), "E_OUTPUT", "output parent does not exist")
    try:
        with output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    except OSError as exc:
        raise AllocationError(f"E_OUTPUT: {output}: {type(exc).__name__}") from exc


def scan(
    repo: Path,
    labels: list[str],
    final_label: str,
    expected_map: dict[str, int] | None,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    if final_label not in labels or len(labels) != len(set(labels)):
        raise AllocationError("E_LABELS: final missing or labels duplicate")
    execute(
        (
            "git",
            "fetch",
            "--prune",
            "--no-tags",
            "origin",
            "+refs/heads/*:refs/remotes/origin/*",
        ),
        repo,
        "E_FETCH",
    )
    origin_dev = execute(
        ("git", "rev-parse", "refs/remotes/origin/dev^{commit}"),
        repo,
        "E_ORIGIN_DEV",
    ).decode("ascii").strip()
    fail(
        origin_dev == manifest["revisions"]["current"],
        "E_ORIGIN_DEV",
        "origin/dev differs from manifest revisions.current",
    )
    claims: dict[int, dict[ClaimIdentity, set[str]]] = {}
    ref_audit: list[dict[str, str]] = []
    for name, oid in refs(repo):
        scan_archive(repo, oid, name, claims)
        ref_audit.append({"ref": name, "oid": oid})
    pr_audit = open_prs(repo)
    for row in pr_audit:
        execute(
            ("git", "fetch", "--no-tags", "origin", f"refs/pull/{row['number']}/head"),
            repo,
            "E_PR_FETCH",
        )
        actual = execute(
            ("git", "rev-parse", "FETCH_HEAD^{commit}"),
            repo,
            "E_PR_FETCH_HEAD",
        ).decode("ascii").strip()
        verify_pr_head(row["number"], row["head_oid"], actual)
        scan_archive(repo, actual, f"pr:{row['number']}", claims)
    worktree_audit: list[dict[str, Any]] = []
    for raw_root in worktree_paths(repo):
        scan_worktree(raw_root, claims, worktree_audit)
    self_claims = planned_self_claims(manifest, repo, expected_map) if expected_map is not None else {}
    external_ids, claim_audit = classify_claims(claims, self_claims, expected_map)
    allocation = allocate_ids(labels, final_label, external_ids)
    return {
        "origin_dev": origin_dev,
        "refs": ref_audit,
        "open_prs": pr_audit,
        "worktrees": worktree_audit,
        "claims": claim_audit,
        "external_used_ids": sorted(external_ids),
        "allocation": allocation,
    }


def expect_error(code: str, operation: Any) -> None:
    try:
        operation()
    except AllocationError as exc:
        fail(str(exc).startswith(f"{code}:"), "E_SELF_TEST", str(exc))
        return
    raise AllocationError(f"E_SELF_TEST: expected {code}")


def task_bytes(task_id: int, label: str, body: str = "fixture") -> bytes:
    return (
        f"---\nid: TASK-{task_id}\n---\n"
        f"<!-- TASK-26000-BATCH: {label} -->\n{body}\n"
    ).encode("utf-8")


def run_self_tests() -> None:
    cases = 0
    expected = {"alpha": 26100}
    self_path = "backlog/tasks/task-26100 - alpha.md"
    self_raw = task_bytes(26100, "alpha", "self")
    self_claim = parse_claim(self_path, self_raw, "self")
    fail(self_claim is not None, "E_SELF_TEST", "self claim")

    collision_claims: dict[int, dict[ClaimIdentity, set[str]]] = {}
    claim(self_path, self_raw, "self", collision_claims)
    claim(
        "backlog/tasks/task-26100 - conflicting.md",
        task_bytes(26100, "alpha", "conflict"),
        "fixture:remote",
        collision_claims,
    )
    expect_error(
        "E_ID_COLLISION",
        lambda: classify_claims(collision_claims, {26100: self_claim.identity}, expected),
    )
    cases += 1

    accepted_claims: dict[int, dict[ClaimIdentity, set[str]]] = {}
    claim(self_path, self_raw, "self", accepted_claims)
    claim(self_path, self_raw, "fixture:exact-copy", accepted_claims)
    external, audit = classify_claims(
        accepted_claims, {26100: self_claim.identity}, expected
    )
    fail(external == set() and audit["26100"][0]["accepted_self"], "E_SELF_TEST", "self copy")
    cases += 1

    expect_error(
        "E_TASK_IDENTITY",
        lambda: parse_claim(
            "backlog/tasks/task-26101 - mismatch.md",
            task_bytes(26102, "alpha"),
            "fixture:mismatch",
        ),
    )
    cases += 1

    verify_pr_head(7, "a" * 40, "a" * 40)
    # A PR that moves after metadata collection must fail closed.
    expect_error("E_PR_HEAD_MOVED", lambda: verify_pr_head(7, "a" * 40, "b" * 40))
    cases += 1
    # A changed replacement head is also rejected rather than silently rescanned.
    expect_error("E_PR_HEAD_MOVED", lambda: verify_pr_head(7, "a" * 40, "c" * 40))
    cases += 1

    with tempfile.TemporaryDirectory() as raw_root:
        root = Path(raw_root)
        fixture_path = root / "backlog/tasks/task-26103 - worktree.md"
        fixture_path.parent.mkdir(parents=True)
        fixture_path.write_bytes(task_bytes(26103, "alpha"))
        worktree_claims: dict[int, dict[ClaimIdentity, set[str]]] = {}
        scan_worktree_files(root, "worktree:fixture", worktree_claims)
        fail(26103 in worktree_claims, "E_SELF_TEST", "worktree claim missing")
    cases += 1

    allocation = allocate_ids(["zeta", "alpha", "final"], "final", {26150})
    fail(
        allocation == {"alpha": 26250, "zeta": 26251, "final": 26252}
        and allocation["final"] == max(allocation.values()),
        "E_SELF_TEST",
        "allocation order",
    )
    cases += 1

    with tempfile.TemporaryDirectory() as raw_root:
        output = Path(raw_root) / "allocation.json"
        write_audit(output, {"fixture": True})
        expect_error("E_OUTPUT", lambda: write_audit(output, {"fixture": True}))
    cases += 1

    fail(cases == 8, "E_SELF_TEST", f"case count {cases}")
    print("allocation scanner self-tests: 8 cases passed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo")
    parser.add_argument("--manifest")
    parser.add_argument("--output")
    parser.add_argument("--expect-map")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        fail(
            all(value is None for value in (args.repo, args.manifest, args.output, args.expect_map)),
            "E_ARGS",
            "--self-test is exclusive",
        )
        run_self_tests()
        return 0
    fail(args.repo is not None and args.manifest is not None and args.output is not None, "E_ARGS", "--repo, --manifest, and --output are required")
    repo = Path(args.repo).resolve()
    manifest = read_json(Path(args.manifest), "E_MANIFEST")
    labels = [row["label"] for row in manifest["batches"]]
    expected: dict[str, int] | None = None
    if args.expect_map:
        expected_raw = read_json(Path(args.expect_map), "E_EXPECT_MAP")["allocation"]
        fail(
            isinstance(expected_raw, dict)
            and set(expected_raw) == set(labels)
            and all(isinstance(value, int) and value > 26000 for value in expected_raw.values())
            and len(set(expected_raw.values())) == len(expected_raw),
            "E_EXPECT_MAP",
            "allocation shape",
        )
        expected = expected_raw
    audit = scan(repo, labels, manifest["final_batch_label"], expected, manifest)
    if expected is not None:
        if audit["allocation"] != expected:
            raise AllocationError(
                f"E_ALLOCATION_MOVED: expected {expected!r}; got {audit['allocation']!r}"
            )
    write_audit(Path(args.output), audit)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AllocationError as exc:
        print(str(exc), file=__import__("sys").stderr)
        raise SystemExit(2)
    except (OSError, UnicodeError, json.JSONDecodeError, tarfile.TarError, KeyError, TypeError, ValueError) as exc:
        print(f"E_SCANNER_IO: {type(exc).__name__}", file=__import__("sys").stderr)
        raise SystemExit(2)
```

`--self-test` invokes no fetch, GitHub CLI, or repository scan. It uses only
temporary fixture bytes and directories to prove that distinct identities sharing
one task ID fail closed, an exact content-bound self copy is accepted, a
filename/frontmatter mismatch fails, moved or changed PR heads fail, worktree files
become claims, allocation leapfrogs deterministically with the final label highest,
and an audit output cannot be overwritten. Each mutation must raise its documented
scanner error; a missed mutation makes the self-test exit nonzero. Successful output
is exactly `allocation scanner self-tests: 8 cases passed`.

The first scan's audit is retained under `task26000_tmp_root/raw/allocation.json`.
Immediately before task-file creation and again immediately before commit, rerun the
scanner with `--expect-map` pointing to that audit. Any `E_ALLOCATION_MOVED` forces a
fresh allocation and regeneration of filenames, frontmatter, manifest bindings, and
final dependencies before staging. Every scan verifies the post-fetch `origin/dev`
commit equals `manifest.revisions.current` before it can write an audit. Before
creation, `cleanup_records` is empty and the expected IDs must be wholly unclaimed.
After rendering, the scanner excludes only manifest-proven self claims whose exact
path, frontmatter ID, batch marker, and content SHA-256 match; a ref, PR, or worktree
copy with different bytes is a distinct identity and therefore `E_ID_COLLISION`.


---

## Appendix D: Exact Cleanup-Record Renderer and Writer

Task 5 materializes this standard-library CLI as
`task26000_tmp_root/task26000_render_cleanup.py`. In `create` mode it validates
the complete batch/allocation boundary before writing and journals the exact empty-
record manifest, absent task/output state, and complete new generation. It creates
every non-final task before the final task, atomically binds canonical
`cleanup_records`, and emits sorted UTF-8 task paths with NUL terminators. If create
is interrupted, `--mode recover` removes partial tasks and handoffs while the old
empty-record manifest remains, or completes the tasks and handoffs after the new
manifest landed; a completed identical rerun remains byte-for-byte idempotent.

The separate `refresh` mode exists only for a Task 7 repin that changes an assigned
batch. It refreshes exactly one named cleanup record. Before replacement it requires
the current task-file SHA-256 to equal that record's old manifest
`task_sha256`; it then writes a durable phase journal containing exact old/new task,
manifest, and optional path-list bytes before mutation. The manifest bytes are the
commit oracle: `--mode recover` rolls back the other entries when the old manifest
remains and completes them when the new manifest landed. It never overwrites an
unbound, missing, locally edited, or third-generation path.

The separate `reallocate` mode is the only collision and structural-regeneration
recovery. It accepts the complete old `cleanup_records` already bound in the
manifest even when their labels differ from the current batch labels, proves every
old path is the exact generated file named by its record and SHA-256, and requires a
fresh allocator audit that treated every old task ID as externally occupied. It
renders every current batch with the fresh allocation, preserves `created_at` only
for surviving labels, and journals exact old/new bytes for every old task, new task,
manifest, NUL path list, and active-state handoff before mutation. The journal is
fsynced outside the repository and remains authoritative across process termination
or power interruption until recovery verifies one complete generation and removes
it. This works identically when the old paths are untracked working-tree files or
tracked files from an earlier commit. Its NUL output is the sorted union of every old
and new path, so an exact index update stages tracked removals while harmlessly
ignoring a retired path that was never tracked.

`atomic_replace` uses a 128-bit random, exclusive sibling name and records the opened
inode identity. Cleanup unlinks that temporary only when `lstat` still matches the
owned device/inode; a colliding or substituted path is never removed.

```python
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import secrets
import stat
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

TOP_KEYS = {
    "schema_version", "generated_at_utc", "tools", "revisions", "commands",
    "source_reachability", "censuses", "identities", "historical_diff",
    "historical_sets", "copy_splits", "classifications", "blockers", "batches",
    "final_batch_label", "cleanup_records",
}
SPEC = "Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md"
EVIDENCE = "Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json"
AC_LINES = [
    "- [ ] After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->",
    "- [ ] Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->",
    "- [ ] Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->",
    "- [ ] Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and significant-token position, preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->",
    "- [ ] Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->",
    "- [ ] Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->",
    "- [ ] `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->",
    "- [ ] The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->",
]
FINAL_AC = "- [ ] After all lower-ID cleanup dependencies pass, the explicit Git-tracked repository-wide command exits zero under the recorded Python 3.12.11 interpreter: `python -m ruff format --check --force-exclude .`; any new unassigned failure blocks this gate and is not absorbed. <!-- TASK-26000-CONTRACT: repository-zero-gate -->"
RECORD_KEYS = {
    "label", "path", "task_id", "final", "dependencies", "paths_sha256",
    "task_sha256", "created_at", "updated_at",
}
SHA256 = re.compile(r"[0-9a-f]{64}")
MINUTE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}")


class RenderError(RuntimeError):
    pass


def need(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise RenderError(f"{code}: {detail}")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def paths_digest(paths: list[str]) -> str:
    raw = json.dumps(sorted(paths), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return digest(raw)


def manifest_minute(stamp: Any) -> str:
    need(isinstance(stamp, str) and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", stamp) is not None, "E_RENDER_MANIFEST", "generated_at_utc")
    return stamp[:10] + " " + stamp[11:16]


def validate_inputs(data: dict[str, Any], audit: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    need(set(data) == TOP_KEYS and data["schema_version"] == 1, "E_RENDER_MANIFEST", "top-level schema")
    need(isinstance(data["batches"], list) and data["batches"], "E_RENDER_MANIFEST", "batches")
    batches: dict[str, dict[str, Any]] = {}
    for row in data["batches"]:
        need(set(row) == {"label", "paths", "owner_basis", "test_surface", "conflict_basis"}, "E_RENDER_MANIFEST", "batch schema")
        label = row["label"]
        need(isinstance(label, str) and re.fullmatch(r"ruff-[a-z0-9]+(?:-[a-z0-9]+)*", label) is not None, "E_RENDER_LABEL", repr(label))
        need(label not in batches, "E_RENDER_LABEL", label)
        need(isinstance(row["paths"], list) and row["paths"] == sorted(set(row["paths"])) and row["paths"] and all(isinstance(path, str) for path in row["paths"]), "E_RENDER_PATHS", label)
        need(isinstance(row["owner_basis"], str) and row["owner_basis"], "E_RENDER_MANIFEST", f"{label}.owner_basis")
        need(isinstance(row["test_surface"], list) and row["test_surface"] == sorted(set(row["test_surface"])) and bool(row["test_surface"]) and all(isinstance(value, str) and value for value in row["test_surface"]), "E_RENDER_MANIFEST", f"{label}.test_surface")
        need(isinstance(row["conflict_basis"], list) and bool(row["conflict_basis"]), "E_RENDER_MANIFEST", f"{label}.conflict_basis")
        batches[label] = row
    need(list(batches) == sorted(batches), "E_RENDER_LABEL", "batch order")
    final_label = data["final_batch_label"]
    need(final_label in batches, "E_RENDER_LABEL", "final_batch_label")
    need(isinstance(audit, dict) and isinstance(audit.get("allocation"), dict), "E_RENDER_ALLOCATION", "allocation object")
    allocation = audit["allocation"]
    need(set(allocation) == set(batches), "E_RENDER_ALLOCATION", "allocation labels")
    need(all(isinstance(value, int) and value > 26000 for value in allocation.values()), "E_RENDER_ALLOCATION", "task IDs")
    need(len(set(allocation.values())) == len(allocation), "E_RENDER_ALLOCATION", "duplicate task ID")
    need(allocation[final_label] == max(allocation.values()), "E_RENDER_ALLOCATION", "final is not highest")
    return batches, allocation


def title_for(label: str) -> str:
    return f"Clean Ruff formatter debt for {label}"


def path_for(task_id: int, label: str) -> str:
    return f"backlog/tasks/task-{task_id} - {title_for(label)}.md"


def render_task(batch: dict[str, Any], task_id: int, dependencies: list[int], final: bool, created_at: str, updated_at: str) -> bytes:
    label = batch["label"]
    path_json = json.dumps(batch["paths"], ensure_ascii=False, indent=2)
    tests_json = json.dumps(batch["test_surface"], ensure_ascii=False)
    lines = [
        "---",
        f"id: TASK-{task_id}",
        f"title: {title_for(label)}",
        "status: To Do",
        "assignee: []",
        f"created_date: '{created_at}'",
        f"updated_date: '{updated_at}'",
        "labels:",
        "  - maintenance",
        "  - formatting",
        "  - quality",
        "dependencies:",
        *(f"  - TASK-{value}" for value in dependencies),
        "references:",
        f"  - {SPEC}",
        f"  - {EVIDENCE}",
        "priority: medium",
        "---",
        "",
        f"<!-- TASK-26000-BATCH: {label} -->",
        f"<!-- TASK-26000-PATHS-SHA256: {paths_digest(batch['paths'])} -->",
        f"<!-- TASK-26000-FINAL: {'true' if final else 'false'} -->",
        "",
        "## Description",
        "",
        "<!-- SECTION:DESCRIPTION:BEGIN -->",
        f"Clean the `{label}` Ruff formatter batch at the owner boundary recorded as: {batch['owner_basis']}. The focused test surface recorded by TASK-26000 is `{tests_json}`.",
        "<!-- SECTION:DESCRIPTION:END -->",
        "",
        "## Assigned Paths",
        "",
        "```json",
        path_json,
        "```",
        "",
        "## Acceptance Criteria",
        "<!-- AC:BEGIN -->",
        *AC_LINES,
    ]
    if final:
        lines.append(FINAL_AC)
    lines.extend(["<!-- AC:END -->", ""])
    return "\n".join(lines).encode("utf-8")


def record_metadata(batch: dict[str, Any], task_id: int, dependencies: list[int], final: bool, path: str, raw: bytes, created_at: str, updated_at: str) -> dict[str, Any]:
    return {
        "label": batch["label"],
        "path": path,
        "task_id": task_id,
        "final": final,
        "dependencies": dependencies,
        "paths_sha256": paths_digest(batch["paths"]),
        "task_sha256": digest(raw),
        "created_at": created_at,
        "updated_at": updated_at,
    }


def inode_identity(fd: int) -> tuple[int, int]:
    info = os.fstat(fd)
    return info.st_dev, info.st_ino


def unlink_owned(path: Path, identity: tuple[int, int]) -> None:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISREG(info.st_mode) and (info.st_dev, info.st_ino) == identity:
        path.unlink()


def fsync_parent(path: Path) -> None:
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def exclusive_or_identical(path: Path, raw: bytes) -> bool:
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        need(path.is_file() and path.read_bytes() == raw, "E_RECORD_EXISTS", str(path))
        return False
    identity = inode_identity(fd)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        unlink_owned(path, identity)
        raise
    fsync_parent(path)
    return True


def atomic_replace(path: Path, raw: bytes) -> None:
    temporary: Path | None = None
    identity: tuple[int, int] | None = None
    fd: int | None = None
    for _attempt in range(32):
        candidate = path.with_name(
            f".{path.name}.task26000-{os.getpid()}-{secrets.token_hex(16)}"
        )
        try:
            fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            continue
        temporary = candidate
        identity = inode_identity(fd)
        break
    need(temporary is not None and identity is not None and fd is not None, "E_ATOMIC_TEMP", str(path))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        fsync_parent(path)
    finally:
        if temporary is not None:
            unlink_owned(temporary, identity)


def atomic_create(path: Path, raw: bytes, mode: int) -> None:
    temporary: Path | None = None
    identity: tuple[int, int] | None = None
    fd: int | None = None
    for _attempt in range(32):
        candidate = path.with_name(
            f".{path.name}.task26000-create-{os.getpid()}-{secrets.token_hex(16)}"
        )
        try:
            fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        except FileExistsError:
            continue
        temporary = candidate
        identity = inode_identity(fd)
        break
    need(temporary is not None and identity is not None and fd is not None, "E_ATOMIC_TEMP", str(path))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RenderError(f"E_ATOMIC_CREATE_EXISTS: {path}") from exc
        fsync_parent(path)
    finally:
        if temporary is not None:
            unlink_owned(temporary, identity)


def paths0_bytes(values: list[str]) -> bytes:
    paths = sorted(set(values))
    need(len(paths) == len(set(paths)) and all(value.startswith("backlog/tasks/") and "\n" not in value and "\x00" not in value for value in paths), "E_PATHS0", "task paths")
    return b"\0".join(value.encode("utf-8") for value in paths) + b"\0"


def write_paths0(path: Path, values: list[str]) -> bool:
    return exclusive_or_identical(path, paths0_bytes(values))


def active_state_bytes(mode: str, allocation: dict[str, int], paths0: Path, paths_raw: bytes, records: list[dict[str, Any]]) -> bytes:
    identities = sorted(
        ({"label": row["label"], "path": row["path"], "task_id": row["task_id"]} for row in records),
        key=lambda row: row["label"],
    )
    return canonical_bytes(
        {
            "schema_version": 1,
            "mode": mode,
            "allocation": {label: allocation[label] for label in sorted(allocation)},
            "paths0_output": os.fspath(paths0),
            "paths0_sha256": digest(paths_raw),
            "record_set_sha256": digest(canonical_bytes(identities)),
        }
    )


def validate_active_state_path(repo: Path, path: Path) -> None:
    need(path.is_absolute() and not path.is_relative_to(repo), "E_ACTIVE_STATE_PATH", str(path))
    need(path.parent.is_dir() and not path.parent.is_symlink(), "E_ACTIVE_STATE_PATH", str(path.parent))
    need(not path.is_symlink() and (not path.exists() or path.is_file()), "E_ACTIVE_STATE_PATH", str(path))


def optional_bytes(path: Path, code: str) -> bytes | None:
    if not os.path.lexists(path):
        return None
    need(not path.is_symlink() and path.is_file(), code, str(path))
    return path.read_bytes()


def encoded(raw: bytes | None) -> str | None:
    return None if raw is None else base64.b64encode(raw).decode("ascii")


def decoded(value: Any, code: str) -> bytes | None:
    if value is None:
        return None
    need(isinstance(value, str), code, "encoded bytes")
    try:
        return base64.b64decode(value, validate=True)
    except ValueError as exc:
        raise RenderError(f"{code}: invalid base64") from exc


def state_entry(path: Path, old: bytes | None, new: bytes | None) -> dict[str, Any]:
    need(path.is_absolute(), "E_JOURNAL_PATH", str(path))
    return {"path": os.fspath(path), "old_b64": encoded(old), "new_b64": encoded(new)}


def entry_values(entry: Any, code: str) -> tuple[Path, bytes | None, bytes | None]:
    need(isinstance(entry, dict) and set(entry) == {"path", "old_b64", "new_b64"}, code, "entry schema")
    path = Path(entry["path"])
    need(path.is_absolute(), code, str(path))
    return path, decoded(entry["old_b64"], code), decoded(entry["new_b64"], code)


def apply_entry(entry: dict[str, Any], side: str, code: str) -> None:
    path, old, new = entry_values(entry, code)
    desired = old if side == "old" else new
    current = optional_bytes(path, code)
    need(current == old or current == new, code, f"unexpected bytes: {path}")
    if current == desired:
        return
    if desired is None:
        need(current is not None, code, str(path))
        path.unlink()
        fsync_parent(path)
    elif current is None:
        atomic_create(path, desired, 0o644)
    else:
        atomic_replace(path, desired)


def journal_payload(
    operation: str,
    repo: Path,
    manifest: dict[str, Any],
    tasks: list[dict[str, Any]],
    paths0: dict[str, Any] | None,
    active_state: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": "prepared",
        "operation": operation,
        "repo": os.fspath(repo),
        "manifest": manifest,
        "tasks": tasks,
        "paths0": paths0,
        "active_state": active_state,
    }


def validate_journal(repo: Path, value: Any) -> dict[str, Any]:
    need(
        isinstance(value, dict)
        and set(value) == {"schema_version", "phase", "operation", "repo", "manifest", "tasks", "paths0", "active_state"}
        and value["schema_version"] == 1,
        "E_JOURNAL_SCHEMA",
        "root",
    )
    need(value["phase"] in {"prepared", "files_applied", "committed"}, "E_JOURNAL_SCHEMA", "phase")
    need(value["operation"] in {"create", "refresh", "reallocate"} and value["repo"] == os.fspath(repo), "E_JOURNAL_SCHEMA", "identity")
    manifest_path, manifest_old, manifest_new = entry_values(value["manifest"], "E_JOURNAL_MANIFEST")
    need(manifest_path.is_relative_to(repo) and manifest_old is not None and manifest_new is not None, "E_JOURNAL_MANIFEST", str(manifest_path))
    need(isinstance(value["tasks"], list) and value["tasks"], "E_JOURNAL_SCHEMA", "tasks")
    seen: set[Path] = set()
    for entry in value["tasks"]:
        path, _old, _new = entry_values(entry, "E_JOURNAL_TASK")
        need(path.parent == repo / "backlog/tasks" and path not in seen, "E_JOURNAL_TASK", str(path))
        seen.add(path)
    for name in ("paths0", "active_state"):
        entry = value[name]
        if entry is not None:
            path, _old, _new = entry_values(entry, "E_JOURNAL_OUTPUT")
            need(not path.is_relative_to(repo), "E_JOURNAL_OUTPUT", str(path))
    return value


def write_journal_phase(journal: Path, value: dict[str, Any], phase: str) -> bytes:
    updated = dict(value)
    updated["phase"] = phase
    raw = canonical_bytes(updated)
    atomic_replace(journal, raw)
    value.clear()
    value.update(updated)
    return raw


def remove_exact(path: Path, raw: bytes, code: str) -> None:
    need(not path.is_symlink() and path.is_file() and path.read_bytes() == raw, code, str(path))
    path.unlink()
    fsync_parent(path)


def recover_transaction(repo: Path, journal: Path) -> str:
    need(journal.is_absolute() and not journal.is_relative_to(repo), "E_JOURNAL_PATH", str(journal))
    need(journal.parent.is_dir() and not journal.parent.is_symlink() and journal.is_file() and not journal.is_symlink(), "E_JOURNAL_PATH", str(journal))
    journal_raw = journal.read_bytes()
    value = validate_journal(repo, json.loads(journal_raw.decode("utf-8")))
    manifest_path, manifest_old, manifest_new = entry_values(value["manifest"], "E_JOURNAL_MANIFEST")
    current_manifest = optional_bytes(manifest_path, "E_JOURNAL_MANIFEST")
    if current_manifest == manifest_old:
        side = "old"
        outcome = "rolled-back"
    elif current_manifest == manifest_new:
        side = "new"
        outcome = "committed"
    else:
        raise RenderError("E_JOURNAL_MANIFEST: manifest matches neither journal generation")
    for entry in value["tasks"]:
        apply_entry(entry, side, "E_JOURNAL_TASK")
    for name in ("paths0", "active_state"):
        if value[name] is not None:
            apply_entry(value[name], side, "E_JOURNAL_OUTPUT")
    need(optional_bytes(manifest_path, "E_JOURNAL_MANIFEST") == (manifest_old if side == "old" else manifest_new), "E_JOURNAL_MANIFEST", "changed during recovery")
    remove_exact(journal, journal_raw, "E_JOURNAL_CHANGED")
    return outcome


def run_transaction(repo: Path, journal: Path, value: dict[str, Any]) -> str:
    need(journal.is_absolute() and not journal.is_relative_to(repo), "E_JOURNAL_PATH", str(journal))
    need(journal.parent.is_dir() and not journal.parent.is_symlink() and not os.path.lexists(journal), "E_TRANSACTION_PENDING", str(journal))
    validate_journal(repo, value)
    for entry in value["tasks"]:
        path, old, _new = entry_values(entry, "E_JOURNAL_TASK")
        need(optional_bytes(path, "E_JOURNAL_TASK") == old, "E_JOURNAL_TASK", f"preflight: {path}")
    for name in ("paths0", "active_state"):
        if value[name] is not None:
            path, old, _new = entry_values(value[name], "E_JOURNAL_OUTPUT")
            need(optional_bytes(path, "E_JOURNAL_OUTPUT") == old, "E_JOURNAL_OUTPUT", f"preflight: {path}")
    manifest_path, manifest_old, manifest_new = entry_values(value["manifest"], "E_JOURNAL_MANIFEST")
    need(optional_bytes(manifest_path, "E_JOURNAL_MANIFEST") == manifest_old, "E_JOURNAL_MANIFEST", "preflight")
    atomic_create(journal, canonical_bytes(value), 0o600)
    try:
        for entry in value["tasks"]:
            apply_entry(entry, "new", "E_JOURNAL_TASK")
        for name in ("paths0", "active_state"):
            if value[name] is not None:
                apply_entry(value[name], "new", "E_JOURNAL_OUTPUT")
        write_journal_phase(journal, value, "files_applied")
        need(optional_bytes(manifest_path, "E_JOURNAL_MANIFEST") == manifest_old, "E_JOURNAL_MANIFEST", "before commit")
        atomic_replace(manifest_path, manifest_new)
        write_journal_phase(journal, value, "committed")
        return recover_transaction(repo, journal)
    except BaseException as exc:
        try:
            outcome = recover_transaction(repo, journal)
        except BaseException as recovery_exc:
            raise RenderError(
                f"E_TRANSACTION_RECOVERY: run --mode recover --repo {repo} --journal {journal}: {type(recovery_exc).__name__}: {recovery_exc}"
            ) from exc
        if outcome == "committed":
            return outcome
        raise


def expected_create_records(data: dict[str, Any], batches: dict[str, dict[str, Any]], allocation: dict[str, int]) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    existing = {row["label"]: row for row in data["cleanup_records"]} if data["cleanup_records"] else {}
    need(not existing or len(existing) == len(data["cleanup_records"]) == len(batches), "E_RENDER_STATE", "partial or duplicate cleanup_records")
    final_label = data["final_batch_label"]
    non_final_ids = sorted(value for label, value in allocation.items() if label != final_label)
    stamp = manifest_minute(data["generated_at_utc"])
    records: list[dict[str, Any]] = []
    rendered: dict[str, bytes] = {}
    for label in sorted(batches):
        batch = batches[label]
        final = label == final_label
        dependencies = sorted({26000} | (set(non_final_ids) if final else set()))
        old = existing.get(label)
        created_at = old["created_at"] if old else stamp
        updated_at = old["updated_at"] if old else stamp
        task_path = path_for(allocation[label], label)
        raw = render_task(batch, allocation[label], dependencies, final, created_at, updated_at)
        record = record_metadata(batch, allocation[label], dependencies, final, task_path, raw, created_at, updated_at)
        if old is not None:
            need(old == record, "E_RENDER_STATE", label)
        records.append(record)
        rendered[label] = raw
    return records, rendered


def create_records(repo: Path, manifest: Path, paths0: Path, active_state: Path, journal: Path, data: dict[str, Any], batches: dict[str, dict[str, Any]], allocation: dict[str, int], original_manifest: bytes) -> None:
    records, rendered = expected_create_records(data, batches, allocation)
    final_label = data["final_batch_label"]
    order = sorted(label for label in batches if label != final_label) + [final_label]
    need(manifest.read_bytes() == original_manifest, "E_MANIFEST_MOVED", str(manifest))
    updated = dict(data)
    updated["cleanup_records"] = records
    manifest_raw = canonical_bytes(updated)
    paths_raw = paths0_bytes([record["path"] for record in records])
    validate_active_state_path(repo, active_state)
    state_raw = active_state_bytes("create", allocation, paths0, paths_raw, records)
    task_entries: list[dict[str, Any]] = []
    for label in order:
        record = next(row for row in records if row["label"] == label)
        target = repo / record["path"]
        need(
            target.parent == repo / "backlog/tasks"
            and target.parent.is_dir()
            and not target.parent.is_symlink(),
            "E_RECORD_PATH",
            str(target),
        )
        current = optional_bytes(target, "E_RECORD_EXISTS")
        if data["cleanup_records"]:
            need(current == rendered[label], "E_RECORD_EXISTS", str(target))
        else:
            need(current is None, "E_RECORD_EXISTS", str(target))
        task_entries.append(
            state_entry(target, rendered[label] if data["cleanup_records"] else None, rendered[label])
        )
    if data["cleanup_records"]:
        need(manifest_raw == original_manifest and not os.path.lexists(journal), "E_RENDER_STATE", "completed create rerun")
        need(optional_bytes(paths0, "E_RECORD_EXISTS") in {None, paths_raw}, "E_RECORD_EXISTS", str(paths0))
        need(optional_bytes(active_state, "E_RECORD_EXISTS") in {None, state_raw}, "E_RECORD_EXISTS", str(active_state))
        if not paths0.exists():
            atomic_create(paths0, paths_raw, 0o644)
        if not active_state.exists():
            atomic_create(active_state, state_raw, 0o644)
        return
    need(optional_bytes(paths0, "E_RECORD_EXISTS") is None, "E_RECORD_EXISTS", str(paths0))
    need(optional_bytes(active_state, "E_RECORD_EXISTS") is None, "E_RECORD_EXISTS", str(active_state))
    value = journal_payload(
        "create",
        repo,
        state_entry(manifest, original_manifest, manifest_raw),
        task_entries,
        state_entry(paths0, None, paths_raw),
        state_entry(active_state, None, state_raw),
    )
    run_transaction(repo, journal, value)


def refresh_record(repo: Path, manifest: Path, paths0: Path | None, journal: Path, label: str, data: dict[str, Any], batches: dict[str, dict[str, Any]], allocation: dict[str, int], original_manifest: bytes) -> None:
    records = data["cleanup_records"]
    need(isinstance(records, list) and len(records) == len(batches), "E_REFRESH_STATE", "cleanup_records incomplete")
    by_label = {row["label"]: row for row in records}
    need(len(by_label) == len(records) and set(by_label) == set(batches), "E_REFRESH_STATE", "cleanup_records labels")
    need(label in by_label, "E_REFRESH_LABEL", label)
    old = by_label[label]
    need(old["task_id"] == allocation[label] and old["path"] == path_for(allocation[label], label), "E_REFRESH_STATE", label)
    target = repo / old["path"]
    current_raw = target.read_bytes()
    need(digest(current_raw) == old["task_sha256"], "E_REFRESH_DIRTY", old["path"])
    final = label == data["final_batch_label"]
    non_final_ids = sorted(value for candidate, value in allocation.items() if candidate != data["final_batch_label"])
    dependencies = sorted({26000} | (set(non_final_ids) if final else set()))
    updated_at = manifest_minute(data["generated_at_utc"])
    desired = render_task(batches[label], old["task_id"], dependencies, final, old["created_at"], updated_at)
    replacement = record_metadata(batches[label], old["task_id"], dependencies, final, old["path"], desired, old["created_at"], updated_at)
    new_records = [replacement if row["label"] == label else row for row in records]
    new_records.sort(key=lambda row: row["label"])
    updated = dict(data)
    updated["cleanup_records"] = new_records
    manifest_raw = canonical_bytes(updated)
    need(manifest.read_bytes() == original_manifest, "E_MANIFEST_MOVED", str(manifest))
    paths_entry = None
    if paths0 is not None:
        paths_entry = state_entry(
            paths0,
            optional_bytes(paths0, "E_JOURNAL_OUTPUT"),
            paths0_bytes([record["path"] for record in new_records]),
        )
    value = journal_payload(
        "refresh",
        repo,
        state_entry(manifest, original_manifest, manifest_raw),
        [state_entry(target, current_raw, desired)],
        paths_entry,
        None,
    )
    run_transaction(repo, journal, value)


def task_target(repo: Path, value: Any, code: str) -> Path:
    need(isinstance(value, str), code, repr(value))
    relative = PurePosixPath(value)
    need(
        not relative.is_absolute()
        and len(relative.parts) == 3
        and relative.parts[:2] == ("backlog", "tasks")
        and relative.parts[2] not in {".", ".."}
        and "\n" not in value
        and "\x00" not in value,
        code,
        value,
    )
    target = repo.joinpath(*relative.parts)
    need(
        target.parent == repo / "backlog/tasks"
        and target.parent.is_dir()
        and not target.parent.is_symlink(),
        code,
        value,
    )
    return target


def old_record_files(repo: Path, data: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, bytes]]:
    records = data["cleanup_records"]
    need(isinstance(records, list) and records, "E_REALLOCATE_STATE", "cleanup_records must be non-empty")
    labels: list[str] = []
    task_ids: list[int] = []
    paths: list[str] = []
    raw_by_path: dict[str, bytes] = {}
    for row in records:
        need(isinstance(row, dict) and set(row) == RECORD_KEYS, "E_REALLOCATE_STATE", "cleanup record schema")
        label = row["label"]
        task_id = row["task_id"]
        need(isinstance(label, str) and re.fullmatch(r"ruff-[a-z0-9]+(?:-[a-z0-9]+)*", label) is not None, "E_REALLOCATE_STATE", repr(label))
        need(isinstance(task_id, int) and not isinstance(task_id, bool) and task_id > 26000, "E_REALLOCATE_STATE", f"{label}.task_id")
        need(row["path"] == path_for(task_id, label), "E_REALLOCATE_STATE", f"{label}.path")
        need(isinstance(row["final"], bool), "E_REALLOCATE_STATE", f"{label}.final")
        need(isinstance(row["dependencies"], list) and row["dependencies"] == sorted(set(row["dependencies"])) and all(isinstance(value, int) and not isinstance(value, bool) for value in row["dependencies"]), "E_REALLOCATE_STATE", f"{label}.dependencies")
        need(isinstance(row["paths_sha256"], str) and SHA256.fullmatch(row["paths_sha256"]) is not None, "E_REALLOCATE_STATE", f"{label}.paths_sha256")
        need(isinstance(row["task_sha256"], str) and SHA256.fullmatch(row["task_sha256"]) is not None, "E_REALLOCATE_STATE", f"{label}.task_sha256")
        need(isinstance(row["created_at"], str) and MINUTE.fullmatch(row["created_at"]) is not None, "E_REALLOCATE_STATE", f"{label}.created_at")
        need(isinstance(row["updated_at"], str) and MINUTE.fullmatch(row["updated_at"]) is not None, "E_REALLOCATE_STATE", f"{label}.updated_at")
        target = task_target(repo, row["path"], "E_REALLOCATE_PATH")
        need(not target.is_symlink() and target.is_file(), "E_REALLOCATE_MISSING", row["path"])
        raw = target.read_bytes()
        need(digest(raw) == row["task_sha256"], "E_REALLOCATE_DIRTY", row["path"])
        text = raw.decode("utf-8")
        need(re.search(rf"(?m)^id: TASK-{task_id}$", text) is not None, "E_REALLOCATE_IDENTITY", row["path"])
        need(text.count(f"<!-- TASK-26000-BATCH: {label} -->") == 1, "E_REALLOCATE_IDENTITY", row["path"])
        need(text.count(f"<!-- TASK-26000-PATHS-SHA256: {row['paths_sha256']} -->") == 1, "E_REALLOCATE_IDENTITY", row["path"])
        need(text.count(f"<!-- TASK-26000-FINAL: {'true' if row['final'] else 'false'} -->") == 1, "E_REALLOCATE_IDENTITY", row["path"])
        labels.append(label)
        task_ids.append(task_id)
        paths.append(row["path"])
        raw_by_path[row["path"]] = raw
    need(labels == sorted(set(labels)), "E_REALLOCATE_STATE", "cleanup record labels")
    need(len(task_ids) == len(set(task_ids)) and len(paths) == len(set(paths)), "E_REALLOCATE_STATE", "duplicate task ID or path")
    final_rows = [row for row in records if row["final"]]
    need(len(final_rows) == 1 and final_rows[0]["task_id"] == max(task_ids), "E_REALLOCATE_STATE", "old final record")
    non_final_ids = {row["task_id"] for row in records if not row["final"]}
    for row in records:
        expected = sorted({26000} | (non_final_ids if row["final"] else set()))
        need(row["dependencies"] == expected, "E_REALLOCATE_STATE", f"{row['label']}.dependencies")
    return records, {row["label"]: row for row in records}, raw_by_path


def expected_reallocated_records(data: dict[str, Any], batches: dict[str, dict[str, Any]], allocation: dict[str, int], old_by_label: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    final_label = data["final_batch_label"]
    non_final_ids = sorted(value for label, value in allocation.items() if label != final_label)
    stamp = manifest_minute(data["generated_at_utc"])
    records: list[dict[str, Any]] = []
    rendered: dict[str, bytes] = {}
    for label in sorted(batches):
        batch = batches[label]
        final = label == final_label
        dependencies = sorted({26000} | (set(non_final_ids) if final else set()))
        old = old_by_label.get(label)
        created_at = old["created_at"] if old is not None else stamp
        task_path = path_for(allocation[label], label)
        raw = render_task(batch, allocation[label], dependencies, final, created_at, stamp)
        records.append(record_metadata(batch, allocation[label], dependencies, final, task_path, raw, created_at, stamp))
        rendered[label] = raw
    return records, rendered


def write_exclusive(path: Path, raw: bytes, mode: int) -> None:
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    except FileExistsError as exc:
        raise RenderError(f"E_REALLOCATE_TEMP_EXISTS: {path}") from exc
    identity = inode_identity(fd)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        unlink_owned(path, identity)
        raise
    fsync_parent(path)


def exact_unlink(path: Path, raw: bytes, errors: list[str], code: str) -> None:
    if not os.path.lexists(path):
        return
    try:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
            errors.append(f"{code}:{path}:changed")
            return
        path.unlink()
    except OSError as exc:
        errors.append(f"{code}:{path}:{type(exc).__name__}")


def reallocate_records(repo: Path, manifest: Path, paths0: Path, active_state: Path, journal: Path, data: dict[str, Any], audit: dict[str, Any], batches: dict[str, dict[str, Any]], allocation: dict[str, int], original_manifest: bytes) -> None:
    old_records, old_by_label, old_raw_by_path = old_record_files(repo, data)
    external_ids = audit.get("external_used_ids")
    need(isinstance(external_ids, list) and external_ids == sorted(set(external_ids)) and all(isinstance(value, int) and not isinstance(value, bool) for value in external_ids), "E_REALLOCATE_AUDIT", "external_used_ids")
    old_ids = {row["task_id"] for row in old_records}
    need(old_ids <= set(external_ids), "E_REALLOCATE_AUDIT", "fresh audit did not observe every old task ID")
    need(old_ids.isdisjoint(allocation.values()), "E_REALLOCATE_AUDIT", "fresh allocation reuses an old task ID")
    new_records, rendered = expected_reallocated_records(data, batches, allocation, old_by_label)
    old_paths = {row["path"] for row in old_records}
    new_paths = {row["path"] for row in new_records}
    need(old_paths.isdisjoint(new_paths), "E_REALLOCATE_AUDIT", "old and new task paths overlap")
    for row in new_records:
        target = task_target(repo, row["path"], "E_REALLOCATE_PATH")
        need(not os.path.lexists(target), "E_REALLOCATE_TARGET_EXISTS", row["path"])
    updated = dict(data)
    updated["cleanup_records"] = new_records
    manifest_raw = canonical_bytes(updated)
    paths_raw = paths0_bytes(sorted(old_paths | new_paths))
    validate_active_state_path(repo, active_state)
    old_state_raw = optional_bytes(active_state, "E_JOURNAL_OUTPUT")
    state_raw = active_state_bytes("reallocate", allocation, paths0, paths_raw, new_records)
    need(manifest.read_bytes() == original_manifest, "E_MANIFEST_MOVED", str(manifest))
    task_entries = [
        state_entry(repo / row["path"], None, rendered[row["label"]])
        for row in new_records
    ]
    task_entries.extend(
        state_entry(repo / row["path"], old_raw_by_path[row["path"]], None)
        for row in old_records
    )
    value = journal_payload(
        "reallocate",
        repo,
        state_entry(manifest, original_manifest, manifest_raw),
        task_entries,
        state_entry(paths0, optional_bytes(paths0, "E_JOURNAL_OUTPUT"), paths_raw),
        state_entry(active_state, old_state_raw, state_raw),
    )
    run_transaction(repo, journal, value)


def self_test_batch(label: str, path: str) -> dict[str, Any]:
    return {
        "label": label,
        "paths": [path],
        "owner_basis": f"self-test:{label}",
        "test_surface": ["Tests/self_test.py"],
        "conflict_basis": [{"source": "none", "reference": "none-at-self-test", "paths": []}],
    }


def self_test_manifest(rows: list[dict[str, Any]], final_label: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at_utc": "2026-08-30T20:00:00Z",
        "tools": {}, "revisions": {}, "commands": {}, "source_reachability": {},
        "censuses": {}, "identities": [], "historical_diff": [],
        "historical_sets": {}, "copy_splits": [], "classifications": {},
        "blockers": [], "batches": rows, "final_batch_label": final_label,
        "cleanup_records": [],
    }


def self_test_seed(root: Path, external: Path) -> tuple[Path, Path, Path, dict[str, int]]:
    (root / "backlog/tasks").mkdir(parents=True)
    rows = [self_test_batch("ruff-alpha", "a.py"), self_test_batch("ruff-final-gate", "z.py")]
    data = self_test_manifest(rows, "ruff-final-gate")
    manifest = root / "manifest.json"
    manifest.write_bytes(canonical_bytes(data))
    allocation = {"ruff-alpha": 30000, "ruff-final-gate": 30001}
    batches, checked = validate_inputs(data, {"allocation": allocation})
    paths0 = external / "created-paths0"
    active = external / "active.json"
    journal = external / "create-journal.json"
    create_records(root, manifest, paths0, active, journal, data, batches, checked, manifest.read_bytes())
    return manifest, paths0, active, allocation


def run_self_tests() -> None:
    global atomic_replace, run_transaction, write_journal_phase
    original_atomic = atomic_replace
    original_run_transaction = run_transaction
    original_phase = write_journal_phase
    cases = 0
    try:
        with tempfile.TemporaryDirectory(prefix="task26000-render-selftest-") as temporary_root:
            sandbox = Path(temporary_root)

            collision_target = sandbox / "atomic-target"
            tokens = iter(("a" * 32, "b" * 32))
            collision = collision_target.with_name(f".{collision_target.name}.task26000-{os.getpid()}-{'a' * 32}")
            collision.write_bytes(b"unowned-decoy")
            original_token_hex = secrets.token_hex
            secrets.token_hex = lambda _size: next(tokens)
            try:
                atomic_replace(collision_target, b"owned-result")
            finally:
                secrets.token_hex = original_token_hex
            need(collision.read_bytes() == b"unowned-decoy" and collision_target.read_bytes() == b"owned-result", "E_SELF_TEST", "atomic temp ownership")
            cases += 1

            create_repo = sandbox / "create-repo"
            create_external = sandbox / "create-external"
            create_external.mkdir()
            (create_repo / "backlog/tasks").mkdir(parents=True)
            create_rows = [self_test_batch("ruff-alpha", "a.py"), self_test_batch("ruff-final-gate", "z.py")]
            create_data = self_test_manifest(create_rows, "ruff-final-gate")
            create_manifest = create_repo / "manifest.json"
            create_manifest.write_bytes(canonical_bytes(create_data))
            create_allocation = {"ruff-alpha": 30000, "ruff-final-gate": 30001}
            create_batches, create_checked = validate_inputs(create_data, {"allocation": create_allocation})
            create_paths0 = create_external / "paths0"
            create_active = create_external / "active.json"
            create_journal = create_external / "journal.json"
            create_original = create_manifest.read_bytes()
            create_first = create_repo / path_for(create_allocation["ruff-alpha"], "ruff-alpha")
            create_final = create_repo / path_for(create_allocation["ruff-final-gate"], "ruff-final-gate")

            def partial_create(repo: Path, journal: Path, value: dict[str, Any]) -> str:
                validate_journal(repo, value)
                atomic_create(journal, canonical_bytes(value), 0o600)
                apply_entry(value["tasks"][0], "new", "E_SELF_TEST")
                raise OSError("injected partial create before manifest")

            run_transaction = partial_create
            try:
                create_records(create_repo, create_manifest, create_paths0, create_active, create_journal, create_data, create_batches, create_checked, create_original)
            except OSError as exc:
                need("injected partial create" in str(exc), "E_SELF_TEST", "create injection")
            else:
                raise RenderError("E_SELF_TEST: create injection accepted")
            run_transaction = original_run_transaction
            need(create_journal.is_file() and create_first.is_file() and not os.path.lexists(create_final), "E_SELF_TEST", "partial create state")
            need(create_manifest.read_bytes() == create_original and not os.path.lexists(create_paths0) and not os.path.lexists(create_active), "E_SELF_TEST", "partial create commit boundary")
            need(recover_transaction(create_repo, create_journal) == "rolled-back", "E_SELF_TEST", "create rollback outcome")
            need(not create_journal.exists() and not os.path.lexists(create_first) and not os.path.lexists(create_final), "E_SELF_TEST", "create rollback files")
            create_records(create_repo, create_manifest, create_paths0, create_active, create_journal, create_data, create_batches, create_checked, create_manifest.read_bytes())
            created_data = json.loads(create_manifest.read_text(encoding="utf-8"))
            need(len(created_data["cleanup_records"]) == 2 and create_first.is_file() and create_final.is_file(), "E_SELF_TEST", "create retry tasks")
            need(create_active.is_file() and create_paths0.is_file() and not create_journal.exists(), "E_SELF_TEST", "create retry handoffs")
            cases += 1

            refresh_repo = sandbox / "refresh-repo"
            refresh_external = sandbox / "refresh-external"
            refresh_external.mkdir()
            refresh_manifest, refresh_paths0, _refresh_active, refresh_allocation = self_test_seed(refresh_repo, refresh_external)
            refresh_data = json.loads(refresh_manifest.read_text(encoding="utf-8"))
            refresh_data["batches"][0] = self_test_batch("ruff-alpha", "changed.py")
            refresh_manifest.write_bytes(canonical_bytes(refresh_data))
            refresh_original = refresh_manifest.read_bytes()
            refresh_target = refresh_repo / refresh_data["cleanup_records"][0]["path"]
            refresh_task_original = refresh_target.read_bytes()
            refresh_batches, refresh_checked = validate_inputs(refresh_data, {"allocation": refresh_allocation})
            refresh_journal = refresh_external / "journal.json"

            def refresh_boundary(path: Path, raw: bytes) -> None:
                if path == refresh_manifest:
                    raise OSError("injected refresh manifest boundary")
                original_atomic(path, raw)

            atomic_replace = refresh_boundary
            try:
                refresh_record(refresh_repo, refresh_manifest, refresh_paths0, refresh_journal, "ruff-alpha", refresh_data, refresh_batches, refresh_checked, refresh_original)
            except OSError as exc:
                need("injected refresh" in str(exc), "E_SELF_TEST", "refresh injection")
            else:
                raise RenderError("E_SELF_TEST: refresh injection accepted")
            atomic_replace = original_atomic
            need(refresh_manifest.read_bytes() == refresh_original and refresh_target.read_bytes() == refresh_task_original and not refresh_journal.exists(), "E_SELF_TEST", "refresh rollback")
            cases += 1

            reallocate_repo = sandbox / "reallocate-repo"
            reallocate_external = sandbox / "reallocate-external"
            reallocate_external.mkdir()
            reallocate_manifest, _created_paths0, reallocate_active, _old_allocation = self_test_seed(reallocate_repo, reallocate_external)
            reallocate_data = json.loads(reallocate_manifest.read_text(encoding="utf-8"))
            reallocate_data["batches"] = [self_test_batch("ruff-alpha", "a.py"), self_test_batch("ruff-gamma", "g.py")]
            reallocate_data["final_batch_label"] = "ruff-gamma"
            reallocate_manifest.write_bytes(canonical_bytes(reallocate_data))
            reallocate_audit = {"external_used_ids": [30000, 30001], "allocation": {"ruff-alpha": 30200, "ruff-gamma": 30201}}
            reallocate_batches, reallocate_checked = validate_inputs(reallocate_data, reallocate_audit)
            reallocate_paths0 = reallocate_external / "reallocated-paths0"
            reallocate_journal = reallocate_external / "journal.json"
            fail_committed = {"armed": True}

            def phase_boundary(path: Path, value: dict[str, Any], phase: str) -> bytes:
                if phase == "committed" and fail_committed["armed"]:
                    fail_committed["armed"] = False
                    raise OSError("injected post-commit journal boundary")
                return original_phase(path, value, phase)

            write_journal_phase = phase_boundary
            reallocate_records(reallocate_repo, reallocate_manifest, reallocate_paths0, reallocate_active, reallocate_journal, reallocate_data, reallocate_audit, reallocate_batches, reallocate_checked, reallocate_manifest.read_bytes())
            write_journal_phase = original_phase
            reallocated = json.loads(reallocate_manifest.read_text(encoding="utf-8"))
            need({row["label"] for row in reallocated["cleanup_records"]} == {"ruff-alpha", "ruff-gamma"} and not reallocate_journal.exists(), "E_SELF_TEST", "post-commit resume")
            cases += 1

            corrupt_repo = sandbox / "corrupt-repo"
            corrupt_external = sandbox / "corrupt-external"
            corrupt_external.mkdir()
            corrupt_manifest, _corrupt_created_paths, corrupt_active, _corrupt_old_allocation = self_test_seed(corrupt_repo, corrupt_external)
            corrupt_data = json.loads(corrupt_manifest.read_text(encoding="utf-8"))
            corrupt_data["batches"] = [self_test_batch("ruff-alpha", "a.py"), self_test_batch("ruff-gamma", "g.py")]
            corrupt_data["final_batch_label"] = "ruff-gamma"
            corrupt_manifest.write_bytes(canonical_bytes(corrupt_data))
            corrupt_audit = {"external_used_ids": [30000, 30001], "allocation": {"ruff-alpha": 30200, "ruff-gamma": 30201}}
            corrupt_batches, corrupt_checked = validate_inputs(corrupt_data, corrupt_audit)
            corrupt_paths0 = corrupt_external / "reallocated-paths0"
            corrupt_journal = corrupt_external / "journal.json"

            def corrupt_boundary(path: Path, raw: bytes) -> None:
                if path == corrupt_manifest:
                    original_atomic(corrupt_active, b"third-party-corruption")
                    raise OSError("injected corruption boundary")
                original_atomic(path, raw)

            atomic_replace = corrupt_boundary
            try:
                reallocate_records(corrupt_repo, corrupt_manifest, corrupt_paths0, corrupt_active, corrupt_journal, corrupt_data, corrupt_audit, corrupt_batches, corrupt_checked, corrupt_manifest.read_bytes())
            except RenderError as exc:
                need(str(exc).startswith("E_TRANSACTION_RECOVERY:"), "E_SELF_TEST", "corruption stop")
            else:
                raise RenderError("E_SELF_TEST: active-state corruption accepted")
            atomic_replace = original_atomic
            need(corrupt_journal.is_file(), "E_SELF_TEST", "corruption journal retained")
            journal_value = json.loads(corrupt_journal.read_text(encoding="utf-8"))
            _active_path, active_old, _active_new = entry_values(journal_value["active_state"], "E_SELF_TEST")
            need(active_old is not None, "E_SELF_TEST", "old active state")
            original_atomic(corrupt_active, active_old)
            need(recover_transaction(corrupt_repo, corrupt_journal) == "rolled-back" and not corrupt_journal.exists(), "E_SELF_TEST", "explicit corruption recovery")
            cases += 1
    finally:
        atomic_replace = original_atomic
        run_transaction = original_run_transaction
        write_journal_phase = original_phase
    need(cases == 5, "E_SELF_TEST", f"case count {cases}")
    print("cleanup renderer self-tests: 5 cases passed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("create", "refresh", "reallocate", "recover"))
    parser.add_argument("--repo")
    parser.add_argument("--manifest")
    parser.add_argument("--allocation")
    parser.add_argument("--paths0-output")
    parser.add_argument("--active-state-output")
    parser.add_argument("--refresh-label")
    parser.add_argument("--journal")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        need(all(value is None for value in (args.mode, args.repo, args.manifest, args.allocation, args.paths0_output, args.active_state_output, args.refresh_label, args.journal)), "E_ARGS", "--self-test is exclusive")
        run_self_tests()
        return 0
    need(args.mode is not None and args.repo is not None, "E_ARGS", "--mode and --repo are required")
    repo = Path(args.repo).resolve()
    journal = Path(args.journal).absolute() if args.journal else None
    if args.mode == "recover":
        need(journal is not None and all(value is None for value in (args.manifest, args.allocation, args.paths0_output, args.active_state_output, args.refresh_label)), "E_ARGS", "recover requires only --repo and --journal")
        print(f"cleanup renderer recovery: {recover_transaction(repo, journal)}")
        return 0
    need(args.manifest is not None and args.allocation is not None, "E_ARGS", "operation requires --manifest and --allocation")
    manifest = (repo / args.manifest).resolve() if not Path(args.manifest).is_absolute() else Path(args.manifest).resolve()
    need(manifest.is_relative_to(repo), "E_MANIFEST_PATH", str(manifest))
    original_manifest = manifest.read_bytes()
    data = json.loads(original_manifest.decode("utf-8"))
    audit = json.loads(Path(args.allocation).read_text(encoding="utf-8"))
    batches, allocation = validate_inputs(data, audit)
    paths0 = Path(args.paths0_output).resolve() if args.paths0_output else None
    active_state = Path(args.active_state_output).absolute() if args.active_state_output else None
    if args.mode == "create":
        need(args.refresh_label is None and paths0 is not None and active_state is not None and journal is not None, "E_ARGS", "create requires --paths0-output, --active-state-output, and --journal and forbids --refresh-label")
        create_records(repo, manifest, paths0, active_state, journal, data, batches, allocation, original_manifest)
    elif args.mode == "refresh":
        need(args.refresh_label is not None and active_state is None and journal is not None, "E_ARGS", "refresh requires exactly one --refresh-label and --journal and forbids --active-state-output")
        refresh_record(repo, manifest, paths0, journal, args.refresh_label, data, batches, allocation, original_manifest)
    else:
        need(args.refresh_label is None and paths0 is not None and active_state is not None and journal is not None, "E_ARGS", "reallocate requires --paths0-output, --active-state-output, and --journal and forbids --refresh-label")
        reallocate_records(repo, manifest, paths0, active_state, journal, data, audit, batches, allocation, original_manifest)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RenderError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
```

A repin refresh uses the already revalidated allocation and exactly one stable label.
Set `task26000_refresh_label` from the reviewed repin delta, and invoke the command
once for each label whose assigned paths or rendered ownership evidence changed:

```bash
"${task26000_python}" \
  "${task26000_tmp_root}/task26000_render_cleanup.py" \
  --mode refresh \
  --refresh-label "$task26000_refresh_label" \
  --repo "$PWD" \
  --manifest Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
  --allocation "${task26000_tmp_root}/active-cleanup-state.json" \
  --journal "${task26000_tmp_root}/cleanup-render-transaction.json"
```

Repeat the command once per actually changed label. An `E_REFRESH_DIRTY` result is
a hard stop: inspect the task diff and do not replace it through another tool. Refresh
does not rewrite `active-cleanup-state.json` because its closed record identity,
allocation, and task-path set are unchanged. If the command is interrupted or reports
`E_TRANSACTION_RECOVERY`, run the exact Task 5 Step 6 `--mode recover` command before
retrying the label. The retained journal either restores the exact old task/path-list
generation or completes the exact new generation according to the manifest bytes;
it refuses independently changed content.
