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
  task26000_python="$("${TASK26000_PYTHON}" -c 'import os, sys; print(os.path.abspath(sys.executable))')"
  task26000_resolved_python="${task26000_python}"
  case "${task26000_resolved_python}" in
    /*) ;;
    *) echo 'E_TASK26000_PYTHON: resolved executable must be absolute' >&2; exit 2 ;;
  esac
  test "$("${task26000_python}" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')" = '3.12.11' || { echo 'E_TASK26000_PYTHON: expected Python 3.12.11' >&2; exit 2; }
  test "$("${task26000_python}" -m ruff --version)" = 'ruff 0.15.22' || { echo 'E_TASK26000_PYTHON: expected ruff 0.15.22' >&2; exit 2; }
  export task26000_python task26000_resolved_python
  ```

- Immutable authority cut `S`, manifest `task_base`, and manifest `current` are all
  `e555df102c950c29beed5e7119f433d35eee1f3c`; common is
  `f0e8961222fe1a7a3ac7566f7f78142e717358f3`. Later allocation audits accept an
  observed `origin/dev` equal to `S` or a verified fast-forward descendant. Missing
  or divergent/force-pushed ancestry fails `E_ORIGIN_DEV_DIVERGED`; ordinary
  fast-forward movement never restarts the full evidence cycle.
- The current/common snapshots contain `5,056/1,966` and `4,643/1,746`
  entries/failures. The exhaustive classes are `historical_still_current=44`,
  `historical_no_longer_current=17`, `shared_ancestor_debt=1,603`, and
  `current_line_drift=319`, represented by 2,096 identities and 83 batches with zero
  blockers. The pre-record authority-cut manifest contained zero cleanup records.
- The Task 5/pre-record execution used the current/common raw, lineage,
  replay-cache, pre-record manifest, PR snapshot, materializer, producer, checker,
  allocator, and renderer SHA-256 values respectively
  `f888cf9351f1c41f66fb98b4ec218c9268beb9b23295037320f725cec567ae10`,
  `c34c5fe9d8e3154c3450f1cf28d4c9a6f1f631feb4735296fc6b891af5de1b15`,
  `b9f9876d438b4b6770e84013c515ae54791b14f0e740de67283fb3de20f655a6`,
  `0026dce1124fb3e9fc027dca785101c76a77b63882deac9e1951d5ce2d46a1df`,
  `0f1a8ca2652e7537628c82885f5d5d0cb4421189c31255bb0f05648991083022`,
  `46282d8e81b1bd512263443e97955b1650944684f6c1d0ccd1341f52218bd8d5`,
  `69817bd0bac15097f80c6d194b7b27618bc96f494aab806aeb6d009a9c384c5c`,
  `fd33448f2841d0502509201a5bf6fd2f279f3f2c67cff8f3d4391b9ed7d9ce3e`,
  `a003aee74e01c2729136e244474f1fac08a06ae9ee9331752f56d1bfbffe9e79`,
  `6d7559449c35cd6db3dca31dbbdb510efbb45d1dc0a96c4f01f59c6a8461403b`,
  and `4a08b6a5a9a8b12926ab9417bc330a4e94eb60c3b4afe88226ef232e2653a17a`.
- Current post-failure/Task 7 materializer and allocator SHA-256 values are
  `353160bc073aef50dfcf51f55bd18e261c58e91147db9df30a6e3d0d0f5a2977` and
  `2e456e41bdd2b4f357d181a32b91efdfd07060c33a8f23cc1622d3ef8a4bd432`.
- After Task 5, the canonical manifest contains 83 cleanup records with contiguous
  IDs `TASK-26933` through `TASK-27015`; `TASK-27015` is the final record. Its
  post-record SHA-256 is
  `ded7288d8580367842110dd1a9e79976dc9c00663361251bb9212ca717cea0b9`.
- The 83 exact sorted labels are `ruff-active-pr-1655`,
  `ruff-active-pr-1655-2059`, `ruff-active-pr-1903-2196`,
  `ruff-active-pr-2196`, `ruff-active-pr-2230`, `ruff-active-pr-2264`,
  `ruff-active-pr-2265`, `ruff-agents-runtime`, `ruff-api`,
  `ruff-character-persona`, `ruff-chat-agents-tools`, `ruff-chat-citations`,
  `ruff-chat-console-context`, `ruff-chat-console-fleet`,
  `ruff-chat-console-foundation`, `ruff-chat-console-interaction`,
  `ruff-chat-console-library`, `ruff-chat-console-observability`,
  `ruff-chat-general`, `ruff-chat-media`, `ruff-chat-metrics`,
  `ruff-chat-persistence`, `ruff-chat-providers`, `ruff-chat-retrieval`,
  `ruff-chat-trajectory`, `ruff-chunking`, `ruff-console-character-media`,
  `ruff-console-composer`, `ruff-console-fleet-ui`,
  `ruff-console-foundation-ui`, `ruff-console-inspection`,
  `ruff-console-knowledge-ui`, `ruff-console-layout-rails`,
  `ruff-console-modals`, `ruff-console-runtime`, `ruff-console-session-send`,
  `ruff-console-transcript-selection`, `ruff-console-workspaces`,
  `ruff-core-runtime`, `ruff-database`, `ruff-evals`, `ruff-generation-media`,
  `ruff-ingestion-web-media`, `ruff-integration-live`, `ruff-library`,
  `ruff-library-screen-large`, `ruff-mcp-runtime`, `ruff-model-artifacts-tests`,
  `ruff-notes`, `ruff-performance`, `ruff-personas-screen-large`,
  `ruff-providers-prompts`, `ruff-rag-research`, `ruff-rag-search-tests`,
  `ruff-root-ci-architecture-final`, `ruff-root-test-infrastructure`,
  `ruff-scheduling-notifications`, `ruff-skills-runtime`, `ruff-speech-audio`,
  `ruff-state-sync-wizards-tests`, `ruff-tests-misc`, `ruff-tools-runtime`,
  `ruff-ui-evals`, `ruff-ui-file-dialogs`, `ruff-ui-library`,
  `ruff-ui-mcp-tools`, `ruff-ui-model-management`, `ruff-ui-navigation-shell`,
  `ruff-ui-personas`, `ruff-ui-prompts-workbench`,
  `ruff-ui-remaining-screens`, `ruff-ui-research`, `ruff-ui-scheduling`,
  `ruff-ui-settings`, `ruff-ui-speech`, `ruff-ui-visual-css`,
  `ruff-ui-watchlists`, `ruff-ui-wizards`, `ruff-utils-config`,
  `ruff-watchlists-screen-large`, `ruff-watchlists-subscriptions`,
  `ruff-widgets`, and `ruff-workspaces-runtime`.
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
- The final allocation audit is one canonical
  `raw/allocation-closeout-rescan.json`. Record its canonical final allocation-audit
  SHA-256 plus bound `manifest_pin`, `observed_origin_dev`, and
  `origin_dev_ancestry` in Task 7 Implementation Notes, retain the raw audit through
  review and integration, and do not fetch after that scan.
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
  --revision e555df102c950c29beed5e7119f433d35eee1f3c \
  --label current \
  --output "${task26000_tmp_root}/raw/current.json"
```

An optional scoped call passes `--paths0 PATH`, where `PATH` is a NUL-terminated
byte stream of Git paths. TASK-26000 uses unscoped whole-tree snapshots for all five
revisions and projects `M` into the base/pre-closeout snapshots afterward. The
example current SHA is the owner-approved immutable authority cut; later
`origin/dev` fast-forwards are handled by allocation ancestry audits.

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
  then-recorded `task_base` from the task plan (at that historical step
  `fa0017351ceb375fcb70a0af7cce82dc3d3d4814`) into `task26000_previous_base`, and
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

Pre-Task 3 authority repin (2026-08-30): before any census execution, the recorded
base/current pin `ceac56e06eda4d3d2995a2f5ac8010a7a1821ed2` was refreshed to
`3e5e75e4aa884d4f362aa63c1e151c3855f07a36`. The clean, verified fourteen-commit
TASK-26000 task/spec/plan slice rebased only onto that fresh pin so Task 3 cannot
begin from a stale current-development authority; the derived common ancestor remains
`f0e8961222fe1a7a3ac7566f7f78142e717358f3`.

Task 3 pre-commit authority refresh (2026-08-30): after the first five-census run,
`origin/dev` advanced from `3e5e75e4aa884d4f362aa63c1e151c3855f07a36`
to `57ffb893670ebee744da00c85c0c2c87318357d5`. The clean, verified fifteen-commit
TASK-26000 task/spec/plan slice rebased only onto the fresh pin. The derived common
ancestor remained `f0e8961222fe1a7a3ac7566f7f78142e717358f3`, so only the detached
current checkout, full current census, complete lineage, and downstream digests were
regenerated.

Task 3 final pre-stage authority refresh (2026-08-30): the mandatory final fetch
observed one documentation-only upstream commit and advanced current from
`57ffb893670ebee744da00c85c0c2c87318357d5` to
`857747d3d4e8d048d7c763a65d2a05d9104fc52e`. The same verified fifteen-commit
TASK-26000 documentation slice rebased cleanly. Common and the tracked-Python
contents were unchanged, but the exact-revision current census and complete lineage
were regenerated before staging.

Task 3 spec-review correction repin (2026-08-30): before the follow-up correction
commit, current advanced from `857747d3d4e8d048d7c763a65d2a05d9104fc52e`
to `ae863bfc0e5b33d29a9423e4dcc70664d490cc12`. The clean, verified sixteen-commit
TASK-26000 documentation slice rebased only onto that fresh pin; common remained
`f0e8961222fe1a7a3ac7566f7f78142e717358f3`, and the exact current census and
corrected interval-aware lineage were regenerated.

Task 2/3 executable-provenance correction repin (2026-08-30): the mandatory final
fetch advanced current from `ae863bfc0e5b33d29a9423e4dcc70664d490cc12` to
`747042659706d68861d6e8d88da7a3bbc139f247`. The verified 22-commit TASK-26000
documentation slice rebased only onto that fresh pin; common remained
`f0e8961222fe1a7a3ac7566f7f78142e717358f3`. The detached current checkout, full
current census, complete lineage, and all pin-dependent digests were regenerated
after correcting the executable-provenance contract.

Task 2/3 direct-executable portability correction repin (2026-08-30): the
mandatory final fetch advanced current from
`747042659706d68861d6e8d88da7a3bbc139f247` to
`fa0017351ceb375fcb70a0af7cce82dc3d3d4814`. The verified 23-commit TASK-26000
documentation slice rebased only onto that fresh pin; common remained
`f0e8961222fe1a7a3ac7566f7f78142e717358f3`. The detached current checkout, full
current census, complete lineage, and all pin-dependent digests were regenerated
after making the self-test portable to valid direct executables.

Bounded TASK-26000 pin refresh (2026-08-30): the recorded task base and current
pin advanced from `fa0017351ceb375fcb70a0af7cce82dc3d3d4814` to
`4ae04314c49c54d9241aae8275b5d4b8e14b254e`. The clean 24-commit TASK-26000
task/spec/plan slice was rebased only from that recorded base onto the new pin.
The closeout/current merge base remained
`f0e8961222fe1a7a3ac7566f7f78142e717358f3`, so only the detached current
checkout, current raw census, and current-dependent complete lineage were rebuilt.

Bounded TASK-26000 current-only refresh (2026-08-30): the recorded task base and
current pin advanced from `4ae04314c49c54d9241aae8275b5d4b8e14b254e` to
`872a325483679d2880fcfe2a6e2b9fc82e12f42d`. The clean 26-commit TASK-26000
task/spec/plan slice was rebased only from that recorded base onto the new pin.
The closeout/current merge base remained
`f0e8961222fe1a7a3ac7566f7f78142e717358f3`, so historical/common raws stayed
unchanged while the detached current checkout, current raw census, and complete
lineage were rebuilt.

Bounded TASK-26000 current refresh (2026-08-31): the recorded task base and
current pin advanced from `872a325483679d2880fcfe2a6e2b9fc82e12f42d` to
`05c858e87cc1f11c96d6b384b34fdaf914efc51e`. The clean TASK-26000 committed
range was rebased only from that recorded base onto the new pin. The
closeout/current merge base remained `f0e8961222fe1a7a3ac7566f7f78142e717358f3`,
so the historical/common raws stayed unchanged while the detached current checkout,
current raw census, and complete lineage were rebuilt.

Bounded TASK-26000 current refresh (2026-08-31): the recorded task base and
current pin advanced from `05c858e87cc1f11c96d6b384b34fdaf914efc51e` to
`41176579f185cd4080d0b77441f86db4320a2254`. The clean TASK-26000 committed
range was rebased only from that recorded base onto the new pin. The
closeout/current merge base remained `f0e8961222fe1a7a3ac7566f7f78142e717358f3`,
so historical/common raws stayed unchanged while the detached current checkout,
current raw census, and complete lineage were rebuilt.

Bounded TASK-26000 current refresh (2026-08-31): the recorded task base and
current pin advanced from `41176579f185cd4080d0b77441f86db4320a2254` to
`51d3fbdbf20ff9fc2cf3a3ea3c7f71fef308339a`. The clean TASK-26000 committed
range was rebased only from that recorded base onto the new pin. The
closeout/current merge base remained `f0e8961222fe1a7a3ac7566f7f78142e717358f3`,
so historical/common raws stayed unchanged while the detached current checkout,
current raw census, and complete lineage were rebuilt.

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
  canonical absolute invocation executable without dereferencing a replayable
  virtual-environment symlink; do not copy its absolute path into child-task
  requirements.

---

### Task 3: Reconstruct historical scope and run all pinned censuses

**Files:**

- Create temporarily: `task26000_tmp_root/evidence-repo/`
- Create temporarily: `task26000_tmp_root/checkouts/`
- Create temporarily: `task26000_tmp_root/raw/`
- Create temporarily: `task26000_tmp_root/m-identities.json`

**Interfaces:**

- Consumes: Task 1's five revisions and Task 2's census tool.
- Produces: raw snapshots for `base`, `pre_closeout`, `closeout`, `common`, and
  `current`, plus revision-path identities for the original changed manifest.

- [x] **Step 1: Create clean detached worktrees at every exact revision**

  Reuse Task 2's validated temporary root. Never add evidence worktrees from this
  working repository: its shared Git common directory has active `info/exclude`
  patterns, which the census correctly rejects. Instead create a local, no-network
  clone with its own common Git directory, prove that it has no local external
  excludes and every required object, then add detached evidence worktrees from that
  isolated clone:

  ```bash
  task26000_evidence_repo="${task26000_tmp_root}/evidence-repo"
  git clone --local --no-checkout "$PWD" "${task26000_evidence_repo}"
  task26000_source_common="$(git rev-parse --git-common-dir)"
  case "${task26000_source_common}" in
    /*) ;;
    *) task26000_source_common="$PWD/${task26000_source_common}" ;;
  esac
  task26000_source_common="$(cd "${task26000_source_common}" && pwd -P)"
  task26000_evidence_common="$(git -C "${task26000_evidence_repo}" rev-parse --git-common-dir)"
  case "${task26000_evidence_common}" in
    /*) ;;
    *) task26000_evidence_common="${task26000_evidence_repo}/${task26000_evidence_common}" ;;
  esac
  task26000_evidence_common="$(cd "${task26000_evidence_common}" && pwd -P)"
  test "${task26000_evidence_common}" != "${task26000_source_common}"
  if git -C "${task26000_evidence_repo}" config --local --get core.excludesFile; then
    echo 'E_EXTERNAL_EXCLUDES: isolated clone core.excludesFile is set' >&2; exit 2
  else
    test "$?" = 1
  fi
  task26000_evidence_exclude="$(git -C "${task26000_evidence_repo}" rev-parse --git-path info/exclude)"
  case "${task26000_evidence_exclude}" in
    /*) ;;
    *) task26000_evidence_exclude="${task26000_evidence_repo}/${task26000_evidence_exclude}" ;;
  esac
  test -f "${task26000_evidence_exclude}"
  if awk 'NF && $1 !~ /^#/' "${task26000_evidence_exclude}" | grep -q .; then
    echo 'E_EXTERNAL_EXCLUDES: isolated clone info/exclude has active patterns' >&2; exit 2
  fi
  for task26000_revision in \
    31ed49bb368f54211d6482599e00a5c1340f80b2 \
    1f4f72ac5ff02f5237a4946745e82e8932cd41cf \
    642b1c782fe6c066a781314dae669a55b05b62ad \
    "${task26000_common_ancestor}" \
    "${task26000_current_pin}"; do
    git -C "${task26000_evidence_repo}" cat-file -e "${task26000_revision}^{commit}"
  done
  git -C "${task26000_evidence_repo}" worktree add --detach "${task26000_tmp_root}/checkouts/base" \
    31ed49bb368f54211d6482599e00a5c1340f80b2
  git -C "${task26000_evidence_repo}" worktree add --detach "${task26000_tmp_root}/checkouts/pre_closeout" \
    1f4f72ac5ff02f5237a4946745e82e8932cd41cf
  git -C "${task26000_evidence_repo}" worktree add --detach "${task26000_tmp_root}/checkouts/closeout" \
    642b1c782fe6c066a781314dae669a55b05b62ad
  git -C "${task26000_evidence_repo}" worktree add --detach "${task26000_tmp_root}/checkouts/common" \
    "${task26000_common_ancestor}"
  git -C "${task26000_evidence_repo}" worktree add --detach "${task26000_tmp_root}/checkouts/current" \
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

- [x] **Step 2: Reconstruct `M` as stable identities**

  Parse this command as NUL-delimited bytes:

  ```bash
  git -C "${task26000_evidence_repo}" diff --name-status -z -M \
    31ed49bb368f54211d6482599e00a5c1340f80b2..1f4f72ac5ff02f5237a4946745e82e8932cd41cf \
    -- '*.py'
  ```

  Emit identity records with stable IDs beginning at `I-0000` and increasing
  monotonically in tuple sort order `(base_path or "", pre_closeout_path or "")`.
  Modified paths project to the same
  name at both revisions, adds have `base_path: null`, deletes have
  `pre_closeout_path: null`, and renames retain both names and Git's similarity
  score. Assert exactly 99 identity records before continuing.

- [x] **Step 3: Run full base and pre-closeout censuses, then project `M`**

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

- [x] **Step 4: Run whole-repository closeout, common, and current censuses**

  Run the census tool without `--paths0` at all three revisions. Keep each run
  in its own raw JSON file. Any blocker or disagreement between aggregate and
  per-path failure existence stops the task.

- [x] **Step 5: Resolve the complete lineage graph needed by every classification**

  Record optional `base`, `pre_closeout`, `closeout`, `common`, and `current` paths
  for every `M` identity. Separately create an identity for every `F_common` failure
  and project it from common ancestor to current, including non-`H` debt. Use Git
  rename evidence (`git diff --find-renames --name-status`) as a candidate, then
  inspect each moved/deleted/copied identity with `git log --follow` and blob IDs. A
  path string match across divergent branches is insufficient by itself. Record
  explicit `rename`, `delete`, `add`, `copy`, `unchanged`, or `ambiguous` lineage
  entries with source/target revisions and paths. Any ambiguity is a blocker rather
  than a classification guess.

- [x] **Step 6: Prove TASK-22514's final projected invariant**

  Assert:

  ```text
  F_closeout & project(M, closeout) == project(H, closeout)
  ```

  Expected: exactly the projected 61 identities. A mismatch blocks the task and is
  reported as a TASK-22514 evidence inconsistency; do not change the expected set to
  make the assertion pass.

#### Task 3 Execution Record (2026-08-30)

- Authority remained pinned at base `31ed49bb368f54211d6482599e00a5c1340f80b2`,
  pre-closeout `1f4f72ac5ff02f5237a4946745e82e8932cd41cf`, closeout
  `642b1c782fe6c066a781314dae669a55b05b62ad`, common
  `f0e8961222fe1a7a3ac7566f7f78142e717358f3`, and current
  `51d3fbdbf20ff9fc2cf3a3ea3c7f71fef308339a`; the first evidence run was pinned at
  `3e5e75e4aa884d4f362aa63c1e151c3855f07a36`, then the current census and lineage
  were regenerated after all ten post-initial authority advances.
- The no-network clone `/tmp/task26000.b0z8M0/evidence-repo` had a distinct Git
  common directory, local `core.excludesFile` rc 1, no active `info/exclude`
  patterns, and clean detached worktrees under `/tmp/task26000.b0z8M0/checkouts/`.
  The provenance- and portability-corrected census tool was rematerialized at
  SHA-256 `dc665997e31040be0b16701a83b83890fbc555f93116430691f1e6eb1f860cc0`.
  Its 20-case self-test accepts both a valid direct executable and the retained
  `.venv/bin/python` symlink while requiring metadata and every Ruff argv to use
  the exact invocation path; symlink-dereference, relative, non-executable,
  wrong-version, and missing/wrong-Ruff controls remain fail-closed.
- Raw snapshots under `/tmp/task26000.b0z8M0/raw/` recorded
  base `4,648/1,741` entries/failures (SHA-256
  `7d2c0b02695fc6a05ebe294f629389348b68403f8433466f2ca6bd4d88f8ae17`),
  pre-closeout `4,653/1,754`
  (`073db424a2bc1ba7d0af7a047120c9d3e996eb1f71934fd8f83e823e68fd77ae`),
  closeout `4,653/1,738`
  (`5d29afd7294cbf7149676287edbf7b1f1c3a13824634d98eea7668579fd74e56`),
  common `4,643/1,746`
  (`c34c5fe9d8e3154c3450f1cf28d4c9a6f1f631feb4735296fc6b891af5de1b15`),
  and current `5,028/1,946`
  (`b2c5bb2b56c1357625d79f9ef0189af2751baec5f51782170c16a39787afeab7`);
  every snapshot had zero blockers and aggregate exit 1 reconciled with per-path
  membership. All five were rerun with the final portability-corrected producer;
  the four historical snapshots remained byte-identical because their schema does
  not embed the producer-source digest, while current changed with the required
  authority repin. Each rerun records the canonical absolute `.venv/bin/python`
  invocation identically in `toolchain.resolved_python`, `command_template[0]`, and
  `aggregate_command[0]`; verbatim Appendix B remained SHA-256
  `b16cfb7bdbd94fe0946cad99a4225f8981de87c27df324e78516f5556459a413`, its
  2-positive/14-mutation self-test passed, and its callable census validator accepted
  all five corrected snapshots. Relative to the superseded current snapshot, upstream added two
  tracked Python files, introduced no new formatter failures, and resolved
  `tldw_chatbook/Utils/input_validation.py`. The executable-provenance correction repin added no new
  failures and resolved two Console tests plus
  `tldw_chatbook/UI/Console_Modules/session.py`; the later portability correction
  repin held the tracked-Python count at 4,947 and added two formatter failures.
- `/tmp/task26000.b0z8M0/m-identities.json` is deterministic at SHA-256
  `4118abc9a37988580b43cde8e4733d8e7bc33270e962b8b64c3878d446fca6d0`.
  Identity arithmetic is `M=99`, `B=64`, `C=77`, `C-B=16`, `B-C=3`, and
  `H=61`. Complete lineage contains 94 M identities projected through common and
  current, five feature-branch-only additions, and all 1,746 common failures:
  1,742 unchanged plus four deletes. The aggregate lineage categories are
  `unchanged=2,123`, `add=5`, `delete=4`, `rename=0`, `copy=0`, and
  `ambiguous=0`; each delete records target-anchored `git log --follow`, an exact NUL
  name-status row spanning both interval endpoints, its correlated deletion commit,
  source blob ID, and zero exact-current-blob matches. The four interval records
  require deletion commits `38dbb58a21`, `f9a06ff625` (two paths), and `489a57b050`.
  Before deriving any lineage, the helper now authenticates the isolated repository,
  full pins/ancestry and the unique closeout/current merge base, canonical
  closed-schema snapshots, exact tree and
  configuration inventories, approved toolchain/scope, aggregate reconciliation, and
  the complete M input against the authentic NUL historical diff. Git reads sanitize
  hostile `GIT_*`/configuration/replacement inputs; D/R/C inspection carries both
  paths and fails closed on ambiguity; atomic publication reuses Appendix A's
  owner-safe file/parent-fsync implementation. Same-path projections are rejected if
  a source-descended commit-parent history contains A/D replacement evidence, while
  unrelated merge parents cannot create false continuity breaks. R/C proof propagates
  one identity state through the source-descended commit-parent DAG using event-level
  A/D/R/C rows. Deleting one of several active copy paths removes and records only that
  path; the identity becomes dead only after its final active path is deleted, after
  which resurrection fails closed. Merge parents must agree on the identity-bearing
  active/dead state; historical aliases and tombstones are conservatively unioned so
  a retired alias on either parent still blocks later replacement or resurrection.
  The ADRC candidate scan gates per-parent proof diffs while unchanged
  commits still propagate topology; the performance control reduces 20 unrelated
  commits plus one rename from 21 proof diffs to one. Every category proof persists
  the exact executed command, raw-output digest, and all parsed NUL rows. Deletion
  proofs additionally identify the actual parent and full-row index, and replay
  controls reproduce every pointer and command framing. Full endpoint blob/path maps
  make stationary duplicates ambiguous and fail closed.
  The four real intervals contain neither R/C nor same-path replacement projections.
  Temporary TDD helper/test digests are
  `13f8718bcfc59d96bdd7221a7875fe0806c83566752888003a908cf32b03de67` and
  `2a7f8c519a70ebfc956a50a9c5a3f3db7c62184f3df75c0d3abbfdd8ad89f60a`;
  49 helper controls pass, including merge-parent retired-alias union and later
  alias-resurrection rejection, direct/merge copy survival after deleting the
  original, three-copy partial deletion, all-path deletion/resurrection, exact
  merge-deletion parent/row pointers, candidate-filter call count, prior direct/merge
  rename and copy deletion/reuse, command/digest/row replay, merge-base authority,
  multi-hop/merge-parent R/C chains, duplicate ambiguity, end-to-end D/R/C, odd-path,
  snapshot/M mutation, hostile-environment, strict NUL, and atomic-output cases.
- Blockers remained zero. The historical invariant passed exactly:
  `F_closeout & project(M, closeout) == project(H, closeout)` with 61 projected
  identities.

---

### Task 4: Build the durable manifest and prove its negative cases

**Files:**

- Create:
  `Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json`
- Create temporarily: `task26000_tmp_root/task26000_manifest_producer.py`,
  `task26000_tmp_root/task26000_manifest_check.py`, and
  `task26000_tmp_root/task26000_tool_authority.py`.

**Interfaces:**

- Consumes: all Task 3 raw snapshots and identity/lineage records.
- Produces: the single durable manifest and a validator that exits zero only for a
  complete, consistent artifact.

- [x] **Step 1: Assemble the manifest with canonical JSON serialization**

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
  `current` must include `refs/remotes/origin/dev`. These arrays are authenticated
  point-in-time capture evidence at `generated_at_utc`: ordinary artifact validation
  checks their closed syntax plus every captured revision/tree/blob object, but does
  not require the mutable live remote-ref set to remain equal. The orthogonal
  `--require-live-current` gate is only an immediate authority-cut capture diagnostic:
  after the one refresh-start fetch it proves live `origin/dev` equals the newly
  frozen `revisions.current`. It is not a later pre-records/final closeout invariant.
  The committed appendices are the durable tool source;
  `tools`, `commands`, and every complete census snapshot record the exact runtime
  and invocation provenance without a pre-records/final schema transition.

- [x] **Step 2: Classify current failures exhaustively**

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

  Every current-line-drift row stores an ordered authenticated candidate ledger and
  an independently derived complete `source_aliases` rename chain. Candidate
  `kind`/`causes` are derived from Git path/config/exclusion transitions rather than
  trusted labels. Ruff exit 2 is `invalid` only when an independent Python compile
  of that candidate's exact source blob proves `python_syntax_error`; malformed
  config, invocation, or runtime exit 2 is a blocking non-formatter error.

  `project(F_common identities, current)` comes from Task 3's complete
  common-to-current lineage graph. A duplicate current projection, unexplained copy,
  or ambiguous mapping is a blocker.

- [x] **Step 3: Define owner-aligned stable batches**

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

- [x] **Step 4: Implement the manifest checker**

  Materialize Appendix B verbatim. The checker owns the exact 16-key schema shown in
  Step 1; every nested object is closed to extra keys, every array has a declared
  order/uniqueness rule, every revision, blob, command, source-reachability claim,
  census control, lineage transition, provenance record, stored set, classification,
  blocker, batch, and cleanup-record binding is checked recursively. Do not replace
  the oracle with a JSON Schema or a looser producer-local assertion set.

  `--phase pre-records` and `--phase final` are point-in-time artifact phases.
  They authenticate all stored Git objects and replayable evidence even after an
  unrelated live remote advances. `--require-live-current` is an explicit immediate
  authority-cut capture diagnostic/self-test only; ordinary Task 5 and later Task 7
  review/closeout commands omit it.

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
  Appendix A.1, Appendix B, Appendix C, and Appendix D are the complete durable
  producer/checker/allocator/renderer authority; Appendix B.1 authenticates their marker hashes before atomically materializing
  temporary copies. Task 5 and Task 7 must run that hash preflight before any
  producer/checker/allocator/renderer authority sequence; `/tmp` is never the source of truth.

- [x] **Step 5: Prove the checker fails for corrupt manifests**

  Generate JSON-normalized temporary mutations and require nonzero exits for the
  original manifest corruptions plus captured-ref syntax, explicit live-current,
  canonical-byte, and temporal-ledger corruptions:

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
  missing-post-cut-correction     E_FINAL_GATE
  malformed-captured-ref          E_REACHABILITY
  live-current-mismatch           E_ORIGIN_DEV
  noncanonical-whitespace         E_CANONICAL_BYTES
  noncanonical-key-order          E_CANONICAL_BYTES
  missing-final-newline           E_CANONICAL_BYTES
  nonexistent-temporal-commit     E_TEMPORAL_RANGE
  out-of-range-temporal-commit    E_TEMPORAL_RANGE
  omitted-first-evidence          E_TEMPORAL_EVIDENCE
  reordered-temporal-candidate    E_TEMPORAL_ORDER
  wrong-temporal-result           E_TEMPORAL_RESULT
  wrong-temporal-path-blob        E_TEMPORAL_PATH
  wrong-temporal-config-blob      E_TEMPORAL_CONFIG
  wrong-temporal-exclusion-blob   E_TEMPORAL_EXCLUSION
  missing-prior-clean-state       E_TEMPORAL_COMPLETENESS
  missing-prior-invalid-state     E_TEMPORAL_COMPLETENESS
  wrong-temporal-invalid-reason   E_TEMPORAL_RESULT
  false-temporal-cause            E_TEMPORAL_CAUSE
  missing-pre-rename-alias-segment E_TEMPORAL_COMPLETENESS
  nonformatter-exit-two           E_TEMPORAL_NONFORMATTER
  ```

  Restore the unmodified manifest after every mutation. Run the Appendix B built-in
  fixtures first. The arithmetic fixture contains exactly 99 `M` identities plus one
  shared-ancestor and one current-only identity, passes both positive phases, and
  applies the original 13 mutations independently. Authentic Git fixtures cover
  `A`, `D`, `M`, `R100`, point-in-time ref churn, explicit live-current equality and
  mismatch, canonical raw bytes, and clean/invalid/failing revision-local Ruff
  histories. Each mutation must produce the exact first code recorded by Appendix B.

  ```bash
  "${task26000_python}" \
    "${task26000_tmp_root}/task26000_manifest_check.py" --self-test
  ```

  Expected stdout is exactly:

  ```text
  manifest self-tests: 2 positive phases and 34 deterministic mutations passed
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

  Task 4 intentionally omits `--require-live-current`. Its self-test preserves that
  gate solely as an immediate authority-cut capture diagnostic: equality succeeds and
  mismatch fails exact `E_ORIGIN_DEV`. Once the cut is frozen, all ordinary validation
  is object-ID based and a later fast-forward does not invalidate the artifact.

- [x] **Step 6: Record derived counts and stable labels in both plans**

  After all Task 4 arithmetic checks pass, append an `Execution Record` section to
  this detailed plan containing the exact pins, `M/B/C/H/F_closeout/F_common/current`
  counts, the four comparison counts, blocker count, and sorted stable batch labels.
  Append the same counts and labels—never higher task IDs—to TASK-26000's concise
  Implementation Plan. Regenerate canonical JSON after any resulting label change.

#### Historical Task 4 Execution Record (2026-08-30)

- Pins: task base/current
  `51d3fbdbf20ff9fc2cf3a3ea3c7f71fef308339a`, common
  `f0e8961222fe1a7a3ac7566f7f78142e717358f3`, historical base
  `31ed49bb368f54211d6482599e00a5c1340f80b2`, pre-closeout
  `1f4f72ac5ff02f5237a4946745e82e8932cd41cf`, and closeout
  `642b1c782fe6c066a781314dae669a55b05b62ad`.
- Historical/current counts: `M=99`, `B=64`, `C=77`, `C-B=16`, `B-C=3`,
  `H=61`, `F_closeout=1,738`, `F_common=1,746`, and current failures `=1,946`.
  The exhaustive comparison is `historical_still_current=43`,
  `historical_no_longer_current=18`, `shared_ancestor_debt=1,600`, and
  `current_line_drift=303`; blockers remained zero.
- All 303 current-line-drift rows carry ordered, self-contained first-parent
  integration ledgers authenticated against the captured Git objects. The frozen
  producer replayed 1,211 revision-local Ruff states: 709 failing, 499 clean, and
  three transient invalid states; the invalid states remain explicit and no
  ambiguous final chronology survived. The replay used full detached Git trees,
  tracked ignore/config context, NUL-safe path plumbing, and pinned Ruff 0.15.22.
  Candidate causes use every commit parent, including merge-side path transitions
  that `git log --follow` omits; the independent audit matched all 1,211 candidates.
- Point-in-time ownership capture inspected all 11 open PRs at
  `2026-08-31T04:46:24Z`. Paginated API paths at exact recorded heads matched local
  `51d3fbdb...head` diffs for every PR. Exact current-failure overlaps were
  `#2252=5`, `#2251=4`, `#2230=1`, `#2196=12`, `#2059=1`, `#1903=1`, and
  `#1655=2`; `#2026`, `#1991`, `#1851`, and `#1651` had zero. Worktree and
  14-day recent-Python evidence were projected into every batch's nonempty
  `conflict_basis`.
- The 87 sorted stable labels are `ruff-active-pr-1655`,
  `ruff-active-pr-1655-2059`, `ruff-active-pr-1903-2196`,
  `ruff-active-pr-2196`, `ruff-active-pr-2196-2252`, `ruff-active-pr-2230`,
  `ruff-active-pr-2251`, `ruff-active-pr-2252`, `ruff-agents-runtime`, `ruff-api`,
  `ruff-api-client-large`, `ruff-chachanotes-db-large`,
  `ruff-character-persona`, `ruff-chat-agents-tools`, `ruff-chat-citations`,
  `ruff-chat-console-context`, `ruff-chat-console-fleet`,
  `ruff-chat-console-foundation`, `ruff-chat-console-interaction`,
  `ruff-chat-console-library`, `ruff-chat-console-observability`,
  `ruff-chat-general`, `ruff-chat-media`, `ruff-chat-metrics`,
  `ruff-chat-persistence`, `ruff-chat-providers`, `ruff-chat-retrieval`,
  `ruff-chat-trajectory`, `ruff-chunking`, `ruff-console-character-media`,
  `ruff-console-composer`, `ruff-console-fleet-ui`,
  `ruff-console-foundation-ui`, `ruff-console-inspection`,
  `ruff-console-knowledge-ui`, `ruff-console-layout-rails`, `ruff-console-modals`,
  `ruff-console-runtime`,
  `ruff-console-session-send`, `ruff-console-transcript-selection`,
  `ruff-console-workspaces`, `ruff-core-runtime`, `ruff-database`, `ruff-evals`,
  `ruff-generation-media`, `ruff-ingestion-web-media`, `ruff-integration-live`,
  `ruff-library`, `ruff-library-screen-large`, `ruff-mcp-runtime`,
  `ruff-model-artifacts-tests`, `ruff-notes`, `ruff-performance`,
  `ruff-personas-screen-large`, `ruff-providers-prompts`, `ruff-rag-research`,
  `ruff-rag-search-tests`, `ruff-root-ci-architecture-final`,
  `ruff-root-test-infrastructure`, `ruff-scheduling-notifications`,
  `ruff-settings-screen-large`, `ruff-skills-runtime`, `ruff-speech-audio`,
  `ruff-state-sync-wizards-tests`, `ruff-tests-misc`, `ruff-tools-runtime`,
  `ruff-ui-evals`, `ruff-ui-file-dialogs`, `ruff-ui-library`, `ruff-ui-mcp-tools`,
  `ruff-ui-model-management`, `ruff-ui-navigation-shell`, `ruff-ui-personas`,
  `ruff-ui-prompts-workbench`, `ruff-ui-remaining-screens`, `ruff-ui-research`,
  `ruff-ui-scheduling`, `ruff-ui-settings`, `ruff-ui-speech`, `ruff-ui-visual-css`,
  `ruff-ui-watchlists`, `ruff-ui-wizards`, `ruff-utils-config`,
  `ruff-watchlists-screen-large`, `ruff-watchlists-subscriptions`, `ruff-widgets`,
  and `ruff-workspaces-runtime`.
- Canonical manifest SHA-256 is
  `e012b77f091dee598cfb4495ab4ed7c14f236c86f2ae308c324b32d682afb49f`;
  the frozen producer SHA-256 is
  `dc89d298c9df801c0b08dcf76d9f0a0a2e3669a0cb168b9ee80bde4144085872`;
  verbatim Appendix B/checker SHA-256 is
  `4e89b960c0efe85fb22bde53b9e7b38444e1bb2d44abaee64cc65ff72c3a21a0`,
  and Appendix B.1 authority SHA-256 is
  `6486a50497e6dbef847b10447f190877f90d5215ddc259fd77c08c4100545ac3`.
  Its built-in suite printed exactly
  `manifest self-tests: 2 positive phases and 33 deterministic mutations passed`.
  Ordinary repo-aware `--phase pre-records` accepted the canonical 16-key manifest
  with 2,080 identities, 87 batches, zero blockers, and zero
  cleanup records.
- The manifest is point-in-time evidence at `51d3fbdb...`. During validation,
  live `origin/dev` advanced to
  `1f2d03beb0a2cd82985e395f94bfb05ee992ca7f`; ordinary artifact validation
  intentionally remained green, while the explicit
  `--require-live-current` authority gate failed exactly
  `E_ORIGIN_DEV: origin/dev differs from captured current`. Task 7 owns the deferred
  refresh/reconciliation; no live movement was silently ignored or folded into this
  Task 4 artifact.

#### Historical Task 7 Step 1 Refresh Record (2026-08-31, superseded cut)

- The mandatory live refresh rebased only the 31-commit TASK-26000 range onto task
  base/current `0577884cf24a358a86c2e8711f3b4d5933d6d564`; the resulting uncommitted
  task HEAD is `442e823d9c799ad84a47a04b9fc6b95deae96434`. Common remains
  `f0e8961222fe1a7a3ac7566f7f78142e717358f3`; historical base,
  pre-closeout, and closeout remain `31ed49bb368f54211d6482599e00a5c1340f80b2`,
  `1f4f72ac5ff02f5237a4946745e82e8932cd41cf`, and
  `642b1c782fe6c066a781314dae669a55b05b62ad`.
- The recreated clean detached current checkout contained 5,047 tracked Python
  entries and 1,958 failures; the recreated common checkout contained 4,643 entries
  and 1,746 failures. Both had zero blockers and aggregate exit 1. Current/common
  raw SHA-256 values are
  `5a1c03ac7d56f7aaff4961b7dc8f67b8ca7fd672bb56eb2f10d2432b7c6f4ebf`
  and `c34c5fe9d8e3154c3450f1cf28d4c9a6f1f631feb4735296fc6b891af5de1b15`;
  rebuilt lineage SHA-256 is
  `86999279bfa65dadd17774308f238e63c9e3830348471430a63f0fc5d75fefef`.
- Historical arithmetic remains `M=99`, `B=64`, `C=77`, `C-B=16`, `B-C=3`,
  `H=61`, and `F_closeout=1,738`. The exhaustive comparison is
  `historical_still_current=43`, `historical_no_longer_current=18`,
  `shared_ancestor_debt=1,602`, and `current_line_drift=313`; the refreshed manifest
  has 2,090 identities, 90 batches, zero blockers, and zero cleanup records.
- The cache-cold temporal replay checked 313 rows and 1,257 revision-local Ruff
  candidates: 725 failing, 529 clean, and three transient syntax-invalid states.
  Its SHA-256 is
  `7b2f5c5eda6bb20bbacfff1847eec9fca14e33f34d0f563e1cf052cd11101287`.
- Point-in-time ownership capture inspected all 15 open PRs at
  `2026-08-31T15:51:54Z`; every paginated API file list matched its exact pinned
  local `0577884c...head` diff. Exact current-failure overlaps were `#2262=7`,
  `#2261=4`, `#2257=7`, `#2230=1`, `#2196=12`, `#2059=1`, `#1903=1`,
  and `#1655=2`; `#2263`, `#2258`, `#2254`, `#2026`, `#1991`, `#1851`,
  and `#1651` had zero overlap.
- The 90 sorted stable labels are `ruff-active-pr-1655`,
  `ruff-active-pr-1655-2059`, `ruff-active-pr-1903-2196`,
  `ruff-active-pr-2196`, `ruff-active-pr-2196-2257-2261-2262`,
  `ruff-active-pr-2230`, `ruff-active-pr-2257`,
  `ruff-active-pr-2257-2261`, `ruff-active-pr-2261`, `ruff-active-pr-2262`,
  `ruff-agents-runtime`, `ruff-api`, `ruff-api-client-large`,
  `ruff-app-shell-large`, `ruff-chachanotes-db-large`, `ruff-character-persona`,
  `ruff-chat-agents-tools`, `ruff-chat-citations`, `ruff-chat-console-context`,
  `ruff-chat-console-fleet`, `ruff-chat-console-foundation`,
  `ruff-chat-console-interaction`, `ruff-chat-console-library`,
  `ruff-chat-console-observability`, `ruff-chat-general`, `ruff-chat-media`,
  `ruff-chat-metrics`, `ruff-chat-persistence`, `ruff-chat-providers`,
  `ruff-chat-retrieval`, `ruff-chat-trajectory`, `ruff-chunking`,
  `ruff-console-character-media`, `ruff-console-composer`,
  `ruff-console-fleet-ui`, `ruff-console-foundation-ui`,
  `ruff-console-inspection`, `ruff-console-knowledge-ui`,
  `ruff-console-layout-rails`, `ruff-console-modals`, `ruff-console-runtime`,
  `ruff-console-session-send`, `ruff-console-transcript-selection`,
  `ruff-console-workspaces`, `ruff-core-runtime`, `ruff-database`, `ruff-evals`,
  `ruff-generation-media`, `ruff-ingestion-web-media`, `ruff-integration-live`,
  `ruff-library`, `ruff-library-screen-large`, `ruff-mcp-runtime`,
  `ruff-model-artifacts-tests`, `ruff-notes`, `ruff-performance`,
  `ruff-personas-screen-large`, `ruff-providers-prompts`, `ruff-rag-research`,
  `ruff-rag-search-tests`, `ruff-root-ci-architecture-final`,
  `ruff-root-test-infrastructure`, `ruff-scheduling-notifications`,
  `ruff-settings-screen-large`, `ruff-skills-runtime`, `ruff-speech-audio`,
  `ruff-state-sync-wizards-tests`,
  `ruff-tests-misc`, `ruff-tools-runtime`, `ruff-ui-evals`,
  `ruff-ui-file-dialogs`, `ruff-ui-library`, `ruff-ui-mcp-tools`,
  `ruff-ui-model-management`, `ruff-ui-navigation-shell`, `ruff-ui-personas`,
  `ruff-ui-prompts-workbench`, `ruff-ui-remaining-screens`, `ruff-ui-research`,
  `ruff-ui-scheduling`, `ruff-ui-settings`, `ruff-ui-speech`,
  `ruff-ui-visual-css`, `ruff-ui-watchlists`, `ruff-ui-wizards`,
  `ruff-utils-config`, `ruff-watchlists-screen-large`,
  `ruff-watchlists-subscriptions`, `ruff-widgets`, and `ruff-workspaces-runtime`.
- Canonical manifest, frozen producer, checker, and authority SHA-256 values are
  `94997c33c057f82b91afc758ec0b94d577692a56aea970ba96647d4cf86b6e3d`,
  `c9b1d49521f475715b7b44ff906b4bf5c1279153f851de922970a0025fff4f50`,
  `4e89b960c0efe85fb22bde53b9e7b38444e1bb2d44abaee64cc65ff72c3a21a0`,
  and `6486a50497e6dbef847b10447f190877f90d5215ddc259fd77c08c4100545ac3`.
  Cleanup records remain intentionally absent, so no renderer refresh ran.

#### Pre-Record Owner-Approved Authority-Cut Record (2026-08-31)

- Immutable `task_base`/`current` authority cut is
  `e555df102c950c29beed5e7119f433d35eee1f3c`; common remains
  `f0e8961222fe1a7a3ac7566f7f78142e717358f3`. HEAD at capture was
  `e322f68ec4b50b5f1b28570d946876618db2059d`. A later allocation audit accepts
  `origin/dev` equal to the cut or a verified fast-forward descendant and rejects
  missing/divergent ancestry without refreshing the evidence again.
- Current/common snapshots contain `5,056/1,966` and `4,643/1,746`
  entries/failures with zero blockers. Historical arithmetic remains `M=99`,
  `B=64`, `C=77`, `C-B=16`, `B-C=3`, `H=61`, `F_closeout=1,738`, and
  `F_common=1,746`. Current classes are `historical_still_current=44`,
  `historical_no_longer_current=17`, `shared_ancestor_debt=1,603`, and
  `current_line_drift=319`, represented by 2,096 identities, the 83 exact Global
  Constraints labels, zero blockers, and—at this pre-record capture—zero cleanup
  records.
- The 319 temporal ledgers contain 1,272 candidates: 736 failing, 533 clean, and
  three transient syntax-invalid states. The 13-PR snapshot at
  `2026-08-31T17:40:01Z` has exact current-failure overlaps `#2265=6`, `#2264=4`,
  `#2230=1`, `#2196=12`, `#2059=1`, `#1903=1`, and `#1655=2`; six PRs have
  zero overlap.
- Current/common raw, lineage, replay-cache, pre-record manifest, and PR snapshot
  SHA-256 values are
  `f888cf9351f1c41f66fb98b4ec218c9268beb9b23295037320f725cec567ae10`,
  `c34c5fe9d8e3154c3450f1cf28d4c9a6f1f631feb4735296fc6b891af5de1b15`,
  `b9f9876d438b4b6770e84013c515ae54791b14f0e740de67283fb3de20f655a6`,
  `0026dce1124fb3e9fc027dca785101c76a77b63882deac9e1951d5ce2d46a1df`,
  `0f1a8ca2652e7537628c82885f5d5d0cb4421189c31255bb0f05648991083022`,
  and `46282d8e81b1bd512263443e97955b1650944684f6c1d0ccd1341f52218bd8d5`.
- The post-Task5 canonical manifest contains 83 cleanup records, allocated as
  `TASK-26933` through `TASK-27015` with final record `TASK-27015`, and has SHA-256
  `ded7288d8580367842110dd1a9e79976dc9c00663361251bb9212ca717cea0b9`.
- Task 5 execution materializer, producer, checker, allocator, and renderer SHA-256
  values were
  `69817bd0bac15097f80c6d194b7b27618bc96f494aab806aeb6d009a9c384c5c`,
  `fd33448f2841d0502509201a5bf6fd2f279f3f2c67cff8f3d4391b9ed7d9ce3e`,
  `a003aee74e01c2729136e244474f1fac08a06ae9ee9331752f56d1bfbffe9e79`,
  `6d7559449c35cd6db3dca31dbbdb510efbb45d1dc0a96c4f01f59c6a8461403b`,
  and `4a08b6a5a9a8b12926ab9417bc330a4e94eb60c3b4afe88226ef232e2653a17a`.
- Current post-failure/Task 7 materializer and allocator SHA-256 values are
  `353160bc073aef50dfcf51f55bd18e261c58e91147db9df30a6e3d0d0f5a2977` and
  `2e456e41bdd2b4f357d181a32b91efdfd07060c33a8f23cc1622d3ef8a4bd432`.

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

Before any Task 5 authority sequence, materialize Appendix B.1 verbatim, authenticate
the materializer itself, extract the tracked Appendix A.1/B/C/D sources, and establish
this fail-closed preflight. Re-run `task26000_verify_tool_authority` immediately
before every later Task 5/Task 7 producer, checker, allocator, or renderer sequence; a mismatch stops
before that sequence executes.

```bash
task26000_tool_authority="${task26000_tmp_root}/task26000_tool_authority.py"
task26000_plan='Docs/superpowers/plans/2026-08-30-task-26000-ruff-formatter-debt.md'
task26000_manifest_producer="${task26000_tmp_root}/task26000_manifest_producer.py"
task26000_manifest_checker="${task26000_tmp_root}/task26000_manifest_check.py"
task26000_allocator="${task26000_tmp_root}/task26000_allocate_ids.py"
task26000_renderer="${task26000_tmp_root}/task26000_render_cleanup.py"
task26000_verify_child_hashes() {
  printf '%s  %s\n' \
    'fd33448f2841d0502509201a5bf6fd2f279f3f2c67cff8f3d4391b9ed7d9ce3e' "${task26000_manifest_producer}" \
    'a003aee74e01c2729136e244474f1fac08a06ae9ee9331752f56d1bfbffe9e79' "${task26000_manifest_checker}" \
    '2e456e41bdd2b4f357d181a32b91efdfd07060c33a8f23cc1622d3ef8a4bd432' "${task26000_allocator}" \
    '4a08b6a5a9a8b12926ab9417bc330a4e94eb60c3b4afe88226ef232e2653a17a' "${task26000_renderer}" | shasum -a 256 -c - || {
      echo 'E_TOOL_AUTHORITY_PREFLIGHT: child digest mismatch' >&2
      exit 2
    }
}
printf '%s  %s\n' \
  '353160bc073aef50dfcf51f55bd18e261c58e91147db9df30a6e3d0d0f5a2977' \
  "${task26000_tool_authority}" | shasum -a 256 -c - || {
    echo 'E_TOOL_AUTHORITY_PREFLIGHT: materializer digest mismatch' >&2
    exit 2
  }
"${task26000_python}" "${task26000_tool_authority}" \
  --plan "${task26000_plan}" \
  --producer "${task26000_manifest_producer}" \
  --checker "${task26000_manifest_checker}" \
  --allocator "${task26000_allocator}" \
  --renderer "${task26000_renderer}" || {
    echo 'E_TOOL_AUTHORITY_PREFLIGHT: materialization failed' >&2
    exit 2
  }
task26000_verify_child_hashes
task26000_verify_tool_authority() {
  printf '%s  %s\n' \
    '353160bc073aef50dfcf51f55bd18e261c58e91147db9df30a6e3d0d0f5a2977' \
    "${task26000_tool_authority}" | shasum -a 256 -c - || {
      echo 'E_TOOL_AUTHORITY_PREFLIGHT: materializer digest mismatch' >&2
      exit 2
    }
  task26000_verify_child_hashes
  "${task26000_python}" "${task26000_tool_authority}" \
    --plan "${task26000_plan}" \
    --producer "${task26000_manifest_producer}" \
    --checker "${task26000_manifest_checker}" \
    --allocator "${task26000_allocator}" \
    --renderer "${task26000_renderer}" \
    --verify-only || {
      echo 'E_TOOL_AUTHORITY_PREFLIGHT: verification failed' >&2
      exit 2
    }
}
task26000_verify_tool_authority
```

- [ ] **Step 1: Allocate IDs against the live repository and in-flight work**

  Materialize Appendix C, then run:

  ```bash
  task26000_verify_tool_authority
  "${task26000_python}" \
    "${task26000_tmp_root}/task26000_allocate_ids.py" --self-test || {
      echo 'E_ALLOCATOR_SELF_TEST' >&2
      exit 2
    }
  task26000_verify_tool_authority
  "${task26000_python}" \
    "${task26000_tmp_root}/task26000_allocate_ids.py" \
    --repo "$PWD" \
    --manifest Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
    --output "${task26000_tmp_root}/raw/allocation.json" || {
      echo 'E_INITIAL_ALLOCATION' >&2
      exit 2
    }
  "${task26000_python}" -m pytest \
    Tests/CI/test_backlog_task_id_uniqueness.py -q || {
      echo 'E_INITIAL_IDENTITY_TEST' >&2
      exit 2
    }
  ```

  Expected scanner self-test stdout is exactly:

  ```text
  allocation scanner self-tests: 40 cases passed
  ```

  Inspect the scanner's claims for title-fragment renumbered twins before accepting
  its leapfrog allocation. Do not assume the local maximum is authoritative; the
  scanner's refreshed origin branches, paginated PR-head snapshots, and all local
  worktrees are mandatory inputs. Its canonical audit records `manifest_pin`,
  `observed_origin_dev`, and `origin_dev_ancestry`. Equal tips and verified
  fast-forward descendants are accepted; a missing pin/tip or non-ancestor fails
  exact `E_ORIGIN_DEV_DIVERGED`.

- [ ] **Step 2: Create non-final cleanup records first**

  Materialize Appendix D verbatim. If the create journal exists, recover it before
  the precreate scanner: recovery removes a partial uncommitted generation or
  completes a generation whose manifest commit landed. Then repeat Appendix C with
  the initial audit as `--expect-map`; an external identity on any reserved ID blocks
  rendering, while unrelated claims still recompute the precreate allocation and
  therefore raise `E_ALLOCATION_MOVED`. Then invoke the renderer once:

  ```bash
  task26000_verify_tool_authority
  "${task26000_python}" \
    "${task26000_tmp_root}/task26000_render_cleanup.py" --self-test || {
      echo 'E_RENDERER_SELF_TEST' >&2
      exit 2
    }
  if test -e "${task26000_tmp_root}/cleanup-render-transaction.json"; then
    task26000_verify_tool_authority
    "${task26000_python}" \
      "${task26000_tmp_root}/task26000_render_cleanup.py" \
      --mode recover \
      --repo "$PWD" \
      --journal "${task26000_tmp_root}/cleanup-render-transaction.json" || {
        echo 'E_RENDERER_RECOVERY' >&2
        exit 2
      }
  fi
  test ! -e "${task26000_tmp_root}/cleanup-render-transaction.json" || {
    echo 'E_RENDERER_RECOVERY: journal remains' >&2
    exit 2
  }
  task26000_verify_tool_authority
  "${task26000_python}" \
    "${task26000_tmp_root}/task26000_allocate_ids.py" \
    --repo "$PWD" \
    --manifest Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
    --output "${task26000_tmp_root}/raw/allocation-precreate.json" \
    --expect-map "${task26000_tmp_root}/raw/allocation.json" || {
      echo 'E_PRECREATE_ALLOCATION' >&2
      exit 2
    }
  task26000_verify_tool_authority
  "${task26000_python}" \
    "${task26000_tmp_root}/task26000_render_cleanup.py" \
    --mode create \
    --repo "$PWD" \
    --manifest Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
    --allocation "${task26000_tmp_root}/raw/allocation.json" \
    --paths0-output "${task26000_tmp_root}/raw/new-task-paths0" \
    --active-state-output "${task26000_tmp_root}/active-cleanup-state.json" \
    --journal "${task26000_tmp_root}/cleanup-render-transaction.json" || {
      echo 'E_CREATE_RENDER' >&2
      exit 2
    }
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
  `cleanup renderer self-tests: 9 cases passed` before any real create or reallocate
  invocation; its ninth case proves the superseded refresh entry point changes no bytes.

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
  task26000_verify_tool_authority
  "${task26000_python}" \
    "${task26000_tmp_root}/task26000_manifest_check.py" \
    --phase final \
    --manifest Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
    --repo "$PWD" || {
      echo 'E_TASK5_MANIFEST_CHECK' >&2
      exit 2
    }
  ```

  Expected: exit zero with counts for historical sets, current classifications,
  batches, cleanup records, and zero blockers. Task 5 deliberately omits
  `--require-live-current`: it validates cleanup records against the frozen
  authority-cut artifact.

- [ ] **Step 5: Run Backlog identity and platform-name guards**

  Run:

  ```bash
  "${task26000_python}" -m pytest \
    Tests/CI/test_backlog_task_id_uniqueness.py -q || {
      echo 'E_TASK5_IDENTITY_TEST' >&2
      exit 2
    }
  git diff --check || {
    echo 'E_TASK5_DIFF_CHECK' >&2
    exit 2
  }
  ```

  Expected: three task-ID tests pass, every new basename is checkout-safe, and diff
  check is clean. Existing pytest temporary-directory cleanup warnings are recorded
  separately and do not change the test result.

- [ ] **Step 6: Commit the evidence and cleanup records**

  First repeat the remote-ref, open-PR-head, candidate-ID, filename/frontmatter,
  and uniqueness scans from Step 1 as a precommit collision probe. This is not the
  final claim audit; the final authenticated scan occurs after the active allocation
  is resolved below. The normal no-collision probe is:

  ```bash
  task26000_reallocation_required=0
  task26000_rescan_stderr="${task26000_tmp_root}/raw/allocation-rescan.stderr"
  : > "${task26000_rescan_stderr}" || {
    echo 'E_ALLOCATION_RESCAN: cannot create stderr capture' >&2
    exit 2
  }
  task26000_verify_tool_authority
  if "${task26000_python}" \
      "${task26000_tmp_root}/task26000_allocate_ids.py" \
      --repo "$PWD" \
      --manifest Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
      --output "${task26000_tmp_root}/raw/allocation-rescan.json" \
      --expect-map "${task26000_tmp_root}/raw/allocation.json" \
      2> "${task26000_rescan_stderr}"; then
    test ! -s "${task26000_rescan_stderr}" || {
      echo 'E_ALLOCATION_RESCAN: unexpected stderr on success' >&2
      exit 2
    }
  else
    task26000_rescan_status=$?
    if test "${task26000_rescan_status}" -eq 2 && \
        "${task26000_python}" -c \
          'import sys; from pathlib import Path; lines=Path(sys.argv[1]).read_text(encoding="utf-8").splitlines(); raise SystemExit(0 if len(lines)==1 and lines[0].split(":",1)[0]=="E_ID_COLLISION" else 1)' \
          "${task26000_rescan_stderr}"; then
      cat "${task26000_rescan_stderr}" >&2
      task26000_reallocation_required=1
    else
      cat "${task26000_rescan_stderr}" >&2
      echo 'E_ALLOCATION_RESCAN: non-collision allocator failure' >&2
      exit 2
    fi
  fi
  ```

  The scanner accepts an observed `origin/dev` equal to the manifest pin or a verified
  fast-forward descendant and records both object IDs plus the ancestry result. Exact
  `E_ORIGIN_DEV_DIVERGED` means the pin/tip is missing or the observed tip is not a
  descendant (including force-push/divergence), and is a hard stop. A normal
  descendant does not trigger another full census/review cycle.

  With complete `cleanup_records`, `--expect-map` authenticates every manifest-bound
  record, requires each exact self identity to appear in the live claim census, and
  classifies unexpected identities before retaining the active allocation. A new
  external maximum outside the active IDs remains audit evidence but does not move
  the already-created records. Only an unexpected identity on an active ID raises
  `E_ID_COLLISION` and authorizes the bounded reallocation path below.

  An `E_ID_COLLISION` after rendering, or a reviewed change that adds, removes, or
  renames any batch label, must not be repaired with manual renames or a sequence of
  `refresh` calls. Preserve the old manifest `cleanup_records` and their task files,
  update only the reviewed `batches` / `final_batch_label` structure when applicable,
  and run this exact recovery at most once. The allocator intentionally runs without
  `--expect-map`: its fresh audit must observe every old generated ID as occupied,
  and Appendix D rejects an audit that does not contain all of those old IDs.

  ```bash
  if test "${task26000_reallocation_required}" -eq 1; then
    task26000_verify_tool_authority
    "${task26000_python}" \
      "${task26000_tmp_root}/task26000_allocate_ids.py" \
      --repo "$PWD" \
      --manifest Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
      --output "${task26000_tmp_root}/raw/allocation-recovery.json" || {
        echo 'E_REALLOCATION_AUDIT' >&2
        exit 2
      }
    task26000_verify_tool_authority
    "${task26000_python}" \
      "${task26000_tmp_root}/task26000_render_cleanup.py" \
      --mode reallocate \
      --repo "$PWD" \
      --manifest Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
      --allocation "${task26000_tmp_root}/raw/allocation-recovery.json" \
      --paths0-output "${task26000_tmp_root}/raw/reallocated-task-paths0" \
      --active-state-output "${task26000_tmp_root}/active-cleanup-state.json" \
      --journal "${task26000_tmp_root}/cleanup-render-transaction.json" || {
        echo 'E_REALLOCATION_RENDER' >&2
        exit 2
      }
  fi
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

  A normal reallocation removes its journal only after the selected
  generation is fully verified. After an interruption, or whenever the journal
  remains, run this exact idempotent recovery command before any renderer, scanner,
  staging, or commit command; do not delete or edit the journal manually:

  ```bash
  task26000_verify_tool_authority
  "${task26000_python}" \
    "${task26000_tmp_root}/task26000_render_cleanup.py" \
    --mode recover \
    --repo "$PWD" \
    --journal "${task26000_tmp_root}/cleanup-render-transaction.json" || {
      echo 'E_REALLOCATION_RECOVERY' >&2
      exit 2
    }
  test ! -e "${task26000_tmp_root}/cleanup-render-transaction.json" || {
    echo 'E_REALLOCATION_RECOVERY: journal remains' >&2
    exit 2
  }
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
  so an ordinary verified fast-forward of `origin/dev` does not invalidate the allocation handoff.
  `paths0_sha256` separately binds the exact NUL path-list bytes, including the
  old-plus-new union required after reallocation. Before any later task consumes the
  allocation or path file, run this exact resolver; a
  schema, digest, allocation, path, or NUL-list mismatch is
  `E_ACTIVE_ALLOCATION` and blocks all staging or scans:

  ```bash
  task26000_resolve_active_allocation() {
  task26000_active_state="${task26000_tmp_root}/active-cleanup-state.json"
  task26000_active_exports="$(
  env -u PYTHONOPTIMIZE "${task26000_python}" -c 'if True:
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
  eval "${task26000_active_exports}" || {
    echo 'E_ACTIVE_ALLOCATION: export evaluation failed' >&2
    exit 2
  }
  }
  task26000_resolve_active_allocation
  ```

  After either the normal probe or the one permitted reallocation completes, run the
  resolver above, then perform this authenticated scan against the newly active
  allocation. This is Task 5's last fetch and claim scan. A renewed
  `E_ID_COLLISION` is a hard stop: do not run a second reallocation and do not stage
  or commit. Only this successful canonical audit may be carried forward as the
  Task 5 allocation authority.

  ```bash
  test ! -e "${task26000_tmp_root}/raw/allocation-closeout-rescan.json" || {
    echo 'E_FINAL_ALLOCATION_AUDIT: output already exists' >&2
    exit 2
  }
  task26000_verify_tool_authority
  "${task26000_python}" \
    "${task26000_tmp_root}/task26000_allocate_ids.py" \
    --repo "$PWD" \
    --manifest Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
    --output "${task26000_tmp_root}/raw/allocation-closeout-rescan.json" \
    --expect-map "${task26000_active_allocation}" || {
      echo 'E_FINAL_ALLOCATION_AUDIT: final allocation scan failed' >&2
      exit 2
    }
  ```

  A mismatch is a hard stop, so a process interruption before the manifest commit
  cannot activate a prematurely published recovery handoff. Create mode is
  byte-idempotent: if only handoff
  publication fails after a successful create, rerun the exact Step 2 renderer
  command to publish it without changing generated records. Reallocate publishes the
  new handoff before mutation and restores the exact prior handoff on every caught
  rollback; it performs no fallible cleanup or handoff write after manifest commit.

  After the final successful allocation audit, rerun the positive checker,
  identity guard, and diff guard unconditionally:

  ```bash
  task26000_verify_tool_authority
  "${task26000_python}" \
    "${task26000_tmp_root}/task26000_manifest_check.py" \
    --phase final \
    --manifest Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
    --repo "$PWD" || {
      echo 'E_FINAL_MANIFEST_CHECK' >&2
      exit 2
    }
  "${task26000_python}" -m pytest \
    Tests/CI/test_backlog_task_id_uniqueness.py -q || {
      echo 'E_FINAL_IDENTITY_TEST' >&2
      exit 2
    }
  git diff --check || {
    echo 'E_FINAL_DIFF_CHECK' >&2
    exit 2
  }
  ```

  If the normal rescan succeeded, stage and commit with the original create-mode
  path list:

  ```bash
  git add Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json || {
    echo 'E_TASK5_STAGE' >&2
    exit 2
  }
  git add Docs/superpowers/plans/2026-08-30-task-26000-ruff-formatter-debt.md \
    'backlog/tasks/task-26000 - Characterize-current-dev-inherited-Ruff-formatter-drift.md' || {
      echo 'E_TASK5_STAGE' >&2
      exit 2
    }
  git add --pathspec-from-file="${task26000_tmp_root}/raw/new-task-paths0" \
    --pathspec-file-nul || {
      echo 'E_TASK5_STAGE' >&2
      exit 2
    }
  git commit -m "chore(backlog): partition current Ruff formatter debt" || {
    echo 'E_TASK5_COMMIT' >&2
    exit 2
  }
  ```

  If recovery ran, stage the same durable inputs plus the exact removal/addition
  union, inspect its name-status, and commit instead with:

  ```bash
  git add Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json || {
    echo 'E_TASK5_STAGE' >&2
    exit 2
  }
  git add Docs/superpowers/plans/2026-08-30-task-26000-ruff-formatter-debt.md \
    'backlog/tasks/task-26000 - Characterize-current-dev-inherited-Ruff-formatter-drift.md' || {
      echo 'E_TASK5_STAGE' >&2
      exit 2
    }
  git update-index --add --remove -z --stdin \
    < "${task26000_tmp_root}/raw/reallocated-task-paths0" || {
      echo 'E_TASK5_STAGE' >&2
      exit 2
    }
  git diff --cached --name-status -- backlog/tasks || {
    echo 'E_TASK5_STAGED_DIFF' >&2
    exit 2
  }
  git commit -m "chore(backlog): partition current Ruff formatter debt" || {
    echo 'E_TASK5_COMMIT' >&2
    exit 2
  }
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
  APPROVED. Task 6 review is point-in-time artifact review and therefore omits
  `--require-live-current`; reviewers must still authenticate every captured
  revision/tree/blob and must not reinterpret a later mutable ref as captured truth.

- [ ] **Step 3: Commit reviewed corrections**

  Rerun Task 5's exact active-state resolver against the committed manifest rather
  than assuming collision recovery did not run. A valid create handoff selects
  `new-task-paths0` with `git add`; a valid reallocation handoff selects
  `reallocated-task-paths0` with `git update-index --add --remove`. Any mismatch is
  `E_ACTIVE_ALLOCATION` and blocks the commit. Stage only manifest/task-record
  changes with the selected exact path set and commit:

  ```bash
  task26000_resolve_active_allocation || {
    echo 'E_TASK6_ACTIVE_ALLOCATION' >&2
    exit 2
  }
  git add Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json || {
    echo 'E_TASK6_STAGE' >&2
    exit 2
  }
  git add Docs/superpowers/plans/2026-08-30-task-26000-ruff-formatter-debt.md \
    'backlog/tasks/task-26000 - Characterize-current-dev-inherited-Ruff-formatter-drift.md' || {
      echo 'E_TASK6_STAGE' >&2
      exit 2
    }
  if test "${task26000_active_stage_mode}" = add; then
    git add --pathspec-from-file="${task26000_active_paths0}" --pathspec-file-nul || {
      echo 'E_TASK6_STAGE' >&2
      exit 2
    }
  elif test "${task26000_active_stage_mode}" = update-index; then
    git update-index --add --remove -z --stdin < "${task26000_active_paths0}" || {
      echo 'E_TASK6_STAGE' >&2
      exit 2
    }
  else
    echo 'E_ACTIVE_ALLOCATION: invalid stage mode' >&2; exit 2
  fi
  git diff --cached --name-status -- \
    Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
    Docs/superpowers/plans/2026-08-30-task-26000-ruff-formatter-debt.md \
    'backlog/tasks/task-26000 - Characterize-current-dev-inherited-Ruff-formatter-drift.md' \
    backlog/tasks || {
      echo 'E_TASK6_STAGED_DIFF' >&2
      exit 2
    }
  git commit -m "docs: harden TASK-26000 formatter debt records" || {
    echo 'E_TASK6_COMMIT' >&2
    exit 2
  }
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

- [ ] **Step 1: Authenticate the owner-approved immutable authority cut**

  Do not fetch, rebase, repin, or regenerate the point-in-time evidence again. The
  approved authority cut `S`, manifest `task_base`, and manifest `current` are
  `e555df102c950c29beed5e7119f433d35eee1f3c`; common is
  `f0e8961222fe1a7a3ac7566f7f78142e717358f3`. Repeat Task 5's exact Appendix B.1
  materialization/function block, call `task26000_verify_tool_authority`, verify the
  manifest's canonical authority-cut provenance, revisions, zero blockers, and
  completed label-to-record bindings, and obtain independent approval of the final
  allocated artifact. All census,
  lineage, provenance, PR, canonical, mutation, and review evidence remains bound to
  immutable object IDs. A mutable `origin/dev` fast-forward does not invalidate the
  artifact; Step 2 owns the one later ancestry and collision audit.

  ```bash
  task26000_verify_tool_authority
  "${task26000_python}" \
    "${task26000_tmp_root}/task26000_manifest_check.py" \
    --phase final \
    --manifest Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
    --repo "$PWD" || {
      echo 'E_TASK7_INITIAL_MANIFEST_CHECK' >&2
      exit 2
    }
  ```

  This is the only accepted cut for allocation. Do not run the checker's
  `--require-live-current` diagnostic here: exact live equality was proved at capture,
  while Step 2 deliberately accepts equality or verified fast-forward ancestry.

- [ ] **Step 2: Unconditionally rescan live task-ID claims**

  Rerun Task 5's exact active-allocation resolver, then execute Appendix C against
  that selected audit after Step 1 authenticates the immutable artifact. Preserve
  Task 5's final audit under the no-overwrite archival name below before creating the
  Task 7 audit. The first authenticated allocator invocation captures stderr and
  authorizes recovery only when exit 2 carries exactly one `E_ID_COLLISION` error
  code; every other exit or stderr shape stops before reading audit fields. Run the
  Task-7-only fresh allocation and rollback-backed `reallocate` branch below exactly
  once, obtain independent review, rerun the active-allocation resolver, prove the
  closeout output is still absent, and run the authenticated closeout allocator once
  more. A renewed
  `E_ID_COLLISION` is a hard stop; do not reallocate again and do not continue
  closeout. The first successful `raw/allocation-closeout-rescan.json` is the last
  fetch and claim scan and the only audit allowed to populate the final evidence
  variables.

  Because the manifest now has complete cleanup records, this `--expect-map` scan
  retains the authenticated active allocation after proving every exact self identity
  and rejecting every unexpected identity on its IDs. New unrelated task IDs remain
  in `external_used_ids`; they do not authorize reallocation or produce
  `E_ALLOCATION_MOVED`.

  ```bash
  task26000_task5_final_allocation_audit="${task26000_tmp_root}/raw/allocation-task5-final-rescan.json"
  if test -e "${task26000_tmp_root}/raw/allocation-closeout-rescan.json"; then
    test ! -e "${task26000_task5_final_allocation_audit}" || {
      echo 'E_TASK5_AUDIT_ARCHIVE: archive already exists' >&2
      exit 2
    }
    mv "${task26000_tmp_root}/raw/allocation-closeout-rescan.json" \
      "${task26000_task5_final_allocation_audit}" || {
        echo 'E_TASK5_AUDIT_ARCHIVE: move failed' >&2
        exit 2
      }
  fi
  test ! -e "${task26000_tmp_root}/raw/allocation-closeout-rescan.json" || {
    echo 'E_FINAL_ALLOCATION_AUDIT: output already exists' >&2
    exit 2
  }
  task26000_task7_reallocation_required=0
  task26000_task7_scan_stderr="${task26000_tmp_root}/raw/allocation-task7-closeout.stderr"
  : > "${task26000_task7_scan_stderr}" || {
    echo 'E_FINAL_ALLOCATION_AUDIT: cannot create stderr capture' >&2
    exit 2
  }
  task26000_verify_tool_authority
  if "${task26000_python}" \
      "${task26000_tmp_root}/task26000_allocate_ids.py" \
      --repo "$PWD" \
      --manifest Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
      --output "${task26000_tmp_root}/raw/allocation-closeout-rescan.json" \
      --expect-map "${task26000_active_allocation}" \
      2> "${task26000_task7_scan_stderr}"; then
    test ! -s "${task26000_task7_scan_stderr}" || {
      echo 'E_FINAL_ALLOCATION_AUDIT: unexpected stderr on success' >&2
      exit 2
    }
  else
    task26000_task7_scan_status=$?
    if test "${task26000_task7_scan_status}" -eq 2 && \
        "${task26000_python}" -c \
          'import sys; from pathlib import Path; lines=Path(sys.argv[1]).read_text(encoding="utf-8").splitlines(); raise SystemExit(0 if len(lines)==1 and lines[0].split(":",1)[0]=="E_ID_COLLISION" else 1)' \
          "${task26000_task7_scan_stderr}"; then
      cat "${task26000_task7_scan_stderr}" >&2
      task26000_task7_reallocation_required=1
    else
      cat "${task26000_task7_scan_stderr}" >&2
      echo 'E_FINAL_ALLOCATION_AUDIT: non-collision allocator failure' >&2
      exit 2
    fi
  fi
  ```

  The next conditional block runs only when the preceding closeout allocator produced
  the exact `E_ID_COLLISION` authorization; it is skipped after a successful scan.
  Its Task-7-only audit path cannot collide with Task 5's preserved recovery audit.
  It performs only the fresh allocation and rollback-backed reallocation, writes a
  canonical review-request artifact binding the recovery audit, active handoff, and
  resulting manifest, and then exits with `E_TASK7_REVIEW_REQUIRED`. It MUST NOT run
  the active-allocation resolver or closeout allocator in the same invocation.

  ```bash
  if test "${task26000_task7_reallocation_required}" -eq 1; then
    task26000_task7_review_request="${task26000_tmp_root}/raw/task7-reallocation-review-request.json"
    task26000_task7_review_approval="${task26000_tmp_root}/raw/task7-reallocation-review-approval.json"
    test ! -e "${task26000_tmp_root}/raw/allocation-task7-recovery.json" || {
      echo 'E_TASK7_REALLOCATION: recovery audit already exists' >&2
      exit 2
    }
    test ! -e "${task26000_task7_review_request}" || {
      echo 'E_TASK7_REVIEW_REQUEST: request already exists' >&2
      exit 2
    }
    test ! -e "${task26000_task7_review_approval}" || {
      echo 'E_TASK7_REVIEW_REQUEST: premature approval exists' >&2
      exit 2
    }
    task26000_verify_tool_authority
    "${task26000_python}" \
      "${task26000_tmp_root}/task26000_allocate_ids.py" \
      --repo "$PWD" \
      --manifest Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
      --output "${task26000_tmp_root}/raw/allocation-task7-recovery.json" || {
        echo 'E_TASK7_REALLOCATION: fresh allocation failed' >&2
        exit 2
      }
    task26000_verify_tool_authority
    "${task26000_python}" \
      "${task26000_tmp_root}/task26000_render_cleanup.py" \
      --mode reallocate \
      --repo "$PWD" \
      --manifest Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
      --allocation "${task26000_tmp_root}/raw/allocation-task7-recovery.json" \
      --paths0-output "${task26000_tmp_root}/raw/reallocated-task-paths0" \
      --active-state-output "${task26000_tmp_root}/active-cleanup-state.json" \
      --journal "${task26000_tmp_root}/cleanup-render-transaction.json" || {
        echo 'E_TASK7_REALLOCATION: renderer failed' >&2
        exit 2
      }
    test ! -e "${task26000_tmp_root}/raw/allocation-closeout-rescan.json" || {
      echo 'E_FINAL_ALLOCATION_AUDIT: recovery produced an unexpected closeout audit' >&2
      exit 2
    }
    "${task26000_python}" \
      -c 'import hashlib,json,os,sys
from pathlib import Path
allocation,active,manifest,output = map(Path,sys.argv[1:])
def digest(path):
    if not path.is_file() or path.is_symlink():
        raise SystemExit("invalid review-request input")
    return hashlib.sha256(path.read_bytes()).hexdigest()
value={
    "active_state_sha256":digest(active),
    "allocation_audit_sha256":digest(allocation),
    "decision_required":"independent_review",
    "manifest_sha256":digest(manifest),
    "schema_version":1,
}
raw=(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n").encode()
fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,"wb") as stream:
    stream.write(raw)
' \
      "${task26000_tmp_root}/raw/allocation-task7-recovery.json" \
      "${task26000_tmp_root}/active-cleanup-state.json" \
      Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
      "${task26000_task7_review_request}" || {
        echo 'E_TASK7_REVIEW_REQUEST: materialization failed' >&2
        exit 2
      }
    echo 'E_TASK7_REVIEW_REQUIRED: independent approval artifact required before resume' >&2
    exit 2
  fi
  ```

  The collision invocation ends at that boundary. An independent reviewer must inspect
  the bound request, the reallocated task records, and the manifest, then separately
  create canonical `raw/task7-reallocation-review-approval.json` with exactly
  `schema_version: 1`, `decision: "APPROVED"`, and the SHA-256 of the exact request
  bytes as `request_sha256`. No workflow command creates that approval. In a new
  explicitly resumed shell, repeat Step 1's authority materialization and then
  define the resolver without invoking it by running this exact definition-only
  block:

  ```bash
  task26000_resolve_active_allocation() {
  task26000_active_state="${task26000_tmp_root}/active-cleanup-state.json"
  task26000_active_exports="$(
  env -u PYTHONOPTIMIZE "${task26000_python}" -c 'if True:
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
  eval "${task26000_active_exports}" || {
    echo 'E_ACTIVE_ALLOCATION: export evaluation failed' >&2
    exit 2
  }
  }
  ```

  The definition block performs no reads or validation until called. After it is
  defined, set `task26000_task7_resume_reallocation=1` and run only the resume block
  below. Its first resolver invocation remains after canonical request and approval
  validation. Missing,
  noncanonical, open-schema, stale, or mismatched request/approval bytes stop before
  the resolver and final scan:

  ```bash
  test "${task26000_task7_resume_reallocation:-0}" = 1 || {
    echo 'E_TASK7_REVIEW_APPROVAL: explicit resume authorization missing' >&2
    exit 2
  }
  task26000_task7_review_request="${task26000_tmp_root}/raw/task7-reallocation-review-request.json"
  task26000_task7_review_approval="${task26000_tmp_root}/raw/task7-reallocation-review-approval.json"
  task26000_task7_scan_stderr="${task26000_tmp_root}/raw/allocation-task7-closeout.stderr"
  "${task26000_python}" -c 'import hashlib,json,sys
from pathlib import Path
request_path,approval_path,allocation_path,active_path,manifest_path=map(Path,sys.argv[1:])
def canonical(path):
    if not path.is_file() or path.is_symlink():
        raise SystemExit("missing or linked review artifact")
    raw=path.read_bytes()
    value=json.loads(raw)
    expected=(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n").encode()
    if raw != expected:
        raise SystemExit("noncanonical review artifact")
    return raw,value
def digest(path):
    if not path.is_file() or path.is_symlink():
        raise SystemExit("invalid reviewed input")
    return hashlib.sha256(path.read_bytes()).hexdigest()
request_raw,request=canonical(request_path)
_,approval=canonical(approval_path)
if set(request)!={"schema_version","decision_required","manifest_sha256","allocation_audit_sha256","active_state_sha256"}:
    raise SystemExit("open review-request schema")
if request["schema_version"] != 1 or request["decision_required"] != "independent_review":
    raise SystemExit("invalid review request")
if request["allocation_audit_sha256"] != digest(allocation_path):
    raise SystemExit("stale recovery allocation")
if request["active_state_sha256"] != digest(active_path):
    raise SystemExit("stale active state")
if request["manifest_sha256"] != digest(manifest_path):
    raise SystemExit("stale manifest")
if set(approval)!={"schema_version","decision","request_sha256"}:
    raise SystemExit("open review-approval schema")
if approval != {"schema_version":1,"decision":"APPROVED","request_sha256":hashlib.sha256(request_raw).hexdigest()}:
    raise SystemExit("invalid review approval")
' \
    "${task26000_task7_review_request}" \
    "${task26000_task7_review_approval}" \
    "${task26000_tmp_root}/raw/allocation-task7-recovery.json" \
    "${task26000_tmp_root}/active-cleanup-state.json" \
    Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json || {
      echo 'E_TASK7_REVIEW_APPROVAL: validation failed' >&2
      exit 2
    }
  task26000_resolve_active_allocation || {
    echo 'E_TASK7_REVIEW_APPROVAL: active allocation resolution failed' >&2
    exit 2
  }
  test ! -e "${task26000_tmp_root}/raw/allocation-closeout-rescan.json" || {
    echo 'E_FINAL_ALLOCATION_AUDIT: reviewed resume found an existing closeout audit' >&2
    exit 2
  }
  : > "${task26000_task7_scan_stderr}" || {
    echo 'E_FINAL_ALLOCATION_AUDIT: cannot reset stderr capture' >&2
    exit 2
  }
  task26000_verify_tool_authority
  if "${task26000_python}" \
      "${task26000_tmp_root}/task26000_allocate_ids.py" \
      --repo "$PWD" \
      --manifest Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
      --output "${task26000_tmp_root}/raw/allocation-closeout-rescan.json" \
      --expect-map "${task26000_active_allocation}" \
      2> "${task26000_task7_scan_stderr}"; then
    test ! -s "${task26000_task7_scan_stderr}" || {
      echo 'E_FINAL_ALLOCATION_AUDIT: unexpected stderr on reviewed success' >&2
      exit 2
    }
  else
    task26000_task7_scan_status=$?
    if test "${task26000_task7_scan_status}" -eq 2 && \
        "${task26000_python}" -c \
          'import sys; from pathlib import Path; lines=Path(sys.argv[1]).read_text(encoding="utf-8").splitlines(); raise SystemExit(0 if len(lines)==1 and lines[0].split(":",1)[0]=="E_ID_COLLISION" else 1)' \
          "${task26000_task7_scan_stderr}"; then
      cat "${task26000_task7_scan_stderr}" >&2
      echo 'E_FINAL_ALLOCATION_AUDIT: renewed collision after reviewed reallocation' >&2
    else
      cat "${task26000_task7_scan_stderr}" >&2
      echo 'E_FINAL_ALLOCATION_AUDIT: reviewed post-reallocation scan failed' >&2
    fi
    exit 2
  fi
  ```

  Only after the first successful closeout allocator scan—whether the initial scan
  or the one rerun after the bounded recovery—extract its canonical evidence and
  rerun the authenticated final checker:

  ```bash
  task26000_final_allocation_audit="${task26000_tmp_root}/raw/allocation-closeout-rescan.json"
  task26000_final_allocation_audit_sha256=$("${task26000_python}" -c \
    'import hashlib,json,sys; from pathlib import Path; p=Path(sys.argv[1]); raw=p.read_bytes(); value=json.loads(raw); canonical=(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n").encode(); assert raw==canonical,"allocation audit is not canonical"; print(hashlib.sha256(raw).hexdigest())' \
    "${task26000_final_allocation_audit}") || {
      echo 'E_FINAL_EVIDENCE: audit digest extraction failed' >&2
      exit 2
    }
  task26000_final_manifest_pin=$("${task26000_python}" -c \
    'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["manifest_pin"])' \
    "${task26000_final_allocation_audit}") || {
      echo 'E_FINAL_EVIDENCE: manifest pin extraction failed' >&2
      exit 2
    }
  task26000_final_observed_origin_dev=$("${task26000_python}" -c \
    'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["observed_origin_dev"])' \
    "${task26000_final_allocation_audit}") || {
      echo 'E_FINAL_EVIDENCE: observed origin extraction failed' >&2
      exit 2
    }
  task26000_final_origin_dev_ancestry=$("${task26000_python}" -c \
    'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["origin_dev_ancestry"])' \
    "${task26000_final_allocation_audit}") || {
      echo 'E_FINAL_EVIDENCE: ancestry extraction failed' >&2
      exit 2
    }
  test "${task26000_final_manifest_pin}" = \
    'e555df102c950c29beed5e7119f433d35eee1f3c' || {
      echo 'E_FINAL_MANIFEST_PIN' >&2
      exit 2
    }
  case "${task26000_final_origin_dev_ancestry}" in
    equal|fast_forward_descendant) ;;
    *) echo 'E_ORIGIN_DEV_DIVERGED: invalid final ancestry' >&2; exit 2 ;;
  esac
  task26000_verify_tool_authority
  "${task26000_python}" \
    "${task26000_tmp_root}/task26000_manifest_check.py" \
    --phase final \
    --manifest Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
    --repo "$PWD" || {
      echo 'E_TASK7_MANIFEST_CHECK' >&2
      exit 2
    }
  "${task26000_python}" -m pytest \
    Tests/CI/test_backlog_task_id_uniqueness.py -q || {
      echo 'E_TASK7_IDENTITY_TEST' >&2
      exit 2
    }
  ```

  Appendix C fetches every mandatory remote branch, paginated PR head, and worktree
  task-ID source before comparing the full content-bound claim identities. It records
  the manifest pin, observed `origin/dev`, and exact ancestry result; equality or a
  verified fast-forward descendant is accepted. Missing/non-ancestor state fails
  exact `E_ORIGIN_DEV_DIVERGED` and stops. Collision recovery is bounded by the
  one-attempt procedure above. No fetch occurs after this final scan.
  Retain the canonical
  `raw/allocation-closeout-rescan.json` through review and integration; its
  canonical final allocation-audit SHA-256 and bound `manifest_pin`,
  `observed_origin_dev`, and `origin_dev_ancestry` variables are closeout evidence,
  not replaceable summaries.
  The ordinary point-in-time manifest oracle and task-ID guard must pass; PR-head
  movement and task-ID/worktree collision checks remain strict.

- [ ] **Step 3: Complete task hygiene before the final gate**

  Check all four TASK-26000 acceptance criteria, add concise Implementation Notes
  naming the pins, historical/current counts, comparison categories, batch labels,
  created record count, validator/mutation/review evidence, targeted test result,
  no-Python diff, and `ADR required: no`. The Implementation Notes must also record
  `task26000_final_allocation_audit_sha256`, `task26000_final_manifest_pin`,
  `task26000_final_observed_origin_dev`, and
  `task26000_final_origin_dev_ancestry` under their durable meanings: canonical final
  allocation-audit SHA-256, `manifest_pin`, `observed_origin_dev`, and
  `origin_dev_ancestry`. Set status to Done only after every item is true. Do not mark
  any cleanup record Done.

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
  git add -- 'backlog/tasks/task-26000 - Characterize-current-dev-inherited-Ruff-formatter-drift.md' || {
    echo 'E_TASK7_STAGE' >&2
    exit 2
  }
  if test -n "${task26000_lesson_path}"; then
    case "${task26000_lesson_path}" in
      backlog/docs/lessons-*.md) git add -- "${task26000_lesson_path}" || { echo 'E_TASK7_STAGE' >&2; exit 2; } ;;
      *) echo 'invalid TASK-26000 lesson path' >&2; exit 2 ;;
    esac
  fi
  task26000_verify_tool_authority
  "${task26000_python}" \
    "${task26000_tmp_root}/task26000_manifest_check.py" \
    --phase final \
    --manifest Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json \
    --repo "$PWD" || {
      echo 'E_TASK7_FINAL_MANIFEST_CHECK' >&2
      exit 2
    }
  "${task26000_python}" -m pytest \
    Tests/CI/test_backlog_task_id_uniqueness.py -q || {
      echo 'E_TASK7_FINAL_IDENTITY_TEST' >&2
      exit 2
    }
  task26000_task_base=$("${task26000_python}" \
    -c 'import json; print(json.load(open("Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json"))["revisions"]["task_base"])') || {
      echo 'E_TASK7_TASK_BASE' >&2
      exit 2
    }
  git diff --check "${task26000_task_base}" || {
    echo 'E_TASK7_DIFF_CHECK' >&2
    exit 2
  }
  task26000_python_diff=$(git diff --name-only "${task26000_task_base}" -- '*.py') || {
    echo 'E_TASK7_PYTHON_DIFF: git diff failed' >&2
    exit 2
  }
  test -z "${task26000_python_diff}" || {
    echo 'E_TASK7_PYTHON_DIFF' >&2
    exit 2
  }
  task26000_untracked=$(git ls-files --others --exclude-standard) || {
    echo 'E_TASK7_UNTRACKED_FILES: git ls-files failed' >&2
    exit 2
  }
  test -z "${task26000_untracked}" || {
    echo 'E_TASK7_UNTRACKED_FILES' >&2
    exit 2
  }
  ```

  Expected: manifest checker zero; three task-ID tests pass; diff check clean; the
  Python-path diff is empty. These commands do not claim the repository-wide Ruff
  gate is green—the final cleanup record owns that future outcome.

- [ ] **Step 6: Commit the characterization closeout**

  ```bash
  git commit -m "docs: close TASK-26000 formatter debt characterization" || {
    echo 'E_TASK7_COMMIT' >&2
    exit 2
  }
  git status --short || {
    echo 'E_TASK7_STATUS' >&2
    exit 2
  }
  ```

  Expected: clean worktree. Preserve the temporary raw snapshots, including the exact
  canonical `raw/allocation-closeout-rescan.json`, through review and integration;
  the committed JSON remains the durable point-in-time formatter evidence afterward,
  while Task 7 Implementation Notes retain the final allocation-audit digest and
  ancestry binding.
  Remove every clean detached worktree with:

  ```bash
  git -C "${task26000_evidence_repo}" worktree remove "${task26000_tmp_root}/checkouts/base"
  git -C "${task26000_evidence_repo}" worktree remove "${task26000_tmp_root}/checkouts/pre_closeout"
  git -C "${task26000_evidence_repo}" worktree remove "${task26000_tmp_root}/checkouts/closeout"
  git -C "${task26000_evidence_repo}" worktree remove "${task26000_tmp_root}/checkouts/common"
  git -C "${task26000_evidence_repo}" worktree remove "${task26000_tmp_root}/checkouts/current"
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
import secrets
import shlex
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
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("GIT_", "RUFF_"))
    }
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


def require_python_executable(executable_value: str) -> str:
    require(
        os.path.isabs(executable_value),
        "E_PYTHON_EXECUTABLE",
        "invocation executable must be absolute",
    )
    executable = os.path.abspath(executable_value)
    require(
        os.path.isfile(executable) and os.access(executable, os.X_OK),
        "E_PYTHON_EXECUTABLE",
        f"invocation executable is not executable: {executable}",
    )
    return executable


def require_clean_checkout(
    repo: Path,
    runner: Callable[[tuple[str, ...], Path], subprocess.CompletedProcess[bytes]] = run,
) -> None:
    status = runner(
        ("git", "status", "--porcelain=v1", "-z", "--untracked-files=all"), repo
    )
    require(
        status.returncode == 0 and not status.stdout,
        "E_DIRTY_CHECKOUT",
        "status is nonempty",
    )
    ignored = runner(
        ("git", "ls-files", "--others", "--ignored", "--exclude-standard", "-z"), repo
    )
    require(
        ignored.returncode == 0 and not ignored.stdout,
        "E_IGNORED_RESIDUE",
        "ignored files exist",
    )
    local_excludes = runner(
        ("git", "config", "--local", "--get", "core.excludesFile"), repo
    )
    require(
        local_excludes.returncode in (0, 1),
        "E_EXTERNAL_EXCLUDES",
        f"cannot inspect local core.excludesFile: exit {local_excludes.returncode}",
    )
    require(
        local_excludes.returncode == 1 or not local_excludes.stdout.strip(),
        "E_EXTERNAL_EXCLUDES",
        "local core.excludesFile is set",
    )
    info = runner(("git", "rev-parse", "--git-path", "info/exclude"), repo)
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
        require(
            not active, "E_EXTERNAL_EXCLUDES", ".git/info/exclude has active patterns"
        )


def aggregate_blocker(
    entries: list[dict[str, object]],
    aggregate_exit: int,
    blockers: list[dict[str, object]],
) -> dict[str, object]:
    aggregate = {"exit_code": aggregate_exit}
    if aggregate_exit not in (0, 1):
        blockers.append(
            {"category": "aggregate_nonformatter_exit", "exit_code": aggregate_exit}
        )
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
    require(
        re.fullmatch(r"[0-9a-f]{40}", expected_revision) is not None,
        "E_REVISION",
        "full lowercase SHA required",
    )
    top_level = runner(("git", "rev-parse", "--show-toplevel"), repo)
    require(
        top_level.returncode == 0, "E_CHECKOUT_ROOT", "cannot resolve checkout root"
    )
    checkout_root = Path(top_level.stdout.decode("utf-8", "strict").strip()).resolve()
    require(
        checkout_root == repo,
        "E_CHECKOUT_ROOT",
        f"expected {repo}; got {checkout_root}",
    )
    head_result = runner(("git", "rev-parse", "HEAD^{commit}"), repo)
    require(head_result.returncode == 0, "E_REVISION", "cannot resolve checkout HEAD")
    head = head_result.stdout.decode("ascii").strip()
    require(
        head == expected_revision,
        "E_REVISION",
        f"expected {expected_revision}; got {head}",
    )
    require_clean_checkout(repo, runner)
    python_executable = require_python_executable(sys.executable)
    ruff_version = require_toolchain(
        sys.version_info[:3],
        runner((python_executable, "-m", "ruff", "--version"), repo),
    )
    tree = tree_loader(repo, head)
    universe = sorted(raw for raw in tree if raw.endswith(b".py"))
    chosen = universe if selected is None else selected
    require(
        len(chosen) == len(set(chosen)),
        "E_SELECTION_DUPLICATE",
        "selected paths repeat",
    )
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
        argv = (python_executable, *RUFF, f"./{path}")
        cp = runner(argv, repo)
        result = (
            "not_failing"
            if cp.returncode == 0
            else "would_reformat"
            if cp.returncode == 1
            else "blocked"
        )
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
        cp = runner((python_executable, *RUFF, "."), repo)
        aggregate = aggregate_blocker(entries, cp.returncode, blockers)
        aggregate.update(
            stdout=cp.stdout.decode("utf-8", "backslashreplace"),
            stderr=cp.stderr.decode("utf-8", "backslashreplace"),
        )
    after = runner(
        ("git", "status", "--porcelain=v1", "-z", "--untracked-files=all"), repo
    )
    require(
        after.returncode == 0, "E_CHECKOUT_STATUS", "cannot inspect checkout after Ruff"
    )
    ignored_after = runner(
        ("git", "ls-files", "--others", "--ignored", "--exclude-standard", "-z"),
        repo,
    )
    require(
        ignored_after.returncode == 0,
        "E_CHECKOUT_STATUS",
        "cannot inspect ignored residue after Ruff",
    )
    if after.stdout or ignored_after.stdout:
        blockers.append(
            {
                "category": "checkout_mutated",
                "status_b64": base64.b64encode(after.stdout).decode("ascii"),
                "ignored_b64": base64.b64encode(ignored_after.stdout).decode("ascii"),
            }
        )
    config_names = {
        b"pyproject.toml",
        b"ruff.toml",
        b".ruff.toml",
        b".gitignore",
        b".ignore",
    }
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
            "resolved_python": python_executable,
            "ruff": ruff_version,
        },
        "scope": "all_tracked_dot_py" if selected is None else "selected",
        "command_template": [python_executable, *RUFF, "./PATH_FROM_GIT"],
        "aggregate_command": [python_executable, *RUFF, "."] if selected is None else None,
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


def snapshot_exit_code(snapshot: dict[str, object]) -> int:
    return 2 if snapshot["blockers"] else 0


def cleanup_owned_temp(path: Path, owner: os.stat_result) -> None:
    try:
        current = path.lstat()
    except FileNotFoundError:
        return
    if (current.st_dev, current.st_ino) == (owner.st_dev, owner.st_ino):
        path.unlink()


def write_payload(
    descriptor: int,
    payload: bytes,
    file_sync: Callable[[int], None] = os.fsync,
) -> None:
    with os.fdopen(descriptor, "wb", closefd=False) as handle:
        handle.write(payload)
        handle.flush()
        file_sync(handle.fileno())


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(
    output: Path,
    snapshot: dict[str, object],
    token_factory: Callable[[], str] | None = None,
    replacer: Callable[[Path, Path], None] = os.replace,
    writer: Callable[[int, bytes], None] | None = None,
    file_sync: Callable[[int], None] = os.fsync,
    directory_sync: Callable[[Path], None] = fsync_directory,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    token = token_factory or (lambda: secrets.token_hex(16))
    temporary: Path | None = None
    owner: os.stat_result | None = None
    descriptor: int | None = None
    try:
        for _ in range(16):
            candidate = output.with_name(f".{output.name}.task26000-{token()}")
            try:
                descriptor = os.open(
                    candidate,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
            except FileExistsError:
                continue
            temporary = candidate
            owner = os.fstat(descriptor)
            os.fchmod(descriptor, 0o600)
            if writer is None:
                write_payload(descriptor, payload, file_sync)
            else:
                writer(descriptor, payload)
            os.close(descriptor)
            descriptor = None
            break
        require(
            temporary is not None and owner is not None,
            "E_ATOMIC_WRITE",
            "temporary-name collision",
        )
        replacer(temporary, output)
        directory_sync(output.parent)
    except OSError as exc:
        if descriptor is not None:
            try:
                os.close(descriptor)
            finally:
                descriptor = None
        if temporary is not None and owner is not None:
            cleanup_owned_temp(temporary, owner)
        raise EvidenceError(f"E_ATOMIC_WRITE: {type(exc).__name__}") from exc
    except BaseException:
        if descriptor is not None:
            try:
                os.close(descriptor)
            finally:
                descriptor = None
        if temporary is not None and owner is not None:
            cleanup_owned_temp(temporary, owner)
        raise


def make_repo(root: Path, files: dict[bytes, bytes], config: bytes) -> tuple[Path, str]:
    root.mkdir()
    require(run(("git", "init"), root).returncode == 0, "E_SELFTEST", "git init failed")
    for raw_path, content in files.items():
        full = os.path.join(os.fsencode(root), raw_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as handle:
            handle.write(content)
    (root / "pyproject.toml").write_bytes(config)
    require(
        run(("git", "add", "--all"), root).returncode == 0,
        "E_SELFTEST",
        "git add failed",
    )
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
    head = (
        run(("git", "rev-parse", "HEAD^{commit}"), root).stdout.decode("ascii").strip()
    )
    return root, head


def expect_error(code: str, callback: Callable[[], object]) -> None:
    try:
        callback()
    except EvidenceError as exc:
        require(
            str(exc).startswith(f"{code}:"), "E_SELFTEST", f"expected {code}; got {exc}"
        )
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
        by_path = {row["path"]: row for row in snapshot["entries"] if "path" in row}
        require(
            by_path["clean.py"]["result"] == "not_failing", "E_SELFTEST", "clean status"
        )
        require(
            by_path["fail.py"]["result"] == "would_reformat",
            "E_SELFTEST",
            "fail status",
        )
        require(
            by_path["excluded.py"]["result"] == "not_failing",
            "E_SELFTEST",
            "exclude status",
        )
        require(
            by_path["-lead.py"]["result"] == "not_failing", "E_SELFTEST", "dash status"
        )
        require(
            by_path["space name.py"]["result"] == "not_failing",
            "E_SELFTEST",
            "space status",
        )
        require(
            by_path["line\nbreak.py"]["result"] == "not_failing",
            "E_SELFTEST",
            "newline status",
        )
        require(
            snapshot["aggregate"]["exit_code"] == 1, "E_SELFTEST", "aggregate status"
        )
        require(not snapshot["blockers"], "E_SELFTEST", "unexpected basic blocker")

        original_executable = sys.executable

        def prove_executable_provenance(expected_executable: str) -> None:
            observed_ruff_argv: list[tuple[str, ...]] = []

            def provenance_runner(
                argv: tuple[str, ...],
                cwd: Path,
            ) -> subprocess.CompletedProcess[bytes]:
                if len(argv) >= 3 and argv[1:3] == ("-m", "ruff"):
                    observed_ruff_argv.append(argv)
                return run(argv, cwd)

            provenance_snapshot = build_snapshot(
                str(basic), basic_head, "selftest", runner=provenance_runner
            )
            require(
                provenance_snapshot["toolchain"]["resolved_python"]
                == expected_executable,
                "E_SELFTEST",
                "absolute invocation executable metadata",
            )
            require(
                provenance_snapshot["command_template"][0] == expected_executable
                and provenance_snapshot["aggregate_command"][0] == expected_executable
                and observed_ruff_argv
                and all(argv[0] == expected_executable for argv in observed_ruff_argv),
                "E_SELFTEST",
                "invocation executable must be identical in metadata and every Ruff command",
            )

        invocation_executable = os.path.abspath(original_executable)
        prove_executable_provenance(invocation_executable)
        if os.path.islink(original_executable):
            require(
                invocation_executable != os.path.realpath(original_executable),
                "E_SELFTEST",
                "symlinked invocation executable was dereferenced",
            )

        direct_executable = temp / "direct-python"
        direct_executable.write_text(
            "#!/bin/sh\nexec " + shlex.quote(invocation_executable) + ' "$@"\n',
            encoding="utf-8",
        )
        direct_executable.chmod(0o700)
        require(
            not direct_executable.is_symlink(),
            "E_SELFTEST",
            "direct executable fixture is a symlink",
        )
        try:
            sys.executable = str(direct_executable)
            prove_executable_provenance(str(direct_executable))
        finally:
            sys.executable = original_executable

        absent = build_snapshot(str(basic), basic_head, "selftest", [b"missing.py"])
        require(
            absent["blockers"]
            == [{"path": "missing.py", "category": "selected_path_absent"}],
            "E_SELFTEST",
            "absent selection blocker",
        )
        require(absent["entries"] == [], "E_SELFTEST", "absent selection entries")
        require(
            absent["aggregate"] == {"status": "not_run_selected_scope"},
            "E_SELFTEST",
            "absent selection aggregate",
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
            non_utf8["blockers"]
            == [
                {
                    "path_b64": base64.b64encode(b"bad-\xff.py").decode("ascii"),
                    "category": "non_utf8_path",
                }
            ],
            "E_SELFTEST",
            "non-UTF-8 blocker",
        )
        require(
            non_utf8["entries"]
            == [
                {
                    "path_b64": base64.b64encode(b"bad-\xff.py").decode("ascii"),
                    "mode": "100644",
                    "blob_id": "0" * 40,
                    "result": "blocked",
                }
            ],
            "E_SELFTEST",
            "non-UTF-8 entry",
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
            malformed_snapshot["blockers"]
            == [
                {
                    "path": "clean.py",
                    "category": "ruff_nonformatter_exit",
                    "exit_code": 2,
                    "stdout": "",
                    "stderr": malformed_snapshot["blockers"][0]["stderr"],
                }
            ],
            "E_SELFTEST",
            "malformed-config blocker",
        )
        require(
            malformed_snapshot["entries"][0]["result"] == "blocked"
            and malformed_snapshot["entries"][0]["exit_code"] == 2,
            "E_SELFTEST",
            "malformed-config entry",
        )

        def nonformatter_runner(
            argv: tuple[str, ...],
            cwd: Path,
        ) -> subprocess.CompletedProcess[bytes]:
            if "format" in argv and argv[-1] == "./clean.py":
                return subprocess.CompletedProcess(
                    argv, 2, b"", b"injected Ruff failure"
                )
            return run(argv, cwd)

        injected = build_snapshot(
            str(basic),
            basic_head,
            "selftest",
            [b"clean.py"],
            runner=nonformatter_runner,
        )
        require(
            injected["blockers"]
            == [
                {
                    "path": "clean.py",
                    "category": "ruff_nonformatter_exit",
                    "exit_code": 2,
                    "stdout": "",
                    "stderr": "injected Ruff failure",
                }
            ],
            "E_SELFTEST",
            "injected nonformatter blocker",
        )
        require(
            injected["entries"][0]["result"] == "blocked"
            and injected["entries"][0]["exit_code"] == 2,
            "E_SELFTEST",
            "injected nonformatter entry",
        )

        fake_ok = subprocess.CompletedProcess(("ruff",), 0, b"ruff 0.15.22\n", b"")
        fake_bad = subprocess.CompletedProcess(("ruff",), 0, b"ruff 0.15.21\n", b"")
        fake_missing = subprocess.CompletedProcess(("ruff",), 1, b"", b"no Ruff")
        expect_error(
            "E_PYTHON_EXECUTABLE",
            lambda: require_python_executable("relative/python"),
        )
        non_executable = temp / "non-executable-python"
        non_executable.write_text("not executable\n", encoding="utf-8")
        non_executable.chmod(0o600)
        expect_error(
            "E_PYTHON_EXECUTABLE",
            lambda: require_python_executable(str(non_executable)),
        )
        expect_error(
            "E_PYTHON_VERSION", lambda: require_toolchain((3, 12, 10), fake_ok)
        )
        expect_error(
            "E_RUFF_VERSION", lambda: require_toolchain(EXPECTED_PYTHON, fake_bad)
        )
        expect_error(
            "E_RUFF_VERSION", lambda: require_toolchain(EXPECTED_PYTHON, fake_missing)
        )
        expect_error(
            "E_REVISION",
            lambda: build_snapshot(str(basic), "not-a-full-sha", "selftest"),
        )

        mismatch_blockers: list[dict[str, object]] = []
        mismatch_aggregate = aggregate_blocker(
            [{"result": "would_reformat"}],
            0,
            mismatch_blockers,
        )
        require(
            mismatch_blockers
            == [
                {
                    "category": "aggregate_mismatch",
                    "aggregate_exit": 0,
                    "per_path_failure_count": 1,
                }
            ],
            "E_SELFTEST",
            "aggregate mismatch blocker",
        )
        require(
            mismatch_aggregate == {"exit_code": 0},
            "E_SELFTEST",
            "aggregate mismatch exit",
        )

        require(snapshot_exit_code(snapshot) == 0, "E_SELFTEST", "clean CLI exit")
        for negative, detail in (
            (absent, "absent CLI exit"),
            (non_utf8, "non-UTF-8 CLI exit"),
            (malformed_snapshot, "malformed CLI exit"),
            (injected, "injected CLI exit"),
            ({"blockers": mismatch_blockers}, "aggregate mismatch CLI exit"),
        ):
            require(snapshot_exit_code(negative) == 2, "E_SELFTEST", detail)

        def excludes_128_runner(
            argv: tuple[str, ...],
            cwd: Path,
        ) -> subprocess.CompletedProcess[bytes]:
            if argv == ("git", "config", "--local", "--get", "core.excludesFile"):
                return subprocess.CompletedProcess(
                    argv, 128, b"", b"injected config failure"
                )
            return run(argv, cwd)

        expect_error(
            "E_EXTERNAL_EXCLUDES",
            lambda: require_clean_checkout(basic, runner=excludes_128_runner),
        )

        hostile = {
            "GIT_DIR": str(temp / "not-a-git-dir"),
            "GIT_WORK_TREE": str(temp / "not-a-work-tree"),
            "GIT_INDEX_FILE": str(temp / "not-an-index"),
            "GIT_CONFIG_GLOBAL": str(temp / "hostile.gitconfig"),
        }
        saved = {key: os.environ.get(key) for key in hostile}
        os.environ.update(hostile)
        try:
            hostile_snapshot = build_snapshot(
                str(basic), basic_head, "selftest", [b"clean.py"]
            )
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        require(
            hostile_snapshot["blockers"] == [],
            "E_SELFTEST",
            "hostile Git environment blockers",
        )
        require(
            hostile_snapshot["entries"][0]["result"] == "not_failing"
            and hostile_snapshot["entries"][0]["exit_code"] == 0,
            "E_SELFTEST",
            "hostile Git environment entry",
        )

        def wrong_root_runner(
            argv: tuple[str, ...],
            cwd: Path,
        ) -> subprocess.CompletedProcess[bytes]:
            if argv == ("git", "rev-parse", "--show-toplevel"):
                return subprocess.CompletedProcess(argv, 0, b"/wrong/root\\n", b"")
            return run(argv, cwd)

        expect_error(
            "E_CHECKOUT_ROOT",
            lambda: build_snapshot(
                str(basic), basic_head, "selftest", runner=wrong_root_runner
            ),
        )

        atomic_target = temp / "atomic.json"
        directory_syncs: list[Path] = []
        atomic_write_json(
            atomic_target,
            {"z": [3, 2, 1], "a": True},
            token_factory=lambda: "published",
            directory_sync=lambda path: directory_syncs.append(path),
        )
        expected_atomic = (
            json.dumps(
                {"z": [3, 2, 1], "a": True},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        require(
            atomic_target.read_bytes() == expected_atomic,
            "E_SELFTEST",
            "atomic success bytes",
        )
        require(
            atomic_target.stat().st_mode & 0o777 == 0o600,
            "E_SELFTEST",
            "atomic success mode",
        )
        require(
            directory_syncs == [temp],
            "E_SELFTEST",
            "atomic success directory fsync",
        )
        require(
            not list(temp.glob(".atomic.json.task26000-*")),
            "E_SELFTEST",
            "atomic success sibling residue",
        )
        atomic_target.write_text("old\\n", encoding="utf-8")

        def fail_write(descriptor: int, payload: bytes) -> None:
            raise OSError("injected write failure")

        expect_error(
            "E_ATOMIC_WRITE",
            lambda: atomic_write_json(atomic_target, {"ok": True}, writer=fail_write),
        )
        require(
            atomic_target.read_text(encoding="utf-8") == "old\\n",
            "E_SELFTEST",
            "write failure changed output",
        )
        require(
            not list(temp.glob(".atomic.json.task26000-*")),
            "E_SELFTEST",
            "write failure leaked owned sibling",
        )

        def fail_file_sync(descriptor: int) -> None:
            raise OSError("injected file fsync failure")

        expect_error(
            "E_ATOMIC_WRITE",
            lambda: atomic_write_json(
                atomic_target,
                {"ok": True},
                file_sync=fail_file_sync,
            ),
        )
        require(
            atomic_target.read_text(encoding="utf-8") == "old\\n",
            "E_SELFTEST",
            "file fsync failure changed output",
        )
        require(
            not list(temp.glob(".atomic.json.task26000-*")),
            "E_SELFTEST",
            "file fsync failure leaked owned sibling",
        )

        def fail_replace(
            source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        ) -> None:
            raise OSError("injected interruption")

        expect_error(
            "E_ATOMIC_WRITE",
            lambda: atomic_write_json(
                atomic_target, {"ok": True}, replacer=fail_replace
            ),
        )
        require(
            atomic_target.read_text(encoding="utf-8") == "old\\n",
            "E_SELFTEST",
            "atomic failure changed output",
        )
        require(
            not list(temp.glob(".atomic.json.task26000-*")),
            "E_SELFTEST",
            "owned atomic temp leaked after failure",
        )

        def replace_then_fail(
            source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        ) -> None:
            source_path = Path(source)
            source_path.unlink()
            source_path.write_text("replacement\\n", encoding="utf-8")
            raise OSError("injected substitution")

        expect_error(
            "E_ATOMIC_WRITE",
            lambda: atomic_write_json(
                atomic_target,
                {"ok": True},
                token_factory=lambda: "substituted",
                replacer=replace_then_fail,
            ),
        )
        substituted = temp / ".atomic.json.task26000-substituted"
        require(
            substituted.read_text(encoding="utf-8") == "replacement\\n",
            "E_SELFTEST",
            "unowned atomic temp was removed",
        )
        substituted.unlink()
    print("census self-tests: 20 cases passed")


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
    require(
        all((args.checkout, args.revision, args.label, args.output)),
        "E_ARGS",
        "missing required argument",
    )
    checkout = Path(args.checkout).resolve()
    output = Path(args.output).resolve()
    require(
        not output.is_relative_to(checkout),
        "E_OUTPUT_SCOPE",
        "output must be outside checkout",
    )
    selected = read_paths0(Path(args.paths0)) if args.paths0 else None
    snapshot = build_snapshot(args.checkout, args.revision, args.label, selected)
    atomic_write_json(output, snapshot)
    return snapshot_exit_code(snapshot)


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
`census self-tests: 20 cases passed` only after clean/fail/excluded,
dash/space/newline, non-UTF-8, absent-selection, malformed-config/nonformatter,
direct and symlinked absolute invocation-path replay, relative/non-executable interpreter rejection,
tool-version/no-Ruff, aggregate-mismatch, abnormal `core.excludesFile`, hostile Git
environment, checkout-root, and atomic-output ownership assertions all pass. The
negative snapshots assert their exact blocker data and `snapshot_exit_code()` is the
CLI-equivalent exit-code helper used by `main`; it returns 2 for every blocked
snapshot. Git subprocesses remove caller `GIT_*` and `RUFF_*` variables before adding
the fixed config environment, and `build_snapshot()` requires `--show-toplevel` to
equal the resolved checkout. `atomic_write_json()` writes a 128-bit random,
owner-created sibling, fsyncs it, atomically replaces the output, fsyncs the parent,
and only unlinks a failed temporary when its device/inode still match the file it
created. The expanded atomic probes assert canonical successful bytes, final mode
0600, deterministic parent-sync invocation, no sibling residue, and cleanup after
injected write or file-sync failure.

---

## Appendix A.1: Exact Durable Manifest Producer

Task 4 and any separately owner-approved manifest rebuild materialize this complete producer source through Appendix B.1's authority tool. The approved Task 7 authority-cut workflow does not rebuild or refresh the manifest. The marker digest is authoritative; private temporary copies are never sources of truth.

<!-- TASK-26000-PRODUCER-SOURCE-BEGIN sha256=fd33448f2841d0502509201a5bf6fd2f279f3f2c67cff8f3d4391b9ed7d9ce3e -->
```python
from __future__ import annotations

import argparse
import atexit
import datetime as dt
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


LABELS = ("base", "pre_closeout", "closeout", "common", "current")
TMP = Path("/tmp/task26000.b0z8M0")
RAW = TMP / "raw"
REPO = Path("/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/track-ruff-format-debt")
MANIFEST = REPO / "Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json"
TASK_BASE = "e555df102c950c29beed5e7119f433d35eee1f3c"
RECENT_SINCE = "2026-08-16T00:00:00Z"
LEGACY_OPEN_PRS = [
    {
        "number": 2244,
        "head": "codex/task-23026-trace-ledger-plan",
        "title": "Implement reference-backed semantic trace ledger lifecycle",
        "paths": [
            "Docs/superpowers/qa/2026-08-15-local-thinking-controls-live-verification/live-verify.py",
            "Packaging/check_manifest.py",
            "Tests/ChaChaNotesDB/test_index_census.py",
            "Tests/Chat/test_assistant_generation_state_roundtrip.py",
            "Tests/Chat/test_console_agent_bridge.py",
            "Tests/Chat/test_console_chat_controller.py",
            "Tests/Chat/test_console_context_compaction.py",
            "Tests/Chat/test_console_ephemeral.py",
            "Tests/Chat/test_console_exchange_capture.py",
            "Tests/Chat/test_console_prepared_request.py",
            "Tests/Chat/test_console_provider_continuation.py",
            "Tests/Chat/test_console_provider_gateway.py",
            "Tests/Chat/test_console_terminal_citation_persistence.py",
            "Tests/Chat/test_console_video_actions.py",
            "Tests/Chat/test_console_visual_evaluation.py",
            "Tests/DB/test_chachanotes_message_exchanges.py",
            "Tests/DB/test_chachanotes_v55_console_memory_selection_migration.py",
            "Tests/Packaging/test_installed_distribution.py",
            "Tests/Research/test_chat_handoff.py",
            "Tests/Sync_Interop/test_provider_continuation_reconciliation.py",
            "Tests/UI/test_chat_screen_console_inspector_loader.py",
            "tldw_chatbook/Agents/agent_service.py",
            "tldw_chatbook/Chat/Chat_Functions.py",
            "tldw_chatbook/Chat/chat_persistence_service.py",
            "tldw_chatbook/Chat/console_agent_bridge.py",
            "tldw_chatbook/Chat/console_chat_store.py",
            "tldw_chatbook/Chat/console_context_compaction.py",
            "tldw_chatbook/Chat/console_conversation_hydration.py",
            "tldw_chatbook/Chat/console_exchange_capture.py",
            "tldw_chatbook/Chat/console_provider_gateway.py",
        ],
    },
    {
        "number": 2230,
        "head": "fix/console-voice-chip-width",
        "title": "Console voice chip width",
        "paths": ["Tests/UI/test_console_dictation_streaming.py"],
    },
    {
        "number": 2196,
        "head": "perf/burndown-0828",
        "title": "Console keystroke and boot work",
        "paths": [
            "Tests/Performance/test_boot_worker_census.py",
            "tldw_chatbook/Chat/console_chat_store.py",
            "tldw_chatbook/Chunking/engine/strategies/semantic.py",
            "tldw_chatbook/Chunking/engine/strategies/tokens.py",
            "tldw_chatbook/Subscriptions/briefing_audio.py",
            "tldw_chatbook/UI/Console_Modules/prompt_queue.py",
            "tldw_chatbook/UI/Console_Modules/workspace.py",
            "tldw_chatbook/UI/MCP_Modules/mcp_workbench.py",
            "tldw_chatbook/UI/Screens/chat_screen.py",
            "tldw_chatbook/__init__.py",
            "tldw_chatbook/config.py",
        ],
    },
    {
        "number": 2059,
        "head": "fix/task-21969-test-workflow-pr-cancellation",
        "title": "Pull-request workflow cancellation",
        "paths": ["Tests/CI/test_github_actions_test_workflow.py"],
    },
    {
        "number": 1655,
        "head": "codex/task-13208-windows-audio-cpp",
        "title": "Windows audio.cpp lifecycle parity",
        "paths": [
            "Tests/CI/test_github_actions_test_workflow.py",
            "Tests/UI/test_speech_playground_pane_lifecycle.py",
        ],
    },
]

PR_SNAPSHOT_PATH = TMP / "open-pr-snapshot.json"
TEMPORAL_CACHE_PATH = TMP / f"temporal-replay-cache.{TASK_BASE}.json"
TEMPORAL_CACHE_VERSION = "first-valid-failure-v5-authenticated-alias-syntax-ledger"
TEMPORAL_CACHE_SCHEMA = 3
_ANCESTOR_CACHE: dict[tuple[str, str, str], bool] = {}
_REV_LIST_CACHE: dict[tuple[str, str, str, str], list[str]] = {}


def producer_source_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def temporal_cache_fingerprint(
    *,
    common: str,
    current: str,
    resolved_python: str,
    python_version: str,
    ruff_version: str,
    current_census_sha256: str,
) -> dict[str, Any]:
    return {
        "algorithm_version": TEMPORAL_CACHE_VERSION,
        "schema_version": TEMPORAL_CACHE_SCHEMA,
        "common": common,
        "current": current,
        "resolved_python": resolved_python,
        "python_version": python_version,
        "ruff_version": ruff_version,
        "current_census_sha256": current_census_sha256,
        "producer_source_sha256": producer_source_digest(),
    }


def load_temporal_cache(fingerprint: dict[str, Any]) -> dict[str, Any]:
    if TEMPORAL_CACHE_PATH.exists():
        try:
            raw = TEMPORAL_CACHE_PATH.read_bytes()
            data = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {"fingerprint": fingerprint, "rows": {}}
        if (
            isinstance(data, dict)
            and data.get("fingerprint") == fingerprint
            and isinstance(data.get("rows"), dict)
            and raw == canonical_bytes(data)
        ):
            return {"fingerprint": fingerprint, "rows": data["rows"]}
    return {"fingerprint": fingerprint, "rows": {}}


def write_temporal_cache(cache: dict[str, Any]) -> None:
    encoded = json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = TEMPORAL_CACHE_PATH.with_suffix(".json.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(TEMPORAL_CACHE_PATH)


def load_open_pr_snapshot() -> tuple[str, int, list[dict[str, Any]]]:
    data = json.loads(PR_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    captured_at = data.get("captured_at_utc")
    per_page = data.get("per_page")
    prs = data.get("prs")
    if not isinstance(captured_at, str) or not captured_at:
        raise RuntimeError("missing open PR capture timestamp")
    if per_page != 100 or not isinstance(prs, list) or not prs:
        raise RuntimeError("invalid open PR snapshot control")
    numbers: set[int] = set()
    for pr in prs:
        required = {
            "number",
            "base",
            "base_sha",
            "head",
            "head_sha",
            "title",
            "paths",
            "file_count",
            "page_counts",
            "head_object_present",
            "base_object_present",
            "local_diff_match",
            "open_at_capture",
        }
        if not isinstance(pr, dict) or not required <= set(pr):
            raise RuntimeError("incomplete open PR snapshot row")
        number = pr["number"]
        if not isinstance(number, int) or number in numbers:
            raise RuntimeError(f"duplicate or invalid open PR number: {number}")
        numbers.add(number)
        paths = pr["paths"]
        if paths != sorted(set(paths)) or pr["file_count"] != len(paths):
            raise RuntimeError(f"invalid open PR file census: {number}")
        page_counts = pr["page_counts"]
        if not isinstance(page_counts, list) or sum(page_counts) != len(paths):
            raise RuntimeError(f"invalid open PR page census: {number}")
        if any(count != per_page for count in page_counts[:-1]):
            raise RuntimeError(f"short non-final open PR page: {number}")
        if (
            not pr["head_object_present"]
            or not pr["base_object_present"]
            or pr["local_diff_match"] is not True
        ):
            raise RuntimeError(f"open PR lacks exact local diff proof: {number}")
        if pr["open_at_capture"] is not True:
            raise RuntimeError(f"PR was not open at capture: {number}")
    return captured_at, per_page, sorted(prs, key=lambda row: row["number"], reverse=True)


PR_CAPTURED_AT, PR_PER_PAGE, OPEN_PRS = load_open_pr_snapshot()
PR_BY_NUMBER = {pr["number"]: pr for pr in OPEN_PRS}


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git(repo: Path, *argv: str, binary: bool = False) -> bytes | str:
    completed = subprocess.run(
        ("git", *argv), cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.decode("utf-8", "backslashreplace").strip())
    return completed.stdout if binary else completed.stdout.decode("utf-8").strip()


def transition_kind(
    source_path: str | None,
    target_path: str | None,
    source_blob: str | None,
    target_blob: str | None,
) -> tuple[str, str]:
    if source_path is None and target_path is None:
        return "absent", "absent"
    if source_path is None:
        return "add", "A"
    if target_path is None:
        return "delete", "D"
    if source_path != target_path:
        return "rename", "R100"
    if source_blob == target_blob:
        return "unchanged", "same"
    return "modify", "M"


def lineage_row(
    revisions: dict[str, str],
    source: str,
    target: str,
    paths: dict[str, str | None],
    blobs: dict[str, str | None],
    *,
    git_status: str | None = None,
    commits: list[str] | None = None,
    rationale: str,
) -> dict[str, Any]:
    kind, derived_status = transition_kind(
        paths[source], paths[target], blobs[source], blobs[target]
    )
    return {
        "kind": kind,
        "source_revision": revisions[source],
        "target_revision": revisions[target],
        "source_path": paths[source],
        "target_path": paths[target],
        "source_blob": blobs[source],
        "target_blob": blobs[target],
        "git_status": git_status or derived_status,
        "follow_commits": sorted(set(commits or [revisions[target]])),
        "rationale": rationale,
    }


def entry_indexes(censuses: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    return {label: {row["path"]: row for row in censuses[label]["entries"]} for label in LABELS}


def build_identities(
    source: dict[str, Any], censuses: dict[str, Any], revisions: dict[str, str]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    entries = entry_indexes(censuses)
    result: list[dict[str, Any]] = []
    by_source_id: dict[str, dict[str, Any]] = {}
    current_owner: dict[str, str] = {}
    common_by_m: dict[str, dict[str, Any]] = {}
    for common in source["common_failure_identities"]:
        for source_id in common["m_identity_ids"]:
            common_by_m[source_id] = common

    def append_identity(
        identity_id: str,
        paths: dict[str, str | None],
        blobs: dict[str, str | None],
        *,
        historical_status: str | None = None,
        common_source: dict[str, Any] | None = None,
    ) -> None:
        common_commits: list[str] = []
        if common_source is not None:
            inspection = common_source["lineage"].get("inspection", {})
            common_commits = inspection.get("interval_commit_oids", [])
        lineage = {
            "base_to_pre": lineage_row(
                revisions,
                "base",
                "pre_closeout",
                paths,
                blobs,
                git_status=historical_status,
                rationale="TASK-22514 historical diff identity projection.",
            ),
            "pre_to_closeout": lineage_row(
                revisions,
                "pre_closeout",
                "closeout",
                paths,
                blobs,
                rationale="TASK-22514 pre-closeout to closeout projection.",
            ),
            "common_to_current": lineage_row(
                revisions,
                "common",
                "current",
                paths,
                blobs,
                commits=common_commits or None,
                rationale="Current-line projection from the branches' common ancestor; closeout is not used as a temporal ancestor.",
            ),
        }
        row = {"id": identity_id, "paths": paths, "blobs": blobs, "lineage": lineage}
        result.append(row)
        current_path = paths["current"]
        if current_path is not None:
            if current_path in current_owner:
                raise RuntimeError(f"duplicate current owner: {current_path}")
            current_owner[current_path] = identity_id

    for historical in source["identities"]:
        identity_id = historical["identity_id"]
        paths = dict(historical["paths"])
        blobs = dict(historical["blob_ids"])
        append_identity(
            identity_id,
            paths,
            blobs,
            historical_status=historical["change"],
            common_source=common_by_m.get(identity_id),
        )
        by_source_id[identity_id] = result[-1]

    next_number = len(result)
    for common in source["common_failure_identities"]:
        if common["m_identity_ids"]:
            continue
        identity_id = f"I-{next_number:04d}"
        next_number += 1
        paths = {label: None for label in LABELS}
        blobs = {label: None for label in LABELS}
        paths.update(common["paths"])
        blobs.update(common["blob_ids"])
        append_identity(identity_id, paths, blobs, common_source=common)

    current_failures = {
        row["path"]
        for row in censuses["current"]["entries"]
        if row["result"] == "would_reformat"
    }
    for current_path in sorted(current_failures - set(current_owner)):
        identity_id = f"I-{next_number:04d}"
        next_number += 1
        paths = {label: None for label in LABELS}
        blobs = {label: None for label in LABELS}
        paths["current"] = current_path
        blobs["current"] = entries["current"][current_path]["blob_id"]
        if current_path in entries["common"]:
            paths["common"] = current_path
            blobs["common"] = entries["common"][current_path]["blob_id"]
        append_identity(identity_id, paths, blobs)

    result.sort(key=lambda row: row["id"])
    return result, current_owner


def collect_paginated_pr_files(
    fetch_page: Callable[[int, int], list[dict[str, Any]]], *, per_page: int = 100
) -> dict[str, Any]:
    if per_page <= 0:
        raise RuntimeError("invalid PR page size")
    paths: list[str] = []
    page_counts: list[int] = []
    seen: set[str] = set()
    page = 1
    while True:
        rows = fetch_page(page, per_page)
        if not isinstance(rows, list):
            raise RuntimeError(f"invalid PR files page {page}")
        page_counts.append(len(rows))
        for row in rows:
            path = row.get("filename") if isinstance(row, dict) else None
            if not isinstance(path, str) or not path:
                raise RuntimeError(f"invalid PR file path on page {page}")
            if path in seen:
                raise RuntimeError(f"duplicate PR file path across pages: {path}")
            seen.add(path)
            paths.append(path)
        if len(rows) < per_page:
            break
        page += 1
    return {
        "paths": sorted(paths),
        "file_count": len(paths),
        "page_counts": page_counts,
    }


def git_exists(repo: Path, object_name: str) -> bool:
    completed = subprocess.run(
        ("git", "cat-file", "-e", object_name),
        cwd=repo,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def commit_parents(repo: Path, commit: str) -> list[str]:
    return str(git(repo, "show", "-s", "--format=%P", commit)).split()


def is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    cache_key = (str(repo.resolve()), ancestor, descendant)
    if cache_key in _ANCESTOR_CACHE:
        return _ANCESTOR_CACHE[cache_key]
    completed = subprocess.run(
        ("git", "merge-base", "--is-ancestor", ancestor, descendant),
        cwd=repo,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError(f"cannot compare commit topology: {ancestor} -> {descendant}")
    result = completed.returncode == 0
    _ANCESTOR_CACHE[cache_key] = result
    return result


def cached_revision_list(
    repo: Path, common: str, current: str, mode: str
) -> list[str]:
    cache_key = (str(repo.resolve()), common, current, mode)
    if cache_key not in _REV_LIST_CACHE:
        if mode == "topology":
            argv = ("rev-list", "--reverse", "--topo-order", f"{common}..{current}")
        elif mode == "first-parent":
            argv = ("rev-list", "--first-parent", "--reverse", f"{common}..{current}")
        else:
            raise ValueError(f"unknown revision-list mode: {mode}")
        _REV_LIST_CACHE[cache_key] = str(git(repo, *argv)).splitlines()
    return _REV_LIST_CACHE[cache_key]


def first_integration_index(
    repo: Path, commit: str, first_parent: list[str]
) -> int:
    """Return the earliest first-parent state containing commit in O(log n) checks."""
    if not first_parent or not is_ancestor(repo, commit, first_parent[-1]):
        raise RuntimeError(f"no first-parent integration interval for {commit}")
    lower = 0
    upper = len(first_parent) - 1
    while lower < upper:
        middle = (lower + upper) // 2
        if is_ancestor(repo, commit, first_parent[middle]):
            upper = middle
        else:
            lower = middle + 1
    return lower


def config_paths_for(path: str) -> list[str]:
    parent = Path(path).parent
    directories = [Path(".")]
    if parent != Path("."):
        current = Path()
        for part in parent.parts:
            current /= part
            directories.append(current)
    result: list[str] = []
    for directory in directories:
        for name in ("pyproject.toml", "ruff.toml", ".ruff.toml"):
            candidate = str(directory / name)
            if candidate.startswith("./"):
                candidate = candidate[2:]
            result.append(candidate)
    return sorted(set(result))


def nul_commits(repo: Path, *argv: str) -> list[str]:
    raw = git(repo, *argv, binary=True)
    assert isinstance(raw, bytes)
    values = [value.strip() for value in raw.split(b"\0") if value.strip()]
    try:
        commits = [value.decode("ascii") for value in values]
    except UnicodeDecodeError as exc:
        raise RuntimeError("non-ASCII commit id in NUL-delimited Git output") from exc
    if any(len(value) != 40 for value in commits):
        raise RuntimeError("invalid commit id in NUL-delimited Git output")
    return commits


def name_status_z(repo: Path, parent: str, commit: str) -> list[tuple[str, list[str]]]:
    raw = git(
        repo, "diff", "--name-status", "-z", "-M", parent, commit, "--", binary=True
    )
    assert isinstance(raw, bytes)
    if raw and not raw.endswith(b"\0"):
        raise RuntimeError("unterminated NUL-delimited Git path output")
    tokens = raw[:-1].split(b"\0") if raw else []
    result: list[tuple[str, list[str]]] = []
    index = 0
    while index < len(tokens):
        status = tokens[index].decode("ascii")
        index += 1
        count = 2 if status.startswith(("R", "C")) else 1
        if index + count > len(tokens):
            raise RuntimeError(f"truncated Git name-status record: {status}")
        paths = [token.decode("utf-8") for token in tokens[index:index + count]]
        result.append((status, paths))
        index += count
    return result


def rename_lineage_paths(
    repo: Path, common: str, current: str, current_path: str
) -> tuple[set[str], dict[str, tuple[str, str]]]:
    commits = nul_commits(
        repo, "log", "--topo-order", "--format=%H%x00", "--follow",
        f"{common}..{current}", "--", current_path,
    )
    tracked = current_path
    paths = {current_path}
    renames: dict[str, tuple[str, str]] = {}
    for commit in commits:
        parents = commit_parents(repo, commit)
        transitions: set[tuple[str, str]] = set()
        for parent in parents:
            for status, fields in name_status_z(repo, parent, commit):
                if status.startswith("R") and len(fields) == 2:
                    source_path, target_path = fields
                    if target_path == tracked:
                        transitions.add((source_path, target_path))
        if len(transitions) > 1:
            raise RuntimeError(
                f"ambiguous rename ancestry for {current_path} at {commit}: {sorted(transitions)}"
            )
        if transitions:
            source_path, target_path = next(iter(transitions))
            paths.update((source_path, target_path))
            renames[commit] = (source_path, target_path)
            tracked = source_path
    return paths, renames


def unique_merge_transition(
    current_path: str, transitions: set[tuple[str, str]]
) -> tuple[str, str] | None:
    relevant = {transition for transition in transitions if transition[1] == current_path}
    if len(relevant) > 1:
        raise RuntimeError(
            f"ambiguous merge/rename chronology for {current_path}: {sorted(relevant)}"
        )
    return next(iter(relevant)) if relevant else None


def all_parent_path_transition(
    repo: Path, commit: str, path: str
) -> str | None:
    """Derive a candidate's path cause from every parent with NUL-safe plumbing."""
    parents = commit_parents(repo, commit)
    if not parents:
        raise RuntimeError(f"root temporal candidate: {commit}")
    rename_sources: set[str] = set()
    path_changed = False
    for parent in parents:
        for status, fields in name_status_z(repo, parent, commit):
            if status.startswith("R") and len(fields) == 2 and fields[1] == path:
                rename_sources.add(fields[0])
                path_changed = True
            elif (
                len(fields) == 1
                and fields[0] == path
                and status[:1] in {"A", "M"}
            ):
                path_changed = True
    if len(rename_sources) > 1:
        raise RuntimeError(
            f"ambiguous all-parent rename chronology for {path} at {commit}: "
            f"{sorted(rename_sources)}"
        )
    if rename_sources:
        return "path_rename"
    if not path_changed:
        return None
    return (
        "path_modify"
        if any(git_exists(repo, f"{parent}:{path}") for parent in parents)
        else "path_add"
    )


def relevant_state_candidates(
    repo: Path, common: str, current: str, current_path: str
) -> tuple[list[dict[str, Any]], list[str]]:
    path_commits = nul_commits(
        repo, "log", "--topo-order", "--format=%H%x00", "--follow",
        f"{common}..{current}", "--", current_path,
    )
    tracked = current_path
    lineage_paths = {current_path}
    path_events: dict[str, dict[str, str]] = {}
    for commit in path_commits:
        parents = commit_parents(repo, commit)
        if not parents:
            continue
        transitions: set[tuple[str, str]] = set()
        for parent in parents:
            for status, fields in name_status_z(repo, parent, commit):
                if status.startswith("R") and len(fields) == 2:
                    transitions.add((fields[0], fields[1]))
        transition = unique_merge_transition(tracked, transitions)
        target_path = tracked
        if transition is not None:
            source_path, target_path = transition
            lineage_paths.update((source_path, target_path))
            kind = "path_rename"
            tracked = source_path
        else:
            existed = any(git_exists(repo, f"{parent}:{target_path}") for parent in parents)
            kind = "path_modify" if existed else "path_add"
        if git_exists(repo, f"{commit}:{target_path}"):
            path_events[commit] = {
                "commit": commit,
                "path": target_path,
                "kind": kind,
                "causes": [kind],
            }
    config_paths = sorted(
        set().union(*(set(config_paths_for(path)) for path in lineage_paths))
    )
    raw_config_commits = nul_commits(
        repo, "log", "--full-history", "--topo-order", "--format=%H%x00",
        f"{common}..{current}", "--", *config_paths,
    )
    config_events: dict[str, dict[str, str]] = {}
    for commit in raw_config_commits:
        relevant = False
        for config_path in config_paths:
            before_values = []
            for parent in commit_parents(repo, commit):
                before_values.append(
                    git(repo, "show", f"{parent}:{config_path}", binary=True)
                    if git_exists(repo, f"{parent}:{config_path}")
                    else b""
                )
            after = (
                git(repo, "show", f"{commit}:{config_path}", binary=True)
                if git_exists(repo, f"{commit}:{config_path}")
                else b""
            )
            if not any(before != after for before in before_values):
                continue
            if config_path.endswith(("ruff.toml", ".ruff.toml")) or b"[tool.ruff" in b"".join(before_values) + after:
                relevant = True
                break
        if not relevant:
            continue
        present_paths = sorted(
            path for path in lineage_paths if git_exists(repo, f"{commit}:{path}")
        )
        if not present_paths:
            continue
        if len(present_paths) != 1:
            raise RuntimeError(
                f"ambiguous Ruff-config path chronology for {current_path} at {commit}: {present_paths}"
            )
        config_events[commit] = {
            "commit": commit,
            "path": present_paths[0],
            "kind": "ruff_config",
            "causes": ["ruff_config"],
        }

    exclusion_specs = [".gitignore", ".ignore", ":(glob)**/.gitignore", ":(glob)**/.ignore"]
    exclusion_commits = nul_commits(
        repo, "log", "--full-history", "--topo-order", "--format=%H%x00",
        f"{common}..{current}", "--", *exclusion_specs,
    )
    exclusion_events: dict[str, dict[str, Any]] = {}
    for commit in exclusion_commits:
        present_paths = sorted(
            path for path in lineage_paths if git_exists(repo, f"{commit}:{path}")
        )
        if len(present_paths) > 1:
            raise RuntimeError(
                f"ambiguous exclusion path chronology for {current_path} at {commit}: {present_paths}"
            )
        if present_paths:
            exclusion_events[commit] = {
                "commit": commit,
                "path": present_paths[0],
                "kind": "exclusion_change",
                "causes": ["exclusion_change"],
            }
    topology = cached_revision_list(repo, common, current, "topology")
    topology_index = {commit: index for index, commit in enumerate(topology)}
    first_parent = cached_revision_list(repo, common, current, "first-parent")
    events: dict[str, dict[str, Any]] = {}
    for source in (config_events, exclusion_events, path_events):
        for commit, event in source.items():
            if commit not in events:
                events[commit] = event
            else:
                events[commit]["causes"] = sorted(
                    set(events[commit]["causes"]) | set(event["causes"])
                )
                if event["kind"].startswith("path_"):
                    events[commit]["kind"] = event["kind"]
                    events[commit]["path"] = event["path"]
    for commit, event in events.items():
        path_cause = all_parent_path_transition(repo, commit, event["path"])
        if path_cause is not None:
            event["causes"] = sorted(set(event["causes"]) | {path_cause})
            event["kind"] = path_cause
    candidates: list[dict[str, Any]] = []
    for commit, event in events.items():
        integration_index = first_integration_index(repo, commit, first_parent)
        candidates.append(
            {
                **event,
                "integration_index": integration_index,
                "integration_commit": first_parent[integration_index],
                "topology_index": topology_index[commit],
            }
        )
    candidates.sort(
        key=lambda row: (row["integration_index"], topology_index[row["commit"]])
    )
    if not candidates:
        raise RuntimeError(f"no relevant current-line state candidates for {current_path}")
    return candidates, config_paths


def tracked_inventory(root: Path) -> dict[str, dict[str, str]]:
    raw = subprocess.run(
        ("git", "ls-files", "-s", "-z"), cwd=root, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    if raw.returncode or (raw.stdout and not raw.stdout.endswith(b"\0")):
        raise RuntimeError(raw.stderr.decode("utf-8", "backslashreplace").strip() or "invalid ls-files output")
    inventory: dict[str, dict[str, str]] = {}
    for token in raw.stdout[:-1].split(b"\0") if raw.stdout else []:
        meta, raw_path = token.split(b"\t", 1)
        mode, blob, stage = meta.decode("ascii").split(" ")
        if stage != "0":
            raise RuntimeError("non-stage-zero entry in detached replay checkout")
        path = raw_path.decode("utf-8")
        inventory[path] = {"path": path, "mode": mode, "blob_id": blob}
    return inventory


def is_ruff_config(root: Path, path: str) -> bool:
    name = Path(path).name
    if name in {"ruff.toml", ".ruff.toml"}:
        return True
    return name == "pyproject.toml" and b"[tool.ruff" in (root / path).read_bytes()


class ReplayWorkspace:
    """One sanitized shared-object clone reused for all revision-local Ruff states."""

    def __init__(self, repo: Path) -> None:
        self.repo = repo.resolve()
        self.parent = Path(tempfile.mkdtemp(prefix="temporal-replay-v4-", dir=TMP))
        self.root = self.parent / "checkout"
        self.home = self.parent / "home"
        self.home.mkdir()
        completed = subprocess.run(
            ("git", "clone", "-q", "--shared", "--no-checkout", str(self.repo), str(self.root)),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        if completed.returncode:
            shutil.rmtree(self.parent, ignore_errors=True)
            raise RuntimeError(completed.stderr.decode("utf-8", "backslashreplace").strip())
        self.current: str | None = None
        self.cache: dict[tuple[str, str, str], dict[str, Any]] = {}

    def checkout(self, commit: str) -> None:
        if self.current == commit:
            return
        completed = subprocess.run(
            ("git", "checkout", "-q", "--detach", "--force", commit), cwd=self.root,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr.decode("utf-8", "backslashreplace").strip())
        resolved = subprocess.run(
            ("git", "rev-parse", "HEAD^{commit}"), cwd=self.root, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        status = subprocess.run(
            ("git", "status", "--porcelain=v1", "-z"), cwd=self.root,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        if resolved.returncode or resolved.stdout.strip() != commit or status.returncode or status.stdout:
            raise RuntimeError(f"unauthenticated or dirty replay checkout for {commit}")
        self.current = commit

    def run(self, commit: str, path: str, resolved_python: str) -> dict[str, Any]:
        key = (commit, path, resolved_python)
        if key in self.cache:
            return self.cache[key]
        self.checkout(commit)
        inventory = tracked_inventory(self.root)
        if path not in inventory:
            raise RuntimeError(f"missing replay path {commit}:{path}")
        configs = [
            row for candidate, row in sorted(inventory.items())
            if is_ruff_config(self.root, candidate)
        ]
        exclusions = [
            row for candidate, row in sorted(inventory.items())
            if Path(candidate).name in {".gitignore", ".ignore"}
        ]
        argv = [
            resolved_python, "-m", "ruff", "format", "--check",
            "--force-exclude", "--no-cache", f"./{path}",
        ]
        env = {
            "HOME": str(self.home),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONHASHSEED": "0",
        }
        completed = subprocess.run(
            argv, cwd=self.root, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False,
        )
        if completed.returncode not in {0, 1, 2}:
            detail = completed.stderr.decode("utf-8", "backslashreplace").strip()
            raise RuntimeError(f"Ruff replay failed for {commit}:{path}: {detail}")
        exit_class, invalid_reason = classify_ruff_exit(
            completed.returncode,
            (self.root / path).read_bytes(),
            path,
        )
        result = {
            "path_mode": inventory[path]["mode"],
            "path_blob": inventory[path]["blob_id"],
            "config_inputs": configs,
            "exclusion_inputs": exclusions,
            "command_argv": argv,
            "exit_code": completed.returncode,
            "exit_class": exit_class,
            "invalid_reason": invalid_reason,
            "stdout_sha256": digest(completed.stdout),
            "stderr_sha256": digest(completed.stderr),
        }
        self.cache[key] = result
        return result

    def close(self) -> None:
        shutil.rmtree(self.parent, ignore_errors=True)


_REPLAY_WORKSPACES: dict[str, ReplayWorkspace] = {}


def replay_workspace(repo: Path) -> ReplayWorkspace:
    key = str(repo.resolve())
    if key not in _REPLAY_WORKSPACES:
        _REPLAY_WORKSPACES[key] = ReplayWorkspace(repo)
    return _REPLAY_WORKSPACES[key]


def close_replay_workspaces() -> None:
    for workspace in _REPLAY_WORKSPACES.values():
        workspace.close()
    _REPLAY_WORKSPACES.clear()


atexit.register(close_replay_workspaces)


def classify_ruff_exit(
    exit_code: int,
    source: bytes,
    path: str,
) -> tuple[str, str | None]:
    """Classify Ruff only after independently authenticating an exit-2 source error."""
    if exit_code == 0:
        return "clean", None
    if exit_code == 1:
        return "failing", None
    if exit_code != 2:
        raise RuntimeError(f"unexpected Ruff exit {exit_code} for {path}")
    try:
        compile(source, path, "exec", dont_inherit=True)
    except SyntaxError:
        return "invalid", "python_syntax_error"
    raise RuntimeError(
        f"non-source-syntax Ruff exit 2 for {path}; config, invocation, or runtime failure"
    )


def ruff_state_fails(
    repo: Path, commit: str, path: str, resolved_python: str
) -> tuple[bool, list[str], str]:
    """Compatibility shim for the producer tests; durable details live in the ledger."""
    result = replay_workspace(repo).run(commit, path, resolved_python)
    configs = [row["path"] for row in result["config_inputs"]]
    status = "unparsable" if result["exit_class"] == "invalid" else "formatter-result"
    return result["exit_class"] == "failing", configs, status


def first_failing_state(
    repo: Path,
    common: str,
    current: str,
    current_path: str,
    resolved_python: str,
) -> dict[str, Any]:
    if not is_ancestor(repo, common, current):
        raise RuntimeError(f"common revision is not an ancestor of current: {common}")
    source_aliases, _rename_events = rename_lineage_paths(
        repo, common, current, current_path
    )
    candidates, _config_paths = relevant_state_candidates(
        repo, common, current, current_path
    )
    results: list[dict[str, Any]] = []
    first: dict[str, Any] | None = None
    integration_indexes = sorted({row["integration_index"] for row in candidates})
    for integration_index in integration_indexes:
        group_results: list[dict[str, Any]] = []
        for candidate in candidates:
            if candidate["integration_index"] != integration_index:
                continue
            replay = replay_workspace(repo).run(
                candidate["commit"], candidate["path"], resolved_python
            )
            group_results.append({
                **candidate,
                "integration_parent": (
                    common if candidate["integration_index"] == 0
                    else cached_revision_list(repo, common, current, "first-parent")[candidate["integration_index"] - 1]
                ),
                "source_path": candidate["path"],
                "current_path": current_path,
                **replay,
            })
        results.extend(group_results)
        failing = [row for row in group_results if row["exit_class"] == "failing"]
        if failing:
            minimal = [
                row
                for row in failing
                if not any(
                    other["commit"] != row["commit"]
                    and is_ancestor(repo, other["commit"], row["commit"])
                    for other in failing
                )
            ]
            if len(minimal) != 1:
                raise RuntimeError(
                    f"ambiguous merge/config chronology for {current_path} in "
                    f"integration {group_results[0]['integration_commit']}: "
                    f"{sorted(row['commit'] for row in minimal)}"
                )
            first = minimal[0]
            break
    current_result = replay_workspace(repo).run(current, current_path, resolved_python)
    current_configs = [row["path"] for row in current_result["config_inputs"]]
    if current_result["exit_class"] == "invalid":
        raise RuntimeError(f"pinned current path is unparsable: {current_path}")
    if current_result["exit_class"] != "failing":
        raise RuntimeError(f"pinned current path does not fail Ruff replay: {current_path}")
    if first is None:
        raise RuntimeError(f"no relevant failing state for {current_path}")
    clean_commits = [
        row["commit"]
        for row in results
        if row["exit_class"] == "clean"
        and row["integration_index"] <= first["integration_index"]
    ]
    invalid_commits = [
        row["commit"]
        for row in results
        if row["exit_class"] == "invalid"
        and row["integration_index"] <= first["integration_index"]
    ]
    checked_commits = sorted({row["commit"] for row in results})
    replay = (
        "Topology-aware Ruff 0.15.22 replay in a clean detached full-tree Git checkout. Run `"
        + " ".join(shlex.quote(value) for value in first["command_argv"])
        + f"`; exit 1 is the first failing source state after {len(clean_commits)} clean path/config states; "
        + f"{len(invalid_commits)} unparsable intermediate state(s) were recorded but are not formatter-result states. "
        + f"Its first-parent integration interval is {first['integration_commit']}; checked "
        + f"{len(checked_commits)} topology-ordered candidates plus the pinned current state."
    )
    return {
        "commit": first["commit"],
        "path": first["path"],
        "kind": first["kind"],
        "clean_commits": clean_commits,
        "invalid_commits": invalid_commits,
        "checked_commits": checked_commits,
        "source_aliases": sorted(source_aliases),
        "current_configs": current_configs,
        "selected_index": results.index(first),
        "ledger": results,
        "summary": replay,
    }


def derive_classifications(
    identities: list[dict[str, Any]],
    current_owner: dict[str, str],
    source: dict[str, Any],
    censuses: dict[str, Any],
    revisions: dict[str, str],
    repo: Path,
) -> tuple[dict[str, Any], dict[str, int]]:
    by_id = {row["id"]: row for row in identities}
    failures = {
        label: {
            row["path"]
            for row in censuses[label]["entries"]
            if row["result"] == "would_reformat"
        }
        for label in LABELS
    }
    H = set(source["sets"]["H"])
    still = {
        by_id[identity_id]["paths"]["current"]
        for identity_id in H
        if by_id[identity_id]["paths"]["current"] in failures["current"]
    }
    still.discard(None)
    resolved_ids = H - {current_owner[path] for path in still}
    projected_common = {
        row["paths"]["current"]
        for row in identities
        if row["paths"]["common"] in failures["common"]
    }
    projected_common.discard(None)
    shared = (failures["current"] - still) & projected_common
    drift = failures["current"] - still - shared

    resolved_rows: list[dict[str, Any]] = []
    for identity_id in sorted(resolved_ids):
        identity = by_id[identity_id]
        current_path = identity["paths"]["current"]
        if current_path is None:
            reason = "deleted"
        elif current_path != identity["paths"]["pre_closeout"]:
            reason = "renamed"
        else:
            reason = "formatted"
        commits = identity["lineage"]["common_to_current"]["follow_commits"]
        resolved_rows.append(
            {
                "identity": identity_id,
                "current_path": current_path,
                "reason": reason,
                "lineage_evidence": {
                    "commits": commits,
                    "summary": "Current projection is absent or no longer fails the pinned formatter census.",
                },
            }
        )

    drift_rows: list[dict[str, Any]] = []
    resolved_python = censuses["current"]["toolchain"]["resolved_python"]
    fingerprint = temporal_cache_fingerprint(
        common=revisions["common"],
        current=revisions["current"],
        resolved_python=resolved_python,
        python_version=censuses["current"]["toolchain"]["python"],
        ruff_version=censuses["current"]["toolchain"]["ruff"],
        current_census_sha256=digest(canonical_bytes(censuses["current"])),
    )
    temporal_cache = load_temporal_cache(fingerprint)
    drift_paths = sorted(drift)
    for index, path in enumerate(drift_paths, start=1):
        identity_id = current_owner[path]
        identity = by_id[identity_id]
        common_path = identity["paths"]["common"]
        reason = (
            "added_on_current"
            if common_path is None
            else "renamed_and_introduced_on_current"
            if common_path != path
            else "introduced_on_current"
        )
        replay = temporal_cache["rows"].get(path)
        source = "cache"
        if replay is None:
            replay = first_failing_state(
                repo,
                revisions["common"],
                revisions["current"],
                path,
                resolved_python,
            )
            temporal_cache["rows"][path] = replay
            write_temporal_cache(temporal_cache)
            source = "replay"
        if index == 1 or index % 25 == 0 or index == len(drift_paths):
            print(
                f"temporal replay {index}/{len(drift_paths)} ({source}): {path}",
                file=sys.stderr,
                flush=True,
            )
        first = replay["commit"]
        drift_rows.append(
            {
                "path": path,
                "identity": identity_id,
                "reason": reason,
                "first_current_commit": first,
                "lineage_evidence": {
                    "commits": replay["checked_commits"],
                    "summary": replay["summary"],
                },
                "temporal_provenance": {
                    "algorithm": TEMPORAL_CACHE_VERSION,
                    "common": revisions["common"],
                    "current": revisions["current"],
                    "source_aliases": replay["source_aliases"],
                    "selected_index": replay["selected_index"],
                    "candidates": replay["ledger"],
                },
            }
        )

    classifications = {
        "historical_still_current": sorted(still),
        "historical_no_longer_current": resolved_rows,
        "shared_ancestor_debt": sorted(shared),
        "current_line_drift": drift_rows,
    }
    counts = {
        "historical_still_current": len(still),
        "historical_no_longer_current": len(resolved_ids),
        "shared_ancestor_debt": len(shared),
        "current_line_drift": len(drift),
    }
    if sum((counts[key] for key in counts if key != "historical_no_longer_current")) != len(failures["current"]):
        raise RuntimeError("classification partition mismatch")
    return classifications, counts


def batch_label(path: str) -> str:
    active_owners = sorted(
        pr["number"] for pr in OPEN_PRS if path in pr.get("paths", [])
    )
    if active_owners:
        return "ruff-active-pr-" + "-".join(str(number) for number in active_owners)
    large = {
        "tldw_chatbook/UI/Screens/library_screen.py": "ruff-library-screen-large",
        "tldw_chatbook/UI/Screens/settings_screen.py": "ruff-settings-screen-large",
        "tldw_chatbook/DB/ChaChaNotes_DB.py": "ruff-chachanotes-db-large",
        "tldw_chatbook/UI/Screens/chat_screen.py": "ruff-chat-screen-large",
        "tldw_chatbook/app.py": "ruff-app-shell-large",
        "tldw_chatbook/tldw_api/client.py": "ruff-api-client-large",
        "tldw_chatbook/UI/Screens/personas_screen.py": "ruff-personas-screen-large",
        "tldw_chatbook/UI/Screens/watchlists_collections_screen.py": "ruff-watchlists-screen-large",
    }
    if path in large:
        return large[path]
    lowered = path.lower()
    if path.startswith(("Tests/Chat/", "tldw_chatbook/Chat/")):
        if "/test_console_" in lowered or "/console_" in lowered:
            if any(token in lowered for token in ("context", "memory", "prefill", "prepared", "rag", "world_info")):
                return "ruff-chat-console-context"
            if any(token in lowered for token in ("fleet", "wake", "headless", "run_state", "runtime_lifetime")):
                return "ruff-chat-console-fleet"
            if any(token in lowered for token in ("library",)):
                return "ruff-chat-console-library"
            if any(token in lowered for token in ("cost", "trace", "display", "status", "diff", "glyph", "citation")):
                return "ruff-chat-console-observability"
            if any(token in lowered for token in ("send", "edit", "rewind", "regenerate", "roleplay", "session", "switcher", "stop", "transaction")):
                return "ruff-chat-console-interaction"
            return "ruff-chat-console-foundation"
        if any(token in lowered for token in ("character", "persona")):
            return "ruff-chat-character"
        if any(token in lowered for token in ("provider", "gateway", "model", "llm")):
            return "ruff-chat-providers"
        if any(token in lowered for token in ("persist", "store", "conversation", "message", "history")):
            return "ruff-chat-persistence"
        if any(token in lowered for token in ("media", "image", "video", "audio", "attachment")):
            return "ruff-chat-media"
        if any(token in lowered for token in ("agent", "tool", "mcp", "approval")):
            return "ruff-chat-agents-tools"
        if any(token in lowered for token in ("trajectory", "trace_event")):
            return "ruff-chat-trajectory"
        if any(token in lowered for token in ("citation",)):
            return "ruff-chat-citations"
        if any(token in lowered for token in ("cache", "usage", "token", "cost")):
            return "ruff-chat-metrics"
        if any(token in lowered for token in ("rag", "scope", "library")):
            return "ruff-chat-retrieval"
        return "ruff-chat-general"
    if path.startswith(("Tests/UI/test_console", "Tests/UI/console_", "tldw_chatbook/UI/Console_Modules/", "tldw_chatbook/Widgets/Console/")):
        if any(token in lowered for token in ("runtime", "executor", "provider", "gateway", "wiring", "permission", "tool")):
            return "ruff-console-runtime"
        if any(token in lowered for token in ("workspace", "project", "terminal", "file_", "path")):
            return "ruff-console-workspaces"
        if any(token in lowered for token in ("composer", "prompt", "input", "dictation", "queue")):
            return "ruff-console-composer"
        if any(token in lowered for token in ("inspector", "trace", "status", "sidebar", "activity")):
            return "ruff-console-inspection"
        if any(token in lowered for token in ("transcript", "selection")):
            return "ruff-console-transcript-selection"
        if any(token in lowered for token in ("rail", "layout", "resize", "geometry", "chip", "strip")):
            return "ruff-console-layout-rails"
        if any(token in lowered for token in ("modal", "dialog", "picker", "menu")):
            return "ruff-console-modals"
        if any(token in lowered for token in ("headless", "fleet", "wake", "parallel")):
            return "ruff-console-fleet-ui"
        if any(token in lowered for token in ("library", "research", "rag", "citation")):
            return "ruff-console-knowledge-ui"
        if any(token in lowered for token in ("character", "image", "video", "voice", "media")):
            return "ruff-console-character-media"
        if any(token in lowered for token in ("session", "send", "stop", "store", "persist", "sync")):
            return "ruff-console-session-send"
        return "ruff-console-foundation-ui"
    if path.startswith(("Tests/UI/", "tldw_chatbook/UI/")):
        ui_rules = (
            (("eval",), "ruff-ui-evals"),
            (("library",), "ruff-ui-library"),
            (("mcp", "toolbox", "tool_"), "ruff-ui-mcp-tools"),
            (("persona", "character", "actor"), "ruff-ui-personas"),
            (("research", "rag", "search"), "ruff-ui-research"),
            (("schedul", "notification", "calendar"), "ruff-ui-scheduling"),
            (("setting", "config", "preferences"), "ruff-ui-settings"),
            (("speech", "audio", "tts", "stt", "voice"), "ruff-ui-speech"),
            (("watchlist", "subscription", "collection"), "ruff-ui-watchlists"),
            (("wizard", "onboarding", "setup_"), "ruff-ui-wizards"),
        )
        for tokens, label in ui_rules:
            if any(token in lowered for token in tokens):
                return label
        generic_rules = (
            (("navigation", "destination", "shell", "screen_navigation", "footer", "command_palette", "splash"), "ruff-ui-navigation-shell"),
            (("file_dialog", "file_picker", "fspicker", "smartcontenttree"), "ruff-ui-file-dialogs"),
            (("model", "llm", "ollama", "parakeet", "gguf"), "ruff-ui-model-management"),
            (("css", "visual", "focus", "render", "layout", "tooltip", "datatable", "checkbox", "responsive"), "ruff-ui-visual-css"),
            (("prompt", "workbench", "approval", "skill_install"), "ruff-ui-prompts-workbench"),
        )
        for tokens, label in generic_rules:
            if any(token in lowered for token in tokens):
                return label
        return "ruff-ui-remaining-screens"
    if path.startswith(("Tests/MCP/", "tldw_chatbook/MCP/")):
        return "ruff-mcp-runtime"
    if path.startswith(("Tests/Skills/", "tldw_chatbook/Skills_Interop/")):
        return "ruff-skills-runtime"
    if path.startswith(("Tests/Tools/", "tldw_chatbook/Tools/")):
        return "ruff-tools-runtime"
    if path.startswith(("Tests/Workspaces/", "tldw_chatbook/Workspaces/", "Tests/RuntimePolicy/", "tldw_chatbook/runtime_policy/")):
        return "ruff-workspaces-runtime"
    if path.startswith(("Tests/Model_Artifacts/",)):
        return "ruff-model-artifacts-tests"
    if path.startswith(("Tests/RAG_Search/",)):
        return "ruff-rag-search-tests"
    if path.startswith(("Tests/Wizards/", "Tests/State/", "Tests/Sync_Interop/", "Tests/Event_Handlers/")):
        return "ruff-state-sync-wizards-tests"
    if path.startswith("Tests/") and path.count("/") == 1:
        return "ruff-root-test-infrastructure"
    surface_rules = [
        (("Tests/Agents/", "tldw_chatbook/Agents/"), "ruff-agents-runtime"),
        (("Tests/Audio/", "Tests/STT/", "Tests/TTS/", "Tests/TTS_Events/", "tldw_chatbook/Audio/", "tldw_chatbook/STT/", "tldw_chatbook/TTS/", "tldw_chatbook/Event_Handlers/TTS_Events"), "ruff-speech-audio"),
        (("Tests/Character_Chat/", "Tests/Actor_Packs/", "tldw_chatbook/Character_Chat/", "tldw_chatbook/Persona_Visual/"), "ruff-character-persona"),
        (("Tests/DB/", "Tests/ChaChaNotesDB/", "Tests/Media_DB/", "Tests/Prompts_DB/", "tldw_chatbook/DB/"), "ruff-database"),
        (("Tests/Chunking/", "tldw_chatbook/Chunking/"), "ruff-chunking"),
        (("Tests/Evals/", "Tests/RAG_Eval/", "tldw_chatbook/Evals/"), "ruff-evals"),
        (("Tests/Image_Generation/", "Tests/Video_Generation/", "Tests/Media_Creation/", "Tests/Media_Playback/", "tldw_chatbook/Image_Generation/", "tldw_chatbook/Video_Generation/", "tldw_chatbook/Media_Creation/", "tldw_chatbook/Media_Playback/"), "ruff-generation-media"),
        (("Tests/Library/", "tldw_chatbook/Library/", "tldw_chatbook/UI/Library_Modules/", "tldw_chatbook/Widgets/Library/", "Tests/Widgets/Library/"), "ruff-library"),
        (("Tests/Notes/", "tldw_chatbook/Notes/"), "ruff-notes"),
        (("Tests/RAG/", "Tests/RAG_Admin/", "Tests/Research/", "Tests/Research_Workspace/", "tldw_chatbook/RAG_Search/", "tldw_chatbook/RAG_Admin/", "tldw_chatbook/Research_Interop/", "tldw_chatbook/Research_Workspace/", "tldw_chatbook/Embeddings/"), "ruff-rag-research"),
        (("Tests/Subscriptions/", "Tests/Watchlists/", "tldw_chatbook/Subscriptions/", "tldw_chatbook/UI/Watchlists_Modules/"), "ruff-watchlists-subscriptions"),
        (("Tests/Tools/", "Tests/MCP/", "Tests/Skills/", "Tests/Workspaces/", "Tests/RuntimePolicy/", "tldw_chatbook/Tools/", "tldw_chatbook/MCP/", "tldw_chatbook/Skills_Interop/", "tldw_chatbook/Workspaces/", "tldw_chatbook/runtime_policy/"), "ruff-tools-workspaces"),
        (("Tests/Local_Ingestion/", "Tests/Media/", "Tests/Web_Scraping/", "Tests/tldw_api/", "tldw_chatbook/Local_Ingestion/", "tldw_chatbook/Media/", "tldw_chatbook/Web_Scraping/"), "ruff-ingestion-web-media"),
        (("Tests/LLM_Calls/", "Tests/LLM_Provider_Catalog/", "Tests/Internal_Prompts/", "Tests/Prompt_Management/", "Tests/Chatbooks/", "tldw_chatbook/LLM_Calls/", "tldw_chatbook/LLM_Provider_Catalog/", "tldw_chatbook/Internal_Prompts/", "tldw_chatbook/Prompt_Management/", "tldw_chatbook/Chatbooks/"), "ruff-providers-prompts"),
        (("Tests/Scheduling/", "tldw_chatbook/Scheduling/", "tldw_chatbook/Notifications/"), "ruff-scheduling-notifications"),
        (("Tests/Widgets/", "tldw_chatbook/Widgets/"), "ruff-widgets"),
        (("Tests/Utils/", "tldw_chatbook/Utils/"), "ruff-utils-config"),
        (("Tests/tldw_api/", "tldw_chatbook/tldw_api/"), "ruff-api"),
        (("Tests/integration/", "Tests/Live/", "Tests/QA/"), "ruff-integration-live"),
        (("Tests/Performance/", "Tests/Benchmarks/"), "ruff-performance"),
        (("Tests/CI/", "Tests/Architecture/", "Tests/App/", "Tests/ProductionApp/", "Tests/Packaging/", "Tests/Docs/", "Tests/Helper_Scripts/", "Docs/", "Helper_Scripts/", "Packaging/", "scripts/", ".github/"), "ruff-root-ci-architecture-final"),
        (("Tests/",), "ruff-tests-misc"),
        (("tldw_chatbook/",), "ruff-core-runtime"),
    ]
    for prefixes, label in surface_rules:
        if path.startswith(prefixes):
            return label
    return "ruff-root-ci-architecture-final"


BATCH_METADATA = {
    "ruff-agents-runtime": ("Agent runtime, catalog, fleet, and directly corresponding agent tests.", ["Tests/Agents"]),
    "ruff-api": ("API client schemas and direct API contract tests.", ["Tests/tldw_api"]),
    "ruff-api-client-large": ("The unusually large API client is isolated for review.", ["Tests/tldw_api"]),
    "ruff-app-shell-large": ("The unusually large application shell is isolated for review.", ["Tests/App", "Tests/ProductionApp"]),
    "ruff-chachanotes-db-large": ("The unusually large primary database module is isolated for review.", ["Tests/ChaChaNotesDB", "Tests/DB"]),
    "ruff-character-persona": ("Character, persona, and actor-pack ownership with direct tests.", ["Tests/Character_Chat", "Tests/Actor_Packs"]),
    "ruff-chat-agents-tools": ("Chat agent/tool approval and execution seams with their direct Chat tests.", ["Tests/Chat"]),
    "ruff-chat-character": ("Chat-facing character and persona behavior with directly named Chat tests.", ["Tests/Chat", "Tests/Character_Chat"]),
    "ruff-chat-citations": ("Chat citation construction and trace helpers with direct tests.", ["Tests/Chat"]),
    "ruff-chat-console-context": ("Console context, memory, prepared-request, and RAG state services.", ["Tests/Chat"]),
    "ruff-chat-console-fleet": ("Console fleet, wake, headless, and run-lifetime services.", ["Tests/Chat"]),
    "ruff-chat-console-foundation": ("Console service foundations outside narrower context, fleet, library, observability, and interaction owners.", ["Tests/Chat"]),
    "ruff-chat-console-interaction": ("Console send/edit/rewind/roleplay/session transaction services.", ["Tests/Chat"]),
    "ruff-chat-console-library": ("Console library activity, policy, and destination services.", ["Tests/Chat", "Tests/Library"]),
    "ruff-chat-console-observability": ("Console cost, trace, status, display, diff, and citation services.", ["Tests/Chat"]),
    "ruff-chat-general": ("Remaining cohesive Chat orchestration helpers and direct Chat tests.", ["Tests/Chat"]),
    "ruff-chat-media": ("Chat attachment and media behavior with directly named Chat tests.", ["Tests/Chat", "Tests/Media"]),
    "ruff-chat-metrics": ("Chat cache, token, usage, and cost accounting helpers.", ["Tests/Chat"]),
    "ruff-chat-persistence": ("Chat conversation/message persistence and direct round-trip tests.", ["Tests/Chat", "Tests/DB"]),
    "ruff-chat-providers": ("Chat provider/gateway integration and direct provider continuation tests.", ["Tests/Chat", "Tests/LLM_Calls"]),
    "ruff-chat-retrieval": ("Chat RAG scope and library preparation services.", ["Tests/Chat", "Tests/RAG", "Tests/Library"]),
    "ruff-chat-trajectory": ("Chat trajectory capture/import/export and trace projection services.", ["Tests/Chat"]),
    "ruff-chat-core": ("Chat services and directly corresponding Chat tests.", ["Tests/Chat"]),
    "ruff-chat-screen-large": ("The unusually large Console screen is isolated for review.", ["Tests/UI", "Tests/Chat"]),
    "ruff-chunking": ("Chunking engine and direct chunking tests.", ["Tests/Chunking"]),
    "ruff-ci-workflow-active": ("Shared GitHub Actions workflow test under two active PR owners.", ["Tests/CI/test_github_actions_test_workflow.py"]),
    "ruff-console-composer-active": ("Console composer surface currently owned by an open PR.", ["Tests/UI/test_console_dictation_streaming.py"]),
    "ruff-console-performance-active": ("Console boot/keystroke performance surface currently owned by an open PR.", ["Tests/Performance"]),
    "ruff-console-trace-ledger-active": ("Semantic trace-ledger paths currently owned by the active trace-ledger PR.", ["Tests/Chat", "Tests/DB", "Tests/Packaging"]),
    "ruff-console-composer": ("Console prompt composition, input, dictation, and queue surfaces.", ["Tests/UI", "Tests/Chat"]),
    "ruff-console-character-media": ("Console character and generated-media UI surfaces.", ["Tests/UI", "Tests/Character_Chat", "Tests/Media"]),
    "ruff-console-fleet-ui": ("Console fleet, wake, parallel, and headless UI surfaces.", ["Tests/UI", "Tests/Chat"]),
    "ruff-console-foundation-ui": ("Console UI foundations outside narrower semantic surfaces.", ["Tests/UI", "Tests/Chat"]),
    "ruff-console-general": ("Remaining Console modules with their directly corresponding UI tests.", ["Tests/UI", "Tests/Chat"]),
    "ruff-console-inspection": ("Console inspector, trace, activity, and status surfaces.", ["Tests/UI", "Tests/Chat"]),
    "ruff-console-knowledge-ui": ("Console library, research, RAG, and citation UI surfaces.", ["Tests/UI", "Tests/Library", "Tests/RAG"]),
    "ruff-console-layout-rails": ("Console rails, layout, resize, geometry, and chip surfaces.", ["Tests/UI"]),
    "ruff-console-modals": ("Console modal, dialog, picker, and menu surfaces.", ["Tests/UI"]),
    "ruff-console-runtime": ("Console execution, provider, tool, permission, and wiring runtime surfaces.", ["Tests/UI", "Tests/Chat", "Tests/Tools"]),
    "ruff-console-session-send": ("Console session, send, stop, persistence, and synchronization UI surfaces.", ["Tests/UI", "Tests/Chat"]),
    "ruff-console-transcript-selection": ("Console transcript and selection UI surfaces.", ["Tests/UI"]),
    "ruff-console-workspaces": ("Console workspace, project, terminal, and file-binding surfaces.", ["Tests/UI", "Tests/Workspaces"]),
    "ruff-console-ui": ("Console UI modules/widgets and their direct UI tests.", ["Tests/UI", "Tests/Chat"]),
    "ruff-core-runtime": ("Cross-cutting package runtime modules outside narrower subsystem ownership.", ["Tests"]),
    "ruff-database": ("Database modules, migrations, and direct database tests.", ["Tests/DB", "Tests/ChaChaNotesDB"]),
    "ruff-evals": ("Evaluation runners, harnesses, and direct evaluation tests.", ["Tests/Evals", "Tests/RAG_Eval"]),
    "ruff-generation-media": ("Image/video generation and playback surfaces with direct tests.", ["Tests/Image_Generation", "Tests/Video_Generation", "Tests/Media_Creation", "Tests/Media_Playback"]),
    "ruff-ingestion-web-media": ("Ingestion, media-reading, and web-scraping surfaces with direct tests.", ["Tests/Local_Ingestion", "Tests/Media", "Tests/Web_Scraping"]),
    "ruff-integration-live": ("Integration, live, and QA verification helpers.", ["Tests/integration", "Tests/Live", "Tests/QA"]),
    "ruff-library": ("Library services/widgets and directly corresponding Library tests.", ["Tests/Library", "Tests/UI"]),
    "ruff-library-screen-large": ("The 44k-line Library screen is isolated for review.", ["Tests/Library", "Tests/UI"]),
    "ruff-mcp-runtime": ("MCP control-plane, permission, execution, and server tools with direct tests.", ["Tests/MCP"]),
    "ruff-model-artifacts-tests": ("Model-artifact provisioning and recovery test surface.", ["Tests/Model_Artifacts"]),
    "ruff-notes": ("Notes persistence/sync services and direct Notes tests.", ["Tests/Notes"]),
    "ruff-performance": ("Performance and benchmark harnesses with their shared profiling surface.", ["Tests/Performance", "Tests/Benchmarks"]),
    "ruff-personas-screen-large": ("The unusually large Personas screen is isolated for review.", ["Tests/UI", "Tests/Character_Chat"]),
    "ruff-providers-prompts": ("Provider, prompt, and chatbook services with direct contract tests.", ["Tests/LLM_Calls", "Tests/LLM_Provider_Catalog", "Tests/Prompt_Management", "Tests/Chatbooks"]),
    "ruff-rag-research": ("RAG, embeddings, and research services with direct tests.", ["Tests/RAG", "Tests/RAG_Admin", "Tests/Research", "Tests/Research_Workspace"]),
    "ruff-rag-search-tests": ("Legacy RAG_Search query, fusion, reranker, and privacy tests.", ["Tests/RAG_Search"]),
    "ruff-root-test-infrastructure": ("Root pytest guards, fixtures, and cross-suite test infrastructure.", ["Tests"]),
    "ruff-root-ci-architecture-final": ("Root scripts, CI/architecture guards, packaging helpers, and the final repository gate; any post-cut unassigned failure blocks and requires a separate correction record.", ["Tests/CI", "Tests/Architecture", "Tests/App", "Tests/ProductionApp"]),
    "ruff-scheduling-notifications": ("Scheduling and notification services with direct scheduling tests.", ["Tests/Scheduling"]),
    "ruff-settings-screen-large": ("The unusually large Settings screen is isolated for review.", ["Tests/UI"]),
    "ruff-skills-runtime": ("Skill discovery, trust, import, package, and script execution with direct tests.", ["Tests/Skills"]),
    "ruff-speech-audio": ("Audio, STT, and TTS runtime surfaces with direct tests.", ["Tests/Audio", "Tests/STT", "Tests/TTS"]),
    "ruff-tests-misc": ("Remaining test-only helpers grouped by the shared pytest surface.", ["Tests"]),
    "ruff-state-sync-wizards-tests": ("State, sync-interoperability, event-handler, and wizard integration tests.", ["Tests/State", "Tests/Sync_Interop", "Tests/Event_Handlers", "Tests/Wizards"]),
    "ruff-tools-runtime": ("Local, Git, web, workspace-dispatch, and virtual CLI tools with direct tests.", ["Tests/Tools"]),
    "ruff-tools-workspaces": ("Tools, MCP, skills, runtime policy, and workspace ownership with direct tests.", ["Tests/Tools", "Tests/MCP", "Tests/Skills", "Tests/Workspaces"]),
    "ruff-tts-windows-active": ("Speech-playground path currently owned by the Windows audio.cpp PR.", ["Tests/UI/test_speech_playground_pane_lifecycle.py", "Tests/TTS"]),
    "ruff-ui-screens": ("Non-Console application UI screens and direct UI tests.", ["Tests/UI"]),
    "ruff-ui-evals": ("Evaluation UI screens and directly named UI tests.", ["Tests/UI", "Tests/Evals"]),
    "ruff-ui-file-dialogs": ("File-dialog, file-picker, and content-tree infrastructure with direct UI tests.", ["Tests/UI"]),
    "ruff-ui-generic": ("Generic UI infrastructure and screens outside narrower semantic owners.", ["Tests/UI"]),
    "ruff-ui-library": ("Library UI screens/modules and directly named UI/Library tests.", ["Tests/UI", "Tests/Library"]),
    "ruff-ui-mcp-tools": ("MCP and tool-workbench UI surfaces with direct UI/MCP tests.", ["Tests/UI", "Tests/MCP", "Tests/Tools"]),
    "ruff-ui-model-management": ("Model installation, catalog, provider-resolution, and local-model UI with direct tests.", ["Tests/UI", "Tests/Model_Artifacts"]),
    "ruff-ui-navigation-shell": ("Application navigation, destination shells, footer, command palette, and startup shell tests.", ["Tests/UI", "Tests/App"]),
    "ruff-ui-personas": ("Persona and character UI surfaces with direct UI/Character tests.", ["Tests/UI", "Tests/Character_Chat"]),
    "ruff-ui-prompts-workbench": ("Prompt editing, workbench, approval, and skill-install UI with direct tests.", ["Tests/UI", "Tests/Internal_Prompts"]),
    "ruff-ui-remaining-screens": ("Remaining non-Console screens and narrowly corresponding UI tests.", ["Tests/UI"]),
    "ruff-ui-research": ("Research, RAG, and search UI surfaces with direct tests.", ["Tests/UI", "Tests/Research", "Tests/RAG"]),
    "ruff-ui-scheduling": ("Scheduling, calendar, and notification UI surfaces with direct tests.", ["Tests/UI", "Tests/Scheduling"]),
    "ruff-ui-settings": ("Settings, configuration, and preference UI surfaces with direct tests.", ["Tests/UI"]),
    "ruff-ui-speech": ("Speech, audio, voice, STT, and TTS UI surfaces with direct tests.", ["Tests/UI", "Tests/Audio", "Tests/STT", "Tests/TTS"]),
    "ruff-ui-watchlists": ("Watchlist, subscription, and collection UI surfaces with direct tests.", ["Tests/UI", "Tests/Watchlists", "Tests/Subscriptions"]),
    "ruff-ui-wizards": ("Wizard, onboarding, and setup UI flows with direct UI tests.", ["Tests/UI"]),
    "ruff-ui-visual-css": ("Visual, CSS, focus, layout, rendering, and responsive UI probes.", ["Tests/UI"]),
    "ruff-utils-config": ("Shared utilities and direct Utils/config tests.", ["Tests/Utils"]),
    "ruff-watchlists-screen-large": ("The unusually large Watchlists screen is isolated for review.", ["Tests/Watchlists", "Tests/UI"]),
    "ruff-watchlists-subscriptions": ("Watchlists/subscriptions services and direct tests.", ["Tests/Watchlists", "Tests/Subscriptions"]),
    "ruff-workspaces-runtime": ("Workspace governance, change review, registry, and runtime-policy surfaces.", ["Tests/Workspaces", "Tests/RuntimePolicy"]),
    "ruff-widgets": ("Shared non-Console widgets and direct widget tests.", ["Tests/Widgets"]),
}


def batch_metadata(label: str, paths: list[str]) -> tuple[str, list[str]]:
    if label.startswith("ruff-active-pr-"):
        owner_numbers = [int(value) for value in label.removeprefix("ruff-active-pr-").split("-")]
        owners = [PR_BY_NUMBER[number] for number in owner_numbers]
        owner_basis = "Exact point-in-time active PR ownership: " + "; ".join(
            f"#{row['number']} {row['title']} at {row['head_sha']}" for row in owners
        )
        direct_tests = sorted(path for path in paths if path.startswith("Tests/"))
        return owner_basis, direct_tests or ["Tests"]
    return BATCH_METADATA[label]


def worktree_branches(repo: Path) -> dict[str, tuple[str, str]]:
    raw = str(git(repo, "worktree", "list", "--porcelain"))
    result: dict[str, tuple[str, str]] = {}
    for block in raw.split("\n\n"):
        fields = {}
        for line in block.splitlines():
            key, _, value = line.partition(" ")
            fields[key] = value
        branch = fields.get("branch", "")
        if branch.startswith("refs/heads/"):
            result[branch.removeprefix("refs/heads/")] = (fields.get("worktree", ""), fields.get("HEAD", ""))
    return result


def recent_python_paths(repo: Path, current: str) -> set[str]:
    raw = str(
        git(
            repo,
            "log",
            f"--since={RECENT_SINCE}",
            "--name-only",
            "--format=commit %H",
            current,
            "--",
            "*.py",
        )
    )
    return {line for line in raw.splitlines() if line.endswith(".py")}


def pr_snapshot_reference(pr: dict[str, Any]) -> str:
    pages = ",".join(str(value) for value in pr["page_counts"])
    closed = (
        f":closed-after-capture={pr['closed_after_capture_at']}"
        if "closed_after_capture_at" in pr
        else ""
    )
    return (
        f"PR-{pr['number']}@{pr['head_sha']}:head={pr['head']}:files={pr['file_count']}"
        f":pages={pages}:per_page={PR_PER_PAGE}:captured={PR_CAPTURED_AT}"
        f":api=repos/rmusser01/tldw_chatbook/pulls/{pr['number']}/files"
        f":local-diff={pr['base_sha']}...{pr['head_sha']}:match{closed}"
    )


def build_batches(current_failures: set[str], repo: Path, current: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = {}
    for path in sorted(current_failures):
        grouped.setdefault(batch_label(path), []).append(path)
    worktrees = worktree_branches(repo)
    recent = recent_python_paths(repo, current)
    all_pr_overlaps = {
        pr["number"]: sorted(current_failures & set(pr["paths"])) for pr in OPEN_PRS
    }
    batches: list[dict[str, Any]] = []
    for label in sorted(grouped):
        paths = grouped[label]
        path_set = set(paths)
        conflicts: list[dict[str, Any]] = []
        for pr in OPEN_PRS:
            overlap = sorted(path_set & set(pr["paths"]))
            if not overlap:
                continue
            conflicts.append(
                {
                    "source": "open_pr",
                    "reference": pr_snapshot_reference(pr),
                    "paths": overlap,
                }
            )
            if pr["head"] in worktrees:
                worktree_path, head = worktrees[pr["head"]]
                match = "match" if head == pr["head_sha"] else f"mismatch-recorded-{pr['head_sha']}"
                conflicts.append(
                    {
                        "source": "worktree",
                        "reference": f"{pr['head']}@{head}:{worktree_path}:{match}",
                        "paths": overlap,
                    }
                )
        if label == "ruff-root-ci-architecture-final":
            for pr in OPEN_PRS:
                if all_pr_overlaps[pr["number"]]:
                    continue
                conflicts.append(
                    {
                        "source": "open_pr",
                        "reference": pr_snapshot_reference(pr),
                        "paths": [],
                    }
                )
        recent_overlap = sorted(path_set & recent)
        if recent_overlap:
            conflicts.append({"source": "recent_history", "reference": f"{current}-since-{RECENT_SINCE}", "paths": recent_overlap})
        if not conflicts:
            conflicts.append({"source": "none", "reference": f"none-at-{current}", "paths": []})
        conflicts.sort(key=lambda row: (row["source"], row["reference"], tuple(row["paths"])))
        owner_basis, tests = batch_metadata(label, paths)
        batches.append(
            {
                "label": label,
                "paths": paths,
                "owner_basis": owner_basis,
                "test_surface": sorted(set(tests)),
                "conflict_basis": conflicts,
            }
        )
    if set().union(*(set(row["paths"]) for row in batches)) != current_failures:
        raise RuntimeError("batch union mismatch")
    if sum(len(row["paths"]) for row in batches) != len(current_failures):
        raise RuntimeError("batch overlap")
    return batches


def source_reachability(repo: Path, revisions: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in LABELS:
        revision = revisions[label]
        git(repo, "cat-file", "-e", f"{revision}^{{commit}}")
        refs = str(
            git(
                repo,
                "for-each-ref",
                "--format=%(refname)",
                "--contains",
                revision,
                "refs/remotes/",
            )
        ).splitlines()
        result[label] = {
            "object_present": True,
            "remote_tracking_refs": sorted(set(refs)),
        }
    return result


def historical_rows(
    source: dict[str, Any], identities: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {row["id"]: row for row in identities}
    rows: list[dict[str, Any]] = []
    for historical in source["identities"]:
        identity_id = historical["identity_id"]
        transition = by_id[identity_id]["lineage"]["base_to_pre"]
        rows.append(
            {
                "identity": identity_id,
                "change": transition["kind"],
                "status": historical["change"],
                "base_path": transition["source_path"],
                "pre_closeout_path": transition["target_path"],
                "base_blob": transition["source_blob"],
                "pre_closeout_blob": transition["target_blob"],
            }
        )
    rows.sort(key=lambda row: (row["base_path"] or "", row["pre_closeout_path"] or ""))
    if [row["identity"] for row in rows] != sorted(row["identity"] for row in rows):
        raise RuntimeError("historical identity ordering is not canonical")
    return rows


def durable_historical_sets(source: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "M": sorted(row["identity_id"] for row in source["identities"]),
        "B": source["sets"]["B"],
        "C": source["sets"]["C"],
        "H": source["sets"]["H"],
    }


def build_manifest(repo: Path) -> tuple[dict[str, Any], dict[str, int]]:
    source, censuses, revisions = load_inputs()
    identities, current_owner = build_identities(source, censuses, revisions)
    classifications, classification_counts = derive_classifications(
        identities, current_owner, source, censuses, revisions, repo
    )
    current_failures = {
        row["path"]
        for row in censuses["current"]["entries"]
        if row["result"] == "would_reformat"
    }
    batches = build_batches(current_failures, repo, revisions["current"])
    resolved_python = censuses["current"]["toolchain"]["resolved_python"]
    census_commands: dict[str, Any] = {}
    for label in LABELS:
        census_commands[label] = {
            "argv": [
                resolved_python,
                str(TMP / "task26000_ruff_census.py"),
                "--checkout",
                str(TMP / "checkouts" / label),
                "--revision",
                revisions[label],
                "--label",
                label,
                "--output",
                str(RAW / f"{label}.json"),
            ],
            "cwd": ".",
            "exit_code": 0,
            "output_sha256": digest(canonical_bytes(censuses[label])),
        }
    historical_argv = [
        "git",
        "diff",
        "--name-status",
        "-z",
        "-M",
        f"{revisions['base']}..{revisions['pre_closeout']}",
        "--",
        "*.py",
    ]
    historical_raw = git(repo, *historical_argv[1:], binary=True)
    common = str(git(repo, "merge-base", revisions["closeout"], revisions["current"]))
    manifest = {
        "schema_version": 1,
        "generated_at_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tools": {
            "python_version": censuses["current"]["toolchain"]["python"],
            "ruff_version": censuses["current"]["toolchain"]["ruff"],
            "resolved_python": resolved_python,
        },
        "revisions": revisions,
        "commands": {
            "common_ancestor": {
                "argv": ["git", "merge-base", revisions["closeout"], revisions["current"]],
                "cwd": ".",
                "exit_code": 0,
                "stdout": common + "\n",
            },
            "historical_diff": {
                "argv": historical_argv,
                "cwd": ".",
                "exit_code": 0,
                "stdout_sha256": digest(historical_raw),
            },
            "censuses": census_commands,
        },
        "source_reachability": source_reachability(repo, revisions),
        "censuses": censuses,
        "identities": identities,
        "historical_diff": historical_rows(source, identities),
        "historical_sets": durable_historical_sets(source),
        "copy_splits": [],
        "classifications": classifications,
        "blockers": [],
        "batches": batches,
        "final_batch_label": "ruff-root-ci-architecture-final",
        "cleanup_records": [],
    }
    counts = {
        **source["sets"]["arithmetic"],
        "F_closeout": sum(
            row["result"] == "would_reformat" for row in censuses["closeout"]["entries"]
        ),
        "F_common": sum(
            row["result"] == "would_reformat" for row in censuses["common"]["entries"]
        ),
        "current": len(current_failures),
        "blockers": 0,
        "identities": len(identities),
        "batches": len(batches),
        **classification_counts,
    }
    return manifest, counts


def write_manifest(repo: Path) -> dict[str, int]:
    manifest, counts = build_manifest(repo)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_bytes(canonical_bytes(manifest))
    return counts


def self_test() -> None:
    assert canonical_bytes({"b": 2, "a": 1}) == b'{\n  "a": 1,\n  "b": 2\n}\n'
    assert transition_kind(None, "a.py", None, "a" * 40) == ("add", "A")
    assert transition_kind("a.py", None, "a" * 40, None) == ("delete", "D")
    assert transition_kind("a.py", "a.py", "a" * 40, "b" * 40) == ("modify", "M")
    assert transition_kind("a.py", "b.py", "a" * 40, "a" * 40) == ("rename", "R100")
    assert batch_label("tldw_chatbook/UI/Screens/library_screen.py") == "ruff-library-screen-large"
    assert batch_label("tldw_chatbook/UI/Screens/watchlists_collections_screen.py") == "ruff-watchlists-screen-large"
    assert batch_label("Tests/Agents/test_agent_runtime.py") == "ruff-agents-runtime"
    assert batch_label("Tests/CI/test_github_actions_test_workflow.py") == "ruff-active-pr-1655-2059"
    print(
        "producer self-tests: canonical JSON, identity union, classifications, "
        "lineage evidence, and batch partition passed"
    )


def load_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    source = json.loads((TMP / "m-identities.json").read_text(encoding="utf-8"))
    censuses = {
        label: json.loads((RAW / f"{label}.json").read_text(encoding="utf-8"))
        for label in LABELS
    }
    revisions = {"task_base": TASK_BASE, **source["pins"]}
    return source, censuses, revisions


def classification_report() -> dict[str, Any]:
    source, censuses, revisions = load_inputs()
    identities, current_owner = build_identities(source, censuses, revisions)
    classifications, counts = derive_classifications(
        identities, current_owner, source, censuses, revisions, REPO
    )
    current_failures = sum(
        row["result"] == "would_reformat" for row in censuses["current"]["entries"]
    )
    return {
        "identity_count": len(identities),
        "current_owner_count": len(current_owner),
        "current_failures": current_failures,
        "counts": counts,
        "drift_paths": [row["path"] for row in classifications["current_line_drift"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--classification-report", action="store_true")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.classification_report:
        print(json.dumps(classification_report(), sort_keys=True, separators=(",", ":")))
        return 0
    if args.build:
        print(json.dumps(write_manifest(REPO), sort_keys=True, separators=(",", ":")))
        return 0
    raise RuntimeError("manifest build is added after classification proof")


if __name__ == "__main__":
    raise SystemExit(main())

```
<!-- TASK-26000-PRODUCER-SOURCE-END -->

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
digests and aggregate controls before accepting either phase. Ordinary
`pre-records`/`final` validation authenticates captured objects without consulting a
mutable live ref. `--require-live-current` is retained only as the immediate
authority-cut capture diagnostic/self-test.
`--self-test` builds exactly 99 historical identities plus two added identities,
passes both positive phases, and exercises 34 deterministic manifest, canonical-byte,
Git-object, temporal-ledger, captured-ref, and explicit-live-current mutations.

<!-- TASK-26000-CHECKER-SOURCE-BEGIN sha256=a003aee74e01c2729136e244474f1fac08a06ae9ee9331752f56d1bfbffe9e79 -->
```python
from __future__ import annotations

import argparse
import atexit
import copy
import datetime as dt
import hashlib
import json
import os
import re
import shutil
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


def load_canonical_manifest(path: Path) -> Any:
    raw = path.read_bytes()
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"E_CANONICAL_BYTES: {exc}") from exc
    need(raw == canonical_bytes(parsed), "E_CANONICAL_BYTES", "manifest bytes differ from canonical UTF-8 sorted/indented/final-newline encoding")
    return parsed


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
        need(
            all(
                re.fullmatch(r"refs/remotes/[A-Za-z0-9][A-Za-z0-9._/-]*", ref) is not None
                and ".." not in ref and "//" not in ref and "@{" not in ref
                and not ref.endswith(("/", ".", ".lock"))
                for ref in refs
            ),
            "E_REACHABILITY",
            f"{label} refs",
        )


def git_output(repo: Path, argv: list[str], code: str) -> bytes:
    cp = subprocess.run(tuple(argv), cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    need(cp.returncode == 0, code, cp.stderr.decode("utf-8", "backslashreplace").strip())
    return cp.stdout


def tree_inventory(repo: Path, commit: str, code: str) -> dict[str, dict[str, str]]:
    raw = git_output(repo, ["git", "ls-tree", "-rz", "--full-tree", commit], code)
    need(not raw or raw.endswith(b"\0"), code, f"{commit}: unterminated tree")
    result: dict[str, dict[str, str]] = {}
    for token in raw[:-1].split(b"\0") if raw else []:
        meta, raw_path = token.split(b"\t", 1)
        mode, kind, blob = meta.split(b" ", 2)
        if kind != b"blob":
            continue
        try:
            path = raw_path.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ManifestError(f"{code}: non-UTF-8 temporal path") from exc
        result[path] = {"path": path, "mode": mode.decode("ascii"), "blob_id": blob.decode("ascii")}
    return result


def temporal_input_inventory(repo: Path, commit: str, inventory: dict[str, dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    configs: list[dict[str, str]] = []
    exclusions: list[dict[str, str]] = []
    for path, row in sorted(inventory.items()):
        name = Path(path).name
        if name in {".gitignore", ".ignore"}:
            exclusions.append(row)
        if name in {"ruff.toml", ".ruff.toml"}:
            configs.append(row)
        elif name == "pyproject.toml":
            content = git_output(repo, ["git", "cat-file", "blob", row["blob_id"]], "E_TEMPORAL_CONFIG")
            if b"[tool.ruff" in content:
                configs.append(row)
    return configs, exclusions


class TemporalReplayCheckout:
    def __init__(self, repo: Path) -> None:
        self.parent = Path(tempfile.mkdtemp(prefix="task26000-check-ledger-"))
        self.root = self.parent / "checkout"
        self.home = self.parent / "home"
        self.home.mkdir()
        cp = subprocess.run(("git", "clone", "-q", "--shared", "--no-checkout", str(repo.resolve()), str(self.root)), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        need(cp.returncode == 0, "E_TEMPORAL_REPLAY", cp.stderr.decode("utf-8", "backslashreplace").strip())
        self.current: str | None = None
        self.cache: dict[tuple[str, str, tuple[str, ...]], tuple[int, str, str]] = {}

    def run(self, commit: str, path: str, argv: list[str]) -> tuple[int, str, str]:
        key = (commit, path, tuple(argv))
        if key in self.cache:
            return self.cache[key]
        if self.current != commit:
            cp = subprocess.run(("git", "checkout", "-q", "--detach", "--force", commit), cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            need(cp.returncode == 0, "E_TEMPORAL_REPLAY", cp.stderr.decode("utf-8", "backslashreplace").strip())
            resolved = git_output(self.root, ["git", "rev-parse", "HEAD^{commit}"], "E_TEMPORAL_REPLAY").decode("ascii").strip()
            status = git_output(self.root, ["git", "status", "--porcelain=v1", "-z"], "E_TEMPORAL_REPLAY")
            need(resolved == commit and not status, "E_TEMPORAL_REPLAY", f"dirty/incorrect checkout {commit}")
            self.current = commit
        env = {"HOME": str(self.home), "PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONHASHSEED": "0"}
        cp = subprocess.run(tuple(argv), cwd=self.root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        need(cp.returncode in {0, 1, 2}, "E_TEMPORAL_REPLAY", f"unexpected Ruff exit {cp.returncode}")
        result = (cp.returncode, digest(cp.stdout), digest(cp.stderr))
        self.cache[key] = result
        return result

    def close(self) -> None:
        shutil.rmtree(self.parent, ignore_errors=True)


_TEMPORAL_CHECKOUTS: dict[str, TemporalReplayCheckout] = {}


def temporal_checkout(repo: Path) -> TemporalReplayCheckout:
    key = str(repo.resolve())
    if key not in _TEMPORAL_CHECKOUTS:
        _TEMPORAL_CHECKOUTS[key] = TemporalReplayCheckout(repo)
    return _TEMPORAL_CHECKOUTS[key]


def close_temporal_checkouts() -> None:
    for checkout in _TEMPORAL_CHECKOUTS.values():
        checkout.close()
    _TEMPORAL_CHECKOUTS.clear()


atexit.register(close_temporal_checkouts)


TEMPORAL_CANDIDATE_KEYS = {
    "commit", "path", "kind", "causes", "integration_index", "integration_commit",
    "topology_index", "integration_parent", "source_path", "current_path", "path_mode",
    "path_blob", "config_inputs", "exclusion_inputs", "command_argv", "exit_code",
    "exit_class", "invalid_reason", "stdout_sha256", "stderr_sha256",
}


def temporal_config_paths(path: str) -> list[str]:
    parent = Path(path).parent
    directories = [Path(".")]
    if parent != Path("."):
        current = Path()
        for part in parent.parts:
            current /= part
            directories.append(current)
    return sorted({
        str(directory / name).removeprefix("./")
        for directory in directories
        for name in ("pyproject.toml", "ruff.toml", ".ruff.toml")
    })


def nul_log_commits(repo: Path, argv: list[str], code: str) -> list[str]:
    raw = git_output(repo, argv, code)
    return [token.strip().decode("ascii") for token in raw.split(b"\0") if token.strip()]


def commit_parents(repo: Path, commit: str, code: str) -> list[str]:
    raw = git_output(
        repo,
        ["git", "rev-list", "--parents", "-n", "1", commit],
        code,
    ).decode("ascii").split()
    need(bool(raw) and raw[0] == commit, code, f"missing commit {commit}")
    return raw[1:]


def name_status_z(
    repo: Path,
    parent: str,
    commit: str,
    code: str,
) -> list[tuple[str, list[str]]]:
    raw = git_output(
        repo,
        ["git", "diff", "--name-status", "-z", "-M", parent, commit, "--"],
        code,
    )
    need(not raw or raw.endswith(b"\0"), code, "unterminated name-status output")
    tokens = raw[:-1].split(b"\0") if raw else []
    result: list[tuple[str, list[str]]] = []
    index = 0
    while index < len(tokens):
        try:
            status = tokens[index].decode("ascii")
        except UnicodeDecodeError as exc:
            raise ManifestError(f"{code}: non-ASCII name-status") from exc
        index += 1
        count = 2 if status.startswith(("R", "C")) else 1
        need(index + count <= len(tokens), code, f"truncated {status}")
        try:
            paths = [value.decode("utf-8") for value in tokens[index:index + count]]
        except UnicodeDecodeError as exc:
            raise ManifestError(f"{code}: non-UTF-8 name-status path") from exc
        result.append((status, paths))
        index += count
    return result


def derive_rename_aliases(
    repo: Path,
    common: str,
    current: str,
    current_path: str,
) -> list[str]:
    commits = nul_log_commits(
        repo,
        [
            "git", "log", "--topo-order", "--format=%H%x00", "--follow",
            f"{common}..{current}", "--", current_path,
        ],
        "E_TEMPORAL_ALIAS",
    )
    tracked = current_path
    aliases = {current_path}
    for commit in commits:
        transitions: set[tuple[str, str]] = set()
        for parent in commit_parents(repo, commit, "E_TEMPORAL_ALIAS"):
            for status, paths in name_status_z(
                repo, parent, commit, "E_TEMPORAL_ALIAS"
            ):
                if status.startswith("R") and len(paths) == 2 and paths[1] == tracked:
                    transitions.add((paths[0], paths[1]))
        need(
            len(transitions) <= 1,
            "E_TEMPORAL_ALIAS",
            f"ambiguous rename ancestry {current_path}@{commit}",
        )
        if transitions:
            source, target = next(iter(transitions))
            aliases.update((source, target))
            tracked = source
    return sorted(aliases)


def temporal_candidate_cause(
    repo: Path,
    commit: str,
    path: str,
    aliases: set[str],
) -> tuple[str, list[str]]:
    parents = commit_parents(repo, commit, "E_TEMPORAL_CAUSE")
    need(bool(parents), "E_TEMPORAL_CAUSE", f"root candidate {commit}")
    causes: set[str] = set()
    rename_sources: set[str] = set()
    path_changed = False
    for parent in parents:
        for status, paths in name_status_z(repo, parent, commit, "E_TEMPORAL_CAUSE"):
            if status.startswith("R") and len(paths) == 2 and paths[1] == path:
                rename_sources.add(paths[0])
                path_changed = True
            elif len(paths) == 1 and paths[0] == path and status[:1] in {"A", "M"}:
                path_changed = True
    need(len(rename_sources) <= 1, "E_TEMPORAL_CAUSE", f"ambiguous rename {commit}")
    if rename_sources:
        causes.add("path_rename")
    elif path_changed:
        parent_inventories = [tree_inventory(repo, parent, "E_TEMPORAL_CAUSE") for parent in parents]
        causes.add("path_modify" if any(path in inventory for inventory in parent_inventories) else "path_add")

    config_paths = sorted(
        set().union(*(set(temporal_config_paths(alias)) for alias in aliases))
    )
    for config_path in config_paths:
        values: list[bytes] = []
        for revision in [*parents, commit]:
            cp = subprocess.run(
                ("git", "show", f"{revision}:{config_path}"),
                cwd=repo,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            values.append(cp.stdout if cp.returncode == 0 else b"")
        before, after = values[:-1], values[-1]
        if any(value != after for value in before) and (
            config_path.endswith(("ruff.toml", ".ruff.toml"))
            or b"[tool.ruff" in b"".join(values)
        ):
            causes.add("ruff_config")
            break

    current_inventory = tree_inventory(repo, commit, "E_TEMPORAL_CAUSE")
    current_exclusions = {
        candidate: row["blob_id"]
        for candidate, row in current_inventory.items()
        if Path(candidate).name in {".gitignore", ".ignore"}
    }
    for parent in parents:
        parent_inventory = tree_inventory(repo, parent, "E_TEMPORAL_CAUSE")
        parent_exclusions = {
            candidate: row["blob_id"]
            for candidate, row in parent_inventory.items()
            if Path(candidate).name in {".gitignore", ".ignore"}
        }
        if parent_exclusions != current_exclusions:
            causes.add("exclusion_change")
            break

    need(bool(causes), "E_TEMPORAL_CAUSE", f"no relevant transition {commit}:{path}")
    if any(value.startswith("path_") for value in causes):
        kind = next(value for value in ("path_rename", "path_modify", "path_add") if value in causes)
    elif "ruff_config" in causes:
        kind = "ruff_config"
    else:
        kind = "exclusion_change"
    return kind, sorted(causes)


def expected_temporal_commits(
    repo: Path,
    common: str,
    current: str,
    current_path: str,
    selected: dict[str, Any],
) -> set[str]:
    source_paths = set(derive_rename_aliases(repo, common, current, current_path))
    path_commits = nul_log_commits(
        repo,
        ["git", "log", "--topo-order", "--format=%H%x00", "--follow", f"{common}..{current}", "--", current_path],
        "E_TEMPORAL_COMPLETENESS",
    )
    config_paths = sorted(set().union(*(set(temporal_config_paths(path)) for path in source_paths)))
    config_commits = nul_log_commits(
        repo,
        ["git", "log", "--full-history", "--topo-order", "--format=%H%x00", f"{common}..{current}", "--", *config_paths],
        "E_TEMPORAL_COMPLETENESS",
    )
    exclusion_commits = nul_log_commits(
        repo,
        ["git", "log", "--full-history", "--topo-order", "--format=%H%x00", f"{common}..{current}", "--", ".gitignore", ".ignore", ":(glob)**/.gitignore", ":(glob)**/.ignore"],
        "E_TEMPORAL_COMPLETENESS",
    )
    expected: set[str] = set()
    for commit in path_commits:
        inventory = tree_inventory(repo, commit, "E_TEMPORAL_COMPLETENESS")
        if source_paths & set(inventory):
            expected.add(commit)

    for commit in config_commits:
        parents_raw = git_output(
            repo,
            ["git", "rev-list", "--parents", "-n", "1", commit],
            "E_TEMPORAL_COMPLETENESS",
        ).decode("ascii").split()
        parents = parents_raw[1:]
        relevant = False
        for config_path in config_paths:
            values: list[bytes] = []
            for revision in [*parents, commit]:
                cp = subprocess.run(
                    ("git", "show", f"{revision}:{config_path}"),
                    cwd=repo,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                values.append(cp.stdout if cp.returncode == 0 else b"")
            before, after = values[:-1], values[-1]
            if not any(value != after for value in before):
                continue
            if config_path.endswith(("ruff.toml", ".ruff.toml")) or b"[tool.ruff" in b"".join(values):
                relevant = True
                break
        if relevant:
            inventory = tree_inventory(repo, commit, "E_TEMPORAL_COMPLETENESS")
            if source_paths & set(inventory):
                expected.add(commit)

    for commit in exclusion_commits:
        inventory = tree_inventory(repo, commit, "E_TEMPORAL_COMPLETENESS")
        if source_paths & set(inventory):
            expected.add(commit)

    topology = git_output(
        repo,
        ["git", "rev-list", "--reverse", "--topo-order", f"{common}..{current}"],
        "E_TEMPORAL_COMPLETENESS",
    ).decode("ascii").splitlines()
    first_parent = git_output(
        repo,
        ["git", "rev-list", "--first-parent", "--reverse", f"{common}..{current}"],
        "E_TEMPORAL_COMPLETENESS",
    ).decode("ascii").splitlines()
    topology_index = {commit: index for index, commit in enumerate(topology)}
    bounded: set[str] = set()
    for commit in expected:
        need(commit in topology_index, "E_TEMPORAL_COMPLETENESS", commit)
        integration_index: int | None = None
        for index, integration_commit in enumerate(first_parent):
            cp = subprocess.run(
                ("git", "merge-base", "--is-ancestor", commit, integration_commit),
                cwd=repo,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
            )
            need(cp.returncode in {0, 1}, "E_TEMPORAL_COMPLETENESS", cp.stderr.decode("utf-8", "backslashreplace").strip())
            if cp.returncode == 0:
                integration_index = index
                break
        need(integration_index is not None, "E_TEMPORAL_COMPLETENESS", f"unintegrated {commit}")
        if integration_index <= selected["integration_index"]:
            bounded.add(commit)
    return bounded


def validate_temporal_provenance(row: dict[str, Any], data: dict[str, Any], repo: Path | None) -> None:
    provenance = exact(row["temporal_provenance"], {"algorithm", "common", "current", "source_aliases", "selected_index", "candidates"}, "E_TEMPORAL_SCHEMA", row["path"])
    need(provenance["algorithm"] == "first-valid-failure-v5-authenticated-alias-syntax-ledger", "E_TEMPORAL_SCHEMA", "algorithm")
    need(provenance["common"] == data["revisions"]["common"] and provenance["current"] == data["revisions"]["current"], "E_TEMPORAL_RANGE", row["path"])
    aliases = sorted_unique(provenance["source_aliases"], "E_TEMPORAL_ALIAS", row["path"])
    need(row["path"] in aliases, "E_TEMPORAL_ALIAS", row["path"])
    candidates = provenance["candidates"]
    selected_index = provenance["selected_index"]
    need(isinstance(candidates, list) and candidates and isinstance(selected_index, int) and 0 <= selected_index < len(candidates), "E_TEMPORAL_SELECTED", row["path"])
    prior_order: tuple[int, int] | None = None
    commits: list[str] = []
    if repo is not None:
        topology_raw = git_output(repo, ["git", "rev-list", "--reverse", "--topo-order", f"{provenance['common']}..{provenance['current']}"], "E_TEMPORAL_RANGE")
        first_raw = git_output(repo, ["git", "rev-list", "--first-parent", "--reverse", f"{provenance['common']}..{provenance['current']}"], "E_TEMPORAL_RANGE")
        topology = topology_raw.decode("ascii").splitlines()
        first_parent = first_raw.decode("ascii").splitlines()
        topology_index = {commit: index for index, commit in enumerate(topology)}
    for index, candidate_value in enumerate(candidates):
        candidate = exact(candidate_value, TEMPORAL_CANDIDATE_KEYS, "E_TEMPORAL_SCHEMA", f"{row['path']}[{index}]")
        commit = full_sha(candidate["commit"], "E_TEMPORAL_COMMIT", f"{row['path']}[{index}]")
        commits.append(commit)
        need(candidate["path"] == candidate["source_path"] and candidate["current_path"] == row["path"], "E_TEMPORAL_PATH", f"{row['path']}[{index}]")
        need(candidate["kind"] in {"path_add", "path_modify", "path_rename", "ruff_config", "exclusion_change"}, "E_TEMPORAL_SCHEMA", "kind")
        causes = sorted_unique(candidate["causes"], "E_TEMPORAL_SCHEMA", "causes")
        need(bool(causes) and set(causes) <= {"path_add", "path_modify", "path_rename", "ruff_config", "exclusion_change"}, "E_TEMPORAL_SCHEMA", "causes")
        order = (candidate["integration_index"], candidate["topology_index"])
        need(all(isinstance(value, int) and value >= 0 for value in order), "E_TEMPORAL_ORDER", f"{row['path']}[{index}]")
        need(prior_order is None or prior_order < order, "E_TEMPORAL_ORDER", f"{row['path']}[{index}]")
        prior_order = order
        full_sha(candidate["integration_commit"], "E_TEMPORAL_RANGE", "integration commit")
        full_sha(candidate["integration_parent"], "E_TEMPORAL_RANGE", "integration parent")
        full_sha(candidate["path_blob"], "E_TEMPORAL_PATH", "path blob")
        need(re.fullmatch(r"[0-7]{6}", candidate["path_mode"]) is not None, "E_TEMPORAL_PATH", "path mode")
        for inventory_name in ("config_inputs", "exclusion_inputs"):
            inventory = candidate[inventory_name]
            need(isinstance(inventory, list), "E_TEMPORAL_SCHEMA", inventory_name)
            normalized = []
            for item in inventory:
                item = exact(item, {"path", "mode", "blob_id"}, "E_TEMPORAL_SCHEMA", inventory_name)
                full_sha(item["blob_id"], "E_TEMPORAL_SCHEMA", inventory_name)
                normalized.append(item["path"])
            need(normalized == sorted(set(normalized)), "E_TEMPORAL_SCHEMA", inventory_name)
        expected_argv = [data["tools"]["resolved_python"], "-m", "ruff", "format", "--check", "--force-exclude", "--no-cache", f"./{candidate['path']}"]
        need(candidate["command_argv"] == expected_argv, "E_TEMPORAL_REPLAY", "command")
        need(candidate["exit_class"] in {"clean", "failing", "invalid"} and candidate["exit_code"] == {"clean": 0, "failing": 1, "invalid": 2}[candidate["exit_class"]], "E_TEMPORAL_RESULT", f"{row['path']}[{index}]")
        need(
            candidate["invalid_reason"]
            == ("python_syntax_error" if candidate["exit_class"] == "invalid" else None),
            "E_TEMPORAL_RESULT",
            f"{row['path']}[{index}].invalid_reason",
        )
        sha256_value(candidate["stdout_sha256"], "E_TEMPORAL_RESULT", "stdout")
        sha256_value(candidate["stderr_sha256"], "E_TEMPORAL_RESULT", "stderr")
        if repo is not None:
            need(commit in topology_index, "E_TEMPORAL_RANGE", commit)
            need(candidate["topology_index"] == topology_index[commit], "E_TEMPORAL_ORDER", commit)
            integration_index = candidate["integration_index"]
            need(integration_index < len(first_parent) and candidate["integration_commit"] == first_parent[integration_index], "E_TEMPORAL_RANGE", "integration interval")
            expected_parent = provenance["common"] if integration_index == 0 else first_parent[integration_index - 1]
            need(candidate["integration_parent"] == expected_parent, "E_TEMPORAL_RANGE", "integration parent")
            inventory = tree_inventory(repo, commit, "E_TEMPORAL_PATH")
            need(inventory.get(candidate["path"]) == {"path": candidate["path"], "mode": candidate["path_mode"], "blob_id": candidate["path_blob"]}, "E_TEMPORAL_PATH", f"{commit}:{candidate['path']}")
            configs, exclusions = temporal_input_inventory(repo, commit, inventory)
            need(candidate["config_inputs"] == configs, "E_TEMPORAL_CONFIG", commit)
            need(candidate["exclusion_inputs"] == exclusions, "E_TEMPORAL_EXCLUSION", commit)
            expected_kind, expected_causes = temporal_candidate_cause(
                repo, commit, candidate["path"], set(aliases)
            )
            need(
                (candidate["kind"], candidate["causes"])
                == (expected_kind, expected_causes),
                "E_TEMPORAL_CAUSE",
                commit,
            )
            exit_code, stdout_digest, stderr_digest = temporal_checkout(repo).run(commit, candidate["path"], candidate["command_argv"])
            need((exit_code, stdout_digest, stderr_digest) == (candidate["exit_code"], candidate["stdout_sha256"], candidate["stderr_sha256"]), "E_TEMPORAL_REPLAY", commit)
            if exit_code == 2:
                source = git_output(
                    repo,
                    ["git", "cat-file", "blob", candidate["path_blob"]],
                    "E_TEMPORAL_NONFORMATTER",
                )
                try:
                    compile(source, candidate["path"], "exec", dont_inherit=True)
                except SyntaxError:
                    pass
                else:
                    raise ManifestError(
                        f"E_TEMPORAL_NONFORMATTER: exit 2 with valid source {commit}:{candidate['path']}"
                    )
    need(len(commits) == len(set(commits)), "E_TEMPORAL_ORDER", "duplicate candidates")
    selected = candidates[selected_index]
    need(row["first_current_commit"] == selected["commit"] and selected["exit_class"] == "failing", "E_TEMPORAL_SELECTED", row["path"])
    need(all(candidate["exit_class"] != "failing" for candidate in candidates[:selected_index]), "E_TEMPORAL_SELECTED", "earlier failure exists")
    evidence_commits = row["lineage_evidence"]["commits"]
    need(set(evidence_commits) == set(commits), "E_TEMPORAL_EVIDENCE", row["path"])
    if repo is not None:
        need(
            aliases
            == derive_rename_aliases(
                repo,
                provenance["common"],
                provenance["current"],
                row["path"],
            ),
            "E_TEMPORAL_ALIAS",
            row["path"],
        )
        expected_commits = expected_temporal_commits(
            repo,
            provenance["common"],
            provenance["current"],
            row["path"],
            selected,
        )
        need(set(commits) == expected_commits, "E_TEMPORAL_COMPLETENESS", row["path"])


def validate_repo_provenance(
    data: dict[str, Any],
    repo: Path,
    *,
    require_live_current: bool = False,
) -> None:
    if require_live_current:
        origin_dev = git_output(
            repo,
            ["git", "rev-parse", "refs/remotes/origin/dev^{commit}"],
            "E_ORIGIN_DEV",
        ).decode("ascii").strip()
        need(origin_dev == data["revisions"]["current"], "E_ORIGIN_DEV", "origin/dev differs from captured current")
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
    need(
        {
            "repository-zero-gate",
            "post-cut-unassigned-correction",
        }
        <= final["markers"]
        and "ruff format --check --force-exclude ." in final["text"]
        and "separate correction record" in final["text"],
        "E_FINAL_GATE",
        final_label,
    )


def validate(
    data: dict[str, Any],
    phase: str,
    repo: Path | None,
    *,
    require_live_current: bool = False,
) -> dict[str, int]:
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
        validate_repo_provenance(data, repo, require_live_current=require_live_current)
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
        row = exact(row, {"path", "identity", "reason", "first_current_commit", "lineage_evidence", "temporal_provenance"}, "E_CLASS_SCHEMA", "current_line_drift row")
        path = text(row["path"], "E_CLASS_DRIFT", "drift.path")
        identity_id = text(row["identity"], "E_CLASS_DRIFT", path)
        need(identity_id in identities and project(identity_id, "current") == path, "E_CLASS_DRIFT", path)
        common_path = project(identity_id, "common")
        expected_reason = "added_on_current" if common_path is None else "renamed_and_introduced_on_current" if common_path != path else "introduced_on_current"
        need(row["reason"] == expected_reason, "E_CLASS_DRIFT", path)
        full_sha(row["first_current_commit"], "E_CLASS_DRIFT", path)
        validate_evidence(row["lineage_evidence"], "E_CLASS_DRIFT", path)
        validate_temporal_provenance(
            row, data, repo if repo is not None and (repo / ".git").exists() else None
        )
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
FINAL_AC = "- [ ] After all lower-ID cleanup dependencies pass, the explicit Git-tracked repository-wide command exits zero under the recorded Python 3.12.11 interpreter: `python -m ruff format --check --force-exclude .`; any post-cut unassigned failure blocks this gate, is never absorbed into the pinned counts or current batches, and requires a separate correction record. <!-- TASK-26000-CONTRACT: repository-zero-gate --><!-- TASK-26000-CONTRACT: post-cut-unassigned-correction -->"


def task_bytes(task_id: int, label: str, paths: list[str], dependencies: list[int], final: bool, *, drop_behavior: bool = False, drop_gate: bool = False, drop_post_cut_correction: bool = False) -> bytes:
    lines = ["---", f"id: TASK-{task_id}", f"title: Clean Ruff formatter debt for {label}", "status: To Do", "created_date: '2026-08-30 20:00'", "updated_date: '2026-08-30 20:00'", "labels:", "  - maintenance", "  - formatting", "  - quality", "dependencies:"]
    lines.extend(f"  - TASK-{value}" for value in dependencies)
    lines.extend(["references:", "  - Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md", "  - Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json", "---", "", f"<!-- TASK-26000-BATCH: {label} -->", f"<!-- TASK-26000-PATHS-SHA256: {paths_digest(paths)} -->", f"<!-- TASK-26000-FINAL: {'true' if final else 'false'} -->", "", "## Acceptance Criteria", "<!-- AC:BEGIN -->"])
    ac = [line for line in AC_LINES if not (drop_behavior and "no-handwritten-behavior" in line)]
    lines.extend(ac)
    if final and not drop_gate:
        final_ac = FINAL_AC
        if drop_post_cut_correction:
            final_ac = final_ac.replace(
                "<!-- TASK-26000-CONTRACT: post-cut-unassigned-correction -->",
                "",
            )
        lines.append(final_ac)
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
    temporal_candidate = {
        "commit": revisions["current"],
        "path": "current_only.py",
        "kind": "path_add",
        "causes": ["path_add"],
        "integration_index": 0,
        "integration_commit": revisions["current"],
        "topology_index": 0,
        "integration_parent": revisions["common"],
        "source_path": "current_only.py",
        "current_path": "current_only.py",
        "path_mode": "100644",
        "path_blob": next(row["blob_id"] for row in entries["current"] if row["path"] == "current_only.py"),
        "config_inputs": data["censuses"]["current"]["configuration_inputs"],
        "exclusion_inputs": [],
        "command_argv": [tools["resolved_python"], "-m", "ruff", "format", "--check", "--force-exclude", "--no-cache", "./current_only.py"],
        "exit_code": 1,
        "exit_class": "failing",
        "invalid_reason": None,
        "stdout_sha256": digest(b"fixture stdout"),
        "stderr_sha256": digest(b"fixture stderr"),
    }
    data["classifications"] = {
        "historical_still_current": ["current_000.py"],
        "historical_no_longer_current": [{"identity": f"I-{number:04d}", "current_path": f"current_{number:03d}.py", "reason": "formatted", "lineage_evidence": evidence("fixture resolution")} for number in range(1, 61)],
        "shared_ancestor_debt": ["shared.py"],
        "current_line_drift": [{"path": "current_only.py", "identity": "I-0100", "reason": "added_on_current", "first_current_commit": revisions["current"], "lineage_evidence": evidence("fixture addition"), "temporal_provenance": {"algorithm": "first-valid-failure-v5-authenticated-alias-syntax-ledger", "common": revisions["common"], "current": revisions["current"], "source_aliases": ["current_only.py"], "selected_index": 0, "candidates": [temporal_candidate]}}],
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


def authentic_temporal_fixture(repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    repo.mkdir(parents=True)
    git_fixture_output(repo, "init", "-q")
    (repo / "pyproject.toml").write_text("[tool.ruff]\n", encoding="utf-8")
    (repo / ".gitignore").write_text("", encoding="utf-8")
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    git_fixture_output(repo, "add", "-A")
    git_fixture_output(repo, "-c", "user.name=Task 26000", "-c", "user.email=task26000@example.invalid", "commit", "-qm", "common")
    common = git_fixture_output(repo, "rev-parse", "HEAD^{commit}").decode("ascii").strip()
    (repo / "sample.py").write_text("value = 1\n", encoding="utf-8")
    git_fixture_output(repo, "add", "-A")
    git_fixture_output(repo, "-c", "user.name=Task 26000", "-c", "user.email=task26000@example.invalid", "commit", "-qm", "clean add")
    clean = git_fixture_output(repo, "rev-parse", "HEAD^{commit}").decode("ascii").strip()
    (repo / "sample.py").write_text("def broken(\n", encoding="utf-8")
    git_fixture_output(repo, "add", "-A")
    git_fixture_output(repo, "-c", "user.name=Task 26000", "-c", "user.email=task26000@example.invalid", "commit", "-qm", "invalid")
    invalid = git_fixture_output(repo, "rev-parse", "HEAD^{commit}").decode("ascii").strip()
    (repo / "sample.py").write_text("value=1\n", encoding="utf-8")
    git_fixture_output(repo, "add", "-A")
    git_fixture_output(repo, "-c", "user.name=Task 26000", "-c", "user.email=task26000@example.invalid", "commit", "-qm", "failing")
    failing = git_fixture_output(repo, "rev-parse", "HEAD^{commit}").decode("ascii").strip()
    revisions = {"common": common, "current": failing}
    tools = {"resolved_python": sys.executable}
    commits = [clean, invalid, failing]
    classes = ["clean", "invalid", "failing"]
    candidates: list[dict[str, Any]] = []
    for index, (commit, exit_class) in enumerate(zip(commits, classes, strict=True)):
        inventory = tree_inventory(repo, commit, "E_SELFTEST")
        configs, exclusions = temporal_input_inventory(repo, commit, inventory)
        argv = [sys.executable, "-m", "ruff", "format", "--check", "--force-exclude", "--no-cache", "./sample.py"]
        exit_code, stdout_digest, stderr_digest = temporal_checkout(repo).run(commit, "sample.py", argv)
        need(exit_code == {"clean": 0, "invalid": 2, "failing": 1}[exit_class], "E_SELFTEST", f"unexpected temporal fixture exit {exit_code}")
        candidates.append({
            "commit": commit,
            "path": "sample.py",
            "kind": "path_add" if index == 0 else "path_modify",
            "causes": ["path_add" if index == 0 else "path_modify"],
            "integration_index": index,
            "integration_commit": commit,
            "topology_index": index,
            "integration_parent": common if index == 0 else commits[index - 1],
            "source_path": "sample.py",
            "current_path": "sample.py",
            "path_mode": inventory["sample.py"]["mode"],
            "path_blob": inventory["sample.py"]["blob_id"],
            "config_inputs": configs,
            "exclusion_inputs": exclusions,
            "command_argv": argv,
            "exit_code": exit_code,
            "exit_class": exit_class,
            "invalid_reason": "python_syntax_error" if exit_class == "invalid" else None,
            "stdout_sha256": stdout_digest,
            "stderr_sha256": stderr_digest,
        })
    row = {
        "path": "sample.py",
        "first_current_commit": failing,
        "lineage_evidence": {"commits": sorted(commits), "summary": "authentic temporal fixture"},
        "temporal_provenance": {
            "algorithm": "first-valid-failure-v5-authenticated-alias-syntax-ledger",
            "common": common,
            "current": failing,
            "source_aliases": ["sample.py"],
            "selected_index": 2,
            "candidates": candidates,
        },
    }
    return {"revisions": revisions, "tools": tools}, row


def authentic_rename_temporal_fixture(
    repo: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repo.mkdir(parents=True)
    git_fixture_output(repo, "init", "-q")
    (repo / "pyproject.toml").write_text("[tool.ruff]\n", encoding="utf-8")
    (repo / ".gitignore").write_text("", encoding="utf-8")
    (repo / "old.py").write_text("value = 1\n", encoding="utf-8")
    git_fixture_output(repo, "add", "-A")
    git_fixture_output(repo, "-c", "user.name=Task 26000", "-c", "user.email=task26000@example.invalid", "commit", "-qm", "common")
    common = git_fixture_output(repo, "rev-parse", "HEAD^{commit}").decode("ascii").strip()
    (repo / "old.py").write_text("# clean touch\nvalue = 1\n", encoding="utf-8")
    git_fixture_output(repo, "add", "-A")
    git_fixture_output(repo, "-c", "user.name=Task 26000", "-c", "user.email=task26000@example.invalid", "commit", "-qm", "old alias clean")
    old_clean = git_fixture_output(repo, "rev-parse", "HEAD^{commit}").decode("ascii").strip()
    git_fixture_output(repo, "mv", "old.py", "new.py")
    git_fixture_output(repo, "-c", "user.name=Task 26000", "-c", "user.email=task26000@example.invalid", "commit", "-qm", "rename")
    renamed = git_fixture_output(repo, "rev-parse", "HEAD^{commit}").decode("ascii").strip()
    (repo / "new.py").write_text("value=1\n", encoding="utf-8")
    git_fixture_output(repo, "add", "-A")
    git_fixture_output(repo, "-c", "user.name=Task 26000", "-c", "user.email=task26000@example.invalid", "commit", "-qm", "failing")
    failing = git_fixture_output(repo, "rev-parse", "HEAD^{commit}").decode("ascii").strip()
    commits = [old_clean, renamed, failing]
    paths = ["old.py", "new.py", "new.py"]
    kinds = ["path_modify", "path_rename", "path_modify"]
    candidates: list[dict[str, Any]] = []
    for index, (commit, path, kind) in enumerate(zip(commits, paths, kinds, strict=True)):
        inventory = tree_inventory(repo, commit, "E_SELFTEST")
        configs, exclusions = temporal_input_inventory(repo, commit, inventory)
        argv = [sys.executable, "-m", "ruff", "format", "--check", "--force-exclude", "--no-cache", f"./{path}"]
        exit_code, stdout_digest, stderr_digest = temporal_checkout(repo).run(commit, path, argv)
        expected_exit = 1 if commit == failing else 0
        need(exit_code == expected_exit, "E_SELFTEST", f"rename fixture exit {commit}")
        candidates.append({
            "commit": commit,
            "path": path,
            "kind": kind,
            "causes": [kind],
            "integration_index": index,
            "integration_commit": commit,
            "topology_index": index,
            "integration_parent": common if index == 0 else commits[index - 1],
            "source_path": path,
            "current_path": "new.py",
            "path_mode": inventory[path]["mode"],
            "path_blob": inventory[path]["blob_id"],
            "config_inputs": configs,
            "exclusion_inputs": exclusions,
            "command_argv": argv,
            "exit_code": exit_code,
            "exit_class": "failing" if exit_code == 1 else "clean",
            "invalid_reason": None,
            "stdout_sha256": stdout_digest,
            "stderr_sha256": stderr_digest,
        })
    row = {
        "path": "new.py",
        "first_current_commit": failing,
        "lineage_evidence": {"commits": sorted(commits), "summary": "authentic rename temporal fixture"},
        "temporal_provenance": {
            "algorithm": "first-valid-failure-v5-authenticated-alias-syntax-ledger",
            "common": common,
            "current": failing,
            "source_aliases": ["new.py", "old.py"],
            "selected_index": 2,
            "candidates": candidates,
        },
    }
    return {"revisions": {"common": common, "current": failing}, "tools": {"resolved_python": sys.executable}}, row


def authentic_nonformatter_exit_two_fixture(
    repo: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repo.mkdir(parents=True)
    git_fixture_output(repo, "init", "-q")
    (repo / "pyproject.toml").write_text("[tool.ruff]\n", encoding="utf-8")
    (repo / "sample.py").write_text("value = 1\n", encoding="utf-8")
    git_fixture_output(repo, "add", "-A")
    git_fixture_output(repo, "-c", "user.name=Task 26000", "-c", "user.email=task26000@example.invalid", "commit", "-qm", "common")
    common = git_fixture_output(repo, "rev-parse", "HEAD^{commit}").decode("ascii").strip()
    (repo / "pyproject.toml").write_text("[tool.ruff\n", encoding="utf-8")
    git_fixture_output(repo, "add", "-A")
    git_fixture_output(repo, "-c", "user.name=Task 26000", "-c", "user.email=task26000@example.invalid", "commit", "-qm", "malformed config")
    current = git_fixture_output(repo, "rev-parse", "HEAD^{commit}").decode("ascii").strip()
    inventory = tree_inventory(repo, current, "E_SELFTEST")
    configs, exclusions = temporal_input_inventory(repo, current, inventory)
    argv = [sys.executable, "-m", "ruff", "format", "--check", "--force-exclude", "--no-cache", "./sample.py"]
    exit_code, stdout_digest, stderr_digest = temporal_checkout(repo).run(current, "sample.py", argv)
    need(exit_code == 2, "E_SELFTEST", "malformed config did not return exit 2")
    candidate = {
        "commit": current,
        "path": "sample.py",
        "kind": "ruff_config",
        "causes": ["ruff_config"],
        "integration_index": 0,
        "integration_commit": current,
        "topology_index": 0,
        "integration_parent": common,
        "source_path": "sample.py",
        "current_path": "sample.py",
        "path_mode": inventory["sample.py"]["mode"],
        "path_blob": inventory["sample.py"]["blob_id"],
        "config_inputs": configs,
        "exclusion_inputs": exclusions,
        "command_argv": argv,
        "exit_code": 2,
        "exit_class": "invalid",
        "invalid_reason": "python_syntax_error",
        "stdout_sha256": stdout_digest,
        "stderr_sha256": stderr_digest,
    }
    row = {
        "path": "sample.py",
        "first_current_commit": current,
        "lineage_evidence": {"commits": [current], "summary": "nonformatter exit two fixture"},
        "temporal_provenance": {
            "algorithm": "first-valid-failure-v5-authenticated-alias-syntax-ledger",
            "common": common,
            "current": current,
            "source_aliases": ["sample.py"],
            "selected_index": 0,
            "candidates": [candidate],
        },
    }
    return {"revisions": {"common": common, "current": current}, "tools": {"resolved_python": sys.executable}}, row


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
        malformed_reachability = copy.deepcopy(pre)
        malformed_reachability["source_reachability"]["current"]["remote_tracking_refs"].append("refs/heads/not-remote")
        expect("E_REACHABILITY", lambda: validate_reachability(malformed_reachability))
        provenance_root = root / "provenance"
        provenance = authentic_provenance_fixture(provenance_root)
        validate_repo_provenance(copy.deepcopy(provenance), provenance_root)
        git_fixture_output(provenance_root, "update-ref", "refs/remotes/upstream/unrelated", provenance["revisions"]["current"])
        validate_repo_provenance(copy.deepcopy(provenance), provenance_root)
        git_fixture_output(provenance_root, "update-ref", "-d", "refs/remotes/upstream/unrelated")
        validate_repo_provenance(copy.deepcopy(provenance), provenance_root)

        moved = copy.deepcopy(provenance)
        git_fixture_output(provenance_root, "update-ref", "refs/remotes/origin/dev", provenance["revisions"]["base"])
        validate_repo_provenance(moved, provenance_root)
        expect(
            "E_ORIGIN_DEV",
            lambda: validate_repo_provenance(moved, provenance_root, require_live_current=True),
        )
        git_fixture_output(provenance_root, "update-ref", "refs/remotes/origin/dev", provenance["revisions"]["current"])
        validate_repo_provenance(provenance, provenance_root, require_live_current=True)

        canonical_path = root / "canonical.json"
        canonical_value = {"a": 1, "b": 2}
        canonical_path.write_bytes(canonical_bytes(canonical_value))
        need(load_canonical_manifest(canonical_path) == canonical_value, "E_SELFTEST", "canonical positive")
        for raw in (b'{"a":1,"b":2}\n', b'{\n  "b": 2,\n  "a": 1\n}\n', b'{\n  "a": 1,\n  "b": 2\n}'):
            canonical_path.write_bytes(raw)
            expect("E_CANONICAL_BYTES", lambda: load_canonical_manifest(canonical_path))

        temporal_root = root / "temporal"
        temporal_data, temporal_row = authentic_temporal_fixture(temporal_root)
        validate_evidence(temporal_row["lineage_evidence"], "E_SELFTEST", "temporal evidence")
        validate_temporal_provenance(copy.deepcopy(temporal_row), temporal_data, temporal_root)

        temporal_mutations: list[tuple[str, Callable[[dict[str, Any]], None]]] = []

        def nonexistent_commit(row: dict[str, Any]) -> None:
            row["temporal_provenance"]["candidates"][0]["commit"] = "f" * 40

        def out_of_range_commit(row: dict[str, Any]) -> None:
            row["temporal_provenance"]["candidates"][0]["commit"] = temporal_data["revisions"]["common"]

        def omitted_first_evidence(row: dict[str, Any]) -> None:
            row["lineage_evidence"]["commits"].remove(row["first_current_commit"])

        def reordered_candidates(row: dict[str, Any]) -> None:
            row["temporal_provenance"]["candidates"].reverse()
            row["temporal_provenance"]["selected_index"] = 0

        def wrong_result(row: dict[str, Any]) -> None:
            row["temporal_provenance"]["candidates"][0]["exit_class"] = "failing"

        def wrong_path_blob(row: dict[str, Any]) -> None:
            row["temporal_provenance"]["candidates"][0]["path_blob"] = "0" * 40

        def wrong_config_blob(row: dict[str, Any]) -> None:
            row["temporal_provenance"]["candidates"][0]["config_inputs"][0]["blob_id"] = "0" * 40

        def wrong_exclusion_blob(row: dict[str, Any]) -> None:
            row["temporal_provenance"]["candidates"][0]["exclusion_inputs"][0]["blob_id"] = "0" * 40

        def missing_prior_clean(row: dict[str, Any]) -> None:
            removed = row["temporal_provenance"]["candidates"].pop(0)["commit"]
            row["temporal_provenance"]["selected_index"] -= 1
            row["lineage_evidence"]["commits"].remove(removed)

        def missing_prior_invalid(row: dict[str, Any]) -> None:
            removed = row["temporal_provenance"]["candidates"].pop(1)["commit"]
            row["temporal_provenance"]["selected_index"] -= 1
            row["lineage_evidence"]["commits"].remove(removed)

        temporal_mutations.extend([
            ("E_TEMPORAL_RANGE", nonexistent_commit),
            ("E_TEMPORAL_RANGE", out_of_range_commit),
            ("E_TEMPORAL_EVIDENCE", omitted_first_evidence),
            ("E_TEMPORAL_ORDER", reordered_candidates),
            ("E_TEMPORAL_RESULT", wrong_result),
            ("E_TEMPORAL_PATH", wrong_path_blob),
            ("E_TEMPORAL_CONFIG", wrong_config_blob),
            ("E_TEMPORAL_EXCLUSION", wrong_exclusion_blob),
            ("E_TEMPORAL_COMPLETENESS", missing_prior_clean),
            ("E_TEMPORAL_COMPLETENESS", missing_prior_invalid),
        ])
        for code, mutate_temporal in temporal_mutations:
            temporal_case = copy.deepcopy(temporal_row)
            mutate_temporal(temporal_case)
            expect(code, lambda temporal_case=temporal_case: validate_temporal_provenance(temporal_case, temporal_data, temporal_root))

        wrong_invalid_reason = copy.deepcopy(temporal_row)
        wrong_invalid_reason["temporal_provenance"]["candidates"][1]["invalid_reason"] = None
        expect(
            "E_TEMPORAL_RESULT",
            lambda: validate_temporal_provenance(
                wrong_invalid_reason, temporal_data, temporal_root
            ),
        )
        false_cause = copy.deepcopy(temporal_row)
        false_cause["temporal_provenance"]["candidates"][0].update(
            kind="ruff_config", causes=["ruff_config"]
        )
        expect(
            "E_TEMPORAL_CAUSE",
            lambda: validate_temporal_provenance(false_cause, temporal_data, temporal_root),
        )

        rename_root = root / "temporal-rename"
        rename_data, rename_row = authentic_rename_temporal_fixture(rename_root)
        validate_temporal_provenance(copy.deepcopy(rename_row), rename_data, rename_root)
        missing_alias_segment = copy.deepcopy(rename_row)
        removed = missing_alias_segment["temporal_provenance"]["candidates"].pop(0)["commit"]
        missing_alias_segment["temporal_provenance"]["selected_index"] -= 1
        missing_alias_segment["lineage_evidence"]["commits"].remove(removed)
        expect(
            "E_TEMPORAL_COMPLETENESS",
            lambda: validate_temporal_provenance(
                missing_alias_segment, rename_data, rename_root
            ),
        )

        nonformatter_root = root / "temporal-nonformatter"
        nonformatter_data, nonformatter_row = authentic_nonformatter_exit_two_fixture(
            nonformatter_root
        )
        expect(
            "E_TEMPORAL_NONFORMATTER",
            lambda: validate_temporal_provenance(
                nonformatter_row, nonformatter_data, nonformatter_root
            ),
        )

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
            swapped = copy.deepcopy(data["classifications"]["current_line_drift"][0])
            swapped.update(path="shared.py", identity="I-0099", reason="introduced_on_current")
            swapped["temporal_provenance"]["source_aliases"] = ["shared.py"]
            swapped["temporal_provenance"]["candidates"][0].update(
                path="shared.py", source_path="shared.py", current_path="shared.py",
                command_argv=[data["tools"]["resolved_python"], "-m", "ruff", "format", "--check", "--force-exclude", "--no-cache", "./shared.py"],
            )
            data["classifications"]["current_line_drift"] = [swapped]

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

        def missing_post_cut_correction(data: dict[str, Any], repo: Path) -> None:
            record = data["cleanup_records"][1]
            raw = task_bytes(
                record["task_id"],
                record["label"],
                data["batches"][1]["paths"],
                record["dependencies"],
                True,
                drop_post_cut_correction=True,
            )
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
            ("missing-post-cut-correction", "E_FINAL_GATE", missing_post_cut_correction, "final"),
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
    print("manifest self-tests: 2 positive phases and 34 deterministic mutations passed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pre-records", "final"))
    parser.add_argument("--manifest")
    parser.add_argument("--repo")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--require-live-current", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        need(
            args.phase is None
            and args.manifest is None
            and args.repo is None
            and not args.require_live_current,
            "E_ARGS",
            "--self-test is exclusive",
        )
        run_self_tests()
        return 0
    need(args.phase is not None and args.manifest is not None, "E_ARGS", "--phase and --manifest are required")
    need(args.phase != "final" or args.repo is not None, "E_ARGS", "final requires --repo")
    need(not args.require_live_current or args.repo is not None, "E_ARGS", "--require-live-current requires --repo")
    manifest = load_canonical_manifest(Path(args.manifest))
    counts = validate(
        manifest,
        args.phase,
        Path(args.repo).resolve() if args.repo else None,
        require_live_current=args.require_live_current,
    )
    print(json.dumps(counts, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ManifestError, KeyError, IndexError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)

```
<!-- TASK-26000-CHECKER-SOURCE-END -->

The self-test's deterministic first failures include:

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
malformed-captured-ref          E_REACHABILITY
live-current-mismatch           E_ORIGIN_DEV
noncanonical-bytes (three)      E_CANONICAL_BYTES
temporal-range (two)            E_TEMPORAL_RANGE
temporal-evidence               E_TEMPORAL_EVIDENCE
temporal-order                  E_TEMPORAL_ORDER
temporal-result                 E_TEMPORAL_RESULT
temporal-path                   E_TEMPORAL_PATH
temporal-config                 E_TEMPORAL_CONFIG
temporal-exclusion              E_TEMPORAL_EXCLUSION
temporal-completeness (two)     E_TEMPORAL_COMPLETENESS
```

It prints exactly
`manifest self-tests: 2 positive phases and 34 deterministic mutations passed`
only after both positive phases and all mutations pass.

---

## Appendix B.1: Durable Tool Authority Materializer

Materialize this file verbatim as `task26000_tmp_root/task26000_tool_authority.py`. Before executing it, require SHA-256 `353160bc073aef50dfcf51f55bd18e261c58e91147db9df30a6e3d0d0f5a2977`. It authenticates its own tracked source against the executing file, then deterministically extracts the tracked producer, checker, allocator, and renderer sources and requires both every adjacent marker and every extracted byte digest to equal the closed approved child hashes embedded below. The shell independently rechecks the same four reviewed literals after extraction.

<!-- TASK-26000-AUTHORITY-MATERIALIZER-BEGIN sha256=353160bc073aef50dfcf51f55bd18e261c58e91147db9df30a6e3d0d0f5a2977 -->
```python
from __future__ import annotations

import argparse
import hashlib
import os
import re
import tempfile
from pathlib import Path


class AuthorityError(RuntimeError):
    pass


EXPECTED_CHILD_SHA256 = {
    "producer": "fd33448f2841d0502509201a5bf6fd2f279f3f2c67cff8f3d4391b9ed7d9ce3e",
    "checker": "a003aee74e01c2729136e244474f1fac08a06ae9ee9331752f56d1bfbffe9e79",
    "allocator": "2e456e41bdd2b4f357d181a32b91efdfd07060c33a8f23cc1622d3ef8a4bd432",
    "renderer": "4a08b6a5a9a8b12926ab9417bc330a4e94eb60c3b4afe88226ef232e2653a17a",
}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def extract_source(plan: Path, name: str) -> tuple[bytes, str]:
    raw = plan.read_bytes()
    upper = name.upper()
    pattern = re.compile(
        rb"<!-- TASK-26000-"
        + upper.encode("ascii")
        + rb"-SOURCE-BEGIN sha256=([0-9a-f]{64}) -->\n```python\n"
    )
    matches = list(pattern.finditer(raw))
    if len(matches) != 1:
        raise AuthorityError(f"E_TOOL_AUTHORITY: {name} source marker count={len(matches)}")
    match = matches[0]
    end_marker = (
        b"\n```\n<!-- TASK-26000-"
        + upper.encode("ascii")
        + b"-SOURCE-END -->"
    )
    end = raw.find(end_marker, match.end())
    if end < 0 or raw.find(end_marker, end + 1) >= 0:
        raise AuthorityError(f"E_TOOL_AUTHORITY: {name} source end marker")
    source = raw[match.end():end]
    expected = match.group(1).decode("ascii")
    anchored = EXPECTED_CHILD_SHA256.get(name)
    if anchored is None or expected != anchored:
        raise AuthorityError(f"E_TOOL_AUTHORITY: {name} anchored digest mismatch")
    if not source.endswith(b"\n") or digest(source) != anchored:
        raise AuthorityError(f"E_TOOL_AUTHORITY: {name} tracked digest mismatch")
    return source, anchored


def extract_materializer(plan: Path) -> tuple[bytes, str]:
    raw = plan.read_bytes()
    pattern = re.compile(
        rb"<!-- TASK-26000-AUTHORITY-MATERIALIZER-BEGIN sha256=([0-9a-f]{64}) -->\n```python\n"
    )
    matches = list(pattern.finditer(raw))
    if len(matches) != 1:
        raise AuthorityError(
            f"E_TOOL_AUTHORITY: materializer source marker count={len(matches)}"
        )
    match = matches[0]
    end_marker = b"\n```\n<!-- TASK-26000-AUTHORITY-MATERIALIZER-END -->"
    end = raw.find(end_marker, match.end())
    if end < 0 or raw.find(end_marker, end + 1) >= 0:
        raise AuthorityError("E_TOOL_AUTHORITY: materializer source end marker")
    source = raw[match.end():end]
    expected = match.group(1).decode("ascii")
    if not source.endswith(b"\n") or digest(source) != expected:
        raise AuthorityError("E_TOOL_AUTHORITY: materializer tracked digest mismatch")
    return source, expected


def verify_self(plan: Path, executable: Path) -> str:
    source, expected = extract_materializer(plan)
    try:
        actual = executable.read_bytes()
    except OSError as exc:
        raise AuthorityError(
            f"E_TOOL_AUTHORITY: materializer unavailable: {exc}"
        ) from exc
    if actual != source or digest(actual) != expected:
        raise AuthorityError("E_TOOL_AUTHORITY: materializer bytes mismatch")
    return expected


def publish(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_value = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_value)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def expected_sources(plan: Path) -> dict[str, tuple[bytes, str]]:
    return {
        name: extract_source(plan, name)
        for name in ("producer", "checker", "allocator", "renderer")
    }


def verify_materialized(
    plan: Path,
    producer: Path,
    checker: Path,
    allocator: Path,
    renderer: Path,
) -> dict[str, str]:
    expected = expected_sources(plan)
    result: dict[str, str] = {}
    for name, path in (
        ("producer", producer),
        ("checker", checker),
        ("allocator", allocator),
        ("renderer", renderer),
    ):
        source, expected_digest = expected[name]
        try:
            actual = path.read_bytes()
        except OSError as exc:
            raise AuthorityError(f"E_TOOL_AUTHORITY: {name} unavailable: {exc}") from exc
        if actual != source or digest(actual) != expected_digest:
            raise AuthorityError(f"E_TOOL_AUTHORITY: {name} materialized digest mismatch")
        result[f"{name}_sha256"] = expected_digest
    return result


def materialize(
    plan: Path,
    producer: Path,
    checker: Path,
    allocator: Path,
    renderer: Path,
) -> dict[str, str]:
    expected = expected_sources(plan)
    publish(producer, expected["producer"][0])
    publish(checker, expected["checker"][0])
    publish(allocator, expected["allocator"][0])
    publish(renderer, expected["renderer"][0])
    return verify_materialized(plan, producer, checker, allocator, renderer)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--checker", type=Path, required=True)
    parser.add_argument("--allocator", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        verify_self(args.plan, Path(__file__))
        result = (
            verify_materialized(
                args.plan, args.producer, args.checker, args.allocator, args.renderer
            )
            if args.verify_only
            else materialize(
                args.plan, args.producer, args.checker, args.allocator, args.renderer
            )
        )
    except AuthorityError as exc:
        print(str(exc), file=os.sys.stderr)
        return 2
    print(
        f"producer={result['producer_sha256']} checker={result['checker_sha256']} "
        f"allocator={result['allocator_sha256']} renderer={result['renderer_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

```
<!-- TASK-26000-AUTHORITY-MATERIALIZER-END -->

## Appendix C: Exact Collision-Safe Task-ID Scanner

Task 5 materializes this temporary scanner. It accepts `--manifest`, `--output`,
optional `--expect-map`, and fixture-only `--self-test`; it reads batch labels plus
`final_batch_label` from the
manifest, accepts only an exact canonical closed and immutable-OID-bound scanner audit or validated active-state
handoff as `--expect-map`, writes a canonical closed nine-key audit JSON, and exits 2 on a moved PR head, malformed task
identity, inaccessible checkout/ref, self-claim mismatch, external ID collision, or
changed precreate allocation. An immutable commit with no `backlog` tree contributes zero
claims; invalid OIDs, non-commit objects, and unexpected tree probes fail `E_ARCHIVE`.

<!-- TASK-26000-ALLOCATOR-SOURCE-BEGIN sha256=2e456e41bdd2b4f357d181a32b91efdfd07060c33a8f23cc1622d3ef8a4bd432 -->
```python
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

FILE_ID = re.compile(r"^task-(\d+)(?:\.\d+)* - .+\.md$", re.IGNORECASE)
FRONT_ID = re.compile(r"(?m)^id:[ \t]*(?i:TASK)-(\d+)(?:\.\d+)*[ \t]*$")
BUCKETS = (
    "backlog/tasks/",
    "backlog/completed/",
    "backlog/archive/tasks/",
    "backlog/drafts/",
)
SCANNER_AUDIT_KEYS = frozenset(
    {
        "manifest_pin",
        "observed_origin_dev",
        "origin_dev_ancestry",
        "refs",
        "open_prs",
        "worktrees",
        "claims",
        "external_used_ids",
        "allocation",
    }
)
ACTIVE_STATE_KEYS = frozenset(
    {
        "schema_version",
        "mode",
        "allocation",
        "paths0_output",
        "paths0_sha256",
        "record_set_sha256",
    }
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


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def read_canonical_json(path: Path, code: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AllocationError(f"{code}: {path}: {type(exc).__name__}") from exc
    fail(isinstance(value, dict), code, "root is not an object")
    fail(raw == canonical_json_bytes(value), code, "JSON bytes are not canonical")
    return value


def valid_oid(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def valid_digest(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_scanner_audit(
    value: dict[str, Any], manifest: dict[str, Any], repo: Path, code: str
) -> None:
    fail(set(value) == SCANNER_AUDIT_KEYS, code, "scanner audit keys")
    fail(
        all(valid_oid(value[key]) for key in ("manifest_pin", "observed_origin_dev")),
        code,
        "scanner audit revisions",
    )
    fail(
        value["origin_dev_ancestry"] in {"equal", "fast_forward_descendant"},
        code,
        "scanner audit ancestry",
    )
    refs_value = value["refs"]
    fail(isinstance(refs_value, list), code, "scanner audit refs")
    for row in refs_value:
        fail(
            isinstance(row, dict)
            and set(row) == {"ref", "oid"}
            and isinstance(row["ref"], str)
            and re.fullmatch(r"refs/(?:heads|remotes/origin)/.+", row["ref"])
            is not None
            and valid_oid(row["oid"]),
            code,
            "scanner audit ref row",
        )
    fail(
        refs_value == sorted(refs_value, key=lambda row: row["ref"]),
        code,
        "scanner audit ref order",
    )

    prs = value["open_prs"]
    fail(isinstance(prs, list), code, "scanner audit open PRs")
    fail(
        all(
            isinstance(row, dict)
            and set(row) == {"number", "head_oid"}
            and type(row["number"]) is int
            and row["number"] > 0
            and valid_oid(row["head_oid"])
            for row in prs
        )
        and prs == sorted(prs, key=lambda row: row["number"])
        and len({row["number"] for row in prs}) == len(prs),
        code,
        "scanner audit open PR rows",
    )

    worktrees = value["worktrees"]
    fail(isinstance(worktrees, list), code, "scanner audit worktrees")
    decoded_paths: list[bytes] = []
    for row in worktrees:
        fail(
            isinstance(row, dict)
            and set(row) == {"path_b64", "head", "dirty"}
            and isinstance(row["path_b64"], str)
            and valid_oid(row["head"])
            and type(row["dirty"]) is bool,
            code,
            "scanner audit worktree row",
        )
        try:
            decoded = base64.b64decode(row["path_b64"], validate=True)
        except (ValueError, binascii.Error):
            decoded = b""
        fail(
            bool(decoded)
            and base64.b64encode(decoded).decode("ascii") == row["path_b64"],
            code,
            "scanner audit worktree path",
        )
        decoded_paths.append(decoded)
    fail(len(set(decoded_paths)) == len(decoded_paths), code, "scanner audit worktree duplicates")

    claims = value["claims"]
    fail(isinstance(claims, dict), code, "scanner audit claims")
    for task_id, rows in claims.items():
        fail(
            isinstance(task_id, str)
            and re.fullmatch(r"[0-9]+", task_id) is not None
            and str(int(task_id)) == task_id
            and isinstance(rows, list)
            and bool(rows),
            code,
            "scanner audit claim group",
        )
        for row in rows:
            fail(
                isinstance(row, dict)
                and set(row)
                == {
                    "path",
                    "batch_label",
                    "content_sha256",
                    "sources",
                    "accepted_self",
                }
                and isinstance(row["path"], str)
                and row["path"].startswith(BUCKETS)
                and "\n" not in row["path"]
                and "\x00" not in row["path"]
                and (
                    row["batch_label"] is None
                    or (
                        isinstance(row["batch_label"], str)
                        and re.fullmatch(
                            r"[a-z0-9]+(?:-[a-z0-9]+)*", row["batch_label"]
                        )
                        is not None
                    )
                )
                and valid_digest(row["content_sha256"])
                and isinstance(row["sources"], list)
                and bool(row["sources"])
                and row["sources"] == sorted(set(row["sources"]))
                and all(isinstance(source, str) and source for source in row["sources"])
                and type(row["accepted_self"]) is bool,
                code,
                "scanner audit claim row",
            )

    external_ids = value["external_used_ids"]
    fail(
        isinstance(external_ids, list)
        and external_ids == sorted(set(external_ids))
        and all(type(task_id) is int and task_id >= 0 for task_id in external_ids),
        code,
        "scanner audit external IDs",
    )
    allocation = value["allocation"]
    fail(
        isinstance(allocation, dict)
        and all(
            isinstance(label, str)
            and re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", label) is not None
            and type(task_id) is int
            and task_id > 26000
            for label, task_id in allocation.items()
        )
        and len(set(allocation.values())) == len(allocation),
        code,
        "scanner audit allocation",
    )

    revisions = manifest.get("revisions")
    manifest_pin = revisions.get("current") if isinstance(revisions, dict) else None
    fail(
        valid_oid(manifest_pin) and value["manifest_pin"] == manifest_pin,
        "E_ORIGIN_DEV_DIVERGED",
        "scanner audit manifest pin differs from manifest.revisions.current",
    )
    origin_dev = [
        row["oid"]
        for row in refs_value
        if row["ref"] == "refs/remotes/origin/dev"
    ]
    fail(
        len(origin_dev) == 1 and origin_dev[0] == value["observed_origin_dev"],
        "E_ORIGIN_DEV_DIVERGED",
        "scanner audit origin/dev snapshot is missing, duplicated, or mismatched",
    )
    fail(
        len({row["ref"] for row in refs_value}) == len(refs_value),
        code,
        "scanner audit duplicate ref",
    )
    pin = value["manifest_pin"]
    observed = value["observed_origin_dev"]
    for name, oid in (("manifest pin", pin), ("observed origin/dev", observed)):
        try:
            commit = subprocess.run(
                ("git", "cat-file", "-e", f"{oid}^{{commit}}"),
                cwd=repo,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except OSError as exc:
            raise AllocationError(
                f"E_ORIGIN_DEV_DIVERGED: {name} object check: {type(exc).__name__}"
            ) from exc
        fail(
            commit.returncode == 0,
            "E_ORIGIN_DEV_DIVERGED",
            f"{name} commit is unavailable",
        )
    if value["origin_dev_ancestry"] == "equal":
        fail(
            observed == pin,
            "E_ORIGIN_DEV_DIVERGED",
            "equal ancestry has distinct immutable OIDs",
        )
    else:
        fail(
            observed != pin,
            "E_ORIGIN_DEV_DIVERGED",
            "fast-forward ancestry has equal immutable OIDs",
        )
        try:
            ancestor = subprocess.run(
                ("git", "merge-base", "--is-ancestor", pin, observed),
                cwd=repo,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except OSError as exc:
            raise AllocationError(
                f"E_ORIGIN_DEV_DIVERGED: immutable ancestry check: {type(exc).__name__}"
            ) from exc
        fail(
            ancestor.returncode == 0,
            "E_ORIGIN_DEV_DIVERGED",
            "stored origin/dev OID does not descend from the manifest pin",
        )


def valid_task_path(value: Any) -> bool:
    if not isinstance(value, str) or "\n" in value or "\x00" in value:
        return False
    relative = PurePosixPath(value)
    return (
        not relative.is_absolute()
        and len(relative.parts) == 3
        and relative.parts[:2] == ("backlog", "tasks")
        and relative.parts[2] not in {".", ".."}
    )


def manifest_record_identities(manifest: dict[str, Any], code: str) -> list[dict[str, Any]]:
    records = manifest.get("cleanup_records")
    fail(isinstance(records, list), code, "manifest cleanup records")
    identities: list[dict[str, Any]] = []
    for row in records:
        fail(
            isinstance(row, dict)
            and isinstance(row.get("label"), str)
            and valid_task_path(row.get("path"))
            and type(row.get("task_id")) is int
            and row["task_id"] > 26000,
            code,
            "manifest cleanup record identity",
        )
        identities.append(
            {"label": row["label"], "path": row["path"], "task_id": row["task_id"]}
        )
    identities.sort(key=lambda row: row["label"])
    fail(
        len({row["label"] for row in identities}) == len(identities),
        code,
        "duplicate manifest cleanup label",
    )
    return identities


def validate_active_state(
    value: dict[str, Any], path: Path, manifest: dict[str, Any], repo: Path, code: str
) -> None:
    fail(set(value) == ACTIVE_STATE_KEYS, code, "active-state keys")
    fail(
        type(value["schema_version"]) is int
        and value["schema_version"] == 1
        and value["mode"] in {"create", "reallocate"},
        code,
        "active-state schema or mode",
    )
    identities = manifest_record_identities(manifest, code)
    expected_allocation = {row["label"]: row["task_id"] for row in identities}
    fail(value["allocation"] == expected_allocation, code, "active allocation")
    expected_record_digest = hashlib.sha256(canonical_json_bytes(identities)).hexdigest()
    fail(
        value["record_set_sha256"] == expected_record_digest,
        code,
        "active record-set digest",
    )

    raw_paths0 = value["paths0_output"]
    fail(isinstance(raw_paths0, str), code, "active paths0 path")
    paths0_candidate = Path(raw_paths0)
    fail(
        paths0_candidate.is_absolute()
        and not paths0_candidate.is_symlink()
        and paths0_candidate.is_file(),
        code,
        "active paths0 authority",
    )
    paths0 = paths0_candidate.resolve()
    expected_parent = path.resolve().parent / "raw"
    expected_name = (
        "new-task-paths0" if value["mode"] == "create" else "reallocated-task-paths0"
    )
    fail(
        paths0.parent == expected_parent
        and paths0.name == expected_name
        and not expected_parent.is_symlink(),
        code,
        "active paths0 authority",
    )
    try:
        paths_raw = paths0.read_bytes()
    except OSError as exc:
        raise AllocationError(f"{code}: active paths0: {type(exc).__name__}") from exc
    fail(
        isinstance(value["paths0_sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", value["paths0_sha256"]) is not None
        and hashlib.sha256(paths_raw).hexdigest() == value["paths0_sha256"],
        code,
        "active paths0 digest",
    )
    fields = paths_raw.split(b"\0")
    fail(bool(fields) and fields[-1] == b"" and all(fields[:-1]), code, "active paths0 framing")
    try:
        decoded = [field.decode("utf-8") for field in fields[:-1]]
    except UnicodeDecodeError as exc:
        raise AllocationError(f"{code}: active paths0 encoding") from exc
    fail(
        decoded == sorted(set(decoded)) and all(valid_task_path(item) for item in decoded),
        code,
        "active paths0 entries",
    )
    current_paths = {row["path"] for row in identities}
    fail(current_paths <= set(decoded), code, "active current paths")
    retired_paths = set(decoded) - current_paths
    fail(value["mode"] == "reallocate" or not retired_paths, code, "active retired paths")
    fail(
        not any(
            os.path.lexists(repo.joinpath(*PurePosixPath(item).parts))
            for item in retired_paths
        ),
        code,
        "active retired path still exists",
    )


def read_expected_allocation(
    path: Path, manifest: dict[str, Any], repo: Path
) -> dict[str, int]:
    value = read_canonical_json(path, "E_EXPECT_MAP")
    if set(value) == SCANNER_AUDIT_KEYS:
        validate_scanner_audit(value, manifest, repo, "E_EXPECT_MAP")
    elif set(value) == ACTIVE_STATE_KEYS:
        validate_active_state(value, path, manifest, repo, "E_EXPECT_MAP")
    else:
        raise AllocationError("E_EXPECT_MAP: unrecognized closed document shape")
    allocation = value["allocation"]
    fail(isinstance(allocation, dict), "E_EXPECT_MAP", "allocation is not an object")
    return allocation


def parse_claim(path: str, raw: bytes, source: str) -> ParsedClaim | None:
    basename = PurePosixPath(path).name
    filename = FILE_ID.fullmatch(basename)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AllocationError(
            f"E_TASK_IDENTITY: {source}:{path}: invalid UTF-8"
        ) from exc
    front = ""
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        fail(end >= 0, "E_TASK_IDENTITY", f"{source}:{path}: unterminated frontmatter")
        front = text[4:end]
    front_ids = FRONT_ID.findall(front)
    if filename is None:
        fail(
            not front_ids and not basename.lower().startswith("task-"),
            "E_TASK_IDENTITY",
            f"{source}:{path}",
        )
        return None
    fail(len(front_ids) <= 1, "E_TASK_IDENTITY", f"{source}:{path}")
    fail(
        not front_ids or filename.group(1) == front_ids[0],
        "E_TASK_IDENTITY",
        f"{source}:{path}",
    )
    batch_markers = re.findall(
        r"(?m)^<!-- TASK-26000-BATCH: ([a-z0-9]+(?:-[a-z0-9]+)*) -->$",
        text,
    )
    fail(len(batch_markers) <= 1, "E_TASK_IDENTITY", f"{source}:{path}: batch markers")
    return ParsedClaim(
        task_id=int(filename.group(1)),
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
    fail(
        re.fullmatch(r"[0-9a-f]{40}", revision) is not None,
        "E_ARCHIVE",
        f"{source}: revision is not an immutable OID",
    )
    backlog_tree = execute(
        (
            "git",
            "ls-tree",
            "-d",
            "--name-only",
            f"{revision}^{{commit}}",
            "--",
            "backlog",
        ),
        repo,
        "E_ARCHIVE",
    )
    fail(
        backlog_tree in {b"", b"backlog\n"},
        "E_ARCHIVE",
        f"{source}:{revision}: unexpected backlog tree probe {backlog_tree!r}",
    )
    if not backlog_tree:
        return
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


def boundary_error(path: str, detail: str) -> AllocationError:
    return AllocationError(f"E_WORKTREE_BOUNDARY: {path}: {detail}")


def stable_identity(before: os.stat_result, after: os.stat_result) -> bool:
    return before.st_dev == after.st_dev and before.st_ino == after.st_ino


def open_directory_nofollow(
    path: str | Path, before: os.stat_result, display: str, parent_fd: int | None = None
) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise boundary_error(display, "O_NOFOLLOW is unavailable")
    flags = (
        os.O_RDONLY
        | nofollow
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags, dir_fd=parent_fd)
        opened = os.fstat(descriptor)
    except OSError as exc:
        raise boundary_error(display, type(exc).__name__) from exc
    if not stat.S_ISDIR(opened.st_mode) or not stable_identity(before, opened):
        os.close(descriptor)
        raise boundary_error(display, "directory changed during no-follow open")
    return descriptor


def read_regular_nofollow(
    parent_fd: int, name: str, before: os.stat_result, display: str
) -> bytes:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise boundary_error(display, "O_NOFOLLOW is unavailable")
    flags = (
        os.O_RDONLY
        | nofollow
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise boundary_error(display, type(exc).__name__) from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not stable_identity(before, opened):
            raise boundary_error(display, "file changed during no-follow open")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        finished = os.fstat(descriptor)
        if (
            not stable_identity(opened, finished)
            or opened.st_size != finished.st_size
            or opened.st_mtime_ns != finished.st_mtime_ns
        ):
            raise boundary_error(display, "file changed during read")
        return b"".join(chunks)
    except OSError as exc:
        raise boundary_error(display, type(exc).__name__) from exc
    finally:
        os.close(descriptor)


def scan_directory_nofollow(
    descriptor: int,
    relative: str,
    source: str,
    claims: dict[int, dict[ClaimIdentity, set[str]]],
) -> None:
    try:
        names = sorted(os.listdir(descriptor))
    except OSError as exc:
        raise boundary_error(relative, type(exc).__name__) from exc
    for name in names:
        path = f"{relative}/{name}"
        try:
            entry = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except OSError as exc:
            raise boundary_error(path, type(exc).__name__) from exc
        if stat.S_ISLNK(entry.st_mode):
            raise boundary_error(path, "symlink entry")
        if stat.S_ISDIR(entry.st_mode):
            child = open_directory_nofollow(name, entry, path, descriptor)
            try:
                scan_directory_nofollow(child, path, source, claims)
                if not stable_identity(entry, os.fstat(child)):
                    raise boundary_error(path, "directory changed during scan")
            finally:
                os.close(child)
        elif stat.S_ISREG(entry.st_mode):
            claim(
                path,
                read_regular_nofollow(descriptor, name, entry, path),
                source,
                claims,
            )
        else:
            raise boundary_error(path, "non-regular entry")


def open_bucket_nofollow(
    root_descriptor: int, root: Path, bucket: str
) -> int | None:
    current = root_descriptor
    owned = False
    display = root
    try:
        for component in PurePosixPath(bucket).parts:
            display /= component
            try:
                entry = os.stat(
                    component, dir_fd=current, follow_symlinks=False
                )
            except FileNotFoundError:
                if owned:
                    os.close(current)
                return None
            except OSError as exc:
                raise boundary_error(
                    display.as_posix(), type(exc).__name__
                ) from exc
            if not stat.S_ISDIR(entry.st_mode):
                raise boundary_error(
                    display.as_posix(), "bucket component is not a real directory"
                )
            child = open_directory_nofollow(
                component, entry, display.as_posix(), current
            )
            if owned:
                os.close(current)
            current = child
            owned = True
        return current
    except BaseException:
        if owned:
            os.close(current)
        raise


def scan_worktree_files(
    root: Path,
    source: str,
    claims: dict[int, dict[ClaimIdentity, set[str]]],
) -> None:
    try:
        root_entry = os.lstat(root)
    except OSError as exc:
        raise boundary_error(root.as_posix(), type(exc).__name__) from exc
    if not stat.S_ISDIR(root_entry.st_mode):
        raise boundary_error(root.as_posix(), "worktree root is not a real directory")
    root_descriptor = open_directory_nofollow(
        root, root_entry, root.as_posix()
    )
    try:
        for bucket in BUCKETS:
            descriptor = open_bucket_nofollow(root_descriptor, root, bucket)
            if descriptor is None:
                continue
            try:
                bucket_entry = os.fstat(descriptor)
                scan_directory_nofollow(
                    descriptor, bucket.rstrip("/"), source, claims
                )
                if not stable_identity(bucket_entry, os.fstat(descriptor)):
                    raise boundary_error(
                        (root / bucket).as_posix(), "bucket changed during scan"
                    )
            finally:
                os.close(descriptor)
        if not stable_identity(root_entry, os.fstat(root_descriptor)):
            raise boundary_error(root.as_posix(), "worktree root changed during scan")
    finally:
        os.close(root_descriptor)


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
    for task_id, allowed in sorted(self_claims.items()):
        fail(
            allowed in claims.get(task_id, {}),
            "E_SELF_CLAIMS",
            f"TASK-{task_id}: authenticated self identity absent from live claims",
        )
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


def select_allocation(
    labels: list[str],
    final_label: str,
    external_ids: set[int],
    expected_map: dict[str, int] | None,
    self_claims: dict[int, ClaimIdentity],
) -> dict[str, int]:
    if not self_claims:
        return allocate_ids(labels, final_label, external_ids)
    fail(expected_map is not None, "E_SELF_CLAIMS", "expected allocation is missing")
    fail(
        set(self_claims) == set(expected_map.values()),
        "E_SELF_CLAIMS",
        "authenticated self IDs differ from expected allocation",
    )
    return dict(expected_map)


def verify_pr_head(number: int, expected: str, actual: str) -> None:
    if actual != expected:
        raise AllocationError(
            f"E_PR_HEAD_MOVED: PR {number} expected {expected}; fetched {actual}"
        )


def verify_origin_dev_ancestry(repo: Path, manifest_pin: str) -> dict[str, str]:
    """Authenticate origin/dev as the pinned authority cut or its descendant."""
    fail(
        isinstance(manifest_pin, str)
        and re.fullmatch(r"[0-9a-f]{40}", manifest_pin) is not None,
        "E_ORIGIN_DEV_DIVERGED",
        "manifest pin is missing or malformed",
    )
    pin = subprocess.run(
        ("git", "cat-file", "-e", f"{manifest_pin}^{{commit}}"),
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    fail(pin.returncode == 0, "E_ORIGIN_DEV_DIVERGED", "manifest pin is unavailable")
    tip = subprocess.run(
        ("git", "rev-parse", "--verify", "refs/remotes/origin/dev^{commit}"),
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    observed = tip.stdout.decode("ascii", "replace").strip()
    fail(
        tip.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", observed) is not None,
        "E_ORIGIN_DEV_DIVERGED",
        "origin/dev tip is missing or malformed",
    )
    if observed == manifest_pin:
        ancestry = "equal"
    else:
        ancestor = subprocess.run(
            ("git", "merge-base", "--is-ancestor", manifest_pin, observed),
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        fail(
            ancestor.returncode == 0,
            "E_ORIGIN_DEV_DIVERGED",
            f"manifest pin {manifest_pin} is not an ancestor of origin/dev {observed}",
        )
        ancestry = "fast_forward_descendant"
    return {
        "manifest_pin": manifest_pin,
        "observed_origin_dev": observed,
        "origin_dev_ancestry": ancestry,
    }


def verify_origin_dev_snapshot_binding(
    authority: dict[str, str], ref_snapshot: list[tuple[str, str]]
) -> None:
    """Bind the ancestry observation to the exact origin/dev ref being audited."""
    origin_dev_oids = [
        oid for name, oid in ref_snapshot if name == "refs/remotes/origin/dev"
    ]
    fail(
        len(origin_dev_oids) == 1,
        "E_ORIGIN_DEV_DIVERGED",
        "origin/dev is missing or duplicated in the ref audit snapshot",
    )
    fail(
        authority.get("observed_origin_dev") == origin_dev_oids[0],
        "E_ORIGIN_DEV_DIVERGED",
        "origin/dev changed between ancestry verification and the ref audit snapshot",
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
    authority = verify_origin_dev_ancestry(repo, manifest["revisions"]["current"])
    ref_snapshot = refs(repo)
    verify_origin_dev_snapshot_binding(authority, ref_snapshot)
    claims: dict[int, dict[ClaimIdentity, set[str]]] = {}
    ref_audit: list[dict[str, str]] = []
    for name, oid in ref_snapshot:
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
    allocation = select_allocation(
        labels, final_label, external_ids, expected_map, self_claims
    )
    return {
        **authority,
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

    for prefix in (b"task", b"TaSk"):
        lowercase_claim = parse_claim(
            self_path,
            self_raw.replace(b"id: TASK", b"id: " + prefix),
            "refs/heads/fixture",
        )
        fail(
            lowercase_claim is not None and lowercase_claim.task_id == 26100,
            "E_SELF_TEST",
            "case-insensitive TASK prefix",
        )
    cases += 1

    dotted_claim = parse_claim(
        "backlog/tasks/task-26100.3 - alpha.md",
        b"---\nid: task-26100.7\n---\nfixture\n",
        "refs/heads/fixture",
    )
    fail(
        dotted_claim is not None and dotted_claim.task_id == 26100,
        "E_SELF_TEST",
        "lowercase dotted task identity",
    )
    cases += 1

    for raw in (
        b"legacy task body\n",
        b"---\nid: taREDACTED-26100\n---\nlegacy task body\n",
    ):
        filename_claim = parse_claim(self_path, raw, "refs/heads/fixture")
        fail(
            filename_claim is not None and filename_claim.task_id == 26100,
            "E_SELF_TEST",
            "filename-only task identity",
        )
    cases += 1

    for raw in (
        b"---\nid: TASK-26100\nlegacy task body\n",
        b"---\nid: TASK-26100\nid: task-26100\n---\nlegacy task body\n",
    ):
        expect_error(
            "E_TASK_IDENTITY",
            lambda raw=raw: parse_claim(self_path, raw, "refs/heads/fixture"),
        )
    cases += 1

    expect_error(
        "E_TASK_IDENTITY",
        lambda: parse_claim(
            "backlog/tasks/task-26100-invalid.md",
            b"legacy task body\n",
            "refs/heads/fixture",
        ),
    )
    cases += 1

    expect_error(
        "E_TASK_IDENTITY",
        lambda: parse_claim(self_path, b"\xff", "refs/heads/fixture"),
    )
    cases += 1

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

    preserved = select_allocation(
        ["alpha"], "alpha", {26834}, expected, {26100: self_claim.identity}
    )
    refreshed = select_allocation(["alpha"], "alpha", {26834}, expected, {})
    fail(
        preserved == expected and refreshed == {"alpha": 26934},
        "E_SELF_TEST",
        "allocation lifecycle",
    )
    cases += 1

    expect_error(
        "E_SELF_CLAIMS",
        lambda: classify_claims({}, {26100: self_claim.identity}, expected),
    )
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

    for entry_kind in (
        "symlinked-backlog",
        "symlinked-archive",
        "symlinked-bucket",
        "symlinked-file",
        "symlinked-directory",
        "non-directory-bucket",
        "fifo",
    ):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root) / "repo"
            root.mkdir()
            backlog = root / "backlog"
            bucket = backlog / "tasks"
            outside = Path(raw_root) / "outside"
            outside.mkdir()
            outside_task = outside / "task-26100 - outside.md"
            if entry_kind in {"symlinked-backlog", "symlinked-archive"}:
                (outside / "tasks").mkdir()
                (outside / "tasks" / outside_task.name).write_bytes(
                    task_bytes(26100, "alpha")
                )
            else:
                outside_task.write_bytes(task_bytes(26100, "alpha"))
            if entry_kind == "symlinked-backlog":
                backlog.symlink_to(outside, target_is_directory=True)
            elif entry_kind == "symlinked-archive":
                backlog.mkdir()
                (backlog / "archive").symlink_to(
                    outside, target_is_directory=True
                )
            else:
                backlog.mkdir()
            if entry_kind not in {"symlinked-backlog", "symlinked-archive"}:
                if entry_kind == "symlinked-bucket":
                    bucket.symlink_to(outside, target_is_directory=True)
                elif entry_kind == "non-directory-bucket":
                    bucket.write_text("not a directory\n", encoding="utf-8")
                else:
                    bucket.mkdir()
                    if entry_kind == "symlinked-file":
                        (bucket / outside_task.name).symlink_to(outside_task)
                    elif entry_kind == "symlinked-directory":
                        (bucket / "linked").symlink_to(
                            outside, target_is_directory=True
                        )
                    else:
                        os.mkfifo(bucket / "task-26100 - fifo.md")
            expect_error(
                "E_WORKTREE_BOUNDARY",
                lambda root=root: scan_worktree_files(
                    root, "worktree:fixture", {}
                ),
            )
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

    with tempfile.TemporaryDirectory() as raw_root:
        root = Path(raw_root)
        repo = root / "repo"
        repo.mkdir()
        execute(("git", "init", "-q", "-b", "main"), repo, "E_SELF_TEST")
        execute(("git", "config", "user.name", "Task 26000 Test"), repo, "E_SELF_TEST")
        execute(("git", "config", "user.email", "task26000@example.invalid"), repo, "E_SELF_TEST")
        (repo / "fixture.txt").write_text("authority\n", encoding="utf-8")
        execute(("git", "add", "fixture.txt"), repo, "E_SELF_TEST")
        execute(("git", "commit", "-q", "-m", "authority"), repo, "E_SELF_TEST")
        manifest_pin = execute(
            ("git", "rev-parse", "HEAD"), repo, "E_SELF_TEST"
        ).decode("ascii").strip()
        task_path = "backlog/tasks/task-26100 - alpha.md"
        manifest = {
            "revisions": {"current": manifest_pin},
            "cleanup_records": [
                {"label": "alpha", "path": task_path, "task_id": 26100}
            ]
        }
        scanner_path = root / "allocation.json"
        scanner_audit = {
            "manifest_pin": manifest_pin,
            "observed_origin_dev": manifest_pin,
            "origin_dev_ancestry": "equal",
            "refs": [
                {"ref": "refs/remotes/origin/dev", "oid": manifest_pin}
            ],
            "open_prs": [],
            "worktrees": [],
            "claims": {},
            "external_used_ids": [],
            "allocation": expected,
        }
        scanner_path.write_bytes(canonical_json_bytes(scanner_audit))
        fail(
            read_expected_allocation(scanner_path, manifest, repo) == expected,
            "E_SELF_TEST",
            "canonical scanner audit",
        )
        cases += 1
        scanner_path.write_text(json.dumps(scanner_audit), encoding="utf-8")
        expect_error(
            "E_EXPECT_MAP",
            lambda: read_expected_allocation(scanner_path, manifest, repo),
        )
        cases += 1
        paths0 = root / "raw/new-task-paths0"
        paths0.parent.mkdir()
        paths_raw = task_path.encode("utf-8") + b"\0"
        paths0.write_bytes(paths_raw)
        identities = [{"label": "alpha", "path": task_path, "task_id": 26100}]
        active_path = root / "active-cleanup-state.json"
        active = {
            "schema_version": 1,
            "mode": "create",
            "allocation": expected,
            "paths0_output": os.fspath(paths0),
            "paths0_sha256": hashlib.sha256(paths_raw).hexdigest(),
            "record_set_sha256": hashlib.sha256(
                canonical_json_bytes(identities)
            ).hexdigest(),
        }
        active_path.write_bytes(canonical_json_bytes(active))
        fail(
            read_expected_allocation(active_path, manifest, repo) == expected,
            "E_SELF_TEST",
            "canonical active state",
        )
        cases += 1
        active["record_set_sha256"] = "f" * 64
        active_path.write_bytes(canonical_json_bytes(active))
        expect_error(
            "E_EXPECT_MAP",
            lambda: read_expected_allocation(active_path, manifest, repo),
        )
        cases += 1
        scanner_audit["unexpected"] = True
        scanner_path.write_bytes(canonical_json_bytes(scanner_audit))
        expect_error(
            "E_EXPECT_MAP",
            lambda: read_expected_allocation(scanner_path, manifest, repo),
        )
        cases += 1
        scanner_audit.pop("unexpected")
        scanner_audit["open_prs"] = [
            {"number": 1, "head_oid": manifest_pin, "extra": True}
        ]
        scanner_path.write_bytes(canonical_json_bytes(scanner_audit))
        expect_error(
            "E_EXPECT_MAP",
            lambda: read_expected_allocation(scanner_path, manifest, repo),
        )
        cases += 1

    with tempfile.TemporaryDirectory() as raw_root:
        repo = Path(raw_root)
        execute(("git", "init", "-q", "-b", "main"), repo, "E_SELF_TEST")
        execute(("git", "config", "user.name", "Task 26000 Test"), repo, "E_SELF_TEST")
        execute(("git", "config", "user.email", "task26000@example.invalid"), repo, "E_SELF_TEST")
        fixture = repo / "fixture.txt"
        fixture.write_text("common\n", encoding="utf-8")
        execute(("git", "add", "fixture.txt"), repo, "E_SELF_TEST")
        execute(("git", "commit", "-q", "-m", "common"), repo, "E_SELF_TEST")
        common = execute(("git", "rev-parse", "HEAD"), repo, "E_SELF_TEST").decode("ascii").strip()
        archive_claims: dict[int, dict[ClaimIdentity, set[str]]] = {}
        scan_archive(repo, common, "refs/heads/no-backlog", archive_claims)
        fail(not archive_claims, "E_SELF_TEST", "no-backlog archive claims")
        cases += 1
        expect_error(
            "E_ARCHIVE",
            lambda: scan_archive(repo, "f" * 40, "refs/heads/invalid", {}),
        )
        cases += 1
        fixture.write_text("authority cut\n", encoding="utf-8")
        execute(("git", "add", "fixture.txt"), repo, "E_SELF_TEST")
        execute(("git", "commit", "-q", "-m", "authority cut"), repo, "E_SELF_TEST")
        manifest_pin = execute(("git", "rev-parse", "HEAD"), repo, "E_SELF_TEST").decode("ascii").strip()
        execute(("git", "update-ref", "refs/remotes/origin/dev", manifest_pin), repo, "E_SELF_TEST")
        equal = verify_origin_dev_ancestry(repo, manifest_pin)
        fail(equal["origin_dev_ancestry"] == "equal", "E_SELF_TEST", "equal authority cut")
        cases += 1

        fixture.write_text("descendant\n", encoding="utf-8")
        execute(("git", "add", "fixture.txt"), repo, "E_SELF_TEST")
        execute(("git", "commit", "-q", "-m", "descendant"), repo, "E_SELF_TEST")
        descendant = execute(("git", "rev-parse", "HEAD"), repo, "E_SELF_TEST").decode("ascii").strip()
        execute(("git", "update-ref", "refs/remotes/origin/dev", descendant), repo, "E_SELF_TEST")
        advanced = verify_origin_dev_ancestry(repo, manifest_pin)
        fail(
            advanced == {
                "manifest_pin": manifest_pin,
                "observed_origin_dev": descendant,
                "origin_dev_ancestry": "fast_forward_descendant",
            },
            "E_SELF_TEST",
            "descendant authority cut",
        )
        cases += 1

        audit_manifest = {"revisions": {"current": manifest_pin}}
        stored_equal = {
            **equal,
            "refs": [
                {"ref": "refs/remotes/origin/dev", "oid": manifest_pin}
            ],
            "open_prs": [],
            "worktrees": [],
            "claims": {},
            "external_used_ids": [],
            "allocation": expected,
        }
        stored_advanced = {
            **advanced,
            "refs": [
                {"ref": "refs/remotes/origin/dev", "oid": descendant}
            ],
            "open_prs": [],
            "worktrees": [],
            "claims": {},
            "external_used_ids": [],
            "allocation": expected,
        }
        validate_scanner_audit(stored_equal, audit_manifest, repo, "E_SELF_TEST")
        validate_scanner_audit(stored_advanced, audit_manifest, repo, "E_SELF_TEST")
        cases += 1
        for forged in (
            {**stored_advanced, "origin_dev_ancestry": "equal"},
            {**stored_equal, "origin_dev_ancestry": "fast_forward_descendant"},
            {**stored_equal, "refs": []},
            {**stored_equal, "refs": [*stored_equal["refs"], *stored_equal["refs"]]},
            {
                **stored_advanced,
                "observed_origin_dev": "f" * 40,
                "refs": [
                    {"ref": "refs/remotes/origin/dev", "oid": "f" * 40}
                ],
            },
            {**stored_advanced, "refs": stored_equal["refs"]},
        ):
            expect_error(
                "E_ORIGIN_DEV_DIVERGED",
                lambda forged=forged: validate_scanner_audit(
                    forged, audit_manifest, repo, "E_SELF_TEST"
                ),
            )
        expect_error(
            "E_ORIGIN_DEV_DIVERGED",
            lambda: validate_scanner_audit(
                stored_advanced,
                {"revisions": {"current": descendant}},
                repo,
                "E_SELF_TEST",
            ),
        )
        cases += 1

        expect_error(
            "E_ORIGIN_DEV_DIVERGED",
            lambda: verify_origin_dev_snapshot_binding(
                advanced,
                [("refs/remotes/origin/dev", manifest_pin)],
            ),
        )
        cases += 1

        execute(("git", "checkout", "-q", "-b", "replacement", common), repo, "E_SELF_TEST")
        fixture.write_text("replacement\n", encoding="utf-8")
        execute(("git", "add", "fixture.txt"), repo, "E_SELF_TEST")
        execute(("git", "commit", "-q", "-m", "replacement"), repo, "E_SELF_TEST")
        replacement = execute(("git", "rev-parse", "HEAD"), repo, "E_SELF_TEST").decode("ascii").strip()
        expect_error(
            "E_ORIGIN_DEV_DIVERGED",
            lambda: validate_scanner_audit(
                {
                    **stored_advanced,
                    "observed_origin_dev": replacement,
                    "refs": [
                        {"ref": "refs/remotes/origin/dev", "oid": replacement}
                    ],
                },
                audit_manifest,
                repo,
                "E_SELF_TEST",
            ),
        )
        cases += 1
        execute(("git", "update-ref", "refs/remotes/origin/dev", replacement), repo, "E_SELF_TEST")
        expect_error("E_ORIGIN_DEV_DIVERGED", lambda: verify_origin_dev_ancestry(repo, manifest_pin))
        cases += 1

        execute(("git", "update-ref", "-d", "refs/remotes/origin/dev"), repo, "E_SELF_TEST")
        expect_error("E_ORIGIN_DEV_DIVERGED", lambda: verify_origin_dev_ancestry(repo, manifest_pin))
        cases += 1
        execute(("git", "update-ref", "refs/remotes/origin/dev", replacement), repo, "E_SELF_TEST")
        expect_error("E_ORIGIN_DEV_DIVERGED", lambda: verify_origin_dev_ancestry(repo, "f" * 40))
        cases += 1

    fail(cases == 40, "E_SELF_TEST", f"case count {cases}")
    print("allocation scanner self-tests: 40 cases passed")


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
        expected_raw = read_expected_allocation(
            Path(args.expect_map), manifest, repo
        )
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
    validate_scanner_audit(audit, manifest, repo, "E_AUDIT")
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
<!-- TASK-26000-ALLOCATOR-SOURCE-END -->

`--self-test` invokes no fetch, GitHub CLI, or repository scan. It uses only
temporary fixture bytes and directories to prove that distinct identities sharing
one task ID fail closed, an exact content-bound self copy is accepted, a
valid task filename reserves its numeric ID even when legacy frontmatter lacks a
usable numeric ID, filename/frontmatter numeric mismatches and ambiguous numeric
frontmatter fail closed without consulting headings, invalid UTF-8 maps to exact
`E_TASK_IDENTITY`, commits without a `backlog` tree contribute zero claims while
invalid archive OIDs fail closed, moved or changed PR heads fail, worktree files
become claims, fresh and precreate allocation leapfrog deterministically with the
final label highest, complete manifest-bound self claims preserve their authenticated
allocation despite an unrelated higher external maximum, missing planned live self
claims fail closed,
an audit output cannot be overwritten, equal and fast-forward-descendant authority
tips are distinguished, and missing/divergent authority fails closed. Each mutation must raise its documented
scanner error; canonical scanner-audit and active-state inputs are accepted while
noncanonical, open-shaped, or digest-invalid handoffs fail closed. A missed mutation
makes the self-test exit nonzero. Successful output is exactly
`allocation scanner self-tests: 40 cases passed`.

The first scan's audit is retained under `task26000_tmp_root/raw/allocation.json`.
Immediately before task-file creation, rerun the scanner with `--expect-map` pointing
to that audit. While `cleanup_records` is empty, it recomputes the allocation from the
live external maximum; `E_ALLOCATION_MOVED` forces a fresh allocation and regeneration
before any task file is created. After rendering and again immediately before commit,
the same option authenticates the complete manifest-bound records, requires every
planned self identity to appear in the live claim census, rejects a different identity
on any active ID with `E_ID_COLLISION`, and retains the expected allocation even when
an unrelated external maximum has advanced. Every scan verifies that the post-fetch
`origin/dev` commit equals `manifest.revisions.current` or is its fast-forward
descendant, captures the complete audited ref snapshot once, mechanically binds
the ancestry observation to that snapshot's `origin/dev` OID, and records both
SHAs plus the exact ancestry result. Missing, moved-between-observation-and-snapshot, or
non-ancestor state fails `E_ORIGIN_DEV_DIVERGED`. Before
creation, `cleanup_records` is empty and the expected IDs must be wholly unclaimed.
After rendering, the scanner excludes only manifest-proven self claims whose exact
path, frontmatter ID, batch marker, and content SHA-256 match; a ref, PR, or worktree
copy with different bytes is a distinct identity and therefore `E_ID_COLLISION`.
Fresh scans without `--expect-map` continue to treat the old generated IDs as external
and allocate above every observed claim for the bounded reallocation workflow.


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

The renderer retains an unreachable legacy `refresh` implementation only as
historical recovery code. Its authenticated CLI rejects every `refresh` request with
exact `E_REFRESH_SUPERSEDED` immediately after argument parsing and before reading a
repository path, allocation, manifest, task, path list, or journal. The negative
self-test proves this gate changes no bytes. Task 7 MUST NOT invoke renderer refresh:
ordinary `origin/dev` equality or verified fast-forward advancement leaves the pinned
batches and records unchanged, while an ID or batch collision uses only the bounded
`reallocate` workflow.

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

<!-- TASK-26000-RENDERER-SOURCE-BEGIN sha256=4a08b6a5a9a8b12926ab9417bc330a4e94eb60c3b4afe88226ef232e2653a17a -->
```python
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
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
SCANNER_AUDIT_KEYS = {
    "manifest_pin", "observed_origin_dev", "origin_dev_ancestry", "refs",
    "open_prs", "worktrees", "claims", "external_used_ids", "allocation",
}
ACTIVE_STATE_KEYS = {
    "schema_version", "mode", "allocation", "paths0_output", "paths0_sha256",
    "record_set_sha256",
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
FINAL_AC = "- [ ] After all lower-ID cleanup dependencies pass, the explicit Git-tracked repository-wide command exits zero under the recorded Python 3.12.11 interpreter: `python -m ruff format --check --force-exclude .`; any post-cut unassigned failure blocks this gate, is never absorbed into the pinned counts or current batches, and requires a separate correction record. <!-- TASK-26000-CONTRACT: repository-zero-gate --><!-- TASK-26000-CONTRACT: post-cut-unassigned-correction -->"
RECORD_KEYS = {
    "label", "path", "task_id", "final", "dependencies", "paths_sha256",
    "task_sha256", "created_at", "updated_at",
}
SHA256 = re.compile(r"[0-9a-f]{64}")
MINUTE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}")
TASK_BUCKETS = (
    "backlog/tasks/",
    "backlog/completed/",
    "backlog/archive/tasks/",
    "backlog/drafts/",
)


class RenderError(RuntimeError):
    pass


def need(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise RenderError(f"{code}: {detail}")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def valid_oid(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def validate_scanner_audit(
    value: dict[str, Any], data: dict[str, Any], repo: Path
) -> None:
    need(set(value) == SCANNER_AUDIT_KEYS, "E_RENDER_ALLOCATION", "scanner audit schema")
    need(
        all(valid_oid(value[key]) for key in ("manifest_pin", "observed_origin_dev")),
        "E_RENDER_ALLOCATION",
        "scanner audit revisions",
    )
    need(
        value["origin_dev_ancestry"] in {"equal", "fast_forward_descendant"},
        "E_RENDER_ALLOCATION",
        "scanner audit ancestry",
    )
    refs_value = value["refs"]
    need(isinstance(refs_value, list), "E_RENDER_ALLOCATION", "scanner audit refs")
    for row in refs_value:
        need(
            isinstance(row, dict)
            and set(row) == {"ref", "oid"}
            and isinstance(row["ref"], str)
            and re.fullmatch(r"refs/(?:heads|remotes/origin)/.+", row["ref"])
            is not None
            and valid_oid(row["oid"]),
            "E_RENDER_ALLOCATION",
            "scanner audit ref row",
        )
    need(
        refs_value == sorted(refs_value, key=lambda row: row["ref"]),
        "E_RENDER_ALLOCATION",
        "scanner audit ref order",
    )
    prs = value["open_prs"]
    need(isinstance(prs, list), "E_RENDER_ALLOCATION", "scanner audit open PRs")
    need(
        all(
            isinstance(row, dict)
            and set(row) == {"number", "head_oid"}
            and type(row["number"]) is int
            and row["number"] > 0
            and valid_oid(row["head_oid"])
            for row in prs
        )
        and prs == sorted(prs, key=lambda row: row["number"])
        and len({row["number"] for row in prs}) == len(prs),
        "E_RENDER_ALLOCATION",
        "scanner audit open PR rows",
    )
    worktrees = value["worktrees"]
    need(isinstance(worktrees, list), "E_RENDER_ALLOCATION", "scanner audit worktrees")
    decoded_paths: list[bytes] = []
    for row in worktrees:
        need(
            isinstance(row, dict)
            and set(row) == {"path_b64", "head", "dirty"}
            and isinstance(row["path_b64"], str)
            and valid_oid(row["head"])
            and type(row["dirty"]) is bool,
            "E_RENDER_ALLOCATION",
            "scanner audit worktree row",
        )
        try:
            decoded = base64.b64decode(row["path_b64"], validate=True)
        except (ValueError, binascii.Error):
            decoded = b""
        need(
            bool(decoded)
            and base64.b64encode(decoded).decode("ascii") == row["path_b64"],
            "E_RENDER_ALLOCATION",
            "scanner audit worktree path",
        )
        decoded_paths.append(decoded)
    need(
        len(set(decoded_paths)) == len(decoded_paths),
        "E_RENDER_ALLOCATION",
        "scanner audit worktree duplicates",
    )
    claims = value["claims"]
    need(isinstance(claims, dict), "E_RENDER_ALLOCATION", "scanner audit claims")
    for task_id, rows in claims.items():
        need(
            isinstance(task_id, str)
            and re.fullmatch(r"[0-9]+", task_id) is not None
            and str(int(task_id)) == task_id
            and isinstance(rows, list)
            and bool(rows),
            "E_RENDER_ALLOCATION",
            "scanner audit claim group",
        )
        for row in rows:
            need(
                isinstance(row, dict)
                and set(row)
                == {
                    "path",
                    "batch_label",
                    "content_sha256",
                    "sources",
                    "accepted_self",
                }
                and isinstance(row["path"], str)
                and row["path"].startswith(TASK_BUCKETS)
                and "\n" not in row["path"]
                and "\x00" not in row["path"]
                and (
                    row["batch_label"] is None
                    or (
                        isinstance(row["batch_label"], str)
                        and re.fullmatch(
                            r"[a-z0-9]+(?:-[a-z0-9]+)*", row["batch_label"]
                        )
                        is not None
                    )
                )
                and isinstance(row["content_sha256"], str)
                and SHA256.fullmatch(row["content_sha256"]) is not None
                and isinstance(row["sources"], list)
                and bool(row["sources"])
                and row["sources"] == sorted(set(row["sources"]))
                and all(isinstance(source, str) and source for source in row["sources"])
                and type(row["accepted_self"]) is bool,
                "E_RENDER_ALLOCATION",
                "scanner audit claim row",
            )
    external_ids = value["external_used_ids"]
    need(
        isinstance(external_ids, list)
        and external_ids == sorted(set(external_ids))
        and all(type(task_id) is int and task_id >= 0 for task_id in external_ids),
        "E_RENDER_ALLOCATION",
        "scanner audit external IDs",
    )
    allocation = value["allocation"]
    need(
        isinstance(allocation, dict)
        and all(
            isinstance(label, str)
            and re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", label) is not None
            and type(task_id) is int
            and task_id > 26000
            for label, task_id in allocation.items()
        )
        and len(set(allocation.values())) == len(allocation),
        "E_RENDER_ALLOCATION",
        "scanner audit allocation",
    )
    revisions = data.get("revisions")
    manifest_pin = revisions.get("current") if isinstance(revisions, dict) else None
    need(
        valid_oid(manifest_pin) and value["manifest_pin"] == manifest_pin,
        "E_ORIGIN_DEV_DIVERGED",
        "scanner audit manifest pin differs from manifest.revisions.current",
    )
    origin_dev = [
        row["oid"]
        for row in refs_value
        if row["ref"] == "refs/remotes/origin/dev"
    ]
    need(
        len(origin_dev) == 1 and origin_dev[0] == value["observed_origin_dev"],
        "E_ORIGIN_DEV_DIVERGED",
        "scanner audit origin/dev snapshot is missing, duplicated, or mismatched",
    )
    need(
        len({row["ref"] for row in refs_value}) == len(refs_value),
        "E_RENDER_ALLOCATION",
        "scanner audit duplicate ref",
    )
    pin = value["manifest_pin"]
    observed = value["observed_origin_dev"]
    for name, oid in (("manifest pin", pin), ("observed origin/dev", observed)):
        commit = subprocess.run(
            ("git", "cat-file", "-e", f"{oid}^{{commit}}"),
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        need(
            commit.returncode == 0,
            "E_ORIGIN_DEV_DIVERGED",
            f"{name} commit is unavailable",
        )
    if value["origin_dev_ancestry"] == "equal":
        need(
            observed == pin,
            "E_ORIGIN_DEV_DIVERGED",
            "equal ancestry has distinct immutable OIDs",
        )
    else:
        need(
            observed != pin,
            "E_ORIGIN_DEV_DIVERGED",
            "fast-forward ancestry has equal immutable OIDs",
        )
        ancestor = subprocess.run(
            ("git", "merge-base", "--is-ancestor", pin, observed),
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        need(
            ancestor.returncode == 0,
            "E_ORIGIN_DEV_DIVERGED",
            "stored origin/dev OID does not descend from the manifest pin",
        )


def read_allocation_authority(
    path: Path, mode: str, data: dict[str, Any], repo: Path
) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    need(isinstance(value, dict) and raw == canonical_bytes(value), "E_RENDER_ALLOCATION", "canonical JSON")
    if mode in {"create", "reallocate"}:
        validate_scanner_audit(value, data, repo)
        return value
    need(mode == "refresh" and set(value) == ACTIVE_STATE_KEYS, "E_RENDER_ALLOCATION", "active-state schema")
    need(value["schema_version"] == 1 and value["mode"] in {"create", "reallocate"}, "E_RENDER_ALLOCATION", "active-state identity")
    paths0 = value["paths0_output"]
    need(isinstance(paths0, str) and Path(paths0).is_absolute(), "E_RENDER_ALLOCATION", "active-state paths0_output")
    paths0_path = Path(paths0)
    need(not paths0_path.is_symlink() and paths0_path.is_file(), "E_RENDER_ALLOCATION", "active-state paths0 file")
    paths_raw = paths0_path.read_bytes()
    need(isinstance(value["paths0_sha256"], str) and SHA256.fullmatch(value["paths0_sha256"]) is not None and digest(paths_raw) == value["paths0_sha256"], "E_RENDER_ALLOCATION", "active-state paths0 digest")
    records = data.get("cleanup_records")
    need(isinstance(records, list), "E_RENDER_ALLOCATION", "active-state cleanup records")
    need(all(isinstance(row, dict) and isinstance(row.get("label"), str) and isinstance(row.get("path"), str) and row["path"].startswith("backlog/tasks/") and "\n" not in row["path"] and "\x00" not in row["path"] and type(row.get("task_id")) is int and row["task_id"] > 26000 for row in records), "E_RENDER_ALLOCATION", "active-state record identities")
    identities = sorted(
        ({"label": row["label"], "path": row["path"], "task_id": row["task_id"]} for row in records),
        key=lambda row: row["label"],
    )
    expected_allocation = {row["label"]: row["task_id"] for row in identities}
    need(len(expected_allocation) == len(identities) and value["allocation"] == expected_allocation, "E_RENDER_ALLOCATION", "active-state allocation")
    need(isinstance(value["record_set_sha256"], str) and SHA256.fullmatch(value["record_set_sha256"]) is not None and digest(canonical_bytes(identities)) == value["record_set_sha256"], "E_RENDER_ALLOCATION", "active-state record-set digest")
    return value


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


def reject_superseded_refresh(mode: str | None) -> None:
    if mode == "refresh":
        raise RenderError("E_REFRESH_SUPERSEDED")


def run_self_tests() -> None:
    global atomic_replace, run_transaction, write_journal_phase
    original_atomic = atomic_replace
    original_run_transaction = run_transaction
    original_phase = write_journal_phase
    cases = 0
    try:
        read_authority = globals().get("read_allocation_authority")
        need(callable(read_authority), "E_SELF_TEST", "mode-specific allocation authority reader")
        expected_final_ac = "- [ ] After all lower-ID cleanup dependencies pass, the explicit Git-tracked repository-wide command exits zero under the recorded Python 3.12.11 interpreter: `python -m ruff format --check --force-exclude .`; any post-cut unassigned failure blocks this gate, is never absorbed into the pinned counts or current batches, and requires a separate correction record. <!-- TASK-26000-CONTRACT: repository-zero-gate --><!-- TASK-26000-CONTRACT: post-cut-unassigned-correction -->"
        final_task = render_task(
            self_test_batch("ruff-final-gate", "z.py"),
            30001,
            [26000, 30000],
            True,
            "2026-08-30 20:00",
            "2026-08-30 20:00",
        ).decode("utf-8")
        need(FINAL_AC == expected_final_ac, "E_SELF_TEST", "exact final acceptance criterion")
        need(final_task.count(expected_final_ac) == 1, "E_SELF_TEST", "rendered final acceptance criterion")
        need("any new unassigned failure" not in final_task, "E_SELF_TEST", "stale final acceptance criterion")
        cases += 1

        with tempfile.TemporaryDirectory(prefix="task26000-render-selftest-") as temporary_root:
            sandbox = Path(temporary_root)

            refresh_sentinel = sandbox / "refresh-superseded-sentinel"
            refresh_sentinel.write_bytes(b"unchanged\n")
            refresh_before = refresh_sentinel.read_bytes()
            try:
                reject_superseded_refresh("refresh")
            except RenderError as exc:
                need(str(exc) == "E_REFRESH_SUPERSEDED", "E_SELF_TEST", "refresh supersession error")
            else:
                raise RenderError("E_SELF_TEST: refresh supersession accepted")
            need(refresh_sentinel.read_bytes() == refresh_before, "E_SELF_TEST", "refresh supersession mutated bytes")
            cases += 1

            authority_data = self_test_manifest(
                [self_test_batch("ruff-final-gate", "z.py")],
                "ruff-final-gate",
            )
            authority_repo = sandbox / "authority-repo"
            authority_repo.mkdir()

            def authority_git(*argv: str) -> str:
                completed = subprocess.run(
                    ("git", *argv),
                    cwd=authority_repo,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                need(completed.returncode == 0, "E_SELF_TEST", completed.stderr)
                return completed.stdout.strip()

            authority_git("init", "-q", "-b", "main")
            authority_git("config", "user.name", "Task 26000 Test")
            authority_git("config", "user.email", "task26000@example.invalid")
            authority_fixture = authority_repo / "fixture.txt"
            authority_fixture.write_text("common\n", encoding="utf-8")
            authority_git("add", "fixture.txt")
            authority_git("commit", "-q", "-m", "common")
            common = authority_git("rev-parse", "HEAD")
            authority_fixture.write_text("authority\n", encoding="utf-8")
            authority_git("add", "fixture.txt")
            authority_git("commit", "-q", "-m", "authority")
            pin = authority_git("rev-parse", "HEAD")
            authority_data["revisions"] = {"current": pin}
            scanner = {
                "manifest_pin": pin,
                "observed_origin_dev": pin,
                "origin_dev_ancestry": "equal",
                "refs": [{"ref": "refs/remotes/origin/dev", "oid": pin}],
                "open_prs": [],
                "worktrees": [],
                "claims": {},
                "external_used_ids": [],
                "allocation": {"ruff-final-gate": 30001},
            }
            scanner_path = sandbox / "allocation.json"
            scanner_path.write_bytes(canonical_bytes(scanner))
            need(
                read_authority(
                    scanner_path, "create", authority_data, authority_repo
                )
                == scanner,
                "E_SELF_TEST",
                "canonical equal scanner audit",
            )
            authority_fixture.write_text("descendant\n", encoding="utf-8")
            authority_git("add", "fixture.txt")
            authority_git("commit", "-q", "-m", "descendant")
            descendant = authority_git("rev-parse", "HEAD")
            fast_forward = {
                **scanner,
                "observed_origin_dev": descendant,
                "origin_dev_ancestry": "fast_forward_descendant",
                "refs": [
                    {"ref": "refs/remotes/origin/dev", "oid": descendant}
                ],
            }
            scanner_path.write_bytes(canonical_bytes(fast_forward))
            need(
                read_authority(
                    scanner_path, "reallocate", authority_data, authority_repo
                )
                == fast_forward,
                "E_SELF_TEST",
                "canonical fast-forward scanner audit",
            )
            authority_git("checkout", "-q", "-b", "replacement", common)
            authority_fixture.write_text("replacement\n", encoding="utf-8")
            authority_git("add", "fixture.txt")
            authority_git("commit", "-q", "-m", "replacement")
            replacement = authority_git("rev-parse", "HEAD")
            for changed in (
                {**scanner, "extra": None},
                {key: value for key, value in scanner.items() if key != "claims"},
                {**scanner, "manifest_pin": "a" * 64},
                {**scanner, "origin_dev_ancestry": "divergent"},
                {**scanner, "claims": []},
                {**fast_forward, "manifest_pin": descendant},
                {**fast_forward, "origin_dev_ancestry": "equal"},
                {**scanner, "origin_dev_ancestry": "fast_forward_descendant"},
                {**scanner, "refs": []},
                {**scanner, "refs": [*scanner["refs"], *scanner["refs"]]},
                {
                    **fast_forward,
                    "observed_origin_dev": "f" * 40,
                    "refs": [
                        {"ref": "refs/remotes/origin/dev", "oid": "f" * 40}
                    ],
                },
                {**fast_forward, "refs": scanner["refs"]},
                {
                    **fast_forward,
                    "observed_origin_dev": replacement,
                    "refs": [
                        {"ref": "refs/remotes/origin/dev", "oid": replacement}
                    ],
                },
                {
                    **scanner,
                    "open_prs": [
                        {"number": 1, "head_oid": pin, "extra": True}
                    ],
                },
            ):
                scanner_path.write_bytes(canonical_bytes(changed))
                try:
                    read_authority(
                        scanner_path, "reallocate", authority_data, authority_repo
                    )
                except RenderError as exc:
                    need(
                        str(exc).startswith(
                            ("E_RENDER_ALLOCATION:", "E_ORIGIN_DEV_DIVERGED:")
                        ),
                        "E_SELF_TEST",
                        "scanner authority error",
                    )
                else:
                    raise RenderError("E_SELF_TEST: scanner mutation accepted")
            scanner_path.write_text(json.dumps(scanner), encoding="utf-8")
            try:
                read_authority(
                    scanner_path, "create", authority_data, authority_repo
                )
            except RenderError as exc:
                need(str(exc).startswith("E_RENDER_ALLOCATION:"), "E_SELF_TEST", "scanner canonical error")
            else:
                raise RenderError("E_SELF_TEST: noncanonical scanner audit accepted")
            cases += 1

            paths_authority = sandbox / "active-paths0"
            paths_raw = b"backlog/tasks/task-30001 - final.md\0"
            paths_authority.write_bytes(paths_raw)
            active_identity = {
                "label": "ruff-final-gate",
                "path": "backlog/tasks/task-30001 - final.md",
                "task_id": 30001,
            }
            authority_data["cleanup_records"] = [active_identity]
            active = {
                "schema_version": 1,
                "mode": "create",
                "allocation": {"ruff-final-gate": 30001},
                "paths0_output": os.fspath(paths_authority),
                "paths0_sha256": digest(paths_raw),
                "record_set_sha256": digest(canonical_bytes([active_identity])),
            }
            active_path = sandbox / "active.json"
            active_path.write_bytes(canonical_bytes(active))
            need(read_authority(active_path, "refresh", authority_data, authority_repo) == active, "E_SELF_TEST", "canonical active state")
            active["allocation"] = {"ruff-final-gate": 30002}
            active_path.write_bytes(canonical_bytes(active))
            try:
                read_authority(active_path, "refresh", authority_data, authority_repo)
            except RenderError as exc:
                need(str(exc).startswith("E_RENDER_ALLOCATION:"), "E_SELF_TEST", "active allocation error")
            else:
                raise RenderError("E_SELF_TEST: active-state allocation mutation accepted")
            active["allocation"] = {"ruff-final-gate": 30001}
            active["paths0_sha256"] = "0" * 64
            active_path.write_bytes(canonical_bytes(active))
            try:
                read_authority(active_path, "refresh", authority_data, authority_repo)
            except RenderError as exc:
                need(str(exc).startswith("E_RENDER_ALLOCATION:"), "E_SELF_TEST", "active digest error")
            else:
                raise RenderError("E_SELF_TEST: active-state digest mutation accepted")
            cases += 1

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
    need(cases == 9, "E_SELF_TEST", f"case count {cases}")
    print("cleanup renderer self-tests: 9 cases passed")


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
    reject_superseded_refresh(args.mode)
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
    audit = read_allocation_authority(Path(args.allocation), args.mode, data, repo)
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
<!-- TASK-26000-RENDERER-SOURCE-END -->

The former Task 7 repin-refresh procedure is closed and superseded. It intentionally
has no executable mutation command in this plan; any compatibility invocation fails
exact `E_REFRESH_SUPERSEDED` before reading or changing bytes. Do not derive a refresh
label. A collision follows Task 5's one-attempt `reallocate` branch;
ordinary equal or verified-fast-forward remote state proceeds without rewriting any
cleanup record.
