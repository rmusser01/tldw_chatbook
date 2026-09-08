# TASK-31942: fresh macOS concurrency evidence

## Scope and authority

The user confirmed the local Mac had not restarted after the recorded semaphore
exhaustion, then approved setting up and running a targeted job on fresh macOS CI.
Task14 qualifies only the eleven previously host-blocked repository concurrency
cases. No local semaphore control or product case is rerun, and no local host or
dependency repair is authorized by this job.

ADR required: no new ADR. Existing
[ADR125](../../../backlog/decisions/125-lock-safe-private-sqlite-validation.md)
applies unchanged; this is evidence for the existing concurrency contract.

Implementation commit: `13fe672c648f8fe329bcca6c070e853a376a27e1`.
Only the new branch-restricted workflow and its focused CI-contract test are
included in that commit. The dedicated remote branch is
`codex/task-31942-macos-concurrency-evidence`; no main/dev update, PR, rebase or
merge is part of this operation.

## Job contract and local checks

The workflow uses `macos-15`, Python `3.12.11`, a 30-minute job deadline,
read-only contents permission, and exact-SHA checkout without persisted
credentials. Fixed SHA/Python/SQLite/macOS/architecture metadata and an isolated
stdlib spawn-Lock allocation/acquisition/disposal control precede installation
of the existing dev extra on the runner. A failed control prevents product tests.

One serial pytest invocation selects exactly these groups from
`Tests/TTS/test_profile_repository.py`:

| Test | Parameter IDs | Count |
| --- | --- | --- |
| `test_spawned_repositories_open_one_fresh_store_concurrently` | 0–2 | 3 |
| `test_spawned_repositories_resolve_sqlite_constraint_race_safely` | 0–4 | 5 |
| `test_spawned_set_delete_race_is_serialized_without_partial_mutation` | 0–2 | 3 |

Existing assertions/timeouts are unchanged. No xdist, retry, skip/deselect,
keyword expansion, or source-path override is added. Pipeline status preserves
pytest failure. The always-run JUnit check requires exact identities and zero
errors/failures/skips; only fixed metadata, control, pytest log and JUnit files
under runner temporary storage are uploaded, including on failure.

Implementer evidence: initial missing-workflow RED (12 failures), then a separate
aggregate-error regression RED (1 failure, 8 passes), and final GREEN (15 passes,
0.61s). Root independently ran the focused file before commit: **15 passed in
1.26s**; after commit: **15 passed in 1.12s**, exit0 with no warnings. Command:

```text
../../.venv/bin/python -m pytest Tests/CI/test_task31942_macos_concurrency_evidence.py --confcutdir=Tests/CI -q
```

Root new-file Ruff, formatter and whitespace checks passed. The implementer
also parsed YAML, checked Bash syntax for all five run blocks, and compiled both
inline Python bodies without executing the local lock control. `actionlint` is
unavailable and was not installed; no actionlint result is claimed.

## Independent review and remote result

Independent scoped review approved both spec compliance and task quality with
no Critical, Important, or Minor findings. It checked the preserved semaphore
control against the prior diagnosis and the behavioral contracts against the
immutable two-file diff. The complete review is retained in the Task14 SDD
workspace. Actual remote execution remains the review's unverified item.

## Remote attempt 1: setup failure, no product execution

The root pushed the exact reviewed commit without force to the previously absent
dedicated branch. GitHub accepted the workflow and ran one push-triggered job:

- [Run 34292597991, attempt1](https://github.com/rmusser01/tldw_chatbook/actions/runs/34292597991)
- [Job 102282182152](https://github.com/rmusser01/tldw_chatbook/actions/runs/34292597991/job/102282182152)
- SHA `13fe672c648f8fe329bcca6c070e853a376a27e1`, branch
  `codex/task-31942-macos-concurrency-evidence`.
- GitHub Actions runner `GitHub Actions 1000450687`, label `macos-15`;
  started `2026-09-08T23:53:13Z`, completed `2026-09-08T23:53:35Z`.
- Conclusion: **failure during Python setup**. Checkout succeeded.

The primary error was:

```text
The version '3.12.11' with architecture 'arm64' was not found for macOS 15.7.9.
```

The pinned Python was absent from the runner cache and the official download
manifest. Read-only inspection of GitHub's
[Python build manifest](https://raw.githubusercontent.com/actions/python-versions/main/versions-manifest.json)
confirmed that 3.12.11 through 3.12.14 have no Darwin builds, while 3.12.10 has
both Darwin ARM64 and x64 builds. The controller selected the unavailable pin
without checking the platform-specific manifest; this is a CI setup mistake,
not evidence of a SQLite product defect or fresh-host semaphore exhaustion.

Fixed runtime metadata, the lock control, dependency installation, and all
eleven product cases were skipped because setup failed. Thus Python/SQLite
runtime metadata and concurrency results are **unavailable**, not passing.
The always-run validator correctly failed on missing JUnit. Artifact upload
was attempted and failed because none of its four allowlisted files existed.
The run artifact API confirmed `total_count=0`; no JUnit or artifact could be
downloaded. Root inspected the exact job and failed-step logs rather than
substituting another run's evidence. GitHub also emitted Node20 migration and
`punycode` deprecation warnings; neither was the reported setup error.

No retry, pin change, alternative installer, or product repair was performed.
The proposed minimal correction is a **CI-only** pin to the available Python
3.12.10 build, retaining native ARM64, the 3.12 runtime family, exact test set,
and all failure/evidence gates. It would not downgrade the local environment or
change product requirements, but would qualify 3.12.10 rather than reproduce
the local 3.12.11 build. Following the CI-fix skill's approval gate, that change
and a new exact-commit run await user approval. AC15 remains unchecked.

## Qualification limits

No result from this job qualifies the unchanged local Mac, historical skipped
platform/optional cases, or the nonzero aggregate static gate. Earlier failures,
skips, deselections, and diagnostic evidence remain preserved. TASK-31942 remains
In Progress and Canvas V2 remains disabled until its remaining required gates
are resolved. This report does not authorize another suite, benchmark, host
cleanup, dependency repair, main/dev change, PR, or merge.
