# TASK-32160: fresh macOS concurrency evidence

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

## Approved setup correction: CI-only Python3.12.10

The user subsequently approved the proposed pin change and rerun. Task14's
current plan/brief now specifies3.12.10; the unavailable3.12.11 first attempt
above remains historical evidence. The bounded fix changed only the workflow
Python value and its existing contract expectation. No new installer or
action-version upgrade was included. Independent fix-only review and committed
focused verification preceded the exact-SHA fast-forward push to the existing
evidence branch.

Fix commit: `90e60aea42910096a5d59f5c03b7f3eee749f00d`, exactly two substitutions.
The amended existing expectation failed against the old pin (1 failure0.16s),
then passed after the workflow change (1 pass0.12s). Implementer covering file:
15passed0.78s. Root independently verified the covering file before and after
commit: **15 passed in0.61s**, exit0 with no warnings for each run. Ruff,
formatter and whitespace checks pass. No local lock/product probe was run.

Independent fix-only review found the unavailable-build issue **addressed**,
with no new Critical/Important breakage or other findings in the fix. Its two
execution limitations are resolved by the exact remote evidence below; local
`actionlint` remains unavailable and no result is claimed for it.

## Remote attempt 2: all eleven exact cases pass

- [Run34294412119, attempt1](https://github.com/rmusser01/tldw_chatbook/actions/runs/34294412119),
  triggered by the new push, not a blind retry of the failed run.
- [Job102287782445](https://github.com/rmusser01/tldw_chatbook/actions/runs/34294412119/job/102287782445),
  conclusion **success**, all steps successful; duration54s.
- SHA `90e60aea42910096a5d59f5c03b7f3eee749f00d`; branch
  `codex/task-31942-macos-concurrency-evidence`.
- Runner `GitHub Actions 1000450693`, image `macos-15-arm64`, image version
  `20260829.0321.1`; job ran `2026-09-09T00:18:10Z`–`00:19:04Z`.
- Runtime metadata: **Python3.12.10, SQLite3.49.1, macOS15.7.9, arm64**.

The isolated stdlib spawn-Lock control reported:

```json
{"acquired": true, "allocated": true, "owned_cleanup": "finalizer_completed", "probe": "stdlib_spawn_lock"}
```

The single serial product invocation completed with **11 passed in11.61s**, no
pytest warnings, and no skip/failure/error. JUnit records11cases in11.605s with
zero errors, failures, and skips. Every parameter ID in the three-group table
above appears exactly once. The in-job validator reported exact11passing cases.

Root downloaded only the exact run's named artifact to
`/private/tmp/task-31942-macos-evidence.YJFavY`. Independent local XML analysis
checked the complete testcase identity multiset, one suite, exact11count, no
outcome children, and zero aggregate errors/failures/skips. It also checked the
four-file artifact allowlist, exact commit/Python/architecture metadata and all
control fields. The analysis exited0; the full pytest log and relevant exact-job
setup/control/test/validation/upload log lines corroborate those results. No
Chatbook import, semaphore allocation or product test occurred locally.

[Artifact10082585312](https://github.com/rmusser01/tldw_chatbook/actions/runs/34294412119/artifacts/10082585312)
is1390bytes; the API and upload log agree on zip SHA256
`49d5a41e78faa8b1a8590bce9ba0178d26b6f4df1ac20182198b1d5300fb0ce0`.
GitHub reports expiry `2026-12-08T00:18:02Z`. Exact downloaded file bytes and
their SHA256 hashes are also preserved as base64 in the retained SDD archive
`task-14-run-34294412119-artifact.json`, so evidence does not depend solely on
the remote retention window or the temporary download directory.

| Downloaded file | Bytes | SHA256 |
| --- | --- | --- |
| `task-31942-junit.xml` | 1900 | `1c22d50452d6dd68eb1792e1f5fd0a7eb1c3a402029f4ee72396fbfad419c589` |
| `task-31942-lock-control.json` | 108 | `1e3b43da18c08b573e21bd6c621a1c96b9b9e1e711ac77b74db9cfd1de426766` |
| `task-31942-metadata.txt` | 117 | `ce063abf9d74f28fdf88c02b2fabec54aadb4e3f5f500977c9a7ec4602a11eb7` |
| `task-31942-pytest.log` | 446 | `7e97c8966f3996f8b9453a9ed9f9a70dc2c2ce5ac9a1e40702021030eb352662` |

The CI service output is not warning-free: the existing action versions emit
Node20-to24 migration, `punycode`, and `url.parse()` deprecation warnings.
Those are retained as tooling limitations, not hidden or fixed by this pin-only
change. The pytest result itself emitted no warnings.

This closes Task14's fresh-runner concurrency qualification and AC15. It does
not establish results for local Python3.12.11, the unchanged exhausted Mac,
other platforms, optional cases, or the aggregate static gate. No additional
run was triggered after collecting this successful attempt.

## Qualification limits

No result from this job qualifies the unchanged local Mac, historical skipped
platform/optional cases, or the nonzero aggregate static gate. Earlier failures,
skips, deselections, and diagnostic evidence remain preserved. TASK-32160 remains
In Progress and Canvas V2 remains disabled until its remaining required gates
are resolved. This report does not authorize another suite, benchmark, host
cleanup, dependency repair, main/dev change, PR, or merge.
