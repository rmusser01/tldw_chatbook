---
id: TASK-32912
title: CI has executed zero tests for days, and no test job runs on PRs at all
status: To Do
assignee: []
created_date: '2026-09-22 09:00'
labels:
  - tier2-review
  - review-testing
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**This supersedes the stated cause of TASK-32908.** That task said `Tests/UI` was excluded from the PR gate
by `test.yml:121`'s `--ignore=Tests/UI`. That line is a red herring — it is merely the core/UI job split
inside a workflow that also has a 12-shard `ui-tests` job. The real problem is four lines higher, and it is
much larger.

## 1. `test.yml` never runs on a pull request

```yaml
on:
  push:
    branches: ["main"]
  workflow_dispatch:
```

There is **no `pull_request` trigger**. So the entire `Tests` workflow — not just `Tests/UI` — gates nothing
on any PR. `gh pr checks` on an open PR confirms it: `PR Fast Lane`, `Derived artifacts`, and the GGUF /
platform evidence jobs, and **no `Tests` check at all**. `dev`'s only required context is
`Derived artifacts reproduce from their sources`.

## 2. The nightly, which is the only thing that runs tests, has executed ZERO of them

`nightly-deep.yml` runs `pytest ./Tests/` with no `--continue-on-collection-errors`, so a single bad import
aborts the whole run. Verified against run `35706024071` (2026-09-22), on all three platforms:

```
ubuntu-latest   collected 101851 items / 1 error / 8 skipped
                !!!! Interrupted: 1 error during collection !!!!
macos-latest    collected 101854 items / 1 error / 8 skipped
                !!!! Interrupted: 1 error during collection !!!!
windows-latest  collected 101580 items / 3 errors / 11 skipped
                !!!! Interrupted: 3 errors during collection !!!!
```

`gh run list --workflow=nightly-deep.yml` shows **failure on 09-19, 09-20, 09-21 and 09-22** — four
consecutive nights, and the pattern is identical each time.

**Taken together: this repository has had no executed automated test coverage, anywhere, for at least four
days.** Not a degraded signal — zero tests run.

## 3. The collection error is a live product defect

`tldw_chatbook/LLM_Calls/LLM_API_Calls.py:3435`:
```python
    if tools:
        payload["tools"] = _google_tools_payload(tools)
```
`grep -rn 'def _google_tools_payload' tldw_chatbook/` returns **zero definitions**. It was deleted; the call
site was not. Any Google request carrying tools raises `NameError`.

So one live defect on the Google native-tools path is also the thing blanking the entire nightly.

## Fixes, in order of how much they buy

1. **Define or remove `_google_tools_payload`.** Unblocks the nightly immediately and fixes a real crash.
2. **Add `--continue-on-collection-errors` to `nightly-deep.yml`.** A single bad import must never cost
   101,851 tests. This is the structural fix; item 1 is the current instance.
3. **Give `test.yml` a `pull_request` trigger**, or move a meaningful subset into a job that has one. See
   TASK-32908 for the measured cost of the full suite (~40% of `Tests/UI` is red) and the fast-lane subset
   shipped as an interim.

Note the ordering argument: (2) matters more than (1), because without it the next deleted symbol does the
same thing again and nobody finds out for four days.

Source: tier-2 code review 2026-09-21, found while implementing TASK-32908.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `_google_tools_payload` is defined or its call site removed, with a test covering the Google tools path
- [ ] #2 `nightly-deep.yml` survives a collection error and still runs the tests it could collect
- [ ] #3 Some test job runs on `pull_request` and is a required check
- [ ] #4 A check or alert fires when a scheduled run executes zero tests — "failed" and "ran nothing" must not look the same
<!-- AC:END -->
