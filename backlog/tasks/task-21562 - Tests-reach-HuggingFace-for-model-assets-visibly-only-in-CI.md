---
id: TASK-21562
title: 'Tests reach HuggingFace for model assets, visibly only in CI'
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-08-24 06:06'
labels:
  - testing
  - test-integrity
  - ci
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The test sandbox points the HuggingFace cache at a directory inside itself, which never
exists, so any code path that reaches the hub attempts a real download. Nothing forbids that
download, so the attempt goes out, the network guard refuses it, and the client retries
several times before the guard's record fails the test during teardown.

The cost is almost entirely hidden from developers. A single core shard in CI recorded 188
such errors; a full local run of the same tests records none, because whatever triggers the
hub path in CI is not triggered on a machine with the full optional dependency set installed.
So this is a failure mode that only the gate can see — and until recently the gate could not
report at all, which is why a defect worth roughly a fifth of a shard's outcomes went
unexamined.

Two suites already forbid downloads for their own scope, and one of them explains why the
environment variable has to be set before the client is imported rather than from a fixture.
That reasoning applies to the whole suite, not just to those two directories.

The immediate goal is not to make those tests pass. It is to make them fail the same way on
a laptop as they do in CI, because a defect that only one environment can see is one nobody
will fix.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No test can download a model asset, and this holds for code that reads the setting at import time as well as code that consults it per request
- [x] #2 The guarantee is a property of the run rather than a claim in a docstring: both halves of it are asserted, and each is shown to fail when removed
- [x] #3 A test that genuinely needs a live fetch can still ask for one, and asking does not turn the suite red
- [x] #4 Turning the guarantee off is visible in a run's summary rather than looking like tests that passed without asserting anything
- [x] #5 Forbidding downloads does not change the outcome of the suites most likely to touch model assets
- [x] #6 The claim that this is CI-visible-only is stated with the evidence for it, and the part that remains unexplained is named rather than glossed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Establish where the cache resolves under test and whether downloads are currently forbidden.
2. Confirm from a CI log what is actually being reached, and how often, rather than inferring it.
3. Check that forbidding downloads is neutral locally before changing anything.
4. Set the environment before any client import; patch the frozen constant for the case where
   the client got in first.
5. Assert both halves, prove each fails when removed, and make the opt-out skip visibly.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Set `HF_HUB_OFFLINE=1` in `Tests/conftest.py`'s pre-import bootstrap, and added an autouse
fixture that patches `huggingface_hub.constants.HF_HUB_OFFLINE` when the module is already
in `sys.modules`.

**Why two halves (AC#1).** `huggingface_hub` freezes `constants.HF_HUB_OFFLINE` at *its* import
time, so an env var written from a fixture is read too late to matter, while
`is_offline_mode()` consults the constant on every request. The bootstrap covers the normal
case (nothing has imported the client yet); the fixture covers the case where something got
in first. `Tests/RAG_Eval/conftest.py` already documents this split for its own scope, with
the measurement — this generalises it. The fixture looks the module up through `sys.modules`
and never imports it: importing the hub stack in every session to assert a property of
sessions that never touch it would cost more than the guard is worth.

**What was measured (AC#6).** Under pytest the cache resolves to
`<sandbox>/home/.cache/huggingface/hub`, which does **not exist**, and `HF_HUB_OFFLINE` was
**False** — so any code path reaching the hub attempted a real download. From CI core shard
0/6 of run 32676835700: **188 of its 189 errors** were the network guard refusing egress —
709 recorded attempts to a CDN address and 65 to `huggingface.co:443`, the excess over the
error count being `huggingface_hub` retrying five times per call (visible as
`Retrying in 2s [Retry 2/5]`).

**Named rather than glossed:** the same tests record **zero** egress attempts locally
(`test_console_agent_bridge.py`: 254 passed, 0 blocks; CI: 19 errors). The cache is cold in
both, so the cold cache is not the differentiator, and I did not establish what is. The most
likely candidate is the dependency set — CI installs `.[embeddings_rag,websearch,chunker]`
while this venv has every optional group, and optional-dependency branches decide whether the
hub path is entered at all. **This is not confirmed.** What is confirmed is that the hub path
runs in CI, that nothing forbade the download, and that this change forbids it.

An earlier hypothesis — that tiktoken's temp-dir cache was responsible — was **wrong** and is
recorded so it is not retried: a cold `TIKTOKEN_CACHE_DIR` reproduced nothing (269 passed),
and the CI log shows `huggingface_hub` retrying, not tiktoken. The tiktoken `ERROR` lines in
that log are incidental.

**AC#3/#4.** `TLDW_TEST_ALLOW_HF_DOWNLOADS=1` opts out. The guard module carries a
`pytestmark` skipif on it rather than asserting the flag is unset — a developer who asks for
live fetches should not get a red suite — and it *skips* rather than returning early, so the
opt-out shows up in the run summary instead of looking like three tests that passed while
asserting nothing. Verified: 3 passed normally, **3 skipped** when opted out.

**AC#2 — mutation-proven.** Replacing the bootstrap condition with `if False` reddens
`test_the_environment_declares_offline_before_any_import`. Notably the *other* two tests still
passed under that mutation, because the autouse fixture patched the constant — the two halves
are genuinely belt-and-braces rather than one guard written twice.

**AC#5 — neutral locally.** Two A/Bs, offline forced on versus opted out:
`test_console_agent_bridge` + `test_fleet_runtime` + `test_console_provider_gateway`:
**633 passed** both ways. `Tests/RAG_Search` + `Tests/Chunking` + `Tests/Transcription`:
**4 failed / 1214 passed / 74 skipped / 1 xfailed** both ways, identical.

**Not claimed:** that this makes CI green. It makes the failure mode reproducible and
deterministic instead of CI-only, which is the precondition for fixing whatever those tests
then do. Whether the affected tests pass offline or fail fast can only be answered by a CI
run, so the PR is opened for that evidence rather than merged on local results.

A broader local sweep was attempted and discarded rather than reported: concurrent sessions
had 12–21 pytest processes running and it crashed xdist workers repeatedly. CI shard logs show
**zero** `node down` events, so those crashes are this machine, not the code.

Added: `Tests/test_huggingface_offline.py`. Modified: `Tests/conftest.py`.
<!-- SECTION:NOTES:END -->
