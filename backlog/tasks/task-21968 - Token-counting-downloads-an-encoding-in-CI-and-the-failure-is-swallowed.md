---
id: TASK-21968
title: 'Token counting downloads an encoding in CI, and the failure is swallowed'
status: Done
assignee: []
created_date: ''
updated_date: '2026-08-24 16:51'
labels:
  - testing
  - test-integrity
  - ci
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The single largest source of errors in the core test job.

Token counting and the chunking engine obtain their encodings from a library that downloads
its tables on first use and caches them somewhere the test sandbox does not control. That
cache is warm on any machine that has run the suite before and cold on every CI run, so the
download is attempted on every call there, refused by the egress guard, and then hidden — the
caller wraps the lookup in a broad exception handler and returns nothing. The only reason it
is visible at all is that the guard records the attempt, and the test then fails during
teardown pointing at a network address rather than at tokenizing.

Two things made it hard to see. The library is a dependency the application declares but
which was absent from at least one working environment, so the code path returned early
locally and never reached the network. And a second library appearing in the same logs prints
retry lines while this one does not, which drew the initial diagnosis to the wrong place.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No test attempts to download an encoding, shown by the attempts falling to zero rather than by an error message disappearing
- [x] #2 What the suite actually requests is established by observation, and only that is vendored
- [x] #3 The arrangement cannot rot quietly: both an incomplete set and a renamed entry fail, and each is shown to fail
- [x] #4 Tokenizing is shown to work with the egress guard active, so the evidence is behavioural rather than a file listing
- [x] #5 Production keeps its normal download behaviour; only tests read from the vendored copy
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Vendored the tables under `Tests/fixtures/tiktoken_cache/` and pointed `TIKTOKEN_CACHE_DIR`
at them from the conftest bootstrap.

**Why it was invisible locally (AC#2's harder half).** `tiktoken` is declared in
`pyproject.toml` as hard-required, and CI installs it — but it was **not installed in the
working venv here**, so `import tiktoken` failed, the broad `except` returned `None`, and no
socket was ever opened. Installing it reproduced the CI failure on a laptop immediately:
`Tests/Chunking` went to 14 errors with 32 recorded attempts. That reproduction is what made
the rest of this tractable, and it is the reason the earlier attribution went wrong twice.

**What is actually requested.** Not the fallback everyone assumes. The chunking engine's
default tokenizer is `gpt2`, which uses the two-file data-gym format
(`vocab.bpe` + `encoder.json`) rather than a single `.tiktoken` table — vendoring
`cl100k_base` alone left 33 attempts outstanding. Vendoring `gpt2` as well took it to zero.
`p50k_base`, `r50k_base` and `o200k_base` were fetched during the investigation and removed
again: nothing asked for them, and `o200k_base` alone is 3.5 MB. Final size **3.0 MB** for
three files.

**Evidence (AC#1).** `Tests/Chunking`, same command, vendored cache absent then present:

| | before | after |
|---|---|---|
| errors | **14** | **0** |
| failed | 2 | 1 |
| passed | 603 | 604 |
| recorded download attempts | **32** | **0** |

Widened to `Tests/Chunking` + `Tests/Subscriptions` + `Tests/RAG_Search` + `Tests/Utils`:
**2,756 passed, 0 attempts**, and no encoding requested beyond the three vendored. The two
residual failures both fail identically on dev with only a workflow change applied
(`test_fts5_match_forms_shared` is TASK-19642.19.1; `test_chunking_interop_v7`'s
`TestNoIsSystemAnywhere` is likewise pre-existing).

**AC#3/#4.** `Tests/test_tiktoken_vendored_cache.py` recomputes each entry's key as
`sha1(<download url>)` — tiktoken's own scheme, not a name we chose — so a rename reads to
tiktoken exactly like a missing file and would silently restore downloading. Mutation-proven
twice: removing an entry and renaming one each fail the guard, and in both cases the
*behavioural* test fails too, because the encoding then gets fetched and the egress guard
records it. That is the assertion that makes this evidence rather than a file listing.

**AC#5.** Only `TIKTOKEN_CACHE_DIR` is set, and only from the test conftest. Production
downloads and caches as before, which is correct for a real user.

Added: `Tests/fixtures/tiktoken_cache/` (3 tables + README explaining the opaque filenames),
`Tests/test_tiktoken_vendored_cache.py`. Modified: `Tests/conftest.py`.
<!-- SECTION:NOTES:END -->
