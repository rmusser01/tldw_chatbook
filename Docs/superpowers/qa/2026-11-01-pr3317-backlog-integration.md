# PR3317 integration evidence

Source: a60e027f5db751f57f0256d6b132d08ca32f5679; true parent: 383bf3da799b4276fb124e9225e494a468ae9a6d.
Final base: confirmed PR3316 merge a68971781f3826a108d8116f06bb778aeffa3ae4.
Tested integration head: 37861286e969bcfed14d207c80dc4169a220a6f0.

All three preparation commits retain identical patches after rebase. The PR now targets dev after its dependency merged. Parsed workflow comparison against the final base confirms every other job is identical. The aggregate retains every prior verdict step and adds only actions: read plus the final bounded diagnostic report. Current 32 shards, preflight, obsolete-run cancellation, UAT and the newer architecture/runtime files are preserved.

## Focused execution

195 tests passed, 32 unchanged exhaustive shard-distribution cases deselected, no failures/errors/skips. The run covered the five related modules: cost report, derived-artifacts workflow, queue-pressure contracts, obsolete-run cancellation and merge-queue workflow. It used --noconftest with addopts cleared and cache disabled; no application initialization or full suite was requested. Source hashes before and after are identical. Receipt: sibling pr3298-resync/pr3317-final-b907bde1 (command, result XML/log, source hashes and parsed preservation).

The report tests execute the actual inline JavaScript and cover incomplete timestamps, independent wait/execution validity, current versus cumulative attempts, deduplication, medians, thresholds and API failures. Their fake paginate API verifies endpoint/arguments and returned data handling; it does not qualify real multi-page HTTP delivery. The earlier 19-case report-only pass remains preliminary evidence, not an additional final-run total.

The new Python test parses; scoped Ruff and git diff --check pass. Initial independent source review found no findings; final immutable-head integration review precedes publication.

ADR-103 records the existing diagnostic-only contract. No additional ADR is required. Local tests do not establish hosted summary rendering, timestamp availability, cancellation delivery, throughput or billing accuracy. Hosted CI and module-size ceilings are waived by the user; all existing verdict protections remain in code. Draft PRs and PR #3267/#3297 are excluded. No release or main promotion is part of this work.
