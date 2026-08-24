---
id: TASK-20010
title: Confirm steady-state three-turn Console latency
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-21 21:21'
updated_date: '2026-08-22 16:09'
labels:
  - console
  - performance
  - diagnostics
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run a separately pre-registered steady-state confirmation of the real-provider three-turn Console comparison after balanced burn-in, preserving the original inconclusive benchmark evidence unchanged. Latest-dev integration will canonically renumber that benchmark TASK-20009 while leaving its retained artifacts' internal TASK-19641 label byte-identical.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Five complete balanced burn-in blocks run after one warmup per arm, every burn-in conversation satisfies the full product, privacy, cleanup, and ownership contract, and burn-in is excluded from all measured summaries by a predeclared rule.
- [ ] #2 Thirty fresh measured three-turn samples per arm use the same pinned control and candidate, endpoint-reported model alias, retained server contract, fixtures, request parameters, isolation, and 10% non-regression gates as the original benchmark (canonical TASK-20009 after latest-dev integration); the report discloses that the original evidence did not retain a model-weight digest.
- [ ] #3 All ninety measured conversations complete the exact 1/3/1 provider-round, `load_tools`, confined `fs_write`, terminal-follow-up path with zero prompt loss and clean final ownership.
- [ ] #4 Before filtering, the complete 108-conversation terminal-row identity/order sequence exactly matches the predeclared schedule with global sample-ID uniqueness; retained artifacts collectively preserve phase provenance, summaries retain only excluded burn-in counts and contract status, and no artifact makes a performance claim from burn-in samples.
- [ ] #5 Independent recomputation, privacy scans, focused tests, and static checks exactly validate the retained evidence and verdict.
- [ ] #6 The original benchmark evidence remains byte-identical, including its internal pre-integration TASK-19641 label, and the confirmatory evidence is stored separately under canonical TASK-20010.
- [ ] #7 The first complete protocol-valid attempt is definitive regardless of verdict; correctable derived-artifact defects are fixed and re-reviewed without reacquisition, only uncorrectable acquisition or raw-evidence failures may be retried, and all attempt states remain linked and retained.
- [ ] #8 The original harness revision and runner digest are explicit, its digest-verified original statistics module produces the measured summary, and publication atomically promotes only artifacts whose canonical digest and attempt ID exactly match an approving independent-review receipt.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the pinned candidate object, rebase onto the exact refreshed `origin/dev`, preserve dev's colliding TASK-19641/ADR-077, renumber the branch-owned benchmark task/Change Review ADR without modifying immutable evidence or protocol fixtures, and pin the exact post-integration implementation baseline.
2. Extend the existing benchmark schedule with predeclared balanced burn-in and validate the complete terminal-row order and identity before statistical filtering.
3. Pin and digest-check the original harness/evidence, reuse its validators and statistics directly, and fail closed on protocol, revision, workspace, or listener drift.
4. Add an append-only attempt ledger and atomic acquisition lock that make the first complete protocol-valid attempt definitive and preserve retry lineage.
5. Wire confirmatory acquisition through the existing parent/child runner without changing production Console code.
6. Bind independent review to the exact canonical artifact digest and publish approved evidence through verified sibling-copy plus atomic rename.
7. Verify the harness, run a disposable live smoke against port 9099, acquire and independently review the official 30-block confirmation, then publish the retained evidence.
8. Run the full test/lint/format gates and final evidence checks, record the measured verdict without altering the original evidence, and complete Backlog hygiene only after every gate passes.

Detailed executable plan: `Docs/superpowers/plans/2026-08-22-confirmatory-steady-state-console-latency.md`

ADR required: no

ADR path: `backlog/decisions/079-change-review-consent-and-asynchronous-finalization.md` after latest-dev integration (existing governing ADR, renumbered from branch-local ADR-077)

Reason: this task changes benchmark-only tooling and retained evidence, while the renumbered ADR-079 already governs the Change Review behavior being measured.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria Evidence

- **#1:** The definitive schedule ran three warmups followed by five balanced
  burn-in blocks (15 conversations) and 30 measured blocks. All burn-in product,
  privacy, cleanup, and ownership contracts passed; `phase == measured only`
  excluded burn-in before statistics, and retained summaries contain only its
  count and contract status.
- **#2:** The 90 measured conversations comprise 30 samples per arm against
  control `5f720a40417eaa78f33619d5cbc82effc470104b` and candidate
  `eb8225a32f88ea43c337aff99804d360384e7668`. Protocol equivalence and the
  endpoint-reported model alias passed. The report explicitly states that the
  original benchmark retained no model-weight digest.
- **#3:** All 108 warmup, burn-in, and measured conversations completed the
  exact `1/3/1` provider-round sequence, one `load_tools(local:fs_write)`, one
  confined `fs_write`, and the terminal follow-up, with zero prompt loss and
  clean final ownership.
- **#4:** The raw evidence has 222 rows: 111 child starts, three protocol
  preflight results, and 108 terminal samples. All 108 sample IDs and schedule
  positions are unique and match the predeclared order. The 540 provider calls
  (450 measured) have coherent token accounting; no retained artifact makes a
  burn-in performance claim.
- **#5:** Digest-verified recomputation, JSON/JSONL parsing, exact-inventory and
  attempt-lineage checks, privacy scans, focused harness tests, affected suites,
  scoped static checks, and independent review passed. Repository-wide baseline
  failures are separately disclosed below and are not represented as green.
- **#6:** The five original benchmark files remain byte-identical at the hashes
  recorded below, including the embedded pre-integration `TASK-19641` label.
  Confirmatory evidence is separately retained under canonical TASK-20010.
- **#7:** `attempt-0001` is the first and only complete protocol-valid attempt;
  no retry, replacement, or reacquisition occurred. The initial approval was
  reopened by `review-002` for a derived-manifest omission, then
  `correction-001` was independently approved by `review-003`. The raw hash,
  append-only lineage, corrected artifact digest, approving receipt, and
  published attempt ID agree.
- **#8:** The acquisition harness revision is
  `1275ffc39f81c38821fdf1c6b3cae42da53287ba` with runner SHA-256
  `6591f6755897c73d03abe1e7481659f6f28a6260e25f63d57e88a895659ca9a2`.
  The digest-verified original statistics runner is revision
  `eb8225a32f88ea43c337aff99804d360384e7668`, SHA-256
  `fbca69703b771f7b7b27fa78ef9bf095fb30712435743877e20fcb01bb6d06ae`.
  The corrected manifest records implementation base
  `77c5e9f487af79391a479deb85e712163bfed909`. Publication is bound to
  `attempt-0001` and the approving receipt's exact five-artifact digest.

## Implementation Notes

The implementation extended the benchmark-only harness with predeclared
balanced burn-in, exact schedule validation, append-only attempt lineage,
digest-bound independent review, governed same-acquisition correction, and
fail-closed atomic publication. TASK-20010 did not change production Console
behavior. The separately pre-registered official verdict is **inconclusive**
and is not reinterpreted here.

### Published evidence and immutable identities

The published inventory is the exact five review-bound artifacts plus the
separate approving receipt:

| File | SHA-256 |
| --- | --- |
| `README.md` | `c6e13a0bac597384d323d0585eaa4322e9984854713d13c2c90f4ae6f222db9d` |
| `real-provider-three-turn.raw.jsonl` | `2cdda7f369979fb1ac65f4f668bfd8ad4a28b4d2502f081eb767dc766d944a8c` |
| `real-provider-three-turn.manifest.json` | `cb725ea68dec8cee9621309664721c9ca19c755d8d3955707284f69475209ac1` |
| `real-provider-three-turn.summary.json` | `3be4fb9d451ff9794f0409337e16b461290abe669d9e2e1c785fc62279877f8b` |
| `real-provider-three-turn-summary.md` | `ab6a3a7481f3d4fe3b3f16a76f366707f20b1f4904409a144754b09c98ea636b` |
| `confirmatory-review-receipt.json` | `889b3f8382d7ac78fa931bd8ea50dda5aa78fc8e17cf08d193616702d6a2c95d` |

The canonical five-artifact set digest is
`c04acca85762c5f2cbfe05113223049d907ad2c8436b0ce8909f7ae78267ee49`.
The receipt approves that digest for `attempt-0001`, records no findings,
confirms privacy, and preserves verdict `inconclusive`. The corrected manifest
adds only `implementation_base_revision` with exact value
`77c5e9f487af79391a479deb85e712163bfed909`. Its raw JSONL, summary JSON,
human report, and README are byte-identical to the initial reviewed package.
The retained campaign also preserves initial `review-001` SHA-256
`dcacc70ecf3234960f8a600c6f79407927d5de60d9af188d3c1a63b5443110c8` and
reopening `review-002` SHA-256
`cdc00f8823e4202e88f7b260213cccf882ebbc97d6dacfe38efdb9ec356253a5`.
The original benchmark hashes remain:

- `README.md`: `724be0f80eff3c9a2eced35b86ae4ce2e6f9a7524d44016cd3f49b61752bd491`
- `real-provider-three-turn-summary.md`: `fdb4528bd82a33f244b4e6fbcfe3b739bd2374006cfea2df878f2e0d27a7d5c2`
- `real-provider-three-turn.manifest.json`: `f5dec9153845b585d32660ca87f8d4aef7ad31be4dc431bb52e64fdc29187bb6`
- `real-provider-three-turn.raw.jsonl`: `82150cd55ba701b5a2680f87fce43b15676004fc1609f477f458a7abb2078319`
- `real-provider-three-turn.summary.json`: `edec5d347427748e26c93d21da7ecf121cccedb41ea7d304fb6cdad684f3668a`

Privacy verification found no prompt, response, tool-result, generated-file
body, credential, header, environment dump, absolute workstation path, or
personal workspace content. Across all 108 conversations, 540 provider calls
reported 2,028,888 total tokens; the 450 measured calls used 563,580 tokens in
each candidate arm.

### Verdict and measurements

Values are measured-sample median / nearest-rank p95:

| Metric | Control | Disabled | Enabled |
| --- | ---: | ---: | ---: |
| Third send → worker | 587.70 / 779.44 ms | 171.42 / 288.08 ms | 189.53 / 327.94 ms |
| Event-loop-lag p95 | 14.93 / 24.09 ms | 9.52 / 14.10 ms | 16.14 / 22.78 ms |
| Assistant durable → release | 0.142 / 0.234 ms | 0.049 / 0.069 ms | 0.280 / 0.386 ms |
| Terminal provider → third provider | 2.068 / 2.431 s | 1.490 / 1.837 s | 1.720 / 2.086 s |

Disabled passed the third-send and event-loop-lag gates with upper p95-ratio
bounds `0.4254` and `0.8318`. Enabled passed third-send at `0.5171`, but its
event-loop-lag interval was `0.6037`–`1.3179`, crossing the `1.10` ceiling and
therefore making the official overall verdict inconclusive.

### Verification state

- `.venv/bin/pytest Tests/Performance/test_console_three_turn_profile.py -q`:
  **649 passed, 2 dependency warnings in 75.04s** after corrected publication.
- `.venv/bin/pytest Tests/UI/test_console_native_chat_flow.py Tests/Chat/test_console_agent_bridge.py Tests/Workspaces/test_change_review_consent.py Tests/Workspaces/test_change_review_finalization.py -q`:
  **602 passed, 3 warnings in 372.16s**.
- Focused Ruff, `py_compile`, evidence parsing/recomputation, privacy, digest,
  receipt, and `git diff --check` gates passed.
- Repository-wide Ruff and format remain baseline-red but are exactly identical
  to `refs/benchmarks/task-20010-implementation-base`: Ruff reports 551
  violations across 234 files on both revisions; format reports 1,591 files
  would be reformatted and 2,719 already formatted on both revisions. No mass
  formatting was applied.
- Repository-wide pytest also remains baseline-red. The final unrestricted
  attempt was externally terminated by SIGTERM at 71% after emitting 134
  failure and 17 error markers. It produced no terminal summary, so those
  numbers are explicitly not a complete failure inventory and no green
  full-suite claim is made.
- The branch-level production-module inventory relative to the implementation
  base now prints `tldw_chatbook/config.py` solely because the separately scoped
  TASK-20013 lock-order blocker was fixed during closeout. TASK-20010 itself
  changes only benchmark tooling, tests, retained evidence, and documentation.

Separately scoped blockers discovered and completed during aggregate/full-suite
verification were TASK-20011 (File Notes decorator drift), TASK-20012 (Console
rail recompose wait), TASK-20013 (config/settings lock inversion), TASK-19642.23
(tool-catalog snapshot test boundary), TASK-19642.20 (persistent diagnostic
inventory reconciliation), TASK-20014 (retired Library diagnostic ledger), and
TASK-19574.1 (offline-safe chunking sync and temporary-clone cleanup). Their
changes and gates are recorded in their own Done task files and are not treated
as TASK-20010 implementation.

ADR required: no. Existing
[ADR-079](../decisions/079-change-review-consent-and-asynchronous-finalization.md)
governs the Change Review behavior being measured; this task introduces no new
production storage, ownership, privacy, provider, or runtime boundary.
