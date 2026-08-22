# Confirmatory steady-state three-turn Console latency design

Date: 2026-08-21
Task: TASK-19642
Status: approved after independent written-spec review

## Context

TASK-19641 completed its pre-registered real-provider comparison with 30 measured
three-turn conversations per arm. All 90 measured conversations satisfied the mounted
Console, provider, tool, persistence, isolation, and ownership contracts, but three of
the four primary confidence bounds crossed the 10% non-regression ceiling. The retained
verdict is therefore `inconclusive`.

The largest event-loop and third-send observations clustered in measured iterations
2–4. A diagnostic-only, post-hoc view beginning at iteration 5 put all four primary
upper bounds at or below 1.10, but that view cannot change the original verdict because
the original design declared all 30 blocks measured. It supplies a testable hypothesis:
the first few complete rotations include host, Textual, Python, repository, or model
steady-state effects that are not removed by one warmup per arm.

This task performs one separate, pre-registered confirmation. It introduces five
complete balanced burn-in blocks before a fresh 30-block measurement. Burn-in validates
the full product contract and is retained for audit, but it is never used in a
performance-summary calculation, bootstrap, verdict, improvement claim, or replacement
analysis of TASK-19641.

## Goals

1. Test the steady-state hypothesis with a predeclared exclusion rule rather than a
   post-hoc reanalysis.
2. Preserve an apples-to-apples comparison by pinning the same production revisions,
   endpoint-reported model alias, retained server contract, fixtures, provider request
   settings, mounted workflow, isolation, metrics, bootstrap, and decision thresholds
   used by TASK-19641.
3. Retain enough phase-labelled evidence to prove exactly which samples were excluded
   and independently reproduce the measured verdict.
4. Preserve the original TASK-19641 evidence byte-for-byte and report the confirmation
   as a separate result, including another `inconclusive` result if that is what the
   pre-registered rules produce.

## Non-goals

- This task does not revise, supersede, merge with, or selectively filter TASK-19641.
- It does not tune Change Review, the Console, prompts, sampling, llama.cpp, GPU
  placement, or the local model.
- It does not change the 10% ceiling, bootstrap method, confidence level, primary
  metrics, sample count, or pass/regression/inconclusive rules after observing results.
- It does not use burn-in samples to improve statistical power or make a performance
  claim.
- It does not add production telemetry or change production application behavior.

## Immutable comparison and original-evidence guard

The compared production revisions remain exactly:

- control: `5f720a40417eaa78f33619d5cbc82effc470104b`;
- candidate for both review-disabled and review-enabled arms:
  `eb8225a32f88ea43c337aff99804d360384e7668`.

The original benchmark harness is also pinned explicitly:

- original harness revision:
  `eb8225a32f88ea43c337aff99804d360384e7668`;
- original `Tests/Performance/run_console_three_turn_profile.py` SHA-256:
  `fbca69703b771f7b7b27fa78ef9bf095fb30712435743877e20fcb01bb6d06ae`.

The parent resolves and records both hashes once, creates detached worktrees for both,
and executes target application imports only from those worktrees. Benchmark runner
and validation changes may live at a later source revision, but the runner refuses to
start unless its source checkout is clean. Before provider preflight it records the
resolved full `harness_revision` and SHA-256 of the tracked runner file, then verifies
that `git status --porcelain --untracked-files=all` is empty and that the runner bytes
still match the recorded digest. This pins the code that owns scheduling, validation,
statistics, and evidence even though it is intentionally newer than the production
candidate. Preflight repeats the arm-specific behavior fingerprints from TASK-19641
before any sample runs.

Before modifying the harness and again before retaining the confirmation, the task
records SHA-256 values for all five original evidence files under
`Docs/superpowers/qa/console-three-turn-real-provider/`. The task fails its evidence
gate if those values differ from the following immutable baseline:

| File | SHA-256 |
| --- | --- |
| `README.md` | `724be0f80eff3c9a2eced35b86ae4ce2e6f9a7524d44016cd3f49b61752bd491` |
| `real-provider-three-turn-summary.md` | `fdb4528bd82a33f244b4e6fbcfe3b739bd2374006cfea2df878f2e0d27a7d5c2` |
| `real-provider-three-turn.manifest.json` | `f5dec9153845b585d32660ca87f8d4aef7ad31be4dc431bb52e64fdc29187bb6` |
| `real-provider-three-turn.raw.jsonl` | `82150cd55ba701b5a2680f87fce43b15676004fc1609f477f458a7abb2078319` |
| `real-provider-three-turn.summary.json` | `edec5d347427748e26c93d21da7ecf121cccedb41ea7d304fb6cdad684f3668a` |

The runner writes into one explicit confirmation campaign root outside the repository.
That root contains an append-only attempt ledger and a new, uniquely numbered staging
directory for each attempt; it never writes measurement files directly into
`Docs/superpowers/qa/console-three-turn-real-provider-confirmatory/`. Each attempt
records `running`, `failed`, `invalid`, `complete_pending_review`, or
`changes_required` state and is preserved on failure with its last flushed boundary.
The ledger links every attempt in order with a content-free failure/invalid reason.

Starting or recovering an attempt requires one atomically acquired campaign lock.
`running`, `complete_pending_review`, and `changes_required` all block new acquisition.
A normal process exit releases the lock after recording terminal state. If abrupt
termination leaves `running`, an explicit recovery operation may append
`failed:interrupted` only after it holds the campaign lock and verifies the recorded
owner process is no longer live; it never completes or repairs raw evidence. Otherwise
recovery refuses to proceed.

The first complete protocol-valid attempt is definitive whether its result is `pass`,
`regression`, or `inconclusive`. A retry is allowed only when the preceding attempt is
objectively invalid or failed because an uncorrectable provider, acquisition, raw-data,
product-contract, completeness, isolation, privacy, or ownership gate failed. A
correctable manifest, summary, human-report, README, digest, receipt, or presentation
defect enters `changes_required`; those derived artifacts must be corrected and reviewed
again against the same raw acquisition. A slow, noisy, regressing, inconclusive, or
correctable derived-artifact result is not a retry reason. The reviewed manifest retains
the ordered content-free attempt lineage through `complete_pending_review`, and earlier
staging roots remain available for audit.

After the human report and README are complete, the harness creates one canonical
artifact-set digest: SHA-256 of a deterministic, sorted mapping from every retained
relative artifact path to that file's SHA-256. Independent review records the attempt
ID, exact set digest, verdict, and content-free findings in a review receipt. A
`changes_required` receipt is bound to the rejected digest and blocks new acquisition;
after correction, a new set digest requires a new review. An approving receipt is the
immutable, non-circular marker that the referenced `complete_pending_review` attempt is
definitive; neither the reviewed manifest nor reviewed ledger snapshot is rewritten
after approval.

Promotion refuses a missing/non-approving receipt or any attempt-ID/digest mismatch;
rehashes the source immediately before copying; uses standard-library copy into a new
sibling temporary directory; rehashes the copy; and atomically renames that directory to
`Docs/superpowers/qa/console-three-turn-real-provider-confirmatory/`. The final
directory must not already exist. The receipt is copied beside the reviewed files but
is not part of the reviewed set, avoiding a circular digest. No bespoke copy protocol
or resumable publisher is introduced.

## Pre-registered schedule

The schedule has three ordered phases:

1. **Warmup:** one successful sample per arm, in `control, disabled, enabled` order.
2. **Burn-in:** five complete balanced blocks, 15 samples total. Arm order rotates by
   block with the same three permutations used by TASK-19641. Burn-in iterations are
   numbered 0–4 within the `burn_in` phase.
3. **Measured:** 30 new complete balanced blocks, 90 samples total. Rotation continues
   from global block index 5, so the first measured order is
   `enabled, control, disabled`; measured iterations are still numbered 0–29. This
   avoids giving control two consecutive samples at the burn-in/measurement seam while
   retaining ten occurrences of each arm-order position, exactly as TASK-19641 did.

Every warmup and burn-in sample must satisfy `validate_sample`, privacy rules, cleanup,
and ownership checks before the next sample begins. Any failure stops the run and makes
the run invalid. A phase is determined entirely by the generated schedule before
provider preflight; elapsed time, observed latency, host load, model response, or arm
cannot reclassify a sample.

Only the 90 rows whose phase is exactly `measured` enter metric arrays, arm summaries,
paired blocks, confidence intervals, validity cardinality for the reported measurement,
or verdict. Before any filtering, the phase wrapper extracts every terminal sample row
from raw evidence and compares its complete `(sample_id, phase, iteration, arm)`
sequence with the predeclared 3 + 15 + 90 schedule. Row position supplies ordering; no
duplicate identity field is added. The wrapper also enforces global sample-ID uniqueness
and invokes the digest-verified original runner's `validate_sample` for every row.
Missing, extra, duplicated, misordered, unknown-phase, or sample-invalid rows invalidate
the run; they cannot be silently dropped. Only after that exact-sequence check does the
wrapper remove the 15 burn-in rows and pass exactly three warmups plus 90 measured rows
onward.

## Product path, provider, and isolation

Apart from the schedule and immutable-candidate worktree ownership, TASK-19641's
approved design remains normative. Each sample is a fresh mounted Console conversation
using the same synthetic corpus, explicit `rw` binding, permission definition, three
composer sends, common terminal-turn-two trigger, and exact 1/3/1 provider rounds.
Turn two calls only `load_tools(local:fs_write)`, then only `fs_write`, then completes
with a terminal assistant follow-up consuming the successful result. The write remains
confined to `measured/turn-two.txt` with the same fixed bytes.

The endpoint remains `http://127.0.0.1:9099` and must identify
`gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf`. Every request retains
temperature `0`, 512 maximum output tokens, streaming with usage, and reasoning effort
`none`. Listener isolation, loopback-only validation, provider/tool fail-fast checks,
sample deadlines, child-process isolation, path and credential privacy rules, source
write inventory, and final ownership checks are unchanged.

TASK-19641 retained the endpoint-reported model alias and sanitized llama-server
metadata, but not a model-weight digest. This confirmation therefore cannot claim that
the GGUF bytes are historically identical. It fails closed unless every retained
`provider_server` field matches the original manifest—build info, context tokens,
endpoint capabilities, sleep readiness, modalities, model alias, and total slots—and
unless the complete original `runtime` object matches, including Python implementation
and version, dependency versions, and SQLite. It also verifies one listener process
identity remains continuous throughout each attempt. The report states the historical
weight-identity limitation explicitly. Computing a new weight digest is intentionally
out of scope because no historical digest exists to compare it with.

The operator dedicates the single-slot listener to this run and does not edit either
target worktree or the harness while it is active. The manifest records runtime,
provider, host-load, and listener-resource metadata at the same boundaries as the
original run.

Before any conversation, a pure protocol preflight loads the retained TASK-19641
manifest and machine summary and compares them with the confirmatory harness's declared
protocol. It fails closed unless all of the following match:

- target revision hashes, provider kind, endpoint-reported model alias, retained server
  build/runtime contract, temperature, maximum tokens, reasoning effort, stream options,
  fixture IDs, prompt hash, mutation hash, corpus digest, and per-arm tool-definition
  hashes;
- the exact six metric names visible in the original machine summary and the two
  primary gate names;
- nearest-rank p95, 30 measured blocks, paired complete-block resampling, 10,000
  resamples, seed `19_641`, 95% two-sided and one-sided confidence levels, 1.10
  non-regression ceiling, and 1.00 improvement ceiling as fixed by the pinned original
  harness revision and runner digest.

The preflight derives prompt and mutation hashes from the current harness bytes rather
than trusting labels, derives the corpus digest from the harness-generated corpus,
derives tool-definition hashes through the pinned target adapters, and checks that the
original summary exposes exactly the expected metrics and gate structure. It verifies
the pinned original runner bytes before using that runner as the machine source for
original statistical constants and the executed statistics implementation; Markdown
prose is not a machine-validation input. A mismatch makes the attempt invalid before a
warmup; it cannot be accepted with a warning.

## Harness changes

The smallest harness extension is preferred:

1. `SamplePlan.phase` accepts `warmup`, `burn_in`, and `measured`.
2. `sample_schedule` receives an explicit burn-in block count, defaulting to zero for
   existing callers. The confirmatory entry point supplies exactly five and continues
   rotation across the phase seam.
3. A small phase wrapper compares the full terminal-row identity/order sequence with
   the exact predeclared schedule, enforces global uniqueness, and runs the full sample
   contract before filtering anything.
4. After verifying its SHA-256, parent mode loads the pinned original runner as an
   isolated module and directly invokes that module's original `validate_run` and
   `build_summary` on the filtered three-warmup + 90-measured rows. This retains the
   complete transitive statistics path without a custom dependency-digest walker. Tests
   prove changing burn-in latency alone leaves the entire summary byte-equivalent.
5. Parent mode creates and removes detached control and candidate target worktrees,
   while rejecting a dirty harness and recording `harness_revision` plus the runner
   file SHA-256.
6. A pure protocol-equivalence guard checks the current declared protocol against the
   immutable TASK-19641 manifest and summary plus the explicitly pinned original runner
   before sampling; it does not parse the design document.
7. The manifest records the phase schedule, `burn_in_blocks: 5`, the exclusion rule,
   protocol-equivalence result, pinned original evidence hashes, target revisions,
   harness revision, and runner digest.
8. Parent mode owns the ordered campaign ledger and unique staging-attempt state; a
   small promotion path verifies the review-bound artifact digest and uses a
   standard-library sibling copy plus atomic rename.

The existing command remains backward-compatible with zero burn-in unless an explicit
confirmatory option is supplied. A dedicated documented confirmatory command must make
five burn-in blocks and 30 measured blocks visible at invocation; hidden environment
switches are prohibited.

## Retained evidence

The confirmatory directory contains:

- a raw JSONL file with child-start and terminal rows for all 108 conversations;
- a manifest with the full predeclared 3 + 15 + 90 schedule and exclusion rule;
- a machine summary derived from only the 90 measured rows;
- a human report that clearly labels the result confirmatory and describes burn-in only
  with contract-completeness and provenance facts, not latency comparisons;
- a README with the exact preflight, smoke, full-run, and independent-recomputation
  commands;
- a review receipt bound to the canonical digest of the other retained artifacts.

Raw rows retain `phase`, phase-local iteration, arm, and schedule position. The machine
summary reports measured sample counts and may report burn-in only as excluded counts
and contract status. It contains no burn-in latency distribution, candidate/control
ratio, confidence bound, or verdict input. The human report may state why burn-in was
pre-registered and whether all burn-in contracts passed; it must not describe post-hoc
burn-in performance.

As before, evidence contains no prompt, response, tool-result, or generated-file body;
no API key, header, environment dump, absolute workstation path, or personal workspace
content; and no secret-bearing listener command line.

## Statistics and decision rules

The 30 measured rotated iteration triples are the only pairing blocks. The report uses
the same nearest-rank p95, medians, deterministic 10,000-resample paired block bootstrap,
seed `19_641`, two-sided 95% interval, and one-sided 95% bounds as TASK-19641.

The two primary metrics remain:

- turn-three `third_send_requested` to worker start;
- per-sample p95 10 ms Textual heartbeat lag.

Each candidate arm passes only when both primary p95-ratio upper bounds are at or below
1.10. A metric is a measured regression only when its lower bound is above 1.10;
otherwise it is `inconclusive`. Critical-path improvement claims retain the original
application-owned metrics and require an upper bound below 1.00. Provider latency and
complete wall time remain descriptive only.

There is no optional stopping, sample replacement, variance override, second bootstrap
seed, alternative quantile, or threshold change. If the complete run is still
inconclusive, the report says so and the task does not manufacture a pass. Provider,
contract, isolation, privacy, completeness, or ownership failure makes the result
invalid.

The campaign ledger enforces this rule operationally: `complete_pending_review` and
`changes_required` refuse another acquisition even if the observed result is
unfavorable. Only an uncorrectable acquisition/raw-evidence defect can make an attempt
invalid and allow a retry. Invalid and failed attempts retain their objective gate
failure and remain linked from the definitive receipt's reviewed manifest.

The confirmatory report stands beside TASK-19641. It may say whether the independent
pre-registered confirmation passed, regressed, or was inconclusive; it may not rewrite
TASK-19641's `inconclusive` verdict or pool the two datasets.

## Verification

Test-driven implementation adds focused tests for:

- exact warmup, five-block burn-in, and 30-block measured schedule/order;
- continued global rotation at the burn-in/measurement seam;
- backward-compatible zero-burn-in scheduling;
- burn-in sample, phase, identity, ordering, contract, and completeness failures;
- whole-run sequence mismatches, unknown phases, cross-phase duplicate IDs, and rows
  that would otherwise disappear during filtering;
- measured-only summary and bootstrap inputs, including extreme burn-in latency that
  leaves the entire summary byte-equivalent;
- separate detached target revisions, direct loading of the digest-verified original
  runner for summary generation, and recorded current harness revision/digest;
- dirty-harness refusal and runner-file digest mismatch;
- fixture, request-setting, metric, bootstrap, confidence-level, and threshold
  protocol mismatches against the retained evidence and pinned original runner;
- original-evidence SHA-256 guard and separate output-root enforcement;
- ordered attempt-ledger behavior, first-valid-attempt finality, and permitted invalid
  retries;
- atomic campaign locking, interrupted-owner recovery, and refusal to reacquire while
  `complete_pending_review` or `changes_required`;
- derived-artifact correction and digest-bound re-review without provider reacquisition;
- canonical artifact-set and review-receipt digest binding, source/destination rehash,
  approving-receipt finality, pre-existing-destination refusal, and sibling-temp atomic
  publication;
- manifest and privacy validation for the new phase and exclusion metadata.

Before the long run, one confirmation smoke uses one warmup per arm, one burn-in block,
and one measured block to prove the phase plumbing and mounted product path. The smoke
is not part of the retained measurement. The final evidence is accepted only after:

1. focused benchmark tests, changed-surface Console tests, Ruff, `py_compile`, and
   `git diff --check` pass;
2. raw JSONL independently recomputes the manifest cardinality, all machine summary
   metrics, confidence bounds, gate verdicts, provider/tool contract, prompt ordering,
   isolation, ownership, and token-use totals;
3. privacy and absolute-path scans pass;
4. all original TASK-19641 evidence hashes match the immutable table above;
5. an independent review finds no Important evidence or correctness defect.

## ADR check

ADR required: no.

This task changes opt-in benchmark tooling and adds separate retained evidence. It does
not alter production storage, runtime boundaries, provider contracts, ownership,
security posture, or long-lived UX. ADR-077 remains the governing decision for Change
Review consent, asynchronous finalization, and advisory behavior.
