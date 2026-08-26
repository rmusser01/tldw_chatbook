# TASK-602 Native Parakeet Platform Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task by task.

**Goal:** Close TASK-602 AC7 with strict, path-private, same-commit native
evidence for the pinned Parakeet ONNX CPU path on Linux x86_64/aarch64,
Windows x86_64, and macOS arm64/x86_64.

**Architecture:** Reuse the production artifact service, Parakeet runtime,
coordinator, and local STT executor. Add one explicit five-lane workflow, one
spawn-safe smoke entry point, and one strict normalizer/aggregator. Do not add
general CI, another downloader, another provider, or a persistent model cache.

**Tech stack:** Python 3.12, pytest, GitHub Actions, `onnx-asr[cpu]==0.12.0`,
ONNX Runtime CPU, existing `ModelArtifactService`, existing
`ParakeetOnnxRuntime`, existing `LocalSTTExecutor`.

**Approved design:**
`Docs/superpowers/specs/2026-08-12-task-602-platform-evidence-design.md`

**ADR required:** no.

**ADR path:**
`backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md`.

**Reason:** ADR-025 already governs the runtime, artifact, platform,
cancellation, reuse, and recovery boundaries. This work supplies evidence only.

## Global constraints

- Test-drive each behavior before implementation.
- Run no repository-wide suite and no unrelated general CI.
- All five native lanes must pass the same executable commit.
- Model network access is allowed only during explicit parent-side acquisition.
- Runtime and worker paths remain offline.
- Evidence must contain no local paths, commands, transcripts, exceptions,
  environment values, PIDs, handles, or credentials.
- Keep semantic-default promotion and legacy-provider removal closed for
  TASK-605.
- Do not check AC7 or mark TASK-602 Done until the strict aggregate validates.

## Task 1: Add strict evidence-schema and workflow RED tests

**Files:**

- Create: `Tests/CI/test_task602_platform_evidence.py`
- Create: `Tests/STT/test_task602_platform_smoke.py` only if needed for
  dependency-injected smoke components.

1. Add tests that initially fail because the TASK-602 scripts and workflow do
   not exist.
2. Cover the five host/name pairs, package/provider schema, exact artifact
   identities, seven required checks, bounded duration/failure schemas, nested
   privacy, CLI exclusivity, exact five-input aggregation, workflow triggers,
   matrix, failure normalization, validation, and upload.
3. Add dependency-injected smoke tests for package probe, offline v2/v3 calls,
   VAD cancellation, resident reuse, retry action, and cleanup without loading
   real models locally.
4. Run the exact new test files and record the genuine RED.

## Task 2: Implement the evidence normalizer and aggregator

**Files:**

- Create: `.github/scripts/task602_platform_evidence.py`

1. Implement standard-library initialization, result validation, atomic write,
   strict CLI modes, and exact five-platform aggregation.
2. Bind evidence names to normalized native hosts and construct the canonical
   workflow URL from the allowlisted repository plus numeric run ID.
3. Reject unknown fields, malformed/unbounded numbers and strings, path-like or
   secret content, missing/skipped checks, non-CPU providers, and failed
   external step outcomes.
4. Run the focused schema/CLI/aggregation tests to GREEN.
5. Perform and restore the required failed-check, host-mismatch,
   commit/run-mismatch, and skipped-check mutations.

## Task 3: Implement the production-path smoke runner

**Files:**

- Create: `.github/scripts/task602_platform_smoke.py`
- Modify: focused test files from Task 1 only as needed.

1. Implement a spawn-safe entry point with bounded acquisition, execution, and
   cleanup.
2. Download and checksum the CC BY 4.0 VOiCES fixture.
3. Acquire exact v2/v3 INT8 plus shared VAD through existing preflight and
   provision APIs into a lane-owned temporary store.
4. Disable acquisition/network seams before runtime load.
5. Exercise package probe, v2/v3 CPU inference, real long-form VAD,
   cancellation before the second segment batch, same-identity resident reuse,
   lease retention, and normalized retry eligibility.
6. Return only allowlisted bounded observations to the normalizer; suppress
   path-bearing exception context from the artifact.
7. Run the focused smoke tests to GREEN and perform at least one cancellation
   and one reuse mutation.

## Task 4: Add the explicit five-lane workflow

**Files:**

- Create: `.github/workflows/task-602-platform-evidence.yml`
- Modify: `Tests/CI/test_task602_platform_evidence.py`

1. Add only `workflow_dispatch` and `pull_request:labeled` triggers, the exact
   label gate, read-only permissions, and five native matrix entries.
2. Check out the exact selected commit and set up Python 3.12.
3. Initialize failure evidence, install the exact Parakeet/recovery profiles,
   run the smoke, normalize every reachable failure, strictly validate, and
   upload one JSON artifact per lane even on failure.
4. Keep the job red after any dependency, acquisition, smoke, validation,
   timeout, or cleanup failure.
5. Run workflow semantic tests plus YAML/static checks to GREEN.

## Task 5: Local verification and review

1. Run the exact new evidence/smoke tests.
2. Run directly affected existing Parakeet package/runtime/executor tests.
3. Run Ruff check and format-check only on changed Python files, py_compile on
   the two scripts, a YAML parse/semantic check, JSON fixtures, and
   `git diff --check`.
4. Perform a Ponytail pass: remove duplicate helpers/tests without weakening
   trust-boundary validation.
5. Self-review for implicit runtime downloads, fake v3 language enforcement,
   stale generation acceptance, path leakage, permissive failure evidence,
   broad triggers/permissions, and accidental default promotion.
6. Commit the executable evidence implementation, rebase on current
   `origin/dev`, rerun affected local gates, push, and open a focused PR.

## Task 6: Run and close native evidence

1. Dispatch the exact workflow against the PR head.
2. Monitor all five lanes without retrying a failed executable commit. On a
   native RED, stop before evidence aggregation, capture the exact failure,
   test-drive the minimal fix, and run a brand-new five-lane workflow on the
   repaired commit.
3. After all five lanes pass, download artifacts into separate temporary
   directories and validate each with the checked-in script.
4. Aggregate only through the script and validate the aggregate.
5. Update `Docs/STT_Evaluation/task-602/README.md` and add
   `platform-evidence.json` with the tested commit/run/matrix/scope and VOiCES
   attribution.
6. Through Backlog CLI only, append evidence notes, check AC7, and set TASK-602
   Done after confirming all Definition-of-Done items.
7. Commit only evidence docs and Backlog metadata after the frozen executable
   commit.
8. Address all actionable PR comments, rebase onto current `origin/dev`, repeat
   only verification invalidated by the rebase, push, and merge normally.
