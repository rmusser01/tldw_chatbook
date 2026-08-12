# TASK-602 Native Parakeet Platform Evidence Design

**Status:** approved by the user on 2026-08-12

**Date:** 2026-08-12

**Related task:** TASK-602

**Canonical ADR:**
[ADR-025](../../../backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md)

## Purpose

Close TASK-602 acceptance criterion 7 with reproducible evidence that the
pinned Parakeet ONNX CPU path installs and runs on every required wheel target:

- Linux x86_64;
- Linux aarch64;
- Windows x86_64;
- macOS arm64; and
- macOS x86_64.

All five lanes must pass on one executable commit. The prior Apple-silicon
smoke remains useful historical evidence, but it cannot be combined with newer
platform results to close the gate.

This work verifies the existing implementation. It does not promote semantic
defaults, remove legacy providers, add accelerator support, or become general
CI.

## Governing decision

**ADR required:** no new ADR.

**ADR path:**
`backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md`.

**Reason:** ADR-025 already defines the CPU dependency profile, five required
wheel targets, offline managed-artifact runtime, cancellation, resident reuse,
and explicit recovery contracts. This design supplies the deferred evidence
without changing those boundaries.

## Approaches considered

### Explicit five-lane release-evidence workflow — selected

Run a label-gated or manually dispatched GitHub Actions matrix on one current
hosted runner for each required operating-system/architecture pair. Each lane
cleanly resolves the pinned runtime, acquires the exact managed v2/v3 INT8 and
Silero VAD artifacts, exercises production runtime paths, and writes a bounded
JSON result.

This is the smallest reproducible proof that covers the real native wheels and
CPU execution providers without burdening ordinary CI.

### Manual native hosts — rejected

Manual runs are harder to reproduce, bind to one commit, validate uniformly,
and rerun after a fix. They also leave Linux aarch64 and macOS x86_64 host
availability unresolved.

### General CI — rejected

Downloading about 1.34 GB of exact model data per lane on every pull request is
expensive and noisy. A release gate should run only when explicitly requested.

## Native matrix and dependency contract

The workflow uses Python 3.12 and these current hosted-runner labels:

| Evidence name | Runner | Expected host |
| --- | --- | --- |
| `linux-x86_64` | `ubuntu-24.04` | Linux x86_64 |
| `linux-aarch64` | `ubuntu-24.04-arm` | Linux aarch64 |
| `windows-x86_64` | `windows-2022` | Windows x86_64 |
| `macos-arm64` | `macos-15` | Darwin arm64 |
| `macos-x86_64` | `macos-15-intel` | Darwin x86_64 |

Each lane installs the repository with the exact Parakeet CPU and
faster-whisper recovery extras in a clean environment. The evidence runner
records the resolved `onnx-asr`, `onnxruntime`, `faster-whisper`, and
`ctranslate2` versions and requires `CPUExecutionProvider`. Accelerator ONNX
Runtime distributions are forbidden.

The package-resolution claim is limited to TASK-602's selected Parakeet and
recovery profiles. It does not claim that unrelated optional stacks in
`all-tools` are supported on every host.

## Model and fixture acquisition

The user approved network acquisition in the explicit evidence workflow.
Every lane downloads through Chatbook's production `ModelArtifactService` and
existing Parakeet preflight/provision helpers:

- exact managed Parakeet v2 INT8;
- exact managed Parakeet v3 INT8; and
- the exact pinned Silero VAD dependency shared by both closures.

The runner verifies the descriptor-declared sizes and SHA-256 digests before a
runtime receives local paths. Runtime inference is then offline; no provider or
worker may call a hub or download API.

The speech fixture is PyTorch Audio's 16 kHz mono VOiCES tutorial sample,
`Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav`, released under CC BY 4.0.
The workflow downloads it from PyTorch's tutorial-assets host and requires
SHA-256
`c65fcd726d6b08c82c1e5dc7558f863cd8d483e3ed2f4a7bcf271dc1865ada14`.
It is not committed to the repository. A checksum mismatch fails before
inference.

## Required smoke contract

One spawn-safe Python evidence runner exercises the existing production
boundaries. It does not implement a second provider or artifact path.

Every lane must prove:

1. **Package resolution and probe:** the pinned CPU runtime imports, exposes
   `CPUExecutionProvider`, and does not load native modules during the cheap
   application probe.
2. **v2 INT8 CPU inference:** the exact managed v2 closure loads offline and
   performs nonempty inference through `ParakeetOnnxRuntime`.
3. **v3 INT8 CPU inference:** the exact managed v3 closure performs inference
   while retaining requested-language routing semantics: effective language is
   `auto`, detected language is null, and
   `requested_language_not_enforced` is present.
4. **Long-form VAD:** a deterministic long fixture made from the verified
   sample and silence uses the exact managed VAD with ASR batch size one.
5. **Cancellation:** cancellation becomes visible before a second VAD segment
   batch, and that second ASR call does not occur.
6. **Batch reuse:** two same-identity requests use one resident runtime and the
   second result reports no model reload. The root and VAD leases remain held
   until the resident closes.
7. **Retry wiring:** an eligible clear Parakeet failure produces normalized
   failure provenance and the exact `retry_faster_whisper` recovery action; no
   cross-provider retry happens silently.

All waits and network operations are bounded. Cleanup closes the executor and
artifact handles before deleting only the lane-owned temporary directory. A
timeout or cleanup failure keeps the lane red.

## Evidence workflow

Create `.github/workflows/task-602-platform-evidence.yml` with:

- `workflow_dispatch` and `pull_request` `labeled` triggers only;
- label `task-602-platform-evidence`;
- read-only repository permissions;
- checkout of the exact selected commit;
- Python 3.12 and `fail-fast: false`;
- the five native lanes above;
- a bounded per-job timeout;
- failure-document initialization before dependency installation;
- `continue-on-error` only where needed to normalize failure evidence;
- strict validation that restores a red job after artifact creation; and
- one JSON artifact per lane, uploaded on every reachable outcome.

The workflow has no push, schedule, secret, write permission, broad cache, or
general-suite responsibility. A failed or timed-out lane remains an open
release gate.

## Evidence format and validation

Add a TASK-602-specific standard-library normalizer. It owns initialization,
bounded validation, and exact aggregation; the smoke runner supplies only
allowlisted observations.

Each platform document contains:

- schema version and evidence name;
- passed/failed status and stable failure stage/code;
- tested commit, workflow run ID, run attempt, and canonical run URL;
- normalized OS, architecture, and Python identity;
- allowlisted package versions and CPU provider;
- exact root/VAD artifact identities and closure fingerprints;
- the seven required check outcomes and bounded durations; and
- cleanup completion.

It excludes commands, exception text, tracebacks, transcripts, environment
variables, credentials, local paths, usernames, PIDs, handles, and temporary
names. Unknown keys, skipped/missing checks, unexpected hosts, accelerator
providers, malformed identities, unbounded values, or a nominally successful
external step outcome with failed observations are rejected.

The aggregate mode accepts exactly the five validated platform documents and
requires one shared tested commit, workflow run, canonical URL, and passed
status. It writes atomically and cannot manufacture passing evidence from a
failure document.

## Testing and mutation requirements

Dependency-free tests cover:

- all accepted host/name pairs and mismatches;
- initialized, dependency, acquisition, smoke, timeout, and cleanup failures;
- strict package/provider/artifact/check schemas;
- path and secret privacy throughout nested content;
- workflow triggers, permissions, matrix, exact checkout, failure plumbing,
  validation, and upload semantics; and
- five-input aggregation plus every commit/run/platform/check mismatch.

Required mutations must prove the tests reject:

1. a passing document when a required smoke check failed;
2. a host/evidence-name mismatch;
3. aggregation across different commits or runs; and
4. a skipped or absent required check.

The real five-lane run is the native evidence. Unit tests must not fake or
substitute for it.

## Completion

After all five lanes pass one executable commit:

1. Download and individually validate each named artifact.
2. Aggregate only through the checked-in normalizer.
3. Add `Docs/STT_Evaluation/task-602/platform-evidence.json` and update the
   README with the exact workflow URL, tested commit, package/model identities,
   outcomes, fixture attribution, and scope.
4. Update TASK-602 only through Backlog CLI, check AC7, and set the task Done
   only if every Definition-of-Done item still holds.
5. Rebase, rerun only gates affected by the rebase, address every actionable PR
   comment, and merge normally. General unrelated CI is not part of this gate.

Any production, test, workflow, smoke-runner, or normalizer change after the
matrix run invalidates the evidence and requires a new five-lane run. A later
commit may contain only aggregate documentation and Backlog metadata.

## Expected file boundary

- Create `.github/scripts/task602_platform_evidence.py`.
- Create `.github/scripts/task602_platform_smoke.py`.
- Create `.github/workflows/task-602-platform-evidence.yml`.
- Create `Tests/CI/test_task602_platform_evidence.py`.
- Create `Tests/STT/test_task602_platform_smoke.py` only if spawn-safe smoke
  components need focused dependency-injected tests.
- Create `Docs/STT_Evaluation/task-602/platform-evidence.json` only after the
  green native matrix.
- Update `Docs/STT_Evaluation/task-602/README.md` only after the green matrix.
- Update TASK-602 only through Backlog CLI.

No production change is planned. If a native lane exposes a product defect,
stop, record the exact RED, deliberately extend the plan, and prove the fix on
that platform before rerunning all five lanes.
