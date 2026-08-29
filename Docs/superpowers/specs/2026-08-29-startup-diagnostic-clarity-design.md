# Startup Diagnostic Clarity Design

**Date:** 2026-08-29
**Status:** Approved for implementation planning

## Problem

The submitted startup log mixes optional-feature absence, degraded platform
verification, recoverable cache rejection, and fatal exceptions at the same
visual severity. Two optional subsystems use warning language even though the
base application continues normally, while the OpenTelemetry import and
initialization paths can report the same absence more than once. Security
posture and cache warnings are meaningful but do not state clearly what the
application does next.

## Goals

- Optional HuggingFace-evaluation and OpenTelemetry absence is reported once
  per normal startup path as informational, explicitly naming the disabled
  optional capability.
- Missing optional dependencies still produce actionable errors when a user
  invokes a feature that requires them.
- SQLite and runtime-policy unverified-platform diagnostics remain warnings and
  state that platform permission verification was unavailable while the named
  operation continues with an unverified posture.
- Rejected model-catalog cache entries remain a warning and state that accepted
  entries continue loading and discovery can repopulate missing data.
- No diagnostic exposes paths, credentials, cache contents, or other private
  data.

## Non-goals

- A centralized diagnostic registry, new logging framework, or persisted
  warning ledger.
- Suppressing security-posture or cache-integrity warnings.
- Installing optional dependencies automatically.
- Changing privacy admission, runtime-policy decisions, cache validation, or
  model discovery behavior.

## Design

### Optional subsystems

Use informational severity for expected optional dependency absence. Each
message names the unavailable capability and the opt-in installation action.
Remove duplicate reporting within the same OpenTelemetry initialization flow;
retain one authoritative initialization message rather than logging once at
import and again when initialization is attempted.

Feature-use boundaries continue to raise or return their existing actionable
missing-dependency errors. Startup severity changes do not weaken those guards.

### Security posture

Keep `SQLitePrivacyUnverifiedWarning` and the runtime-policy unverified posture
at warning severity. Update their copy to distinguish "verification unavailable"
from "privacy failure" and explicitly state that the operation continues with
an unverified posture. Preserve existing once-per-owner or operation context and
do not claim that permissions are safe.

### Model catalog cache

Keep rejected cache entries at warning severity. The message reports only the
bounded rejection count and explains that valid entries remain usable and model
discovery may refresh missing entries. Rejection criteria and recovery behavior
do not change.

## Failure Handling

This work changes presentation, severity, and duplicate emission only. Missing
required dependencies at feature-use time, verified privacy failures, runtime
policy denials, and invalid cache records keep their current behavior.

## Verification

Focused tests capture logging/warnings and assert:

- absent HuggingFace and OpenTelemetry dependencies produce one informational
  startup/initialization message per subsystem with actionable capability copy;
- repeated OpenTelemetry initialization does not duplicate the absence notice
  within the supported startup flow;
- SQLite and runtime-policy unverified-platform cases remain warnings, include
  unverified-continuation language, and preserve their current deduplication
  scope;
- model-cache rejection remains a warning, exposes only the count, and states
  valid-entry continuation/recovery;
- actual optional-feature use still returns its existing missing-dependency
  failure.

Run the focused Evals dependency, metrics, private SQLite/private-path,
runtime-policy source-state, and model-catalog disk-cache tests. Do not run the
full suite without explicit user opt-in.

## Delivery and ADR Check

This is one atomic diagnostic-hygiene task and one PR-sized change.

ADR required: no
ADR path: N/A
Reason: severity and wording are corrected without changing security policy,
privacy ownership, dependency boundaries, cache schema, or recovery behavior.
