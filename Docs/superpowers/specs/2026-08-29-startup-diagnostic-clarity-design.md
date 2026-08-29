# Startup Diagnostic Clarity Design

**Date:** 2026-08-29
**Status:** Revised after adversarial review; awaiting approval

## Problem

The submitted startup log mixes optional-feature absence, degraded platform
verification, recoverable cache rejection, and fatal exceptions at the same
visual severity. Two optional subsystems use warning language even though the
base application continues normally, while the OpenTelemetry import and
initialization paths can report the same absence more than once. Security
posture and cache warnings are meaningful but do not state clearly what the
application does next.

## Goals

- The HuggingFace-evaluation import boundary reports optional dependency
  absence once as informational, explicitly naming the disabled capability.
- OpenTelemetry absence is reported once when initialization is actually
  attempted. Importing the module remains silent, and startup paths that do not
  initialize telemetry do not claim an outcome.
- Missing optional dependencies still produce actionable errors when a user
  invokes a feature that requires them, including the evaluation runner paths
  used by production orchestration.
- Prometheus and OpenTelemetry initializers are the sole emitters of their
  normal success/unavailable outcomes; application callers never add a false
  or duplicate success message.
- SQLite and runtime-policy unverified-platform diagnostics remain warnings and
  state that platform permission verification was unavailable while the named
  operation continues with an unverified posture.
- Rejected model-catalog cache entries remain a warning and state that accepted
  entries continue loading and discovery can repopulate missing data.
- Changed diagnostics expose no paths, credentials, environment-provided
  service names, cache contents, or arbitrary third-party exception text.

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
The HuggingFace notice is emitted at the existing task-loader import boundary;
normal Python import caching supplies the once-per-process behavior without a
new diagnostic registry. The OpenTelemetry module emits nothing merely because
it was imported.

`Metrics.Otel_Metrics.init_metrics` is the authoritative OpenTelemetry outcome
boundary. It reports and returns `False` when the optional dependency is
unavailable, reports successful initialization and returns `True` when enabled,
and is idempotent after either outcome so repeat calls do not replace the global
provider or duplicate notices. Protect initialization state with the module's
thread-safety discipline so concurrent first calls cannot initialize twice.
The success message is static and does not interpolate `OTEL_SERVICE_NAME` or
other environment-provided values.

The alternate `python -m tldw_chatbook.app` startup path continues to attempt
OpenTelemetry initialization but no longer emits an unconditional success
message afterward. Its exception diagnostic reports only a bounded static
message plus the exception type, never arbitrary exception text. The installed
`tldw-cli` entry point currently does not initialize OpenTelemetry; this change
does not add that runtime side effect or promise a telemetry notice on that
path.

Apply the same ownership correction to the adjacent Prometheus startup flow.
`init_metrics_server` returns `False` when the client is unavailable and `True`
after the server starts, while remaining the sole emitter of its normal
unavailable/success outcome. The application removes its unconditional success
message and sanitizes unexpected exception diagnostics to the exception type.
Because `prometheus_client` is supplied by optional development/debugging
extras rather than the base dependencies, its unavailable outcome is
informational; server-start failures remain warnings. Server-start behavior
otherwise remains unchanged.

Feature-use boundaries retain actionable missing-dependency failures. In
addition to the existing `TaskLoader` guard, both dataset-loader implementations
used by evaluation code must recognize an `owner/dataset` identifier before
checking availability. When `datasets` is absent they return the specific
install-the-dependency error as a typed `DatasetLoadingError`, rather than
letting the private loader's `ImportError` become the generic "Unexpected
error: ImportError" response or the generic "cannot determine dataset type"
response. Tests assert the public error message and suggestion seen by callers,
not only the private helper exception. Local path detection and invalid
identifier behavior do not change, and no shared abstraction is introduced for
this small correction.

### Security posture

Keep `SQLitePrivacyUnverifiedWarning` and the runtime-policy unverified posture
at warning severity. Update their copy to distinguish "verification unavailable"
from "privacy failure" and explicitly state that the operation continues with
an unverified posture. Preserve existing once-per-owner or operation context and
do not claim that permissions are safe.

### Model catalog cache

Keep rejected cache entries at warning severity. The message reports only the
bounded rejection count and explains that any valid entries remain usable and
model discovery may refresh missing entries. Rejection criteria and recovery
behavior do not change.

## Failure Handling

This work primarily changes presentation, severity, and duplicate emission. It
also fixes two existing evaluation routing guards so a recognized HuggingFace
identifier reaches the already-defined missing-dependency outcome. Missing
required dependencies at feature-use time, verified privacy failures, runtime
policy denials, and invalid cache records otherwise keep their current behavior.

## Verification

Focused tests capture logging/warnings and assert:

- an isolated subprocess import proves absent HuggingFace support produces one
  informational task-loader message with actionable capability copy and no
  import-time warning; subprocess isolation is required because test collection
  imports this module before ordinary log capture starts;
- a fresh isolated subprocess also proves importing the OpenTelemetry module
  emits no optional-absence warning; repeated and concurrent unavailable
  initialization returns `False`, reports one informational notice, and
  produces zero success messages;
- available OpenTelemetry initialization returns `True`, reports one static
  success message, and repeat calls preserve the same initialized state without
  resetting process-global SDK providers or instrumentation. Tests stub the SDK
  collaborators rather than mutating the test process's global provider, and a
  fixture resets the module's new initialization state between unavailable and
  available cases;
- Prometheus initialization returns the correct boolean outcome, emits one
  authoritative success/unavailable message at the specified severity, and the
  alternate application startup path adds no unconditional success message for
  either metrics system. The available-path test stubs `start_http_server` and
  never binds a real network listener;
- both evaluation dataset-loader paths produce the actionable missing-`datasets`
  failure for an `owner/dataset` identifier while local and invalid-source
  routing remains unchanged;
- SQLite and runtime-policy unverified-platform cases remain warnings, include
  unverified-continuation language, and preserve their current deduplication
  scope;
- model-cache rejection remains a warning, exposes only the count, and states
  valid-entry continuation/recovery;
- every changed diagnostic excludes representative credential, local-path, and
  cache-content sentinels, not only real production values; telemetry success
  also excludes a service-name sentinel, and unexpected initializer failures
  exclude exception-message sentinels while retaining the exception type.

Run the focused Evals dependency, metrics, private SQLite/private-path,
runtime-policy source-state, and model-catalog disk-cache tests. Do not run the
full suite without explicit user opt-in.

## Delivery and ADR Check

This is one atomic diagnostic-hygiene task and one PR-sized change.

ADR required: no
ADR path: N/A
Reason: severity and wording are corrected without changing security policy,
privacy ownership, dependency boundaries, cache schema, or recovery behavior.
