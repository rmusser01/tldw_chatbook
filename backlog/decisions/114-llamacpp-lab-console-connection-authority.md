# ADR-114: Own llama.cpp process, readiness, Console adoption, and defaults separately

Status: Accepted
Date: 2026-09-03
Related Tasks: TASK-31200 through TASK-31206
Extends: ADR-002, ADR-025, ADR-095

## Decision

Chatbook will represent a verified llama.cpp connection with one sanitized,
process-local descriptor:

```text
LlamaCppConnectionTarget
  provider_key: canonical llama_cpp identity
  base_url: canonical credential-free persisted endpoint root
  model_id: exact non-path-identifying model ID returned by the verified endpoint
  runtime_owner: lab_process | external_server
  verification_generation: process-local opaque generation
```

`base_url` is produced by `resolve_provider_endpoint`; the descriptor never carries
a chat-completions suffix. `model_id` is endpoint-reported identity, not a GGUF
path, managed artifact path, or filename-derived global identity. `runtime_owner`
is informational lifecycle provenance, not authority for Console to stop a process.
`verification_generation` exists only to reject stale probes and is never persisted.
Executable paths, external GGUF paths, and managed store paths remain Lab-owned.
Credentials, raw commands, and raw log output are also excluded from the descriptor;
the Privacy and observability boundary below governs what Lab may retain or render.

For every Lab-owned llama.cpp launch, Chatbook reserves the stable model alias
`chatbook-llamacpp`. The launch builder emits exactly one
`--alias chatbook-llamacpp`; the value is never derived from the selected GGUF path
or filename. Expert/raw arguments cannot supply any `-a` or `--alias` option,
including `-a=...` and `--alias=...` attached-value forms; they cannot replace the
reserved value or add another model alias. A Lab-owned endpoint is model-ready only
when `/v1/models` reports that exact reserved alias.

For an existing server, Chatbook preserves an accepted selected model ID exactly,
but first classifies it without publishing it. An ID is path-identifying only when
its raw text, viewed with surrounding ASCII whitespace removed for classification,
has an unambiguous filesystem marker: a case-insensitive `file:` URI; a POSIX
absolute, explicit relative, or home-relative prefix (`/`, `./`, `../`, or `~/`); a
Windows drive-root prefix such as `C:\` or `C:/`; a UNC prefix (`\\` or `//`);
any backslash path separator; a `.` or `..` path segment; or a final path
component ending in `.gguf` case-insensitively. An interior forward slash alone is
not path-identifying, so ordinary namespace-style IDs such as `owner/model` remain
valid.

A path-identifying candidate never enters a model selector, descriptor, handoff,
display, copy payload, or log. If the selected existing-server identity is
path-identifying, verification fails closed before descriptor creation or adoption
and shows only a bounded recovery message directing the user to restart or configure
`llama-server` with a non-path `--alias`, then check again. The rejected value is
never interpolated into that message.

The descriptor uses `canonical_connection_identity` wherever a provider and endpoint
must be compared. Its GGUF boundary remains governed by ADR-025: managed artifacts
and external user-owned paths are distinct launch authorities, and neither path nor
filename becomes model identity. Its Console boundary remains governed by ADR-095:
provider endpoints are configuration-owned and never enter conversation generation
metadata.

Readiness is not process liveness. The contract has three distinct vocabularies:

```text
Runtime truth: unclaimed -> reserved -> process_alive -> process_dead
Connection truth: unchecked -> checking -> api_healthy -> model_available -> stale_or_failed
Product state: not_configured | checking | starting | loading_model | api_ready | console_connected | needs_attention
```

`process_alive` alone may project to `starting` or `loading_model`; it can never
project to `api_ready`. API ready requires both a successful health-compatible probe
and a successful `/v1/models` response containing the selected exact admissible
model ID: the reserved alias for a Lab-owned launch or the non-path-identifying exact
ID selected from an existing server. A port collision may offer **Connect to it**
only after that exact endpoint passes the same llama.cpp-compatible health and model
checks.

Every probe and handoff carries the exact `verification_generation`. Cancellation,
process death, target edit, model change, screen recomposition, or a newer probe
invalidates older evidence. A stale result cannot expose **Use in Console** or modify
Console.

## Context

The existing llama.cpp flow has three context-specific defaults. The Lab launch
fallback is port `8001` in `llm_management_events.py`; configured provider and local
discovery defaults use port `8080` in `config.py` and `local_server_discovery.py`;
and the Console direct-path fallback is port `9099` in
`console_session_settings.py`. This makes the same absent user value resolve to
different servers depending on the entry point.

The current Lab command builder supplies the selected file through `--model` but
does not force an alias. llama.cpp consequently reports that model path as the
default `/v1/models` ID; its documented `--alias` option replaces the API identity.
Without a reserved alias, requiring an exact endpoint-reported ID would copy the
very filesystem identity this descriptor prohibits.

Subprocess liveness currently drives the running presentation while stdout and
stderr are discarded. A live process therefore supplies no evidence that the API is
healthy or that the selected model is available.

Console detected-server adoption already applies a discovered endpoint safely to
the current session and preserves a different configured endpoint, but its behavior
of filling missing configuration is not the Lab handoff contract. TASK-16473 already
requires a warning when an active Console endpoint will not survive restart. Lab
handoff must retain that session-only truth rather than imply persistence. TASK-16476
also protects a different configured endpoint from silent replacement.

TASK-26837 records a provider-setup path that can report a successful connection
test without a durable `api_settings` entry. **Make default** must not inherit or
normalize that false-success behavior.

ADR-002 otherwise limits OpenAI-compatible discovery to configured providers and
explicit persistence. ADR-025 owns managed and external GGUF authority, and ADR-095
separates active Console session settings, conversation-safe metadata, and durable
provider defaults. This decision extends those boundaries for the explicit Lab to
Console workflow rather than introducing a competing provider, artifact, or
conversation owner.

## Ownership and state transitions

Ownership is split as follows:

| Concern | Owner | Authority |
|---|---|---|
| Exact launch reservation, process identity, and stop | `server_lifecycle.py` | The exact claim and process generation |
| Verified connection target and readiness evidence | TASK-31201 app-scoped llama.cpp connection owner | `LlamaCppConnectionTarget` plus generation-fenced HTTP/model evidence |
| User-facing Lab state | Lab | Projection of runtime and connection owners into product state |
| Active-session adoption | Console | Whether the current Console session adopted the verified target |
| Durable provider endpoint/defaults | Settings/config | Configuration read by restart resolution |

For a managed-artifact launch, this decision preserves ADR-025's exact process-lease
lifetime. The artifact lease is acquired before spawn, transferred atomically to the
exact process claim, and retained until that claim proves its exact process dead. A
cancellation request or UI stop completion is not permission to release the lease
early. A stubborn process retains both its claim and lease, and a stale generation
cannot detach or release either. Lease release occurs only through identity-matched
claim settlement after confirmed process death.

The Lab never collapses these owners into one running flag. Runtime state can advance
from `unclaimed` through `reserved` and `process_alive` while connection state remains
`unchecked` or `checking`. Only current-generation health and exact-model evidence can
advance connection truth through `api_healthy` to `model_available`, permitting the
Lab to project `api_ready`. Console adoption may then project `console_connected` for
the active session. Failure, invalidation, or process death projects
`needs_attention` or an earlier honest state without treating prior evidence as
current.

The supported actions preserve those owners:

| Action | Owner and effect | Persistence |
|---|---|---|
| Start on this computer | Lab reserves and owns one exact process claim | None |
| Connect to existing server | Lab verifies one user-entered endpoint | None |
| Use in Console | Console applies provider, exact model, and base URL to the active session | Process-local session only |
| Make default | Full Settings commit path applies its existing checked-endpoint rules | Explicit config mutation |
| Stop server | Lab stops only its exact owned process claim | Does not alter Console or defaults |

**Use in Console** never calls the detected-server path that auto-fills missing
configuration. It never persists `base_url` in conversation metadata, consistent
with ADR-095. It preserves a different configured endpoint, labels the adopted
target **Session only**, and refreshes Console readiness in the same application
process. Console adoption grants no authority to stop the Lab process, and stopping
the server does not rewrite the adopted session or any configured default.

**Make default** opens or delegates to the full Settings commit path; Lab verification
is not permission to write configuration. It reports success only after the
normalized provider endpoint is durably present in the configuration layer read by
restart resolution. The commit path retains TASK-16473's distinction between
session-only and restart-safe endpoints and TASK-16476's protection against silently
replacing a different configured endpoint. TASK-26837's missing-`api_settings`
success state is an unresolved defect that the path must prevent or surface, never
accepted behavior.

## Default and compatibility policy

Absent values resolve in this order:

```text
explicit user-entered or launch endpoint
  -> exact current-session target
  -> configured provider endpoint
  -> canonical llama.cpp absent-value default http://127.0.0.1:8080
```

New Lab launches default to loopback `127.0.0.1:8080` only when the user supplied no
value. Existing explicit `8001`, `9099`, LAN, HTTPS, and reverse-proxy-prefix
endpoints remain valid and are neither migrated nor rewritten. Endpoint parsing,
suffix removal, safe display, and equality use `resolve_provider_endpoint` and
`canonical_connection_identity` rather than raw string comparison.

A user-entered existing-server check is an explicit, exact-endpoint exception to
ADR-002's configured-provider-only discovery rule. It authorizes only the endpoint
the user entered; it does not enable ambient LAN scanning or background remote
discovery. Discovery and verification do not persist model IDs or endpoints. Any
durable change remains an explicit Settings action under ADR-002 and this decision.

## Privacy and observability boundary

The narrow display allowlist governs every projection out of Lab into cross-surface
UI, Console, app-global metadata, or application/unrestricted logs. Those projections
may carry or show only canonical provider identity, the canonical credential-free
endpoint, an accepted endpoint-reported model ID, coarse lifecycle state, and a
bounded failure category. App-global state and logs must not retain raw executable
or model paths, credentials, raw command arguments, or unbounded process output.
Query strings and fragments never enter `LlamaCppConnectionTarget` or Console
conversation metadata.

Lab itself may retain and render its surface-owned executable and GGUF selections,
the owned process PID, and expert launch configuration. User-entered arguments may
appear in their owning editor; every derived command or argument presentation is
redacted. TASK-31206 may add a bounded, sanitized runtime diagnostic tail and
sanitized copy action within Lab. The same bound and redaction apply before render or
copy, including suppression of any rejected endpoint model ID.

These Lab-local details never enter `LlamaCppConnectionTarget`, Console, active
session or conversation metadata, app-global connection metadata, or application
logs. Lab's bounded diagnostic buffer is not an unrestricted application log sink.

## Alternatives Considered

| Alternative | Why rejected |
|---|---|
| Treat process liveness as readiness | A process can be alive while loading, bound to the wrong endpoint, unable to answer the API, or serving a different model. |
| Auto-write the Lab endpoint into provider configuration | It confuses verification with consent, can overwrite a user's durable endpoint, and makes session-only adoption look restart-safe. |
| Reuse `_apply_detected_local_server` unchanged for Lab handoff | That path may fill missing configuration and update defaults; Lab's **Use in Console** action is explicitly session-only. |
| Persist the complete Lab launch command in Console conversation metadata | It leaks paths and arguments, violates ADR-095's allowlist, and makes runtime launch details conversation-owned. |
| Retain three context-specific defaults | The same absent value would continue resolving differently across Lab, discovery, and Console. |
| Generalize all local runtimes before the llama.cpp path is proven | Other runtimes have different lifecycle, endpoint, and model contracts; early generalization would weaken the llama.cpp contract. |

## Consequences

- TASK-31201 implements the app-scoped connection owner, generation-fenced
  verification, and active Console adoption.
- TASK-31202 and TASK-31203 project the ownership and readiness vocabulary into
  guided onboarding and a keyboard-efficient, narrow-width Lab experience.
- TASK-31204 and TASK-31205 retain launch configuration separately through current
  versus next configuration, restart-last behavior, and durable named profiles.
- TASK-31206 adds bounded, sanitized Lab-local diagnostics without expanding the
  handoff descriptor.
- Lab, Console, and Settings must display whether a value is runtime-only,
  session-only, or durable instead of deriving persistence from successful probing.
- Existing configured endpoints remain compatible; the unified default affects only
  absent values, so no endpoint migration or synthesized configuration is required.
- Readiness now requires network evidence and exact model identity, adding probe,
  cancellation, and stale-result handling to the local launch workflow.
- Lab-owned launches now carry one stable Chatbook-reserved model alias, while an
  existing server must expose a non-path-identifying model ID before handoff.

## Rollback plan

Disable the new Lab handoff and fall back to existing Console Settings/discovery.
Retain every existing explicit configuration value and do not synthesize migrations.
Disabling the handoff also disables its app-scoped connection projection; it does not
stop unrelated processes, alter active Console sessions, or rewrite defaults.

## Verification obligations

- Contract tests must prove the exact default-resolution order and preserve explicit
  `8001`, `9099`, LAN, HTTPS, and reverse-proxy-prefix endpoints.
- Readiness tests must prove that `process_alive` without current-generation health
  and `/v1/models` evidence cannot expose `api_ready` or **Use in Console**.
- Probe settlement tests must invalidate evidence after cancellation, process death,
  target edit, model change, screen recomposition, or a newer
  `verification_generation`.
- Managed-artifact lifecycle tests must prove lease acquisition before spawn, atomic
  transfer to the exact process claim, retention until that claim proves the exact
  process dead, retention for a stubborn process, and rejection of cancellation, UI
  stop completion, or stale-generation attempts to release the lease early.
- Adoption tests must prove **Use in Console** is session-only, does not call the
  detected-server persistence path, preserves a different configured endpoint, omits
  `base_url` from conversation metadata, and refreshes readiness in process.
- Persistence tests must prove **Make default** reports success only after the
  normalized endpoint exists in the durable `api_settings` layer consumed on
  restart, while preserving TASK-16473, TASK-16476, and TASK-26837 protections.
- Lab launch-construction tests must prove starts and restarts using distinct GGUF
  paths each emit exactly one `--alias chatbook-llamacpp`, and that separated and
  equals-attached `-a` and `--alias` raw-argument forms are rejected before they can
  replace or duplicate it.
- Existing-server tests must prove path-identifying IDs fail closed before selector,
  descriptor, display, copy, log, or adoption; the recovery names `--alias` without
  echoing a sentinel path. The same tests must accept ordinary namespace-style IDs
  such as `owner/model` and preserve accepted IDs exactly.
- Privacy tests must prove the descriptor, app-global state, logs, and Console
  metadata exclude executable/model paths, credentials, raw commands, and unbounded
  process output while Lab retains only its explicitly permitted bounded local
  diagnostics. They must also prove that query strings and fragments never enter
  `LlamaCppConnectionTarget` or Console conversation metadata.

| Contract | Required future evidence |
|---|---|
| Canonical endpoint | Pure normalization/default-precedence tests in `Tests/Chat/test_provider_endpoint_contract.py` and Console settings tests |
| Process versus readiness | Lifecycle plus real loopback HTTP tests proving live-process/not-ready and model-ready transitions |
| Stale-result fencing | Generation replacement tests for process exit, model edit, endpoint edit, cancellation, and recomposition |
| Console adoption | Mounted Lab-to-Console test proving exact provider/base URL/model apply without restart |
| Persistence boundary | Regression test proving Use in Console does not write config and Make default preserves unrelated or newer fields |
| Managed model privacy | Tests proving no filesystem path enters the descriptor, rendered authority text, app-global metadata, or Console settings |
| Compact UX | Production-stylesheet 80x24, 100x30, and 120x40 compositor/focus tests |
| Live qualification | Scratch-profile run against a real llama-server, with default-profile fingerprints checked before cleanup |

## Links

- [ADR-002: OpenAI-compatible model discovery](002-openai-compatible-model-discovery.md)
- [ADR-025: Shared artifacts and runtime routing](025-shared-stt-artifacts-and-runtime-routing.md)
- [ADR-095: Console generation settings ownership](095-conversation-owned-console-generation-settings.md)
- [TASK-16473: Warn when a Console endpoint will not survive restart](../tasks/task-16473%20-%20Console-warn-when-a-saved-provider-endpoint-will-not-survive-restart.md)
- [TASK-16476: Preserve a configured endpoint during server adoption](../tasks/task-16476%20-%20Console-server-adoption-must-not-clobber-a-configured-endpoint.md)
- [TASK-26837: Provider setup can omit durable API settings](../tasks/task-26837%20-%20Provider-setup-can-report-a-successful-connection-test-yet-write-no-api_settings-block.md)
- [llama.cpp HTTP server README: model IDs and `--alias`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#get-v1models-openai-compatible-model-info-api)
