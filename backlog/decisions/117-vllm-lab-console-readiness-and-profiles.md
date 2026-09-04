# ADR-117: Separate vLLM launch, readiness, profiles, and Console adoption

Status: Accepted
Date: 2026-09-03
Related Tasks: TASK-31282, TASK-31283, TASK-31284, TASK-31285, TASK-31286, TASK-31287
Extends: ADR-002, ADR-006, ADR-095
Related: ADR-114

## Decision

Chatbook will treat vLLM setup as a verified connection workflow rather than a
subprocess-launch form. Four owners remain separate:

1. Lab owns environment selection, launch intent, the exact process claim, and Stop.
2. An app-scoped vLLM connection owner holds generation-fenced readiness evidence.
3. Console owns active-session provider/model adoption.
4. Settings/config owns durable provider endpoints and new-chat defaults.

No successful action at one boundary implies success at another. In particular, a
live process is not an API-ready model, verification is not permission to persist,
and Console adoption is not authority to stop Lab's process.

### Verified connection descriptor

Only a current, verified target may cross from Lab into Console:

```text
VllmConnectionTarget
  provider_key: "vllm"
  api_url: canonical Chatbook-persisted chat-completions URL
  model_id: exact admissible ID returned by /v1/models
  runtime_owner: lab_process | external_server
  verification_generation: process-local opaque generation
  credential_source: configured | none
```

`api_url` is the `persisted_endpoint` returned by
`resolve_provider_endpoint("vllm", value)`, normally ending in
`/v1/chat/completions`. Query strings, fragments, userinfo, and raw credentials are
rejected before descriptor creation. `verification_generation` is never persisted.
`credential_source` is a non-secret classification; the descriptor never contains a
credential or credential reference value.

For a Lab-owned launch, Chatbook supplies exactly one reserved served-model alias:

```text
--served-model-name chatbook-vllm
```

The alias is not derived from a Hugging Face repository, local directory, filename,
or profile name. Raw arguments cannot provide `--served-model-name`, `--model`,
`--host`, `--port`, `--api-key`, `--hf-token`, or equivalent attached-value forms.
The launch is model-ready only when `/v1/models` returns the exact reserved alias.

For an existing server, endpoint model IDs are untrusted. The admissibility and
path-identifying rules from ADR-114 apply without cleaning or truncation: the ID must
be a canonical printable string of 1 through 120 Unicode code points, contain no
control, format, surrogate, line-separator, or paragraph-separator characters, and
must not carry an unambiguous filesystem marker. Namespace IDs such as
`organization/model` remain valid. Rejected IDs never enter UI, copy payloads,
descriptors, Console state, application logs, or profile storage.

### Launch modes and environment preflight

The vLLM pane has two explicit modes:

- **Start on this computer** selects a Python environment and model source, then
  launches one Chatbook-owned process.
- **Connect to existing server** verifies one explicitly entered endpoint. It never
  performs ambient LAN discovery or starts/stops that process.

The local launch path resolves the `vllm` console executable installed beside the
selected environment and invokes the public command:

```text
vllm serve <model> --host <bind> --port <port>
  --served-model-name chatbook-vllm <approved options>
```

Chatbook does not depend on private `vllm.entrypoints.*` module paths. Preflight
fails closed when the chosen environment cannot resolve both its Python executable
and matching `vllm` console executable, cannot report an installed vLLM version, or
cannot import `vllm`. The UI gives install/reselect recovery but does not install
packages automatically.

Model source is explicit:

- **Hugging Face repository** accepts a validated repository identifier.
- **Local model directory** uses a directory picker plus path validation.

The selected source remains Lab-local. A local directory path never becomes the
served model ID or crosses into Console. Preflight also validates port availability,
bind-address syntax, managed/raw argument conflicts, and parseability before launch.
Hardware suitability is reported as detected/unknown rather than guessed; lack of a
supported accelerator is not fabricated from platform name alone.

The local launch default is an actual value, not placeholder behavior:

```text
bind address = 127.0.0.1
port = 8000
```

`0.0.0.0`, `::`, and non-loopback binds display a persistent **Network exposed**
warning before Start. For client use, wildcard IPv4 maps to `127.0.0.1` and wildcard
IPv6 maps to `::1`; the bind value remains visible only in Lab. Existing-server URLs
may use HTTP or HTTPS and keep accepted reverse-proxy prefixes.

Credentials are never entered as raw launch arguments. A Lab-owned process may
inherit vLLM's established environment-based authentication, including
`VLLM_API_KEY`; Lab records only whether a configured credential source exists.
Existing-server verification uses the configured vLLM credential-resolution path.
No credential is displayed, copied, saved in a launch profile, or written to logs.

### Readiness and generation fencing

The owners expose three distinct vocabularies:

```text
Runtime truth: unclaimed -> reserved -> process_alive -> process_dead
Connection truth: unchecked -> checking -> health_ok -> model_available -> stale_or_failed
Product state: not_configured | checking | launching | loading_model |
               api_ready | console_connected | stopping | needs_attention
```

For a Lab-owned process, `api_ready` requires all of the following for the same
generation:

1. the exact owned process claim is alive;
2. bounded `GET /health` succeeds;
3. bounded `GET /v1/models` succeeds;
4. the response includes exact `chatbook-vllm` identity.

For an existing server, steps 2 and 3 apply and the user selects one admissible
chat-capable returned model. A process claim is neither created nor required.

Every preflight, launch, probe, retry, and handoff captures the exact opaque
generation. Environment, model, bind, port, endpoint, profile, or mode changes;
process death; cancellation; screen replacement; or a newer operation invalidates
older evidence. A stale result may not change product state, enable Console actions,
or publish a connection target.

Probe retry is bounded and cancellable. `loading_model` is shown while the process
is alive but health/model evidence is incomplete. Timeout is a retryable
`needs_attention` state, not proof the process died. Stop settles only the exact
owned process claim and invalidates its connection generation.

### Console adoption and durable scope

After `api_ready`, Lab exposes two distinct actions:

- **Use in Console** applies provider `vllm`, the exact descriptor model, and the
  descriptor API URL to the current Console session. It performs no config write.
- **Make default for new chats** opens/delegates to the established full Console
  Settings default transaction with the verified draft prefilled. It succeeds only
  after the existing provider/default persistence contract confirms the durable
  write.

Ordinary session adoption reuses Console's provider-aware rebase/application owner
from ADR-095. It does not call an adoption path that silently fills missing durable
configuration. It preserves a different configured endpoint, labels the active
connection **Session only**, refreshes in-process readiness, and never stores
`api_url` in conversation metadata.

Stopping a Lab-owned server does not rewrite Console or defaults. Console then shows
its normal endpoint-unavailable recovery. Reverification can produce a newer target;
an older ready target cannot be reused after its generation becomes stale.

### Current server and next-launch intent

An active process has an immutable app-scoped snapshot:

```text
VllmLaunchSnapshot
  generation
  profile_id | none
  environment_display
  model_source_kind
  model_source_display
  bind_address
  port
  structured_options
  redacted_argument_summary
```

The snapshot is captured before launch and belongs to the exact process claim until
that claim proves the process dead. It is not reconstructed from mutable widgets.
While active, the form represents **Next restart configuration**. A semantic
fingerprint determines whether the draft differs from the current snapshot. The UI
shows a dirty marker and enables **Restart with draft** only when the exact current
process can be stopped and a valid preflight for the draft remains current.

Restart is a two-generation transaction: stop/settle the old exact process first,
then reserve a new claim and launch only if the draft fingerprint still matches. A
failed or stubborn stop never spawns a second process.

### Device-local launch profiles

Named vLLM launch profiles are device-local convenience, not Console provider
configuration, model artifacts, or syncable user content. They are stored in one
versioned JSON document beneath the active profile's device-local application data
directory and written with the repository's atomic replace and restrictive-file
patterns. No database migration is required.

Version 1 stores at most 32 profiles:

The exact V1 names below incorporate the accepted TASK-31286 implementation plan.
They replace the earlier design-draft spellings in this ADR so the durable contract
uses the same field vocabulary as `VllmLaunchDraft`. Revision belongs to the whole
CAS document, so V1 deliberately omits a per-profile `updated_at`; this keeps the
schema minimal and avoids timestamp-only write conflicts.

```text
VllmLaunchProfileV1
  profile_id: stable UUID
  name: canonical printable 1..120 Unicode code points
  python_environment: local absolute path or safe bare executable name
  model_source: hugging_face | local_directory
  model_value: valid namespaced Hugging Face repository ID or safe absolute local path
  bind_address
  port
  dtype: auto | half | float16 | bfloat16 | float32
  tensor_parallel_size: positive integer | none
  maximum_model_length: positive integer | none
  gpu_memory_utilization: finite float in (0, 1] | none
  trust_remote_code: boolean

VllmProfileDocumentV1
  version: exactly 1
  revision: non-negative integer
  selected_profile_id: UUID of one contained profile
  profiles: array of 1..32 exact V1 profile objects
```

These are exact key sets: unknown or missing keys are corruption, not extension
points. A Hugging Face value must pass repository-ID validation. A local-directory
value may name a nonexistent directory so the user can repair it later, but it must
be an absolute local path shape without option, URL/userinfo, control, or traversal
syntax. The document does not store credentials,
environment-variable values, API keys, Hugging Face tokens, unrestricted raw
arguments, command strings, probe responses, process IDs, readiness evidence, or
Console adoption state. Local paths are permitted only in this device-local Lab
owner and must not appear in app-global metadata or application logs.

Create, rename, duplicate, and delete are explicit foreground actions. Profile
selection updates the editable draft; it never restarts a process. A missing
environment or local model path leaves the profile selected but marks it **Needs
setup**, preserves the stored value for repair, and blocks Start. Corrupt/future
profile documents fail closed, remain recoverable, and are not overwritten by an
older reader without explicit reset.

Raw advanced arguments remain launch-draft-only and are never written to a profile.
The UI states this adjacent to the editor. Common durable expert needs use the
structured fields above.

### Lab information architecture and keyboard ownership

The vLLM pane follows this reading and focus order:

1. readiness and authority summary;
2. launch/connect mode;
3. profile and basic environment/model setup;
4. network configuration;
5. Check setup and Start/Stop/Restart;
6. collapsed Advanced options;
7. compact Activity and recovery;
8. verified Console actions.

The fixed empty 15-row log is removed. Activity shows bounded state transitions and
safe recovery. Detailed diagnostics are secondary, bounded, redacted, and Lab-local.

At compact widths labels, inputs, and actions stack. At 100 columns or below the
Inspector auto-collapses for this workflow; at 80 columns the provider catalog may
also collapse after vLLM selection. Every focusable descendant must remain within
its owning pane at 80x24, 100x30, and 120x40.

Tab traversal is constrained to the active provider body. Lifecycle transitions
move focus from Start to Stop, from Stop to Start, and to Use in Console when the
user initiated a successful verification. Brackets retain their Lab-level meaning;
provider selection uses the catalog with arrows and Enter. Hidden digit/provider
shortcuts that cover only part of the catalog are removed rather than advertised.

## Context

The current vLLM pane launches
`python -m vllm.entrypoints.api_server`, a private/legacy module path that is not the
current documented server command. It accepts a Python string, model string, bind,
port, unrestricted arguments, and displays a mostly empty `RichLog`.

Current status is derived from process liveness. The child process stdout/stderr are
discarded, so Lab can say running while the model is loading or the API cannot serve
requests, and the visible log cannot explain most failures.

Console already supports vLLM endpoint/model discovery and ADR-095 session/default
scope, but Lab does not call those paths. Custom Lab host, port, and model values must
be reconstructed manually in Console.

Official vLLM documentation defines `vllm serve`, `/health`, `/v1/models`,
OpenAI-compatible `/v1/chat/completions`, environment-based API-key support, and
`--served-model-name`. It also specifies that the selected model path/name becomes
the served identity when no served-model name is supplied. Reserving a safe alias is
therefore necessary to keep local directory paths out of Console.

ADR-114 established the analogous separation for llama.cpp but explicitly rejected
prematurely generalizing all local runtimes. This decision applies that proven shape
to vLLM's distinct environment, command, model-source, authentication, and profile
requirements without amending ADR-114.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Treat process liveness as ready | vLLM can remain alive while loading or fail to expose the requested model. |
| Keep the private Python module entry point | It couples Chatbook to vLLM internals while the public `vllm serve` command is documented. |
| Copy the model path/repository directly into Console | Local paths leak device identity, and endpoint truth must come from `/v1/models`. |
| Persist Lab endpoint/model automatically | Verification is not consent to change durable provider settings. |
| Store profiles inside `api_settings.vllm` | Launch environment and local paths are device-local concerns, not provider request configuration. |
| Save unrestricted raw arguments in profiles | Arguments can contain secrets and unstable provider-owned flags. Structured non-secret options cover the durable case. |
| Stream all subprocess output into the UI | Raw output can contain paths, tokens, model metadata, or unbounded content. Bounded classifications preserve recovery without creating a new log sink. |
| Create one generic local-server framework first | vLLM and llama.cpp have materially different launch and model semantics; a shared abstraction should emerge only from implemented contracts. |

## Consequences

- TASK-31283 adds the guided launch/connect setup and preflight boundary.
- TASK-31284 adds the app-scoped target owner, snapshots, generation fencing, and
  API/model readiness.
- TASK-31285 connects verified targets to Console's existing session/default owners.
- TASK-31286 adds device-local structured profiles and current-versus-next restart.
- TASK-31287 completes compact layout, focus containment, and production-stylesheet
  verification.
- Existing explicit vLLM provider endpoints remain unchanged. No migration rewrites
  `api_settings.vllm`.
- The local launch command changes to the current public CLI and fails with recovery
  when the matching executable is not present.
- Readiness adds bounded HTTP work and cancellation/invalidation handling.
- Local model paths may be retained only by the device-local Lab profile store and
  current process snapshot.

## Rollback plan

Feature-gate the redesigned vLLM connection workflow and restore the legacy Lab
launcher. Disabling it removes profile editing, verified handoff, and the app-scoped
connection projection but does not stop unrelated processes, rewrite provider
configuration, or change existing Console sessions. Keep the profile JSON document
untouched so rollback is non-destructive; an older build ignores it.

## Verification obligations

- Command tests prove the public CLI shape, exactly one reserved served-model alias,
  and rejection of every managed or secret-bearing raw-argument override form.
- Preflight tests cover missing/non-executable Python, missing/mismatched vLLM CLI,
  import failure, invalid repository/directory, busy port, wildcard/network binds,
  and paths containing spaces or Unicode.
- Loopback HTTP tests prove process-alive/not-ready, health-ready/model-missing,
  exact-model-ready, timeout, and recovery.
- Generation tests invalidate results after every launch field, mode, profile,
  process, cancellation, recomposition, and newer-operation transition.
- Privacy tests prove no credential, local path, raw command, raw child output, or
  rejected model ID enters the descriptor, application logs, Console metadata, or
  copied diagnostics.
- Console tests prove exact in-process session adoption without config writes and
  durable default delegation without silent endpoint replacement.
- Profile tests cover atomic round trips, limits, corruption, future versions,
  missing local dependencies, rename/duplicate/delete, and exclusion of raw args and
  secrets.
- Production-stylesheet Textual tests cover descendant containment and complete
  keyboard traversal at 80x24, 100x30, and 120x40 for first-run, loading, ready,
  failure, dirty-restart, and Console-handoff states.
- Live qualification uses a scratch profile and a real compatible vLLM server when
  the host has an eligible environment; otherwise the PR records the exact missing
  capability and retains loopback protocol evidence without claiming real-server
  coverage.

## Links

- [Design specification](../../Docs/superpowers/specs/2026-09-03-vllm-lab-console-complete-redesign.md)
- [ADR-002: OpenAI-compatible model discovery](002-openai-compatible-model-discovery.md)
- [ADR-006: Provider-aware generation settings](006-provider-aware-generation-settings.md)
- [ADR-095: Console generation settings ownership](095-conversation-owned-console-generation-settings.md)
- [ADR-114: llama.cpp Lab-to-Console connection authority](114-llamacpp-lab-console-connection-authority.md)
- [Official vLLM online serving documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)
- [Official `vllm serve` CLI reference](https://docs.vllm.ai/en/latest/cli/serve/)

## Numbering provenance

This decision was originally added as ADR-115 by
`ffc4f9d8f8343169097dcac40d3ba4ed0a2177c0`. During branch integration,
current `origin/dev` already shipped the unrelated
`115-personas-demand-mounted-center-views.md` at add commit
`2516735cfd27df249ab45e96c96f15b8aee35d15`. The unmerged vLLM decision moved
to collision-free ADR-117, and every live vLLM reference moved with it.
