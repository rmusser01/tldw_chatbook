# vLLM Lab-to-Console Complete Redesign

**Date:** 2026-09-03

**Status:** Proposed for user review

**Tasks:** TASK-31213, TASK-31214, TASK-31215, TASK-31217, TASK-31219, TASK-31221

**Decision:** [ADR-115](../../../backlog/decisions/115-vllm-lab-console-readiness-and-profiles.md)

## Summary

Redesign Lab's vLLM pane as an end-to-end operating workflow. A first-time user
can choose an environment and model, understand what is missing, start vLLM, wait
through truthful model loading, and move the verified endpoint into Console. An
experienced user can restore profiles, distinguish the running server from a
future restart draft, diagnose bounded failures, and restart without reconstructing
the command.

The finish line is not a child process. It is a current-generation vLLM endpoint
whose health and served model are verified and can be adopted into Console with an
explicit session or durable scope.

## Design classification and direction

This is an architectural extension of an established **Operate** surface. It keeps
Chatbook's existing terminal-native Lab visual system, navigation, density, semantic
colors, and control vocabulary. It changes information hierarchy, workflow, state,
and ownership inside the vLLM pane; it does not create a new visual identity or
rewrite `DESIGN.md`.

Direction contract:

- **Thesis:** a calm launch console that always distinguishes setup, process,
  connection, and Chatbook-use state; it refuses the generic form-plus-log layout.
- **Own world:** compact Textual sections, explicit text state, restrained semantic
  emphasis, and one primary action matching the current phase.
- **Story:** resolve missing prerequisites, launch or connect, prove the model,
  then use it in Console; returning operators compare current and next state.
- **First viewport:** readiness precedes configuration; basic setup and the current
  primary action remain above the fold; Advanced and diagnostics are secondary.
- **Finish:** every focusable descendant is visible at 80x24, 100x30, and 120x40,
  and every completion claim is backed by current-generation evidence.

## Users and jobs

### Jordan: first-time local-model user

Jordan knows they want to use a local model in Chatbook but may not know the
difference between a Python environment, vLLM installation, Hugging Face repository,
local model directory, bind address, and client endpoint.

Jordan's job is:

> Tell me what I need, help me satisfy it without leaving a broken draft, and make
> the model available in Console when it is actually ready.

### Alex: experienced operator

Alex manages multiple environments and models, knows vLLM flags, changes tensor and
memory settings, and repeatedly starts, stops, diagnoses, and switches configurations.

Alex's job is:

> Restore my known setup, show me exactly what is running versus what I changed,
> let me restart safely, and keep Console synchronized without hiding persistence.

### Riley: keyboard and security-conscious operator

Riley expects every action to remain reachable in narrow terminals, wants public
network exposure called out, and does not want secrets or local paths copied into
conversation state or logs.

## Goals

- Make first-run prerequisites and recovery visible before Start.
- Support explicit local launch and existing-server connection modes.
- Treat process liveness, API health, model availability, Console adoption, and
  durable defaults as separate states.
- Complete the Lab-to-Console handoff without retyping endpoint or model.
- Restore reusable, device-local, non-secret launch profiles.
- Separate immutable current-server state from editable next-restart intent.
- Provide bounded, sanitized activity and recovery without creating a raw log sink.
- Remain usable and focus-correct at 80x24, 100x30, and 120x40.

## Non-goals

- Installing Python, vLLM, accelerator drivers, or models automatically.
- Ambient LAN scanning or automatic connection to unconfigured remote endpoints.
- Replacing Console's provider settings/default persistence contracts.
- Persisting unrestricted vLLM command lines, credentials, raw arguments, or output.
- Managing multiple concurrent Chatbook-owned vLLM processes.
- Supporting dynamic LoRA loading, pooling-only, embedding-only, audio, or Omni
  workflows in this chat-oriented pane.
- Generalizing every local runtime behind one framework before the vLLM contract is
  implemented and measured.

## Experience architecture

### Stable sections

The pane has seven stable sections in reading order:

1. **Readiness** — current product state, endpoint/model when safe, and blockers.
2. **Mode** — Start on this computer or Connect to existing server.
3. **Setup** — profile, environment, and source-specific model controls.
4. **Network** — bind/client endpoint and exposure meaning.
5. **Actions** — Check setup plus the single phase-appropriate primary action.
6. **Advanced and Activity** — progressive disclosure and bounded recovery.
7. **Use in Console** — available only for a current verified generation.

The right Inspector becomes contextual when visible. It shows selected profile,
process ownership, safe current endpoint/model, persistence scope, and the next
available action. It does not repeat six generic server rows while vLLM is selected.

### Progressive disclosure

First paint shows Readiness, Mode, Setup, Network, and Check setup. The following are
collapsed or conditional:

- Structured expert options are under **Advanced**.
- Raw launch-only arguments are nested under **Advanced arguments**.
- Diagnostics are under **Activity details** and appear automatically on failure.
- Current server appears only when a launch claim exists or a verified external
  endpoint is selected.
- Next restart appears only while an owned process is active.
- Console actions appear only after verification.

## First-time workflow

### 1. Arrival

The initial title is `vLLM · Setup incomplete`, never the destination-global
`Ready`. The readiness checklist contains:

```text
○ Environment
○ vLLM installation
○ Model
○ Network
```

Each failed or missing row provides one next action. Start is disabled with a visible
summary: `Choose a model and check setup first.`

### 2. Mode selection

`Start on this computer` is the initial mode. Switching modes preserves each mode's
draft independently and invalidates prior readiness evidence.

`Connect to existing server` replaces environment/model-source/bind fields with:

- Server URL
- `Use configured vLLM credentials` status
- Check connection
- Returned model selector

The URL is probed only after the explicit user action.

### 3. Environment

The control label is **Python environment**, with value plus Browse. Check setup
reports the resolved Python version, matching `vllm` executable, and installed vLLM
version. The display never prints an unbounded path outside the Lab-owned setup area.

States and recovery:

| State | Copy | Action |
|---|---|---|
| Blank/unresolved | `Choose the Python environment that contains vLLM.` | Browse |
| Python missing | `Python cannot be run from this location.` | Choose another |
| vLLM missing | `vLLM is not installed in this environment.` | Show install command |
| CLI mismatch | `The environment's vLLM command is unavailable.` | Choose another |
| Ready | `Python 3.x · vLLM x.y` | Change |

The install command is copyable factual guidance such as
`python -m pip install "tldw_chatbook[local_vllm]"`; Chatbook does not execute it.

### 4. Model source

Source is a two-option selector:

```text
Hugging Face repository | Local model directory
```

Repository mode accepts `organization/model`-style identity and explains that
downloads and gated access remain vLLM/Hugging Face concerns. Local mode uses a
directory picker and explains that the directory stays on this device.

The UI never asks for a GGUF file and never titles the picker `checkpoint or GGUF`.

### 5. Network

The initial fields contain real values:

```text
Bind address  127.0.0.1
Port          8000
```

Loopback copy reads `Only this computer can connect.` Wildcard or non-loopback copy
reads `Network exposed` and remains visible through launch and readiness. A warning
does not silently change the user's explicit bind value.

### 6. Check setup

Check setup runs environment, model, port, argument, and exposure checks without
starting a server. Results update one checklist in place and retain every draft
value. The first failed field receives focus only when the user initiated the check;
background invalidation does not steal focus.

When all blocking checks pass, the primary action becomes **Start vLLM**.

### 7. Launch and loading

Start captures the validated semantic draft, reserves one exact process generation,
and uses the resolved environment's public `vllm serve` command. Focus moves to Stop.

The Activity row advances truthfully:

```text
Launching process…
Loading model… 00:47
API health confirmed; checking served model…
Ready at http://127.0.0.1:8000/v1
```

Elapsed time is display-only and not evidence. `Ready` requires current-generation
health and model-list evidence. If the process is alive after timeout, the state is
`Needs attention · Model is still unavailable`, with Retry check and Stop.

### 8. Console adoption

Ready renders:

```text
✓ Ready at http://127.0.0.1:8000/v1
  Served model: chatbook-vllm

[Use in Console]  [Make default for new chats]
```

Use in Console applies to the active chat and navigates to Console only after the
originating session accepts the target. Copy reads `Session only · restart uses your
saved provider endpoint.`

Make default delegates to the existing full settings transaction with provider,
model, and endpoint prefilled. Lab does not report success on its behalf.

## Experienced workflow

### Restore and compare

On re-entry, Lab restores the last selected provider and vLLM profile. It shows:

- **Current server** — immutable values captured by the exact active process.
- **Next restart configuration** — editable profile/draft values.

Changing a draft value produces `Modified for next restart` and a field-level marker.
It never visually implies the live process changed.

### Profiles

The profile selector supports:

- New profile
- Save changes
- Save as
- Rename
- Duplicate
- Delete with confirmation

The initial profile is `Default vLLM`. Profile names are unique under Unicode
casefold plus canonical whitespace. A duplicate is named `<name> copy`, with a
bounded numeric suffix when required.

Profiles persist environment, model source, network, and structured expert options.
The raw arguments editor states: `Launch only · not saved in profiles.` Selecting a
profile never starts, stops, or restarts a server.

### Structured expert options

The first durable expert fields are:

- dtype
- tensor parallel size
- maximum model length
- GPU memory utilization
- trust remote code

`trust remote code` defaults off and includes consequence copy. Blank numeric values
inherit vLLM behavior. Validation occurs before the launch command is built.

### Restart with draft

When the next-launch fingerprint differs and preflight is current, the action reads
**Restart with draft**. Confirmation summarizes safe changes without echoing local
paths or raw arguments outside their editors.

Restart must prove the exact old process dead before claiming a new generation. A
stubborn process keeps Stop/recovery visible and never permits a second launch.

## State model

| Product state | Required evidence | Primary action | Console action |
|---|---|---|---|
| Not configured | Missing required draft | Check setup disabled or first recovery | Hidden |
| Checking | Current preflight worker | Cancel check | Hidden |
| Ready to start | Current successful preflight | Start vLLM | Hidden |
| Launching | Exact claim reserved, process not yet proven | Stop starting | Hidden |
| Loading model | Exact process alive, readiness incomplete | Stop | Hidden |
| API ready | Current health + exact model evidence | Stop | Enabled |
| Console connected | Current target adopted by active session | Stop | `Open Console` |
| Stopping | Exact stop settlement in progress | Wait | Disabled |
| Needs attention | Current bounded failure | Retry/repair/Stop by category | Hidden or disabled |

Connection state is invalidated by any semantic target change. Editing only a profile
name does not invalidate the running connection; changing profile launch fields does.

## Data and service boundaries

### Focused modules

Implementation should introduce focused owners rather than further expanding the
large management window:

```text
UI/LLM_Management_Window.py
  Compose the vLLM pane and project service state into widgets.

UI/LLM_Management/vllm_setup.py
  Pure draft, preflight, snapshot, target, state, and command-building contracts.

UI/LLM_Management/vllm_profiles.py
  Versioned device-local profile validation and atomic repository.

Event_Handlers/LLM_Management_Events/llm_management_events_vllm.py
  Translate widget messages into service operations; no duplicated validation.

UI/Screens/llm_screen.py
  Own app/screen-level workers, generation invalidation, Inspector projection,
  and Console navigation/adoption coordination.
```

Exact filenames may be refined in the implementation plan after verifying existing
package conventions, but responsibilities may not collapse back into one UI module.

### Core types

The implementation plan must define typed immutable values corresponding to:

- `VllmModelSource`
- `VllmLaunchDraft`
- `VllmPreflightResult`
- `VllmLaunchSnapshot`
- `VllmConnectionTarget`
- `VllmReadinessState`
- `VllmLaunchProfileV1`

Drafts are mutable only at the UI/controller boundary. Worker inputs and settled
results are immutable snapshots carrying generation plus semantic fingerprint.

### Profile persistence

One versioned JSON document lives beneath the active profile's device-local data
directory. Writes use atomic replacement, validate before replacing, and preserve a
future version rather than overwriting it. The store is capped at 32 profiles and is
never synchronized to tldw_server.

Unrestricted raw arguments, credentials, environment-variable values, process IDs,
probe output, and Console state are excluded by schema, not merely by UI policy.

## Validation and error recovery

### Bounded failure taxonomy

| Code | User copy | Recovery |
|---|---|---|
| `python_unavailable` | Python cannot be run from this location. | Browse environment |
| `vllm_unavailable` | vLLM is not installed in this environment. | Copy install guidance |
| `vllm_cli_mismatch` | The environment's vLLM command is unavailable. | Choose environment |
| `model_required` | Choose a model source. | Focus model |
| `model_invalid` | This repository ID or directory is not usable. | Repair value |
| `port_busy` | Port N is already in use. | Change port or check existing server |
| `network_exposed` | Network exposed at this bind address. | Confirm or choose loopback |
| `arguments_conflict` | Advanced arguments duplicate a managed setting. | Remove named flag |
| `process_exited` | vLLM exited before the API became ready. | Review safe details/retry |
| `health_timeout` | The process is alive, but the API is not ready yet. | Retry check or Stop |
| `model_missing` | The API is healthy, but the expected chat model is unavailable. | Retry/restart |
| `credential_required` | The server requires configured vLLM credentials. | Open Settings |
| `profile_unavailable` | This saved profile needs repair on this computer. | Edit profile |
| `profile_store_unavailable` | Profiles could not be read safely. | Retry/export/reset review |

Exception messages, raw HTTP bodies, local paths, rejected model IDs, and process
output are never interpolated into global notifications or application logs.

### Activity details

Activity retains a bounded in-memory sequence of allowlisted events for the current
operation. Copy diagnostics includes versions, safe state codes, endpoint host/port,
elapsed ranges, and exit code when available. It excludes paths, commands, model
source values, credentials, HTTP payloads, and raw stdout/stderr.

## Responsive and keyboard contract

### 120x40 and wider

- Catalog, central work area, and contextual Inspector may remain visible.
- Setup fields can use compact horizontal label/control rows.
- Readiness, basic setup, and current primary action remain above the fold.

### 100x30

- Inspector auto-collapses.
- Label/control/action groups stack when their measured descendants do not fit.
- A visible fold cue appears when Activity or Console actions are below the viewport.

### 80x24

- Catalog collapses after vLLM selection and remains recoverable with the standard
  rail action.
- Each label, input, and action occupies a complete row.
- Browse actions are full width and never remain focusable outside the compositor.
- Readiness plus the next required action remain in the first viewport.

Tab visits only displayed controls in the active vLLM pane. Disabled actions are
skipped. Escape returns to the catalog according to existing Lab convention. Arrow
keys plus Enter select provider rows. Brackets keep the Lab mode-focus meaning shown
in the footer; hidden provider-view bracket/digit bindings are removed.

## Wireframes

### Local setup, normal width

```text
┌ Lab / Models / vLLM ───────────────────────────────────────────────────────┐
│ vLLM · Setup incomplete                         Servers: none running      │
├──────────────┬──────────────────────────────────────────┬──────────────────┤
│ Local servers│ READINESS                                │ vLLM INSPECTOR   │
│ ...          │ ✓ Python 3.12                            │ State: Stopped   │
│ > vLLM       │ ✕ vLLM not installed   [View recovery]  │ Profile: Default │
│ ...          │ ○ Model                                  │ Endpoint: —      │
│              │ ○ Network                                │ Model: —         │
│              │                                          │                  │
│              │ MODE  [Start here ●] [Connect existing] │ Next action      │
│              │ PROFILE [Default vLLM ▼] [Manage]       │ Check setup      │
│              │ Python  [/env/bin/python] [Browse]      │                  │
│              │ Model   [HF repository ▼]               │                  │
│              │         [organization/model]             │                  │
│              │ Bind    [127.0.0.1] Port [8000]         │                  │
│              │                                          │                  │
│              │ [Check setup]              [Start vLLM] │                  │
│              │ ▸ Advanced                               │                  │
│              │ ACTIVITY · Waiting for setup             │                  │
└──────────────┴──────────────────────────────────────────┴──────────────────┘
```

### Running with a modified restart draft

```text
CURRENT SERVER · Ready
  Endpoint  http://127.0.0.1:8000/v1
  Model     chatbook-vllm
  Profile   Default vLLM

NEXT RESTART CONFIGURATION · Modified
  Model     organization/larger-model               changed
  dtype     bfloat16                                changed

[Stop] [Restart with draft]          [Use in Console] [Make default…]
```

### Compact setup

```text
┌ vLLM · Setup incomplete ─────────────┐
│ ✓ Python 3.12                        │
│ ✕ vLLM not installed                │
│ ○ Model                              │
│ ○ Network                            │
│                                      │
│ Python environment                   │
│ [/env/bin/python                  ]  │
│ [Browse environment]                 │
│                                      │
│ Model source [HF repository ▼]       │
│ [organization/model               ]  │
│                                      │
│ Bind address [127.0.0.1]             │
│ Port         [8000]                  │
│                                      │
│ [Check setup]                        │
│ [Start vLLM]                         │
│ ▸ Advanced                           │
└──────────────────────────────────────┘
```

## Testing and evidence

### Pure/unit tests

- Source-specific validation, path acceptance, and field error projection.
- Public CLI construction and managed/secret raw-argument rejection.
- Bind-to-client endpoint normalization, including wildcard IPv4/IPv6.
- Semantic fingerprints and generation invalidation.
- Served-model admissibility and path-ID rejection.
- Product-state projection from runtime and connection truth.
- Profile schema limits, exact round trip, atomic failure, corruption, and future
  version preservation.

### Integration tests

- Real loopback `/health` and `/v1/models` fixtures for loading, ready, missing
  model, auth failure, timeout, cancellation, and stale settlement.
- Exact process-claim launch/stop/restart sequencing with a controllable process.
- Mounted Lab-to-Console session adoption without config writes.
- Durable default delegation preserving a different configured endpoint and
  unrelated concurrent config fields.
- Screen recomposition restoring view/profile while invalidating obsolete workers.

### Production-stylesheet UI tests

At 80x24, 100x30, and 120x40, measure every visible focusable descendant against its
owning pane for:

- first-run incomplete setup;
- successful preflight;
- launching/loading;
- ready and Console handoff;
- failed startup with recovery;
- current server plus dirty next-restart draft;
- profile management and deletion confirmation.

Walk the complete Tab order, verify no focus enters hidden provider bodies, and prove
each lifecycle transition lands on the intended action.

### Live evidence

Use a disposable HOME/XDG/config/data profile and fingerprint the real profile before
and after. If the host has an eligible vLLM environment, launch a small chat-capable
model and verify Check setup → Start → loading → `/health` → `/v1/models` → Use in
Console → one response. If capability is absent, record the missing prerequisite and
do not describe loopback fixtures as real vLLM qualification.

## Delivery sequence

1. **TASK-31213 — Contract:** ADR-115 and this specification only.
2. **TASK-31214 — Guided preflight:** launch/connect modes, source-specific setup,
   environment/network checks, safe public CLI builder.
3. **TASK-31215 — Readiness:** app-scoped connection owner, current snapshot,
   generation fencing, health/model probing, Activity.
4. **TASK-31217 — Console adoption:** session Apply, Settings default delegation,
   navigation, stale/rollback behavior.
5. **TASK-31219 — Profiles:** device-local repository, management actions,
   current-versus-next projection, restart with draft.
6. **TASK-31221 — Responsive completion:** compact composition, Inspector behavior,
   focus containment, production-stylesheet matrix, final end-to-end evidence.

Each task is one reviewable PR and may begin only after its listed dependencies are
complete. Production acceptance criteria may not be moved into a later task merely
to close an earlier PR.

## ADR check

ADR required: yes

ADR path: `backlog/decisions/115-vllm-lab-console-readiness-and-profiles.md`

Reason: the redesign changes provider/runtime ownership, cross-screen service
contracts, device-local persistence, privacy boundaries, and long-lived UX structure.

## References

- [vLLM online serving](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)
- [`vllm serve` CLI](https://docs.vllm.ai/en/latest/cli/serve/)
- [ADR-095](../../../backlog/decisions/095-conversation-owned-console-generation-settings.md)
- [ADR-114](../../../backlog/decisions/114-llamacpp-lab-console-connection-authority.md)
