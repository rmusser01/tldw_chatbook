# llama.cpp Lab-to-Console handoff wireframe

## Design brief

- **Job and audience:** Help a first-time local-model user reach a verified Chatbook response, while giving an experienced user repeatable launch, restart, and diagnosis controls.
- **Outcome and proof:** Success means the selected llama.cpp API is healthy, its model is discoverable, and the exact provider, endpoint, and model are active in the Console session. A live subprocess alone is not success.
- **Direction:** Keep the terminal-native Lab workbench, but replace the flat launcher form with a four-step readiness path. Present one primary action for the current state and disclose expert configuration separately.
- **Boundary:** Lab owns local process launch, model-path authority, and runtime output. Console owns the active conversation destination. Provider defaults remain configuration-owned and change only through an explicit Make default action.
- **Responsive rule:** At compact widths, collapse navigation before setup content, stack all label/input/action groups, and keep the current recovery or completion action painted without horizontal panning.

## Shared state language

| State | Meaning | Primary action |
|---|---|---|
| Not configured | Required runtime, model, or endpoint is missing | Complete the current checklist item |
| Checking | Chatbook is probing the exact endpoint | Cancel |
| Starting | Chatbook owns a pending local process claim | Stop starting |
| Loading model | Process is alive but the selected model is not API-ready | Stop |
| API ready | Health and model discovery passed | Use in Console |
| Console connected | The active Console session uses the verified destination | Open Console |
| Needs attention | A specific preflight, process, or API check failed | Context-specific recovery |

`Ready` is not used for a mounted screen or process liveness.

## First-run — start on this computer

```text
┌ LAB ─ Models ─ llama.cpp                                      Not configured ┐
│ Catalog        Set up llama.cpp                                                │
│                Run a local GGUF model and connect it to Chatbook.              │
│ ● llama.cpp                                                                    │
│   Ollama       [ Start on this computer ] [ Connect to existing server ]       │
│   vLLM                                                                         │
│                SETUP                                                           │
│ Models         1  Runtime       llama-server found                  [Change]   │
│   Installed    2  Model         Not selected                        [Choose]   │
│   Remote       3  Endpoint      127.0.0.1:8080                      [Edit]     │
│                4  Chatbook      Waiting for steps 1–3                          │
│                                                                                 │
│                Choose a model                                                   │
│                (•) Installed in Chatbook — recommended                         │
│                    No compatible GGUF models installed                          │
│                    [ Find a model ]  [ Import a GGUF ]                          │
│                ( ) Existing GGUF file                                           │
│                                                                                 │
│                ▸ Expert launch settings                                         │
│                                                                                 │
│                Complete step 2 to continue.                   [ Start & check ] │
└─────────────────────────────────────────────────────────────────────────────────┘
```

Behavior:

- Detect runs automatically once when no executable is known; Browse remains available.
- Find a model opens Remote and preserves return intent for the exact llama.cpp setup.
- Start & check preflights the executable, model, and endpoint; it becomes available only when the adjacent reason is satisfied.
- Expert launch settings contains host binding, port override, context/hardware presets, raw arguments, and command preview.
- Every Lab-owned start reserves the stable API model alias `chatbook-llamacpp`; expert arguments cannot replace or duplicate it.

## First-run — connect to an existing server

```text
┌ LAB ─ Models ─ llama.cpp                                      Not configured ┐
│                Set up llama.cpp                                                │
│                [ Start on this computer ] [ Connect to existing server ]       │
│                                                                                 │
│                Server URL                                                      │
│                [ http://127.0.0.1:8080_____________________ ] [ Check ]         │
│                                                                                 │
│                Checking…  Health endpoint reached                              │
│                Model       [ community/gemma-3-4b-it                 ▾ ]         │
│                                                                                 │
│                This changes the current Console session only.                   │
│                Your saved endpoint remains unchanged.                           │
│                                                                                 │
│                                             [ Cancel ] [ Use in Console ]        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

This path never asks for a local executable or GGUF path.

An existing server that exposes a filesystem- or GGUF-path model ID fails before the
ID is offered or copied. Recovery says to configure `llama-server --alias` and check
again without echoing the rejected value. A forward slash by itself remains valid
for namespace-style model IDs such as `community/gemma-3-4b-it`.

Before path classification, the endpoint-provided ID must be 1–120 Unicode code
points, already use canonical whitespace, be printable, and contain no control,
format/bidi-control, or surrogate character. Chatbook rejects unsafe values rather
than cleaning or truncating them, and does so before selector, display, copy, log,
descriptor, or adoption. The recovery message never echoes the rejected value.

## Verified handoff

```text
┌ LAB ─ Models ─ llama.cpp                                           API ready ┐
│                llama.cpp is ready for Chatbook                                │
│                                                                                │
│                Endpoint     http://127.0.0.1:8080                              │
│                Model        chatbook-llamacpp                                  │
│                Runtime      Local process · PID 4812                           │
│                Health       API healthy · model available                      │
│                Console      Not connected                                      │
│                                                                                │
│                This applies to the active Console session.                     │
│                It will not replace a different saved endpoint.                 │
│                                                                                │
│                [ Stop server ] [ Make default… ] [ Use in Console ]            │
└────────────────────────────────────────────────────────────────────────────────┘

                              ↓ Use in Console

┌ CONSOLE ─ New chat                                      llama.cpp · Connected ┐
│ Model: chatbook-llamacpp                                                   [⌄] │
│ Endpoint: 127.0.0.1:8080 · Session only                                      │
│                                                                                │
│ Try a message to verify generation.                                            │
│ [ Say hello in one sentence.__________________________________ ] [ Send ]      │
└────────────────────────────────────────────────────────────────────────────────┘
```

## Experienced user — current and next launch

```text
┌ LAB ─ Models ─ llama.cpp                            API ready · Console active ┐
│ Profile  [ Coding · 8k context ▾ ]          [ Duplicate ] [ Save changes ]    │
│                                                                                │
│ CURRENT LAUNCH                         NEXT LAUNCH                              │
│ Model     Qwen2.5-Coder-7B Q4          Model     Qwen2.5-Coder-14B Q4          │
│ Endpoint  127.0.0.1:8080               Endpoint  127.0.0.1:8080                │
│ Health    API ready                     Preflight Changes need checking          │
│ Started   14:32 · PID 4812              Arguments --ctx-size 16384 --ngl 35     │
│                                                                                │
│ [ View command ] [ Server log ]         [ Check changes ] [ Restart with next ]│
│                                                                                │
│ Server log · last 200 sanitized lines                              [ Copy ]     │
│ 14:32:08 model loaded · 5.1 GiB                                                │
│ 14:32:09 listening on 127.0.0.1:8080                                           │
│ 14:32:09 health check passed                                                    │
│                                                                                │
│ [ Stop server ]                                               [ Open Console ] │
└────────────────────────────────────────────────────────────────────────────────┘
```

Current launch is immutable process truth. Next launch is an editable candidate. Restart performs preflight before stopping the current healthy process.

## Compact 80-column topology

```text
┌ LAB · Models · llama.cpp ─────────────────────── Not configured ┐
│ Set up llama.cpp                                                 │
│ [ Start here ] [ Connect existing ]                             │
│                                                                 │
│ 1 Runtime                                            Incomplete │
│ Server executable                                               │
│ [ /opt/llama.cpp/build/bin/llama-server___________ ]            │
│ [ Detect ] [ Browse ]                                           │
│                                                                 │
│ 2 Model                                              Waiting    │
│ 3 Endpoint                                           Waiting    │
│ 4 Chatbook                                           Waiting    │
│                                                                 │
│ Choose or detect a server executable.                           │
│                                                [ Continue ]     │
└─────────────────────────────────────────────────────────────────┘
```

The provider catalog and Inspector collapse before the setup column. Nothing relies on horizontal scrolling. F6 enters the setup pane; a pane-local action returns to the current primary action.

## Error and recovery examples

| Failure | Message | Recovery |
|---|---|---|
| Port occupied | Port 8080 is already in use. Chatbook found a llama.cpp-compatible API there. | Connect to it, or choose another port |
| Executable unavailable | llama-server could not be executed. Check file permission or choose another binary. | Browse, Detect again |
| Model rejected | llama.cpp started, but could not load this model. | View sanitized log, choose another model |
| Health timeout | The process is running, but its API did not become ready within the expected time. | Keep waiting, view log, stop |
| Saved endpoint differs | Console is saved to 192.168.1.20:8080. Use this local server for the current session only? | Use for session, cancel, make default |
| Unsafe model identity | This server did not return a model name Chatbook can safely use. | Configure `llama-server --alias`, check again |

## Scope boundaries

- Do not auto-start a runtime on app launch.
- Do not auto-persist discovered endpoints or models.
- Do not copy executable or GGUF paths into Console session metadata.
- Do not project Lab-owned executable/GGUF selection, PID, expert arguments, command preview, or diagnostics into the handoff descriptor, Console, conversation metadata, or app-global metadata.
- Do not display, log, or copy a rejected path-identifying endpoint model ID; show generic `--alias` recovery instead.
- Do not generalize the first implementation into a universal local-server framework before the llama.cpp contract is proven.
- Do not expose unbounded or unsanitized subprocess output.

## Architecture authority

- [ADR-114](../../../backlog/decisions/114-llamacpp-lab-console-connection-authority.md) owns endpoint defaults, readiness truth, cross-surface handoff, persistence scope, and privacy boundaries.
- [ADR-025](../../../backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md) continues to own Managed versus External GGUF authority and exact process-lifetime artifact leases.
- [ADR-095](../../../backlog/decisions/095-conversation-owned-console-generation-settings.md) continues to own Console session settings and explicit default persistence.

## Backlog mapping

1. TASK-31200 defines the cross-surface contract and ADR.
2. TASK-31201 implements verified readiness and Console adoption.
3. TASK-31202 implements the first-run paths and documentation.
4. TASK-31203 implements compact layout and keyboard behavior.
5. TASK-31204 separates current and next launch and adds Restart last.
6. TASK-31205 adds durable named profiles.
7. TASK-31206 adds bounded diagnostics and recovery.
