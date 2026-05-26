# Console Session Settings Design

Date: 2026-05-25
Status: Approved by user and spec review; amended with post-review clarifications
Primary Repo: `tldw_chatbook`
Scope: Console screen UX, per-tab session settings, provider/model discoverability, sampling controls, and context summary

## Summary

Console needs a first-class settings surface that is visible, compact, and owned by the native Console flow. The current Console can send through the configured `llama_cpp` provider, but provider/model selection is not discoverable in the rendered Console UI because the existing compact model bar is hidden. This design adds a small `Console Settings` summary box under the left `Context` rail content and a dedicated modal for editing current Console tab settings.

The first version is deliberately scoped to the current Console chat tab. It exposes provider/model and sampling settings, shows read-only context usage, and displays current persona/character identity as read-only. It does not write global chat defaults and does not implement persona/character selection yet.

## Current Context

Console already has a native chat core and rail workbench:

- `ChatScreen` owns Console composition, native send state, rail preferences, and provider/model resolution.
- `ConsoleChatController` owns send, continue, regenerate, retry, stop, and session switching behavior.
- `ConsoleProviderGateway` resolves and streams native Console provider calls. Today, `llama_cpp` and `local_llamacpp` are wired; other providers return explicit WIP blocked copy.
- `ConsoleControlBar` renders a single status line and composes `CompactModelBar`, but hides it with `_hide_layout_widget(...)` and `.console-hidden-control`.
- `ConsoleStagedContextTray` owns the left rail `Context` content.
- `ConsoleRunInspector` owns right rail readiness and run inspection.
- Existing token utilities in `Utils/token_counter.py` can estimate used tokens and model limits.

The design should not couple settings to staged context. The left rail can contain both `Context` and `Console Settings`, but each section should have separate state and widget ownership.

## Goals

- Make the active Console provider/model discoverable without reopening legacy Chat settings.
- Let users edit provider/model and core sampling parameters for the current Console tab.
- Keep the left rail compact and prevent the Console from becoming cramped again.
- Show current context usage as an estimate only.
- Show current persona/character identity without making identity editing part of the first slice.
- Keep settings scoped to the active Console tab/session.
- Allow unsupported providers to be selected and shown as WIP/blocked, without pretending they can send.
- Preserve existing native Console send, transcript, rail, composer, and message action behavior.
- Leave a clear future path for `Save as workspace default`.

## Non-Goals

- Do not implement global settings writes.
- Do not implement `Save as workspace default` in this slice.
- Do not implement full persona/character selection in this slice.
- Do not wire every provider into native Console sending in this slice.
- Do not redesign the entire Console header, top status line, or right inspector.
- Do not make context estimate control truncation or prompt packing.
- Do not mutate hidden legacy Chat widgets as the source of truth for native Console settings.
- Do not replace Tools Settings or the legacy Chat settings screen.

## Approved Direction

The selected approach is a left-rail `Console Settings` summary plus a dedicated modal.

Rejected alternatives:

- Header status line plus modal: always visible, but risks reintroducing the cramped top rows that were recently removed.
- Right inspector settings tab: keeps the left rail pure, but weakens the inspector boundary by mixing editable session settings with run readiness.

The left-rail summary works best because settings are important before sending, while inspector details are mostly useful during or after a run.

## UX Surface

Add a compact `Console Settings` box below the existing `Context` section inside the left rail. It is not nested inside staged context and should be implemented as its own Console widget.

Target shape:

```text
Context                         <
Staged Context
No staged work.
[Attach]

Console Settings
Model: llama.cpp / model-name
Context: 1.2k / 32k
Sampling: T 0.7, P 0.95
Persona: General
[Settings]
```

Rules:

- Show at most four summary rows plus the button.
- Use terse labels: `Model`, `Context`, `Sampling`, `Persona`.
- Prefer graceful truncation over wrapping long model names across many rows.
- Keep the `Settings` button visually obvious but not dominant over the composer.
- When the left rail is collapsed, do not show settings detail in the collapsed handle.
- The summary refreshes when the active Console tab changes or settings are saved.
- The top status line may continue to exist, but it should not be the only provider/model affordance.

## Modal UX

Clicking `Settings` opens a Console-owned modal scoped to the active Console tab.

Target structure:

```text
Console Settings

Session Model
Provider        [OpenAI             v]   Ready / WIP / Missing key
Model           [gpt-4.1            v]
Base URL        [http://127.0.0.1:9099]  shown only when relevant

Sampling
Temperature     [0.70]
Top P           [0.95]
Min P           [0.05]
Top K           [40]
Max Tokens      [4096]
Streaming       [x]

Context
Current         1.2k / 32k tokens
Sources         0 staged
Note            Estimate only; no truncation changes in this version.

Identity
Current         General
Persona         Read-only for this version
Character       Read-only for this version

[Cancel] [Save]
```

Behavior:

- Fields edit a draft copy.
- `Save` validates and applies the draft to the current Console tab.
- `Cancel` discards draft changes.
- No changes apply live while the modal is open.
- The modal may open while a run is active, but `Save` is disabled unless the current Console run state allows a new send. `Cancel` remains available.
- Saved settings affect the next send, continue, or regenerate operation only. They must not alter an active stream in flight.
- The modal closes on successful save.
- Invalid fields keep the modal open and show concise inline error copy.
- Readiness copy updates when provider changes in the modal draft.
- Missing API keys and unsupported native providers warn but do not block save.
- Invalid base URLs block save for URL-based providers.

## Provider Selection

The modal should show all configured providers, not only Console-ready providers.

Provider/model options are config-backed in this slice:

- Provider options come from the same configured provider/model registry used by current Chat controls, such as `get_cli_providers_and_models()`.
- Model options come from that provider/model registry.
- The current configured model from the active session or `api_settings.<provider>.model` should be included even when it is not in the registry list.
- If no model options are known for a provider, the modal should show an editable model field rather than attempting live discovery.
- Live model discovery is not part of the modal. The only allowed discovery behavior in this slice is the existing `llama_cpp` send-time fallback that can query `/v1/models` when no explicit/configured model is available.
- Modal readiness is static/config-backed. It should not make network calls such as `/health` or `/v1/models`; send-time provider resolution remains the authoritative reachability check.

Provider readiness labels:

- `Ready`: provider appears usable for native Console.
- `Missing key`: provider requires a key and none is configured.
- `WIP`: provider is configured but native Console send is not wired yet.
- `Invalid URL`: URL-based provider has an invalid endpoint.
- `Unknown`: provider cannot be matched to known readiness rules.

When more than one readiness condition applies, use this precedence:

1. `Invalid URL`
2. `WIP`
3. `Missing key`
4. `Ready`
5. `Unknown`

For example, an OpenAI provider with no API key is still labeled `WIP` in native Console because the larger blocker is that native Console sending is not wired for OpenAI yet. The modal may include secondary explanatory copy such as `also missing API key`, but tests should assert the primary readiness label from the precedence list.

For this slice:

- `llama_cpp` and `local_llamacpp` use the existing Console native gateway path.
- Other providers may be selected and saved, but sends continue to surface the existing WIP/blocked provider copy until those providers are implemented.
- The summary box should make blocked/WIP state visible, for example `Model: OpenAI / gpt-4.1 (WIP)`.

URL-based providers for base URL display/validation are providers with a configured `api_url`, `base_url`, or `api_base`, plus known local/OpenAI-compatible provider keys such as `llama_cpp`, `local_llamacpp`, `ollama`, `vllm`, `koboldcpp`, and `oobabooga`. In this slice, only `llama_cpp` and `local_llamacpp` use the base URL for native Console sends; other URL-based providers may store the value in session settings but remain WIP for sending.

## Session Settings State

Introduce a Console-owned session settings object. It should be independent of legacy sidebar widgets.

Suggested shape:

```python
@dataclass(frozen=True)
class ConsoleSessionSettings:
    provider: str
    model: str | None = None
    base_url: str | None = None
    temperature: float = 0.7
    top_p: float = 0.95
    min_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    streaming: bool = True
    persona_label: str = "General"
    character_label: str = ""
```

The exact object can be adjusted during implementation, but it should satisfy these boundaries:

- It is Console-owned.
- It can be serialized later for workspace defaults.
- It can be copied into a modal draft.
- It is safe to store per active native Console session.
- It is owned by the native Console session lifecycle, either as a field on `ConsoleChatSession` or in a private settings map keyed by Console session id.
- Creating a new Console session creates a settings snapshot from defaults. Switching sessions restores that session's snapshot. Closing a session removes its settings snapshot.
- If an existing session is encountered without settings during migration, initialize that session once from current Console/default selection rather than sharing another session's mutable settings.
- It does not rely on querying legacy `#chat-api-provider`, `#chat-api-model`, or `#chat-temperature`.

## Data Flow

Initial settings:

1. New Console tab/session is created.
2. Settings initialize from `chat_defaults` plus provider-specific `api_settings`.
3. The active session stores its own settings snapshot.
4. The summary box renders from that snapshot plus context estimate.

Editing settings:

1. User opens `Console Settings`.
2. Modal receives a draft copy of the active session settings.
3. User edits fields.
4. `Save` validates the draft.
5. Active session settings are replaced.
6. Summary box, provider readiness, and controller selection refresh.
7. The next send uses the updated settings.

Sending:

1. `ChatScreen` builds `ConsoleProviderSelection` from active Console session settings.
2. `ConsoleChatController` passes the selection to `ConsoleProviderGateway`.
3. `ConsoleProviderGateway` includes sampling params and streaming preference in supported provider payloads.
4. Unsupported providers continue through the existing blocked path.

This flow should replace the hidden compact model bar as the Console-native settings source of truth.

## Context Estimate

The `Context` row is read-only in the first version.

Estimate inputs:

- Native Console transcript messages for the active tab.
- System prompt/persona prompt when available.
- Staged source count and staged context summary when available.
- Selected provider/model.

Behavior:

- Use existing token-counter helpers where possible.
- Show `used / limit` when a model limit is known.
- Show `used / unknown` when the model limit is unknown.
- Degrade to `Context: unknown` if counting fails.
- Do not truncate history.
- Do not change prompt packing.
- Do not imply context safety beyond an estimate.

## Sampling Parameters

Editable v1 parameters:

- `temperature`
- `top_p`
- `min_p`
- `top_k`
- `max_tokens`
- `streaming`

These values apply only to the current Console tab. For `llama_cpp`, they should be mapped into the OpenAI-compatible request payload when supported:

- `temperature`: top-level `temperature`.
- `top_p`: top-level `top_p`.
- `min_p`: top-level `min_p` when not blank.
- `top_k`: top-level `top_k` when greater than `0`; blank or `0` means provider default and should be omitted.
- `max_tokens`: top-level `max_tokens` when not blank.
- `streaming`: top-level `stream`.

Add one Console gateway payload builder for native `llama_cpp` chat completions and use it for both streaming and non-streaming fallback calls. If `streaming` is false, the gateway should still expose an async iterator to the controller by making one non-streaming completion request and yielding the returned text once.

If a provider does not support a parameter, implementation may omit it from that provider payload. The UI can still store the session value because it represents user intent for that session.

## Identity

The modal displays identity but does not edit it in this slice.

Identity display:

- `Current`: `General`, persona name, or character name.
- `Persona`: read-only row.
- `Character`: read-only row.

Future persona/character editing should reuse the Console session settings object rather than binding the modal directly to legacy character loading widgets.

## Validation

Save validation rules:

- `provider`: required.
- `model`: required unless the provider can discover a default model.
- `base_url`: when shown and required for the selected provider, must be a valid URL; blank handling follows the provider-specific rule below.
- `temperature`: float from `0.0` through `2.0`.
- `top_p`: float from `0.0` through `1.0`.
- `min_p`: blank or float from `0.0` through `1.0`.
- `top_k`: blank or integer `>= 0`; `0` means provider default.
- `max_tokens`: blank or integer `>= 1`.
- `streaming`: boolean.

Validation copy should be specific and short, for example `Temperature must be between 0 and 2.`

For v1, "provider can discover a default model" means only the existing `llama_cpp` / `local_llamacpp` send-time fallback that can query `/v1/models`. Other providers require an explicit or configured model even if a future native implementation might support discovery.

Blank base URL handling is scoped to providers where the modal shows a base URL field. A blank `llama_cpp` / `local_llamacpp` base URL should fall back to the existing default origin. A blank base URL for a WIP URL-based provider should not block Save unless that field is shown as required for the selected provider in the modal.

## Error Handling

- Unsupported providers can be saved but show WIP/blocked readiness.
- Missing API keys can be saved but show missing-key readiness.
- Invalid base URLs block save for URL-based providers.
- Context estimate failures show `Context: unknown`.
- Provider readiness failures should not crash the modal or summary.
- If settings cannot be initialized, fall back to safe chat defaults and show `Model: not selected`.
- No v1 settings save writes to global config, so persistence failures should be limited to in-memory session state.

## Code Architecture

Add small, focused Console modules rather than expanding `ChatScreen` with modal internals.

Suggested modules:

- `Chat/console_session_settings.py`: pure settings dataclass, defaults builder, validation, readiness summary, context summary helpers.
- `Widgets/Console/console_settings_summary.py`: left-rail summary widget.
- `Widgets/Console/console_settings_modal.py`: modal draft editor and validation display.

`ChatScreen` responsibilities:

- Own the active settings per native Console session.
- Mount the summary widget below `ConsoleStagedContextTray`.
- Open the modal on `Settings`.
- Apply validated modal output to the active session.
- Push settings into `ConsoleChatController` before send/continue/regenerate.
- Refresh summary and control/readiness labels after save.

`ConsoleProviderGateway` responsibilities:

- Accept effective sampling settings for supported providers.
- Include supported parameters in `llama_cpp` payloads.
- Preserve current WIP blocked behavior for unsupported providers.

Avoid:

- Making hidden `CompactModelBar` the active state source.
- Querying legacy Chat sidebar widgets to determine Console session settings.
- Writing `chat_defaults` during normal modal save.
- Putting settings editing into `ConsoleRunInspector`.

## Persistence And Future Workspace Defaults

V1 is current Console tab only.

The settings object should be shaped so a later `Save as workspace default` can persist the active settings snapshot under workspace/session scope. That later feature should be explicit and separate from modal `Save`.

Expected future model:

- `Save`: current Console tab only.
- `Save as workspace default`: writes workspace-scoped defaults.
- `Reset to workspace default`: restores persisted workspace defaults.
- `Reset to global defaults`: restores `chat_defaults`.

Do not implement those future actions in this slice.

## Accessibility And Keyboard Behavior

- The summary `Settings` button must be keyboard focusable.
- Modal controls should follow logical tab order from provider/model through sampling, context, identity, then actions.
- `Escape` should cancel/close the modal if consistent with existing modal patterns.
- `Enter` on `Save` should only apply when focus is on the Save button or Textual modal conventions support default buttons safely.
- Inline errors should be readable in terminal/canvas rendering, not color-only.

## Testing

Unit tests:

- Defaults initialize from `chat_defaults` and provider-specific `api_settings`.
- Per-session settings snapshots are independent.
- Validation accepts valid numeric values and rejects invalid ranges.
- Readiness summary labels distinguish ready, missing key, WIP, invalid URL, and unknown provider.
- Context estimate degrades safely when token counting fails.
- Provider/model option helpers use configured provider/model data, include the current configured model when it is absent from the registry list, fall back to editable model input when no options are known, and do not perform modal-time live discovery.

Controller/gateway tests:

- Active session settings update `ConsoleProviderSelection`.
- `llama_cpp` streaming payload includes selected model and supported sampling params.
- Unsupported provider selection remains blocked with visible WIP copy.
- Saving settings in one Console tab does not mutate another tab.

UI tests:

- Left rail renders `Console Settings` below `Context`.
- Summary shows model, context, sampling, and persona rows.
- `Settings` opens the modal.
- `Cancel` discards edits.
- `Save` updates the active tab summary.
- `Save` is disabled while a Console run is active and becomes available again when the run reaches a send-allowed state.
- Invalid numeric input keeps the modal open and shows error copy.
- Provider WIP state is visible in modal and summary.

UAT:

- Open Console and confirm settings summary is visible but compact.
- Switch provider/model in modal.
- Save and confirm summary updates.
- Send a basic chat through `llama_cpp`.
- Change temperature/top_p/min_p/top_k/max_tokens and confirm the next request payload changes.
- Open a second Console tab and confirm settings remain isolated.
- Select unsupported provider and confirm WIP/blocked state is honest.
- Capture default-width and compact-width screenshots with a long provider/model name to confirm the left rail remains readable and does not re-cramp the transcript/composer.

## Risks

- The left rail may become crowded if summary rows wrap. Mitigation: hard cap rows, truncate long model names, keep detail in modal.
- Provider readiness can be confusing if unsupported providers are selectable. Mitigation: show WIP clearly in modal and summary.
- Sampling params may not map uniformly across providers. Mitigation: store user intent in session settings and only send supported payload fields.
- Context estimate may be mistaken for truncation safety. Mitigation: label it as estimate only and avoid editable context controls in v1.
- Legacy Chat state may drift from Console state. Mitigation: make Console session settings the explicit source of truth and avoid hidden widget sync.
