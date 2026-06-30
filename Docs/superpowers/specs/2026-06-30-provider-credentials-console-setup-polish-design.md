# Provider Credentials And Console Setup Polish Design

Date: 2026-06-30
Status: Implemented; user-approved visual proof captured
Primary Repo: `tldw_chatbook`
Related Task: `TASK-145`
Related ADR: `backlog/decisions/012-provider-credential-settings-boundary.md`

## Summary

Chatbook will make provider API-key setup discoverable and usable from Settings, while keeping the environment-variable path visible as the safer power-user workflow. Console will route missing-key recovery to the exact provider credential controls and will stop letting blocked setup copy or action chips obscure the composer input.

This is a narrow remediation on top of the approved Workbench UI system. It does not redesign all Console widgets again. It fixes the broken first-run provider path and the visible Console setup state that currently feels unfinished.

## Problem

The current Console can say "Add API Key" and navigate to Settings, but Settings only exposes `api_key_env_var`. A new user cannot paste and save an API key from the app. They must know to edit TOML by hand or launch Chatbook with an environment variable already set.

The current Console blocked-state UI also has visual and interaction problems:

- disabled/recovery copy can compete with or obscure the composer input.
- structural lines and panels can read as the same color as their backgrounds.
- empty horizontal chrome can appear even when it contains no meaningful status or actions.
- key recovery is technically routed, but not visibly precise enough for a first-time user.

## Goals

- Add direct local API-key entry and clearing under Settings > Providers & Models.
- Preserve the env-var credential path and label it as the safer/power-user path.
- Never display raw saved API keys in labels, summaries, diagnostics, or tests.
- Make Console missing-key recovery navigate to the provider credential controls, not only the category.
- Keep Console composer input readable and stable in blocked setup states.
- Remove or hide blank Console bands when no useful status or action exists.
- Improve Console separator contrast using semantic Textual tokens, not hardcoded decorative borders.

## Non-Goals

- No new encrypted credential store.
- No keyring migration.
- No provider-specific live key validation beyond existing readiness checks.
- No command-palette-only recovery.
- No broad all-screen redesign in this task.
- No decorative side-stripe accents, gradient text, glass effects, or marketing-style cards.

## User Flows

### Beginner: first key setup

```text
------------------ Console ------------------+
| OpenAI setup blocked: missing API key       |
| Sending is disabled until a key is set.     |
| [Add API key] [Choose provider] [Choose model]
+---------------------------------------------+
                          |
                          v
+------------- Settings > Providers & Models ---------------+
| Provider: [ OpenAI v ]        Model: [ gpt-4.1 v ]         |
|                                                              |
| Endpoint                                                     |
| [ https://api.openai.com/v1                              ]   |
|                                                              |
| Credentials                                                  |
| Status: Missing API key                                      |
|                                                              |
| API key                                                      |
| [ Paste key, saved locally in config                    ]    |
| [Save key] [Clear saved key]                                 |
|                                                              |
| Env var                                                      |
| [ OPENAI_API_KEY                                        ]    |
| Env vars are safer for shells, shared machines, and CI.      |
|                                                              |
| [Save provider settings]                                     |
+-------------------------------------------------------------+
```

Expected result: the user can paste a key, save, return to Console, and send without editing config files.

### Advanced: env-var setup

```text
+------------- Settings > Providers & Models ---------------+
| Credentials                                                  |
| Status: Env var OPENAI_API_KEY is configured                 |
|                                                              |
| API key                                                      |
| [ Local saved key not set                               ]    |
| [Save key] [Clear saved key]                                 |
|                                                              |
| Env var                                                      |
| [ OPENAI_API_KEY                                        ]    |
| Env vars are preferred when keys are managed by the shell.   |
+-------------------------------------------------------------+
```

Expected result: the user can keep env-var credentials, edit only the env-var name if needed, and does not see a raw key.

### Blocked Console polish

```text
Before
+-------------------------------------------------------------+
| same-color chrome / empty bar                                |
| transcript area with weak separators                         |
| [input squeezed by blocked reason + recovery action chips]   |
+-------------------------------------------------------------+

After
+-------------------------------------------------------------+
| OpenAI setup blocked: missing API key       [Add API key]    |
| Impact: sending is disabled until provider credentials exist |
+---------------- Transcript ---------------------------------+
| ...                                                         |
+---------------- Composer -----------------------------------+
| [ Ask, command, or paste task...                         ]  |
| [Attach] [Library RAG] [Save Chatbook] [Help]        [Send] |
+-------------------------------------------------------------+
```

Expected result: recovery details live above the composer, while the composer keeps enough width for input.

## Architecture

The change stays inside existing ownership boundaries:

```text
SettingsScreen
  owns persisted provider defaults:
    api_settings.<provider>.endpoint
    api_settings.<provider>.api_key_env_var
    api_settings.<provider>.api_key

provider_readiness
  reads config/env sources and reports ready/missing/recovery state

ChatScreen / Console
  owns blocked-send recovery and navigation context:
    category = providers-models
    provider = current provider
    model = current model
    field = credential or endpoint

ConsoleComposerBar / Console CSS
  owns composer layout and visible blocked-state polish
```

Direct secret mutation must use the existing Settings adapter and config-save path instead of adding a second configuration writer.

## Settings Design

The Providers & Models category gets a `Credentials` block. It replaces the current ambiguous single "Credential env" presentation with a two-path credential setup:

```text
Credentials
  Status:
    API key source: env:OPENAI_API_KEY
    API key source: local config
    API key source: missing

  API key:
    masked/password input
    placeholder explains local config storage
    save path: api_settings.<provider>.api_key
    clear action writes an empty local config value or removes the saved key if the adapter supports removal

  Env var:
    existing api_settings.<provider>.api_key_env_var field
    copy says this stores the variable name, not the secret value
```

Raw key display is forbidden. The UI may show:

```text
local config key saved
env:OPENAI_API_KEY
missing
```

The UI must not show:

```text
sk-proj-...
ghp_...
AIza...
```

## Console Design

Console missing-key recovery updates from "go to Settings category" to "go to provider credentials":

```text
NavigateToScreen(
  TAB_SETTINGS,
  screen_context={
    "category": "providers-models",
    "provider": provider_key,
    "model": model_key,
    "field": "api_key",
  },
)
```

Endpoint recovery uses the same pattern with `"field": "endpoint"`.

The blocked setup copy must state owner, problem, impact, and action:

```text
Owner: OpenAI
Problem: missing API key
Impact: sending is disabled
Action: Add API key in Settings > Providers & Models
```

The composer row must only contain composer content and compact actions. Long blocked reasons and recovery copy belong in the setup callout or inspector.

## Error Handling And Security

- Blank pasted API keys clear the local config key only through an explicit clear action or equivalent tested behavior.
- Placeholder values such as `<API_KEY_HERE>` remain invalid.
- Save success copy must not include the key.
- Test fixtures must use obvious fake keys and must assert redaction.
- Existing privacy/security diagnostics must continue to count and redact `api_key`.
- If env-var and local config key are both available, readiness copy may prefer the source order already used by provider readiness, but must remain explicit about the active source.

## Testing

Write failing tests before implementation:

```text
Settings tests
  - Providers & Models renders API key and Env var controls.
  - Missing-key navigation context focuses or marks the API-key field.
  - Saving a fake key persists api_settings.openai.api_key.
  - Clearing the key removes or blanks api_settings.openai.api_key.
  - Raw fake key is never visible in rendered status or summaries.

Console tests
  - Missing-key recovery action navigates with field=api_key.
  - Endpoint recovery action navigates with field=endpoint.
  - Blocked setup state renders recovery outside the composer input.
  - Composer input retains a stable minimum width in blocked state.
  - No empty Console header/status band is displayed when no content exists.

CSS/Textual contract tests
  - Separators use semantic tokens with stronger contrast than panel background.
  - Composer blocked-state helper labels do not take fixed width from input.
```

## ADR Check

ADR required: yes
ADR path: `backlog/decisions/012-provider-credential-settings-boundary.md`
Reason: The task exposes direct in-app mutation of local provider API keys, which is a credential and privacy boundary.
