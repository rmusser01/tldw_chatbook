# ADR 012: Provider Credential Settings Boundary

Status: Accepted (amended 2026-09-26: explicit key check by model listing)
Date: 2026-06-30
Related Task: [backlog/tasks/task-145 - Restore-provider-credential-onboarding-and-polish-Console-setup-UX.md](../tasks/task-145%20-%20Restore-provider-credential-onboarding-and-polish-Console-setup-UX.md)
Supersedes: N/A

## Decision

Settings may expose provider credential setup for both environment-variable names and local config-backed API keys under the existing `api_settings.<provider>` boundary, while Console owns blocked-send recovery and navigation into the exact Settings credential controls.

## Context

Chatbook already supports provider API keys through environment variables and through the existing `api_settings.<provider>.api_key` fallback in local TOML config. Provider readiness and config loading can already inspect that fallback, but the Settings UI only exposes the environment-variable name. This leaves new users with an `Add API Key` recovery action that opens Settings without an actual API-key entry path.

The product contract requires visible setup and recovery states. Users should not need to discover a TOML file, edit it by hand, or infer that a "Credential env" field means "paste the variable name, not the key." At the same time, direct local API-key storage is less secure than environment variables and must be labeled as local config storage, masked in the UI, and redacted from diagnostics.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Keep env-var-only setup | This preserves the safest path but fails first-run setup and contradicts the existing local config fallback. |
| Add direct API-key entry only | This hides the safer and more portable env-var workflow from power users. |
| Store API keys in a new credential store | Keyring/encrypted storage is valuable but larger than this recovery fix and would duplicate the existing config fallback before a migration design exists. |
| Put provider setup only in Console settings | Console owns session recovery, not durable global provider defaults. Settings remains the configuration owner. |

## Consequences

Settings Providers & Models must distinguish "API key saved in local config", "API key available from env var", and "missing API key" without displaying raw secrets. Direct key entry must use masked input and must support clearing the saved local key. Env-var setup remains visible and described as the safer/power-user path.

Console missing-key recovery must navigate to Settings with provider and credential intent, not only to a generic settings category. Console can surface setup blockers and recovery actions, but durable provider credentials remain owned by Settings.

This ADR does not introduce encrypted credential storage, keyring migration, or provider-specific secret validation. Those require separate UX, storage, and migration decisions.

## Web search setup extension (TASK-32188 / TASK-32189, 2026-09-09)

The canonical F9 Settings screen exposes a staged **Web Search** category.
It owns the existing `[SearchSettings] search_provider_default` and curated
`[SearchEngines]` fields. Selecting which backend to edit does not change the
default. Save applies the complete staged delta atomically through the config
owner; Revert discards it. Provider and category navigation preserve drafts.
The category uses ADR-033's staged model and ADR-032's shared-default policy.

A shared search-backend field catalog defines labels, required fields,
secret flags, environment names, legacy aliases, and setup guidance. Runtime
requests and setup checks resolve the same environment > local config values.
No additional credential store is introduced. Saved secrets are never
prefilled into widgets; blank replacement fields keep existing keys, explicit
Clear stages deletion, and an active environment override remains visible.
Tests and diagnostics use closed, secret-safe messages rather than provider
exception bodies. No secret or probe result is persisted as UI metadata.

Setup checks are local. **Test saved settings** is a separate explicit action:
it sends the visible sample query to the selected saved backend, may consume
provider quota, does not invoke a synthesis LLM, and never silently saves a
draft. Test state is invalidated by edits and stale results cannot update a
different backend or a later Settings view. SearX endpoints may be local, as
already allowed by ADR-032. Tests do not grant Console tool permission.

Backend requests resolve current config without rewriting a process-global
request snapshot. Shipped Bing/SearX field spellings remain compatible with
legacy aliases; retired/restricted services are labeled instead of appearing
as interchangeable first-run recommendations.

Alternatives rejected: raw-TOML-only setup retains the discovery and recovery
failure; automatically making the edited provider the default changes query
destination while configuring an alternative; automatic network validation
would spend quota and send queries during ordinary editing; storing test
results permanently would imply readiness beyond the configuration tested.

The existing product identity and Settings layout remain authoritative. The
new panel is modular; it does not add another Settings surface or redesign the
application shell.

## Subscription setup preservation (TASK-32711, 2026-09-17)

Completing first-run setup with Anthropic's existing `claude_subscription`
auth source and no API-key edit preserves the inactive `api_key`,
`api_key_env_var`, and `credential_source` fields. Subscription readiness is
a secret-free observation; its lack of an API-key value is not a Clear
instruction. The borrowed token remains outside setup drafts and mutations.

The shared provider-setup builder owns this distinction. It may issue a sparse
credential-preserving mutation only for an unchanged (`none`) draft under
Anthropic subscription configuration. That mutation omits all three credential
fields from writes and deletes. Its `none` semantic identity describes the
setup observation, not the persisted API-key routing choice. Explicit Clear
and typed replacement retain their existing deletion and replacement behavior.
Immutable issuance checks, provider ownership validation, and atomic config
preconditions continue to apply. The auth source is part of those preconditions,
so changing from subscription to API-key auth invalidates a queued setup save.

Alternatives rejected: inferring deletion from missing borrowed API-key material
silently destroys unrelated configuration; copying inactive secrets into the
mutation needlessly rewrites values and can erase concurrent edits; rewriting
an issued immutable mutation in the wizard bypasses the shared owner's
validation boundary. No new credential storage or authentication fallback is
introduced.

## Amendment 2026-09-19: stored key over environment variable (TASK-32806.2)

The core-runtime review of 2026-09-17 found two accessors resolving provider
credentials in opposite orders, with **both orders asserted by test name**:

- `config.py` `_normalize_legacy_provider_api_key` (the chat spend path):
  modern `api_settings.<provider>.api_key` > environment variable > legacy
  `[API]`.
- `config.py` `get_api_key` (MCP tools, Console realtime): the
  `api_key_env_var` environment variable > stored `api_key` > legacy.

Reproduced with `[api_settings.anthropic] api_key = "sk-ant-modern"` and
`ANTHROPIC_API_KEY=sk-ant-from-env` set together:

    bridge anthropic_api.api_key : 'sk-ant-modern'
    get_api_key('anthropic')     : 'sk-ant-from-env'

So `chat_api_call` spent the Settings key while MCP tools and Console
realtime spent the environment key. The shipped configuration template gives
every `[api_settings.<provider>]` table an `api_key_env_var`, so the
disagreement is reachable by default rather than by unusual configuration.
This is the same "readiness and spend disagree" failure this ADR exists to
prevent, relocated from two readers to two accessors.

**Ruling: the stored `api_key` outranks the environment variable it names.**

This is not a new decision. It is the rule already recorded in CLAUDE.md --
"Priority: env vars -> config.toml -> defaults, EXCEPT provider API keys,
where an explicit `api_settings.<provider>.api_key` now outranks the matching
environment variable" -- and already implemented on the spend path by PR-T2.
`get_api_key` was the accessor out of line, and now matches it.

The reason the exception exists at all: a key the user types into Settings
must take effect. Under environment-first, a stale `OPENAI_API_KEY` exported
in a shell profile silently outranked what the UI showed and what the user
had just entered, with no surface anywhere saying which one would be spent.

The environment variable is unchanged as the FALLBACK, so an env-only
deployment keeps working exactly as before. Precedence below those two --
legacy `[API] <provider>_api_key`, then the conventional `<NAME>_API_KEY` --
is untouched.

`Tests/Utils/test_config_api_key_resolution.py` carried the contradicting
assertion. It was rewritten to the ruling rather than deleted, and a
companion test pins the fallback, so both halves of the order are asserted.

## Amendment 2026-09-26: explicit key check by model listing (model-config redesign, D2)

The model-configuration review of 2026-09-26 found that Settings "Test
Provider" never checks a cloud key. For OpenAI it reports:

    OpenAI configuration is complete. Credential is present; provider
    acceptance has not been tested.

That text comes from `settings_screen.py:15247-15251`, and a fake key passes.
Only URL-based providers get a live probe: `_provider_live_probe_base_url`
returns "" for everything else. The behaviour is by design. This ADR's
Consequences exclude provider-specific secret validation, and TASK-386's copy
calls the action a local readiness check. The button still says "Test",
though, and a test that cannot fail on a wrong key is the Settings surface's
worst status problem (`qa/model-config-ux-review-2026-09-26/report.md`).

**Ruling: in Settings ▸ Providers & Models, `t` on a cloud provider runs one
explicit, non-generating, authenticated model listing and reports what it
proved: "key accepted (models listed); generation not tested". Cloud providers
are never probed automatically.**

This narrows the Consequences sentence "This ADR does not introduce ...
provider-specific secret validation" without removing it. The check is the
provider's own authentication of one listing request, made when the user asks
for it. It is not a local validator of key format. Encrypted storage and
keyring migration stay out of scope.

The rules:

- **Explicit only.**
  - The check runs only when the user presses `t` or its button.
  - Nothing else triggers it: not opening Settings, not editing or saving the
    key, not opening the Console model switcher, not startup.
  - ADR-020's consent-gated catalog refresh is unchanged, and it is not a key
    check. Its results never show a key as verified.
- **Non-generating.**
  - One model-listing request goes through the existing discovery client,
    `LLM_Provider_Catalog/openai_compatible_model_discovery.py`. That client
    already sends Anthropic's `x-api-key` and maps 401/403 (`:726-737`).
  - There is no completion call. The paid 1-token test stays a separate,
    consented action.
- **Same destination, same credential.**
  - The request goes to the endpoint the spend path would use for the draft
    under test, with the credential that path would use, resolved by the
    precedence in the 2026-09-19 amendment. The key goes nowhere a send would
    not already take it.
  - The check never saves the draft.
- **Honest words.** The result carries the shared readiness vocabulary:

  | Outcome | Shown as |
  |---|---|
  | The listing returns models | `Ready · verified HH:MM`, with "key accepted (models listed); generation not tested" |
  | 401 or 403 | `Not ready · key rejected` |
  | Connection refused or timed out | a distinct `Not ready` reason |
  | The listing needs no key (OpenRouter's catalog is public, per ADR-020) | "models listed; key not checked". Never "accepted" |
  | The provider has no listing endpoint | today's local readiness check, labelled as local |
- **Nothing persists.**
  - The result lives only in process memory, keyed by the draft identity, and
    any semantic edit invalidates it.
  - No result, key or response body is persisted, logged or displayed.
    Messages stay closed and secret-safe, as for the web search Test above.

Alternatives rejected:
- **Keep the local-only check under a "Test" label.** The label promises a key
  check that never happens.
- **Probe automatically on edit, save or switcher open.** That spends provider
  rate limits and contacts a third party without a user action. ADR-020
  needed a consent gate for exactly this.
- **Use a 1-token generation as the key check.** It costs money and needs
  consent, and a listing proves the key without generating anything.
- **Validate the key's format locally.** That says nothing about whether the
  provider accepts the key.

Unchanged:
- Settings remains the only place credentials are entered. Owner decision D4
  of the same redesign reaffirms this: there is no inline key entry in
  Console.
- Console still only surfaces blockers and routes recovery to the exact
  Settings credential control.
- Key precedence (2026-09-19) is unchanged.

## Links

- [Design spec](../../Docs/superpowers/specs/2026-06-30-provider-credentials-console-setup-polish-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-06-30-provider-credentials-console-setup-polish.md)
- [ADR 006: Provider-Aware Generation Settings](006-provider-aware-generation-settings.md)
- [ADR 011: Chatbook Workbench UI System](011-chatbook-workbench-ui-system.md)
- [Model configuration redesign spec (2026-09-26 amendment)](../docs/spec-2026-09-26-model-config-redesign.md)
