# ADR 012: Provider Credential Settings Boundary

Status: Accepted
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

## Links

- [Design spec](../../Docs/superpowers/specs/2026-06-30-provider-credentials-console-setup-polish-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-06-30-provider-credentials-console-setup-polish.md)
- [ADR 006: Provider-Aware Generation Settings](006-provider-aware-generation-settings.md)
- [ADR 011: Chatbook Workbench UI System](011-chatbook-workbench-ui-system.md)
