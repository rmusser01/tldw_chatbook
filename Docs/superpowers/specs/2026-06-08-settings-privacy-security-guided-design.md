# Settings Privacy And Security Guided Posture Design

Date: 2026-06-08
Status: Approved for implementation
Target branch: `dev`
Backlog: `TASK-82`

## Purpose

Make `Settings > Privacy & Security` useful as a read-only privacy posture and recovery panel.

The screen should answer the user's immediate questions:

- Are secrets protected from display?
- Are provider credentials coming from environment variables or config?
- Is config encryption enabled?
- What local/server data boundary applies right now?
- Where should I go to fix a privacy or credential problem?

This slice must not add unsafe credential editing or password/encryption mutation. Those require a separate password-gated recovery flow.

## Current State

The current Settings screen has a `Privacy & Security` category with a working privacy check, but the visible category content is mostly static copy:

- Secrets are described generically.
- Encryption says it is not configured from the slice.
- Credential mutation is marked unavailable/WIP.
- The only action is `Check Privacy`.
- The inspector mostly repeats generic category guidance.

Existing helper logic already computes useful redacted facts:

- config encryption enabled or disabled
- sensitive config field count
- provider environment variable presence and missing counts
- provider config secret count
- local/server data-boundary copy
- redaction safety copy

The issue is presentation and recovery: the user cannot see a structured privacy posture before running the check, and cannot navigate to likely recovery surfaces from this category.

## Design

`Privacy & Security` remains read-only in this slice, but becomes a structured guided panel.

### Center Pane

Render three compact sections:

- `Privacy posture`
  - Config encryption: `enabled` or `disabled`
  - Redaction: `active`
  - Sensitive config fields: count only
  - Provider config secrets: count only

- `Credential sources`
  - Provider env vars: present / missing / configured
  - Preferred source: environment variable
  - Config secrets: counted, never shown

- `Data boundary`
  - Local data remains local unless explicit server handoff or sync is enabled
  - Server tokens are reported as configured/missing only
  - Encryption setup or credential mutation is deferred to a password-gated flow

### Actions

Use terminal-native buttons in the existing Settings style:

- `Check Privacy`
  - Runs the existing worker-backed privacy check.
  - Updates visible redacted status rows.

- `Open Providers & Models`
  - Navigates to the provider/model category where users can correct provider, model, endpoint, and credential-source defaults.

- `Open Advanced Config`
  - Navigates to the advanced config category for expert inspection and recovery.
  - The Privacy pane must make clear this is expert recovery, not a normal place to reveal secrets.

`Save` and `Revert` remain disabled for `Privacy & Security` because this category does not mutate config in this slice.

### Inspector

Replace generic category text with category-specific guidance:

- what each posture row means
- why secrets are never displayed
- why environment variables are preferred for provider credentials
- why encryption setup/change/disable is deferred
- what each recovery action does

### Error Handling

- Missing config or malformed config should render safe defaults, not crash.
- Secret-looking values must never be printed in full.
- Counts and status labels must be stable even if provider config values are absent, non-mapping, or partially configured.
- Running the privacy check should keep using a stable config snapshot so live config changes do not race the worker.

## Non-Goals

- Do not implement encryption enable/disable/change-password flows.
- Do not edit provider secrets directly from this category.
- Do not display raw API keys, tokens, passwords, secrets, or credential values.
- Do not move provider/model runtime selection out of `Providers & Models`.
- Do not add sync, server handoff, or workspace execution behavior.

## ADR Check

ADR required: no

ADR path: N/A

Reason: This slice presents existing privacy/config state and adds recovery navigation while preserving the current credential/encryption service boundary. It does not introduce a storage schema, sync/conflict policy, data ownership change, provider/runtime boundary, security policy, dependency, or long-lived application structure.

## Verification

Automated verification:

- helper tests for privacy posture calculation and redaction
- mounted Settings tests for Privacy & Security rendering
- mounted Settings tests for recovery navigation
- existing privacy check worker tests
- `git diff --check`

Manual verification:

- actual Textual-web/CDP screenshot of `Settings > Privacy & Security`
- user approval of the rendered screen before PR
