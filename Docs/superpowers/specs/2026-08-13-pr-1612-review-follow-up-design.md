# PR 1612 Review Follow-Up Design

## Context

PR #1612 merged after rebasing onto `dev`. Two inline Qodo findings arrived
immediately after the merge: the shared hosted transport can expose its private
base-URL validation exception, and Z.ai's adapter-only retry-delay fallback
differs from the canonical Console/config default.

This is a correction to TASK-15676 and its existing ADR-063 boundary, not a new
feature or architectural decision.

## Design

`owned_json_post()` will catch only `HostedChatBaseURLValidationError` from its
existing normalizer and translate it to the same context-free
`ChatProviderError` used for invalid transport configuration. The malformed URL
and any embedded credentials remain absent from the public error.

`resolve_zai_request()` will use `5.0` seconds when no retry delay is configured,
matching both the shipped `[api_settings.zai]` default and Console's hosted
transport policy. Explicit configuration and explicit call arguments retain
their existing precedence.

Two narrow regression tests will pin the exception type/redaction and the Z.ai
fallback. No shared retry-default abstraction or unrelated provider change is
introduced.

## Verification

- Run the two new tests RED before production changes and GREEN afterward.
- Run the complete hosted Chat and Z.ai unit modules.
- Run Ruff lint/format checks and compile the two touched production modules.
- Use GitHub checks and resolve both original PR review threads before merging
  the follow-up PR.

## ADR Check

ADR required: no new ADR

ADR path:
`backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md`

Reason: this directly restores the typed-error and provider-default contracts
already established by ADR-063; it does not change a boundary or policy.
