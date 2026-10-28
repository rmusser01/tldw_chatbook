# ADR-252: Preserve tool availability refusals before hook preparation

Status: Accepted
Date: 2026-10-28
Related task: TASK-37146
Related decisions: ADR-104 (replaceable tool working sets), ADR-078 (tool outcome provenance), ADR-154 (authoritative denial facts)

## Context

An allowed catalog tool without a loaded schema needs reload guidance, not an authorization-denial message. With hooks v2 enabled, definition resolution rejects that call during preparation before the ordinary dispatch gate, and the runtime hides every preparation exception behind `hook: preparation refused`. Native and fenced real-service regressions reproduced that diagnostic loss.

## Decision

The injected `LoopDeps.prepare_hook_call` callback may return either the existing prepared `ToolCall` or a host-authored `ToolResult` that refuses before preparation. The refusal must be unsuccessful, carry `blocked` and `not_started` provenance, and contain no authoritative approval decision. The runtime preserves its nonempty error with a fixed `ERROR:` prefix so refusal data cannot equal the guard's reserved `proceed` verdict; it uses that text as the blocked call result, skips acceptance/review/dispatch for that call, and retains coherent batch history. Successes and malformed or authority-bearing refusal results fail closed with the existing generic preparation error.

The service checks catalog availability before calling the hook preparer. Runtime schemas retain their existing independent gates. An excluded name or an allow-listed name absent from the actual registry retains the generic refusal; only an allowed, registered catalog name without a loaded schema receives reload guidance. This does not load schemas, run hooks, approve calls, or grant authority.

Ordinary exceptions from preparation remain generic. Never forward arbitrary exception text or infer permission/denial facts from error strings. Actual hook preparation, fresh identity validation, required checkpoints and dispatch for eligible calls stay unchanged.

## Alternatives

- Forward preparation exception text: rejected because it can contain private or untrusted data and loses provenance.
- Prepare hooks for undisclosed tools: rejected because it runs handlers for calls that cannot execute.
- Automatically load the guessed tool or accumulate schemas: rejected because it changes ADR-104 disclosure and budgeting behavior.
- Add another runtime callback or exception hierarchy: unnecessary; the existing preparation boundary and ToolResult already represent the refusal.

## Verification

Targeted tests cover native/fenced hooked recovery, unavailable and unknown allow-listed names, typed refusal propagation, invalid/successful/authority-bearing preparation returns, generic exception redaction, unchanged transformation/approval ordering, and no early dispatch. No storage, provider-wire, permission-store, or dependency changes are introduced.
