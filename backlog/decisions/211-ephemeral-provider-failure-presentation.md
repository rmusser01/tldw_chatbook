# ADR-211: Carry provider failure presentation separately from diagnostic custody

Status: Accepted
Date: 2026-10-03
Related task: [TASK-33664](../tasks/task-33664%20-%20Keep-Console-Resend-out-of-empty-startup-imports.md)
Related plan: [Agent orchestration burndown](../../Docs/superpowers/plans/2026-09-29-agent-orchestration-burndown.md)
Applies alongside: ADR-063, ADR-179 and ADR-200.

## Decision

The gateway keeps known provider failures in their existing semantic exception
classes, with content-free default diagnostic message, original status/provider
and rate-limit retry data. Only its already sanitized queue presentation crosses
to the Console in a dedicated in-memory exception field. The existing diagnostic
description remains the source for durable agent error steps.

When AgentService contains that exception, its RunOutcome may carry the same
sanitized presentation separately from its diagnostic steps. This field is
keyword-only and excluded from repr. It is transient UI data: SQL rows, run logs,
manifests, provider history, assistant content, retained child transcripts and
fallback decisions do not serialize or consume it. The Console's existing
visible failure projection consumes it only for failed outcomes. A stuck/budget
outcome and errors without presentation retain their existing behavior.

The presentation comes from the gateway's existing bounded credential sanitizer,
provider display-name and selected-model recovery functions. It never copies a
provider response body or raw exception message through this field. This adds
no persistence schema, owner, permission, authority, import module, dependency,
retry/fallback policy or performance ceiling.

## Context and evidence

At composed immutable762a7f7, the exact prior typed projection discarded the
incoming TASK33922 display/model/400/404 recovery text stored beside it in the
queue. The original gateway selection was8PASS/1FAIL. Direct Console and the
default primary-agent path consequently showed only default diagnostic prose.
Actual admitted-profile tests reproduce missing visible copy for400/401/404/429.
AgentService persists describe_stream_failure in STEP_ERROR, so changing the
exception message would also change durable diagnostics.

## Alternatives

Replacing the typed exception with a generic wrapper would erase fallback and
refusal semantics. Putting presentation in str/message would change audit
custody. Reconstructing copy from mutable settings after failure could name a
different model/provider from the failed request. Storing UI copy on a diagnostic
step would persist it. A separate transient field retains the frozen, sanitized
gateway projection without changing those established contracts.
