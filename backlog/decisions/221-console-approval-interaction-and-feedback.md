# ADR 221 Console approval interaction and feedback

Date: 2026-10-05

Status: Accepted after owner approval of the reviewed spec and continuation into implementation planning.

Design: [Console approval clarity and responsiveness](../../Docs/superpowers/specs/2026-10-05-console-approval-ux-and-responsiveness-design.md).

Related: ADR-032, ADR-067, ADR-090, ADR-093, ADR-150, ADR-161, ADR-195, ADR-210.

Implementation: [plan](../../Docs/superpowers/plans/2026-10-05-console-approval-ux-and-responsiveness.md); existing tasks TASK-34564 through TASK-34569.

## Decision

The Console approval card presents the action and target before technical arguments. A single call offers immediate Allow once and Deny; broader grants move under More options with an explanation and commit action. Until Chatbook exits labels the existing temporary lifetime, but its explanation comes from the provider: ordinary MCP-backed grants use an exact permission profile across chats, while raw-shell grants cover future commands in this Console chat and clear on Disarm or exit. Persistent Default grants also explain their existing inheritance into named profiles. No grant domain is broadened to fit generic copy.

Ordinary batches offer counted immediate one-time approval and denial when supported. Bulk approval excludes individual-review requests and never falls back to a broader grant. Mixed batches visibly stage row choices and submit one complete map with a final count and broader scopes. Ordinary rows retain Once initially; raw shell retains Deny but needs a deliberate gesture, including a deliberate re-selection of Deny. Deny all remains an explicit negative action for eligible mixed batches. Grouped copy names every captured call or distinct argument set rather than falsely describing one call.

Withhold Remember these inputs in repeated-tool batches whose shared stamp cannot honor independent exact-input choices. Explain this limitation and retain one-time approval; do not migrate stamp ownership in this work. Matching rules use canonical JSON of original captured arguments, not backend-normalized or redacted display text. More options eligibility considers the full batch, not just each row's nominal options.

A committing gesture locks duplicate submissions and paints feedback immediately. A newer authoritative state can replace Applying before the first paint; feedback must not introduce a minimum display delay or defer worker release. Controller receipt, host settlement after cancellation and timeout arbitration, and grant application are distinct owner-reported observations. Do not infer them from the resolver's or best-effort writer's void return. Failed remembering is reported honestly without altering the existing current-call execution behavior or automatically re-prompting.

Early dispatch uses Starting tool, not a claim that a backend has started. Tool running needs the execution owner's observation. Route late feedback to the owning tool or run status, not a replacement card; older observations cannot overwrite newer states. These display facts cannot grant access, keep a resolved card alive, or delay FIFO promotion.

The controller and interrupt host remain the sole decision authorities. Preserve gesture-generation and round-id guards, FIFO routing, current permission-profile binding, risk floors, hooks, file authority, exclusions, kill switch, definition checks, cancellation and deadlines. Alt+A lands on a neutral summary or Details control, never an approval action. More options and Details are card-owned disclosures; closing them does not decide the request or change its answerable-time clock. Retain stable focus, row identity, staged choices and scroll. Complete redacted Details is lazy and rendered in bounded pages rather than eagerly mounting large bodies or target lists.

Measure complete request availability, local checks, actionable paint, input arrival before handler queueing, feedback, receipt, settlement, grant application, worker release, dispatch, backend start and provider continuation separately. Also measure the complete last-output-to-card gap. Diagnose blocking through sampled UI stacks. Browser input and paint use the browser clock; server spans use the server clock. Metrics crossing clocks need calibration and bounded error or remain unverified; server frame emission does not establish browser paint. Report transport-inclusive feedback separately. Local bounded diagnostics contain no tool or conversation content. The proposed 100 ms feedback and 200 ms card p95 targets are on recorded hardware, not current measured behavior.

This extends ADR-195's lifecycle presentation for decision acknowledgment while keeping the card as the decision surface. Styling and ownership continue under ADR-150, ADR-161 and ADR-210. Permission storage, grant lifetimes, provider wire contracts and configuration defaults are unchanged.

## Context

Users report cluttered permission cards, unclear approval targets and scopes, pauses before the card appears, and delays after answering. The current single-call UI duplicates immediate buttons with a scope selector and batch toolbar. Approve all changes selections without submitting and has exceptions its label does not explain. This session describes an app-lifetime grant, not a chat-lifetime grant.

Source inspection identifies intervals to measure, including undisplayed tool-call collection, preparation and guards, synchronous card projection, and execution or provider continuation. The advisory explanation already runs asynchronously and setting the approval Event wakes the wait immediately. Neither source inspection nor the diagnostic launches stopped by storage recovery establish a measured cause for users' pauses.

## Alternatives

| Alternative | Reason not selected |
| --- | --- |
| Rename controls without changing the interaction | Leaves redundant controls and no distinction between click receipt, runtime acceptance and later work. |
| Keep broad grants beside the common actions | Adds choices and scope ambiguity to every request; the user chose More options. |
| Change temporary permission to this chat | The user retained the current lifetime and requested an explicit Until Chatbook exits label. |
| Make bulk approval choose the nearest available scope | Can grant broader access than a one-time label promises. |
| Display approval success as soon as the button is pressed | A stale or cancelled round can reject that gesture; the claim needs runtime acknowledgment. |
| Request an AI explanation before displaying the card | Adds optional model latency to a decision path and contradicts ADR-090's asynchronous advisory boundary. |
| Reduce wait polling or skip permission checks to improve speed | The Event wakes on an answer already; enforcement must not be weakened to mask a measured bottleneck. |

## Consequences

The presentation must receive effective legal choices and the captured authority needed to explain scope. The runtime exposes a minimal noncontrolling acknowledgment for the exact round, and mounted tests distinguish applying, accepted, rejected and execution states. More options and grouped-call presentation need stale-gesture protection just as the primary buttons do.

There is no new database schema, permission-store format, external telemetry, model call, permission lifetime or partial-batch execution. Exact performance fixes depend on the isolated baseline and belong in the subsequent implementation plan. Targeted automated and live checks qualify behavior, rendering and timing separately; full-suite execution still requires explicit user opt-in.

The owner accepted the reviewed spec and requested continuation into planning. The existing implementation tasks link this decision; their execution plans and final notes must retain that link. Plan review and execution-method selection still precede product implementation.
