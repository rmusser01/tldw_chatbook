# ADR-080: Advisory context summaries on Console permission requests

Status: Accepted
Date: 2026-08-31
Related Task: Implementation tasks will be created from the approved design during planning.
Related: ADR-069, ADR-032

## Decision

Console approval cards gain two advisory, display-only context lanes:

1. A per-row **model context** line carrying the working model's rationale for
   the call, captured passively at parse time — an explicit `rationale` key in
   the fence tool-call protocol when present, otherwise the model's preamble
   text (fence visible text, or the native turn's text). This lane is
   unconditional, needs no network, and adds one optional `ToolCall` field
   that is excluded from every persisted serialization.
2. A card-level **summary** line: one short paragraph per approval round,
   produced by an external fast LLM whose provider and model the user
   designates in a new `[permission_summary]` config section. The trigger is a
   user-configured mode (`off` / `fallback` / `always`), default `off`; the
   call is one-per-round, executed synchronously on its own thread off the
   approval wait loop, and delivered asynchronously. Late or failed results
   are dropped silently; the card and its auto-deny deadline are never blocked
   or extended.

Summaries and rationales are untrusted content. They never alter reason codes,
risk badges, path-precheck warnings, decision options, defaults, deadlines, or
any verdict; they are never persisted, synchronized, logged with content, or
fed back to the working model. The external call's prompt is bounded to stored
user/assistant visible text plus redacted tool-call information; automatic
project-instruction bodies never reach it (per ADR-069). Both lanes are
length-capped, control-stripped, plain-text rendered, and visually subordinate
to the machine-owned fields. The default summarizer prompt is neutral and
forbids approve/deny recommendations.

Rationale flows through the existing pending-call chain (`ToolCall` →
`MCPPendingCall` → wire payload → `ChatApprovalCard`), covering all three tool
owners (MCP, builtin, local). The summary is carried in the payload (source of
truth, so re-mounts re-render it) and patched live by a guarded
`set_summary(round_id, text)` that never re-runs `set_batch` and so cannot
clobber in-progress row decisions.

## Context

Approval cards today show only machine-owned data (tool name, reason code,
redacted arguments). The model's stated intent is discarded at parse time
(`split_visible_text_and_tool_call` returns it; the loop drops it; native turn
text sits unused beside `tool_calls`). Users approve or deny without the
"why". Two cheap sources exist: text the working model already emits, and —
when richer synthesis is wanted — a one-shot call to a fast model the user
designates, for which the repo already has per-feature provider+model
precedents (`[analysis_defaults]`, the auxiliary completion gateway).

ADR-069 established the governing principle: optional context preparation is
kept strictly separate from the security-review boundary and its failure
posture. This design keeps both lanes entirely outside `review_tool_calls`'
verdict mechanics — they are decoration on the decision surface, never inputs
to it.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Compute summaries inside `review_tool_calls` before card mount | Puts a network call inside the security-review boundary, delays every approval, and eats the auto-deny deadline — the coupling ADR-069 exists to prevent. |
| Card widget requests summaries itself | Widgets making network calls breaks the controller/screen worker pattern and complicates headless testing. |
| External-only summaries (no model-authored lane) | Loses free context the working model already produces; the passive lane is unconditional and costs nothing. |
| Preamble harvest only (no fence `rationale` key) | Leaves fence-protocol users without an explicit, per-call precise channel. |
| Explicit fence key only | Native tool-calling — the common path — would get nothing. |
| Per-row external summaries | N calls per round for marginal gain; one batch-level summary bounds cost. |
| Separate `enabled` flag plus trigger mode | `mode = "off"` already expresses disabled; a second flag is redundant state. |

## Consequences

- `ToolCall` and `MCPPendingCall` gain one optional field each; the three
  pending-row constructors copy it; the wire payload gains `rationale` per row
  and `summary` at card level.
- The fence parser accepts exactly one new top-level key; all other unknown
  keys remain dropped. Wrong-typed values are ignored, never fatal.
- A new sync service module owns config resolution, prompt assembly, the call,
  and output normalization; it never raises across its public API and is
  testable with an injected call function (no network in tests).
- Enabling the external path discloses bounded conversation content to the
  designated endpoint. This is an explicit, opt-in user choice; the settings
  copy states it.
- Residual risk: an untrusted summary line can still attempt to influence the
  user. Mitigated by neutral prompting, labeling, display caps, and visual
  subordination to machine-owned fields; accepted, not eliminated.
- Same-name calls still collapse into `×N` rows because the wire payload drops
  `call_id`; grouped rows show the first non-empty rationale. Fixing that is a
  documented, separable follow-up.
- No schema migrations, no sync-visible state, no new persisted data anywhere.

## Links

- [Approved design spec](../../Docs/superpowers/specs/2026-08-31-permission-request-summaries-design.md)
- [ADR-069: Console project-instruction local state and preflight](069-console-project-instruction-local-state-and-preflight.md)
- [ADR-032: Local agent tool permission boundary](032-local-agent-tool-permission-boundary.md)
