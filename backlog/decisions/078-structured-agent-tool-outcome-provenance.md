# ADR-078: Structured agent tool outcome provenance

Status: Accepted
Date: 2026-08-21
Related Task: [TASK-19426](../tasks/task-19426%20-%20Group-Console-tool-activity-inside-assistant-turns.md)
Supersedes: N/A

## Decision

Carry tool success, ordinary failure, and policy refusal as optional structured
facts across the internal agent provider/runtime step boundary, and use payload
classification only as a compatibility fallback for legacy or malformed steps.

The contract is:

1. `ToolResult.outcome` is an optional bounded internal value. Providers use
   `ToolResult.blocked(error)` for permission, approval, kill-switch, timeout,
   workspace-root, and other policy refusals. Ordinary dispatch, execution,
   lookup, and formatting failures remain `ToolResult(ok=False, error=...)`
   without blocked provenance.
2. Before flattening tool content or errors into `AgentStep.result`, the runtime
   writes `AgentStep.tool_outcome` using protocol facts: a non-`proceed` review
   verdict is `blocked`; a dispatched `ToolResult(ok=True)` is `success`; a
   dispatched `ok=False` result with blocked provenance is `blocked`; every
   other dispatched `ok=False` result is `failed`. `ok=True` is authoritative,
   so successful content that begins with `ERROR:` or equals denial copy cannot
   be mislabeled.
3. Live Console presentation and resumed marker reconstruction trust a valid
   `tool_outcome` value. A missing or unknown value falls back to the previous
   conservative result-text classifier, preserving old run records and failing
   safely for malformed per-step JSON.
4. `ConsoleActivityPresentation` remains session-only display metadata. It does
   not enter conversation persistence or provider history.

## Context

TASK-19426 originally derived Console activity status solely from the flattened
`AgentStep.result` string. That lost the `ToolResult.ok` fact: valid successful
payloads such as `ERROR: harmless successful payload`, or content equal to a
canonical denial message, collided with the bridge's legacy failure/refusal
grammar and rendered as `failed` or `blocked`.

The runtime already receives the structured `ToolResult` and any pre-dispatch
review verdict before it builds the display/log step. That is the last seam
where outcome provenance is unambiguous. Providers likewise know whether an
`ok=False` result represents policy refusal or ordinary execution failure; the
Console bridge should not recover that distinction from human-readable copy.

`AgentService._persist` serializes `AgentStep` with `dataclasses.asdict` into
the existing per-run `steps` JSON list stored in SQLite. Resume code consumes
those entries as dictionaries. Adding an optional key therefore needs no
SQLite migration or new storage table/column. Existing records omit the key,
and malformed values are ignored in favor of the legacy classifier. The
internal `ToolResult` dataclass is not an external LLM provider wire schema, so
no hosted-provider request/response migration is involved.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Keep parsing `AgentStep.result` and escape successful collisions | Payload text is untrusted tool output; escaping invents a second wire grammar and still cannot prove whether denial-looking content is data or policy. |
| Move all refusal-copy recognition into the runtime | This relocates string heuristics instead of preserving provenance and couples the pure runtime to Console presentation policy. |
| Persist `ConsoleActivityPresentation` with conversation messages | TOOL markers and their presentation are session-local display projections, not durable chat-tree nodes or provider history. Persisting them would change the wrong ownership boundary. |
| Add normalized SQLite columns or a run-log schema migration | The existing steps JSON list already carries additive dataclass fields. A migration would add storage machinery without improving compatibility or authority. |
| Make the new fields required | Old persisted runs and manual/test `AgentStep` constructors legitimately lack them. Optional defaults plus conservative fallback preserve compatibility. |

## Consequences

- Console status is derived from protocol facts for new runs, so successful
  payload text cannot impersonate failure/refusal control data.
- Provider adapters must mark policy refusals with `ToolResult.blocked`; plain
  `ok=False` continues to mean an ordinary failed dispatch.
- New persisted step dictionaries include `tool_outcome` through the existing
  `asdict` path. Old and malformed records remain readable without migration.
- The bridge retains refusal-string parsing only as a compatibility floor for
  steps that lack a valid structured outcome; new runtime steps must not rely
  on it.
- The decision changes an internal provider/runtime interface but does not
  change external provider payloads, the run-log record schema, conversation
  message persistence, or SQLite layout.

## Links

- [TASK-19426 — Group Console tool activity inside assistant turns](../tasks/task-19426%20-%20Group-Console-tool-activity-inside-assistant-turns.md)
- [Console Assistant Turn Grouping Design](../../Docs/superpowers/specs/2026-08-21-console-assistant-turn-grouping-design.md)
- [Console Assistant Turn Grouping Implementation Plan](../../Docs/superpowers/plans/2026-08-21-console-assistant-turn-grouping.md)
