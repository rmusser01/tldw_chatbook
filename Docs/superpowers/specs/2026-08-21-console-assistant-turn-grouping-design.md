# Console Assistant Turn Grouping Design

**Task:** TASK-19426
**Date:** 2026-08-21
**Status:** Approved interaction direction; implementation pending

## Goal

Make the Console transcript read as a causal conversation. A user query is followed by one visually coherent Assistant turn. Any thinking summaries, tool calls, results, approvals, diffs, and the final answer produced during that run appear inside that Assistant turn instead of as neighboring top-level messages.

The final answer remains immediately readable. Operational detail starts collapsed and expands only when the user asks for it.

## Approved Interaction

The transcript renders this hierarchy:

```text
You
<query>

Assistant
  ▸ Thinking                                      done
  ▸ fs_list · workspace root                     success
  ▸ Thinking                                      done

  <final answer>
```

The Assistant label, activity disclosures, and answer share one containing surface. This is not a separate activity card followed by an assistant message, and it is not an answer row followed by detached Tool rows.

Each activity disclosure is collapsed by default. Its one-line header identifies the activity and terminal status. Expanding a row reveals the existing preview/full-output content and any associated diff. Expansion state is independent per activity row and remains view-only.

## Turn Ownership

The persisted conversation tree remains unchanged. TOOL markers stay display-only messages and never become tree nodes, parents, or durable chat messages.

The transcript derives Assistant turns from the existing active-path view:

1. An ASSISTANT message starts a turn.
2. Contiguous TOOL markers following that message belong to that turn.
3. The next USER, SYSTEM, or ASSISTANT message closes the turn.
4. Orphan TOOL markers with no assistant owner remain visible as standalone system activity rather than being silently attached to the wrong answer.
5. Resume-derived markers use the same grouping because `inject_resume_agent_markers` already anchors each primary run's marker block to its persisted assistant reply.

This preserves branch correctness: when the store drops markers whose assistant anchor is off the active branch, the transcript cannot misattribute them to the visible branch.

## Transcript Components

### Assistant turn container

A new transcript-level container owns one assistant message plus its activity markers. It renders:

1. Assistant header and terminal state.
2. Zero or more activity disclosures.
3. Assistant response body and footer.
4. Assistant message actions, citations, attachments, generated media, and variant controls that already belong to that response.

The container is keyed by the assistant message id. Streaming updates sync the answer body in place. Adding a new tool marker may rebuild only that turn's activity stack; unrelated turns retain their mounted widgets and scroll position.

The new turn/disclosure widgets live in a focused Console widget module rather than adding another subsystem to the already-large `console_transcript.py`. The transcript remains responsible for derivation, selection, reconciliation, and pruning; the child widget is responsible only for rendering and locally syncing one turn.

### Activity presentation contract

Collapsed headers must not parse user-facing marker strings. Live and resume marker builders attach a session-only `ConsoleActivityPresentation` value to each display-only TOOL message. It carries a bounded enum-like kind, a literal label, and a terminal status (`success`, `blocked`, `failed`, or `done`). Existing `content`, `tool_output_full`, `tool_diff`, and `change_review_run_id` fields remain the detail/action payloads.

The presentation value is never persisted in the conversation database and never enters provider history. Every known TOOL-marker builder attaches it: step results/spawns/errors/timeouts, task snapshots, live and resumed change summaries, concurrent/sub-agent change notices, change-tracking failures, and live/resumed diff-feedback disclosures. Only truly legacy or unknown markers use the neutral fallback. `AgentStep` has no outcome field, so step-driven live and resume builders use one bridge-owned `classify_activity_status(step_kind, result)` helper before constructing the presentation value:

- a successful `STEP_TOOL_RESULT` is `success`;
- a `STEP_TOOL_RESULT` is compared directly against the Console controller's review-hook denial and global kill-switch verdicts, because those pre-dispatch refusals reach the step without an `ERROR:` envelope; a match is `blocked`;
- a failed dispatched `STEP_TOOL_RESULT` first removes the runtime's exact `ERROR:` envelope, then compares the remaining error to the canonical built-in, local, and MCP denial/timeout/kill-switch refusal constants; a match is `blocked`;
- every other `ERROR:`-wrapped `STEP_TOOL_RESULT` is `failed`;
- `STEP_APPROVAL_TIMEOUT` is `blocked`, `STEP_ERROR` is `failed`, and non-tool activity is `done`.

This classifier reads the agent-step protocol result, not the rendered `⚙ … → …` marker string. The order is material: `agent_runtime` emits controller review refusals directly but wraps unsuccessful dispatched `ToolResult` values as `ERROR: <provider refusal>`. Treating all non-enveloped results as success would mislabel the former; checking for generic errors before unwrapping would mislabel the latter. Tests use the controller's and providers' exported refusal constants in their actual direct/enveloped shapes so copy drift cannot silently change status. No `AgentStep`, run-log, or persisted-step contract changes. Unknown or legacy transcript markers with no derived presentation value receive a neutral `Activity · done` header while preserving their original content; they are not hidden or guessed into a tool identity.

### Activity disclosure

Each completed tool execution becomes one compact disclosure inside its owning Assistant turn. The existing bridge emits a transcript marker at `STEP_TOOL_RESULT`; that marker represents the call and its terminal result together. `STEP_TOOL_CALL` does not create a second transcript row, and this task does not add a speculative live `running` row or a call-state machine.

The collapsed header uses the structured presentation value and names the tool or activity plus `success`, `blocked`, or `failed`. The body carries the existing result preview or full result. Approval timeouts and execution errors remain their own terminal activity disclosures when they do not produce a normal tool-result marker.

The existing `o` full-output action and mouse/Enter disclosure toggle converge on the same per-message expansion set. File-write diffs render inside the expanded disclosure. Selecting a tool marker for inspector/actions remains supported.

### Reasoning disclosure

The agent bridge adds one `Thinking` activity marker for every safe intermediate primary-agent `STEP_MODEL` whose following primary step proves that model turn initiated tool work. The proving step may be `STEP_TOOL_CALL`, `STEP_SPAWN`, or a direct `STEP_TOOL_RESULT` produced by a pre-dispatch review/continuation refusal. It must not create a marker for the final answer turn. Live rendering buffers the primary `STEP_MODEL` until the next primary step determines whether it led to tool work; resume rendering performs the equivalent look-ahead over persisted steps. This makes live and resumed marker order identical without changing the agent-step schema.

Reasoning disclosure does not expose hidden chain-of-thought. It may show only a safe, already-visible intermediate preamble from `STEP_MODEL.summary`. Sanitization is conservative: strip tool-call fence payloads, reject thinking-tag/provider-private-reasoning shapes, flatten control markup, and apply the existing display cap. Provider-private reasoning content is never inferred or surfaced.

When no safe summary remains, render `Thinking · done` as a non-expandable status row. It must not show a chevron or accept a toggle that reveals nothing. When a safe summary exists, the row is a collapsed disclosure and expands to that summary. `Thinking` describes that an intermediate model turn occurred; it never promises raw reasoning.

Live and resumed runs must derive the same Thinking marker order from the recorded primary-run steps. Sub-agent internals remain in the run inspector; they are not mixed into the primary Assistant turn.

## Rendering and Interaction Details

- Assistant turns use one left ownership accent and shared surface; nested disclosures use quieter borders.
- The answer is visually separated from activity by spacing or a subtle rule, but remains inside the same block.
- Activity headers are keyboard focusable and toggle with Enter/Space through Textual's native disclosure behavior.
- Visual-order selection is `USER -> owned activity rows -> assistant answer`. `j/k` uses that derived order even though the store's causal sequence remains `USER -> ASSISTANT placeholder -> TOOL markers`.
- Selecting an activity highlights its disclosure header and renders its existing contextual action row immediately after the header, outside the hidden detail body. This keeps actions reachable while collapsed. Clicking or pressing Enter/Space on the header both selects the activity and toggles its detail. Collapsing does not clear selection.
- The transcript `o` binding and the disclosure toggle use the same per-message expansion set. A selected tool row therefore expands/collapses identically by mouse, Enter/Space, action button, or `o`. The Inspector continues resolving the original TOOL message id.
- Selecting the assistant response highlights the answer region/actions, not every activity row in the containing surface.
- No new screen-wide keybinding or footer hint is introduced. Existing transcript `o` behavior remains truthful under ADR-031.
- User, system, and orphan activity messages continue to render as independent transcript rows.
- Empty, streaming, stopped, and failed assistant bodies retain their existing copy and status semantics.
- Collapsed state is the default on initial render, session switch, and resume. Expansion is not persisted.

## Data Flow

```text
Agent runtime steps
  -> ConsoleAgentBridge display-only activity markers
  -> ConsoleChatStore active-path view (assistant + anchored TOOL markers)
  -> ConsoleTranscript turn derivation
  -> AssistantTurn container
       -> collapsed ActivityDisclosure rows
       -> visible assistant answer
```

No database migration, provider contract change, or persisted message-role change is required.

## Error and Edge Cases

- A failed or blocked tool call stays inside the owning turn and uses a failed/blocked header; its diagnostic body remains expandable.
- An approval timeout remains explicitly identified as auto-denied and not run.
- An agent run with tools but no final text still renders its Assistant container, activity rows, and the existing `No response was generated.` fallback.
- A streaming answer grows below the activity stack without remounting unrelated turns.
- Rewinding, regenerating, deleting, or switching branches cannot transfer activity markers to a different assistant response.
- Pruning removes a whole Assistant turn as a group; it never leaves an orphaned activity disclosure or answer fragment.
- Pruning walks top-level display units instead of assuming every message row is a direct transcript child. An Assistant turn is protected when its answer is streaming or any nested message is selected; committing the prune adds every message id owned by that turn to the pruned-id set in one operation.
- Plain-text export preserves causal order: user, Assistant heading, activity headers plus the existing bounded result previews, then final answer. It does not include hidden full tool output or diffs, and it is independent of ephemeral disclosure expansion state.

## Verification

Focused tests will cover:

- turn derivation for user/assistant/tool/tool/user sequences;
- tools rendered inside, and before the answer body of, their Assistant container;
- independent collapsed-by-default expansion;
- full-output and diff visibility through both disclosure and `o` paths;
- streaming updates preserving existing turn widget identity;
- completed, failed/stopped, empty-final, and resumed marker blocks;
- branch changes and pruning dropping whole owned groups;
- keyboard and selection behavior for assistant and tool messages;
- structured activity metadata for live, resumed, legacy, and unknown marker shapes;
- visual-order `j/k`, selected nested action placement, and `o` parity;
- plain-text export ordering;
- stylesheet parity between source partial and compiled stylesheet.

Live verification will run against an isolated scratch profile and inspect the real Console at supported wide and narrow terminal sizes. It will exercise a real local tool call, expansion/collapse, final-answer placement, and a resumed conversation.

## Implementation Isolation

The design was authored while the primary checkout was on an unrelated dirty video-generation branch. Implementation must not continue in that checkout. Create a dedicated `codex/` worktree from the latest fetched `origin/dev`, carry the TASK-16324 task/spec commits into it, and perform all code/test work there. This prevents unrelated user changes and feature history from entering the implementation diff.

## ADR Check

**ADR required:** no
**ADR path:** N/A
**Reason:** this is a focused transcript presentation and interaction change. It preserves existing storage, marker ownership, agent-step/run-log contracts, provider/runtime boundaries, security policy, and application navigation. Activity status is derived at the existing bridge presentation seam rather than added to the runtime contract. ADR-031 remains applicable for keybinding and footer-hint truthfulness.
