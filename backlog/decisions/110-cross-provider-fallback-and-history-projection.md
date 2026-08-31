# ADR-110: Cross-provider model fallback and history projection

Status: **Accepted**
Date: 2026-08-31
Accepted: 2026-08-31 — owner approved the full-fallback-with-projection option over the narrower first-call-only and same-family-only alternatives recorded below
Related Task: [TASK-25902](../tasks/task-25902%20-%20Agent-loop-cross-provider-fallback-chain.md)
Builds on: [TASK-25901](../tasks/task-25901%20-%20Agent-loop-classified-retry-with-bounded-backoff-on-model-errors.md) (in-loop retry — fallback is what happens when retry is exhausted)
Number swept against all remote refs and worktrees 2026-08-31; 108–110 were free (max in use: 106).

## Context

TASK-25902 asked for an ordered fallback provider list consulted when the
primary provider exhausts retries or returns a credit-terminal error. Its
acceptance criteria say "model-specific request shaping (tool schema, thinking,
caching) is re-resolved for the fallback provider rather than carried over".

Implementation found that understates the problem. **The accumulated message
history is structurally provider-shaped, not just the outgoing request.**

`Agents/agent_runtime.py:836-848` (`_append_tool_result`) branches on protocol:

- **Native protocol** (`call.call_id` set): the tool result is a `role="tool"`
  message paired to the assistant turn's `tool_calls` entry by `tool_call_id`.
- **Fence protocol** (`call.call_id` unset): the tool result is plain text
  appended as a **user-role** message with a `FENCE_TOOL_RESULT_PREFIX`.

`agent_runtime.py:1481` appends `turn.assistant_message` — the raw
provider-shaped assistant turn — into the same history.

Which branch applies is decided by `provider_supports_native_tools(api_endpoint)`
(`Agents/native_tools.py:63`), currently true for **openai, anthropic, google,
cohere** and false for every other provider.

So a mid-run switch from OpenAI to Groq hands the fence provider a history
containing `tool_calls` and `role:"tool"` messages it cannot interpret; the
reverse hands a native provider fence-prefixed user text where it expects
paired tool results. Neither fails loudly — they produce a confused model.

A second constraint sits in the same seam. `AgentService._make_call_model`
resolves `native` once at closure-build time and caches rendered protocol text,
with an explicit comment that "byte-stable repeated turns are the precondition
for provider-side prompt caching". A provider switch invalidates that by
construction.

## Decision

Cross-provider fallback is supported **at any turn, across any pair of
providers**, and the accumulated history is **projected into the target
provider's protocol** before the fallback request is made.

1. **Projection is a pure, total function of (history, target protocol).**
   `project_history_for_protocol(messages, native: bool) -> list[dict]` returns
   a history valid for the target. It never mutates its input and never drops a
   turn: every exchange present before the switch is present after it, in
   order.

2. **Native → fence.** An assistant turn carrying `tool_calls` becomes an
   assistant turn whose text contains the equivalent fenced call. Each paired
   `role:"tool"` message becomes the `FENCE_TOOL_RESULT_PREFIX` user message
   the fence protocol already uses. The pairing is resolved by
   `tool_call_id`; an unpaired `tool_calls` entry (a call whose result never
   arrived) projects as a fenced call with an explicit "no result recorded"
   marker rather than being dropped.

3. **Fence → native.** A fenced call in assistant text is parsed back into a
   `tool_calls` entry with a synthesised, run-unique `tool_call_id`, and its
   `FENCE_TOOL_RESULT_PREFIX` user message becomes the paired `role:"tool"`
   message. Assistant text that merely *looks* like a fence (the look-alike
   tags `agent_runtime.py:193,253` already guard against) is left as text.

4. **Projection is applied once, at the switch, to the whole history** — not
   incrementally per turn. After a switch the run continues in the target
   provider's protocol natively; there is no permanent translation layer.

5. **A failed projection aborts the fallback, not the run's integrity.** If the
   history cannot be projected faithfully, the fallback is refused and the
   original provider's error is raised. A confused model is worse than an
   honest failure.

6. **Prompt-cache state is invalidated on switch, deliberately.** The cached
   protocol text and any provider cache breakpoints are rebuilt for the target.
   The first post-switch request is a cache miss and that is correct — the
   alternative is sending another provider's cache markers.

7. **Readiness gates entry to the chain.** A fallback candidate that the
   existing per-provider readiness check reports as unconfigured is skipped
   without an attempt, and the skip is visible.

8. **The switch is never silent.** It emits a trace step naming the failing
   provider, the chosen provider, and the reason (retries exhausted, or the
   terminal class), and it is visible in the transcript.

9. **Fallback does not rebuild `LoopDeps`.** TASK-25913's review established
   that reconstructing deps mid-run resets the wall-budget origin its tool
   clamp reads from. Provider switching re-resolves the *model-call seam* only.

10. **Off by default.** With no fallback chain configured the behaviour is
    byte-identical to today, including the absence of any projection call.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| First-call-only fallback (before any tool results exist) | Safe and cheap, and it covers the common "provider down at run start" case — but it abandons a run that fails on turn 12, which is exactly when the accumulated work is most expensive to lose. Recorded here because it remains the correct fallback position if projection proves unreliable. |
| Same-family only (native→native, fence→fence) | Avoids the problem instead of solving it, and silently narrows a user's configured chain: a user listing `openai, groq` would get a chain that skips groq forever without saying so. Surfacing that honestly costs nearly as much as projecting. |
| Translate on every request rather than once at the switch | A permanent translation layer means every subsequent turn pays the cost and carries the risk, and it makes the native path permanently non-native. Projecting once at the switch keeps the steady state clean. |
| Restart the run on the fallback provider | Discards the tool results the fallback exists to preserve — it is the failure this task set out to prevent, with extra steps. |
| Let the provider reject the malformed history | Providers do not reliably reject it; they accept it and produce a confused model. A silent quality collapse is worse than an error. |

## Consequences

- **A new correctness surface.** Projection is the kind of code that is right
  for the cases someone thought of. It needs round-trip property tests
  (project native→fence→native and assert semantic equivalence), explicit
  coverage of unpaired calls, multi-call batches, and look-alike fences, and it
  must be exercised against a real second provider before release.
- **The first request after a switch is a cache miss**, so a fallback costs
  more than a normal turn. Acceptable: the alternative was losing the run.
- **`provider_supports_native_tools` becomes load-bearing.** It currently
  decides request shaping; it now also decides history shape at switch time. A
  provider added to that set without projection coverage is a latent bug, so
  the projection tests should be driven from that same list rather than a
  hand-copied one.
- **Fallback composes with retry, not instead of it.** Retry (TASK-25901)
  handles a provider that will recover in seconds. Fallback handles one that
  will not. The chain is only consulted after retries are exhausted or on a
  terminal credit/quota class.
- **Scope explicitly excluded here:** per-provider model *selection* within the
  chain (which model to use on the fallback provider) is out of scope; the
  chain entry names a provider and the existing model-resolution rules apply.

## Links

- [TASK-25902](../tasks/task-25902%20-%20Agent-loop-cross-provider-fallback-chain.md)
- `Agents/agent_runtime.py:836-848` — the protocol branch this ADR exists for
- `Agents/native_tools.py:63` — `provider_supports_native_tools`
- `Agents/agent_service.py` `_make_call_model` — the closure holding per-provider state
- ADR-063 — private continuation (a separate provider-specific history concern)
