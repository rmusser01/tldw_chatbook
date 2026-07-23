# Token budget for the agent RunBudget (TASK-326)

**Date**: 2026-07-23
**Status**: Approved design, pending implementation plan
**Base**: origin/dev @ `c10b5eb9e` (contains 322 + 320/321/325)
**Backlog**: TASK-326 (harness-review stream; no listed deps)

## Why

The agent runtime bounds steps, model-turns, wall-clock, and sub-agent count
(`agent_models.py` `RunBudget`) but has **no token budget** — a grep for
`token_budget`/`cost_budget`/`max_tokens` in `Agents/` returns nothing. A run on
a large-context model with big, growing tool-loop transcripts can burn
substantial tokens within the model-turn cap with no spend ceiling and no
accounting to stop on. This is the documented complement to task-322: **322
bounds the conversation history *into* the loop at initial dispatch; 326 bounds
the loop *itself*** — the tool-call/tool-result messages that accumulate across
iterations within a single turn. Providers already return usage on the
runtime's non-streaming calls, so it can be tracked cheaply.

## Decisions (user-approved)

1. **Hybrid token source.** Use the real `resp["usage"]` (prompt+completion)
   when the provider reports it; fall back to the task-321 `token_counter`
   estimate when it doesn't (some local providers). Accurate where possible,
   and always counts *something* so the budget can't be silently defeated.
2. **Tokens only** (`max_total_tokens`). AC#1 says "max_total_tokens (and/or
   cost)"; a dollar-cost budget needs a per-model price table that doesn't
   exist yet (`model_capabilities` has no pricing) — a clean follow-up.
3. **Sentinel default `max_total_tokens = 0` = unlimited**, so default runs are
   byte-identical until a consumer opts in (mirroring how `max_model_turns` is
   unreachable at defaults).

## Architecture

The agent loop `run_agent_loop` (`agent_runtime.py`) is a **pure function**
driven by `deps.call_model(messages, active) -> ModelTurn`; the existing caps
are checked at the top of each iteration, each producing `add(STEP_ERROR,
summary="… budget exhausted")` + `return RunOutcome(RUN_STUCK, …)`. The token
budget mirrors this exactly, so AC#3 (clear terminal status/step, not silent
truncation) falls out of the existing pattern. Three small units:

### Unit A — `ModelTurn` carries its token cost (`agent_models.py`)

Add one field to the frozen `ModelTurn` dataclass:

```python
@dataclass(frozen=True)
class ModelTurn:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    assistant_message: dict | None = None
    tokens: int = 0   # prompt+completion tokens for THIS provider call (0 = unknown)
```

`tokens` is the total (prompt + completion) tokens attributable to that single
provider call. Default `0` keeps fence-protocol/test turns and any construction
site that doesn't set it valid and non-contributing. This is the seam that lets
the pure loop accumulate spend without knowing the source.

### Unit B — `_make_call_model` populates `ModelTurn.tokens` (`agent_service.py`)

`_make_call_model.call_model` already does `resp = self.chat_call(..., streaming=
False, model=config.model, ...)` and extracts `text`/message. After that, derive
the turn's tokens with a hybrid source and set `tokens=` on **both** return
paths (non-native `ModelTurn(text=text, tokens=…)` and the native
`ModelTurn(text=…, tool_calls=…, assistant_message=…, tokens=…)`):

```python
def _usage_total_tokens(resp) -> int | None:
    """Prompt+completion tokens from a provider's OpenAI-shaped usage block,
    or None when the provider didn't report usage."""
    try:
        usage = resp["usage"]
    except (KeyError, TypeError):
        return None
    if not isinstance(usage, dict):
        return None
    total = usage.get("total_tokens")
    if isinstance(total, int) and total > 0:
        return total
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if isinstance(prompt, int) and isinstance(completion, int):
        return prompt + completion
    return None
```

In `call_model`, after building `payload` and getting `resp`/`text`:

```python
tokens = _usage_total_tokens(resp)
if tokens is None:
    # Provider reported no usage — estimate from what we sent + received.
    tokens = count_tokens_messages(payload, config.model) + estimate_tokens(
        text, config.model
    )
```

`payload` is the full sent message list (system + accumulated history), so
`count_tokens_messages(payload, model)` estimates the prompt and
`estimate_tokens(text, model)` the completion — summing to this call's spend.
(Agent payloads are text-only — tool calls/results are JSON/text strings — so
`count_tokens_messages`'s string-content assumption holds; no multimodal
flattening is needed here.)

### Unit C — `RunBudget.max_total_tokens` + accumulation + terminal stop

**`agent_models.py`:**
- Add `max_total_tokens: int = 0` to `RunBudget` (0 = unlimited).
- `clamp_child_budget` passes it through unchanged (`max_total_tokens=
  child.max_total_tokens`) — each run, parent or sub-agent, carries its own
  token budget, exactly as `max_steps` is per-run. (Sub-agent spend counts
  against the *sub-agent's* budget, not the parent's; cross-run token
  accounting is out of scope.)
- Add `total_tokens: int = 0` to `RunOutcome` for observability (the run's
  measured spend), answering the description's "no accounting" gap. The cap is
  the AC; this is the cheap accounting complement.

**`agent_runtime.py` `run_agent_loop`:**
- Initialize `total_tokens = 0` beside `model_turns = 0`.
- After each model turn (`turn = deps.call_model(...)`; `model_turns += 1`),
  add `total_tokens += turn.tokens`.
- Add a top-of-loop check alongside the step/model-turn/wall-clock checks:
  ```python
  if budget.max_total_tokens and total_tokens >= budget.max_total_tokens:
      add(STEP_ERROR, summary="token budget exhausted")
      return RunOutcome(RUN_STUCK, steps, subagents_spawned=spawned,
                        total_tokens=total_tokens)
  ```
  The `budget.max_total_tokens and …` guard makes `0` a true "unlimited"
  sentinel that never trips.
- **Thread `total_tokens=total_tokens` into every `RunOutcome(...)` return** in
  the loop (done, stuck, cancelled, other caps, loop-detection), so the spend
  is always reported. Checking at the *top* (before the next call) means a run
  that produces its final answer on the turn it crosses the budget still
  returns `RUN_DONE` — the budget stops the loop from making *more* calls, it
  does not retroactively fail a completed answer (AC#2/#3).

## Testing (AC#4)

- **Loop budget exhaustion (pure, `test_agent_runtime.py` style):** inject a
  fake `call_model` returning tool-calling `ModelTurn(text=…, tokens=N)` turns;
  `RunBudget(max_total_tokens=small)`; assert the run returns `RUN_STUCK` with a
  final `STEP_ERROR` summary `"token budget exhausted"`, and the reported
  `RunOutcome.total_tokens` reflects the accumulation.
- **Sentinel:** `max_total_tokens=0` never trips — a run whose turns carry large
  `tokens` still completes normally (`RUN_DONE`).
- **Boundary:** a run that crosses the budget on the SAME turn that yields the
  final answer completes `RUN_DONE` (stop-the-loop, not fail-the-answer).
- **Accounting:** `RunOutcome.total_tokens` equals the sum of the turns' tokens
  on a normal `RUN_DONE` run.
- **Hybrid source (`_usage_total_tokens` + call_model, `test_agent_service.py`
  style):** a `resp` with `usage.total_tokens` uses it; a `resp` with only
  `prompt_tokens`+`completion_tokens` sums them; a `resp` with no/`malformed`
  usage returns `None` and the call_model estimate fallback yields `tokens > 0`.
- **Child budget:** `clamp_child_budget` preserves `max_total_tokens`.

## Out of scope

- Dollar-cost budgets and per-model price tables (follow-up; AC#1 "and/or").
- Cross-run/parent-inclusive token accounting for sub-agents (each run has its
  own budget).
- Streaming-usage capture — the agent runtime calls providers with
  `streaming=False`, so usage is already on the response.
- Surfacing/enforcing the budget in any specific consumer UI; `CONSOLE_RUN_
  BUDGET` may set a real `max_total_tokens` later, but this task ships the
  mechanism with the safe unlimited default.
