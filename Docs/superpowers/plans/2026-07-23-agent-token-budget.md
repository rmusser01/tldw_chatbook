# Agent RunBudget Token Budget Implementation Plan (TASK-326)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the agent runtime a token-spend ceiling — `RunBudget.max_total_tokens` — that accumulates prompt+completion tokens per run and stops the loop cleanly (like the existing caps) when exceeded, with real provider usage when reported and a token_counter estimate otherwise.

**Architecture:** Three files. `agent_models.py` gains the data fields (`ModelTurn.tokens`, `RunBudget.max_total_tokens`, `RunOutcome.total_tokens`). `agent_service.py` populates `ModelTurn.tokens` from `resp["usage"]` or the task-321 estimator. `agent_runtime.py`'s pure loop accumulates the spend and adds a top-of-loop budget check mirroring the step/model-turn/wall-clock caps, reporting spend via an `_outcome` helper closure.

**Tech Stack:** Python, pytest. The runtime calls providers with `streaming=False`, so usage is on the response dict.

## Global Constraints

- **Worktree:** all work in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook-agent-token-budget` (branch `feat/agent-token-budget`, off `origin/dev @ c10b5eb9e`). Never touch the main checkout `/Users/macbook-dev/Documents/GitHub/tldw_chatbook`.
- **Test command (venv is in the main checkout):**
  `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook-agent-token-budget && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <path> -v`
- **Sentinel default `max_total_tokens = 0` = unlimited** — default runs must be byte-identical (the check is guarded `if budget.max_total_tokens and …`).
- **`max_total_tokens` is cumulative SPEND** (`Σ(prompt_i + completion_i)` across turns, since the growing prompt is re-sent each call), not a context-window size.
- New fields `ModelTurn.tokens` and `RunOutcome.total_tokens` are **trailing and defaulted (0)** — backward compatible with every existing construction.
- **Terminal on exhaustion:** `add(STEP_ERROR, summary="token budget exhausted")` then return `RUN_STUCK` — a clear step, not silent truncation. Placed **after** the existing wall-clock check.
- **Commit only the files each task names**, with explicit `git add <paths>` — never `git add -A`/`-am` (the repo has a tracked `.superpowers/` scratch area and other stray files).

---

## Task 1: Data-model fields (`agent_models.py`)

**Files:**
- Modify: `tldw_chatbook/Agents/agent_models.py` (`ModelTurn`, `RunBudget`, `RunOutcome`, `clamp_child_budget`)
- Test: `Tests/Agents/test_agent_models.py`

**Interfaces:**
- Produces: `ModelTurn(text, tool_calls, assistant_message, tokens: int = 0)`; `RunBudget(..., max_total_tokens: int = 0)`; `RunOutcome(status, steps, final_text="", subagents_spawned=0, total_tokens: int = 0)`; `clamp_child_budget` preserves `max_total_tokens`.

- [ ] **Step 1: Write the failing tests**

Add to `Tests/Agents/test_agent_models.py` (import what's missing at the top of the file: `ModelTurn`, `RunBudget`, `RunOutcome`, `clamp_child_budget` from `tldw_chatbook.Agents.agent_models`):

```python
def test_modelturn_tokens_defaults_zero():
    from tldw_chatbook.Agents.agent_models import ModelTurn
    assert ModelTurn(text="hi").tokens == 0
    assert ModelTurn(text="hi", tokens=42).tokens == 42


def test_runbudget_max_total_tokens_defaults_zero():
    from tldw_chatbook.Agents.agent_models import RunBudget
    assert RunBudget().max_total_tokens == 0
    assert RunBudget(max_total_tokens=5000).max_total_tokens == 5000


def test_runoutcome_total_tokens_defaults_zero():
    from tldw_chatbook.Agents.agent_models import RunOutcome, RUN_DONE
    assert RunOutcome(RUN_DONE, []).total_tokens == 0
    assert RunOutcome(RUN_DONE, [], total_tokens=123).total_tokens == 123


def test_clamp_child_budget_preserves_max_total_tokens():
    from tldw_chatbook.Agents.agent_models import RunBudget, clamp_child_budget
    child = RunBudget(max_total_tokens=7000)
    assert clamp_child_budget(child, 10.0).max_total_tokens == 7000
```

- [ ] **Step 2: Run to verify failure**

Run: `... -m pytest Tests/Agents/test_agent_models.py -v -k "tokens or clamp_child_budget_preserves"`
Expected: FAIL (`ModelTurn` has no `tokens`; `RunBudget` has no `max_total_tokens`; etc.).

- [ ] **Step 3: Add `ModelTurn.tokens`**

In `agent_models.py`, add a trailing field to the frozen `ModelTurn` dataclass:

```python
@dataclass(frozen=True)
class ModelTurn:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    assistant_message: dict | None = None
    tokens: int = 0
```

- [ ] **Step 4: Add `RunBudget.max_total_tokens`**

Add a trailing field to the frozen `RunBudget` dataclass (after `max_model_turns`):

```python
    max_model_turns: int = 8
    # task-326: cumulative prompt+completion token spend ceiling for one run.
    # 0 = unlimited (default), keeping existing runs byte-identical. This is a
    # SPEND ceiling (the growing prompt is re-sent each call), not a window size.
    max_total_tokens: int = 0
```

- [ ] **Step 5: Add `RunOutcome.total_tokens`**

Add a trailing field to the `RunOutcome` dataclass:

```python
@dataclass
class RunOutcome:
    status: str
    steps: list[AgentStep]
    final_text: str = ""
    subagents_spawned: int = 0
    total_tokens: int = 0
```

- [ ] **Step 6: Preserve `max_total_tokens` in `clamp_child_budget`**

In `clamp_child_budget`, add `max_total_tokens=child.max_total_tokens` to the returned `RunBudget(...)` (alongside the other pass-through fields):

```python
    return RunBudget(
        max_steps=child.max_steps,
        max_wall_seconds=min(
            child.max_wall_seconds, max(parent_remaining_seconds, 1.0)
        ),
        max_subagents=0,
        max_active_tools=child.max_active_tools,
        max_subagent_result_chars=child.max_subagent_result_chars,
        max_model_turns=child.max_model_turns,
        max_total_tokens=child.max_total_tokens,
    )
```

- [ ] **Step 7: Run to verify pass**

Run: `... -m pytest Tests/Agents/test_agent_models.py -v`
Expected: PASS (new tests + all pre-existing model tests).

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Agents/agent_models.py Tests/Agents/test_agent_models.py
git commit -m "feat(agents): RunBudget.max_total_tokens + ModelTurn.tokens + RunOutcome.total_tokens (TASK-326)"
```

---

## Task 2: Loop accumulation + token-budget stop (`agent_runtime.py`)

**Files:**
- Modify: `tldw_chatbook/Agents/agent_runtime.py` (`run_agent_loop`)
- Test: `Tests/Agents/test_agent_runtime.py`

**Interfaces:**
- Consumes: `ModelTurn.tokens`, `RunBudget.max_total_tokens`, `RunOutcome.total_tokens` (Task 1).
- Produces: `run_agent_loop` accumulates `total_tokens += turn.tokens`, stops with `RUN_STUCK` + a `"token budget exhausted"` step when `max_total_tokens` is exceeded, and reports `RunOutcome.total_tokens` on every terminal path.

- [ ] **Step 1: Write the failing tests**

Add to `Tests/Agents/test_agent_runtime.py` (the module already imports `RUN_STUCK`, `RUN_DONE`, `AgentConfig`, `ModelTurn`, `RunBudget`, `run_agent_loop`, `SPAWN_TOOL_NAME`, and defines `fence`, `CFG`, `CALC`, `run`, `make_deps`):

```python
def test_token_budget_trips_to_stuck():
    # Tool-calling turns that never finish; each carries 100 tokens so the
    # cumulative spend crosses max_total_tokens before the (raised) step/turn caps.
    turns = [
        ModelTurn(text=fence("calculator", {"expression": str(i)}), tokens=100)
        for i in range(50)
    ]
    cfg = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=("calculator", SPAWN_TOOL_NAME),
        budget=RunBudget(max_steps=99, max_model_turns=99, max_total_tokens=250),
    )
    out = run(turns, config=cfg)
    assert out.status == RUN_STUCK
    assert out.steps[-1].summary == "token budget exhausted"
    assert out.total_tokens >= 250


def test_token_budget_sentinel_zero_never_trips():
    # Huge per-turn tokens but max_total_tokens=0 (default) -> completes normally.
    turns = [
        ModelTurn(text=fence("calculator", {"expression": "1"}), tokens=10_000_000),
        ModelTurn(text="all done", tokens=10_000_000),
    ]
    out = run(turns)  # default CFG budget has max_total_tokens=0
    assert out.status == RUN_DONE
    assert out.final_text == "all done"


def test_token_budget_done_on_crossing_turn_completes():
    # The final-answer turn itself crosses the budget -> still RUN_DONE
    # (stop-the-loop, not fail-the-answer), and total_tokens is reported.
    cfg = AgentConfig(model="m", system_prompt="s", budget=RunBudget(max_total_tokens=50))
    out = run([ModelTurn(text="the answer", tokens=100)], config=cfg)
    assert out.status == RUN_DONE
    assert out.final_text == "the answer"
    assert out.total_tokens == 100


def test_run_outcome_reports_total_tokens_accounting():
    turns = [
        ModelTurn(text=fence("calculator", {"expression": "1"}), tokens=30),
        ModelTurn(text="done", tokens=12),
    ]
    out = run(turns)
    assert out.status == RUN_DONE
    assert out.total_tokens == 42
```

- [ ] **Step 2: Run to verify failure**

Run: `... -m pytest Tests/Agents/test_agent_runtime.py -v -k "token_budget or total_tokens_accounting"`
Expected: FAIL (no token accumulation/check yet; `out.total_tokens` is always 0, `RUN_STUCK` step summary absent).

- [ ] **Step 3: Add the `total_tokens` accumulator + `_outcome` helper**

In `run_agent_loop`, initialize `total_tokens` beside the other counters. Find:

```python
    started = deps.clock()
    spawned = 0
    model_turns = 0
```

Change to:

```python
    started = deps.clock()
    spawned = 0
    model_turns = 0
    total_tokens = 0
```

Immediately after the existing `add` closure definition (the `def add(kind: str, **kw) -> AgentStep:` block ending in `return step`), add a sibling `_outcome` helper:

```python
    def _outcome(status: str, **kw) -> RunOutcome:
        # Reports run spend on every terminal path; reads enclosing steps/
        # spawned/total_tokens at call time (no nonlocal, like add()).
        return RunOutcome(
            status, steps, subagents_spawned=spawned, total_tokens=total_tokens, **kw
        )
```

- [ ] **Step 4: Accumulate tokens after each model turn**

Find the model-turn block:

```python
        turn = deps.call_model(messages, tuple(active))
        model_turns += 1
        add(STEP_MODEL, summary=turn.text[:200])
```

Change to:

```python
        turn = deps.call_model(messages, tuple(active))
        model_turns += 1
        total_tokens += turn.tokens
        add(STEP_MODEL, summary=turn.text[:200])
```

- [ ] **Step 5: Add the token-budget check after the wall-clock check**

Find the wall-clock check:

```python
        if deps.clock() - started > budget.max_wall_seconds:
            add(STEP_ERROR, summary="wall-clock budget exhausted")
            return RunOutcome(RUN_STUCK, steps, subagents_spawned=spawned)
```

Add immediately after it:

```python
        if budget.max_total_tokens and total_tokens >= budget.max_total_tokens:
            add(STEP_ERROR, summary="token budget exhausted")
            return _outcome(RUN_STUCK)
```

- [ ] **Step 6: Route every terminal return through `_outcome`**

Convert all **8** existing `return RunOutcome(...)` call-sites in `run_agent_loop` to `_outcome(...)` so each reports `total_tokens`:
- Every `return RunOutcome(<STATUS>, steps, subagents_spawned=spawned)` → `return _outcome(<STATUS>)` (there are six: the `should_cancel`, step-budget, model-turn-budget, wall-clock-budget, the in-tool-loop cancel, and the loop-detection returns).
- The two with `final_text=turn.text`:
  `return RunOutcome(RUN_CANCELLED, steps, final_text=turn.text, subagents_spawned=spawned)` → `return _outcome(RUN_CANCELLED, final_text=turn.text)`
  `return RunOutcome(RUN_DONE, steps, final_text=turn.text, subagents_spawned=spawned)` → `return _outcome(RUN_DONE, final_text=turn.text)`

Then verify no call-site was missed — only the single definition inside `_outcome` may remain:

Run: `grep -c "RunOutcome(" tldw_chatbook/Agents/agent_runtime.py`
Expected: `1` (the lone `RunOutcome(...)` constructor inside `_outcome`).

- [ ] **Step 7: Run to verify pass**

Run: `... -m pytest Tests/Agents/test_agent_runtime.py -v`
Expected: PASS — the four new tests plus every pre-existing loop test (the `_outcome` conversion preserves each prior return's status/final_text; `total_tokens` defaults 0 so untouched-behavior tests are unaffected).

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Agents/agent_runtime.py Tests/Agents/test_agent_runtime.py
git commit -m "feat(agents): accumulate token spend and stop the loop on max_total_tokens (TASK-326)"
```

---

## Task 3: Populate `ModelTurn.tokens` — real usage or estimate (`agent_service.py`)

**Files:**
- Modify: `tldw_chatbook/Agents/agent_service.py` (add `_usage_total_tokens`, imports, populate `tokens=` in `_make_call_model.call_model`)
- Test: `Tests/Agents/test_agent_service.py`

**Interfaces:**
- Consumes: `ModelTurn.tokens` (Task 1); `count_tokens_messages`, `estimate_tokens` from `tldw_chatbook.Utils.token_counter`.
- Produces: `_usage_total_tokens(resp) -> int | None`; `call_model` returns `ModelTurn` with `tokens=` populated (real usage when present, else estimate).

- [ ] **Step 1: Write the failing tests**

Add to `Tests/Agents/test_agent_service.py` (it already imports `AgentService`, `ToolCatalogRegistry`, `BuiltinToolProvider`, and defines the `db` fixture; add `from tldw_chatbook.Agents.agent_models import AgentConfig` and `from tldw_chatbook.Agents.agent_service import _usage_total_tokens` at the top):

```python
def test_usage_total_tokens_reads_total():
    assert _usage_total_tokens({"usage": {"total_tokens": 150}}) == 150


def test_usage_total_tokens_sums_prompt_and_completion():
    assert _usage_total_tokens(
        {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}
    ) == 150


def test_usage_total_tokens_none_when_absent_or_malformed():
    assert _usage_total_tokens({"choices": []}) is None
    assert _usage_total_tokens("a string") is None
    assert _usage_total_tokens({"usage": "bad"}) is None
    assert _usage_total_tokens({"usage": {"prompt_tokens": 10}}) is None


def _service_with_chat(db, chat_call):
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    return AgentService(db=db, registry=registry, chat_call=chat_call)


def test_call_model_uses_real_provider_usage(db):
    def chat(**kwargs):
        return {
            "choices": [{"message": {"content": "hello there"}}],
            "usage": {"total_tokens": 150},
        }
    service = _service_with_chat(db, chat)
    cfg = AgentConfig(model="gpt-4o", system_prompt="s", native_tools=False)
    call_model = service._make_call_model(cfg, "openai", [])
    turn = call_model([{"role": "user", "content": "hi"}], ())
    assert turn.tokens == 150


def test_call_model_estimates_when_no_usage(db):
    def chat(**kwargs):
        return {"choices": [{"message": {"content": "hello there world"}}]}
    service = _service_with_chat(db, chat)
    cfg = AgentConfig(model="gpt-4o", system_prompt="s", native_tools=False)
    call_model = service._make_call_model(cfg, "openai", [])
    turn = call_model([{"role": "user", "content": "count these tokens"}], ())
    # No provider usage -> estimate of sent payload + response text, always > 0.
    assert turn.tokens > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `... -m pytest Tests/Agents/test_agent_service.py -v -k "usage_total_tokens or call_model_uses_real or call_model_estimates"`
Expected: FAIL (`cannot import name '_usage_total_tokens'`).

- [ ] **Step 3: Add imports**

At the top of `agent_service.py`, add (with the other `tldw_chatbook` imports):

```python
from tldw_chatbook.Utils.token_counter import count_tokens_messages, estimate_tokens
```

- [ ] **Step 4: Add the `_usage_total_tokens` helper**

Add near the existing `_response_text`/`_response_message` module functions:

```python
def _usage_total_tokens(resp) -> int | None:
    """Prompt+completion tokens from a provider's OpenAI-shaped usage block,
    or None when the provider didn't report usage.

    Args:
        resp: The provider response (dict when the provider reports usage).

    Returns:
        The total tokens for the call, or None to signal "estimate instead".
    """
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

- [ ] **Step 5: Populate `tokens=` on both `ModelTurn` returns**

In `_make_call_model.call_model`, find:

```python
            text = _response_text(resp)
            if not native:
                return ModelTurn(text=text)
```

Change to (compute tokens once, before both returns):

```python
            text = _response_text(resp)
            tokens = _usage_total_tokens(resp)
            if tokens is None:
                # Provider reported no usage -> estimate from sent payload +
                # response text (native tool_calls JSON is not separately
                # counted here; the prompt term dominates the per-turn total).
                tokens = count_tokens_messages(payload, config.model) + estimate_tokens(
                    text, config.model
                )
            if not native:
                return ModelTurn(text=text, tokens=tokens)
```

And the native return at the end of `call_model`, find:

```python
            return ModelTurn(
                text=text, tool_calls=tool_calls, assistant_message=assistant_message
            )
```

Change to:

```python
            return ModelTurn(
                text=text,
                tool_calls=tool_calls,
                assistant_message=assistant_message,
                tokens=tokens,
            )
```

- [ ] **Step 6: Run to verify pass**

Run: `... -m pytest Tests/Agents/test_agent_service.py -v`
Expected: PASS — the new tests plus every pre-existing service test (adding `tokens=` to the returns doesn't change text/tool_calls behavior).

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Agents/agent_service.py Tests/Agents/test_agent_service.py
git commit -m "feat(agents): populate ModelTurn.tokens from provider usage or estimate (TASK-326)"
```

---

## Final verification (after all tasks)

- [ ] Run the full agent suite + the token_counter consumers:
  `... -m pytest Tests/Agents/ Tests/Chat/test_console_agent_bridge.py -q`
  Expected: all pass (no behavior change at defaults; `max_total_tokens=0` sentinel).
- [ ] Confirm the loop has exactly one `RunOutcome(` constructor:
  `grep -c "RunOutcome(" tldw_chatbook/Agents/agent_runtime.py` → `1`.
- [ ] Update the `task-326` backlog file (check ACs, add Implementation Notes) as the final step before finishing the branch.
