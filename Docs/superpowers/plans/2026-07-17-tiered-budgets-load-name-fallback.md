# Tiered Budget Accounting + load_tools Name-Fallback — Implementation Plan (task-244)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Agent runs are limited primarily by model turns (provider round-trips) instead of raw step entries, and `load_tools` resolves bare tool names a model echoes from `find_tools` output instead of burning a round on a generic error.

**Architecture:** `RunBudget` gains `max_model_turns` (appended last — positional compat); `run_agent_loop` counts `STEP_MODEL` calls and returns `RUN_STUCK` with a distinct `model-turn budget exhausted` error when exceeded. Engine defaults are chosen so the new check is **provably unreachable** at `RunBudget()` (each turn adds ≥1 step, so `max_model_turns=8 == max_steps=8` can never fire first) — zero engine-default behavior change. The Console budget flips to model-turn-primary (`max_model_turns=8`, steps raised to a pure backstop of 32). `agent_service.load_schemas` falls back to `registry.resolve_name()` when the direct catalog-id lookup KeyErrors.

**Tech Stack:** Python 3.11, existing agent runtime, pytest.

## Global Constraints

- Backlog ACs (task-244): (1) the run budget's primary limiter is (or is accompanied by) a model-turn/provider-call count; (2) the documented 10-step/4-model-turn discovery floor (console_agent_bridge.py's comment block, lines ~42-61) leaves headroom for ≥2 additional real tool rounds under the recommended budget; (3) `load_schemas` resolves a bare tool name via `registry.resolve_name()` as a fallback when the direct catalog-id lookup fails, before giving up on that id; (4) a new test reproduces a model calling `load_tools` with a bare name and confirms the tool loads instead of erroring.
- Engine-default behavior must be byte-identical: `RunBudget()` runs behave exactly as today (the new check unreachable at defaults — prove it in a comment and pin defaults in a test).
- `clamp_child_budget` constructs `RunBudget` field-by-field — it MUST carry `max_model_turns` through (`child.max_model_turns`), or children would silently get the default.
- The `"step budget exhausted"` copy is pinned by `Tests/Chat/test_console_agent_swap.py:504` — do not change it; the new copy is `"model-turn budget exhausted"`.
- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/agent-runtime`, branch `claude/tiered-budgets-244` (off origin/dev 496a5319). Tests: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest`, FOREGROUND always; `timeout` command unavailable.

## File Structure

- Modify `tldw_chatbook/Agents/agent_models.py` — `RunBudget.max_model_turns` + `clamp_child_budget` passthrough.
- Modify `tldw_chatbook/Agents/agent_runtime.py` — model-turn check in the loop.
- Modify `tldw_chatbook/Chat/console_agent_bridge.py` — `CONSOLE_RUN_BUDGET` + its comment block.
- Modify `tldw_chatbook/Agents/agent_service.py` — `load_schemas` name fallback.
- Tests: `Tests/Agents/test_agent_models.py`, `Tests/Agents/test_agent_runtime.py`, `Tests/Agents/test_agent_service.py`, `Tests/Chat/test_console_agent_bridge.py` (only if it pins budget values — check).

---

### Task 1: Model-turn budget tier (ACs #1, #2)

**Files:**
- Modify: `tldw_chatbook/Agents/agent_models.py` (`RunBudget` ~line 80; `clamp_child_budget` ~line 116)
- Modify: `tldw_chatbook/Agents/agent_runtime.py` (budget checks at top of the `while True:` loop, ~lines 243-252)
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py` (comment block ~42-61 + `CONSOLE_RUN_BUDGET` line 62)
- Test: `Tests/Agents/test_agent_runtime.py`, `Tests/Agents/test_agent_models.py`

**Interfaces:**
- Produces: `RunBudget.max_model_turns: int = 8` (LAST field); loop returns `RunOutcome(RUN_STUCK, ...)` with step summary `"model-turn budget exhausted"` when the count of prior `STEP_MODEL` steps reaches the cap; `CONSOLE_RUN_BUDGET = RunBudget(max_steps=32, max_wall_seconds=480.0, max_model_turns=8)`.

AC #2 arithmetic to encode in the rewritten comment: the documented discovery floor is 3 tool rounds + 1 wrap-up = **4 model turns / 10 steps**. Two additional real tool rounds cost 2 turns / 6 steps → 6 turns / 16 steps total. Under the new budget the primary limiter is `max_model_turns=8` → the floor + 2 extra rounds fits with 2 turns to spare; `max_steps=32` is a pure backstop (≈4 steps/turn worst case) that can no longer starve a discovery run one step short of its wrap-up. `max_wall_seconds` stays 480 (25-50s/turn × 8 turns).

- [ ] **Step 1: Write the failing tests**

In `Tests/Agents/test_agent_runtime.py` (reuse the file's existing deps/fake helpers):

```python
def test_model_turn_budget_exhaustion_is_distinct_from_step_budget():
    """A run that spends its model turns on tool rounds must stop with the
    model-turn copy while raw steps are still well under max_steps."""
    # budget: max_model_turns=2, max_steps=99 (steps can never bind)
    # script: every model turn returns a fence tool call (never a final
    # answer) -> turn 1 (3 steps), turn 2 (3 steps), then the check fires
    # BEFORE a third provider call.
    outcome = run_agent_loop(cfg_with(budget=RunBudget(max_steps=99,
                                                       max_model_turns=2)),
                             [{"role": "user", "content": "go"}], [schema],
                             deps)
    assert outcome.status == RUN_STUCK
    assert outcome.steps[-1].summary == "model-turn budget exhausted"
    assert sum(1 for s in outcome.steps if s.kind == STEP_MODEL) == 2
    assert len(outcome.steps) < 99


def test_console_budget_floor_plus_two_extra_rounds_completes():
    """AC #2: the documented 4-turn/10-step discovery floor plus TWO more
    real tool rounds (6 turns / 16 steps) completes under CONSOLE_RUN_BUDGET."""
    from tldw_chatbook.Chat.console_agent_bridge import CONSOLE_RUN_BUDGET
    # script: 5 fence tool-call turns then a final answer (6 model turns,
    # 16 steps) with a generous fake clock -> RUN_DONE, not stuck.
    assert outcome.status == RUN_DONE
```

Write both fully against the file's real helpers (fence-turn scripting exists in the file). In `Tests/Agents/test_agent_models.py`:

```python
def test_run_budget_default_model_turns_equal_steps_and_child_clamp_carries():
    b = RunBudget()
    # Unreachability invariant: each model turn appends >=1 step, so with
    # max_model_turns == max_steps the step check always fires first (or
    # ties) at defaults -> engine-default behavior byte-identical.
    assert b.max_model_turns == b.max_steps == 8
    child = clamp_child_budget(RunBudget(max_model_turns=3),
                               parent_remaining_seconds=30.0)
    assert child.max_model_turns == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Agents/test_agent_runtime.py Tests/Agents/test_agent_models.py -v -k "model_turn or floor_plus_two"`
Expected: FAIL (`RunBudget` has no `max_model_turns`).

- [ ] **Step 3: Implement**

`agent_models.py` — `RunBudget` (field appended LAST; docstring on the class noting the unreachable-at-default invariant):

```python
@dataclass(frozen=True)
class RunBudget:
    max_steps: int = 8
    max_wall_seconds: float = 240.0
    max_subagents: int = 2
    max_active_tools: int = 8
    max_subagent_result_chars: int = 4000
    # Primary provider-call limiter (task-244): counts STEP_MODEL turns.
    # At defaults it equals max_steps, which makes it provably unreachable
    # (every model turn appends >=1 step, so the step check fires first) —
    # engine-default behavior is byte-identical to the pre-task-244 loop.
    max_model_turns: int = 8
```

`clamp_child_budget` gains `max_model_turns=child.max_model_turns,` in its `RunBudget(...)` construction.

`agent_runtime.py` — inside the `while True:`, after the step-budget check and before the wall-clock check (order preserves the tie-break proof above):

```python
        if sum(1 for s in steps if s.kind == STEP_MODEL) >= budget.max_model_turns:
            add(STEP_ERROR, summary="model-turn budget exhausted")
            return RunOutcome(RUN_STUCK, steps, subagents_spawned=spawned)
```

(If the loop already tracks a cheap counter, prefer incrementing one next to `add(STEP_MODEL, ...)` — implementer's choice; the summary copy and placement are binding.)

`console_agent_bridge.py` — replace `CONSOLE_RUN_BUDGET` and rewrite its comment block for the new accounting (keep the Skills Phase-2 gate provenance, state the 4-turn/10-step floor, the +2-round headroom arithmetic above, and that steps are now a pure backstop):

```python
CONSOLE_RUN_BUDGET = RunBudget(
    max_steps=32, max_wall_seconds=480.0, max_model_turns=8)
```

Update the `run_agent_loop` docstring's budget sentence to mention both limiters.

- [ ] **Step 4: Run the suites**

Run: `pytest Tests/Agents/ Tests/Chat/test_console_agent_bridge.py Tests/Chat/test_console_agent_swap.py -q`
Expected: PASS — every pre-existing step-budget test unchanged (defaults unreachable), swap-test's pinned `"step budget exhausted"` copy untouched.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/agent_models.py tldw_chatbook/Agents/agent_runtime.py tldw_chatbook/Chat/console_agent_bridge.py Tests/Agents/test_agent_runtime.py Tests/Agents/test_agent_models.py
git commit -m "feat(agents): model-turn budget tier — provider calls become the primary run limiter (task-244)"
```

---

### Task 2: `load_schemas` bare-name fallback (ACs #3, #4)

**Files:**
- Modify: `tldw_chatbook/Agents/agent_service.py` (`load_schemas` closure, ~lines 216-250)
- Test: `Tests/Agents/test_agent_service.py`

**Interfaces:**
- Consumes: `ToolCatalogRegistry.resolve_name(name: str) -> str | None` (tool_catalog.py:325 — returns the catalog id or None; cache shared with `invoke_by_name`).
- Produces: unchanged `load_schemas` signature; a bare name that resolves is treated exactly as if its catalog id had been passed (allow-list gate, disclosed-dedup gate, and room slice all still apply, in the same order).

- [ ] **Step 1: Write the failing tests**

In `Tests/Agents/test_agent_service.py` (reuse `make_service`, `fence`, `CFG`; the file already has find/load flow tests — mirror their budget/catalog setup, including the >DIRECT_DISCLOSE_THRESHOLD catalog trick they use to force the find/load path):

```python
def test_load_tools_with_bare_name_loads_via_resolve_name_fallback(db):
    """AC #4: models echo the tool NAME from a find_tools result line, not
    the catalog id. load_tools(ids=["calculator"]) must load the tool."""
    # script: fence load_tools {"ids": ["calculator"]} -> fence
    # calculator call -> final answer. Assert the load result step contains
    # "loaded: calculator" (not "No valid tools found to load"), the
    # calculator round-trips, and the run completes done.


def test_load_tools_bare_name_still_respects_allow_list(db):
    """A resolvable bare name OUTSIDE config.allowed_tools stays refused
    with the generic load error (Q7(c) gate unchanged)."""


def test_load_tools_unresolvable_junk_still_errors_generically(db):
    """ids=["definitely-not-a-tool"] -> 'No valid tools found to load'."""
```

Write all three fully against the file's real fixtures.

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Agents/test_agent_service.py -v -k "bare_name or unresolvable_junk"`
Expected: the bare-name test FAILS (generic load error); the other two may already pass (pinned regressions).

- [ ] **Step 3: Implement**

In `load_schemas`, replace the lookup block only — the allow-list, dedup, and room-slice logic below it must stay byte-identical:

```python
            for tool_id in ids:
                try:
                    schema = self.registry.load_schema(str(tool_id))
                except KeyError:
                    # task-244 AC#3: models often echo a bare tool NAME from
                    # a find_tools result line instead of the catalog id —
                    # resolve it before giving up on this entry, instead of
                    # burning the whole round on a generic load error.
                    resolved = self.registry.resolve_name(str(tool_id))
                    if resolved is None:
                        continue
                    try:
                        schema = self.registry.load_schema(resolved)
                    except KeyError:
                        continue
```

- [ ] **Step 4: Run the service suite**

Run: `pytest Tests/Agents/test_agent_service.py -q`
Expected: PASS (all, including every pre-existing find/load gate test).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/agent_service.py Tests/Agents/test_agent_service.py
git commit -m "feat(agents): load_tools resolves bare tool names via resolve_name fallback (task-244)"
```

---

### Close-out (coordinator)

Cross-suite sweep (`Tests/Agents/ + test_console_agent_bridge.py + test_console_agent_swap.py + test_console_provider_gateway.py + Tests/UI/test_console_agent_rail.py`), backlog ACs checked + Implementation Notes, final review, PR.

## Self-Review

- AC #1 → Task 1 (`max_model_turns` primary in the Console budget; distinct-copy test). AC #2 → Task 1's `test_console_budget_floor_plus_two_extra_rounds_completes` (6 turns/16 steps under 8/32). AC #3/#4 → Task 2 (fallback + three tests incl. allow-list and junk pins).
- Engine-default byte-identity: proven by construction (8==8, ≥1 step per turn) and pinned by the defaults test; `clamp_child_budget` passthrough covered by test.
- Type consistency: `max_model_turns` named identically across models/loop/bridge/tests; copy string `"model-turn budget exhausted"` used verbatim in loop and test.
