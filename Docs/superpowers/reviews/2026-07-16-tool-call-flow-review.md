# Agent runtime tool-call flow review — task-231

Read-only investigation. No production code changed. Probe scripts used for
measurements live in the scratchpad (not committed); their commands and
output are inlined below so the numbers are reproducible.

## 1. End-to-end map (file:line)

```
User submits draft
  -> ConsoleChatController.submit_draft / retry_message / regenerate_message /
     continue_from_message  (Chat/console_chat_controller.py)
       - _apply_skill_substitution (console_chat_controller.py:619)   [/skill-name user-invocation, unrelated to model-driven tool calls]
       - _provider_messages_for_session -> _provider_message_payloads
         (console_chat_controller.py:1118-1216)                      -- ONE full transcript rebuild
         per user turn (not per agent step): walks store.messages_for_session(),
         reserves an image budget, builds role/content payload dicts.
  -> _run_agent_reply (console_chat_controller.py:930-1025)
       - splits leading system message off into session_system_prompt
       - asyncio.to_thread(self._agent_bridge.run_reply, ...)         -- ONE call per user turn
  -> ConsoleAgentBridge.run_reply (Chat/console_agent_bridge.py:534-628)
       - if skills_service: asyncio.run(get_context(mode="local"))    -- ONE disk read (skills index
         (console_agent_bridge.py:551)                                   JSON) per user turn, not per step
       - _compose_run_registry_and_allowed -> fresh ToolCatalogRegistry +
         allow-list every run_reply call (console_agent_bridge.py:438-471)
       - AgentConfig(budget=CONSOLE_RUN_BUDGET) (console_agent_bridge.py:560-564)
         CONSOLE_RUN_BUDGET = RunBudget(max_steps=16, max_wall_seconds=480.0)
         (console_agent_bridge.py:61)
       - one asyncio event loop for the WHOLE run, reused across every
         chat_call (console_agent_bridge.py:574, PR #629 Fix 1(c))
       - _StreamingModelAdapter.chat_call -> StreamGate() PER TURN
         (console_agent_bridge.py:288-334, agent_stream.py:49-176)
       - on_step hook: appends an in-memory-only TOOL marker
         (persist=False) for STEP_SPAWN / non-quiet STEP_TOOL_RESULT / STEP_ERROR
         (console_agent_bridge.py:588-606, 731-743) -- NOT a ChaChaNotes DB write
  -> AgentService.run_turn -> _run_one (Agents/agent_service.py:144-393)
       - db.create_run(...)              (agent_service.py:149)   -- 1 write txn
       - initial_disclosure(registry, budget) (tool_catalog.py:347-358)
       - builds find_tools/load_schemas/spawn/invoke_tool closures
       - LoopDeps -> run_agent_loop(...)
       - db.append_steps(...) + db.set_status(...) via _persist    -- 2 write txns,
         (agent_service.py:133-142, called once at agent_service.py:342)   ONCE AT RUN END, not per-step
  -> run_agent_loop (Agents/agent_runtime.py:198-385)  -- pure, no I/O
       - per turn: deps.call_model(messages, active) -> ModelTurn
       - _make_call_model (agent_service.py:107-121): re-renders
         render_tool_protocol(runtime_schemas + active_schemas) EVERY turn,
         rebuilds the full payload list EVERY turn (required — stateless
         chat-completion transport), calls self.chat_call(...) with
         streaming=False and NO tools=/tool_choice= kwarg
       - fence parse: split_visible_text_and_tool_call (agent_runtime.py:70-98)
       - find_tools/load_tools/spawn_subagent handled in-loop
         (agent_runtime.py:313-373, 290-312)
       - other tool names -> deps.invoke_tool -> registry.invoke_by_name
         or (skill) skill_runner.run -> asyncio.run(execute_skill(...))
         (agent_service.py:294-321, console_agent_bridge.py:495-508)
         -- execute_skill re-reads the skills index JSON from disk
         (local_skills_service.py:785-802 -> get_skill:608-611 -> _load_index:127-135)
         ONCE PER SKILL INVOCATION (fresh-trust-at-call-time by design)
  -> provider gateway: chat_api_call / LLM_API_Calls.py per-provider handler
       - ONE HTTP request per deps.call_model() call == one per fence turn
       - tools=/tool_choice= ALREADY plumbed end-to-end for OpenAI, Anthropic,
         Cohere, OpenRouter, Mistral, Gemini, Groq(partial), HF, DeepSeek,
         Moonshot (LLM_API_Calls.py: e.g. lines 510, 790, 1094, 1433, 1593,
         1892, 2141, 2365, 2518, 2807, 3040) -- but llama_cpp's own
         PROVIDER_PARAM_MAP entry has BOTH commented out
         (Chat_Functions.py:301,312-313: `#'tools': 'tools'`, `#'tool_choice':
         'tool_choice'`) and the agent runtime never passes tools= to
         chat_api_call regardless of provider (agent_service.py:117-119) --
         100% fence-text protocol today, for every provider.
```

## 2. Measured costs (AC 1)

All numbers below come from real code (`render_tool_protocol`,
`ToolCatalogRegistry`, `AgentService`, `AgentRunsDB`) exercised against the
actual builtin tool schemas (`Calculator`, `get_current_datetime`) and the
real 14-skill `obra/superpowers` corpus (the project's own standing skills
test corpus — `/Users/macbook-dev/.codex/.tmp/plugins/plugins/superpowers/skills/*/SKILL.md`),
not synthetic stand-ins. Probe scripts: `probe_tool_protocol.py`,
`probe_timing.py`, `probe_db_writes.py` in the scratchpad.

### Token overhead (`render_tool_protocol`, agent_runtime.py:139-168)

| Scenario | Catalog size | Disclosure mode | Protocol chars | ~tokens (chars/4) |
|---|---|---|---|---|
| Builtins only (Calculator+DateTime) | 2 (+spawn) | direct-disclose (≤8 threshold) | 1,477 | ~369 |
| Builtins + 14 real skills, **initial** turn | 16 (+spawn) | disclosure (find/load offered) | 1,356 | ~339 |
| Same, **after** find+load fills active set to cap (8) | 16 (+spawn) | disclosure, post-load | 5,317 | ~1,329 |
| Hypothetical: all 16 catalog tools direct-disclosed | 16 (+spawn) | (no gating) | 8,568 | ~2,142 |
| `find_tools("")` catalog-lines result (one-liners only) | 16 | — | 2,887 | ~722 |

Because `_make_call_model` re-renders the protocol **every** provider call
(agent_service.py:110-111), a post-load 16-tool run resends ~1,329 protocol
tokens on **every single turn** for the rest of the run — at 16 turns that's
~21,268 tokens of protocol text alone, byte-identical each time whenever the
active set hasn't changed.

### CPU cost — measured, all negligible

```
render_tool_protocol(11 schemas, post-load-cap):        80.8 microseconds/call
per-turn call_model payload rebuild @ 2-60 history msgs: ~73-74 microseconds/call
StreamGate full turn (176 chunks, 3510 chars, no fence):  223.1 microseconds/turn (1.27us/chunk)
```
CPU is not a bottleneck anywhere in this pipeline — `StreamGate` already
scans incrementally from a persisted `_scan_from` offset (agent_stream.py:103),
so it's not accidentally quadratic. The real costs below are round-trips and
tokens, not CPU.

### Round-trips per disclosure cycle (measured with a real scripted run)

Scripted `AgentService.run_turn` against the real 16-tool registry
(`probe_db_writes.py`), model turns: find_tools → load_tools →
calculator → final answer:

```
Step count: 10
  [0] model        (```tool_call find_tools...)
  [1] tool_call    find_tools
  [2] tool_result  find_tools    "builtin:calculator — calculator: ..."
  [3] model        (```tool_call load_tools...)
  [4] tool_call    load_tools
  [5] tool_result  load_tools    "loaded: calculator"
  [6] model        (```tool_call calculator...)
  [7] tool_call    calculator
  [8] tool_result  calculator    "{"expression": "2+2", "result": 4, ...}"
  [9] model        "The answer is 4."
```
This is exactly the 10-step floor documented in
`console_agent_bridge.py:44-52` — confirmed by execution, not just by
re-reading the comment. **4 provider calls** for one real tool use. The
direct-disclose equivalent (Scenario A, catalog ≤8) needs only **2 provider
calls** (tool-call round + wrap-up). Disclosure gating costs **+6 steps
(+150%), +2 provider calls (+100%)** versus direct-disclose for a
single-tool-use turn — the exact overhead the `>8` threshold buys in
exchange for keeping the active set small when the catalog is large.

### Provider calls per multi-tool turn

`ModelTurn.tool_calls` is **always** the empty default tuple —
`_make_call_model` (agent_service.py:107-121) never populates it from a
provider response, and never passes `tools=`/`tool_choice=` to
`self.chat_call(...)`. So even though `run_agent_loop`'s per-turn dispatch
already iterates `for call in calls:` (agent_runtime.py:276) and could
process several `ToolCall`s from one native multi-call reply, that path is
dead code today: `calls = list(turn.tool_calls)` is always `[]`, so
`calls = [fenced]` (agent_runtime.py:257-273) is always exactly one call.
**One provider call buys at most one tool call, always**, regardless of what
the underlying provider natively supports.

### IO — persistence and skills reads

* **AgentRunsDB writes are already batched at run end, not per-step.**
  Measured via a counting wrapper around `AgentRunsDB.transaction()`
  (`probe_db_writes.py`): the entire 10-step run above produced exactly
  **3** write transactions total — `create_run` (agent_service.py:149),
  `append_steps` (agent_service.py:140, batches all 10 steps in one UPDATE),
  `set_status` (agent_service.py:141-142). The live `on_step` hook fires
  once per step (agent_runtime.py:231-241) but is UI-only (feeds the rail
  snapshot + in-memory TOOL marker); it never touches the DB. Each spawned
  sub-agent is its own `_run_one` call and adds its own 3 write txns
  (create_run/append_steps/set_status), so an N-subagent run tree costs
  `3 * (1 + N)` write transactions — still O(runs), not O(steps).
* **ChaChaNotes writes per marker: zero.** `_append_marker` calls
  `store.append_message(..., persist=False)` (default) — an in-memory
  transcript row only (console_agent_bridge.py:731-743). Resume
  re-derives markers from `AgentRunsDB` (`resume_marker_messages`,
  console_agent_bridge.py:684-727), so there is no per-marker DB
  amplification to worry about — already correctly designed.
* **Skills catalog composition: once per user turn**, not per step —
  `get_context(mode="local")` is called once in `run_reply`
  (console_agent_bridge.py:551), building a fresh registry/allow-list. Each
  **skill invocation** (not each turn) triggers one more full
  `_load_index()` disk read+JSON-parse via `execute_skill` → `get_skill`
  (local_skills_service.py:785-802, 608-611, 127-135) — reasonable at
  today's catalog scale (tens of skills) but would not scale to hundreds
  without an in-request cache; `_load_index()` has no cache today at all
  (every call re-reads and re-parses the whole index file).

## 3. Opportunities, prioritized (AC 2)

### 1. Wire native provider tool-calls, fence-text as fallback — **High value, Medium-High effort**
The vertical-slice design spec explicitly called for this
(`Docs/superpowers/specs/2026-07-12-agent-runtime-vertical-slice-design.md:119-125`:
*"the runtime supports native tool-calls where the provider advertises
them, and a text-protocol fallback otherwise"*) but it was never
implemented — `_make_call_model` only ever builds the fence prompt and
never sets `tools=`/`tool_choice=` (agent_service.py:107-121), so
`ModelTurn.tool_calls` is always empty. Most cloud providers already accept
`tools=`/`tool_choice=` at the raw `chat_api_call` layer (see table above);
`llama_cpp`'s own `PROVIDER_PARAM_MAP` entry has both commented out
(Chat_Functions.py:312-313), so a real switch needs a per-provider/model
capability check with fence-fallback for local backends that don't support
it — matching the design doc's own stated intent. Payoff: removes the ~339-
2,142-token protocol block from every turn for capable providers, collapses
the fence-parsing risk surface, and — since `run_agent_loop` already loops
over multiple `ToolCall`s per turn (agent_runtime.py:276) — unlocks
multiple-tool-calls-per-reply "for free" the moment `ModelTurn.tool_calls`
is populated from a real response, cutting the measured 4-provider-call
discovery round down toward 1-2 calls for capable models.

### 2. Budget accounting: charge model turns, not raw step entries; fix load_tools name-fallback — **High value (AC 4), Low-Medium effort**
`RunBudget.max_steps` counts `STEP_MODEL` + `STEP_TOOL_CALL` +
`STEP_TOOL_RESULT` identically (agent_runtime.py:231-241, 247-249), but
only `STEP_MODEL` costs a provider round-trip — the other two are
near-instant bookkeeping entries for the *same* round. The measured 10-step
discovery floor is therefore only **4 real model turns**; `CONSOLE_RUN_BUDGET`
(max_steps=16) buys ~5 model-turn rounds total, i.e. roughly 1-2 rounds of
real work beyond discovery — thin, and exactly why the Skills Phase-2 gate
hit exhaustion on a *successful* discovery run (console_agent_bridge.py:44-52).
Recommendation: track/budget on **provider-call count** (STEP_MODEL) as the
primary limiter, with total `AgentStep` entries as a much looser secondary
safety net — this gives 2-3x more real headroom for the same conceptual
budget size without inflating `max_wall_seconds` further. Bundle in the
cheap, related fix: `load_schemas` (agent_service.py:175-209) only accepts
catalog ids (`"builtin:calculator"`); a model that echoes back the bare
`name` half of a `find_tools` result line (`_catalog_lines`,
agent_runtime.py:191-195: `"{e.id} — {e.name}: ..."`) gets silently dropped
(`except KeyError: continue`) and burns a whole extra round on "No valid
tools found to load" — `ToolCatalogRegistry.resolve_name()` already exists
(tool_catalog.py:325-327) and is unused here; falling back to it costs
one `if` and removes a common, wasted retry round for exactly the
weaker/local models this budget is tuned for.

### 3. Memoize the rendered protocol per active-schema-set; evaluate provider prompt-caching — **Medium value, Low effort for the memo, Medium for provider caching**
`render_tool_protocol` is deterministic over its input list
(agent_runtime.py:139-168) and CPU cost is negligible (measured ~81us/call)
so this is *not* a performance bug today — but it is a clean, free win: cache
the rendered string keyed on the active-schema-name tuple in
`_make_call_model`'s closure (agent_service.py:107-121) so identical turns
(the common case — most rounds don't change the active set) skip the
rebuild, and — more importantly — the resulting system-prompt prefix stays
byte-identical across turns, which is the precondition for provider-side
prompt caching (e.g. Anthropic `cache_control` breakpoints). No provider
call in `LLM_API_Calls.py` sets `cache_control`/prompt-cache fields today
(grep found none) — wiring that up is a separate, provider-specific lift,
but the render-memoization here is the cheap first half that makes it
possible later, and directly reduces the measured ~21k-token/16-turn
resend cost for providers that do support caching.

### Other, lower-priority observations
- **Persistence and marker writes are already well-designed** — batched
  end-of-run DB writes, in-memory-only markers re-derived from
  `AgentRunsDB` on resume. No action needed; documented above so this
  doesn't get "fixed" again.
- **`_load_index()` has no in-process cache** (local_skills_service.py:127-135)
  — fine at current skill counts (14-ish real, tens expected); worth an
  invalidation-aware cache only if/when the skill catalog grows to
  hundreds.
- **`StreamGate` per-chunk scanning is already efficient** (persisted
  `_scan_from`, no rescanning of settled buffer regions) — confirmed by
  timing (1.27us/chunk); not a target.

## 4. Budget recommendation summary (AC 4)

Real multi-step run shape (measured): discovery (find+load) = 6 steps / 2
model turns; one tool/skill execution round = 3 steps / 1 model turn;
wrap-up = 1 step / 1 model turn. A "does one real thing after discovering
it" run is **10 steps / 4 model turns** — already at the console
`max_steps=16` budget's ~62% mark before any retry, double-check, or
second tool round. Recommendation: keep `max_steps=16`/`max_wall_seconds=480`
as the *entry-count* safety net (cheap, already tuned, don't touch without
re-validating against this same corpus), but add a **model-turn-count**
budget as the primary limiter (e.g. 8-10 real provider calls), since that
number tracks actual cost (latency + tokens) far more accurately than raw
`AgentStep` count — a discovery-heavy run and a straight-answer run
currently consume the identical nominal budget unit for very different
real costs.

## Follow-up tasks (AC 3) — draft `backlog task create` commands (NOT run; coordinator files after ID-collision check against origin/dev)

```
backlog task create "Wire native provider tool-calls with fence-protocol fallback" \
  -d "The vertical-slice design spec called for native tool-calls where the provider advertises them, with the fence-first text protocol as a fallback for tool-incapable models (Docs/superpowers/specs/2026-07-12-agent-runtime-vertical-slice-design.md line 119). Only the fallback was ever implemented: agent_service.py's _make_call_model never sets tools=/tool_choice= and ModelTurn.tool_calls is always empty, so every provider — including ones that already support native function-calling end-to-end in LLM_API_Calls.py (OpenAI, Anthropic, Cohere, OpenRouter, Mistral, Gemini, HF, DeepSeek, Moonshot) — pays the fence-protocol's per-turn prompt overhead and one-tool-call-per-reply limit today. run_agent_loop already iterates multiple ToolCall entries per turn (agent_runtime.py line 276), so native multi-call batching is unlocked as soon as ModelTurn.tool_calls is populated from a real response." \
  -l agents,console,performance \
  --ac "A per-model/provider capability check selects native tool-calls when supported and falls back to the fence protocol otherwise (llama_cpp and other local backends without tools= support in PROVIDER_PARAM_MAP keep working via fallback)" \
  --ac "ModelTurn.tool_calls is populated from a real native tool-call response for at least one cloud provider end-to-end" \
  --ac "A native multi-tool-call reply is dispatched as multiple ToolCall entries in one run_agent_loop turn without engine changes" \
  --ac "Existing fence-protocol tests and the Console agent-reply integration tests still pass unchanged for tool-incapable models"

backlog task create "Tiered agent-run budget accounting (model-turn count) + load_tools name-fallback" \
  -d "CONSOLE_RUN_BUDGET.max_steps counts STEP_MODEL/STEP_TOOL_CALL/STEP_TOOL_RESULT identically even though only STEP_MODEL costs a provider round-trip, so a measured 10-step discovery-plus-one-tool-call run (4 real model turns) already consumes 62% of the 16-step budget before any retry or second tool round — the exact shape that hit step exhaustion on the Skills Phase-2 gate's successful discovery run. Separately, agent_service.py's load_schemas only resolves catalog ids (e.g. builtin:calculator), silently dropping a bare tool name a model echoes back from a find_tools result line, burning a full extra round on a generic 'No valid tools found to load' error even though ToolCatalogRegistry.resolve_name() already exists and is unused there." \
  -l agents,console,performance \
  --ac "The run budget's primary limiter is (or is accompanied by) a model-turn/provider-call count rather than only raw AgentStep entries" \
  --ac "The documented 10-step/4-model-turn discovery-plus-execution floor (console_agent_bridge.py's own comment) leaves headroom for at least 2 additional real tool rounds under the recommended budget" \
  --ac "load_schemas resolves a bare tool name via registry.resolve_name() as a fallback when the direct catalog-id lookup fails, before giving up on that id" \
  --ac "A new test reproduces a model calling load_tools with a bare name (not a catalog id) and confirms the tool loads instead of erroring"

backlog task create "Memoize per-turn tool-protocol render and prepare for provider prompt-caching" \
  -d "_make_call_model re-renders render_tool_protocol(runtime_schemas + active_schemas) from scratch on every provider call within a run, even though the active set is unchanged on most turns (measured: a post-load 16-tool catalog's protocol text is ~1,329 tokens, resent unchanged on every one of up to 16 turns — ~21k tokens of byte-identical text over one run). CPU cost of the render itself is negligible (measured ~81 microseconds), so this is not a CPU fix; it is the precondition for later provider-side prompt caching (e.g. Anthropic cache_control breakpoints), which no provider call in LLM_API_Calls.py currently sets." \
  -l agents,console,performance \
  --ac "The rendered protocol string is cached/reused across consecutive turns within one run when the active schema set (by name) has not changed since the last render" \
  --ac "A cache invalidates correctly the moment load_tools admits a new schema into the active set" \
  --ac "A short design note records what would be needed to additionally mark this stable prefix for provider-side prompt caching, as a distinct follow-up rather than in-scope work here"
```
