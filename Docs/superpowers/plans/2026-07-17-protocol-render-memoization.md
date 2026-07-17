# Tool-Protocol Render Memoization + Prompt-Caching Prep Note — Implementation Plan (task-245)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The fence-protocol system-prompt section is rendered once per active-set change within a run (byte-stable across unchanged turns — the precondition for provider-side prompt caching), and a design note records what provider-side caching would additionally need.

**Architecture:** `_make_call_model` already builds one `call_model` closure per run (`_run_one` constructs it once), so the memo is plain closure state: a name-tuple key + the cached rendered string, re-rendered only when the key changes (which is exactly when `load_tools` admits a new schema — AC #2). Native mode (task-243) never renders the protocol and is untouched. The note (AC #3) is a standalone doc, explicitly out-of-scope work.

**Tech Stack:** Python 3.11, existing agent service, pytest.

## Global Constraints

- Backlog ACs (task-245): (1) the rendered protocol string is cached/reused across consecutive turns within one run when the active schema set (BY NAME) has not changed since the last render; (2) the cache invalidates the moment `load_tools` admits a new schema into the active set; (3) a short design note records what provider-side prompt caching (e.g. Anthropic `cache_control`) would additionally need, as a distinct follow-up.
- Rendered output byte-identical to today for every turn (memoization must never change WHAT is sent, only avoid re-computing it); fence tests pass unchanged; native mode untouched.
- The cache must be per-run (closure state in `call_model`, built per `_run_one`) — never on `self` (AgentService is shared across runs incl. sub-agents, and a run's sub-agent gets its OWN closure via its own `_run_one`).
- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/agent-runtime`, branch `claude/protocol-memoize-245` (off origin/dev bbca2db8). Tests via `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest`, FOREGROUND; `timeout` unavailable.

---

### Task 1: Memoize the per-run protocol render + design note

**Files:**
- Modify: `tldw_chatbook/Agents/agent_service.py` (`_make_call_model`, ~lines 119-162)
- Create: `Docs/superpowers/reviews/2026-07-17-provider-prompt-caching-note.md`
- Test: `Tests/Agents/test_agent_service.py`

**Interfaces:**
- Consumes: `render_tool_protocol(schemas) -> str` (imported by name into agent_service — patchable at `tldw_chatbook.Agents.agent_service.render_tool_protocol`).
- Produces: no signature changes anywhere.

- [ ] **Step 1: Write the failing tests**

In `Tests/Agents/test_agent_service.py` (reuse `make_service`, `fence`, `CFG`, `FakeBigProvider`, and the find/load fixtures already present):

```python
def test_protocol_render_memoized_across_unchanged_turns(db, monkeypatch):
    """AC #1: three fence turns with an unchanged active set must render the
    protocol exactly once; the payload text stays byte-identical per turn."""
    import tldw_chatbook.Agents.agent_service as svc
    real_render = svc.render_tool_protocol
    calls = []

    def counting_render(schemas):
        calls.append(tuple(s.name for s in schemas))
        return real_render(schemas)

    monkeypatch.setattr(svc, "render_tool_protocol", counting_render)
    # script: two calculator fence rounds + final answer = 3 model turns,
    # active set never changes (direct-disclose catalog, no load_tools).
    service, chat = make_service(db, [
        fence("calculator", {"expression": "1+1"}),
        fence("calculator", {"expression": "2+2"}),
        "done"])
    _run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "go"}],
        config=CFG, api_endpoint="llama_cpp", should_cancel=lambda: False)
    assert outcome.status == RUN_DONE
    assert len(calls) == 1                       # rendered once, reused twice
    first_system = chat.calls[0]["messages_payload"][0]["content"]
    for later in chat.calls[1:]:
        assert later["messages_payload"][0]["content"] == first_system  # byte-stable


def test_protocol_rerenders_when_load_tools_admits_new_schema(db, monkeypatch):
    """AC #2: the cache invalidates the moment load_tools grows the active
    set — the very next turn's protocol includes the new tool."""
    # Mirror the file's existing find/load test setup (FakeBigProvider forces
    # the load path). Script: load_tools fence -> calculator fence -> "done".
    # Assert: counting_render was called exactly twice; the second recorded
    # name-tuple includes the newly loaded tool; the post-load turn's system
    # content contains the new tool's name while the pre-load turn's does not.
```

Write the second test fully against the file's real find/load fixtures.

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Agents/test_agent_service.py -v -k "memoized or rerenders"`
Expected: the first FAILS on `len(calls) == 1` (currently one render per turn); the second's render-count assert FAILS (currently 3).

- [ ] **Step 3: Implement**

In `_make_call_model`, fence branch only (native branch untouched):

```python
    def _make_call_model(self, config: AgentConfig, api_endpoint: str,
                         runtime_schemas: list):
        native = (config.native_tools
                  and provider_supports_native_tools(api_endpoint))
        # task-245: one render per active-set change, not per turn. Keyed by
        # schema NAMES (the set only ever grows via load_tools — AC #2), and
        # scoped to this closure = this run, so sub-agents (their own
        # _run_one -> their own closure) never share a cache. Byte-stable
        # repeated turns are the precondition for provider-side prompt
        # caching (see Docs/superpowers/reviews/
        # 2026-07-17-provider-prompt-caching-note.md).
        protocol_key: tuple | None = None
        protocol_text = ""

        def call_model(messages: list[dict], active_schemas: tuple) -> ModelTurn:
            nonlocal protocol_key, protocol_text
            schemas = runtime_schemas + list(active_schemas)
            system_content = config.system_prompt
            call_kwargs: dict = {}
            if native:
                tools = schemas_to_openai_tools(schemas)
                if tools:
                    call_kwargs["tools"] = tools
            else:
                key = tuple(s.name for s in schemas)
                if key != protocol_key:
                    protocol_text = render_tool_protocol(schemas)
                    protocol_key = key
                if protocol_text:
                    system_content = f"{config.system_prompt}\n\n{protocol_text}"
            ...rest unchanged...
```

(Preserve the native branch and everything below exactly as-is; only the fence-side render is keyed.)

Create `Docs/superpowers/reviews/2026-07-17-provider-prompt-caching-note.md` (~40 lines): what provider-side caching additionally needs, as a follow-up — (a) Anthropic: `cache_control: {"type": "ephemeral"}` breakpoints on the system block, which requires the system prompt as structured content blocks (list form) rather than a plain string, set inside `chat_with_anthropic`'s payload build — moot until task-263 makes Anthropic tool-capable, but equally applicable to plain chats; (b) OpenAI/compatible: automatic prefix caching needs no API marks — this task's byte-stability plus the stable `[system, ...history]` message ordering already qualifies the prefix; note that history APPEND-only growth (never rewriting earlier messages) is the other half, which the loop already satisfies; (c) native mode symmetry: `schemas_to_openai_tools` is likewise recomputed per turn and byte-stable per active set — a follow-up could memo it identically, and provider-side `tools` arrays participate in OpenAI prefix caching only if serialized stably (dict insertion order already deterministic here); (d) measurement hook suggestion: log prompt-cache hit metrics from provider usage fields (`cached_tokens` etc.) before investing further.

- [ ] **Step 4: Run the suites**

Run: `pytest Tests/Agents/ -q` then `pytest Tests/Chat/test_console_agent_bridge.py Tests/Chat/test_console_agent_swap.py -q`
Expected: PASS (all — repeated-turn payloads were already byte-identical, so no existing assertion can change).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/agent_service.py Tests/Agents/test_agent_service.py Docs/superpowers/reviews/2026-07-17-provider-prompt-caching-note.md
git commit -m "perf(agents): memoize fence-protocol render per active-set change + prompt-caching prep note (task-245)"
```

## Self-Review

- AC #1 → memo test (1 render / 3 turns + byte-stability assert). AC #2 → invalidation test (re-render on load, new tool present). AC #3 → the note, explicitly follow-up-scoped.
- Per-run scoping constraint encoded in the comment and guaranteed by construction (closure per `_run_one`).
- Byte-identity: render function unchanged; memo returns the identical string object — no output change possible.
