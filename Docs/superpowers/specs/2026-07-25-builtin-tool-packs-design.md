# Built-in tool packs for the agent runtime

**Date:** 2026-07-25
**Status:** Design approved; Phase 0 + Phase 1 ready for planning
**Related:** TASK-545 (built-in tool permission gate), TASK-547 (dead `[tools]` config),
TASK-656 (surface `agent:builtin` in a permissions UI), TASK-659 (agent settings screen)

## Scope of this document

This is a **program-level** spec. It defines the full candidate catalog and the phase
ordering at program altitude, then designs **Phase 0 and Phase 1 only** in
implementation detail. Phases 2–7 are sketched with their blocking dependencies
recorded; each earns its own spec before it is planned.

Do not treat the Phase 2–7 sketches as approved designs.

## 1. Problem

The agent runtime's built-in tool surface is two tools: `calculator` and `datetime`
(`Agents/tool_catalog.py`, `BuiltinToolProvider`). Everything else an agent can reach
comes from user-installed skills or user-configured MCP servers. An out-of-the-box
agent therefore cannot read a file, search the user's own corpus, fetch a page, or
take any action against the substantial subsystems this application already owns.

Meanwhile `Tools/tool_executor.py` (System A) holds working `read_file`,
`list_directory`, `write_file`, `rag_search`, `create_note`, `search_notes`,
`update_note`, and `web_search` implementations that the Console **never reaches** —
they are wired only to the deprecated legacy chat path. TASK-545 established that
System A and the agent runtime (System B) are disjoint systems, and that the
resolution is to port System A's tools into System B and gate them there.

This spec covers that port and what comes after it.

## 2. Verified current state

Facts established by reading the code on `origin/dev`, not from memory. Each is
load-bearing for a decision below.

| Fact | Location |
|---|---|
| `BuiltinToolProvider` holds only `CalculatorTool` and `DateTimeTool`, and takes no dependencies | `Agents/tool_catalog.py` |
| The permission gate exists and is wired unconditionally; namespace `agent:builtin` | `Agents/builtin_tool_gate.py`, `MCP/permission_store.py` |
| `HIGH_RISK_TAGS = {"mutates", "process"}`, consumed by both the MCP and built-in resolvers | `MCP/permission_store.py:69,637,706` |
| `DIRECT_DISCLOSE_THRESHOLD = 8`; above it the model must use `find_tools`/`load_tools` | `Agents/agent_models.py:48` |
| `max_active_tools = 8` is a hard ceiling; `active` is **only ever extended**, never reduced | `Agents/agent_runtime.py:570-572` |
| `load_tools` returns `"no room"` once the ceiling is reached; there is no unload path | `Agents/agent_runtime.py:579-580` |
| `active` is passed as the provider `tools=` array every turn, so today it is monotonically growing | `Agents/agent_runtime.py:393` |
| `disclosed_names` is a grow-only set kept deliberately in lockstep with `active`, and is what `invoke_tool` gates on | `Agents/agent_service.py:448-449, 394-398, 510` |
| Tool results enter conversation history **untruncated**; only the persisted step record is capped at 2000 chars | `Agents/agent_runtime.py:615-616` |
| Every tool call runs on a fresh per-call daemon thread; on timeout the worker is **abandoned still running** | `Agents/agent_service.py:168-213` |
| `max_tool_call_seconds = 300.0` is per-run; there is no per-tool override | `Agents/agent_models.py:180` |
| Built-ins are registered first and win name collisions; colliding MCP tools are silently dropped from the run | `Chat/console_agent_bridge.py`, `_non_colliding_mcp_names` |
| Console budget constants are sized together: turns 20, steps 64, wall 1200s | `Chat/console_agent_bridge.py:119-133` |
| Sub-agents deliberately inherit `max_model_turns` and `max_steps` (operator decision, 2026-07-25) | `Agents/agent_models.py`, `clamp_child_budget` |
| `max_total_tokens = 0` — run spend is unlimited by default | `Agents/agent_models.py` |
| DB connections are `threading.local` with an idle reaper assuming long-lived thread connections | `DB/ChaChaNotes_DB.py:2601, 2635` |
| `Media.content_hash` is `UNIQUE NOT NULL`; ingestion already performs duplicate-skip | `DB/Client_Media_DB_v2.py:246,381,3472` |
| `Scheduling/` mirrors the server model (`automation_definitions`, `approval_policy`, `AutomationFamily.AGENT_TASK`) | `Scheduling/models.py`, `Scheduling/db/migrations/v0_to_v1.py` |
| The scheduler dispatches on `task["type"]` against a registry holding only `reminder` and `watchlist_check` — **no `agent_task` handler exists** | `Scheduling/scheduler/loop.py` |

## 3. Decisions

1. **Direction:** all four — app-native data operation, general agent capability,
   producer/ingestion, and agent self-management.
2. **Granularity:** hybrid. Read paths collapse behind `corpus_search`/`corpus_read`
   with a `kind` discriminator; every mutation and action stays its own named tool so
   the approval card is always specific about what will happen.
3. **Disclosure:** user-enabled packs bound the catalog, plus an upgraded `find()` so
   cross-pack discovery works.
4. **Risk appetite:** full shell execution is in scope, with a command denylist,
   explicit disclosure that it acts outside the workspace, and a sandbox seam. Users
   are to be encouraged toward constrained environments; that guidance is the real
   safety story, the denylist is not.
5. **Backing:** local SQLite DBs only, behind an injected service object so a
   server-backed implementation can slot in later without changing tool schemas.
6. **Risk vocabulary:** `network` joins `HIGH_RISK_TAGS`, accepting that MCP tools
   already declaring that tag begin prompting.
7. **Turn budget:** Console turn cap rises to 30, with the two derived backstops
   resized. Sub-agent turn inheritance is **retained** per the 2026-07-25 operator
   decision; runaway spend is bounded instead by a non-zero `max_total_tokens`.

### 3.1 Retracted proposals

Recorded so they are not re-proposed:

- **Tool eviction / LRU on `active`.** Retracted. `active` is the provider `tools=`
  array and is currently grow-only, so a tool referenced by conversation history is
  always still declared. Eviction would be the first time that array shrinks, and
  native tool-calling runs against anthropic, google, and cohere with delicate history
  round-tripping (including Gemini 3 thought-signature pairing). Sending history whose
  `tool_use` references an undeclared tool is provider-undefined. Additionally,
  evicting from `active` without evicting from `disclosed_names` makes the tool
  permanently unloadable, because the re-load dedupe at `agent_service.py:510` skips
  any name already in `disclosed_names`. Replaced by simply raising the ceiling.
- **MCP tools winning name collisions.** Retracted. It would let a malicious or
  compromised MCP server name-squat an audited built-in (`write_file`, `run_command`)
  and intercept calls the user believes are gated. Built-ins keep winning; the
  shadowing is made visible instead.
- **`reads_private` as a non-flooring tag.** Retracted as dead code: a non-flooring
  tag on an otherwise-untagged tool resolves to `allow`, so no approval card ever
  renders for the copy to appear on. Conversation privacy is handled by defaulting
  `kind=conversation` off in the `corpus` pack config.
- **Job-handle shape for ingestion.** Retracted as unnecessary. `Media.content_hash`
  is unique and ingestion already duplicate-skips, so a post-timeout retry is a no-op.
  Only `run_command` genuinely cannot be made idempotent; it gets a per-tool timeout
  and an explicit "may have executed" message instead of a job store.
- **Clamping sub-agent turns.** Retracted; contradicts a documented operator decision.

## 4. Architecture

### 4.1 Pack layout

```
Agents/builtin_packs/__init__.py   # PACKS registry: name -> module
Agents/builtin_packs/files.py
Agents/builtin_packs/corpus.py
Agents/builtin_packs/authoring.py
Agents/builtin_packs/web.py
Agents/builtin_packs/shell.py
Agents/builtin_packs/library.py
Agents/builtin_packs/schedule.py
Agents/builtin_packs/meta.py
```

Each pack module exports:

- `TOOLS: tuple[type[Tool], ...]`
- `REQUIRES: tuple[str, ...]` — optional-dependency feature names checked through
  `optional_deps.py`

**Tool metadata must be readable without instantiation.** `name`, `description`,
`parameters`, and `risk_tags` are class-level and require neither a constructed
instance nor injected services. This is not stylistic: TASK-656's permissions
enumeration builds a bare `BuiltinToolProvider()` and calls `tool_ref(tool)` to resolve
each row's state. If pack tools can only be described once constructed with live
services, that enumeration degrades to listing `calculator`/`datetime` and rendering
every pack tool with a stored decision as an empty-description orphan. Enumeration must
read the **pack registry**, not a constructed provider.

Correspondingly, TASK-656's `orphaned` flag needs splitting into two states:

- *removed* — a stored decision for a tool no release provides any more; invite the user
  to clear it
- *pack disabled* — the tool exists and its decision is still meaningful; show it, allow
  pre-configuration before the pack is enabled, and do not invite clearing

A pack whose dependencies are unmet is **absent from the catalog**, never a tool that
fails at invoke time. The model must not spend turns discovering a tool is broken.

Because pack availability varies per machine, the resolved pack set is logged at run
start and `find_tools` reports the count hidden by unmet dependencies. Without this,
whether progressive disclosure engages at all differs between users and bug reports
become unreproducible.

### 4.2 Dependency injection

`BuiltinToolProvider.__init__` gains an optional `services: BuiltinToolServices` — a
frozen dataclass of local service seams (`notes_library`, `media_reading`,
`prompt_service`, `chunk_service`, `rag_search`, …), constructed once per run on the
main loop by `console_agent_bridge._compose_run_registry_and_allowed` and passed in.
Tools receive it at construction.

`BuiltinToolProvider()` with no services still yields `calculator`/`datetime`, so
existing tests remain valid unchanged.

**Services contract** — every injected service MUST be:

- thread-safe, because tools execute on a fresh per-call daemon thread
- free of event-loop-bound state (no `httpx.AsyncClient` bound to the app loop), because
  `BuiltinToolProvider.invoke` drives async tools through `asyncio.run`
- free of Textual/UI handles

This is written down because violating it produces failures that are miserable to
diagnose from a worker thread.

### 4.3 Configuration

Pack enablement lives under a new config section. Two known traps must be avoided:

- TASK-547: the existing `[tools]` section is unreachable via `get_cli_setting`. The
  new section must not repeat the defect.
- Dotted-section lookups have silently failed before (`chat.images`). The new keys
  require an **unmocked** integration test that reads them exactly as the app does.

**TASK-547 is resolved by deletion, not repair.** `[tools]`'s only live consumers are
System A's file tools (`file_sandbox_root`, §4.6) and `UI/Tools_Settings_Window.py`'s
per-tool enable/disable switches — and System A executes nothing (§4.7). Making the
section reachable would restore configurability to a system that cannot run. The new
pack config section owns the sandbox root instead; `[tools]` and the tool switches are
removed.

This also bounds the surfaces answering "what can the agent do." Without the deletion
there would be four: the dead Tools settings switches, TASK-656's permissions matrix,
pack enablement, and TASK-659's agent settings screen. Phase 1 removes the first.

Phase 1 owns the config **keys and defaults**. TASK-659's agent settings screen owns
**rendering** them. They must not grow competing surfaces.

### 4.4 Risk vocabulary

`HIGH_RISK_TAGS` becomes `{"mutates", "process", "network"}`. The `network` addition
closes the exfiltration leg: `fetch_url` with a model-chosen URL is a data channel, and
combined with corpus reads (private data) and ingested media (attacker-controlled text)
it completes an injection-to-exfiltration chain that would otherwise never prompt.

Because `HIGH_RISK_TAGS` is shared with MCP resolution
(`permission_store.py:637` and `:706`), any MCP tool already declaring a `network` tag
begins flooring to `ask`. This is a deliberate, documented behavior change.

### 4.5 Error handling

Unchanged contract: tools return `ToolResult` and never raise;
`BuiltinToolProvider.invoke` already wraps. Long-running mutators rely on backing-store
idempotency (content-hash duplicate-skip) rather than a job-handle protocol.
`run_command` is the sole exception and is handled in §8.4.

### 4.6 Filesystem root policy

System A's file tools confine every read, write and listing to a sandbox root —
`<user data dir>/tool_sandbox` by default, overridable via `[tools]
file_sandbox_root` (`file_operation_tools._tool_sandbox_root`). Porting those tools
without restating the policy would silently widen them from *a scratch directory* to
*the entire filesystem*. That is a privilege escalation, and it must be a decision, not
a side effect of a port.

Policy for ported and new file tools:

1. **Workspace-rooted by default.** File tools operate under the active workspace root,
   plus any additional roots the user configures. This is what makes the tools useful —
   a general agent confined to a scratch directory cannot do the work — while keeping
   the reachable surface something the user chose.
2. **Credential and application-state paths are refused regardless of root.** `~/.ssh`,
   `~/.aws`, `~/.gnupg`, keyring stores, `~/.config/tldw_cli/config.toml`, the SQLite
   DBs, and `mcp_permissions.json` are denied even when a configured root would
   otherwise contain them. This mirrors §8.4's shell denylist, and for the same reason:
   `read_file` is untagged and therefore silent, so an unconfined read is a
   zero-prompt path from a credential file into a persisted transcript that may be sent
   to any provider.
3. **Reads outside the root are refused, not prompted.** An `ask` on every stray path
   trains reflexive approval; the user widens the root deliberately instead.
4. All resolution uses `Utils/path_validation.validate_path` and the existing
   `_is_within` containment check against fully-resolved paths, so symlink escapes are
   rejected.

The incoherence to avoid: `run_command` has no root confinement by nature, so a
sandbox-confined `read_file` beside an unconfined shell is security theatre. The
resolution is that they share one denylist for the paths that matter (rule 2) and differ
only in default reach — which is honest, because the shell is `ask`-gated per call and
`read_file` is not.

### 4.7 System A is already dead

TASK-545's description states System A is reached from the legacy chat path via
`Event_Handlers/worker_events.py` and `chat_streaming_events.py`. **That is no longer
true.** `chat_streaming_events.py` was deleted by the TASK-577 campaign,
`worker_events.py` holds zero tool-executor references, and a sweep of
`tldw_chatbook/**.py` finds no caller of `get_tool_executor`, `execute_tool_call`, or
`execute_tool_calls` outside `Tools/` itself and `UI/Tools_Settings_Window.py` — which
only *lists* tools behind enable/disable switches.

Two consequences:

1. **TASK-545 P3 collapses.** Its criterion "no tool anywhere executes ungated" is
   already satisfied, because System A cannot execute at all. P3 reduces from "port,
   gate, or remove" to removing dead code: `Tools/tool_executor.py`'s registry and
   registration block, the `Tools_Settings_Window` switches, and `[tools]`. The tool
   *implementations* (`file_operation_tools.py`, `note_management_tools.py`,
   `rag_search_tool.py`, `web_search_tool.py`) are the porting source for Phases 1–3
   and outlive the executor.
2. **The `Tool` ABC survives the removal.** `Agents/builtin_tool_gate.py` imports `Tool`
   from `Tools.tool_executor`, and `risk_tags` lives on it. Removal must relocate the
   ABC rather than delete the module wholesale.

This is recorded here because TASK-545's own text predates the deletion campaign and
will otherwise send an implementer looking for callers that no longer exist.

## 5. The catalog

`[P2]` marks tools already inside TASK-545 P2's acceptance criteria.

### `files`
| Tool | Tags |
|---|---|
| `read_file` `[P2]` | — |
| `list_directory` `[P2]` | — |
| `glob_files` | — |
| `grep_files` | — |
| `write_file` `[P2]` | mutates |
| `edit_file` (exact-string replace) | mutates |

`edit_file` requires a read-before-edit invariant: the file must have been read in the
current run before an edit is accepted, and the target string must match uniquely.

### `corpus`
| Tool | Tags |
|---|---|
| `corpus_search(kind, query, filters, limit)` | — |
| `corpus_read(kind, id, offset, limit)` | — |
| `corpus_facets(kind)` | — |
| `rag_search` `[P2]` | — |

`kind ∈ {note, media, conversation, character, prompt, watchlist_item}`.
`kind=conversation` defaults **off** in pack config: it exposes every message the user
has ever sent an LLM. `corpus_facets` returns available keywords, authors and date
ranges so the agent can construct meaningful filters rather than guessing.

`rag_search` stays separate from `corpus_search` — semantic and lexical retrieval are
different affordances, and it carries a RAG profile parameter.

### `authoring`
| Tool | Tags |
|---|---|
| `chunk_text` | — |
| `create_note` `[P2]` | mutates |
| `update_note` `[P2]` | mutates |
| `create_prompt` | mutates |

`chunk_text` returns a handle and chunk count, never the chunks inline.

### `web`
| Tool | Tags |
|---|---|
| `web_search` `[P2]` | network |
| `fetch_url` | network |

`fetch_url` uses `Article_Extractor_Lib` for readable-text extraction and
`Utils/egress.py` for SSRF policy.

### `shell`
| Tool | Tags |
|---|---|
| `run_python` | process |
| `run_command` | process, mutates |

Detailed in §8.4.

### `library`
| Tool | Tags |
|---|---|
| `export_chatbook` | mutates |
| `ingest_url` | mutates, network |
| `ingest_file` | mutates |
| `generate_image` | mutates |
| `speak_text` | mutates |
| `transcribe_media` | process, mutates |
| `index_media` | process, mutates |

### `schedule`
| Tool | Tags |
|---|---|
| `schedule_list` | — |
| `schedule_preview(spec)` | — |
| `schedule_create(preview_id)` | mutates |
| `schedule_pause` / `schedule_resume` / `schedule_delete` | mutates |
| `reminder_create` | mutates |

### `meta`
| Tool | Tags |
|---|---|
| `recall` | — |
| `remember` | mutates |
| `todo_write` | — |

### Excluded

- **Evals tools** — agent-driven benchmarking is a different product.
- **`think` / scratchpad** — dead weight with reasoning models.
- **Multi-page `scrape_site`** — a crawl budget is its own design problem;
  `fetch_url` in a loop is honest and bounded.
- **Conversation and character writes** — an agent silently editing character cards is
  a trust cliff, not a feature.
- **`ask_user`** — not implementable as a tool. It would block a daemon thread on a
  human inside the per-call timeout, with abandon-on-timeout semantics, so an answer
  arriving after the ceiling lands nowhere. It belongs at the runtime as a suspending
  step kind, not in a pack.

## 6. Phase 0 — runtime substrate

No pack ships before this. Each item is a prerequisite, not a follow-up.

### 6.1 Raise `max_active_tools` and `DIRECT_DISCLOSE_THRESHOLD`

| Constant | From | To | Derivation |
|---|---|---|---|
| `max_active_tools` | 8 | 24 | A typical enabled set is three packs — `files` (6) + `corpus` (4) + `authoring` (4) = 14 — and the ceiling must clear that with room for a fourth. Runtime schemas (`spawn_subagent`, `find_tools`, `load_tools`, skill tools) are appended separately and do not consume this budget |
| `DIRECT_DISCLOSE_THRESHOLD` | 8 | 16 | See below |

`active` remains grow-only; no eviction is introduced (see §3.1).

**The threshold must rise with the ceiling.** `initial_disclosure` runs per run, so a
catalog above the threshold costs `find_tools` + `load_tools` round trips **before any
real work, on every user message**. With packs enabled that becomes the common case,
turning a two-turn tax into the default experience. Raising the threshold to 16 means a
modest pack set is disclosed directly and only genuinely large catalogs — many packs,
or packs plus a large MCP surface — pay for progressive disclosure.

This genuinely trades prompt tokens for round trips, and the trade is only favourable
because of how Console runs are distributed. Direct-disclosing 16 schemas costs roughly
2.4k tokens **every turn**, where progressive disclosure leaves the model carrying only
the three or four schemas it actually loaded. Over a 30-turn run direct disclosure is
the more expensive option.

It wins because **most Console messages are short.** On a one-to-three-turn message —
the common case by a wide margin — two round trips spent on `find_tools`/`load_tools`
before any real work is most of the run, and that overhead repeats on every message. The
long run where progressive disclosure pays off is the exception, and it is exactly the
run that can afford the tokens. If telemetry later shows long runs dominating, the
threshold is a single constant to lower.

**Tests:** a run with a full pack set can load past the old ceiling and still invoke the
last-loaded tool; a run with a 14-tool catalog discloses directly and issues no
`find_tools` call.

### 6.2 Runtime-enforced tool result caps

Add `max_tool_result_chars` to `RunBudget`, defaulting to **16,000**, enforced in
`_append_tool_result` — **not** in individual tools. Over-cap content is truncated with
an explicit trailer naming the tool and the pagination arguments that retrieve the
remainder.

The default is derived from the existing sibling constant `max_subagent_result_chars`
(4,000): four times a sub-agent's whole result is generous for a single tool call while
still bounding a 30-turn run to a tractable history. It is configurable, and `0` means
unlimited so an operator can restore today's behavior exactly.

This applies uniformly to built-in, MCP, and skill results. That is a behavior change
for existing MCP users, so the MCP documentation records it.

**Test:** a tool returning far over the cap leaves conversation history bounded, and the
trailer names a usable continuation.

### 6.3 Per-tool timeout override

`max_tool_call_seconds` is currently per-run only. Add a per-tool override so slow
ingestion and transcription are not forced to share a ceiling with a calculator, and so
`run_command` can carry a tighter one.

**Test:** a tool declaring its own ceiling is bounded by that value, not the run default.

### 6.4 Name-shadowing visibility

Built-ins continue to win collisions with MCP tools. When a collision drops an MCP
tool from a run, emit a log line and surface a row in the MCP workbench so the user
learns their configured tool was shadowed rather than discovering it silently stopped
working.

**Test:** a run composed with a colliding MCP tool records the shadowing and still
resolves the built-in.

### 6.5 `network` risk tag

Extend `HIGH_RISK_TAGS`. Verify the MCP resolver's existing tests still pass and add
one asserting a `network`-tagged tool floors to `ask`.

### 6.6 Console budget resize

| Constant | From | To | Derivation |
|---|---|---|---|
| `CONSOLE_MAX_MODEL_TURNS` | 20 | 30 | The primary limiter |
| `CONSOLE_MAX_STEPS` | 64 | 96 | A fence round costs 3 steps, so N turns need `3*(N-1)+1` = 88 at N=30. 64 would fire around round 22, making 30 unreachable. 96 clears 88 with backstop headroom comparable to 64-over-58 |
| `CONSOLE_MAX_WALL_SECONDS` | 1200 | 1800 | 25–50s/turn at the slow local-model pace the live gate exercises × 30 = 750–1500s |
| `CONSOLE` `max_total_tokens` | 0 (unlimited) | 1,000,000 | Sub-agents inherit turns by the 2026-07-25 decision, so worst case is `30 × (1 + 2)` = 90 provider turns. That is bounded in time but not in spend. At a ~20k-token working prompt across 90 turns a runaway approaches ~1.8M tokens; a 1M ceiling stops that while sitting far above any normal 30-turn run. `0` remains available to restore unlimited |

`DEFAULT_MAX_MODEL_TURNS` also moves to 30 for consistency, with its docstring keeping
the note that only the Console sizing makes it operative — engine `max_steps=8` fires
first by design.

`test_console_budget_step_cap_admits_a_full_model_turn_run` exists specifically to fail
if `max_steps` drops below the derived minimum, and validates the new numbers.

## 7. Phase 1 — pack seam and the read-only `files` pack

### 7.1 Contents

- `Agents/builtin_packs/` registry, `TOOLS`/`REQUIRES` protocol, optional-dep gating
- `BuiltinToolServices` frozen dataclass and its injection at
  `_compose_run_registry_and_allowed`
- Pack-enablement config keys and defaults, with an unmocked read test (§4.3)
- TASK-547's `[tools]` reachability fix, since it governs §4.6's root (§4.3)
- The §4.6 filesystem root policy, shared with §8.4's denylist
- `files` pack, **read-only subset**: `read_file`, `list_directory`, `glob_files`,
  `grep_files`

The `find()` upgrade moves **out** of Phase 1. With `DIRECT_DISCLOSE_THRESHOLD` at 16
(§6.1), a one-to-three-pack user never reaches `find_tools` at all, so improving it
buys nothing until catalogs actually grow. It returns when a phase pushes a realistic
enabled set past the threshold — Phase 5 (`library`) is the likely trigger.

### 7.2 Why read-only

`write_file` and `edit_file` are deferred to Phase 2 behind TASK-656. TASK-545 P1
shipped session-scoped decisions deliberately, because no UI can currently reverse a
persistent built-in deny. Shipping mutating tools before that surface exists forces
users to re-approve every session, which trains them to approve reflexively — the
opposite of what the gate is for.

The read-only subset still proves everything architectural: pack registry, dependency
injection, config, catalog composition, and disclosure. The gate's ask/deny path is
already covered by P1's own tests against synthetic tools.

### 7.3 Non-goals

Phase 1 does **not** close TASK-545 P2. P2's criteria list `rag_search`, `web_search`,
and the note tools alongside the filesystem ones; TASK-545 stays open until Phase 3.

## 8. Later phases

Sketches, not approved designs. Each needs its own spec.

### 8.1 Phase 2 — `corpus` + `authoring`
Closes TASK-545 P2's tool list together with Phase 3. Blocked on TASK-656 so mutating
tools ship with reversible persistent permissions. Adds `write_file`/`edit_file`
deferred from Phase 1.

### 8.2 Phase 3 — `web`
Blocked on the `network` tag. Completes TASK-545 P2.

### 8.3 Phase 4 — `shell`
Own PR, own security review.

### 8.4 `shell` safeguards

A denylist on shell input is a guardrail against accidents and naive injected payloads,
**not a security boundary**. `rm -rf /` is catchable; `python -c "import shutil; …"` is
not. The boundary is user approval plus the sandbox.

- **Approval:** tagged `process` + `mutates`, floors to `ask`. P1 already excludes
  `always_allow` for built-in rows; `run_command` narrows further via per-row `options`
  to `approve_once` / `deny` only. "Approve shell for the session" is indistinguishable
  from no gate.
- **Refuse outright** (not prompt), after `shlex`-parsing the full compound including
  `;`, `&&`, `||`, `|`, backticks and `$(…)`:
  - destructive-root patterns: `rm -rf /`, `mkfs*`, `dd of=/dev/*`, partition tools,
    fork bombs, `shutdown`/`reboot`
  - credential paths: `~/.ssh`, `~/.aws`, `~/.gnupg`, keyring stores
  - **this application's own state**: `~/.config/tldw_cli/config.toml`, the SQLite DBs,
    and `mcp_permissions.json`. Without this, `run_command` is a one-line bypass of the
    entire permission gate — rewrite the permission store and every `ask` becomes
    `allow`. Path matching will not stop a determined `python -c`, which is the
    strongest argument for the sandbox track.
- **Environment scrubbing is mandatory.** A bare `env` would dump every API key into the
  model's context and the persisted transcript. Reuse `skill_script_runner`'s scrubbed
  environment.
- **Out-of-workspace disclosure:** the approval card shows the resolved cwd and flags
  argument paths resolving outside it. A one-time first-use notice states that the tool
  acts with the user's full account privileges and points at sandbox guidance.
- **Sandbox seam, shipped inert:** `[tools.shell] sandbox = "none"`, designed for
  `docker` / `sandbox-exec` / `bubblewrap`. Every run records its sandbox mode in the
  step record so the audit trail never overstates containment.
- **Timeout:** uses §6.3's per-tool override, with a message stating explicitly that the
  command may have executed and must not be retried blindly. `run_command` is the one
  tool where abandon-on-timeout cannot be resolved by idempotency.

### 8.5 Phase 5 — `library`
Relies on `content_hash` duplicate-skip for idempotency under retry. Optional-dep gated.

### 8.6 Phase 6 — `schedule`

Mirrors the server control plane (`POST /previews` → `preview_id` → `POST /definitions`,
plus pause/resume/archive). The preview-then-commit split is what makes scheduling safe
for agent use: the approval card shows a rendered schedule — "every day at 07:00, run:
summarize new watchlist items" — rather than an opaque cron string the user must decode
under time pressure.

Three things do not exist yet and are this phase's real work:

1. **An `agent_task` handler.** `AutomationFamily.AGENT_TASK` exists as an enum value;
   no handler is registered in `scheduler/loop.py`. It must run an agent turn headless,
   which forces decisions about where output lands and what happens on failure.
2. **A no-user-present authorization story.** A scheduled run has nobody to approve
   anything, so the gate correctly fails closed and the agent gets only untagged tools —
   meaning a nightly ingestion job silently does nothing useful. The `approval_policy`
   column on `automation_definitions` exists for exactly this and must actually be read
   by the gate.
3. **A recursion guard.** A scheduled agent able to call `schedule_create` is a runaway
   generator. The `schedule` pack is excluded from scheduled runs outright. This
   generalizes: scheduled runs need their own pack policy, not the interactive one.

Server-side, `tldw_server` TASK-2286 ("Add agentic scheduling and wakeup tools") is a
To Do stub with no acceptance criteria, so there is nothing upstream to port.

### 8.7 Phase 7 — `meta`

`remember`/`recall` must not use the ordinary notes scope: the bidirectional file sync
engine would turn every agent memory into a file on disk and churn the sync loop. A
reserved non-synced scope or a dedicated table is required.

## 9. Known gaps and risks

- **Prompt injection remains the dominant risk.** `corpus_read` over ingested media
  returns attacker-controlled text into the loop. With `files` and `shell` enabled, the
  classic trifecta — private data, untrusted content, exfiltration — is fully present.
  The `network` tag prompts on the exfiltration leg; nothing prevents a user from
  approving it. This is inherent to the capability, not a defect to be closed.
- **Connection churn.** Tools run on per-call daemon threads while DB connections are
  `threading.local` with an idle reaper assuming long-lived threads. Each DB-touching
  call opens and orphans a connection. Negligible at two calls per run; measurable at
  thirty. **To be filed** as a follow-up task before Phase 2 — no task exists yet.
- **Denylist evasion.** Stated plainly in §8.4. The sandbox track is the answer.
- **Per-machine catalog variance.** Mitigated by logging the resolved pack set (§4.1),
  not eliminated.

## 10. Testing

- Per-pack unit tests against real in-memory SQLite, following house style — no mocks
  for DB behavior.
- Phase 0 gets dedicated tests per §6.1–6.6.
- Config keys get an **unmocked** integration test reading them as the app does (§4.3).
- One live end-to-end run per pack. Note that **Phase 1 has no allow/ask/deny to
  drive** — its tools are read-only and untagged, so every one resolves to the `allow`
  floor. Phase 1's live gate covers pack resolution, disclosure, and invocation;
  TASK-545's allow/ask/deny criterion is exercised from Phase 2, where the first
  `mutates`-tagged tools ship.

## 11. Coordination

- **TASK-656** (`agent:builtin` permissions UI) gates Phase 2 and is **in progress in
  parallel** on branch `feat/builtin-permissions-ui`. Two coordination points:
  - Its enumerator constructs a bare `BuiltinToolProvider()`; §4.1 records the
    class-level-metadata requirement and the `orphaned` split that keeps it working
    once packs exist. Designing for that now is far cheaper than retrofitting.
  - §6.4's name-shadowing surface belongs in whatever permissions view 656 builds
    rather than a second, competing MCP-workbench affordance.

  The branch's commits are labeled `[TASK-627]`, but the dedup round-2 merge (#904)
  renumbered that work to TASK-656 — on dev, `task-627` is now an unrelated Settings/RAG
  mouse-capture bug. Because the branch is 57 commits behind dev, its `backlog/tasks/`
  still holds the *old* task-627 file; rebasing will otherwise reintroduce a duplicate
  ID. The rebase must delete that file (its content already lives at task-656) rather
  than merge it.
- **TASK-659** (agent settings screen) renders the config keys Phase 1 defines, and
  already documents the budget constants §6.6 changes.
- **TASK-547** (dead `[tools]` config) is the defect §4.3 must not repeat.
