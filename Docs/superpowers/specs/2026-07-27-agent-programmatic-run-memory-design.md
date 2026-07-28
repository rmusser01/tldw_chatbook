# Programmatic run memory for the agent runtime

**Date:** 2026-07-27
**Status:** Design approved; not implemented
**Baseline:** `origin/dev` @ `c2e6483c4`
**Prior art:** PRO-LONG (arXiv 2607.20064v2; `github.com/alexisfox7/PRO-LONG`)

## 1. Problem

The agent runtime's memory *is* its message list. Every tool result, every model
turn, and every sub-agent result accumulates in `run_agent_loop`'s `messages`,
and the whole list is re-sent on every provider call. Nothing ever evicts, and
nothing outside that list is queryable while a run is in flight.

That was survivable when runs were short. They are not short any more. The
Console budget on `origin/dev` is:

```python
CONSOLE_MAX_MODEL_TURNS = 30
CONSOLE_MAX_STEPS       = 96
CONSOLE_MAX_WALL_SECONDS = 1800.0
CONSOLE_MAX_TOTAL_TOKENS = 1_000_000
```

Thirty tool-calling rounds over thirty minutes, with each tool result admitted
into history at up to `max_tool_result_chars = 16_000`. A run that actually uses
that budget can push well past 100k tokens of prompt by its final rounds. A
200k-context cloud model tolerates it expensively; a 32k local model does not
reach round ten.

**The budgets were raised for long-horizon work. The memory architecture was
never changed to match.** This design changes it.

Three further consequences of the same root cause:

- **Nothing survives a crash.** `agent_service._persist` writes `agent_runs.steps`
  as a single JSON blob *after* the loop terminates. A run killed at round 28 of
  30 leaves no record of the first 27.
- **Truncation is lossy, permanently.** When `_truncate_tool_result` cuts a
  result to 16,000 characters, the remainder is gone. The trailer tells the model
  to "re-issue the call with a narrower query" — re-doing work whose answer was
  already computed.
- **Sub-agent work is discarded.** A parent sees `max_subagent_result_chars =
  4000` of its child's entire run. Everything else the child learned is lost at
  the `spawn` boundary.

## 2. Approach

PRO-LONG's finding is that an agent does not need its history *in context*; it
needs history to be *reachable*. Append every action and observation losslessly
to a structured log, and give the agent a way to search it. The paper's ablation
(GPT-5.5, pass@1) shows how much the read side matters: read-only 23.1%, plus
grep 27.2%, plus model-authored Python 38.3%, plus write/edit 41.2%.

Applied here, in one sentence: **the log becomes the lossless record, and the
message list becomes a cache of it.**

Phase 1 is deliberately *additive* — the log is written and made searchable, but
`messages` is not touched, so existing runs behave byte-identically. The format
and seams are designed so that a later phase can evict from context (§10), which
is where goals 1 and 2 are fully realised.

### 2.1 Goals

1. Long-horizon runs that are not bounded by the context window.
2. Small-context local models running the same workloads as large cloud models.
3. Queryable, crash-durable run history for the user and the UI.
4. A later 1:1 PRO-LONG mode (full accessed/accessible split) reachable as a
   configuration change, not a rewrite.

### 2.2 Non-goals

- Model-authored code execution over the log. Deferred; see §11.
- Semantic/embedding retrieval over history. PRO-LONG's central claim is that
  programmatic search beats it, and the app's RAG stack is a separate concern.
- Changing `grep_files` / `glob_files`. See §9.4.
- Replacing `agent_runs` DB persistence. The log is a second, independent record.

### 2.3 What `origin/dev` already provides

| Capability | State |
| --- | --- |
| Sandboxed file read/write/list | `read_file`, `write_file`, `list_directory` — consult `allowed_file_roots`, so they reach workspace folders |
| Sandboxed search | `glob_files`, `grep_files` — **sandbox root only** (see §9.4) |
| Constrained script execution | `run_skill_script` — trusted-skill bundles only, scrubbed env, rlimits |
| Long budgets | 30 model turns / 96 steps / 1800s / 1M tokens |
| Per-result history cap | `max_tool_result_chars = 16_000` |
| Run lineage + supersession | `AgentRunsDB`, `supersede_run_tree` |
| Append-only lossless log | **Missing — this design** |
| Context eviction | **Missing — deferred to Phase 3** |

## 3. Architecture

### 3.1 Writer ownership

```
run_turn()                            RunLogWriter constructed ONCE here, UNBOUND
  └─ _run_one(primary)                bind(run_id) — directory created here
       └─ spawn() → _run_one(child)   SAME writer, already bound
```

The writer is constructed in `AgentService.run_turn` and threaded through the
`_run_one` recursion. Two requirements make this mandatory rather than stylistic:

- **One monotonic record counter per run *tree*.** `AgentStep.index` is
  `len(steps)` — per-run. Sub-agents run inline via `spawn` → `_run_one`, so a
  per-run writer would give parent and child each their own counter and produce
  duplicate record numbers in a shared log.
- **Every caller gets logging.** `on_step` is passed *into* `AgentService` by the
  Console bridge. Anything built on the hook would silently produce no log for
  any non-Console caller.

**Construction is two-phase, and must be.** The run id does not exist when
`run_turn` runs — `run_id = self.db.create_run(...)` is the first statement of
`_run_one`, so the writer cannot name its `<run_id>/` directory at construction
time. The writer is therefore constructed *unbound* in `run_turn` (fixing the
counter to the tree) and `bind(run_id)` is called by the **primary** `_run_one`,
which creates the directory. Child `_run_one` calls find the writer already
bound and never rebind it.

### 3.2 Capture points

The step record is not a viable source: `agent_runtime.py` truncates model turns
to 200 characters (`summary=turn.text[:200]`) and tool results to 2,000
(`result=content[:2000]`) — strictly *less* than history retains. So capture must
happen somewhere the full value exists.

Capture is a **single new injected callable, `LoopDeps.on_record`**, called at
exactly two points inside `run_agent_loop`:

| Call site | Record types | Carries |
| --- | --- | --- |
| Immediately after `turn = deps.call_model(...)`, before `add(STEP_MODEL, …)` | `model` | full `turn.text`, `turn.tokens`, and any `tool_calls` with their `call_id`s |
| At the point `content` is assembled for a dispatched call, **before** `_truncate_tool_result` is applied | `tool_call`, `tool_result` | full args; full result or error text |
| `on_step` (existing hook) | `error` | budget exhaustion, cycle detection |

**Why one hook in the loop rather than wrappers in the service.** An earlier
draft wrapped `call_model`, `invoke_tool`, and `spawn` individually in
`agent_service`. That is both more code and *incomplete*: the loop dispatches
`find_tools`, `load_tools`, `spawn_subagent`, `skill_file`, `install_skill`, and
`run_skill_script` through their own branches, never through `deps.invoke_tool`,
so none of their results would have been logged — `run_skill_script`'s script
output least excusably of all. The loop assembles `content` for **every** branch
at one point, immediately before truncation. Capturing there is uniform by
construction: builtin, MCP, skill, and runtime tools are all covered, and so is
the `review_tool_calls` refusal path.

It also removes the need for a special `spawn` capture. A child's own model and
tool records are already written by the same hook during its own loop, so the
child's untruncated final answer is in the log as its last `model` record —
independently of the `max_subagent_result_chars = 4000` cut applied to what the
*parent* receives. The parent's `spawn` record needs only the task text and the
returned summary.

**This does modify `agent_runtime.py`**, and an earlier draft of this document
wrongly claimed otherwise. The modification follows the module's established
pattern exactly: `on_record` is an optional injected callable on `LoopDeps`,
defaulting to a no-op, in the same shape as the existing `on_step` and
`review_tool_calls`. The loop stays pure — it calls an injected function and
owns no I/O. `agent_service` remains the only impure module in `Agents/`.

**Anti-duplication rule.** `on_record` owns `model`, `tool_call`, `tool_result`,
and `spawn`. `on_step` owns *only* `STEP_ERROR`; its `STEP_TOOL_CALL` and
`STEP_MODEL` firings are ignored because `on_record` holds the complete versions.

**Failure isolation.** `on_record` must be wrapped in the same
catch-and-continue that `add()` already applies to `on_step`: a failing log write
can never abort a run.

### 3.3 Location

```
<workspace>/agent-runs/            undotted — dotted dirs are excluded by _is_hidden_within
  .gitignore                       contains "*"; created only if absent
  <run_id>/
    MANIFEST                       convenience metadata only (§4.3)
    logs.0001.txt
    logs.0002.txt
```

The log is written into the run's bound workspace folder, making it a
user-visible artifact they can open, diff, and keep. Path resolution goes through
the same chain the file tools use — `allowed_file_roots(write=True,
sandbox_root=_tool_sandbox_root())` → `is_within` → `is_sensitive_path` — and
never through bespoke path logic, so the writer cannot become a validation
bypass.

When no read-write workspace folder is bound, `allowed_file_roots` degrades to
sandbox-only; the writer follows it and logs under the sandbox root. The log is
always written somewhere.

## 4. Log format

### 4.1 Record structure

```
#@# 000412 run=a3f9c1 kind=primary type=tool_result tool=grep_files status=ok call=call_7 ts=2026-07-27T18:22:31.004Z bytes=1834
<content verbatim — exactly 1834 UTF-8 bytes, excluding the newline that ends it>
#@# 000413 run=b7e2d4 kind=subagent type=model tool=- status=- call=- ts=2026-07-27T18:22:48.221Z bytes=94
<content verbatim>
```

| Element | Rationale |
| --- | --- |
| `#@#` anchor | Never occurs naturally. `###` was rejected: it is a markdown H3, so every heading in fetched or generated content would false-positive. |
| **One physical line** per header | A wrapped header breaks `^#@# ` matching and detaches fields onto a continuation line no search can associate. ~130 chars — inside grep's 500-char window. |
| Zero-padded monotonic `000412` | Unique across the whole run tree; sortable; stable to reference from a truncation trailer. |
| `run=` / `kind=` | Sub-agent records are distinguishable in a shared log — a parent can search its child's *entire* trace rather than the 4,000-char summary it receives. |
| `call=<tool_call_id>` | Lets Phase 3 reconstruct valid assistant-echo/`role="tool"` pairs (§10). |
| `bytes=N` | **UTF-8 bytes of content, excluding the terminating newline. Files are parsed in binary.** Exact slicing means content containing a literal `#@#` cannot corrupt parsing. |
| `truncated=<original_bytes>` | Present **only** on a record cut by the per-record ceiling (§8.1); absent otherwise. |
| Content verbatim, unwrapped | Losslessness requires no transformation to reverse. |

### 4.2 The one accepted trade-off

Content is not wrapped, so a long content line is searchable by `grep_files` only
in its first 500 characters (`_MAX_GREP_LINE_SEARCH_CHARS`). This is accepted
because `search_run_log` (§5) is the primary read path and has no such limit.
Header lines — which carry every structured field — are short and fully
searchable by any tool.

### 4.3 Segmentation

Segments roll at 4 MB (`logs.0002.txt`, …), under `grep_files`'
`_MAX_GREP_FILE_BYTES` ceiling of 5 MB.

**A record never spans segments.** The roll decision is made *before* writing a
record that would carry the current segment past the threshold, not after
exceeding it. A split record would break the `bytes=`-exact parsing model, which
assumes a record's content lies wholly within one file. Because
`run_log_max_record_bytes` (1 MB) is well under the segment threshold (4 MB), a
single record always fits.

**Readers must tolerate a partially written trailing record.** The agent searches
its own log *while the writer is appending to it*. `search_run_log` therefore
parses up to the last record whose declared `bytes=` is fully present on disk and
ignores any trailing remainder.

**Segment discovery is glob + sort, not `MANIFEST`.** A crashed run never writes
a manifest; making it load-bearing would render exactly the runs that most need
inspection unreadable. `MANIFEST` carries convenience metadata only: run header
(model, budget, allowed tools, workspace), segment ranges, final status, and
supersession.

## 5. `search_run_log`

### 5.1 Registration

Registered as a **runtime tool**, alongside `find_tools`, `load_tools`,
`spawn_subagent`, `skill_file`, `install_skill`, and `run_skill_script` — not as
a catalog tool.

Registration follows the pattern those six already establish: a schema in
`tool_catalog`, an optional callable on `LoopDeps`, and a dispatch branch in
`run_agent_loop` guarded by `deps.<name> is not None`, appended to the existing
`elif` chain. An unrecognised name still falls through to the chain's `else`,
reaching `deps.invoke_tool` and its permission gate — so a stray call when the
tool is not offered is refused normally rather than mishandled.

Three consequences, each resolving a specific problem:

- **Consumes no `max_active_tools` slot.** That ceiling is a one-way ratchet
  (`load_tools` refuses past it, nothing unloads). A memory tool should not
  compete with real capabilities for it.
- **No config gate and no approval prompt.** `grep_files` carries the `"reads"`
  risk tag, which floors it to `ask`; an autonomous run would stall on its first
  log search. The exemption is justified rather than assumed: `search_run_log`
  reads only the current run's own log — content this agent already produced or
  received. The disclosure risk the `"reads"` tag guards against is absent.
- **Offered only when logging is active** and the log is confirmed writable
  (§7).

### 5.2 Sub-agents do not get the tool

In Phase 1, `search_run_log` is offered to the **primary agent only**.

The isolation argument is the first reason: `spawn_subagent`'s contract is *"It
sees only the task text you pass"*, and granting a child its parent's entire
history as an unremarked side effect would break that.

But scoping a child to its *own* records — the obvious middle path, and what an
earlier draft specified — turns out to be pointless. `clamp_child_budget` sets
`max_subagents=0`, so a child's subtree is only itself; children are short; and a
short run's entire history is already in its context. The tool would add surface
area and a scoping rule to buy a child nothing. Not offering it is simpler and
strictly safer.

Giving children scoped or full access is a deliberate future decision, and one
worth revisiting only once Phase 3 eviction makes a child's own history capable
of exceeding its window.

### 5.3 Interface

```
search_run_log(
    contains,                  literal substring — DEFAULT search mode
    tool, kind, type, status,  structured filters
    from_record, to_record,    range slice
    context,                   N records either side of each hit
    pattern,                   OPT-IN regex, bounded (§5.4)
)
```

Literal-first is a safety property, not a convenience. Python's `re` has no match
timeout, and `agent_service._call_with_timeout` abandons rather than kills its
worker thread, so a catastrophic-backtracking pattern keeps burning CPU past its
deadline — the documented reason `grep_files` searches only 500 characters per
line. Literal substring search is linear and cannot backtrack, so the default
mode needs no line-length cap and searches the log without limit.

### 5.4 Regex mode

`pattern=` is opt-in and carries the same 500-character-per-line guard as
`grep_files`, for the same reason. Its limitation is stated in the tool
description so the model can choose `contains=` when it needs unbounded reach.

## 6. Prompt integration

### 6.1 Truncation trailer

`_truncate_tool_result`'s trailer currently reads:

> `[truncated: {tool} returned {n} characters; showing the first {max}. Re-issue
> the call with a narrower query, or use the tool's offset/limit arguments to
> read the rest.]`

It gains a pointer to the exact record:

> `… The full result is recorded at record 000412 — search_run_log(from_record=412).`

This is the single change that makes an additive Phase 1 pay off immediately:
every truncation in the run becomes a pointer to a lossless copy, instead of an
instruction to redo the work.

### 6.2 System prompt section

A short section (~8 lines) appended when and only when logging is active and the
log is writable: the log's location, the record format in one line, the tool
name, and the operative instruction — *history in your context is truncated but
the log is complete; search it rather than guessing or re-running work.*

## 7. Failure and degradation

| Condition | Behaviour |
| --- | --- |
| Write failure (any cause) | Log once at warning; **suppress the prompt section and withdraw the tool**. The model is never told to search a log that does not exist. |
| No `rw` workspace folder bound | Fall back to the sandbox root. |
| `agent-runs/` unwritable | As write failure. |
| Concurrent access | Lock plus `O_APPEND`. Tool calls run under `_call_with_timeout` on a worker thread, so the writer is reachable from more than one thread. |
| Durability | `flush()` per record (survives process crash); `fsync` at segment roll and run end (survives power loss). Per-record `fsync` into a user's project directory is rejected as wasteful. |

`agent_runtime.add()` swallows hook exceptions by design, so a broken log can
never abort a run. That safety property is exactly why silent absence must be
made loud at the prompt/tool level instead.

## 8. Configuration

`[agents]`:

| Key | Default | Meaning |
| --- | --- | --- |
| `run_log_enabled` | `true` | Master switch |
| `run_log_dir_name` | `agent-runs` | Directory name within the workspace |
| `run_log_max_record_bytes` | `1_000_000` | Per-record ceiling (§8.1) |
| `run_log_segment_bytes` | `4_000_000` | Segment roll threshold |
| `run_log_evict_enabled` | `false` | Phase 3 (§10) SEND-payload eviction. Off by default: existing runs stay byte-identical until opted in. |

**Retention: never auto-delete.** Silently pruning files from a user's project
directory is the wrong default. The Console surfaces total size and offers
explicit cleanup instead.

### 8.1 The one bounded exception to losslessness

`run_log_max_record_bytes` is a deliberate exception to §4's losslessness
guarantee, and the only one. A single record exceeding it is written truncated,
and:

- `bytes=` reports the **bytes actually written**, never the original — so
  parsing stays exact and a truncated record cannot desynchronise the file.
- The record header gains `truncated=<original_bytes>` so both a human and
  `search_run_log` can see that content was dropped and how much.

Without a per-record ceiling, a single pathological tool result (a multi-gigabyte
file read) could fill a user's disk inside one run. The default of 1 MB is far
above `max_tool_result_chars` (16,000), so in normal operation no record is ever
truncated and the log remains fully lossless.

## 9. Security

### 9.1 Content sensitivity

The log contains full, untruncated tool results — `read_file` contents, web
fetches, MCP responses. Today that material lives only in memory and in a
2,000-character DB field. Writing it into a workspace folder, which is very
likely a git repository, is a new disclosure path. Mitigations:

- `.agent-runs/.gitignore` containing `*`, created **only if absent**, never
  overwritten — creating files in a user's repository is itself a mutation.
- **Corrected by TASK-1270 (2026-07-28, shipped), superseding the paragraph
  below as originally written.** This section originally said the
  directory was undotted *deliberately* everywhere, on the premise that
  `glob_files`/`grep_files` glob `_tool_sandbox_root()` alone and could
  never reach a workspace folder root — so dotting there would only ever
  hide the log from `read_file`/`write_file`/`list_directory`, tools that
  were never going to see it anyway. **TASK-850 made that premise false**:
  both `glob_files` and `grep_files` now resolve every root
  `allowed_file_roots()` returns — the sandbox root *and* every bound
  workspace folder (`Tools/file_operation_tools.py`,
  `GlobFiles.execute`/`GrepFiles.execute` via
  `_iter_candidates_across_roots`) — so an undotted log directory landing
  in a bound workspace folder became a root those two tools search
  directly. A spawned sub-agent, which inherits its parent's tool
  allow-list, could therefore `grep_files`/`glob_files` its way to the
  parent's entire log — the exact disclosure the sandbox-fallback dotting
  (below) was introduced to prevent, reopened for the workspace case by an
  unrelated change. TASK-1270 reproduced this with a planted secret
  (`Tests/Agents/test_run_log_workspace_isolation.py`,
  `Tests/Agents/test_run_log_sandbox_isolation.py::
  test_bound_workspace_folder_also_gets_dotted_and_hidden_from_grep`,
  `Tests/Agents/test_run_log_writer.py::
  test_workspace_folder_outside_the_sandbox_also_gets_dotted_and_hidden`)
  and **shipped the fix**: `RunLogWriter.bind()` now dots the directory
  name **unconditionally**, in both the sandbox-fallback and the
  bound-workspace case, with the sandbox-fallback-only conditional (and
  the `_root_kind` side channel it depended on) deleted entirely. The
  directory is `.agent-runs` in every configuration. This costs nothing
  for the log's original purpose: a dotted directory stays a fully
  visible, ordinary directory to the *user* — `ls -a` lists it, editors
  show it, it is fully diffable and keepable in the user's own repository.
  It is hidden only from this app's own sandboxed file tools, which is
  exactly the intent regardless of which root the log happened to land
  under. `search_run_log`/`load_records` are unaffected either way: they
  glob `log_dir` directly and never route through
  `validate_path`/`_is_hidden_within`.
- Sensitive paths remain refused at the writer via `is_sensitive_path`.

### 9.2 Path validation

The writer resolves through `allowed_file_roots` → `is_within` →
`is_sensitive_path`. It never constructs or validates paths itself.

### 9.3 Regex exposure

Covered in §5.3–5.4: literal default, bounded opt-in regex.

### 9.4 `grep_files` / `glob_files` now also search workspace folders (TASK-850)

**Corrected by TASK-850, which shipped the "its own task" this section
originally deferred to.** This section originally said `grep_files`/
`glob_files` glob and containment-check against `_tool_sandbox_root()`
alone and never consult `allowed_file_roots`, unlike `read_file` — making a
workspace-folder log readable by exact path but **not searchable** by
them — and that correcting the asymmetry belonged in its own task (also
listed under §11 Deferred).

That task shipped. TASK-850 ("Scope glob_files and grep_files to workspace
folder roots") made both tools resolve every root `allowed_file_roots()`
returns, exactly like `read_file`/`write_file`/`list_directory` already
did (`_iter_candidates_across_roots` and its two callers in
`Tools/file_operation_tools.py`). The path-vs-search asymmetry this
section described no longer exists: a workspace-folder log is now
reachable by search too, not just by exact path.

That closed the original gap for legitimate use, but reopened the
sub-agent run-log disclosure the sandbox-fallback directory dotting was
meant to prevent, for the bound-workspace case specifically — this
design's premise that a workspace-folder log is "not searchable" is what
made an undotted directory name seem safe there. See §9.1 for TASK-1270,
which closed that follow-up by dotting the directory name
unconditionally, in every configuration.

Ripgrep-as-search-backend, the other half of this section, remains
out of scope for the reasons originally given: it would remove every cap
listed in §5.3, but it introduces an external binary dependency this
repository has no precedent for, and its discovery would conflict with the
established rule that binaries are never resolved via ambient `PATH`
(`resolve_interpreter` searches only `SCRUBBED_PATH`).

## 10. Phasing

| Phase | Contents |
| --- | --- |
| **1 — this spec** | Writer, format, `search_run_log`, prompt section, truncation trailer. Additive: `messages` untouched, existing runs byte-identical. |
| **2** | `run_log_stats` / `run_log_slice` aggregation tools; TASK-870's Console reader. |
| **3 — 1:1 PRO-LONG** | Eviction at the history-assembly seam: keep recent rounds, replace older ones with a pointer. |

Phase 3 is cheap by construction because the log is authoritative by then and
`call=` preserves pairing. **The hazard it must respect, recorded now because it
is expensive to retrofit:** the native protocol pairs an assistant `tool_calls`
echo with its `role="tool"` replies by `tool_call_id`. Eviction that orphans
either half produces a request strict providers reject. Any eviction policy must
operate on whole call/result groups.

**Implemented by TASK-1272 (`Agents/run_log_eviction.py`).** Applied at the SEND
seam only (`agent_service._make_call_model`'s `call_model` closure), immediately
before the provider call — `run_agent_loop`'s own `messages` list is never
touched, so cycle detection, retries, and step accounting are unaffected; only
what is SENT shrinks. Reuses `bound_messages_to_window` (§14.1) rather than
reimplementing it, extended with an optional `is_turn_boundary` predicate
(`console_history_budget.py`, default unchanged so every Console call site
stays byte-identical).

One design decision worth recording because it is not obvious from §14.1's
"reuse, don't reimplement" framing alone: the reused primitive's unit of
"turn" is Console's own — anchored on the last human-authored message —
which, inside a single agent run (only one such message, at the start),
would collapse the *entire* run's own growth into one undroppable "current
turn" and evict nothing while a run is actually in progress. That defeats
goals 1 and 2 (§10.1) for exactly the long-single-run, small-context-model
case this phase exists to serve. TASK-1272 therefore uses a finer *round*
boundary instead — every assistant-authored message starts a new round; a
native `role="tool"` reply or a fence `role="user"` tool-result row is a
continuation of it, never a boundary — verified to still avoid orphaning any
call/result pair for both protocols. `console_history_budget`'s own
Console-facing default is untouched.

Gated on `log_active` (§7's gate on the tool and the prompt section, reused
verbatim — eviction never runs without a durable log to point the model back
at) AND the new `[agents] run_log_evict_enabled` flag (§8), off by default.
When something is dropped, a synthetic `role="user"` note (not `role=
"system"`, which several local chat templates reject mid-conversation) is
spliced in where the dropped rounds were, naming a count and
`search_run_log` — never a specific record number, which the loop cannot
derive accurately for a dropped round.

### 10.1 Which goals each phase actually delivers

| Goal (§2.1) | Phase 1 | Phase 2 | Phase 3 |
| --- | --- | --- | --- |
| 1. Long-horizon runs unbounded by context | — | — | **Yes** |
| 2. Small-context local models | — | — | **Yes** |
| 3. Queryable, crash-durable run history | **Durability + per-run search** | Console reader, aggregation, cross-run search | — |
| 4. 1:1 PRO-LONG reachable as config | Format and seams built for it | — | **Delivered** |

Stated plainly so nobody mistakes Phase 1 for the whole thing: **Phase 1 does not
reduce context usage.** It makes truncated content *recoverable* and run history
*durable*. Goals 1 and 2 are delivered by Phase 3.

### 10.2 A known Phase 1 failure mode

Because Phase 1 evicts nothing, a `search_run_log` result enters history like any
other tool result — capped at `max_tool_result_chars`, counted against
`max_total_tokens`. An agent that searches its log heavily therefore *grows* its
context faster than one that does not, and in the worst case can thrash: search →
large result → context pressure → truncation → search again.

Three things bound this rather than solve it: search results are capped at 16,000
characters like every other result; the `context=` parameter defaults low so hits
return records rather than whole regions; and the system-prompt section (§6.2)
directs the model to search for *specific* content it knows it needs, not to
browse. It is genuinely resolved only by Phase 3, and that is a further argument
for not treating Phase 1 as the finished feature.

## 11. Deferred

- **Model-authored code over the log.** The paper's largest single gain
  (+10.1pp). `skill_script_runner` provides most of the hardening already
  (scrubbed `PATH`, `RLIMIT_CPU/NOFILE/FSIZE/AS`, process-group kill, capped
  sinks, throwaway `HOME`, `shell=False`), but it is resource-confined, not
  filesystem-confined, and its trust model assumes a *trusted skill bundle* —
  model-authored code is neither. Needs its own confinement design.
- **Ripgrep backend** (§9.4).
- ~~Extending `glob_files`/`grep_files` to workspace roots~~ — shipped as
  TASK-850 (§9.4); see TASK-1270 for the sub-agent-disclosure follow-up it
  reopened.
- **Cross-run search.** The log is per-run; searching *across* runs is a natural
  Phase 2+ extension.

## 12. Testing

- **Format round-trip:** write → parse → byte-identical content, including
  content containing a literal `#@#` and multi-byte UTF-8 at a `bytes=` boundary.
- **Counter monotonicity** across a parent plus two sub-agents; no duplicate
  record numbers.
- **Segmentation:** roll at threshold; search spanning segments; a run with a
  missing `MANIFEST` remains fully searchable; **no record ever spans a segment
  boundary**, including one written when the segment is just under threshold.
- **Concurrent read during write:** a search issued while a record is
  half-written returns every complete record and ignores the remainder.
- **Capture completeness:** results from `find_tools`, `load_tools`,
  `run_skill_script`, an MCP tool, a skill tool, and a `review_tool_calls`
  refusal each appear in the log — the branches a service-side wrapper would have
  missed.
- **Primary-only offer:** a sub-agent is not given `search_run_log`, and a stray
  call to it from a child is refused through the normal permission path.
- **Capture fidelity:** a 50,000-character tool result appears in full in the log
  while history shows the 16,000-character truncation plus a record pointer.
- **Degradation:** no workspace bound; unwritable directory; write failure
  mid-run — each suppresses the prompt section and withdraws the tool.
- **Sub-agent scoping:** a child cannot retrieve parent-only records.
- **Concurrency:** simultaneous runs do not interleave or corrupt records.
- **Live verification:** a real provider run per the repository's
  live-verification rule — tests alone have repeatedly missed defects here.

## 13. Review findings register

Twenty-eight issues were found across three review passes — the first over the
architecture, the second over the record format, the third over this document
after it was written. Each is addressed above; they are recorded here so
reviewers can check the reasoning rather than rediscover it.

**First pass — architecture.**

| # | Finding | Resolution |
| --- | --- | --- |
| 1 | `on_step` truncates to 200/2000 chars — cannot carry a lossless log | Capture in service closures (§3.2) |
| 2 | `grep_files` searches only 500 chars per line | `search_run_log` is the primary path; headers stay short (§4.2) |
| 3 | Grep caps (5 MB/file, 200k lines, 200 matches) degrade on long runs | Segmentation; `search_run_log` is uncapped (§4.3, §5.3) |
| 4 | `grep_files` is `ask`-floored on every call | `search_run_log` is a runtime tool, justified exemption (§5.1) |
| 5 | Wiring on the caller's `on_step` misses non-Console callers | Writer built inside `AgentService` (§3.1) |
| 6 | Step indices collide across the run tree | Tree-scoped monotonic counter (§3.1, §4.1) |
| 7 | Secrets reach a user's git repository | `.gitignore`, undotted dir, sensitive-path refusal (§9.1) |
| 8 | Silent log failure while the prompt advertises the log | Suppress prompt section and withdraw tool (§7) |
| 9 | Phase 3 eviction can orphan `tool_call_id` pairs | `call=` recorded now; whole-group eviction required (§10) |
| 10 | Writer could bypass path validation | Reuses the file tools' chain (§9.2) |
| 11 | Unbounded disk growth | Per-record cap, segmentation, never auto-delete (§8, §8.1) |
| 12 | Supersession invisible in the log | Recorded in `MANIFEST` (§4.3) |

**Second pass — record format.**

| # | Finding | Resolution |
| --- | --- | --- |
| 13 | **Blocker:** a workspace-folder log is unsearchable by `grep_files`, which globs the sandbox root alone and never consults `allowed_file_roots` — unlike `read_file` | The workspace location stands; `search_run_log` replaces grep as the Phase 1 read path (§5, §9.4) |
| 14 | Dropping ripgrep re-arms catastrophic backtracking in `search_run_log` | Literal-substring default; regex opt-in and bounded (§5.3–5.4) |
| 15 | `###` anchor collides with markdown H3 in generated and fetched content | `#@#` (§4.1) |
| 16 | A header wrapped across two lines breaks `^#@# ` matching and detaches fields | One physical line, mandatory (§4.1) |
| 17 | `bytes=N` ambiguous — UTF-8 bytes or characters, newline included or not | Defined exactly: UTF-8 bytes, excluding the terminating newline, parsed in binary (§4.1) |
| 18 | `MANIFEST` as a correctness dependency makes crashed runs unreadable | Discovery is glob + sort; manifest is metadata only (§4.3) |
| 19 | Writer reachable from multiple threads via `_call_with_timeout`; fsync policy unstated | Lock + `O_APPEND`; flush per record, fsync at roll and run end (§7) |
| 20 | `tool_call` args were unspecified — `write_file` args *are* file contents | Full args logged (§3.2) |

**Third pass — this document.**

| # | Finding | Resolution |
| --- | --- | --- |
| 21 | **Contradiction:** §3.2 claimed `agent_runtime.py` is unmodified while §5.1 registered a runtime tool, which requires a `LoopDeps` field and a dispatch branch in the loop | Claim corrected; the modification is stated and shown to follow the pattern the six existing runtime tools already establish (§3.2, §5.1) |
| 22 | **Incomplete capture:** service-side wrappers miss every runtime-tool result — `find_tools`, `load_tools`, `skill_file`, `install_skill`, `run_skill_script` — because the loop dispatches those outside `deps.invoke_tool` | Replaced three wrappers with one `on_record` hook at the loop's single `content`-assembly point, which is uniform across all branches (§3.2) |
| 23 | **Ordering bug:** the writer cannot name its `<run_id>/` directory in `run_turn`; `create_run` is the first statement of `_run_one` | Two-phase construction: unbound in `run_turn`, `bind(run_id)` by the primary `_run_one` (§3.1) |
| 24 | Sub-agent scoping was specified but buys a child nothing — `max_subagents=0` makes its subtree just itself, and its short history is already in context | Tool is offered to the primary agent only in Phase 1 (§5.2) |
| 25 | A record could span a segment boundary, breaking `bytes=`-exact parsing | Roll decided *before* writing; a 1 MB record cap under a 4 MB threshold guarantees fit (§4.3) |
| 26 | Readers would hit a partially written trailing record — the agent searches the log while the writer appends | Parse to the last complete record; ignore the remainder (§4.3) |
| 27 | The document did not say which goals each phase delivers, and never stated that Phase 1 can *increase* context pressure | Goal/phase matrix and the thrash failure mode both documented (§10.1, §10.2) |
| 28 | task-322 was described as an unshipped, competing proposal. It shipped 2026-07-22 as `console_history_budget.py`, already bounds the agent run's *starting* history, and is the primitive Phase 3 should reuse — but its `_group_turns` splits on `role == "user"`, which the fence protocol uses for tool results | §14.1 rewritten with the reuse requirement and the fence-protocol grouping trap |

## 14. Related work in this repository

- **TASK-870** — Console tool-result display caps: Settings control and
  full-content access. The Console currently shows 160–200 characters of a result
  the model received at up to 16,000. The log is what makes "read the full
  content" answerable.
- **task-1265** — `glob_files`/`grep_files` ignore workspace folder roots that
  `read_file` honours (§9.4). Filed rather than fixed here.
- **task-326 / task-327** — token budget and durability hardening; already merged
  into the substrate this builds on.

### 14.1 task-322 is Done, and Phase 3 must build on it

**task-322 shipped on 2026-07-22**, adding `Chat/console_history_budget.py`
(`count_console_messages_tokens`, `_group_turns`, `bound_messages_to_window`).
An earlier draft of this document described it as an unshipped proposal for lossy
truncation and an alternative to Phase 3. Both halves were wrong.

What it means in practice:

- **The entry point is already bounded.** `console_chat_controller.py:4395` does
  `agent_messages = list(provider_messages)`, so the history an agent run *starts*
  from is already trimmed to the model window. The remaining unbounded quantity
  is growth *during* a run — which sharpens this design's problem statement
  rather than competing with it.
- **Phase 3 must reuse `bound_messages_to_window`, not reimplement it.** Window
  lookup, safety margin, reply reservation, and system-prefix preservation are
  all solved there and should not be written twice.

**The trap, recorded now because it will not be obvious later.** `_group_turns`
splits history on `role == "user"` boundaries, and its docstring notes that
dropping a whole group never splits a tool_call/tool_result pair *"were tool rows
ever present in the payload"* — they never are, on the Console send path it was
built for. In an agent run they are, and the two tool-call protocols encode them
differently (`agent_runtime._append_tool_result`):

- **Native protocol** appends `{"role": "tool", "tool_call_id": …}`. `_group_turns`
  keeps that inside the current group. Correct as-is.
- **Fence protocol** appends `{"role": "user", "content": "Tool result for …"}`.
  `_group_turns` reads that as **the start of a new turn**, splitting an
  assistant turn from the tool result that answers it.

So reuse is safe for native-protocol runs and **incorrect for fence-protocol
runs** until grouping understands the fence convention. Phase 3 must either teach
`_group_turns` about it or group on the run's own record structure, where `call=`
already pairs them unambiguously (§4.1).
