# hermes-agent vs tldw_chatbook — full feature-surface parity report

## 1. Header

### Tips compared

| repo | ref | tip | date | subject |
|---|---|---|---|---|
| hermes-agent (NousResearch) | `origin/main` | `a0a63a1bc21115ba8da7fed6fdb695522dd37c96` | 2026-08-31 06:28:25 -0700 | `fix(gateway): username-based DISCORD_ALLOWED_USERS no longer locks out the operator after one turn` |
| tldw_chatbook (rmusser01) | `origin/dev` | `1f2d03beb0a2cd82985e395f94bfb05ee992ca7f` | 2026-08-30 22:53:54 -0700 | `Merge pull request #2253 from rmusser01/codex/task-23113-6-trace-privacy` |

The local hermes clone was 114 days stale (`HEAD` = `dae94fa6526dec0c7660276a4d875cebc6e344f6`, 2026-05-09). It was fetched
and every hermes claim below is read from `origin/main`, never from that HEAD. Chatbook was read from a detached
`origin/dev` worktree, never from the dirty working tree.

Scale, for context only (not a verdict): hermes 1,283 `.py` / 1,020,506 LOC (excl. tests, node_modules) plus 2,745
`.ts`/`.tsx` / 652,730 LOC; chatbook 1,996 `.py` / 1,390,088 LOC in the `tldw_chatbook/` package.

### Hermes source mix

Priority order used: code first, then `RELEASE_v*.md`, then the releases API.

- **`RELEASE_v*.md` no longer exist in the repo.** They were deleted at `64202200a6` (2026-06-03),
  *"chore: remove committed RELEASE_v\*.md changelogs from repo root (#37855)"*. The last committed file was
  `RELEASE_v0.15.1.md`. Everything after v0.15.1 is only available from
  `gh api repos/NousResearch/hermes-agent/releases`.
- **30 releases via the API**, v0.2.0 (2026-03-12) through **v0.20.6 (2026-08-27)**, scraped to
  `hermes-releases-all.md` (825 KB) for grepping.
- **Claim mix: ~230 capability rows; roughly 24 rest on release notes alone (~10%), the rest are `[CODE]`.**
  The release-note-only claims cluster in two places: hermes desktop/Electron features that have no in-tree Python
  to read, and date anchors for when a feature was introduced. Every scheduling row was deliberately re-derived
  from `cron/` source because the prior audit was release-note driven.

### Delta since the prior parity pass

TASK-18936 read release notes **v0.13 – v0.20.4**. The delta is **v0.20.5** and **v0.20.6** — both thin rollups with
no curated feature list (curated notes deferred to v0.21.0). Their headline paragraphs name, as RELEASE-NOTE-only
evidence: keyless web tier with 5-vendor ring failover, fuzzy `/model` picker, Ctrl+P command palette, `hermes update`
receipts, fleet `--plan`, `hermes worktree list/prune`, multi-question clarify, **cron jobs gaining persistent memory
and per-job reasoning effort**; then consent-gated real-profile browsing, a **50+ server remote MCP catalog
expansion**, TTL caching for `web_search`/`web_extract`, **lean-tail compression as the default**, **multi-query
`tool_search` with stemming**, **opt-in OS-keychain encryption for stored secrets**, and **cron durable-incident acks**.

### Corrections to the brief

- **The ADR is 077, not 076.** `backlog/decisions/077-server-offloaded-scheduled-agent-tasks.md`; its own header
  records `Renumbered: 072 → 076 → 077`. `076` on origin/dev is `076-library-lifecycle-progressive-disclosure.md`.
  (Note `077` now itself collides with `077-console-bounded-rail-section-scrolling.md`.)
- **`environments/` is not a top-level hermes directory** — it is `tools/environments/`.
- Task states on origin/dev at `1f2d03beb`: **Done** 18926, 18932(+.1–.4), 18937, 18938, 18939 · **In Progress**
  18940 · **To Do** 18920, 18921, 18922, 18923, 18924, 18925, 18927, 18928, 18929, 18930, 18931, 18933, 18934,
  18935, 18936.

### Left UNVERIFIED

Carried forward honestly, with the check that would settle each:

| # | Claim | Check to run |
|---|---|---|
| 1 | Server-side status of ADR-077 (definition scheduler feed, `agent_task` consumer, timeout status, run-now endpoint) | Lives in the `tldw_server` repo — neither worktree has it. Read `scheduled_tasks_control_plane.py` / `reminders_scheduler.py` there. |
| 1b | Hermes's local-inference process management — **resolved during synthesis**, listed here only because it was on an area agent's list | `grep -rn "Popen\|subprocess.run\|create_subprocess" hermes_cli/ agent/ tools/ \| grep -iE "llama\|ollama\|vllm\|lmstudio\|mlx"` → 0 hits; hermes talks to local servers but does not launch them |
| 2 | Whether hermes resumes an in-flight run after a hard crash (vs. restoring history via `--resume`) | `grep -rn "def .*reclaim\|orphan\|stale_lease" gateway/turn_lease.py gateway/run.py` |
| 3 | Hermes `ui-tui/` (Ink/React, 463 files) pane and command inventory | `ls ui-tui/src/components/`; grep `ui-tui/src/app/` for `useInput(`, diff against `hermes_cli/commands.py` |
| 4 | Hermes desktop app (`apps/`, 2113 files) surface depth — watch-windows, rebindable shortcuts, marketplace themes are RELEASE-NOTE only | `ls apps/desktop/src/`; grep for `notification`, `shortcut`, `theme` registration |
| 5 | Whether chatbook's `mcp-unified==0.2.1` dependency itself implements HTTP/SSE serving that `MCP/server.py:1016` merely refuses to expose | Read the installed `mcp_unified/gateway/__init__.py` for a `serve_http`/`serve_sse` export |
| 6 | Whether hermes's non-docker sandboxes (modal/daytona/vercel_sandbox) actually enforce network isolation or are merely trusted as isolated by `tools/approval.py:4075` | `grep -n "network\|egress" tools/environments/{modal,daytona,vercel_sandbox}.py` |
| 7 | Whether hermes creates Gemini explicit cached content (only `cachedContentTokenCount` read-back was found, `agent/gemini_native_adapter.py:804`) | `grep -rn "cachedContents\|create_cached_content" agent/ providers/` |
| 8 | Whether chatbook's per-turn `cache_control` breakpoint is enabled on the default Console path | `grep -n "prompt_caching" tldw_chatbook/Chat/console_provider_gateway.py`, follow into `chat_with_anthropic` |
| 9 | Whether chatbook enforces skill-frontmatter `allowed-tools` at invocation (`intersect_skill_tools` exists at `Agents/tool_catalog.py:1139`; caller not traced) | `grep -rn "intersect_skill_tools" tldw_chatbook/` and read the caller |
| 10 | Whether hermes publishes a wheel to PyPI (`package.json:5` is `"private": true`; v0.19.0 notes warn pip/Homebrew installs unsupported) | `curl https://pypi.org/pypi/hermes-agent/json` |
| 11 | Whether chatbook's `TLDW_CONFIG_PATH` override rejects symlink redirection | Read `tldw_chatbook/Utils/private_paths.py` for `PrivatePathStatus` symlink handling |
| 12 | Hermes's per-tool `get_max_result_size` default table, needed to state its spill threshold numerically against chatbook's 32 KiB | `grep -n "get_max_result_size\|max_result_size" tools/registry.py tools/budget_config.py` |

Resolved during synthesis (were on the area agents' unverified lists, checked directly):

- `local_watchlist_runs` **is** a real per-run ledger — `DB/Subscriptions_DB.py:936-950` (status/started_at/finished_at/
  stats_json/error_msg/log_text), with orphan reconciliation at `Subscriptions/startup_reconcile.py:164,195`. The
  "no execution history" gap therefore applies to **reminders and briefings only**, not watchlists.
- **No agent-facing scheduling tool.** `grep -rn "reminder|scheduled_task|SchedulingService" tldw_chatbook/Tools
  tldw_chatbook/Agents --include="*.py"` → 0 hits.
- **`Tools/file_operation_hooks.py` is dead scaffolding** — `Tests/Tools/test_system_a_is_retired.py:73,80` pins that
  `install_claude_code_hooks` has no callers.
- **The permission audit trail is MCP-only.** `mcp_execution_log.jsonl` is written solely from
  `MCP/unified_control_plane_service.py:2261`; raw-shell and local-builtin decisions are not recorded.
- **Chatbook's MCP client is stdio-only.** `grep -rn "streamable|sse_client|transport" tldw_chatbook/MCP/*.py`
  yields only `"stdio"` (`MCP/server.py:1010,1016`) and `"in_process"` (`MCP/local_runtime_delegate.py:310,379`);
  `MCP/local_store.py` has no URL field.
- **`Terminal/` is unwired to any UI** — grep over `UI/` + `Widgets/` returns one unrelated comment.
- **`register_fallback_resolver` has no production call site** — `Chat/console_skill_resolver.py:19-21` documents the
  factory was deleted; bare `/skill-name` does not resolve.
- **Per-tool timeout is not clamped to remaining wall budget** — `_call_with_timeout(fn, seconds, …)`
  (`Agents/agent_service.py:1522`) takes an absolute bound; engine default 300 s (`Agents/agent_models.py:400`),
  Console raises it to 3600 s (`Chat/console_agent_bridge.py:408`), while `max_wall_seconds` is checked only at loop
  top (`Agents/agent_runtime.py:1175`). A single tool call can overshoot the run budget.
- **Prometheus listener binds without a config gate** — `app.py:16900` calls `init_metrics_server()` unconditionally
  at boot; the only gate is dependency presence (`Metrics/metrics.py:255`). Installing the `[debugging]` or `[dev]`
  extras binds an unauthenticated HTTP listener on port 8000 with no opt-in setting.

---

## 2. Matrix

Verdicts are one of `PARITY` / `CHATBOOK AHEAD` / `HERMES AHEAD` / `N/A BY DESIGN`. The **Source** column is the
hermes evidence tag plus the prior-task reference where one exists (`NEW` = not covered by 18920–18940 or ADR-077).

### 2.1 Agent control loop

| capability | hermes | chatbook (file:line) | verdict | source |
|---|---|---|---|---|
| Main loop shape | `run_conversation` while-loop over API calls, `agent/conversation_loop.py:1899,2094` | pure `run_agent_loop` with injected `LoopDeps`; service owns all impurity — `Agents/agent_runtime.py:890,1159`, `:353` | PARITY | CODE |
| Budget dimensions | one `IterationBudget`, thread-safe consume/refund, `agent/iteration_budget.py:17` | four: `max_steps`, `max_model_turns`, `max_wall_seconds`, `max_total_tokens` — `Agents/agent_runtime.py:1169-1179`; `agent_models.py:358-400` | PARITY | CODE |
| Shipped wall-clock default | `run_budget_seconds` default `None` = unbounded, `hermes_cli/config_defaults.py:56` | 86400 s wall / 3600 s per tool / 25 M tokens, re-read per run — `Chat/console_agent_bridge.py:452,455,459,493` | CHATBOOK AHEAD | CODE / NEW |
| Graceful budget wrap-up | at 80 % appends a cache-safe notice to the newest tool msg; on exhaustion a tools-stripped summary call — `agent/conversation_loop.py:195-240`, `run_agent.py:8650` | hard stop: `STEP_ERROR "step budget exhausted"` → `RUN_STUCK` — `Agents/agent_runtime.py:1169-1179` | HERMES AHEAD | CODE / NEW |
| Mid-run steering of the **primary** agent | `AIAgent.steer(text)` drains onto the last tool result before the next model call, plus a pre-API drain — `run_agent.py:3544,3692`; `agent/conversation_loop.py:2180-2235` | no channel: `drain_mailbox` is `None` for a primary by design; typed text queues for the next turn — `Agents/agent_runtime.py:1196-1230`; `Agents/agent_service.py:3486`; `Chat/console_prompt_queue.py:60` | HERMES AHEAD | CODE / NEW |
| Active-turn redirect (cancel model call, keep partials, retry same turn) | `AIAgent.redirect(text)` aborts only the in-flight request, keeps completed tool results, re-runs the turn — `run_agent.py:3591-3690` | Stop is terminal → `RUN_CANCELLED`, stream settled "Response stopped."; correction becomes a new turn — `Chat/console_chat_controller.py:13048-13126`; `Agents/agent_runtime.py:1352-1362` | HERMES AHEAD | CODE / NEW |
| Busy-input policy | 3-way `display.busy_input_mode` ∈ interrupt/queue/steer + `/busy` — `gateway/run.py:10116,10153-10178` | queue-only, but with a full pause state machine and edit/move/remove/clear — `Chat/console_prompt_queue.py:60-76`; `Chat/console_chat_controller.py:4279,4306,4326,4340` | HERMES AHEAD (mode choice) · CHATBOOK AHEAD (queue mgmt) | CODE / NEW |
| Stop / interrupt | `interrupt()` + `hard_interrupt()` fanned out to tool threads and children — `run_agent.py:3304,3466`; `tools/interrupt.py:1` | `stop_active_run` → `_signal_stop` → `should_cancel`, polled at loop top, post-transport, per tool — `UI/Screens/chat_screen.py:15479`; `Agents/agent_runtime.py:1160,1352,1634` | PARITY | CODE |
| Approval revocation on stop | not found (`grep -rn "revoke" agent/ tools/`) | emits `STEP_APPROVAL_REVOKED` per already-approved call — `Agents/agent_runtime.py:1634-1652` | CHATBOOK AHEAD | CODE / NEW |
| Global resumable pause | `ESTOP` sentinel file, checked by cron + kanban + gateway, fail-safe — `agent/estop.py:1-174` | none (named grep `estop|emergency stop|pause_all_agents` → 9 hits, all TTS false positives) | HERMES AHEAD | CODE / NEW |
| In-loop error retry | inner retry loop w/ jittered backoff, 429 rotation, OAuth refresh, compression restart, length continuation — `agent/conversation_loop.py:2999-3009`; `agent/turn_retry_state.py:1`; `agent/error_classifier.py:30-60` | none: `call_model` exception → `STEP_MODEL_ERROR` → raise → `RUN_ERROR` (named grep `fallback_model\|retry_model\|max_retries\|backoff` over `Agents/` → 0) — `Agents/agent_runtime.py:1291-1302` | HERMES AHEAD | CODE / NEW |
| Model fallback chain | `_try_activate_fallback()` walks the chain and resets retry count — `agent/conversation_loop.py:3516-3521,3592` | none (same grep) | HERMES AHEAD | CODE / NEW |
| Outer-loop error cap | `_MAX_OUTER_LOOP_ERRORS = 8` per turn — `agent/conversation_loop.py:358,8783` | none — first loop exception ends the run, `Agents/agent_runtime.py:1293` | N/A BY DESIGN (fails fast on first error; nothing to count) | CODE / NEW |
| Empty-response handling | two consecutive zero-token empties ⇒ deterministic, skip retries and fall back; cost-aware retry budget — `agent/empty_response_guard.py:1-75` | empty text + no tool calls = `RUN_DONE` — `Agents/agent_runtime.py:1427` | HERMES AHEAD | CODE / NEW |
| Repetition / no-progress breaker | two layers: `agent/tool_guardrails.py:109-127` thresholds + `agent/repetition_guard.py:1-20` | `_detect_cycle` smallest-repeating-period over `(name,args)` tail → `RUN_STUCK` naming the tool — `Agents/agent_runtime.py:853,1654-1689` | PARITY | CODE · denial-specific breaker = task-18929 |
| Premature-finish nudges | `agent/verification_stop.py:1-16` (edited code w/o evidence), `agent/kanban_stop.py:1-20` | none — no tool calls = `RUN_DONE` unconditionally, `Agents/agent_runtime.py:1427` | HERMES AHEAD | CODE / NEW |
| Parallel tool batch | concurrent dispatch under a shared, dynamically-extended batch deadline w/ worker cap — `agent/tool_executor.py:122,195-205,448-463` | strictly sequential `for call in calls:`, per-call daemon-thread timeout — `Agents/agent_runtime.py:1612`; `Agents/agent_service.py:1522` | HERMES AHEAD | CODE / NEW |
| Human-wait pauses the tool deadline | approval waits excluded from the batch clock — `agent/tool_executor.py:448-463` | refcounted marks; `_call_with_timeout` re-arms while `human_input_wait_active(run_id)` — `Agents/human_input_wait.py:1-85`; `agent_service.py:1527,1558` | PARITY | CODE |
| Unified deadline primitive | `agent/deadline.py:1-66` — config-first resolution, macOS `time_t` clamp, timer-driven bounded exec, process-tree kill | three unrelated per-site mechanisms (httpx read timeout, daemon thread, loop-top check) — `Chat/console_provider_gateway.py:150`; `Agents/agent_service.py:1522`; `agent_runtime.py:1175` | HERMES AHEAD | CODE / NEW |
| Stream stall watchdog | stale-reasoning kill + cross-turn breaker after 5 — `agent/reasoning_timeouts.py:116`; `agent/stream_single_writer.py:1-22` | none; only the httpx read timeout bounds a dribbling stream — `Chat/console_provider_gateway.py:150` | HERMES AHEAD | CODE / NEW |
| Stream fence gate | provider-native streaming behind a single-writer fence — `agent/stream_single_writer.py:1` | incremental `StreamGate`: proves each fence candidate, resumes past look-alikes, streamed text is always an exact prefix of final visible text — `Agents/agent_stream.py:52-176` | CHATBOOK AHEAD | CODE / NEW |
| Sub-agent spawn | `delegate_task` single/batch/parallel, fresh conversation, inherited toolsets minus blocked — `tools/delegate_tool.py:1-60,1861` | `spawn_subagent` inline or threaded; child budget derived from parent, depth clamped to 1 — `Agents/agent_runtime.py:1791-1854`; `agent_models.py:684-696` | PARITY | CODE |
| Background delegation | `delegate_task(background=true)` → completion queue drained into a new turn — `tools/async_delegation.py:1-35` | threaded fleet children outlive their turn; `_FleetDrainedSignal`, `wait_agents`/`check_agents` — `Chat/console_agent_bridge.py:1940-2000,4076-4099` | PARITY | CODE |
| Steering a running sub-agent | model-facing `delegate_task(action=steer)` w/ ownership check and `missed_steer` preservation — `tools/delegate_tool.py:272,529-639` | `send_to_agent` tool + user-facing steering bar, drained at the one protocol-coherent point, never splits a tool pair — `Agents/agent_service.py:5006,5033`; `agent_runtime.py:1196-1230` | PARITY | CODE |
| Cancelling a sub-agent | `action=stop` → `request_hard_interrupt`, ownership-scoped — `tools/delegate_tool.py:589-608` | `cancel_subagent` / `cancel_all_subagents`, child Event OR'd with parent cancel — `Chat/console_agent_bridge.py:6876,6919` | PARITY | CODE |
| Sub-agent filesystem isolation | per-child git worktree — `tools/subagent_worktree.py:1` (352 ln) | none per-subagent; children share parent workspace roots (grep `worktree` over `Agents/` → only `WorkspaceKind.GIT_WORKTREE` as a user workspace type) | HERMES AHEAD | CODE / NEW |
| Live child observability | per-child tail-able log — `tools/delegation_live_log.py:1-20` | segmented run log for the whole run tree + `search_run_log`/`run_log_stats`/`run_log_slice` tools + eviction — `Agents/run_log.py:1-19`; `agent_runtime.py:440-460` | CHATBOOK AHEAD | CODE / NEW |
| Per-step run persistence | session log per turn via `atomic_json_write` — `run_agent.py:3290-3300` | every `AgentStep` inserted incrementally, `capture_failed` diagnostic step on write failure — `Agents/agent_service.py:5769-5789`; `DB/AgentRuns_DB.py:1539` | CHATBOOK AHEAD | CODE / NEW |
| Crash recovery | durable turn lease + `--resume`/`/resume` restore history/model/route — `run_agent.py:8745-8770`; `cli.py:9048` | `reconcile_orphaned_runs()` flips crashed `running` rows to `error` (not resumed); separately a mid-tool-call turn IS resumable from `ProviderContinuationCheckpoint` — `DB/AgentRuns_DB.py:1689-1745`; `Chat/provider_continuation.py:1-46` | PARITY | CODE / NEW |
| Filesystem checkpoints | shadow-git store, snapshot per turn before mutating tools, `list/diff/session_diff/restore` + `safe_restore_plan` — `tools/checkpoint_manager.py:755-1089` | `ChangeTurnTracker` B/E shadow-repo snapshots per turn w/ concurrent-subagent + post-turn window kinds; `preflight_revert`/`revert_paths` per file — `Workspaces/change_turn_tracker.py:172,315,396`; `Workspaces/change_revert.py:94,154` | PARITY | CODE / NEW |
| Conversation rewind | `/undo N` durable truncate preserving the compaction handoff — `cli.py:10558-10740` | regenerate-with-snapshot only; no multi-turn undo — `Chat/console_chat_store.py:448,9768,14630-14642` | HERMES AHEAD | CODE / NEW |
| Alternate loop shapes | MoA loop gathers reference-model context per iteration — `agent/moa_loop.py:1-14` (2459 ln) | none in the runtime | HERMES AHEAD | CODE · task-18931 |
| Out-of-loop forks | `background_review`, `curator`, `/btw` side-question — cache-parity forks that never touch main history — `agent/background_review.py:1-16`; `agent/curator.py:1-18`; `agent/side_question.py:1-20` | `agent_lesson_promotion` runs *inside* the run as an approval-gated tool — `Agents/agent_lesson_promotion.py:1` | HERMES AHEAD | CODE / NEW |

### 2.2 Tools & skills

| capability | hermes | chatbook (file:line) | verdict | source |
|---|---|---|---|---|
| Tool registration model | decorator/registry singleton + AST-scan auto-discovery w/ on-disk cache — `tools/registry.py:74,87,111,165,452` | provider protocol (`list_catalog`/`load_schema`/`invoke`) + `ToolCatalogRegistry.register_provider` — `Agents/tool_catalog.py:593,1273,1322` | PARITY | CODE / NEW |
| Named toolsets / bundles of tools | 100+ named toolsets w/ `includes` recursion, aliases, plugin-contributed names — `toolsets.py:103,707,748,879` | no grouping; per-tool gating via `_GATEABLE_BUILTINS` + permission store — `Agents/tool_catalog.py:656,714,723` | HERMES AHEAD | CODE / NEW |
| Progressive tool disclosure | `tool_search`/`tool_describe`/`tool_call`; `tool_call` invokes without loading the schema — `tools/tool_search.py:64-68,755` | `find_tools`/`load_tools`; `load_tools` replaces the live schema set, then a normal call — `Agents/tool_catalog.py:238,249`; `agent_runtime.py:1910` | HERMES AHEAD (marginal) | CODE / NEW |
| Catalog ranking | BM25 + Snowball stemming + per-source corpus stats, up to 10 queries/call — `tools/tool_search.py:79,405,506,551` | deterministic 4-tier substring rank, single query, limit 8 — `Agents/tool_catalog.py:1408-1448`; `agent_models.py:177` | HERMES AHEAD | CODE + RELEASE-NOTE v0.20.6 / NEW |
| Disclosure activation threshold | `listing_token_budget` = min(max_tokens, pct of ctx); always-defer for MCP/plugin — `tools/tool_search.py:331,361` | `DIRECT_DISCLOSURE_CONTEXT_FRACTION = 0.10` of model context — `Agents/agent_models.py:174` | PARITY | CODE / NEW |
| Deferred-tool listing kept visible | name/group listing embedded **inside** the `tool_search` description — `tools/tool_search.py:639,672,755,786` | no listing; deferred tools reachable only by guessing a `find_tools` query — `Agents/tool_catalog.py:239-247` | HERMES AHEAD | CODE / NEW |
| Skill format | `SKILL.md` + YAML frontmatter (own dialect) — `skills/media/songsee/SKILL.md:1-13` | `SKILL.md` + Claude-Agent-Skills-spec frontmatter w/ per-field length caps — `Skills_Interop/local_skills_service.py:41,77-103,505-580` | PARITY | CODE / NEW |
| Bundled skill catalog | 261 `SKILL.md` in `skills/` (13 categories) + 137 in `optional-skills/` (22 categories) | ships zero: `find . -name SKILL.md` = 7, all `Tests/fixtures/` + `Docs/Examples/` | HERMES AHEAD | CODE / NEW |
| Skill discovery scopes | home `~/.hermes/skills`, hub taps, plugin-namespaced — `tools/skills_tool.py:986,1001,1057` | user store, project `.SKILLS`/`.skills` walked upward, server backend — `Skills_Interop/project_skills_discovery.py:17,102,144`; `skills_scope_service.py:24` | PARITY | CODE / NEW |
| Skill security gate | regex static scan + trust tiers; community w/ any finding blocked — `tools/skills_guard.py:1-20` | sha256 directory snapshot + trust store; install lands **pending user review**, inert until approved — `Skills_Interop/skill_trust_scanner.py:34`; `Agents/tool_catalog.py:282-300` | PARITY (different axis) | CODE / NEW |
| Remote skill install | hub sources, GitHub, taps, `HubLockFile` provenance, quarantine, audit log, index cache — `tools/skills_hub.py:5-13` | `install_skill` from a GitHub URL or `.zip`, SSRF-hardened; no registry/tap/index — `Skills_Interop/skill_remote_fetch.py:1-8` | HERMES AHEAD | CODE / NEW |
| Skill script execution | via prerequisites/commands + the general `terminal` tool — `tools/skills_tool.py:376-418` | dedicated `run_skill_script`: setrlimit trampoline, `start_new_session`, bounded reader threads, process-group SIGKILL, 600 s ceiling — `Agents/tool_catalog.py:377`; `Skills_Interop/skill_script_runner.py:1-31` | CHATBOOK AHEAD | CODE / NEW |
| Skill usage telemetry / mutation ledger | `.usage.json` counters + `.curator_ledger.jsonl` w/ content-addressed before/after blobs — `tools/skill_usage.py:1-11`; `tools/skill_ledger.py:1-11` | none (named grep `usage\|ledger\|invocation_count` over `Skills_Interop/local_skills_service.py` → 0) | HERMES AHEAD | CODE / NEW |
| Skill authoring lint | soft-convention linter on top of hard validation — `tools/skill_linter.py:1-11` | hard validation only (name regex, field caps) — `Skills_Interop/local_skills_service.py:91,93-103` | HERMES AHEAD | CODE / NEW |
| Skills as scheduled automations | "Blueprint" = a skill whose frontmatter carries `metadata.hermes.blueprint.schedule` — `tools/blueprints.py:3-12` | none (named grep `blueprint\|schedule` over `Skills_Interop/` → 0) | HERMES AHEAD | CODE / NEW |
| Skill bundles behind one slash command | YAML bundles + `hermes bundles list/show/create/delete/reload` — `hermes_cli/bundles.py:1-13` | none | HERMES AHEAD | CODE / NEW |
| Plugin/extension package format | two: in-repo `plugin.yaml` (`plugins/image_gen/fal/plugin.yaml:1-8`) and portable **Agent Plugins v1** w/ JSON schema, `extensions` map, `skills/` discovery, `mcpServers` — `hermes_cli/agent_plugins.py:21,34,143,192,456` | none; extension = an MCP server or a skill — `Agents/mcp_tool_provider.py:178` | HERMES AHEAD | CODE / NEW |
| Shell tool | `terminal` + `process` in the core toolset, dangerous-pattern detection, per-session approval, persistent allowlist — `toolsets.py:38`; `tools/approval.py:2-8` | `shell_exec`, ask-only + explicitly armed, scrubbed env, byte/timeout caps, delete-on-close spool — `Agents/raw_shell_tool_provider.py:31,36-40`; `Tools/raw_cli_executor.py:144,236,479` | PARITY | CODE · task-18926 (Done) |
| Shell failure hints | output-pattern tier → one actionable recovery hint on non-zero exit — `tools/terminal_hints.py:1-24` | none (named grep `hint\|suggest\|recovery` over `Tools/raw_cli_executor.py` → 0); raw stderr passthrough | HERMES AHEAD | CODE / NEW |
| Filesystem tool set | `read_file/write_file/patch/search_files` — `toolsets.py:44` | `fs_list/read/write/edit/patch/glob/grep` + `git_status/diff/log/blame/branches` — `Agents/local_tool_provider.py:143-149,2328-2453` | CHATBOOK AHEAD (git tools) | CODE / NEW |
| Edit self-recovery | 9-strategy fuzzy chain + `is_already_applied` + escape-drift guards + re-indent + `find_closest_lines` — `tools/fuzzy_match.py:74,126,263,333,394,1091` | exact `str.count`/`str.replace` only; error is `old_string not found in X` with no near-match — `Tools/local_tool_impls.py:786,839-847` | HERMES AHEAD | CODE · task-18927 (open) |
| Patch self-recovery | patch routes through the same fuzzy chain — `tools/patch_parser.py` + `tools/fuzzy_match.py:126` | strict unified diff; bare `patch_context_mismatch` — `Tools/patch_tool_impls.py:211,223-227,407` | HERMES AHEAD | CODE · task-18927 (open) |
| Concurrent-edit / stale-write guard | `FileStateRegistry` read stamps, global last-writer, `check_stale` before write — `tools/file_state.py:1-30` | per-key write lock + expected-target assertion, but no cross-agent staleness check (named grep `stale\|read_before_write\|concurrent` over `Tools/local_tool_impls.py` → 0) — `Tools/local_tool_impls.py:494,680,725` | HERMES AHEAD | CODE / NEW |
| Tool-output spill to disk | 3 tiers: per-tool pre-truncate → `maybe_persist_tool_result` writes full text to a re-readable path (sandbox-translated) → 200 K per-turn aggregate sweep — `tools/tool_result_storage.py:1-42,64-65` | hard truncate: `_fit_result` cuts at 32 KiB + `… [truncated]`, tail unrecoverable (named grep `spill\|spillover` over `tldw_chatbook/**/*.py` → 2 unrelated hits) — `Agents/local_tool_provider.py:158,320-324` | HERMES AHEAD | CODE · task-18927 (open) |
| Per-turn aggregate output budget | `enforce_turn_budget`, 200 K chars — `tools/tool_result_storage.py:38-42` | none (named grep `turn_budget\|aggregate.*budget\|MAX_TURN` over `Agents/*.py` → 0) | HERMES AHEAD | CODE / NEW |
| Configurable output limits | `tool_output.{max_bytes,max_lines,max_line_length}` in config — `tools/tool_output_limits.py:20-27` | hardcoded module constants — `Tools/local_tool_impls.py:73-83`; `Agents/local_tool_provider.py:158` | HERMES AHEAD (minor) | CODE / NEW |
| Code execution | `execute_code` — Programmatic Tool Calling: model writes Python that RPCs back into the tool registry over UDS/file-RPC; only stdout enters context — `tools/code_execution_tool.py:1-30`; `toolsets.py:80` | none (named grep `execute_code\|code_execution\|jupyter\|sandbox_exec\|docker` over `Tools/*.py` + `Agents/*.py` → 0) | HERMES AHEAD | CODE / NEW |
| Pluggable execution environments | 10 backends: local, docker, ssh, modal, managed_modal, daytona, singularity, vercel_sandbox, file_sync — `tools/environments/` | one-shot subprocess worker with a pinned root — isolation, not a remote backend — `Tools/workspace_tool_worker.py:1,29`; `Tools/workspace_root_pin.py` | HERMES AHEAD | CODE / NEW |
| Browser tools | 13 browser tools in core + camofox + CDP + extension router + supervisor — `toolsets.py:52-58`; `tools/browser_tool.py` | none (named grep `browser_navigate\|playwright\|selenium\|computer_use` over `Tools/*.py` + `Agents/*.py` → 1 comment). Web surface is `web_fetch`/`web_search`/`web_crawl`/`web_deep_search` — `Agents/local_tool_provider.py:2471,2506,2538,3272` | HERMES AHEAD | CODE / NEW |
| Computer use | `computer_use` tool, macOS, cua-driver gated — `toolsets.py:88`; `tools/computer_use/` | none | N/A BY DESIGN (TUI has no GUI-driver surface) | CODE / NEW |
| Image / video generation as **tools** | `image_generate` in core + `video_generate`; 7 pluggable backends — `toolsets.py:46`; `tools/image_generation_tool.py` | exists as **UI screens + config only**; named grep `image_generate\|image_gen\|video_gen` over `Agents/*.py` + `Tools/*.py` → 0 | HERMES AHEAD | CODE / NEW |
| Subagent-as-tool | `delegate_task` + per-child git-worktree isolation — `toolsets.py:80`; `tools/subagent_worktree.py:1-12` | `spawn_subagent` w/ per-run named-agent roster as an enum; `wait_agents`/`check_agents`/`send_to_agent` — `Agents/tool_catalog.py:89,109,154,180,198` | PARITY on delegation · HERMES AHEAD on isolation | CODE / NEW |
| Per-tool call caps | not found (`grep -rn "max_calls_per_turn\|call_cap"` over `toolsets.py`, `model_tools.py`, `tools/registry.py` → 0) | `RunToolPolicy` at the single `invoke_by_name` choke point, keyed `(run_id, name)`, narrowing-only — `Agents/run_tool_policy.py:1-16,27,39` | CHATBOOK AHEAD | CODE / NEW |
| Tool-arg coercion for sloppy models | `coerce_tool_args` — JSON-string → native, recursive, schema-aware — `model_tools.py:825,845,911,953` | `json.loads(call.arguments)` straight through, no repair layer — `Agents/agent_runtime.py:1149` | HERMES AHEAD | CODE / NEW |
| MCP-sourced tools | OAuth manager, schema cache, stdio watchdog — `tools/mcp_tool.py`; `tools/mcp_oauth_manager.py` | `MCPToolProvider` w/ per-tool permission states, pending-call flow, session-approval short-circuit — `Agents/mcp_tool_provider.py:110,153,170,592` | PARITY | CODE / NEW |
| Library / RAG as tools | n/a (no local media library) | `LocalLibraryToolService` + bounded `search_library_rag` w/ capped excerpts and scrubbed provenance + 14 `watchlists_*` tools — `Agents/library_tool_provider.py:1-13`; `library_rag_tool_provider.py:1-15`; `local_tool_provider.py:2589-3092` | CHATBOOK AHEAD | CODE / NEW |
| Toolset probability distributions | toolset→selection % for batch datagen runs — `toolset_distributions.py:28-60` | none | N/A BY DESIGN (chatbook is not a trajectory generator) | CODE / NEW |

### 2.3 Permissions & guardrails

| capability | hermes | chatbook (file:line) | verdict | source |
|---|---|---|---|---|
| Approval modes | `manual`/`smart`/`off`, profile-scoped — `hermes_cli/approval_mode.py:15`; `tools/approval.py:3450` | no global mode enum; posture is `global_default` ∈ ask/allow/deny per profile — `MCP/permission_store.py:79,386,394` | HERMES AHEAD | CODE / NEW |
| Yolo / bypass-all | `--yolo` process-scoped + `/yolo` session-scoped — `tools/approval.py:2977,3029` | none (named grep `yolo\|YOLO` over `tldw_chatbook/` → 0) | N/A BY DESIGN (no session escape hatch by stance) | CODE / NEW |
| Kill switch | not found (`grep -rn "kill_switch"` over hermes root → 0) | global stop consulted before any stamp or session grant — `MCP/permission_store.py:363,373`; `Agents/builtin_tool_gate.py:340` | CHATBOOK AHEAD | CODE / NEW |
| Decision grain | per dangerous-**pattern-key** + per-command-text glob in `command_allowlist` — `tools/approval.py:3145,3179` | per tool `(server_key, tool_name)`, per server default, per global default, chained across named profiles — `MCP/permission_store.py:790,190` | HERMES AHEAD on arg/path grain · CHATBOOK AHEAD on tool/server grain | CODE / NEW |
| Per-arg / per-path allow rules | command-text globs match args and paths (`git push --force *`) — `tools/approval.py:3145` | none: keys are `(server_key, tool_name)` only — `MCP/permission_store.py:472,489` | HERMES AHEAD | CODE / NEW |
| Risk tagging | descriptions on ~330 regex patterns + hardline floor — `tools/approval.py:945,551` | declarative `risk_tags` per tool + high-risk floor — `Tools/tool_executor.py:43`; `Tools/file_operation_tools.py:270,393,611`; `MCP/permission_store.py:80,97` | PARITY (different axis) | CODE / NEW |
| High-risk floor on inherited allow | not found (`grep -rn "risk" tools/approval.py` → prose only) | inherited `allow` downgraded to `ask` when tags ∩ `{mutates,process}` — `MCP/permission_store.py:912-918,995-1000` | CHATBOOK AHEAD | CODE / NEW |
| Rug-pull guard (definition changed) | not found (`grep -rn "definition_hash\|schema_hash\|tool_hash"` → only DB hashing) | explicit `allow` reverts to `ask` on `definition_hash` mismatch — `MCP/permission_store.py:604,884-892` | CHATBOOK AHEAD | CODE / NEW |
| Unconditional hardline deny | `rm -rf /`, mkfs, dd-to-device, fork bomb, shutdown — fires **before** yolo/off; sudo-stdin guard — `tools/approval.py:551,735,4751` | none (named grep `rm -rf\|hardline\|mkfs\|fork bomb` over `tldw_chatbook/` → 0 guard hits) | HERMES AHEAD | CODE / NEW |
| User-defined deny globs | `approvals.deny` fnmatch globs over de-obfuscated variants, before yolo/off — `tools/approval.py:794,826` | per-tool `deny` state only — `Agents/builtin_tool_gate.py:355`; `Agents/persona_policy.py:131` | HERMES AHEAD | CODE / NEW |
| Shell command risk detection | ~330 compiled patterns + de-obfuscation variants + quote-aware tokenizer — `tools/approval.py:2412` | **none**: validates caller/shell/size/timeout/cwd only — `Tools/raw_cli_executor.py:144`; gate is state-only — `Agents/raw_shell_tool_provider.py:291` | HERMES AHEAD | CODE / NEW |
| Shell-free structured CLI | not found (`tools/terminal_tool.py` executes shell strings) | 10 allowlisted read-only commands, argv via argparse, never parsed by a host shell — `Tools/virtual_cli_impls.py:20`; `Agents/virtual_cli_provider.py:225` | CHATBOOK AHEAD | CODE / NEW |
| LLM guardian ("smart approvals") | aux-LLM APPROVE/DENY/ESCALATE w/ injection-hardened prompt + operator policy — `tools/approval.py:3625,3640` | none (named grep `guardian\|smart approval\|auto.approve` over `tldw_chatbook/` → 2 prose hits) | HERMES AHEAD | CODE + RELEASE-NOTE (default since ~v0.14) / NEW |
| Consecutive-denial breaker | `approvals.denial_breaker_threshold` (default 3) appends a hard-stop to model-facing deny text — `tools/approval.py:2753,2785` | absent | HERMES AHEAD | CODE · task-18929 |
| Deny-with-reason to the model | `/deny <reason>` → `Reason given by the user: "…"` — `tools/approval.py:3958-3968` | fixed copy only — `Agents/mcp_tool_provider.py:87`; `builtin_tool_gate.py:359` | HERMES AHEAD | CODE · task-18920 |
| Denial semantics | "Do NOT retry, do NOT rephrase… Silence is not consent." — `tools/approval.py:4028-4034` | distinct unresolved/timeout/deny/kill-switch copy, "do not retry" on timeout — `Agents/mcp_tool_provider.py:78-95` | PARITY on provenance · HERMES AHEAD on anti-rephrase | CODE / NEW |
| Always-allow suggestions from history | `hermes approvals suggest` mines the session DB, never auto-applies — `hermes_cli/approvals_suggest.py:1-33` | tool-level allow recommendations from the execution log, applied via the control plane — `MCP/permission_prompt_reducer.py:146`; `MCP/unified_control_plane_service.py:2677`; `UI/Screens/chat_screen.py:263` | PARITY at tool grain · HERMES AHEAD at command grain | CODE · task-18928 (narrower than filed) |
| Dry-run "what would happen" | `hermes approvals test` replays real evaluators, exit 0/2/3 — `hermes_cli/approvals_test.py:1-31` | not found (`grep -rn "dry.run\|would_approve\|approval.*preview"` over `Agents/`, `MCP/` → 0) | HERMES AHEAD | CODE / NEW |
| Non-interactive fail-closed policy | per-context `cron_mode`/`single_query_mode`/`unattended_mode`, each default deny — `hermes_cli/config_defaults.py:2542-2544` | approval timeout → `TIMEOUT_REFUSAL` fail-closed; one context only — `Agents/mcp_tool_provider.py:93`; `MCP/unified_control_plane_service.py:3006` | HERMES AHEAD | CODE / NEW |
| Out-of-band approval transports | Discord/Telegram/Slack buttons + pluggable transports w/ request-digest binding — `tools/approval.py:3898-3960`; `hermes_cli/approval_transport.py:1-46`; `acp_adapter/permissions.py:40` | TUI approval card only — `Widgets/Chat_Widgets/chat_approval_card.py:49` | HERMES AHEAD | CODE / NEW |
| OS / container sandboxing | 7 backends; docker hardened `--cap-drop ALL`, no-new-privileges, `--pids-limit`, `--cpus`/`--memory`, `--user`, `--network=none` **verified post-start**; isolated backends skip approval — `tools/environments/docker.py:375,957-977,1457-1463`; `tools/approval.py:4075` | **none** (named grep `docker\|container\|podman` over `Tools/`, `Agents/`, `Coding/` → 0 execution hits). Isolation is process + path: fd-pinned root in a one-shot worker — `Tools/workspace_root_pin.py:99`; `Tools/workspace_tool_worker.py:53` | HERMES AHEAD | CODE / NEW |
| OS MAC sandboxes (seatbelt/bwrap/landlock) | not found (grep over hermes root → 0) | not found (same grep over `tldw_chatbook/` → 0) | PARITY | CODE / NEW |
| Path jail | `HERMES_WRITE_SAFE_ROOT` multi-root write jail + cross-profile classifiers — `agent/file_safety.py:93,498,616` | `allowed_file_roots(write=…)` w/ rw/ro bindings re-validated per call against symlink/mount swap; TOCTOU-safe `pin_workspace_root`; traversal + hidden-file checks — `Tools/workspace_file_roots.py:369`; `Tools/workspace_root_pin.py:99`; `Utils/path_validation.py:56` | CHATBOOK AHEAD | CODE / NEW |
| Credential-path file guard | hard-deny set + prefixes; `~/.ssh/config` approval-gated not denied — `agent/file_safety.py:28,74,111,200` | hard deny (no approval path) for `~/.ssh`,`~/.aws`,`~/.gnupg`,`~/.docker`,`~/.kube`,`~/.config/gh`, keyrings + name rules; also translated into git `:(exclude)` pathspecs — `Utils/sensitive_paths.py:218,241,716,895` | PARITY | CODE / NEW |
| Secret redaction | value-shape battery: key prefixes, env assigns, YAML/JSON/dotted config, JWT, private keys, DB conn-strings, auth headers — `agent/redact.py:80-409`, applied to approval payloads `tools/approval.py:3925` | key-name + CLI-arg + URL-query redaction only; `MCP/redaction.py:64` documents its own bypass — `MCP/redaction.py:1-114`; `Widgets/Chat_Widgets/chat_approval_card.py:44` | HERMES AHEAD | CODE / NEW |
| Env scrubbing before exec | tiered strip list + suffix rules + plugin keys + skill-declared passthrough allowlist — `tools/environments/local.py:415,505,511`; `tools/env_passthrough.py:1` | copy-only allowlist of 24 usability vars into an empty env — `Tools/raw_cli_executor.py:60,236` | PARITY (chatbook stricter, hermes richer) | CODE / NEW |
| Pre/post-tool hooks | plugin `pre_tool_call` (block/escalate/allow) + `post_tool_call` observer + approval-specific hooks + file-based event hooks — `model_tools.py:1425-1435,1184`; `tools/approval.py:108,4166`; `gateway/hooks.py:1` | `review_tool_calls(calls, run_id)` pre-dispatch batch hook + `before_tool_dispatch`; **no post-tool hook, no user-authored hooks** — `Agents/agent_runtime.py:1510,1610`. (`Tools/file_operation_hooks.py` is dead — pinned by `Tests/Tools/test_system_a_is_retired.py:73,80`) | HERMES AHEAD | CODE / NEW |
| Audit trail of decisions | no ledger — `hermes_cli/approvals_suggest.py:1-8` states this explicitly; mining reconstructs from the session DB | bounded metadata-only JSONL w/ `decision` ∈ allowed/approved/denied/downgraded — `MCP/execution_log.py:1,27`; `MCP/unified_control_plane_service.py:3054,2979,3235`. **Scope: MCP tools only** — raw-shell and local-builtin decisions are not recorded (only `Agents/` importer is `mcp_tool_provider.py:57`, a constant) | CHATBOOK AHEAD (scoped) | CODE / NEW |
| MCP server-spawn guard | blocks shell-interpreter egress/persistence shapes + hardcoded IOCs at save and spawn — `hermes_cli/mcp_security.py:1-25` | not found (`grep -rn "authorized_keys\|IOC\|suspicious"` over `MCP/` → 0) | HERMES AHEAD | CODE / NEW |
| SSRF / egress guard | SSRF-safe httpx clients, metadata hosts blocked — `tools/url_safety.py:1-27` | per-hop redirect re-validation + proxy-bypass guard — `Tools/web_tool_impls.py:68-118,747,880,949`; app-wide policy `Utils/egress.py:1-11` | PARITY | CODE / NEW |
| Content-level pre-exec scanner | Tirith subprocess scanner (homograph URLs, pipe-to-interpreter, terminal injection) — `tools/tirith_security.py:1-21`; wired at `tools/approval.py:4838` | not found (same-purpose grep over `Agents/`, `Tools/` → 0) | HERMES AHEAD | CODE / NEW |
| Cgroup / orphan reaping | `ExecStopPost` per-PID SIGKILL over the unit cgroup — `gateway/cgroup_cleanup.py:1,54` | process-tree cleanup only — `Tools/raw_cli_executor.py:826` | HERMES AHEAD | CODE / NEW |
| Per-run approval stamps | per-call approval, not run-scoped | stamps keyed by `run_id`, so one run's approval can't permit a sibling child's call — `Agents/builtin_tool_gate.py:177,195,357` | CHATBOOK AHEAD | CODE / NEW |

### 2.4 Context & memory

| capability | hermes | chatbook (file:line) | verdict | source |
|---|---|---|---|---|
| Automatic threshold compaction | ratio trigger 0.50 + <512 K-window floor at 0.75, absolute cap, target-ratio tail — `hermes_cli/config_defaults.py:807-830` | `decide_compaction()` tri-state w/ enforced hysteresis, wired into send preflight — `Chat/console_context_compaction.py:770`; `console_context_policy.py:398`; `console_chat_controller.py:18054` | PARITY | CODE · **refutes** the 2026-07 audit's "no compaction anywhere" |
| Compaction mode (off/ask/auto) | auto only; manual via `/compress` — `hermes_cli/commands.py:179` | tri-state `ASK/AUTOMATIC/OFF`; ASK blocks the send with an approval prompt — `Chat/console_context_policy.py:29`; `console_chat_controller.py:18072` | CHATBOOK AHEAD | CODE / NEW |
| Manual compaction | `/compress` w/ `here [N] \| focus topic \| --preview` — `hermes_cli/commands.py:179` | `plan_manual_prefix()` / `plan_manual_range()` — arbitrary message-range, not just prefix — `Chat/console_context_compaction.py:1016,1047` | PARITY | CODE / NEW |
| Manual preview / dry-run | `--preview\|--dry-run` — `hermes_cli/commands.py:179` | plan object exists but no surfaced preview flag (grep `--preview\|dry_run` over `Chat/` → 0) | HERMES AHEAD | CODE / NEW |
| Focus-directed compaction | `/compress focus <topic>` — `hermes_cli/commands.py:179` | no topic arg (grep `focus` in `console_context_compaction.py` → 0) | HERMES AHEAD | CODE / NEW |
| Micro-compaction (per-turn amortized) | folds the oldest un-absorbed exchange into a rolling summary each turn; cadence + defrag tunable — `agent/context_compressor.py:253,2276`; `agent/turn_finalizer.py:416-440`; `hermes_cli/config_defaults.py:879-899` | none (named grep `micro_compact\|micro-compact` over `tldw_chatbook/` → 0) | HERMES AHEAD | CODE + RELEASE-NOTE (#75345) / NEW |
| Proactive tool-result pruning (no LLM) | `prune_tool_results_only()` w/ min-reclaim cache-break gate — `agent/context_compressor.py:4232`; `config_defaults.py:857-878` | none (named grep `tool_result.*prune\|strip.*tool.*output` over `Chat/` + `Agents/` → 0); only whole-turn-group drops — `Chat/console_history_budget.py:266` | HERMES AHEAD | CODE / NEW |
| Stale-image retirement | `_retire_stale_tool_result_images()` replaces old image payloads with placeholders — `agent/context_compressor.py:1651,1680` | none (named grep `retire.*image` over `Chat/` → 0); images charged forever — `Chat/console_history_budget.py:184` | HERMES AHEAD | CODE / NEW |
| Non-LLM window bounding | tail retention lean/legacy, `protect_last_n`/`protect_first_n` — `hermes_cli/config_defaults.py:833-856` | `bound_messages_to_window()` turn-group-aware drop loop, reused for agent run-logs — `Chat/console_history_budget.py:266`; `Agents/run_log_eviction.py:58` | PARITY | CODE + RELEASE-NOTE v0.20.6 ("lean-tail default") |
| Per-image token budgeting | flat `_IMAGE_TOKEN_ESTIMATE = 1600` — `agent/context_compressor.py:1249` | `per_image_tokens` threaded through budget + prepared request — `Chat/console_history_budget.py:119,184`; `console_prepared_request.py:1321` | PARITY | CODE / NEW |
| Compaction failure behavior | `abort_on_summary_failure` (freeze vs placeholder-drop) — `hermes_cli/config_defaults.py:944-953` | user-selectable `STOP_AND_ASK` / `OMIT_OLDER_CONTEXT` in settings — `Chat/console_context_policy.py:42`; `UI/Screens/settings_screen.py:14145` | PARITY | CODE / NEW |
| Visual-transcript compaction | none (named grep `visual transcript\|render.*transcript.*image` over `agent/` → 0) | renders transcript pages as images for a vision model; `TEXT_SUMMARY`/`VISUAL_TRANSCRIPT`/`HYBRID` — `Chat/console_visual_transcript.py:346,463`; `console_context_policy.py:36` | CHATBOOK AHEAD | CODE / NEW |
| Branch/fork-aware compaction memory | none (named grep `branch_memory\|fork.*compaction` over `agent/` → 0) | `BranchMemoryCommit`, `MemorySelectionFence`, lineage fences — `Chat/console_context_repository.py:285,294,324,371` | CHATBOOK AHEAD | CODE / NEW |
| Compaction provenance | summary carrier + display projection — `agent/compaction_display.py:24` | `CompactionAdmission`, `prefix_digest`, `provenance_payload`, per-attempt pricing provenance — `Chat/console_context_compaction.py:258,304,375,2115`; `console_context_repository.py:81,108` | CHATBOOK AHEAD | CODE / NEW |
| Anthropic `cache_control` | 4 breakpoints, TTL 5 m/1 h w/ per-route measured allow-list, stable-prefix boundary registry — `agent/prompt_caching.py:157,190-230,343` | 3 breakpoints (system, last tool, opt-in per-turn last block) + capability gate + `[caching].anthropic_enabled` kill-switch + one-shot degrade-retry — `LLM_Calls/LLM_API_Calls.py:1466,1511,1544,1066,1597-1615` | PARITY | CODE · **refutes** the 2026-07 audit's "cache_control never enabled" |
| Cache TTL tiers (1 h) | `_build_marker(ttl)` emits `{"ttl":"1h"}` per route — `agent/prompt_caching.py:157,56` | ephemeral 5 m only — all three sites are bare `{"type":"ephemeral"}` (grep `"ttl"\|1h` in `LLM_API_Calls.py` → 0) — `LLM_Calls/LLM_API_Calls.py:1466,1511,1544` | HERMES AHEAD | CODE / NEW |
| OpenAI/Codex `prompt_cache_key` | content-addressed key derivation + rotation-stable scope — `agent/auxiliary_client.py:1819-1834`; `agent/transports/codex.py:692` | none (named grep `prompt_cache_key\|cachedContent\|implicit_cach` over `tldw_chatbook/` → 0) | HERMES AHEAD | CODE / NEW |
| Cached-token accounting | Gemini `cachedContentTokenCount` — `agent/gemini_native_adapter.py:804` | `cache_read_input_tokens`/`cache_creation_input_tokens`/OpenAI `cached_tokens` normalized and priced — `Chat/provider_usage.py:235,285-286`; `LLM_Calls/pricing_catalog.py:233` | PARITY | CODE / NEW |
| Context breakdown by category | 8 categories + 100-cell glyph grid, `/context` and `/usage` — `agent/context_breakdown.py:89-170,244` | 2-row request/conversation split + overhead only — `Widgets/Console/console_context_controls.py:105,118` | HERMES AHEAD | CODE + RELEASE-NOTE (#55204) / NEW |
| Live token/budget display | usage-anchored `context_used`/`context_percent` from provider-exact last usage — `agent/context_breakdown.py:134-150` | `ConsoleSettingsContextEstimate` (used/limit + verified flag + source provenance) in rail and settings — `Chat/console_session_settings.py:317-326,1219`; `UI/Screens/chat_screen.py:5702` | PARITY | CODE · task-18923 |
| Model-window honesty | threshold clamped to model context length — `hermes_cli/config_defaults.py:824` | `model_window_verified`, `token_limit_source`, `UNKNOWN_WINDOW`, "estimated input; model unverified" label — `Widgets/Console/console_context_controls.py:110-116`; `Chat/console_context_compaction.py:781` | CHATBOOK AHEAD | CODE / NEW |
| Pluggable per-session context engine | `ContextEngine` ABC + `load_context_engine(name)` w/ session lifecycle hooks — `agent/context_engine.py:89`; `plugins/context_engine/__init__.py:79` | single hard-wired compaction service — `Chat/console_context_compaction.py:1627` | HERMES AHEAD | CODE / NEW |
| Context references (`@file`, `@diff`, `@folder`, `@git`) | `parse_context_references`/`preprocess_context_references`; builtin prefixes diff/staged/file/folder/git/url, line-range slicing, plugin-registered prefixes + autocomplete — `agent/context_references.py:24,148,212,368-471` | none (named grep `expand.*reference\|@folder\|@diff\|@staged` over `Chat/` + `Widgets/Console/` → 0). `$`-mentions exist for **skills only** — `Chat/console_command_suggestions.py:163` | HERMES AHEAD | CODE + RELEASE-NOTE / NEW |
| Persistent cross-session memory | `MEMORY.md` + `USER.md` store, nudge interval, memory toolset; `MemoryManager` prefetch/sync/session-boundary/pre-compress checkpoints — `agent/agent_init.py:1855-1870`; `agent/memory_manager.py:403,564,714,1083` | none (named grep `MEMORY\.md\|persistent memory\|memory_store\|user_profile` over `Agents/` + `Chat/` → 0). "Memory" = per-conversation compaction summaries — `Chat/console_context_repository.py:136` | HERMES AHEAD | CODE / NEW |
| External memory-provider plugins | 9 providers behind a `MemoryProvider` ABC — `plugins/memory/`; `agent/memory_provider.py:110` | none (named grep `mem0\|supermemory\|honcho\|MemoryProvider` over `tldw_chatbook/` → 0) | HERMES AHEAD | CODE + RELEASE-NOTE / NEW |
| Learning graph / journey | skill nodes + memory cards + edges + density stats, user-editable — `agent/learning_graph.py:254,171`; `agent/learning_mutations.py:1` | none (named grep `learning_graph\|journey\|skill_graph` over `Agents/` + `Chat/` → 0) | HERMES AHEAD | CODE / NEW |
| Curator (automatic memory/skill hygiene) | scheduled stale/archive/prune/consolidate + LLM review + run reports — `agent/curator.py:233,305,1518,2023` | manual-gated `agent_lesson_promotion`: evidence assessment → AGENTS.md / managed-skill proposal behind an approval gate — `Agents/agent_lesson_promotion.py:208,286,343,396` | HERMES AHEAD | CODE + RELEASE-NOTE / NEW |
| Project instruction files | SOUL.md / AGENTS.md / CLAUDE.md / .cursorrules / .hermes.md — `agent/prompt_builder.py:2149,2196` | `AGENTS.md` + `AGENTS.override.md` w/ nested-scope resolution chain and safety metadata — `Agents/project_instruction_resolver.py:163,192,298-316` | PARITY | CODE / NEW |
| Trajectory compression | standalone `TrajectoryCompressor` w/ metrics, for dataset generation — `trajectory_compressor.py:83,332` | `derive_trajectory()` builds/exports but no compression pass — `Chat/trajectory.py:440` | N/A BY DESIGN (hermes's is an offline RL-dataset tool, not a runtime feature) | CODE / NEW |
| RAG-into-context | memory-provider prefetch + `session_search` pointers — `agent/memory_manager.py:564` | `get_rag_context_for_chat()` w/ plain/full/hybrid pipelines + user-set max context length + persisted per-conversation scope — `Event_Handlers/Chat_Events/chat_rag_events.py:177,225,279,1883,2051`; `Chat/rag_scope.py:435` | CHATBOOK AHEAD | CODE / NEW |
| Compaction timeouts | progress-aware inactivity budgets + absolute ceilings — `hermes_cli/config_defaults.py:905-931` | none (grep `timeout` in `console_context_compaction.py` → 0) | HERMES AHEAD | CODE / NEW |
| Provider-native compaction | Codex `thread/compact/start`, OpenAI Responses server-side compaction opt-in — `agent/native_compaction.py:320`; `config_defaults.py:975-991` | none (named grep `native_compaction\|server.*compact` over `Chat/` → 0) | HERMES AHEAD | CODE / NEW |

### 2.5 Providers & auth

| capability | hermes | chatbook (file:line) | verdict | source |
|---|---|---|---|---|
| Provider registry size | **~48**: 38 `ProviderConfig` entries — `hermes_cli/auth.py:249-573` — auto-extended by 39 declarative `ProviderProfile` plugins — `plugins/model-providers/`; `providers/__init__.py:57-77`; user/pip plugins can override any | **23 distinct** from a 30-entry hardcoded dict: 12 hosted (openai, anthropic, cohere, groq, openrouter, deepseek, mistral, google, huggingface, moonshot, zai, qwencloud), 9 local, 2 custom — `Chat/Chat_Functions.py:120-152`. No plugin point — `Chat/provider_readiness.py:108` | HERMES AHEAD | CODE / NEW |
| Provider adapter shape | declarative `ProviderProfile` dataclass; transport reads it — `providers/base.py:1-60` | per-provider Python function + hand-maintained param map + parallel frozensets kept in sync by hand — `Chat/Chat_Functions.py:186-217`; `Chat/provider_readiness.py:50-107` | HERMES AHEAD | CODE / NEW |
| Claude Pro/Max subscription auth | full PKCE S256 against `claude.ai/oauth/authorize`; also reads Claude Code's `~/.claude/.credentials.json`; single-use refresh rotation w/ spent-token fingerprint — `agent/anthropic_credentials.py:88-93,108-133,920-1013` | none (grep `code_verifier\|code_challenge\|device_code\|PKCE\|claude.ai` over `tldw_chatbook/` → 0). Anthropic is API-key-only — `Chat/provider_readiness.py:52` | HERMES AHEAD | CODE / NEW |
| ChatGPT / Codex subscription auth | `oauth_external` provider, device-code persisted, talks to `chatgpt.com/backend-api/codex`, dedicated Responses adapter — `plugins/model-providers/openai-codex/__init__.py:12`; `agent/credential_pool.py:3201` | none. (`Auth_Account_Interop/auth_account_scope_service.py:720` `get_openai_oauth_status` is a proxy to tldw_server for **user-account** auth, not an LLM subscription) | HERMES AHEAD | CODE / NEW |
| GitHub Copilot auth | 736-line `hermes_cli/copilot_auth.py`, gh-CLI token source, plus a separate `copilot-acp` provider — `plugins/model-providers/copilot/__init__.py:77` | none (grep `copilot` over `tldw_chatbook/` → 0) | HERMES AHEAD | CODE / NEW |
| Other subscription OAuth (xAI, Qwen, MiniMax, Nous) | `xai-oauth`, `qwen-oauth`, `minimax-oauth`, `nous` device-code — `hermes_cli/auth.py:273-278`; `plugins/model-providers/{qwen-oauth,minimax,nous}/__init__.py:104,91,139` | none | HERMES AHEAD | CODE + RELEASE-NOTE v0.16.x (Grok Composer) / NEW |
| Enterprise identity (AWS/GCP/Azure) | Bedrock SigV4 — `agent/bedrock_adapter.py`; Vertex GCP SA token mint/refresh; Azure — `agent/azure_identity_adapter.py`; `hermes_cli/azure_detect.py` | none (grep `bedrock\|vertex_ai\|sigv4\|AzureOpenAI\|DefaultAzureCredential` over `tldw_chatbook/` → 0) | HERMES AHEAD | CODE + RELEASE-NOTE v0.13.x (Vertex) / NEW |
| Credential pooling (N accounts/provider) | 3566-line `CredentialPool` w/ per-entry status/priority/lease, seeded from env, `.env`, claude_code, PKCE store, qwen-cli, gh CLI, config, manual — `agent/credential_pool.py:722-2762`; `agent/credential_sources.py:5-16` | none — one key per provider (grep `credential_pool\|key_pool\|multi_account\|rotate_key` over `tldw_chatbook/` → 1 unrelated policy action) — `config.py:7423-7478` | HERMES AHEAD | CODE / NEW |
| Pool selection strategy | `fill_first` / `round_robin` / `least_used`, per provider — `agent/credential_pool.py:119-121,544-557` | n/a (single credential) | HERMES AHEAD | CODE + RELEASE-NOTE v0.5.0 / NEW |
| Rate-limit → rotate account | `mark_exhausted_and_rotate` w/ parsed `retry_delay`/reset-time TTL, `STATUS_DEAD` for terminal OAuth failure — `agent/credential_pool.py:401-462,2404` | retries the **same** key: `Retry-After` then exponential backoff, urllib3 `Retry(status_forcelist=[429,…])`, finally `ChatRateLimitError` — `LLM_Calls/hosted_chat.py:864-881`; `LLM_API_Calls.py:878-884`; `Chat/Chat_Functions.py:1180-1184` | HERMES AHEAD | CODE + RELEASE-NOTE v0.11.x / NEW |
| Cross-provider failover chain | ordered `fallback_providers`, main + auxiliary; 402/credit-exhaustion auto-retry down the chain — `agent/auxiliary_client.py:39-45,6063-6092` | none — the two `fallback_*` hits are config-key fallbacks, not error-driven switching — `Chat/console_session_settings.py:725-727` | HERMES AHEAD | CODE + RELEASE-NOTE v0.4.x / NEW |
| Rate-limit header telemetry | parses 12 `x-ratelimit-*` into RPM/RPH/TPM/TPH for `/usage` — `agent/rate_limit_tracker.py:1-48` | none (grep `x-ratelimit` over `tldw_chatbook/` → 0) | HERMES AHEAD | CODE / NEW |
| Credential storage | `~/.hermes/auth.json`, atomic `O_EXCL` + 0600, TOCTOU-closed; disk-boundary sanitization refuses to persist borrowed secrets; no OS keyring for LLM keys — `hermes_cli/auth.py:1441-1486,2837-2844`; `agent/credential_persistence.py:19-26` | env → TOML config file; **no keyring for chat providers** (grep `keyring` over `Chat/`, `LLM_Calls/`, `LLM_Provider_Catalog/` → 1 unrelated HMAC hit). Keyring **is** used for image-gen, video-gen, and server creds — `config.py:7423-7478`; `Image_Generation/config.py:223`; `runtime_policy/server_credentials.py:324` | PARITY | CODE / NEW |
| Credential lifecycle CLI | `hermes auth add/remove/list` w/ unified per-source `RemovalStep` so removal sticks — `agent/credential_sources.py:18-50`; `hermes_cli/auth_commands.py` (901 ln) | Settings-screen key entry + per-provider readiness verdict — `config.py:7460-7469`; `Chat/provider_readiness.py:351` | HERMES AHEAD | CODE / NEW |
| Model metadata catalog | models.dev: 4000+ models / 109+ providers, ETag conditional GET, disk cache, stale-while-revalidate, no-network-on-hot-path — `agent/models_dev.py:1-40` + 4012-line `agent/model_metadata.py` | regex pattern table for vision/context-window, config-overridable — `model_capabilities.py:27-45` | HERMES AHEAD | CODE / NEW |
| Pricing catalog | models.dev cost/M-token fields, community-maintained — `agent/models_dev.py:5-8` | hand-seeded $/M table w/ `_SEED_AS_OF` staleness stamp + `[pricing]` override — `LLM_Calls/pricing_catalog.py:1-24,41-48,532` | HERMES AHEAD | CODE / NEW |
| Live model discovery (`/v1/models`) | `ProviderProfile.fetch_models` w/ UA to dodge WAFs — `providers/base.py:24-36` | full subsystem: OpenAI-compatible probe + memory cache + disk cache + merge + persistence + auto-refresh w/ UI notification + Ollama `/api/tags` fallback — `LLM_Provider_Catalog/openai_compatible_model_discovery.py:1-35`; `model_auto_refresh.py:21-83`; `Chat/local_server_discovery.py:455-471` | PARITY | CODE / NEW |
| Per-turn / one-shot model override | `/model --once` | absent (grep `model_once\|--once\|once_model\|one_turn` over `tldw_chatbook/` → 0) | HERMES AHEAD | RELEASE-NOTE · task-18922 |
| Fast/cheap vs strong routing | `prefer_fast_model` per auxiliary task; 3-tier aux-model resolution + free-only OpenRouter cost guard — `agent/auxiliary_client.py:971,1025-1109` | auxiliary endpoints go through the **same** dispatch table at the same model; audited but not routed — `Chat/Chat_Functions.py:186-217` | HERMES AHEAD | CODE / NEW |
| Local inference runtimes | lmstudio, ollama + ollama-cloud as providers; llama.cpp/vllm/lmstudio in the error surface — `hermes_cli/auth.py:285-292,537`; `agent/error_surface.py:86-90` | 9 dispatchable: llama_cpp, koboldcpp, oobabooga, tabbyapi, vllm, local-llm, ollama, aphrodite, mlx_lm + keyless readiness — `Chat/Chat_Functions.py:130-152`; `Chat/provider_readiness.py:84-107` | CHATBOOK AHEAD | CODE / NEW |
| Local server process management | talks to local servers, does not launch or administer them — named grep `Popen\|subprocess.run\|create_subprocess` over `hermes_cli/`, `agent/`, `tools/` filtered to `llama\|ollama\|vllm\|lmstudio\|mlx` → 0; `api/pull\|/api/create\|/api/delete` → 0 | launches/stops MLX-LM via `Popen`; full Ollama model admin (list/pull/push/copy/delete/create/embeddings/ps); auto-discovery across ollama/llamacpp/vllm — `Local_Inference/mlx_lm_inference_local.py:22-177`; `Local_Inference/ollama_model_mgmt.py:261-386`; `Chat/local_server_discovery.py:201-232` | CHATBOOK AHEAD | CODE / NEW |
| Embedding providers | none (named grep `rerank\|/v1/embeddings\|embedding` over `agent/`, `hermes_cli/` → only models.dev noise filters + Bedrock exclusions) | HF local sentence-transformers + OpenAI-compatible w/ retry, plus Ollama embeddings — `Embeddings/Embeddings_Lib.py:236-286,513-575`; `Local_Inference/ollama_model_mgmt.py:374` | CHATBOOK AHEAD | CODE / NEW |
| TTS providers | registry + 11 built-in names + plugin registration — `agent/tts_registry.py:41-60` | 7 backends + audio.cpp adapter + adapter registry + profiles/migrations/playground — `TTS/backends/`; `TTS/adapter_registry.py` | PARITY | CODE / NEW |
| STT providers | 6 providers w/ per-provider language resolution, keys from the credential pool — `tools/transcription_tools.py:5-14,71-84,182-201` | sealed capability registry w/ declared-vs-observed validation, executor/process-tree layers, parakeet ONNX, whisper.cpp, faster-whisper/qwen2audio/canary/lightning-whisper-mlx — `STT/registry.py:1-66`; `Local_Ingestion/transcription_service.py:3347-3424` | CHATBOOK AHEAD | CODE / NEW |
| Image-gen providers | 7 plugins + registry w/ fallback + vision/image routing — `plugins/image_gen/`; `agent/image_gen_registry.py:38-178` | 10 adapters incl. self-hosted ComfyUI/SwarmUI/stable-diffusion.cpp + keyring secrets + 429 classification — `Image_Generation/adapters/`; `Image_Generation/config.py:223-229` | CHATBOOK AHEAD (slightly) | CODE / NEW |
| Video-gen providers | `agent/video_gen_provider.py` + registry + xAI video tools | 2 adapters (comfyui, minimax) + worker/store/templates/workflows + keyring secrets — `Video_Generation/adapters/`; `Video_Generation/config.py:158-168` | PARITY | CODE / NEW |
| Billing / credits / usage | 14 `x-nous-credits-*` headers as micros ints; Claude/Codex account usage windows w/ reset times; remote-spending screens w/ `Decimal` discipline; per-provider billing deep links — `agent/credits_tracker.py:1-30`; `agent/account_usage.py:26-30`; `agent/billing_view.py:1-15`; `agent/billing_links.py:55` | token usage dataclass → cost via `PricingCatalog`; composer next-send price estimate w/ honest "cost unavailable" fallback — `Chat/provider_usage.py:59-209`; `LLM_Calls/pricing_catalog.py:532`; `Widgets/Console/console_composer_bar.py:542,1917` | HERMES AHEAD | CODE / NEW |

### 2.6 Scheduling / cron / automations

Rows marked **PRIOR** were covered by TASK-18936; the rest is what a code-level pass adds. Corrections to that audit
are in §3.6.

| capability | hermes | chatbook (file:line) | verdict | source |
|---|---|---|---|---|
| Core loop / tick | 60 s file-locked tick, cross-process `flock`/`msvcrt`, EMFILE not swallowed — `cron/scheduler.py:7812-7880` | 30 s asyncio poll + wake-on-reload event, single in-process loop — `Scheduling/scheduler/loop.py:326`; `constants.py:15` | PARITY | CODE · PRIOR |
| Schedule grammar | `once`/`interval`/`cron` **plus natural language** ("every monday 9am", "weekdays at 9am", "in 30m") — `cron/jobs.py:941-1090` | ISO `one_time` / raw cron `recurring`; presets in the form (grep `parse_schedule\|natural` over `Scheduling` → 0) — `UI/Screens/scheduling/forms/reminder_form.py:492` | HERMES AHEAD | CODE / NEW |
| Per-job timezone | one configured tz for all jobs — `cron/jobs.py:1104-1119` | per-task IANA tz, required for recurring, applied at next-run compute — `Scheduling/models.py:104-116`; `db/scheduled_tasks_db.py:856-864` | CHATBOOK AHEAD | CODE / NEW |
| Missed-fire accounting | fast-forward past `grace = period/2` clamped 120 s–2 h, fire once, bump a counter — `cron/jobs.py:1177-1192,4004-4039` | `missed_at` + `missed_count` written on late dispatch, skipped occurrences counted-not-replayed, capped w/ a `-1` overflow sentinel — `db/scheduled_tasks_db.py:772-870,888` | PARITY | CODE · PRIOR gap 18937 → **shipped** |
| Missed-fire **cause** attribution | not present (grep `away\|stalled\|busy` over `cron/` → no classifier) | 3-way `away`/`busy`/`stalled` w/ evidence from `_running_since` + prior-tick dispatch span, emitted as `scheduler_dispatch_late{cause}` — `Scheduling/scheduler/loop.py:444-521` | CHATBOOK AHEAD | CODE / NEW |
| Missed-fire UI surfacing | `cron status` counter + per-job diagnostic file — `hermes_cli/cron.py:375` | `◇` glyph + filter + legend in the Queue tab; detail banner w/ exact skipped count — `UI/Screens/scheduling/schedules_workbench.py:439-444,1312`; `task_detail.py:696-726` | PARITY | CODE · PRIOR 18937 |
| Run now / manual fire | `trigger_job` re-arms `next_run_at=now`, un-pauses, carries a transient `manual_run_prompt` — `cron/jobs.py:2724-2762` | `run_reminder_now` shares the dispatch seam, dequeues first to prevent double-fire, works on disabled tasks — `Scheduling/scheduler/loop.py:557-608`; `queue.py:116`; `schedules_workbench.py:127,807` | PARITY | CODE · PRIOR gap 18938 → **shipped** |
| Run-now extra context | per-run `extra_prompt` injected once — `cron/jobs.py:2724` | none (grep `extra_prompt\|manual_run_prompt` over `Scheduling` → 0) | HERMES AHEAD (minor) | CODE / NEW |
| Retry semantics | resume/re-arm + persisted-error auto-recovery — `cron/jobs.py:1310,1211-1254` | Run-now relabels to "Run now (retry)" on `missed`/`timed_out`; no automatic retry — `UI/Screens/scheduling/task_detail.py:643-651` | PARITY | CODE · PRIOR 18938 |
| Execution timeout | many independent sub-timeouts (script/media/session/cleanup); **no single per-job wall clock** — `cron/scheduler.py:3973,4009,4044,5401` | per-task `timeout_seconds` overriding `[scheduling] handler_timeout_seconds`; cancel → distinct `timed_out` status; schedule still advances — `Scheduling/scheduler/loop.py:365-451,533-555`; `models.py:103`; `db/migrations/v2_to_v3.py:45` | CHATBOOK AHEAD | CODE · PRIOR gap 18939 → **shipped, verdict inverted** |
| Durable per-run execution ledger | `executions` table w/ owner pid + process-start-time, claimed/running/completed/failed/unknown, dead-owner reclaim, 1000-row prune; `hermes cron runs` — `cron/executions.py:46-72,141-241`; `hermes_cli/cron.py:259` | **none for reminders/briefings** — only `last_status`/`last_run_at` on the row — `db/scheduled_tasks_db.py:772-830`. **Watchlists do have one**: `local_watchlist_runs` — `DB/Subscriptions_DB.py:936-950`; orphan reconcile `Subscriptions/startup_reconcile.py:164,195` | HERMES AHEAD (reminders/briefings only) | CODE / NEW — **corrects 18936** |
| Server-side automation run history | n/a (gateway is the executor) | Automations tab pulls the server audit trail — `schedules_workbench.py:1043`; `services/server_client.py:315` | CHATBOOK AHEAD (server-scoped) | CODE / NEW |
| Failure incidents (dedupe + ack) | `cron_incidents` keyed `(job_id, normalized-error-sig)`, detected→alerted→closed, redaction, failure classification; `hermes cron incidents` — `cron/incidents.py:91-116,133-222,256` | none (`grep -rin "incident"` over `Scheduling`, `UI/Screens/scheduling` → 0). Failures = `last_status` + a Home count — `Home/dashboard_state.py:1030` | HERMES AHEAD | CODE + RELEASE-NOTE v0.20.6 (durable-incident acks) / NEW — **audit missed** |
| Per-job durable KV notepad | `cron_notepad` w/ 16 KB/key, 64 KB/job caps, rendered into the job prompt; `hermes cron notepad` — `cron/notepad.py:55-96,169` | none (`grep -rin "notepad"` over `Scheduling` → 0) | HERMES AHEAD | CODE + RELEASE-NOTE v0.20.5 ("cron jobs gaining persistent memory") / NEW — **audit missed** |
| Change-detection ("monitor") jobs | `monitor_script`/`monitor_url` hashed exact-bytes; unchanged ⇒ silent `no_change`, changed ⇒ unified diff injected — `cron/monitor.py:148-198` | shipped as watchlists: scheduled check handler w/ its own run rows, projected into the queue — `Scheduling/scheduler/handlers/watchlist_check_handler.py:56-102`; `services/watchlist_projection.py:60` | PARITY | CODE / NEW |
| Job chaining (`context_from`) | prior job's latest output injected into the next job's prompt — `cron/jobs.py:2186` | none (`grep -rin "context_from"` over `Scheduling` → 0) | HERMES AHEAD | CODE / NEW |
| Repeat / occurrence limit | `repeat: N` w/ auto-delete on exhaustion — `cron/jobs.py:777-806,2974-2990` | none — a recurring reminder runs forever | HERMES AHEAD (minor) | CODE / NEW |
| Pause with reason / terminal states | `pause_job(reason)`, `paused_at`/`paused_reason`, `effective_job_state`, `is_terminal_job` — `cron/jobs.py:2683-2723,618-642` | boolean `enabled` only; `TaskStatus.PAUSED` exists but only watchlists set it — `Scheduling/models.py:100`; `services/watchlist_projection.py:60` | HERMES AHEAD (minor) | CODE / NEW |
| Global emergency stop | `hermes pause` ESTOP sentinel checked every tick, in-flight untouched — `cron/scheduler.py:7888` | none (`grep -rin "estop\|emergency stop"` over `Scheduling` → 0) | HERMES AHEAD | CODE / NEW |
| Concurrency / overlap guard | in-flight registry, stale-claim sweep, forced release, fire-claims w/ heartbeat + machine id, parallel + sequential pools — `cron/scheduler.py:797,982-1181,1451` | serial inline `await` per tick; no overlap possible, no claim protocol — `Scheduling/scheduler/loop.py:342-363` | N/A BY DESIGN (single process; hermes needs claims because CLI+gateway+daemon race) | CODE / NEW |
| Scheduler liveness / heartbeat | `record_ticker_heartbeat`, `get_ticker_heartbeat_age`, last-error file, EMFILE backoff — `cron/jobs.py:1465-1608`; `cron/scheduler_provider.py:36-67` | none (`grep -rin "heartbeat"` over `Scheduling` → 0); startup logs unhandled types + `scheduler_tasks_unhandled` — `loop.py:349-362` | HERMES AHEAD | CODE / NEW |
| Preflight validation | provider key / delivery target / skills checked before firing, one-shot alert flags — `cron/scheduler.py:5127-5400`; `cron/jobs.py:2922-2937` | none for reminders (grep `preflight` over `Scheduling` → 0); handler-registry misconfig warned at startup — `loop.py:349` | HERMES AHEAD | CODE / NEW |
| Prompt-injection / exfil guards on job payloads | assembled-prompt scan + credential-exfil guard + auto-pause; gateway-restart cron rejected at create — `cron/scheduler.py:4869,4939,4985`; `cron/lifecycle_guard.py` | n/a — chatbook reminders carry title/body only, no prompt, no script, no shell — `Scheduling/models.py:88-95` | N/A BY DESIGN | CODE / NEW |
| Output storage + retention | every run's output to `cron/output/<job>/<ts>.md` w/ a retention cap — `cron/jobs.py:4189-4262` | none — reminders dispatch a notification — `handlers/reminder_handler.py:20-30` | HERMES AHEAD | CODE / NEW |
| Delivery routing / media | multi-target delivery, origin/local/platform/bot-chat tokens, mirror, MEDIA: attachments — `cron/scheduler.py:2884-3107,2938` | single channel: `NotificationDispatchService` — `handlers/reminder_handler.py:22-29` | N/A BY DESIGN (TUI has one notification surface) | CODE · PRIOR (listed, undecided) |
| Per-job model / provider / reasoning effort | `model`/`provider`/`base_url`/`reasoning_effort` per job w/ snapshot resolution — `cron/jobs.py:2016-2148,2186-2210` | server-owned definitions: `input.provider`/`input.model` rendered as the execution-target column, `auto` when unpinned — `schedules_workbench.py:83-110` | PARITY for server-scoped; no local equivalent (local tasks run no LLM) | CODE + RELEASE-NOTE v0.20.5 · PRIOR (folded into 18940) |
| `agent_task` execution | gateway executes LLM cron jobs — `cron/scheduler.py:5525,7151` | server executes; client dispatches `run_automation_definition_now`, results arrive as notifications, local loop refuses server-scoped rows — `services/server_client.py:356`; `schedules_workbench.py:1103-1165`; `queue.py:77-86`; `loop.py:583-591` | PARITY (control plane + dispatch shipped; execution is server-side) | CODE · PRIOR 18940 → **client half shipped** |
| Blueprints / templates | 16 typed-slot blueprints; one schema → form + slash command + deeplink + `create_job` kwargs — `cron/blueprint_catalog.py:120,578,602,623,747`; `hermes_cli/blueprint_cmd.py`; `tools/blueprints.py` | none (`grep -rin "blueprint"` over `Scheduling` → 0). Nearest: automation previews / `payload_hash` (validation, not templates) — `Scheduling/models.py:159-186` | HERMES AHEAD | CODE · PRIOR (mischaracterized — see §3.6) |
| Agent-proposed cron suggestions | `add_suggestion`/`accept_suggestion` inbox + seeded catalog — `cron/suggestions.py:127,225`; `cron/suggestion_catalog.py:44-124` | none (`grep -rin "suggestion"` over `Scheduling` → 0) | HERMES AHEAD | CODE / NEW — **audit missed** |
| Cron CRUD as an agent tool | `cronjob` model tool, full create/edit/run/list — `tools/cronjob_tools.py:1460,1956` | none (`grep -rn "reminder\|scheduled_task\|SchedulingService" Tools/ Agents/` → 0) | HERMES AHEAD | CODE / NEW |
| Pluggable scheduler provider | `CronScheduler` ABC + `resolve_cron_scheduler` + Chronos NAS plugin — `cron/scheduler_provider.py:93-263`; `plugins/cron_providers/chronos/` | none | N/A BY DESIGN (hosted scale-to-zero) | CODE · PRIOR — rejection still correct |
| Sync / offline / conflicts | none — `jobs.json` is gateway-local | full sync engine: mappings, tombstones, pending mutations, server-wins conflicts + Conflicts tab — `db/migrations/v0_to_v1.py:132-190`; `services/sync_engine.py`; `UI/Screens/scheduling/conflicts_tab.py` | CHATBOOK AHEAD | CODE · PRIOR |
| Storage substrate | JSON file + advisory `flock` + merge-unexpected-disk-jobs repair — `cron/jobs.py:1610-1834` | versioned SQLite w/ 3 forward migrations — `db/migrations/v0_to_v1.py`, `v1_to_v2.py:54`, `v2_to_v3.py:45` | CHATBOOK AHEAD | CODE · PRIOR |

### 2.7 Interop

| capability | hermes | chatbook (file:line) | verdict | source |
|---|---|---|---|---|
| ACP agent/server (be an ACP agent for Zed/VS Code/JetBrains) | full JSON-RPC server: initialize/authenticate, session new/load/resume/set_mode/set_model, streaming updates, available_commands — `acp_adapter/server.py:1305,1626,1690,2224,2570,2604` (2640 ln + 11 modules) | no ACP protocol. `ACP_Interop/` only spawns a configured subprocess and tracks status; the screen renders "Diffs unavailable" — `ACP_Interop/runtime_process.py:185,203,266`; `UI/Screens/acp_screen.py:277,280` | HERMES AHEAD | CODE / NEW |
| ACP permission / edit-approval bridge | `session/request_permission` mapped to hermes approval outcomes + diff/edit approval — `acp_adapter/permissions.py:22`; `acp_adapter/edit_approval.py` | none (grep `request_permission` over `tldw_chatbook/` → 0) | HERMES AHEAD | CODE / NEW |
| ACP tool-call typing / provenance | `TOOL_KIND_MAP` → ACP ToolKind — `acp_adapter/tools.py:24`; `acp_adapter/provenance.py` | absent | HERMES AHEAD | CODE / NEW |
| ACP as a **client** (drive another ACP agent as a provider) | `copilot --acp` driven as an OpenAI-shaped backend — `agent/copilot_acp_client.py:1`; `agent/acp_openai_bridge.py:1` | absent | HERMES AHEAD | CODE / NEW |
| MCP client — stdio | yes — `tools/mcp_tool.py:5,344` | yes, hand-rolled JSON-RPC over stdio pipes — `MCP/client.py:815,822,851` | PARITY | CODE / NEW |
| MCP client — Streamable HTTP / SSE | both — `tools/mcp_tool.py:5,349,384` | **none** (grep `httpx\|aiohttp\|sse_client\|streamable` over `MCP/*.py` → 0; only `connect_to_server(command, args, env)`) — `MCP/client.py:815` | HERMES AHEAD | CODE / NEW |
| MCP client — OAuth 2.1 / PKCE / DCR | 1957-line stack: PKCE, dynamic client registration, token storage, CIMD, registration-poisoning guard; TUI login flows — `tools/mcp_oauth.py:5,457,686`; `tui_gateway/mcp_oauth_sessions.py:1` | none; secrets are env placeholders only — `MCP/local_store.py:38` | HERMES AHEAD | CODE / NEW |
| MCP client — sampling | `sampling/createMessage` w/ rate limiting + tool-use responses — `tools/mcp_tool.py:1771,2028` | server→client: only `ping`; everything else `-32601` — `MCP/client.py:744,754` | HERMES AHEAD | CODE / NEW |
| MCP client — elicitation | form-mode routed to approval surfaces, URL-mode declined — `tools/mcp_tool.py:2203,2245,2299` | same `-32601` fallthrough — `MCP/client.py:754` | HERMES AHEAD | CODE + RELEASE-NOTE / NEW |
| MCP client — resources & prompts | exposed to the model as synthetic tools `mcp__<srv>__{list_resources,read_resource,list_prompts,get_prompt}` — `tools/mcp_tool.py:6420-6944` | client can `read_resource`/`get_prompt`, surfaced through the control plane, but the **agent** provider exposes tools only — `MCP/client.py:1044,1076`; `MCP/unified_control_plane_service.py:1177,1183`; `Agents/mcp_tool_provider.py:307,653` | PARITY (protocol) · HERMES AHEAD (agent exposure) | CODE / NEW |
| MCP client — curated catalog + installer | 65 vendored manifests + `hermes mcp catalog/install/picker` — `hermes_cli/mcp_catalog.py:1`; `hermes_cli/mcp_picker.py`; `optional-mcps/` | no catalog. Claude-Desktop `mcpServers` JSON import + server-side registry browsing via tldw_server — `MCP/mcp_import.py:2,51`; `MCP_Governance_Interop/server_mcp_governance_service.py:202,228` | HERMES AHEAD | CODE + RELEASE-NOTE v0.20.6 (50+ vendor servers) / NEW |
| MCP client — governance on external tools | approval routing + exfil-shaped stdio config blocking — `hermes_cli/mcp_security.py`; `hermes_cli/mcp_config.py:88` | deeper: per-tool persisted permission store w/ definition hashes, kill switch, prompt-reduction analytics, execution log — `MCP/permission_store.py:1`; `permission_prompt_reducer.py:1`; `execution_log.py:1` | CHATBOOK AHEAD | CODE / NEW |
| MCP client — stdio recycle / watchdog | idle + lifetime recycle, watchdog, schema cache — `tools/mcp_tool.py:26`; `tools/mcp_stdio_watchdog.py` | connection deadline + bounded teardown + forced process stop; no idle/lifetime recycle (grep `idle_timeout\|max_lifetime` over `MCP/` → 0) — `MCP/client.py:840,1280,1298` | PARITY | CODE / NEW |
| MCP **server** — stdio | 10 messaging tools — `mcp_serve.py:4,1053` | 9 built-ins + resources + prompts over `mcp-unified` stdio — `MCP/server.py:515,593,1010`; `MCP/tools.py:64`; `resources.py:29`; `prompts.py:27` | PARITY | CODE / NEW |
| MCP **server** — HTTP/SSE | stdio only (grep `streamable\|sse\|http` over `mcp_serve.py` → stdio only) | config advertises `transport = "http"` (`config.py:4856`) but code raises `NotImplementedError("Only stdio transport is supported")` — `MCP/server.py:1016` | PARITY (both stdio-only; chatbook's config string is misleading) | CODE / NEW |
| MCP **server** — surface breadth | conversations/messages/attachments/events/permissions across gateway platforms — `mcp_serve.py:11` | chat, RAG search, media, notes, characters + opt-in `fs_*`/`git_*`/`web_*`/`watchlists_*` and 24 `library_*`, all permission-gated — `MCP/server.py:11,518`; `MCP/local_server_tools.py:130,239` | CHATBOOK AHEAD | CODE / NEW |
| OpenAI-compatible API server | 7937-line server: `/v1/chat/completions`, `/v1/responses`, `/v1/models`, `/v1/capabilities`, `/v1/runs` (+SSE events/approval/steer/stop), per-profile `/p/<profile>/v1/*` — `gateway/platforms/api_server.py:5-36,2219-2247` | none (grep `FastAPI\|aiohttp.web\|uvicorn\|starlette` over `tldw_chatbook/` → doc-comments/TTS only). Only HTTP surface is textual-serve — `Web_Server/serve.py:224,232` | HERMES AHEAD | CODE / NEW |
| Remote/browser access to the agent UI | React dashboard + WS/RPC host — `tui_gateway/server.py` (17775 ln); `tui_gateway/ws.py`; `web/` | textual-serve over aiohttp+WS w/ font-size/viewport patching — `Web_Server/serve.py:151,160,232,304` | PARITY | CODE / NEW |
| Chat-platform gateways (Discord/Slack/Telegram/WhatsApp/Matrix/Teams/Feishu/DingTalk/…) | 22 platform plugins + core adapters + a 109-file `gateway/` runtime — `plugins/platforms/`; `gateway/platforms/` | none (grep over `tldw_chatbook/**/*.py` → only `_RESULT_WAIT_SLACK_SECONDS`, unrelated) | N/A BY DESIGN (hosted multi-user chat-bot fan-out; chatbook is a single-user local TUI with no always-on daemon) | CODE / NEW |
| Relay connector contract | generic `RelayAdapter` dialing a Node connector over WS w/ capability handshake — `gateway/relay/adapter.py:1`; `docs/relay-connector-contract.md` | absent | N/A BY DESIGN (exists solely to front hosted chat platforms) | CODE / NEW |
| Hosted multi-agent rooms / peer replicas | durable room identity + event log, drivers, peers, replicas, execution policy — `gateway/hosted_rooms.py:1` and siblings | absent | N/A BY DESIGN (multi-user hosted concept) | CODE / NEW |
| Inbound webhooks (GitHub/JIRA/Stripe → agent run) | aiohttp webhook platform w/ per-route HMAC (V2 timestamped), rate limit, idempotency, delivery routing; MS Graph change notifications — `gateway/platforms/webhook.py:1-33`; `msgraph_webhook.py:1` | none locally; only remote-server webhook config passthrough — `runtime_policy/registry.py:1280` | HERMES AHEAD | CODE / NEW |
| Outbound signed webhooks (run lifecycle → your systems) | HMAC-SHA256 signed fire-and-forget off the hook manager — `agent/outbound_webhooks.py:1-25` | absent (grep `webhook` over `tldw_chatbook/**/*.py` excl. `_Interop`/`tldw_api` → 1 capability-id string) | HERMES AHEAD | CODE + RELEASE-NOTE (Herald) / NEW |
| Agent-to-agent across processes (A2A v1.0) | agent card at `/.well-known/agent-card.json`, JSON-RPC `message/send`, `message/stream` SSE, `tasks/*`, HMAC push notifications + 5 outbound client tools — `plugins/platforms/a2a/adapter.py:1-17` | in-process fleet only: `FleetCoordinator` state machine over asyncio handles; no wire protocol, no discovery — `Agents/fleet_coordinator.py:148` | HERMES AHEAD | CODE + RELEASE-NOTE (A2A v1.0) / NEW |
| Local IPC control surface | versioned JSON control socket (`identify`/`status`) — `gateway/control_socket.py:1-18` | absent | HERMES AHEAD | CODE / NEW |
| Connectors — Drive / cloud docs | `tools/drive_preview_tool.py`, `feishu_drive_tool.py`, `feishu_doc_tool.py` | Drive/Notion/Gmail/OneDrive/Zotero w/ OAuth authorize+callback+browse/import/sync — but every call is proxied to a remote tldw_server — `External_Connectors_Interop/server_connectors_service.py:96,114,210` | PARITY (hermes local-tool vs chatbook server-delegated) | CODE / NEW |
| Connectors — Home Assistant / Spotify / Meet / Teams | `tools/homeassistant_tool.py` (514 ln), `plugins/spotify/tools.py` (454 ln), `plugins/google_meet/`, `plugins/teams_pipeline/` | none for HA/Spotify (grep → 0); meetings exist but are remote-server calls — `Meetings_Interop/server_meetings_service.py:1` | HERMES AHEAD | CODE / NEW |
| Connector — GitHub | `optional-mcps/gitlab` + GitHub via webhook delivery — `gateway/platforms/webhook.py:15` | local GitHub API client for repo copy-paste, skill fetch, subscription scraping — `Utils/github_api_client.py:1,46`; `UI/CodeRepoCopyPasteWindow.py:36`; `Subscriptions/scrapers/github_scraper.py:56,170` | CHATBOOK AHEAD | CODE / NEW |
| Browser extension as a controlled surface | registry-level router dispatching `browser_*` to an attached extension w/ bound-identity fail-closed semantics; broker + WS endpoints — `tools/browser_extension_router.py:1-30`; `gateway/browser_control_broker.py` | web-clipper capture API is a client to a remote server; no extension, no local receiver (`find . -name manifest.json` → tiktoken cache + docs only) — `Web_Clipper_Interop/server_web_clipper_service.py:14,94` | HERMES AHEAD | CODE / NEW |
| Device-to-device encrypted sync | not found (grep `outbox\|conflict_review\|envelope_applier` over `gateway/`, `agent/`, `tui_gateway/` → 0) | 36-file encrypted local-first sync: envelope build/apply, crypto, key recovery, conflict review, per-domain adapters — `Sync_Interop/local_first_sync_service.py:1`; `crypto.py:1`; `envelope_applier.py:1`; `conflict_review.py:1` | CHATBOOK AHEAD | CODE / NEW |

> **Naming note.** Chatbook's ~30 `*_Interop` packages are *not* protocol interop — they are local/remote dual-mode
> service facades over `tldw_api/client.py`, each pairing a `*_scope_service` (capability resolution) with a
> `server_*_service` (policy-enforced remote calls). Only `MCP*` and `ACP_Interop` are protocol-shaped, and
> `ACP_Interop` is a process launcher.

### 2.8 User surfaces

| capability | hermes | chatbook (file:line) | verdict | source |
|---|---|---|---|---|
| Shell shape | prompt_toolkit single-stream REPL + `!bang` shell + curses checklists — `hermes_cli/bang_shell.py:1`; `hermes_cli/curses_ui.py:1` | full Textual multi-screen TUI, 60+ screens, left/right rails, tabs — `UI/Screens/chat_screen.py:1725-1733` | N/A BY DESIGN (shape, not capability) | CODE / NEW |
| Slash-command registry size | **101** `CommandDef` + ~24 aliases in one registry feeding CLI/gateway/Telegram/Slack/autocomplete — `hermes_cli/commands.py:149-416` | **10** `ConsoleCommand`s, no aliases — `Chat/console_command_grammar.py:194-272` | HERMES AHEAD | CODE / NEW |
| Command palette | `/palette` + Ctrl+P fuzzy modal over the registry — `cli.py:11248-11361,18955-18975` | Textual palette Ctrl+P w/ `ConsoleCommandProvider` (13 actions), `ThemeProvider`, dev-commands provider — `app.py:7099,1075`; `UI/console_command_provider.py:33-97` | PARITY | CODE / NEW |
| Shell completion (bash/zsh/fish) | generated from the argparse tree — `hermes_cli/completion.py:55,146,251` | none — `tldw_chatbook/cli.py` is 22 lines, one entrypoint — `cli.py:6` | N/A BY DESIGN (no CLI subcommand tree to complete) | CODE / NEW |
| In-composer autocomplete | slash popup + `@file:`/`@folder:` path completer + subcommand completion — `hermes_cli/commands.py:1865-1935` | `/`-prefix command rows + `/skills <name>` arg rows + `$`-sigil skill mentions; **no file-path completion** — `Chat/console_command_suggestions.py:100-173`; `console_skill_resolver.py:36,156` | HERMES AHEAD | CODE · partly task-18921 |
| Slash popup usage ranking | leads with most-used skills | absent | HERMES AHEAD | RELEASE-NOTE v0.20.6 · task-18921 |
| i18n / UI localization | 17 locale YAMLs + `t()` runtime, `HERMES_LANGUAGE`/`display.language` — `agent/i18n.py:43,91,232`; `locales/*.yaml` | **none** (named grep `gettext\|babel\|i18n\|ngettext` over `tldw_chatbook/` → 4 hits, all false positives: a `.babelrc` filename constant and a git config flag) | HERMES AHEAD | CODE / NEW |
| RTL / bidi | auto RTL/bidi direction; Arabic across desktop/dashboard/agent | none (named grep `rtl\|bidi` over `Widgets/Console/` → 0 real hits) | HERMES AHEAD | RELEASE-NOTE v0.17.0, v0.20.6 / NEW |
| CJK wide-char table alignment | `wcwidth`-based markdown table re-padding — `agent/markdown_tables.py:1-22` | `wcwidth` used only in `UI/character_display_text.py` and `Terminal/screen_model.py`, not for model markdown | HERMES AHEAD (narrow) | CODE / NEW |
| Themes / skins | 9 built-in skins + `/skin` — `hermes_cli/skin_engine.py:201`; desktop can install any VS Code Marketplace theme | **58** registered Textual themes + palette theme picker — `css/Themes/themes.py:1425-1484`; `app.py:13206-13207,1075` | CHATBOOK AHEAD (TUI) · HERMES AHEAD (desktop marketplace) | CODE + RELEASE-NOTE v0.17.0 / NEW |
| Session browsing / resume | `/sessions`, `/resume <name>`, `-c/--continue` per-terminal, lost-and-found, recap, foreign sessions — `hermes_cli/cli_commands_mixin.py:1271,1074`; `session_lost_and_found.py` | Ctrl+K fuzzy switcher + conversation-browser rail w/ workspace groups, starred sort, run markers, relative ages — `UI/Screens/chat_screen.py:1664,3687`; `Workspaces/conversation_browser_state.py:420,691,787` | PARITY | CODE · task-18924 (dividers) |
| Branch / fork / undo | `/branch`(`/fork`), `/undo N`, `/retry`, `/rollback` — `hermes_cli/commands.py:164-180`; `hermes_cli/checkpoints.py:1` | `/rewind` + fork lineage/fences projection — `Chat/console_command_grammar.py:69-72,260`; `Chat/console_chat_fork.py:170-273` | PARITY | CODE / NEW |
| Diff / edit review UI | `/diff [staged\|all\|session] [--stat]` → colorized text, 400-line cap — `hermes_cli/cli_commands_mixin.py:188-320` | dedicated `ChangeReviewScreen`: per-file diff pane, j/k navigate, `u` revert file, `U` undo all, `C` comment, `g` commit, `p` push, `P` open PR — `UI/Screens/change_review_screen.py:1339-1366,1258` | CHATBOOK AHEAD | CODE / NEW |
| Filesystem checkpoints (user-facing) | `hermes checkpoints status/list/prune/clear`, `/rollback`, `/diff session` baseline — `hermes_cli/checkpoints.py:1-20` | file revert via change-review `u`/`U` only; no time-indexed session baseline — `UI/Screens/change_review_screen.py:1347-1348` | HERMES AHEAD | CODE / NEW |
| Doctor / diagnostics UX | `hermes doctor` (3395 ln) + `doctor_live` parallel probes for MCP/firecrawl/fal/browser/TTS/STT + `/debug` upload — `hermes_cli/doctor.py`; `doctor_live.py:267,320` | **none** (named grep `def .*doctor\|class .*Doctor\|diagnostic\|healthcheck\|self_test` over `tldw_chatbook/` → only a log sink and TTS health intervals) | HERMES AHEAD | CODE + RELEASE-NOTE v0.19.x / NEW |
| Onboarding / first-run | `hermes setup` interactive (3876 ln, curses prompts, back-nav, Quick vs Full) + `hermes init` — `hermes_cli/setup.py:161-474,951` | 11-step `FirstRunSetupWizard` (welcome/provider/model/voice/RAG/speech/tools/notes/appearance/protect-keys/summary) + in-Console "Get started" card — `UI/Wizards/first_run_setup_state.py:915-937`; `Chat/console_onboarding_state.py:13` | CHATBOOK AHEAD | CODE / NEW |
| Voice: wake word | `/wake` "Hey Hermes"; openWakeWord/sherpa/Porcupine providers, open-vocabulary, multi-profile routing — `hermes_cli/config_defaults.py:1975-2003`; `cli_commands_mixin.py:4082-4107` | **none** (named grep `wake_word\|wakeword\|openwakeword\|porcupine` over `tldw_chatbook/` → 0) | HERMES AHEAD | CODE · task-18933 |
| Voice: barge-in / hands-free | streaming clause-by-clause TTS, acoustic barge-in w/ threshold multiplier + grace window — `cli.py:15362-15424` | headless hands-free FSM (idle→listening→countdown→awaiting_reply→speaking), acoustic barge-in, spoken "stop" — `Chat/console_hands_free.py:1-30`; `UI/Console_Modules/hands_free.py:126,651,827` | PARITY | CODE / NEW |
| Voice: auto-speak replies | auto-TTS, platform-aware | eligibility + consent policy, per-message speak/stop action — `Chat/console_auto_speak.py:1`; `Chat/console_message_actions.py:226` | PARITY | RELEASE-NOTE v0.20.6 / NEW |
| Voice authoring surfaces | TTS/STT provider registries + `hermes tools` STT category — `hermes_cli/tts_registry.py` | STTS playground screen, voice-cloning window, TTS profile library, audiobook generator — `UI/STTS_Window.py`; `UI/Voice_Cloning_Window.py`; `UI/stts_profile_library.py:1592`; `TTS/audiobook_generator.py` | CHATBOOK AHEAD | CODE / NEW |
| Images in terminal | kitty/iTerm2/sixel/unicode half-block w/ `--mode` override — `hermes_cli/pets.py:146,490`; `config_defaults.py:1640-1650` | kitty→sixel→halfcell→ascii detection + inline view modes + per-message state + render cache — `Media_Playback/render_mode.py:16-62`; `Chat/console_image_view.py:1-8`; `Utils/terminal_utils.py:110-172` | PARITY | CODE / NEW |
| Video in terminal | none (named grep `video.*render\|play video\|mpv\|ffplay` over `hermes_cli/` → 0) | `VideoPlayerScreen` w/ space/s/←/→/q transport over the same 4-rung renderer; `/stream-video`, `/generate-video` — `UI/Screens/video_player_screen.py:74-83,122-130`; `Chat/console_command_grammar.py:57-67` | CHATBOOK AHEAD | CODE / NEW |
| Clipboard | `/copy`, `/paste` clipboard image, `/image <path>`; pngpaste/osascript/xclip — `hermes_cli/commands.py:401-405`; `hermes_cli/clipboard.py:135-191` | Alt+V paste clipboard image + palette route; per-message Copy action — `UI/Screens/chat_screen.py:1683`; `Chat/console_message_actions.py:143,195` | PARITY | CODE / NEW |
| Notifications | in-chat + gateway modes; **native OS notifications w/ per-type toggles** on desktop; kanban OS notifications | Textual toasts + a full notification delivery/presentation service; **no OS-level notify** (grep `osascript\|notify-send\|plyer\|terminal-notifier` over `tldw_chatbook/` → 0) — `app.py:8382`; `Notifications/notification_presentation.py:49-120` | HERMES AHEAD (OS-level) | CODE + RELEASE-NOTE v0.17.0, v0.20.6 · task-18925 |
| Kanban / work board | `hermes kanban` CLI (3542 ln): multi-board, workspaces, dispatcher, swarm, decompose, transfer + web dashboard plugin + `/kanban` — `hermes_cli/kanban.py:216-265`; `plugins/kanban/dashboard/plugin_api.py` | data services only, **zero UI** (named grep `kanban\|Kanban` over `UI/`, `Widgets/` → 0); only `Kanban_Interop/{local,server}_kanban_service.py` | HERMES AHEAD | CODE · task-18934 |
| Prompt stash / drafts | Ctrl+S stash panel w/ up/down/enter/d/D/esc; `/prompt` composes in `$EDITOR` — `hermes_cli/cli_commands_mixin.py:3405-3449` | per-session composer draft persistence w/ auto-swap on session change; no multi-draft stash (grep `stash` over `tldw_chatbook/` → 0 feature hits) — `UI/Console_Modules/session.py:3835-3901` | HERMES AHEAD | CODE · task-18930 |
| Status / busy indicator | `/statusbar`, `/battery`, `/indicator` (4 styles), `/timestamps`, `/focus` — `hermes_cli/commands.py:269-287`; `hermes_cli/focus_view.py:1-21` | status-chip strip w/ position + collapse prefs, live-work status card, glyph vocabulary — `UI/Console_Modules/status_row.py:1-11`; `Chat/console_live_work.py:13-15`; `Chat/console_glyphs.py:13-40` | PARITY | CODE · task-18923 |
| Keyboard model | prompt_toolkit ~50 context-filtered bindings + kitty CSI-u normalization; rebindable shortcuts on desktop — `cli.py:18955+`; `hermes_cli/pt_input_extras.py:14,281,449` | 26 `Binding`s on ChatScreen + 6 app-level, F6 pane cycling, "Show Keybindings" palette action — `UI/Screens/chat_screen.py:1651-1733`; `app.py:7098-7110,2004-2006` | PARITY | CODE + RELEASE-NOTE v0.17.0 · task-18935 |
| Help surface | `/help`, `/help skills`, `/help <filter>`, paginated `/commands` browser, `/whoami` — `hermes_cli/commands.py:381-386,245` | F1 workbench help + a palette action; **no `/help` command** — `app.py:7100,2004` | HERMES AHEAD | CODE / NEW |
| Embedded terminal pane | `!bang` inline shell through the same approval gate, zero tokens — `hermes_cli/bang_shell.py:1-16`; resizable terminal pane on desktop | `Terminal/` backend + screen model exist but are **unwired to any UI** (grep over `UI/`, `Widgets/` → 1 unrelated comment) | HERMES AHEAD | CODE + RELEASE-NOTE v0.17.0 / NEW |

**Slash-command head-to-head.** Hermes registers 101 commands in `hermes_cli/commands.py:149-416`; chatbook registers 10
in `Chat/console_command_grammar.py:194-272` (`/prompt`, `/system`, `/skills`, `/fewer-permission-prompts`, `/prefill`,
`/generate-image`, `/generate-video`, `/stream-video`, `/rewind`, `/research`) plus `$name` skill mentions and 13
non-slash palette actions. Chatbook has, hermes lacks: `/generate-video`, `/stream-video`, `/prefill`,
`/fewer-permission-prompts`, `/system`. The ~80 hermes commands chatbook lacks as *typed names* include `/help`,
`/commands`, `/status`, `/context`, `/config`, `/model`, `/save`, `/history`, `/retry`, `/undo`, `/branch`,
`/compress`, `/sessions`, `/resume`, `/new`, `/clear`, `/copy`, `/paste`, `/diff`, `/rollback`, `/queue`, `/steer`,
`/stop`, `/approve`, `/deny`, `/tools`, `/memory`, `/kanban`, `/usage`, `/version`, `/update`, `/debug`, `/export`,
`/import`. **Many of these exist in chatbook as screens or keybindings** (model→Alt+M, sessions→Ctrl+K,
diff→ChangeReviewScreen, skin→palette theme picker) — the gap is the typed-command route, not always the capability.

### 2.9 Ops

| capability | hermes | chatbook (file:line) | verdict | source |
|---|---|---|---|---|
| Config format + location | YAML at `~/.hermes/config.yaml` + `.env` sidecar, `HERMES_HOME` override — `hermes_cli/config.py:770-776` | TOML at `~/.config/tldw_cli/config.toml`, `TLDW_CONFIG_PATH` override — `config.py:91,96` | PARITY | CODE / NEW |
| Config layering | 4 layers: defaults → user → `${VAR}` expansion → managed `/etc/hermes/config.yaml` winning per-leaf — `hermes_cli/config.py:3792-3900`; `managed_scope.py:1-17` | 2 layers: defaults deep-merged with user TOML + scattered per-key `os.getenv`; no admin/system layer — `config.py:5113,5128,1727-1955` | HERMES AHEAD | CODE / NEW |
| Per-project config | per-user + per-profile home only; no cwd-scoped file — `hermes_cli/config.py:5` | per-user only (named grep `cwd()/".tldw"\|project_config\|\.tldwrc` over `tldw_chatbook/` → 0) | PARITY | CODE / NEW |
| Profiles / multi-instance | full system: isolated `profiles/<name>/` HERMES_HOME, create/list/delete, wrapper scripts, aliases, per-profile skill seeding + config migration — `hermes_cli/profiles.py:282,505,578,1029,1177` | no config-profile system ("profile" in chatbook = RAG/GitHub/trace-safety profiles); one advisory instance lock — `Utils/instance_lock.py:1-8` | HERMES AHEAD | CODE / NEW |
| Config schema migrations | versioned 12→39 w/ a `MIGRATIONS` table, `run_migrations`, `check_config_version` — `hermes_cli/config_migrations.py:869,896`; `config.py:2043,2365` | **none** (grep `config_version\|CONFIG_VERSION\|migrat` over `config.py` → only DB/feature-flag comments) | HERMES AHEAD | CODE / NEW |
| DB schema migrations | state-DB `SCHEMA_VERSION` + stepwise in-code migrations — `hermes_state.py:1033,1069,1301` | `db_schema_version` table (currently v62) + stepwise SQL chain + `DB/migrations/*.sql` — `DB/ChaChaNotes_DB.py:582,605,1382` | PARITY | CODE / NEW |
| Unknown-key / deprecated-key validation | `validate_config_structure` + `ConfigIssue` + typo suggestions via `get_close_matches` — `hermes_cli/config.py:2130,2138,2301` | adapter-level TOML parse + table-shape check only, surfaced in Settings → Diagnostics; no unknown/deprecated detection — `UI/Screens/settings_screen.py:9030,9044-9060` | HERMES AHEAD | CODE / NEW |
| Corrupt-config resilience | backs up the corrupt file **and** serves last-known-good in-process config — `hermes_cli/config.py:48,3862-3893` | falls back to bare internal defaults, records `ConfigLoadFailure`, raises a notification; no last-known-good, no backup — `config.py:5052,5079,5165-5178` | HERMES AHEAD | CODE / NEW |
| Config write safety | `atomic_config_write` + `require_readable_config_before_write` — `hermes_cli/config.py:3468,3552` | atomic private write under RLock + **cross-process `portalocker` EXCLUSIVE lock** + `.bak` before editor replacement + timestamped snapshots — `config.py:5285,5326,5789,5845,5945` | PARITY | CODE / NEW |
| Hot reload of external edits | cache keyed on (mtime,size) of user + managed files plus an env snapshot → picked up on next read — `hermes_cli/config.py:3798-3838` | cache keyed on path only; needs `force_reload=True` or a manual "Reload config" button; no file watcher (named grep `watchdog\|Observer(\|hot.reload` → 0 config hits) — `config.py:5101-5107`; `UI/Screens/settings_screen.py:9037` | HERMES AHEAD | CODE / NEW |
| Secrets at rest | plaintext `.env` chmod 0600 + redaction helpers; no encryption — `hermes_cli/config.py:898-911` | **AES-256-GCM + scrypt** over sensitive values, password-gated, enable/disable/change-password, single-source `is_sensitive_config_key` — `Utils/config_encryption.py:1-6`; `config.py:7283,7310,7350`; `Utils/sensitive_config_keys.py:1` | CHATBOOK AHEAD | CODE + RELEASE-NOTE v0.20.6 (hermes added opt-in OS-keychain encryption) / NEW |
| Telemetry — remote sink | no remote metrics sink (`telemetry.shared_metrics.enabled=False`, "no remote sink exists"); local SQLite aggregates. Opt-in `hermes debug share --nous` uploads a redacted bundle on explicit command; third-party PostHog explicitly disabled — `hermes_cli/config_defaults.py:3470-3475,3751-3757`; `observability/shared_metrics.py:1` | **zero vendor analytics and no upload path at all** (named grep `posthog\|sentry\|mixpanel\|amplitude\|segment\|google-analytics\|plausible\|umami` over `tldw_chatbook/` → 1 false positive, a splash string at `Utils/Splash_Strings.py:300`) | PARITY (both no-phone-home) · chatbook has no upload command at all | CODE / NEW |
| Metrics backend | local SQLite shared-metrics store (opt-in) + optional OTLP export — `observability/shared_metrics.py:28`; `agent/monitoring/otlp_exporter.py:107,233` | Prometheus client + OTel meter w/ `PrometheusMetricReader` (local scrape, no OTLP push) + system-metrics instrumentation — `Metrics/metrics.py:245,261`; `Metrics/Otel_Metrics.py:61,105,110` | PARITY | CODE + RELEASE-NOTE v0.20.0 / NEW |
| Traces / spans | OTLP span export w/ resource attrs, batch + streaming exporters — `agent/monitoring/otlp_exporter.py:118-182` | none — OTel integration is metrics-only, no `TracerProvider` — `Metrics/Otel_Metrics.py:154` | HERMES AHEAD | CODE / NEW |
| Log rotation | `RotatingFileHandler` (Concurrent on Windows), config-driven level/size/backups — `hermes_logging.py:66,721,750-801` | `PrivateRotatingFileHandler` w/ config-driven size/backups/level, **re-hardens permissions on every rotation generation** — `Logging_Config.py:193,230-266,355-365` | PARITY | CODE / NEW |
| Log redaction | `RedactingFormatter` on every file sink — `hermes_logging.py:14,316,384`; `agent/redact.py:774,1103` | `RedactingFileFormatter` on the private sink + package-wide loguru `diagnose=False` so exception frames never leak locals — `Logging_Config.py:267-309,345-347`; `__init__.py:26-72` | PARITY | CODE / NEW |
| Structured / metadata-only log admission | session-id record factory + component prefix filters; no strict schema — `hermes_logging.py:165,183,226-230` | strict admission: the persistent sink accepts only `log_persistent_metadata` records w/ a small schema, enforced by a handler filter **and a repo check script** — `Utils/persistent_diagnostics.py:1-8,17`; `Logging_Config.py:368`; `scripts/check_persistent_diagnostic_inventory.py` | CHATBOOK AHEAD | CODE / NEW |
| Log viewing | `hermes logs` CLI: tail/follow/session/level/component/time filters — `hermes_cli/logs.py:1-12` | in-TUI Logs tab over a `RichLogHandler` queue; no CLI log command — `Logging_Config.py:43-157`; `Utils/log_widget_manager.py:17` | PARITY (different surface) | CODE / NEW |
| Doctor / health check | `hermes doctor` (3395 ln, ~30 checks: DB journal modes, cert bundle, version consistency, s6, systemd linger, macOS TCC/FDA, managed scope, provider keys, toolsets, deprecated config) + `--fix` + `--live` real-call probes — `hermes_cli/doctor.py:641-1236`; `doctor_live.py:1-18` | **none** (named grep `\bdoctor\b` over `tldw_chatbook/`, `Packaging/`, `scripts/` → only TTS abbreviation expansions and an eval fixture) | HERMES AHEAD | CODE + RELEASE-NOTE v0.19.x / NEW |
| Self-repair / install repair | stdlib-only `_early_recovery.recover_if_needed` runs before any third-party import to repair a wiped venv; shared reinstall executor w/ uv→pip fallback; venv-blocker JSON probe; lazy dep bootstrap — `hermes_cli/_early_recovery.py:1-23`; `_install_repair.py:1-20`; `_scan_venv_blockers.py:1-10`; `dep_ensure.py:1-14` | none (named grep `self.?update\|auto.?update\|check.?for.?updates` over `tldw_chatbook/` → only unrelated widget methods). Only recovery surface is an interrupted-first-run dialog — `UI/Wizards/first_run_recovery_dialog.py:1-15` | HERMES AHEAD | CODE / NEW |
| Self-update | `hermes update` w/ cross-process lock, machine-readable receipts, pre-update snapshot modes, runtime inventory + `--plan`, image-managed refusal contract, restart recovery — `hermes_cli/update_cmd.py:1`; `update_lock.py:1-12`; `update_receipt.py:1-13`; `update_contract.py:1-12` | none (same grep); upgrades go through pip/installer | HERMES AHEAD | CODE + RELEASE-NOTE v0.20.6 / NEW |
| Backup / export of user state | `hermes backup` zips all of `~/.hermes/`, `hermes import` restores; pre-update snapshots — `hermes_cli/backup.py:1-9` | per-DB `backup_database()` (sqlite online backup) on three DBs + config snapshot export + Chatbooks content export; no whole-home archive — `DB/ChaChaNotes_DB.py:3508`; `DB/Client_Media_DB_v2.py:7445`; `config.py:5936` | HERMES AHEAD | CODE / NEW |
| Crash forensics | `faulthandler.enable(all_threads=True)` to a dedicated log + SIGUSR2 dump; lifecycle ledger persisting termination reason across SIGKILL/OOM — `gateway/run.py:13159-13180`; `gateway/lifecycle_ledger.py:1-15` | none (named grep `excepthook\|faulthandler` over `tldw_chatbook/` → 3 comment hits). Targeted crash *guards* exist — `Utils/text_selection_crash_guard.py:1-15`; `Utils/fd_protection.py:1-10` | HERMES AHEAD | CODE / NEW |
| Resource monitoring | periodic structured `[MEMORY]` RSS+GC time series; `/api/status` disk rollup; drain-control marker — `gateway/memory_monitor.py:1-15`; `gateway/disk_status.py:1-14` | `log_resource_usage()` gauges RSS + CPU% on demand; `DBStatusManager` caches DB file sizes. No disk-free, no leak time series — `Metrics/metrics_logger.py:164-179`; `Utils/db_status_manager.py:1-12` | HERMES AHEAD | CODE / NEW |
| Diagnostics bundle / share | `hermes debug share` collects system info + logs, redacts at capture, uploads to paste.rs or a 14-day-expiring Nous bucket; `--no-redact` opt-out — `hermes_cli/debug.py:4-12,459,642`; `diagnostics_upload.py:1-20` | no bundle/share command; diagnostics read locally from the rotating log — `Logging_Config.py:321-365` | HERMES AHEAD | CODE / NEW |
| Packaging targets | curl installer (`scripts/install.sh`/`.ps1`), Docker + compose + s6, Nix flake / NixOS + home-manager, npm workspaces, Termux constraints, install-method detection driving per-method update commands — `Dockerfile`; `flake.nix`; `nix/nixosModules.nix`; `package.json:6-13`; `constraints-termux.txt:1-7`; `hermes_cli/config.py:472,623` | PyPI sdist+wheel w/ manifest verify, macOS `.app` (py2app/Nuitka) + `.dmg`, Windows Nuitka `.exe` + NSIS. **No Docker, Nix, npm, Termux, or Homebrew** (`find . -iname "Dockerfile*" -o -iname "*.nix" -o -iname "docker-compose*"` → 0) — `Packaging/build_dist.sh:1-30`; `Packaging/macos/scripts/package_dmg.sh:33`; `Packaging/windows/installer.nsi` | HERMES AHEAD (breadth) · CHATBOOK AHEAD (native signed desktop installers) | CODE + RELEASE-NOTE v0.19.0 / NEW |
| Container / service lifecycle | s6-rc per-profile reconciliation on container boot; systemd/launchd service manager; baked build SHA — `hermes_cli/container_boot.py:1-13`; `service_manager.py`; `build_info.py:1-12` | n/a | N/A BY DESIGN (terminal app, not a hosted service) | CODE / NEW |
| Egress / network policy | `urllib_security.py` + security-audit startup | app-wide SSRF policy: every resolved IP must be public and non-metadata unless user-seeded trusted or in `[web_security] allowed_hosts` — `Utils/egress.py:1-11` | CHATBOOK AHEAD | CODE / NEW |

---

## 3. Per-area detail (non-PARITY rows only)

Each entry: what hermes does · what chatbook does instead · one line on what closing it would take. No design work,
no task drafts.

### 3.1 Agent control loop

**Mid-run steering of the primary agent.** Hermes stashes user text and drains it onto the last tool result before
the next model call, with a pre-API drain so a steer sent *during* an API call still lands on that iteration
(`run_agent.py:3544,3692`; `agent/conversation_loop.py:2180-2235`). Chatbook has the whole machinery —
`format_steering_message`, a drain point at `Agents/agent_runtime.py:1196-1230`, a steering bar — but wires it only
for threaded fleet children; `drain_mailbox` is `None` for a primary by explicit design. Typed text goes to the
prompt queue for the next turn. *Closing it:* give the primary a mailbox and pass `drain_mailbox`; the drain point,
the formatter, and the 4000-char cap already exist.

**Active-turn redirect.** Hermes's `redirect(text)` aborts only the in-flight model request, keeps every completed
message and tool result, records the displayed partial reasoning as assistant context, appends the correction as a
real user message, and re-runs the same turn (`run_agent.py:3591-3690`). Chatbook's Stop is terminal: `RUN_CANCELLED`,
the stream settles as "Response stopped.", and the correction becomes a new turn with the tool results discarded
(`Chat/console_chat_controller.py:13048-13126`). *Closing it:* a cancel-reason distinction in `_signal_stop` plus a
re-entry path that preserves `run_messages`.

**In-loop retry and model fallback.** Hermes classifies every API failure (`agent/error_classifier.py:30-60`) and
drives jittered backoff, credential rotation, OAuth refresh, compression restart, and a fallback chain that resets the
retry counter per hop (`agent/conversation_loop.py:2999-3009,3516-3521`). Chatbook re-raises the first `call_model`
exception straight out of the loop into `RUN_ERROR` (`Agents/agent_runtime.py:1291-1302`); the named grep for
`fallback_model|retry_model|max_retries|backoff` over `Agents/` returns zero. *Closing it:* a retry wrapper around the
`deps.call_model` call site with a classified-exception predicate and bounded backoff.

**Graceful budget wrap-up and empty-response handling.** Hermes warns at 80 % of the wall budget by appending a
cache-safe notice to the newest tool message, and on exhaustion makes one tools-stripped summary call; it also treats
two consecutive zero-token empties from the same (model, provider, finish_reason) as deterministic and skips the
remaining retries (`agent/empty_response_guard.py:1-75`). Chatbook hard-stops to `RUN_STUCK` with a bare error step,
and an empty text turn with no tool calls is simply `RUN_DONE` (`Agents/agent_runtime.py:1169-1179,1427`).
*Closing them:* an elapsed-fraction check at the loop top, and a token/text check before the `RUN_DONE` return.

**Parallel tool batch and stream stall watchdog.** Hermes dispatches a tool batch concurrently under a shared,
dynamically-extended deadline with a worker cap (`agent/tool_executor.py:122,195-205`), and kills a stale reasoning
stream with a cross-turn breaker after five (`agent/reasoning_timeouts.py:116`). Chatbook iterates `for call in calls:`
sequentially (`Agents/agent_runtime.py:1612`) and relies solely on the httpx read timeout, so a provider dribbling
keep-alive bytes holds the run until `max_wall_seconds`. *Closing them:* a thread pool over the dispatch branch (the
`run_context` ContextVar bindings already anticipate off-thread tools), and a last-chunk timestamp check in the
streaming adapter.

**Global resumable pause, sub-agent worktree isolation, conversation rewind, MoA, out-of-loop forks.** Hermes has an
`ESTOP` sentinel file that pauses all *new* work without touching in-flight runs (`agent/estop.py:1-174`), per-child
git worktrees (`tools/subagent_worktree.py:1`), a durable `/undo N` that preserves the compaction handoff
(`cli.py:10558-10740`), an MoA loop (`agent/moa_loop.py:1-14` — task-18931), and detached cache-parity forks for
background review, curation, and `/btw` side questions. Chatbook has per-session Stop only, shared workspace roots for
children, regenerate-with-snapshot but no multi-turn undo, and runs lesson promotion *inside* the run as an
approval-gated tool. *Closing the cheapest:* ESTOP is one boolean checked at `AgentService.run_turn` entry.

**Footnote — a real overshoot.** `max_wall_seconds` is checked only at loop top (`Agents/agent_runtime.py:1175`) while
`_call_with_timeout(fn, seconds, …)` takes an absolute bound (`Agents/agent_service.py:1522`) that Console raises to
3600 s (`Chat/console_agent_bridge.py:408`). A single tool call can therefore exceed the run's wall budget.

### 3.2 Tools & skills

**Edit and patch self-recovery.** Hermes runs `old_string` through nine progressively fuzzier strategies (exact →
line_trimmed → whitespace_normalized → indentation_flexible → escape_normalized → trimmed_boundary →
unicode_normalized → block_anchor → context_aware) plus already-applied detection, escape-drift and backslash-doubling
guards, automatic re-indentation, and a whitespace-visualized `find_closest_lines` report
(`tools/fuzzy_match.py:74,126,263,333,394,1091`); `patch` routes through the same chain. Chatbook's `fs_edit` is
`content.count()` + `content.replace()`, returning `old_string not found in X` with no near-match
(`Tools/local_tool_impls.py:839-847`), and `fs_patch` returns a bare `patch_context_mismatch`
(`Tools/patch_tool_impls.py:211,407`). *Closing it:* a 2–3 strategy subset plus `is_already_applied`, and nearest-line
context on the not-found error. This is precisely task-18927's scope; the code shows it unshipped.

**Tool-output spill.** Hermes preserves oversized results in three tiers — per-tool pre-truncate, then
`maybe_persist_tool_result` writing full text to `$HERMES_HOME/cache/spillover/{tool_use_id}.txt` and handing the model
a preview plus a re-readable (sandbox-translated) path, then a 200 K per-turn aggregate sweep that spills the largest
survivors (`tools/tool_result_storage.py:1-42`). Chatbook's `_fit_result` cuts at 32 KiB and appends `… [truncated]`;
the tail is unrecoverable and there is no aggregate turn budget (`Agents/local_tool_provider.py:320-324`).
*Closing it:* a write-preview-and-path shim at the single `_fit_result` seam; `Skills_Interop/atomic_write.py` already
exists.

**Catalog discovery quality.** Hermes ranks with BM25 + Snowball stemming over a pre-tokenized corpus, accepts up to
ten parallel queries, and embeds the deferred-tool *name listing* directly in the `tool_search` description so the
model can never conclude a capability is unavailable (`tools/tool_search.py:405,506,551,672,786`). Chatbook does
single-query 4-tier substring matching with limit 8 and no listing (`Agents/tool_catalog.py:1408-1448`) — so
"find files by name" will not match `fs_glob`. *Closing it:* a `difflib.SequenceMatcher` fifth tier, plus the name
listing in the `find_tools` description (the bigger win of the two).

**Skill/plugin distribution.** Hermes ships 398 skills across two catalogs plus a hub with GitHub/tap sources,
lockfile provenance, quarantine, an audit log, a soft authoring linter, per-skill usage counters, an append-only
mutation ledger with content-addressed blobs, YAML skill *bundles* behind one slash command, and "blueprint" skills
that self-schedule via cron frontmatter. Chatbook has a well-hardened skill *runtime* but ships zero skills
(`find . -name SKILL.md` = 7, all fixtures and docs) and has no registry, bundles, telemetry, or lint. *Closing the
cheapest slice:* the format is already Claude-Agent-Skills compatible, so ship a small curated `.SKILLS/` set; a hub is
a separate project.

**Code execution, browser, image/video-gen as tools; execution environments.** Hermes has `execute_code` implementing
Programmatic Tool Calling (the model writes Python that RPCs back into the tool registry, so intermediate results never
enter context), 10 execution backends, 13 browser tools, and image/video generation with 7 pluggable backends. Named
greps over `tldw_chatbook/Tools/*.py` and `Agents/*.py` for `execute_code|code_execution|jupyter|docker`,
`browser_navigate|playwright|selenium`, and `image_generate|image_gen|video_gen` return zero tool-surface hits;
chatbook's image/video generation lives only in UI screens. *Closing the highest-leverage one:* PTC needs a subprocess
and an RPC back into `ToolCatalogRegistry.invoke_by_name` — both already exist
(`Tools/workspace_tool_worker.py:29`, `Agents/tool_catalog.py:1534`).

**Three small affordances.** `tools/terminal_hints.py:1-24` maps known non-zero-exit output shapes to one actionable
recovery line; `coerce_tool_args` repairs models that emit stringified JSON arrays (`model_tools.py:845,911`);
`FileStateRegistry` blocks a write to a file another agent wrote since this agent read it (`tools/file_state.py:1-30`).
Chatbook has none (named greps: `hint|suggest|recovery` over `raw_cli_executor.py` = 0; `Agents/agent_runtime.py:1149`
is a bare `json.loads`; `stale|concurrent` over `local_tool_impls.py` = 0). *Cheapest and highest-value:* a
schema-aware `json.loads` retry at `agent_runtime.py:1149`.

### 3.3 Permissions & guardrails

**Sandboxing — the code-level blind spot the release-note pass could not see.** Hermes routes every `terminal` and
`execute_code` call through a pluggable execution environment (`tools/environments/`), and the docker backend is
genuinely hardened: `--cap-drop ALL`, `--security-opt no-new-privileges`, `--pids-limit`, `--cpus`/`--memory`,
`--user`, and `--network=none` with a **post-start verification** that the container's actual network mode matches the
requested air-gap (`tools/environments/docker.py:1457-1463`). Isolated backends make the whole approval layer moot
(`tools/approval.py:4075`). Chatbook executes in-process or as a direct host subprocess; its isolation is a path jail
plus a one-shot worker holding an fd-pinned root. *Closing it:* an execution-environment abstraction plus a docker
backend — large, and chatbook has no seam for it today.

**No shell-command risk detection and no unbypassable floor.** `validate_raw_cli_request`
(`Tools/raw_cli_executor.py:144`) checks caller, shell, size, timeout and cwd — nothing about what the command *does*.
`rm -rf ~` arrives as an ordinary approval card, and once `approve_session` is granted
(`Agents/raw_shell_tool_provider.py:48`) it runs unreviewed for the rest of the Console session. Hermes has a hardline
list plus a sudo-stdin guard plus user `approvals.deny` globs, all of which fire *before* yolo/`mode: off`
(`tools/approval.py:551,735,794,4751`), and a 330-pattern detector with de-obfuscation variants
(`tools/approval.py:2412`). *Closing it:* port `detect_hardline_command` + `_command_detection_variants` (a
self-contained regex + tokenizer block) and call it from `validate_raw_cli_request` so it also covers `approve_session`.

**No per-arg allowlist grain.** Hermes stores command-text globs (`podman *`) so "always allow" can scope to a command
family (`tools/approval.py:3145`). Chatbook's "Always allow" is whole-tool — which is exactly why the approval card
deliberately withholds `always_allow` for raw shell (`Widgets/Chat_Widgets/chat_approval_card.py:57`). The result is a
real capability gap dressed as a safety choice: repetitive safe commands can never be made quiet. *Closing it:* an
argument-predicate layer on `set_tool_state`.

**No LLM guardian, no dry-run, no out-of-band transport, no post-tool hook.** Hermes's default posture is an aux-LLM
reviewer clearing false positives with an injection-hardened prompt (`tools/approval.py:3640`); `hermes approvals test`
replays the real evaluators for a dry run (`hermes_cli/approvals_test.py:1-31`); approvals can be presented on
Discord/Telegram/Slack or a plugin transport with digest-bound requests and fail-closed timeouts; and plugins can
observe `post_tool_call` (`model_tools.py:1184`). Chatbook asks the human every time, in the TUI card only, with no
post-tool seam and no user-authored hooks (`Tools/file_operation_hooks.py` is dead, pinned by
`Tests/Tools/test_system_a_is_retired.py:73,80`). *Closing the cheapest:* a post-tool emit in the dispatch path.

**Weaker secret redaction.** `MCP/redaction.py` matches on key *names* only and documents its own bypass at line 64
(a secret value starting with `-` survives). Hermes matches value shapes — JWTs, private-key blocks, DB connection
strings, bearer headers, provider key prefixes (`agent/redact.py:80-409`). Approval cards and the execution log are the
exposure surfaces. *Closing it:* add value-shape patterns to `redaction.py`; it is a pure function.

**Scope note on chatbook's audit trail.** The decision ledger is real but MCP-only — `mcp_execution_log.jsonl` is
written solely from `MCP/unified_control_plane_service.py:2261`. Raw-shell and local-builtin decisions are not
recorded, so "chatbook has an audit trail and hermes has none" is true only for MCP tools.

### 3.4 Context & memory

**Micro-compaction, proactive tool-result pruning, stale-image retirement.** Hermes folds one oldest exchange into a
rolling summary after each turn (`agent/turn_finalizer.py:416-440`), keeping the window flat instead of sawtoothing;
it separately runs a deterministic no-LLM prune of large old tool results with a min-reclaim gate so prompt-cache
breaks stay episodic (`agent/context_compressor.py:4232`); and it replaces image payloads in older tool results with
text placeholders, reclaiming ~1600 tokens each (`agent/context_compressor.py:1651`). Chatbook compacts in one batch at
the trigger ratio (`Chat/console_chat_controller.py:18054`) — a visible stall — only drops whole turn-groups
(`Chat/console_history_budget.py:266`), and charges `per_image_tokens` forever without ever stripping images.
*Closing them:* a post-turn hook calling the existing `plan_manual_range` over the single oldest unit; a tool-role
filter inside the existing bounding pass; image-part stripping in that same pass. All three sit on machinery that
already ships.

**Cache TTL tiers and non-Anthropic caching.** Hermes emits `ttl: 1h` per measured route and derives a
`prompt_cache_key` for OpenAI/Codex/xAI (`agent/prompt_caching.py:157`; `agent/transports/codex.py:692`). Chatbook
emits bare 5-minute ephemeral markers at three sites (`LLM_Calls/LLM_API_Calls.py:1466,1511,1544`) and has nothing for
OpenAI or Gemini. *Closing it:* 1 h is a one-key addition behind a config flag; `prompt_cache_key` is one kwarg.

**Context breakdown and context references.** Hermes's `/context` splits the window into 8 named categories plus a
100-cell glyph grid (`agent/context_breakdown.py:89`), and `@file:path:10-40`, `@folder`, `@diff`, `@staged`, `@git`,
`@url` expand inline before send with plugin-registered prefixes and autocomplete
(`agent/context_references.py:148,212`). Chatbook shows request-vs-conversation totals only
(`Widgets/Console/console_context_controls.py:105`) and has `$`-mentions for skills only. *Closing them:* extend
`PreparedConsoleRequest.accounting` to name system/tool/RAG buckets; add a composer-submit preprocessor reusing the
existing attachment reader (the autocomplete surface already exists).

**Persistent cross-session memory.** Hermes carries `MEMORY.md`/`USER.md`, a `MemoryProvider` ABC with 9 external
backends, a scheduled curator that ages/archives/consolidates learned skills, and a journey graph over skills and
memory cards. Chatbook's "memory" is strictly per-conversation compaction summaries
(`Chat/console_context_repository.py:136`); its only learning path is the human-gated `agent_lesson_promotion` into
AGENTS.md. *Closing it:* a whole subsystem, not a patch — the nearest cheap step is persisting promoted lessons to a
user-scoped store and injecting them at prompt build.

**Compaction timeouts, preview, focus, pluggable engine, provider-native compaction.** Hermes bounds the summarizer
with progress-aware inactivity budgets plus absolute ceilings, offers `/compress --preview` and `focus <topic>`, loads
a per-session `ContextEngine` plugin, and can delegate compaction to Codex/OpenAI server-side. Chatbook builds a
`ManualMemoryPlanResult` internally but exposes no preview or topic argument, has no timeout config, and hard-wires
one compaction service. *Closing the cheapest:* wrap the auxiliary call in `asyncio.wait_for`; both preview and topic
are thin (the plan object is already returned before commit).

### 3.5 Providers & auth

**OAuth / subscription auth.** Hermes ships real OAuth — PKCE S256 for Anthropic (`agent/anthropic_credentials.py:920-1013`),
device-code and external-browser flows for six more providers — and it *borrows* credentials other tools already
minted (`~/.claude/.credentials.json`, `~/.qwen/oauth_creds.json`, `gh auth token`), so a Claude Max subscriber pays
$0 marginal per token. Chatbook has zero LLM-provider OAuth; every hosted provider is API-key-only
(`Chat/provider_readiness.py:50-66`). Its only OAuth surfaces are Confluence scraping and a proxy to tldw_server for
*user-account* login. *Closing the highest-ROI slice:* not a full OAuth stack but a **credential-borrow reader** —
read `~/.claude/.credentials.json` and add the `Authorization: Bearer` + `anthropic-beta: oauth-*` header path in
`chat_with_anthropic`. Full PKCE with refresh rotation is a much larger second step.

**Credential pooling and cross-provider failover.** Hermes has a 3566-line pool with three selection strategies,
per-entry exhaustion TTL parsed from `Retry-After`/reset timestamps, leases, terminal-failure `STATUS_DEAD`, and an
ordered `fallback_providers` chain that also triggers on 402 credit exhaustion. Chatbook has one key per provider
(`config.py:7423-7478`); a 429 retries the same key with backoff and then surfaces `ChatRateLimitError`
(`Chat/Chat_Functions.py:1180-1184`), and there is no error-driven provider switching. *Closing them:* accept a list
for `api_settings.<provider>.api_key` with a per-key `exhausted_until`; and a `[chat] fallback_providers` list looped
around the existing `handler(...)` call at `Chat/Chat_Functions.py:1037`.

**Model metadata and pricing catalog.** Hermes pulls models.dev (4000+ models, 109+ providers) with
ETag/disk-cache/stale-while-revalidate and gets context windows, capabilities, and $/M-token for free
(`agent/models_dev.py:1-40`). Chatbook hand-maintains a regex capability table (`model_capabilities.py:27-45`) and a
$/M price table with a `_SEED_AS_OF` staleness stamp it must re-verify by hand (`LLM_Calls/pricing_catalog.py:41-48`).
*Closing it:* chatbook already has the disk-cache and merge machinery in `LLM_Provider_Catalog/`; point one more source
at `https://models.dev/api.json` and fold it in as a lower-priority merge layer. The plumbing is the expensive part and
it already exists.

**Auxiliary model routing and rate-limit telemetry.** Hermes routes side tasks (compression, titles, vision, search) to
a cheap model with three-tier resolution and a free-only cost guard (`agent/auxiliary_client.py:1025-1109`), and parses
12 `x-ratelimit-*` headers plus 14 `x-nous-credits-*` into usage windows. Chatbook sends auxiliary requests through the
same dispatch table at the same model — the auxiliary set is *audited* but not *routed*
(`Chat/Chat_Functions.py:186-217`) — and captures no rate-limit headers. *Closing them:* one `[chat] aux_model` config
key consulted at the auxiliary call sites; and capturing response headers already in hand at
`LLM_Calls/hosted_chat.py:847`.

### 3.6 Scheduling / cron / automations

This area had a prior audit (TASK-18936). Below is only what a code-level pass adds.

**Corrections to TASK-18936.**

1. *"Storage durability + audit — PARITY OR BETTER locally … hermes's durable execution-audit history is matched in
   concept"* — **wrong.** Chatbook's `automation_audit_events` covers definition CRUD only
   (`db/migrations/v0_to_v1.py:114`); `mark_reminder_dispatched` overwrites one `last_status`/`last_run_at` pair
   (`db/scheduled_tasks_db.py:772-830`), so run *N−1* is unrecoverable. Hermes has a real ledger
   (`cron/executions.py:46`) plus incidents (`cron/incidents.py:91`). **Scope correction to that correction:**
   watchlists *do* have per-run rows (`DB/Subscriptions_DB.py:936-950`), so the gap is reminders and briefings only.
2. *"Missed-fire — GAP … no code path populates `missed_at`"* — **stale.** `db/scheduled_tasks_db.py:830-849` writes
   both `missed_at` and `missed_count`, with `_count_missed_occurrences` at `:892` and an overflow cap at `:888`; UI at
   `task_detail.py:696-726`. Task-18937 shipped, and chatbook now *exceeds* hermes on lateness-cause attribution
   (`loop.py:444-521`).
3. *"Run now — GAP. No run-now action on any task"* — **stale.** `loop.py:557-608`, `services/scheduling_service.py:331`,
   `schedules_workbench.py:127,807`, including a dequeue-before-dispatch double-fire guard (`queue.py:116`).
4. *"Execution-timeout configurability — GAP"* — **stale and now inverted.** Chatbook has a per-task wall-clock
   timeout (`models.py:103`; `loop.py:533-555`; `migrations/v2_to_v3.py:45`); hermes has no per-job equivalent, only
   script/media/session sub-timeouts (`cron/scheduler.py:3973,4009,4044`). This row is now CHATBOOK AHEAD.
5. *"`agent_task` automations do not execute … NET GAP in practice"* — **stale.** The ADR-077 client seam shipped:
   server-scoped rows are filtered out of the local queue (`queue.py:77-86`), Run-now honestly refuses on them
   (`loop.py:583-591`), and an Automations tab lists definitions, shows the server audit trail, and dispatches server
   run-now (`schedules_workbench.py:940,1043,1103`). TASK-18940's **client half is done**; what remains is server-side.
6. *"Per-job model picker — GAP (moot)"* — **stale.** ADR-077 AC#7 rendering exists at
   `schedules_workbench.py:83-110` (`provider/model`, or `auto` when unpinned).
7. *"chatbook has the richer underlying definition model … ahead on modeling"* — **half wrong.** Chatbook's
   previews/policies are richer for *validation*; hermes's blueprints are a *template catalog* with typed slots
   rendering to a form, a slash command, and a deeplink from one schema
   (`cron/blueprint_catalog.py:120,578,602,623,747`). Different axes; chatbook has nothing on the template axis.
8. **The audit's capability list was incomplete.** Release notes did not surface: incidents with signature dedupe and
   ack, the per-job notepad, the cron suggestions inbox, job chaining, repeat limits, ticker heartbeat/liveness,
   preflight checks, output retention, global ESTOP, and the `cronjob` agent tool.

**New gaps.** *Durable per-run execution history* for reminders and briefings (`cron/executions.py:46-241` vs. a single
overwritten status row) — closing it is one `scheduled_task_runs` table (v3→v4), an insert in `dispatch_reminder`, and
a history list in `task_detail`. *Failure incidents with dedupe and ack* (`cron/incidents.py:133-262`) — chatbook
re-notifies on every failed dispatch; closing it is an error-signature hash plus an ack flag on that same run table.
*Scheduler liveness* (`cron/jobs.py:1465-1608`) — chatbook cannot distinguish a dead loop from an idle one; closing it
is stamping `last_tick_at` per tick. *Preflight validation* (`cron/scheduler.py:5127-5400`) — a per-handler optional
`preflight(task)` in `_dispatch_due`. *Natural-language schedule input* (`cron/jobs.py:941-1090`) — already filed as
task-23102/23103. *Global ESTOP* (`cron/scheduler.py:7888`) — one boolean in `_dispatch_due` plus a workbench binding,
genuinely ~10 lines. *Job chaining, repeat limits, pause-with-reason, run-now extra prompt* — each is a column plus a
UI field; none is architectural. *Cron CRUD as an agent tool* (`tools/cronjob_tools.py:1460`) — chatbook's scheduling
is UI/service-only; given ADR-077 decision 4, treat this as a decision to record rather than an accident.

### 3.7 Interop

**ACP, both directions.** Hermes is a full ACP agent — `initialize`/`authenticate`, session new/load/resume,
history replay as `session/update` notifications, advertised slash commands, `set_session_mode`/`set_session_model`,
and permission plus edit-diff bridges (`acp_adapter/server.py:1626,2224,2570,2604`; `acp_adapter/permissions.py:22`) —
*and* it can consume an ACP agent as a chat backend (`agent/copilot_acp_client.py:1`). Chatbook's `ACP_Interop` is a
subprocess supervisor: `start_session` shells out with `Popen` and records a status enum
(`ACP_Interop/runtime_process.py:185,203`), and the screen renders "Diffs unavailable"
(`UI/Screens/acp_screen.py:277`). *Closing it:* implement the ACP JSON-RPC surface over stdio — schema, session
registry, update streaming — roughly the 8 k lines hermes carries.

**MCP client transports and OAuth — the single biggest interop gap.** Hermes reaches remote MCP servers over
Streamable HTTP and SSE (`tools/mcp_tool.py:349,384`) with a full OAuth 2.1/PKCE/DCR token stack
(`tools/mcp_oauth.py:5,457`) and interactive login flows for CLI and TUI. Chatbook connects only by spawning a local
process (`MCP/client.py:815,851`); `MCP/local_store.py` has `command`/`args`/`env` and no URL field, and the only
`transport` values in the package are `"stdio"` and `"in_process"`. The entire hosted-MCP ecosystem — Linear, Sentry,
Notion, Stripe remote endpoints, and hermes's own 50+ vendor catalog — is unreachable. *Closing it:* adopt the official
`mcp` Python SDK's `streamablehttp_client`/`sse_client` instead of the hand-rolled stdio JSON-RPC, then layer token
storage.

**MCP sampling and elicitation.** Hermes answers server-initiated `sampling/createMessage` with rate limits and
tool-use support, and routes `elicitation/create` to whichever surface owns the session
(`tools/mcp_tool.py:1771,2203,2299`). Chatbook replies `-32601` to every server-initiated method except `ping`
(`MCP/client.py:744,754`). *Closing it:* two callback handlers in `_handle_server_request`, reusing the existing
approval-card path for elicitation and the existing chat provider for sampling.

**OpenAI-compatible API server.** Hermes exposes the agent as `/v1/chat/completions`, `/v1/responses`, and a
`/v1/runs` lifecycle with SSE events, steer, and stop (`gateway/platforms/api_server.py:5-22,2219-2247`), so Codex,
Aider, or Cline can drive it. Chatbook exposes no HTTP API; its only server is textual-serve rendering the TUI in a
browser (`Web_Server/serve.py:232`). Even for a single-user local-first tool, "point my editor at my own agent" is a
real local analogue chatbook lacks. *Closing it:* one small ASGI app over the existing `agent_service` run loop; the
`/v1/runs` half is optional.

**Webhooks and A2A.** Hermes accepts signed inbound webhooks that start agent runs
(`gateway/platforms/webhook.py:1-33`) and emits HMAC-signed outbound lifecycle notifications
(`agent/outbound_webhooks.py:1-25`); it also serves an A2A v1.0 agent card and JSON-RPC task surface
(`plugins/platforms/a2a/adapter.py:1-17`). Chatbook has neither; `FleetCoordinator` manages sub-agents inside one
process with no wire protocol (`Agents/fleet_coordinator.py:148`). Inbound webhooks are arguably gateway-shaped, but
**outbound** ("tell my dashboard when a run finishes") is squarely local-first. *Closing outbound:* an HMAC POST
callback on the existing run-log hook points.

**Connectors, browser extension, local IPC.** Hermes ships local tools for Home Assistant, Spotify, Google Meet, and
the Teams pipeline; routes `browser_*` to an attached extension with bound-identity fail-closed dispatch
(`tools/browser_extension_router.py:1-30`); and exposes a versioned JSON control socket for sibling processes
(`gateway/control_socket.py:1-18`). Chatbook has none of these locally — its connectors and clipper are API clients to
a remote tldw_server (`External_Connectors_Interop/server_connectors_service.py:96`;
`Web_Clipper_Interop/server_web_clipper_service.py:94`). *Closing any one connector:* a tool descriptor plus an HTTP
client in `Tools/` — per-integration work, not architectural.

**MCP resources and prompts reaching the model.** Chatbook can read resources and prompts (`MCP/client.py:1044,1076`)
but `MCPToolProvider` composes a *tool* catalog only (`Agents/mcp_tool_provider.py:307`). Hermes synthesizes
`list_resources`/`read_resource`/`list_prompts`/`get_prompt` as callable tools (`tools/mcp_tool.py:6905-6944`).
*Closing it:* four synthetic tool registrations over the existing client methods.

### 3.8 User surfaces

**Slash-command breadth and `/help`.** Hermes routes ~101 typed commands through one registry that simultaneously
feeds CLI help, gateway dispatch, Telegram BotCommands, the Slack manifest, and the completer
(`hermes_cli/commands.py:149-416`), with `/help`, `/help <filter>`, and a paginated `/commands` browser. Chatbook
registers 10 (`Chat/console_command_grammar.py:194-272`) and puts the rest behind keybindings and screens, so anything
not on a key has no keyboard-discoverable name — and there is no `/help` at all. *Closing it:* add `ConsoleCommand`
entries dispatching to the `action_*` methods that already exist (`UI/console_command_provider.py:33-97`); one `/help`
rendering `registry.available_names()` + `_COMMAND_DESCRIPTIONS`, both already present at
`Chat/console_command_suggestions.py:34`. The grammar and suggestion layers need no change.

**i18n and RTL.** Hermes ships 17 locale YAMLs and a `t(key, **kwargs)` runtime resolved from
`HERMES_LANGUAGE`/`display.language` (`agent/i18n.py:43,91,232`), plus automatic RTL/bidi direction. Chatbook has
literally zero localization machinery — the named grep over the whole package returns only a `.babelrc` filename
constant and a git config flag. *Closing it:* a whole-codebase string-extraction project across 60+ screens, not a
bolt-on; realistically scope it to the Console screen first with stdlib `gettext`. RTL additionally needs
`python-bidi` at the transcript render seam — Rich/Textual has no bidi engine.

**Doctor / diagnostics UX.** `hermes doctor` runs ~30 static checks with a `--fix` repair path and an opt-in `--live`
real-call probe of every configured backend (`hermes_cli/doctor.py:641-1236`; `doctor_live.py:1-18`). Chatbook has a
Settings pane that parses and reloads the TOML (`UI/Screens/settings_screen.py:9030-9042`) and a speech-deps status
pane. *Closing it:* a `doctor` surface aggregating what already exists — `Utils/optional_deps.py` `DEPENDENCIES_AVAILABLE`,
`DB/base_db.py:257` `check_integrity()`, `config.py:5079` `get_config_load_failure()`, `config.py:7262`
`get_detected_api_providers()`, plus `settings_endpoint_probe.py` for provider reachability.

**Wake word, kanban UI, prompt stash, OS notifications.** Hermes ships three on-device hotword providers behind
`/wake` (`hermes_cli/config_defaults.py:1975-2003`), a 3542-line multi-board kanban CLI plus a web dashboard, a Ctrl+S
stash panel with delete/restore, and native OS notifications with per-type toggles. Chatbook's voice stack is
push-to-talk/hands-free only (wake grep = clean zero); `Kanban_Interop` has services and **zero UI** (named grep over
`UI/` + `Widgets/` = 0); drafts are one-per-session (`UI/Console_Modules/session.py:3835-3901`); notifications are
in-TUI toasts only. These are tasks 18933, 18934, 18930, 18925 respectively — all still To Do, all confirmed still open
in code.

**Filesystem checkpoints and the embedded terminal.** Hermes snapshots the working tree per turn and offers
`/rollback`, `/diff session`, and a `hermes checkpoints` prune/clear CLI; `!bang` runs a shell inline through the same
approval gate at zero tokens. Chatbook can revert a file from change-review but has no time-indexed session baseline,
and its `Terminal/` package is fully built and **unwired to any UI** (grep over `UI/` + `Widgets/` returns one
unrelated comment). *Closing the terminal one:* this is wiring, not building — mount a widget over
`Terminal/session_manager.py`.

**Minor.** CJK wide-character markdown table re-padding (`agent/markdown_tables.py:1-22`) has no chatbook equivalent
for model output; `wcwidth` is used only in `UI/character_display_text.py` and `Terminal/screen_model.py`. And
`register_fallback_resolver` (`Chat/console_command_grammar.py:144`) has no production call site —
`Chat/console_skill_resolver.py:19-21` documents that the factory was deleted — so bare `/skill-name` does not
resolve; only `$name` and `/skills <name>` do.

### 3.9 Ops

**Config layering, migrations, and corrupt-config resilience.** Hermes merges defaults → user → `${VAR}` expansion →
a root-owned `/etc/hermes/config.yaml` overlay winning per-leaf (`hermes_cli/config.py:3792-3900`), runs 20 numbered
config-schema migrations behind a version check (`hermes_cli/config_migrations.py:869`), and on a parse failure copies
the corrupt file aside while continuing to serve last-known-good config — explicitly so security-critical deny rules
survive a mid-edit break (`hermes_cli/config.py:3862-3893`). Chatbook merges defaults + one user TOML with ~30 ad-hoc
`os.getenv` fallbacks, has no config schema version at all (so key renames leave orphans forever), and drops to bare
internal defaults on a `TOMLDecodeError` — silently reverting encryption settings, DB path overrides, and provider
config (`config.py:5165-5178`). *Closing the highest-value one:* keep the last good dict in a module global and return
it instead of defaults in the decode-error branch — about five lines.

**Hot reload of external edits.** Hermes keys its config cache on (mtime, size) of both files plus an env snapshot, so
an edit made outside the app is picked up on the next read (`hermes_cli/config.py:3798-3838`). Chatbook's cache is
keyed on path only (`config.py:5101-5107`); an external edit is invisible until the user clicks Settings → Reload
config. *Closing it:* stat the file and compare `(st_mtime_ns, st_size)` in the cache-hit branch.

**Doctor, self-repair, self-update, whole-state backup, crash forensics.** Hermes has a stdlib-only pre-import venv
repair pass (`hermes_cli/_early_recovery.py:1-23`), an update pipeline with cross-process locks, machine-readable
receipts, pre-update snapshots, and an image-managed refusal contract, a `hermes backup`/`import` pair that zips all of
`~/.hermes/`, and `faulthandler` enabled to a dedicated log with all-threads dumps plus a lifecycle ledger recording
unclean deaths (`gateway/run.py:13159-13180`). Chatbook has none of these: per-DB `backup_database()` calls exist
(`DB/ChaChaNotes_DB.py:3508` and two siblings) but nothing rolls them up, and the named grep for
`excepthook|faulthandler` returns three comments. *Closing the cheapest:*
`faulthandler.enable(file=<log dir>/faulthandler.log, all_threads=True)` in `configure_application_logging`
(`Logging_Config.py:459`) — one line, and the private log directory resolution already exists at `config.py:7849`.

**Traces and packaging.** Hermes exports OTLP spans (`agent/monitoring/otlp_exporter.py:118-182`); chatbook's OTel
integration is metrics-only through a `PrometheusMetricReader` with no `TracerProvider`. Hermes packages for curl
installer, Docker+s6, Nix/NixOS/home-manager, npm, and Termux with install-method detection driving the right update
command; chatbook covers PyPI plus real signed native desktop installers hermes lacks. *Closing the one genuinely
missing target for a terminal app:* a Dockerfile.

**Footnote — an unauthenticated listener.** `app.py:16900` calls `init_metrics_server()` unconditionally at boot; the
only gate is dependency presence (`Metrics/metrics.py:255`). `prometheus_client` ships in the `[dev]` and `[debugging]`
extras (`pyproject.toml:312,315`), so a user who installs either gets `start_http_server(8000)` bound with no config
opt-in and no auth. Verify with `ss -ltnp | grep 8000` after launching a `[debugging]` install.

---

## 4. Top gaps, ranked

Ranked for a **local-first TUI** — weighted toward what a single user running their own stack on their own machine
actually hits. `NEW` means not covered by tasks 18920–18940 or ADR-077.

| # | Gap | Why this rank | Covered? |
|---|---|---|---|
| 1 | **MCP client is stdio-only** — no HTTP/SSE transport, no OAuth (`MCP/client.py:815,851`; `MCP/local_store.py` has no URL field) | MCP is chatbook's declared extension mechanism, and half the ecosystem lives behind remote endpoints. This is the only gap that shrinks *what the user can connect to* rather than how nicely it works — and hermes just expanded to 50+ vendor-hosted servers. | **NEW** |
| 2 | **No in-loop retry or model fallback** — first `call_model` exception ends the run (`Agents/agent_runtime.py:1291-1302`) | Local-first users run flaky local servers and hit 429s constantly. A single transient error currently discards a long agent run and every tool result in it. Highest damage-per-occurrence on the list. | **NEW** |
| 3 | **`fs_edit`/`fs_patch` are exact-match only** (`Tools/local_tool_impls.py:839-847`; `Tools/patch_tool_impls.py:211,407`) | The highest-frequency turn-waster in any agent loop: a whitespace-off `old_string` costs a read-retry cycle *and* an approval cycle, every time. Hermes ships nine recovery strategies. | task-18927 (To Do) |
| 4 | **No mid-run steering or active-turn redirect for the primary agent** (`Agents/agent_runtime.py:1196-1230`; `Chat/console_chat_controller.py:13048-13126`) | This is the core TUI interaction. Today a correction means killing the run and losing every completed tool result. The drain point, formatter, and steering bar all exist and are wired for children only. | **NEW** |
| 5 | **Tool output is hard-truncated at 32 KiB with no spill** (`Agents/local_tool_provider.py:320-324`) | Silent, unrecoverable data loss *inside* the agent loop — the model cannot re-read what was cut, so it re-runs the tool or guesses. Also no per-turn aggregate budget. Cheap to fix at one seam. | task-18927 (partial) |
| 6 | **No shell-command risk floor** — `validate_raw_cli_request` checks caller/shell/size/timeout/cwd, never what the command does (`Tools/raw_cli_executor.py:144`) | `rm -rf ~` is an ordinary approval card, and `approve_session` (`Agents/raw_shell_tool_provider.py:48`) then runs everything unreviewed for the session. Hermes's hardline list is explicitly unbypassable even under `--yolo`. Safety-shaped, and the detector is portable. | **NEW** |
| 7 | **No doctor / health-check surface** (named grep for `doctor\|healthcheck\|self_test` over `tldw_chatbook/` → nothing user-facing) | Local-first means the user owns the whole stack — local servers, optional extras, keys, DB integrity — and has no way to ask "what's broken?". Every ingredient already exists (`optional_deps`, `check_integrity`, `get_config_load_failure`, endpoint probes); only the aggregation is missing. | **NEW** |
| 8 | **No cross-session persistent memory** (`Chat/console_context_repository.py:136` — memory is per-conversation compaction summaries) | Hermes's `MEMORY.md`/`USER.md` + curator + learning graph is the difference between an assistant that accumulates knowledge of you and one that starts fresh every conversation. Largest build on this list, which is why it is not higher. | **NEW** |
| 9 | **10 slash commands vs 101, and no `/help`** (`Chat/console_command_grammar.py:194-272` vs `hermes_cli/commands.py:149-416`) | Keyboard-only discoverability. Most capabilities *exist* as screens or keybindings — the gap is that they have no typed name, and nothing enumerates what is available. `/help` alone is a few lines over data that already exists. | task-18921 (partial — ranking only) |
| 10 | **Compaction is one visible stall** — no micro-compaction, no tool-result pruning, no stale-image retirement (`Chat/console_chat_controller.py:18054`; `Chat/console_history_budget.py:266,184`) | Chatbook's compaction machinery is genuinely good (better than hermes on mode, provenance, and honesty), but it sawtooths where hermes amortizes, and never reclaims the cheapest tokens — old tool results and images. All three sit on shipped machinery. | **NEW** |

**Considered and ranked below the line, with why:**

- **OAuth / subscription auth** (`Chat/provider_readiness.py:50-66` — API-key-only) — high user value, but the cheap
  slice (read `~/.claude/.credentials.json`) is narrow and the full stack is large. Would be #4 if scored on user
  money saved rather than local-first friction.
- **Credential pooling** — matters less for one user with one key than for a hosted fleet.
- **Sandboxing** (no container/OS isolation anywhere) — the largest single build on the list, and chatbook's path jail
  plus approval gate covers the common case; it becomes urgent only if unattended execution ships.
- **Per-run execution history for reminders/briefings** (`db/scheduled_tasks_db.py:772-830`) — real, but scoped: the
  watchlists path already has one (`DB/Subscriptions_DB.py:936-950`).
- **Config schema migrations and corrupt-config resilience** (`config.py:5165-5178`) — the last-known-good fallback is
  ~5 lines and worth doing, but the failure is rare.
- **Models.dev metadata/pricing feed** — removes hand-maintenance of `_SEED_AS_OF`, but the existing tables are
  correct today.
- **i18n / RTL** — a genuine total absence, but it is a 60-screen extraction project with no current user demand
  recorded anywhere in the repo.

---

## 5. Where chatbook is ahead

Not a courtesy list — every entry is a capability hermes lacks, verified by a named grep over named hermes
directories in addition to the chatbook citation.

### Safety and permissions

- **Kill switch** — one global stop consulted before any approval stamp or session grant.
  `MCP/permission_store.py:363,373`; `Agents/builtin_tool_gate.py:340`. (`grep -rn "kill_switch"` over hermes → 0.)
- **Rug-pull guard** — an explicit `allow` reverts to `ask` when a tool's description/schema hash changes.
  `MCP/permission_store.py:604,884-892`. Hermes has no tool-definition hashing at all.
- **High-risk floor on inherited allow** — inherited `allow` downgrades to `ask` when risk tags intersect
  `{mutates, process}`. `MCP/permission_store.py:80,97,912-918`.
- **Real decision audit trail** (MCP-scoped) — bounded metadata-only JSONL recording allowed/approved/denied/downgraded.
  `MCP/execution_log.py:1,27`; `MCP/unified_control_plane_service.py:3054`. Hermes explicitly has none — its own
  source says so at `hermes_cli/approvals_suggest.py:3`.
- **Per-run approval stamps** keyed by `run_id`, so one run's approval cannot permit a sibling sub-agent's call.
  `Agents/builtin_tool_gate.py:177,195,357`.
- **Fail-closed synthetic `gate_error` verdict** when permission resolution itself raises — rendered "Unknown", never
  "Off". `MCP/permission_store.py:756,777`.
- **Distinct model-facing refusal provenance** — deny / unresolved / timeout / kill-switch / Off are never conflated.
  `Agents/mcp_tool_provider.py:78-95`.
- **TOCTOU-hardened path jail** — fd/handle-pinned root with an ancestor-identity chain, roots re-validated against
  symlink and mount swap on every call. `Tools/workspace_root_pin.py:99,135,185`; `Tools/workspace_file_roots.py:369`.
- **Read-only vs read-write folder bindings** as first-class permission grain. `Tools/workspace_file_roots.py:410`.
- **Credential denial extended to git output** via computed `:(exclude)` pathspecs — closes reading `~/.ssh/id_rsa`
  out of history from a clean tree. `Utils/sensitive_paths.py:895`; `Tools/git_tool_impls.py:393`.
- **Shell-free structured CLI** — 10 allowlisted read-only commands parsed by argparse; argv never touches a host
  shell. `Tools/virtual_cli_impls.py:20,44`; `Agents/virtual_cli_provider.py:225`.
- **Per-run tool-call caps** at a single choke point, thread-safe and narrowing-only.
  `Agents/run_tool_policy.py:1-16,27,39`. (`grep -rn "max_calls_per_turn\|call_cap"` over hermes → 0.)
- **App-wide SSRF egress policy** — every resolved IP must be public and non-metadata unless user-seeded trusted;
  shared pipeline code is forbidden from auto-trusting its own input URL. `Utils/egress.py:1-11`.

### Skills

- **Sandboxed skill-script execution** — `run_skill_script` over a genuinely hardened runner: setrlimit trampoline
  instead of `preexec_fn` (fork-deadlock safe in a threaded process), bounded per-stream reader threads instead of
  `communicate()` (no OOM on a spewing script), process-group SIGKILL teardown, 600 s ceiling.
  `Agents/tool_catalog.py:377`; `Skills_Interop/skill_script_runner.py:1-31`. Hermes runs skill code through the
  general `terminal` tool.
- **Trust-gated install with mandatory human review** — an installed skill is inert until approved in Library ▸ Skills,
  backed by a deterministic sha256 directory snapshot and revocable script grants.
  `Agents/tool_catalog.py:282-300`; `Skills_Interop/skill_trust_scanner.py:34`. Hermes's `skills_guard` is regex
  matching with an auto-allow tier for `trusted` sources (`tools/skills_guard.py:11-14`).
- **`prepare_managed_skill_promotion`** — turns verified lesson evidence into a reviewable skill-update proposal with
  `expected_version` + `expected_trust_state` + `current_sha256` optimistic-concurrency fields; never writes, never
  re-trusts. `Agents/tool_catalog.py:309-370`. No hermes equivalent.

### Agent loop and runtime

- **`StreamGate`** — incremental fence classification that resumes past look-alikes and guarantees streamed text is
  always an exact prefix of the final visible text. `Agents/agent_stream.py:52-176`.
- **`STEP_APPROVAL_REVOKED`** audit event when a cancel lands after approval but before dispatch.
  `Agents/agent_runtime.py:1636-1651`.
- **Incremental per-step SQLite persistence** with a `capture_failed` diagnostic step if the write itself fails.
  `Agents/agent_service.py:5769-5789`; `DB/AgentRuns_DB.py:1539`. Hermes persists per turn.
- **Crash sweep** that deterministically reconciles orphaned `running` rows, guarded against double-sweep and
  registered only post-commit. `DB/AgentRuns_DB.py:1689-1745`.
- **Durable provider-continuation checkpoints** that resume a turn interrupted mid-tool-call, with bounded parsing and
  an inline recovery callout. `Chat/provider_continuation.py:1-46`; `UI/Console_Modules/provider_continuation_recovery.py:1`.
- **Whole-run-tree segmented run log** plus `search_run_log` / `run_log_stats` / `run_log_slice` runtime tools and
  eviction. `Agents/run_log.py:1-19`; `Agents/agent_runtime.py:440-460`; `Agents/run_log_eviction.py:1`. Hermes's live
  log is per-delegated-child only.
- **Four independent budget dimensions** (steps, model turns, wall, tokens) with shipped defaults sized for real work
  and re-read per run. `Agents/agent_runtime.py:1169-1179`; `Chat/console_agent_bridge.py:452,455,459`.
- **Prompt-queue management surface** — edit, move, remove, clear queued prompts, plus a pause state machine with
  typed reasons. `Chat/console_prompt_queue.py:60-76`; `Chat/console_chat_controller.py:4279,4306,4326,4340`.
- **Change-window taxonomy** distinguishing a turn's own writes from concurrent-subagent and post-turn-survivor
  writes, so a diff never implies authorship it cannot prove. `DB/AgentRuns_DB.py:335-345`.
- **Causal step lineage** — `parent_event_id` / `source_event_id` / `next_owner_seq` threaded through steering and
  dispatch. `Agents/agent_runtime.py:959,1219-1224,548`.

### Context and compaction

- **Ask / automatic / off compaction mode** with an approval gate that blocks the send until the user consents.
  `Chat/console_context_policy.py:29`; `Chat/console_chat_controller.py:18072`.
- **Visual-transcript compaction** — render conversation history to image pages for a vision model, with
  `TEXT_SUMMARY`/`VISUAL_TRANSCRIPT`/`HYBRID` representations. `Chat/console_visual_transcript.py:346,463`.
  (Named grep `visual transcript\|render.*transcript.*image` over hermes `agent/` → 0.)
- **Branch/fork-aware compaction memory** — memory records fenced to conversation lineage so a fork inherits the right
  summary. `Chat/console_context_repository.py:285,294,324,371`. (Named grep `branch_memory\|fork.*compaction` over
  hermes `agent/` → 0.)
- **Compaction provenance and admission** — prefix digests, admission matching, per-attempt pricing provenance,
  stale/cancelled terminals. `Chat/console_context_compaction.py:258,304,375,2115`.
- **Arbitrary-range manual compaction**, not just prefix. `Chat/console_context_compaction.py:1047`.
- **Model-window honesty surface** — `model_window_verified`, `token_limit_source`, an `UNKNOWN_WINDOW` decision, and
  an "estimated input; model unverified" label rather than silently trusting a guess.
  `Widgets/Console/console_context_controls.py:110-116`.
- **User-selectable compaction failure behavior** in settings. `Chat/console_context_policy.py:42`;
  `UI/Screens/settings_screen.py:14145`.
- **Anthropic cache degrade-retry** — on a 400 mentioning `cache_control`, retry once with all breakpoints stripped
  instead of failing the turn. `LLM_Calls/LLM_API_Calls.py:1597-1615`.
- **Three-tier context policy precedence** (application → global → per-conversation sparse overrides) with hysteresis
  validation. `Chat/console_context_policy.py:246,279,398`.
- **First-class RAG-into-context** with plain/full/hybrid pipelines and persisted per-conversation scope.
  `Event_Handlers/Chat_Events/chat_rag_events.py:177,225,279,2051`; `Chat/rag_scope.py:435`.

### Providers, local inference, media

- **Local server process management** — starts and stops MLX-LM via `Popen`
  (`Local_Inference/mlx_lm_inference_local.py:22-177`); hermes only *talks to* local servers.
- **Ollama model administration** — pull/push/copy/delete/create/list/ps/embeddings.
  `Local_Inference/ollama_model_mgmt.py:261-386`.
- **Local server auto-discovery** across ollama/llamacpp/vllm with bounded reads and an `/api/tags` fallback.
  `Chat/local_server_discovery.py:201-232,455-471`.
- **Local runtime breadth** — koboldcpp, oobabooga, tabbyapi, aphrodite have no hermes equivalent.
  `Chat/Chat_Functions.py:130-152`.
- **Embedding providers** — HF local sentence-transformers + OpenAI-compatible with retry.
  `Embeddings/Embeddings_Lib.py:236-286,513-575`. Hermes has none (named grep over `agent/`, `hermes_cli/` → only
  models.dev noise filters).
- **STT depth** — a sealed capability registry validating declared vs. runtime-observed capabilities
  (`STT/registry.py:1-66`), a process-tree executor, parakeet ONNX, whisper.cpp, and five engines in
  `Local_Ingestion/transcription_service.py:3347-3424`.
- **Image-gen adapter breadth** — 10 adapters including self-hosted ComfyUI/SwarmUI/stable-diffusion.cpp
  (`Image_Generation/adapters/`); hermes has 7, all hosted.
- **Honest cost-unavailable UX** — refuses to fabricate a dollar figure when pricing is unknown.
  `LLM_Calls/pricing_catalog.py:11-13`; `Widgets/Console/console_composer_bar.py:542,1917`.
- **Discovery cache layering** — separate memory/disk/merge/persistence modules with provider-identity normalization.
  `LLM_Provider_Catalog/model_discovery_merge.py`, `model_discovery_persistence.py`.

### Tools

- **Git tool suite as first-class tools** — `git_status`/`git_diff`/`git_log`/`git_blame`/`git_branches`.
  `Agents/local_tool_provider.py:2328,2351,2392,2420,2453`. Hermes reaches git through `terminal`.
- **Library and RAG tool providers** with Console/MCP-identical payload and error shapes, capped excerpts, scrubbed
  provenance, plus 14 `watchlists_*` tools. `Agents/library_tool_provider.py:1-13`;
  `Agents/library_rag_tool_provider.py:1-15`; `Agents/local_tool_provider.py:2589-3092`.
- **Path-aware provider protocol** — `PathAwareToolProvider.path_targets` + `redact_root_locator` let the permission
  gate reason about which paths a call will touch *before* dispatch. `Agents/tool_catalog.py:612,759,955`.
- **Pinned-root workspace worker** — filesystem ops run in a one-shot subprocess with the root pinned by
  directory-chain identity, so a TOCTOU root swap cannot redirect a write. `Tools/workspace_tool_worker.py:1,29`.

### Scheduling

- **Per-task IANA timezone** on recurring tasks. `Scheduling/models.py:104-116`; `db/scheduled_tasks_db.py:856-864`.
  Hermes has one global timezone (`cron/jobs.py:1104-1119`).
- **Lateness cause attribution** with falsifiable evidence (`away`/`busy`/`stalled`), metric-labelled.
  `Scheduling/scheduler/loop.py:444-521`. No hermes equivalent.
- **Per-task execution timeout** with a distinct `timed_out` terminal status and schedule-still-advances semantics.
  `Scheduling/scheduler/loop.py:365-451,533-555`. Hermes has no per-job wall clock.
- **Exact skipped-occurrence count with an explicit overflow sentinel** rather than silent truncation.
  `db/scheduled_tasks_db.py:888-931`; rendered honestly at `UI/Screens/scheduling/task_detail.py:702-720`.
- **Versioned SQLite schema with forward migrations** vs. hermes's `jobs.json` plus repair heuristics.
  `db/migrations/v0_to_v1.py`, `v1_to_v2.py:54`, `v2_to_v3.py:45`.
- **Offline sync** with mappings, tombstones, pending mutations, server-wins conflicts, and a Conflicts tab.
  `db/migrations/v0_to_v1.py:132-190`; `services/sync_engine.py`; `UI/Screens/scheduling/conflicts_tab.py`.
- **Single-owner execution contract** enforced at the queue seam so client and server cannot double-fire.
  `Scheduling/scheduler/queue.py:23,77-86`; `loop.py:583-591`.
- **Manual-run double-fire guard** (dequeue before dispatch, reload after). `Scheduling/scheduler/queue.py:116-130`.
- **Config-junk tolerance** — TOML values coerced to documented defaults instead of mis-classifying every dispatch.
  `Scheduling/constants.py:29-64`.

### Interop

- **Much larger own-MCP-server surface** — 9 built-ins plus opt-in `fs_*`/`git_*`/`web_*`/`watchlists_*` and 24
  `library_*` tools, each routed through the same permission gate as the Console. `MCP/server.py:11,518`;
  `MCP/local_server_tools.py:130,239`.
- **MCP resources and prompts served to external clients** — conversation/note/character/media/rag-chunk URIs and 5
  prompt templates. `MCP/resources.py:29,108,174,240`; `MCP/prompts.py:27,105,163,238`. Hermes's own MCP server
  exposes tools only (`mcp_serve.py:11`).
- **MCP Hub readiness model** — priority-ordered reason codes and allowed-action sets driving the UI.
  `MCP/readiness.py:1,22,36`.
- **Secret-shaped value rejection in stored MCP env** — refuses to persist `sk-`/`ghp_`/`xox`/JWT-shaped literals,
  forcing `$ENV` placeholders. `MCP/local_store.py:42-50`.
- **Encrypted local-first device sync** — outbox producers, envelope build/apply, crypto plus key recovery, conflict
  review, per-domain adapters. `Sync_Interop/local_first_sync_service.py:1`; `crypto.py:1`; `envelope_applier.py:1`;
  `conflict_review.py:1`. (Named grep `outbox\|conflict_review\|envelope_applier` over hermes `gateway/`, `agent/`,
  `tui_gateway/` → 0.)
- **Local GitHub API client** for repo copy-paste, skill fetch, and issue/PR subscription scraping.
  `Utils/github_api_client.py:1,46`; `Subscriptions/scrapers/github_scraper.py:56,170`.

### User surfaces

- **58 registered Textual themes** vs. hermes's 9 CLI skins. `css/Themes/themes.py:1425-1484`; `app.py:13206-13207`.
- **Interactive change-review screen** — per-file diff pane, j/k navigation, `u` revert file, `U` undo all, `C`
  comment, `g` commit, `p` push, `P` open PR. `UI/Screens/change_review_screen.py:1339-1366`. Hermes prints a
  400-line-capped colorized diff with no interaction.
- **In-terminal video player** with pause/stop/seek/close over kitty→sixel→halfcell→ascii, plus `/stream-video` and
  `/generate-video` as composer commands. `UI/Screens/video_player_screen.py:74-83,122-130`;
  `Chat/console_command_grammar.py:57-67`. (Named grep `video.*render\|mpv\|ffplay` over `hermes_cli/` → 0.)
- **11-step first-run setup wizard** with resumable draft state, vs. hermes's linear stdin prompt sequence.
  `UI/Wizards/first_run_setup_state.py:915-937`.
- **Dedicated speech workbench** — STTS playground, voice-cloning window, TTS profile library, audiobook generator.
  `UI/STTS_Window.py`; `UI/Voice_Cloning_Window.py`; `UI/stts_profile_library.py:1592`; `TTS/audiobook_generator.py`.
- **Per-message action strip** — copy, speak/stop, view image, save image, diff, review changes, variant navigation.
  `Chat/console_message_actions.py:143-247`.
- **Workspace-scoped conversation browser** with starred-recency sort, run-urgency markers, and collapse groups.
  `Workspaces/conversation_browser_state.py:691,787,802,842`.
- **Structured notification delivery/presentation service** with delivered/failed/suppressed states, beyond simple
  toasts. `Notifications/notification_presentation.py:49-120`.

### Ops

- **Encrypted secrets at rest** — AES-256-GCM with scrypt over sensitive config values, password-gated, with
  enable/disable/change-password flows and a guard that blocks persisting plaintext secrets once encryption is on.
  `Utils/config_encryption.py:1-6`; `config.py:7283,7310,5734,5881`. Hermes stores API keys as plaintext in
  `~/.hermes/.env` protected only by mode 0600 (`hermes_cli/config.py:898-911`).
- **Single-source secret-key classification** that fixed four previously-disagreeing notions of "is this a secret",
  used by both the encryptor and the log redactor. `Utils/sensitive_config_keys.py:1-20`.
- **Strict metadata-only admission for the persistent log sink** — records must go through `log_persistent_metadata`
  with a bounded schema; no `repr`, no exception text. Enforced by a handler filter *and* a repo-level inventory check.
  `Utils/persistent_diagnostics.py:1-8`; `Logging_Config.py:368`; `scripts/check_persistent_diagnostic_inventory.py`.
  Hermes redacts by pattern-matching the formatted line — a weaker guarantee.
- **Package-wide `diagnose=False` enforcement** installed at import time so no forgotten `logger.add()` can leak local
  variables into an exception trace. `__init__.py:26-72`; `Metrics/logger_config.py:36-45`.
- **Cross-process config write lock** — `portalocker` EXCLUSIVE over a sibling `.lock` wrapping every whole-file config
  transaction. `config.py:5285-5322`. Hermes's `atomic_config_write` is atomic-replace only, with no interprocess
  mutual exclusion.
- **Config generation protocol** — `_CONFIG_GENERATION` sandwich and `run_if_runtime_config_generation_current` so
  callers can validate their view is still current. `config.py:165,5606,5699,5995`.
- **Log-file permission hardening across rotation generations** — re-hardened on open and on every rollover.
  `Logging_Config.py:230-266`.
- **Native signed desktop installers** — macOS `.app` + `.dmg`, Windows Nuitka `.exe` + NSIS.
  `Packaging/macos/scripts/package_dmg.sh:33`; `Packaging/windows/installer.nsi`. Hermes has no in-tree `.dmg`/`.msi`.
- **Zero upload paths** — no diagnostics-share command exists at all, so no code path can send user data anywhere.
  Hermes's equivalent is explicit and redacted, but it does exist (`hermes_cli/diagnostics_upload.py:1-20`).
