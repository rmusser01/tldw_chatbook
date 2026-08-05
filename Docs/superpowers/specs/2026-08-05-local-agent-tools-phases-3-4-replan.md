# Local Agent Tools — Phases 3-4 Re-plan (tldw_server-informed)

**Date:** 2026-08-05
**Status:** Draft (pending spec review + user review)
**Supersedes:** phase 3 and phase 4 scope of `Docs/superpowers/specs/2026-08-04-local-agent-tools-design.md` (phases 1-2 shipped unchanged: PRs #1352, #1358)
**ADR:** new ADR required (ADR-033) — git tools spawn subprocesses and the virtual-CLI model defines a process-execution boundary; both are security/runtime-boundary decisions per backlog.md rules. ADR-032 remains in force for naming, confinement, and approval discipline.
**Reference:** tldw_server @ dev — `apps/mcp-unified` (gateway/policy) + `tldw_Server_API/app/core/MCP_unified/modules/implementations/` (tool modules). Research notes: tool modules register via `BaseModule.get_tools()/execute_tool()`, mostly opt-in/disabled by default, strict `domain.action` naming, systematic byte-caps with `truncated` flags, structured `{ok, reason_code, message}` errors.

## 1. What changed and why

The original phase 3 (web_fetch + web_search + todo_write) and phase 4 (MCP exposure) were scoped before reviewing tldw_server's unified MCP tool surface. That review found mature, portable implementations for several planned tools and a deliberate design answer for the deferred shell question. This re-plan ports rather than reinvents, and adds two tools the original roadmap lacked (git, fs_patch).

## 2. Phase 3 (revised): research + git + patch tools

All tools register as `LocalToolSpec`s in `Agents/local_tool_provider.py` under ADR-032 discipline (workspace confinement, permission store under `local:__local__`, fail-closed, pinned refusal strings, 32 KiB result fitting where applicable). New core modules by domain rather than growing `local_tool_impls.py`:

### 2.1 `web_fetch` — PORT, not scratch

Port tldw_server's `web_fetch_module.py` + `web_tool_base.py` + `web_rate_limit.py` + `web_cache.py` (~900 lines) into `Tools/web_tool_impls.py`, adapted to the sync-core shape:

- Keep: redirect cap (5), 30 s timeout, byte caps (1 MB default / 5 MB hard max) with explicit `truncated` flags, per-domain rate limiting, response caching, structured error reasons.
- Keep especially: the SSRF/egress guard (`Web_Scraping.outbound_policy` equivalent) — a scratch fetcher without it is a SSRF hole. If `outbound_policy` doesn't port cleanly, implement its checks (scheme allowlist, private/loopback IP refusal, redirect re-validation) directly.
- Replace: tldw_server internals with chatbook equivalents; HTML→text via `trafilatura` as the original spec decided.
- Result discipline: byte-fitted per ADR-030; `truncated` flag semantics folded into the text result.

### 2.2 `web_search` — migrate, adopt patterns

Migrate the existing `Tools/web_search_tool.py` onto the catalog as planned (delegates to `Web_Scraping/WebSearch_APIs.perform_websearch`). Adopt from tldw_server's `web_search_module.py`: bounded per-result size (~4 KiB/result) and structured error reasons.

### 2.3 `todo_write` — unchanged from original spec

Session-scoped, in-memory on `ConsoleChatSession`, claude-code `TodoWrite` semantics. **Decision (user):** do NOT port tldw_server's `notes.tasks.*` system — too heavy for the goal; noted here so the question doesn't get re-asked.

### 2.4 `fs_patch` — PORT

Port tldw_server's `filesystem_diff.py` (~290 lines, self-contained unified-diff applier) into `Tools/patch_tool_impls.py`. `fs_patch` tool: applies a unified diff to workspace files; `mutates` tag; confinement via `resolve_workspace_path` per target file; dry-run/--check mode returns the would-be result without writing.

### 2.5 Git tools — PORT (read-only)

Port tldw_server's `git_module.py` (~2,100 lines) into `Tools/git_tool_impls.py`, as `git_status`, `git_diff`, `git_log`, `git_blame`, `git_branches` (snake_case per ADR-032; `git_conflicts_*` deferred unless trivially included):

- Read-only only. Subprocess `git` with a subcommand allowlist, 30 s timeout, 1 MB output cap, workspace-root confinement (repo discovery confined to the workspace root; bare `git -C <root>`).
- These are the first model-invocable tools that spawn a process — the ADR-033 boundary: fixed argv arrays (no shell interpolation), allowlisted subcommands/flags, cwd confined, timeouts, output caps. Risk tag: none (read-only), but a new `process` tag applies per the permission store's HIGH_RISK_TAGS if flags ever expand — document why read-only git doesn't floor to ask while still being process-spawning.
- tldw_server's only local deps (`tool_observability`, workspace-root resolver) are thin — inline/shim them.

### 2.6 `web_research` — subagent/skill, NOT a tool

**Decision (user):** research composition is subagent functionality, not a catalog tool. Implement as a bundled skill (e.g. `web-research`) whose prompt orchestrates `web_search` + `web_fetch` through a `spawn_subagent` run — using the existing SkillToolProvider/spawn infrastructure. The spec deliverable is the skill definition + docs, not new runtime code. tldw_server's `web_research_module.py` (~410 lines) serves as the orchestration reference.

## 3. Phase 4 (revised): MCP exposure + shell design + permission notes

### 3.1 MCP server exposure (as originally specced, expanded set)

Expose the full local tool set (fs_* + web_* + todo_write + git_* + fs_patch) through `MCP/server.py` backed by the same cores, giving external MCP clients parity. Adopt from tldw_server: structured error reasons in tool results, and the domain-grouped catalog presentation. Naming stays `fs_*`/`git_*` snake_case (ADR-032); tldw_server's dotted convention is noted as the alternative and rejected for consistency with the chatbook registry.

### 3.2 Shell — adopt the virtual-CLI model (design only)

**Decision (user):** no raw shell. The re-plan adopts tldw_server's governed virtual-CLI design as the answer to the shell question, to be implemented as a future phase:

- A `run`-style tool whose command registry maps allowlisted commands (`ls`, `cat`, `grep`, `find`, `stat`, …) onto the existing policy-checked `fs_*`/`git_*` cores — no `subprocess` to a host shell at all.
- Profile-granted commands only; output spill-to-disk past ~64 KiB with preview caps.
- ADR-033 records this as the deliberate rejection of a raw bash tool, with the virtual-CLI as the accepted alternative.

### 3.3 Permission-model upgrades — deferred, recorded

**Decision (user):** defer. Noted as future work in ADR-033: TTL-bound approval grants (replace/augment permanent `always_allow`), claude-code-style path/rule syntax (`Read(/docs/**)`), and `explain-policy`-style dry-run evaluation UX.

## 4. Testing (phase 3 specifics)

- Port fidelity tests: ported modules' behavior pinned against fixtures (git repo in tmp_path; httpx.MockTransport for web).
- ADR-033 boundary tests for git tools: subcommand/flag injection attempts refused, timeout enforcement, output cap, non-workspace repo refused.
- fs_patch: multi-hunk apply, conflict/fuzz failure returns model-actionable error, dry-run writes nothing, confinement per target file.
- web_fetch: SSRF guard tests (loopback/private IP, file:// scheme, redirect-to-private).
- Skill: web-research skill smoke test through the skill resolver.
- Hypothesis where it pays (patch parsing).

## 5. Phasing and task breakdown

- **Phase 3a:** web_fetch port + web_search migration + todo_write (research cluster)
- **Phase 3b:** fs_patch + git tools (workspace cluster, needs ADR-033 first)
- **Phase 3c:** web-research skill
- **Phase 4:** MCP exposure + ADR-033 shell/permission design sections

Each phase gets its own plan + backlog task, per the established pipeline.

## 6. Explicit non-goals (updated)

Raw shell/bash (virtual-CLI is the adopted answer), notes.tasks port, TTL grants / rule-syntax permissions (recorded future work), kanban/codegraph/browser/slides/flashcards/rpg ports (tldw_server-internal coupling), dotted tool naming.
