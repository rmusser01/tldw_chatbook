# Gap candidates for filing — review list

Source: `report.md` (hermes `origin/main` a0a63a1bc2 vs chatbook `origin/dev`). **156 rows** carry a
`HERMES AHEAD` verdict. This file strips the ones already covered and lists what remains, so you are
reviewing real candidates rather than a raw dump.

- **Size** — S = under ~100 lines on shipped machinery · M = a real feature, one PR · L = needs an ADR / multi-PR
- **Rec** — ✅ file it · ➖ file only if you want breadth · ❌ recommend against (design reason given)

---

## Already handled — not candidates

**Filed this session (25900–25914):** MCP HTTP/SSE transport · in-loop retry · model fallback chain ·
primary-agent steering · tool-output spill + per-turn aggregate budget · shell hardline floor (covers
"shell command risk detection", "unconditional hardline deny", "user deny globs") · doctor (covers both the
2.8 and 2.9 rows) · cross-session memory · `/help` · slash-command surface · micro-compaction ·
tool-result pruning · stale-image retirement · tool-timeout clamp · Prometheus listener gate.

**Pre-existing tasks:** 18920 deny-with-reason · 18921 slash ranking + composer autocomplete ·
18922 one-shot model override · 18925 OS notifications · 18927 edit/patch self-recovery (AC #7 added this
session for `fs_patch`) · 18928 always-allow mining · 18929 denial breaker · 18930 prompt stash ·
18931 MoA/council · 18933 wake word · 18934 kanban UI.

---

## 1. Agent control loop

| Capability | What's missing | Size | Rec |
|---|---|---|---|
| Active-turn redirect | Cancel the model call, keep completed tool results, re-run the turn with a correction. Stop is terminal today. | M | ✅ |
| Graceful budget wrap-up | Warn the model at ~80% of wall budget; on exhaustion make one tools-stripped summary call instead of `RUN_STUCK`. | S | ✅ |
| Empty-response handling | Two consecutive zero-token empties = deterministic; skip remaining retries. Today an empty turn is silently `RUN_DONE`. | S | ✅ |
| Stream stall watchdog | Kill a stream that dribbles keep-alive bytes; only the httpx read timeout bounds it now. | S | ✅ |
| Global resumable pause (ESTOP) | One sentinel that stops *new* work without killing in-flight runs. ~10 lines. | S | ✅ |
| Parallel tool batch | Dispatch a tool batch concurrently under a shared deadline; strictly sequential today. | M | ➖ |
| Conversation rewind (`/undo N`) | Durable multi-turn truncate. Only regenerate-with-snapshot exists. | M | ➖ |
| Sub-agent worktree isolation | Per-child git worktree so parallel children can't collide on files. | M | ➖ |
| Premature-finish nudges | Convert a suspicious clean exit (code edited, no verification) into a bounded continue. | M | ➖ |
| Busy-input policy | User-selectable interrupt / queue / steer. Queue-only today. | S | ➖ (depends on 25903) |
| Unified deadline primitive | Consolidate three per-site timeout mechanisms into one. | M | ❌ refactor, no user-visible capability beyond the stall watchdog |
| Out-of-loop forks | Detached post-turn review/curator forks. | L | ❌ overlaps 25907; revisit after that ADR |

## 2. Tools & skills

| Capability | What's missing | Size | Rec |
|---|---|---|---|
| Tool-arg coercion | Repair models that emit stringified JSON arrays. `agent_runtime.py:1149` is a bare `json.loads`. | S | ✅ |
| Shell failure hints | Map known non-zero-exit shapes to one actionable recovery line. | S | ✅ |
| Catalog ranking + visible listing | Substring match misses paraphrase; deferred tools are invisible. `difflib` tier + name listing in the tool description. | S | ✅ |
| Configurable output limits | Byte/line caps are hardcoded module constants. | S | ➖ |
| Concurrent-edit / stale-write guard | Block a write to a file another agent wrote since this agent read it. | M | ➖ |
| Bundled skill catalog | Chatbook ships **zero** skills; the runtime is good and the format is Claude-Agent-Skills compatible. | M | ✅ |
| Skill authoring lint | Soft-convention linter on top of the existing hard validation. | S | ➖ |
| Skill usage telemetry / ledger | Per-skill counters + append-only mutation ledger. | M | ➖ |
| Remote skill install (hub/taps/lockfile) | Direct GitHub/zip install exists; no registry, provenance lockfile, or quarantine. | L | ➖ |
| Named toolsets / bundles | Group tools under one name; per-tool gating only today. | M | ➖ |
| Skill bundles behind one slash command | YAML bundle → one `/name`. | M | ❌ depends on a catalog existing first |
| Skills as scheduled automations | Skill frontmatter carrying a cron schedule. | L | ❌ overlaps the scheduling program; decide there |
| Plugin/extension package format | Portable dir bundling skill + MCP server + config. | L | ❌ MCP already covers tool extension |
| Progressive disclosure `tool_call` | Invoke without loading the schema; marginal over `find_tools`/`load_tools`. | S | ❌ marginal |
| Code execution (PTC) | Model writes Python that RPCs back into the tool registry; intermediate results never enter context. | L | ➖ high leverage, big build |
| Pluggable execution environments | 10 remote backends (docker/ssh/modal/…). | L | ❌ see sandboxing below |
| Browser tools | 13 browser tools + CDP + extension. | L | ❌ separate product decision |
| Image/video gen as tools | Exists as UI screens only; not callable by the agent. | M | ➖ |

## 3. Permissions & guardrails

| Capability | What's missing | Size | Rec |
|---|---|---|---|
| Secret redaction by value shape | `MCP/redaction.py` matches key *names* only and documents its own bypass at line 64. Pure function, no callers change. | S | ✅ |
| Post-tool hook | Pre-dispatch hook exists; no post-tool seam. (`Tools/file_operation_hooks.py` is dead — pinned retired.) | S | ✅ |
| Denial semantics | Add the anti-rephrase instruction to denied-tool copy so the model stops retrying the same call. | S | ✅ |
| Non-interactive fail-closed policy | Per-context deny defaults (cron / unattended / single-query). One context today. | S | ➖ |
| Dry-run "what would happen" | Replay the real evaluators against a hypothetical call. | M | ➖ |
| Per-arg / per-path allow rules | "Always allow" is whole-tool, which is why raw shell deliberately withholds it. Blocks quieting safe repeats. | M | ✅ |
| Approval modes | A cross-cutting manual/smart/off mode covering shell too. | M | ➖ |
| LLM guardian ("smart approvals") | Aux-LLM clears false positives. | M | ❌ pays off only after 25905 lands |
| Out-of-band approval transports | Approve from Discord/Slack/a plugin transport. | M | ❌ hosted-shaped; an unattended TUI run has no second channel |
| OS / container sandboxing | No container or OS isolation anywhere; isolation is path jail + approval gate. | L | ➖ big; urgent only if unattended execution ships |
| MCP server-spawn guard | Block exfil-shaped stdio configs at save/spawn. | S | ✅ |
| Content-level pre-exec scanner | Homograph URLs, pipe-to-interpreter, terminal injection. | M | ➖ |
| Cgroup / orphan reaping | Process-tree cleanup exists; no cgroup-level sweep. | S | ❌ Linux-service-shaped |

## 4. Context & memory

| Capability | What's missing | Size | Rec |
|---|---|---|---|
| Cache TTL tiers (1h) | Three sites emit bare 5-minute ephemeral markers. One key behind a flag. | S | ✅ |
| OpenAI/Codex `prompt_cache_key` | No cache key derivation for non-Anthropic providers. One kwarg. | S | ✅ |
| Compaction timeouts | Auxiliary compaction call is unbounded. `asyncio.wait_for` + a config value. | S | ✅ |
| Manual preview / dry-run | The plan object is already returned before commit; just isn't surfaced. | S | ✅ |
| Focus-directed compaction | `/compress focus <topic>` — one string appended to the compaction prompt. | S | ✅ |
| Context breakdown by category | Only request-vs-conversation totals; no system/tools/RAG/skills split. | M | ✅ |
| Context references (`@file`, `@diff`, `@folder`) | No file/diff expansion; `$`-mentions are skills-only. Autocomplete surface already exists. | M | ✅ |
| Pluggable per-session context engine | Swap the compaction strategy per session. | M | ❌ one implementation; premature abstraction |
| Provider-native compaction | Delegate compaction to Codex/OpenAI server-side. | M | ➖ |
| External memory-provider plugins | mem0/supermemory/etc. behind an ABC. | L | ❌ decide inside 25907's ADR |
| Learning graph / journey | Graph over skills + memory cards. | L | ❌ decide inside 25907's ADR |
| Curator | Scheduled aging/consolidation of learned facts. | L | ❌ decide inside 25907's ADR |

## 5. Providers & auth

| Capability | What's missing | Size | Rec |
|---|---|---|---|
| Claude Pro/Max credential borrow | Read `~/.claude/.credentials.json` + the oauth header path. A Max subscriber pays $0 marginal instead of API rates. | M | ✅ highest user-money item in the report |
| Model metadata + pricing from models.dev | Hand-maintained regex capability table and a `_SEED_AS_OF` price table. The cache/merge plumbing already exists. | M | ✅ |
| Fast/cheap vs strong routing | Auxiliary calls (titles, compaction, vision) use the same model as the main chat. One config key. | S | ✅ |
| Rate-limit header telemetry | 12 `x-ratelimit-*` headers discarded; response headers already in hand. | S | ➖ |
| Credential pooling (N keys/provider) | One key per provider; a 429 retries the same key. | M | ➖ less relevant for one user |
| Pool selection strategy | round_robin / least_used / fill_first. | S | ❌ needs pooling first |
| Rate-limit → rotate account | Exhaustion TTL + rotate. | M | ❌ needs pooling first |
| Full OAuth (ChatGPT/Codex, Copilot, xAI, Qwen, MiniMax) | Device-code and PKCE flows per provider. | L | ➖ after the Claude borrow proves the shape |
| Enterprise identity (Bedrock/Vertex/Azure) | SigV4, GCP SA tokens, Azure identity. | L | ❌ enterprise-shaped; not local-first |
| Provider registry size / adapter shape | 23 providers via a hardcoded dict + hand-synced frozensets vs ~48 declarative profiles. | L | ➖ refactor with real payoff |
| Credential lifecycle CLI | `auth add/remove/list` with per-source removal. | M | ❌ chatbook has no CLI subcommand tree |
| Billing / credits / usage | Account usage windows, credit headers, spend screens. | M | ➖ |
| Cross-provider failover | **already filed as 25902** | — | — |

## 6. Scheduling / cron

| Capability | What's missing | Size | Rec |
|---|---|---|---|
| Global emergency stop | One boolean in `_dispatch_due` + a workbench binding. ~10 lines. | S | ✅ |
| Scheduler liveness / heartbeat | A dead loop is indistinguishable from an idle one. Stamp `last_tick_at`. | S | ✅ |
| Durable per-run execution ledger | Reminders/briefings keep one overwritten status row. (Watchlists already have `local_watchlist_runs`.) | M | ✅ |
| Failure incidents (dedupe + ack) | Re-notifies on every failed dispatch; no acking. Rides on the ledger above. | M | ✅ |
| Preflight validation | Nothing checks a handler's prerequisites before firing. | S | ✅ |
| Natural-language schedule input | ISO or raw cron only. | M | ❌ already filed as task-23102/23103 |
| Repeat / occurrence limit | A recurring reminder runs forever. One column. | S | ➖ |
| Pause with reason | Boolean `enabled` only. One column. | S | ➖ |
| Run-now extra context | One-shot prompt attached to a manual run. | S | ➖ |
| Job chaining (`context_from`) | Feed one job's output into the next. | M | ➖ |
| Output storage + retention | Reminders produce a notification and no artifact. | M | ➖ |
| Per-job notepad | Durable KV carried between wake-ups. | M | ❌ only matters once local tasks do stateful work |
| Blueprints / templates | Typed-slot templates → form + command + deeplink. | L | ➖ |
| Agent-proposed cron suggestions | An inbox of agent-proposed jobs. | M | ➖ |
| Cron CRUD as an agent tool | Let the model create/edit jobs. | M | ❌ conflicts with ADR-077 decision 4; record as a decision, not a task |

## 7. Interop

| Capability | What's missing | Size | Rec |
|---|---|---|---|
| MCP sampling + elicitation | Server-initiated methods all return `-32601`. Two handlers over existing plumbing. | M | ✅ |
| MCP resources/prompts as agent tools | Client can read them; the agent provider exposes tools only. Four synthetic registrations. | S | ✅ |
| Outbound signed webhooks | "Tell my dashboard when a run finishes." HMAC POST on existing hook points. | S | ✅ |
| MCP OAuth 2.1 / PKCE / DCR | Needed for most hosted MCP servers. | M | ✅ (after 25900) |
| MCP curated catalog + installer | No offline curated server list. | M | ➖ |
| OpenAI-compatible API server | "Point my editor at my own agent." One ASGI app over `agent_service`. | M | ✅ |
| Local IPC control surface | Versioned control socket for sibling processes. | S | ➖ |
| ACP agent/server | Be an ACP agent for Zed/VS Code. `ACP_Interop` is only a subprocess launcher. | L | ➖ real, big |
| ACP client / permission bridge / tool typing | The other three ACP rows. | L | ❌ all depend on the ACP server above |
| Agent-to-agent (A2A v1.0) | Agent card + JSON-RPC task surface. | L | ➖ |
| Inbound webhooks | External events start agent runs. | M | ❌ needs an always-on listener; gateway-shaped |
| Browser extension surface | Local WS receiver + an extension. | L | ❌ separate product |
| Connectors (Home Assistant / Spotify / Meet) | Per-integration tools. | M each | ➖ pick individually if wanted |

## 8. User surfaces

| Capability | What's missing | Size | Rec |
|---|---|---|---|
| Embedded terminal pane | `Terminal/` is fully built and **unwired to any UI**. This is wiring, not building. | S | ✅ |
| Filesystem checkpoints | Per-turn working-tree baseline + rollback. Change-review reverts per file only. | M | ✅ |
| CJK wide-char table alignment | Model markdown tables misalign with CJK text; `wcwidth` is already a dependency. | S | ➖ |
| i18n / UI localization | Zero localization machinery anywhere. | L | ➖ 60-screen extraction; no recorded demand |
| RTL / bidi | Needs `python-bidi`; Textual has no bidi engine. | L | ❌ depends on i18n |
| Themes / skins | Chatbook is ahead (58 vs 9); only desktop marketplace themes are hermes-only. | — | ❌ N/A for a TUI |

## 9. Ops

| Capability | What's missing | Size | Rec |
|---|---|---|---|
| Corrupt-config last-known-good | A broken TOML silently reverts encryption settings and DB paths to defaults. ~5 lines. | S | ✅ |
| Crash forensics (faulthandler) | One line in `configure_application_logging`; the private log dir already resolves. | S | ✅ |
| Hot reload of external edits | Config cache keyed on path only; stat and compare `(mtime, size)`. | S | ✅ |
| Unknown-key / deprecated-key validation | No typo or stale-key detection. | S | ✅ |
| Config schema migrations | No config version at all; renames leave orphan keys forever. | M | ✅ |
| Whole-state backup / export | Per-DB backups exist but nothing rolls them up. | M | ➖ |
| Config layering (admin overlay) | No system/admin layer. | M | ❌ multi-user-shaped |
| Profiles / multi-instance | Isolated config homes. | M | ➖ |
| Self-repair / self-update | Pre-import venv repair; update pipeline with receipts. | L | ➖ cheap slice = a PyPI version check at startup |
| Traces / spans | OTel is metrics-only; no `TracerProvider`. | M | ❌ needs a collector; low value for a local TUI |
| Resource monitoring | No disk-free or leak time series. | S | ➖ |
| Diagnostics bundle / share | No bundle command (and no upload path at all — arguably correct). | M | ❌ chatbook's zero-upload stance is a feature |
| Packaging: Dockerfile | The one genuinely missing target for a terminal app. | S | ➖ |
| Packaging: Nix / npm / Termux / Homebrew | Other distribution targets. | M each | ❌ low demand |
