# Hermes parity: deferred items

**66 entries** covering the ~73 capability rows from the 2026-08-31 parity report that were **reviewed and not
filed** (a few entries bundle closely related rows — the three ACP-client rows are one entry, for instance).
Kept here so they are recoverable rather than lost in a report nobody re-reads.

Structure: 9 blocked behind filed work · 21 recommended for closing with reasons · 36 worth revisiting.

- Source: `qa/hermes-parity-2026-08-31/report.md` (hermes `origin/main` a0a63a1bc2 vs chatbook `origin/dev`),
  full candidate list with evidence in `qa/hermes-parity-2026-08-31/gap-candidates.md`.
- Filed instead: **TASK-25900–25914** and **TASK-26000–26040** (56 tasks); their execution ordering is in [hermes-parity-burndown-plan.md](hermes-parity-burndown-plan.md).
- **Review these after that burn-down**, not before — several are blocked behind filed work, and a few will be
  answered by it rather than needing their own task.

Two things to re-check at review time, because both will have moved:

1. **Hermes will have moved.** The comparison is pinned to a August 2026 tip and the v0.21.0 curated notes had
   not shipped. Re-fetch before assuming any row still describes reality.
2. **Chatbook will have moved.** Several rows here may be closed by the filed 56 as a side effect. Re-verify
   against code before filing anything; this file is a pointer, not evidence.

---

## Blocked behind filed work — decide these *inside* the blocking task

| Row | Blocked behind | Note |
|---|---|---|
| External memory-provider plugins | TASK-25907 | Its ADR settles what may persist; a provider interface is meaningless before that |
| Learning graph / journey | TASK-25907 | Same |
| Curator (automatic memory hygiene) | TASK-25907 | Same |
| Out-of-loop forks | TASK-25907 | What a detached fork may write is exactly 25907's question |
| Pool selection strategy | credential pooling | Needs pooling to exist first |
| Rate-limit → rotate account | credential pooling | Same |
| ACP client / permission bridge / tool typing (3 rows) | ACP agent-server | All three need the server half first |
| Skill bundles behind one slash command | curated skill catalog (TASK-26008) | Needs a catalog to bundle |
| LLM guardian ("smart approvals") | TASK-25905 | Only pays off once command-risk detection exists to feed it |

## Recommend closing — with the reason

These were judged not worth filing. Recorded so the judgement is auditable and can be overturned.

| Row | Reason |
|---|---|
| Themes / skins | Chatbook is **ahead** (58 themes vs 9 skins). Only hermes's desktop VS Code Marketplace themes are missing, which has no TUI analogue. |
| Diagnostics bundle / share | Chatbook has **no upload path at all**. That is a privacy feature, not a gap. |
| Traces / spans | Needs a collector running; low value for a single-user local TUI. Metrics already exist. |
| Config layering (admin overlay) | A root-owned `/etc` overlay is a multi-user fleet concept. |
| Credential lifecycle CLI | Chatbook has no CLI subcommand tree to hang `auth add/remove/list` on. |
| Enterprise identity (Bedrock / Vertex / Azure) | Enterprise-shaped; not local-first. |
| Inbound webhooks | Requires an always-on listener; gateway-shaped. |
| Out-of-band approval transports | An unattended TUI run has no second channel to approve from. |
| Cgroup / orphan reaping | Linux-service-shaped; process-tree cleanup already exists. |
| Browser extension surface | A separate product, not a parity gap. |
| Pluggable execution environments | Ten remote backends; see sandboxing, which is the real question. |
| Browser tools (13 tools + CDP) | Separate product decision. |
| Plugin / extension package format | MCP already covers tool extension. |
| Skills as scheduled automations | Overlaps the scheduling program; decide there. |
| Progressive disclosure `tool_call` | Marginal over the existing `find_tools`/`load_tools`. |
| Unified deadline primitive | Pure refactor; delivers nothing user-visible beyond TASK-26003. |
| Per-job notepad | Only matters once local scheduled tasks do stateful work. They do not. |
| Cron CRUD as an agent tool | Conflicts with ADR-077 decision 4 (nothing auto-approved unattended). Record as a decision, not a task. |
| Natural-language schedule input | **Already filed** as task-23102 / task-23103. |
| RTL / bidi | Depends on i18n; Textual has no bidi engine. |
| Packaging: Nix / npm / Termux / Homebrew | Low demand. |

## Worth revisiting — ranked within each area

Nothing here is blocked or rejected; these are real gaps that lost on priority.

### Agent control loop
- **Parallel tool batch dispatch** — independent tool calls serialize today.
- **Conversation rewind (undo N turns)** — the compaction-boundary interaction is the hard part.
- **Sub-agent worktree isolation** — parallel children share one tree; relates to the change-attribution taxonomy.
- **Premature-finish nudges** — narrow value, real nagging risk.
- **Busy-input policy** — needs TASK-25903 first; then it is a setting.

### Tools & skills
- **Code execution / Programmatic Tool Calling** — highest leverage item in this whole file. Model writes Python
  that RPCs back into the tool registry so intermediate results never enter context. Big build; both halves
  (subprocess worker, `invoke_by_name`) already exist.
- **Concurrent-edit / stale-write guard** — matters more once sub-agents run in parallel.
- **Named toolsets / bundles** — grouping; per-tool gating only today.
- **Image / video generation as agent tools** — exists as UI screens, not callable by the agent.
- **Remote skill install (hub, taps, lockfile provenance, quarantine)** — after a curated catalog proves demand.
- **Skill usage telemetry / mutation ledger**, **skill authoring lint** — polish on the skills surface.
- **Configurable tool output limits** — hardcoded constants today.

### Permissions & guardrails
- **OS / container sandboxing** — the largest single build in the report. Becomes urgent the moment unattended
  or server-side execution ships; until then the path jail plus approval gate covers the common case.
- **Content-level pre-exec scanner** — homograph URLs, pipe-to-interpreter, terminal injection.
- **Approval modes** — a cross-cutting manual/smart/off covering shell too.
- **Non-interactive fail-closed policy** — per-context deny defaults; one context exists today.
- **Dry-run "what would happen"** — replay the real evaluators against a hypothetical call.

### Providers & auth
- **Full subscription OAuth** (ChatGPT/Codex, Copilot, xAI, Qwen, MiniMax) — revisit after TASK-26022 proves
  the credential-borrow shape.
- **Credential pooling** — less relevant for a single user with one key; matters if OAuth refresh lands.
- **Provider registry / adapter shape** — 23 providers via a hardcoded dict and hand-synced frozensets vs ~48
  declarative profiles. A refactor with real payoff and real risk.
- **Billing / credits / usage surfaces**, **rate-limit header telemetry** — cheap once there is somewhere to show it.

### Scheduling
- **Blueprints / templates** — typed-slot templates rendering to a form and a command from one schema. The axis
  chatbook has nothing on (TASK-18936 mischaracterized this as chatbook being ahead on modeling).
- **Agent-proposed cron suggestions**, **job chaining**, **output storage + retention** — each a real feature.
- **Repeat limit**, **pause with reason**, **run-now extra context** — each is a column plus a UI field.

### Interop
- **ACP agent/server** — be an ACP agent for Zed/VS Code/JetBrains. `ACP_Interop` is currently a subprocess
  launcher that renders "Diffs unavailable". Real and large.
- **A2A v1.0** — agent card plus JSON-RPC task surface.
- **MCP curated catalog + installer** — after TASK-25900/26032 make remote servers reachable.
- **Local IPC control surface** — versioned control socket for sibling processes.
- **Connectors (Home Assistant, Spotify, Meet)** — per-integration, pick individually if wanted.

### User surfaces
- **i18n / UI localization** — a total absence, but a 60-screen string-extraction project with no recorded user
  demand. Scope to the Console screen first if it is ever taken.
- **CJK wide-char table alignment** — model markdown tables misalign; `wcwidth` is already a dependency.

### Ops
- **Self-repair / self-update** — cheap slice is a PyPI version check at startup; the full pipeline is large.
- **Whole-state backup / export** — per-DB backups exist and nothing rolls them up.
- **Profiles / multi-instance** — isolated config homes.
- **Packaging: Dockerfile** — the one genuinely missing distribution target for a terminal app.
- **Resource monitoring** — no disk-free or leak time series.
