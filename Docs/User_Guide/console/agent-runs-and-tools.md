# Agent runs & tools — what happens when a reply uses tools, skills, and sub-agents

## What this page covers

When a Console reply needs more than plain text — running a tool, spawning a
sub-agent, executing a skill, or calling an MCP server — it becomes an *agent
run*. This page covers what you see while a run is in flight, how tool-call
approvals work, how background runs in other tabs surface, and how skills and
MCP tools plug in. For the Console screen itself see [Console](../console.md).

## Getting there

Open Console (**Ctrl+2**) and send a message — runs happen wherever you
chat, no separate mode to enable. The surfaces this page covers: the
transcript's inline tool rows, the **Agent** section in the left "Console
context" rail, the Inspector's status rows, the status chips above the
composer, and the approval and confirm cards that appear above the
transcript.

## Layout tour — what you see during a run

Each Console tab runs its own agent, and a run keeps going in the background
while you're on another tab. The first time you open a second tab (**Ctrl+T**),
a one-time banner spells it out:

> Each tab runs its own agent — up to 3 in parallel (change in Settings >
> Console Behavior).

The number is your configured cap (default 3). Sending past the cap is
refused with a message like "2 agents already running (…). Wait for one to
finish or interrupt it." Runs live only while Console stays open — see
[Console agent runs are screen-scoped](../index.md#console-agent-runs-are-screen-scoped).

**In the transcript** — inline `Tool` rows appear between your message and the
reply:

- `⚙ toolname → result preview` — a tool call and a preview of its result,
  truncated with an `… (+N chars)` suffix past the display cap.
- `⤷ spawned sub-agent: …` — the agent delegated work to a sub-agent.
- `⚠ …` — an error summary.

**In the left rail** — expand the **Agent** section (collapsed by default):

- Status line: `Agent: idle`, or `Agent: running · step N` while working.
- One `·`-prefixed line per step.
- One line per sub-agent, prefixed `✓` (done), `●` (running), `⚠` (stuck), or
  `✗` (error/cancelled). Several of these can read `●` at once — sub-agents
  the supervisor spawned in the same reply run in parallel, not one at a
  time; see [Parallel sub-agents](#parallel-sub-agents-the-fleet). **Click a
  sub-agent line to drill in** — the status line becomes `Sub-agent ·
  <status> (Back)` with that run's own step lines; the **Back** button
  returns to the main run.
- **View full log** opens the "Full run log — <run id>" window: the complete,
  untruncated record ("what the model actually saw, before the Console's
  display cap trimmed it"). **Close** or **Esc** dismisses it.

**In the Inspector** (right rail) — the "Status:" line tracks the run
(`Status: Ready` / `Status: Generating…` / `Status: Needs approval` /
`Status: Source blocked` / `Status: Blocked`), and the "Run recipe" row summarizes provider / model /
sources / tools / approvals for the next send.

**In the status chips** (above the composer) — "Tools: N ready" counts the
tools available to the agent (the chip stays hidden until tools are counted,
which happens after your first send), and
"Approvals: N pending" counts tool calls waiting on you. The Approvals chip is
clickable: it jumps you to the pending approval card (with nothing pending it
just says "No approval is pending.").

## Features & controls

### Approvals — tools ask before they run

Nothing is ever auto-approved, and built-in tools always ask first. When the
agent wants to run a tool, the run pauses and an **"Approval required"** card
appears above the transcript:

![The "Approval required" card with a pending tool call](../images/console/approval-card.svg)

- Each pending tool call gets a row with a decision select: **Approve once**
  (the default), **Approve for session**, **Always allow**, or **Deny**.
  Built-in tools don't offer "Always allow" — decisions for them last at most
  the session.
- Bulk controls: **Approve all** sets every row to Approve once, **Submit**
  applies each row's selected decision and resumes the run, **Deny all** sets
  every row to Deny.
- When exactly one tool call is pending, the row also gets fast **Approve
  once** and **Deny** buttons that resume immediately, skipping Submit.
- Watch the badges on a row's header: **(definition changed)** means the
  tool's definition differs from what you previously approved; **(high risk)**
  flags reads that could exfiltrate file contents; and a path warning —
  "path outside allowed folders; will fail even if approved" — means the file
  path will be rejected regardless of your decision.

**Always allow** (MCP tools only) is remembered per tool, tied to the tool's
current definition — if the server later changes the tool, the approval card
comes back with a "(definition changed)" badge. Review or change a remembered
allow from the tool's row on the [MCP screen](../mcp.md) 🚧.

With sub-agents running in parallel (see below), more than one approval card
can be pending at once — cards aren't merged across sub-agents: each is
scoped to the one run that raised it, so deciding one card never resolves or
touches another's. Cancelling that sub-agent (see
[Stopping & leaving](#stopping--leaving)) withdraws only its own still-pending
cards; a sibling sub-agent's card, or the parent's, is left exactly as it
was.

### Background & parked runs

Tabs with unwatched activity carry a status marker, listed in F1 help:

> Status markers: ● running · ◆ needs approval · ✓ finished · ✗ failed —
> clears once you visit that tab.

- A background run that hits a tool approval **parks**: its tab gets a `◆`
  badge and you get exactly one toast — "Agent in <tab> (<workspace>) needs
  approval." Switch to that tab to review the card; parked approvals wait,
  they never resolve themselves.
- A background run that ends also toasts once: "Agent in <tab> (<workspace>)
  finished." (or "failed.").
- The left rail pins a fleet summary line whenever other tabs are busy:
  "N other agents running, M waiting for approval."

### Named agents

Beyond a plain, generic sub-agent, the supervisor can delegate to a **named
agent definition** — a reusable persona with its own instructions and
optionally a narrower tool list or a different model. Create and manage
definitions in **Settings ▸ Agents** (Troubleshooting group); changes there
apply immediately but only take effect on the **next** reply, never one
already streaming.

- A definition's instructions are **appended** to the built-in sub-agent
  prompt, not swapped in — every sub-agent still starts from the same base
  identity.
- A definition's tools can only **narrow** what the sub-agent inherits from
  the parent (never grant something the parent itself couldn't use); its
  model override stays on the same provider.
- When a reply spawns a named agent, the transcript's `⤷ spawned sub-agent: …`
  marker and the Agent rail's per-sub-agent line both show it as
  `[<name>] <task>` while the run is live. That prefix is a display detail of
  the running turn, not a stored field — a session you reopen after the app
  restarts shows the sub-agent's task text without the `[<name>]` prefix, and
  the drill-in "Sub-agent · \<status\>" view never shows the name either.
- The run log durably records which definition ran, for future audit tooling
  — `agent_runs.agent_definition` (the definition's name) and
  `definition_fingerprint` (a content hash of its instructions, tools, and
  model at spawn time) are written on every named-agent spawn. Neither is
  currently surfaced in **View full log** or anywhere else in the UI.

### Parallel sub-agents (the fleet)

Sub-agents the supervisor spawns within a **single reply** no longer run one
at a time — up to a configured number can be live together, each working its
own task concurrently. The Agent rail shows this directly: several
`●`-prefixed sub-agent lines can be running at once (see
[Layout tour](#layout-tour--what-you-see-during-a-run) above).

- **How results come back.** The supervisor has to explicitly collect a
  sub-agent's result before it can use it — spawning one hands back a
  handle, not an answer. It gathers results with its own internal
  `wait_agents` step (optionally for just one sub-agent, to get that one's
  answer back in full rather than sharing a combined budget with its
  siblings) and can check progress without blocking via `check_agents`. This
  is internal turn mechanics, not something you drive — the reply you see
  simply arrives once every sub-agent it waited on has finished, and by then
  it has already folded each result into its answer.
- **Skills still run one at a time.** Running a skill (`$name`) always
  returns that skill's own output directly into the same turn, never a
  fleet handle — skills are not part of the parallel fleet.
- **How many can run at once.** Capped at `[agents] max_live_subagents` in
  `config.toml` (default **3**; no Settings UI switch — hand-edit the file).
  Setting it to `1` turns the fleet off entirely: sub-agents go back to
  running one at a time, synchronously, exactly as before. Trying to spawn
  a sub-agent past the live cap is refused ("live sub-agent limit reached
  (N already running); call wait_agents to collect a finished sub-agent
  before starting another") rather than queued — the supervisor collects
  a finished sub-agent to free a slot, then retries, and the refusal itself
  doesn't count against its per-turn spawn budget. A bad value in the
  config file never stops a run: zero or a negative number floors to `1`
  (fleet off), and anything that isn't a number at all (letters, a blank)
  falls back to the default of `3` (fleet on) — either way the run
  proceeds instead of erroring.

### Skills

Skills are reusable instruction packs kept in Library ▸ Skills.

- **Run one** by starting your message with `$name` (arguments can follow:
  `$name your input here`).
- **`/skills`** lists what's installed as `$name — description` lines in the
  transcript. With nothing installed it says: "No skills yet — create them in
  Library ▸ Skills."
- **`/skills <name>`** doesn't run anything — it replies "Run skills by typing
  $name — /skills only lists them."
- A skill that hasn't been reviewed yet refuses to run: "Skill "name" isn't
  trusted (…) — review and approve it in Library ▸ Skills before running it."

When an **agent** (not you) tries something skill-related, a confirm card
appears above the transcript:

- **Skill install** — "An agent wants to install a skill:" with the source
  URL, buttons **Allow** / **Deny**, and the note: "It will be installed
  pending your review and cannot run until you approve it in Library >
  Skills." Allowing installs it, but it still can't run until you review it.
- **Skill script** — "An agent wants to run a script from a skill:" with the
  target and arguments, buttons **Allow once** / **Always allow this skill** /
  **Deny**, and the note: "It runs with a scrubbed environment in a temporary
  folder (not the skill's own folder); only its output comes back."

### MCP tools

Servers you configure on the [MCP screen](../mcp.md) 🚧 surface in Console as
extra tools the agent can call. The Inspector's **MCP** row (under Tools)
shows their state: "N tools ready", or "N servers enabled, not connected" when
servers are configured but unreachable. MCP tool calls go through the same
"Approval required" card as everything else.

### Web research tools

Console's standard web tools are `web_search` (find links), `web_fetch`
(extract one URL), and `web_crawl` (bounded same-host crawl). They are local
agent tools, not tools supplied by an external MCP server. They are registered
by default. Configure the master switch and confinement directory in **MCP →
Tools → Local workspace + web tools**, then choose Allow, Ask, or Off for each
tool in MCP Permissions. Master/root changes apply to the next Console agent
run. `[mcp] expose_local_tools` is only for external MCP clients and does not
enable these tools in Console.

Web-tool results are ephemeral. To persist a page in Library, use **Library →
Import…** and submit its URL; Console does not advertise the retired
`ingest_media` placeholder.

### Stopping & leaving

- **Stop** (appears next to Send while a run is active) stops **this tab's
  run only** — other tabs keep going. Any sub-agents still working for that
  run are cancelled with it, and any of their approval cards still pending
  are withdrawn (denied) at the same time — a sibling sub-agent's card
  belonging to a *different* tab's run is untouched. The partial reply is
  tagged `[stopped]` and a System row records "Response stopped by user."
- Leaving the Console screen is different: after the "Leave Console?" confirm,
  **every** in-flight run is cancelled and every pending or parked approval is
  denied — never approved. Details in
  [Console agent runs are screen-scoped](../index.md#console-agent-runs-are-screen-scoped).

## Common tasks

1. **Run two agents in parallel.** Send a prompt, press **Ctrl+T** for a new
   tab, and send another. The first tab's strip entry shows `●` while its run
   continues; when it finishes unseen the marker flips to `✓` (cleared when
   you visit the tab).
2. **Approve a tool call once.** When the "Approval required" card appears
   with a single pending tool, click its fast **Approve once** button — the
   run resumes immediately.
3. **Deny a risky tool call.** On the card, check the row's badges (e.g.
   "(high risk)"), set its select to **Deny** (or click the fast **Deny**
   button when it's the only row), then **Submit** if needed. The agent
   continues without that tool result.
4. **Check what a finished background run did.** Open its tab, expand the
   **Agent** rail section, skim the step and sub-agent lines, then click
   **View full log** for the untruncated record.
5. **Run a skill.** Type `/skills` to see what's installed, then send
   `$name` (plus any input after it). If it refuses as untrusted, approve it
   in Library ▸ Skills first.

## Keyboard & commands

| Key / command | Action |
|---|---|
| `/skills` | List installed skills in the transcript |
| `/skills <name>` | Points you at the `$name` form (never runs) |
| `$name …` | Run a skill, with everything after the name as its input |
| Enter / Space (on the Approvals chip) | Jump to the pending approval card |

Approval-card decisions are mouse-driven (or Tab to a control and press
Enter). Tab-fleet keys (Ctrl+T, Alt+1…9, Ctrl+K) are covered in
[Sessions, tabs & workspaces](sessions-tabs-workspaces.md).

## Related settings & docs

- **Settings > Console Behavior** — "Max parallel agent runs" (saved as
  `console.max_parallel_runs`; raising it is allowed without limit) and
  "Tool result display cap" (how much of a tool result the transcript
  preview shows). This caps parallel **tabs**, a different knob from
  `[agents] max_live_subagents` below, which caps parallel **sub-agents
  within one reply**.
- **`[agents] max_live_subagents`** in `config.toml` — how many sub-agents
  of one reply may run at once (default 3; `1` disables the fleet). No
  Settings UI switch; see [Parallel sub-agents](#parallel-sub-agents-the-fleet)
  above.
- **Settings > Agents** — create and manage the named agent definitions the
  supervisor can delegate to; see [Named agents](#named-agents) above.
- [Library ▸ Skills](../library/skills.md) — create, import, review, and
  approve skills.
- [MCP](../mcp.md) 🚧 — servers, tools, and permissions.
- [Console agent runs are screen-scoped](../index.md#console-agent-runs-are-screen-scoped)
  — what leaving Console does to runs and approvals.
- [Console](../console.md) — the screen itself.

## Quirks & troubleshooting

- **Inline tool markers vanish after your next action.** The `⚙` / `⤷` / `⚠`
  rows are display-only: the next send, variant swipe, or delete rebuilds the
  thread view and drops them (task-570). The Agent rail summary and **View
  full log** keep the durable record.
- **Tool results are previews.** Transcript markers truncate at the display
  cap; the full text is always in "Full run log — <run id>".
- **No Tools chip before your first send.** The Tools chip in the status
  strip stays hidden until tools are counted (after your first send in a
  session); it then reads "Tools: N ready".
- Tab status markers clear as soon as you visit the tab — a missing `✓` just
  means you already looked.

—
*Verified against dev @ ff435772c — 2026-07-31. Named agents section added
against dev @ 3dd3e7431 — 2026-08-09 (fleet PR-1: driven live — Console
delegated to a real named definition, the transcript showed the
`[researcher]` sub-agent marker, and the reply visibly honored the
definition's instructions; the rest of this page's content unchanged from
the prior stamp).*
