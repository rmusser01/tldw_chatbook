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
- A **Sub-agents** panel appears once the reply has spawned at least one
  sub-agent — see [The fleet panel](#the-fleet-panel--three-states) below
  for its three states (collapsed summary, expanded rows, drilled into one
  child), how to cancel a child, and how its token spend shows up.
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
> clears once you visit that tab. `Qn` is the tab's unsent prompt count.

- A background run that hits a tool approval **parks**: its tab gets a `◆`
  badge and you get exactly one toast — "Agent in <tab> (<workspace>) needs
  approval." Switch to that tab to review the card; parked approvals wait,
  they never resolve themselves.
- A background run that ends also toasts once: "Agent in <tab> (<workspace>)
  finished." (or "failed.").
- The left rail pins a fleet summary line whenever other tabs are busy:
  "N other agents running, M waiting for approval."
- Open session tabs show `Qn`, and open conversation rows show `Queue n`,
  without revealing queued text. Switch to that tab to see its queue shelf.
- A successful multi-prompt drain reports one final background completion;
  intermediate queued turns do not add completion toasts or finished markers.

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
  The cap applies to the **conversation**, not to one reply: a sub-agent
  that is still working when its reply finishes keeps holding its slot, so
  the next message you send can only start as many new sub-agents as there
  are free slots left (see [When a sub-agent outlives the
  reply](#when-a-sub-agent-outlives-the-reply)). Be aware of what the cap
  is *not*: it is per conversation **and** per running app, so two
  conversations can each run the full cap at the same time, and nothing
  caps the total across all of them.
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

#### The fleet panel — three states

The **Sub-agents** panel inside the Agent rail section has its own header
(title + chevron), independent of the Agent section's own collapse state —
it only appears once the reply has spawned at least one sub-agent, and it
reaches a real terminal status (done/error/stuck/cancelled) for each child
**while the turn is still running**, not only after the whole reply
finishes.

1. **Collapsed** (its default state the first time it appears). Just the
   header: "Sub-agents" plus a right-aligned summary — one status glyph per
   child, in spawn order (e.g. `●●✓`), then "N working, M done". "Working"
   means still running; done/error/stuck/cancelled all count toward "done"
   here.
2. **Expanded** — click the chevron: one two-line row per child.
   - Primary line: status glyph, the child's name/task, and — for a child
     still live in the current process — an elapsed segment (`· 12s`,
     `· 1m 4s`). A historical/resumed row (a conversation reopened after a
     restart, or one this process never ran live) shows no elapsed segment;
     see *Known gaps* below.
   - Secondary line: the child's last step, result, or error text, dimmed,
     with the child's measured token spend appended once it finishes — see
     *Token spend*, below. Both are **transient**: they come from the live
     fleet, so when the whole turn ends every row falls back to the sparser
     historical rendering (name and task only). See *Known gaps*.
3. **Drilled in** — click a specific row: the whole Agent section switches
   to that one child's own view (`Sub-agent · <status> (Back)` plus its own
   step lines), and the Sub-agents panel itself is hidden while you're
   drilled in. **Back** returns to the overview. Each row resolves directly
   to its own run — clicking never cycles you through other sub-agent runs
   first.

**Cancel one child.** Focus a still-running row (Tab into the panel, or
click a row then Tab) and press **Delete** — this cooperatively cancels
just that child and withdraws (denies) any of its own approval cards still
pending, the same mechanism **Stop** uses for a whole run; a sibling child
keeps running, untouched. A row for a finished, errored, or already-
cancelled child — or any historical/resumed row — doesn't offer this
gesture at all, since there's nothing left to stop.

**Token spend.** A live child's measured token spend (prompt and
completion combined) appears on its row once it finishes — but only while
some part of the turn is still live; see *Known gaps* for what happens to
the row afterwards. The same figure is folded into the Console cost chip's
token total — the chip's
tooltip breaks it out separately as "Sub-agents: N tok (not priced)". It
never becomes a dollar figure: the measurement is one combined number with
no input/output split, so there is no honest per-model rate to price it
at — an unpriced count was chosen over either fabricating a dollar amount
or discounting the primary transcript's own already-priced total just
because a fleet ran underneath it.

**Known gaps** (filed, not fixed):
- Historical/resumed rows never show elapsed time. The timestamps exist in
  the run database, but the code path that rebuilds a resumed row doesn't
  read them yet (task-15200). The same task also covers `stuck` and
  `cancelled` rows not getting their own status color — they still render
  distinctly by glyph (`⚠`/`✗`), just not by color, so they're
  distinguishable but not visually called out.
- Row detail is transient, not durable. Elapsed time, the secondary line
  and the token count are read from the live fleet, so they exist only for
  a child this app process actually ran. A row for a child that has already
  reached a terminal status is dropped from the panel when your **next**
  message starts, and a conversation you reopen later (or after a restart)
  falls back to the sparser historical rendering — name and task only. A
  child that is *still working* keeps its full row across the reply and
  across later turns; that part changed in fleet PR 3a-1 (task-15200 covers
  restoring the elapsed time and the secondary line from the run database;
  the token count cannot be restored that way at all — `agent_runs` has no
  column for it — so that dimension stays live-only until the schema gains
  one).
- A still-working row's elapsed segment does not tick on its own. It is
  rewritten only when something else repaints the rail — the child's own
  next step, your next message, drilling into the row and back. Between
  those, a sub-agent that has been working for a minute can still show
  `· 1s`. Observed live during PR 3a-1's verification pass; the status
  glyph and the "N working" summary stay correct throughout (task-15664).
- There is no "View all" tail, and expanding the panel does not scroll it
  into view. With a dozen-plus children, or several rail sections open
  above it, you may need to scroll the rail manually to reach the last
  rows (task-15201).

#### When a sub-agent outlives the reply

The supervisor doesn't have to wait for every sub-agent it started. If it
answers you without collecting one, that sub-agent **keeps working after
the reply finishes** — the turn is over for you, not for it.

What this means in practice:

- **The reply you read does not contain that sub-agent's result.** The
  supervisor answered without it, deliberately. Today nothing carries a
  late result back into the conversation on its own: the finished work
  lands in the sub-agent's own run record (**View full log**) and in any
  files it edited, not in a new message. Ask in your next message if you
  want it folded into the thread.
- **It stays visible.** The **Sub-agents** panel keeps its row — glyph,
  name/task, elapsed — after the reply lands and across the turns that
  follow, and clicking that row still drills into that child. The summary
  keeps counting it under "N working". (The elapsed number goes stale
  between repaints; see *Known gaps* above.)
- **It stays cancellable.** Focus the row and press **Delete** (see
  [The fleet panel](#the-fleet-panel--three-states)). The cancel is
  *cooperative*: the child notices between its own steps, so if it is
  waiting on a model response the row can stay `●` for another several
  seconds before flipping to cancelled — whatever it had produced up to
  that point is kept as its result.
- **Stop is different before and after.** Pressing **Stop** during the
  turn that spawned it cancels the whole tree, survivors included — Stop
  stays a kill switch for everything that turn started. Once that turn
  has returned, Stop no longer reaches the child; the row's **Delete** is
  the gesture that does.

**What bounds it.** Three separate limits, none of which is a promise the
others make:

- **Wall clock, per child** — `[agents] child_max_wall_seconds` in
  `config.toml` (default **1800**, i.e. 30 minutes). A background child
  gets its own ceiling rather than whatever was left of the turn's.
  The ceiling is checked *between* the child's steps, so a child stuck
  inside a single long provider call is not cut off until that call
  returns.
- **How many at once** — `[agents] max_live_subagents`, which counts
  survivors from earlier messages against the same cap. Per conversation
  and per running app: N conversations can hold N × the cap between them.
- **Tokens, per run** — each child runs against a run token ceiling of
  its own rather than a slice of the parent's remainder, so a fleet's
  worst-case spend scales with the number of children, not with what the
  parent had left.

**Changes it makes to files.** Change review keeps a survivor's edits in
their own record instead of folding them into whatever turn happens to be
running: a turn that ends with children still working gets a
"✎ A sub-agent edited N files after this turn" row, and a turn that
*starts* while an earlier turn's child is still writing is stamped
"⚠ a sub-agent from an earlier turn was still writing during this turn —
some of these changes may be its, not this turn's". Change tracking diffs
a working tree and cannot tell two writers apart, so it discloses the
overlap rather than implying sole authorship.

**If the app restarts.** A child still running when the app exits cannot
survive the process. The next time Console opens the run database (once
per app run), every row left `running` is swept to **error** with the
result "Interrupted by app restart" — so a killed child shows up as
errored, not silently missing. The sweep assumes one app instance per data
directory; a second instance sharing the same directory would flip the
first's genuinely-running rows.

**Honest limits of the current release:**

- A survivor's token spend reaches the **cost chip only**. Hover it and
  the tooltip breaks the figure out as `Sub-agents: N tok (not priced)`,
  and it is folded into the chip's own token total. It is *not* on the
  assistant message's own usage row and *not* in conversation exports, and
  it is remembered only for as long as the Console screen stays open.
- Nothing wakes the supervisor when the last child finishes, and nothing
  notifies you from another tab or screen.

**Turning it off.** Set `[agents] subagents_outlive_turn = false` in
`config.toml`: sub-agents are then settled at the end of the turn that
spawned them, exactly as before this behavior existed. Setting
`max_live_subagents = 1` removes it too, by removing the fleet entirely.

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
  `[stopped]` and a System row records "Response stopped by user." If that
  tab has queued prompts, they pause. Choose **Retry stopped** to retry the
  stopped turn before continuing, or **Resume next** to keep the stopped turn
  and continue with the next prompt.
- Leaving the Console screen is different: after the "Leave Console?" confirm,
  **every** in-flight run is cancelled and every pending or parked approval is
  denied — never approved. The warning also counts queued sessions and unsent
  prompts. Staying leaves the queue and manager focus untouched; leaving
  clears process-memory queues. Closing one tab uses the same count-aware
  warning for that tab, and quitting the app reports the whole fleet. Details in
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
  of one conversation may run at once, counting any still working from an
  earlier message (default 3; `1` disables the fleet). No
  Settings UI switch; see [Parallel sub-agents](#parallel-sub-agents-the-fleet)
  above.
- **`[agents] child_max_wall_seconds`** in `config.toml` — how long one
  background sub-agent may keep working, in seconds (default `1800`).
  Checked between the child's steps, so it does not interrupt a provider
  call already in flight. No Settings UI switch; see [When a sub-agent
  outlives the reply](#when-a-sub-agent-outlives-the-reply) above.
- **`[agents] subagents_outlive_turn`** in `config.toml` — whether a
  sub-agent may keep working after the reply that spawned it finishes
  (default `true`). Set it to `false` to settle every sub-agent at the end
  of its own turn. No Settings UI switch.
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
the prior stamp). Parallel sub-agents section, concurrent-approval-card
scoping, and the `max_live_subagents` knob added @ d21a91649 — 2026-08-10
(fleet PR2a Task 8: driven live — a single reply spawned two sub-agents at
once, both appeared in the Agent rail with their own handle ids and
results, the reply incorporated both, `sqlite3` showed two terminal child
run rows, and Stop mid-fleet cancelled two live children with zero rows
left `running`; the rest of this page's content unchanged from the prior
stamp). Fleet panel (three states, per-row drill-in and cancel, token
spend) added @ 41cfc5ca4 — 2026-08-11 (fleet PR2b Task 6: driven live —
the collapsed summary showed `2 working, 0 done`, expanding gave one
two-line row per child, a row flipped `● -> ✓` while the turn banner still
read Running, and clicking the second row drilled straight into that
child. Two checks NOT confirmed live and reported as such: Delete-to-cancel
(lost the race against child completion across ~7 attempts; covered by
passing tests) and a durable token figure on a finished row — which
surfaced the transience now documented under Known gaps). "When a
sub-agent outlives the reply", the per-conversation/per-process cap
wording, and the two Known-gaps corrections added @ d87bef16d —
2026-08-11 (fleet PR3a-1 Task 7: driven live against a real Anthropic
model on an isolated scratch profile. Confirmed by pane and by
`agent_runs`: a child was still `running` when its reply rendered
"STARTED" and its primary row read `done`; it reached `done` 31.7s later
with a 6,104-character result; a whole later turn ran start-to-finish
while it stayed `running`, never `superseded`; the Sub-agents panel showed
`● 1 working, 0 done` with the child's row after the turn returned and
again after a later turn; focusing that row and pressing Delete flipped it
to `cancelled` ~18s later with its partial text preserved; SIGKILL with a
child live, then relaunch, left its row `error` / "Interrupted by app
restart"; and hovering the cost chip showed `Sub-agents: 1.8k tok (not
priced)`. One thing found and NOT fixed here: a still-working row's
elapsed segment froze at `· 1s` for a child a minute old until something
else repainted the rail — now documented under Known gaps as
task-15664.)*
