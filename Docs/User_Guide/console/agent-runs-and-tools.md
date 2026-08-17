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
context" rail, the Inspector's status rows, the status chips below the
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

**In the status chips** (below the composer) — "Tools: N ready" counts the
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

An armed approval card does not expire — the run waits for your decision
however long you take. Stopping the run or closing the session withdraws a
pending card; nothing else does. If you'd rather have undecided calls
auto-denied on a clock, set `[mcp] approval_timeout_seconds` in
`config.toml` (seconds; `0`, the default, waits indefinitely — the skill
install and run-script confirm cards follow the same rule).

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

### Interrupted provider tool runs — Resume, Take over, or Discard

For a provider integration that has opted into exact tool continuation, Console
checkpoints private state on the assistant reply that owns it. The checkpoint
can include private model state, tool arguments, and the exact bounded result
returned to the provider. It is not shown in the transcript, search, summaries,
run logs, errors, or usage displays.

Reopening, importing, or syncing a conversation never runs a tool. Instead, an
interrupted card offers recovery actions:

- **Resume** validates the original provider, model, API mode, and normalized
  base URL, resolves the credential from your current Settings/environment,
  and asks for fresh approval before any still-pending call runs. A rotated key
  therefore does not require editing saved continuation data.
- **Take over** is the corresponding explicit action for a checkpoint known to
  have arrived from another device. Sync does not provide a distributed lock;
  confirm the other device is no longer running the turn before taking over.
- **Discard** never executes a tool. It removes the private checkpoint; a blank
  assistant placeholder is removed, while already-visible assistant text is
  kept as ordinary non-resumable history.

A call saved as completed or failed is replayed as recorded and is never
executed again. A call saved as executing is deliberately **ambiguous**: the
side effect may have happened before the result was saved, so Resume is blocked
to avoid repeating it. Discard the interrupted run and start a new turn after
checking the external system.

For local-first Sync v2, each checkpoint change first commits with the message
and its local sync intent, then is projected idempotently into the durable
encrypted outbox. A configured but unavailable or memory-only outbox blocks a
new side effect; Console does not wait for remote acknowledgement. This is
crash recovery, not an exactly-once guarantee across devices, and it makes no
claim about a model provider's own retention or caching.

### Background & parked runs

Tabs with unwatched activity carry a status marker, listed in F1 help:

> Status markers: ● running · ◆ needs approval · ✓ finished · ✗ failed ·
> ◈ sub-agent ended in background — clears once you visit that tab. `Qn`
> is the unsent prompt count.

The `◈` marker is the cross-conversation completion indicator: a
background sub-agent of that conversation finished while you weren't
looking. Unlike the other markers it is **durable** — it survives leaving
Console and even an app restart — and it normally arrives together with
an auto-wake of that conversation's supervisor; see
[When a background sub-agent finishes — auto-wake](#when-a-background-sub-agent-finishes--auto-wake).
A completion that lands while you're on another screen also stages a deep
link: the next time Console opens, it switches you straight to that
conversation's session if it is still open — and when it isn't, the `◈`
marker stays the durable pointer to what finished.

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

### Change review — reviewing a turn's file changes

When an agent turn edits files, the transcript shows a **turn file card**
directly under that turn instead of a plain summary line: a header with the
counts ("✎ Edited 3 files  +92 −468"), then one row per changed file. Press
**Enter** or click a row to expand its diff in place — expanding is
per-row and the diff loads (and is cached) the first time you open it, so
collapsing and reopening a row is instant. The card never mutates
anything; there is no undo or revert control on it.

**`v`** still opens the full **Review** screen for that turn — the same
key as before the card shipped. Reach it from the selected transcript row
or the run inspector's **Review changes** action. Revert-all and the other
destructive actions live only on that screen, behind a confirm, never as a
one-keystroke action in the transcript.

If change tracking failed for one of a turn's roots, that failure shows up
as its own plain-text disclosure row next to the card, not inside it — the
same row the transcript rendered before the card shipped. When *every*
root in a turn failed to track, there is nothing left to count, so the
turn gets no card and no summary row at all — only the per-root failure
disclosures. And if a change spans a nested repository the tracker
excluded, the card and the marker row are both silent about it; that
exclusion is visible only in the full **Review** screen (`v`), which reads
it from the stored snapshot.

**`[console] turn_file_cards`** in `config.toml` (default `true`) is a pure
presentation kill switch: set it to `false` to fall back to the original
plain-text marker row (`` ✎ Edited N files  +A −D — review with `v` ``,
byte-identical to the pre-card behavior). `v` and the inspector's Review
changes action work identically either way.

The "✎ A sub-agent edited N files after this turn" row (see [Parallel
sub-agents](#parallel-sub-agents-the-fleet) below) renders a card too, and
by design it covers the same run's full set of tracked changes — turn and
post-turn windows alike — the same union the `v` Review screen shows for
that run.

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

When the **last child of a conversation's fleet finishes**, the whole
turn's provider-reported usage — the reply's own calls plus everything its
sub-agents billed, survivors included — is re-attached to the originating
assistant message's own usage row, saved with the conversation, and the
chip's unpriced sub-agent line falls back to zero. Until that moment a
survivor's post-turn spend shows on the chip line only.

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
  supervisor answered without it, deliberately. When the sub-agent later
  finishes, its completion **wakes the supervisor**, which acts on the
  result in a fresh, clearly machine-triggered turn — see
  [When a background sub-agent finishes — auto-wake](#when-a-background-sub-agent-finishes--auto-wake)
  below. The finished work itself is always durable in the sub-agent's
  own run record (**View full log**) and in any files it edited.
- **It stays visible.** The **Sub-agents** panel keeps its row — glyph,
  name/task, elapsed — after the reply lands and across the turns that
  follow, and clicking that row still drills into that child. The summary
  keeps counting it under "N working". While only survivors are running,
  a once-a-second tick keeps the elapsed segment advancing on its own and
  paints the row's terminal glyph the moment the child settles; the tick
  stops itself as soon as nothing is live (task-15664).
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

- A survivor's token spend reaches the cost chip's `Sub-agents: N tok
  (not priced)` line **while some child of the conversation is still
  running**. Once the last child finishes, the spend is folded into the
  assistant message's own usage row, saved with the conversation (it
  survives closing and reopening Console), and included in the JSON
  conversation export's per-message `usage` records. Two limits remain:
  quitting the **app** before the last child finishes loses whatever a
  survivor billed after its turn — that remainder is recorded nowhere
  durable — and the plain-text conversation export contains no token
  figures at all.
- The supervisor can act while you are not watching — deliberately. A
  finished background sub-agent wakes its supervisor and notifies you
  from any screen (see
  [When a background sub-agent finishes — auto-wake](#when-a-background-sub-agent-finishes--auto-wake)),
  and as long as Console is open the wake turn fires immediately rather
  than waiting for you to look at it: acting on results without a human
  keystroke is what auto-wake is for. "Open but not watched" is the
  ordinary case here — a dialog, the command palette or the destination
  menu covering Console, or a different session tab in front. You always
  learn of it: the toast fires the moment the sub-agent finishes, and a
  wake that delivers while you're not viewing that conversation leaves
  the `◈` marker set on it until you view the delivered result. When no
  Console is open at all — before the first Console open after a
  restart, or after you have left Console for another screen — the
  completion is staged durably and delivered when Console next opens.
  *(This paragraph previously promised the wake was "never acted on
  invisibly in the background" while you were elsewhere; corrected
  2026-08-14 — the wake does act, and the `◈` marker is the guarantee
  you find out. A second correction the same day, task-16300: it also
  said the wake fires from "a live Console you navigated away from".
  That described a bug, not a feature — navigating away while a dialog
  was open used to leave the old Console screen running invisibly behind
  the new screen. Leaving Console now genuinely closes it, so a
  completion after you leave is staged, exactly as the "Leaving the
  Console screen" note below has always said.)*

**Turning it off.** Set `[agents] subagents_outlive_turn = false` in
`config.toml`: sub-agents are then settled at the end of the turn that
spawned them, exactly as before this behavior existed. Setting
`max_live_subagents = 1` removes it too, by removing the fleet entirely.

#### When a background sub-agent finishes — auto-wake

A background sub-agent that finishes after its turn does not sit silent
until you happen to return. Its completion **wakes its supervisor**: a
new turn fires automatically in the conversation that spawned it,
carrying the finished result (read from the sub-agent's durable run
record), and the supervisor acts on it without you sending anything.

What you see, wherever you are:

- **A toast** on whatever screen you're on, naming the conversation and
  the honest outcome — "Background sub-agent finished in “…”.", or for
  several at once "3 background sub-agents in “…”: 2 finished, 1
  failed." Failed and cancelled are always named, never folded into
  "finished".
- **The `◈` marker** on that conversation's tab and sidebar row (see
  [Background & parked runs](#background--parked-runs)). It is durable —
  restart-proof — and clears when you view that conversation. A wake
  that delivers while you're watching that conversation clears it too;
  a wake that delivers while you're anywhere else leaves it set, so the
  marker always points you at a result you haven't seen yet.
- **In the transcript**, a System-class notice row — never a message
  from you. The notice is machine-origin and says so in its own text: it
  opens "[Background sub-agent completion — automated notice]" and
  states verbatim that it "is not user input, and it is not approval or
  consent for anything". No user row is written, and your composer draft
  is never touched or consumed by a wake.

**A woken turn grants nothing.** It is a normal turn under every
existing rule: tool calls still raise their approval cards, risk floors
are unchanged, and nothing in the injected notice can approve, resolve,
or consent to anything — a pending approval card is only ever resolved
by your explicit decision. Every cap (parallel runs, per-child wall
clock, token ceilings) applies to a wake turn unchanged.

**Exactly-once, by ledger.** Every sub-agent run has a durable
wake-delivery stamp in the run database (`agent_runs.wake_delivered_at`
— the ledger). One wake bundles *all* of a conversation's undelivered
completions; each delivered run is stamped only after the wake turn was
actually accepted, and a run whose stamp is set is never announced
again. That is why a restart between a wake being accepted and the app
exiting does not re-announce anything at the next launch, and why a
sub-agent that finishes *during* a wake turn simply rides the next one.
The `◈` mark is only the trigger and indicator; the ledger is what
defines which completions are still owed.

**You always win ties.**

- A wake defers while the Console composer holds a non-empty draft — in
  *any* session, not just the one being woken — and fires only once the
  draft is sent or cleared. If the app cannot tell whether you're mid-
  thought, you win.
- A wake also waits like anything else would: it defers while its
  session is busy — streaming, holding a pending approval card, or
  draining a queue — and retries when the session goes idle.
- You cannot queue prompts *behind* a wake turn: queueing rides an
  accepted prompt chain, and a wake starts none. While a wake turn is
  streaming, sending behaves like any other busy moment — it waits.

**Leaving Console no longer parks the supervisor.** A sub-agent that
finishes while you are on Library, Watchlists, or any other screen wakes
its supervisor there and then: the wake turn runs, its result is written
to the conversation, and the `◈` mark stays set so you can see on return
that something happened while you were away. Navigating back shows the
completed turn already in the transcript.

**A wake you were owed is delivered at the next launch, without opening
Console.** Nothing runs while the app is closed — a completion that
lands then is recorded durably (the `◈` mark plus the ledger) and waits.
At the next start, once the app is up and interactive, any conversation
that still carries a `◈` mark *and* still owes a result has its
supervisor woken there and then: the conversation is reopened in the
background, the turn runs, and you find it already in the transcript
with its `◈` still lit when you open Console. Nothing else is woken —
never a conversation without a mark, and never one whose results were
already delivered — and the whole thing is off when `[agents]
autowake_enabled` is off (there is no separate launch switch). If you
have never run a background sub-agent, launch does exactly what it did
before: one indexed check that finds nothing.

If Console is your startup tab, the woken conversation opens as another
tab beside the one you landed on; it never switches you away from the tab
you started in.

One case cannot be delivered and is cleaned up instead: sub-agent work
started in a **temporary (unsaved) chat** belongs to a session that does
not survive the app, so there is no conversation left to wake. Its `◈`
mark is cleared at the next launch rather than left pointing at nothing.
Save the chat before starting long background work you want to come back
to.

A wake turn spends model tokens with no window open — the `◈` mark is
your signal that it did.

**A headless wake that needs approval asks you, wherever you are.** When
a woken turn reaches a tool that requires your approval, a toast names it
on whatever screen you're on ("Agent in “…” needs approval to use a tool.
Open Console to review — nothing runs until you answer."), the session
picks up its usual approval badge, and the card is waiting, already
mounted, the moment you open Console. The tool does not run until you
answer it. Nothing auto-approves: navigating away from Console again
denies the request (the same rule as any card you leave unanswered), and
so does quitting the app. If you have set a positive `[mcp]
approval_timeout_seconds`, it still expires the request on schedule —
being away does not buy the request extra time. The shipped default is
`0`, which means no deadline: the request waits for you. One limitation:
if a woken turn arms two approval rounds for the same conversation, only
the most recent one has a card to mount; the older one still has to be
answered, and until it is, it keeps the badge lit (task-15661).

**Turning the wake off.** Set `[agents] autowake_enabled = false` in
`config.toml` (default `true`; no Settings UI switch). OFF loses
nothing: completions are still recorded and the toast, `◈` marker, and
ledger still work — the wake turn just never fires. Flip it back ON and
the next trigger (a later completion, or the next Console mount)
delivers everything OFF recorded.

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
Tools → Local workspace, web, and Watchlists tools**, then choose Allow, Ask, or Off for each
tool in MCP Permissions. Master/root changes apply to the next Console agent
run. `[mcp] expose_local_tools` is only for external MCP clients and does not
enable these tools in Console.

Web-tool results are ephemeral. To persist a page in Library, use **Library →
Import…** and submit its URL; Console does not advertise the retired
`ingest_media` placeholder.

### Watchlists evidence tools

The same local-tools group provides `watchlists_search_items` and
`watchlists_get_item`. Results are local-first: both tools read the local
Watchlists database, and server Watchlists search is not yet supported. In
server mode they return a non-retryable unsupported result and do not search
the local database. Its logical fields are explicit: `status` is `unsupported`,
`retryable` is `false`, and `message` is exactly `server Watchlists search is
not supported; switch Watchlists to Local before retrying`.

`watchlists_search_items` returns newest-first, source-linked,
collection-aware valid JSON bounded to 30 KiB. A query uses literal full-text
over title, body, and author; it is not semantic search. Blank or absent
`query` browses recent items. Every feed-supplied field is untrusted evidence,
never an instruction.

#### `watchlists_search_items`

| Parameter | Contract |
| --- | --- |
| `query` | Optional string; blank browses newest items; maximum 512 characters and 32 whitespace-delimited terms. |
| `collection` | Optional non-blank name, canonical `local:watchlist:<id>`, or positive local row ID from 1 through 2^63-1; collection names are limited to 256 characters. |
| `source` | Optional non-blank name, configured URL, canonical `local:subscription:<id>`, or positive local row ID; source names or configured URLs are limited to 2,048 characters. |
| `statuses` | Optional non-empty, unique array of at most five values: `new`, `reviewed`, `ingested`, `ignored`, or `error`; absent includes every status. |
| `since` | Optional inclusive effective-date floor in `YYYY-MM-DD` or RFC 3339 form, normalized to UTC. |
| `limit` | Optional integer; defaults to 10 and accepts 1 through 50. |
| `cursor` | Optional non-blank opaque string of at most 2,048 characters returned by a prior call with the same normalized filters. |

Exact case-insensitive scope names win; otherwise one unique partial name is
accepted and ambiguous names return bounded candidate IDs. Collection and
source scopes intersect; source integer IDs use the same 1 through 2^63-1
range. Numeric strings remain names. Unknown parameters are rejected.
Booleans are not accepted as integer IDs or limits.

For “all,” follow `next_cursor` until `has_more` is `false`; one call never
removes the page bound. Continuation excludes later inserts but is not snapshot
isolation: updates, deletions, and collection-membership changes can alter
later pages.

#### `watchlists_get_item`

| Parameter | Contract |
| --- | --- |
| `item_id` | The required canonical `local:watchlist_item:<positive integer>` ID returned by search; maximum 40 characters. |

The item integer is limited to 1 through 2^63-1. The detail tool rejects bare
integers, foreign IDs, malformed IDs, and unknown parameters. Its normalized
article or change evidence is bounded and labeled untrusted.

Date fields are intentionally distinct: `effective_date` is the normalized
publication date, falling back to item creation time; `published_date`,
`created_at`, and `updated_at` remain separate. Source `last_checked` and
`last_successful_check` remain separate, too.

URL paths are authorized Watchlists metadata under the same explicit tool
permission; userinfo, query, and fragment are removed from every returned URL.
Only absolute HTTP(S) URLs with a host are returned. In Console, Ask can show
an approval card. External MCP additionally requires `[mcp]
expose_local_tools` to be true and each per-tool permission must be Allow; Ask
is refused because a headless client cannot show that card. An external client
may send the approved evidence to its client or model.

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
  every in-flight **turn** is cancelled and every pending or parked approval is
  denied — never approved. One thing survives the leave: a background
  sub-agent that already outlived its turn **keeps running** — its result
  lands durably, you get the completion toast + `◈` marker wherever you
  are, and the staged wake is claimed when Console next mounts (see
  [auto-wake](#when-a-background-sub-agent-finishes--auto-wake)). The next
  Console mount reports both fates honestly: "N agent runs were cancelled
  when you left Console." and/or "… sub-agents kept running in the
  background when you left Console — you'll be notified as they finish."
  The warning also counts queued sessions and unsent
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
- **`[agents] autowake_enabled`** in `config.toml` — whether a background
  sub-agent finishing after its turn wakes its supervisor (default
  `true`). `false` still records every completion (toast, `◈` marker,
  delivery ledger); only the wake turn is suppressed, and flipping back
  to `true` delivers what was recorded. No Settings UI switch; see
  [When a background sub-agent finishes — auto-wake](#when-a-background-sub-agent-finishes--auto-wake)
  above.
- **Settings > Agents** — create and manage the named agent definitions the
  supervisor can delegate to; see [Named agents](#named-agents) above.
- **`[console] turn_file_cards`** in `config.toml` — whether a turn's file
  changes render as the expandable card described in [Change
  review](#change-review--reviewing-a-turns-file-changes) above, or the
  original plain-text marker row (default `true`, card on). No Settings UI
  switch.
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
else repainted the rail — task-15664, fixed in fleet PR 3a-2 by the
survivor tick described above.) Auto-wake, the cross-screen completion
indicator, the kill switch, user-wins-ties, and the restart story
verified @ e38e62a2f — 2026-08-13 (fleet PR3a-2 Task 7: driven live
against a real Anthropic model on an isolated scratch profile. Confirmed
by pane and by both databases: a survivor's completion woke its
supervisor with the machine-origin System notice (verbatim "not user
input" marking, fenced result, truncation note for a long result) and a
reply that referenced the child's result, `agent_runs.wake_delivered_at`
stamped once per delivered run; the wake fired while another Console
session was in view; a completion landing while on Library toasted there
by conversation name, staged durably (mark + NULL stamp), and the wake
ran at the next Console mount with the mark cleared after delivery;
`autowake_enabled = false` recorded everything (toast, `◈` badge, mark,
owed ledger row) and fired nothing over a watched quiet window, and the
owed wake was delivered after flipping back on; a non-empty composer
draft held a due wake back for the full 50s it existed and the wake fired
seconds after the draft cleared; SIGKILL with a wake owed left the mark
and the NULL stamp in place, relaunch swept the mid-run child to `error`
/ "Interrupted by app restart", and the owed wake was delivered exactly
once with no previously-stamped run re-announced. Found and NOT fixed
here, filed as follow-ups: a wake turn's UI can go stale until the
session is next viewed (stuck `●` on the tab, an unpainted reply row, a
misleading "finish provider setup" composer state — the delivery itself
was always correct and durable); one deferred wake's notice labeled a
`done` child "running"; and after a restart the staged wake's `◈` badge
did not render on the sidebar row, with delivery waiting on the next
retry trigger rather than on opening the conversation.) The off-view
delivery contract (the honest-limits correction above and the `◈`
clear-semantics rewrite) verified @ 9144b235e — 2026-08-14
(wake-integrity arc, tasks 15970/15971: driven live against a real
Anthropic model on an isolated scratch profile. Confirmed by pane and by
both databases: a draft typed with real keys DURING the spawning turn
held a due wake back for the full ~90 seconds it existed (child `done`
with a NULL `wake_delivered_at` stamp throughout) and clearing the draft
delivered it within ~2 seconds; a completion landing while the
conversation was not in view was DELIVERED immediately (stamped while a
palette covered Console and another session tab was active) and left the
`◈` marker set on the conversation's tab, which cleared — mark row gone —
the moment the conversation was activated and viewed; and restart staging
still held: SIGKILL with a wake owed left mark + NULL stamp in place, the
relaunch rendered `◈` on the sidebar row before the conversation was
opened, and one click on that row delivered the owed wake exactly once,
stamped ~2s later. Every one of the five sub-agent runs in the session's
ledger ended stamped exactly once.) The "navigated away from" clause in
the honest-limits bullet corrected @ HEAD — 2026-08-14 (task-16300,
documentation-only for this page: navigating away from Console under an
open dialog used to leave the old Console screen resident and running,
and the bullet had described that leak as intended behavior. Navigation
now unmounts the outgoing screen, so leaving Console stages. Pinned by
`Tests/UI/test_screen_residency.py`; the live off-view evidence above is
unaffected — it was gathered with a palette covering an open Console and
a different session tab active, not by navigating away.) The "Change
review" section added @ 6fc069cc0 — 2026-08-15 (turn file card feature:
docs-only pass against shipped code and the whole-module test run —
`Tests/Chat/test_console_turn_file_entries.py`,
`Tests/UI/test_console_turn_file_card.py`,
`Tests/UI/test_console_turn_file_card_factory.py`,
`Tests/UI/test_console_native_chat_flow.py`, and
`Tests/UI/test_console_internals_decomposition.py`, 457 passed — not an
interactive live-tmux walkthrough of the card itself. The card is a pure
presentation layer over TASK-1972's existing change-review subsystem; the
`[console] turn_file_cards` kill switch reverts to the pre-card plain-text
marker row byte-for-byte, confirmed by
`test_summary_row_stays_plain_marker_when_disabled`.)*
