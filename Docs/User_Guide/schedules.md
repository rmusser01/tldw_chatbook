# Schedules — When scheduled tasks fire and recurring questions run

## Recurring Watchlists briefings

An “every 24 hours” Watchlists briefing uses an interval of 86,400 seconds from its saved schedule; it is not a promise to run at local midnight. The Console tool reports whether the scheduler acknowledged the reload, and the saved job appears in Settings because both surfaces project the same durable schedule.

Recurring briefing model resolution is independent of the currently open chat.
The scheduler first uses the collection's persisted briefing provider/model
preset. Without one, it reads the persisted `chat_defaults` provider and model
at run time; if that model is empty, it uses the configured model saved for that
same provider. Changing a conversation's model does not silently change future
briefings; edit a persisted briefing or provider setting when you want the
recurrence to use a different model.

If the cadence is saved while no usable briefing provider/model route is
available, the save remains successful and still requests a scheduler reload.
Its receipt reports `briefing_route_ready: false` and directs you to Settings;
no briefing model call is attempted until a persisted route is available.

If creation is accepted but no completed briefing appears, inspect the exact briefing receipt before editing or duplicating the schedule.

## What this screen is for

Schedules controls when scheduled tasks fire and recurring questions run. It is a
**single surface**: one list of everything scheduled, a detail pane for
whatever is highlighted, and an inspector — no tabs. A one-line
scheduler-liveness indicator sits above them, and a status strip (sync
bar + Conflicts count) runs along the bottom — see "Status strip",
below. Views that are not about one selected row — results, a
definition's run history, sync conflicts — open **over** the list and
close with **Escape**, rather than living in a tab bar you have to
remember to visit.

## The unified list

The Schedule Queue lists **both** reminders and recurring-question
automations in one table, spanning both owners (this device and any
connected server) — one place to see whether something is scheduled.
A chip row above the table (**All · Active · Paused · Completed**)
narrows it: **Active** is everything still armed to run (an enabled
reminder or a `configured` automation, including one mid-transfer to the
server — it keeps running locally until the server actually accepts
it); **Paused** is anything disabled or explicitly paused; **Completed**
is a fired one-time reminder or an archived automation, kept out of the
default **All** view so it stays uncluttered (a one-time reminder that
has already fired stays under **Completed** even if you re-enable it —
re-enabling gives it no new run time; edit its schedule to arm it
again). Each row shows a status
glyph (`○` recurring, `▶` one-shot, `⏸` paused, `✓` completed), the
title with the same owner/transfer-badge suffixes described below, and
a subtitle line (the next run with its relative time, or
"— (paused)"/"— (disabled)" when nothing will fire, then the schedule
summary; the next run comes first so a narrow queue cuts the summary,
not the time). An automation row
with unread results carries a bold unread dot after its title.

Highlighting a row opens its detail pane beside the list — a reminder's
or an automation's, routed by the row you are on. **Both kinds are
fully actionable from here**: edit, run now, pause/resume and move
between owners all work on an automation row exactly as they do on a
reminder. The few genuinely reminder-only verbs (delete, mark, enable/
disable) say so when pressed on an automation row ("Automations don't
support delete — use the actions in this automation's own detail pane")
rather than doing nothing.

Watchlist checks and briefings are not in this list — they have their
own home in Watchlists' **Sources** pane, whose **Next check** column
shows the same "when will this run again" signal the Queue used to
project for them (see the [Watchlists guide](watchlists.md)).

Every row's relative next-run text ("in 25m", "overdue 2h") stays
current on its own: the visible Queue rows repaint once a minute
without reloading the list, so a bucket a row was in a minute ago
("in 1h") reads correctly once it crosses into the next one ("in 59m")
even if you never touch the screen. Switching away to another screen
pauses that repaint; coming back refreshes it immediately so nothing is
left stale from the time it was hidden — and re-reads the automations if
anything changed them while you were gone.

## The rail

The list pane's header is a rail:

- **Create ▾** opens the chooser described below (also **n**).
- **Mark all read** appears only once an automation result somewhere in
  the queue is unread, and clears every one of them in one press (also
  **a**).
- **Results** opens the results inbox over the list — always available,
  since browsing already-read results is useful too. Its label carries
  the unread count ("Results (3)").

The rail and the per-row unread dots track results as they change: a
result arriving from the server makes them appear on their own, and
marking everything read inside the results view clears them here too.
Below the chip row, the filter box narrows the list by title or
question/body text as you type (**/** focuses it) — not by status words
like `missed`, which the unified list dropped along with the Status/Type
columns.

## Status strip

A status strip runs along the bottom of the screen: the sync bar
(Local/Server, last pull/push — see "Sync bar honesty", below) on the
left, and a **Conflicts** button on the right showing the current
owner's scheduled-task conflict count ("Conflicts (2)", or plain
"Conflicts" with none). Clicking it opens the conflicts view over the
list; **Escape** returns, and the count re-reads itself in case you
resolved something. At narrow terminal widths the sync bar's timestamps
drop to save space; the owner buttons, any sync error, and the
Conflicts button stay visible either way.

## Getting there

- Press **Ctrl+7**, click **⌃7 Schedules** in the nav bar, or press
  **Ctrl+P** → "Tab Navigation: Switch to Schedules".

## Keyboard map

Every key acts on the highlighted row unless noted. Up/Down move the
list cursor; inside a detail pane, Up/Down move between its editable
rows.

| Key | Action |
| --- | --- |
| `1` `2` `3` `4` | Show **All** / **Active** / **Paused** / **Completed** |
| `f` | Cycle those four filters in order |
| `/` | Focus the filter box |
| `n` | Create — opens the Reminder / Recurring question chooser |
| `p` | Pause or resume (a reminder's enable state, an automation's lifecycle) |
| `m` | Open the highlighted row's **Runs on** dropdown — the move-between-owners flow. Below 84 columns this opens the row's full-screen detail first, since the docked pane is hidden there |
| `r` | Mark the highlighted automation's unread results read |
| `e` | Edit in full — opens the row's own edit form |
| `space` | Enable or disable (reminders) |
| `d` | Delete (reminders) |
| `x` | Mark or unmark the row for a bulk action |
| `Esc` | Leave the filter box (when it has focus) — otherwise clear marks, or, in a view opened over the list, close it |
| `s` | Sync now |
| `a` | Mark every unread automation result read |

**Run now is not a key.** It is a button in the detail pane of the row
it belongs to, so a dispatch always names what it is about to run. The
same is true of the results, run-history and conflicts views: they are
opened from the affordance that describes them, not from a global
shortcut.

Keys that only make sense for one row kind say so when pressed on the
other, rather than silently doing nothing.

## Narrow terminals

The screen has a floor of **80×24** and degrades in two steps:

- **Below ~118 columns** the inspector yields (it summarises what the
  detail pane already shows). A one-line notice under the queue says so.
- **Below 84 columns** the detail pane yields too, and the list takes the
  whole width. **Enter on a row opens that row's detail full-screen**,
  with everything it normally offers — the in-pane row editors, the
  Runs-on transfer dropdown, Pause/Resume, Run now, Edit in full. It is
  a fresh view of the same pane, not a moved one, so the list keeps its
  place behind it. **Escape** closes it (and, while a row editor is
  open, Escape closes *the editor* first). It stays on the row you opened
  it for: if that row leaves the queue, it closes with "'*Title*' is no
  longer in the queue." The four filter chips
  collapse into one **Filter: …** button that cycles them, and the rail
  keeps its buttons on a single row — Create ▾ and Results, plus
  **Mark all read** whenever there is anything unread to mark (that
  button is hidden otherwise, at every width).

At the floor every operation is still reachable: create through **n** or
**Create ▾**; edit, pause/resume, run now and transfer inside the pushed
detail; results from the rail; conflicts from the status strip.

## Sync bar honesty

Pressing **s** reports what actually happened: "Sync completed." only
when a pull or push was recorded (the Last pull/push timestamps update),
and "Sync finished — nothing was pulled or pushed." otherwise. With a
Local owner and no scheduling server connected, the bar collapses to a
single line ("Local schedules — no scheduling server connected; sync is
off"), and the **Clear** button only appears once a sync error exists.
With no server to reach, **s** syncs nothing and says so: "Local only —
nothing to sync (no server connection).", or "Server configured but not
reachable — nothing to sync."

A sync cycle runs several independent phases (reminders, then automation
review/definition pushback, definitions pull, results pull); one of them
failing never overwrites the others' honest report. If your reminders
synced cleanly but, say, an automation results pull hit a stale server,
you get the success toast **and** a separate "Sync completed with
issues — …" notice naming that one phase — never a blanket "Sync failed"
that would make a genuinely successful push look lost.

## Scheduler liveness

Below the sync bar, a one-line **scheduler liveness** indicator distinguishes
three states that used to look identical: *not started* (the loop has never
ticked on this machine), *live* (with the age of the last tick, e.g. "live ·
last tick 12s ago"), and **STALLED** — the loop has not ticked in several
poll intervals, so reminders have silently stopped; a stall shows the last
error the loop hit, if any. Staleness is judged against your configured poll
interval, so a deliberately long interval is not mistaken for a stall. The
signal is durable (a small heartbeat file), so a stalled or dead loop is
distinguishable from an idle one even across a restart. The line refreshes
every 5 seconds.

When the Console's **emergency stop** is active, the line adds "scheduled
dispatches are held by the emergency stop (/emergency-stop clear in
Console)". The scheduler records that on each tick, so the note appears (and
goes away after a clear) at the scheduler's next tick, within one poll
interval. Held tasks are not lost: they run once the stop is cleared.

## Preflight checks

A handler can declare a **preflight** check that runs immediately before a
task fires — verifying, say, that a watchlist's source still exists or a
briefing's provider key is still configured. A failed preflight is a
distinct, legible outcome (shown as *preflight failed* in the run history,
not confused with a handler crash), never runs the handler, and keeps the
task visibly needing attention. Repeated preflight failures compose with the
incident grouping above — you're told once per condition, not once per
occurrence — and the check is time-bounded so it can't wedge the scheduler.
Handlers without a preflight fire exactly as before.

## Repeated-failure incidents

A briefing that fails the same way over and over no longer floods you with
identical notifications. Repeated failures of one watchlist's brief that
share a normalized error signature (timestamps, ids, paths, and numbers are
stripped so cosmetic variation doesn't defeat grouping) are grouped into a
single durable **incident** — only the first failure of a signature alerts;
the rest are recorded silently. A different error opens its own incident. A
successful run resolves the incident, so a later recurrence alerts afresh.
Incidents survive restarts, and acknowledging one suppresses its
notifications only — it never disables the task or removes it from the queue. The Task Detail pane lists a task's open incidents and offers an **Acknowledge incident** button when one is alerting.

## Run history

The **Task Detail** pane now shows a durable **Recent runs** list for
reminders and briefings — not just the latest outcome. Each dispatch
records its own row (start time, status, and any error), so run N-1 is
recoverable rather than overwritten. History is capped per task (the newest
50 runs are kept), a run interrupted by an app exit is reconciled to
*failed* on next start rather than left hanging, and server-scoped tasks
keep their history server-authoritative (per ADR-077) rather than
duplicating it locally. The existing missed-fire accounting is unchanged.

## The reminder detail pane — grouped fields

A reminder's detail pane shows its body text in a card, then its
fields as three labeled groups instead of a flat list, matching the same
label-left/value-right row style used throughout the pane (a reminder with
no body text shows no card):

- **Details** — `Runs on` (the owner: "This device" for a reminder this
  device runs, or the server's own id for one the server runs, with the
  transfer badge text appended while a move is in flight, e.g.
  "This device (Moving to server…)"). The automation detail pane words
  this row exactly the same way.
- **Frequency** — `Repeat`, `At`, `Timezone`, and `Notifications` (always
  "Inbox + toast" — every reminder dispatch writes an inbox row and
  attempts a toast; there is no per-reminder channel setting).
- **History** (collapsed by default — click its title to expand) —
  `Last fire` (the last run's time and outcome) and a `Run history`
  pointer row to the **Recent runs** list, which stays exactly where it
  was below.

Each group's title is keyboard-focusable: Tab reaches it and Enter
toggles the group, so collapsing and expanding never needs the mouse.
That does put three focus stops ahead of the pane's action buttons.

A watchlist or briefing projection (not a reminder) still shows the older
Type/Schedule rows unchanged — the grouped layout is reminder-specific.

### Editing rows in place

Click a row (or Tab to it and press Enter) to edit it where it sits,
instead of opening the create/edit form. The dimmed `▾` next to a row's
value marks it as editable; a row without one has no single-field write
target and only changes through **Edit in full…** or the create form.

- **Repeat**, **Timezone**, and `Runs on` open a dropdown pre-selected to
  the current value; picking a new one commits immediately. **At** opens
  a text box; press Enter to commit the typed value (the same forgiving
  local-time parsing the create form uses).
- **Escape** closes an open editor without saving — a plain cancel, not a
  commit of whatever is currently showing.
- A bad **At** value (unparseable text) commits nothing: the row shows
  the error inline underneath and restores the last-saved value rather
  than left holding your typo.
- Only one of **Repeat** or **At** is ever editable at a time — a
  one-time reminder's Repeat row and a recurring reminder's At row have
  no sensible single-field target for the other schedule kind, so they
  stay read-only rows until you switch kind via the full edit form.
  `Notifications` never opens an editor for a reminder — see the
  Frequency bullet above.
- A row that is locked (a transfer is in flight, or the row's owner is a
  dormant server-release copy waiting to arm) still responds to a click:
  it shows the lock reason inline instead of silently doing nothing.
- Selecting a different row in the list closes any editor you left open
  and clears any inline error under it — an edit always commits against
  the row you opened it on, never against whichever row the pane happens
  to be showing by the time you press Enter.
- An automation whose pending **move** (to the server, or back to this
  device) has not reached the server yet refuses an in-place edit and
  says which change is waiting ("A move to the server is waiting to sync
  — edit this automation once it lands."). An edit and a move share one
  queue slot, so saving the edit would throw the move away without
  telling you. Sync (or reconnect), then edit. A pending **Pause** or
  **Resume** does *not* refuse an edit — lifecycle changes queue
  separately, so you can pause an automation while offline, edit it, and
  have both land on the next sync.
- `Runs on`'s dropdown is the transfer flow described under "Moving a
  task between this device and the server", below — picking the other
  owner runs the refusal check and confirmation dialog. Once a move is
  queued or failed, its own **Cancel transfer** / **Retry transfer**
  buttons appear directly under the row (the dropdown is unavailable
  while a move is in flight — that is the locked state above). This row
  is the **only** transfer surface: the pane's older Move/Cancel/Retry
  buttons, and the automation-only transfer keys that went with them,
  are retired.

### Duplicate and View runs

A second row of buttons sits under Edit/Acknowledge/Run now/Enable/
Disable/Delete:

- **Duplicate** creates a new local copy of the task, named "*Title*
  (copy)" — same schedule, body, and linked item, but a fresh id, no
  transfer state, and no borrowed run history. It always lands on this
  device, even when the task you duplicated is server-owned: duplicating
  is a plain new draft, not an implicit move. It is disabled (with the
  reason shown under the button, same as Edit/Enable/Disable/Delete)
  while the task is mid-transfer, for the same reason those four are.
- **View runs** scrolls the pane down to the **Recent runs** list — a
  reminder has no separate run-history screen of its own, only that
  inline section, so this is a shortcut to it rather than a new view.

There is no **View results** button here: results (`automation_results`
rows) are a recurring-question concept a reminder has no equivalent of,
and a permanently-disabled button plus a permanent explanation for it
was standing weight this pane doesn't need — the automation detail
pane's own **View results** (below) covers the action wherever it
actually applies.

The automation detail pane carries **Duplicate**, **View runs**, and
**View results** — see "The automation detail pane", below, for what
they do there.

## Creating a scheduled task

Press **n**, or click **Create ▾** in the rail header. Both ask which
kind of task you want — **Scheduled task…** or **Recurring question…** —
since a recurring question is a different kind of definition, not just
another schedule shape. The form scrolls when the terminal is short; the
live "Runs: …" preview, validation, and Save/Cancel stay pinned at the
bottom while you edit. Closing a form that has unsaved edits asks first
(**Discard changes?** — **Keep editing** returns to the form), and so
does **Ctrl+Q** while the form is open (**Discard changes and quit?**).

Every create/edit form also has a **Runs on** selector — **This device**
or **Server (\<id\>)** when a scheduling server is connected — defaulting
to whatever owner the Schedules screen is currently showing. Choosing the
server writes the task there directly when the server is reachable, or
authors it locally and queues it to sync up on the next successful sync
when it is not (the sync bar and the task's own state say which
happened -- a recurring question in that state is listed as
*\[\<server id\> · pending sync\]*, and Run now/run history say so
rather than asking the server about a task it has never seen). If the server refuses the save outright rather than being
unreachable, the form says so and nothing is queued -- retrying it later
would only hit the same refusal. An existing reminder's owner is fixed
once created — the selector shows it but is not editable — moving a task
between owners is a separate action, not part of editing.

- **One-time**: type a plain local time like `2026-08-28 09:00` — no
  offset needed. It is interpreted in your machine's timezone and the
  preview confirms the interpretation ("Runs: 2026-08-28 09:00 PDT (your
  local time)") before you save. Full ISO-8601 with an offset is still
  accepted and kept as written.
- **Recurring**: pick a frequency preset — *Every day at…*, *Every
  weekday at…*, *Every Monday at…* (each with an editable 24-hour time of
  day), or *Every hour* — no cron required. *Custom cron…* reveals a raw
  5-field cron expression with the same live preview.
- **Timezone** is a selectable list, not free text: your machine's zone
  is preselected and listed first, followed by common zones and any zone
  your existing tasks already use.

## Next Run readability

Next Run pairs the absolute time with a relative form: the detail pane
shows "2026-08-31 14:30 UTC (in 2d)" and the queue column a shorter
"2026-08-31 14:30 (in 2d)". Overdue times read "(overdue 2h)"; times
within a minute read "(due now)".

A recurring task keeps showing its next run after it has run. Its badge
reads **Completed**, **Missed** or **Timed out** for the last run, and
Next Run still names the next occurrence. A one-time task that has fired
has no next run and reads a bare **—** (see Disabled tasks below). Edit
its schedule to give it a new time and, once it is enabled, Next Run
shows that time under the same **Completed** badge; until you enable it,
it reads **— (disabled)**.

## Bulk actions — marking rows

Press **x** to mark or unmark the highlighted row (●). While any rows
are marked, a legend under the queue states the count and the keys:
"2 marked — space toggles all · d deletes all · esc clears". The ◇
missed-while-away glyph also gets an on-screen explanation ("◇ = ran
late (dispatched after its scheduled time)") whenever a visible row
carries it.

## Disabled tasks

Press **space** on a highlighted task (or the **Disable** button in the
detail pane) to disable it. A disabled task shows the text status
**Disabled** in the detail badge, the `⏸` glyph in the queue row (the
unified list conveys status via glyph, not a separate Status column),
and its Next Run reads **— (disabled)** instead of a concrete time it
will not honor, in both the detail pane and the queue row's subtitle.
Enabling it restores the recorded last outcome and the real next run.

This does **not** cover a one-time reminder that has already fired:
dispatching it also disables it and clears its next run internally, but
the screen reads that specific shape as **finished**, not disabled —
the detail badge and Next Run agree with the queue row: **Completed**,
Next Run **—**, the `✓` glyph, and the **Completed** chip, everywhere at
once. Every other disabled reminder reads **Disabled** /
**— (disabled)**, whether or not a next run is still stored underneath:
for example one you disabled yourself, a fired one-time reminder you
gave a new time and have not enabled yet, or an overdue one-time
reminder the server parked disabled when you moved it there (see Moving
a task between this device and the server below).

## Ran late — what happens to overdue reminders

Reminders only fire while the app is running (they execute locally, even
with no server configured). If the app is closed — or asleep — when a
reminder's scheduled time passes, the reminder is **not replayed**: when you
next open the app, the overdue occurrence fires once, immediately, and the
task records honestly that it was late.

Other causes produce the same "ran late" state, which is why the notice no
longer claims the app was closed. The scheduler runs each poll's due
handlers one after another, so a slow handler (a watchlist check may run up
to its 300-second execution timeout) holds the loop and pushes the *next*
poll — and everything due by then — past the grace window. A machine that
sleeps or suspends with the app still open does the same thing without any
handler being slow. The app log names which it was: `busy` when the previous
poll's handlers account for the delay, `stalled` when the scheduler was up
but nothing it ran explains it, and `away` when the scheduler genuinely was
not running at the scheduled time.

- **One-time reminders** fire once, late, and then complete as usual. Their
  detail pane shows "Ran late: the \<scheduled time\> occurrence dispatched
  well after its scheduled time".
- **Recurring reminders** fire once for the overdue occurrence and count
  how many earlier occurrences were skipped ("2 earlier occurrence(s) were
  skipped, not replayed") — the schedule then continues from the current
  time, not from where it left off. Skipped occurrences are counted and
  surfaced, never re-run.
- **The queue marks late tasks with a ◇ glyph** before the title, so a
  glance at the queue shows what fell behind. The queue filter
  searches title and body text only (it no longer matches status words
  like `missed` — the unified list dropped the old status/type keyword
  search along with the Status/Type columns); search for the task's own
  title to find it, or scan for the ◇ glyph.

This state describes the **last** dispatch and heals itself: the next
on-time firing clears the marker and the notice.

**"Missed" vs "Ran late".** A task whose *status* reads **Missed** had a
dispatch that ran and *failed* (its handler raised — it tried and errored).
"Ran late" (the ◇ marker and detail notice) means the scheduled time passed
without the work being dispatched — because the app was away, or because
the scheduler was busy with an earlier task in the same tick — and it then
fired late. A task can be one, the other, or neither; neither implies the
other.

### Tuning the grace window

A dispatch counts as "late" only when it lands more than
`missed_fire_grace_seconds` after its scheduled time (default **60**
seconds — twice the default 30-second scheduler poll, so a dispatch that
lands within one poll of its schedule is on time, not missed). Raise it
under `[scheduling]` in `config.toml` if your machine sleeps briefly and
you don't want those counted; the same section's
`scheduler_poll_interval_seconds` controls the poll itself.

Reminders you create or edit while the app is running reach the live
scheduler queue immediately — they do not wait for the periodic reload,
and creating one for a time that already passed reports as missed-while-
away only if it genuinely outruns the grace window.

## Run now — dispatch immediately

Click **Run now** in the highlighted row's detail pane to dispatch it
right away, without waiting for its schedule. (There is no Run-now key:
the button names what it is about to run, a global shortcut would not.)
This is a **real dispatch** through the same path the scheduler uses —
not a preview:

- A **recurring** reminder's next occurrence is computed from the moment
  you run it, exactly as a scheduled firing would.
- A **one-time** reminder is consumed: it completes and disables itself.
- A reminder whose last dispatch **failed** (status *Missed* — it ran and
  raised) offers the same action labeled **Run now (retry)**: its retry
  is simply another real dispatch.

**Disabled reminders can be run manually** — manual intent outranks the
schedule — and running one does **not** re-enable it: the row stays
disabled, and the toast says so ("…ran now (still disabled)."). If you
then enable it, it will not double-fire: the manual run already advanced
the schedule.

A reminder that is both queued and run manually dispatches exactly once —
the pending scheduled occurrence is claimed by the manual run rather than
firing twice.

**Server-scheduled reminders cannot be run from here.** When the Schedules
owner is a connected server, reminders synced to it carry a server scope:
the **server** executes them on schedule and delivers the notification
through the server feed, and the local scheduler never fires them (no
double execution — one owner, one executor). Run now on one refuses with
a toast saying so. This is the single-owner execution rule the
server-offload design is built on; local-owner reminders are unaffected.

## Moving a task between this device and the server

Every move — reminder or automation, either direction — goes through the
**`Runs on`** row in that row's detail pane. Open its dropdown (click it,
press Enter on it, or press **m** with the row highlighted) and pick the
other owner. One surface, one flow: the older Move/Cancel/Retry buttons
and the automation-only `M`/`m`/`y`/`k` transfer keys are retired.

A move that cannot proceed is refused inline under the row with the
reason — no server connection, no server identity, a transfer already
running, or, moving a `recurring_question` mirror to this device, the
same local-health reason the automation's own pane reports.

"Configured" and "reachable" are two different answers and the screen
says which one it means. A profile with no scheduling server at all
refuses with *"No server connection is configured."*; a server that is
configured but has not answered refuses with *"The configured server is
not reachable right now."* Nothing is offered on a hope: the answer
comes from a real round trip (the same capabilities probe the sync
layer uses), it is re-taken when you change the owner or press **s**,
and until one has succeeded the create forms' `Runs on` dropdown lists
only **This device** — the server option is omitted rather than offered
and then refused.

Picking the other owner opens a confirmation listing anything worth knowing before
you commit: an imminent or already-passed one-time run time ("server
behavior this close to run time is unverified"), and, for a reminder,
that its per-run timeout is local-only and will not transfer.

What the server actually does with an already-passed one-time run was
measured on 2026-09-02: it **accepts the transfer and parks the task
disabled** — no catch-up run, no error, no rejection — and leaves it
that way. This screen mirrors it as a Disabled row carrying the
missed-while-away marker. The warning above is still worth reading,
because a run time only *seconds* away is a different race, but an
overdue one-time reminder does not fire twice by being moved. (Note the
device keeps ownership until the server accepts, so an overdue task will
usually have already fired here first.)

Confirming
a **local → server** move only queues it — the task keeps running on
this device until the server actually accepts the transfer, the toast
says so, and the queue row shows "(Moving to server…)" for as long as
that stays true. A **server → local** move creates a dormant local copy
immediately (shown as "(Waiting for server release)"); it stays inert
until the server's release is acknowledged, at which point it arms and
starts running here.

When Schedules is showing a server owner's queue, each row's title also
carries a "(server: \<id\>)" owner suffix — the same wording the results
view uses for its own rows. It is hidden whenever the window is narrow
enough to trigger the compact layout, so a squeezed terminal never
truncates a title to make room for it.

**While a move is in flight the task is read-only.** Edit, Delete and
Enable/Disable are refused with the reason stated for a queued
local → server move, one already sent, and a
dormant server → local copy. This is not fussiness: the move takes a
snapshot of the task when you start it, so an edit made afterwards would
be sent nowhere and then overwritten by the first sync. Cancel the
transfer first, edit, then move it. A move the server *rejected* is not
in flight — that task is editable again, and retryable.

**Cancel transfer** is available on any in-flight state except one
already sent to the server (too late by then — the button says so and
suggests moving it back once it lands). Cancelling drops the queued
transfer, so nothing further is sent: an unattempted local → server
queue and a definitively failed one both simply stay here. Note the one
case cancel cannot undo — a server → local release whose delete already
reached the server but whose acknowledgement was lost still looks
"waiting for release" locally, and cancelling then removes the dormant
copy without bringing the server's task back. That is why the
confirmation says "nothing further will be sent" rather than "nothing
happened".

**Retry transfer** appears only alongside a local → server move the
server definitively rejected; the stored reason is shown beside the
button, and retrying resubmits the same task.

A queued move can also be stranded rather than rejected — you point the
app at a different server, or drop the connection entirely, while a
local → server transfer is still waiting. That queued mutation can never
be delivered, so the next time the Schedules screen opens (this check
does not need a reachable server, or even a configured one — it is a
local check, so it also covers the connection-dropped case, not only a
sync-eligible server switch) settles it to *failed* with the reason
*"The server this move was queued for is no longer configured."* rather
than leaving the row reading "(Moving to server…)" with nothing behind
it. The task is editable again and carries the usual **Retry
transfer** / **Cancel transfer** pair.

**A disabled server reminder stays disabled** when it is released to this
device — the release moves the task, not its on/off state.

**A recurring question's run/result history does not follow a
server → local release.** A local → server move keeps the same task
identity (this device's row just gains a server link), so its run and
result history stays visible under either id, before and after the
move. A server → local release is different: it creates a brand-new,
independent local row rather than converting the existing one, so the
findings and run history the automation built up while server-owned
stay attached to the now-archived server-owned row and do not carry
over to the new one. The new row starts a clean history from the
moment it arms. There is currently no way to view or reattach the prior
history from the new row.

## Creating a recurring question

A recurring question runs a scoped search on a schedule and reports what
it finds — a different kind of task than a reminder. Open its create
form from the **Create ▾** chooser ("Recurring question…"). Its v1
fields: a name, the
question itself, which sources to search (all readable library sources,
or a specific choice of Media / Notes / Chats — collections, tags, and
saved searches are not offered yet), the schedule (the same one-time/
recurring controls as a reminder), when to generate a draft answer
(always, only when something new is found, or never), a finding-policy
preset, whether to be notified, and an optional provider/model pin.

**Preview** runs the same validation the save itself will run and shows
the next few scheduled occurrences without saving anything; a rejected
preview highlights the specific field that needs fixing. **Save** always
previews first — an invalid definition is never written. While saving, the
button shows **Saving…** and repeated Save actions cannot start another
save. Validation errors, server refusals and failed writes keep your draft
open so you can correct it or retry. Only the
`recurring_question` family can be authored here; agent-task automations
are not yet supported from this form.

## Automations in the list

Automation definitions from **both** owners share the one list: this
device's own local `recurring_question` automations and whatever a
connected server reports. Each row carries its owner suffix so the two
are never ambiguous side by side; saving a new "Runs on: This device"
automation appears immediately. With no server connected you see the
local ones alone rather than an empty list.

Click **Run now** in a highlighted definition's detail pane to run it
immediately — a real dispatch, not a preview, routed by that row's own
owner. A local
automation runs through the same claim/spawn machinery the scheduler's
own tick uses (no risk of it double-firing against a scheduled run) and
refuses honestly when it is missing, paused/archived, mid-transfer, or
its read-time health is not ready — the toast says which. A server
automation dispatches **on the server** through its own control-plane
pipeline; the toast reports the run slot (and whether the server
collapsed it into an already-queued run), and the result arrives through
the server's notification feed, not the local queue. A paused or
archived server definition refuses with the server's own reason.

Press **e** (or **Edit in full…** in the pane) on a highlighted
`recurring_question` definition, either owner, to edit it — the same
form Save opens, pre-filled from the row. Editing a server automation
that has never synced to this device mirrors it locally first
(automatic, no extra step); agent-task automations are not yet editable
here.

The pane's **Model** row shows the automation's pinned execution target
— `provider/model` when the definition carries its own selection (the
executor honors it per run), or `auto` when it pins nothing and the
executor resolves the target itself (config defaults, then the provider
default). Per-task selection rides the definition payload, so one
automation can run on a different model than the default without
touching config.

Activating the pane's **Last run** row opens that definition's **run
history** over the list ("Run history — \<name\>"). For a server
automation this is the server's durable audit trail, newest first (time,
event, summary), fetched when the view opens: queued, succeeded, failed,
timed out, skipped — the same events the server records for
reconciliation. **Local automations do not have a durable run history
yet**, and the view says so rather than showing an empty server-shaped
trail. **Escape** closes it.

### The automation detail pane

Highlighting a definition row opens its detail pane beside the list.
**Pause**/**Resume** and **Run now** buttons sit above the body
(Pause/Resume toggles the definition's lifecycle; Archive stays a list
action, not this button), and **p** does the same as the Pause/Resume
button from the list. The pane shows the question text in a card next,
then the same grouped-row layout as the reminder detail pane:

- **Details** — `Runs on` (owner + transfer badge, same wording as the
  reminder pane; its dropdown is the same transfer flow described under
  "Moving a task between this device and the server"), `Model` (the
  pinned `provider/model`, or `auto` when the definition pins nothing),
  `Generation` (always/only-new/never), `Finding policy` (the preset
  name), and `Sources` (the selected library sources, or "All searchable
  library" for the default scope). A row the definition does not carry a
  value for reads **"Not set"** — a definition authored on the server
  often carries none of these, and the pane never fills the gap in with
  the create form's defaults.
- **Frequency** — the schedule summary (repeat/at/timezone, or the raw
  cron expression for a custom schedule) and `Notifications` (a real
  On/Off toggle here, unlike a reminder's fixed "Inbox + toast" label).
- **History** (collapsed by default) — `Last run` (activate it to open
  the run history described above), total run count, and `Unread
  results` (activate it to open the results view scoped to just this
  automation; **r** marks them all read without opening anything). Those
  counts are this device's own execution record, so a server-owned
  definition reads "Kept on the server — see Run history" instead: only
  the server holds that definition's execution history.

Every Details/Frequency row except the schedule-kind mismatch (Repeat on
a one-time definition, or At on a recurring one — same rule the reminder
pane follows) edits in place the same way the reminder pane's rows do —
see "Editing rows in place", above — with one addition: **Sources**
opens three checkboxes (Media/Notes/Chats) plus an **Apply** button
rather than a dropdown, since a checkbox has no single commit event of
its own. Ticking every box and applying writes the explicit three-source
selection, not "all searchable library" — pick **Edit in full…** if you
want the scope to keep resolving to whatever sources are readable at
each run rather than freezing today's three.

Below Pause/Resume/Run now sits a second row: **Duplicate**, **View
runs**, and **View results**. **View runs** and **View results** are
plain shortcuts onto the `Last run` and `Unread results` rows above —
same destination, same "always reachable, viewing history is never
gated" rule — so they work for any definition regardless of family or
lock state. **Duplicate** creates a new local copy named "*Name*
(copy)" with the same question, schedule, model pin, generation mode,
sources, and notification policy, and a fresh id. Its **paused or
active state carries over** — pausing before duplicating, or after, is
your call, but Duplicate itself never turns a paused definition into a
running one behind your back. (An archived or otherwise inactive
source collapses to paused, not archived — a brand-new row starting
out archived would be a copy you could not even see turned on.) If
pausing the copy fails, a warning says so: the copy exists and is active
until you pause it from its detail pane. A
generation mode/sources/finding-policy/retention field the source left
unset ("Not set") is written as its concrete default in the copy
(e.g. "Balanced findings") rather than staying unset — display only;
what actually runs is unaffected either way, since an unset field
already resolved to that same default. Duplicate always lands on this
device, the same "new draft, not an implicit move" rule the reminder
pane's own Duplicate follows, and is disabled (with the reason shown
under the button) while the definition is mid-transfer or is a family
this pane cannot author (`agent_task`).

At narrow widths this pane opens full-screen over the list instead of
beside it — see "Narrow terminals", above. Everything described here
works there unchanged.

## The results view — the automation findings inbox

The **Results** button in the rail opens the `automation_results` rows a
recurring question has produced, across **both** owners (this device and
any connected server) in one inbox over the list, newest first.
**Escape** closes it and re-syncs the rail. The button carries an unread
badge — "Results (3)" — updating after every sync and after every action
below, and reading plain "Results" with nothing unread.

An automation's own **Unread results** row opens the same view scoped to
just that automation (both of its id spaces, so a definition that has
moved between owners keeps its earlier results), with the heading naming
it.

The table holds the newest 200 results: the same window a sync pull
mirrors down, so it shows everything a sync could have fetched. Past
that the heading says which slice you are looking at ("Automation
results — showing newest 200 of 214"). It has to, because the unread
badge counts *every* unread result — a silently truncated table beside
that number would misreport how much is there. Older results stay in
the database; there is no paging here.

Each row shows a kind glyph (● for a **finding**, ✕ for a **failure** —
failure rows are styled distinctly since they are diagnostic, not
hidden), the result's title (with a "(server: \<id\>)" suffix for a
server-owned row), how long ago it was created, and its review state
("● unread" in bold, or plain "read"/"dismissed"). Selecting a row shows
its answer, evidence (the stored source references), and review
metadata (who reviewed it and when, plus any review note) in the detail
pane below the table.

While connected to a server, results refresh on their own the moment the
server reports that an automation run finished — no need to press **s**.
A short pause (well under a second) absorbs a burst of several finish
notifications arriving close together into a single pull, so a chatty
automation does not fire one network round trip per event. Opening the
screen pulls once as well, so results announced while you were elsewhere
are picked up rather than waiting for the next notification.

Actions inside the view are keys:

| Key | Action |
| --- | --- |
| `r` | Mark the selected result **read** |
| `d` | **Dismiss** the selected result |
| `o` | Mark the selected finding's automation **solved** |
| `a` | Mark **every** unread result in this view’s scope read, including rows beyond the table window |

These four belong to the results view itself — it is a screen of its
own, so its keys never collide with the list's behind it. A server-owned
row's read/dismiss writes locally and automatically queues the matching
pushback to the server; nothing extra to do.

While one Results action is pending, another `r`, `d`, `o` or `a` shows
a busy notice. Wait for completion, then press the desired key again. An accepted
action keeps its original result or unread batch even if you move the selection;
new results arriving afterward wait for a later batch. Mark all read reports how
many succeeded and how many failed; failed rows stay unread and can be retried.

**Mark solved** only applies to a `finding` whose automation is not
already solved — a `failure` row, or a finding whose automation was
already marked solved, refuses with the specific reason instead of
attempting the call. A finding's detail pane says which applies: "Solve:
eligible" or "Solve: not eligible — \<reason\>". Marking solved on a server-owned automation
requires that server connection right now: this repo does not yet queue
a solved/reopen mutation for later delivery, so an offline attempt
refuses honestly ("…this action requires a server connection") rather
than silently queuing something that might already be stale by the time
it would send.

A definite server-side refusal shows the server's own reason (for
example "the server has archived this automation") rather than the
connection message — only genuine connectivity failures mention the
network.

**A server too old for this surface says so.** Before pulling, the app
asks the server what it supports. A server that does not answer that
question at all is not asked for automations or results — nothing is
pulled, no error is invented, and the run-history pane reads *"This
server does not support scheduled task automation (server too old)."* A
server new enough to answer but not yet serving the results route
reports *"This server does not provide the results inbox (server too
old)."* — carried in the sync notice as *"Sync completed with issues —
Automation results pull: …"*, so the missing route names itself instead
of surfacing as a server-worded error about a task that was never the
problem. The results view itself still reads "No results yet." when
nothing has arrived; the sync notice is where the *why* lives.

## Execution timeouts

A scheduled task's handler is bounded: if it is still running after its
execution timeout, it is **cancelled** and the task records the status
**Timed out** — the schedule advances to the next occurrence regardless, so
one hung job (say, a watchlist check against an unresponsive URL) can never
stall the rest of the scheduler. Timed out is its own outcome, distinct
from *Missed* (the dispatch ran and raised an error) and from *Missed while
away* (the scheduled time passed with the scheduler not running) — and a
timed-out task offers **Run now (retry)** like a failed one.

The default bound is `handler_timeout_seconds` under `[scheduling]` in
`config.toml` (**300** seconds). Set it to `0` (or negative) to disable the
bound entirely — every handler may then run as long as it likes, and a
wedged handler will wedge the scheduler, which is why the default is on.
