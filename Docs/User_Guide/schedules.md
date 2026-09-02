# Schedules — When jobs, watchlists, and workflows run

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

> 🚧 **This page is a stub.** The full write-up is planned; the sections
> below cover orientation only. See the [guide index](index.md).

## What this screen is for

Schedules controls when jobs, watchlists, and workflows run. The screen
has no single on-screen title; it opens on a sync status bar (Local /
Server, last pull/push) above **Queue**, **Automations**, **Conflicts**,
and **Results** tabs, with panels for the Schedule Queue, Task Detail, and
Inspector.

## Getting there

- Press **Ctrl+7**, click **⌃7 Schedules** in the nav bar, or press
  **Ctrl+P** → "Tab Navigation: Switch to Schedules".

## Sync bar honesty

Pressing **s** reports what actually happened: "Sync completed." only
when a pull or push was recorded (the Last pull/push timestamps update),
and "Sync finished — nothing was pulled or pushed." otherwise. With a
Local owner and no scheduling server connected, the bar collapses to a
single line ("Local schedules — no scheduling server connected; sync is
off"), and the **Clear** button only appears once a sync error exists.

## Scheduler liveness

Below the sync bar, a one-line **scheduler liveness** indicator distinguishes
three states that used to look identical: *not started* (the loop has never
ticked on this machine), *live* (with the age of the last tick, e.g. "live ·
last tick 12s ago"), and **STALLED** — the loop has not ticked in several
poll intervals, so reminders have silently stopped; a stall shows the last
error the loop hit, if any. Staleness is judged against your configured poll
interval, so a deliberately long interval is not mistaken for a stall. The
signal is durable (a small heartbeat file), so a stalled or dead loop is
distinguishable from an idle one even across a restart.

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

## Creating a scheduled task

Press **c**, or click **+ New** in the Queue tab's pane header, to open
the create form. Clicking **+ New** first asks which kind of task you
want — **Reminder…** or **Recurring question…** — since a recurring
question is a different kind of definition, not just another schedule
shape; **c** always opens the reminder form directly. The form scrolls
when the terminal is short; the live "Runs: …" preview, validation, and
Save/Cancel stay pinned at the bottom while you edit.

Every create/edit form also has a **Runs on** selector — **This device**
or **Server (\<id\>)** when a scheduling server is connected — defaulting
to whatever owner the Schedules screen is currently showing. Choosing the
server writes the task there directly when the server is reachable, or
authors it locally and queues it to sync up on the next successful sync
when it is not (the sync bar and the task's own state say which
happened -- a recurring question in that state is listed on the
Automations tab as *\[\<server id\> · pending sync\]*, and Run now/run
history say so rather than asking the server about a task it has never
seen). If the server refuses the save outright rather than being
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
**Disabled** in both the queue row and the detail badge, and its Next Run
reads **— (disabled)** instead of a concrete time it will not honor.
Enabling it restores the recorded last outcome and the real next run.

This covers a one-time reminder that has already fired: running it
disables it and clears its next run, so it reads **Disabled** with a Next
Run of **— (disabled)** — the same as a task you disabled by hand.

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
  glance at the Queue tab shows what fell behind. Typing `missed` into the
  queue filter finds them (it also matches tasks whose last dispatch
  *failed*, which is a different thing — see below).

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

## Run now — dispatch a reminder immediately

Press **r** on a highlighted reminder (or its **Run now** button in the
task detail pane) to dispatch it right away, without waiting for its
schedule. This is a **real dispatch** through the same path the scheduler
uses — not a preview:

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
double execution — one owner, one executor). Pressing **r** on one refuses
with a toast saying so. This is the single-owner execution rule the
server-offload design is built on; local-owner reminders are unaffected.

## Moving a task between this device and the server

A reminder's detail pane offers **Move to server** (on a local task) or
**Move to local** (on a server-owned mirror), plus **Cancel transfer**
once a move is in flight and **Retry transfer** after one fails. Each
button is visible whenever it could apply and disabled with a stated
reason otherwise — no server connection, no server identity, a transfer
already running, or, moving a `recurring_question` mirror to this
device, the same local-health reason the Automations tab already
surfaces.

Clicking Move opens a confirmation listing anything worth knowing before
you commit: an imminent or already-passed one-time run time ("server
behavior this close to run time is unverified"), and, for a reminder,
that its per-run timeout is local-only and will not transfer. Confirming
a **local → server** move only queues it — the task keeps running on
this device until the server actually accepts the transfer, the toast
says so, and the queue row shows "(Moving to server…)" for as long as
that stays true. A **server → local** move creates a dormant local copy
immediately (shown as "(Waiting for server release)"); it stays inert
until the server's release is acknowledged, at which point it arms and
starts running here.

When Schedules is showing a server owner's queue, each row's title also
carries a "(server: \<id\>)" owner suffix — the same wording the Results
tab uses for its own rows. It is hidden whenever the window is narrow
enough to trigger the compact layout (side panes already hidden), so a
squeezed terminal never truncates a title to make room for it.

**While a move is in flight the task is read-only.** Edit, Delete and
Enable/Disable are disabled with the reason stated, on the buttons and
as text, for a queued local → server move, one already sent, and a
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

**A disabled server reminder stays disabled** when it is released to this
device — the release moves the task, not its on/off state.

### Moving an automation (Automations tab)

Automations use keys instead of buttons, because that tab has no
per-row detail pane:

| Key | Action |
| --- | --- |
| `M` | Move the selected local automation **to the server** |
| `m` | Move the selected server-owned automation **to this device** |
| `y` | Retry a local → server move the server rejected |
| `k` | Cancel the selected automation's in-progress move |

These four are **Automations-tab only**, even though the footer
advertises them on every tab: pressing them elsewhere answers with a
"Switch to the Automations tab…" notice rather than acting on whatever
the Queue tab happens to have selected. They run the same confirmation,
warnings and honest toasts as the reminder buttons above, and a refusal
appears inline in the Automations pane's notice line rather than as a
toast.

## Creating a recurring question

A recurring question runs a scoped search on a schedule and reports what
it finds — a different kind of task than a reminder. Open its create
form from the Queue tab's **+ New** chooser ("Recurring question…") or
the Automations tab's own **+ New** button. Its v1 fields: a name, the
question itself, which sources to search (all readable library sources,
or a specific choice of Media / Notes / Chats — collections, tags, and
saved searches are not offered yet), the schedule (the same one-time/
recurring controls as a reminder), when to generate a draft answer
(always, only when something new is found, or never), a finding-policy
preset, whether to be notified, and an optional provider/model pin.

**Preview** runs the same validation the save itself will run and shows
the next few scheduled occurrences without saving anything; a rejected
preview highlights the specific field that needs fixing. **Save** always
previews first — an invalid definition is never written. Only the
`recurring_question` family can be authored here; agent-task automations
are not yet supported from this form.

## Automations tab — local and server automations

The **Automations** tab lists automation definitions from **both**
owners: this device's own local `recurring_question` automations and
whatever a connected server reports — the server ones execute there,
which is why they do not appear in the local Queue. Each row's Name cell
is prefixed with its owner (`[This device] …` or `[<server id>] …`) so
the two are never ambiguous side by side; saving a new "Runs on: This
device" automation shows up here immediately (the tab refreshes after
every save). With no server connected the tab shows local automations
alone instead of an empty list.

> **Known gap (live verification, 2026-09-02).** Against a real server
> every row is currently prefixed `[This device]`, including genuine
> server automations, because the owner id the server sends is not in the
> form this screen recognises. The pane's own count line ("2 automations
> on the server") stays correct, so the two disagree. The same
> misreading routes the keys below: **r** on a server automation refuses
> with the *local* health message instead of dispatching on the server,
> and **m** refuses with "This automation no longer exists." Until this
> is fixed, treat the count line — not the row prefixes — as the truth
> about ownership, and drive server automations from the server.

Press **r** on a highlighted definition to run it immediately — a real
dispatch, not a preview, routed by that row's own owner. A local
automation runs through the same claim/spawn machinery the scheduler's
own tick uses (no risk of it double-firing against a scheduled run) and
refuses honestly when it is missing, paused/archived, mid-transfer, or
its read-time health is not ready — the toast says which. A server
automation dispatches **on the server** through its own control-plane
pipeline; the toast reports the run slot (and whether the server
collapsed it into an already-queued run), and the result arrives through
the server's notification feed, not the local queue. A paused or
archived server definition refuses with the server's own reason.

Press **e** on a highlighted `recurring_question` definition (either
owner) to edit it — the same form Save opens, pre-filled from the row.
Editing a server automation that has never synced to this device mirrors
it locally first (automatic, no extra step); agent-task automations are
not yet editable here.

The **Model** column shows each automation's pinned execution target —
`provider/model` when the definition carries its own selection (the
executor honors it per run), or `auto` when it pins nothing and the
executor resolves the target itself (config defaults, then the
provider default). Per-task selection rides the definition payload, so
one automation can run on a different model than the default without
touching config.

The right half of the tab is that definition's **Run history**. For a
server automation this is the server's durable audit trail, newest first
(time, event, summary) — it loads when you highlight the row and
refreshes right after a Run-now dispatch, so the run you just triggered
appears without re-selecting it. **Local automations do not have a
durable run history yet** — the pane says so honestly rather than
showing an empty server-shaped trail; every execution still leaves its
trail here for server automations: queued, succeeded, failed, timed out,
skipped, the same events the server records for reconciliation.

## Results tab — the automation findings inbox

The **Results** tab lists every `automation_results` row a recurring
question has produced, across **both** owners (this device and any
connected server) in one inbox, newest first. Its tab label is meant to
carry an unread badge — "Results (3)" — updating after every sync and
after every action below, and reading plain "Results" with nothing
unread.

> **Known gap (live verification, 2026-09-02).** The badge does not
> render today: the unread count is computed correctly but written to an
> attribute this Textual version's tab bar never reads, so the label
> stays plain "Results" whatever the count is. The Conflicts tab's count
> has the same problem. The per-row "● unread" state in the table below
> is unaffected, and is the honest reading until this is fixed.

Each row shows a kind glyph (● for a **finding**, ✕ for a **failure** —
failure rows are styled distinctly since they are diagnostic, not
hidden), the result's title (with a "(server: \<id\>)" suffix for a
server-owned row), how long ago it was created, and its review state
("● unread" in bold, or plain "read"/"dismissed"). Selecting a row shows
its answer, evidence (the stored source references), and review
metadata (who reviewed it and when, plus any review note) in the detail
pane below the table.

While connected to a server, this tab also refreshes on its own the
moment the server reports that an automation run finished — no need to
press **s**. A short pause (well under a second) absorbs a burst of
several finish notifications arriving close together into a single
pull, so opening a chatty automation's run history does not fire one
network round trip per event.

Actions are keys, not buttons — the tab has no per-row detail widget
here either:

| Key | Action |
| --- | --- |
| `r` | Mark the selected result **read** |
| `d` | **Dismiss** the selected result |
| `o` | Mark the selected finding's automation **solved** |
| `a` | Mark **every** currently-listed unread result read |

`r`/`d` are the same keys the Queue tab uses for Run now/Delete —
reused here because Read/Dismiss are the natural reading of those same
letters on this tab. A server-owned row's read/dismiss writes locally
and automatically queues the matching pushback to the server; nothing
extra to do. `o`/`a` are Results-tab only, the same way the Automations
tab's `m`/`M`/`y`/`k` are: pressed elsewhere they answer with a "Switch
to the Results tab…" notice instead of acting on another tab's
selection.

**Mark solved** only applies to a `finding` whose automation is not
already solved — a `failure` row, or a finding whose automation was
already marked solved, refuses with the specific reason instead of
attempting the call. Marking solved on a server-owned automation
requires that server connection right now: this repo does not yet queue
a solved/reopen mutation for later delivery, so an offline attempt
refuses honestly ("…this action requires a server connection") rather
than silently queuing something that might already be stale by the time
it would send.

> **Known gap (live verification, 2026-09-02).** For a result that came
> down from a server, mark-solved never becomes eligible: the detail pane
> reports "This result's automation definition could not be found" even
> when that definition is mirrored on this device, because the result
> carries the server's id for it and the eligibility check looks the id
> up among local ones. `o` therefore refuses on exactly the rows the
> action exists for. Read/dismiss (`r`/`d`) and their server pushback are
> unaffected.

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

*Verified against a running tldw_server — 2026-09-02 (schedules-handoff
PR-6 task 6, the program's §10 live gate: real TUI in tmux against
tldw_server `origin/dev` 25fb0eca59 on a local single-user profile.
Verified live: authoring a local recurring question, transferring it to
the server (server returns 201) and the local row rebinding to the
server owner; server-owned results syncing down as unread with the
kind/owner/created/review-state columns and the answer/evidence/review
detail pane; `r` marking a result read and that read reaching the server
(`review_state: read` + `reviewed_at`); and the Results tab refreshing
on a server finish notification with no `s` pressed. Not verified in
that run: reminder transfer round-trips, including the past-`run_at`
one-time case — the create form could not be driven to Save. The three
"Known gap" callouts above were found by this run. Full record:
`.superpowers/sdd/plan-2026-09-02-schedules-handoff-pr6/task-6-report.md`.
Supersedes the same-day PR-6 task
4 stamp: the Queue tab's own "(server: \<id\>)" owner suffix, hidden at
compact width; and the Results tab's automatic refresh on a server
finish notification, debounced so a burst of notifications settles into
one pull; supersedes the same-day PR-6 task 3 stamp, which covered the
new Results tab — unread badge, kind/owner/created/review-state
columns, failure styling, the answer/evidence/review-metadata detail
pane, and the r/d/o/a read/dismiss/mark-solved/mark-all-read keys with
their tab-scoping and refusal reasons; and, before that, PR-5's final
fix wave: the Automations tab's
M/m/y/k transfer keys and their tab-scoping, the read-only-except-cancel
rule on in-flight rows, and the corrected cancel copy; plus PR-5 task 7's
the detail pane's Move to server/Move to local/Cancel transfer/Retry
transfer buttons, the confirm dialog and its warnings, honest transfer
toasts, and the queue row's transfer-state suffix; and the 2026-09-01
stamp before that, which covered recurring-question create/edit form,
the Queue tab's New/Reminder/Recurring-question chooser, the Automations
tab's own New button, the "Runs on" selector on both forms, the merged
local+server Automations listing including not-yet-synced server-owned
rows, and its local run-now/edit routing).*
