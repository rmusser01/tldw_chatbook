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
Server, last pull/push) above **Queue**, **Automations**, and
**Conflicts** tabs, with panels for the Schedule Queue, Task Detail, and
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
happened). An existing reminder's owner is fixed once created — the
selector shows it but is not editable — moving a task between owners is
a separate action, not part of editing.

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

## Automations tab — server-scheduled automations

The **Automations** tab lists the automation definitions that live on a
connected server (name, family, lifecycle, health) — the server owns
their execution, which is why they do not appear in the local Queue. With
no server connected the tab says so instead of showing an empty list.
Automations you create with a **Runs on: This device** owner are not
shown in this list yet — it currently reflects the connected server only.

Press **r** on a highlighted definition to dispatch one immediate run
**on the server** through the same pipeline its schedule uses — a real
dispatch, not a preview. The toast reports the run slot (and whether the
server collapsed it into an already-queued run); the result comes back
through the server's notification feed, not into the local queue. A
paused or archived definition refuses with the server's own reason.

The **Model** column shows each automation's pinned execution target —
`provider/model` when the definition carries its own selection (the
server executor honors it per run), or `auto` when it pins nothing and
the server resolves the target itself (its automation-config executor
defaults, then the server default — both live in server config, not the
definition). Per-task selection rides the definition payload, so one
automation can run on a different model than the server-wide default
without touching server config.

The right half of the tab is that definition's **Run history** — the
server's durable audit trail, newest first (time, event, summary). It
loads when you highlight a definition and refreshes right after a
Run-now dispatch, so the run you just triggered appears without
re-selecting the row. Every execution leaves its trail here: queued,
succeeded, failed, timed out, skipped — the same events the server
records for reconciliation.

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

*Verified against working tree — 2026-09-01 (schedules-handoff PR-4 task 5:
recurring-question create form, the Queue tab's New/Reminder/Recurring-
question chooser, the Automations tab's own New button, and the "Runs on"
selector on both forms).*
