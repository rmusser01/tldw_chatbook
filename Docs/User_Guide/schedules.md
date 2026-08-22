# Schedules — When jobs, watchlists, and workflows run

> 🚧 **This page is a stub.** The full write-up is planned; the sections
> below cover orientation only. See the [guide index](index.md).

## What this screen is for

Schedules controls when jobs, watchlists, and workflows run. The screen
has no single on-screen title; it opens on a sync status bar (Local /
Server, last pull/push) above **Queue** and **Conflicts** tabs, with
panels for the Schedule Queue, Task Detail, and Inspector.

## Getting there

- Press **Ctrl+7**, click **⌃7 Schedules** in the nav bar, or press
  **Ctrl+P** → "Tab Navigation: Switch to Schedules".

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
