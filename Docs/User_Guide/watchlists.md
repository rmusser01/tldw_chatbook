# Watchlists — Monitored sources, runs, alerts, and recovery

> 🚧 **This page is a stub.** The full write-up is planned; the sections
> below cover orientation only. See the [guide index](index.md).

## What this screen is for

Watchlists tracks monitored sources, their runs, alerts, and recovery. The
header shows "Watchlists | Monitored sources, runs, alerts, recovery |
Mixed | Local/Server". This screen was previously called "Subscriptions."

## Getting there

- Press **Ctrl+6**, click **⌃6 Watchlists** in the nav bar, or press
  **Ctrl+P** → "Tab Navigation: Switch to Watchlists".

## Adding sources and making a Watchlist

In **Sources**, use **New source** for one feed or **Add several…** for a
repeatable bulk pass. Bulk entry accepts one HTTP(S) URL per line, up to 50
nonblank lines, with one shared Type and optional Tags. **Validate and create**
keeps the draft visible and reports each URL in its original order as Created,
Existing, or Invalid. A validation or save failure never clears the draft.

When only part of a batch succeeds, the app pauses with exactly two choices:
**Continue with successful sources** or **Return to draft**. Continuing opens
**All Sources** with the successful source IDs selected; it does not silently
add them to a Watchlist. The **Next** choice either focuses the All Sources
table or focuses the enabled **Create Watchlist from selected…** action; it
never presses that action for you. Pressing the action creates the Watchlist
and its memberships atomically, after one name prompt. While an admitted batch
is being saved, Create, Cancel, and Escape wait for its result so a successful
write cannot be hidden behind a dismissed dialog.

The source table also supports focus-scoped keyboard selection:

- **Space** toggles the highlighted source and starts a range.
- **Shift+Up/Down** extends or contracts that range in the current visible
  order.
- **v** selects all currently visible filtered rows; press it again to clear
  only those visible rows.
- **x** clears every selection, including rows hidden by filters.

Selections follow source identities, not row positions, so sorting and
filtering do not move the selection to a different source. The status line
reports both the total selected count and how many are hidden by filters.
These shortcuts appear in Help and the command palette only while the Sources
table owns focus.

## Checking a source while a check is already running

One check of a given page runs at a time. If you press **Check now** for a
source whose page is already being checked — usually because a scheduled
check of it is mid-flight — the app does not run a second, duplicate check
of the same page: your run completes immediately without touching the
network, the toast says the check was skipped because one is already
running, and the running check's result appears in the Runs section the
next time the screen refreshes.
The skipped run stays honest in the Runs section too: its detail line
counts the page under "skipped (check already running)" rather than
reading like a clean check that found nothing. Different sources are
unaffected — they check concurrently, exactly as before.

## Recovering a failed source check

Select a failed run to see a short failure reason and its **Next** action. The
app keeps request addresses, response bodies, credentials, and internal error
details out of this view. Older run records that predate classified failures
show a generic failure and ask you to review the source configuration.

| Failure | What to do | Re-run unchanged? |
|---|---|---|
| Access denied | Check whether the source permits automated access. | No |
| Authentication required | Check the source credentials and authentication settings. | No |
| Rate limited | Wait for the source's stated delay, when one is available. | Yes |
| Invalid feed | Check the source address and feed format. | No |
| Connection failure | Re-run when the network or source is available. | Yes |
| Temporary server error | Re-run later. | Yes |
| Blocked by network safety policy | Choose a public HTTP(S) source allowed by the network safety policy. | No |

The **Re-run source** action is available for a failed run only when the
recorded outcome says the same input can be retried. For authentication,
access, invalid-feed, and policy failures, make the stated configuration or
content change first and then use **Check now**.

## Exporting briefings and podcast feeds

The **Artifacts** section of a watchlist holds its briefings — text digests
of what its sources did — along with any scripts cast from them and any
audio synthesized from those scripts. **Generate briefing**, **Refresh**, and
the generation/schedule controls stay in the primary workflow. Export, Keep,
and feed-serving actions live under **More briefing actions** so an empty
watchlist foregrounds creation instead of unavailable downstream actions.

Refreshing or generating does not blank a briefing you are already reading.
The last good table, selection, Markdown body, and citations remain visible
with an inline loading state. A failure keeps that content and offers
**Retry**. If generation was durably accepted or completed but the refreshed
view cannot find its row, Artifacts reports “Briefing saved, but this view
could not reload it” and offers both **Retry** and **Inspect Runs**; this is a
storage/reload diagnostic, not an empty state.

Under **More briefing actions**:

- **Export** saves the selected briefing as a Markdown file. It is enabled
  once you have selected a briefing that has finished generating — a
  finished briefing that isn't selected, or nothing selected at all, leaves
  the button disabled.
- **Export Feed** writes a **podcast feed directory**: a `feed.xml` plus a
  copy of every finished audio episode in the watchlist. It is enabled once
  at least one episode exists.

If some episodes cannot be exported — a file has been moved or deleted
since it was synthesized — the rest are still written and the app tells you
how many of how many succeeded, and why the others were skipped.

### Listening to an exported feed

The exported folder is self-contained: `feed.xml` refers to the audio files
beside it by name, never by absolute path, so you can copy, sync, or zip the
whole directory and it keeps working on another machine.

Some podcast clients accept a local folder or a `file://` URL directly. For
clients that require HTTP, the Artifacts toolbar has two more buttons next
to Export Feed:

- **Serve Feed** starts a small, built-in server for the directory you most
  recently exported, and shows a toast with the URL to point your podcast
  client at (e.g. `http://127.0.0.1:54231/feed.xml`). It is disabled until
  you have exported a feed, and disabled again while a feed is already
  being served.
- **Stop Serving** stops it. Switching away from Watchlists, or closing the
  app, also stops it — serving never outlives the screen it started on.

This is unrelated to the app's `[web_server]` setting: that runs the whole
chatbook UI in a browser instead of your terminal, and it has no way to
serve a directory you choose at all. If you would rather not use the
built-in server, serving the folder yourself works exactly as before — from
a terminal, in the exported directory:

```bash
python3 -m http.server 8000
```

Then point the client at `http://localhost:8000/feed.xml` (or your machine's
LAN address, if the client runs on another device). Stop the server with
Ctrl+C when you are done.

#### Security posture

The built-in Serve Feed server is opt-in and session-only: nothing is ever
served until you press **Serve Feed**, and it stops the moment you press
**Stop Serving**, switch away from Watchlists, or close the app — it never
starts on its own, and no setting can make it start on its own.

- **No authentication.** Anyone who can reach the served address can read
  every file in the exported directory **and every subdirectory beneath
  it** — serving is recursive, not limited to `feed.xml` and the episodes
  next to it — for as long as it is running. The toast that appears when
  you press Serve Feed states this every time, not just here. Point Serve
  Feed at a dedicated export folder, not a general-purpose directory like
  your home folder, for exactly this reason: everything under whatever
  folder you export into becomes reachable while serving is on.
- **No directory browsing.** The server refuses to render a listing of the
  served folder (or any subfolder) — only a file whose exact name a client
  already knows (from `feed.xml`, or a URL you typed yourself) is
  fetchable. This narrows the exposure above; it does not remove it, since
  every filename in the tree is still fetchable if it is known.
- **Loopback by default.** The server binds `127.0.0.1` (this machine
  only) unless you change `bind` under `[briefings_feed_server]` in
  `config.toml` to something wider (e.g. `0.0.0.0`, to reach it from
  another device on your network). A blank or otherwise invalid `bind`
  value can never silently widen this — it falls back to `127.0.0.1`
  instead — and whenever the server does end up bound to a non-loopback
  address, a warning is logged and the Serve Feed toast says so plainly.
  Only widen this if you understand that doing so removes the one thing
  standing between "only this machine" and "anyone who can reach it" —
  there is still no authentication either way. Note that a wildcard IPv6
  bind (`::`) is usually dual-stack: it accepts plain IPv4 connections as
  well, so the reachable surface can be wider than the address suggests.
- **Confined to the exported directory.** The server refuses (with a plain
  404) any request that would read a file outside the directory you
  exported to, including through a symlink planted inside it that points
  elsewhere on disk.
- **One directory at a time.** Serve Feed refuses (naming the URL already
  in use) rather than switching directories out from under a client that
  might still be connected. Press Stop Serving first to serve a different
  export.

## Scheduled briefings

By default, a briefing is written only when you press **Generate briefing** in the
Artifacts section — nothing runs unless you ask for it. The **cadence**
picker next to the selection-mode and default-preset pickers in that same
toolbar turns this into a recurring job for one watchlist: choose **Every
12 hours**, **Every 24 hours**, or **Every 7 days**, and a new briefing is written on that
schedule without you pressing anything. Choose **Off** — the default — to
turn scheduling back off.

A few things worth knowing before turning it on:

- **It runs only while the app is open.** There is no background service —
  a scheduled briefing fires from inside this app's own process, so closing
  the app pauses the schedule. It picks back up, on the same rhythm, the
  next time you open the app; nothing is generated while it is closed.
- **It is opt-in, per watchlist, and off by default.** Generating a briefing
  spends the LLM tokens your briefing preset is configured to use, so
  turning scheduling on for one watchlist never turns it on for any other.
- **A failed run is retried at the next scheduled time, not immediately.**
  If a scheduled briefing fails, the schedule doesn't skip ahead to the
  next period — but it also doesn't retry right away. The next attempt
  lands one cadence period after the failure, the same timing a normal
  run would have used.
- **A saved cadence requests an immediate scheduler reload.** The receipt
  distinguishes the durable save and reload request from a reload the running
  scheduler has actually acknowledged. If the scheduler is stopped or the
  acknowledgement times out, the cadence remains stored and is loaded when
  the scheduler next runs.
- **An app-level setting can turn scheduling off entirely.** The
  `[scheduling] briefing_schedules_enabled` setting in `config.toml` (`true`
  by default, hand-edit only today — there is no in-app control for it)
  gates whether this app ever fires ANY scheduled briefing, for every
  watchlist at once. Turning it off does not clear a watchlist's stored
  cadence: the cadence picker in Artifacts still shows what is stored, but
  greys out, and the schedule resumes exactly where it left off the moment
  the setting is turned back on.

The Artifacts section's scope line states plainly which of these applies:
"on request" when no cadence is stored, the actual cadence — "scheduled
every 24 hours while the app is open", for example — when one is stored and
scheduling is enabled for the app, or, when a cadence is stored but
`briefing_schedules_enabled` is off, a third line naming the stored cadence
and stating plainly that it will not fire — "stored to run every 24 hours, but
scheduled briefings are turned off for this app — this schedule will not
fire", for example.

## Console agents and external MCP

The Console can drive the same local Watchlists workflow through approved
tool calls: create sources and collections, start source checks or briefing
generation, follow durable operation receipts, and read a completed briefing
with its ordered source/item provenance on the user's behalf. These tools use
domain services rather than driving the Textual controls, so the Watchlists
screen does not need to be mounted.

External MCP is deliberately narrower. With an operator-recorded permission it
may read bounded source, collection, operation, and briefing-receipt metadata;
it cannot receive item evidence or full briefing Markdown and cannot invoke
Watchlists mutations or network/model work. Settings remains the owner of
global permission and scheduling gates.
