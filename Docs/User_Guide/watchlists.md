# Watchlists — Monitored sources, runs, alerts, and recovery

> 🚧 **This page is a stub.** The full write-up is planned; the sections
> below cover orientation only. See the [guide index](index.md).

## What this screen is for

Watchlists tracks monitored sources, their runs, alerts, and recovery. The
header shows "Watchlists | Monitored sources, runs, alerts, recovery |
Mixed | Local/Server". This screen was previously called "Subscriptions."

## Getting there

- Click **Watchlists** in the nav bar, or press **Ctrl+P** → "Switch to
  Watchlists". (Or press **Ctrl+6** from anywhere.)

## Exporting briefings and podcast feeds

The **Artifacts** section of a watchlist holds its briefings — text digests
of what its sources did — along with any scripts cast from them and any
audio synthesized from those scripts. Two export actions live on that
section's top toolbar, and both write to a location you pick:

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
clients that require HTTP, serve the folder yourself — from a terminal, in
the exported directory:

```bash
python3 -m http.server 8000
```

Then point the client at `http://localhost:8000/feed.xml` (or your machine's
LAN address, if the client runs on another device). Stop the server with
Ctrl+C when you are done.

The app does not serve the folder for you. Its `[web_server]` setting is
unrelated — that runs the whole chatbook UI in a browser instead of your
terminal, and it is not a way to publish a feed.

## Scheduled briefings

By default, a briefing is written only when you press **Generate** in the
Artifacts section — nothing runs unless you ask for it. The **cadence**
picker next to the selection-mode and default-preset pickers in that same
toolbar turns this into a recurring job for one watchlist: choose **Every
12h**, **Daily**, or **Weekly**, and a new briefing is written on that
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

The Artifacts section's scope line states plainly which of these applies:
"on request" when scheduling is off, or the actual cadence — "scheduled
daily while the app is open", for example — when it is on.
