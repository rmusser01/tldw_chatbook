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
