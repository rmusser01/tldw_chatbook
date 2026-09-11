# Artifacts — Generated outputs, bundles, reports, datasets, and Chatbooks

> 🚧 **This page is a stub.** The full write-up is planned; the sections
> below cover orientation only. See the [guide index](index.md).

## What this screen is for

Artifacts is where generated outputs collect: bundles, reports, datasets,
drafts, exports, and Chatbooks. The on-screen filter bar lets you narrow
by type (All, Chatbooks, Reports, Datasets, Drafts, Exports) and sort
(default: Recent).

## Reports

The **Reports** slot lists recent Daily Briefs. With no reports yet, use
**Create Your First Daily Report** to run the wired demo (it seeds a
"Daily Brief" watchlist from live RSS and drafts a brief with your
configured LLM provider). If a run fails — for example, no API key is
configured — the failed report is listed and a **Run the Daily Report
demo again** button stays on the screen so you can retry after fixing the
provider in **Settings (F9) → API Keys**. Report timestamps are shown in
your local time.

## Import

**Import Artifact** is not yet available in this shell; the button is
disabled and labelled with that precondition. Create artifacts through
Console (Chatbooks) or generate them from Library sources instead.

## Getting there

- Press **Ctrl+4**, click **⌃4 Artifacts** in the nav bar, or press
  **Ctrl+P** → "Tab Navigation: Switch to Artifacts".

## Sharing artifacts

You can hand Chatbook artifacts to other people by hosting a temporary
web page from this screen. Press **s** or click **Share artifacts**
(next to the Chatbooks actions); recipients browse the page, download
the bundles, and import them into their own tldw_chatbook via
**Chatbooks → Import**. Web sharing needs the optional extras —
`pip install tldw_chatbook[web]` — or the screen tells you that hint
instead of opening the dialog.

### Starting a share

The share dialog lists your local Chatbook artifacts for multi-select.
Artifacts whose exported bundle is no longer on disk are listed but
grayed out ("no exported bundle on disk") — only artifacts with an
actual `.zip` behind them can be shared.

- **Share name** (optional) becomes the page's title.
- **Require a password** adds a single shared username/password
  (HTTP Basic). Every recipient you give the login to uses the same
  pair — leave it off for an open share.
- **Who can reach it**: *This computer only (localhost)* keeps the page
  on your machine; *Local network (all interfaces)* makes it reachable
  from other devices on your LAN. Sharing on the local network
  **without** a password requires typing `share` as confirmation.
- **Port** (optional). Leave it blank and a free port is picked
  automatically.

Starting a share stages immutable copies of the selected bundles (plus
a pre-built download-all bundle) and serves them from a small child web
server. While it runs, the Artifacts screen shows a banner with the
URL(s) — loopback always, plus your LAN address when bound wide — and
the artifact count. The share is owned by the app, not the screen: it
survives switching tabs, and only one share runs at a time (starting a
new one stops the previous). Deleting or editing the artifacts in your
library mid-share does not change what recipients get — they see the
staged copies as they were at start.

### What recipients see

A plain HTML page (no JavaScript) listing each artifact with its
description, kind, and size, plus:

- a **Download** button per artifact,
- a **Download all** button (one bundle containing every artifact), and
- `index.json`, machine-readable metadata including sha256 checksums.

Recipients import the downloaded `.zip` files through
**Chatbooks → Import**.

### Plain HTTP — read this before sharing wide

Share traffic is plain HTTP. A password is an access gate, not
encryption: on an untrusted network, anyone between you and a recipient
can read the traffic. Home and office LANs are the intended setting.
For hostile networks, front the share with a reverse proxy that adds
TLS and point recipients at that instead.

### Stopping

Click **Stop sharing** (the banner's stop button) or close the app.
Access is revoked immediately — the server is stopped and the staged
copies are deleted, so the URLs die with the share.
