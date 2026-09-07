# Watchlists reading path — walked with real content, 2026-07-28

`origin/dev` `79152bbb6`, clean profile, 235x52. First walk with genuine
scraped content, now that `Check now` works (task-1100).

Fetched `https://summitroute.com/blog/feed.xml`: **10 items**, with real article
bodies stored — 6,194 and 12,193 bytes for the first two.

## Feeds → Items: works

The Items table lists all ten with title, source, status `new` and created date:

```
Title                                                                Source        Status
Lightsail object storage concerns - Part 2                           Summit Route  new
Lightsail object storage concerns - Part 1                           Summit Route  new
S3 backups and other strategies for ensuring data durability ...     Summit Route  new
```

Clicking an item selects it and the Inspector names it correctly.

## Items → Content: there is no reader

The Content region is still the Phase D placeholder:

```
Content
Reader arrives in the next slice.
```

So the path ends at the list. Ten articles with full bodies are sitting in
`subscription_items.content` and nothing in the app can display them.

## Found

**task-1120 (high)** — a selected item is classified as a source. The Inspector
reports `Type: source` and offers `Preview` / `Check now`; the item actions
(`Mark reviewed`, `Ingest`, `Ignore`) never appear, so an item cannot be acted
on at all.

## Where this leaves Phase D

Phase D is the reader, and it now has real content to build against rather than
placeholders — which is the first time that has been true. Two things should be
fixed before or as part of it: task-1120 above, and task-1105 (clicking a row
other than the first does not move the selection), since both sit directly on
the path a reader user takes.
