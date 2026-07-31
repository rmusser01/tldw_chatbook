---
id: TASK-1537
title: 'Remote inline transcript images behind a security setting'
status: Done
assignee: []
created_date: '2026-07-30 17:20'
labels: [enhancement, console, security]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Assistant replies that reference images by link (markdown `![](url)` or a
bare image-extension URL, http(s) only) can render them inline. OFF by
default -- fetching a model-suggested URL leaks the reader's IP/UA to that
host -- behind `[chat.images] render_remote_images = true`. Fetches go
through the egress-hardened image GET
(`image_format_utils.fetch_image_bytes`: per-hop SSRF policy, credential
stripping on cross-origin redirects, Content-Length + streamed byte caps at
8 MB), require an image/* content type, scan only the most recent 20
assistant replies, attempt each URL at most once per screen lifetime, and
share the bounded transcript render cache under per-URL `remote:` keys
(failures negative-cache). Rows appear on the next transcript sync tick
after a fetch lands and reuse the existing image-row pipeline (modes,
viewer, budgets).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Default off: no fetch is dispatched and no row rendered without the setting.
- [x] #2 Enabled + cached link renders an image row for the message; uncached link dispatches exactly one egress-gated fetch per URL.
- [x] #3 URL extraction accepts markdown image links and bare image-extension URLs only, http(s) only, deduped and capped.
- [x] #4 All fetches route through the per-hop-validated egress GET with byte caps and content-type check.
<!-- AC:END -->

## Implementation Notes (live-verification addendum)

Live end-to-end run (textual-serve + Playwright + local llama.cpp): the model
emitted the requested markdown image line, the egress GET fetched 44 KB of
image/jpeg from gstatic.com, and the row rendered inline. Two findings from
the live pass, both fixed:

1. The Console sync is DEMAND-driven, not a free-running timer -- after the
   fetch cached the image, nothing requested a re-render, so the row stayed
   invisible until an unrelated UI action. The fetch worker now requests
   `_sync_native_console_chat_ui()` after a successful prepare.
2. The persistent app log admits only diagnostics events (ADR-029 sink), so
   module-level logger warnings from this path are NOT visible there --
   debugging used a temporary file probe, since removed.

Also: the viewer modal needed explicit inline cell sizes -- the app-tier CSS
bundle outranks a modal's DEFAULT_CSS and a Container's default 100% height
collapses to zero inside the auto-sized modal (caught live, pinned by a
region-size assertion; harness CSS does NOT reproduce bundle interference).
