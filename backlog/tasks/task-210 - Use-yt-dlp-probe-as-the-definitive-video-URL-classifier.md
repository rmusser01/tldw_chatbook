---
id: TASK-210
title: Use yt-dlp probe as the definitive video-URL classifier
status: To Do
assignee: []
created_date: '2026-07-12 18:56'
labels:
  - follow-up
  - ingest
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
URL ingest (task 162) classifies media-vs-article by a URL pattern heuristic (host + extension). For ambiguous URLs, borrow tldw_server2's approach: try yt_dlp.extract_info(url, download=False) (catch DownloadError/UnsupportedError) as the definitive 'is this a playable video' check instead of a hand-rolled host list. More accurate for media on unknown hosts. Discovered mining the server media pipeline during task 162.
<!-- SECTION:DESCRIPTION:END -->
