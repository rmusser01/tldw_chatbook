---
id: TASK-20973
title: >-
  Two public media seams let a caller supply the URL that vouches for itself
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - security
  - egress
  - media
  - architecture
priority: medium
dependencies:
  - TASK-19556
---

## Description

Source: raised by **TASK-19556**'s reviewer while verifying the yt-dlp egress
guard. Re-verified at `684c6aba4`.

TASK-19556 put the app's egress policy in front of the two yt-dlp seams, as
`check_url_or_raise(url, trusted_origins=origin_set(url))`
(`Local_Ingestion/video_processing.py:85`). The URL is its own trusted origin.
That is a deliberate and correct choice *for a URL the user typed into the
ingest form*: `config.py`'s `[web_security]` contract permits an explicitly
configured URL to be private, because an intranet media server is a legitimate
source. The function's own docstring states this reasoning
(`video_processing.py:56-62`).

The reasoning holds only as long as every URL arriving at that check is
user-entered. Today it is — but by **wiring, not by invariant**. Two public,
test-covered methods accept caller-supplied URLs and forward them with no
provenance check of any kind:

- `LocalMediaReadingService.process_video(urls=…)`
  (`Media/local_media_reading_service.py:1217`)
- `MediaReadingScopeService.process_video(urls=…)`
  (`Media/media_reading_scope_service.py:2112`)

Neither has an in-app caller today — the only production reference to a
similarly named method is the unrelated private
`BackendIntegration._process_video` — so nothing is exposed right now. But
`trusted_origins=origin_set(url)` means a URL that reaches that seam from
anywhere else instructs the policy to trust its own host, which is precisely
what the policy exists to refuse for untrusted input. Wiring either seam to any
source that is not the ingest Input — a config file, an API payload, a feed, an
agent tool — silently converts a correct guard into no guard, with no test
failing and no reviewer prompted to look.

This is the "safe because of who happens to call it" shape rather than "safe by
construction". The fix wanted is a provenance decision at the boundary, not a
larger denylist.

## Acceptance Criteria

- [ ] A URL reaching the yt-dlp egress check is trusted as its own origin only
      when its provenance actually establishes that trust; a caller-supplied URL
      of unknown provenance is not self-trusting
- [ ] The two `process_video(urls=…)` seams either establish provenance
      explicitly or cannot reach the self-trusting path
- [ ] The trust decision is expressed where it is made rather than inferred from
      the current caller set, so adding a caller cannot silently change it
- [ ] A test proves that wiring a new, non-user-entered source into these seams
      does not grant self-trust — i.e. the guarantee survives a hypothetical
      future caller, which is the property that does not hold today
- [ ] Existing user-entered ingest of a private or intranet media URL still
      works, and a test pins it, so the fix does not close a legitimate case
- [ ] The residual limits TASK-19556 documented (yt-dlp's own redirect hops,
      extractor-discovered per-format URLs, resolve-then-connect TOCTOU) remain
      accurately stated and are not implied to be closed by this work

## Notes

Filed medium with no live exposure, deliberately. The severity is not what an
attacker can do today — nothing reaches these seams — it is that the security
property is held by a fact nobody is watching, one wiring commit from being
false, and that the commit which makes it false will look entirely ordinary.
