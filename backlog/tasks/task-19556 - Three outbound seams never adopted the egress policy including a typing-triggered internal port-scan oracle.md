---
id: TASK-19556
title: >-
  Three outbound seams never adopted the egress policy, including a
  typing-triggered internal port-scan oracle
status: To Do
assignee: []
created_date: '2026-08-21 20:06'
labels:
  - security
  - egress
  - ssrf
priority: high
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 2 (security & privacy) — its **Tier 2
#6**. Re-verified at this branch base.

The egress policy itself is **well built** and was verified by live probe in
this review: DNS-resolution based, `follow_redirects=False`, per-hop recheck,
on by default, and it blocks cloud metadata IPs, loopback, RFC1918, CGNAT,
link-local, SSDP and non-HTTP schemes. The defect is three seams that never
call it.

**(a) CONFIRMED LIVE — ingest preflight is an internal host+port scanning
oracle.** `tldw_chatbook/Library/ingest_preflight.py:15,221` uses a bare
`urlopen` HEAD (`from urllib.request import Request, urlopen`), which
auto-follows redirects, and it fires on a **0.8 s debounce while the user is
typing** (`library_screen.py`) — *before* `validate_url` runs.
`analyze_path` (`ingest_preflight.py:255`) returns distinguishable outcomes —
refused / answered-{code} / clean — so a pasted link drives an
attacker-readable probe of the user's internal network. There is no egress
call anywhere in the module.

**(b) CONFIRMED LIVE — yt-dlp handed unchecked URLs.**
`tldw_chatbook/Local_Ingestion/video_processing.py` passes the URL straight to
`yt_dlp.YoutubeDL(...).extract_info(url, ...)` for both the probe and the
download, and contains **zero** references to the egress helpers.
`audio_processing.py` is the instructive contrast: it **imports the egress
helpers at line 265** — but only for its plain-HTTP branch. The article arm of
the same entry point *is* guarded; the media arm is not. This is a seam that
half-adopted the primitive.

**(c) CONFIRMED code shape / LATENT reach — the guard is handed the attacker's
own hostname as trusted.** Sitemap and crawl paths call the checker with
`trusted_origins=origin_set(<the content-derived URL itself>)` —
e.g. `Web_Scraping/Article_Extractor_Lib.py:1032`,
`Web_Scraping/Article_Scraper/crawler.py:352`, and the same shape at
`UI/Screens/settings_image_gen_defaults.py:768`. Trusting the origin of the
URL you are about to fetch, when that URL came from fetched content, defeats
the check for exactly the input it needs to catch. This **contradicts
`config.py:3731` verbatim**, which states that content-derived URLs are the
ones the policy exists to re-check. Reachability is honestly LATENT for the
sitemap arm: the create form omits the sitemap field, though the database
permits the row — so the code path is real but not currently reachable through
the shipped UI. It should be fixed as a correctness defect and its
reachability re-checked, not treated as an active exploit.

## Acceptance Criteria

- [ ] `ingest_preflight` performs no network request that bypasses the egress
      policy, and no network request at all before URL validation
- [ ] The preflight no longer returns outcomes that distinguish "refused" from
      "answered" for private/internal addresses — the typing-debounced path
      must not be usable as a scanning oracle
- [ ] The debounce-while-typing behaviour is reviewed: probing on every
      keystroke pause is the wrong default even for allowed hosts
- [ ] `video_processing.py`'s URL entry points apply the same egress check the
      article arm of `audio_processing.py` already applies, so the media and
      article arms of the same entry point behave identically
- [ ] Content-derived URLs are never passed as their own `trusted_origins` —
      the sitemap/crawl call sites are corrected to match the stated contract
      at `config.py:3731`
- [ ] Tests pin each of the three seams against a private/internal target and
      fail if the guard is removed (mutation-checked)
- [ ] A guard test fails if a new outbound call site reaches the network
      without passing through the egress policy, so the next seam cannot
      silently skip it
