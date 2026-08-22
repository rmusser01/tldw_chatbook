---
id: TASK-19556
title: >-
  Three outbound seams never adopted the egress policy, including a
  typing-triggered internal port-scan oracle
status: Done
assignee:
  - '@claude'
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

- [x] `ingest_preflight` performs no network request that bypasses the egress
      policy, and no network request at all before URL validation
- [x] The preflight no longer returns outcomes that distinguish "refused" from
      "answered" for private/internal addresses — the typing-debounced path
      must not be usable as a scanning oracle
- [x] The debounce-while-typing behaviour is reviewed: probing on every
      keystroke pause is the wrong default even for allowed hosts
- [x] `video_processing.py`'s URL entry points apply the same egress check the
      article arm of `audio_processing.py` already applies, so the media and
      article arms of the same entry point behave identically
- [x] Content-derived URLs are never passed as their own `trusted_origins` —
      the sitemap/crawl call sites are corrected to match the stated contract
      at `config.py:3731`
- [x] Tests pin each of the three seams against a private/internal target and
      fail if the guard is removed (mutation-checked)
- [x] A guard test fails if a new outbound call site reaches the network
      without passing through the egress policy, so the next seam cannot
      silently skip it

## Implementation Plan

1. Prove each defect before touching it: a born-red test per seam, at the
   branch base, using in-process transports only (the suite forbids real
   sockets) and synthetic sentinels.
2. (a) Close the oracle in three moves, not one: route the probe through the
   egress policy, collapse the outcome vocabulary, and take the probe off
   the typing path.
3. (b) Apply at the two yt-dlp seams the same check `audio_processing`
   already applies, and state honestly what a pre-check cannot cover.
4. (c) Make `trusted_origins` an explicit, fail-closed parameter on the
   sitemap/crawl seams, applied to the caller-named entry URL only; re-check
   reachability and record it rather than describing latent code as live.
5. Add the census guard so the next seam cannot skip the policy silently.
6. Baseline every failure against the merge base before attributing it.

## Implementation Notes

### (a) The typing-triggered port-scan oracle — CONFIRMED LIVE, closed

Reproduced first. At base, `analyze_path("http://10.255.255.1:8080/report.pdf")`
opens a real TCP connection — the suite's network guard recorded
`('socket.create_connection', '10.255.255.1:8080')` — and the three targets
below produced three *different* user-visible results:

```
(('URL unreachable — the connection was refused.',), (), (), 0, False)   # :8080 refused
((), ('Could not check the link',), ('pdf',), 1, False)                  # :9200 answered 403
((), (), ('pdf',), 1, False)                                            # :22 answered 200
```

That is the oracle, verbatim. Three changes close it.

**The debounce default — what was actually done.** The probe is now OFF by
default (`[library] ingest_url_preflight_probe = false`, documented in the
shipped TOML), so the typing path issues no network request at all: a URL is
classified by name, exactly as a local path is classified by `stat`. The
config gate alone was judged insufficient, because a user who opted in would
be back to hitting the host on every 0.8 s pause — the thing the AC calls the
wrong default *even for allowed hosts*. So the typing timer additionally
passes `allow_probe=False` unconditionally
(`library_screen._run_debounced_library_ingest_preflight` →
`_trigger_library_ingest_preflight(path, allow_probe=False)` →
`analyze_path(..., probe_url=False)`). Probing, when enabled, runs only from
the deliberate triggers: blur, Enter, Browse…, the retry button. The 0.8 s
timer itself was left alone — it exists for local path/directory feedback,
which is cheap and local, and changing the interval would have been arbitrary.

**The outcome vocabulary — what was actually done.** `_probe_url` calls
`check_url_or_raise` before any transport call and returns ONE constant note
(`_UNVERIFIABLE_NOTE`) for every declined reason. `EgressBlockedError.reason`
is deliberately not read: private vs. dns_failure vs. metadata vs. bad-scheme
is precisely the difference an internal scan wants, and it is not actionable
for the user anyway. Verified: a metadata endpoint and an RFC1918 host now
reduce to the identical observable. The policy is consulted with **no**
trusted origins — an automatic probe is not the user asking to contact a
host, and self-trusting would have made the check a no-op for exactly the
private hosts at issue. (The deliberate ingest that follows keeps its own
trust; `[web_security]` says configured URLs may be private.)

Two further hardenings fell out of the same read: `validate_url` now runs
*before* anything else (at base, `http://user:secret@example.com/doc.pdf`
was accepted with zero errors and its embedded credential put on the wire),
and the probe uses a no-redirect opener, since `urlopen` auto-followed
redirects and a public host answering `302 Location: http://10.0.0.5:8080/`
walked the probe into internal space *after* the check had passed.

### (b) yt-dlp — CONFIRMED LIVE, guarded, with the gap stated

`check_media_url_egress` applies `check_url_or_raise(url,
trusted_origins=origin_set(url))` — the same shape
`audio_processing.download_audio_file` uses — at both yt-dlp seams:
`download_video` (which covers its size probe *and* its download) and
`extract_metadata`. At base, `extract_metadata` fetched
`http://169.254.169.254/latest/meta-data/iam/security-credentials/` and
returned its metadata.

**What the fix covers:** cloud metadata endpoints (blocked regardless of
trust — the policy's one hard rule), non-http(s) schemes (yt-dlp will hand
`file://` and dozens of other protocols to its extractors), and the entry
URL's classification. A user-typed private URL still works, matching the
audio arm and the `[web_security]` contract — pinned by a test, so the guard
cannot be "fixed" into breaking intranet media ingest.

**What it does NOT cover, plainly:** yt-dlp does its own fetching. This is a
pre-check on the entry URL only. It cannot re-validate yt-dlp's own redirect
hops, the per-format media URLs an extractor discovers inside a page, or a
DNS answer that changes between the check and yt-dlp's own resolution — the
TOCTOU window `Utils/egress.py` documents as a residual for every consumer of
a resolve-then-connect check. Closing those needs a yt-dlp request hook and
was out of scope. This is stated in the function docstring and the test
module docstring, not just here.

### (c) trusted_origins — corrected as a correctness defect; reachability re-checked

Corrected: `Article_Extractor_Lib.scrape_from_sitemap`,
`Article_Extractor_Lib.collect_internal_links`,
`Article_Extractor_Lib.recursive_scrape`,
`Article_Scraper/crawler.get_urls_from_sitemap`, and
`Article_Scraper/crawler.crawl_site`. Each gained an explicit
`trusted_origins` keyword defaulting to `frozenset()` (matching
`scrape_article`, `get_page_title` and `ScraperConfig`, which were already
fail-closed), and each now applies it to the caller-named entry URL **only**.
Two distinct halves were wrong, and both are fixed:

* self-trusting the URL about to be fetched (`origin_set(sitemap_url)`) —
  against `Utils/egress.py`'s "shared pipeline code must NEVER auto-trust its
  own input URL";
* forwarding that trust to what the fetched content named —
  `scrape_article(url.text, trusted_origins=origins)` for every `<loc>` in
  the sitemap, and every link discovered mid-crawl. `config.py`'s
  `[web_security]` block names "sitemap/crawl discoveries" verbatim among the
  content-derived URLs that must resolve to public IPs.

**Reachability, re-checked at this base and unchanged from the filing:
LATENT, and described as such.** None of the five has an in-app caller.
`scrape_from_sitemap` is reached only from `scrape_and_convert_with_filter`
(no callers); `get_urls_from_sitemap` and `crawl_site` only from their own
docstring examples; `collect_internal_links` from `scrape_entire_site` /
`create_filtered_sitemap` (no callers). The one shipped path that fetches a
sitemap is the Watchlists `sitemap` source type, and it goes through
`Subscriptions/local_watchlists_service._urls_for_sitemap`, whose
`origin_set(source)` **is** provenance-correct — `source` is the subscription
URL the user configured. So this change is behaviour-neutral for the shipped
app, which is also why it carried no regression risk.

**`UI/Screens/settings_image_gen_defaults.py:768` — checked, NOT changed, and
why.** The shape is identical (`check_url_or_raise(url,
trusted_origins=origin_set(url))`) but the *provenance* is not: every caller
of `_guarded_get` derives its URL from a base_url the user typed into the
Settings ▸ Image Generation form (`_probe_swarmui`,
`_probe_reachability_only`, and `_probe_comfyui` via
`normalize_comfyui_image_origin`). That is an explicitly configured URL,
which `config.py:3807`'s contract permits to be private — and the shipped
default backend is a **localhost** SwarmUI instance, so removing the
self-trust would break the zero-key default probe. Existing tests already pin
both halves of that intent
(`Tests/UI/test_settings_image_gen_defaults.py::test_probe_egress_allows_private_base_url_via_self_trust`
and `::test_probe_egress_blocks_metadata_ip_even_self_trusted`). A comment
was added at the call site recording the provenance so the two shapes are not
confused again.

### The census guard (last AC)

`Tests/Utils/test_egress_adoption_census.py` fails when a module under
`tldw_chatbook/` uses a `urllib.request` opener or constructs a
`yt_dlp.YoutubeDL` without naming any egress-policy symbol. Run at base it
independently rediscovered exactly the two seams this task names:
`Library/ingest_preflight.py` and `Local_Ingestion/video_processing.py`.
One allowlisted exemption, with its reason recorded in the file:
`Local_Ingestion/parakeet_v2_installer.py` (constant huggingface.co URL, no
caller-supplied component, size- and SHA-256-pinned). Its scope is stated
rather than overclaimed: it does not census bare `requests`/`httpx`, because
this app has dozens of legitimate fixed-endpoint calls in `LLM_Calls/` and
that census would be noise that gets suppressed rather than a guard that gets
read. Two self-tests keep it honest — the detector is proved to detect, and a
stale exemption (a module that no longer opens URLs) fails.

### Born-red evidence

Every test was run against the unmodified base `f12bb21ad` first. Counts:
(a) 8 failed / 0 passed; (b) 4 failed / 2 passed (the two passes are the
must-not-over-block parity pins); (c) 10 failed / 0 passed; census 1 failed /
2 passed. Signatures are quoted in each test module's docstring. Because the
base *is* the pre-fix code, these runs are also the mutation check for every
born-red assertion. The tests written after the fix (the debounce wiring
pair) are mutation-checked separately, recorded below.

### Tests changed rather than added

Three existing tests asserted the retired behaviour and were updated with the
reason inline, not silently:
`Tests/Web_Scraping/test_web_fetch_wiring.py` pinned
`trusted_origins == frozenset({"sitemap.internal"})` — i.e. it pinned the
defect; `Tests/Library/test_ingest_preflight.py`'s probe tests now opt into
the (default-off) probe via a `probing_allowed` fixture and patch the new
`_open_probe` seam; `Tests/Library/test_library_ingest_state.py` no longer
needs a `urlopen` stub to classify a URL. Two UI stubs
(`test_library_ingest_inline_consent.py`, `test_library_ingest_retry_last.py`)
gained `**_kwargs` for `analyze_path`'s new keyword.

### Modified / added files

Source: `tldw_chatbook/Library/ingest_preflight.py`,
`tldw_chatbook/UI/Screens/library_screen.py`,
`tldw_chatbook/Local_Ingestion/video_processing.py`,
`tldw_chatbook/Web_Scraping/Article_Extractor_Lib.py`,
`tldw_chatbook/Web_Scraping/Article_Scraper/crawler.py`,
`tldw_chatbook/UI/Screens/settings_image_gen_defaults.py` (comment only),
`tldw_chatbook/config.py` (new `[library] ingest_url_preflight_probe`).
Docs: `Docs/User_Guide/library/import-and-export.md`.
Tests added: `Tests/Library/test_ingest_preflight_egress.py`,
`Tests/Local_Ingestion/test_video_egress_guard.py`,
`Tests/Web_Scraping/test_sitemap_crawl_trusted_origins.py`,
`Tests/Utils/test_egress_adoption_census.py`.

### Independent review (2026-08-22, adversarial)

The oracle closure was re-verified from scratch against base `f12bb21ad`,
not taken on report. Confirmed independently, with in-process transports
and a stubbed resolver (no real sockets):

* **Both halves hold.** Gate off by default -> `analyze_path` on a URL
  issues zero transport calls *and* zero DNS lookups. Opted in, the typing
  path (`probe_url=False`) still issues neither, while a deliberate trigger
  does. No third caller reaches `analyze_path`/`_probe_url`: six call sites
  in `library_screen`, one of them the typing timer with `allow_probe=False`.
* **The vocabulary really is collapsed.** Fourteen declined targets --
  private/loopback/CGNAT/multicast/metadata literals, the same via DNS, a
  DNS failure, `metadata.google.internal`, `[::1]`, and an IPv4-mapped v6
  private address -- produce ONE identical observable, and none of them
  reaches the transport at all.
* **No usable timing distinguisher.** Nothing connects for a declined
  address, so there is no connect latency to time. The only latency split
  is "IP literal (no DNS)" vs "hostname (one DNS lookup)", which discloses
  nothing an attacker choosing the address does not already know.
* **Both incidental findings reproduce at base and are closed here.**
  `http://user:secret@example.com/doc.pdf` -> at base: `errors=[]`,
  `warnings=[]`, classified `pdf`, and the credential-bearing URL recorded
  on the wire; now `path_invalid=True` and nothing is opened. The 302 walk:
  at base the probe opened `http://public.example.test/doc.pdf` **and then**
  `http://10.0.0.5:8080/`; now only the first, surfaced as an answered-302
  note.

**Two gaps found and closed** (AC 6 says "mutation-checked"; these two
survived mutation):

1. `_run_library_ingest_preflight`'s `probe_url=None if allow_probe else
   False` -- the step that JOINS the debounce wiring to `analyze_path` --
   was pinned by nothing. Rewriting it to a bare `probe_url=None` left 509
   tests green (only the pre-existing red) while putting an opted-in user
   back on a per-0.8 s-pause probe.
2. The no-redirect opener was pinned by nothing. Reverting `_open_probe` to
   `build_opener()` left 2252 tests green, because every other test in the
   module patches `OpenerDirector.open` and so never reaches the handler
   chain that would follow a redirect.

Three tests added to `Tests/Library/test_ingest_preflight_egress.py`; each
is born-red at `f12bb21ad` and red under its mutation. Module: 13 tests,
12F/1P at base.

**Residual, filed rather than changed here:** `Input.Blurred` is posted by
Textual on `AppBlur` (terminal focus loss) as well as on a focus move --
demonstrated live -- so for an *opted-in* user, alt-tabbing away with a URL
staged fires the probe with `allow_probe=True`. Same category as the
debounce (not a gesture aimed at the field), far lower frequency, and inert
under the default-off gate; changing that blur handler is a behaviour change
with its own live-verification cost (its docstring records a prior reversal
after an xhigh review + live-verify round). Recommended as a separate LOW
follow-up rather than folded in here; no task id minted by the reviewer, to
avoid an id collision against `origin/dev`.

Also corrected: the `settings_image_gen_defaults` provenance comment
enumerated three of its five callers (omitting `_probe_openai_compatible`
and `_probe_gemini`, the two that attach a credential header) and said
"typed into this settings form" when a blank field falls back to the
resolved `[image_generation]` config value. The exception's CONCLUSION
survives -- every source is operator-configured, and the only writer of that
config section is the Settings screen's own save -- but the reasoning as
written did not match the code.

Baselines re-measured, not accepted: `Tests/UI/test_library_ingest_*` is
9F/233P on BOTH sides with identical failure sets, and `Tests/integration`
is 3F/69P on both sides (the commit message says 2 integration reds; it is
3, all pre-existing). Post-review sweep across
`Tests/{Library,Web_Scraping,Local_Ingestion,Utils}` + the image-gen
settings suite: **3685 passed / 5 skipped / 0 failed**.
