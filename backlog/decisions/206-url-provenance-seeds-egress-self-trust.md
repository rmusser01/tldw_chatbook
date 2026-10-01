# ADR-206: URL provenance, not caller identity, seeds egress self-trust

Status: Accepted
Date: 2026-09-30
Related Task: [task-20973 - Two public media seams let a caller supply the URL that vouches for itself](../tasks/task-20973%20-%20Two%20public%20media%20seams%20let%20a%20caller%20supply%20the%20URL%20that%20vouches%20for%20itself.md)
Supersedes: N/A (extends the contract recorded in `Utils/egress.py`'s module
docstring and `Docs/superpowers/specs/2026-07-23-web-fetch-hardening-design.md`;
TASK-19556 implemented that contract's media arm and is the direct predecessor)

## Decision

A URL is its own egress `trusted_origins` only when an explicit
`Utils.egress.UrlProvenance.USER_ENTERED` value, threaded down from the
boundary that established user entry, says so. Every seam the value has not
reached defaults to `UrlProvenance.UNKNOWN` (fail closed: no self-trust).
The enum is minted in exactly one place today — the Library ingest queue
(`app_ingest_queue._ingest_job_options`), derived from the job's own
lineage: general import-form submissions are `USER_ENTERED`,
research-source catalog jobs are `UNKNOWN`. No pipeline or service may
mint it.

## Context

TASK-19556 put the egress policy in front of the two yt-dlp seams as
`check_url_or_raise(url, trusted_origins=origin_set(url))` — the URL
vouching for itself. That is correct for a URL the user typed into the
ingest form (`[web_security]` permits configured URLs to be private; an
intranet media server is a legitimate source), and the property held —
but "by wiring, not by invariant": `LocalMediaReadingService.
process_video(urls=…)` and `MediaReadingScopeService.process_video(urls=…)`
are public, test-covered methods that forwarded any caller-supplied URL
into the self-trusting check. A research-run catalog shares the same
parse pipeline today; a config value, API payload, feed or agent tool is
one ordinary-looking wiring commit away. The guard would silently become
no guard, with no test failing and no reviewer prompted to look.

`Utils/egress.py`'s contract already said "trust is seeded only at
boundaries where user intent is known and threaded down"; the media arm
was the one place that sentence was not actually enforced.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Keep self-trust, document the caller set | The exact defect: safety held by a fact nobody watches; adding a caller silently changes the trust decision. |
| A larger denylist (block private ranges harder) | Closes the legitimate intranet-media case the `[web_security]` contract explicitly permits; also does not answer WHO may vouch. |
| A boolean `user_entered` flag | Same idea, weaker vocabulary: cannot grow (configured/discovered provenances), and a bare `True` in a call reads as informational, not as a trust mint. |
| Mint `USER_ENTERED` inside `parse_local_file_for_ingest` for every URL | Moves the "who happens to call" assumption one level down; research-catalog URLs share that seam. |
| Thread provenance to the server backend too | The server fetches in another process whose own policy this parameter cannot reach, and a trust claim crossing a process boundary is unverifiable — sending it would be theatre. |

## Consequences

- The two `process_video` seams (and `download_video`/`extract_metadata`/
  `process_videos`/`parse_local_file_for_ingest`/`run_parse_job` below
  them) carry a keyword-only `url_provenance` defaulting to `UNKNOWN`;
  callers that have established user entry must pass it explicitly.
- Research-source URLs no longer self-trust (a real tightening: they are
  agent-discovered content; a private target among them is now refused,
  matching the policy's stance on content-derived URLs).
- The minted enum rides the pickled parse-`options` dict across the spawn
  pool; `run_parse_job` translates it back and accepts the enum only — a
  plain string cannot launder trust.
- Deliberately unchanged: the audio arm
  (`audio_processing.download_audio_file`'s own `origin_set(url)`,
  TASK-19556's parity choice) — same shape, different task; and the
  metadata-endpoint and non-http(s) refusals, which apply regardless of
  provenance.
- TASK-19556's residual limits are unchanged and NOT claimed closed:
  yt-dlp's own redirect hops, extractor-discovered per-format URLs, and
  the resolve-then-connect TOCTOU window.

## Links

- TASK-19556 (predecessor; its guard is the one whose justification this
  contract replaces):
  `backlog/tasks/task-19556 - Three outbound seams never adopted the egress policy including a typing-triggered internal port-scan oracle.md`
- `Utils/egress.py` module docstring (the contract's canonical text)
- `Docs/superpowers/specs/2026-07-23-web-fetch-hardening-design.md`
- Tests: `Tests/Local_Ingestion/test_video_url_provenance.py`
