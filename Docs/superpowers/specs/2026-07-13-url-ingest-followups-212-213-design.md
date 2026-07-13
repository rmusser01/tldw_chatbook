# URL-ingest follow-up hardening (tasks 212 + 213)

**Status:** Design approved (brainstorm), pending spec review.
**Backlog:** task-212 (loosen `validate_url`) + task-213 (URL-ingest edge tests) — both from #614's whole-branch review of task 162.
**Branch:** `claude/followups-212-213` off dev `2c5aa25a` (already carries the task-192→214 id-fix commit).

## Problem

Task 162 made `Utils/input_validation.py:validate_url` load-bearing at the Library ingest form. Its regex requires a 2-6 char TLD and has no IPv6/IDN support, so valid URLs — `https://blog.example.software/post` (long TLD), `https://[::1]/x` (IPv6), Unicode-domain URLs (IDN) — are falsely rejected there. It's a **shared** validator (~13 call sites: LLM/provider endpoint URLs, article scraping, RAG state, console gateway, settings probes), so any change must not regress those.

Separately, #614's review verified two ingest seams by hand that needed automated coverage. One — a `socket.gaierror`-caused httpx error mapping to `PermanentIngestError` — was **already covered** by tests added during #614's bot-resolution (`test_dns_failure_is_permanent`). The other — a URL payload flowing through `persist_parsed_media` with no filesystem access — is still uncovered (the existing persist test is file-only).

## Goal / Acceptance

- **AC1 (212)** — a long-TLD URL (e.g. `.software`) is accepted by the Library ingest form; IPv6 and IDN hosts are supported; existing `validate_url` callers are unaffected (no regression).
- **AC2 (213)** — a URL payload (`media_type='article'`, canonical url, `file_path`=URL string) is accepted end-to-end by `persist_parsed_media` with no filesystem access. (The DNS→permanent seam is already covered — see Non-goals.)

## Component 1 (212): urlparse-based `validate_url`

Replace the brittle regex with `urllib.parse`-based validation, preserving the length cap and the metrics/logging:

```python
def validate_url(url: str) -> bool:
    start_time = time.time()
    log_counter("input_validation_url_attempt")
    if not url or len(url) > 2000:
        log_counter("input_validation_url_invalid", labels={"reason": "empty" if not url else "too_long"})
        return False
    if any(c.isspace() for c in url):          # a valid URL contains no raw whitespace
        result = False
    else:
        from urllib.parse import urlparse
        try:
            parsed = urlparse(url)
            _ = parsed.port                     # raises ValueError on a malformed/out-of-range port
            result = parsed.scheme in ("http", "https") and bool(parsed.hostname)
        except ValueError:
            result = False
    # ... existing duration/result/length metrics unchanged ...
    return result
```

**Why urlparse over patching the regex:** it natively handles long TLDs, IPv6 (`[::1]` → host `::1`), IDN (Unicode host), localhost, IPs, and ports — cleanly satisfying the AC's "supported" branch, versus a growing, fragile regex. The `scheme in ("http","https")` check preserves the security property (rejects `file:`/`javascript:`/`data:`/`vbscript:`/`about:`/`chrome:`/`ftp:`).

**Verified no-regression (empirical):** the proposed function was run against every existing assertion — all 7 `valid_urls` → True, all 13 `invalid_urls` → False, all 7 `test_security.py` malicious URLs → False, and all 3 AC cases → True. Zero regressions.

**The whitespace guard is load-bearing:** `http://exam ple.com` (space in domain) is in the existing "should be False" list, and urlparse alone gives it a non-empty hostname → it would wrongly pass. Rejecting any raw-whitespace URL fixes that (and rejects tabs/newlines, which the old regex tolerated via `$`-before-`\n` — an accepted, more-correct change).

**Two intentional broadenings** (no test or caller depends on the old behavior; security unchanged):
- single-label hosts (`https://foo`) now validate — correct for internal/docker hosts;
- out-of-range ports (`:99999999999`) now rejected — the old `:\d+` regex accepted them.

**Testing:** extend `Tests/Web_Scraping/test_input_validation.py` with the new accept cases (long-TLD, IPv6, IDN) and the whitespace-/bad-port-reject cases. The existing `test_input_validation.py`, `Tests/Web_Scraping/test_security.py`, and `Tests/UI/test_library_url_ingest_submit.py` suites are the regression gate — all must stay green unchanged.

## Component 2 (213): URL-payload persist test

Add one test near `test_persist_writes_payload_and_returns_media_id` in `Tests/Local_Ingestion/test_ingest_parse_worker.py` (which currently covers only a *file* payload):

- Build a URL payload dict directly (mirroring the URL-source tail of `parse_local_file_for_ingest`): `media_type='article'`, `file_type='article'`, `url='https://example.com/post'`, `file_path='https://example.com/post'`, `title`, `content`, `keywords=[]`, `analysis_content=''`, `chunks=None`, `chunk_options=None`.
- `persist_parsed_media(payload, MediaDatabase(":memory:", client_id="test-url-persist"))`.
- Assert the returned `media_id` is an int and `db.get_media_by_id(media_id)` has `row["url"] == "https://example.com/post"` and `row["type"] == "article"`.
- **No-filesystem-access proof:** the payload's `file_path` is a URL string that is *not* a real file. `persist_parsed_media` never stats/opens it (it only passes `url`/`content`/… to `add_media_with_keywords`), so the test succeeding is itself the proof — no file exists at `https://example.com/post`, so any filesystem dependency on `file_path` would fail. `MediaDatabase(":memory:")` keeps the DB write in RAM.

## Data flow

212 changes only the internal validation mechanism of `validate_url`; its `(str) -> bool` contract and metrics are unchanged. 213 adds a test only. No production behavior changes beyond `validate_url` accepting the previously-rejected valid URLs.

## Error handling

`validate_url` returns `False` (never raises) for every invalid input, including a malformed/out-of-range port (the `.port` `ValueError` is caught) and `None`/empty/over-length — matching the current contract.

## Testing summary

- `validate_url` unit tests (new accepts + rejects) in `test_input_validation.py`; existing `test_input_validation` / `test_security` / `test_library_url_ingest_submit` as the regression gate.
- One URL-payload `persist_parsed_media` test in `test_ingest_parse_worker.py`.

## Scope / non-goals

- **213 test #1 (DNS→permanent) is already done** (`test_dns_failure_is_permanent` + `test_connection_error_without_dns_cause_is_retryable`, shipped in #614). Not re-implemented.
- No changes to the ~13 `validate_url` callers, no UI copy changes (the AC's "actionable rejection copy" branch is moot — IPv6/IDN are now *supported*, so the form accepts them).
- The `_has_valid_url_port` helper in `console_session_settings.py` (an extra per-caller port check) is left as-is.
