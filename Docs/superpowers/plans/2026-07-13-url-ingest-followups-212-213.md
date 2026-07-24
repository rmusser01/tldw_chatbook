# URL-ingest follow-up hardening (212 + 213) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax. Spec: `Docs/superpowers/specs/2026-07-13-url-ingest-followups-212-213-design.md`. Branch `claude/followups-212-213` off dev `2c5aa25a` (already carries the task-192→214 id-fix commit).

**Goal:** Replace `validate_url`'s brittle regex with `urllib.parse`-based validation (long TLDs / IPv6 / IDN supported, no caller regression), and add the one missing URL-ingest edge test (a URL payload through `persist_parsed_media`).

**Architecture:** Task 1 rewrites `Utils/input_validation.py:validate_url` (production) + extends its unit tests. Task 2 adds one characterization test locking the URL-payload persist seam. Both changes are small and independent.

**Tech Stack:** Python `urllib.parse`, pytest, `MediaDatabase(":memory:")`.

## Global Constraints

- **`validate_url` stays `(str) -> bool`, never raises**, and preserves its existing metrics calls (`log_counter`/`log_histogram`) and the `len(url) > 2000` cap — only the *validation mechanism* changes.
- **No regression:** the existing `Tests/Web_Scraping/test_input_validation.py`, `Tests/Web_Scraping/test_security.py`, and `Tests/UI/test_library_url_ingest_submit.py` suites must stay green **unchanged** (verified empirically: every `valid_urls` → True, every `invalid_urls` + malicious URL → False, all AC cases → True).
- **Security property preserved:** `scheme in ("http","https")` rejects `file:`/`javascript:`/`data:`/`vbscript:`/`about:`/`chrome:`/`ftp:`.
- **No production change for 213** — test-only; `persist_parsed_media` already handles URL payloads.
- **`persist_parsed_media` reads these payload keys** (a missing one `KeyError`s): `file_type, title, media_type, content, keywords, url, analysis_content, author, chunks, chunk_options`. It does NOT read `file_path`.
- **Staging:** explicit paths only. Every commit ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- **Test command** (venv, isolated HOME):
  ```
  HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
    .venv/bin/python -m pytest <target> -q -p no:cacheprovider -o addopts="" --timeout=120
  ```
  (Use the repo-root-relative `.venv/bin/python`.)

---

### Task 1: urlparse-based `validate_url` + unit tests (task 212)

**Files:**
- Modify: `tldw_chatbook/Utils/input_validation.py` (`validate_url` body + a module-level import)
- Modify: `Tests/Web_Scraping/test_input_validation.py` (add test methods to `TestURLValidation`)
- Modify: `backlog/tasks/task-212 - Loosen-validate_url-for-URL-ingest-long-TLDs-IPv6-IDN.md`

**Interfaces:**
- Produces: `validate_url(url: str) -> bool` — unchanged signature; now accepts long-TLD/IPv6/IDN/single-label http(s) URLs and rejects whitespace / malformed-or-out-of-range ports.

- [ ] **Step 1: Write the failing tests**

Add these methods inside `class TestURLValidation:` in `Tests/Web_Scraping/test_input_validation.py`:
```python
    def test_loosened_urls_now_accepted(self):
        """URLs the old TLD/IPv6/IDN-blind regex wrongly rejected must now validate."""
        for url in [
            "https://blog.example.software/post",   # long TLD (>6 chars)
            "https://[::1]/x",                       # IPv6 literal host
            "https://münchen.de/p",                  # IDN (Unicode host)
            "https://foo",                           # single-label host (internal/docker)
        ]:
            assert validate_url(url) is True, f"URL should now be valid: {url}"

    def test_whitespace_and_bad_ports_rejected(self):
        """Raw whitespace and malformed/out-of-range ports are rejected."""
        for url in [
            "http://exam ple.com",                   # space in host
            "https://example.com\t/x",               # tab
            "https://example.com:foo/",              # non-integer port
            "https://example.com:99999999999/x",     # out-of-range port
        ]:
            assert validate_url(url) is False, f"URL should be invalid: {url}"
```

- [ ] **Step 2: Run to verify it fails**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  .venv/bin/python -m pytest \
  "Tests/Web_Scraping/test_input_validation.py::TestURLValidation::test_loosened_urls_now_accepted" \
  "Tests/Web_Scraping/test_input_validation.py::TestURLValidation::test_whitespace_and_bad_ports_rejected" \
  -q -p no:cacheprovider -o addopts="" --timeout=60
```
Expected: FAIL — the old regex rejects the long-TLD/IPv6/IDN/single-label URLs (accepted test fails) and accepts `:99999999999` (rejected test fails on that line).

- [ ] **Step 3: Rewrite `validate_url` (urlparse-based)**

In `tldw_chatbook/Utils/input_validation.py`, add a module-level import near the top (with the other imports, e.g. after `import time`):
```python
from urllib.parse import urlparse
```
Then replace the regex block in `validate_url` — everything from the `# Basic URL pattern` comment through `result = bool(pattern.match(url))` — with:
```python
    if any(c.isspace() for c in url):
        # A valid URL contains no raw whitespace (spaces are %-encoded).
        result = False
    else:
        try:
            parsed = urlparse(url)
            _ = parsed.port  # raises ValueError on a malformed/out-of-range port
            result = parsed.scheme in ("http", "https") and bool(parsed.hostname)
        except ValueError:
            result = False
```
Leave the `if not url or len(url) > 2000:` guard above it and the metrics tail below it (`log_histogram("input_validation_url_duration", ...)`, `log_counter("input_validation_url_result", ...)`, `log_histogram("input_validation_url_length", ...)`, `return result`) exactly as they are. The now-unused `re` import may stay if other functions in the file use it (do NOT remove `re` without confirming no other user).

- [ ] **Step 4: Run the new tests to verify they pass**

Run the Step-2 command. Expected: 2 passed.

- [ ] **Step 5: Run the full regression gate**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  .venv/bin/python -m pytest \
  Tests/Web_Scraping/test_input_validation.py Tests/Web_Scraping/test_security.py \
  Tests/UI/test_library_url_ingest_submit.py \
  -q -p no:cacheprovider -o addopts="" --timeout=120
```
Expected: ALL pass (the existing `valid_urls`/`invalid_urls`/malicious-URL assertions still hold — the rewrite is a verified superset-of-accepts that still rejects every previously-rejected URL, incl. `http://exam ple.com`).

- [ ] **Step 6: Mark backlog task 212 Done**

```bash
perl -0pi -e 's/- \[ \] #1/- [x] #1/' "backlog/tasks/task-212 - Loosen-validate_url-for-URL-ingest-long-TLDs-IPv6-IDN.md"
perl -0pi -e 's/^status: .*/status: Done/mi' "backlog/tasks/task-212 - Loosen-validate_url-for-URL-ingest-long-TLDs-IPv6-IDN.md"
```
Add a short `## Implementation Notes`: replaced the regex with urlparse-based validation (scheme + host + port-guard + whitespace-reject); long-TLD/IPv6/IDN now supported; verified no regression against the three existing suites; single-label hosts now accepted and out-of-range ports now rejected (both intentional).

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Utils/input_validation.py Tests/Web_Scraping/test_input_validation.py \
  "backlog/tasks/task-212 - Loosen-validate_url-for-URL-ingest-long-TLDs-IPv6-IDN.md"
git commit -m "fix(validation): urlparse-based validate_url (long TLD / IPv6 / IDN); task 212 done (212)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: URL-payload persist test (task 213)

**Files:**
- Modify: `Tests/Local_Ingestion/test_ingest_parse_worker.py` (add one test near `test_persist_writes_payload_and_returns_media_id`)
- Modify: `backlog/tasks/task-213 - Add-URL-ingest-edge-path-tests-DNS-permanent-persist-URL-payload.md`

**Interfaces:**
- Consumes: `persist_parsed_media(payload, media_db)` and `MediaDatabase` (both already imported in the test module at lines ~32/42).

- [ ] **Step 1: Write the coverage-lock test**

Add to `Tests/Local_Ingestion/test_ingest_parse_worker.py` (after `test_persist_writes_payload_and_returns_media_id`):
```python
def test_persist_url_payload_writes_article_row_no_filesystem() -> None:
    """A URL-source payload (media_type=article, canonical url, URL string as
    file_path) persists to a media row without any filesystem access -- the
    payload's file_path is a URL that is not a real file, and persist never
    stats/opens it (it only forwards url/content/etc. to the DB)."""
    payload = {
        "file_type": "article",
        "media_type": "article",
        "title": "Kept article",
        "content": "Extracted article body.",
        "keywords": [],
        "url": "https://example.com/post",
        "analysis_content": "",
        "author": "Unknown",
        "chunks": None,
        "chunk_options": None,
        "file_path": "https://example.com/post",  # a URL, NOT a real file -> never accessed
    }

    db = MediaDatabase(":memory:", client_id="test-url-persist")
    media_id, media_uuid, message = persist_parsed_media(payload, db)

    assert isinstance(media_id, int)
    assert isinstance(media_uuid, str) and media_uuid
    row = db.get_media_by_id(media_id)
    assert row is not None
    assert row["url"] == "https://example.com/post"
    assert row["type"] == "article"
```

- [ ] **Step 2: Run to verify it PASSES (coverage lock, not RED-first)**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  .venv/bin/python -m pytest \
  "Tests/Local_Ingestion/test_ingest_parse_worker.py::test_persist_url_payload_writes_article_row_no_filesystem" \
  -q -p no:cacheprovider -o addopts="" --timeout=60
```
Expected: PASS. This is a **characterization/coverage test** — `persist_parsed_media` already handles URL payloads (that is exactly the hand-verified seam #614's review wanted locked), so there is no RED phase. The test succeeding with a non-existent-file URL as `file_path` is itself the "no filesystem access" proof. If the DB assertion names differ (e.g. `row["type"]` vs `row["media_type"]`), align to what `get_media_by_id` returns — the existing `test_persist_writes_payload_and_returns_media_id` uses `row["type"]`.

- [ ] **Step 3: Mark backlog task 213 Done**

```bash
perl -0pi -e 's/- \[ \] #1/- [x] #1/' "backlog/tasks/task-213 - Add-URL-ingest-edge-path-tests-DNS-permanent-persist-URL-payload.md"
perl -0pi -e 's/^status: .*/status: Done/mi' "backlog/tasks/task-213 - Add-URL-ingest-edge-path-tests-DNS-permanent-persist-URL-payload.md"
```
Add a short `## Implementation Notes`: AC #1 (DNS→permanent) was already covered by `test_dns_failure_is_permanent` (shipped in #614); this task adds the missing URL-payload `persist_parsed_media` coverage test.

- [ ] **Step 4: Commit**

```bash
git add Tests/Local_Ingestion/test_ingest_parse_worker.py \
  "backlog/tasks/task-213 - Add-URL-ingest-edge-path-tests-DNS-permanent-persist-URL-payload.md"
git commit -m "test(ingest): lock URL-payload persist_parsed_media seam; task 213 done (213)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Final gate (after Task 2)

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  .venv/bin/python -m pytest \
  Tests/Web_Scraping/test_input_validation.py Tests/Web_Scraping/test_security.py \
  Tests/UI/test_library_url_ingest_submit.py Tests/Local_Ingestion/test_ingest_parse_worker.py \
  -q -p no:cacheprovider -o addopts="" --timeout=180
```
Expected all pass. Then the whole-branch review and finishing-a-development-branch.
