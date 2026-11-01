# S14-web validation

Note: `git diff --stat 3722a85748..HEAD -- tldw_chatbook/Personal_Context tldw_chatbook/Web_Scraping tldw_chatbook/Web_Server tldw_chatbook/WebClipper` is empty — none of these files changed since the review commit, so line numbers below are unchanged from the slice.

## 1. P1 [D1] — Every production Confluence API call bypasses the module's own egress guard
- Verdict: CONFIRMED
- Site now: `Web_Scraping/Confluence/confluence_auth.py:289` (guard predicate `set(kwargs) <= {"headers", "timeout"}`)  (was same)
- Proof: `grep -rn "make_request" tldw_chatbook/Web_Scraping/Confluence/confluence_scraper.py tldw_chatbook/Web_Scraping/Confluence/confluence_crawler.py` → 7 call sites at :78,150,214,376 (scraper) and :239,271,292 (crawler); reading each shows every one passes `params={...}` on the next line(s), so `set(kwargs) <= {"headers","timeout"}` is False at every production call, routing all 7 through the raw `self.session.request(method, url, **kwargs)` else-branch. `sed -n '54,72p' Tests/Web_Scraping/test_web_fetch_wiring.py` shows `test_confluence_make_request_gets_timeout_and_guard` calls `auth.make_request("GET", "/rest/api/content/123")` with no `params=` — the one shape production never uses.

## 2. P1 [D1] — `Article_Scraper/__init__.py` is empty, so `Subscriptions`' generic scraper silently disables article extraction
- Verdict: CONFIRMED
- Site now: `Web_Scraping/Article_Scraper/__init__.py` (0 bytes)  (was same)
- Proof: `wc -c tldw_chatbook/Web_Scraping/Article_Scraper/__init__.py` → `0`. Reproduced: `.venv/bin/python -c "from tldw_chatbook.Web_Scraping.Article_Scraper import Scraper"` → `ImportError: cannot import name 'Scraper'...`. `generic_scraper.py:50-57` wraps both `scrape_article` and `Scraper`/`ScraperConfig` imports in one `try/except ImportError`, so the failed second import sets `ARTICLE_EXTRACTOR_AVAILABLE = False` and nulls `scrape_article` too, even though only `scrape_article` is used (`fetch_content` at :129 gates on `ARTICLE_EXTRACTOR_AVAILABLE`).

## 3. P2 [D1/D4] — `WebSearch_APIs.py`'s provider fetches skip the size cap and cross-origin header strip
- Verdict: CONFIRMED
- Site now: `Web_Scraping/WebSearch_APIs.py:2710` (Bing), `:2945` (Brave), others per slice  (was same)
- Proof: `grep -n "^import\|^from" WebSearch_APIs.py | grep -i egress` → only `is_public_http_url` imported, no `guarded_fetch_*`. `grep -n "allow_redirects" WebSearch_APIs.py` → zero matches (default `True` everywhere). `grep -n "search_url" WebSearch_APIs.py` confirms `:2653 search_url = search_settings["bing_search_api_url"]` is config-supplied, and `search_backend_settings.py`'s `"bing"` BackendSpec declares only the API-key field, no URL-shape validation.

## 4. P2 [D3] — ~1,050 lines of `Article_Extractor_Lib.py` and the whole `Confluence/` subsystem have zero production importers
- Verdict: CONFIRMED
- Site now: same paths/lines as filed  (was same)
- Proof: `grep -rn "Article_Extractor_Lib" tldw_chatbook/ --include='*.py' | grep -v Article_Extractor_Lib.py` → only `scrape_article`/`scrape_article_sync` imported anywhere (`WebSearch_APIs.py:88`, `generic_scraper.py:51`, `local_media_reading_service.py:4276`). `grep -rln -i confluence tldw_chatbook/ --include='*.py' | grep -v Web_Scraping/Confluence` → only `config.py`. `grep -rn get_auth_headers tldw_chatbook/ Tests/` → 1 def + 2 test refs, 0 production. `sed -n '1900,1919p' Article_Extractor_Lib.py` confirms `recursive_scrape`'s bare `finally:` block ends with `return scraped_articles` after unconditionally `os.remove(resume_file)` (ruff B012). `grep -c "^def test_" WebSearch_APIs.py` → 21. `grep -n "print(" Article_Extractor_Lib.py` → 3 non-docstring hits at :833,848,1249, matching exactly.

## 5. P2 [D1] — Two open entity-expansion parsers in this slice are still unhardened
- Verdict: CONFIRMED
- Site now: `Article_Extractor_Lib.py:48/879,1064,1213`, `Article_Scraper/crawler.py:38/403`  (was same)
- Proof: `grep -n "_KNOWN_UNHARDENED" -A5 ... ; sed -n '55,86p' Tests/Subscriptions/test_watchlist_opml_entity_expansion.py` shows the 7-entry register still contains `"Web_Scraping/Article_Extractor_Lib.py"` and `"Web_Scraping/Article_Scraper/crawler.py"`. `pytest Tests/Subscriptions/test_watchlist_opml_entity_expansion.py -q` → `14 passed`, matching the review's cited number exactly.

## 6. P2 [D1] — The artifact-share index page escapes four untrusted manifest fields and interpolates the fifth raw
- Verdict: CONFIRMED
- Site now: `Web_Server/artifact_share_server.py:355` `href="/artifact/{item.key}"`  (was same)
- Proof: `sed -n '340,362p' artifact_share_server.py` — `name`, `description`, `item.kind`, `share_name` all pass through `_escape_fragment(...)`; `item.key` at the `href=` interpolates raw. `grep -n "class SharedArtifact" -A11 artifact_share_manifest.py` → `key: str` with no `Field(pattern=...)`. `load_manifest` (`:276-287`) validates only `schema_version` and non-empty `artifacts`, nothing on `key`. Key is normally `secrets.token_urlsafe(16)` (`:46`) but the server is a separate process trusting whatever manifest path it's handed, per the review's framing.

## 7. P3 [D4] — `key_protector._write_private` re-rolls `atomic_write_bytes` with stronger hardening but drops a safety
- Verdict: CONFIRMED
- Site now: `Personal_Context/key_protector.py:348-372` vs `Utils/atomic_file_ops.py:139-196`  (was same)
- Proof: `sed -n '348,372p' key_protector.py` shows `O_EXCL|O_NOFOLLOW`, mode `0o600` at `os.open()`, `secrets.token_hex(8)` temp name, and `os.write(descriptor, payload)` with the return value discarded, no parent-dir fsync. `sed -n '139,196p' atomic_file_ops.py` shows `mkstemp` + post-hoc `os.chmod(temp_path, mode)`, `f.write()` on a buffered fdopen object (raises on short write), also no parent-dir fsync — both helper and local copy lack it, matching the finding precisely.

## 8. P3 [D1] — Four naive-local `datetime.now()` timestamps survive in `Web_Scraping/`
- Verdict: CONFIRMED
- Site now: `Article_Extractor_Lib.py:846,1178,1630`; `Confluence/confluence_main.py:341`  (was same)
- Proof: `sed -n '844,848p;1176,1180p;1628,1632p' Article_Extractor_Lib.py` and `sed -n '339,343p' confluence_main.py` show all four `datetime.now().strftime(...)` calls exactly as cited; `:846` and `:1630` both set an `"ingestion_date"`-shaped value with no `Z`/`+00:00` suffix.

## 9. P3 [D2] — `re.compile` inside a nested function called per result in a fetch loop
- Verdict: CONFIRMED
- Site now: `WebSearch_APIs.py:3091`  (was same)
- Proof: `sed -n '3080,3095p' WebSearch_APIs.py` → `REGEX_STRIP_TAGS = re.compile("<.*?>")` inside `_normalize`, itself nested inside `parse_duckduckgo_results`.

## 10. P3 [D3] — `extract_domain` re-imports stdlib per call and strips `www.` anywhere in the netloc
- Verdict: CONFIRMED
- Site now: `WebSearch_APIs.py:3264-3275`  (was same)
- Proof: `sed -n '3260,3280p' WebSearch_APIs.py` → inner `from urllib.parse import urlparse` (module already imports it at top), `except (ImportError, ValueError, AttributeError)`, and `domain.replace("www.", "")` — `"newww.example.com".replace("www.", "")` yields `"neexample.com"` (substring replace, not prefix-only), exactly as claimed.

## 11. P3 [D3] — Two of the repo's largest ungoverned modules have no size-ratchet row
- Verdict: CONFIRMED
- Site now: `Personal_Context/repository.py` (4360 lines), `Web_Scraping/WebSearch_APIs.py` (4249 lines)  (was same)
- Proof: `wc -l` on both files → 4360 and 4249, matching exactly. `grep -n "_BUDGETS" -A8 Tests/Architecture/test_module_size_ratchet.py` → 7-row dict, lowest entry 6760 (`mcp_workbench.py`); neither file appears in it.

## 12. P3 [D3] — `ConfluenceAuth` logs the authenticating user's email address at INFO on every configure
- Verdict: CONFIRMED
- Site now: `Confluence/confluence_auth.py:88,103`  (was same)
- Proof: `sed -n '80,105p' confluence_auth.py` → `logger.info(f"Configured API token authentication for user: {username}")` at :88 and the basic-auth twin at :103. `grep -rln log_sanitizer tldw_chatbook/Personal_Context tldw_chatbook/Web_Scraping tldw_chatbook/WebClipper tldw_chatbook/Web_Server` → zero hits.
- Note: repo-wide `log_sanitizer` importer count is 21 today, not the slice's cited 19 — a minor mechanical drift elsewhere in the repo, not a refutation of the zero-hits-in-this-slice claim.

TOTALS: confirmed=12 fixed=0 wrong=0 demoted=0 promoted=0
