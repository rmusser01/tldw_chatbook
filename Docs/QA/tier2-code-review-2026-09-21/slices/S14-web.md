# S14 — `Personal_Context/` + `Web_Scraping/` + `WebClipper/` + `Web_Server/`

**Coverage:** files read in full: 9 | sampled: 16 | mechanical only: 25 (of 50).
Full: `Personal_Context/{crypto,key_protector}.py`, `Web_Server/{artifact_share,artifact_share_manifest,
artifact_share_server}.py`, `Web_Scraping/search_backend_settings.py`,
`WebClipper/{server_web_clipper_service,__init__}.py`, `Web_Scraping/Confluence/confluence_auth.py`.
Mechanical only: 25 — ruff `E,F,B,S,ASYNC,RUF` sweep + symbol/import/reachability greps, no line-by-line read.

## Findings

### P1 [D1] — Every production Confluence API call bypasses the module's own egress guard; the pinning test exercises the one call shape production never uses
- Where: `Web_Scraping/Confluence/confluence_auth.py:288-303` — the `set(kwargs) <= {"headers","timeout"}` branch
  predicate. Call sites: `confluence_scraper.py:78,150,214,376`, `confluence_crawler.py:239,271,292` — **all seven
  pass `params=`**.
- Evidence:
  ```
  no params (test-only shape):        guarded_fetch_requests=True   raw session.request=False
  with params (every prod call site): guarded_fetch_requests=False  raw session.request=True
  ```
  `grep -rn "make_request"` → 7 production call sites, 7 with `params=`, 0 without.
  `Tests/Web_Scraping/test_web_fetch_wiring.py:71` calls `auth.make_request("GET", "/rest/api/content/123")` — no
  `params` — **i.e. the guard test is green on a shape no caller uses.**
- Why it matters: the `else` branch does a one-shot `check_url_or_raise(url)` then
  `self.session.request(method, url, **kwargs)` with `allow_redirects` defaulted True. Three protections are lost:
  (a) **per-hop redirect re-validation** — `evaluate_url_policy`'s `_post_resolution` blocks cloud-metadata IPs
  *before* the `trusted_origins` check, so `guarded_fetch_requests` refuses `169.254.169.254` even for a trusted
  Confluence host, while the raw path follows the redirect and hands the body back; (b) the `MAX_FETCH_BYTES_PAGE`
  cap — `response.json()` buffers unbounded; (c) `strip_cross_origin_request_headers`. Verified
  `requests.SessionRedirectMixin.rebuild_auth` strips only `Authorization`.
- **Not P0 because** no production entry point reaches this subsystem today (see the dead-code finding), but the
  defect and its false-green test are shipped and will stay silent the day someone wires the UI.
- Recommended correction: delete the branch predicate; route every method through the guard (extend
  `guarded_fetch_requests` to accept `params`/`method`/`json`, or add `guarded_send_requests` in `Utils/egress.py`
  taking a prepared request — the per-hop loop at `egress.py:1065-1110` is already method-agnostic). Then change the
  test at `:71` to pass `params=`.
- Size: M · Confidence: **verified**
- Pinning test: `test_confluence_make_request_gets_timeout_and_guard` — asserts the guard IS used, but only on the
  unused shape. **False green.**

### P1 [D1] — `Article_Scraper/__init__.py` is empty, so `Subscriptions`' generic scraper silently disables article extraction on every clip
- Where: `Web_Scraping/Article_Scraper/__init__.py` (**0 bytes**). Consumer:
  `Subscriptions/scrapers/generic_scraper.py:50-57`.
- Evidence:
  ```
  IMPORT FAILS: cannot import name 'Scraper' from 'tldw_chatbook.Web_Scraping.Article_Scraper'
  ARTICLE_EXTRACTOR_AVAILABLE = False
  scrape_article = None
  ```
  Reachability: `Subscriptions/scrapers/__init__.py:18` registers `GenericWebScrapingPipeline` as the **default**
  subscription/watchlist scraper.
- Why it matters: `generic_scraper.py:50` does `from ...Article_Scraper import Scraper, ScraperConfig` inside **one**
  `try/except ImportError` that also wraps the *working* `scrape_article` import. The failed second import poisons
  the whole block: `scrape_article` becomes `None` and `ARTICLE_EXTRACTOR_AVAILABLE=False`, so `fetch_content`
  permanently takes the raw-httpx + BeautifulSoup fallback and **never uses trafilatura/readability extraction**.
  Every generic watchlist clip has been getting degraded content; nothing logs it. `Scraper`/`ScraperConfig` are
  never actually used in that module — only `scrape_article` is.
- Recommended correction: export the names in `Article_Scraper/__init__.py` (preferred —
  `Confluence/confluence_scraper.py:25-27` already reaches past the empty `__init__` with submodule paths, the same
  symptom), or drop the unused second import. Separately: split the two imports so one failing cannot null the other.
- Size: S · Confidence: **verified**
- Pinning test: none. Existing generic-scraper tests mock `fetch_content` and never assert which branch ran.
- Already covered: none. **This is the exact D3 shape the brief names** ("function-body imports of modules that no
  longer exist; mocked tests never catch these"), here at module scope.

### P2 [D1/D4] — `WebSearch_APIs.py`'s eight provider fetches are the only outbound HTTP in `Web_Scraping/` that skip the size cap and cross-origin header strip; several carry a non-`Authorization` API key across redirects
- Where: `WebSearch_APIs.py:2710` (Bing, `Ocp-Apim-Subscription-Key`, **config-supplied** `search_url`), `:2945`
  (Brave, `X-Subscription-Token`), `:3103` (DuckDuckGo POST), `:3438` (Kagi), `:3644` (Google), `:3790` (SearX),
  `:3916` (Serper), `:3987` (Exa), `:4081`, `:4183`.
- Evidence: `grep -rn "guarded_fetch\|check_url_or_raise" Web_Scraping/` → `Article_Extractor_Lib.py`,
  `Article_Scraper/{crawler,scraper}.py`, `Confluence/{confluence_auth,confluence_scraper}.py` all route through
  `guarded_fetch_*` with `MAX_FETCH_BYTES_*`; `WebSearch_APIs.py` imports only `is_public_http_url` and uses bare
  `requests` for every provider call. `inspect.getsource(requests.sessions.SessionRedirectMixin.rebuild_auth)` →
  **strips `Authorization` only**; `X-Subscription-Token` / `X-API-KEY` / `Ocp-Apim-Subscription-Key` / `x-api-key`
  are forwarded verbatim to the redirect target.
- Why it matters: (a) unbounded buffering — `response.json()`/`.content`/`document_fromstring(response.content)`
  with no cap, while the same package defines and uses `MAX_FETCH_BYTES_PAGE = 10MB` everywhere else;
  (b) credential forwarding on a cross-origin redirect — for the nine hard-coded vendor endpoints this needs a
  compromised vendor, but **`bing_search_api_url` is config-supplied and not validated**
  (`search_backend_settings.BACKENDS["bing"]` declares only the API-key field; `searx_url_issue` has no Bing
  counterpart), which is precisely the class `Tests/Utils/test_egress_adoption_census.py` excuses itself from
  ("fixed, non-user-supplied API endpoints").
- Recommended correction: minimum correct fix is `allow_redirects=False` on every credentialed provider call (a
  search API has no legitimate redirect) plus a byte cap. `guarded_fetch_requests` is GET-only, so the POSTs need
  the `guarded_send_requests` seam proposed above or a local `allow_redirects=False`.
- Size: M · Confidence: verified
- Already covered: none. Adjacent to TASK-32806.6 but that task is scoped to disk, not the wire.

### P2 [D3] — ~1,050 lines of `Article_Extractor_Lib.py` and the whole 2,100-line `Confluence/` subsystem have zero production importers, including the two open XXE register entries and a credential-materializing helper
- Where: `Article_Extractor_Lib.py` — `recursive_scrape` (1765), `sync_recursive_scrape` (1747),
  `scrape_from_sitemap` (1029), `collect_internal_links` (1086), `scrape_by_url_level`,
  `generate_temp_sitemap_from_links` (1239), `scrape_and_convert_with_filter`, `parse_csv_urls` (1428),
  `scrape_and_no_summarize_then_ingest` (824). `Web_Scraping/Confluence/*` — 6 modules, 2,100 lines, plus
  `confluence_auth.get_auth_headers` (246) which materializes `Authorization: Basic <b64 user:token>`.
- Evidence: `grep -rn "Article_Extractor_Lib" tldw_chatbook/` → only `scrape_article` (`WebSearch_APIs.py:88`,
  `generic_scraper.py:51`) and `scrape_article_sync` (`Media/local_media_reading_service.py:4276`) are imported
  anywhere in production. `Web_Scraping/__init__.py` is 0 bytes, so no star re-export.
  `grep -rn -il confluence tldw_chatbook/ | grep -v Web_Scraping/Confluence` → **`config.py` only** (a full
  `[Confluence]` credential section with no consumer). `grep -rn "get_auth_headers"` → 1 def, 2 test refs, 0
  production.
- Why it matters: three things ride on the dead code. (1) **Both of this slice's `_KNOWN_UNHARDENED` entries live
  here**, so the register's threat statements describe paths nothing calls. (2) `recursive_scrape:1922` is a
  `return` inside `finally` (ruff `B012`) that swallows every exception **including `CancelledError`**, *and* its
  `finally` unconditionally `os.remove(resume_file)` — destroying the resume state the function exists to
  maintain — while `save_progress` does blocking `open()`/`json.dump` on the event loop (`ASYNC230`) to a relative
  default path `"scrape_progress.json"` in the process CWD. (3) **21 shipped `test_*` functions inside
  `WebSearch_APIs.py` (2252-3804)**, one of which (`test_search_searx`, 3804) issues a real request to
  `https://searx.be`, plus three bare `print()` calls in `Article_Extractor_Lib.py:833,848,1249`.
- Recommended correction: delete the unreferenced functions and the `Confluence/` package (or park it out of the
  import graph), then drop the two `_KNOWN_UNHARDENED` entries — `test_known_unhardened_entries_are_still_unhardened`
  will demand it. If `Confluence/` is kept, fix the P1 first. Move the 21 `test_*` functions to `Tests/`.
- Size: M · Confidence: verified
- Already covered: **`Web_Scraping/` is in none of TASK-32807's five sub-tasks — a new cluster, and the largest found.**

### P2 [D1] — Two open entity-expansion parsers in this slice are still unhardened, against a core dependency that fixes them in one line each
- Where: `Article_Extractor_Lib.py:48` (`import xml.etree.ElementTree as xET`), parses at `:879, 1064, 1213`;
  `Article_Scraper/crawler.py:38`, parse at `:403`.
- Evidence: `pytest Tests/Subscriptions/test_watchlist_opml_entity_expansion.py -q` → **14 passed** — the register
  still lists both and `test_known_unhardened_entries_are_still_unhardened` confirms both still parse unhardened.
  Ruff agrees: `S314` ×3, `S318` ×2.
- Why it matters: the register's own note is right that `MAX_FETCH_BYTES_SITEMAP` (50 MB) does not bound expansion.
  **`crawler.py:403` is the one that will be live the moment the P1 import is fixed.**
- Recommended correction: `import defusedxml.ElementTree as xET` (core dependency;
  `Subscriptions/watchlist_opml_service.py` is the shape). `EntitiesForbidden` subclasses `ValueError` and both
  modules already catch `ParseError`. Delete the two register entries in the same commit. · Size: S · Confidence: verified

### P2 [D1] — The artifact-share index page escapes four untrusted manifest fields and interpolates the fifth raw
- Where: `Web_Server/artifact_share_server.py:355` — `href="/artifact/{item.key}"`. Siblings at
  `:344,345,354,361` all go through `_escape_fragment`.
- Evidence: `SharedArtifact.key` (`artifact_share_manifest.py:105`) is `key: str` with **no** `Field` pattern;
  `load_manifest` (`:263`) validates only `schema_version` and non-emptiness. The value is
  `secrets.token_urlsafe(16)` when *this app* stages the share, **but the server is a separate process**
  (`python -m …artifact_share_server <manifest>`) that trusts whatever manifest path it is handed.
- Why it matters: the whole `_render_index` design is "escape every field" — `_escape_fragment`'s docstring states
  that as policy. One field escaping the policy is the bug. `_BASE_HEADERS` sets
  `Content-Security-Policy: default-src 'none'; style-src 'unsafe-inline'`, which blocks external and inline
  `<script>` but not every vector — and CSP is defence-in-depth, not the escape.
- Recommended correction: `_escape_fragment(item.key)` at `:355`, and add `pattern=r"^[A-Za-z0-9_-]{1,64}$"` to
  `SharedArtifact.key` so `load_manifest` fails closed on a hand-edited manifest. · Size: S · Confidence: **inferred**

### P3 [D4] — `key_protector._write_private` re-rolls `Utils/atomic_file_ops.atomic_write_bytes` with *stronger* hardening, and drops the one safety the helper has
- Where: `Personal_Context/key_protector.py:348-372` vs `Utils/atomic_file_ops.py:139-196`.
- Evidence: the local copy adds `O_EXCL | O_NOFOLLOW`, `0o600` **at `open()`** (not a post-hoc `chmod`), and a
  `secrets.token_hex(8)` temp name — all absent from the shared helper, which `mkstemp`s then `chmod`s. Both
  `os.fsync` the file descriptor; **neither fsyncs the parent directory** after `os.replace`. The local copy uses
  raw `os.write(descriptor, payload)` and **discards the return value**; the helper's `f.write()` on a buffered
  object raises on a short write.
- Why it matters: **the divergence runs the wrong way — the stricter primitive is the one-off, the widely-adopted
  helper is the weaker one.** The short-write window is narrow (~250-byte payload) but the failure mode is total: a
  truncated bundle replaces a good one, `_deserialize_material` rejects it (`len(payload) != 65`), and **every
  encrypted Personal Context object becomes permanently unreadable.**
- Recommended correction: lift `O_EXCL|O_NOFOLLOW` + mode-at-open into `atomic_write_bytes` (gated on a
  `private=True` kwarg), then have `_write_private` call it.
- Size: S · Confidence: verified (read both implementations)
- Already covered: **TASK-32808.5 is Done — this is a surviving non-adopter *and* evidence the adopted helper is the
  weaker of the two.** *(Third independent confirmation of the `atomic_file_ops` weakness; see S25 and the
  lead's `phase4-verification.md`.)*

### P3 [D1] — Four naive-local `datetime.now()` timestamps survive in `Web_Scraping/`, one of them writing to storage
- Where: `Article_Extractor_Lib.py:846,1178,1630`; `Confluence/confluence_main.py:341`.
  (`WebSearch_APIs.py:1870` and `artifact_share_server.py:415` use `time.strftime`/`time.gmtime` — the latter is
  correct UTC.)
- Why it matters: `:1630` sets `"ingestion_date"` on an article dict destined for media storage; `:846` does the
  same. A local-time `"%Y-%m-%d %H:%M:%S"` sorts wrong against the canonical `…T…Z` shape (`' '` 0x20 < `'T'` 0x54).
  Mitigated in practice only because all four sites are in the dead code above. · Size: S · Confidence: verified
- Already covered: TASK-32803.5 is **Done** — four non-adopters survived the pass.

### P3 [D2] — `re.compile` inside a nested function called per result in a 5-iteration fetch loop
- `WebSearch_APIs.py:3091` — `REGEX_STRIP_TAGS = re.compile("<.*?>")` inside `_normalize`, inside
  `parse_duckduckgo_results`' page loop. `re` caches internally, so the real cost is the cache lookup +
  `re.compile` call overhead per invocation. Module constant. · Size: S

### P3 [D3] — `extract_domain` re-imports a stdlib module per call, catches `ImportError` for it, and strips `www.` anywhere in the netloc
- `WebSearch_APIs.py:3264-3275` — `from urllib.parse import urlparse` inside the body while it is already imported
  at module scope; `except (ImportError, ValueError, AttributeError)` with a comment conceding "ImportError if
  urllib.parse not available (very unlikely)"; **`domain.replace("www.", "")` → `"newww.example.com"` becomes
  `"neexample.com"`**. Ruff also flags `F841`. Fix: drop the inner import and the `ImportError` arm;
  `removeprefix("www.")`. · Size: S · Confidence: verified

### P3 [D3] — Two of the repo's largest ungoverned modules have no size-ratchet row
- `Personal_Context/repository.py` (4,360) and `Web_Scraping/WebSearch_APIs.py` (4,249) vs
  `Tests/Architecture/test_module_size_ratchet.py:_BUDGETS` (7 rows, lowest 6,760). `WebSearch_APIs.py` carries ≥5
  responsibilities (9 provider clients, LLM sub-question analysis, the deep-search orchestration pipeline, config
  resolution, and a 21-function embedded test harness); `repository.py` carries schema init, envelope crypto key
  custody, outbox, proposals, undo, quarantine, first-link reconciliation, and rebaselining. · Size: S
- Already covered: TASK-32809.2 (In Progress) — concrete additions. *(Third slice to find a missed god module.)*

### P3 [D3] — `ConfluenceAuth` logs the authenticating user's email address at INFO on every configure
- `Confluence/confluence_auth.py:88` and `:103`. Reproduced: `INFO … Configured API token authentication for user:
  u@e.com`. No module in this slice imports `Utils/log_sanitizer.py` (0 hits; 19 repo-wide). PII, not a credential;
  currently unreachable. Flag on TASK-32806.7 rather than filing separately. · Size: S

## Candidate triage
**CONFIRMED:** `os_replace_no_atomic` `key_protector.py:364` (reframed — atomicity is fine and *stronger* than the
shared helper; the gaps are the unchecked `os.write` return and the absent parent-dir fsync **which the shared
helper also lacks**); `Article_Extractor_Lib.py:1808` (`save_progress` writes unfsynced JSON to a relative CWD path,
on the event loop, inside a `finally` that then deletes the file — dead code); `re_compile_in_def`
`WebSearch_APIs.py:3091`; `function_body_import` `WebSearch_APIs.py:3264` and `confluence_auth.py:133` (ruff `F811`,
shadows the module-level import); `strftime` 4 of 6.
**RETIRED:** `except_exception_pass` `interview_coordinator.py:405,417` (documented best-effort markers after records
are durable), `repository.py:650,1412` (rollback-inside-rollback; key cleanup on a re-raising `BaseException` path).
`except_exception_return` `serve.py:907` (`_canvas_enabled` fails closed, comment says so),
`search_backend_settings.py:277` (deliberate UI-boundary suppression with `noqa: BLE001` and the reason).
`fetchall_no_limit` 33 rows (31 in `repository.py`) — `_iter_head_rows` is a bounded keyset pager
(`_COLLECTION_PAGE_SIZE`) and the rest are singleton/PRAGMA/`WHERE id = ?`; `cookie_cloner.py` ×3 — bounded by a
host-scoped `LIKE` over the user's own local cookie DB read from a private 0600 clone.
`function_body_import` aiohttp ×25 (deliberate optional-dep gating; `_require_aiohttp` documents the one-gate
design), keyring ×3, and the `Personal_Context` ×11 (documented cycle-breakers).
`mutable_class_attr` `artifact_share_manifest.py:118` — `model_config` is the pydantic v2 contract.
`tempfile_no_secure` ×3 — `_private_cookie_clone` resolves the temp root, calls `secure_private_directory`, opens
`O_EXCL|O_NOFOLLOW` at `0o600`, `fchmod`s and `fsync`s; `artifact_share.py:156` is an anonymous `TemporaryFile()`
for child stdout (the comment explains why a file beats a pipe); `artifact_share_manifest.py:238` is
`mkstemp(dir=share_dir)` at 0600 then `os.replace`, with a comment saying that is exactly to avoid a 0644 window.
`raw_mkdir` `artifact_share_manifest.py:169` — followed by an explicit `os.chmod(root, 0o700)`.
`legacy_markers` ×17 — accurate provider-retirement notices (Bing retired 2025-08-11, Google CSE closed to new
customers, Kagi v0 deprecated) plus live `LEGACY_FIELD_ALIASES` config-migration code. `try_import_guard` ×18 —
optional-dep guards with user-facing install hints. The scope-scaffold DUP clusters — **no call site in this slice
passes a sync argument that does blocking I/O on the loop**; `WebClipper` members are pure protocol forwarding.
DUP `_default_clock`/`_now` ×2 — same two-line body; consolidating into `Utils/timestamps.utc_now` is the right move
but it is the P3 timestamp finding, not a separate one. DUP `replace`/`delete` (`key_protector` ↔
`link_key_custody`) — `keyring` wrappers over *different* key namespaces; same shape, correctly separate.
DUP `pid_alive` ↔ `_posix_group_exists`, `handle_textual_js` ↔ `handle_index` — shape coincidences.
**UNVERIFIED:** `except_exception_pass` `cookie_cloner.py:397,658`; `except_exception_return`
`Article_Extractor_Lib.py:2085`; DUP `_json_wire@serve.py:140` (4-member cluster, overlaps task-32862).

## D4 observations for repo-wide Phase 3
1. **`_as_dict` — no helper exists, two copies drifted, and the drift changes the failure mode.** Scope-service
   copies (`WebClipper/server_web_clipper_scope_service.py:53` + 4 siblings, hash `471e51c38f2d`) end with
   `return dict(value)`; service copies (`server_web_clipper_service.py:48` + 2 siblings, hash `2a9381e5cafb`) end
   with `raise TypeError("Expected a mapping or Pydantic model payload.")`. **Same name, same first three branches,
   opposite contract on the fallthrough.** These results feed server API payloads and UI rendering, so the drift is
   a D1 in its own right. Home: `runtime_policy/` alongside the scope-service scaffold — a member of TASK-32808.6.
2. **`get_content_hash` / `content_changed` — byte-identical, two copies, one package.**
   `Article_Extractor_Lib.py:1708,1722` and `Article_Scraper/utils.py:101,115`. Zero drift. Home:
   `Article_Scraper/utils.py` (`ContentMetadataHandler` is already the production-imported surface).
   **Caveat: the 17-reference count came from a bare-name grep and will include unrelated symbols — resolve first.**
3. **The atomic-private-write primitive is duplicated in the wrong direction.** `key_protector.py:348`,
   `Utils/private_paths.atomic_private_write_bytes`, and `Utils/atomic_file_ops.atomic_write_bytes` are three
   implementations; **the two local ones are stronger than the shared one.** TASK-32808.5 is Done, so this is the
   post-adoption state: **the helper everyone adopted is the weakest of the three.** Recommend re-opening `.5`'s
   scope as "lift the hardening into the shared helper", not "adopt the shared helper".
4. **`Utils/log_sanitizer.py` has zero importers across all four packages in this slice** (19 repo-wide). Given
   `confluence_auth.py` logs an email at INFO and `WebSearch_APIs.py` handles five API-key families, this is a
   coverage gap worth a census row rather than per-site findings.
5. **`Tests/Architecture/test_module_size_ratchet.py::_BUDGETS` is a 7-row hand-picked list with no glob.** A
   measured "top N by line count, minus already-ratcheted" sweep would beat hand-picking again.

## Left UNVERIFIED
| Claim | Why | Command |
|---|---|---|
| The artifact-share `item.key` XSS is exploitable end to end | read-only; the CSP (`default-src 'none'`) may block the payload class in practice | render `_render_index()` with a crafted manifest whose `key` is `" onerror="alert(1)` and grep the output for the raw `onerror=` |
| `PassphraseProfileKeyProtector._read_private` refusing on `PrivatePathStatus.HARDENED_PRIVATE` (`key_protector.py:285`) is deliberate, not inverted | `HARDENED_PRIVATE` is in `verified_private`, and the only two other consumers treat it as a benign INFO. Refusing key material that was just tightened is defensible and self-heals on the next read — but there is no comment saying so and no test naming it | `git log -L 280,290:tldw_chatbook/Personal_Context/key_protector.py --oneline \| head -40` and `grep -rn "not private" Tests/Personal_Context/` |
| The `_json_wire`/`_canonical_json_value` 4-member cluster has no datetime/Decimal drift | mechanical row only; overlaps open task-32862 | diff the four bodies at `MCP/permission_store.py:152`, `Canvas/gateway.py:2898`, `Subscriptions/briefing_voices.py:131`, `Web_Server/serve.py:140` |
| `cookie_cloner.py:397,658` `except: pass` | mechanical only | `sed -n '385,402p;648,665p' tldw_chatbook/Web_Scraping/cookie_scraping/cookie_cloner.py` |
| `Personal_Context/{service,link_service,reconciliation,proposal_service}.py` contain no further D1 | mechanical only — ruff `E,F,B,S,ASYNC` clean beyond the triaged rows, no dotted `get_cli_setting`, no `run_worker`, no dual-logger, no raw `sqlite3.connect`, no bare `requests`/`httpx`. **That is a negative scan, not a read.** | `ruff check tldw_chatbook/Personal_Context/{service,link_service,reconciliation,proposal_service}.py --select ALL --ignore D,ANN,COM,E501,TD,FIX,ERA --output-format concise` |
| **Cross-slice:** `Subscriptions/scrapers/generic_scraper.py:47,133` calls `origin_set(url)` on its own input, which `Utils/egress.py`'s docstring forbids | out of slice | `grep -rn "origin_set(" tldw_chatbook/ \| grep -v Utils/egress.py` — *(lead: this is the same defect class S08 filed as its P1 sitemap finding, at a second site)* |
