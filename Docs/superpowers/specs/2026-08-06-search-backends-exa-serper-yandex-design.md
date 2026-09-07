# Search backends: Exa, Serper, Yandex — design (task-1355)

- Date: 2026-08-06
- Backlog: task-1355 (Complete Exa, Serper, and Yandex search engine backends)
- Context: `search_web_serper`/`search_web_yandex` are `pass` stubs in `Web_Scraping/WebSearch_APIs.py` — worse than dead options, selecting them today feeds `None` into `process_web_search_results` and fails confusingly downstream (the dispatcher even calls them with zero arguments). Exa is absent entirely. This spec makes engine choice real on every surface that offers it.

## Owner rulings (brainstorm 2026-08-06)

1. **Live verification with real keys.** The owner supplies keys as repo-root files — `exa-api-key.txt`, `serper-api-key.txt`, `yandex-api-key.txt` + `yandex-folder-id.txt` — whichever subset exists; live tests light up per-engine and skip cleanly otherwise.
2. **All three engines join the agent tool's `SEARCH_ENGINES` enum** in `Tools/web_tool_impls.py`, so the model can select them from Console/MCP.

## §1 Security prerequisite: tracked ignore rules for credential files

The tracked `.gitignore` names only `/openai-api-key.txt` and `/moonshot-api-key.txt`; the catch-all `*api-key*.txt` lives in `.git/info/exclude` — local to this machine, absent on every other clone. `yandex-folder-id.txt` matches no pattern anywhere. **First commit of the branch:** add `*-api-key.txt` and `/yandex-folder-id.txt` to the tracked `.gitignore` (keeping the two existing specific lines; the glob subsumes them but they stay for history clarity). This lands before any instruction that invites key files into the tree.

Handling rule for every implementer working this plan: **never print key-file contents** — not even head/tail fragments — into logs, test output, or reports (a prior programme leaked key bytes into a transcript this way; the key had to be rotated).

## §2 The three backends (`Web_Scraping/WebSearch_APIs.py`)

House pattern throughout: `requests`, keys via `loaded_config_data["search_engines"][...]`, missing key → `ValueError` with a clear message (brave's pattern), HTTP failure → `raise_for_status()` propagating into `perform_websearch`'s existing error envelope and the agent tool's `[search-failed]` string. These are fixed HTTPS API endpoints — the SSRF egress guard is for attacker-controlled URLs and deliberately does not apply, same as every sibling backend.

**Serper** (`search_web_serper(search_query, content_country, search_lang, result_count)`):
- `POST https://google.serper.dev/search`, headers `X-API-KEY: <serper_search_api_key>`, `Content-Type: application/json`.
- Body: `{"q": query, "gl": country.lower() (default "us"), "hl": search_lang (default "en"), "num": result_count}`.
- `parse_serper_results`: standardized items from `organic[]` — `title`, `url` ← `link`, `content` ← `snippet`; `metadata.snippet` ← `snippet`, `metadata.date_published` ← `date`, `metadata.position` ← `position`; `metadata.relevance_score` stays `None` (brave parity — mapping position into a "relevance" field would invert its meaning for any consumer sorting descending). `answerBox`/`knowledgeGraph` blocks are deliberately ignored (non-goal; organic web results only, like every sibling parser).

**Exa** (`search_web_exa(search_query, result_count)`):
- `POST https://api.exa.ai/search`, headers `x-api-key: <exa_search_api_key>`, `Content-Type: application/json`.
- Body: `{"query": query, "numResults": result_count, "type": "auto", "contents": {"highlights": true}}`.
- **Cost note (deliberate):** highlights are billed as contents retrieval on top of the search call. A result without any snippet text is nearly useless to the model, so highlights are on by default; this is recorded here as a paid trade, not an accident.
- `parse_exa_results`: `title`, `url`, `content` ← first `highlights` entry (else `""`); `metadata.snippet` ← same, `metadata.date_published` ← `publishedDate`, `metadata.author` ← `author`.

**Yandex** (`search_web_yandex(search_query, result_count)`):
- Cloud Search API v2, **synchronous** REST endpoint — verified at the proto level (`yandex/cloud/searchapi/v2/search_service.proto`: `WebSearchService.Search` → `google.api.http post: "/v2/web/search"`, returns `WebSearchResponse{raw_data}` directly; no Operation polling). The docs site is CAPTCHA-walled; the proto is the citable ground truth, and the live test is the final arbiter.
- `POST https://searchapi.api.cloud.yandex.net/v2/web/search`, headers `Authorization: Api-Key <yandex_search_api_key>`, `Content-Type: application/json`.
- Body: `{"query": {"searchType": "SEARCH_TYPE_COM", "queryText": query}, "folderId": <yandex_search_folder_id>, "responseFormat": "FORMAT_XML"}` — every field verified against the proto/docs; `searchType` is required, `SEARCH_TYPE_COM` (international) is the default. No `groupSpec`: its REST shape is unverified and unnecessary — the agent tool already trims client-side (`results[:count]`), and Yandex's default page (~10 groups) covers the tool's ceiling. Key or folder id missing → the same clear `ValueError` path.
- Response: JSON `{"rawData": "<base64>"}` → `base64.b64decode` → XML parsed with the repo's defusedxml-with-fallback pattern (attacker-adjacent remote content; same import shape as `web_tool_impls`/`Subscriptions/security.py`).
- **In-XML error detection (honesty requirement):** the decoded XML can carry an `<error code="…">` element inside an HTTP 200 (quota exhausted, bad auth, malformed query). The backend raises a clear error carrying the code and text — a quota error must never render as "No results found" about a query that was never searched.
- `parse_yandex_results`: walk `<group><doc>` elements — `url` ← `<url>`, `title` ← flattened `itertext()` of `<title>` (strips inline `<hlword>` highlight tags), `content` ← joined flattened `<passage>` texts (else `""`); `metadata.snippet` ← same. Malformed/undecodable payloads raise through the standard error envelope, never a bare XML exception.

**Dispatch fixes:** `perform_websearch` passes real arguments to all three (today's serper/yandex branches call the stubs bare) and gains an `exa` branch; `process_web_search_results` gains an `exa` branch calling `parse_exa_results`. Both keep the existing metrics/log calls their sibling branches emit.

## §3 Every engine surface updated (the sweep)

A new option must be taught to every surface. Known enumerations and their treatment:

| Surface | Treatment |
|---|---|
| `Tools/web_tool_impls.py` `SEARCH_ENGINES` | append `"exa", "serper", "yandex"` (owner ruling 2) — schema enum for the agent tool updates automatically |
| `Web_Scraping/WebSearch_APIs.py` dispatch | §2 |
| `Tools/web_search_tool.py` (legacy tool) | check liveness first (registration in `Agents/tool_catalog.py` / `_GATEABLE_BUILTINS`); if live, add the three to its enum; if retired, record that and leave untouched |
| `Utils/Utils.py:148` engine list | inspect its consumer; update if it feeds `perform_websearch`, record if not |
| `Research_Interop/local_research_search_service.py` + `tldw_api/research_search_schemas.py` | inspect — these look like research-API contract lists; update only if they route to `perform_websearch`, otherwise record the deliberate skip with the reason |

The plan enumerates each with its verdict; "recorded skip" means a line in the task's Implementation Notes, not silence.

## §4 Config

- Loader (`config.py`, `search_engines` section): add `serper_search_api_key` (default `""`), `exa_search_api_key` (default `""`), `yandex_search_folder_id` (default `""`). Existing `yandex_search_api_key` and legacy `yandex_search_engine_id` stay untouched (the latter belonged to the deprecated XML API; not reused, not removed).
- Config template: the in-file TOML template in `config.py` (`[search_engines]` block, ~line 3695 where `tavily_search_api_key = ""` lives) gains the same three keys with empty defaults and a one-line comment each.
- No env-var plumbing beyond what the section loader already does (parity with siblings).

## §5 Testing

**Mocked unit tests** (new `Tests/Web_Scraping/test_search_backends.py`), monkeypatching `WebSearch_APIs.requests` and `loaded_config_data`:
- Per engine: request shape pinned (exact URL, auth header name+value source, full JSON body incl. serper's lowercased `gl`, exa's `contents.highlights`, yandex's required `searchType`/`folderId`); parser output pinned against a realistic captured-shape fixture (yandex fixtures: a real-structure base64-encoded XML doc with `<hlword>` tags and a passage-less doc, plus an `<error>`-element payload asserting the honest failure path); missing-key `ValueError` copy; HTTP-error propagation into `perform_websearch`'s envelope; `process_web_search_results` end-to-end for each engine.
- The standardized-shape invariant: every parser emits `title`/`url`/`content` strings + `metadata.snippet`, matching what `web_tool_impls.web_search` renders (`snippet` or `content` fallback).

**Live tests** (same file, `@pytest.mark.live` + per-engine skipif): **double-gated** — each test requires BOTH its key file AND `TLDW_LIVE_SEARCH_TESTS=1` in the environment, because once the owner's key files land on this machine a routine gate run of the file would otherwise make paid API calls on every execution. The `live` marker is registered in `pyproject.toml` if not already (unregistered markers warn). Tests read the repo-root key files, monkeypatch them into `loaded_config_data`, one real query per configured engine, assert ≥1 standardized result with non-empty `url`. Never printed key material; failures reported honestly (a 4xx from a bad key is a finding, not a fabricated pass). Live tests are excluded from the standard gates and run once, foreground, during this programme's verification.

**Agent-tool surface:** the existing `test_web_search_spec_schema` asserts membership, not equality — extend it to assert the three new engines are present in the schema enum.

## §6 Non-goals

- answerBox/knowledgeGraph/news/images verticals (organic web results only, all engines).
- Exa contents beyond highlights (full text, summaries, subpages); Exa's deep/reasoning search types.
- Yandex regional search types beyond the `SEARCH_TYPE_COM` default, and the deprecated XML-API `yandex_search_engine_id` path.
- Retrying/backoff beyond the sibling backends' behavior; per-engine rate limiting (the agent layer's caps already bound call volume).
- UI settings surfaces for the new keys (none of the existing engine keys have one; config.toml is the path).
