---
id: TASK-1355
title: 'Complete Exa, Serper, and Yandex search engine backends'
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-05 06:03'
labels:
  - web-tools
dependencies:
  - TASK-1354
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
search_web_serper and search_web_yandex are empty stubs in Web_Scraping/WebSearch_APIs.py and Exa is absent entirely, so the search_engine enum offers dead options. Complete the three backends so engine choice is real.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Spec `Docs/superpowers/specs/2026-08-06-search-backends-exa-serper-yandex-design.md` + plan `Docs/superpowers/plans/2026-08-06-search-backends-1355.md`: 6 SDD tasks — gitignore security prerequisite, Serper, Exa, Yandex (base64-XML + honest in-XML errors), engine-surface sweep, double-gated live tests.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 serper+yandex implemented and wired into perform_websearch,Exa added with API call + result parsing + [SearchEngines] key,Unit tests with mocked responses + optional live tests
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Approach.** Executed as 6 SDD tasks per the spec/plan referenced above: a
gitignore security prerequisite (credential scratch files), then Serper, Exa,
and Yandex each as their own implement-then-review round, then an engine-surface
sweep, then double/triple-gated live tests.

**The three backends.**
- **Serper** (`search_web_serper` / `parse_serper_results`) — POST
  `https://google.serper.dev/search`, `X-API-KEY` header; parses `organic`
  results only (answerBox/knowledgeGraph deliberately ignored, like every
  sibling parser).
- **Exa** (`search_web_exa` / `parse_exa_results`) — POST
  `https://api.exa.ai/search`, `x-api-key` header, requests
  `contents.highlights` (a deliberate paid trade for snippet text — a
  result with no snippet is nearly useless to the model); first highlight
  becomes `content`.
- **Yandex Cloud Search API v2** (`search_web_yandex` / `parse_yandex_results`)
  — POST `https://searchapi.api.cloud.yandex.net/v2/web/search`,
  `Authorization: Api-Key ...`; response wraps base64-encoded XML in
  `rawData`. An in-XML `<error>` element (quota/auth/malformed-query
  arriving inside HTTP 200) is raised as a `ValueError` from the parser so
  it surfaces via `processing_error` instead of silently rendering "no
  results" for a query that was never actually searched.

**Dispatch fixes.** All three wired into both `perform_websearch`'s
if/elif chain and `process_web_search_results`'s parser dispatch. Engine
enums swept for every live surface (task 5): `Tools/web_tool_impls.py`
`SEARCH_ENGINES` and `Research_Interop/local_research_search_service.py`
`LOCAL_SUPPORTED_WEBSEARCH_ENGINES` updated to advertise `exa`/`serper`/`yandex`.
Three recorded skips from that sweep (verbatim from
`.superpowers/sdd/2026-08-06-search-backends-1355/task-5-report.md`):

> 2. **`Utils/Utils.py` `global_search_engines`** — `grep -rn
> "global_search_engines" .` (repo-wide, excluding `.venv`/`.git`) finds
> exactly one hit: its own definition at `Utils.py:144`. Zero consumers
> anywhere (no import, no dotted access, no test). Does not route to
> `perform_websearch` or present engine choices to any dispatcher. →
> **RECORDED SKIP: dead list, no consumer to update.** (Left untouched;
> already happens to contain `yandex` but not `exa`/`serper` — irrelevant
> since nothing reads it.)
>
> 4. **`tldw_api/research_search_schemas.py`
> `SUPPORTED_WEBSEARCH_ENGINES`** — consumed by
> `WebSearchRequest.validate_engine` (pydantic field validator) and by
> `Research_Interop/server_research_search_service.py:100-103`
> (`list_supported_websearch_engines` for the **remote** server backend).
> The *local* service (`local_research_search_service.py`) also
> constructs a `WebSearchRequest`, but only **after** its own
> `LOCAL_SUPPORTED_WEBSEARCH_ENGINES` gate already passed — this set is a
> superset/second gate, not the local dispatcher's binding list, and it
> already contains engines the local dispatcher never supports
> (`firecrawl`, `sogou`, `startpage`, `stract`, `4chan`). It already lists
> `exa`, `serper`, `yandex`, `firecrawl` (confirms the brief's "already
> lists exa and firecrawl" note). → **RECORDED SKIP: verified
> server/schema contract, already a superset, not
> local-dispatch-authoritative — no change needed.**
>
> 5. **`Tools/web_search_tool.py` (`WebSearchTool`)** — exact brief grep
> `grep -rn "web_search_tool" tldw_chatbook/ | grep -v
> "Tools/web_search_tool.py"` finds one hit: `Tools/__init__.py:61`'s lazy
> `_SUBMODULE_BY_NAME` PEP-562 mapping entry (declares availability only —
> resolves on first attribute access). Followed up with `grep -rn
> "WebSearchTool" . --include="*.py"` repo-wide: the only non-definition,
> non-test hits are that same `__init__.py` mapping/docstring/`__all__`
> entry; nothing in `Agents/`, `Agents/tool_catalog.py`'s
> `_GATEABLE_BUILTINS`, or any app wiring ever accesses
> `Tools.WebSearchTool`. All other hits are tests
> (`Tests/Tools/test_web_search_tool.py`,
> `Tests/Utils/test_optional_import_deferral.py`) exercising the
> class/lazy-import mechanism directly, not production callers. →
> **RECORDED SKIP: retired/dead file, no production consumer — not
> extended.**

**Config keys.** `[SearchEngines]` section: `serper_search_api_key`,
`exa_search_api_key`, `yandex_search_api_key`, `yandex_search_folder_id`.

**Live tests: triple gate.** `test_live_serper`/`test_live_exa`/`test_live_yandex`
require all three: (1) the pytest CLI flag `--run-live` (registered in
`Tests/conftest.py`, a session-level backstop that force-skips any test
carrying the `live` marker when the flag is absent, mirroring the existing
`--run-slow`/`--run-optional` pattern), (2) `TLDW_LIVE_SEARCH_TESTS=1` in
the environment, and (3) the relevant credential file(s) present at the
checkout root (`serper-api-key.txt`, `exa-api-key.txt`,
`yandex-api-key.txt` + `yandex-folder-id.txt`) — all covered by tracked
`.gitignore` rules so they can't be committed by accident.

**Two Critical fix rounds.**
1. **defusedxml import order.** The optional-`defusedxml` guard import
   (Yandex XML parsing) originally sat *above* the `from loguru import
   logger` import, so the `except ImportError` fallback branch called
   `logger.warning(...)` before `logger` existed — a `NameError` on any
   install without `defusedxml`. Fixed by moving the guard below the
   logger import (commit `a22a0b18b`).
2. **Module-scope `pytest` import (this final review round).** The module
   had `import pytest` at top level for a single `@pytest.mark.asyncio`
   test stub (`test_perplexity_pipeline`), which made
   `tldw_chatbook.Web_Scraping.WebSearch_APIs` unimportable on any
   production install (pytest ships only in the dev/all-tools extras).
   Fixed by deleting the import and that one decorated stub; all other
   undecorated `test_*` stubs in the file were left untouched. Verified
   by simulating pytest-absence (`sys.modules['pytest'] = None` in a
   subprocess) and confirming the module still imports cleanly. Also
   fixed in the same round: Serper's `ValueError` copy cited the wrong
   TOML section (`[search_engines]` instead of the real `[SearchEngines]`
   header — exa/yandex already had this right), added HTTP-error
   propagation tests for all three backends (429/401 →
   `requests.HTTPError`, plus one end-to-end proving
   `perform_websearch` wraps it into `processing_error` rather than
   crashing), and strengthened `test_agent_enum_engines_all_dispatchable`
   — it previously asserted a false claim ("every advertised engine
   reaches a real backend"), when `parse_tavily_results`/
   `parse_searx_results` are `pass` stubs that silently render "No
   results found" for real API responses. Filed as TASK-2990.

**Files modified this round:** `tldw_chatbook/Web_Scraping/WebSearch_APIs.py`,
`Tests/Web_Scraping/test_search_backends.py`, `pyproject.toml`, `.gitignore`.

Status intentionally left **In Progress** — the controller runs the triple
live gate (with real key files copied to the worktree root) before moving
this to Done.
<!-- SECTION:NOTES:END -->
