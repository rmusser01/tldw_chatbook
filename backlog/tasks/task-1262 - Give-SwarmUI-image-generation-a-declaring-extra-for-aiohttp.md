---
id: TASK-1262
title: Give SwarmUI image generation a declaring extra for aiohttp
status: Done
assignee: []
created_date: '2026-07-28 19:13'
labels:
  - packaging
  - imagegen
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
aiohttp is required by SwarmUI image generation but is declared in no extra that names image generation: it ships only in [websearch] (for the article scraper) and [all-tools]. A user who wants image generation is therefore told to install the web-search extra, pulling ten unrelated packages (lxml, beautifulsoup4, pandas, playwright, trafilatura, langdetect, nltk, scikit-learn, defusedxml, tqdm) to obtain one HTTP client. This also forced a workaround in task-1261's sibling fix: the optional_deps guard cannot pass feature_name='websearch' without clobbering the DEPENDENCIES_AVAILABLE key that check_websearch_deps() owns, so the install hint had to be hand-written.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An extra exists that names image generation and provides aiohttp
- [x] #2 The install hint raised by swarmui_client._require_aiohttp() names that extra rather than [websearch]
- [x] #3 The new extra is included in all-tools
- [x] #4 Existing aiohttp consumers keep working: the article scraper via [websearch], the web server via [web] -> textual-serve
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm which aiohttp consumers actually lack a declaring extra.
2. Add an `image_generation` extra; keep aiohttp in `[websearch]` for the scraper.
3. Point `_require_aiohttp()`'s hint at the new extra.
4. Add the matching `OPTIONAL_FEATURES` metadata entry (a test enforces coverage).
5. Prove the resolution with a dry-run install into a throwaway venv.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Scope check first.** aiohttp has three consumers, not one, and only one was actually unserved:

| Consumer | Declaring extra |
|---|---|
| `Web_Scraping/Article_Scraper/crawler.py` | `[websearch]` — correct, left alone |
| `Web_Server/serve.py` | `[web]` → `textual-serve`, which requires `aiohttp>=3.9.5` and `aiohttp-jinja2>=1.6` — covered transitively |
| `Media_Creation/swarmui_client.py` | none — the gap |

So this adds `image_generation = ["aiohttp"]` rather than moving anything. aiohttp stays in `[websearch]`; `all-tools` enumerates packages rather than extras and already listed it, now with a comment noting it serves both.

**Measured effect.** Dry-run resolve into a throwaway uv venv: `[image_generation]` → 59 packages, `[websearch]` → 86. A user who wants image generation no longer installs playwright, trafilatura, pandas, nltk and scikit-learn to obtain one HTTP client.

**Metadata.** `Tests/Utils/test_optional_deps.py::test_optional_feature_metadata_covers_pyproject_extras` asserts every pyproject extra has recovery metadata, and correctly failed on the new extra. Added an `OPTIONAL_FEATURES` entry under a new `AREA_MEDIA_CREATION` — `AREA_MEDIA` is "Media ingestion and transcription", which is the wrong side of the ingest/create line. Recovery action is `Console > /generate-image`: no settings screen owns `media_creation`, so pointing at one would have been invented.

**Not done:** the guard still keys `DEPENDENCIES_AVAILABLE` on `"aiohttp"` rather than switching to `require_dependency("aiohttp", "image_generation")`, which the new extra would now make collision-free. Keying on the package keeps one accurate entry across all three consumers instead of one per feature; the reasoning is recorded in `_safe_import_aiohttp()`'s docstring.

**Modified files.** `pyproject.toml`, `tldw_chatbook/Media_Creation/swarmui_client.py`, `tldw_chatbook/Utils/optional_deps.py`, `backlog/docs/lessons-testing-evidence.md` (dated the now-superseded "[websearch] only" claim).
<!-- SECTION:NOTES:END -->
