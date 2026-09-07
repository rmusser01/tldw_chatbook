# PR #1393 review fixes — RED/GREEN evidence

Branch: `fix/web-tools-polish-2620`
Code: `tldw_chatbook/Tools/web_tool_impls.py`
Tests: `Tests/Tools/test_web_crawl.py`, `Tests/Tools/test_web_tool_impls.py`
Residuals task: `backlog/tasks/task-2910 - Web-tools-residual-hardening.md`

Baseline before any change: `100 passed` (`pytest Tests/Tools/test_web_crawl.py
Tests/Tools/test_web_tool_impls.py -v -p no:randomly`).

## Fix 1 — Named constant for the title cap

No behavior change (refactor only), so no RED/GREEN cycle — the constant's
value (512) is unchanged, only its representation.

- Added `CRAWL_TITLE_MAX_CHARS = 512` beside the other `CRAWL_*` constants
  in `web_tool_impls.py`.
- `_CrawlLinkParser.handle_data` now slices with `[:CRAWL_TITLE_MAX_CHARS]`
  instead of the bare literal `512`.
- `Tests/Tools/test_web_crawl.py::test_title_accumulation_bounded` now
  imports `CRAWL_TITLE_MAX_CHARS` and asserts
  `len(parser.title) <= CRAWL_TITLE_MAX_CHARS` instead of a hardcoded `512`.

GREEN: `test_title_accumulation_bounded` passes in the final full run below.

## Fix 2 — `web_fetch` Args/Returns docstring sections

Documentation-only change; no test required per the brief. Verified by hand
that every existing sentence of the docstring (the SSRF/redirect/rate-limit/
cache paragraph, the HTML/plain-text extraction paragraph, the PDF-detection
paragraph, the structured-reasons bullet list, and the `Raises:` section) is
byte-identical to before — only an `Args:` block (right after the summary
line) and a `Returns:` block (right before `Raises:`) were inserted.

## Fix 3 — `_pymupdf_available` total + single-probe per call

### 3a — total (guard the ValueError)

RED: added `test_pymupdf_available_spec_less_stub_returns_false_not_valueerror`
— monkeypatches `sys.modules["pymupdf"] = SimpleNamespace(__spec__=None)` and
calls `web_fetch` on a PDF route, asserting `LocalToolError` with
`missing-dep`. Before the fix, this failed with an uncaught exception:

```
tldw_chatbook/Tools/web_tool_impls.py:314: in _pymupdf_available
    return importlib.util.find_spec("pymupdf") is not None
ValueError: pymupdf.__spec__ is None
<frozen importlib.util>:111: ValueError
```

(confirmed the underlying `find_spec` behavior with a standalone repro
script before writing the test.)

GREEN after wrapping the call in `try: ... except (ImportError, ValueError):
return False`: test passes (`1 passed`).

### 3b — single probe per call, reused for both decision points

`web_fetch` now computes `pymupdf_ok = _pymupdf_available()` once, before
the redirect loop, and both the `pdf_max_bytes=PDF_MAX_BYTES if pymupdf_ok
else None` selection inside the loop and the post-fetch `[missing-dep]`-vs-
`[too-large]` branch (`if not pymupdf_ok:`) read that one value — `grep -n
"_pymupdf_available()"` shows exactly one call site left in the module
(besides the `def`). No dedicated RED test (behavior-preserving under every
existing test with a stable environment); the two pre-existing PDF tests
that monkeypatch `_pymupdf_available` itself
(`test_fetch_pdf_too_large_message_reflects_configured_ceiling`,
`test_fetch_pdf_missing_dep_skips_20mb_download`) still pass, confirming the
single cached value is read correctly from both branches.

## Fix 4 — `budget_truncated` no-false-positive reorder

RED: added two tests to `Tests/Tools/test_web_crawl.py`:
- `test_sitemap_trailing_offhost_loc_does_not_flip_budget_truncated` (3
  same-host locs + 1 trailing off-host loc, `max_pages=3`)
- `test_sitemap_trailing_duplicate_loc_does_not_flip_budget_truncated` (3
  same-host locs + 1 trailing duplicate-of-first, `max_pages=3`)

Both asserted `out.endswith("Stopped: sitemap exhausted.")`. Pre-fix, both
failed:

```
AssertionError: assert False
 +  where False = '...Stopped: page budget reached.'.endswith('Stopped: sitemap exhausted.')
```

GREEN after reordering `take()` (host filter -> dedup filter -> budget
check, was budget-check-first) and the child-sitemap loop's break (deadline
-> off-host filter -> budget check, was budget-check-before-host-filter):
both new tests pass, and the three shape-preservation cases still pass
unchanged:
- `test_sitemap_budget_truncated_reports_page_budget_reached` (true
  positive: 10 same-host URLs, `max_pages=3` -> "page budget reached")
- `test_sitemap_exactly_consumed_still_reports_exhausted` (exact
  consumption: 3 URLs, `max_pages=3` -> "sitemap exhausted")
- `test_sitemap_index_caps_children_fetched` /
  `test_sitemap_child_budget_reached_stop_reason` (child-loop cap behavior
  unaffected by moving the off-host check ahead of the budget check)

## Fix 5 — task-2910 updated

Struck through and annotated items 1, 2, and 6 as "RESOLVED on PR #1393"
with the concrete mechanism and the RED test name(s) that cover each. Items
3, 4, 5, 7, 8 left untouched (out of scope for this PR). Task status left
`To Do` — five residual items remain open.

## Final full run

```
$ .venv/bin/python -m pytest Tests/Tools/test_web_crawl.py Tests/Tools/test_web_tool_impls.py -v -p no:randomly
...
======================= 103 passed, 5 warnings in 1.22s ========================
```

100 pre-existing + 3 new (1 pymupdf-stub test, 2 sitemap-trailing-candidate
tests) = 103. No regressions, no skips beyond the pre-existing
pymupdf/defusedxml-conditional skips (venv has both installed, so none
triggered in this run).
