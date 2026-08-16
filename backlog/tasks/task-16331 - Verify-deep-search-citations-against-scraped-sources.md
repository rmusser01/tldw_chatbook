---
id: TASK-16331
title: Verify deep-search citations against scraped sources
status: Done
assignee:
  - '@robert'
created_date: '2026-08-15 05:13'
updated_date: '2026-08-15 05:40'
labels:
  - research
  - web-tools
  - citations
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
web_deep_search phase 2 (analyze_and_aggregate) prompt-enforces bracket [n] citation markers but never checks them against the actual scraped content, so fabricated or mismatched citations pass through silently and the confidence score is heuristic rather than evidential. Port the verification patterns from tldw_server dev (Claims_Extraction: verbatim-first alignment ladder, quote checking, verification summary) into the chatbook deep-search pipeline so citations in the final answer are validated against the in-memory evidence before the answer is returned.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every bracket citation marker in the final answer resolves to a real evidence or source id from the run and unknown ids are flagged
- [x] #2 Quoted spans in the answer are checked verbatim (casefold plus whitespace-normalized with fuzzy fallback) against source content captured during phase 2 and mismatches are flagged not silently dropped
- [x] #3 Verification outcome is machine-readable (checked/verified/unverified/misquoted counts) in the deep-search result payload and summarized in the honesty footer
- [x] #4 Flagged statements remain visible in the answer with a marker rather than being deleted
- [x] #5 Verification adds no additional network calls and stays within the existing deep-search wall-clock deadline
- [x] #6 Unit tests pin the matching ladder and flagging behavior
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the deep-search pipeline (aggregate_results + prompts + FinalAnswerDict) and the web_deep_search tool output assembly with their tests
2. TDD a new Web_Scraping/deep_search_citations.py module: citation marker extraction and resolution against evidence ids plus verbatim-first quote matching ladder (exact then casefold or whitespace-normalized then bounded fuzzy)
3. Wire verification into analyze_and_aggregate after final answer generation and expose a citation_verification block in the result payload
4. Surface verification counts in the web_deep_search tool honesty footer without new network calls
5. Run full deep-search test files plus lint
ADR required: no - verification pass internal to the existing deep-search pipeline; additive payload key only. ADR-024 (RAG citation provenance) is related but governs RAG chunks not web deep-search
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- New module `tldw_chatbook/Web_Scraping/deep_search_citations.py` ports tldw_server dev's verbatim-first checking design (Claims_Extraction/alignment.py ladder + guardrails quote citations) onto the chatbook evidence shape. `match_quote_in_sources` runs exact substring → casefold/whitespace-normalized containment → bounded token-level fuzzy (difflib SequenceMatcher ≥ 0.75 over windows of len(q)..len(q)+8 tokens; the server's aligner accepts 0.6 — chatbook stays stricter since a false "verified" is worse than a flagged quote).
- `verify_citations(answer_text, evidence)` resolves `[n]` markers against evidence ids 1..N, flags unknown ids inline as `[n?]` (statements stay visible — nothing deleted), quote-checks spans ≥ 4 chars against the scraped `original_content` (falling back to the LLM `content` summary), and counts sentences with no marker as "uncited" (informational only).
- `aggregate_results` runs verification on the LLM-success branch only and stores counts under a `NotRequired` `citation_verification` key on `FinalAnswerDict`; failure/empty branches omit the key rather than fabricating a clean verdict. The returned `text` is the annotated text.
- `web_deep_search` renders the counts into its honesty footer via `summarize_for_footer` (quiet on clean runs: only "Citations: N/N resolved" is always shown when markers exist; unknown/misquoted/uncited segments appear only when non-zero). Footer is built before the sources byte budget, so length accounting is unchanged.
- Verified: 100 deep-search tests pass (17 new in `Tests/Web_Scraping/test_deep_search_citations.py`, 1 updated in `test_deep_search_pipeline.py`), full `Tests/Web_Scraping/` + `Tests/Tools/` suites pass (757 passed, 3 skipped), 49 Research/Research_Interop tests pass, ruff check adds no new findings over the dev baseline. Tests were written first and watched fail (`ModuleNotFoundError` then assertion failures) per TDD.
- Known scope decisions: misquoted quotes are reported in counts/footer but not rewritten inline (only unknown citation ids get an inline marker); fuzzy threshold and window bounds are module constants, tunable later.
<!-- SECTION:NOTES:END -->
