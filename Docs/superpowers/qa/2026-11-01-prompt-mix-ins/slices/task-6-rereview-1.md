### Finding Verdicts

- **I1: Lossy all-invalid tool-call conversion still claims an exact history end** — **ADDRESSED**. The per-row `source_end_preserved` flag starts true, becomes false only when all assistant tool calls are discarded, and gates only the optional `record_history_end` call (`tldw_chatbook/LLM_Calls/LLM_API_Calls.py:1710,1813,1849-1852`). The legacy plain-content fallback still appends the same content, and valid/mixed tool-call conversion retains its prior count guard. No active-marker logic changes.
- **Required real-adapter regression** — **ADDRESSED**. Parameterized all-invalid and mixed-valid/invalid source tails run through the existing composed-request HTTP harness and assert no history marker, exact surviving text/valid tool blocks, unchanged literal active content and marker, and the fixed-category unavailable counter (`Tests/Chat/test_console_prompt_mixin_caching.py:445-483`).

### New Breakage in the Fix Diff

- None. The product change affects mapping eligibility only; it adds no conversion policy, provider fields, cache configuration, logging or persistence changes.

### Out-of-Scope Observations

- None.

### Checks

- Read the scoped re-review instructions, supplied fix diff `817618048a..2cc49ffc82` once and the fix-round appendix in `task-6-report.md`; evaluated I1 against the retained task brief and initial review.
- The appendix records the actual-HTTP all-invalid RED failure (1 failed, mixed case already passed), followed by 29 passing focused cases covering the caching file, existing legacy fallback/mixed conversion and cache degradation. Its full-file lint comparison reports adapter 248 → 248 and test 0 → 0, with the adapter diagnostic digest unchanged at 135 diagnostics. These are reported execution results, not independent reruns by this reviewer.
- No tests or lint rerun, full-task re-review, product/index/HEAD mutation, subagents or live provider requests. Only this review report was written.

### Verdict

**Fix round:** All findings addressed, no new Critical/Important breakage.

Skipped suite reruns and live-cache verification; the previously qualified manifest baseline and absence of real hit/cost evidence remain unchanged.
