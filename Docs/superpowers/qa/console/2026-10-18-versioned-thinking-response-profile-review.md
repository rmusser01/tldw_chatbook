### Spec Compliance

- ✅ Spec compliant for the supplemental versioned-profile repair. Reviewed base `38cf68612508d23e9ae70a5deecf76a913fff3e0` through head `f9b8746a2ccd35a389923c4e89ee1c009bb224e4`, against Task4 brief, the approved 2026-10-18 design and ADR097 amendment at `backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md:641`. Prior Task1–3 reviews remain closed.
- ✅ Only the two approved production modules change. `tldw_chatbook/Chat/console_semantic_revision.py:110` defines the closed provider/model rule, including nonblank string models and case-insensitive Kimi exclusion. `tldw_chatbook/Chat/console_trace_service.py:6274` uses it for prospective displayable v2 profiles and preserves hosted proprietary v1 profiles.
- ✅ `tldw_chatbook/Chat/console_semantic_revision.py:144` validates exact version-specific keys and exact integer versions; `:167` retains the v1 scalar contract; `:177` requires the canonical ordered frozen source list. `:197` retains terminal-round/single-block selection; `:213` preserves status/type/provider/model/protocol checks; `:228` emits the actual literal source format and text. No old-row rewrite, migration, parser change, equality change or completed-tool-validator change appears in this range.
- ✅ Test changes stay within the approved existing files. `Tests/Chat/test_console_trace_settlement.py:55` adds optional provider/model parameters while preserving defaults. `Tests/Chat/test_console_trace_thinking_settlement.py:50` covers v1/v2 acceptance and refusal, privacy/raw mismatch and source ownership; `:389` covers the closed producer rule. `Tests/Chat/test_console_archive_recovery_followup.py:353` extends the existing real controller/runtime/gateway/SQLite fixture, with only external adapter transport replaced.
- ✅ `Tests/Chat/test_console_archive_recovery_followup.py:632` requires exact terminal `revision/verified_equal`, saved assistant/conversation ownership, literal structured/tagged evidence, frozen tuple and persisted JSON array. `:788` additionally verifies the captured restore successor's revision owner; the following ordinary turn and prior-evidence comparisons remain in the test.
- ⚠️ Live Qwen/template qualification, historical source20f artifacts, normal-profile fingerprint custody and overall AC1 are outside this supplemental diff and remain root-owned, as explicitly recorded in `Docs/superpowers/qa/console/2026-10-18-versioned-thinking-response-profile.md:40`. This approval does not qualify those outcomes or reopen prior reviews.

### Strengths

- The shared helper is small and specific; the reader checks the complete declared contract rather than accepting an arbitrary source list (`tldw_chatbook/Chat/console_semantic_revision.py:110`, `:180`). The actual source evidence is preserved (`:228`).
- The archive regression proves saved ownership and a normal captured successor through the existing production stack, rather than asserting completion alone (`Tests/Chat/test_console_archive_recovery_followup.py:642`, `:791`). Saved header/list representation is tested on both sides of persistence (`:657`, `:671`).
- RED, harness failures, final GREEN and baseline failures are retained separately and honestly described (`Docs/superpowers/qa/console/2026-10-18-versioned-thinking-response-profile.md:21`). The final successful output is warning-free, and intermediate failures are not presented as passing evidence.

### Issues

#### Critical (Must Fix)

- None. Count: **0**.

#### Important (Should Fix)

- None. Count: **0**.

#### Minor (Nice to Have)

- **Count: 1 — diagnostic harness warning noise.** `Docs/superpowers/qa/console/2026-10-18-versioned-thinking-response-profile/baseline-controls.py:9` imports test isolation and related modules before pytest's plugin assertion-rewrite setup at `:24`; retained `baseline-controls-phase2.log.gz` reports three `PytestAssertRewriteWarning` entries (pytest_asyncio, _hypothesis_globals, hypothesis). This does not invalidate the three reproduced baseline failures or the clean final source checks, but makes that diagnostic phase noisier. If the overlay harness is reused, initialize pytest rewrite/plugin handling before those imports while preserving isolation before any production import. Keep the original raw evidence unchanged. Early RED temporary-directory cleanup warnings are also retained and explicitly documented; later phases already use unique basetemp paths.

### Assessment

- **Task quality: Approved.** The minimal two-module change implements the approved prospective contract without widening v1 or altering equality, parser or recovery authority. The remaining issue is diagnostic evidence hygiene, not a source correctness blocker.
- **Counts:** Critical **0**, Important **0**, Minor **1**.

### Focused connected-code and evidence checks

- **Named risk: JSON list versus frozen header representation.** Checked `tldw_chatbook/Chat/console_trace_repository.py:416`: decoded lists are recursively frozen to tuples, matching the reader's exact tuple validation. The persisted JSON-list assertion in the native fixture establishes the wire/storage shape separately.
- **Named risk: structured fixture bypasses the real parser.** Checked `Tests/Chat/test_console_archive_recovery_followup.py:407` through `:444`: the fake returns an ordinary completion dictionary at `ConsoleProviderGateway(chat_api_call_fn=...)`, while the capture factory, runtime and bridge are real. Checked production `_structured_local_thinking` at `tldw_chatbook/Chat/console_provider_gateway.py:2174`, its normal capture path at `:5871`, and the documented adapter dispatch at `:6325`; parser source remains literal reasoning_content and Kimi refusal remains intact. DNS guards and pending-work teardown remain at fixture `:367` and `:455`.
- **Cut-off diff context:** read the complete `_project_revision_response_envelope` once at `tldw_chatbook/Chat/console_semantic_revision.py:128` because its unchanged middle was omitted between diff hunks; confirmed the existing terminal-round/status/single-block checks. The initial full package output was truncated by the tool, so the omitted three test-file sections were read from the package separately. No git commands or broad source crawl were used.
- **Existing evidence inspected, no tests rerun:** decompressed `green-confirmed.log.gz` (**139 passed**, 69.46s), `helper-callers.log.gz` (**13 passed**, 23.13s), `static-final.log.gz` (unchanged baseline Ruff signatures, all five files formatted, scoped whitespace exit0), native RED terminal-link failures and baseline-control failure/warning output. Checked `commands.json` selectors/exits and verified all **35** artifact-manifest SHA256 entries against retained files. Existing baseline static debt is distinguished from new diagnostics.
- **Skipped:** suite reruns, live app/model/network, historical-artifact mutation, cleanup and prior review waves, per custody. **Remaining risk:** root must separately qualify the fresh live journey within its transport/model budget; the three reproduced baseline settlement failures and existing static debt remain outside this repair.
