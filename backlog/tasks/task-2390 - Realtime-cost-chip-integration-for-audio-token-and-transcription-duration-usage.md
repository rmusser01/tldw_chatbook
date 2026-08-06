---
id: TASK-2390
title: >-
  Realtime: cost-chip integration for audio-token and transcription-duration
  usage
status: Done
assignee:
  - '@claude'
created_date: '2026-08-05 04:16'
updated_date: '2026-08-06 07:42'
labels:
  - realtime
  - cost
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-2363 captured realtime's audio/text token split (ProviderUsage.audio_input/audio_output) and input-audio transcription duration (ProviderUsage.transcription_seconds) onto Console turns, but deliberately left them unbilled: pricing_catalog.py's cost math only reads the plain uncached/cache/output token buckets, and realtime is billed per audio MINUTE, not per token, which the current per-mtok pricing model cannot represent as-is. This task is the follow-up that makes the Console cost chip honest about a realtime session's actual cost.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Realtime sessions' cost estimate/display accounts for audio-minute billing using ProviderUsage.audio_input/audio_output and/or transcription_seconds, not just the token buckets pricing_catalog.py already reads
- [x] #2 Pricing catalog entries exist (or a documented decision to omit them) for the realtime model(s) this app supports
- [x] #3 Existing token-based cost math for non-realtime providers is unaffected
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the task's Research section + reread ProviderUsage.from_provider_payload and openai_session.py's ground-truth header to resolve the two traps (cache-vs-audio attribution; audio_input/output inclusivity).
2. Extend ModelPricing/CostBreakdown additively with optional audio_in/audio_out/cached_audio_in/transcription_per_minute fields, defaulting to None/0.0 so every non-realtime construction site is unaffected.
3. Seed direct-mapping catalog entries for openai:gpt-realtime{,-mini,-2.1,-2,-1.5} from the verified rate table, as_of 2026-08-06.
4. Extend PricingCatalog.cost_for_usage to price audio_input/audio_output/transcription_seconds without double-counting the parent buckets, resolving the cache-attribution gap conservatively (documented in the method's own docstring).
5. Thread the new ConsoleCostRow fields through build_cost_rows and show them in the cost-breakdown modal so they are not folded invisibly into one total.
6. TDD: RED tests in Tests/LLM_Calls/test_pricing_catalog.py, Tests/Chat/test_console_cost_tracker.py, Tests/UI/test_console_cost_modal.py, then implement to GREEN; re-run the contract trio + covering suites.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approach: the task's Research section (authoritative, supersedes the Description) established realtime bills per 1M TOKENS like every other model, so ModelPricing/CostBreakdown extend ADDITIVELY with optional audio/transcription fields (defaulting to None/0.0), never repurposing input_per_mtok/cache_read_per_mtok/etc -- AC3's hard constraint. get_pricing()/cost_for_usage() logic for non-realtime models is untouched byte-for-byte when the new fields are absent.

Two traps resolved by reading the code (not guessed):
- Trap 1 (cache-vs-audio attribution): ProviderUsage.from_provider_payload only reads input_token_details.cached_tokens (a TOTAL, text+audio mixed) and .audio_tokens (also a total, cached+uncached mixed) -- it never parses the wire payload's cached_tokens_details sub-object (confirmed live in openai_session.py's ground-truth header, which shows the API DOES publish that finer split, just not something this codebase captures). So ProviderUsage genuinely CANNOT attribute a cache-read token to audio vs text today. Resolution: cost_for_usage never reads ModelPricing.cached_audio_in_per_mtok (seeded for the record per AC2, documented as unused); every audio-input token is priced at the higher UNCACHED audio rate, which is cost-maximizing (never underbills) since audio's cached rate is cheaper than its uncached rate for every seeded model.
- Trap 2 (inclusivity): ProviderUsage's own docstring already states audio_input/audio_output are SUBSETS of uncached_input+cache_read and output respectively, live-confirmed via the ground-truth payload (input_tokens=33 comprises text_tokens=15 + audio_tokens=18). Confirmed by reading from_provider_payload's realtime branch directly. Pricing audio_input on top of the full uncached_input/cache_read totals would double count; cost_for_usage now subtracts audio tokens from those buckets first (uncached_input drained before cache_read, the conservative/cost-maximizing split among the otherwise-underdetermined alternatives) before applying the ordinary text rates to what remains. audio_output is unambiguous (output is never cached) so it needed no such adjustment.

Catalog: seeded direct "openai:<model>" entries for gpt-realtime, gpt-realtime-mini, gpt-realtime-2.1, gpt-realtime-2, gpt-realtime-1.5 from the task's verified rate table, as_of "2026-08-06". Added a flat $0.006/minute transcription_per_minute (Whisper) to every entry, feeding ProviderUsage.transcription_seconds -- billed independently of the token buckets.

Modal: ConsoleCostRow gained audio_input/audio_output/transcription_seconds (trailing, defaulted, so every existing positional construction site keeps working); console_cost_modal.py's _format_row appends them as their own segment when non-zero rather than leaving them folded invisibly into the row's single cost_usd figure.

Files: tldw_chatbook/LLM_Calls/pricing_catalog.py (ModelPricing/CostBreakdown fields, _entry() helper, realtime seed table, cost_for_usage), tldw_chatbook/Chat/console_cost_tracker.py (ConsoleCostRow fields, build_cost_rows), tldw_chatbook/Widgets/Console/console_cost_modal.py (_format_row), plus new/extended tests in Tests/LLM_Calls/test_pricing_catalog.py, Tests/Chat/test_console_cost_tracker.py, Tests/UI/test_console_cost_modal.py (new file).

Verification: RED confirmed before implementing (12 new tests failing on AttributeError/TypeError), GREEN after. Full targeted run: Tests/LLM_Calls/test_pricing_catalog.py + Tests/Chat/test_console_cost_tracker.py + Tests/UI/test_console_cost_modal.py + Tests/Chat/test_provider_usage.py = 85 passed. Contract trio (test_console_hands_free.py, test_console_hands_free_wiring.py, test_console_dictation.py) = 103 passed, files byte-identical (git diff empty). Cost-chip suites (test_console_status_chips_cost.py, test_console_cost_chip_screen.py) = 28 passed. Repo-wide --collect-only sweep: 30735 tests collected, 4 pre-existing unrelated collection errors (TTS/Web_Scraping, untouched by this change).
<!-- SECTION:NOTES:END -->
