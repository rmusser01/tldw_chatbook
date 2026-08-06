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

## Research (verified 2026-08-06, developers.openai.com/api/docs/pricing)

**This task's Description premise is WRONG and should not be built on.** Realtime audio is
billed **per 1M tokens**, not per audio minute — so `ModelPricing`'s existing per-mtok shape
extends additively (new optional audio fields, exactly how `cache_read_per_mtok` expresses
"no published rate") rather than needing a redesign. Only *transcription* is per-minute.

Rates read from the official pricing page, units "per 1M tokens unless noted":

| Model | Text in | Cached text in | Text out | Audio in | Cached audio in | Audio out |
|---|---|---|---|---|---|---|
| gpt-realtime (the app's default) | $4.00 | $0.40 | $16.00 | $32.00 | $0.40 | $64.00 |
| gpt-realtime-mini | $0.60 | $0.06 | $2.40 | $10.00 | $0.30 | $20.00 |
| gpt-realtime-2.1 | $4.00 | $0.40 | $24.00 | $32.00 | $0.40 | $64.00 |
| gpt-realtime-2 | $4.00 | $0.40 | $24.00 | $32.00 | $0.40 | $64.00 |
| gpt-realtime-1.5 | $4.00 | $0.40 | $16.00 | $32.00 | $0.40 | $64.00 |

Whisper transcription: **$0.006 per minute** (the unit `ProviderUsage.transcription_seconds`
feeds).

Design notes for whoever implements this:
- Cached AUDIO input is a *separate rate* from cached text input — they coincide at $0.40 for
  `gpt-realtime` but diverge for `-mini` ($0.30 vs $0.06), so one shared cache field would be
  wrong. Check first whether `ProviderUsage` can even attribute cache reads to audio vs text
  (`input_token_details`), and if it cannot, say so and price conservatively rather than
  guessing.
- Confirm whether `ProviderUsage.audio_input` is inclusive of cached audio tokens before
  summing, or the bill will double-count.
- Re-verify these rates before committing them: the catalog carries `as_of` precisely because
  published rates go stale.

**Note (2026-08-06, post-implementation review): the drain-order fix.** The first
implementation drained audio tokens out of `uncached_input` before `cache_read`, which
*under*-bills on an ordinary multi-turn realtime session (any turn with nonzero `cache_read`)
-- the exact opposite of this section's "price conservatively" instruction. The corrected,
shipped direction drains the CHEAP bucket (`cache_read`) first, leaving text tokens stranded
in the expensive `uncached_input` bucket for as long as possible -- see
`PricingCatalog.cost_for_usage`'s docstring in `pricing_catalog.py` for the full derivation
and a worked numeric example. Left here so a future reader re-deriving this from scratch
starts from the right intuition instead of the tempting-but-wrong one.

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
- Trap 2 (inclusivity): ProviderUsage's own docstring already states audio_input/audio_output are SUBSETS of uncached_input+cache_read and output respectively, live-confirmed via the ground-truth payload (input_tokens=33 comprises text_tokens=15 + audio_tokens=18). Confirmed by reading from_provider_payload's realtime branch directly. Pricing audio_input on top of the full uncached_input/cache_read totals would double count; cost_for_usage now subtracts audio tokens from those buckets first (cache_read drained before uncached_input -- see the "Post-review follow-up" paragraph below for why that order, not the reverse, is the conservative/cost-maximizing split among the otherwise-underdetermined alternatives) before applying the ordinary text rates to what remains. audio_output is unambiguous (output is never cached) so it needed no such adjustment.

Catalog: seeded direct "openai:<model>" entries for gpt-realtime, gpt-realtime-mini, gpt-realtime-2.1, gpt-realtime-2, gpt-realtime-1.5 from the task's verified rate table, as_of "2026-08-06". Added a flat $0.006/minute transcription_per_minute (Whisper) to every entry, feeding ProviderUsage.transcription_seconds -- billed independently of the token buckets.

Modal: ConsoleCostRow gained audio_input/audio_output/transcription_seconds (trailing, defaulted, so every existing positional construction site keeps working); console_cost_modal.py's _format_row appends them as their own segment when non-zero rather than leaving them folded invisibly into the row's single cost_usd figure.

Files: tldw_chatbook/LLM_Calls/pricing_catalog.py (ModelPricing/CostBreakdown fields, _entry() helper, realtime seed table, cost_for_usage), tldw_chatbook/Chat/console_cost_tracker.py (ConsoleCostRow fields, build_cost_rows), tldw_chatbook/Widgets/Console/console_cost_modal.py (_format_row), plus new/extended tests in Tests/LLM_Calls/test_pricing_catalog.py, Tests/Chat/test_console_cost_tracker.py, Tests/UI/test_console_cost_modal.py (new file).

Verification: RED confirmed before implementing (12 new tests failing on AttributeError/TypeError), GREEN after. Full targeted run: Tests/LLM_Calls/test_pricing_catalog.py + Tests/Chat/test_console_cost_tracker.py + Tests/UI/test_console_cost_modal.py + Tests/Chat/test_provider_usage.py = 85 passed. Contract trio (test_console_hands_free.py, test_console_hands_free_wiring.py, test_console_dictation.py) = 103 passed, files byte-identical (git diff empty). Cost-chip suites (test_console_status_chips_cost.py, test_console_cost_chip_screen.py) = 28 passed. Repo-wide --collect-only sweep: 30735 tests collected, 4 pre-existing unrelated collection errors (TTS/Web_Scraping, untouched by this change).

Post-review follow-up (2026-08-06, F1/F2): code review caught a money bug in the Trap-2 fix above -- draining audio out of `uncached_input` BEFORE `cache_read` (as originally shipped and as this section still described until this edit) actually MINIMIZES the bill on any turn with nonzero `cache_read`, the opposite of "price conservatively": since `input_per_mtok` > `cache_read_per_mtok` for every seeded model, stranding text tokens in the CHEAP bucket (by draining the EXPENSIVE bucket first) under-counts them. Fixed by flipping the drain order -- `cache_read` first, `uncached_input` only once `cache_read` is exhausted -- which strands text tokens in the EXPENSIVE bucket instead, the true cost-maximizing split. Reviewer's worked example (uncached_input=2000, cache_read=8000, audio_input=500, gpt-realtime): old order billed $0.0252, corrected order bills $0.0270 (~7% higher, not a rare edge case -- it's the normal state of any multi-turn realtime session with a warm cache). `cost_for_usage`'s docstring rewritten with the full sign derivation and an explicit "don't flip this back" note. The pinned-numbers regression test that covered this case passed throughout because it mirrored the (buggy) implementation's own output instead of an independent expectation -- replaced with `test_cost_for_usage_realtime_picks_the_cost_maximizing_audio_split`, which recomputes each candidate split's total independently in-test and asserts the implementation reaches the true maximum; mutation-verified by manually re-flipping the drain order and confirming both that test and its concrete-numbers companion fail. Separately, this file's own "## Research" section (the verified rate table two source comments point readers at) had been silently dropped by a later `backlog task edit --notes`/`-s Done` call, which regenerates the file from its known sections only and does not preserve custom headings -- restored verbatim from `git show 988a64f5d`, added back via direct file edit (not the CLI) to avoid the same loss recurring.
<!-- SECTION:NOTES:END -->
