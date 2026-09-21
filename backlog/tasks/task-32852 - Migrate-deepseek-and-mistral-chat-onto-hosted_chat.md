---
id: TASK-32852
title: Migrate deepseek and mistral chat handlers onto the hosted_chat engine
status: Done
assignee: []
created_date: '2026-09-19 08:24'
labels:
  - core-review
  - review-cascade
dependencies:
  - TASK-19642.10
  - TASK-32851
parent_task_id: TASK-32850
references:
  - qa/cascade-review-2026-09-19/report.md
  - backlog/decisions/062-hosted-chat-completions-provider-boundary.md
  - backlog/decisions/064-deepseek-dual-api-provider-boundary.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Second ADR-062 migration wave, after groq/openrouter (TASK-32851) prove the pattern. `chat_with_deepseek` (`LLM_Calls/LLM_API_Calls.py:3147`, 254 LOC, payload 100% standard OpenAI keys) and `chat_with_mistral` (`:4987`, 239 LOC) are ~70-75% transport boilerplate. Same free fixes: deepseek's DONE-yield-in-`finally` (`:3310-3313`) and mistral's (`:5135-5138`), dropped streamed usage, and wrong metric labels (deepseek logs `mistral_api_*` `:3337-3367`; mistral logs `openrouter_api_*` `:5116-5127`).

Genuinely provider-specific and preserved: mistral's `random_seed` (not `seed`), `safe_prompt`, `has_system_in_input` dedup, `Accept` header, endpoint key `mistralai`, no stop/penalties.

Sequencing with TASK-15677 (DeepSeek dual-API, ADR-064): migrating first shrinks 15677's surface — the Responses wire mode lands as a mode on the migrated profile instead of on a hand-rolled handler. If 15677 starts first, contribute to it rather than racing (see lessons-backlog-hygiene on duplicate implementations). ADR required: no — executes ADR-062's direction; record the 064 interaction in the implementation notes.

Source: cascade review 2026-09-19 — `qa/cascade-review-2026-09-19/report.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `chat_with_deepseek` and `chat_with_mistral` route transport through `hosted_chat.py` as provider profiles; mistral's provider-specific payload behavior is preserved and pinned
- [x] #2 The DeepSeek dual-API coordination with TASK-15677/ADR-064 is recorded: either this landed first (15677 builds on the profile) or the handoff is written down
- [x] #3 Streaming Stop closes the transport (no yield-in-`finally`); streamed usage is captured for both
- [x] #4 Metrics name the right provider; consumers of the old labels identified
- [x] #5 Provider-neutral contract coverage for both; existing tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extract the shared `LegacyLineStream` from the groq/openrouter shims; refactor both profiles onto it (green before touching the new providers).
2. TDD red: deepseek/mistral preserved-contract pins + flipped defect pins.
3. deepseek.py + mistral.py profiles mirroring the established pattern; splice the old handlers; re-export entry points.
4. Full verification incl. pristine-A/B classification of any adjacent failures; ruff; commit; push.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PR: #2747 (wave 2, stacked on #2746). Landed 2026-09-19, commit `7b78a6da57` on `fix/cascade-prep` (branch rebased onto `origin/dev` `cccf0acdad` first — dev's 22 intervening commits touch none of this branch's files). TDD red→green: 4 pins failed for the right reasons pre-implementation, all 16 green after.

- **Shared shim first:** `tldw_chatbook/LLM_Calls/legacy_line_stream.py` — the ~60-line line-relay shim extracted from the two identical per-provider classes; groq.py/openrouter.py refactored onto it and their own copies deleted (verified green before the new providers).
- **Profiles:** `deepseek.py` (331 lines) and `mistral.py` (324) on the established pattern. Mistral's specifics preserved and pinned: `random_seed`/`safe_prompt` keys, `Accept: application/json` via `extra_headers`, system-message dedup, `mistralai` endpoint key, no stop/penalties/top_k forwarded. deepseek is the standard OpenAI payload.
- **Fixed and pinned:** clean Stop with exactly-once close; requested + forwarded streamed usage; exactly one `[DONE]`; deepseek's non-streaming metrics now name `deepseek_api_*` (was `mistral_api_*`), mistral's streaming metrics now name `mistral_api_*` (was `openrouter_api_*`); stream-read failures raise typed redacted errors instead of yielding synthetic error chunks (deepseek's old behavior). No consumer outside `LLM_Calls/` keys on the old labels (grep-verified, same as 32851).
- **AC#2 coordination:** no in-flight work exists on TASK-15677 (no branch, no PR — checked 2026-09-19). This landed first, so 15677's Responses wire mode builds as a mode on `deepseek.py`'s profile; the module docstring says so.
- **LOC:** `LLM_API_Calls.py` −572 more lines this commit (5,127 → 4,555; **−1,084 on the branch across both waves** against +1,046 of new profile/shim modules — the four OpenAI-compatible providers now share one transport instead of four).
- **Gate reconciliation (post-PR):** inventory re-pinned for the deepseek/mistral statements (`51e7a1a207`, re-based later) and the ui-ready census bumped 1023 -> 1026 for the two profiles plus the shared shim (`37e78ad391`).
- **Verification:** 284 passed / 0 failed across the 4-provider characterization file, both engine contract suites, the streaming suite, and the dispatcher mapping; ruff clean (repo config) on all five provider/shim modules. Adjacent failures classified by pristine-worktree A/B at `cccf0acdad`: all 5 `Tests/Chat/test_openai_streaming_usage.py` failures (drive `chat_with_openai`, untouched here) fail identically at the base — pre-existing on dev, same class of finding as TASK-19642.10's notes; whoever picks up the openai handler's keep-profile treatment should sweep them.
<!-- SECTION:NOTES:END -->
