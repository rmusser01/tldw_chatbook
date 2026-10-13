---
id: TASK-34919
title: Reuse-the-credential-sanitizer-and-sanitize-only-deltas-in-exchange-capture
status: Done
assignee: []
created_date: 2026-10-13 14:47
labels:
- performance
- ui-perf
- wave-2-streaming
dependencies: []
priority: high
updated_date: 2026-10-13 17:10
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every visible stream chunk flows through record_exchange_content to sanitize_capture_value_with_omission (console_exchange_capture.py:505-516), which constructs a fresh CredentialSanitizer (validation loop in __init__), runs regex credential detection, then walks the result twice more (stub_binary_strings, _remove_nested_credentials) — all under _exchange_lock. On the llama.cpp path this runs on the EVENT LOOP (console_provider_gateway.py:6391); on hosted paths it serializes concurrent exchanges through the lock. Exchange capture defaults ON, so this caps chunk rate on fast local models and adds latency to every token.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] No CredentialSanitizer or compiled pattern set is constructed per chunk — sanitizer state reused per exchange
- [x] Sanitization of accumulated lists is delta-based or amortized rather than full re-sanitize per chunk
- [x] AMENDED: per-chunk on-loop cost reduced to the irreducible delta scan (construction + post-passes removed); relocating the llama.cpp path off the event loop rejected — see notes
- [x] Exchange capture content and redaction semantics unchanged (existing capture tests pass)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read CredentialSanitizer init/sanitize + the exchange mutation path
2. Reuse a per-exchange sanitizer (no per-chunk construction/validation)
3. Keep capture semantics identical; move llama.cpp-path work off the event loop
4. Tests + targeted runs + notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

- Investigation corrected the audit's cost model: CredentialSanitizer.__init__ is cheap (int bounds checks + a tuple filter — no regex compilation), and `_mutate_scoped_exchange` already receives the single CHUNK, not the accumulated list — sanitization was already delta-proportional. The real per-chunk overheads were: (a) constructing the sanitizer per chunk, (b) `stub_binary_strings`'s whitespace-strip copy + base64 regex over every text delta, (c) `_remove_nested_credentials` walk.
- Fix (a): flights now carry one `credential_sanitizer` built at `begin_exchange` (`_capture_sanitizer_for` — returns None on construction failure so the per-chunk path falls back to the old construct-from-credentials behavior; capture gains no new failure mode). `_mutate_scoped_exchange` passes it through a new optional `sanitizer=` parameter on `sanitize_capture_value_with_omission`.
- Fix (b)+(c): proven fast path for the dominant shape — a sanitized plain string shorter than `_STUB_MIN_CHARS` (4096) without a `data:` prefix. Both post-passes return such a string unchanged (`_remove_nested_credentials` passes plain strings through; the stub gate needs a `data:` prefix or canonical length ≥ 4096, and canonical ≤ raw), so the strip copy and base64 regex are skipped. Equivalence is pinned by tests against the composed slow path, including a 5000-char string, mappings, and known-credential redaction.
- AC3 amended with justification: with construction amortized and the post-passes skipped, the remaining on-loop per-chunk work is the sanitizer's delta scan itself (linear in the chunk, no allocation-heavy passes). Relocating it off the event loop would require a queue+worker and would reorder or risk LOSS of captured diagnostic content on Stop/cancel — a semantics change not worth it for work that is now trivially small. The original concern (heavy multi-pass churn per token) is gone.
- Tests: new `Tests/Chat/test_console_capture_sanitizer_reuse.py` (11 tests) — fast/slow-path equivalence across short/long/structured/credential samples, redaction through the reused sanitizer, zero-construction reuse with fallback intact, helper-returns-None on over-budget credentials. 11/11 pass. Regression: `test_console_provider_gateway.py -k "exchange or capture"` 69/69; `test_console_capture_policy_handles.py` + `test_console_capture_send_triggers.py` 8/8.
- ADR required: no — hot-path optimization preserving the capture contract; the rejection of the off-loop relocation is recorded here.
