# Task 6 implementation report

Status: implemented and self-reviewed; TASK-38541 remains In Progress for the controller's independent review. BASE: `cb4131210904f4ac4005f638eb90cc4585364a73`. Worktree: `/Users/macbook-dev/.codex/worktrees/prompt-mix-ins/tldw_chatbook`. ADR: `backlog/decisions/286-console-prompt-mix-ins.md`; no new ADR. No subagents, branch changes, push, merge, full suite, live cloud calls or paid cache writes.

## Implementation

- `PreparedConsoleRequest.clean_history_message_index` is derived only from retained complete historical units when the active request carries the composition owner's applied-nonempty marker. Windowing rebuilds it after dropping complete units. Legacy empty, recorded empty and all-off sends add no boundary and retain original wire payloads. No body text search identifies a boundary.
- Frozen `ConsoleCacheBoundary` binds the adapter-input index to the exact serialized payload fingerprint. Its fingerprint and both request-side boundary fields are representation-hidden, ephemeral and absent from request JSON. Preparation maps semantic origins after runtime guidance conversion, private/native-key removal and system extraction; an origin spanning active content is refused.
- The gateway projects the private optional keyword through the canonical dispatcher only for Anthropic. Other adapters receive no extra cache fields; unavailable capability/mapping has a fixed-category `console_clean_history_cache_unavailable` counter. Keyword-only addition preserves positional callers.
- Anthropic verifies the frozen witness against actual adapter input, then tracks the exact converted source end through tool-use/result conversion, parallel result and project-guidance coalescing, text and image conversion. Empty, unsupported or lossy ends omit the optional history marker instead of moving it to earlier content. Unavailable mapping/budget has a body-free `anthropic_clean_history_cache_unavailable` counter.
- Existing system, last-tool and optional active-tail markers remain authoritative. The new clean-history marker uses only a free fourth slot, with the existing model gate, global kill-switch, per-turn opt-in, configured TTL and rejection/degrade path. Both 5-minute and 1-hour rejection retries preserve the entire instruction-bearing payload except cache controls, and existing 1-hour compatibility-header removal remains intact.
- OpenAI automatic prefix caching and canonical-endpoint, opt-in system/tool-derived `prompt_cache_key` remain unchanged. Different selected mix-ins do not alter the key. Custom endpoints receive neither this key nor speculative breakpoint/options fields. Per controller ruling, no new OpenAI capability family/model migration/API mode/TTL/config option was introduced.
- Updated current Console and provider architecture docs. New and extended tests exercise the real composed snapshot → semantic builder → provider preparation → gateway kwargs → canonical dispatcher → actual Anthropic HTTP adapter capture, not a duplicate cache implementation.

## Current official capability checks (2026-11-01)

Read through the official OpenAI documentation tool and the official Anthropic page:

- [OpenAI prompt caching guide](https://developers.openai.com/api/docs/guides/prompt-caching): GPT-5.6 and later support explicit content `prompt_cache_breakpoint` fields, request `prompt_cache_options`, a 1,024-token minimum and four writes. Implicit caching remains available. A hit requires an eligible matching prefix/routing, not merely stable payload bytes.
- [OpenAI Chat Completions create reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create): model-gated cache options, explicit markers and the 30-minute Chat Completions TTL. The exact matching model page, [GPT-5.6](https://developers.openai.com/api/docs/models/gpt-5.6), confirms Chat Completions and Responses support; current advertised text prices are $4 input / $0.4 cached input / $20 output per million, with cache writes at 1.25 times uncached input. This client does **not** advertise a truthful explicit-cache model/endpoint capability, even though those newer official models support it. Therefore explicit clean-history control is unavailable in this client and automatic caching is retained.
- [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching): four breakpoints, 20-block lookup window, supported text/image/tool blocks, nonempty text required. The actual mocked adapter target is `claude-sonnet-4-6`, with a 1,024-token minimum. Its published per-million rates are $3 input / $15 output / $3.75 five-minute writes / $6 one-hour writes / $0.30 reads. Existing defaults, opt-ins and TTL compatibility behavior are preserved. Tests also preserve the existing Opus-5 TTL helper behavior; that model's published minimum is 512 tokens.

No API-key tool was available. No secrets were read/printed. Payload placement and synthetic usage are **not cache-hit evidence**; there is no latency, cost saving or real-hit claim.

## TDD evidence and targeted commands

All invocations use `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`, private basetemps and `-p no:cacheprovider`. Commands below run in the worktree unless explicitly stated. No full-suite command was run.

### RED

```
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_prompt_mixin_caching.py -q -p no:cacheprovider --basetemp=/private/tmp/mixin-task6-red
```

**1 failed** as expected before implementation: `test_two_mixin_turns_keep_clean_history_system_and_tools_stable` found `older answer` with no `cache_control` on the actual HTTP payload. Existing system/tools and history bytes were unchanged; the missing history marker was the failure.

### GREEN and connected coverage

```
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_prompt_mixin_caching.py Tests/Chat/test_anthropic_caching_degrade.py Tests/Chat/test_console_prepared_request.py -q -p no:cacheprovider --basetemp=/private/tmp/mixin-task6-green1
```

**53 passed in 2.19s**, including existing immutable preparation/windowing and cache-degrade behavior.

```
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_prompt_mixin_caching.py Tests/Chat/test_anthropic_caching_degrade.py Tests/Chat/test_cache_usage_metrics.py Tests/Character_Chat/test_prompt_cache_review_regressions.py -q -p no:cacheprovider --basetemp=/private/tmp/mixin-task6-green2
```

**71 passed, 6 failed in 17.28s** before fixture qualification. All new payload, 5m/1h degradation, native warmed world-info/dictionary off-toggle cases passed. Six existing usage tests failed before HTTP entry with `RecoveryRequired("raw_source_selection_changed")`.

Exact original BASE probe (no fixture changes):

```
mkdir -p /private/tmp/mixin-task6-base
git archive cb4131210904f4ac4005f638eb90cc4585364a73 tldw_chatbook Tests pyproject.toml config.toml | tar -x -C /private/tmp/mixin-task6-base
cd /private/tmp/mixin-task6-base
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_cache_usage_metrics.py -q --tb=line -p no:cacheprovider --basetemp=/private/tmp/mixin-task6-baseline-six
```

**The same six failed in 1.36s** on BASE, with the same admission reason. The native adapter-owned usage module now uses the repository's legitimate `bootstrap_profile` marker: it retains the real collection-admitted private config rather than changing a participant's source after import. No admission/transport/service was stubbed to make it pass.

Final affected-module/new-file verification after that correction and the fixed-category status sink:

```
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_prompt_mixin_caching.py Tests/Chat/test_cache_usage_metrics.py -q --tb=short -p no:cacheprovider --basetemp=/private/tmp/mixin-task6-status-final
```

**31 passed in 1.30s**. (Two earlier identical focused passes were 31/31 in 1.54s and 1.56s; they are repetitions, not additional coverage.) A new test fixture initially imported a nonexistent snapshot module; that fixture was corrected to send distinct legacy/recorded empty `MixinReplayRead` values through composition.

```
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_provider_gateway.py -q -k 'prompt_caching or system_message_is_byte_stable or usage_accumulates_per_call_without_leaking_stale_cache_fields' -p no:cacheprovider --basetemp=/private/tmp/mixin-task6-gateway-final
```

**5 passed, 464 deselected in 0.65s**. The earlier same selection also passed 5/5.

### Shared native conversion/streaming qualification

The unchanged native adapter modules initially hit the same configuration-owner failure (43 cases failed before HTTP entry; 2 pure tests passed). They are outside this slice and were not edited. The retained `task-6-evidence/task6_native_bootstrap.py` adds `bootstrap_profile` only to the three exact native adapter file names during collection; it changes no test logic, product admission, timeout or HTTP fake. Exact same plugin and commands were run on BASE and current:

```
PYTHONPATH=/private/tmp /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -p task6_native_bootstrap Tests/Chat/test_anthropic_native_tools.py Tests/Chat/test_anthropic_prefix_stability.py Tests/Chat/test_anthropic_streaming_usage.py -q --tb=line -p no:cacheprovider --basetemp=/private/tmp/mixin-task6-native-base-qualified
```

BASE archive cwd: **45 passed in 1.70s**.

```
PYTHONPATH=/private/tmp /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -p task6_native_bootstrap Tests/Chat/test_anthropic_native_tools.py Tests/Chat/test_anthropic_prefix_stability.py Tests/Chat/test_anthropic_streaming_usage.py -q --tb=short -p no:cacheprovider --basetemp=/private/tmp/mixin-task6-native-current-qualified
```

Current cwd: **45 passed in 1.82s**. Reproduce by copying the retained plugin into `/private/tmp/task6_native_bootstrap.py`. Native coverage includes system/tool/all-off payloads, history/tool and project-guidance coalescing, assistant text/tool calls, malformed calls, 3/4 breakpoint budget, prefix stability and streaming usage.

### Diagnostic inventory qualification

The exploratory combined native/manifest run returned **44 failed, 3 passed, 2 warnings**: the 43 unqualified native cases above plus the existing strict manifest guard. A temporary extra gateway debug statement was removed in favor of the fixed-category status counter. Exact per-file `scan_source` / `diagnostic_digest` comparison now proves unchanged diagnostic sets versus BASE in all three touched logging owners: gateway **21/21**, provider adapter **135/135**, canonical dispatcher **57/57**. No logging inventory was regenerated, weakened or updated.

Final exact manifest command:

```
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/LLM_Calls/test_summarization_diagnostic_privacy.py::test_manifest_boundary_tracks_reconciled_checked_and_generated_baselines Tests/LLM_Calls/test_summarization_diagnostic_privacy.py::test_manifest_boundary_changes_only_summarization_owner_diagnostics -q --tb=short -p no:cacheprovider --basetemp=/private/tmp/mixin-task6-inventory-final
```

**1 passed, 1 failed, 2 warnings in 44.30s**. The failure is the *checked* inventory versus the pre-existing fixture hash (`a103c429...` versus `68a04025...`), before comparing generated inventory. Both checked inventory and fixture are unchanged by this slice. The two warnings are existing invalid escape sequences in `Utils/Splash_Screens/environmental/train_journey.py:31/32` during source scanning. Exact BASE verification (same two unmodified test nodes, archive cwd, `--tb=line`, basetemp `/private/tmp/mixin-task6-inventory-base2`) also returned **1 passed, 1 failed, 2 warnings in 44.24s**, at the same checked-inventory assertion. BASE support files were extracted with:

```
git -C /Users/macbook-dev/.codex/worktrees/prompt-mix-ins/tldw_chatbook archive cb4131210904f4ac4005f638eb90cc4585364a73 scripts Docs/security | tar -x -C /private/tmp/mixin-task6-base
cd /private/tmp/mixin-task6-base
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/LLM_Calls/test_summarization_diagnostic_privacy.py::test_manifest_boundary_tracks_reconciled_checked_and_generated_baselines Tests/LLM_Calls/test_summarization_diagnostic_privacy.py::test_manifest_boundary_changes_only_summarization_owner_diagnostics -q --tb=line -p no:cacheprovider --basetemp=/private/tmp/mixin-task6-inventory-base2
```

An initial archive-support extraction mistakenly invoked Git from the archive (which has no `.git`) and produced a collection-only `ModuleNotFoundError: scripts`; that setup attempt is not counted as a test result. The corrected command above supplied exact BASE support files. No unrelated manifest fix was attempted.

## Whole-file static qualification

`git diff --check` passed. Python AST parsing passed for every touched Python file. Only changed logical ranges in existing files and the entire new test file were formatted using Ruff 0.16.6; no broad formatter cleanup.

Retained `task-6-evidence/lint_differential.py` and `lint-results.json` compare every touched Python file to exact BASE using `python -m ruff check --no-cache --config /Users/macbook-dev/.codex/worktrees/prompt-mix-ins/tldw_chatbook/pyproject.toml --output-format json`. The retained script enumerates paths with `git diff cb4131210904f4ac4005f638eb90cc4585364a73 --name-only`, so it remains reproducible after the implementation commit. Comparison is by full-file `(rule code, message)` multiplicities with diagnostic-message `from line N` offsets normalized; no changed-line filtering. Zero new diagnostics. The same `pyproject.toml` bytes govern both trees. Initial new import-block findings were corrected rather than hidden.

| File | BASE | Current | New |
| --- | ---: | ---: | ---: |
| `Tests/Character_Chat/test_prompt_cache_review_regressions.py` | 0 | 0 | 0 |
| `Tests/Chat/test_anthropic_caching_degrade.py` | 3 | 3 | 0 |
| `Tests/Chat/test_cache_usage_metrics.py` | 0 | 0 | 0 |
| `tldw_chatbook/Chat/Chat_Functions.py` | 204 | 204 | 0 |
| `tldw_chatbook/Chat/console_prepared_request.py` | 0 | 0 | 0 |
| `tldw_chatbook/Chat/console_provider_gateway.py` | 66 | 66 | 0 |
| `tldw_chatbook/LLM_Calls/LLM_API_Calls.py` | 248 | 248 | 0 |
| `Tests/Chat/test_console_prompt_mixin_caching.py` | 0 | 0 | 0 |

## Files and self-review

Product owners changed: `Chat/console_prepared_request.py`, `Chat/console_provider_gateway.py`, canonical `Chat/Chat_Functions.py`, `LLM_Calls/LLM_API_Calls.py`. Tests changed: new `Tests/Chat/test_console_prompt_mixin_caching.py`; existing degradation, usage metrics and Character_Chat prompt-cache review files. Current docs: `Docs/Architecture/console.md`, `Docs/Architecture/llm-providers.md`; task plan/notes: TASK-38541.

Self-review checked all callers of preparation/dispatcher/Anthropic entry points; keyword-only compatibility; source coordinates after system extraction and private-key stripping; windowed remapping; active tool-loop continuation; nonempty applied marker versus empty operational identity; stale/tampered fingerprint refusal; lossy image and unsupported-role omission; exact coalesced block ownership; existing cache slots/kill-switch/opt-in/TTL; no caller mutation; no fingerprint/instruction body in repr, payload or status diagnostics; custom OpenAI endpoint exclusion; stable system/tool key; cache miss/read/write usage remaining provider-reported. No DB, trace restoration, persistence, UI or TTL policy changed. No new generalized lesson beyond the already documented native-profile fixture rule.

Remaining limits: targeted checks only, no real-provider hit/latency/cost measurement, no live UI. OpenAI explicit breakpoints remain unavailable in this client's undeclared capability contract. The unchanged checked-in diagnostic manifest mismatch is separately qualified, not claimed fixed. Independent controller review is required before Done.

Commit: `817618048a` — `perf: preserve clean-history cache boundaries for mix-ins`. Post-commit worktree status: clean. Existing Git GC/unreachable-object warnings were reported; no cleanup was performed.


## Review fix round 1 — I1 (FIX_BASE `817618048a`)

Read the complete `task-6-review.md` and verified the all-invalid assistant-call fallback against the actual converter. The legacy converter intentionally preserves the ordinary text when no tool call survives. The new optional cache mapping previously treated that surviving earlier text as the exact source end; the existing partial-valid count guard did not cover this fallthrough.

Added per-row `source_end_preserved` eligibility in the new boundary mapping only. A fully discarded assistant tool-call tail makes the existing plain-content fallback ineligible for the optional history marker. Conversion, fallback text, valid tool blocks, the active marker/body, existing status categories, cache switches, TTL and degrade behavior remain unchanged. The partial-valid guard remains in place. No new logs, fields, schema, provider capability or converter repairs. ADR check: no new ADR; routine correction under ADR-286. Current architecture docs already state the corrected lossy-omission contract and need no revision. TASK-38541 remains In Progress for independent re-review.

Two parameterized actual-HTTP cases use the same composition → semantic → prepared → gateway → dispatcher → HTTP harness: the requested all-invalid final historical assistant (`surviving text` plus an empty tool name), and a valid tool call followed by the invalid tail. Both assert no historical marker, exact legacy content/valid-tool projection, unchanged literal active body and active marker, and the fixed-category unavailable counter.

RED, before the product fix:

```
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_prompt_mixin_caching.py -q -k discarded_historical_tool_call_tail --tb=short -p no:cacheprovider --basetemp=/private/tmp/mixin-task6-fix1-red
```

**1 failed, 1 passed, 22 deselected in 0.43s.** The all-invalid actual HTTP payload had markers `[(1, 0), (2, 0)]` instead of only the existing active `[(2, 0)]`; mixed-valid-invalid already omitted the optional marker.

GREEN and exact affected legacy/degrade qualification:

```
PYTHONPATH=/private/tmp /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -p task6_native_bootstrap Tests/Chat/test_console_prompt_mixin_caching.py Tests/Chat/test_anthropic_native_tools.py::test_all_junk_tool_calls_fall_back_to_plain_content Tests/Chat/test_anthropic_native_tools.py::test_junk_tool_call_skipped_among_valid_entries Tests/Chat/test_anthropic_caching_degrade.py::test_400_naming_cache_control_retries_stripped Tests/Chat/test_anthropic_caching_degrade.py::test_cache_degrade_preserves_mixin_instructions_at_real_adapter_boundary -q --tb=short -p no:cacheprovider --basetemp=/private/tmp/mixin-task6-fix1-green
```

**29 passed in 1.28s.** The retained `task-6-evidence/task6_native_bootstrap.py` supplies only the existing native-owner profile qualification, as in the first report. No full 45-case native rerun, manifest rerun, live cloud call or full suite was performed for this fix.

Static checks:

```
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python /private/tmp/task6_fix1_lint.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check --no-cache Tests/Chat/test_console_prompt_mixin_caching.py
git diff --check
```

Retained reproducible script/results: `task-6-evidence/fix1-lint-differential.py` and `fix1-lint-results.json`. Both full touched files are compared to exact FIX_BASE with the same explicit current `pyproject.toml`, normalized embedded source offsets, and exact FIX_BASE package `__init__.py` context. Adapter **248 → 248**, new caching test **0 → 0**; **zero introduced diagnostics**. AST parsing passed for both files; the test file is formatted; whitespace check passed. An initial bare-files baseline lacked package initializers and therefore omitted the existing N999 module-name finding; exact package context corrected the comparison, without a product lint change. Adapter source-level diagnostic inventory is **135 → 135 with the same digest**, so there is no new logging to qualify with the known unrelated manifest guard.

Self-review: the eligibility flag starts true per source row, becomes false only on the all-invalid assistant-call fallthrough, and gates only `record_history_end`; valid-call and ordinary role conversions retain their original control flow. The mixed case pins the existing count safeguard and both cases exercise real payload placement/status. No new concerns; the original no-live-hit and exact-BASE manifest qualifications remain unchanged.

Fix commit: `2cc49ffc82ef024d432da2f36550c59667776ac6` — `fix: omit cache boundary after discarded tool-call tails`. Post-commit tracked worktree status: clean. Existing Git maintenance warnings were left untouched.
