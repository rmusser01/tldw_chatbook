---
id: TASK-39061
title: "redact_log_line does not mask bare sk_live_, sk_test_, rk_live_ or rk_test_ secrets, including one the log forwarder renders as '<message>: sk_live_…'"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 12:31
updated_date: 2026-11-06 17:59
labels:
- follow-up
- logging
- security
priority: medium
references:
- tldw_chatbook/Utils/log_sanitizer.py
- tldw_chatbook/Logging_Config.py
- Tests/Utils/test_log_sanitizer.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 10 of the 2026-10-23 burn-down, by the TASK-38989 review; checked on dev b2c618185f and on e8eeae7db9 (`fu-base-r10`), 2026-11-06. `log_sanitizer.py` is identical in both; line numbers are dev's. Ones marked #3436 are from Architecture PR #3436 (head b7c8e809b3).

**The gap.** `_STANDALONE_CREDENTIALS` (`tldw_chatbook/Utils/log_sanitizer.py:104` onward) lists the `sk-` key forms (`:105-116`: `sk-proj-`, `sk-ant-…`, `sk-or-v1-`, generic `sk-`). It has no pattern for the underscore forms `sk_live_`, `sk_test_`, `rk_live_` or `rk_test_`.

**Probe.** A pure `redact_log_line` call through run.sh: synthetic 40-character bodies, no network, scratch profile. Same result at both commits.

| Line | Masked |
|---|---|
| `payment retry failed with sk_live_<40> for job 7` | no |
| `… sk_test_<40> …` | no |
| `… rk_live_<40> …` | no |
| `… rk_test_<40> …` | no |
| `persona_buddy_preferences_save_failed: sk_live_<40>` (the forwarder's shape after TASK-38989) | no |
| `persona_buddy_preferences_save_failed exception_type=sk_live_<40>` | no |
| `config api_key=sk_live_<40>` | yes, by the label rule |
| `token: sk_live_<40>` | yes, by the label rule |
| control: `… sk-<40> …` | yes |

**Why it matters now.**
- After TASK-38989, the forwarder (`Logging_Config._forward_loguru_to_standard`, the only production loguru sink) appends a bound `exception_type` that matches `[A-Za-z][A-Za-z0-9_]{0,63}` to every logger's line, as `f"{record['message']}: {exception_type}"` (#3436 `Logging_Config.py:675`).
- A 48-character `sk_live_…` value matches that rule, so the line would read `persona_buddy_preferences_save_failed: sk_live_…`, and `redact_log_line` would leave it in the file sink and the Logs view.
- TASK-38989 guards the producers: a census of the 7 reviewed bind sites, and a package-wide tripwire. So no current site binds such a value, and the sanitizer is the second line of defence.
- `Tests/Utils/test_log_sanitizer.py` runs on no PR lane.
- TASK-23190 and TASK-19555 (both Done) closed other sanitizer gaps, not these prefixes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `redact_log_line` masks bare `sk_live_`, `sk_test_`, `rk_live_` and `rk_test_` secrets of realistic length, including the forwarder's `<message>: sk_live_…` line and an `exception_type=sk_live_…` line; a test is red before the change on the six unmasked probe lines
- [x] #2 Short identifiers that only share a prefix (for example `sk_live_mode`) stay readable, and the existing sanitizer tests stay green, with counts recorded
- [x] #3 The new test runs on a PR lane (never the fast PR contract), with three timings of that lane's command recorded
<!-- AC:END -->

## Implementation Plan

1. Add a test in `Tests/Utils/test_log_sanitizer.py` for the six unmasked probe lines (through `redact_log_line` and `sanitize_string`), plus negative cases for short or underscore-joined identifiers that only share the prefix. Show it red at base.
2. Add the `sk_`/`rk_` `live`/`test` pattern to the diagnostic path that `sanitize_string` and `redact_log_line` share. Leave `_STANDALONE_CREDENTIALS` alone: `sanitize_trace_credentials_v1` uses it too, and that versioned trace projection must not change.
3. Run the whole sanitizer file and the other sanitizer suites, and record the counts.
4. Lane `Tests/Utils/test_log_sanitizer.py` on the Architecture guards job (append to the step and to `ARCHITECTURE_GUARD_TARGETS`), then time the job command three times.

## Implementation Notes

**Status: Done.**

- `Utils/log_sanitizer.py` has a new `_DIAGNOSTIC_ONLY_CREDENTIALS` tuple, holding `(?<![A-Za-z0-9_-])[rs]k_(?:live|test)_[A-Za-z0-9]{24,}(?![A-Za-z0-9_-])`. `_sanitize_diagnostic_text`, the step both `sanitize_string` and `redact_log_line` run, applies it. So every log caller benefits, including the forwarder's file sink and the Logs view.
- It is not in `_STANDALONE_CREDENTIALS`, because `sanitize_trace_credentials_v1` reads that tuple. That projection is versioned: stored trace artifacts compare against it, and `test_logging_change_preserves_legacy_trace_credential_projection` pins it. Credentials-v1 traces therefore still do not mask these keys (Proposed follow-ups in the report).
- The 24-character floor (Stripe's shortest key body) and a body without `_` keep short or underscore-joined identifiers readable.

**Evidence.**
- AC#1, red before the fix: `test_bare_underscore_live_and_test_keys_are_masked` covers 4 prefixes and 3 line shapes. The shapes are the prose line, the forwarder's `persona_buddy_preferences_save_failed: <key>` and `... exception_type=<key>`, so the six unmasked probe lines are all included. Before the fix 12 failed. Each case checks both `redact_log_line` and `sanitize_string`.
- AC#2: `test_identifiers_that_only_share_the_key_prefix_stay_readable` covers `sk_live_mode`, a bare `rk_test_`, `sk_test_mode=true`, `sk_live_mode_enabled_for_every_customer_account`, a prefix inside a word (`desk_live_…`) and the uppercase form. All 6 pass before and after.
- Whole file: 168 passed (base: 150 passed, plus 18 new), `-n 4`.
- Other sanitizer suites, together with that file (12 files, `-n 4`): 450 passed and 2 failed. Neither failure comes from this change:
  - `test_rag_diagnostic_privacy.py::test_reranker_parse_failure_reports_shape_not_model_reply` fails at base with RecoveryRequired;
  - the persona boundary test is TASK-39060.
- AC#3: the file is laned on the Architecture guards job, which was never the fast PR contract. It is appended to the step and to `ARCHITECTURE_GUARD_TARGETS`. Timings are below. In the job the file costs 40.6 s, 31.8 s of which is the conftest's periodic GC.

Files: `tldw_chatbook/Utils/log_sanitizer.py`, `Tests/Utils/test_log_sanitizer.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`.

**Architecture guards job timings (shared by TASK-39054, TASK-39055, TASK-39060 and TASK-39061).**
- Before adding anything: CI ran the step in 402 s (job 7 m 38 s) on #3436's run 42218508176, 2026-11-06.
- The step's command, run with `bash $FU/jobstep.sh fu-r11-arch "Run Architecture guard contracts"` (serial, `--timeout=180 --tb=short`). Each run of the final 36 targets gave 383 passed:

| Command | Targets | Nodes | pytest time | Load |
|---|---|---|---|---|
| Base 2684e5453f | 29 | 160 | 153.9 s | 7-11 |
| Base 2684e5453f | 29 | 160 | 457.0 s | 27-45 |
| Base 2684e5453f | 29 | 160 | 283.9 s | 32-38 |
| This branch, run 1 | 36 | 383 | 563.4 s | 25-48 |
| This branch, run 2 | 36 | 383 | 451.2 s | 24-48 |
| This branch, run 3 | 36 | 383 | 440.6 s | 24-57 |
| This branch, right after the 283.9 s base run | 36 | 383 | 413.8 s | 32-48 |

- Six agents shared the machine, and the load swung from 24 to 57, so the totals mostly measure the load.
- One more run, with `--durations=0` (862.4 s at load 50-85), attributes setup, call and teardown time per file. The added files cost 146.9 s of its 858.3 s:
  - `test_no_undefined_module_globals.py`: 72.5 s for 3 nodes, with no periodic-GC teardown;
  - `test_log_sanitizer.py`: 40.6 s for 168 nodes, of which 31.8 s is the conftest's every-25th-test `gc.collect()`;
  - the four ChatScreen files: 33.5 s for 51 nodes, 8.9 s of it GC;
  - the two boundary nodes: 0.3 s.
- An extra job run overlapped my temporary source reverts for a boot-census bisect (12 failed). I discarded it.
