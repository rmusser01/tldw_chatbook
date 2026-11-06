---
id: TASK-39061
title: "redact_log_line does not mask bare sk_live_, sk_test_, rk_live_ or rk_test_ secrets, including one the log forwarder renders as '<message>: sk_live_…'"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-06 12:31
updated_date: 2026-11-06 14:42
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
- [ ] #1 `redact_log_line` masks bare `sk_live_`, `sk_test_`, `rk_live_` and `rk_test_` secrets of realistic length, including the forwarder's `<message>: sk_live_…` line and an `exception_type=sk_live_…` line; a test is red before the change on the six unmasked probe lines
- [ ] #2 Short identifiers that only share a prefix (for example `sk_live_mode`) stay readable, and the existing sanitizer tests stay green, with counts recorded
- [ ] #3 The new test runs on a PR lane (never the fast PR contract), with three timings of that lane's command recorded
<!-- AC:END -->

## Implementation Plan

1. Add a test in `Tests/Utils/test_log_sanitizer.py` for the six unmasked probe lines (through `redact_log_line` and `sanitize_string`), plus negative cases for short or underscore-joined identifiers that only share the prefix. Show it red at base.
2. Add the `sk_`/`rk_` `live`/`test` pattern to the diagnostic path that `sanitize_string` and `redact_log_line` share. Leave `_STANDALONE_CREDENTIALS` alone: `sanitize_trace_credentials_v1` uses it too, and that versioned trace projection must not change.
3. Run the whole sanitizer file and the other sanitizer suites, and record the counts.
4. Lane `Tests/Utils/test_log_sanitizer.py` on the Architecture guards job (append to the step and to `ARCHITECTURE_GUARD_TARGETS`), then time the job command three times.
