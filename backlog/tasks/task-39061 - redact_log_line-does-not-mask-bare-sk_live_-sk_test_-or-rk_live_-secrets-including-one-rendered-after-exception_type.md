---
id: TASK-39061
title: "redact_log_line does not mask bare sk_live_, sk_test_ or rk_live_ secrets, including one rendered after exception_type="
status: To Do
created_date: 2026-11-06 12:31
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
Found in round 10 of the 2026-10-23 burn-down, by the TASK-38989 review; checked on dev b2c618185f and on e8eeae7db9 (`fu-base-r10`), 2026-11-06. `log_sanitizer.py` is identical in both; line numbers are from e8eeae7db9.

**The gap.** `_STANDALONE_CREDENTIALS` (`tldw_chatbook/Utils/log_sanitizer.py:104` onward) lists the `sk-` key forms (`:105-116`: `sk-proj-`, `sk-ant-…`, `sk-or-v1-`, generic `sk-`). It has no pattern for the underscore forms `sk_live_`, `sk_test_` or `rk_live_`.

**Probe.** A pure `redact_log_line` call through run.sh: synthetic 40-character bodies, no network, scratch profile. Same result at both commits.

| Line | Masked |
|---|---|
| `payment retry failed with sk_live_<40> for job 7` | no |
| `… sk_test_<40> …` | no |
| `… rk_live_<40> …` | no |
| `persona_buddy_preferences_save_failed exception_type=sk_live_<40>` | no |
| `config api_key=sk_live_<40>` | yes, by the label rule |
| `token: sk_live_<40>` | yes, by the label rule |
| control: `… sk-<40> …` | yes |

**Why it matters now.**
- After TASK-38989, the forwarder (`Logging_Config._forward_loguru_to_standard`, the only production loguru sink) renders a bound `exception_type` that matches `[A-Za-z][A-Za-z0-9_]{0,63}` on every logger's line.
- A 48-character `sk_live_…` value matches that rule, and `redact_log_line` would leave it in the file sink and the Logs view.
- TASK-38989 guards the producers: a census of the 7 reviewed bind sites, and a package-wide tripwire. So no current site binds such a value, and the sanitizer is the second line of defence.
- `Tests/Utils/test_log_sanitizer.py` runs on no PR lane.
- TASK-23190 and TASK-19555 (both Done) closed other sanitizer gaps, not these prefixes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `redact_log_line` masks bare `sk_live_`, `sk_test_`, `rk_live_` and `rk_test_` secrets of realistic length, including one that follows `exception_type=`; a test is red before the change on the four unmasked probe lines
- [ ] #2 Short identifiers that only share a prefix (for example `sk_live_mode`) stay readable, and the existing sanitizer tests stay green, with counts recorded
- [ ] #3 The new test runs on a PR lane (never the fast PR contract), with three timings of that lane's command recorded
<!-- AC:END -->
