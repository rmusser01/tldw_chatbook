---
id: TASK-39127
title: "persona_policy binds a rejected rule's data type as exception_type, so the forwarded log reads \"Dropping non-mapping persona policy rule: str\" as if str were an error class"
status: Done
assignee: ['@claude']
created_date: 2026-11-07 02:03
updated_date: 2026-11-07 06:09
labels:
- follow-up
- logging
priority: low
references:
- tldw_chatbook/Agents/persona_policy.py
- tldw_chatbook/Logging_Config.py
- Tests/Persona_Buddy/test_controller_failure_logging.py
- Tests/test_persistent_diagnostic_boundary.py
- Tests/Architecture/test_persistent_diagnostic_inventory.py
dependencies:
- TASK-39060
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07, and on round-11 branch fu/r11-arch, which lands TASK-39060.

**The ruling.** Owner decision (2026-11-06), quoted from TASK-39060's notes: "Bind class, drop count. All 10 sites log a constant message with logger.bind(exception_type=…) — the forwarder already renders it, so no detail is lost for the class name. Drop rule_count and its test assertion. The boundary stays strict: no runtime values in the message."

**What it produces at one site.**
- On fu/r11-arch, `parse_persona_policy_from_rules` (`Agents/persona_policy.py`) logs a non-mapping rule with `logger.bind(exception_type=type(entry).__name__).warning("Dropping non-mapping persona policy rule")`.
- The forwarder appends a bound `exception_type` as `f"{record['message']}: {exception_type}"` (dev `Logging_Config.py:660-676`). So a string rule logs "Dropping non-mapping persona policy rule: str", which reads as if `str` were an error class.
- Dev labels the same value honestly as `entry_type=` (`persona_policy.py:62-65`).
- It is the only one of the 10 sites that binds something other than an exception class. The branch pins it in its bind-site census (`Tests/Persona_Buddy/test_controller_failure_logging.py`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The owner decides which field, if any, carries a non-mapping rule's type under the strict boundary; the decision is recorded
- [x] #2 The forwarded line no longer presents a data type as an exception class; a test through the real forwarder is red before the change
- [x] #3 The bind-site census, the diagnostic inventory pin and `Tests/test_persistent_diagnostic_boundary.py` stay green, and the production diagnostic inventory is regenerated after its drift is read
<!-- AC:END -->

## Implementation Plan

1. Record the owner decision (2026-11-06): drop the type; log the constant "Dropping a non-mapping policy rule" with no bound value.
2. Add a test through the real loguru -> standard-logging forwarder that a non-mapping rule logs exactly that line (red first: "...: str").
3. Change `parse_persona_policy_from_rules`'s non-mapping branch to a plain constant warning.
4. Update the bind-site tripwire row count, the constant-prefix pin in `Tests/test_persistent_diagnostic_boundary.py`, the diagnostic inventory pin, and regenerate `Docs/security/production-diagnostic-inventory.json` after reading its drift.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Owner decision (2026-11-06), AC#1:** "Drop the type: log a constant message ('Dropping a non-mapping policy rule') with no bound type — the boundary stays strict and nothing pretends to be an exception."

**Approach.** `parse_persona_policy_from_rules` (`Agents/persona_policy.py`) now logs `logger.warning("Dropping a non-mapping policy rule")` for a non-mapping entry, binding nothing. The malformed-rule branch still binds `type(exc).__name__`, which is an exception class, so the function stays a reviewed bind site.

**Red first (AC#2).** `test_non_mapping_policy_rule_forwards_a_constant_line_without_a_type` (`Tests/test_persistent_diagnostic_boundary.py`) sends two non-mapping rules (a string, an int) through the real `_forward_loguru_to_standard` forwarder. Before the change it failed with `'Dropping non-mapping persona policy rule: str' != 'Dropping a non-mapping policy rule'`; green after.

**Pins (AC#3).**
- Bind-site tripwire (`Tests/Persona_Buddy/test_controller_failure_logging.py`): the `persona_policy` keyword row goes from 2 to 1 occurrence. The bind-site census keeps the function (its malformed branch).
- Constant-prefix pin (`Tests/test_persistent_diagnostic_boundary.py`) and diagnostic inventory pin (`Tests/Architecture/test_persistent_diagnostic_inventory.py`, now `"Dropping a non-mapping policy rule": ()`, no runtime values).
- Production diagnostic inventory: the drift was the one `persona_policy.py` row (2 calls, digest changed). `--statements` showed exactly one removed statement (the bound warning) and one added (`logger.warning("Dropping a non-mapping policy rule")`), a constant with no user content, secret, path or URL. Regenerated with `--write` after that review.

**Covering run.** The three files above serially: 94 passed, 1 skipped (pre-existing skip on unfetchable TASK-15743 commits) in 354 s at load 25-44. `Tests/test_persistent_diagnostic_boundary.py` and `Tests/Persona_Buddy/test_controller_failure_logging.py` (where the new test and the tripwire row live) run on the architecture-guards "Run Architecture guard contracts" step; `Tests/Architecture/test_persistent_diagnostic_inventory.py` has one node on a PR lane, and preflight runs the inventory check. No lane change.

**Files.** `tldw_chatbook/Agents/persona_policy.py`, `Tests/test_persistent_diagnostic_boundary.py`, `Tests/Persona_Buddy/test_controller_failure_logging.py`, `Tests/Architecture/test_persistent_diagnostic_inventory.py`, `Docs/security/production-diagnostic-inventory.json`.
<!-- SECTION:NOTES:END -->
