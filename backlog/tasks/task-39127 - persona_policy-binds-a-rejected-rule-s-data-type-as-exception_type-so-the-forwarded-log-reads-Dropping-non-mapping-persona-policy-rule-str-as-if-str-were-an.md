---
id: TASK-39127
title: "persona_policy binds a rejected rule's data type as exception_type, so the forwarded log reads \"Dropping non-mapping persona policy rule: str\" as if str were an error class"
status: To Do
created_date: 2026-11-07 02:03
labels:
- follow-up
- logging
- needs-owner-decision
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
- [ ] #1 The owner decides which field, if any, carries a non-mapping rule's type under the strict boundary; the decision is recorded
- [ ] #2 The forwarded line no longer presents a data type as an exception class; a test through the real forwarder is red before the change
- [ ] #3 The bind-site census, the diagnostic inventory pin and `Tests/test_persistent_diagnostic_boundary.py` stay green, and the production diagnostic inventory is regenerated after its drift is read
<!-- AC:END -->
