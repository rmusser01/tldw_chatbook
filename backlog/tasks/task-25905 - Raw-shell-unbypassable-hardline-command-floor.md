---
id: TASK-25905
title: 'Raw shell: unbypassable hardline command floor'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:09'
updated_date: '2026-09-01 18:02'
labels:
  - security
  - tools
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The raw-shell approval card is the only thing standing between the agent and any command, and once a session grant is given nothing re-checks. Verified on origin/dev: Tools/raw_cli_executor.py:144 validate_raw_cli_request checks caller identity, shell name, size, timeout and cwd but never inspects what the command does; Agents/raw_shell_tool_provider.py:291 gates on permission state alone; and once approve_session is granted (raw_shell_tool_provider.py:48) subsequent commands run unreviewed for the rest of the Console session. A named grep for rm -rf, hardline, mkfs and fork bomb across tldw_chatbook returns no guard hits. Hermes runs an unbypassable hardline list plus a sudo-stdin guard before even its own yolo mode. This is a floor, not a replacement for the approval card.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A small set of catastrophic command shapes (root recursive delete, mkfs, dd to a block device, fork bomb, shutdown) is refused outright, before any permission state or session grant is consulted
- [x] #2 The floor also applies to commands issued under an active approve_session grant - verified by a test that grants a session then attempts a hardline command
- [x] #3 Detection is resistant to trivial obfuscation (quoting, variable indirection, whitespace padding) - adversarial cases are in the tests
- [x] #4 A refusal states plainly which rule fired and is distinguishable from a user denial in the model-facing result
- [x] #5 The floor is not user-configurable off; any user-supplied deny list is additive to it
- [x] #6 False-positive safety: a corpus of ordinary developer commands (git, npm, pytest, rm of a project file) is asserted to pass unaffected
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: 27 hardline cases (incl. adversarial quoting/vars/whitespace) + 17 ordinary-command corpus + boundary test + session-grant provider test\n2. Pure hardline_violation + 5 rule regexes + obfuscation normalizer in raw_cli_executor (module constants, no config)\n3. Enforce inside validate_raw_cli_request (both callers, before all permission state); RawCliHardlineViolation typed error names the rule\n4. Provider invoke catches it distinctly -> blocked copy 'safety floor, not a user denial, cannot be approved'
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
hardline_violation() in Tools/raw_cli_executor.py: 5 rules (recursive-root-delete matched by ARGUMENT SHAPE anchored to command position — so $VAR -rf / trips even with the command word hidden; filesystem-format; dd-to-block-device; fork-bomb via the self-referencing function pattern incl. named variants; system-shutdown in command position only, so 'git commit -m ..shutdown..' passes). Detection runs on both the raw text and an obfuscation-normalized copy (quotes/backslashes stripped, whitespace collapsed) — the AC#3 adversarial corpus (split-quoted flags, $DELETER indirection, padding, sudo/chained forms) is 27 pinned cases; AC#6's 17-command ordinary developer corpus (incl. rm -rf node_modules, dd file-to-file, grep -rf, tar halt.tar) pinned passing. Enforced INSIDE validate_raw_cli_request after command normalization — before any permission state, for BOTH callers — via typed RawCliHardlineViolation carrying the rule name (AC#4); the provider's invoke catches it distinctly and returns a blocked result stating 'built-in safety floor, not a user denial; cannot be approved or overridden'. AC#2 pinned: with an ACTIVE approve_session grant the ordinary command executes and rm -rf / never reaches the runtime. AC#5 structural: the rule tuple is a module constant, zero config reads. pending_gate_for's blanket except means hardline commands never even raise an approval card (nothing to approve). Known limits (documented in-code): a determined adversary can evade a static floor (base64/eval); the approval card remains the real gate. 46 new tests; raw-shell suites at the exact 3-name baseline.
<!-- SECTION:NOTES:END -->
