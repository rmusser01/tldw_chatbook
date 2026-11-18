---
id: TASK-41319
title: "Console Send preparation still decides skill and configuration authority from function and code identity, against ADR-356 rule 4"
status: To Do
created_date: 2026-11-16 19:55
labels:
- follow-up
- architecture
- console
- agents
priority: medium
references:
- backlog/decisions/356-native-phase-orchestration-runtime-ownership.md
- tldw_chatbook/Chat/console_configuration_preparation.py
- tldw_chatbook/app_skill_wiring.py
- tldw_chatbook/Agents/activation.py
- Tests/Architecture/test_code_identity_ratchet.py
- https://github.com/rmusser01/tldw_chatbook/pull/3539
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the 2026-11-13..16 burn-down. TASK-40719 (PR #3539) recorded this as out of its scope. Checked on dev 5e3c8b2bd4 (2026-11-16) by reading the code; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** ADR-356 rule 4: "Orchestration must not inspect function objects, code/defaults/closures, descriptors, caller stacks/locals or coroutine implementation to establish authority." TASK-40719 removed three such checks, and its whole-package ratchet (`Tests/Architecture/test_code_identity_ratchet.py`) caps the rest without removing them. The Send preparation path still has several. If any checked function is wrapped, reloaded or patched, skill trust or the turn's configuration capture is silently refused or not recorded. The user's Send then fails or loses its skills for no reason they can see.

**Remaining sites on the Send path.**
- `Chat/console_configuration_preparation.py`:
  - `_resident_skill_wiring` (`:447-486`) accepts the wiring only if each of `_capture_console_skill_trust_source`, `_console_skill_metadata_current` and `_console_skill_source_current` has its recorded `__code__`, `__globals__`, no `__defaults__`/`__kwdefaults__`/`__closure__`, and its recorded `__wrapped__` (`:473-479`).
  - `_function_unchanged` guards (imported as `callable_source_current` from `Agents/activation.py`, via `Chat/console_ordinary_tree.py:10`) at `:655`, `:891`, `:1024`, `:1134` and `:2145`.
  - Further `__code__` comparisons at `:187`, `:206`, `:210` and `:364`.
- `app_skill_wiring.py`:
  - `_console_skill_metadata_current` (`:73`) checks the identity of `compact_model_bar`'s module, spec and bindings, and `:124-128` checks function code, globals, defaults and closure.
  - `_capture_console_skill_trust_source` (`:175`) requires the app factory's recorded `__code__`, `__globals__`, no defaults and no closure (`:198-208`).
  - The lazy trust factory is checked by code, globals, defaults, free vars and closure cell (`:329-339`), and functions by `__code__`/`__globals__` (`:399-401`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A Send whose skill-trust, skill-metadata or configuration-capture collaborators are wrapped (as a decorator or a test patch would) captures and records the same skill trust and configuration as the stock path; tests show this and are red on dev
- [ ] #2 Each listed check is replaced by a decision from runtime-owned registration and current state, or kept with a written ADR-356 exception naming the attack it blocks
- [ ] #3 The code-identity ratchet is lowered to the new counts and still fails on any increase
<!-- AC:END -->
