---
id: TASK-18810
title: Console modal inventory test dies before its own assertions
status: To Do
assignee: ['@Robert']
created_date: '2026-08-18'
labels: [console, testing, modals]
dependencies: []
priority: medium
---

## Description (the why)

`Tests/UI/test_console_modal_dismissal.py::test_console_modal_inventory_matches_runtime_ast_and_transitive_launches` fails on dev at its FIRST assertion — an undeclared `WorkspaceCreateModal` launch (which itself launches an undeclared `SelectDirectory`, so the drift is two levels deep). Everything after that line is dead-lettered: the modal-count bump and the `reachable_modal_types == all_contract_types` set comparison never execute, so the inventory contract silently stops guarding new modals even while the file looks maintained. Found during task-18515's whole-branch review, which had to re-run the walk with assertions disabled to confirm its own edits were correct.

## Acceptance Criteria (the what)

- [ ] `WorkspaceCreateModal` and its transitive `SelectDirectory` launch are declared (or explicitly excluded with a recorded reason), so the test reaches its later assertions
- [ ] The whole test passes on dev, with the count and set-equality assertions actually executing
- [ ] A guard exists against the same silent-skip class: the assertions that matter are ordered or structured so an early failure cannot mask them (e.g. collect all mismatches and assert once, or split into independent tests)
