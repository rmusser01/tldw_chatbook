---
id: TASK-26005
title: 'Tool calls: coerce malformed model arguments before dispatch'
status: To Do
assignee: []
created_date: '2026-08-31 15:44'
labels:
  - agents
  - tools
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A model that emits a stringified JSON array where an array is expected fails the call outright. Verified on origin/dev: Agents/agent_runtime.py:1149 does json.loads(call.arguments) straight into the provider path with no repair layer, so a common small-model failure mode costs a whole turn. Hermes runs coerce_tool_args, which is schema-aware and recursive: it converts JSON-string values to native arrays and objects where the schema calls for them and sanitizes fenced output.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A tool argument arriving as a JSON string where the schema declares an array or object is coerced to the native type before dispatch
- [ ] #2 Coercion is schema-driven: a string is left alone where the schema declares a string
- [ ] #3 Coercion is recursive through nested objects and arrays
- [ ] #4 A value that cannot be coerced produces the existing validation error, not a silent wrong-type dispatch
- [ ] #5 Every coercion is recorded in the run log so a systematically malformed model is visible rather than masked
- [ ] #6 Tests include the adversarial cases: double-encoded JSON, fenced JSON, and a legitimately string-typed field containing bracket characters
<!-- AC:END -->
