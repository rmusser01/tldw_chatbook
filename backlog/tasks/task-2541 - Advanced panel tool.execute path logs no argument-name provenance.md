---
id: TASK-2541
title: Advanced panel tool.execute path logs no argument-name provenance
status: To Do
assignee: []
created_date: '2026-08-06 09:48'
labels:
  - mcp
  - audit
dependencies: []
priority: low
---

## Description

PR-T3 Task 4 wired real argument NAMES into the execution log for the two paths that
have a schema to source them from: the Test Tool runner and the agent bridge, both of
which resolve a `HubTool.input_schema` before calling `execute_hub_tool()`.

PR-T3 Task 6 routed the Advanced (legacy control plane) panel's `tool.execute` action
through that same gated/logged seam (`execute_advanced_tool()`), closing the ungated
door — but its own docstring says why argument names still don't reach those rows:
"No `registered_argument_names` is supplied: the payload is free-form JSON with no
schema behind it, so this path honestly records no argument provenance rather than
inventing some." The Advanced runner lets an operator type an arbitrary tool name and
arguments as raw JSON; there is no guaranteed `HubTool` behind that name at all.

So of the three execution paths now gated and logged, this is the one still recording
`argument_names: []` / `unknown_argument_count == len(arguments)` regardless of what
was actually supplied — not a lie (the code is honest about not inventing names), but
a coverage gap worth closing where possible.

## Acceptance Criteria

- [ ] When the Advanced panel's `tool.execute` payload names a tool that DOES resolve
      to a known `HubTool` in the catalog, the resulting audit row carries argument
      names the same way a Test Tool run for that tool would.
- [ ] When the payload's tool name does not resolve to any known `HubTool` (or the
      catalog lookup itself fails), the row keeps the existing honest `[]` behavior —
      never invents names from the raw payload's keys.
- [ ] Additive tests cover both the resolved-schema and unresolved-schema cases.
- [ ] No change to the values-never-logged privacy contract (names only, never
      argument values).
