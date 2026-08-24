---
id: TASK-21501
title: >-
  Three documents and the code disagree about the private MCP tool surface
status: In Progress
assignee: []
labels:
  - testing
  - test-integrity
  - documentation
  - mcp
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The contract test that exists to stop the MCP documentation drifting from the code had itself
drifted, and while it was red nobody could see what it was hiding.

The set of Library tools withheld from the standalone server is declared once in the code, in
a descriptor table the local capability manifest is generated from — the manifest builder's
own docstring says the table is never to be hand-maintained. Three documents restate that set
for readers, and the contract test restates it a fourth time in order to check them.

Every restatement had fallen behind, and by different amounts, so the three documents and the
test disagreed with the code and with each other. The failure the suite reported was the one
document that disagreed with the test; the two that agreed with the test were equally wrong
and passed. Most of the reported failures were the contract's own self-checks, which cannot
demonstrate that mutations are rejected while the text they mutate is already non-conforming.

Worth stating separately, because it is the part that matters to a user rather than to the
suite: the tool missing from all three documents is the only **write** tool in the set. The
page that enumerates what the private surface contains did not mention it at all.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The inventory the contract checks against is obtained from the code that defines it, not restated
- [ ] #2 Every document the contract governs agrees with the code, and the count each one states matches the list it prints
- [ ] #3 The code's own prose description of the surface is corrected where it is stale
- [ ] #4 Any tool absent from the documents is identified and reported rather than quietly added, and its significance is stated
- [ ] #5 The contract is shown to reject a document that falls behind the code again
- [ ] #6 Prose assertions in this module cannot be switched off by reflowing a paragraph
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Establish what the code actually declares, from the table the manifest is generated from.
2. Compare all four restatements against it and record how each differs.
3. Replace the test's restatement with a derivation.
4. Correct the documents and the code's stale prose from the derived truth.
5. Mutation-prove both the doc-versus-code link and the prose assertions.
<!-- SECTION:PLAN:END -->
