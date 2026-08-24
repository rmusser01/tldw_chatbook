---
id: TASK-21564
title: >-
  The RAG UI integration harness predates the Library prompt-authority gate
status: To Do
assignee: []
labels: [testing, test-integrity, rag]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Discovered by TASK-21563 while converting `Tests/RAG/test_rag_ui_integration.py` from logged
claims into assertions.

The module drives the chat RAG context path through a hand-built application double. That
double predates the Console Library policy work, which added an authority step between a
search returning candidates and those candidates reaching the prompt. The double does not
satisfy that step, so the pipeline runs, finds its result, and then discards it — the log
line is `RAG candidates excluded; count=1; reason=not_currently_authorized`.

Nothing about this was visible before, because the tests asserted nothing. Two of them now
carry the assertions their log strings always implied, and are marked expected-failure with
this task as the reason rather than deleted, so the coverage they describe stays on the
record instead of disappearing.

The work is to model the authority step in the double — or to decide the seam is better
covered by a test that mounts the real screen, and retire the double.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The two expected-failure tests either pass against the current authority behaviour or are replaced by coverage that does, and the expected-failure marks are removed
- [ ] #2 The double either satisfies the authority step honestly or is retired in favour of a harness that does, with the choice justified
- [ ] #3 The settings-forwarding test still asserts that its search double was actually reached, so it cannot regress to checking nothing
<!-- AC:END -->
