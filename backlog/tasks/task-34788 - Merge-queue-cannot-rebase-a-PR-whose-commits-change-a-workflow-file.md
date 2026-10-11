---
id: TASK-34788
title: Merge queue cannot rebase a PR whose commits change a workflow file
status: To Do
created_date: 2026-10-11 01:44
labels:
- ci
- merge-queue
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
GitHub refuses the queue's rebase of any PR whose own commits change a file under .github/workflows: the built-in token can never hold the workflows permission. Seen live on #3058 (2026-10-10): 'refusing to allow a GitHub App to create or update workflow .github/workflows/derived-artifacts.yml without workflows permission'. The queue treated it as a rebase strike and evicted the PR ten minutes later with a generic 'rebase keeps failing'. PRs here often add tests to that file's lane list, so this recurs, and the author gets no hint that only a hand rebase can help.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A PR the queue cannot rebase because it changes a workflow file is told so in one comment that names the cause and the fix (rebase by hand, then re-arm)
- [ ] #2 Such a PR does not wait out the rebase strike gap before it leaves the line
- [ ] #3 A PR that changes a workflow file and is already up to date with dev still lands through the queue without a rebase
- [ ] #4 CLAUDE.md and AGENTS.md tell agents how to land a workflow-changing PR while the queue is on
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
