---
id: TASK-34788
title: Merge queue cannot rebase a PR whose commits change a workflow file
status: Done
created_date: 2026-10-11 01:44
labels:
- ci
- merge-queue
priority: high
assignee:
- '@claude'
updated_date: 2026-10-11 04:08
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
GitHub refuses the queue's rebase of any PR whose own commits change a file under .github/workflows: the built-in token can never hold the workflows permission. Seen live on #3058 (2026-10-10): 'refusing to allow a GitHub App to create or update workflow .github/workflows/derived-artifacts.yml without workflows permission'. The queue treated it as a rebase strike and evicted the PR ten minutes later with a generic 'rebase keeps failing'. PRs here often add tests to that file's lane list, so this recurs, and the author gets no hint that only a hand rebase can help.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A PR the queue cannot rebase because it changes a workflow file is told so in one comment that names the cause and the fix (rebase by hand, then re-arm)
- [x] #2 Such a PR does not wait out the rebase strike gap before it leaves the line
- [x] #3 A PR that changes a workflow file and is already up to date with dev still lands through the queue without a rebase
- [x] #4 CLAUDE.md and AGENTS.md tell agents how to land a workflow-changing PR while the queue is on
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. In _rebase, recognise GitHub's workflow-permission refusal and evict at once with the file and the fix.\n2. Pin: one comment, no warning first, the line moves on; an up-to-date workflow PR waits for auto-merge.\n3. CLAUDE.md, AGENTS.md, spec and ADR-218.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The queue now recognises GitHub's refusal ('... to create or update workflow <file> without workflows permission') in _rebase and evicts at once with evict-workflow-file, naming the file and the fix (rebase by hand, push with --force-with-lease, re-arm). It no longer posts a rebase-failed warning and waits ten minutes. An up-to-date workflow-changing PR needs nothing from the queue's token and lands through auto-merge like any other. CLAUDE.md and AGENTS.md tell agents to rebase such a PR themselves and arm it when the line is short.

Known limit, by design: the built-in token can never hold the workflows permission and the owner allows no PAT or App key, so on a busy line a workflow-changing PR can be evicted again when dev moves before it lands. Tests: Tests/CI/test_merge_queue_actions.py (test_a_pr_that_changes_a_workflow_file_is_told_why_and_leaves_at_once, test_an_up_to_date_workflow_pr_lands_without_a_rebase); mutation-checked. Files: scripts/merge_queue.py, the spec, ADR-218, CLAUDE.md, AGENTS.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
