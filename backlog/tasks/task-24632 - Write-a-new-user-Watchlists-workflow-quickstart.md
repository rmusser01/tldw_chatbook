---
id: TASK-24632
title: Write a new-user Watchlists workflow quickstart
status: Done
assignee: []
created_date: '2026-08-30 05:48'
updated_date: '2026-08-30 06:09'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give new users one start-to-finish guide for creating feed sources and a Watchlist in Console, following durable operation receipts, generating a briefing, saving an every-24-hours schedule, and verifying the result across Watchlists and Settings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A single beginner page documents prerequisites, the exact Console prompt, approvals, receipt completion, briefing generation, and the every-24-hours schedule.
- [x] #2 The guide shows how to verify the completed workflow in Watchlists and Settings and explains the app-open scheduling limitation.
- [x] #3 The guide includes truthful cost, privacy, and troubleshooting guidance using current product terminology.
- [x] #4 The User Guide index and Watchlists reference page link prominently to the quickstart.
- [x] #5 Documentation links, formatting, and changed-file checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify current labels and reuse the redacted mounted Console capture.
2. Write the focused quickstart and link it from the guide index and Watchlists page.
3. Validate links, formatting, privacy, and changed-file scope.

ADR required: no

ADR path: N/A

Reason: documentation-only guidance for an existing workflow and existing service boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a single beginner-focused feed-to-scheduled-briefing walkthrough. The page covers setup, a copy-paste Console prompt, explicit approval review, durable receipt following, briefing consumption, cross-surface verification, every-24-hours semantics, troubleshooting, cost, and privacy. Linked it from the User Guide index and Watchlists reference. Reused the mounted UAT capture as a self-contained User Guide asset, removed CDN font dependencies, and replaced internal QA-only labels in the copied asset; the original evidence remains unchanged. No ADR was required because this is documentation for existing behavior and boundaries. Files: Docs/User_Guide/watchlists-quickstart.md, Docs/User_Guide/index.md, Docs/User_Guide/watchlists.md, Docs/User_Guide/images/watchlists/console-workflow-complete.svg. Verification: 96 focused documentation/MCP contract tests passed; local-link check passed for all three edited pages; SVG XML, privacy/internal-label scan, visual render, and git diff --check passed.

PR review follow-up: Qodo identified three overstatements in the new quickstart. Corrected Console briefing reads to the bounded, truncation-reporting projection with Watchlists Artifacts as the complete saved view; corrected 86,400-second eligibility to latest activity with never-attempted schedules immediately eligible; and limited approval-card language to mutations initiated through Console tools. The latest-dev rebase also exposed an upstream TASK-23113 collision; applied the repository's older-arrival rule by moving the younger wizard-flake task and its three references to TASK-24652. Verification after the corrections: 96 documentation/MCP tests passed, all edited local links resolved, Backlog Guard checked 2,788 task files green, and git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
New users now have one linked start-to-finish guide for creating feeds and a Watchlist in Console, following receipts, generating and scheduling a briefing, and verifying the durable result.
<!-- SECTION:FINAL_SUMMARY:END -->
