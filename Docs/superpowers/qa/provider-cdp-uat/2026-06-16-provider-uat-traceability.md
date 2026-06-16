# Provider UAT Traceability Repair - 2026-06-16

## Purpose

This document repairs provider UAT traceability after the current `dev` branch was found to contain `TASK-84` Done notes that reference provider sweep evidence files which are missing from current dev.

The missing historical files are:

- `Docs/superpowers/qa/provider-cdp-uat/2026-05-31-provider-cdp-uat.md`
- `Docs/superpowers/qa/provider-cdp-uat/provider-sweep-results.json`

Those historical results are not reconstructed here. The original full sweep table and JSON cannot be recovered from the merged branch without an external artifact source, so this report records the current verified evidence boundary and the residual gap.

## Current Verified Evidence

PR #527 merged provider/Console UAT fixes into `dev`.

Verified before PR #527 was merged:

- Anthropic Console send completed through the rendered Textual-web app using the provider settings flow.
- The selected model was `claude-haiku-4-5-20251001`.
- The assistant response rendered visibly in Console.
- The focused provider regression suite passed with 125 tests.
- Post-rebase critical checks passed.
- `git diff --check` passed.

Transient screenshot evidence from that run was captured outside the repo at:

- `/private/tmp/console-uat-anthropic-ui-response-2.png`

Because that screenshot is outside the repository, it is not counted as durable repo-tracked evidence.

## Residual Gap

`TASK-84` claimed a full provider CDP UAT sweep attempted 11 hosted providers, with 7 passing and 4 classified as external/provider failures or timeouts. The current branch does not contain the referenced report or JSON evidence, so that exact historical claim remains unverifiable from repo contents alone.

Required follow-up if full sweep evidence is needed again:

1. Re-run the provider CDP UAT sweep from the current runtime inventory.
2. Save the durable report and redacted JSON under `Docs/superpowers/qa/provider-cdp-uat/`.
3. Ensure raw API keys are not present in markdown, JSON, terminal logs, or screenshots.
4. Verify each report path referenced by Backlog task notes exists in the repository.

## Acceptance Decision

This file does not re-approve or replace the missing `TASK-84` full sweep. It only records the traceability defect, the currently available provider evidence, and the boundary for future QA.
