# Reviewed MCP catalog and guidance — TASK-32840

## Current merged-dev integration

PR2727 is merged. This saved follow-up is rebased onto the verified merge
`ebee42fab8`; [current evidence and exact conflict choices](CURRENT-DEV-REVIEW.md)
cover 168 distinct targeted passes, seven guards and eight fresh native captures.
The [current gallery](GALLERY.md) is ready for this PR's own visual review.
Current-head CI/review and owner approval remain merge gates.

## Historical original draft qualification

The material below describes saved head `dcdeb2e295` before integration.
Its stacked-base and pending-parent statements are historical. The current
runner reuses shared CLI validation and the app's supported image warm-up.

After accepting a restored-root review, Servers stayed empty because the
completion path cleared historical projections without reading the newly
approved definitions. It now reads the existing passive local catalog once,
retains built-in readiness, and publishes only while the original review view
still owns the result. It does not reload discovery, connect or grant tools.
If this read fails, the warning preserves the successful approval outcome and
offers the existing `r` reload action. Guidance now states that historical rules
and grants remain inactive; fresh Ask/local defaults are separate.

ADR required: no. Existing ADR-126 governs restored authority; ADR-150/161 govern
UI composition. This routine caller/copy repair adds no storage, ownership,
service contract, token or CSS change.

## Verification

[49 distinct targeted cases pass](qualified-cases.json): six new real-owner
catalog cases, 18 parent publication cases, 17 mounted recovery controls,
three owner approval/lifetime checks and five adjacent synchronization/inspector
checks. The new cases cover current success, read failure, mode navigation,
mode and screen round trips, and service replacement during the passive read.
The existing current-success assertion now expects fresh definitions while
still rejecting historical discovery/tools. No full suite ran.

The [two baseline failures](tests/red-001.txt) reproduced the missing passive
catalog read after real owner approval. [The repair passes all 41 recovery checks](tests/green-001.txt).
[Eight adjacent checks](tests/adjacent-001.txt) also pass. The only production
source change after green-001 is a comment documenting the deliberately broad
post-approval error boundary; the final native run hashes the final source.
Two pytest warnings concern unrelated old temporary-directory cleanup.

All [seven preflight guards](preflight.txt) pass. [Static analysis](static-analysis.json)
introduces no diagnostics; unchanged baseline diagnostics remain in the two
production modules, while the new test and runner are clean. [Changed ranges
and test files are formatted](formatting.txt). [Independent review](independent-review.txt)
found no concrete blockers. [The one new diagnostic](diagnostic-review.txt)
uses the existing bounded secret/path redactor; the reviewed inventory pin was
updated. No logging sink was added.

## Native visual evidence

The real TldwCli ran with LinuxDriver and TTY streams in a fresh private
HOME/USERPROFILE/XDG/TLDW profile. A repository fixture creates and selects an
actual restore; the normal app mounts its real service and approval writer.
Only disposable-fixture onboarding and startup model refresh are disabled.
Native MCP client connection, discovery and execution are guarded against use,
and the network guard records no attempts.

[Eight rendered and inspected captures](GALLERY.md) cover dark/light themes
at 120×40 and 170×48. Each cell cancels, reopens and confirms review through
visible keyboard controls, checks fresh Ask/local defaults and retained
historical bytes, then selects the repopulated server row. The first cell
activates the roots; later cells repeat review of the same unchanged generation.
The server remains disconnected, historical rules/approval requests are absent
from fresh stores, and only MCP owners are activated. Other restored owners,
including config and skills, remain unapproved.

[The native receipt](native/result.json) and [lifecycle check](lifecycle.json)
record normal App.run return, exit 0, absent process, released instance lock,
ten healthy private SQLite databases, zero conversations/messages and unchanged
user default files. Twelve source hashes and the runner hash match final source.
The restored fixture logs one unrelated Evals `storage_scope_not_enrolled`
diagnostic; that owner's recovery is not qualified here. No other app errors
were recorded; the faulthandler log is empty.

[Native attempt 001](earlier-native/001-result.json) reached the repopulated
server but the harness assumed a disconnected client object existed; this
profile correctly had no client. Its qualification failed and the app exited
normally. The corrected harness accepts either no client or no sessions.
[Attempt 002](earlier-native/002-result.json) passed all four cells, but preceded
the lint comment; final-source attempt 003 supplies the published captures.
[Export hashes](export-manifest.json) distinguish originals and normalized copies.

## Bounds and continuation

This branch stacks on PR #2727 at `d6c0312bedff15d8c2e8e9589a70c5be7c60379b`
because it relies on that PR's completion ownership guards. Its draft PR targets
`codex/mcp-restored-roots-review`; retarget to dev after the parent merges.
It closes the parent's deferred catalog repopulation and guidance items.
Remaining review includes bulk permission actions, approval workflows and
connected-runtime journeys. The 80×24 layout remains outside this qualification.

Current-head CI and final visual approval remain required before merging.
[Allocation](allocation-census.json), [fresh reference census](precommit-census.json)
and [sole ownership](allocation-owner-check.json) cover TASK-32840.
