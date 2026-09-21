# MCP Audit selection — TASK-32834

The current integration is qualified on merged PR2770 dev. See the
[2026-09-21 qualification](current-dev/README.md): 305 targeted cases, nine
artifact guards and twenty native captures. The owner has now approved PR2720; the [latest-dev rebase replay](approved-rebase/README.md) preserves that approved behavior. Current-head CI/review and final dev review remain merge gates.

## Historical qualification — 2026-09-18

Filtering away or evicting a selected execution now clears its detail and drill
actions. A newest-first refresh preserves the same uniquely identifiable event
and cursor. Retired row keys and delayed selection messages cannot redirect a
gesture to a different execution at the old index.

Same-snapshot filters preserve a selected row even when two records have equal
metadata. After refresh, equal duplicate records are ambiguous because the log
has no event ID; the inspector clears until the user chooses a row again. No log
schema, runtime authority, token, stylesheet or budget changed.

Base: dev `cef6bd2a3e3f0b8de0e166146acb4900ca7ea2b6`. This is an independent
follow-up to merged PR #2707. It does not incorporate the inspector scrolling
repair in PR #2718. Nothing here authorizes a merge.

## Evidence

- **102 targeted cases pass:** 10 new selection regressions, 64 existing Audit
  cases, two MCP table end-to-end cases and 26 design/component governance cases.
  See `tests/final.json` and `tests/final.txt`. Five legacy CSS-source pin tests
  were deselected; four are previously recorded token-literal failures in
  TASK-32796. No full suite was run.
- Before the repair, all three initial regressions failed at the intended
  assertions: stale filtered detail, wrong cursor after prepend, and stale
  detail after eviction. Independent review found duplicate-record filtering;
  its added regression also failed before repair. Red receipts are retained.
- **18 native captures inspected:** real TldwCli, LinuxDriver, TTY rendering and
  isolated HOME/config/data. Dark/light at 80×24 and 170×48 cover selected,
  refreshed, filtered and combined-filter states. Both wide cells also activate
  Open tool and verify the exact `local:audit-beta::review_echo` table selection
  and inspector while `audit-alpha` exposes the same tool name.
- The real JSONL execution log contains synthetic metadata fixtures. Two real
  local stdio catalog connections support drilldown; there are zero tools/call
  requests. This is not an execution-runtime qualification.
- Exit 0, app and both fixture processes absent, instance lock released, ten
  private databases healthy, zero chats/messages, unchanged default-file
  fingerprints and matching captured source hashes: `lifecycle.json`.
- All seven derived-artifact preflight guards pass (`preflight.txt`).
- New test and QA Python files pass Ruff lint/format; production lint diagnostics
  match dev (2 Audit, 65 workbench), with zero added diagnostics. Changed Audit
  regions were formatted. Independent review has no remaining concrete blocker.

Reproduction uses `native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION`; its
profile validator requires every configured data/database path inside that fresh
profile before application imports. `export-manifest.json` hashes retained SVG,
terminal text and native results, recording both original and retained hashes.
Retained SVG/text/test logs normalize trailing line whitespace only. [Gallery](GALLERY.md).

## Remaining visual scope

At 80×24 the existing Audit filter slots squeeze the text field until its value
is unreadable, and the initiator prompt is clipped. The captures retain this
defect; compact filtering was checked through focused controls and actual filter
events, not qualified as a readable keyboard workflow. Repair this toolbar next.
Compact inspector action reachability requires the separate PR #2718 work.
The pre-existing built-in readiness guidance also remains visible above local
execution/tool detail; inspector guidance ownership needs further review.

ADR required: no. Existing ADR-170 governs quiet redraw and selection continuity;
ADR-150/161 retain visual governance. This repairs the existing Audit interaction
without changing storage or runtime boundaries.
