# Approved Audit filters — rebased dev and Qodo closeout

Owner visual approval applies to this filter slice. The branch was rebased without
conflicts onto dev `ea2d7b22a8af93e4a4856d350640f03af5cc4880`. Incoming dev changes
have no paths in common with the PR; startup and MCP support changes warranted
fresh qualification. No new ADR: existing ADR-150/161 govern the layout, and these
review fixes concern callback documentation, fixture consistency and QA publication.

## Review fixes

- Both public focus/resize callbacks document their Textual event parameters.
- `AUDIT_FIXTURE_COUNT` supplies fixture generation and unfiltered counts;
  filtered counts and the final row index derive from it. The value remains 48.
- `Docs/superpowers/qa/export_receipts.py` exports selected UTF-8 receipts with
  repository, home, temporary and pytest-owner paths normalized. Raw evidence is
  retained privately. Each export manifest records original and published hashes.
  The exporter refuses existing/symlink destinations and reads every input before
  creating a new directory, preventing partial exports and raw-evidence overwrite.
- All 25 affected historical/current QA receipts added by this PR are normalized,
  including PR2720's closeout. Existing export hashes were recomputed; original
  hashes remain unchanged. [Normalization manifest](publication-normalization.json)
  records before/after hashes. Future exports must use the same exporter.

Independent review found an initial output-alias bug in the exporter. Its new
regression failed before the exclusive-directory fix and passes afterward.
[Final independent review](independent-review.txt) has no remaining findings.

## Targeted evidence

**381 distinct current-source cases pass:**

- [115 Audit/layout/identity/selection/token and exporter cases](tests/2721-final-integration.txt).
- [133 runner-admission cases](tests/2721-export-green.txt), from a 136-case run
  that also included three exporter cases already counted above.
- [132 incoming MCP activation/store cases](tests/2721-incoming-mcp.txt).
- [Six final exporter cases](tests/2721-export-final3.txt), five already counted
  in the integration run, plus the additional output-alias refusal regression.

The incoming MCP run has six failures. All six reproduce in an archive of
**untouched dev `ea2d7b22a8`**, with PYTHONPATH and subprocesses pinned to that
archive: [baseline output](tests/2721-incoming-baseline.txt). They concern inactive
inspection, the removed external-server secret action, approved remote inspection
and a retained-await timeout. This PR does not modify those runtime/test paths.
The earlier dimension-governance and size-budget debt remains documented in the
[parent qualification](../README.md); no allowance was weakened. No full suite or
all-green repository claim is made.

All [nine artifact preflight guards](tests/2721-final-preflight.txt) pass. New
exporter/tests and the runner pass Ruff lint/format; the changed callback range
passes formatting. Production lint retains the same two baseline findings.
[Red tests](tests/2721-export-red.txt), [pytest-owner red](tests/2721-pytest-owner-red.txt)
and [alias red](tests/2721-alias-red.txt) are retained before their respective fixes.

## Native equivalence and lifecycle

All four native dark/light 80×24/170×48 journeys pass with the real app, service
JSONL records, TTY, keyboard controls and blocked network. All 18 terminal bodies
and SVG bodies exactly match the approved captures after timestamp normalization
and resolving generated style IDs. Nine whole frames also match; the other nine
differ only in the top three navigation rows' horizontal scroll position and
associated generated style numbering. No Audit layout/content/focus difference
remains. Representative compact/wide frames in both themes were rendered and
inspected. [Comparison](capture-comparison.json) and [all captures](GALLERY.md).

[Lifecycle](lifecycle/lifecycle.json) confirms normal app return, exit 0, released
lock, absent process, ten healthy private databases, zero conversations/messages,
unchanged defaults and sentinels, matching source/runner hashes, and zero network
attempts/errors. [Source verification](source-verification.json) matches the final
implementation. [Launch](lifecycle/launch.json) records the rebased source and dirty
review edits at capture. The terminal was closed afterward. An earlier system-
Python preparation failure never launched the app; [retry note](launch-retry.txt).

Fresh current-head CI, accumulated Qodo review and final dev/tree verification
remain required before the already-approved slice merges. TASK-32835 stays In
Progress until merge confirmation; PR2722 inspector guidance is separate.
