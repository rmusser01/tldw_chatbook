# MCP Tools header alignment — TASK-32868

The intermittent header mismatch retained by PR2726 is reproduced on merged dev
`ad0f76e23b8737904f24eb34760bbee9ac01a04c`. When a catalog refresh paints before
Textual 8.2.8's idle measurement, header labels keep their initial widths while
body cells use the measured widths. The State label can start at column 6 while
its data starts at column 28.

`MCPToolsTable._update_dimensions` clears the table's render caches before the
normal measurement. This invalidates premature header renders while letting
Textual populate fresh auto-height cell caches. It changes no sizing rules,
CSS/tokens, catalog ownership, permissions, focus or selection behavior. The
workaround is local to Tools; other tables remain part of their own reviews.
Existing ADR-150/161/170 apply; no new ADR is required.

## Evidence

- [Four regression failures on merged dev](red-reproduction.json), followed by
  [152 passing targeted cases](targeted-tests.json) ([log](targeted-tests.txt)).
  These cover composed header/body positions, Unicode, tags, same-ID catalog
  replacement, actual filter keystrokes, dark/light and compact/wide transitions,
  horizontal scrolling, empty results, focus/identity preservation, quiet refresh
  and the next real Enter selection. Existing production compact/readability and
  gesture tests remain green. No full suite ran.
- [All seven derived-artifact guards pass](preflight.txt). [Ruff and the design
  detector](static-analysis.json) report no findings for the changed targets;
  new files and changed ranges are formatted.
- [Independent review](independent-review.txt) found no blockers.
- The [before](before/result.json) and [after](after/result.json) journeys run the
  real app with LinuxDriver in native tmux, under separate private profiles.
  Both restrict the real catalog to three existing tool identities with synthetic
  display metadata, render the header before queued measurement, then refresh
  labels/tags across dark/light × 120×40/170×48. All four baseline composed
  headers are misaligned; all four fixed headers align with their body columns.
- [Before](before/lifecycle.json) and [after](after/lifecycle.json) lifecycle checks
  verify normal app return, process exit, lock release, healthy private databases,
  zero conversations/messages, unchanged user defaults and no app error logs.
  Permission profiles and execution logs remain unchanged; no tool executes and
  no network connection occurs. Final app/runner/journey hashes match the source.
- [Visual comparisons](visual-comparison.json) isolate every before/after terminal
  difference to the header row. Final fixed PNGs are pixel-identical to the four
  inspected fixed views from the initial pair. [Gallery](GALLERY.md).
  Exported SVGs normalize whitespace-only lines for git; painted content is unchanged.
  The [fixture note](fixture-note.md) records why direct table rendering was
  replaced with composed-screen evidence.

This is a controlled reproduction of the observed paint ordering, not a claim
about its frequency under every runtime load. Native captures qualify the two
named viewport sizes; narrower 80/100-column full-app checks are automated.
The Textual private measurement/cache API remains a dependency of this local
workaround, guarded by the behavioral regressions.

## Reproduce

Run these targeted files with a fresh `--basetemp` and `--timeout=120`:

```sh
.venv/bin/python -m pytest Tests/UI/test_mcp_tools_header_alignment.py \
  Tests/UI/test_mcp_tools_column_readability.py \
  Tests/UI/test_mcp_table_refresh_selection.py Tests/UI/test_mcp_tools_mode.py \
  Tests/UI/test_design_token_governance.py Tests/Utils/test_mcp_native_qa_args.py
```

For native QA, prepare an
unused canonical private profile beneath `/tmp` with `config.toml` and contained
home/config/data/database paths, then invoke `native_check.py ROOT TMUX_SOCKET
SESSION` inside that existing tmux session. The shared validator rejects invalid
arguments, unsafe profiles and reused output before importing the app. Its CLI
regressions are included in `Tests/Utils/test_mcp_native_qa_args.py`.

## Workstream boundary

[PR2726 merge receipt](pr2726-closeout.json) records current-head green CI,
owner visual approval, clean accumulated review and a merged tree identical to
tested head `f42bba8e58`. This follow-up starts from that merge.
The header fix still requires this PR's own final visual approval, current-head
CI/review and live-dev check before merge. Next is existing PR2727's Permissions
restored-root review, followed by the remaining MCP work. PR2707's heartbeat
stays paused.
