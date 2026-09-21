# Current-dev Audit selection qualification — 2026-09-21

PR2720 resumes the saved execution-selection repair on dev
`ba5aa6e9ec60bb93aec4adc6aae219deb9266b78`, the confirmed PR2770 merge.
That merge tree is identical to the qualified parent `d6c15f9e4a`.
The native launch receipt records the pre-commit working tree; its individual
source and runner hashes identify the exact tested implementation.

Filtering away a selected execution clears its detail and actions. Refresh
retains the same uniquely identifiable execution and cursor. Equal records
retain their row identity within a snapshot, but a refresh must be unambiguous
in both old and new snapshots. Leaving Executions retires row keys and selection,
so a delayed gesture cannot reactivate after a mode or subview round trip.

## Verification

- **305 distinct targeted cases pass:** 15 selection, 111 neighboring Audit and
  navigation, 27 architecture/design/CSS governance, and 152 shared native
  admission/fixture/cleanup cases. Final logs are retained alongside this file.
  Four pre-existing CSS literal assertions are deselected in the neighboring
  run; their test file and both relevant CSS sources are byte-identical to
  merged dev (`baseline-css-pins.txt`). No full suite ran.
- Review reproduced duplicate contraction and retired mode/subview gestures
  before the fix (`review-red.txt`). Initial CLI, occupied-profile and cleanup
  failures are also retained. Independent review cleared the final production
  and fixture-cleanup changes.
- All **nine derived-artifact guards** pass (`current-preflight.txt`). Ruff
  adds no production diagnostics (2 Audit, 66 workbench); new test/runner code
  passes lint and formatting, and modified production ranges are formatted.
- **Twenty native captures inspected:** TldwCli with LinuxDriver and both TTY
  streams, dark/light at 80×24 and 170×48. Every cell covers selection, prepend
  refresh, filtering away, combined filters and exact same-name tool drilldown.
  The inspector identifies `audit-beta` while `audit-alpha` advertises the same
  tool. Compact inspector scrolling is inherited from merged PR2770.
- Native records are synthetic metadata in real JSONL storage. Two real local
  stdio catalog connections support navigation; the trace contains no tools/call
  and the network guard records zero external attempts. This does not qualify
  execution runtime or the compact filter toolbar's keyboard usability.
- Exit 0, no app exception, absent app/fixture PIDs, released instance lock,
  ten healthy private databases, zero chats/messages, unchanged default files
  and pre-existing fixture sentinels, no log errors and matching source/runner
  hashes are recorded in `lifecycle.json`. Cleanup requires actual child exit,
  removed connections and deleted owned profiles before declaring success.

`export-manifest.json` records original and retained hashes. SVG/text receipts
normalize trailing whitespace only. The raw result and launch receipt preserve
the actual run provenance; later docs/task commits do not change captured code.
The earlier evidence one directory up remains historical. Its old duplicate
stdio fixture was replaced with the existing validated shared fixture.

## Integration and remaining scope

The saved commit had two documentation conflicts: the MCP review ledger and
testing lessons. Both histories were retained; the old Audit qualification is
labeled historical. There were no product conflicts. Rebase onto the actual
PR2770 merge only changes ancestry because that merge has the qualified tree.

The compact filter field and initiator prompt remain clipped (saved PR2721).
Built-in readiness guidance still appears above unrelated execution/tool detail
(saved PR2722). These defects are visible in the captures and remain separate
bounded work. This selection repair does not change CSS, tokens, schemas,
permission policy or runtime authority.

ADR required: no. Existing ADR-170 governs table selection; ADR-150/161 govern
the inherited visual language. Current-head CI, accumulated review, latest-dev
review and **fresh owner visual approval** remain merge gates for PR2720.

[Compact dark selection](textual-dark-80x24-selected.svg) ·
[Compact light filtered](textual-light-80x24-filtered.svg) ·
[Wide dark refresh](textual-dark-170x48-refreshed.svg) ·
[Wide light tool](textual-light-170x48-exact-tool.svg)
