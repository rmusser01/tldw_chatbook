# TASK-32793 — Ordered MCP master settings

Tools and Servers now admit the same local-tools master choice synchronously to
one app-owned FIFO shared with the root setting. Captured activation targets
survive repaint; observers may disappear without abandoning admitted writes.
Per-key receipts retain pending choices, distinguish committed-file/cache-refresh
failures, and reject later configuration identities. Known sibling-only writes
advance unchanged receipts only across matching locked file/generation fences.
Master polling never refreshes root input. Catalog refresh failure remains a
separate warning until recovery. Servers wraps the full master label and keeps
its dependent controls and saved off-note consistent with pending choices.

ADR-169 extends ADR-168 for exactly these two keys; ADR-033/150/161 still apply.
Runtime permission checks, other settings and tool execution are unchanged.

## Targeted evidence

| Scope | Evidence |
| --- | --- |
| FIFO, both UI paths, lifecycle, root regressions, real atomic partial publication and config retarget: 44 passed before the final dependent-note adjustment | [Lifecycle checks](lifecycle-checks.txt) |
| Final master suite: 19 passed; one harness selector failure, then corrected cross-control replay plus two existing dependent-control cases: 3 passed | [Master run](master-final-with-selector-failure.txt), [corrected replay](dependent-final.txt) |
| Tools and Servers canvases: 98 passed before the final dependent-note adjustment | [Adjacent checks](adjacent-canvases.txt) |
| Four size/theme complete-paint checks plus five existing master/gate cases: 9 passed | [Paint and adjacent checks](paint-and-adjacent.txt) |
| Final token/component governance: 26 passed | [Governance](governance.txt) |

These runs qualify 177 distinct targeted cases; overlapping replays are not
added to that count. No full repository suite was run. The 98-case canvas batch
initially hit eight existing source-bound configuration fixture failures; Servers
now uses the same interpreter-owned private test profile as the two adjacent MCP
modules. No production recovery guard was weakened. The final master run and
native run003 used the nonexistent web_deep_search widget suffix; changing only
the harness to web_deep_search_enabled repaired both assertions.

[Initial red evidence](initial-red.txt) reproduces reordered writes, lost pending
state and false save failure. Independent review reproduced [lost activation
intent](intent-red.txt) and [root edit loss](root-edit-red.txt). [Compact paint
red](paint-red.txt) and [dependent-note red](dependent-red.txt) pin the visual
findings. [Independent review](independent-review.txt) found no remaining issue.
[Preflight](preflight.txt) records clean new-file lint/format, no introduced lint
findings in changed files, formatted changed methods, unique Backlog IDs and
clean diff checks. [Diagnostic review](diagnostic-review.txt) confirms one obsolete
master-save warning removed and no diagnostic added; the regenerated
[inventory check](diagnostic-final.txt) passes.

## Native visual review

Final run004 uses a fresh private profile, LinuxDriver and actual TTY streams.
A bounded thread barrier holds the off-write before calling the real atomic
writer. Read-only refresh retains pending Off; Servers reverses it to On while
the first write remains held. Both writes then finish in order; real TOML and
runtime cache agree. Pending choices do not modify runtime authority. Permission
profiles remain unchanged. All sixteen final screenshots were rendered and
visually inspected.

| Theme / size | Tools pending Off | Servers pending On | Servers saved | Tools saved |
| --- | --- | --- | --- | --- |
| Dark 80x24 | [View](textual-dark-80x24-tools-pending.svg) | [View](textual-dark-80x24-servers-pending.svg) | [View](textual-dark-80x24-servers-saved.svg) | [View](textual-dark-80x24-tools-saved.svg) |
| Dark 170x48 | [View](textual-dark-170x48-tools-pending.svg) | [View](textual-dark-170x48-servers-pending.svg) | [View](textual-dark-170x48-servers-saved.svg) | [View](textual-dark-170x48-tools-saved.svg) |
| Light 80x24 | [View](textual-light-80x24-tools-pending.svg) | [View](textual-light-80x24-servers-pending.svg) | [View](textual-light-80x24-servers-saved.svg) | [View](textual-light-80x24-tools-saved.svg) |
| Light 170x48 | [View](textual-light-170x48-tools-pending.svg) | [View](textual-light-170x48-servers-pending.svg) | [View](textual-light-170x48-servers-saved.svg) | [View](textual-light-170x48-tools-saved.svg) |

[Native result](native-result.json) pins the runner and 34 production files;
[capture hashes](capture-manifest.json) pin sixteen SVGs and terminal transcripts.
[Lifecycle](lifecycle.json) records PID62550, app.run returning, exit0, absent
process before terminal closure, lock reacquisition, ten healthy databases,
zero conversations/messages, unchanged default-profile fingerprints and no error
or faulthandler output. The owned terminal was closed after these checks.

Run001 found the compact master-label clipping; [capture](failed-run001-state.svg),
[result](failed-run001-native-result.json) and [clean exit](failed-run001-lifecycle.json)
are retained. Run002 succeeded before visual review found the stale dependent
note; its [result](pre-final-run002-native-result.json) and
[lifecycle](pre-final-run002-lifecycle.json) are historical only. Run003's incorrect
harness selector is retained with [result](failed-run003-native-result.json) and
[clean exit](failed-run003-lifecycle.json). Run004 qualifies the final source.

This gallery qualifies the two master controls and receipts. Other Servers gate
labels still clip at compact width and remain in the Servers review; provider
execution, broader server lifecycles and Audit remain separate work. The
[MCP review ledger](../../reports/2026-09-18-mcp-review.md) records these bounds.
Draft PR2707 still requires its own visual review and merge approval.
