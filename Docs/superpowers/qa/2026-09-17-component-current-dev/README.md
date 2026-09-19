# Current-dev integration evidence — TASK-32749

Candidate parents: local plan commit `6e0a71dc74` (saved implementation
`9c2edfd67f2ade2df43a90b00ef82ee40d828a91`) and dev
`1c0327b3bb3d95b61e3e1b9a83b30e7030453ad6`. This receipt qualifies the
integration, not every feature in the incoming 71 commits.

## Targeted checks

280 distinct cases passed across the affected selections. Repeated runs are
not counted again. No full suite was run.

| Selection | Distinct cases | Receipt |
| --- | ---: | --- |
| Design tokens, component patterns, Python-style inventory, CSS build/sync, boot budget, Backlog IDs | 101 | `governance-first-summary.txt`, `governance-final-rerun.txt` |
| Sidebar debounce/restoration/source rejection | 11 | `sidebar-private.txt` |
| Console live-work handoffs and bounded sections | 119 | `console-private.txt` |
| File Notes exact path paint, keyboard focus and destructive ink | 7 | `library-private.txt` |
| Covered/transparent/hidden/unmounted Console layout | 8 | `covered-layout.txt` |
| Compact picker plus incoming folder/file initial focus | 17 | `picker-final.txt` |
| Sync review-row disclosure and rejected config-source switch | 2 | `adjacent-summary.txt` |
| Generation defaults | 15 | `settings-private.txt` |

The first governance run passed 91 cases and failed ten. Eight negative-control
cases called newly decorated test functions without executing their async
wrappers; they now explicitly invoke the original assertion under the current
monkeypatch. Two generated/source comparisons raced the final source edits;
the sources were rebuilt and all ten failed cases rerun successfully.
The initial adjacent run passed seven and failed ten picker cases. These tests
expected the older listing-first focus and `Select` label. Incoming dev starts
folder pickers in the path field and labels confirmation `Select folder`.
The updated keyboard journeys exercise that behavior; plain file pickers still
start on the listing.

Incoming config admission binds the selected profile for the interpreter's
lifetime. Old UI fixtures switched it after import, failing before the tested
behavior. `exact-dev-baseline.txt` demonstrates this on the exact incoming dev
commit; the larger exact-dev governance selection similarly had one pass and
40 setup errors. Affected tests now use the existing `private_profile_test`
subprocess helper. The parent skips app import only while launching that child;
the child retains the offline fixture. No production admission guard was relaxed.
The Schedules route checks mutate the preselected private profile instead of
switching source paths after collection. Source-log digests and the original
failed-run summaries are retained in `run-history.json`.

Final preflight passes all seven derived-artifact checks, including hash-pinned
Mermaid reproduction. Boot CSS is **620,062 / 634,050 bytes** (13,988 headroom),
above the unchanged 600,000-byte anti-vacuity floor. `css-preservation.json`
records 45 incoming Notes declarations preserved after token expansion;
`workflow-preservation.json` verifies identical expanded Workflow CSS.

Fatal Ruff across 933 changed Python files reports three diagnostics already
present at the exact incoming dev commit: one F632 in `test_library_screen.py`
and two F821s in `test_mcp_workbench.py`. The incoming missing `Any` import in
the affected File Notes test was repaired. Scoped lint/format comparison against
both parents has no new diagnostic or formatting hunk; inherited debt is not
claimed clean. See `static-delta.json` and `fatal-*.json`.

## Native app

`native_check.py` runs real `TldwCli.run(auto_pilot=...)` in an owned tmux PTY,
using a private HOME, config and data directory established before imports.
The final profile is `/private/tmp/tldw-32749-native-003`, PID 8278.

Four cells cover dark/light at 170×48 and 80×24. Each navigates the unconfigured
Console and empty Library Notes, uses keyboard focus, and records both SVG and
actual terminal paint. All eight captures were inspected. The Console setup
action and Library search/filter focus are visible; the compact Console remains
scrollable. This is startup/navigation rendering evidence, with no provider,
generation, saved-note edit, external server or whole-destination qualification.

**Known visual gap:** both wide Notes captures crop the disabled `Remove
placement` label. Independent review traced this to the incoming empty-selection
branch in `library_notes_canvas.py`; the previous whole-label test only used a
selected placement at 235 columns. TASK-32752 tracks the immediate repair. These
captures must not be described as visually flawless.

`native-result.json` pins runner and production hashes. `capture-manifest.json`
records original and stored capture hashes; only trailing whitespace was trimmed.
`lifecycle.json` records normal return, exit zero, ten healthy private SQLite
databases, zero conversations/messages, empty faulthandler output, no application
errors, unchanged default config/UI-state/policy fingerprints, exact PID absence
before closing the owned terminal, and terminal closure.

Two earlier attempts are explicitly unqualified. Attempt 001 was refused before
app startup because a private HOME was missing for the incoming recovery-control
root. Attempt 002 shut down normally after a runner assertion incorrectly expected
focus on an unavailable composer wrapper. Their separate receipts preserve those
failures and the lifecycle evidence actually obtained.

## Review and scope

Independent read-only review found no integration-specific blocker. It verified
the source-lifetime test isolation, conflict semantics, CSS preservation and the
incoming Notes clipping gap. TASK-32752 and the full completion ledger keep that
gap and remaining feature reviews open. This integration does not merge the PR
into dev or claim CI/full-suite completion.
