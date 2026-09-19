# Generation defaults review — TASK-32748

Generation-profile edits now remain readable and reachable through compact
layout, category return and resize. The existing Settings normalizer rejects
non-finite values before mutation. The disclosure remembers its per-screen open
state, scoped form rules stack compact labels, and deferred scrolling reveals
the same attached focused control after layout changes.

ADR required: no. Existing ADR-006 owns provider-aware defaults; ADR-031/150/161
govern interaction and component patterns. This repair changes no provider or
storage contract and adds no design tokens.

## Reproduced defects

- `initial-red.txt`: eight failures reproduced compact value clipping and a
  disclosure closing after category rebuild; four persistence cases passed.
- `nonfinite-red.txt`: `nan` reached the Settings mutation boundary despite the
  existing range validation. The mounted regression now asserts no write.
- `resize-red.txt`: both themes retained focus on a field outside the viewport
  after 170-to-80-column resize.
- `compact-edit-red.txt`: the initial fixes left two compact cases failing when
  dirty-state copy grew above the focused field. The shared reveal callback now
  runs for staged generation-field changes as well as resize.

## Targeted verification

**173 distinct cases passed**, with overlapping reruns excluded:

| Receipt | Result | Scope |
| --- | --- | --- |
| `final-journeys.txt` | 15 passed | OpenAI/Anthropic supported-control keyboard traversal, complete paint, validation, Save, blank removal, Revert, ownership, category return and resize in dark/light wide/compact production CSS |
| `regressions.txt` | 109 passed | Token/component governance, Python inventory, bundle sync, gallery layout/snapshots, boot budget and adjacent provider journeys |
| `provider-final.txt` | 30 passed, overlap | Final-source rerun of provider keyboard/return cases already counted in the 109 |
| `hub-final.txt` | 43 passed | Selected profile, provider-control, responsive and focus contracts |
| `css-contracts.txt` | 11 passed | Five cases overlap the 43; six compact-field cases are additional |

Five pre-existing static tests still read the former monolithic stylesheet.
`static-baseline.json` reproduces all five against exact HEAD test functions and
HEAD CSS inputs in a disposable projection. This establishes those static
failures only, not a baseline app run. They now read the owning feature/form
sheets and the established compact-height token; their original contracts are
retained.

`static-checks.json` records syntax success, no added lint diagnostics against
HEAD (117 existing Settings and 11 existing hub-test diagnostics), and clean
Ruff check/format for the new tests and native runner. Changed existing ranges
were formatted without rewriting unrelated debt. Boot CSS is **616,786 bytes**
against the unchanged **634,050-byte ceiling**. No full suite ran.

## Private native verification

`native_check.py` ran the real `TldwCli.run(auto_pilot=...)` with the production
CSS and LinuxDriver in an owned terminal. The runner checked its TTY and private
config/data paths before app startup. Final profile:
`/private/tmp/tldw-32748-generation-native-002`, PID 89310.

Four final journeys covered dark/light at 170×48 and 80×24. Each rejected `nan`,
exercised keep-editing/discard, retained `0.35` through category return and
resize, retained the draft after an injected failed writer result, saved through
the real config writer, and removed the override by saving a blank value.
Exact parsed-config comparisons preserved the other model/provider and global
defaults. There were 12 mutation attempts: four injected failures and eight real
successful writes. `result.json` records source/runner hashes and all assertions.

Eight final SVG captures were rendered and inspected in pairs. Temperature's
full label/value, retained focus, pending/clean state and compact stacked fields
are visible in both themes. The pending captures show the retained draft after
the failure toast was dismissed; they do not claim to picture the failure
receipt. `capture-manifest.json` records raw/stored hashes; only trailing
whitespace was removed. `native-terminal.txt` captures the actual terminal paint.

`lifecycle.json` records all 11 private SQLite integrity checks passing, zero
conversations/messages, unchanged default config/UI-state/runtime-policy
fingerprints, empty faulthandler, no error events, normal app stop, exit 0 and
exact process absence before closing the owned terminal. `earlier-run.json`
documents the earlier run's clean lifecycle; that run is not the final resize
qualification. Both owned terminal sessions are closed.

The failure injection qualifies adapter-result handling, not an actual OS
permission error. Native persistence covers OpenAI profile settings;
Anthropic gating is covered by mounted journeys. No external provider request
or generation was made, and no provider availability is claimed.

Independent review found one callback-lifetime guard to add (`is_attached`),
which was fixed. Final source/test/style review found no actionable defect;
`review.json` records the result. The guide, workflow audit, finite-input lesson
and full completion ledger were updated. Current-dev integration is next;
remaining Settings and destination reviews remain open.
