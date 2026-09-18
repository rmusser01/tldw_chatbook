# Empty Notes toolbar — TASK-32752

Base: `36164f9ccc44760346b5a23f283c1ce5526238d8`, the verified current-dev
integration in draft PR #2704. Existing ADR-150/161 apply; no new ADR is needed.

The integration's 170×48 captures cropped `○ Remove placement`. The failing
production-CSS case measured its right edge at 119 while the canvas ended at 113:
the screen supplied a 72-column outer Items contract, but the actual canvas was
68 columns wide. Packing also counted four cells of button chrome while each
tree action consumed five including its right margin.

The tree toolbar now deducts the owning host's stable border/padding from the
screen contract, accounts for actual button and row chrome, and compares the
composed rows against that budget on shrink. The measured fallback already
excludes host chrome. Harmless extra rows remain on growth to preserve widget
identity. Comparing actual composed rows also repairs the first unmeasured frame;
the initial cost-only correction still failed because no generic toolbar decision
changed to request recomposition. Tree caches are cleared before composing a
legacy list, preserving its filter identity after a tree-to-legacy transition.

## Verification

**62 distinct targeted cases passed**, without a full suite:

- `layout-verified.txt`: 18 cases, comprising 13 new geometry/paint/state cases and
  five existing selected-note, narrow-toolbar and resize-identity regressions.
  They cover dark/light, 170×48/80×24, 190→170→80→170→190 resize, selected note,
  protected placement, selected folder, fallback width and tree-to-legacy state.
- `governance.txt`: 44 Python-style, component ratchet, CSS bundle and boot-budget
  cases. No stylesheet or token value changed; generated artifacts remain valid.
- `preflight.txt`: all seven derived-artifact checks pass.
- `static.json` and `static-final.txt`: no new scoped lint/format debt; fatal
  checks pass for the five affected Python files. Inherited whole-file formatting
  debt is not rewritten or represented as clean.

The initial geometry failure and the review finding were reproduced before their
repairs. `red-geometry-excerpt.txt` records the cropped action;
`review-red-excerpt.txt` records filter identity loss with cache clearing removed.
Early harness attempts had a wrong wait-helper call and incomplete `sync_state`
arguments; these are not product failures. `run-history.json` retains source-log
digests and summaries, including those attempts. Independent review verified the
width distinction and requested the cache reset, then confirmed no remaining
blocker after its regression was added.

## Native evidence

The final real `TldwCli.run` used an owned tmux PTY and a private HOME/config/data
profile at `/private/tmp/tldw-32752-native-002`, PID 22778. It rendered empty Notes
in dark/light at 170×48 and 80×24, checking every composed tree action's bounds and
actual compositor text. All four SVG captures and matching terminal paint were
inspected: wide layouts wrap the complete disabled Remove placement action onto
the next row with its selection explanation below; compact layouts retain their
existing single New folder action. This check does not qualify saved-note edits,
sync, remote services or generation.

`native-result.json` records the source/runner hashes and actual action regions.
`lifecycle.json` records normal return, exit zero, ten healthy private SQLite
databases, zero conversations/messages, no application errors, empty faulthandler
output, unchanged default config/UI-state/policy fingerprints, exact PID absence
before owned-terminal closure, and terminal closure. `capture-manifest.json`
records original/stored hashes; only trailing whitespace was trimmed.

Attempt 001 was refused by the pre-import profile validator because database
directories had not yet been created. It never started the app and is not native
qualification; `earlier-run-001.json` records exit 1, exact PID absence, unchanged
defaults and owned-terminal closure. The final profile created and validated all
owned database directories before import.
