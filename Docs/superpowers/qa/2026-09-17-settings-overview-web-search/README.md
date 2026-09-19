# Settings Overview and Web Search — TASK-32753

Base: `807d6c9ce0ef4d619348a9e7cc2da3d09190b959`, saved in draft PR #2704
against `dev`. This is a bounded review of Overview's readiness/navigation and
Web Search's existing staged configuration workflow.

ADR required: no. Existing ADR-012 (credentials), ADR-033 (Settings commit
models), ADR-150 (tokens), and ADR-161 (component patterns) apply. No storage,
provider, permission, or application ownership boundary changed.

## Confirmed defect and repair

At 80×24, Tab reached **Open Providers & Models** while the compositor painted
none of the button. The outer status card was already auto-height, but its
primary `Vertical` still used the default `1fr`: one allocated row contained
15 virtual rows. The diagnostics disclosure consequently covered primary
status and actions. `compact-overview-before.svg` and `compact-geometry-red.txt`
retain the original failing layout.

Giving the primary body auto height exposed the second issue: **Open Privacy &
Security** ended at column 83 while its row clipped at 75. The intermediate
failure is retained in `compact-action-red.txt`. The source Settings sheet now
lets the primary body grow and stacks only its compact action rows. Eleven CSS
lines preserve the detail pane as the scroll owner; existing tokens and values
are unchanged. The lazy Settings sheet was rebuilt from its source.

## Verification

**76 distinct targeted cases pass**; no full suite was run:

- `web-search.txt`: 21 existing behavior/lifecycle cases. Atomic deltas,
  same-field conflicts, secret masking, legacy clear/environment precedence,
  navigation/recreation, queued input, write and reload failures, and discarded
  stale test results remain covered.
- `overview-and-journeys.txt`: 19 existing Overview cases and four production-CSS
  dark/light × 170×48/80×24 journeys. The latter use Tab/Enter through category
  links, check entire focused labels, resize with retained focus, preserve masked
  input across navigation, cancel/confirm Revert, retain failed-save drafts,
  perform a real successful config mutation, and test only on explicit action.
  Their probe outcome and save failure are injected; they do not prove HTTP.
- `governance.txt`: 32 token, component-pattern, generated CSS and boot-budget
  cases. Boot bytes remain 620,062, below the unchanged 634,050 ceiling.
- `review-verified.txt`: the path-label fixture was rerun after review; it is an
  overlapping case, not an additional case in the total.
- `preflight.txt`: all seven derived-artifact checks pass. The first sandboxed
  run could not download a required pinned Mermaid archive; the network-enabled
  run verified its inputs and all generated outputs.
- `static.json` compares Ruff diagnostics and formatter hunks against the exact
  base. There is no new scoped lint/format debt; `static-fatal.txt` passes for
  all five affected Python files. Existing unrelated whole-file debt remains.

The older tests initially changed the selected config path after app imports,
triggering `raw_source_selection_changed`. The affected cases now use the
repository's private-process helper. The Web Search fixture skips the launcher
parent and writes only the child's already-selected profile, retaining its
private data paths. The Overview path-label test patches the path accessor.
Production recovery guards remain unchanged. The Privacy banner assertion was
updated to the existing Canvas settings contract from TASK-31232.

Independent review confirmed the CSS scope and resize regression. Its fixture
finding was repaired and verified. The first geometry diagnostic itself tried
to read an App's nonexistent region; the retained second diagnostic filters to
widgets and reproduces the original product failure.

## Native evidence

Final run `/private/tmp/tldw-32753-native-003`, PID 35146, used real
`TldwCli.run` with an owned tmux PTY, LinuxDriver, private HOME/config/data,
and attached rendering streams. Dark/light × 170×48/80×24 journeys navigated
Overview links, retained a masked Serper draft across category changes,
confirmed Revert without a write, saved an exact SearX URL delta, and explicitly
issued one rejected HTTP 401 request followed by a successful JSON search per
cell. All eight requests reached the owned loopback server through the real
search dispatch; no external provider or LLM was requested. All eight SVGs and
matching terminal text were inspected, including the final visible test status.

`native-result.json` pins the runner and production sources, including the lazy
Settings sheet. `lifecycle.json` records normal exit zero, loopback server closure,
11 healthy private databases, zero conversations/messages, no app ERROR records,
empty faulthandler output, unchanged default config/UI-state/policy fingerprints,
and exact PID absence before closing the owned terminal. `capture-manifest.json`
records original/stored digests; only trailing whitespace was trimmed.

Run 001 is explicitly unqualified: the runner assumed focus remained on Test
while it was temporarily disabled, so retry Enter opened the default selector.
Run 002 retabbed before retry and passed all four journeys, but captured the
entry selectors after completion. Run 003 also retabbed before capturing to
show the actual successful result. The earlier receipts remain in this directory;
no product change was needed for those runner corrections.

## Scope limits

These checks qualify local readiness copy, keyboard/layout, draft persistence,
explicit search testing and recovery. They do not qualify external backend
availability, paid-provider authentication/quota, LLM generation, manual sync,
source switching, or Backup & Restore. Those destination/modal and service
workflows keep their separate review gates in the completion ledger.
