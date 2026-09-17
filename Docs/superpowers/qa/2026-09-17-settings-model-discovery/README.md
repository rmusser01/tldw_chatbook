# Settings model discovery — TASK-32739

2026-09-17, `feat/component-pattern-library`, based on `adbd2fe62a`.

Checked discovery rows were stored only when Save began, and unchanged endpoint
mount events reset the results when the pane rebuilt. Late discovery/save/clear
completions could also replace another connection's form, while Clear reported
success after exceptions. Settings now retains checked IDs, ignores unchanged
connection-field echoes and invalidates pending presentation through a monotonic
revision. Successful saves still publish their durable additions to the original
provider's catalog; stale receipts cannot activate a model in the new form.
Clear failures retain the rows and offer retry; successful reset clears typeahead.

ADR required: no. Existing ADR-002/020 govern discovery and exact provider-list
persistence; ADR-031/150/161 govern keys, tokens and component patterns. This
preserves Settings ownership and existing catalog contracts without a new boundary.

## Automated evidence

**176 distinct targeted cases pass:**

- 24 new production-CSS journeys and eight existing discovery/typeahead checks.
  The [initial targeted selection](initial-targeted.txt) passed 30 cases and
  exposed two obsolete Logs F8 expectations. Both [also failed with the HEAD
  method bodies](logs-baseline.txt), loaded by this [temporary probe](logs-baseline-plugin.txt).
  The unchanged failure-copy helper already advertises F3. After correcting only
  those expectations, [all six existing discovery checks pass](existing-discovery-final.txt).
  Four of those six overlap the initial pass; the two typeahead checks also
  passed there, for 32 distinct discovery cases in total.
- [79 provider regressions](provider-regressions.txt): keyboard editing,
  saved-default returns, atomic provider switching and QwenCloud API modes.
- [65 contract/governance checks](contract-governance.txt): discovery identity,
  merge/persistence and design-token/component patterns (three warnings).

The [initial reproductions](initial-red.txt) failed all 12 selection/late-result
cases. [Recovery reproductions](recovery-red.txt) additionally failed Clear's
false-success and late-receipt cases. Final coverage includes both successful
and exceptional late results, endpoint A→B→A edits, credential edits, blank versus
existing Model values, retained selection with connection drafts, and all three
operations' retry behavior. Trailing whitespace in the red-test logs was
normalized; [hashes](test-log-hashes.json) preserve raw/stored provenance.
No full repository suite ran.

[Static checks](static-checks.json) pass for syntax, changed-range formatting and
new-file lint/format. Existing lint counts remain 118 in Settings and 11 in the
large configuration-hub test file, with no added diagnostics. The final native
result hashes the formatted Settings source and runner. [Independent review](review.json)
found no actionable issues. API mode is not an input to generic model discovery;
a suggested change there was withdrawn after inspecting the catalog service.

## Native evidence

The [runner](native_check.py) uses real TldwCli and production CSS with LinuxDriver,
TTY output, an acquired instance lock, an isolated config/database profile and a
real loopback HTTP catalog. It does not substitute discovery or persistence.
All [four final journeys](result.json) pass at 170×48 and 80×24 in dark/light:

1. Tab/Enter discovers two exact IDs. The first journey receives a real HTTP 503,
   shows recovery copy, then retries successfully once the button accepts input.
2. Tab, arrows and Space check the second model. Keyboard category navigation
   through Appearance and back preserves that selection in rebuilt controls.
3. Save selected writes only the checked ID to the exact `Llama_cpp` catalog key.
   The first save fills an empty Model as a draft; later saves preserve that Model.
   The saved chat default remains blank throughout.
4. Clear empties the actual discovered cache and typeahead while retaining all
   saved model IDs. Each viewport journey starts through category navigation.

Six `/v1/models` GETs and zero generation requests occurred; this includes startup
catalog activity. This establishes local HTTP and config persistence, not external
provider availability or model generation. Async races and save/clear exceptions
are qualified by controlled mounted tests, not by this native fixture.

Eight captures were rendered and inspected in one batch. Focus and selected IDs
are visible in both sizes/themes. Compact rows ellipsize secondary metadata;
the compact saved notification is readable while the full status is above the
scroll viewport. No visual change was required. Only SVG trailing whitespace was
normalized; [hashes](capture-hashes.json) retain raw/stored provenance.

| Theme / width | Selection after category return | Saved notification |
| --- | --- | --- |
| Dark / 170 | [Capture](textual-dark-170-selection.svg) | [Capture](textual-dark-170-saved.svg) |
| Dark / 80 | [Capture](textual-dark-80-selection.svg) | [Capture](textual-dark-80-saved.svg) |
| Light / 170 | [Capture](textual-light-170-selection.svg) | [Capture](textual-light-170-saved.svg) |
| Light / 80 | [Capture](textual-light-80-selection.svg) | [Capture](textual-light-80-saved.svg) |

[Lifecycle evidence](lifecycle.json) confirms exit 0, app-run return, server closure,
exact-PID absence before closing the owned terminal, eleven healthy private
SQLite databases, zero conversations/messages, unchanged default-file fingerprints,
normal app stopping, empty faulthandler logs and no error/critical/traceback lines.

Three earlier fixture runs were closed and are not counted as final qualification:
run-001 added a duplicate provider alias; run-002 retried inside Button's activation
interval; run-003 inherited focus from a disabled Clear across a resize and failed
its paint assertion. The final journey enters each viewport through category
navigation. Resize focus from a disabled action is not qualified by this slice.
The lifecycle checker used the app's Python environment after the system Python's
SQLite open attempt failed; all eleven read-only checks then passed.

The [closeout ID scan](task-id-check.json) checked 258 refs and 27 worktrees with
no collision. The guide and existing live-verification lesson were updated.
This continuation is saved to draft PR #2704 against dev; integration conflicts
remain separate from this bounded review. Next review: automatic model-refresh
controls and their persistence/failure feedback.
