# Report a bug — implementation evidence

Base: `798f8fca75a7acba56da5f523329efa52c4a78f0`, fetched from dev before design.
Implementation: `98b1e50239`, `758079c43c`, `c26076fc0d`, `3edfa2113e`.
Independent-review fixes: `906ffbe710`.

The feature is implemented locally. TASK-37173 remains In Progress: its final
verification criterion requires green shared checks, and existing failures below
prevent that claim. No full-suite result is claimed; the repository's targeted
testing instruction governed verification.

## Focused checks

| Check | Result |
| --- | --- |
| Pure renderer and existing log sanitizer | 163 passed |
| Bounded collectors, native reader and privacy checks before UI wiring | 38 passed |
| Modal and applicable quit census controls | 21 passed; unrelated census failure reported below |
| Combined feature/sanitizer/native/privacy run | 219 passed; three custom-PII subprocess timeouts under concurrent load |
| Existing PII tests after load retired | 12 passed in 6.49 s; no timeout/privacy-policy changes |
| Regression run after all four review fixes | 57 passed in 73.42 s |
| Post-fix entry points and fresh-process import deferral | 5 passed in 65.33 s |
| Affected Quick Actions provider contracts | 7 passed |
| Task IDs/readability and ADR numbering | 5,895 task files and 299 ADRs passed |
| New helpers/tests formatting | Passed |
| Fatal Python checks | No new findings; two existing annotation names reproduced from dev |

The 57-test run includes both collector/trace test files, existing native-reader,
privacy-owner and PII-mask tests, and real Textual modal interactions. Keyboard
traversal reaches every optional field and Copy/Close at 60×24 and 120×40. Coverage
includes stale generation/copy, runtime changes, close/reopen, the real Ctrl+Q
quit walk, oversized edits, missing/deadline-limited sources, redaction re-review,
and unavailable clipboard delivery. Renderer tests cover UTF-8 budgets and hostile
Markdown delimiters. Import tests use a fresh Python process.

## Independent review

A fresh read-only whole-branch reviewer examined `798f8fc..3edfa21` against the
approved spec, plan, ADR-252, and rulings. It found four Important issues and no
Critical or Minor findings. All four reproduced before their fixes; the regression
went from four failures to four passes:

- Move populated log alias projection off the UI loop and use one compiled scan
  per row. The regression uses 500 log rows and 750 identifiers.
- Follow actual capture-service `surface_append` nodes through their preceding
  live call boundary and frozen policy. A real-service regression checks tool
  arguments/results while excluding unrelated assistant/user bodies.
- Keep failed agent step outcomes separate from a run's terminal status, so
  successful recovery cannot hide an earlier tool failure.
- Select agent tails by per-run sequence when clocks move backwards. Run groups
  follow insertion order; unknown cross-run chronology is explicitly disclosed.

The workflow's single fix pass was verified with those reproducing tests and the
57-test affected regression run. No second reviewer was dispatched.

## Native terminal checks

Disposable profiles used real `WindowsDriver`, with stdout/stderr attached to
the PTY (`isatty() == True`). The supported image-protocol warmup ran before
`app.run()`. Synthetic exceptions contained a dummy credential and email; neither
survived reporting. A local Calculator returned 4; its result was persisted through
the real capture service and appeared after tool opt-in. No provider request or
report upload was made.

Palette, Logs and canonical Settings About each generated an editable report,
retained a manual review edit, requested copy, and exercised the discard guard.
About used the supported `screen_context={"category": "about"}` navigation
contract. Earlier probe mistakes (pre-readiness launch, missing synthetic message
version, and an incorrect navigation argument) are not counted as successful
complete journeys. A cold trace read hit its budget and correctly surfaced
unavailable evidence; later reads included the frozen tool result.

Received terminal OSC52 packets were decoded into local text sinks. The request
matched the exact reviewed candidate, including its manual edit:

| Native request | UTF-8 bytes | SHA-256 |
| --- | --- | --- |
| Palette | 2,762 | `a7b93c03c97922f5694226c96c9a6cb3d7e7a44431f920686fc9f4e58e976f3a` |
| About | 2,429 | `b7cf714d028fd67dd3e08c337e31ab6f0c595ad7c924cc5fd0ac816ca982a8d8` |

This verifies client-request bytes, not an OS clipboard acknowledgement; the
product correctly says **Copy requested**. Some bulk terminal rendering output
was truncated by the tool; the complete packets above were independently decoded
and hashed. Sanitized receipts and the ordered execution rulings are preserved
beside this document.

## Shared-check limitations

Native receipts: [About](2026-10-28-report-a-bug-about-receipt.json) and
[Palette/Logs](2026-10-28-report-a-bug-palette-logs-receipt.json). The latter
qualifies its successful first two flows; its subsequent probe navigation error
is preserved, and the separate About run supplies that missing qualification.

- The guarded-modal census finds five unrelated guards without decision entries.
  Its keys were normalized for Windows while registering this feature; this
  report's guard and real quit behavior pass.
- Affected shared tests: 274 passed, 13 failed. Two Quick Actions counts were
  updated for the new command and now pass. Remaining failures concern existing
  profile selection, stale Settings/Library command counts, retired Tools CSS,
  and old focus/token expectations in unchanged sources.
- Shared design/component/bundle checks: 19 passed, 12 failed, including Windows
  path comparisons, existing section-header geometry and governance expectations.
  Generated CSS reproducibility checks pass.
- Fatal lint reports `ThemeChange` and `QueuedLaunchDefault` in
  `app_command_providers.py`; the same findings occur in the original dev source.

These results are not a green repository-wide verification claim. AC #7 remains
unchecked under the repository Definition of Done; unrelated test debt was not
hidden by changing production behavior.

Governance: [ADR-252](../../../backlog/decisions/252-bug-report-evidence-and-sharing.md),
[approved spec](../specs/2026-10-28-report-a-bug-design.md),
[implementation plan](../plans/2026-10-28-report-a-bug.md),
[ordered rulings](2026-10-28-report-a-bug-rulings.md).
