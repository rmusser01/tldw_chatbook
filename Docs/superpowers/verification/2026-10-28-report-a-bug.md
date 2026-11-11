# Report a bug — implementation evidence

Base: `798f8fca75a7acba56da5f523329efa52c4a78f0`, fetched from dev before design.
Implementation: `98b1e50239`, `758079c43c`, `c26076fc0d`, `3edfa2113e`.
Independent-review fixes: `906ffbe710`.
Shared-check completion: `478fd5b328`.

The feature and shared-check follow-up are complete locally. Final verification
at `478fd5b328` covers **605 unique passing targeted cases**. No full suite ran;
the repository's targeted-testing instruction governed verification.

## Focused checks

| Check | Result |
| --- | --- |
| Final report, logs, privacy, native-reader, UI, packaging, focus, palette and full quit census | 395 passed in 239.11 s |
| Design tokens, component patterns and CSS bundle synchronization | 31 passed in 940.49 s |
| Modal lifecycle, covered goal completion, exchange-import boundaries and extraction guards | 179 passed (178 in the affected run; final patch-target guard rerun passed in 28.51 s) |
| Task IDs/readability and ADR numbering | 5,895 task files and 299 ADRs passed |
| Formatting | 19 report/shared-repair modules and test files passed |
| Default Ruff checks | 13 report/shared-repair modules and new tests passed |
| Fatal Python checks | All branch Python changes passed; both annotation findings fixed |
| Whitespace | `git diff --check` passed |

The final combined run includes both collector/trace test files, existing native-reader,
privacy-owner and PII-mask tests, and real Textual modal interactions. Keyboard
traversal reaches every optional field and Copy/Close at 60×24 and 120×40. Coverage
includes stale generation/copy, runtime changes, close/reopen, the real Ctrl+Q
quit walk, oversized edits, missing/deadline-limited sources, redaction re-review,
and unavailable clipboard delivery. Renderer tests cover UTF-8 budgets and hostile
Markdown delimiters. Import tests use a fresh Python process.

## Independent review

A fresh read-only whole-branch reviewer examined `798f8fc..3edfa21` against the
approved spec, plan, ADR-358, and rulings. It found four Important issues and no
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

The feature review's single fix pass was verified with those reproducing tests
and the 57-test affected regression run. The user-authorized shared-check scope
received its own focused review, described below; the feature was not re-reviewed.

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

## Shared-check completion

Native receipts: [About](2026-10-28-report-a-bug-about-receipt.json) and
[Palette/Logs](2026-10-28-report-a-bug-palette-logs-receipt.json). The latter
qualifies its successful first two flows; its subsequent probe navigation error
is preserved, and the separate About run supplies that missing qualification.

The user explicitly requested resolving the shared failures before closing
TASK-37173. That request supersedes the earlier scope deferrals:

- Governance inventories use POSIX keys on Windows, CSS reads specify UTF-8,
  and the standalone pin script binds its own sibling helper and manifest.
  Numeric-literal rejection, canonical sheet ownership, header geometry and
  generated-bundle reproducibility all pass without changing visual tokens.
- Provider tests assert current Settings/Library commands. Focus tests follow
  canonical Settings Tools & MCP (ADR-240), token-backed values, the existing
  four-cell compact action floor and explicit scoped contrast overrides.
- Goal launch, portrait-region saving and response-rule work now answer the
  quit walk. The hook review's unconditional cancel and read-only portrait
  comparison have reviewed census exemptions. Every census row and hook passes.
- The mounted palette test owns a private profile, excludes nonessential boot
  work and awaits its finite Skills workers before closing. A captured stack
  showed native authority initialization still active after the canvas mounted.
  The repeated case passes in the final combined run; retirement budgets and
  storage gates were not changed.
- `ThemeChange` and `QueuedLaunchDefault` are imported for type checking only,
  fixing fatal lint while preserving deferred runtime imports.

A fresh read-only reviewer of the shared-check diff found one Important issue:
goal completion could dismiss the covering quit prompt because Textual's
`is_current` includes background screens. The mounted launch case reproduced
that failure. Exact top-screen ownership now controls delivery, retains the
pending result until resume, and preserves configuration/error updates while
mounted. All three covered-completion cases pass. There were no Critical,
Minor or declined findings; no review findings remain deferred.

The final 395-case command ran the report/collector, sanitizer/log-buffer,
trace/native/privacy, modal/entry/import, focus/palette and complete quit-census
files together. The other 210 distinct passing cases cover the three governance
files plus lifecycle/exchange/extraction checks. Counts are deduplicated by pytest
node ID. Earlier failing runs remain historical evidence, superseded by these
passing results. Native clipboard receipts above remain applicable because this
follow-up did not alter report rendering or delivery.

Governance: [ADR-358](../../../backlog/decisions/358-bug-report-evidence-and-sharing.md),
[approved spec](../specs/2026-10-28-report-a-bug-design.md),
[implementation plan](../plans/2026-10-28-report-a-bug.md),
[ordered rulings](2026-10-28-report-a-bug-rulings.md).
