# Buddy qualification on current dev

TASK-32108; existing ADR-126 profile lifetime and ADR-139 independent Buddy bindings.

Original production source: `64579cce2c8dc64053fb50c00eb4f59b56716b01`. The harness
SHA-256 and separate fresh/upgrade receipts are in
[the verification manifest](artifacts/buddy-v1-32108/current-dev-20260929/verification.json).

The unchanged harness failed all 16 nodes at fixture setup with
`RecoveryRequired: raw_source_selection_changed`. Its in-process profile
redirect predates the recovery source-lifetime contract. Reuse the existing
`private_profile_test` child helper for the two app journeys; retain one
bootstrap profile for the pure harness guards. Seed setup/splash preferences
through the production configuration save seam. No production guard is changed.

Fresh and synthetic schema69-to-current profiles both reached schema73. The
original assertions still verify independent artwork without a Persona, exact
conversation binding across Console/Home, an existing workspace Persona default
cleared to explicit None, settled workspace inbox, Static frame0 and Dynamic
0→1→2→3→0. The hard-coded schema70 expectation now follows the actual head.

Validation: **16 passed in 110.81s**, scoped Ruff and formatting passed, and
`git diff --check` passed. All 11 derived-artifact preflight checks passed.
Independent read-only code review found no actionable
issues. The complete original and amended harness executions were separate; the
manifest records their raw log hashes. SVG captures remain in disposable local
storage, with all capture hashes in the receipts; two were visually inspected.

This is headless Textual rendering with real SQLite and bundled artwork. It does
not qualify a physical terminal, microphone, audible playback, real provider or
full local test suite. TASK-32108 remains In Progress for native acceptance; older
receipts retain their original source attribution.


PR #2910 was rebased onto dev `64e140bb30b476546f561fae0d61f2697fc91325`
after the gateway/provider configuration additions. Range-diff preserved the
reviewed patch exactly. A separate fresh/upgrade headless run passed **16 tests
in 191.58s**, and all 11 derived-artifact preflight guards passed. The
[rebase verification](artifacts/buddy-v1-32108/rebased-dev-20260929/verification.json)
and separate fresh/upgrade receipts retain their tested source and capture hashes.
Earlier evidence keeps its original source attribution; native and physical
voice acceptance remain open. No full suite ran.


CI follow-up: PR Fast Lane hit the pre-existing TASK-33211 missed-click flake
(1 failed, 1170 passed); the admission-sensitive run passed 123 tests with one
expected failure. Reused the one-line repair from [PR #2883](https://github.com/rmusser01/tldw_chatbook/pull/2883),
original commit `d48f92be5a3f974536b5f143a118e10c985827e2`, with attribution retained.
The existing helper waits for the inspector scroll to settle and asserts that the
mouse click reaches Run. At source `1eab2bc8e9519bc9ca73042cc3eda0c94e9c75ee`,
the exact test passed **50/50 across four workers in 32.37s**; suppressing only
the production result toast still failed the original missing-toast assertion.
Ruff diagnostics match the unchanged baseline (five existing findings, zero new).
Raw stress and negative-control logs remain local. No production code or timeout
was changed, and no full local suite ran. ADR required: no; test-only helper reuse.

The following rebase onto the FTS query change in dev
`6423c4fbd1460952dd8cb5df04e51c4e0e7e0d0b` preserved all four PR patches exactly.
At source `52bb6f10d2f7d0b5e7a834286a7cd42f6f55eb59`, a separate Buddy run
passed **16 tests in 93.94s**, and the three incoming FTS query-plan/result cases
passed in 6.37s. Fresh and schema69 upgrade profiles still reach schema73 and
retain the same ownership, workspace-default and Static/Dynamic assertions.
The [FTS rebase receipt](artifacts/buddy-v1-32108/fts-dev-20260929/verification.json)
records the tested source, sanitized profile outcomes and local-log hashes.
Earlier receipts retain their original source attribution. Native and physical
voice acceptance remains open; no full local suite ran.


The later provider-control dev rebase onto
`5216de505826f2fb2036ccbb740fd667802d5b90` preserves all five preceding
PR patches exactly. At source `7659f69fe3d690288bb250e1feda69b6d3da9fd8`,
the two targeted files passed **562 cases in 111.51s**: 16 Buddy qualification
cases and 546 Console provider-control cases. The fresh and schema69 upgrade
journeys still reach schema73 with the same ownership, workspace-default and
Static/Dynamic assertions. The [provider rebase receipt](artifacts/buddy-v1-32108/provider-dev-20260929/verification.json)
records sanitized profile outcomes and the local execution-log hash. Prior
MCP and Buddy evidence retains its original source attribution. No full local
suite or paid provider calls ran; native and physical voice acceptance remains open.


The approval-boundary rebase onto dev
`857b3dd7d058bc4076b9f4b23d31c724620d6193` preserves all six preceding
PR patches exactly. At source `3730c33cf3b18b5b9f2784bb9af842c1ddcd0a2d`,
the Buddy qualification and incoming approval regressions passed **27 cases
in 67.86s**: 16 Buddy cases and 11 built-in, MCP and local approval cases.
Fresh and schema69 upgrade profiles still reach schema73 with the same
ownership, workspace-default and Static/Dynamic assertions. The incoming
regressions retain refusal and audit behavior when a same-name sibling lacks
its own approval. The [approval rebase receipt](artifacts/buddy-v1-32108/approval-dev-20260929/verification.json)
records sanitized outcomes and local-log hashes separately. Earlier evidence
keeps its original source attribution. No full suite or paid provider calls ran;
native and physical voice acceptance remains open.


The provider-preset rebase onto dev `2a74675eea24bc2d96b40bf1da2d6550043a58a1` preserves all seven preceding PR patches exactly. At source `ce02e03442986d706e73455536626700cf52422c`, the targeted Buddy and incoming provider tests passed **187 cases in 66.55s**: 16 Buddy qualification cases and 171 provider-preset, setup-persistence, env-readiness and catalog cases. All 11 derived-artifact preflight guards passed. Fresh and schema69 upgrade profiles still reach schema73 with unchanged ownership, workspace-default and Static/Dynamic assertions. The [preset rebase receipt](artifacts/buddy-v1-32108/preset-dev-20260930/verification.json) records sanitized outcomes and local-log hashes separately. Earlier evidence retains its original source attribution. No full local suite or paid provider calls ran; native and physical voice acceptance remains open. ADR required: no; existing ADR126 and ADR139 govern unchanged Buddy behavior.


The server-session character-boundary rebase onto dev `5980da9c12aa8257db2b3176bd5962b6b6ce00db` preserves all eight preceding PR patches exactly. At source `ba3831f6e46218fabe0a24635c8e917b22287e0d`, the targeted run passed **113 cases in 64.77s**: 16 Buddy qualification cases, 96 MCP-provider/Console-character cases including the nine incoming refusal cases, and the existing MCP toast regression. All 11 preflight guards passed. The incoming checks retain refusal before approval side effects, no persisted grant, switched-backend refusal, allowed reads, local/unbound writes and fail-closed lookup behavior. Fresh and schema69 upgrade profiles still reach schema73 with unchanged ownership, workspace-default and Static/Dynamic assertions. The [character-boundary rebase receipt](artifacts/buddy-v1-32108/character-dev-20260930/verification.json) records sanitized outcomes and local-log hashes separately. Earlier evidence retains its original source attribution. No full local suite or paid provider calls ran; native and physical voice acceptance remains open. ADR required: no; existing ADR126 and ADR139 govern unchanged Buddy behavior.


Qodo review of `b6860cf13feceb4836d300a00196ca9612174ff6` identified missing parent coverage from the isolated child journeys. A real probe reproduced zero child-only hits in serial and xdist reports (two failures, two standalone/disabled controls passing). The shared helper now loads existing pytest-cov only when parent coverage is active, retains source/config/branch/context settings, writes each child dataset privately, and hands a unique parallel file to native parent combination. Standalone behavior, selected profile lifetime, exact-node checks, timeout bounds and parent report thresholds remain unchanged. At fixed source `b007dd43cf70408d88093ec983bf9702a800cb08`, **20 targeted tests passed in 237.45s**: four real coverage regressions and 16 Buddy cases. Both complementary child branches survive the serial/two-worker report, and the real fresh/upgrade journeys each contribute **987 app run-context lines** to parent data (1,074 app XML lines with hits). All 11 preflight guards and scoped Ruff/format/diff checks passed. Fresh/schema69 upgrade profiles retain schema73 and unchanged ownership/default/Static-Dynamic assertions. The [coverage receipt](artifacts/buddy-v1-32108/qodo-coverage-20260930/verification.json) contains sanitized outcomes and local evidence hashes. Raw logs, coverage datasets/XML and captures stay local. Earlier receipts retain original source attribution; native/physical voice acceptance remains open. No full local suite, production change, timeout increase or paid provider call. ADR required: no; this is coverage accounting for the existing test helper under ADR126.

## Console hook consent dev rebase — September 30

2026-09-30 Console hook-consent dev rebase onto 75c06af39a07154560ab1db49561a620a9689e72 (PR2922): ten prior patches remain identical; the final metadata append preserves its content and all three incoming lessons after a documentation-only conflict resolution. At tested source 21eadf3b34e346b610ea1b44a868c5b2bc458bd3, 227 distinct targeted cases pass: 211 incoming hook runtime/config/UI cases and 16 Buddy cases. The first combined run passed225 and failed only the two Buddy capture-root guards because inherited macOS TMPDIR disagreed with the explicit /private/tmp output. Correcting TMPDIR alone passed all16 Buddy cases in74.99s; no source or timeout change. All11 preflight guards and scoped Ruff/format/diff checks pass. Fresh/schema69 upgrade retains schema73 and unchanged ownership/default/Static-Dynamic assertions. Sanitized receipt: Docs/Reviews/artifacts/buddy-v1-32108/hooks-dev-20260930/verification.md. Earlier evidence keeps original attribution. No full sweep or paid provider; native/physical voice acceptance remains open, task In Progress. ADR required: no; existing ADR126/139 apply, incoming hooks follow ADR197.
