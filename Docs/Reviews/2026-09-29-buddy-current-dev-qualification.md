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
