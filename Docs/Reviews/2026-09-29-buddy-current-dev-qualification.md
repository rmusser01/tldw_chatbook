# Buddy qualification on current dev

TASK-32108; existing ADR-126 profile lifetime and ADR-139 independent Buddy bindings.

Production source: `64579cce2c8dc64053fb50c00eb4f59b56716b01`. The harness
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
