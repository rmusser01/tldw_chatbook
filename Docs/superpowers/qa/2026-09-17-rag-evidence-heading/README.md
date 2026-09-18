# Search/RAG evidence heading — TASK-15390

2026-09-17 UTC (2026-09-16 Pacific), on `feat/component-pattern-library`, based on
`6b50af0a33`. This changes one rendering test; production behavior is unchanged.

## Root cause and repair

The original task reported a stale child-order assertion. Commit `13916d8abe`
added the result-count line before the coverage note; `67c667d063` already fixed
the assertion on August 11. The original heading test passes on this branch,
both alone and after the state suite. The complete pre-change gate16 file also
passes: [69 tests](branch-baseline-gate16.txt). Previous audit slices excluded
this test based on the open task, without reproducing that failure on this branch.

There is a separate current isolation problem on saved `origin/dev`
`24094f23d59c7a9d3cfac964c19fd263bc0393b2`. Commit `f875d8225f` made depth
profile-resolved and pinned the heading test to the shipped default, 15. Dev's
later storage admission (`b5251e9a6eb`) can refuse access to profile/config state
in this unit fixture. The existing UI helper then falls back to 5, so the test
[fails expecting 15](saved-dev-red.txt). This is an uncontrolled test input,
not evidence of incorrect heading rendering or a reason to bypass storage checks.

The test now supplies depths 5, 15 and 23 through `library_rag_profile_top_k`.
It retains real panel construction and rendering, both mode suffixes, conditional
coverage text/classes, and result-count → coverage child ordering. Existing
Library state and active-config tests still exercise actual profile resolution,
explicit overrides, bounds and fallback behavior.

ADR required: no. Existing
[ADR-003](../../../../backlog/decisions/003-settings-library-rag-defaults.md)
continues to govern the unchanged display/profile boundary. No production, CSS,
schema, dependency or authority changes were needed.

## Verification

- [Final targeted run](final-targeted-tests.txt): **277 passed in 65.95s**, no
  exclusions. It covers Library RAG state, active-config resolution, and the
  complete Search/RAG gate16 file.
- [Saved-dev repair](saved-dev-green.txt): **3 passed** using the same test-function
  change in a clean source export of the saved dev commit. The package import
  path is asserted to be inside that export. This is a source-snapshot check
  with the existing environment's dependencies, not a fresh dependency install,
  remote fetch, dev integration, or a full dev suite.
- [Isolation checks](saved-dev-isolation.json): all other test AST nodes match
  saved dev, the repaired function matches this branch, and the relevant five
  production modules match their saved-dev bytes.
- [History](history-verification.json) and [diffs](historical-diffs.txt) pin the
  original ordering change, its prior correction, and the profile-depth change.
- [Lint comparison](lint-comparison.json): no new Ruff diagnostics (five existing
  diagnostics remain); the changed range passes formatting. Diff whitespace
  checks pass.
- Independent review of the final helper-level fixture found no issues and
  independently reran the three heading cases successfully.

Branch command:

```sh
.venv/bin/python -m pytest \
  Tests/Library/test_library_rag_state.py \
  Tests/RAG/test_active_config_resolution.py \
  Tests/UI/test_product_maturity_gate16_library_search_rag.py \
  -q --tb=short
```

The saved-dev check exports `tldw_chatbook`, `Tests` and `pyproject.toml` with
`git archive`, replaces only this test function, prepends the export directory
to `sys.path`, verifies `tldw_chatbook.__file__`, then invokes the exact heading
node with `pytest.main`. It does not use the branch's editable package accidentally.

The attempt logs preserve two fixture mistakes: passing an unsupported `top_k`
keyword to the panel constructor, and importing `active_config` before patching
its resolver. The latter still crossed dev's guarded config import. These are
test setup failures, not product regression evidence. The final helper-level
patch avoids that import while leaving the real resolver's dedicated tests intact.

Native verification was not repeated because this is a test-only change. No
full repository suite, push or merge was run. Retrieval/provider journeys remain
separate audit work.
