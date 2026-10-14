# Root-tree cleanup addendum (2026-10-14)

The tree snapshots in this directory (`expected-final-tree.json`,
`final-source-union.json`, `initial-tree-map.json`, and the sibling
task-15 snapshot) pinned the repository root as it stood on
2026-10-04, including several one-off files. Those snapshots are
point-in-time records and are intentionally **not** edited.

On 2026-10-14 the root was deliberately cleaned up; if you are diffing
a current tree against these snapshots, the following divergences are
the known, intentional ones (branch `chore/clean-dev-root`):

- **Removed** (one-off artifacts): `IMPLEMENTATION_PLAN_personal_context_first_link.md`,
  `IMPLEMENTATION_PLAN_personal_context_sync_capabilities.md`,
  `IMPLEMENTATION_PLAN_personal_context_sync_transport.md`,
  `IMPLEMENTATION_PLAN_backup_uat_remediation_20260913.md`,
  `pr-review-fixes.md`, and the tracked-but-gitignored
  `output/ux-review-setup-wizard/findings.md` (`/output/` is ignored
  by `.gitignore`).
- **Relocated**: the debug scripts `check_app.py`, `check_textual.py`,
  `clean_cache.py`, `minimal_app.py`, `verify_ui.py` moved to
  `Helper_Scripts/`. `run_all_tests_with_report.py` and
  `run_rag_tests.py` stay at root — they are the documented test-runner
  entries (`Tests/README.md`, executed by `Tests/run_quick_tests.py`).
- **Relocated**: every `qa/*` evidence directory moved to `Docs/QA/`
  (the established QA-evidence home). Path references in living code
  and tests were updated; references inside dated evidence artifacts
  (including these snapshots) were left as historical records.
