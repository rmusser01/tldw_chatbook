# Console hooks: dev integration and verification

Verified 2026-09-28 on **codex/hook-review-dev**, based on dev commit
e5ac111967bd7310e6e97dec043a559d07e97d30. The original codex/hook-review
branch is preserved. Only the hook design, plan and implementation commits
were carried over; unrelated planning history is excluded.

Implements [ADR-197](../../../backlog/decisions/197-console-hook-configuration-review.md)
and [TASK-33163](../../../backlog/tasks/task-33163%20-%20Add-Console-hook-settings-and-consent-review.md).
The Console has a persistent Hooks action and attention count, expandable
permission review, and a link to canonical Settings → Expert → Hooks.
Existing, new and changed enabled definitions require saved exact-definition
consent on the next actual Send. Cancellation preserves the draft. Shared
admission and process creation enforce the same authority.

## Current evidence

**201 distinct feature and Settings metadata cases passed across targeted
runs.** This is a union of named cases, not a single full-suite result.

| Check | Result |
| --- | --- |
| Hook executor, consent, config snapshots, shared admission and viewless lifetime | 153 passed |
| Latest consent owner regression suite, including decoder failure recovery | 42 passed; overlaps the core run and adds one case |
| Hook identity and inventory | 27 passed |
| Mounted native hook review and staged Settings editor | All 13 cases passed; also repeated with nine admission cases in a completed 22-case run |
| Real canonical Settings metadata contracts | Seven selected checks passed |
| Design tokens and generated CSS integrity | All eight token and 29 build-integrity cases passed |
| ADR-126 raw participant lifetime and config-lock inventory | All 55 raw participant and eight lock-inventory cases passed |
| Authored Python | 48 changed files checked; 11 new files clean; zero Ruff findings on authored lines and zero authored-range formatting failures |
| QA scripts | Ruff and formatting passed |
| Actual native 80×24 and 120×40 application | Both passed all six check groups |

The latest consent run followed the final owner change. The earlier core run
preceded that narrowly scoped decoder-error guard. Whole-file Ruff reports 860
existing-file diagnostics; they were not swept or auto-fixed.

The [machine-readable results](../qa/2026-09-27-console-hook-settings-and-review/test-results.json)
retain named cases, completed versus interrupted runs, baseline comparisons and
superseded diagnostics. [Static analysis](../qa/2026-09-27-console-hook-settings-and-review/static-analysis.json)
records the authored-line checks.

## Integration review

Dev's newer token compiler, dimension scale, responsive modal rules, Console
session callbacks and canonical Settings contracts are retained. Generated
styles were rebuilt from their sources; the obsolete Console stylesheet was
not resurrected.

A real application run exposed the newer raw-file boundary: the retained config
lease could not authorize the separate consent JSON. The concrete Hooks owner
now joins existing ADR-126 admission and drain for only its selected JSON, lock,
and atomic temporary file. The outer config lease and writer locks remain held;
config helpers retain their narrow paths. Real bound-profile tests cover nested
scope restoration, recovery closure, source retargeting and unrelated-file refusal.

Backup discovery recognizes the local consent JSON and lock as intentionally
excluded. Its installed owner rejects restore input. Portable configuration
therefore requires fresh local review. No grants are added to TOML.

Final review also found that a JSON decoder depth exception escaped the
resettable recovery path. A red regression using real private file IO and a
controlled decoder failure now passes after adding that exception to the
existing corrupt-state handler. This proves exception handling; it does not
claim a particular nesting depth fails every decoder environment.

Original security fixes remain covered: stale captured blocking targets refuse
the protected action, stale notification targets are skipped, launch locks are
released before waiting for output, revocation fences future launches, and
failed config refresh retains the local execution fence. Guided saves preserve
unknown fields and invalid originals, protect concurrent edits and transfer
legacy consent only for complete unchanged groups.

No dependency, database migration, watcher, plugin-hook framework or legacy
Settings destination was added.

## Native evidence

Both checks used the real TldwCli with an isolated config/data/database profile,
a harmless real Python marker command and the existing recording gateway.
No real provider generation or user hook was invoked.

The six groups verify private paths; reachable icon and exact details; next
Send plus Escape retains text with no execution; approval launches the hook
once and admits the recording gateway with persisted consent reload; canonical
Settings deep link; and a keyboard master-switch edit that remains staged until
guarded Save.

| Actual terminal | Review capture | Settings capture |
| --- | --- | --- |
| 80×24 | [SVG](../qa/2026-09-27-console-hook-settings-and-review/review-80x24.svg) | [SVG](../qa/2026-09-27-console-hook-settings-and-review/settings-80x24.svg) |
| 120×40 | [SVG](../qa/2026-09-27-console-hook-settings-and-review/review-120x40.svg) | [SVG](../qa/2026-09-27-console-hook-settings-and-review/settings-120x40.svg) |

All four SVGs were rendered and visually inspected. Narrow details and Settings
fields scroll while actions remain reachable; the wider view shows the complete
command and readable control states. The harness owns a fixed-size PTY and
asserts dimensions after interactions: the tool transport otherwise resized a
requested 120-column terminal to 80 columns after yielding. Early unqualified
captures are excluded.

## Baseline failures and limits

**11 remaining failures reproduce against an untouched archive of the exact
dev base**, using the same installed environment:

- Three existing CSS/component ratchets: four unbundled DEFAULT_CSS owners,
  existing dimension literals, and a pre-existing Python style assignment.
- Four existing config-participant lifetime checks: paused cached bootstrap,
  pause during serialization, two-process key preservation and reentrant
  derived publication.
- One wide-modal registry check with stale ConfirmDialog/DeleteConfirmDialog
  skip entries.
- Three installed-owner inventory/capture checks, including the legacy
  collections read-only boundary.

Four existing Settings metadata checks also failed on the base because they
lacked a bound private profile. Those test fixtures are now scoped correctly,
the category count includes Hooks, and all seven selected metadata checks pass.
A temporary wrapper assertion was removed in favor of the real checks.

Broader CSS, storage and UI bundles were interrupted after recording their
completed cases: respectively 84 passes/three failures, 127 passes/four failures,
and 37 passes/no failures. Their remaining cases are unqualified. The completed
inventory/wide-modal run recorded 124 passes and its one reproduced registry
failure. No full suite, global Settings search sweep or full Settings hub was
qualified on this integration.

Executable contents at an unchanged path are not signed. Revocation prevents
future launches and does not terminate a command already running.

## Reproduce

Run from this integration checkout with the existing environment:

    .venv/bin/python -m pytest Tests/Agents/test_hook_permissions.py Tests/Agents/test_run_hooks.py Tests/Agents/test_hook_config_inventory.py Tests/test_hooks_config_snapshot.py Tests/Chat/test_console_hook_admission.py Tests/Chat/test_console_viewless_hooks.py -q --timeout=120
    .venv/bin/python -m pytest Tests/UI/test_console_hooks_review.py Tests/UI/test_settings_hooks.py -q --timeout=120
    .venv/bin/python Docs/superpowers/qa/2026-09-27-console-hook-settings-and-review/check_changed_python.py
    .venv/bin/python Docs/superpowers/qa/2026-09-27-console-hook-settings-and-review/native_check.py 80 24
    .venv/bin/python Docs/superpowers/qa/2026-09-27-console-hook-settings-and-review/native_check.py 120 40

Leave terminal output attached. The native script creates its private profile
and prints the capture/results directory. Exact baseline comparisons used
git archive of the base above; the older baseline_control.py overlay helper
alone is insufficient for tests that import fresh modules or launch children.

The integration is local. Publishing, merging and fixes to unrelated baseline
failures remain separate work.
