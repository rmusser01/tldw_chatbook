# Console hooks: current-dev integration verification

Verified 2026-09-29 on `codex/hook-review-current`, based on
`origin/dev` at `857b3dd7d0`. The three approved design/plan commits
were already present on dev, so this branch carries only the four hook
implementation commits. The earlier `codex/hook-review-dev` branch is
preserved.

Implements [ADR-197](../../../backlog/decisions/197-console-hook-configuration-review.md)
and [TASK-33163](../../../backlog/tasks/task-33163%20-%20Add-Console-hook-settings-and-consent-review.md).
The Console exposes a persistent Hooks attention action, expandable review of
saved exact definitions, and a link to canonical Settings → Expert → Hooks.
Existing, new and changed enabled hooks require one-time local consent on the
next actual Send. Cancellation preserves the draft; shared admission and
subprocess creation enforce the same authority.

## Current-tree verification

**367 distinct named targeted cases passed.** Of these,
201 cover hook execution/consent, Console admission, mounted review and
Settings category contracts. No full suite was run.

| Targeted scope | Result |
| --- | ---: |
| Hook executor, consent owner, identity, config snapshot, admission and viewless lifecycle | 181 passed |
| Mounted review/Settings editor and selected category contracts | 17 passed |
| Remaining category metadata contracts | 3 passed |
| Design-token and generated CSS integrity | 37 passed |
| ADR-126 raw participant lifetime and config lock inventory | 63 passed |
| Existing queue controller compatibility | 27 passed |
| Mounted queue and Settings search compatibility | 39 passed; two existing search failures |

Generated styles were rebuilt from current dev's source modules and the
rebuild produced no diff. Authored Python checks cover 48 changed files:
zero Ruff diagnostics on authored lines and zero authored-range format
failures. The three QA scripts pass full Ruff and format checks.

[Named machine-readable results](../qa/2026-09-27-console-hook-settings-and-review/test-results.json)
and [static checks](../qa/2026-09-27-console-hook-settings-and-review/static-analysis.json)
are retained with the branch.

## Native application

The real TldwCli passed all six groups at **actual 80×24 and 120×40**
using a private config/data/database profile, a harmless real Python hook,
and the recording provider gateway. The checks verify private paths; Hooks
icon and exact details; next-Send/Escape custody with no execution; one
approved launch and persisted consent reload; the canonical Settings link;
and a staged master-switch edit followed by guarded Save. No user hook or
real provider generation ran.

| Terminal | Review | Settings |
| --- | --- | --- |
| 80×24 | [SVG](../qa/2026-09-27-console-hook-settings-and-review/review-80x24.svg) | [SVG](../qa/2026-09-27-console-hook-settings-and-review/settings-80x24.svg) |
| 120×40 | [SVG](../qa/2026-09-27-console-hook-settings-and-review/review-120x40.svg) | [SVG](../qa/2026-09-27-console-hook-settings-and-review/settings-120x40.svg) |

All four captures were rendered and visually inspected. Narrow details
scroll while actions remain reachable; the wide view shows the full command,
matcher and timeout. The QA wrapper owns its PTY and asserts the app's size
after interactions.

The first current-dev native attempt stopped before app import with
`recovery_scope_uncertain`. Root-cause tracing showed that current dev's
startup admission uses a recovery control root under HOME; the existing
user root could not enroll a new isolated config. The QA child now has an
actual private home in its private profile, so neither startup admission
nor the app sees the user's recovery state. Both widths then passed.
This is a QA environment change, not an application bypass.

## Review and limits

The current-dev merge retained newer Console, Settings and token-compiler
contracts. The real bound-profile tests verify narrow nested permission
JSON/lock admission, scope restoration, recovery closure and unrelated-file
refusal. Portable backup discovery excludes local consent and refuses
restore input, so restored config requires fresh review. Stale blocking
targets, revocation, failed refresh, legacy duplicate identity and decoder
failure recovery remain covered by the consent suite.

The completed current-dev Settings search run has two failures outside
Hooks: missing index entries for 17 existing fields and stale OmniVoice
form-table entries. The speech failure is identical on an untouched archive
of 857b3dd7d0. The rendered-index test on that archive first hits the
pre-existing unbound-profile guard; adding only the bound-profile test marker
lets the exact base application source reach an **identical** 17-field
assertion. The queue controller's 27 cases all pass. The mounted queue and
search run records 39 passes alongside those two known failures.

A 45-second exploratory queue run timed out on a mounted geometry case after
36 passes. That same case passed alone and again in the completed
120-second UI run; the exploratory timeout is excluded from current counts.

The previous e5ac111967-based integration recorded 11 broader failures
that reproduced on its exact base. Those findings are historical and are
**not** asserted to reproduce on 857b3dd7d0. Only the named green current
runs above are qualified; the full suite and real provider generation remain
unrun. Same-path executable contents are not signed, and revocation
prevents future launches without terminating a hook already running.

## Reproduce

From this checkout with its existing Python environment:

    .venv/bin/python -m pytest Tests/Agents/test_run_hooks.py Tests/Agents/test_hook_permissions.py Tests/Agents/test_hook_config_inventory.py Tests/test_hooks_config_snapshot.py Tests/Chat/test_console_hook_admission.py Tests/Chat/test_console_viewless_hooks.py -q --timeout=120
    .venv/bin/python -m pytest Tests/UI/test_console_hooks_review.py Tests/UI/test_settings_hooks.py -q --timeout=120
    .venv/bin/python -m pytest Tests/UI/test_design_token_governance.py Tests/UI/test_css_build_integrity.py -q --timeout=120
    .venv/bin/python -m pytest Tests/Backup_Recovery/test_raw_participant_lifetimes.py Tests/Backup_Recovery/test_config_lock_inventory.py -q --timeout=120
    .venv/bin/python -m pytest Tests/Chat/test_console_prompt_queue_coordinator.py Tests/UI/test_console_prompt_queue.py Tests/UI/test_settings_search_index.py -q --timeout=120
    .venv/bin/python Docs/superpowers/qa/2026-09-27-console-hook-settings-and-review/check_changed_python.py
    .venv/bin/python Docs/superpowers/qa/2026-09-27-console-hook-settings-and-review/native_check.py 80 24
    .venv/bin/python Docs/superpowers/qa/2026-09-27-console-hook-settings-and-review/native_check.py 120 40

The native script prints the private capture directory. The branch is published as PR #2922; integration is gated on its required checks.

## PR #2922 review follow-up

The branch was already based on the latest fetched dev (`857b3dd7d0`); the
rebase made no changes. Qodo posted seven findings. The installed TOML encoder
skips `None`, so its absent-section failure did not reproduce; a regression
now covers reading and adding the optional section, and the stamp explicitly
omits absent data. The other findings were corrected: central path validation,
Google-style helper docs, typed widget options, clean-draft reloads that preserve
dirty edits, stale revoke/disable validation before local sealing, and the
existing Pydantic-backed shared validation boundary. Unknown saved fields remain
outside the execution projection.

Required local gate checks also exposed post-await modal lookups, queued
indicator refresh during runtime disposal, eager first-paint hook imports, and
boot CSS growth. Dismissal and shutdown regressions cover the guards. Type-only
imports and indicator disk work now stay past first paint. The review modal and
editor reuse the existing lazy Settings stylesheet; leaf classes avoid adding
ancestor bare-type selectors, and the persistent icon reuses its control-bar
styles. All original boot budget limits remain unchanged. The boot worker
census explicitly records the one off-loop indicator snapshot.

The diagnostic inventory was regenerated only after inspecting every new log
statement and the exact private permission-lock sink. Messages remain bounded
and omit command arguments and private paths. Native 80×24 and 120×40 checks
passed all six groups again; the four updated captures were rendered and
inspected. The QA runner waits for actual marker creation after acceptance,
since accepted Send precedes asynchronous hook execution. No full suite or
real provider generation was used.

Final review checks: **185 hook-runtime cases**, **26 mounted UI/admission
cases**, **21 cases in the complete CI boot-budget group**, and
**37 token/CSS integrity cases** passed. The admission cases overlap the
runtime cohort; these are run counts, not an inflated distinct total.
Authored-line Ruff and formatting are clean across 51 changed Python files;
all three QA scripts pass full checks. Generated sheets reproduce exactly.
The worker contract admits no new unsafe lookup, and the reviewed diagnostic
inventory matches the source tree. Named results and final native captures
are retained in the linked QA artifacts.
