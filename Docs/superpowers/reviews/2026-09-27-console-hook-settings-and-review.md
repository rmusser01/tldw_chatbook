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

A fresh Qodo reviewer guide found one additional projection gap: an invalid
master switch could publish previously granted notification targets. The shared
snapshot now marks enabled rows invalid whenever the container is invalid,
so no event receives a target and launch revalidation rejects captured work.
Six lifecycle regressions failed before this two-line fix; all **78 consent
owner/admission/viewless cases** then passed. Existing disable/re-enable epoch
and failed-write behavior remains covered. The modal focus style also reuses
the existing shared Button emphasis instead of duplicating its literal rule.


The fresh agentic review found a reproducible same-file profile-selection race:
raw username/data-root changes could retain cached grants during the one-second
external-edit throttle. Hook snapshots now carry the profile data directory
resolved from the same raw TOML under the config lease, using the existing pure
profile-path helper. Consent refuses a mismatch with the bound cache before
opening the exact selected permission store; guarded Settings saves also reject
an intervening profile change. Both old-grant regressions failed before the fix.

Shared strict Pydantic adapters now validate container/list, enabled and optional
ID metadata alongside the execution model. Invalid raw rows and unknown fields
remain available for repair. Loader and consent APIs have complete Google-style
documentation. Failed Disable intentionally stays fenced after Recover, as
ADR-197 requires; a new regression verifies explicit current review restores
permission, and the failure notice states that recovery choice directly.

Current follow-up checks pass: **194 runtime**, **25 config/mounted UI**, and
**12 import/first-paint** cases. These overlap prior named cohorts and are not
added to a distinct total. Authored-line Ruff and formatting remain clean across
51 changed Python files. Existing boot budgets, styles and launch semantics are
unchanged; the complete 21-case boot and 37-case CSS qualification above remains
applicable to its unchanged portions. Hosted checks and merge remain pending.


## Latest-dev rebase qualification

Dev advanced to `2a74675eea` during the hosted queue. The rebase was clean;
range-diff confirms all eight implementation/review commits are unchanged.
The upstream provider-preset additions were inspected at the config, Console
readiness and native-tool seams. On this base, **245 distinct named cases**
pass: 195 hook runtime/config/admission/viewless, 16 mounted review/Settings,
21 complete boot-budget and 13 latency/stall-persistence cases. The current
static check uses this base: 51 changed Python files, no authored Ruff
or formatting failures. All three QA scripts pass full Ruff/format checks.
Diagnostic, profile-path and generated stylesheet checks reproduce exactly.

The prior broader search baseline controls and native captures remain scoped
to `857b3dd7d0`; they were not rerun on this new base. UI/style patches are
identical and the current mounted UI and boot/latency cases above passed.
No full suite or real provider generation ran. Qodo resolved all findings on
the prior published, patch-equivalent head; replies explain the verified fixes
and intentional failed-Disable fence. Hosted checks on the rebased head and
final integration remain pending.


Hosted UI Fast Lane and Perf Guard passed on `72f8ceabd2`, and PR Fast Lane's
main contract passed all 1171 cases. Its admission step found four integration
failures. All four reproduced locally: three teardown fixtures bypassed the
ChatScreen constructor and lacked the Hooks controller; one resume assertion
counted the new indicator refresh as an extra reconciliation retry. Minimal
fixture stubs and filtering the reconciliation callback preserve the existing
detach, claim-release and bounded-backoff assertions. Production is unchanged.

All four regressions pass after the test fix. The exact isolated CI admission
invocation passes **123 cases with one pre-existing expected failure**; no new
skip or xfail was added. Combined with the current-base qualification above,
there are **350 distinct local passing cases** (overlapping named cohorts are
not summed). Authored Ruff/format remains clean across 52 changed Python files.
Native auto-merge was held on the hosted failure; fresh hosted gates on the
published test fix must pass before it is enabled again.


## Latest server-boundary dev rebase

Dev advanced again to `5980da9c12` before the repaired head's CI started.
Its per-session MCP character-write refusal was read at the provider and
Console composition seams, including TASK-33106's accepted ADR-183 policy.
The rebase is clean and range-diff preserves all ten hook/qualification patches.
On this base, **417 distinct named cases pass**: 195 hook runtime/config,
123 exact CI admission cases (one existing xfail), 96 MCP provider/character
composition cases, and 21 complete boot-budget cases. Overlapping viewless
cases are counted once. No new xfail or skip was added.

Authored Ruff/format is clean across 52 changed Python files against this base;
all QA scripts pass full checks. Diagnostic and profile-path inventories and
generated stylesheets reproduce. Prior mounted Hooks UI/latency recordings and
native captures retain their explicitly recorded bases; the UI/style patches
are unchanged. The actual dev ref is checked alongside PR state because
GitHub's baseRefOid can retain the old comparison base while dev advances.
The published rebase still requires fresh hosted gates before native merge.
