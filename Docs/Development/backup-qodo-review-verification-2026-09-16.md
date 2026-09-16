# Backup review verification — 2026-09-16

PR [#2642](https://github.com/rmusser01/tldw_chatbook/pull/2642), TASK-32628.
The review fixes are in `6270d26f15`, directly on the rebased PR's dev
base `657f70ffe7`. Fixture/CI corrections through `290dff72fb` leave all
application and packaging source unchanged. This report records that tested
revision; later documentation/task updates are not relabeled as native runs.

| Platform | Review verification |
| --- | --- |
| macOS | 367 distinct targeted cases have passing evidence across the broad and corrective runs; all observed failures were corrected. The fixture/workflow follow-up passes 11 cases, and the final installed F9 plain journey passes in 70.39s. |
| Linux | 96/96 product/regression cases pass in 117.57s; no failures, skips or unfinished cases. |
| Windows | [Run 35111695348](https://github.com/rmusser01/tldw_chatbook/actions/runs/35111695348) passes 96/96 product/regression cases and 59/59 native filesystem/security checks; no failures or skips. |

Both remote native runs use exact committed source. Linux verifies all 16,915
source files and membership before/after execution, and all 2,506 packaged Python
files against the source manifest, wheel and install. Windows verifies all
21 artifact hashes, 16,915 source files (allowing only
native CRLF export conversion), and all 2,506 packaged Python files.
Hashes and the precise per-run scope are in the companion JSON; raw Linux logs
remain on the authorized test host.

Qodo reports zero bugs, rule violations and cross-repository conflicts on the
corrected revision; all nine outstanding threads are resolved. Independent
reviews cover the publication and admission boundaries, exact voice selectors,
provider validation, failed-staging cleanup, and fixture corrections. All 24
touched Python files parse, Ruff has zero new findings (43 existing in that
scope), and Bandit reports zero findings.

The first Windows review run passed 59 native and 95 product cases. Its remaining
fixture attempted a legacy replace onto a held destination; the corrective test
uses two native renames and requires the exact injected exception, retaining its
foreign-file identity/content assertions. The next run passed that correction but
exposed the UI fixture reading config before the real consent-save callback
finished. The fixture now observes completion of the original callback before
asserting persisted values. The complete 96-case repeat above passes. Windows
source CI also uses a verified source copy beneath the real user
Temp directory and a bounded 180s full-app timeout; production guards remain intact.

This is finite review-regression verification, supplementing the earlier
[selectable-group qualification](backup-selectable-groups-verification-2026-09-15.md)
and [first-time/power-user UAT](backup-uat-remediation-2026-09-13.md).
Those historical runs keep their original revisions and limitations. In
particular, the earlier Include Credentials journey acknowledged unavailable
keyring entries and did not establish a complete keyring-secret roundtrip.
The PR's protected-check and merge state is the authoritative integration record.
