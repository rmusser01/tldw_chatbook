# PR3148: bootstrap repair and merge preparation

## Source and scope

The branch was rebased onto dev `69771c1811e734421f0bdf31ed90969f8c1f0c63`.
Qualified combined source: `c9c2af8383ac4151ccec8e188c2c18a19bdfa8aa`.
The later documentation publication changes only this report and task metadata.

All 63 feature commits replayed. Range comparison shows 61 identical patches;
two documentation commits differ only in context. Their testing-lesson conflicts
retain both complete entries. Independent review authenticated the upstream
startup/recovery deltas and unchanged connected plugin authority/custody owners.

The only new test correction adds the existing `bootstrap_profile` marker to
`test_resize_event_width_drives_priority_and_focus` and
`test_console_live_resize_converges_to_cold_start_layout`. Both build real apps.
Without the marker, the fixture changes their already admitted config selection;
the source guard correctly refuses it. Three adjacent controls already use the
marker. Test bodies, helpers, assertions and all production guards are unchanged.

## Verification

| Check | Source / phase | Observed result |
| --- | --- | --- |
| Initial combined selection | `9c4c38311e`, freeze15/ci19 | 39 passed, 2 bootstrap failures, 427.70s |
| Both corrected nodes and three adjacent controls | `be1c9d5fec`, freeze16/ci20 | 5 passed, 51.33s |
| Original selection plus current-dev entry/recovery controls | `c9c2af8383`, freeze17/ci21 | 49 passed, 428.18s |
| Offline derived-artifact checks | `c9c2af8383` | All 11 passed |
| Ruff / changed-range Black | Corrected resize file | Passed |
| Independent integration, bootstrap and latest-dev reviews | Respective immutable checkpoints | 0 Critical / 0 Important / 0 Minor |

The guarded runner selects a fresh private profile before product imports,
retains source-origin checks and external-network refusal, and verifies complete
source inputs before/after each phase. All three runtime phases had unchanged
inputs and zero unexpected outputs. Their exact archives, manifests, commands,
raw failures and outcomes remain under the local
`.superpowers/sdd/2026-10-17-managed-plugins-dev-rebase/` evidence directory.
Source-scope and independent-review receipts are retained under
`.superpowers/pr3148-ci-20261018-02/`.

The initial failing test file was byte-identical to dev. Independent AST checks
confirm that only the two node decorators changed. The corrected file SHA256 is
`57f86a143c28752cd8a1a0af1cd2cc5441b3f5f89a839573af4edd3e5486d9b9`.

## Hosted gates and limits

The original CI repairs were already published and renewed on `7c590d2033`:
required CI, all three native plugin jobs and all three native GGUF jobs passed.
See the [CI repair report](managed-plugins-pr3148-ci-2026-10-18.md) and the PR's
linked completed runs. These remain previous-head results, not evidence that
the newly published head has completed hosted CI.

At this receipt, publication, ready-for-review status, current-head hosted CI
and external-review reconciliation remain merge-preparation steps. Internal
reviews do not replace the repository's Qodo requirement. Auto-merge stays off;
merging requires the user's separate approval.

No full suite, new platform support, real keychain/reboot or vendor-host parity
is claimed. Existing platform skips and previously documented qualification
limits remain. No new ADR is required for this routine test correction and
reconciliation of the existing plugin/Console contracts (ADR162/163).

## Final dev reconciliation and hosted renewal

The user approved the completed independent reviews in place of Qodo for this
PR, then authorized merging into dev once the requirements pass. Automatic
approval review rejected enabling conditional auto-merge while the required
artifact check was queued and the branch was behind. That command did not run.
Auto-merge remains off during the following reconciliation.

Published source `5a24569b4e` passed all 15 required test lanes and all three
native plugin jobs: Linux 461 passed/313 skipped, Windows 308 passed/466 skipped,
and macOS 771 passed/3 skipped. GGUF passed 38 checks on each platform. The
native plugin checkout `b690cdd20e` has the same complete Git tree as `5a24569b4e`.
The final required artifact job was still queued at this receipt; test-lane
success does not establish that final required check is green.

The unarmed branch was subsequently rebased onto dev
`768d58b7564bb2ea6e55a7e0e1f40dbc4016d8fa`, producing qualified source
`5dbd38e85018776df6ba5c70b968a2c69c211225`. All 65 replayed patches are identical.
The three upstream paths are retained byte-for-byte: the notification store,
ADR228 and its task notes. Apart from qualification metadata, no other tracked
source changed. The prior published head remains protected by a backup ref.

Frozen phase `freeze18-dev768-20261019/ci22-notifications-plugin-integration`
passed all 189 selected notification, notification-DB, SQLite kernel/inventory,
Console plugin and cold-start/deferred-import checks in 134.93s. Input manifests
were unchanged, with zero unexpected outputs; 95 bytecode caches were recorded
separately. The existing runner retains all private-profile, null-keyring,
network and native guards. Its historical ancestry pin remains labelled as
historical; current dev ancestry was proved separately.

All 11 offline artifact checks passed on the combined source. A fresh scoped
independent reconciliation review found 0 Critical / 0 Important / 0 Minor
issues and authenticated the replay and notification/kernel/participant/startup
boundaries. This is not a new exhaustive review of the entire feature.

The final report/task publication changes documentation only. Current-head
hosted CI and branch freshness remain merge requirements; previous-head native
results are not represented as new-head passes. Existing ADR162/163/228 apply;
no new ADR, runtime support, full-suite claim or guard relaxation is introduced.
Raw logs, exact replay/source checks and the new review are retained under
`.superpowers/pr3148-publication-retry/` and the frozen phase above.
