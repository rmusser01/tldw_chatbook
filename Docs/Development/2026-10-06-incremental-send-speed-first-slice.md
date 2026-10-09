# Incremental Send speed — first slice verification

Status: implementation and verification in progress. Overall responsiveness acceptance is not yet established.

The qualified stock MCP composition returns its existing empty inspector result before provider construction when the captured maximum is an exact empty frozenset and no plugin maximum is supplied. The original current-source/current-owner checks remain before the return. Unset/nonempty, custom source/factory, and plugin paths retain ordinary composition.

Base: eeaee0328dfb3ba56b47e6d47a90f8e17fae9a32, including dev integration ca31b1c9f7. This is independent of the main chat's remaining uncommitted UI/UAT changes.

## Verified operation counts

Original-code observation on real private native Windows configuration/permission/catalog storage showed the same empty dispatch and preview reductions:

| Operation | Baseline | Candidate |
| --- | ---: | ---: |
| Provider construction | 1 | 0 |
| Catalog preparation | 1 | 0 |
| Native catalog source read | 1 | 0 |

The focused selection passed all 11 candidate tests after the stock tests failed on unchanged source. Unset/nonempty bounds, a custom empty-set subtype, custom factory/service, changed constructor defaults, inspector live/preview transitions, and supplied plugin maxima are covered. Windows currently raises a POSIX-only O_DIRECTORY import error on the original plugin route; that original outcome is preserved, not converted into success. Positive plugin validation still requires supported-host evidence.

## Wider regression evidence

The seven-file affected candidate selection produced 321 passes and three failures. Two cancellation failures reproduce on unchanged integrated source. The writable-selection case passes in isolated selections on both baseline and candidate; the same mixed-collection failure reproduces on unchanged integrated source. The matching unchanged collection produced 318 passes and six failures: those same three baseline failures plus the three expected missing-optimization failures. No new failure is shown by this comparison. No production cancellation or profile-source guard has been changed.

The six original hook consent/queued-epoch safety cases passed. Hook admission/reconciliation remains unchanged. The controller has the same 55 preexisting Ruff findings as the integrated baseline, with no new finding; the new test file and diff checks pass. Source changes are limited to the one MCP controller method.

## Acceptance still requiring evidence

Rendered acknowledgment within 100 ms, actionable input during preparation/approval/refusal, ordinary Send-to-provider below one second, matched cold/warm full-app samples, and Windows/Linux/macOS acceptance remain open. Unit profiling/counts are not elapsed app-speed evidence. Further changes must preserve fresh consent, durable state, cancellation, recovery, resource ownership, and current worker/history ordering.

Existing cross-platform CI workflow: .github/workflows/console-pause-native-evidence.yml (workflow ID 374820507). Earlier exact-bc740 runs failed/cancelled and do not establish current acceptance. Native timing runs remain serialized with the main chat's app tests.

Governance: existing ADR-126, ADR-134, and ADR-197; no schema, dependency, worker, or consent interface changed in this slice.
