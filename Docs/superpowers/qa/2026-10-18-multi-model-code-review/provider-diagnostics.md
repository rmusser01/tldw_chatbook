# Provider failure observation integrated

The qualification's HTTP200 receipt precedes response consumption. Its logging
suppression left the later primary exception unavailable. The existing generic
error category therefore cannot distinguish malformed JSON, interrupted reads,
missing terminal metadata, or other worker failures.

`Helper_Scripts/task35582_provider_failure_observer.py` now observes the existing
Console safe-error-copy boundary before projection. The task-owned helper records
run/actor, original exception class/status, at most16 code locations, and a fixed
allowlisted reason code. It retains no exception text, source lines, locals,
credentials, headers, prompts or response bodies. It passes the same credential
set to the original redactor, preserves failure/closure semantics, and restores
the callback through the runner's existing ExitStack. Product provider behavior,
sealing, accepted decisions, permissions, accounting and child limits are unchanged.

The observer and its maintained tests are committed with the PR. A separate
preparation at `/private/tmp/task35582-provider-diagnostics-prepared` has the patch
applied and the helper included in its receipt hashes. That preparation is
explicitly offline-only: live, metadata and composition entry paths refuse before
profile/credential/transport work. No historical authorization receipt was copied,
and the consumed runner's hash is unchanged. Its `integration-receipt.json` records
file hashes and all three refusal results. It is not an authorized next live scope.

Verification: seven integrated offline tests pass through the real Console
gateway, dispatcher, hosted handler and SSE parser with fake HTTP. Distinct error
causes now retain distinct reason codes even when public error copy is identical.
Controls also cover success, actual HTTP502, actor propagation, private-canary
exclusion, observer-write failure and callback restoration. Ruff check/format and
whitespace checks pass; the three offline-only entry guards pass. No full suite
or completed R1 review was repeated. Self-review found no changes to production
contracts; no new ADR is required (existing ADR179/281).

This closes the demonstrated diagnostic gap, not the historical provider failure.
The original exception/body cannot be recovered by these tests. AC24 stays open
and PR3286 stays draft. This integration consumed zero live requests and zero
product children; cumulative actual usage remains412/28. No budget request,
qualification, automatic retry, merge or raw archive duplication occurred.
