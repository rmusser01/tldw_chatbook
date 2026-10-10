# Console hook preparation demand verification

Task: TASK-34563.12. Base: `41e369c1eb09bb82d2892f68e2c8f068341d60d0`.
Decision: [ADR-225](../../backlog/decisions/225-console-send-preparation-and-io-ownership.md).
Plan: [Demand-driven hook preparation](../superpowers/plans/2026-10-07-console-hook-preparation-demand.md).

## Result

A 12-line guard in the existing runtime hook owner removes an unused workspace/context capture. It runs only after original permission/history reconciliation and applicable native-plugin configuration, and only with ready review, no v2 definitions/invalid admissions/native definitions, and no engine or resident owner entry. Retained/closed/injected owners and uncertain configuration retain their ordinary path; the existing disposal/session fence is rechecked before returning.

Source-current operation counts on cold empty hooks:

| Original operation | Before | After |
| --- | --- | --- |
| HookPermissions.v2_configuration | 1 | 1 |
| ConsoleRuntime._hooks_v2_context_key | 1 | 0 |

The passive wrappers called the original bodies. No permissions cache or authority shortcut was added. This removes one operation; it does not establish an elapsed Send saving.

## Verification

On unchanged base, the cold-empty test failed only because it observed one unused original context result. After implementation, **15 targeted cases passed, zero failures/errors/skips**: ten focused controls, four original permission/refusal controls and one actual mounted Send control. Focused cases cover cold absence, retained engine/closed engine/lifecycle/configured owners, actual unapproved/invalid review, post-reconciliation fencing and applicable native-plugin capture. The native service boundary is controlled; real parsed definitions plus original permission/context readers verify branch demand without executing external hook commands.

Independent source review found no actionable issue. Runtime and new test Ruff checks are clean; syntax/diff checks pass. New tests format cleanly; runtime retains its unchanged 38 pre-existing formatter transformations, so this is not a whole-file format-green claim. Existing module caps elsewhere remain unchanged and are documented in the [admission report](2026-10-07-console-received-admission-verification.md). No ratchet was raised or deadline changed.

Both RED and integrated runs retired their contained Jobs normally: emptiness proved, native identity released, pipe/identity monitor retired, private profile removed, no forced retirement/overflow/lookup race. The final run emitted deprecation warnings but no unawaited-coroutine or destroyed-Task warnings. Local detailed receipts and hashes are in `.superpowers/sdd/2026-10-07-console-hook-preparation-demand/{red-audit,final-audit,final-static}.json`; immutable logs/XML/result/custody use `hook-demand-cold-red` and `hook-demand-integrated` in the existing shared-preparation checks directory.

## Limits

This qualifies the demand boundary and original operation count on this source. Older 0.800–1.024 s whole-method spans do not isolate this read's cost and cannot be claimed as saved time. No new timing campaign or full suite was run. Actual rendered/input feedback within 100 ms and subsecond action-to-adapter remain unqualified, as do existing bare prepromotion native-read cancellation lifetimes. Those broader architecture requirements remain open. No primary checkout edit, push or merge is included.
