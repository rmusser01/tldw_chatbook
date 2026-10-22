# Scoped CI correction review — PR3204

You are a Senior Code Reviewer. Review only the consolidated CI correction after base 007521567a9e961d2cd854432b357b545d8605d4. The parent will supply the immutable exact-range package and current head separately. The whole feature source was independently reviewed previously; this gate concerns new extraction/delegation changes and their evidence.

Read this first — requirements are TASK36006 AC15/16 and Implementation Plan24/25 (in backlog/tasks/task-36006 - Manage-optional-keyword-indexes-in-Settings-and-qualify-release.md), summarized below. Existing ADR241 governs the feature; this correction introduces no storage, authority or dependency choice.

Required behavior: remove the PR-introduced Settings, Console controller and interrupt module-size overruns using cohesive extractions and lower their pins to actual reduced sizes. Lifecycle may not exceed fetched-dev2347 lines; its inherited213-line excess over2134 is explicitly outside this correction. Preserve existing executable bodies and call ownership, controls/order/IDs, full scope/job/epoch fences, current Library choices, initial/retry evidence, review hook late global lookup/monkeypatch behavior, cancellation, and shutdown native ownership. No increased budget or whitespace squeezing. No backend/query/store/scanner/authority changes. No helper should create a circular import, capture controller globals early, spawn work, or shift shutdown to a different caller task.

Binding keyword plan Global Constraints (verbatim):
- Off by default; explicit enablement per local folder binding. SSH indexing is unavailable in this delivery.
- Preserve saved Library controls/scopes; create no Media/Notes records and no global Library search entries.
- Search <=10 results and <=16 KiB serialized output; smaller existing run budgets win.
- Revoke access and publication before cleanup; never deliver stale cached text or restore authority from a matching path string.
- Require expected scope/job state before management effects, and the requesting run's effective exclusion floor before ranking.
- Register storage/activation/maintenance ownership before production opens; no sync/export/ordinary-backup payload for the cache.
- Use only targeted tests unless the user separately requests a full sweep.

Evidence: read the CURRENT CI-correction sections in integration-fix-report.md and referenced ci-ratchet-* raw command/results/body AST/source manifests, plus the hosted failure and exact dev/PR size matrix. Verify wiring and ownership rather than accepting AST equality alone. The earlier same-cache query at source19d is source-qualified; new whole-file hashes are expected after pure extractions. Independently assess that this change did not change query execution before accepting applicability. Diagnostic inventory may change only for exact reviewed moved/removed owners; no unreviewed census repin. Keep adverse setup/broader-ratchet/inherited static receipts in scope as evidence, without treating them as passing executions.

Read-only checkout/index/HEAD; write only the supplied report path. Never dispatch subagents or perform GitHub actions. Do not rerun the implementer's tests on the same source; assess retained raw evidence. No broad source review or full-suite run. Report Spec compliance and Task quality separately, Critical/Important/Minor concrete file:line findings, and every declined-to-judge behavior with a reason. If all clear, give a scoped ready-to-merge verdict qualified by external required CI/queue gates. User requested all issues addressed, including Minor. Return concise status/counts and report path.
