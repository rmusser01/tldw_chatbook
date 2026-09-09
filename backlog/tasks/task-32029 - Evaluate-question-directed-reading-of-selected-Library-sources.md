---
id: TASK-32029
title: Evaluate question-directed reading of selected Library sources
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-08 04:38'
updated_date: '2026-09-08 06:18'
labels: []
dependencies: []
documentation:
  - backlog/decisions/133-question-directed-library-reading-experiment.md
  - Docs/superpowers/specs/2026-09-07-library-source-reader-design.md
  - Docs/superpowers/plans/2026-09-07-library-source-reader.md
  - backlog/docs/library-source-reader-local-qualification-2026-09-08.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Determine whether a constrained smaller-model reader can reduce total cost for questions over selected Library documents and transcripts while preserving evidence accuracy and useful coverage. Compare it with direct reading and source-scoped retrieval before committing to automatic routing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The three comparison paths use the same pinned sources and questions and record their exact submitted evidence.
- [x] #2 Reader results contain only source-validated quotations and distinguish unsupported findings, incomplete coverage, and execution failures.
- [ ] #3 The report includes all model usage, cache effects, latency, answer-quality review, and an explicit adopt, revise, or reject decision.
- [x] #4 The experiment preserves current Library access, provider destination, cancellation, and citation ownership boundaries.
- [x] #5 Worker wire requests contain no tools or unrelated context and run through the sensitive auxiliary gateway; empty, foreign, or stale selection is rejected before content/model dispatch.
- [x] #6 Failed answers and unknown usage cannot improve the financial result; paired comparisons preserve the eligible case set and report retrieval embedding/reranker costs and cache conditions.
- [x] #7 Timeout or cancellation stops further dispatch while any still-running adapter remains accounted for; late content is discarded and unobserved spend remains unknown.
- [x] #8 The experiment uses nonsensitive fixtures and disposable databases; it does not register a Console/MCP tool or write to the user's conversations, fleet history, memory, or citation store.
- [x] #9 Development reader instructions preserve quoted explicit negative facts and statements of unspecified information, while genuine lack of relevant evidence still yields no_evidence_found without an automatic fallback; focused live probes are kept separate from the frozen comparison and do not replace human grading.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Implement bounded source assembly and exact-evidence validation with real SQLite tests. 2. Implement isolated auxiliary requests and owned deadlines with recording-server tests. 3. Implement paired comparison accounting and the fixture CLI. 4. Qualify local boundaries and record remaining live evaluation. 5. Continue with the user-selected DeepSeek provider: Pro main and Flash reader, explicit non-thinking Chat Completions, current peak/off-peak pricing, frozen run metadata, and targeted transport verification before model calls. Prepare a complete capped development run; retain missing retrieval/human grades as inconclusive. ADR required: no new ADR. ADR paths: backlog/decisions/133-question-directed-library-reading-experiment.md and backlog/decisions/064-deepseek-dual-api-provider-boundary.md. Reason: execute the approved provider isolation contract through the existing provider adapter; no new storage/runtime owner.

6. Approved bounded revision: use retained dev-03/dev-04 live failures as baseline evidence; clarify reader relevance for explicit negation, uncertainty, and false question premises without changing the JSON schema or runtime behavior. Run existing targeted reader/transport tests, then use at most four remaining approved requests for one worker/main probe per failed development question. Preserve the original full-matrix run and all held-out questions. Record cumulative spend below $1 and cumulative requests at most 24. ADR required: no. ADR path: backlog/decisions/133-question-directed-library-reading-experiment.md (existing). Reason: prompt clarification under the existing evidence/provider contract; no new storage, ownership, or fallback policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the fixture-only source reader under ADR-133: bounded revision-pinned Library assembly, strict exact-quote validation, isolated auxiliary requests with owned deadlines, paired comparison accounting, and an offline-first CLI with fixed development/held-out fixtures and blind human grading artifacts. Known portions of failed or partly observed calls and late usage remain visible without making unknown totals look complete.

Validation: 143 targeted tests passed across the five Tests/Evals/test_source_reader* modules, including real SQLite and loopback HTTP through the actual provider adapters. Ruff lint and format checks passed for all eleven added Python files. CLI help, prepare, and report completed offline; the generated report is inconclusive. No full suite or paid provider calls were run. The environment emits an existing Requests dependency warning.

Review fixed malformed Unicode fitting, ignored model caps, an overridable deadline, omitted partial/late spend, malformed normalized usage, grading-fact drift, and an explicit reasoning setting that changed the OpenAI API route. A suspected external-database-link issue was retracted after the shared private SQLite boundary rejected both symlinks and hardlinks before connection.

The overall evaluation remains In Progress: live model/endpoint/budget selection, fixture retrieval index and its accounting, held-out runs, and blind human grades remain outstanding. CLI retrieval currently reports index_unavailable. Requests record the auxiliary boundary, not final HTTP bodies; canonical product citation types/persistence remain a later integration requirement. No Console tools/settings or live Library/conversation/fleet changes were made.

Implementation: tldw_chatbook/Evals/source_reader/, Helper_Scripts/Benchmarks/library_source_reader.py, five targeted test modules, and Tests/fixtures/library_source_reader/corpus.json. Design, ADR, implementation plan, and local qualification report are linked in task documentation. Incident added to lessons-live-verification.md.

Additional verification caveat: four existing media namespace cases in Tests/DB/test_core_sqlite_owner_privacy.py fail because they expect PrivatePathError as the deepest cause while the unchanged MediaDatabase constructor raises DatabaseError from None. Independent disposable symlink/hardlink constructor probes confirmed zero raw SQLite connections and unchanged targets. No shared database code was changed; see the local qualification report for the exact command.

DeepSeek continuation: user selected the provider; prepared Pro answers and Flash reader at the official endpoint with thinking explicitly disabled through the native adapter. Added an experiment-only bridge validating cache-hit/miss usage, time-aware peak/off-peak prices, and a no-dispatch dry-run mode. Existing native callers retain their default behavior. ADR-133 and ADR-064 apply; no new runtime owner or ADR. Independent review fixed pricing-boundary calls stopping the matrix; complete bounded costs now continue without becoming exact costs or known subtotals, while incomplete usage still stops. Verification: 163 targeted source-reader tests, three existing DeepSeek chat tests, and Ruff lint/format checks passed. Authenticated model listing confirmed both IDs. Reviewed preflight reserves 24 development calls and 240,000 tokens, with a conservative peak-price estimate of $0.9504 beneath a proposed $1 cap. Manifest: /private/tmp/library-source-reader-qualification-20260908-01/deepseek-development-preflight-reviewed/run_manifest.json. No paid generation occurred. Operator spend approval, fixture retrieval accounting, actual outputs, and blind human grading remain pending. See the updated local qualification report for details and pricing sources.

User approved the prepared DeepSeek development comparison: at most 24 requests and $1 total. Execute the reviewed Pro/Flash non-thinking configuration against nonsensitive development fixtures; retain the frozen held-out set, unavailable retrieval rows, and missing human grades. Do not expand the approved call count or spending ceiling.

Approved live DeepSeek development run completed: 20 of at most 24 calls, estimated normalized model cost $0.003328 under the $1 ceiling. All eight direct attempts returned answers; four of eight reader attempts returned no findings and skipped their main call. The four completed reader pairs cost 1.33–1.74 times direct reading and added 0.64–1.65 seconds. Failures repeat on absent-information and no-approval questions; revise negative-evidence handling before held-out evaluation. Formal report remains inconclusive because retrieval is unavailable and blind human grades are missing. No savings/adoption claim is supported. Independent artifact audit verified matrix/source hashes, quote spans, all 20 usage records, and exact reconciliation of call/attempt totals. Frozen nonsensitive artifacts are preserved under backlog/docs/library-source-reader-runs/2026-09-08-deepseek-development/, with report at backlog/docs/library-source-reader-deepseek-development-2026-09-08.md. Existing ADR-133/064 apply. Updated plan, qualification report, spec status, and live-verification lesson. No runtime code was edited during the run, no held-out request dispatched, and no additional paid call made. Task stays In Progress for reader revision, retrieval accounting, held-out evaluation, and human grades.

Approved prompt-only negative-evidence revision completed under ADR-133. Reader instructions now preserve explicit negative facts and unspecified-information statements, including facts that refute the question premise, while forbidding unsupported claims from silence and preserving epistemic scope. No schema, validator, main prompt, provider, deadline, retry, or fallback change. A prewritten behavior check failed against the original live artifacts, then passed on both targeted questions with required exact quotations and nonempty main answers. Focused probe used four remaining approved requests, estimated $0.001794; cumulative24 requests/$0.005122. All calls had normalized usage. Probe used peak pricing versus the off-peak baseline, so no comparative savings claim. Original artifacts, full matrix, model resolutions, and answer prompt remain unchanged; held-out questions not sent. Verification: 47 affected reader/runner/local-HTTP tests and Ruff lint/format checks passed; existing Requests warning, no full suite. Report: backlog/docs/library-source-reader-negative-evidence-revision-2026-09-08.md; frozen nonsensitive artifacts in backlog/docs/library-source-reader-runs/2026-09-08-deepseek-negative-evidence-probe/. Full development rerun with positive controls, retrieval accounting, held-out comparison, and blind human grading remain pending; the original 24-request allowance is exhausted. Task stays In Progress.
<!-- SECTION:NOTES:END -->
