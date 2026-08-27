# SDD ledger — plan: Docs/superpowers/plans/2026-08-26-console-thinking-blocks-provider-history.md

Setup: isolated worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/console-thinking-blocks`, branch `codex/console-thinking-blocks`.

Dependency: TASK-18932.1 is complete at `9906b4d4bd`; TASK-18932.2 consumes its review-clean envelope, generation ownership, replay policy, Sync, and persistence capability contracts.

Task 1: BASE `9906b4d4bd`; bounded streaming splitter plus compatibility wrapper only.

Task 1: implementation complete at `cd52c97035`; reported 94 focused passes, 5 caller-regression passes, Ruff format/check, and diff-check clean. Fresh review pending.

Task 1: initial review needs fixes (2 Important: bound/terminally clear every parser-owned state, including leading whitespace and overflow; do not fabricate failed thinking from an unconfirmed partial opener; commit `cd52c97035`).

Task 1: fix round 1/5 implemented at `f284bb5438`; reported 516 focused passes, 5 caller-regression passes, Ruff format/check, and diff-check clean. Re-review pending.

Task 1: fix round 1/5 re-review closed both original findings but opened 1 new Important: manual UTF-8 counting accepts lone surrogates that durable envelope validation cannot encode (`f284bb5438`).

Task 1: fix round 2/5 implemented at `4e5a04c2d2`; reported 681 focused passes, 5 caller-regression passes, Ruff format/check, and diff-check clean. Re-review pending.

Task 1: complete (commits `83b1fe3b2b`..`4e5a04c2d2`, review clean after 2/5 fix rounds).

Task 2: BASE `4e5a04c2d2`; explicit provider events, adapter disposition, and adapter-owned persistence-preflight fact only.

Task 2: implementation complete at `5bb018c7e3`; reported 568 provider passes, 692 selected splitter/capability passes, Ruff format/check, and diff-check clean. Implementer-internal review fixed 4 issues; root independent review pending.

Task 2: root review needs fixes (2 Important: direct local paths must obey frozen disposition so ignored/non-thinking resolutions cannot emit evidence or bypass capability preflight; all event provenance fields require strict UTF-8 validation; commit `5bb018c7e3`).

Task 2: fix round 1/5 implemented at `33cbd31a5e`; reported 597 provider passes, 692 selected splitter/capability passes, Ruff check, and diff-check clean. Formatter baseline exception documented; root re-review pending.

Task 2: fix round 1/5 re-review closed strict UTF-8 and the primary direct paths but left 1 Important open: displayable vLLM auxiliary completions do not disposition-normalize their result. The committed report also contradicts itself about formatting. Fresh baseline probe: production gateway was Ruff-format clean at Task 2 base `4e5a04c2d2`, while the large gateway test already was not; current production gateway must be formatted, and the report must state the test-file baseline accurately.

Task 2: fix round 2/5 implemented at `989f48e5e1`; reported 604 provider passes, 692 selected splitter/capability passes, production format, Ruff check, and diff-check clean. Gateway-test baseline format exception documented; root re-review pending.

Task 2: fix round 2/5 re-review closed every code correctness/privacy finding. One evidence-only Important remains: exact base replay with `--stdin-filename` proves both gateway files were already whole-file Ruff-format non-green at `4e5a04c2d2`; the report must remove contrary claims while retaining current production-format pass and test-file baseline exception.

Task 2: fix round 3/5 evidence correction committed at `1a3774cf10`; both base files accurately recorded formatter-non-green, current production format/Ruff lint/diff checks green. Final re-review pending.

Task 2: complete (commits `4e5a04c2d2`..`1a3774cf10`, review clean after 3/5 root fix rounds; implementer-internal review fixes preceded root review).

Task 3: BASE `1a3774cf10`; round-owned accumulator and direct/agent consumer integration only.

Task 3: implementation complete at `876d4d2622`; reported 138 focused passes, 457 adjacent passes with 4 expected skips, 57 persistence/variant passes, Ruff format/check, and diff-check clean. Root independent review pending.

Task 3: root review needs fixes (3 Important: direct Stop must retain a typed evidence item already yielded before the cancel poll; detached agent Stop needs a durable post-detach terminal projection for late evidence; agent retry capability preflight must precede generation-clearing mutation; commit `876d4d2622`).

Task 3: fix round 1/5 implemented at `b06e2b4080`; reported 478 owned passes, 457 adjacent passes with 4 expected skips, Ruff/scoped format/diff checks clean. Root re-review pending.

Task 3: fix round 1/5 re-review closed direct yielded-item Stop and preflight-before-mutation, but detached late settlement remains 1 Important open: check-then-act races controller terminal mutation/commit and can lose evidence or create competing optimistic projections. Safe-case test was serialized and missed both interleavings.

Task 3: fix round 2/5 implemented at `428415970c`; reported 3 barrier cases green, 481 owned passes, 457 adjacent passes with 4 expected skips, Ruff/scoped format/diff checks clean. Per-owner reentrant settlement coordination; root re-review pending.

Task 3: fix round 2/5 re-review confirms same-generation schedules A/B but opens 2 Important findings: detached settlements need a store-owned generation epoch/token, not message-ID-only fencing, so old workers cannot mutate restored/newer generations; the keyed RLock registry needs scoped waiter-aware cleanup. Barrier competitor acknowledgement should also become deterministic.

Task 3: fix round 3/5 implemented at `6edf218638`; reported 9 combined fence/schedule cases, 485 owned passes, 457 adjacent passes with 4 expected skips, Ruff/scoped format/diff checks clean. Store-owned monotonic generation tokens and scoped waiter-aware lock registry; root re-review pending.

Task 3: fix round 3/5 re-review closes scoped-lock cleanup and reviewed stale retry/restore cases but overall generation fencing remains open at superseding seams: content edit, dispatch-recovery reset window, add/select variant must invalidate/advance under owner scope before mutation; continuation-discard removal must clear token runtime. Existing schedule barriers should wait for registered waiter count.

Task 3: fix round 4/5 implemented at `2fd5002703`; reported 8 combined fence/schedule cases, 490 owned passes, 65 dispatch-recovery passes, 457 adjacent passes with 4 expected skips, Ruff/scoped format/diff checks clean. Superseding/removal seam inventory fenced; root final re-review pending.

Task 3: fix round 4/5 re-review closes the replacement/removal inventory except one dispatch rollback Important: a pre-token preflight/refusal release unconditionally invalidates the unchanged prior generation token. Final round must invalidate only the exact replacement token actually issued; refusal before replacement preserves the prior token and late same-generation evidence.

Task 3: fix round 5/5 implemented at `3e8fef02a2`; reported 4 new rollback cases, 494 owned passes, 65 dispatch-recovery passes, 457 adjacent passes with 4 expected skips, Ruff/diff checks clean. Exact optional replacement-token rollback; final re-review pending.

Task 3: fix round 5/5 (3 addressed, 1 open — generic accepted-turn/postcommit/Stop dispatch rollback callers do not retain/pass the replacement token; commits `2fd5002703`..`3e8fef02a2`). Breaker tripped.

Task 3: Ruling: the remaining dispatch rollback finding is real and load-bearing. Do not run a sixth Task 3 round. TASK 4 must begin by making the dispatch recovery owner retain the exact optional replacement token once issued and pass it through every generic release/restore path; pre-token refusal preserves the prior token, post-token rollback invalidates only the matching replacement, and newer tokens survive. Expand Task 4 ownership narrowly to `console_chat_store.py` plus focused controller tests for this prerequisite — this is the smallest safe change because Task 4 already modifies the controller/prepared-request pipeline — cost if wrong: history replay would build on a rollback path that can attach a detached attempt's thinking to a restored generation.

Task 3: complete (commits `1a3774cf10`..`3e8fef02a2`, 1 load-bearing finding carried by ruling into Task 4 at breaker cap).

Task 4: BASE `3e8fef02a2`; first resolve the carried dispatch rollback ruling, then implement owner-atomic replay/prepared-request projection.

Task 4: Ruling: expose normalized `thinking_history_policy` on `ConsoleChatSession` and round-trip it through session creation/hydration/state using Auto for missing/null values. Both direct and agent preparation read that live conversation-owned value. Do not substitute an ad hoc persistence lookup: the spec and Task 4 require session-conversation ownership, and a DB-only lookup would miss temporary sessions and later in-session edits — cost if wrong: this narrowly expands the already-owned store/session DTO, but keeps one authoritative live owner for the later UI child.

Task 4: implementation complete at `00684e7953`; carried recovery-token prerequisite committed separately at `5f31efc6d6`. Root independent review opened 1 Priority-1 lifecycle finding (successful direct/agent/recovery generations retained current attempt tokens, leaking settlement authority) and 1 Priority-2 projection finding (a shared raw continuation/thinking owner marker was popped twice).

Task 4: review fix round 1 implemented at `eff71a5635`; exact compare-and-retire now reclaims only the still-current attempt, direct terminal tasks retire after settlement, agent bridges retire after capture/worker settlement, detached Stop retains authority only until late evidence settles, and shared raw owner markers attach continuation plus thinking independently. Reported 5 review-focused passes, 378 history/prepared/provider passes with 2 expected skips, 489 controller/agent passes, 1 detached-Stop lifetime pass, scoped Ruff format, all-changed-file Ruff lint, and diff-check clean. Root re-review pending.

Task 4: review fix round 1 re-review closed the shared-owner projection finding but left 1 Priority-1 lifecycle finding: direct issuance preceded fallible setup outside its retiring `finally`, and agent issuance/worker scheduling lacked continuous explicit ownership before the bridge worker actually started.

Task 4: review fix round 2 implemented at `1ed4531730`; direct fallible post-issuance setup now retires locally, normal agent issuance moves after preparation/refusal, and a lock-protected controller-to-bridge handoff makes cancellation versus worker acceptance atomic. Preissued recovery tokens remain controller-owned until positive bridge acceptance; accepted workers preserve the existing detached Stop/late-evidence retirement owner. Reported 9 focused lifecycle passes, 498 controller/agent passes, 378 history/prepared/provider passes with 2 expected skips, Ruff/diff checks clean. Root re-review pending.

Task 4: review fix round 2 re-review READY. Independent reviewer reproduced 11 focused lifecycle/Stop cases and verified continuous exact-token ownership, atomic controller-to-bridge handoff, newer-token preservation, reclaimed runtime state, shared raw-owner coexistence, and legacy no-token compatibility.

Task 4: complete (commits `3e8fef02a2`..`a95bc875aa`, review clean after 2/5 fix rounds; carried Task 3 breaker ruling resolved by `5f31efc6d6`).
