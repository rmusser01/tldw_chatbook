# Bounded final qualification — failed acceptance

Source: `47093a3175c59db846a746c70b6f45d8b70cc50c`.
PR: https://github.com/rmusser01/tldw_chatbook/pull/3286, draft against dev.
AC24 remains unchecked; TASK-35582 remains In Progress. This is the completed
single authorized final invocation, not a proposal for another live run.

## Result and first causal boundaries

The runner exited **2**, with `live_report_incomplete_R2`. R1 sealed an empty
report; R2 and V1 remain incomplete. Both reviewers were presented before a
fresh Haiku verifier, with native launches **[2,1]** and one causal chain.

R2 first submitted malformed coverage: two `inspected_refs` entries were objects
instead of strings, and `notes` was forbidden. It corrected coverage under a
new submission ID, so the original rejection remained unresolved even after
`reviewer_complete` was accepted. The narrow offline reproduction demonstrates
new-ID correction stays incomplete, while the same-ID control seals. No schema
or acceptance rule was weakened.

V1 accepted both correct dispositions: the removed empty-input guard is a real
bug, while the claim that strings are silently accepted is refuted. After
history reduction, request 5 had no tool feedback; rereading the initial batch
returned its identical cached revision-0 page despite the revision-1 checkpoint
with no pending units. New wording/new-ID retries conflicted, and V1 never
submitted `verifier_complete`. The offline control reproduces this exact cached
page/checkpoint mismatch and changed-disposition conflict. The earlier generic
project-instruction deferral precedes acceptance; it is not evidence that a
post-commit acknowledgment was replaced. No broad repair follows from this run.

**Mapping and unified output are present:** R2-F1 and R2-F2 each expand exactly
once; the unified report has two readable pages ending with `next_offset=null`.
It preserves the incomplete participant statuses. These useful partial results
are not a successful sealed qualification. The [independent outcome review](outcome-review.md)
records the exact evidence and limits.

## Preflight and spending

The first Haiku preflight used 2 requests and stopped at `invalid_record` because
the private instruction omitted required `schema_version: 1`. That omission was
corrected only in the private prompt. The one permitted manual retry used 3
requests and passed real native read → accepted submission → final response,
with 3 usage payloads settled at the accounting owner: 9,638 tokens, zero reserved.

The final Console run used **54 requests / 3 children**: Qwen 29, Luna 6,
DeepSeek 8, Haiku 11. All 54 HTTP responses were 200; that did not satisfy the
result gate. Total new spending is **59 requests / 3 children**; cumulative
spending is **314 requests / 21 children**. The 321-request ceiling has 7
numerically unused requests (1 preflight, 6 final), but the retry, one final
invocation and all child-launch scope are consumed. **No further live execution
is authorized, and no new allowance is requested.**

Native cleanup completed with no running runs, fleet threads, Workspace
connections/leases or Chat connections; source and temporary team are unchanged.
Accounting for the automatic continuation settled with no reserved/uncertain
rows. [Result and receipt hashes](result.json) preserve attribution and counts.
Existing scripted duplicate/failure/truncation/drift/context/storage controls
remain separately attributed; they are not new live evidence.

## Narrow offline checks and remaining CI blockers

[Boundary controls](boundary-reproduction.json) used zero model requests or
product children. Their first private fixture incorrectly prepared a second
review under an already bound run; it stopped at `run_binding_conflict`. Giving
the second control its own fixture corrected the harness; product code was
unchanged. Both intended boundary reproductions then passed.

The new composer-state CI failure (`own_syncs == ['', '']` instead of `['']`)
did not reproduce in the exact local test: **1 passed in 5.46s**. It is not
claimed fixed. The following module-size CI failures reproduced exactly:
**8 failed in 0.38s**. No budgets were raised and no refactor was begun.

| Module | Lines | Budget | Excess |
|---|---:|---:|---:|
| config.py | 12277 | 12258 | 19 |
| Chat/console_agent_bridge.py | 12328 | 12093 | 235 |
| Agents/agent_service.py | 10386 | 9629 | 757 |
| Chat/console_provider_gateway.py | 8211 | 8184 | 27 |
| Chat/console_trace_service.py | 7433 | 6674 | 759 |
| Agents/local_tool_provider.py | 5785 | 5749 | 36 |
| LLM_Calls/LLM_API_Calls.py | 5217 | 5186 | 31 |
| Workspaces/registry_service.py | 4233 | 4232 | 1 |

CI evidence: [module-size failures](https://github.com/rmusser01/tldw_chatbook/actions/runs/40768235569/job/123104081976),
[composer-state failure](https://github.com/rmusser01/tldw_chatbook/actions/runs/40768235569/job/123104082223).
The prior startup/style corrections, targeted regressions, static guards and
scoped reviews remain valid for their recorded source and scope. The full suite
was not run locally. AC24 and these CI blockers prevent ready-for-review/Done.

## Retention

Raw private evidence remains at `/private/tmp/task35582-bounded-recovery`.
A credential-scanned durable archive and the verified original Git history bundle
are retained under `/Users/macbook-dev/.codex/artifacts/task-35582-bounded-recovery`;
paths/hashes are in result.json. Profiles, databases and synthetic fixture Git
repositories were excluded from the archive; their private originals were not
deleted. The PR carries concise summaries and required fixtures. No merge occurred.
