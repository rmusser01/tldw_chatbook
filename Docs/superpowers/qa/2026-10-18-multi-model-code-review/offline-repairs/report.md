# TASK-35582 bounded offline repairs

PR3286 remains draft. AC24 remains unchecked: these repairs do not replace the
failed live qualification on47093a3175. No new provider request, product child
launch, live qualification or budget request was made for these repairs.
Cumulative physical spend remains314 requests/21 product children; prior unused
financial capacity does not authorize another invocation.

## Completion recovery

Commit3e5054c5ad repairs the two reproduced completion boundaries under ADR281:

- R2 receives bounded field/type feedback with the rejected submission ID in its
  actual native/fenced model history. It must correct that same ID. Completion
  refuses outstanding rejections; retrying the same completion ID after repair
  preserves strict revision/idempotency and terminal/capture sealing.
- V1 can read a current verifier-only checkpoint after history reduction or cached
  batch replay. The old page remains byte-identical. The checkpoint distinguishes
  pending delivered units from completion of every unit and supplies the current
  revision/cursor, so accepted decisions need not be resubmitted.

Offline regressions reproduce the original failures before repair, then exercise
same-ID recovery to a seal and accepted dispositions through history reduction,
immutable replay, current-state recovery, completion, sealing and exact original
expansion. Native and fenced error delivery, role isolation, bounded results and
multiple batches are covered. The V1 test uses constructed history through the
production projection owner; it is not a live-model compliance claim.

Verification:307 affected tests passed, the later recovery run passed9 cases
including the added multi-batch control, and17 focused post-format tests passed.
These runs overlap and are not additive totals. Compilation and changed-code
format checks pass; there are no new Ruff findings relative to the base. Existing
unrelated lint findings are preserved. The [scoped review](completion-review.md)
found no actionable repair findings; completed feature reviews were not repeated.

Private exact commands, red/green receipts and implementer report are retained in
/private/tmp/task35582-offline-completion/. Prior live receipts and archives are
unchanged. Module-size extraction and its checks follow.

## Module-size repair

Commit32f40af781 moves connected request/admission and continuation-validation
bodies into focused helpers, retaining explicit entry points and original-owner
lookups. It also delegates native permission rechecks, metadata target projection
and OpenAI protocol helpers. The nearby typed display-name getter finishes its
existing lazy delegation to make room in config without a new startup import;
workspace binding lookup drops only an unnecessary temporary. No limits rise.

| Owner under tldw_chatbook/ | Before | Old limit | After |
| --- | ---: | ---: | ---: |
| config.py | 12277 | 12258 | 12252 |
| Chat/console_agent_bridge.py | 12328 | 12093 | 11610 |
| Agents/agent_service.py | 10386 | 9629 | 9621 |
| Chat/console_provider_gateway.py | 8211 | 8184 | 8183 |
| Chat/console_trace_service.py | 7433 | 6674 | 6662 |
| Agents/local_tool_provider.py | 5785 | 5749 | 5738 |
| LLM_Calls/LLM_API_Calls.py | 5217 | 5186 | 5176 |
| Workspaces/registry_service.py | 4233 | 4232 | 4232 |

Seven ratchets are lowered to the new measurements; the registry limit is
unchanged. All eight exact ceilings, eight no-slack checks and the helper-governance
check pass:17 tests. All15 extracted method/function bodies pass a normalized AST
comparison, with all175 qualified lookups still declared on their original owners.

Affected runs passed102 agent cases and247 Chat/provider/config cases. Additional
routing/usage/source-permission checks passed20 cases; workspace binding retirement
passed10. These runs also exposed25 failing/error nodes that reproduce identically
on both the pre-extraction3e5054c5ad and original pre-turn d561a27298 candidates:
15 failures and10 fixture errors. The [exact baseline set](baseline-failures.json)
records those unchanged context-target, sidecar, hook and final-event cases. They
were not repaired or relabeled green. No full repository test sweep was run.

Three targeted startup guards pass: UI-ready1031/1031 modules, preimport558/558
modules with424647/425347 lines, and app import679/686 modules. The three diagnostic
headroom warnings are retained; the first two module budgets have no spare room.
No startup budget changed. Subsequent cleanup only restored original import
spellings/comments with the same modules and bindings.

All changed Python compiles; new helper format and full helper Ruff checks pass.
Full extraction lint comparison reports0 new and684 unchanged baseline findings.
A separate fatal-only check over all26 changed Python files reports three F811
TYPE_CHECKING duplicate imports in console_provider_gateway.py; the same three
exist on original d561a27298. Thus static evidence establishes no new findings,
not a globally clean repository. Whitespace checks pass.

The [scoped extraction review](size-review.md) found no actionable findings and
independently checked all eight counts,15 wrapper signatures/forwarding contracts
and175 retained owner declarations. Existing full
feature reviews and the live acceptance gate were not rerun. AC24, the draft PR
and the failed live outcome remain unchanged by the offline results.

The [preservation receipt](preservation.json) identifies a separate verified archive
of69 private command/review/log files. Test-profile and baseline directories stay
private; no original evidence was removed.
