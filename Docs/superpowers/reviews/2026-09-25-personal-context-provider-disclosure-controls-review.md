# Personal Context provider disclosure controls review

Date: 2026-09-25
Task: TASK-25907.7
Status: Reviewed design explicitly approved; accepted design direction only

Design: [Disclosure controls](../specs/2026-09-25-personal-context-provider-disclosure-controls-design.md)
Decision: [Accepted design ADR-187](../../../backlog/decisions/187-personal-context-provider-disclosure-authority.md)
Tracker: [Memory roadmap](../../../backlog/docs/personal-context-memory-roadmap.md)

## Deliverable and evidence boundary

This task proposes restrictive canonical model policy plus encrypted peer-local
user-reviewed destination/purpose grants, distinct from Sync and agent authority.
It explicitly proposes amending ADR-102's device-only wording, deny migration,
current-source/metadata authority, conservative derivative policy, final-entry
controls, restricted diagnostics and a bounded qualified on-device first release.
It changes only documentation/tracking. No model/schema/fixture, permission,
provider, real profile, migration or UI/runtime behavior changed.

TASK-25907.6's design direction was accepted after the user endorsed it subject
to another review. That review fixed AgentRunsDB/filesystem log coverage and
queued-payload versus final-adapter entry semantics, plus exact fresh authoring.
Those are design requirements, not implemented forgetting or disclosure.

## Technical review and fixes

A fresh bounded read-only reviewer inspected the uncommitted spec/ADR/task from
base `66b0b7cf9c` against all seven criteria and native seams. It reported no
Critical or actionable Minor issue and two Important release-boundary gaps:

| Finding | Written resolution |
| --- | --- |
| Managed prompt-cache binaries/live slots can contain earlier governed input even in a new conversation | ADR-119 owner is explicit. Until complete lineage/inherited policy/current controls qualify Save/Restore/every reuse, those paths remain disabled. Local execution needs an explicitly reviewed clean owned process/slot or refuses; no implicit reset/deletion or binary text search. |
| A local model can disclose facts through general web/MCP/skill/shell/file arguments or publication | General tool catalog/invocation and publication owners are explicit. Calls conservatively inherit governed inputs; external/unqualified tool/publication paths stay disabled. Ordinary tool permissions remain necessary floors, not disclosure consent. Future tool egress needs its own purpose/custody contract. |

The parent verified the second finding natively: AgentService `invoke_tool`
passes model arguments to the registry; `Tools/web_tool_impls.py` fetches the
supplied URL through HTTP. Native inspection also confirmed canonical V1 controls,
context provider/model token estimation, unscoped quarantine flag, interview
syncable filtering, destination snapshots and separate exports/compaction owners.
No provider execution was used to confirm these code facts.

Parent self-review clarified reserved on-device audience enrollment, foreground
policy/grant widening, promotions preserving restrictions, current ADR-185 inline
evidence authority and hidden revision diagnostics without a constant-time claim.
Targeted reviewer rereads confirmed both the managed-cache and tool/publication
fixes. No Critical, outstanding Important or actionable Minor finding remained;
the reviewer found no concrete contradiction and recommended written user review.
This verdict concerns documents only, with no runtime qualification.

## Acceptance mapping

| Criterion | Contract evidence |
| --- | --- |
| 1: Local and explicit audiences/migration | Canonical deny/on-device/reviewed destination ceilings; exact native grants; no grandfathering |
| 2: Egress owners | Context/tools/children/interviews/summaries/embeddings/jobs/history/capture/log/cache/export plus generated tool arguments/publication |
| 3: Route/unknown/mixed/cache handling | Exact destination identity, intersection, new admission on route/purpose change and qualified clean model state |
| 4: Next Send/pre-request privacy | Same prepared destination/purpose, authorized candidates only and user-only explicit consent surface |
| 5: ADR-102 and unsupported existence | Explicit proposed device-only/local/export amendment; remove global quarantine flag; matched negative cases |
| 6: Compatibility/testable slice | Profile-wide version gate, local-only grants, trusted-server limits; bounded new V2/new-conversation first release and synthetic cases |
| 7: Reviewed design only | Proposed status; no current authority changes; native source/document verification only |

## Verification and rollout limits

Native scoped checks passed across 14 owned documents: all 215 local Markdown
links and 19 task reference/documentation paths resolve; 59 original child
criteria remain unchanged; 11 family IDs are unique and original dependencies
point backward. Tasks .5/.6 are Done; .7 remains In Progress with seven technical
criteria checked. ADR-186 accepts design direction; ADR-187 is Proposed. New
spec/ADR contain no placeholders; whitespace and the documentation-only diff
boundary pass. Independent evaluation task/roadmap bytes are unchanged.
The offline provisional ADR allocation checked 559 available refs and 77
worktrees; no competing ADR-187 was found. No fetch or open-PR audit occurred;
integration-time recheck remains necessary. Independent evaluation additions
remain outside these commits. No push, PR or merge was performed.

Synthetic acceptance cases, native adapter/grant qualification, schemas/fixtures,
server compatibility, real profile behavior and model outcomes remain future
requirements. No runtime tests or full suite are called for by this design-only
slice. No current device-only model-egress, indirect-tool, cache, complete legacy
history or physical erasure guarantee is established. Native isolated execution
and documentation checks do not establish any such runtime guarantee.

## Written-design approval

The user explicitly approved the reviewed contract. TASK-25907.7 is Done and
ADR-187 accepts design direction only. ADR-102 now links that future amendment,
without claiming it is current model-egress enforcement. No policy, grant, model
schema or runtime behavior changed; enabling consolidation still requires
shipped verified safeguards. Fresh approval/consolidation checkpoint validation passed across 18 owned documents: 136 local Markdown links and 52 task reference/documentation paths resolve; all original 59 child criteria remain unchanged, 11 family IDs are unique, and whitespace/documentation-only boundaries pass. Independent evaluation task/roadmap bytes remain unchanged. No runtime tests or permission changes were performed.
