# PR3323 backlog integration evidence

Source: `874a134f245ee6d23882d60d9e11ba67484bbc8e`.
Integration base: `a9c1fa9d577f623befe0cc65f94d2a003ac6c8f1`.
Rebased baseline: `fcf27a2558cf4c2e7b33c1f850efabebc1991170`.

Twenty-two source commits were retained. Four size-policy-only commits were omitted under the owner's size waiver; existing dev budgets were preserved. Hosted CI and Qodo were owner-waived for this backlog integration. Focused local verification and independent `superpowers:requesting-code-review` replace those gates; drafts, PR3267/3297 and release work remain excluded.

## Resulting behavior

First-use citation keys are acquired outside SQLite write locks and finite DB callbacks. A retained key-only thread completes a Future; a live caller uses a short identity CAS to adopt a fresh unpublished key ID only when no protected records have appeared. Existing protected records require their original key. Repository identity and codec are published together. Issued finite DB work is drained through cancellation; cancelling a key waiter is not physical cancellation of an OS prompt.

Edits and forks await key readiness, then recheck the original store, controller, message or fork fence. Pending readiness cannot permanently mark an unchanged owner mismatched. Authorization filtering preserves original candidate identity; capture failure sends only authorized, unlabelled evidence without a repair contract. Sources distinguishes deterministic incomplete/stale records from retryable loader failures and waits for actual Textual Mount completion before hydration.

ADR disposition: ADR024 was amended before implementation for key publication and physical ownership. ADR094/097/230 runtime, native custody and physical-contact/retry proof remain intact.

## Evidence

The task-private receipt export records exact test identities, source hashes, outcomes and content-free timing properties. Counts below are separate selections, not an additive total.

| Receipt | Qualified result |
| --- | --- |
| `r-f30d94ab`, `r-57423afb` | Causal failures for keyring under a write lock and footer cancellation waiting on keyring work. |
| `r-154a94c1`, `r-daaea631`, `r-8d0360dd`, `r-e96562e4` | Causal failures for retained rank, labelled fallback, pending edit rollback and incomplete-state copy. |
| `r-509ec561`, `r-983e17c2` | Causal failures for eager modal mounting and store replacement during fork readiness. |
| `r-723f3fd7`, `r-bbec9b3f` | Passing final-form issued-CAS cancellation, retained key completion, singleflight, native count drain and held-key host closure; later fixture failures remain recorded. |
| `r-2b034421` | Seven passes including eager modal, five fork controls and the full staged direct send/Sources/second-send/export/restart journey; agent timeout retained. |
| `r-ceca463c` | Full staged agent journey passed; 93.12s runner, 83.949s JUnit case. |
| `r-b1a350a5` | Automatic agent/direct, unavailable secure keyring and lost original key passed before a new test's NOTE/NOTES header typo. |
| `r-4ac9348c` | Both mounted authorization-shrink/restart adapters and staged retrieval-record failure passed before an environmental admission refusal. |
| `r-18f47f3e` | 30 passed: remaining recording-failure variants, corrected unavailable-key assertion, all 12 edit/fork readiness cases, authorization/rank-bound fallbacks and final-form independent-process publication races. |
| `r-6865037e` | 30 passed before the failed-transformed-call error control: all 16 first-use cases, eight terminal/Stop/retirement controls and six successful-transform controls. |
| `r-b21b7ee0` | 46 passed before the layer-direction guard: native/checkpoint/persistence/fork/migration/artifact/Automatic-failure/modal controls. |
| `r-cf89449e` | 158 passed, six failed, 667.50s runner. Config/default-on, Automatic preparation, terminal persistence, trace proof/retry and census/stale-allowlist checks passed. Four hook-initialization refusals and two transformed-call Stop controls require the exact-base disposition below. All 66 enrolled files matched before/after and the current candidate. |
| `r-306266a0` | All four runtime RAG wiring cases passed after the shared fixture used the real runtime store-composition API; 206.98s runner, 62 stable enrolled files matching the final candidate. |

Earlier `r-9e4f0f23` had 22 passes but predates first-use/count refinements. It is historical evidence, not proof of byte-identical final modules. Final-form proofs are mapped to the later receipts above. Provider, keyring and retrieval doubles qualify their seams, not a real desktop keyring prompt or model.

Final preservation selection, its six-case exact-base comparison and the corrected four-case runtime fixture rerun are complete. Counts are not added across overlapping selections.
Independent review cleared the prior source findings (C0/I0/M0); final commit and evidence review is pending. The source snapshot's 54 file hashes were independently matched before the final fixture-only correction and task reconciliation.

### Non-green preservation controls

`r-6865037e` passed 30 preservation cases, then `test_next_send_after_failed_transformed_call[error]` accepted the next send but refused provider dispatch with `completed_tool_turn_unavailable`. `r-1a2839ef` reproduced the identical failure from a clean source archive of base `a9c1fa9d577f623befe0cc65f94d2a003ac6c8f1` (41.11s runner, stable archive hashes). The test and trace service/runtime/continuation/gateway are unchanged from that base. Temporary assertion diagnostics were restored byte-for-byte; no trace or retry guard was relaxed. This control is explicitly not green, and no claim of a fully green preservation suite is made.

The layer-direction guard in `r-b21b7ee0` reported six `app_shutdown` upward imports. Clean-base `r-ad2461d6` reproduced the exact six violations (146.58s runner, stable archive hashes); the shutdown module, layer test and allowlist are byte-identical to that base. The allowlist and production imports remain unchanged. The existing guard correctly detects the violations; its own completed task is not reopened. TASK38414 records the six imports and evidence with all three acceptance criteria open.

Clean-base `r-0bcbb613` reproduced all six failures from the final selection (251.48s runner, 41 stable enrolled files independently matched to the preserved base archive). The four runtime RAG wiring cases omitted `runtime.set_chat_store(store)` from their shared fixture, so hook preparation captured no runtime store and rejected the controller's different store. This cause is established by source inspection; the swallowed exception itself was not captured. The fixture now uses the real composition API, preserving the hook guard and assertions; all four cases pass in `r-306266a0`.

The other two failures remain non-green: `test_streaming_partial_stop_keeps_verified_assistant_on_next_send[True]` timed out waiting for partial output, and `test_next_send_after_failed_transformed_call[stopped]` recorded `error` instead of `stopped` before the successor. Both reproduce on clean base. Their distinct symptoms and the earlier failed-call successor refusal are retained in TASK32032; no common root cause or fix is claimed.

## Governance

The stock diagnostic validator rebuilt and exactly matched the final pin: 720 owners, 1069 TASK-492 calls, 56 TASK-31551 calls, 7794 TASK-494 calls and 16 sink files. The removed provisioning-specific warning was reviewed; only that owner row required regeneration during final reconciliation.

The stock UI census validator passed with 1315 entries. All 1311 base selectors retain their order; four short readiness/modal entries were appended. The 12 readiness cases totalled 24.982s in JUnit. Long full-send/restart variants stay outside the fast lane; TASK33620.8.9 remains open. No runtime or module budget was raised.

Scoped AST/Ruff review covers 18 Python files: no duplicate class methods and no new Ruff findings; 136 existing findings remain. Nine originally scoped files and the corrected runtime fixture pass formatter checks. This is not a claim that the repository-wide lint or full test suite is clean.

## Rulings and limits

- Use a retained key-only daemon thread with no DB/UI owner; physical OS prompt cancellation is not claimed. Cost: blocked key work can outlive its cancelled waiter.
- Provision fresh unpublished IDs and use a short identity CAS; retain inert losing keys rather than risk deleting a published key. Cost: an interrupted attempt can leave an unused keyring entry. Independent-process writer orderings are required evidence.
- Use a Windows-only 60s first-send readiness observer after the original 20s limit captured live storage initialization and the bounded diagnostic completed the same flow. Other-platform, second-send and modal bounds remain 20s. Automatic agent first send measured 32.969s, direct 10.094s. Cost: a Windows latency regression below 60s could be hidden; no performance improvement or threshold compliance is claimed.
- Keep expensive mounted variants outside the fast census. Cost: their broad hosted-gate coverage remains an explicit follow-up despite required local qualification here.
- Relocate test profiles after the scratch parent acquired sandbox Modify permissions. `r-23b7c0f2` collected zero tests with `shared_writable_parent`; the unchanged formerly refused case passed under a separately created private fixture root in `r-18f47f3e`. No existing ACL or application admission guard was changed. Cost: fixture relocation could obscure a source-dependent scope problem, so the identical case and original refusal are retained as evidence.
- Retain the trace-admission guard after the exact failing retry control reproduced on clean `dev`; record a separate follow-up and complete unaffected preservation checks. Cost: the inherited failed-call next-send blockage remains until separately repaired. Independent review must assess this limitation before merge.

TASK33620.8.1/.2/.3/.4/.8/.9 remain separate: pre-send labels, Temporary-chat records, explicit-false migration, initial vector activation, mid-session key recovery and the broad mounted CI gap. A model can ignore title-based guidance and emit unsolicited citation markers; the guide promises no stripping behavior. No full test sweep, live provider/keyring qualification, release or main-checkout edit was performed.
