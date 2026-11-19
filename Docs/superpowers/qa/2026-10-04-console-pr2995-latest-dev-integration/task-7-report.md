# Task7 — verified provider adoption and Console ownership

## Frozen result

Source/derived HEAD `19f2904496c05682ef20f5f21bd75fc172ed2bae`; integrated dev remains `df2ba424de63576d36d9c3e38387c84285303f1c`. The receipts' `dev=a1571e86…` is an observed shared ref, **not integrated ancestry**. Root plan/Backlog changes are deliberately unstaged. Task8 owns the deferred final startup/public-navigation check and latest-dev integration; Task7 completion remains conditional on that qualification and independent review.

Separate commits:

- `9319d87acf8f6ab3b6255ed50a24fd3bee06dd97`: real both-provider adoption and canonical clean settings rebase, targeted admission fixtures/controls.
- `1ce569523be10db32d07976e694ead217f6f562b`: adoption group moved into existing Session with explicit late-bound dependencies and direct consumers retargeted.
- `d05a98c766d9834a9b87f96397be5309dbe12b1b`: existing hooks modal adapter moved to wiring; actual button worker resolves its controller when it executes.
- `b6990a3a75d68e101fc9f925feafff287b118b31`: reviewed diagnostic owner union only.
- `19f2904496c05682ef20f5f21bd75fc172ed2bae`: one test docstring corrected after qualification.

## Behavior and boundaries

The common adoption path now honors each trusted route's intent type, owner/currentness predicate and provider. Both fixed providers run at mount and resume. The existing canonical rebaser uses a complete clean draft: unseen pairs get established target defaults/support filtering; same pairs retain their snapshot. Prompt, character, pinned prefill and context survive. Configured target base URL stays in settings; verified intent URL remains ephemeral. No durable-default, metadata, claim-recovery or approval policy changed.

Existing current controllers are used directly; an absent controller is ensured first. Summary preparation occurs before compensation snapshots, followed by attached/session/intent rechecks. Controls witness real target adoption and real restored rollback, including cancellation, stale/wrong-type/wrong-owner refusal and a valid dropped TopP=None source for both providers.

Session receives five named callables and reads framework attachment live. HooksController, modal, IDs and CSS are unchanged. The local async button callback retains execution-time lookup, exclusive worker group and existing cancellation/eager-task behavior. Existing ADR 114/117/095/148 and DESIGN §7 apply; no new ADR.

## Qualification

Every link below contains exact argv, before/after source hashes, stable run HEAD, exit and log path. Final commit/source equality is recorded in [freeze map](task-7-final-freeze-map.json).

| Receipt | Result |
|---|---|
| [Final affected adoption](task-7-final-adoption.json) | 40 passed; includes real Llama setup Use flow, both-provider adoption/default/authority/compensation and resume |
| [Hooks affected owner](task-7-hooks-affected.json) | 20 passed, including real modal/cancel/eager-worker and actual button lookup |
| [Session live dependency and resume controls](task-7-session-move-wiring-final.json) | 3 passed |
| [Cap and moved-owner fork census](task-7-final-cap-census.json) | 3 passed; **25,188 lines / 759 methods**, unchanged caps 25,218/759 |
| [Source globals](task-7-source-globals.json) | Three changed modules pass unchanged assertion directly |
| [Worker contract](task-7-worker-contract.json), [UI census](task-7-ui-census.json) | Pass; no pin changes (324 DOM lookups, 68 wait pushes; 133 UI files, floor 130) |
| [Fatal](task-7-final-fatal.json), [format](task-7-final-format.json), [whitespace](task-7-final-whitespace.json) | All nine owned Python paths pass |
| [Final diagnostic guard](task-7-diagnostics-final.json) | Pass: 643 owners, 16 sink files; only three reviewed owner rows changed |

Earlier [route-only 7/7](task-7-provider-route-green.json), [compensation 3/3](task-7-compensation-controls-green.json) and [phase 2/2](task-7-vllm-phase-final.json) remain separate historical receipts. They are not relabeled as the final canonical-rebase/move run.

## Retained failures and warnings

The initial/current and immutable admission failures are retained. [Corrected RED](task-7-authority-corrected-red.json) has 8 failures / 11 passes: six actual route/dispatch failures plus two dropped-sampler refusals. The preliminary authority receipt also includes nine setup/instrumentation diagnostics; those are not claimed as functional RED.

Despite their names, `task-7-functional-adoption-green` **failed 1 / passed 39**, and `task-7-controller-order-green` **failed 2**. The original vLLM rollback injection fired during summary preparation before adoption; [immutable overlay](task-7-original-vllm-overlay-proof.json), [original failure](task-7-original-vllm-rollback-immutable.json) and [trace](task-7-original-vllm-rollback-trace.json) preserve that attribution. Only the named two-parameter fixture was phase-armed after real adoption/rollback returns, retaining all original assertions including calls==2 and successful replay. The new cold-controller spy's incidental ensure-count assumption was separately corrected to assert the semantic baseline/order. The initial DI control failed because its bare object lacked required runtime attributes; corrected test double then passed. [Move diagnostics](task-7-move-diagnostics.json) preserves interrupted checked edits/proof failures; these were not committed defects.

[Initial diagnostic failure](task-7-diagnostics-initial.json) remains unchanged. Root reviewed exactly four fixed-label warning statements and unchanged revision/channel/exception-type arguments; [union proof](task-7-diagnostic-union-proof.json), [root review](task-7-diagnostic-root-review.json) and [pin proof](task-7-diagnostic-pin-proof.json) show all other statements/rows/sink topology exact. No diagnostic fields or waivers were added.

The hooks owner emitted an unsuppressed **FD growth 223** warning (start 14 / end 237 / limit 200) at mounted-owner teardown. No correctness assertion failed; untouched HooksController/modal code and prior mounted-owner warnings support an inherited teardown attribution, not a demonstrated general leak fix. No threshold/filter changed. Existing Git auto-GC/unreachable-object warnings were retained; no housekeeping attempted. Historical native-timeout/current-head external CI merge gates remain root-owned.

## Documentation mapping after qualification

Ruling34 corrected only the stale test docstring at `Tests/UI/test_console_provider_apply_defaults_flow.py:1378`. [Docstring proof](task-7-docstring-proof.json) reverses that one literal and proves the full module AST exact against qualified `d05a98c…` and `b6990a3…`; every executable statement/assertion is unchanged. [Scoped format](task-7-docstring-format.json) and [whitespace](task-7-docstring-whitespace.json) pass. The separate copy commit is `19f2904496c05682ef20f5f21bd75fc172ed2bae`; no behavior tests were replayed for prose.

The earlier three-case dependency receipt precedes the hooks move, so it is carried by exact Session/test bytes and exact Session-constructor AST, with the separately qualified hooks reversal covering its two changed container files. The initial handoff sanity assertion deliberately required whole-file equality and rejected this known move; [handoff proof](task-7-final-handoff-proof.json) records the precise permitted carry mappings. It is not claimed as a runtime failure or whole-file equality.

## Preservation and review surface

- [Final test-source proof](task-7-final-test-source-proof.json): all original assertions retained; exact bodies after helper/request/consumer reversal except the two explicitly recorded entered-adoption instrumentation functions.
- [Session move invariants](task-7-adoption-move-invariants.json), [method/dependency map](task-7-adoption-move-proof.json), [hooks move invariants](task-7-hooks-move-invariants.json): exact AST after declared dependency/receiver reversal, unchanged nonmoved methods and full retargeted test-module reversal.
- [Final preservation](task-7-final-preservation.json): all 11,572 historical QA blobs exact, prior 33-commit mapping retained, explicit source exceptions. Unchanged schema/encryption/ledger/reader owners carry prior qualification by identity; no repeated 74 model contracts/five UI / 42 schema/full census run.
- [Freeze map](task-7-final-freeze-map.json): full committed source/test hashes, per-receipt source equality, every final method and caller location. Earlier receipts retain their actual source and HEAD even when they differ from final.
- [Safe manifest](task-7-safe-evidence-manifest.json): 362 exact JSON/log/XML copies under `task-7-safe-evidence/`, each with absolute original/copy paths, SHA256 and byte size. No profiles/config/database/cache copied.

Self-review: examined current-controller/preparation/snapshot order, exact-current claim and session fences, entered rollback/cancellation, provider-aware settings isolation, all actual mount/resume/direct consumers, execution-time hook lookup, untouched controllers/CSS, diagnostic union and cap arithmetic. Independent review should focus on the functional adoption helper, five Session dependencies, two dispatcher retargets, hooks adapter/button branch, two phase-instrumented fixtures, new both-provider controls and mechanical reversal proofs. Task8 must attach final startup/navigation outcomes before publication.
