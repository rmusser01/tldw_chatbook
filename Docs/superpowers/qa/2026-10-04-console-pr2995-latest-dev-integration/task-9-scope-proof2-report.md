# Task 9 scope proof2: private decision-helper wrappers

## Measured result

**Structural fit: 29,315 lines, 52 below the unchanged 29,367 cap.** This is a scope proposal, not architecture selection, production implementation, or test evidence.

Approved source: `d3443b9e4297fa20897cf99e77a3ae2c8c562b10`. No tracked Python delta from that source was present before the scan. All proof1 report/map/five-sketch hashes remain unchanged and are recorded in the new map.

| Accounting item | Lines |
| --- | ---: |
| Approved I1 controller | 33,027 |
| Selected original AST spans removed | −4,644 |
| Actual Ruff-formatted retained forwarding wrappers | +657 |
| Actual construction fragment | +280 |
| Original construction statements replaced | −5 |
| **Projected controller** | **29,315** |

Proof1's permitted documentation-relocation version measured 29,381. Removing seven redundant private wrappers and seven corresponding getter arguments saves **66 additional controller lines**. No comments, unrelated imports, initialization, state, source authority or lock were removed to obtain this reduction. The 52 lines are actual projected headroom; any later source growth still has to fit the original cap.

## Exact private cleanup

Only the following seven helper wrappers disappear from the controller. Their bodies and full original docs still belong to the actual `InterruptRoundHost` owner. Their **12 call sites** are all in methods already selected for that same owner.

| Private wrapper | Frozen I1 def/end lines | Current enclosing callers |
| --- | --- | --- |
| `_expire_answerable_decision_if_due` | 16460–16483 | `expire_pending_decisions` |
| `_mutate_exact_pending_decision` | 16372–16387 | `_expire_answerable_decision_if_due`, `_pause_answerable_decision`, `_refresh_answerable_decision` |
| `_pause_answerable_decision` | 16418–16458 | `_refresh_answerable_decision` |
| `_pause_pending_decision_state_locked` | 16359–16370 | `_pause_answerable_decision` |
| `_pending_decision_payloads_locked` | 16344–16357 | `_publish_pending_decision`, `_refresh_answerable_decision`, `pending_decision_projection` |
| `_pending_round_states_snapshot` | 16329–16342 | `pending_decision_projection` |
| `_settle_pending_decision_timeout_locked` | 16390–16416 | `_expire_answerable_decision_if_due`, `_pause_answerable_decision` |

The map has each call's original line/expression and exact proposed owner expression. Six helpers use ordinary direct same-owner calls. `_settle_pending_decision_timeout_locked` remains a static helper; its two call sites supply the existing named live `read_global_threading` getter explicitly. This preserves global patch lookup without adding another lock or moving timing policy.

All public controller APIs, all 17 importable module helper/builder names, all other externally used/borrowed/patched private names, and all module patch routes keep their forwarding contracts. `capture_worktree_recovery_intent` stays unchanged because its `capture_intent(self, session_id)` call requires the actual controller receiver. The exact bound `_maybe_fire_permission_summary` wrapper remains in `after_remount['approval']`.

## Deadness evidence, including computed and reflective routes

A direct grep alone was not used as proof. The following evidence is recorded in `task-9-scope-proof2-map.json`:

1. Frozen-tree exact-name search across **all tracked file types** finds only controller definitions/calls and one historical QA prose citation. No external Python literal/identifier reference exists for these seven names.
2. AST callers resolve every direct use to `self.<helper>(...)` in a selected moved method. No callback extraction, unbound/class receiver, assignment or borrowed call exists for these names. All callers are listed above and in the call map.
3. Prior exact patch/attribute-assignment inventory contains no target among these seven. Other patch routes remain named and late-bound.
4. Broader receiver census covers **427** controller/interrupt/decision-context Python files and records **217 computed attribute sites** plus **44 reflection sites**. The relevant runtime, test and profiler call sites were inspected rather than inferred from a name grep.
5. Runtime computed remount names are a fixed three-name tuple of retained public/private remount seams. Runtime registry/lock and view-hook inventories contain state/callback names, not these helpers. The host's joined kind dispatch uses the five explicit `set_pending_*` names. Buddy/MCP setter concatenations use that same distinct prefix.
6. The real `dir(controller)` probe filters for callable values whose `__self__` **is the screen**. The six instance helpers are controller-bound and the static helper is unbound, so none enters its patch set. Its generic `setattr(controller, name, ...)` therefore does not preserve a hidden patch route to these helpers.
7. Profiling and QA generic method wrappers have finite call tables; their ConsoleChatController entries are other retained methods. Maintenance reflection builds `_close_admission`, `_drain` and `_resume` suffixes, which cannot select these helpers. `vars(controller)` project-instruction checks inspect instance state and a distinct prefix.
8. Package `__all__` and source inventories were considered. The package does not export these private class members; the UI controller attribute-definition guard is scoped to UI/Console_Modules. General diagnostic/fork-writer inventories still require the owner-path updates already identified in proof1, not resurrection of redundant wrappers.

**Compatibility limit:** this is evidence for the frozen repository's routes. It does not establish that arbitrary external plugins, untracked monkeypatches, externally supplied attribute names or eval-generated access cannot depend on a private attribute. Root must accept the normal private-interface cleanup boundary explicitly; this report makes no stronger claim.

## Dependency and ownership contract

Production construction now contains **86 named controller getters, one named write-through callback and 35 named global getters**. Every removed getter corresponds to one of the seven proven same-owner private routes. All other dependency dispositions from the selected boundary remain named, callable and live. No whole-controller receiver, generic name resolver, proxy/mixin, dependency bag or new-lock shortcut is introduced.

The getter for `_observe_chat_creation_record` remains; I1's capture/read/match/record callbacks and critical-section sequence are not pruned. Shared source records, grants, revoked-run fences, launch/acceptance authority, store, cancellation state and the separate chat-create lock stay controller-owned. Native generic registry/payload/lock aliases keep their actual existing objects. Nullable setters retain None; the one direct reassigned decision counter keeps its explicit write-through callback at the original protected point.

Owner-interface static helper signatures now explicitly include the global getter arguments already required by their function contracts. This makes the proposal internally consistent; no production body or signature was edited. The precise binding retargets are in the new map; actual moved-body implementation remains a separately authorized task.

## Documentation and syntax proof

Root ruling44 permits responsibility documentation to follow its genuine owner. The new owner sketches preserve raw `ast.get_docstring(..., clean=False)` values and presence for **all 105 selected definitions**: **101 full docstrings unchanged and four absent docstrings still absent**. No original full text is compressed, hidden in unrelated strings, assigned dynamically to `__doc__`, or discarded.

Retained controller forwarding docs explicitly name **InterruptRoundHost**; module forwarding docs name **console_interrupt_rounds**. These exact docs were formatted and counted. All **98 retained wrappers** preserve original arguments/defaults, decorators and sync/async shape; the comparison found **zero mismatches** beyond the expressly authorized documentation relocation.

Existing **Ruff 0.16.6** formatted the three new snippets. Each was parsed and compiled with Python's built-in parser/compiler; none was imported or executed. No tests ran.

Artifacts:

- `task-9-scope-proof2-controller.py`
- `task-9-scope-proof2-host-contract.py`
- `task-9-scope-proof2-function-contracts.py`
- `task-9-scope-proof2-map.json`
- `task-9-scope-proof2-report.md`

## Remaining gates and limits

Root still must select the architecture and record the canonical amendment to interrupt-host design section 3.2 and its Backlog/implementation plan. The historical narrow host contract does not already authorize moving bridge-specific behavior. Proof2 only establishes a concrete structural fit after this narrowly evidenced cleanup.

Preserve the exact after-remount callback assertion, all retained patch routes, arm-time event identities, original lock order and non-reentrancy. The old minimal fake-seam fixture compatibility still needs explicit, separate wiring with unchanged assertions; production must not regain a whole-controller receiver to satisfy those fixtures. The snippets remain interface/count artifacts, not a complete body transform or behavioral proof.

Loading remains through the existing host module and existing actual execution boundaries; ready **1,033** and preimport **557** caps are unchanged. Final implementation must verify the exact controller count and module loading once, with this worktree's `PYTHONPATH`. The previously identified targeted interrupt/confirmation/summary/review-builder and I1 routes, diagnostic owner inventory and external-writer source census remain the qualification scope. No full Console/provider/schema/encryption sweep is proposed.

Carry proof1 and I1 evidence only where the actual body/argument/lock semantics remain identical, with the seven exact private-binding changes reviewed explicitly. Re-measure the final formatted source; **52 lines of projected headroom is not permission for an overage**.
