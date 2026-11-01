# Task 9 controller-cap preflight

## Status

**Candidate identified; conditional recommendation, not yet selectable as a proven implementation scope.** No tracked source/tests/plan/index/HEAD mutation, no tests or agents. Only this report and its JSON map were written. Root must freeze I1, select/record scope, and prove the explicit wiring fits before implementation approval.

Frozen source: `40b9aacaa21f7e8c6f2b1989efbb8e8021a1fe77` (32,910 lines). Latest captured working source: 33,027 lines, SHA256 `8a9b447cdebd826ea310745f84370c0fe5dcade79b8cbdfa9acb4b081186a9e0`. The controller changed while inspected; carry current I1 separately, not by shifting frozen citations.

## Recommended owner and alternatives

Recommend extending **`InterruptRoundHost` in `tldw_chatbook/Chat/console_interrupt_rounds.py`** for the human-decision lifecycle: request/resolve bridges, answerable/FIFO projections, permission summaries, and review-hook verdict construction. It already owns generic registries, retained payloads, the non-reentrant lock and round lifecycle. This is a runtime/business owner with no DOM. An arbitrary new UI controller would not improve ownership.

The exact set is **89 controller members plus 17 module functions** in `task-9-cap-preflight-map.json`, with decorated-start/def/end spans, source and AST hashes, direct attributes, globals and local imports. Source intervals are not sufficient: three watchlists receipt methods are explicitly excluded; `_enrich_chat_create_confirm_payload`, `_notify_run_hook_approval`, `_console_tool_kill_switch_reader` and three closed-round stamp helpers are included for their decision role.

Second candidate, rejected: expand `ConsoleChatStartCoordinator` to own chat creation. Its responsibility is bounded launch of an already durable chat; creation/source validation/acceptance is a separate authority, and that extraction alone is too small. Do not copy its whole-controller constructor pattern or transfer launch/source authority merely to reach the cap.

## Exact spans and honest line projection

- Cap remains **29,367**, ready cap **1,033**, preimport allowance **557**.
- Frozen selected AST spans: **4637 lines**; latest current selected spans: **4649**.
- Preserved signature/decorator lines: **271**. Signatures plus three body/layout lines per callable: **589**, a planning allowance, not formatted source.
- With another 250 lines for named controller-side wiring, loading and wrapping: `33027 - 4649 + 589 + 250 = 29217` (150 below cap).
- With 350 instead: **29317** (50 below cap).
- **250 is optimistic, not supported by an exact wiring artifact. 350 is not a proven ceiling either.** The enumerated dependencies below make the risk concrete. The candidate must not be selected on line arithmetic alone. A formatted named wiring/delegator sketch must fit; otherwise root revises the boundary. Never collapse dependencies, cut documentation, raise caps, or weaken assertions to make it fit.
- No initialization/comment/import removal is counted as savings. All moved documentation follows its responsibility. Final measurement must use the frozen I1 tree and the actual formatted implementation.

## Ownership boundaries

Existing host-owned generic registries, payloads and shared lock stay host-owned, with historical alias identity preserved. Decision-only bookkeeping candidates are `_pending_approvals`, `_pending_round_kinds`, `_question_bounces`, `_pending_decision_order`, both answerable maps and announced-decision IDs. Even these require the writer map check; the projection does not rely on an initializer/state move.

Keep `_chat_creation_records`, `_chat_creation_revoked_runs`, `_chat_create_session_grants`, `_pending_chat_create_lock`, `_chat_start`, source observation/record admission, `_agent_bridge`, store, close generations, cancellation events and durable execution authority in their existing owner. The moved **confirmation** accesses them through explicit named live callbacks/getters. Do not merge chat-create's distinct lock into the host's generic lock or move source/launch acceptance into the host.

JSON `attributes` lists each attribute's moved users, remaining users and direct stores. Direct stores are not proof of exclusive ownership: aliases and dict mutation require per-site review. Shared state never moves merely because a moved method reads it. Any retained public state forwarding must accept writes and write through to the actual owner.

## Enumerated dependency cost and routing

Selected member bodies reference **37 moved-method names**, **8 retained controller methods**, and **55 state/service/callback attributes**. Every name is in `dependency_groups` and its sites/users in `attributes`.

The retained-method set is: `_advance_lifecycle_revision`, `_bind_round_cancel_signal`, `_bind_visit_cancel_signal`, `_chat_creation_record_locked`, `_is_session_cancelled`, `_provider_messages_for_session`, `_revoke_chat_create_rounds`, `_run_hooks_engine`.

The 55 state/service names are not 55 new injected arguments: existing host registry/lock aliases remain native ownership, while shared values and attach-time callbacks require separate named live accessors. Conversely the 37 internal routes cannot be silently treated as direct host calls: instance/class patches must still reach the old controller route. Module-level free globals and nested review builders add their own named dependencies; the JSON lists them per function. This is why 250–350 must be a measured budget, not an assumed one.

Use explicit keyword-only dependencies and call-time lambdas, for example `store_accessor=lambda: controller.store`, `app_accessor=lambda: controller.app`, `marshal_pending_chat_create=lambda payload: controller._marshal_pending_chat_create(payload)`. Each nullable setter/timeout remains nullable under a getter; do not replace None with a truthy no-op callback. Shared reassignable state requires explicit setters as well as getters. No whole-controller receiver, reflection through module globals, dependency bag, generated proxy/mixin, or snapshotted bound methods for the new wiring.

Keep the public controller methods and module helper names, signatures, decorators and default-value semantics. Async wrappers remain async and await; blocking worker bridges remain synchronous. Preserve current module patch resolution with named late-bound callables, not simple re-exports, since a re-export does not move function globals.

## Exact legacy `_seams` conversion scope

The existing host has **27 observed direct/getattr sites** below. These are source sites, not just one row per name. Every site needs a named dependency or real host-owned operation in the selected design. The dynamic setter lookup must become five explicit getter callables (approval, skill install, skill script, worktree merge, question), selected by the existing kind contract. A generic `controller_attribute(name)` resolver is not acceptable.

- `267`: `self._seams.store`
- `290`: `self._seams.store`
- `304`: `self._seams.store`
- `328`: `getattr(self._seams, '_refresh_answerable_decision', None)`
- `337`: `self._seams.store`
- `387`: `getattr(self._seams, 'announce_hidden_decision', None)`
- `411`: `self._seams._announce_hidden_decision`
- `416`: `getattr(self._seams, KIND_SETTER_ATTRS[kind], None)`
- `419`: `getattr(self._seams, 'store', None)`
- `470`: `getattr(self._seams, 'app', None)`
- `513`: `getattr(self._seams, 'on_pending_rounds_changed', None)`
- `745`: `getattr(self._seams, '_notify_run_hook_approval', None)`
- `752`: `getattr(self._seams, '_publish_pending_decision', None)`
- `787`: `getattr(self._seams, 'add_pending_round', None)`
- `802`: `getattr(self._seams, 'app', None)`
- `803`: `getattr(self._seams, 'park_pending_approval', None)`
- `807`: `self._seams._approval_view_is_detached`
- `808`: `self._seams._announce_hidden_decision`
- `811`: `getattr(self._seams, 'set_pending_decision', None)`
- `812`: `self._seams._marshal_pending_decision_projection`
- `864`: `self._seams._is_session_cancelled`
- `873`: `self._seams.expire_pending_decisions`
- `886`: `self._seams._is_session_cancelled`
- `904`: `self._seams._forget_hidden_decision`
- `914`: `getattr(self._seams, 'discard_pending_round', None)`
- `920`: `getattr(self._seams, 'set_pending_decision', None)`
- `922`: `self._seams._marshal_pending_decision_projection`

Required named groups:

1. Live `store` and `app` accessors; five per-kind nullable setter getters; `set_pending_decision` getter and `park_pending_approval` getter.
2. Controller-routed callbacks for `_refresh_answerable_decision`, `announce_hidden_decision`, `_announce_hidden_decision`, `on_pending_rounds_changed`, `_publish_pending_decision`, `_notify_run_hook_approval`, `add_pending_round`, `discard_pending_round`, `_forget_hidden_decision`, `_approval_view_is_detached`, `_marshal_pending_decision_projection`, `expire_pending_decisions`.
3. The retained `_is_session_cancelled` callback, forwarding its original session, arm-time visit event and cancellation arguments.

Most groups overlap extracted-member dependencies, but the host adds routes not directly referenced by moved bodies. Count union and each formatted binding site, not a collapsed namespace. Constructor parameter declarations/properties live in the owner; construction call and module-level forwarder injection lines count against the controller cap.

Compatibility: `Tests/Chat/test_console_interrupt_rounds.py` constructs minimal `FakeSeams`/`FakeSeamsFull` and reads `host._seams`; its tests mutate store/setters after construction and delete a setter. Preserve these dynamic behaviors using an explicit test adapter/fixture dependency wiring; unchanged assertion bodies must still observe the same fake. Do not retain broad production receiver access merely to keep a fixture shape. The exact `_seams` attribute reads are at lines 80, 91, 98, 260 and 288 in the observed file.

`Tests/Chat/test_console_interrupt_host_wiring.py` requires original registry/payload/lock object identity and, in `test_approvals_register_the_permission_summary_as_the_after_remount_hook`, **`hook.__func__ is ConsoleChatController._maybe_fire_permission_summary`**. Preserve the existing bound public wrapper in `after_remount['approval']`; replacing it with an anonymous lambda breaks a real contract. The wrapper can forward to host-owned behavior. This existing identity contract must be reviewed separately from new late-binding injection.

## Patch, import and carry maps

The JSON records module import/constant bindings, per-member free globals/local imports, named test references, explicit patch sites and attribute assignments. It also flags current AST equality and enumerates the I1 delta. Static lexical candidates are not a proof for computed `getattr`, JoinedStr dispatch, unbound fake-self calls or callback identity: inspect each relevant receiver in the listed files before freezing implementation.

Load-bearing examples: `test_console_chat_create_confirm.py` patches `_marshal_pending_chat_create` on the live instance, plus `_is_session_cancelled` and `_bind_visit_cancel_signal`. `test_permission_summary_wiring.py` patches the controller module's `threading.Thread` and summary-service imports. Keep module-setting/actor/run/clock/uuid and inter-helper patch routes named and late-bound. Local imports remain local. Keep legacy import aliases if external tests patch them. Bare `object.__new__(ConsoleChatController)` fixtures in local/virtual approval tests exercise provider-composition methods that remain in place; do not add eager host requirements to those paths.

Only selected `request_chat_create_confirm` overlaps I1's current changed methods. Carry its **frozen I1** body; the observation capture/read/match helpers and record/lock routines remain in the original controller. Preserve their exact callback/lock order. All selected frozen members have source and location-independent AST hashes; compare untouched definitions to frozen I1 and moved definitions with a narrowly enumerated dependency-binding diff. Review wrappers/wiring separately. This is structural placement, not a repeat correctness review of I1.

## Loading and concurrency contracts

Prefer the already-resident interrupt host module. Its constructor import is already local to controller construction. Do not add a host-to-controller import cycle or make any existing local import eager. If a separate hook-owner module proves necessary, import only inside the existing actual `build_*_review_hook` invocation; no ready/preimport cap increase or exclusion can fund it. The new module option remains subject to root's final scope, not implicit authorization here.

Preserve one non-reentrant generic host lock, the distinct chat-create lock, existing order, callbacks outside registry locks, event identities captured at arm time, exact revocation/admission decision ordering, and the sync worker/UI-thread handoff. No new await, SQLite read or arbitrary callback under a registry lock. Preserve ADR-067's human-wait/deadline and revoke semantics. Source/durable/launch authority stays where it is.

## Derived inventories and later targeted qualification

Affected inventories:

- `Tests/Architecture/test_module_size_ratchet.py`: actual exact controller count; lower the row on shrink as required, never raise 29,367. Root may add a truthful owner budget separately.
- `Tests/Architecture/test_persistent_diagnostic_inventory.py` and referenced `Docs/security` diagnostic-review JSON: reattribute moved log-owner paths/digests, retaining event/privacy assertions. No diagnostics sweep.
- `Tests/Chat/test_console_fork_transition_census.py::test_external_console_modules_do_not_write_live_fork_fields_directly`: add the extracted owner to its explicit source-owner scan so moved code remains covered. Preserve assertions. Other rollback-contract checks stay on the unmoved owner.
- `test_console_first_chat_handoff.py::test_no_unbounded_resolve_for_send_awaits_remain`: the selected bodies contain no bounded-provider-resolution owner; retain its current controller guard. Broaden only if an actual selected method introduces that dependency.
- Ready/preimport resident-module census: keep 1,033/557 and current exclusions; actual loading must qualify once.

Run no tests for this preflight. The subsequent scope should select the existing interrupt-round/host-wiring/attention tests, confirmation tests for chat-create/ask-user/skill-script/worktree, permission-summary/payload wiring, and only review-hook nodes reaching the selected builders from the JSON map. Carry the established I1 regressions once through the final confirmation route. Add meaningful post-construction patch probes only where existing tests lack coverage. Run the exact module row, changed diagnostic owner inventory, external-writer census and ready/preimport checks once with this worktree's `PYTHONPATH`. No Console/provider/schema/encryption sweep and no repeated passing cohort.

## Governance / selection gate

Read DESIGN.md section 7, migration safety, and bounded decomposition recipe sections on state ownership, monkeypatch/global binding, census spellings and evidence. No DOM/CSS move. ADR required: yes for the final changed runtime dependency contract unless root identifies direct implementation of an existing accepted ownership ADR. ADR path: root selects/creates before implementation; no ADR/plan mutation in this preflight. Preserve ADR-067 and the existing interrupt-host design.

**Selection concern:** the existing owner is coherent and the measured bodies are large enough in principle. However, the actual explicit named dependency/forwarder layout is not yet measured. This report does not prove that it fits the 350-line allowance. Selecting it solely from the optimistic 250-line calculation would be premature. Root should require a concrete formatted wiring sketch or revise the coherent boundary, without compressing code or moving shared authority.
