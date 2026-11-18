# Task 9: permission summary on session switch

Source-only diagnosis at supplied HEAD `432c636fb596c073c5e33ab933ec704a814a3837` in `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`. No tests, source imports, source edits, or Git writes. Only this ignored SDD report was written.

## Contract and caller

ADR-090, `backlog/decisions/090-permission-request-context-summaries.md`, defines one advisory summary call per approval round and retained payload summaries that survive remounts. Its linked design, `Docs/superpowers/specs/2026-08-31-permission-request-summaries-design.md:161`, explicitly evaluates the trigger when a card mounts, including a parked round's later mount. `always` fires for a round with rows; later mounts must not refire.

`ConsoleChatController.switch_session` (`console_chat_controller.py:14502`) clears the outgoing claim and calls the store switch. When `set_pending_decision is None`, line 14548 calls `_reproject_pending_decision_for_session(session_id)`; its wrapper at line 15385 delegates to the host. `InterruptRoundHost._reproject_pending_decision_for_session` (`console_interrupt_rounds.py:2749`) passes the legacy setter the head from `self.payloads["approval"]`, selected with the explicit passed session ID, but does not call the summary trigger. The following `_remount_session_kinds` (line 2707) excludes approval (`SESSION_REMOUNT_KINDS`, line 72) and supplies no approval after-hook.

The attach (line 3454), sibling promotion (line 2597), and unified projection (line 3386; only when its setter reports `True`) routes explicitly fire the summary. `_maybe_fire_permission_summary`'s docstring (line 2067) names activation/switch as required trigger paths. This is a concrete source omission, not an executed functional RED.

## Store fake and setup limits

The fake `store.switch_session = lambda session_id: SimpleNamespace(id=session_id)` violates the real store contract. `ConsoleChatStore.switch_session` (`console_chat_store.py:4602`) calls `_activate_session(session.id)`; `_activate_session` (line 2372) sets `active_session_id` and publishes activation. A narrow fake must assign `ctrl.store.active_session_id = session_id` before returning. This matters to active projection, visibility/clocks, and outgoing-session selection on the second switch.

It cannot alone explain the missing summary in the legacy branch: that branch chooses the head by explicit `session_id` and never invokes the trigger regardless of active ID. Fix the fake before interpreting actual activation behavior, but do not use it to dismiss this source omission.

The payload alias also needs reconstruction: `_bare_controller` builds the host before `_parked_controller` assigns `_parked_approval_payloads`; `make_interrupt_host` currently aliases only approval registry and lock (test binding lines 322–324). The legacy host mount reads `host.payloads["approval"]`, not the later bare-controller dict. Bind that map explicitly or admit/park via the appropriate host API before interpreting mount assertions.

`set_answerable_decision` (host line 5068) also reads absent Console claim state, and `_refresh_answerable_decision` (line 2490) uses answerable/projection state. The complete minimal fixture-field/admission scope remains root's pending reconstruction. This report does not claim those missing fields are exhaustively enumerated.

## Smallest proposals and affected controls

First reconstruct only the scoped fields, host registry/payload aliases, and admission required to reach the original legacy switch phase; make `set_pending_decision=None` explicit and fix the store fake's active-ID update. Preserve the real remount and summary methods. Root records scope before repair/tests.

If that original control then demonstrates the absent trigger, the smallest production change is inside `InterruptRoundHost._reproject_pending_decision_for_session`: store the approval head in a local, pass it to the existing legacy setter, then call the existing `_maybe_fire_permission_summary` seam for a dict payload. Existing once-flag/mode gates provide the semantics. ADR-090 already applies; no new summary policy or ADR is needed.

This affects the legacy approval activation path shared by switch, new-session clear (`console_chat_controller.py:14120`), and close-neighbor activation (`:14859`). A `None` clear must not trigger. Attach, sibling promotion, and unified projection remain controls. The supplied `task-9-final-qualification-safe-evidence/summary-switch-source-observation.json` says this body matched BASE; no extraction/PR regression attribution is established here.

Exact controls in `Tests/Chat/test_permission_summary_wiring.py`:

- `test_switch_session_promotes_parked_round_and_fires_once`: first arrival mounts/fires; departure clears without refiring. The comment mentions returning, but the body only visits s1 then s0; actual revisit evidence needs a final switch to s1.
- `test_parked_round_fires_once_on_attach_remount`; `test_consumed_flag_prevents_refire_on_attach_remount`: attach and once-flag controls.
- `test_remount_head_helper_fires_promoted_sibling_once`; `test_remount_head_helper_skips_non_mcp_payloads`: sibling and non-approval controls.
- `test_mode_off_never_fires`; `test_fallback_fires_only_when_a_rationale_is_missing`; `test_fallback_fires_when_missing_and_always_fires`: mode controls.

Uncertainty: root must reconstruct/adopt fixture aliases/admission and execute the scoped original control before asserting functional RED or a verified repair. Current original-test failure remains setup evidence.
