# Task 17 interrupt review preflight

Read-only source preflight at PR2995 head `06cfe2f6a30236dcd8f81893ebf49bc3d0b78036` in `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`.

Feedback: Qodo comments `4179836586` (stale test construction) and `4179836590` (wide dependency interface), read from `task-17-initial-feedback.json`. No source, index, HEAD, tests, dependencies, profile/configuration, caps, or replay cohorts were changed. No test or production import was executed. Root owns the technical ruling, source selection and independent implementation/review.

## Recommendation

1. Accept `4179836586`. Change only the imports in the two stale test modules to use `Tests.Chat.console_interrupt_test_bindings.make_interrupt_host as InterruptRoundHost`. Preserve all test bodies, assertions, parameterization, deadlines, event identities, cleanup and fakes. No helper or fake attribute additions are needed for these paths.
2. Partially accept the maintainability observation in `4179836590`: the host's `read_global_Any` and `read_global_Mapping` dependencies are annotation-only and can be removed after root selects that narrow scope and clarifies ADR220's exact map. Replace only their annotation uses with the existing imported `Any` and `Mapping`. Remove the two host constructor parameters/assignments and matching entries in production wiring and the test helper. This changes the host from 122 to 120 required keywords, and from 35 to 33 global getters. All 86 controller readers and the one write-through callback remain exact.
3. Reject the proposed wholesale direct-global/one-object interface. Runtime global patch routes and explicit named live controller readers are an approved ownership contract. A Protocol taking the controller would restore broad receiver access; a dataclass of getters would introduce the dependency bag prohibited by the selected plan. Neither fixes the six stale fixtures more cheaply than their existing test helper.

ADR required: existing ADR220 applies; no new architecture boundary is proposed. ADR path: `backlog/decisions/220-console-human-decision-coordination-ownership.md`. Reason: the fixture repair directly follows its legacy test-wiring rule. Annotation-only pruning needs a narrow clarification of the selected 35-global interface/maps, while retaining all actual runtime patch routes. Root must select that change before source implementation.

## Complete production/test construction inventory

`rg` was used first; an AST inventory then confirmed every constructor/helper call under `tldw_chatbook/` and `Tests/`. The current host signature has 122 required keyword-only parameters: 86 controller readers, 35 global readers, one controller write-through callback. There is no positional dependency parameter or compatibility overload.

| File | Call lines | Actual callable and disposition |
| --- | --- | --- |
| `tldw_chatbook/Chat/console_chat_controller.py` | 4276 | Real host, all 122 explicit keywords; correct production construction. |
| `Tests/Chat/console_interrupt_test_bindings.py` | 12 | Real host, all 122 explicit keywords; correct test-only adapter. |
| `Tests/Chat/test_console_decision_clock.py` | 25, 65, 94, 130, 161 | Real host imported at lines8–11; each passes one positional fake; all five stale. |
| `Tests/UI/test_buddy_speech.py` | 49 | Real host imported at11; setup passes one positional `SimpleNamespace(store=store)`; stale. |
| `Tests/Chat/test_console_interrupt_rounds.py` | 47, 133, 147, 162, 175, 188, 202, 215, 232, 247, 274, 302, 321, 341, 356, 376 | `make_interrupt_host` aliased as `InterruptRoundHost` at19–21; already correct. |
| `Tests/Chat/test_console_interrupt_attention.py` | 167, 213, 239 | Same test helper alias at11–13; correct. |
| `Tests/Chat/test_permission_summary_wiring.py` | 29 | Same helper alias at14–16; correct. |
| `Tests/Chat/test_console_local_review_hook.py` | 777 | Same local helper alias at772–774; correct. |
| `Tests/Chat/test_console_virtual_cli_approval.py` | 126, 167, 207 | Direct `make_interrupt_host` calls, imported locally at124,165,205; correct. |
| `Tests/MCP/test_approval_timeout_policy.py` | 43 | Direct helper call, imported at10; correct. |

Total: 33 construction/helper call sites. Two construct the actual host correctly, six construct it incorrectly, and 25 already route through the helper. The controller also imports the class at15398 to call its static `_head_round_payload_locked`; that is not a constructor and needs no change. Other mentions are documentation, class definition and callback qualname assertions. No additional production/test constructor aliases were found.

The stale-call diagnosis follows directly from Python argument binding: each of those six sites gives an argument after `self`, while the constructor accepts only keyword-only arguments. Their failure precedes round/speech assertions. This preflight did not run a failure reproduction, so it claims source-established argument incompatibility, not a test receipt.

## Minimal fixture correction and genuine dependency needs

Both modules should import:

```python
from Tests.Chat.console_interrupt_test_bindings import (
    make_interrupt_host as InterruptRoundHost,
)
```

Decision-clock keeps its separate `KIND_SETTER_ATTRS` import and existing `FakeSeamsFull` import. Buddy speech replaces its current direct host import. The helper already uses `getattr(seams, name, None)` for all controller reads, captures the fake itself rather than callback snapshots, and reads `ccc` globals at invocation. Its constructor stores readers without invoking them. Optional absence is therefore preserved; do not create 86 dummy attributes merely to match the signature.

### Decision-clock paths

`FakeSeamsFull` already provides the actual inputs these five test functions use:

- `store.active_session_id` (initially `sess-A`, changed to `sess-B` and back for the session-away phase).
- `app.call_from_thread`, synchronously executing the actual supplied callback.
- All five `set_pending_*` callbacks, installed by `FakeSeams` from `KIND_SETTER_ATTRS`, appending to the existing `mounted` lists.
- `_is_session_cancelled`, which returns the fake's live `cancelled` value.
- `add_pending_round` and `discard_pending_round`, writing the existing `badges`/`added_kinds` evidence.
- `park_pending_approval = None`, retaining its original no-toast behavior.

Actual call flow: `run_round` -> registration -> optional hook/publisher lookup -> native payload park -> current app/setter -> `refresh_decision_clocks` -> event polling/current cancellation reader -> native unpark/badge discard -> current setter/remount. `_notify_run_hook_approval`, `_publish_pending_decision`, hidden announcer and pending-attention hooks are absent and their guarded branches remain inactive. No retained-decision branch or controller creation/source authority is introduced by the helper.

The patched `rounds.time.monotonic` remains valid: the generic lifecycle/clock methods still use the resident module's `time`, and both modules initially reference the same imported `time` module object. No monotonic fallback, deadline widening, wait sleep, alternate cohort, or assertion relaxation is warranted. The helper's test-only `_seams` assignment preserves historical fixture inspection; production must not regain it.

### Buddy speech paths

The existing `SimpleNamespace(store=store)` is sufficient. Setup constructs the host but does not arm a blocking `run_round`. The speech coordinator inspects native `payloads`, `head_round_payload`, `lock`, and `registries`. The workspace-question test at232 installs the real threading event and parks a native question payload; it checks spoken order, receipt preservation, unset question event and active-session identity. `head_round_payload` uses native payload state and the ordinary resident time module; the test supplies no deadline. These paths need no UI setter, fake app or `_is_session_cancelled` callback. Adding those would enlarge the fake beyond the behavior being checked.

The two named speech/result tests preserve source revalidation after awaits, consent boundaries and all existing cleanup. The host adapter does not add a speech service or change the actual store supplied to `BuddySpeechCoordinator`.

## Annotation-only pruning versus live runtime globals

The host already imports `Any` at line33 and `Mapping` at30, with `from __future__ import annotations` at25. AST ancestor classification found:

- `self.read_global_Any()` has 15 uses: 10 local annotations and five nested-function argument annotations. Local annotation lines:1767,3619,3947,4223,4330,4431,4433,4492,4505,4616. Nested argument lines:1503,1772,1842,2264,2530.
- `self.read_global_Mapping()` has two uses: local annotation1767 and nested-function argument1771.
- Their only other host attribute references are constructor assignments587 and599. No expression read outside annotations exists.

Python does not evaluate the local annotations; the nested function annotations are postponed strings. There is no `get_type_hints`, `get_annotations` or `__annotations__` inspection in the host, controller or live-binding controls. The controller's separate `inspect.signature(provider)` at22727 is unrelated to these nested functions. A source-only candidate rewrite, followed by stripping annotations from both ASTs, proved every host method other than the constructor has identical non-annotation AST. This was an in-memory check; sources were not edited or imported.

Bounded selected change: the two signature entries327/333, two assignments587/599, the 17 annotation references above, controller host-wiring lambdas4487/4499, and test helper lambdas259/271. Use existing `Any`/`Mapping`; add no imports or abstractions. Preserve docs and all other method/signature/runtime statements. The postponed nested annotation strings intentionally become conventional `Any`/`Mapping` spellings; arbitrary external introspection of private nested callbacks is an explicit compatibility limit, not evidence of a runtime dependency.

There is also a separate module-level `build_tool_review_hook` parameter `read_global_Any` at5841, used only in local annotation6063, with controller-wrapper wiring3051. That is outside the host constructor focus and is not needed to repair either stale module or remove the host's two annotation dependencies. Leave that separate interface unchanged unless root explicitly selects it. Likewise leave the compaction collaborator's independent `Any`/`Mapping` getters unchanged.

Runtime globals cannot be generalized from these two type names:

- `_resolve_ask_user_timeout_seconds` reads the controller-module `get_cli_setting` through its named getter at invocation. `test_timeout_reads_console_config_when_no_seam` patches `ccc.get_cli_setting`, observes7.0, then replaces it with an invalid-value callback and observes0.0 on the same controller (Tests/Chat/test_console_ask_user_round.py:185–198). A direct host import would stop following that replacement.
- The committed-Close question control at399 onward patches the controller module's `uuid4` after creating the controller and starting a sibling, using that call to place a real registration barrier at463. Direct host `uuid4` use would bypass that original patch route.
- Moved confirm methods use `read_global_time().monotonic()` at3654,3974,4241,4349,4480,4633 and runtime `read_global_os()` for timeout/bell environment precedence at1897/2788. Threading event/type/thread creation and logger calls also execute in moved methods. Shared module-object member patches sometimes happen to remain visible across imports; replacement of the controller module binding would not. Their existing live accessors preserve both cases.
- `test_interrupt_remount_reads_replaced_sink_and_controller_state` replaces live setters with callbacks then `None`, and replaces the controller-owned pending map. `test_approvals_register_the_permission_summary_as_the_after_remount_hook` pins the exact bound controller wrapper. Broad receiver restoration, snapshots or callback rebinding would weaken those contracts.

ADR220 and the plan at264–266 explicitly require named keyword-only reads, nullability, native aliases, original patch routes, controller-owned authority and the counter write-through. They expressly reject whole-controller receivers, generic resolvers/proxies/mixins, dependency bags, snapshots and new locks. The annotation-only refinement removes non-runtime ceremony; it provides no basis to bypass the live dependency contract or restore the stale positional API.

## Bounded verification selection for the source implementer

Run the two complete affected modules, preserving all current parameterizations:

```text
Tests/Chat/test_console_decision_clock.py
Tests/UI/test_buddy_speech.py
```

The decision-clock module has five test functions expanding to30 cases:10 navigation/session-away cases and four groups of five kinds. Buddy speech has eight test functions expanding to14 cases, including the five source-owner changes and three persona changes. These44 cases directly qualify the corrected fixture imports and their original behavior. Do not replay their unrelated sibling owners or historical extraction/feature cohorts.

If root selects annotation-only production pruning, add only these existing exact controls:

```text
Tests/Chat/test_console_owner_live_bindings.py::test_interrupt_remount_reads_replaced_sink_and_controller_state
Tests/Chat/test_console_interrupt_host_wiring.py::test_legacy_registry_payload_and_lock_names_alias_the_host
Tests/Chat/test_console_interrupt_host_wiring.py::test_approvals_register_the_permission_summary_as_the_after_remount_hook
Tests/Chat/test_console_ask_user_round.py::test_timeout_reads_console_config_when_no_seam
```

These expand to eight cases and witness the retained production construction, setter nullability/replacement, map ownership, native lock/registry identity, bound hook identity and runtime module-binding replacement. Add a narrow AST audit that every changed non-constructor method retains its exact non-annotation AST and that only the two enumerated getter entries were removed. Target Ruff/formatter checks to the touched files and retain all existing size/loading/cap rules; inspect the exact diff to avoid unrelated formatter churn. If the extra production selection remains unapproved, the two test-only import changes require no production replay.

No tests were run by this preflight. No pass/failure totals above are receipts; they are statically derived selections. Root must record fresh results and the exact chosen source before closure.
