# Console approval domain ownership implementation plan

> **For agentic workers:** Use superpowers:executing-plans for this bounded implementation. Root owns integration and independent review; no additional agents or Git mutations are authorized here.

**Goal:** Remove the two new Chat-to-UI feedback imports and restore the three existing module-size limits without changing approval authority or lifetime.

**Architecture:** Existing Chat feedback owns store-only observation bookkeeping; the UI facade retains its projector and real card message alias. Existing import-free approval constants and approval provenance own their unchanged values/functions. Controller, interrupt host and runtime retain their entry points, live dependencies and lifecycle authority.

**Tech Stack:** Python 3.12+, existing callbacks/locks/dataclasses; no new dependency.

**Spec:** [approval design](../specs/2026-10-05-console-approval-ux-and-responsiveness-design.md), [Root integration plan](2026-11-07-console-approval-ci-boundaries.md), Task34569 AC12.

ADR required: existing amendment, written before product edits.
ADR path: `backlog/decisions/235-console-approval-interaction-and-feedback.md`, Feedback domain ownership.
Reason: clarify existing fact/projection ownership and directly implement the existing layer boundary; no new authority, public permission API, persistence or runtime policy.

## Global constraints

- Root's current-source UI3 replay must finish and Root must explicitly release the source/test hold before product edits or pytest.
- Use only reviewed protected-launcher admissions. Root owns launcher, allowlists, module ratchets, global inventories, Task34569 and integration commits/freezes.
- Preserve Event release before receipt, host cancellation/deadline arbitration before settlement, original decision/denial objects, round/revision identity, FIFO and thread/lifetime boundaries.
- No provider/bridge/control-plane/transcript, observation schema, RAG/path policy or shared-plan edits. No App/GUI/full suite, cap increase, whitespace compression or padding.
- The three current ceilings remain at most29180 controller /6303 interrupt host /7069 runtime; Root lowers rows to final measurements when policy requires it.

## Review focus

1. Controller/module monkeypatches remain live: relocated constants are original-name reexports and existing callback accessors remain unchanged.
2. Mutable scope inventory is the same imported object, with no copied dictionary or cached snapshot.
3. Optional observers cannot alter verdicts, Event release, cancellation, FIFO or exception/result identity.
4. UI compatibility names refer to canonical domain functions; the actual card message alias/projector stays UI-owned.
5. First-use store construction, bridge replacement and visit reconciliation keep their original locks, ordering and native lifetimes. These checks provide no physical latency evidence.

## Exact source map

| Source | Change |
| --- | --- |
| `UI/Console_Modules/approval_feedback.py` | Replace four store-only bodies with explicit canonical reexports. Keep `ApprovalFeedbackController`, its `refresh` body and `ApprovalFeedbackChanged` identity. |
| `Chat/console_approval_feedback.py` | Receive the four exact bodies and four controller lifecycle helpers. Store/reducer/formatting bodies remain unchanged. |
| `Chat/approval_constants.py` | Receive unchanged feature constants and their semantic comments. Remain import-free. |
| `Chat/console_chat_controller.py` | Original-name constant imports, ordinary feedback lifecycle delegates, domain publication import. No provider composition or original dependency accessor changes. |
| `Agents/approval_provenance.py` | Receive three exact pure fact/provenance adapters; all existing functions/classes remain unchanged. No edit to `Agents/approval_observation.py`. |
| `Chat/console_interrupt_rounds.py` | Domain feedback imports and direct original-name provenance reexports. Keep all registry, Event, wait, arbitration and sibling-refusal bodies. |
| `Chat/console_chat_models.py` | Receive the exact `ConsoleViewHookSlot` dataclass; no new dependency or persistent model/schema. |
| `Chat/console_runtime.py` | Original-name reexport of that canonical class; retain slot inventory, constructor lookup, remount and every method unchanged. |

### Canonical observation interfaces

Move these complete functions from UI to Chat with original signatures, defaults, docstrings, bodies and explicit owner callbacks:

- `bind_round_observations(pending, view, *, read_feedback, read_publish)`
- `settle_round_observations(round_id, decisions, contexts, outcome, *, read_feedback)`
- `receive_round_observations(round_id, decisions, round_state, *, read_feedback, read_notify)`
- `publish_approval_observation(observation, *, feedback, project, notify)`

UI imports each as the same named object. Controller and interrupt host import Chat directly. No wrapper copies or UI fallback routes.

### Controller constants and lifecycle

Move the following existing definitions, original values and comments to `Chat/approval_constants.py`:

- `_DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS = 0.0`
- `_MCP_APPROVAL_POLL_SECONDS = 1.0`
- `_DEFAULT_SKILL_INSTALL_CONFIRM_TIMEOUT_SECONDS = 0.0`
- `_DEFAULT_SKILL_SCRIPT_CONFIRM_TIMEOUT_SECONDS = 0.0`
- `_DEFAULT_CHAT_CREATE_CONFIRM_TIMEOUT_SECONDS = 0.0`
- `_CHAT_CREATE_SESSION_GONE = "session_gone"`
- `_DEFAULT_WORKTREE_MERGE_CONFIRM_TIMEOUT_SECONDS = 0.0`
- `_DEFAULT_ASK_USER_TIMEOUT_SECONDS = 0.0`
- `ASK_USER_TIMEOUT_ENV_VAR = "TLDW_CONSOLE_ASK_USER_TIMEOUT_SECONDS"`
- `_MAX_TRACKED_QUESTION_BOUNCE_RUNS = 64`
- `_LEGACY_PENDING_APPROVAL_ROUND_ID = "__legacy_pending_approval__"`
- `_APPROVAL_SCOPE_RANK = {"approve_once": 1, "approve_session": 2, "always_allow": 3}`
- `_APPROVING_DECISIONS = ("approve_once", "approve_session", "always_allow")`

Reexport the original controller names. Do not relocate Personal Context setup budgets, dispatch regex, provider identities or unrelated constants. Keep all existing `read_global_*` lambdas in place, so replacing a controller name still changes its original reader. Assert alias identity, including the mutable scope dictionary; compare definition AST/default values without serialization copies.

Preserve the controller property getter/setter and `_notify_approval_feedback`/`_publish_approval_observation` signatures as ordinary delegates. Canonical helpers in Chat feedback are `controller_feedback_store(controller)`, `set_controller_feedback_store(controller, value)`, `notify_controller_approval_feedback(controller, session_id, run_id)` and `publish_controller_approval_observation(controller, observation)`. Move their original bodies mechanically, preserving double-checked locking, existing bridge lookup timing, notification failure isolation and publication evaluation order. The getter's now-redundant self-module import may be removed; its canonical class lookup remains live.

### Interrupt provenance adapters

Relocate only `_approval_decision_fact`, `_review_decision` and `_stamp_answer_provenance` to `Agents/approval_provenance.py`. That owner already defines `ApprovalDecisions` and `approval_was_unanswered`. These three bodies use builtins and explicit named callbacks; preserve their complete signatures/bodies and original host names as direct aliases. `_collect_mcp_pending`, `_sibling_approval_refusals`, approval stamping, registries and every authority/wait body remain in their original owner. Confirm no Domain2 source overlap before edits.

### Runtime view-hook metadata

Move the exact `ConsoleViewHookSlot` dataclass declaration into already-imported `Chat/console_chat_models.py`. Retain the runtime's original name as a direct class import, so the existing `CONSOLE_VIEW_HOOK_SLOTS` construction uses the same runtime lookup. Preserve all four fields (`name`, `target`, `viewless_default=None`, `why=""`), constructor, frozen behavior, class/object identity and historical runtime pickle lookup. The destination is below the existing mandatory-row size threshold and needs no new import. No persistent model, schema, state or authority is added.

The earlier remount-body proposal is explicitly rejected before implementation: ordinary reconciliation would import an approval-only owner and could violate the existing first-use/UI-ready fence. `ConsoleRuntime.remount_pending_approval`, its logger context and callback cadence remain unchanged.

## Targeted RED/GREEN sequence

- [x] Root releases the current-source replay hold and verifies the final launcher hash/admissions.
- [x] Retain exact existing REDs: `Tests/Architecture/test_layer_direction.py` new upward import; `Tests/Architecture/test_module_size_ratchet.py -k "module_does_not_grow_past_its_budget and (console_chat_controller or console_interrupt_rounds or console_runtime)"`.
- [x] Add focused identity/callback cases in existing `Tests/Chat/test_console_approval_feedback.py`, `Tests/Chat/test_approval_constants.py` and `Tests/Chat/test_console_runtime_lifetime.py`; observe actual RED before code. Keep original positive observation, exception, cancellation and first-use concurrency cases.
- [x] Apply only the mapped relocations/delegates. Generate private AST/signature/default/body comparisons against the preserved pre-edit source; explicitly account for controller-owner renaming and any getter self-import removal. Verify exact dataclass AST and unchanged runtime constructor/slot-inventory AST.
- [x] Run admitted feedback, UI feedback, constants, interrupt-host wiring/rounds, runtime-lifetime and runtime-ownership targets. Use selectors for lifecycle tests when appropriate; no full suite or unrelated replay.
- [x] Verify layers, the existing UI-ready forbidden-import guard and all three final size measurements, then report exact moved diagnostic/source symbols to Root for canonical inventory and any lower ratchets. No central artifact edits by this owner.
- [x] Run changed-source static/whitespace checks; record inherited diagnostics separately and request independent review before Root stages files.
- [ ] Independent review and Root-owned integration/staging.

Protected targets requested from Root: architecture layer and module-size files, runtime-lifetime file, approval-constants file. Existing feedback/UI/interrupt/runtime-ownership admissions are reused. No direct pytest, generic target mode or bypass is authorized.

## Implementation verification

The recorded REDs preceded product edits: three constant/reader cases, two domain alias cases, the canonical view-hook model case, the layer guard and all three module ceilings. The layer RED also named Root's independent Utils-to-RAG edge; the final unchanged full layer guard passed after the combined working-tree corrections.

The primary bounded run passed115 cases: constants7, feedback34, UI feedback16, interrupt wiring12, interrupt rounds31, selected runtime lifetime4, selected hook ownership3, three module ceilings3, layer guard4 and UI-ready guard1. After removing three now-unused type-checking imports, final alias/wiring/ceiling repeats passed17 cases; these are repeats, not17 additional distinct cases. RED used reviewed launcher97, GREEN used reviewed98. Every invocation retains argv, stdout/stderr, exit status and verified original-home/origin/startup receipts.

Source proof verifies raw-text and AST equality for all four moved observation functions, all three provenance functions and the dataclass; exact constant definitions; original controller signatures/decorators and mechanically renamed helper bodies; and unchanged remaining module definitions, runtime constructor/slot inventory and model contents. Ruff diagnostic fingerprints remain74/74 (controller58, existing models16, six other owners clean). Only changed helper/test ranges were formatted; owned whitespace checks pass. Final measurements are controller29167, interrupt host6298, runtime7050 and models1686; no limit was raised. Root owns any lower ratchets and canonical inventory update. No logging format/context moved; only the four UI observer exception-site owners and the controller notification exception-site owner move to the named Chat functions.

The UI-ready guard passed with a Windows `subprocess(text=True)` stderr-reader `UnicodeDecodeError` thread warning. Child returncode/stdout census and forbidden-module assertions passed; complete child stderr is not established. This warning has no baseline or product-cause attribution here and was not hidden or retried with changed encoding.

That UI-ready result predates the subsequently discovered nested-child Core-path gap described below. Its original pass is retained, but does not establish a correctly selected-Core child; the source-correct repeat remains a separate required check. The other parent-process ownership tests and source proofs retain their recorded scope.

Private evidence directory: `domain-feedback-extraction-20261107T152500Z` under the existing PR QA root. `summary.json` SHA256 is `680141ebe39e41a942ec3b72372fa42436f4d9c23b1f7cb038f074d4f44e71d3`; `semantic-proof.json` SHA256 is `0c5e9665796edc30831238de699a8cb4a76174ea69dd00825dfc9274bd66e468`. That proof distinguishes raw private bindings from portable UTF8/CRLF-to-LF bindings. No live App capture, full suite, physical timing claim or Git mutation occurred in this implementation.

## Source-binding follow-up plan (before edits)

Root authorized a test-harness correction under AC12 after finding that fresh children could resolve Core from the primary editable checkout. The isolated builders in `Tests/Performance/test_app_import_weight.py::_run_isolated_python`, `test_ui_ready_module_census.py::_boot_and_census` and `test_app_startup_performance.py::_run_isolated_python` override inherited `PYTHONPATH` with Main alone. Change only those three environment values to the selected Main root followed by its `packages/tldw_profile_core/src`, using `os.pathsep`. Keep all command flags, budgets, caches, timeouts, observers, assertions and product imports unchanged. No product security/import deferral is authorized.

ADR required: no new ADR; existing ADR097 and the current source-isolation contract apply. Root owns the protected launcher and any broader admission. Additional Main-only preimport and whitelist first-use builders are reported separately; they require Root's scope decision before edits.

1. Before edits, compile only the actual helper function AST with stdlib globals and intercept its existing subprocess call. Pass the real captured environment/cwd into a fresh stdlib-only origin probe; never run the original App script or import a product/Test package. Retain the actual Main/Core spec paths and proof that product packages remain unloaded. The first real call is the observation boundary; no synthetic module counts or performance result are returned.
2. Demonstrate the existing builders select the wrong Core in that actual-environment probe, then change only the three `PYTHONPATH` values and require the selected Main/Core origins. Preserve exact other helper ASTs and source hashes.
3. After Root's concurrent UI3 case ends, run only the authorized focused original import/UI-ready/preimport checks through the reviewed launcher. Retain any strict688 count failure as current evidence; do not call it inherited without a correctly bound baseline.
4. Keep the Windows import chain explicit: `app` imports storage admission before runtime composition; storage admission/bootstrap import the platform facade, whose Windows branch imports `windows_files`. Core path correction does not justify deferring that security boundary. Both Core checkouts'14 Python files currently have identical UTF8/LF text and no import of Main or the Windows facade; the corrected count must still be observed rather than assumed.

### Stdio and remaining-child amendment (before edits)

Root extended the same correction to `test_screen_preimport_payload_budget.py::_run_census` and the UI-ready first-use test's whitelist environment. Those children also explicitly receive selected Main/Core. All five environment sites set `PYTHONIOENCODING=utf-8`; their six text-capturing subprocess calls explicitly decode with `encoding="utf-8"`. This repairs the retained actual Windows cp1252 reader failure without changing scripts, interpreter/file/cache flags, timeouts, assertions or budgets.

Extend the metadata-only probe to the two additional actual environments and emit non-ASCII stdout/stderr. Intercept the original first-use test script before delegation and substitute only the stdlib metadata probe; stop at the first real subprocess boundary rather than inventing result/census values. Require selected package origins, no loaded product/Test packages, explicit child stream encoding and exact decoded Unicode on both streams. Preserve the earlier three-case RED/GREEN and source-only receipts. Actual performance-node reruns remain held until Root releases the UI3 trace review.

The actual-environment probes reproduced wrong-Core resolution in the original three builders and both additional environments. All five corrected environments resolve selected Main/Core/Tests, and the metadata-only children emit/capture exact non-ASCII stdout and stderr with explicit UTF8 on both sides. No product/Test package or original App script ran in these probes. Whole-module AST comparison proves only five path values, five child stream settings and six capture decoders changed; four-file Ruff/whitespace checks pass. Private receipts are under `test-child-source-binding-20261107T162000Z`: initial RED `125cf630...`, additional RED `e810ad1c...`, final metadata/stdio GREEN `47a2af06...`, source parity `7afdb485...`. These metadata-stage results preceded the separately released original performance checks below and establish no module-count or inheritance result themselves.

After Root released the trace-review hold, the three original protected performance nodes ran once each, sequentially, through reviewed launcher98 `5620c527...`. Strict app import still fails at688<688, with `Utils.windows_files` the sole snapshot addition; no inheritance or speculative product cause is attributed. UI-ready passes1038/1038; preimport passes556/558 modules and426554/426575 LOC at unchanged limits. Only the expected Pydantic/headroom warnings remain; no Unicode reader warning occurs. Full pytest stdout/stderr and the original child data surfaced by those tests are retained. A successful child's captured-but-not-emitted stderr is unavailable because the original helpers discard it; no extra observer/copier or second run was added. No product deferral, cap/assertion/cache/interpreter change or new performance qualification follows.
