# Console configuration capture implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development for the user-authorized parallel lanes, with root as sole integration owner. Steps use checkbox tracking; native runs remain sequential.

**Goal:** Replace the duplicate mounted and runtime configuration composers with one Chat-owned producer, preserving current selected inputs and compatibility while preparing for immediate runtime receipt.

**Architecture:** Thin adapters resolve their existing selection/settings inputs. One named-value producer captures common domain state and builds the complete detached snapshot. The default asynchronous controller route no longer imports or retains the Console session/UI adapter; explicit mounted and custom providers keep their supported routes and freshness checks.

**Tech stack:** Existing Python 3.12, Textual 8, asyncio and frozen Console snapshots; no new dependency, scheduler or cache.

**Spec:** [Console Send preparation architecture](../specs/2026-10-06-console-send-preparation-architecture-design.md).
**Task:** TASK-34563.9.

ADR required: yes, existing ADR-225 applies.
ADR path: backlog/decisions/225-console-send-preparation-and-io-ownership.md.
Reason: direct prerequisite implementation of the accepted screen-free capture boundary, preserving ADR-220 private relocation compatibility.

## Global constraints

- Preserve saved-turn failure: stop dispatch and keep the draft; retain temporary chats, WAL/NORMAL, required trace/consent facts, checkpoint order and awaited history.
- No early receipt or partial custody request in this slice. Existing initial review/admission and staged-input transfer stay in place until all receipt prerequisites are ready.
- No new UI surface, style, keybinding or timing budget. Actual rendered/input feedback within 100 ms and ordinary action-to-adapter under one second remain separate unmet qualification targets.
- No full test sweep. Use the existing private-profile contained runner, unique evidence labels, source-current receipts and positive normal retirement. Coordinate all native runs with UAT; preserve deadlines.
- Preserve callback affinity, call shape and live lookup. No callback dictionary, reflected dependency bag, proxy controller or memoized bound method.

## Interface and ownership

Shared lane owns only `tldw_chatbook/Chat/console_configuration_capture.py` and any explicitly agreed, necessary domain-helper extraction in that same module. Integration lane owns `Chat/console_chat_controller.py`, `UI/Console_Modules/session.py` and, only if necessary, `UI/Console_Modules/wiring.py`. Verification owns `Tests/Chat/test_console_configuration_capture.py`; root owns plan/task/report and final runs. Never edit another lane's files without an explicit handoff.

The shared producer is synchronous and returns `ConsoleTurnConfigurationSnapshot`:

```python
capture_console_turn_configuration(
    app, store, session_id, *,
    provider_selection, scratch_space, presentation_context,
    rag_defaults, tool_configuration, project_authority,
    skill_workspace_id, character_repository,
    tool_policy_profile_id, persona_policy_rules,
    mcp_definition_maximum=None,
)
```

Inputs are existing domain values and named app-owned services, never a screen, controller, bound builder or validator closure. The producer owns common Library scope/policy maximum, workspace review admission, character/prompt/skill capture, capabilities, provider payload fields and final freezing. Project authority is an explicit value resolved by each adapter through the same existing domain helper, preserving its controller-global patch behavior. Policy profile and persona rules are required named values; explicit None preserves the current Snapshot.capture semantics rather than triggering a second resolution.

Relocate the self-contained character capture plus emote projection, prompt capture, skill capture plus empty-result helper, and MCP definition capture plus its single exclusion constant to the new Chat module, with compatibility imports at the former public location. Keep the existing tiny capture_mcp_tool_maximum wrapper and project-selection/validation/root-identity/remote subtree in the controller. Use TYPE_CHECKING for annotations; the producer must not import the controller back or import Console session/UI adapters. Do not duplicate helper bodies or move unrelated remote execution logic.

The existing `capture_turn_configuration_snapshot(session_id, *, context_provider=None)` remains the checked asynchronous entry. Default capture uses the runtime adapter and the existing finite MCP capture/checks without importing the Console UI module. Recognize an explicit stock mounted builder only on the provider route; preserve its asynchronous MCP behavior and before/after mounted owner/callback checks. Custom providers remain one synchronous positional invocation with the existing type/session validation. No new general asynchronous callback seam is needed.

## Deliberate adapter differences

This mechanical consolidation does not normalize previously different configuration policies:

| Value | Mounted adapter | Runtime adapter |
| --- | --- | --- |
| RAG sources/depth | Current selected source types and depth accessor | Existing profile depth, source-types key absent |
| Tool settings | Existing app readiness mapping and strict direct-tools parser | Existing app/CLI split and coercion |
| Agent/project eligibility | Config, prefill and character checks | Additionally existing bridge/injected runtime eligibility |
| Skill workspace | Existing `None` | Owning workspace |
| Presentation/policy | Existing mounted resolvers | Existing app-owned runtime resolvers |

Equivalent explicit inputs must produce the same complete snapshot. Do not force parity for a test fixture's deliberate custom provider configuration. Fresh skill run identifiers remain independently checked; normalize only that nonce for equality, not whole authority fields. Existing CLI/budget read counts remain unchanged this slice; batching their actual source is separate work.

## Review focus

1. Selected session differs from active session: capture its owning workspace/settings, not foreground defaults.
2. Custom mounted callbacks differ from runtime configuration: preserve their current return values, affinity, call count and errors.
3. Source/settings/controller/service changes while the original MCP worker is held: refuse publication and drain the exact worker.
4. A view is detached during a default runtime capture: no Console session/UI import or retained view dependency in that capture; existing explicit mounted preparation may still retain its pre-custody UI owner.
5. Mutable selected maps/maxima change after capture: complete snapshots remain detached, with no widened tool/workspace/skill maximum.

## Implementation and checks

- [x] Record unchanged baseline before product changes: `capture-baseline-1` and `capture-baseline-2`, 18 passed on 0a0582, both normal process retirement. These characterize async stock/custom capture, drift, cancellation, mounted context and viewless selection; they are not new architecture acceptance.
- [x] Review this plan and exact helper relocation before implementation. Independent source review identified the project-helper dependency closure and explicit-None semantics; both are now resolved in the interface above. Constructor correction from TASK-34563.2 must be integrated first.
- [x] Verification lane writes focused controls using actual runtime-created controller/store and the existing native source fixture: complete explicit-input parity, adapter semantic differences, detached maps, custom one-call/error behavior, default route import independence and dropped-view weak reference during held MCP capture. Root observes meaningful RED against the old path before product changes.
- [x] Shared lane implements the producer; integration replaces both duplicate snapshot bodies with adapters and limits UI recognition to explicit provider routes. Keep all existing native identity/settings/service/source checks and custom fallback behavior.
- [x] Root runs the new file and affected existing `test_console_turn_execution_context.py`, `test_console_async_mcp_snapshot.py` and relevant UI wiring/adapter controls after both implementations are ready. Split bounded batches if needed; no changed deadlines or omitted failing assertions. Include exact retirement checks.
- [x] Run scoped Ruff/format/diff checks and compare existing large-module residuals against the committed base. Independently review the final producer, both adapters and tests; fix new issues, repeat only affected controls.
- [x] Update task and phase-attribution report with actual evidence and limitations; commit the independently verified slice. Do not mark the overall latency/feedback goal complete.

## Self-review

The interface uses named existing values rather than introducing another settings schema or general pipeline. All known mounted/runtime differences have an explicit adapter home. No source read or history/durability gate is removed for a count claim. The capture-only view-retention control does not qualify the current screen-owned pre-custody Send closure, rendered feedback or cold-start performance.


## Execution record and compatibility ruling

On committed 372ef191db, `capture-architecture-red` failed the actual default entry's unconditional Console session import and passed the held-worker/detached-view baseline. The original capture worker, leases and runtime disposed normally; the owned process tree and private profile retired with zero diagnostic overflow/races. This is a meaningful behavioral RED, not a missing-new-function assertion.

Both implementation lanes are now integrated for final qualification. Six relocated helper bodies and the single MCP exclusion constant have unchanged ASTs. The old controller keeps direct-call compatibility imports; supported custom turn-context providers, mounted injected callbacks and the existing controller settings-reader seam retain live action-time lookup. Ruling: arbitrary external rebinding of a relocated helper alias in its former module is outside this mechanical relocation contract. No positive repository consumer or documented override contract uses that route; the existing negative hook-currentness control does not traverse the moved producer. Adding four callback getters solely for that hypothetical route would recreate the coupling being removed. Direct helper signatures/calls and the supported provider customization route remain available.

Static preflight adds no Ruff findings: controller 61 to 60, session 10 unchanged, new producer/tests clean. Existing formatter transformation hunks remain9/1 on those large modules; new files are formatted. Controller size drops30484 to30191 lines and session6683 to6601; the new producer is372, yielding three fewer production lines overall. The controller still exceeds its existing29299 ceiling; that pre-existing qualification gap is not hidden by raising the ratchet. Actual whole-app boot/module census and latency remain separately unqualified.


## Final verification

Final source-current targeted verification passed 31 cases: eight new capture controls, ten existing asynchronous capture controls, nine existing helper/context controls, three original wiring controls and one real mounted Send journey. The original `test_console_successful_send_does_not_leave_empty_send_tooltip` exercises the real Console harness, Send click, persistence/controller/gateway and composer clearing with its existing isolated-profile marker and transport substitute. Its existing ordinary-send fixture disables exchange capture; no broader exchange-context qualification is inferred.

One additional original mounted Send-button control failed before Send at `app_factory.load_settings` with `raw_source_selection_changed`. The preserved unchanged `8fff47b36c` baseline reproduced the same failure. Relevant test, fixture and configuration-source files are unchanged between that baseline and this slice. No test assertion, production gate or timeout was changed; the failure remains recorded separately. Existing tests do not directly qualify nonempty dictionary/world-info database capture, although relocated helper ASTs are unchanged.

Every candidate and baseline batch proved normal contained Job emptiness at parent exit, native identity release, pipe/monitor retirement, current containment source, zero diagnostic overflow/races and private-profile removal. This is scoped diagnostic custody, not universal production native-cleanup evidence. Source/test hashes matched throughout. Final Ruff adds no findings (controller 61 to 60; session 10 unchanged); new files pass lint/format and existing formatter transformations are identical. Independent final review found no material issue in this scope.

Evidence: `.superpowers/sdd/2026-10-06-console-shared-tool-preparation/checks/capture-final-*`, `capture-static-final.json`, and baseline `capture-mounted-baseline.*` in the retained comparison worktree. No timing or full-suite run was performed. The pre-existing controller size ratchet, cold boot/module census, other-host checks, actual 100 ms feedback and subsecond action-to-adapter qualification remain open. The next prerequisite is app-owned initial hook review using the resident decision host, followed by exact-generation received admission/promotion.
