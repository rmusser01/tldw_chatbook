# Console approval clarity and responsiveness

Date: 2026-10-05

Status: Approved for implementation planning after owner review and source review corrections. Implementation has not started.

ADR: [ADR-221](../../../backlog/decisions/221-console-approval-interaction-and-feedback.md) (Accepted).

## Purpose

Console users report that permission cards look cluttered, do not explain what they authorize, leave the meaning of Approve all unclear, and appear after a pause in output. Users also report a delay after answering. The common interaction should explain the action and target, state the scope of consent, and acknowledge the answer promptly while the tool and model continue their work.

This design simplifies the existing decision card and adds truthful feedback around its handoff. It retains the current permission lifetimes and enforcement boundaries. The causes of the reported pauses must be measured before choosing performance fixes.

## Agreed decisions

| Area | Decision |
| --- | --- |
| Temporary permission lifetime | Retain each provider's current lifetime and label it **Until Chatbook exits**. Explain its actual scope: most grants use a permission profile, while raw shell is limited to this Console chat and also clears on Disarm. |
| Single request | Show **Allow once**, **Deny**, and **More options**. Allow once and Deny submit immediately. |
| Broader grants | Place supported temporary, exact-input, and permanent tool grants under More options. |
| Ordinary batch | Offer counted, immediate one-time approval and denial when every request supports those actions. |
| Mixed batch | Support individual choices and an explicit final summary such as **Apply: 2 allowed, 1 denied**. Requests requiring individual review need an explicit choice. |
| Feedback | Distinguish submitting a decision, runtime acceptance, tool execution, and subsequent model output. |
| Proposed performance targets | Visible click feedback within 100 ms and an actionable card within 200 ms after completed local checks, measured at p95 on recorded hardware. |

## Current behavior and evidence

The research baseline is commit `0f8e97fe16519455507654f9ac23f1cf97531372`. The sources below explain behavior; they do not establish the reported latency on users' machines.

- [ChatApprovalCard](../../../tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py) renders a scope Select and immediate single-row buttons alongside Approve all, Submit, and Deny all. Approve all changes selections without submitting, deliberately excludes raw shell, and can fall back from approve_once to approve_session for a narrowed row.
- The same card labels approve_session as This session, with supporting copy saying that it lasts until Chatbook exits. [The MCP service](../../../tldw_chatbook/MCP/unified_control_plane_service.py) stores these grants in memory by permission profile, server, and tool. They can cover other chats using that profile and can be revoked before exit. Existing risk floors still apply.
- Argument summaries have an 80-character aggregate budget and normally a 34-character per-value budget. The ordinary approval card has no complete-argument Details disclosure today. Separate transcript tool markers retain an argument display capped at 16,000 characters. Complete redacted Details is part of this redesign, not an existing feature.
- Raw-shell temporary grants belong to one Console chat and clear on Disarm or exit, under [ADR-093](../../../backlog/decisions/093-raw-and-virtual-cli-execution-boundaries.md). They are not the MCP service's profile-scoped grants.
- Integrated MCP review coalesces same-tool approvals into a shared name-keyed stamp. Its scope rank omits allow_matching, so mixing Remember these inputs with Allow once for repeated calls can save no exact-input rule. The new presentation must account for this existing limitation without changing stamp ownership.
- [The provider bridge](../../../tldw_chatbook/Chat/console_agent_bridge.py) collects structured tool-call data without displaying it as transcript text. [The agent runtime](../../../tldw_chatbook/Agents/agent_runtime.py) performs preparation and guards before requesting human review. These intervals are candidates for a perceived pause, not proof of a blocked UI thread.
- [The approval controller](../../../tldw_chatbook/Chat/console_chat_controller.py) starts the optional advisory summarizer on a background thread. The approval resolver sets a threading.Event, which wakes the wait immediately; the one-second polling interval is not a fixed answer delay.
- Two targeted diagnostic launches stopped before tests ran with `Recovery required: recovery_scope_uncertain`, including a launch with a private profile selected before imports. They provide no approval latency measurement and no passing test receipt. A runnable, isolated environment is required for performance qualification; the recovery guard must not be bypassed.

## Request information

The first line names the action and target. For known tool contracts, use descriptions such as **Read a file** with `notes/project.md`, or **Write a file** with its destination. For an unfamiliar MCP tool, use **Call server tool** with its actual name and supplied targets. Do not invent effects from a tool name, a URL-shaped argument, or model prose.

Show the owning workspace, private scratch, remote binding, or server and the captured permission profile name when applicable. Do not substitute whichever workspace happens to be selected when the card updates. Keep code-owned effects and permission reasons visible, including definition changes and paths that will fail even if approved.

Show identifying targets before payload excerpts. Long paths must remain distinguishable: wrap them or use a disclosure that exposes the complete target, rather than presenting an ambiguous truncated prefix as the entire target. Grouped requests expose every target and count actual calls, not just rendered rows. Where the runtime has only a shared name-keyed verdict, state that the grouped requests share one decision; do not offer independent choices the runtime cannot honor.

Details provides access to the complete captured arguments after display redaction, the tool identity, and technical metadata. Build its content on demand and render large payloads or target lists in bounded pages with an explicit omission or continuation count. A collapsed card must not eagerly stringify, mount, or rewrap every argument or target on every update. Every page remains tied to the same request snapshot; none is a new file read or remote fetch. Complete targets remain discoverable even when only a bounded preview is painted. Include large write bodies, long paths, many targets and maximum-size commands in rendering checks.

Raw shell retains its complete command, shell, working directory, timeout, and existing warnings in its primary review surface. Its dedicated read-only command viewport can scroll within the existing command limit; it is not reduced to a friendly paraphrase or buried inside generic Details.

The agent's rationale remains labeled **Agent stated reason** and visually separate from verified request facts. Optional generated summaries remain advisory under ADR-090, arrive asynchronously, and cannot reset choices, change focus, or delay the card. Preserve current redaction, content bounds, control stripping, and markup escaping. Display preparation must not read a file or contact a server to generate an unauthorized preview.

## Single request interaction

The primary actions are **Allow once**, **Deny**, and **More options**. Remove the duplicated Select, Submit, and batch action bar from this common view. More options and Details are noncommitting disclosures.

Opening More options exposes only choices supported by the captured request and its batch. Selecting a broader scope shows its explanation and an explicitly named commit action before it takes effect. The common one-time actions still need only one gesture. More options and Details are card-owned inline disclosures or popovers, not new modal screens. Opening or closing them does not commit, deny, stop the run, reset its deadline, or pause its existing answerable-time clock merely because a disclosure is open.

| Scope | Explanation |
| --- | --- |
| Allow once | Allow this captured call only, or these N captured calls when they share one verdict. |
| Until Chatbook exits | For MCP-backed, built-in and ordinary local grants: allow calls to this tool across chats using this exact permission profile until exit or revocation. For raw shell: allow future raw shell commands in this Console chat until Disarm or exit. Existing mandatory checks still apply. |
| Remember these inputs | Remember the exact captured argument set in the displayed permission profile. For a supported shared-verdict group, name the number of distinct captured argument sets. Show the existing removal route. |
| Always allow this tool | Remember a tool-level allow in the displayed permission profile. Show the existing change or revocation route and any mandatory checks that still apply. |
| Deny | Deny this captured call, or all N calls sharing this verdict; preserve the existing model-facing refusal and optional reason. |

Scope copy comes from the actual provider's grant domain, not the decision string alone. Raw shell must also state that its grant covers future commands, not just the displayed command. A persistent rule written to Default may affect named profiles inheriting that policy; explain this under More options when Default is the write target. Temporary MCP cache grants, by contrast, use the exact profile key.

Do not advertise a remembered choice the consuming provider cannot honor. Exact-input rules are unavailable for tools whose high-risk floor refuses them. In this first version, also withhold per-row Remember these inputs where repeated-tool requests share a stamp that cannot preserve independent exact-input choices. Explain the limitation under More options and retain one-time approval. Do not migrate stamp ownership in this UX work or silently downgrade an already selected remembered scope.

Exact-input rules use canonical JSON of the original captured arguments, not schema-defaulted inputs, resolved paths, redacted text, or clipped previews. The effective option inventory must consider the full batch as well as each request. Redaction affects only presentation; the captured call and its argument identity remain unchanged.

Retain the optional denial reason as a disclosure. It must not add a required step to Deny or change its one-call lifetime.

## Batch interaction

Each request shows its action and target. A compact summary names the total number of calls. A single grouped verdict covering several calls uses the batch presentation and exposes all of its targets, even when it occupies one row.

When every request supports one-time bulk approval and none requires explicit individual review, **Allow these 3 calls once** submits approve_once for the complete captured batch immediately. **Deny all 3 calls** submits denial immediately. Counts are computed from the entire batch, including calls outside the current scroll viewport. Every call must be known before either action becomes available.

Bulk approval never falls back to approval until exit or a persistent grant. A raw shell request and any other request excluded from bulk approval make the batch require individual review. Show **Review individually** instead of a bulk approval action that appears to cover everything while excluding rows.

In individual review, row choices are visibly staged until Apply; ordinary rows retain the current initial Once choice. Use selection controls and a visible **Choices apply when you press Apply** hint, rather than making a row action look like the single-call immediate commit. Broader scopes remain under More options. The final action describes the complete result, for example **Apply: 2 allowed, 1 denied**, and identifies any broader grant beside the relevant row.

Raw shell keeps its displayed Deny default but remains unresolved until an explicit user choice. Deliberately choosing an already displayed Deny counts as review; initializing a control or synchronizing its value does not. Track that gesture separately from the decision value. Other explicit-review requirements come from the producer; do not introduce new per-row review floors merely because a request is unfamiliar. Deny all remains available for mixed and raw-shell batches whenever every call supports denial, and constitutes an explicit denial of the whole captured batch.

One submission resolves one complete round through the existing gate. There is no execution of a partial batch while other rows remain undecided. Counts describe calls; a broader grant's additional scope remains visible and is not represented as merely one allowed call.

## Appearance and keyboard behavior

Compose the card from the existing design tokens and component patterns under ADR-150 and ADR-161. Use compact action controls, one clear approval boundary, and hierarchy through text weight and placement. Avoid repeated headers, nested borders, or decorative motion. Keep the action and target dominant, authority and effects readable, and technical details subordinate.

Keep the actions visible while request details scroll. Verify native terminal geometries of 80x24, 120x40, and 170x48 with Inspect open and closed. The labels, required warnings, and denial controls must remain reachable. Resize must preserve staged decisions, focused controls, Details expansion, and scroll position.

Preserve Alt+A and ordinary Tab navigation. Alt+A lands on a noncommitting request summary or Details control, never Allow once, Apply, or another commit action; Enter there cannot authorize a call. Arrival must not steal composer focus or turn a queued Enter intended for typing into approval. Escape closes the active disclosure without submitting or denying. Closing restores focus to the still-current opener; if the round ended or the opener disappeared, use an eligible neutral target in the owning view. Shortcut hints follow ADR-031 and name only implemented actions.

The layout must remain usable in dark and light themes, with readable rest, hover, focus, disabled, and applying states. Scope and status must be expressed in text, not color alone. Do not change shared token values merely to restyle this card; any necessary feature values belong in the governed token system.

## Decision feedback

| State | Trigger and visible behavior |
| --- | --- |
| Receiving tool request | Show this stage only when the provider exposes actual tool-request activity. If it does not, retain the truthful model-response state. Incomplete arguments are never actionable. |
| Checking permissions | Complete request preparation or permission checks are actually running. The card is not yet ready for a decision. |
| Waiting for your decision | A real, answerable pending round owns the complete card. Generic permission review alone does not imply a human wait. |
| Applying decision | A current card receives a committing gesture. Lock that submission and paint feedback immediately. This does not claim runtime acceptance. A newer authoritative state can replace it before first paint. |
| Decision received | The controller admits the matching gesture. This confirms receipt only; it does not yet claim the host's final outcome, remembered grant success, or tool dispatch. |
| Decision accepted | The host settles the matching round to the user's decision after cancellation and deadline arbitration. Show the actual settled result and next known stage. A remembered grant is described as saved only after its owner confirms success. |
| Starting tool | An accepted call is queued or entering dispatch and its remaining checks. This does not claim an external process or request has started. |
| Tool running | Its execution owner confirms actual start. An early runtime dispatch marker alone is not proof of backend start. Retain Starting tool if no more precise observation exists. |
| Waiting for model response | A subsequent model request is actually in flight. Do not label this as a permission delay. |
| Decision no longer pending | Timeout, cancellation, replacement, or another authoritative resolution wins. Do not show the rejected gesture as accepted or retry it against a newer round. |

Approved one-time copy can read **Allowed once; starting tool...** after host acceptance. Denial can read **Denied; returning to agent...** while retaining the existing refusal outcome. Later cancellation, current-policy refusal, definition changes and grant failures update the owning status honestly; accepting a user choice is not a promise that execution will succeed.

Do not hold a transient Applying state for a minimum duration or wait for an extra paint before releasing the worker. A settled or later execution state painted within the feedback budget also counts as acknowledgment. Route late feedback to the owning session's existing tool or run status after the card clears, not to a replacement approval card. Acknowledgment observers do not retain the card or delay FIFO promotion.

If remembering permission fails but the current gate still permits the call, preserve that existing execution behavior and report **Allowed this call; permission was not remembered** with the existing permissions review route. Future calls may ask again. Do not infer success from a void best-effort persistence call or re-prompt automatically. Each selected grant needs an owner-reported outcome; absent or unresolved evidence never becomes a Saved claim.

Scrolling, navigation, and other chats remain usable throughout. The affected run waits; the application does not intentionally block while waiting for human input. Definitive-after-start tools preserve the existing finishing state and the fact that Stop cannot cancel them.

## Ownership and data flow

The controller and interrupt host remain the authorities for round admission, FIFO ordering, cancellation, deadlines, and resolution. The card renders a captured view and submits a complete decision map for that exact round. Legal choices, ownership, and target grouping derive from that captured request, not mutable global UI state.

Retain both the card's gesture-generation guard and the controller's round-id guard. Display updates and delayed summaries must preserve current choices. A repeated click, stale menu action, replaced round, or session switch cannot authorize a different request.

Decision receipt, host settlement and grant application are distinct ephemeral observations. The current resolver returns no acceptance result, and provider best-effort grant writes return no success receipt; an observer must obtain the fact from the owner at the relevant boundary rather than synthesize it from those calls. Include owning session, round and gesture generation in identity fencing, and captured call identity for per-call observations. Delivery order must not let an old Applying observation overwrite a newer settled or execution state.

These observations never grant access, weaken current-policy checks, or create a second verdict store. Keep display failures isolated from execution and teardown. Use ADR-195's lifecycle observer and ADR-210's region ownership. New Console presentation or controller responsibilities belong under UI/Console_Modules; the reusable card can remain at its existing home. Avoid a general screen or runtime rewrite.

Preserve the existing permission profile binding, file authority, exclusions, risk floors, hook refusals, definition checks, kill switch, positive timeout behavior, default indefinite human wait, and current denial semantics. No permission-store schema, provider wire format, configuration default, or new permission lifetime is introduced by this design.

## Timing investigation and targets

Capture timestamps for complete captured tool-request availability, completion of preparation and local checks, first actionable card paint, input arrival before UI queueing, committing handler dispatch, first visible feedback, controller receipt, host settlement, grant application, worker release, dispatch and confirmed tool start or completion, and next model output. Measure the whole last-visible-output-to-card interval as well as its parts; local budgets must not hide a slow preparation stage.

Measure provider argument generation, local preparation, input queueing, UI dispatch and paint, decision handoff, tool work, and provider response separately. Sample the UI thread during a gap. In the browser, measure input-to-presentation on the browser's own clock and server spans on the server's clock, correlating them by application-generated identifiers. Never subtract uncalibrated browser and server monotonic timestamps. Record transport-inclusive feedback separately from the local application budgets, including input and resize traffic.

The proposed p95 targets are:

- No more than 100 ms from app input arrival, before committing-handler queueing, to visible feedback for that gesture. Feedback can be Applying, authoritative acceptance or a newer execution state; no additional transient frame is required.
- No more than 200 ms from completed local checks to a fully actionable card on the viewed Console. Parked rounds, deliberate FIFO waiting, and hidden Console time are measured separately and are not counted as paint latency.

Record hardware, operating system, terminal or browser transport, app revision, and load. Collect at least 40 samples per warm distribution and report sample count, median, nearest-rank p95 and maximum; retain the bounded raw timing samples. Report first-use behavior separately and do not pool different transports, loads or batch shapes. These are local engineering targets, not current measured performance or guarantees on arbitrary hardware or networks.

Collect timestamps at production boundaries. Do not quote absolute latency from Pilot press plus pause, which this repository has shown can measure harness callbacks. A queued update or mounted widget is not proof of an actionable painted card. Correlate actual native output or browser presentation with independently inspected frames. When endpoints use different clocks, qualify the overall metric only with a calibrated offset and bounded measurement error; otherwise report component spans and leave that overall target unverified. Server frame emission alone is not a browser-visible feedback or actionable-paint receipt. Preserve the browser input-to-visible-feedback result even when the local application budget passes.

Diagnostics contain timings, stage codes, counts, and application-generated correlation identifiers. They must not log arguments, paths, commands, rationale, conversation text, credentials, or externally supplied identifiers. Keep diagnostics local and bounded, with no external telemetry or additional model calls.

Select optimizations from the measured boundary that exceeds its budget. Candidates include repeated preparation, synchronous UI work, widget reconstruction, CSS matching, or blocking state access. Do not assume a cause, shorten the Event wait as an answer-delay fix, bypass checks, or increase bulk permission scope to make the app appear faster.

## Acceptance and verification

1. Users can identify the action, all targets, authority, and current choice's scope before committing. Unknown tools retain honest technical identity rather than a fabricated explanation.
2. Single-call Allow once and Deny submit with one gesture. More options states each provider's real scope, including raw shell's chat and Disarm boundary and Default's persistent inheritance. Unsupported remembered-input combinations are not offered.
3. Counted batch actions commit exactly the captured calls. Bulk approval stays one-time, explicit-review defaults do not substitute for gestures, grouped verdicts do not offer unsupported independent choices, and Deny all remains available for eligible mixed batches.
4. Feedback is visible promptly without an artificial frame delay. Receipt, final acceptance, grant application and backend execution are distinguished. Stale observations cannot update a replacement card or override a newer state.
5. Keyboard access, focus, staged choices, Details, and scrolling survive summary arrival, resize, navigation, remount, and queued rounds. Denial and warnings remain readable at the stated sizes in both theme families.
6. Existing runtime enforcement, raw-shell review, denial reasons, finishing behavior, grant persistence and revocation, timeouts, and cancellation remain correct through real invocation paths.
7. Separate native and browser timing receipts identify the pre-prompt and post-answer costs. Performance claims use recorded conditions and production boundary measurements, and the proposed targets are checked on that hardware.
8. Targeted automated, static, generated-artifact, and live checks cover the changed functionality. Verify token governance and regenerate the CSS bundle from source modules if styles change. The full suite requires the user's opt-in under AGENTS.md.

Use real gestures and rendered frames for appearance, plus controller and invocation tests for actual decisions. Cover mixed same-tool scopes; captured versus normalized or redacted arguments; profile inheritance; raw-shell chat and Disarm scope; deliberately choosing default Deny; large lazy Details; More options returning after replacement; Alt+A then Enter; slow summaries; fast coalesced feedback; reordered observations; cross-chat feedback; Stop and timeout between receipt and settlement; grant-write failures; and normal FIFO promotion. Reuse private-profile isolation before any application import; do not qualify against real user data or bypass storage recovery.

As an optional usability check, use representative rendered cards to ask users to identify the action, affected target, and scope of each choice without explanatory coaching. Include an ordinary file call, a mixed batch and a raw-shell grant. This can expose misleading copy that geometry and invocation tests cannot judge; it does not add a new implementation gate.

## Alternatives

| Approach | Assessment |
| --- | --- |
| Labels and spacing only | Smaller change, but keeps duplicated single-call controls and does not provide a decision acknowledgment contract. |
| Focused card and measured handoff | Selected. Improves the common path and makes pauses observable while retaining existing permission authority and lifetimes. |
| New chat-scoped permission model | Rejected for this work. The user explicitly retained the current temporary lifetime; changing scope would require a separate policy design. |

## ADR check

ADR required: yes

ADR path: backlog/decisions/221-console-approval-interaction-and-feedback.md

Reason: This changes the lasting approval interaction and defines a runtime-to-UI acknowledgment contract. ADR-221 records those choices without replacing existing enforcement. Relevant decisions are [ADR-032](../../../backlog/decisions/032-local-agent-tool-permission-boundary.md), [ADR-067](../../../backlog/decisions/067-indefinite-human-approval-waits.md), [ADR-090](../../../backlog/decisions/090-permission-request-context-summaries.md), [ADR-093](../../../backlog/decisions/093-raw-and-virtual-cli-execution-boundaries.md), [ADR-150](../../../backlog/decisions/150-design-token-system-and-design-language.md), [ADR-161](../../../backlog/decisions/161-component-pattern-library.md), [ADR-195](../../../backlog/decisions/195-console-live-tool-call-presentation.md), and [ADR-210](../../../backlog/decisions/210-console-region-ownership.md).

The [implementation plan](../plans/2026-10-05-console-approval-ux-and-responsiveness.md) records existing TASK-34564 through TASK-34569 in dependency order. They remain To Do until execution starts, when each task receives its Implementation Plan and ADR check. Exact additional performance fixes require baseline attribution before code. Product implementation and latency qualification are subsequent work.

## Research supporting the design

Permission explanations should state how access serves the current task in brief, concrete language. [Apple privacy guidance](https://developer.apple.com/design/human-interface-guidelines/privacy/) and [Android runtime permission guidance](https://developer.android.com/training/permissions/requesting) support contextual explanations and an understandable refusal path.

Action buttons should directly answer the request rather than rely on generic confirmation labels. [Microsoft dialog guidance](https://learn.microsoft.com/en-us/windows/apps/develop/ui/controls/dialogs-and-flyouts/dialogs) supports concise, specific responses. [VS Code approval documentation](https://code.visualstudio.com/docs/agents/run/approvals) provides a relevant example of separately named grant scopes; Chatbook retains its own existing scope semantics.

Synchronous work must stay outside the UI loop, and threaded workers marshal UI changes safely. [Textual worker guidance](https://textual.textualize.io/guide/workers/) supports this implementation constraint. These sources guide the design; they do not supply Chatbook latency measurements.
