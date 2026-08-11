# Console context and memory UAT / UX review

Date: 2026-08-10

Reviewer lens: senior UX/HCI designer

Personas: a first-time terminal user and a first-time Chatbook user

Surface: Console model settings, conversation context policy, first send, compaction, and canonical Settings

## Goal

Verify that a new user can understand and safely control the current conversation's token budget and compaction behavior without being prevented from sending a valid message. Identify defects, friction, trust risks, and opportunities for improvement.

## Method

1. Review the normative context-memory design and the visible product language.
2. Exercise the real Textual controls with `Pilot` in a fully isolated profile.
3. Capture wide and narrow terminal states, keyboard traversal, validation, save feedback, and send admission.
4. Compare the current-conversation modal with the canonical Settings surface.
5. Append every observation to the findings register as it is discovered.

Severity: **S1** blocks the core task; **S2** creates major error risk or confusion; **S3** causes avoidable friction; **S4** is polish.

Disposition: **Fix now** is inside the hotfix contract; **Backlog** needs separate product work; **Keep** is a positive pattern worth preserving.

## Step-by-step UAT journey

| Step | New-user intent | Expected result | Status | Evidence / notes |
| --- | --- | --- | --- | --- |
| 1 | Open Console and identify whether it is ready to accept a message | Readiness and the primary action are obvious | Pass | `01-console-ready-120x42`; header says Ready and composer is visible. |
| 2 | Open the quick model settings | Model, response, conversation, and compaction concepts are distinguishable | Partial | `02-quick-model-settings-default-120x42`; rows are separated, but CM-001 and CM-004 weaken trust and comprehension. |
| 3 | Open Context and memory | The effective model capacity and current conversation policy are understandable | Partial | `03-context-memory-default-120x42`; structure is strong, but capacity provenance and save scope are misleading (CM-001/002). |
| 4 | Set a bounded conversation budget and choose compaction behavior | Validation is immediate, consequences are clear, and Save is scoped correctly | Partial | Blank Custom is rejected inline; 12,000-token override saves. Save scope remains ambiguous (CM-002). |
| 5 | Send the first message using an unknown model window | A fitting request sends; lack of verified capacity does not masquerade as known overflow | Pass with caveat | `06-bounded-first-send-succeeds-120x42`; user and assistant rows complete. The UI still overstates fallback capacity certainty (CM-001). |
| 6 | Reopen the conversation | The saved per-conversation policy is restored and visibly effective | Pass | Reopening quick settings shows the bounded policy still effective; repository/lifecycle regression tests cover durable restore. |
| 7 | Approach the compaction threshold | Ask, Automatic, and Off behavior is predictable and cost-visible | Partial | Controller matrix passes all three modes. Extra-call consequences are absent from the quick surface (CM-004). |
| 8 | Trigger a genuinely impossible request | The block explains the limiting material and offers relevant recovery | Pass after copy fix | The block now leads with the inability to fit, identifies mandatory material, states that summarizing older turns cannot help, and gives recovery. |
| 9 | Configure global defaults in Settings | Scope, save model, advanced behavior, and summary-prompt ownership are clear | Partial | `10-global-console-behavior-120x42`; draft scope and consequences are explicit, but controls are buried (CM-007/008). |
| 10 | Repeat at narrow width and by keyboard only | Controls remain reachable, legible, and ordered | Partial | Modal fits 68×21 and focus order is coherent. Key content is hidden without a fold cue (CM-005/006). |

## Findings register

| ID | Sev | Journey step | Finding | User impact | Recommendation | Disposition | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CM-001 | S2 | 2, 3, 5 | An unrecognized custom model is assigned the provider fallback of 4,096 tokens and labeled `Capacity is verified for the selected model.` | The interface presents a guessed default as provider-verified fact. Users may choose budgets or trust send safety based on false certainty. | Carry model-limit provenance into `ConsoleSettingsContextEstimate`; label fallback values as estimated/unverified and deep-link to Providers & Models for repair. | Backlog | `02`, `03`; `_resolve_token_limit_locally()` returns the provider/default fallback without provenance. |
| CM-002 | S2 | 3, 4 | Context view says `Save applies to this session only` while the product contract calls this a conversation. It also leaves `Save provider defaults` beside Save even though that action intentionally excludes context policy. | A new user can reasonably believe the bounded context policy is temporary or that `Save provider defaults` makes it global. Both mental models are wrong. | Say `Save applies to this conversation`; in Context and memory either hide the provider-default action or rename it `Save model defaults` and explicitly point global context defaults to F9 Settings. | Backlog | `03`, `04b`; `test_provider_defaults_write_excludes_memory_and_prompt_ownership`. |
| CM-003 | S3 | 2, 3 | The model view labels output reservation `Max tokens`, while Context and memory calls the same value `Response max`. | The user may confuse response length with total conversation capacity, the exact ambiguity this feature is meant to remove. | Use `Response max tokens` consistently in both views and validation copy. | Backlog | `console_settings_modal.py:570` vs `:714`. |
| CM-004 | S3 | 2, 7 | Quick settings shows `Compaction at 3,276 tokens` and an `Ask / Automatic / Off` selector without defining compaction, what Ask will do, or that Automatic can add a model call. | The quickest, most discoverable control asks for a consequential choice before the user has enough information to predict latency, cost, or data flow. | Add one concise consequence line, or make the threshold row actionable help: `Summarizes older turns; Automatic may add one model call.` | Backlog | `02`, `07`; global Settings contains the consequence copy but the quick popover does not. |
| CM-005 | S2 | 10 | At 72×24 the quick popover initially stops after the Request row. Conversation usage, compaction threshold/mode, `Context & memory…`, and Apply are below the fold with no `more` indicator. | A mouse-first new user cannot see that the requested feature exists. Keyboard users need nine focus stops before the action scrolls into view. | Add the established conditional `▼ more — scroll` fold indicator and keep the action row pinned, or shorten/collapse model search results at compact heights. | Backlog | `08`, `08b`; focus order recorded by the harness. |
| CM-006 | S2 | 10 | At 72×24 the full modal fits the viewport, but its scroll body clips mid-`Model capacity` after Safety margin with no visible fold indicator. The action row stays visible, making the incomplete body look complete. | Users can save without discovering Conversation budget or Compaction, or conclude that the modal contains only capacity telemetry. | Apply the Settings inspector fold convention to the modal body and focus/scroll the requested Context section to its first editable field. | Backlog | `09-context-memory-72x24`; modal region is 68×21. |
| CM-007 | S3 | 9 | Global context controls live below Rail presentation, paste handling, Chat images, and other Console Behavior sections. The initial 120×42 viewport does not show them. | The long-term/global version of the feature is hard to find by browsing, even after choosing the correct category. | Add an in-category section index or put `Conversation context and memory` nearer the top. Preserve `/` field search as the power-user path. | Backlog | `10-global-console-behavior-120x42`; the existing `▼ more — scroll` cue is a positive mitigation. |
| CM-008 | S3 | 9 | Advanced labels such as `Trigger (%)`, `Compact toward (%)`, `Summary max`, `Failure`, and `Carry forward` rely on the right-side focus inspector for meaning. | Users scanning the form cannot form a causal model before entering the fields; narrow layouts may separate the control from its explanation. | Use outcome-oriented labels (`Summarize when budget reaches`, `Reduce to`, `If summary fails`, `Keep after summary`) and keep exact percentages/tokens in the controls. | Backlog | `settings_screen.py:10844-10912`; focus guidance exists but is secondary. |
| CM-009 | S3 | 8 | The original genuine-overflow error led with `Conversation compaction cannot run safely`, framing compaction as the failure instead of explaining that the request itself cannot fit. | It repeats the alarming phrase from the regression and makes the user troubleshoot the wrong mechanism. | Lead with the failed user goal, name the limiting segment, explain why summarizing cannot help, then offer recovery. | Fix now | Updated controller copy and regression assertion in this PR. |

## Positive patterns to preserve

- The quick surface separates Request, Conversation, and Compaction into distinct rows.
- The full modal separates model capacity, conversation budget, and compaction policy.
- Switching to Custom requires a positive token value and keeps the modal open on error.
- A 12,000-token conversation override is stored independently from provider settings.
- A bounded fitting request sends successfully and receives a complete assistant reply.
- The 72×24 full modal stays inside the terminal; keyboard focus order reaches every context control and the three actions.
- Canonical Settings clearly labels its state as a global draft and explains that compaction adds a model call while preserving the transcript.

## Release assessment

The hotfix is functionally releasable after targeted verification: the original false block is gone, bounded sends work in the integrated app, and a proven overflow still blocks with clearer recovery. The remaining findings are UX and capability-provenance follow-ups. CM-001, CM-002, CM-005, and CM-006 should be prioritized because they affect trust, scope comprehension, or feature reachability; none reproduces the send-blocking regression.
