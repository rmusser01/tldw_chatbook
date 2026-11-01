# Console Copy all response trace

Date: 2026-10-28
Status: Approved for direct implementation on 2026-10-28, including
Select full message and uncapped note capture
Size: One bounded Console feature set

Task: [TASK-37300](../../../backlog/tasks/task-37300%20-%20Console-trace-copying-and-full-message-note-capture.md)

Plan: [Console trace copy and note capture](../plans/2026-10-28-console-trace-copy-and-note-capture.md)

## Intent

Copy one assistant response's entire retained trace as readable Markdown for
pasting into another chat for review. The user requested thinking, tool calls,
full results, and the final answer, including sections collapsed in Console.

The user also approved **Copy all messages**, **Create Note**, and
**Create+Go to Note**, asked to incorporate the review findings, and authorized
the implementation plan. The latest request also asks to expand selection
capture so a whole message can be preserved. The user approved the full-message
selection behavior below and removing the cap from both note actions.

## Approaches

1. **Dedicated response trace formatter (selected by the user).** Format one
   existing assistant turn from its retained message/activity data. Keeps the
   change focused and supports activity ordering and richer tool sections.
2. **Extend conversation Markdown export.** Add response scoping and richer tool
   formatting to the whole-conversation exporter. Shares more formatting code,
   but also changes a separate export surface and requires broader regression
   coverage. Not selected.

## Scope and interaction

- Add **Copy all** after **Copy selection** in the floating text-selection menu
  when the selected text belongs to an assistant response or one of its owned
  activities. It copies the enclosing response, regardless of the highlighted
  span. Existing Copy selection and text-only Copy retain their behavior.
- Also expose **Copy all** in the selected assistant response's
  existing **More...** menu, making the same operation available without a drag
  selection and without widening the primary action row.
- Resolve the response using existing assistant-turn ownership. An owned tool
  or thinking selection targets its assistant response. User messages and
  standalone activities have no Copy all response action.
- Capture the target and its retained content when the action is activated.
  Changes to selection or the active chat must not retarget the operation.
- Copy through the existing app clipboard path, dismiss the invoking menu, and
  restore its normal focus. Preserve the composer draft and message selection.
  Missing/stale targets or formatting failures leave the clipboard unchanged
  and report the problem. No new shortcut, preview dialog, or confirmation.

## Markdown contract

- Start with an Assistant response heading and its current completion state.
  If still running, identify the copy as an incomplete snapshot.
- Emit every retained displayable thinking block and owned activity in the
  existing response activity order, then the response body. Preserve all
  thinking blocks exactly once; do not infer unavailable chronology.
- Use headings for thinking and tool/activity sections. Each tool section
  includes its available name, lifecycle status, arguments, and result. Use
  fenced blocks for arguments and results, with fences that safely contain
  embedded Markdown fences. Preserve response-body Markdown and textual payload
  whitespace. All sections are expanded in the copied document.
- Prefer full retained tool output over the display preview. Include retained
  file-write diffs and other response-owned activity details with clear labels;
  avoid repeating the same result as both preview and full output.
- Collapse state, scrolling, and transcript virtualization do not change the
  copied content. Read underlying response data rather than rendered widgets.
- Apply no new quote or preview cap. Existing truncation markers remain visible;
  explicitly label known missing arguments/results or unreadable thinking.
  Proprietary thinking has an unavailable marker, never invented text.
- Use only retained, displayable response data. Private provider continuation,
  raw request/response captures, system prompts, unrelated messages, other
  branches, and binary attachments are outside this feature. Represent attached
  media by available descriptive labels. This is a readable response trace,
  not a new diagnostic archive or a reconstruction of uncaptured data.

## Conversation copying and incorporated review findings

### Copy all messages

Confirmed scope: the current chat's entire active branch,
including user messages and every assistant response's full retained trace.
Include older messages outside the mounted transcript window; exclude other
chats and inactive variants/branches. A running chat yields one consistently
captured snapshot, clearly marked incomplete where appropriate.

Compose the new response formatter across the current chat
rather than silently routing this to the existing **Copy as > Full transcript**
action. That exporter uses collapsed details blocks, does not currently format
tool arguments, and includes system rows. The new action should share the
response-copy fidelity and exclusions defined above. Existing conversation
export behavior should remain unchanged.

Keep **Copy all** (one response) and **Copy all messages** (current chat)
distinct through explanatory tooltips and scope-specific notifications. The
latter should name the message count and payload size. Large output must not
freeze Console or be silently capped. The current OSC 52 transport can report
that text was sent, but cannot verify terminal acceptance of a large clipboard
payload. A file fallback could be a later improvement, not an implicit addition
to this change.

Show the conversation-wide action after the
response-wide action in the selection menu, also for user-message selections
where no response-wide action applies. Do not add it to every message's primary
action row. Keep menu scrolling and keyboard access with the extra rows.

### Note labels and navigation

Approved labels: **Create Note** and **Create+Go to Note**.
The first stays in Console; the second creates the same note, then opens that
exact note in Library > Notes. A tooltip can spell out the compact label:
"Create a note from the selection and open it in Library > Notes."

In this checkout, the selection menu currently has only **Create note**. Its
handler saves the selected quote and shows a notification; no second
create-and-go selection action was found. A separate answer-capture flow offers
**Open note** after saving, and already uses the canonical note-id navigation
context. Adding the requested selection action therefore needs that behavior
wired explicitly, rather than assuming this is only a label change.

Preserve the note action's selected-text scope, while removing its existing
4000-character truncation as specified below. Saving selected text does not
implicitly include the enclosing response's thinking and tool trace.
Both note actions must share one creation path. Retain the returned note id and
navigate only after a successful save. If navigation fails after saving, report
that the note exists and retry opening it without creating another note. Keep
the Console draft and ongoing run intact, and respect existing note-persistence
policy, including temporary-chat rules.

### Full-message selection and note capture

The selection manager already supports arbitrarily long spans, and **Copy
selection** already reads them without the quote cap. The 4000-character limit
is applied by capped selection getters and action handlers, including note
creation. Raising that shared constant would also change side chats, quoting,
and review feedback, and would still leave another arbitrary cutoff.

The approved behavior is:

- Add **Select full message** to the existing selection menu for selectable
  user/assistant message bodies. It selects the body's complete current source
  text, including offscreen text, and keeps the menu available for copying or
  note creation. It is keyboard accessible through normal menu navigation.
  Tool/diff/thinking-only selections do not advertise this body-specific action.
- Both note actions read the full highlighted text with `capped=False` and save
  it verbatim plus the existing provenance footer. Remove the note handler's
  4000-character validation ceiling as well as the truncation at its event
  producer. Preserve strict text/nonblank validation and Notes authorization.
- Do not add another arbitrary character maximum for copying or note capture.
  A genuine storage/resource failure is reported without saving a partial note.
- Keep `SELECTION_QUOTE_CAP = 4000` for Add to chat, side-chat requests, review
  feedback, and bounded selection-highlight previews. Explain those action
  limits in documentation; a bounded visual preview does not shorten the saved
  selection. Show the full selection count when its preview is abbreviated.
- **Select full message** concerns the message body. **Copy all** remains the
  operation for body plus thinking, calls, results, and diffs. Selection remains
  single-row; no new cross-row selection model or terminal shortcut is needed.

This expands the prior draft's note cap; it does not imply that
every existing selection-driven action now accepts uncapped text.

## Implementation constraints and builder choices

Use a small, pure Python formatter and existing turn grouping and clipboard
transport. No new dependencies, schema, retention policy, provider integration,
settings, or whole-conversation export changes. Keep payloads out of logs.
Formatting large retained results must not block UI input; use the existing
worker conventions where needed. Reuse menu focus, scrolling, and design tokens;
rebuild generated CSS if the editable styles change. Module/API names and exact
heading punctuation remain implementation choices.

## Verification

Targeted formatter tests must compare complete output for thinking, multiple
tool calls, arguments, long full results, embedded fences, diffs, plain answers,
partial/failed responses, and known unavailable/truncated details. Verify response
isolation and that each retained thinking block appears exactly once.

Mounted Console tests must exercise Copy all with collapsed activities and with
an activity expanded, assert identical clipboard content, cover mouse/keyboard
entry points and short-menu containment, and verify stale-target handling plus
draft/focus preservation. Include data outside the mounted transcript window.
Run scoped static checks and affected selection/message-action regressions;
perform a disposable-state live interaction check. No full test sweep without
the user's opt-in.

Extend targeted coverage to whole-chat
ordering and scope, offscreen history, active-branch isolation, live snapshots,
large payloads, menu growth, and both note actions. Verify one note write per
activation, identical saved content, no navigation after a failed save, exact
note-id navigation, and recovery from a successful save followed by failed
navigation without a duplicate note.

For full selection, test exact preservation at 3999, 4000, 4001,
10,000, and 100,000 characters for both note actions; include Unicode and
whitespace, full-message selection from a partial span, offscreen text, empty
and stale rows, and a bounded highlight preview whose saved text remains full.
Check that quoting/side-chat/feedback remain capped at 4000 and that the
terminal-convention/global keybindings are unchanged.

## ADR check

ADR required: no

ADR path: backlog/decisions/068-console-text-selection-and-annotations.md;
backlog/decisions/195-console-live-tool-call-presentation.md;
backlog/decisions/210-console-region-ownership.md

Reason: a small additive copy action using existing response ownership,
displayable activity data, menu structure, and clipboard transport. No storage,
capture, authority, runtime, or long-lived application boundary changes.
ADR-031 key conventions and ADR-150/161 design language continue to apply.

The implementation plan and Backlog task cover these decisions and identify
the approved full-selection scope. Implementation evidence is recorded in the
task's Implementation Notes.
