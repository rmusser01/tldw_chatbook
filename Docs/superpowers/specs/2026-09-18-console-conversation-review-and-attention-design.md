# Console conversation review, rows, and attention

Date: 2026-09-18

Status: Design approved in conversation; written specification awaiting user review.

## Scope and owner decisions

Improve the **Conversation Inspector modal**, the Console Conversations list,
and conversation-row icons/actions. The **Inspector sidebar is explicitly out
of scope**: do not redesign its sections, layout, navigation, or behavior.

The user approved:

- Manual Mark as unread, cleared by successfully opening the chat again.
- A marked current chat remains unread until the user leaves and returns.
- Marks survive restart; automatic restoration alone does not clear them.
- Mark as read is available without opening the chat.
- Context/next-send inspection and usage/costs are the modal's priorities.
- Conversations rows visually follow Workspaces rows.
- One right-hand conversation icon replaces the left appearance button and
  right asterisk. The existing conversation menu includes appearance editing.
- Attention and unread indicators temporarily override the saved custom icon.
- **Every state icon must represent the state it indicates**, not an arbitrary
  geometric code the user must memorize.

The August 18 Inspector spec's goal describes the cost-counter entry point; it
does not establish exchange inspection as the entire modal's primary job.

## Architectural decision

ADR required: yes

ADR path: `backlog/decisions/171-console-conversation-review-and-attention.md`

Reason: establish durable manual-read semantics and consistent cross-surface
attention presentation, and change the modal's long-lived information architecture.
Reuse ADR-010's local marks, ADR-083's workspace ownership, ADR-085's operational
receipt authority, ADR-069's automatic-instruction disclosure limits, ADR-031's
keyboard rules, and ADR-150's design tokens.

## 1. Conversation Inspector modal

### Navigation and entry points

Keep a single modal with **Context**, **Usage & cost**, and **Exchange history**
views. The cost-counter action opens Usage & cost. The existing context action
opens Context at its next-send preview. A generic Inspector command opens Context.
An explicit historical-turn link opens that turn's appropriate detail.

Show the conversation title in the modal header, retaining immutable session,
conversation, and profile identity for every asynchronous operation. Changes in
the underlying target must not silently retarget the modal. Keep Close visible
and support Escape using the existing safe-dismiss contract; restore focus to
the invoking control when it still exists.

Do not add an Overview page. Direct task entry avoids another navigation step.
Do not move inspection into a separate application screen.

### Context

Place a compact summary above the content: model, estimated next-send tokens,
and context capacity/remaining allowance when the existing budget service knows
them. Follow the composer's accounting; distinguish input capacity from output
reservation. Unknown capacity is labeled unavailable, not displayed as zero.

Replace nested Next Send tabs with one section list and a readable detail area.
The list clearly groups **Current conversation** and **Next send preview**.
Current conversation exposes the existing retained-history/context information;
the preview exposes prepared system instructions, messages, tools, and staged
sources when available. Labels and counts come from the actual snapshot, not
invented estimates of which content will reach the provider.

Selecting a section displays readable content. Raw JSON is an explicit alternate
view of that selection. Avoid a chain of turn/call/section/message expanders for
ordinary context inspection. Large bodies are loaded/rendered lazily.

Display preview freshness and whether a run prevents rebuilding it. Refresh
rebuilds the preview through the existing snapshot/preparation seam; it never
sends a message. Preserve the last valid content during refresh and distinguish
loading, stale, failed, empty, and ready states. Place Copy and Save beside the
content they export and label their scope explicitly.

Automatic project-instruction bodies remain confined to the explicit disposable
Next Send preview. Summaries, current-conversation sections, row tooltips, logs,
and incidental tab mounting must not reveal those bodies. Preserve redaction,
ephemeral-save restrictions, capture policy, and explicit disclosure controls.

### Usage & cost

Show conversation totals above an aligned turn list. Rows identify the turn,
input/output tokens, cost, and the basis of the values. Distinguish reported
usage from estimates, and known zero cost from missing pricing. Partial totals
must say that some usage is unpriced or unavailable.

Select a turn to inspect its token categories and individual calls. Keep the
turn identity stable during streaming and refresh. Do not derive accounting
from the currently rendered transcript or replace the existing authoritative
usage/pricing seams. Preserve any abandoned/cancelled capture distinctions.

### Exchange history and controls

Keep historical request/response inspection reachable as a third view. Use
turn/call selection plus a readable detail area and lazy disclosure. Preserve
the adapter-boundary caveat and honest missing-capture states. Capture settings
and safe/full trace-view controls live in the historical inspection context;
they retain all existing guards and do not silently alter Next Send policy.

### Responsive and keyboard behavior

At sufficient width, section/turn selection and detail sit side by side inside
the modal. At narrow sizes, selecting an item opens its detail within the same
modal, with a visible Back control that restores list selection and scroll.
Each pane has one scroll owner. Summary, navigation, and Close remain reachable
at 80x24; content areas may scroll without trapping actions below the viewport.
Tab follows visual order; list arrows select; Enter opens detail. Bindings are
local and truthful and never override reserved terminal/global shortcuts.

## 2. Consistent conversation rows

Conversations stays a flat list; Workspaces keeps its tree and ownership rules.
Match compact resting-row height, title alignment, focus/selection treatment,
and right-edge menu placement. Use the existing design tokens and component
states. Do not add a duplicate Unread or Starred location or change ordering.

Measure titles in terminal cells, reserving space for the right icon before
truncating. Full titles remain available on pointer hover and keyboard focus.
Favourites keep a compact star property indicator; the star is not the action
button. Selected state uses the common row treatment, not another status icon.

Resting rows are compact. Active progress, pending decisions, and other
meaningful activity retain concise textual information, including current
subagent/progress counts. Do not discard existing subagent hierarchy to achieve
density. Preserve stable row IDs, pointer press/release targets, keyed refresh,
scroll positions, drafts, ownership, search, and bounded paging.

## 3. Representative attention icons and one action menu

The right-hand control always opens the conversation's existing menu. Its glyph
reflects status; its activation behavior does not change with that status.
The menu gains **Change icon and colour…**, **Mark as unread**, and **Mark as
read** as appropriate. Existing Favourite, Change status, Archive, Rename,
Copy, and other actions retain their availability and consequences.

Use this representative vocabulary for conversation-row presentation:

| State | Icon | ASCII fallback | Text explanation |
| --- | --- | --- | --- |
| Approval waiting | ✋ raised hand | `[approve]` | Approval required |
| Blocked work needing intervention | ⛔ no-entry sign | `[blocked]` | Blocked — actual reason |
| Failed work needing attention | ✗ cross | `[failed]` | Failed — actual reason |
| Running | ⟳ circular progress arrow | `[running]` | Running |
| Paused work | ⏸ pause | `[paused]` | Paused |
| Stopped/cancelled result awaiting acknowledgement | ⏹ stop | `[stopped]` | Stopped or Cancelled, matching the outcome |
| Manually unread conversation | ✉ envelope | `[unread]` | Unread |
| Successful background result not yet seen | ✓ check | `[ready]` | New result ready |
| Ordinary chat | Saved custom icon, otherwise 💬 speech bubble | `[icon]` or `[chat]` | Conversation actions |

Priority is approval, blocked, failed, running, paused, stopped/cancelled awaiting
acknowledgement, manual unread, successful unseen result, then custom/default.
The first three refine the approved action-required/failure priority. Use
actual operational state and outcome, never infer state from title or glyph.
An ordinary old failure does not remain attention-worthy after its existing
acknowledgement policy says it is handled. Paused states use the pause symbol
only when an authoritative producer actually reports a resumable paused state;
this design does not introduce a new execution state.

Custom icon/colour remain stored while overridden. Attention icons use semantic
status colours; custom colour cannot camouflage a failure or approval. Restore
custom appearance once every applicable override clears.

Explain the dominant state on hover and keyboard focus. Add concise visible
row text for action-required conditions. The menu's status summary lists
simultaneous conditions, for example “Approval required · Unread,” so choosing
one glyph never loses another state. Keyboard access uses the existing row-menu
convention, including `m` when the row owns focus. Opening the menu does not
open the conversation or acknowledge any receipt.

Use the existing glyph fallback seam, with deliberate text presentation where
supported. Reserve a stable action slot within each rendering mode based on
cell width, including the longest ASCII fallback. Status changes must not
move the click target. Qualify the representative symbols in supported
terminals; fall back to the explicit ASCII state when a glyph cannot be
rendered legibly. Do not globally replace unrelated Console/sidebar glyphs.

## 4. Manual unread lifecycle and authority

Store a dedicated `manual_unread` mark using ConversationLocalMarksService.
It is local to the data profile, durable, and separate from conversation
status, custom appearance, sync, server payloads, workspace membership, and
operational receipt acknowledgement. Reuse the existing extensible marks table;
no new schema is intended. If inspection proves a migration necessary, revise
the plan and ADR before implementation.

Mark as unread applies to saved local conversations. Unsaved chats follow the
existing menu pattern with a clear “Send or save this chat first” reason.
Failure to store a mark produces an actionable error and no false success UI;
ordinary chat opening remains available when optional marks storage fails.

A current chat marked unread stays unread through repaint, streaming, opening
or closing menus/modals, screen refresh, and automatic restart restoration.
Clicking its already-current row is a no-op for read state. Leaving the
conversation and deliberately returning clears the mark only after its exact
destination successfully renders. Switching away from Console and deliberately
returning to that chat also counts as leaving and returning; incidental modal
closure and automatic screen restoration do not.

Cover native tabs, Conversations, workspace chat rows, Ctrl+K, and existing
explicit resume/handoff routes. Cancelled, missing, failed, or stale-profile
activation does not clear the mark. Background auto-wake is not a user visit.
Mark as read clears only the manual mark, not failed/blocked/unseen operational
receipts; their existing consequence-aware policies stay authoritative.

Fence post-render clearing to the captured profile, conversation, activation,
and mark revision so an old callback cannot clear a newly reapplied mark.
Serialize mark writes and reconcile all visible projections after commit.
Display unread on an existing Ctrl+K result when present, without changing its
Active membership or operational receipt rules solely because of the mark.

## 5. Ownership and implementation slices

- **TASK-32826:** manual unread, representative conversation attention
  presentation, combined right action control and appearance menu integration.
- **TASK-32827:** compact consistent Conversations/workspace chat rows, using
  TASK-32826's action and attention contract.
- **TASK-32828:** Conversation Inspector modal navigation and context/cost UX.

Likely integration points are the existing local marks service, conversation
menu model/widget, Console workspace/session controllers, conversation/browser
and workspace-tree projections/widgets, and the Inspector modal's snapshot,
usage, capture and export seams. Keep attention presentation as a pure mapping
from authoritative state; do not create a second execution/receipt owner.

Implementation plans must link ADR-171. Add task plans only when their task
enters In Progress. Leave implementation tasks To Do until execution begins.

## 6. Verification and acceptance

Run targeted checks only; a full suite requires a separate user request.

- Real SQLite tests prove local persistence, profile isolation, deletion
  handling and independence from sync/receipts.
- Activation tests cover mark-current, leave/return, same-row click, modal
  close, explicit vs automatic restore, cancelled/failed navigation, stale
  callbacks, rapid re-marking, and native/Ctrl+K/workspace opening paths.
- Pure presentation tests cover each state, overlaps, representative glyphs,
  ASCII mapping, custom appearance restoration and unknown-state honesty.
- Mounted row tests exercise pointer targeting, focus, keyboard menus,
  long/wide titles, updates while pressed, paging and scroll stability.
- Modal tests cover entry-point selection, context/preview separation,
  token/cost bases, unavailable pricing, partial totals, freshness, empty/error
  states, lazy large-history detail, privacy and export guards.
- Render the production stylesheet at 80x24, 120x40 and a wide viewport;
  exercise Back/Close and all important actions with keyboard and pointer.
  Compare representative glyph rendering in iTerm2 and Windows Terminal under
  the existing live-verification requirements; record any unavailable platform
  as missing evidence, not a successful check.
- Run touched-file lint/format checks and relevant token/CSS bundle and startup
  guards. Review the diff to confirm no Inspector sidebar redesign entered.

## UX rationale and references

Consistent row controls and task-specific modal entry follow NN/g's consistency,
visibility, user-control and recognition heuristics. Lazy detail provides
progressive disclosure without making routine summaries require expansion.
Representative icons still have text explanations; recognizability is not
assumed merely because a glyph looks meaningful to the implementer.

- [NN/g usability heuristics](https://www.nngroup.com/articles/ten-usability-heuristics/)
- [NN/g progressive disclosure](https://www.nngroup.com/articles/progressive-disclosure/)
- [NN/g icon usability](https://www.nngroup.com/articles/icon-usability/)
- [Design language](../../../backlog/docs/design-language.md)
- [ADR-171](../../../backlog/decisions/171-console-conversation-review-and-attention.md)

## Spec review

Reviewed for scope, target identity, unread/receipt separation, current vs
historical content, icon precedence, narrow-terminal behavior, and privacy.
The Inspector sidebar is excluded. Glyph appearance and layout remain subjects
of implementation verification; this document does not claim live UX evidence.
