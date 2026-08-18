# Text selection & feedback — quote, ask, and review agent output

## What this is for

Pick out a phrase, a stack trace, or a chunk of a diff in the transcript and
do something with *just that part*. Drag-select any text and a small menu
appears with two families of actions: **bring it into the conversation**
(quote it, or ask about it in a throwaway side chat that never touches your
history), and — when the text came from the agent — **review it** the way
you'd review a pull request, with Request changes / LGTM / Comment.

Reach for it when a reply is mostly right but one paragraph isn't, when you
want a term explained without derailing the main thread, or when you're
supervising an agent run and want your verdict on a specific step recorded.

## Getting there

Everything here starts from a **mouse drag across transcript text** — there
is no keyboard entry point yet. Press and hold on the first character, drag
to the last, and release; the selection highlights and the menu opens just
below where you released.

Two settings feed the side chat (**Settings ▸ Console**, "Side chat model"
and "Side chat prompt template") — see [Console](../console.md) for the
screen itself.

## Layout tour

The menu is a small vertical stack of buttons anchored at your release
point. Which buttons appear depends on what you selected:

| Button | When it appears |
|---|---|
| **Add to chat** | When the selection can be quoted into the composer |
| **More Details** | Always |
| **Ask in Side Chat** | Always |
| **Create note** | Always |
| **Request changes** | Only for agent output (assistant replies, tool markers, diff rows) |
| **LGTM** | Same as above |
| **Comment** | Same as above |

If the menu would run off the bottom of the screen it flips to sit entirely
*above* the selected row, so your highlight stays visible.

The first button takes focus on open: **↑/↓** cycle, **Enter** activates,
**Esc** closes and returns focus wherever it was before.

## Features & controls

### Add to chat

Inserts the selection into the composer as a `>`-quoted block, leaving your
cursor after it so you can type your question. Your existing draft is never
replaced.

### More Details

Opens the side chat and **immediately sends** a prompt about the selection —
no typing needed. The prompt comes from the "Side chat prompt template"
setting, where `{selection}` marks where the selected text goes (if you omit
the placeholder, the selection is appended to the end).

### Ask in Side Chat

Opens the same modal but **waits for you**: the quote is shown read-only and
you type whatever you want to ask about it.

Side chats are **ephemeral by design**. The reply lives in the modal and
nowhere else — it is never written to your conversation, never persisted, and
closing the modal discards it. **Stop** cancels a stream in flight, **Retry**
re-sends, and Escape / clicking the backdrop / **Close** all cancel and
dismiss. By default the side chat uses your current session's model; set
"Side chat model" to pin a different (usually cheaper) one.

### Create note

Saves the selection straight into your notes. The note's title is the
selection's first line (capped at 48 characters); its body is the full
selection plus a provenance line naming the conversation and date. A toast
confirms with the title. Available for any selection — your own messages
included — and works with or without a run.

### Review feedback: Request changes, LGTM, Comment

These only appear when the selected text is **agent output** — an assistant
reply, a tool marker or its diagnostics, or a row inside an expanded
file-write diff. Your own messages are never reviewable.

All three compose a structured message — an action header, your selection as
a `>`-quoted block, and your comment if you left one — and send it as your
next turn. If a run is in progress it queues behind it. Your composer draft
is left completely alone.

**Comment** always opens a box for your note. **Request changes** and **LGTM**
also let you add one, and cancelling that box (Escape, or **Cancel**)
abandons the whole thing — nothing is sent.

**Request changes** and **LGTM** need an active run. Without one they render
disabled with the hint *"No active run — start a run to send review
feedback"*. **Comment** stays available either way.

### What gets recorded

Feedback isn't only sent — it's kept, in two places serving two needs.

**Every action** (Request changes, LGTM, Comment) is written to the session's
trajectory ledger as a **user_feedback** event anchored to the exact message
you were reviewing, carrying the action, the quote, and your comment. To read
it back, open the **trajectory view**: feedback appears nested under the
message it was about, in chronological order, shown as e.g. *"Request
changes: tighten error paths"*. Selecting the row shows the full quote and
comment untruncated. Reviewing the same message several times gives you
several records, not one overwriting the last.

**A Comment with a note** additionally persists as a **transcript
annotation**: the annotated message gains an inline *"Review note"* marker
right in the transcript, listing your note(s), so review notes are visible
while reading — no need to open the trajectory view. Markers come back when
you reopen the conversation.

Both records survive restarting the app, and both are deliberately
best-effort: if a write fails, your feedback is still sent. Losing an audit
line or a marker should never cost you the actual message.

## Common tasks

1. **Quote one paragraph back at the model.** Drag across it, click **Add to
   chat**, type your objection after the quote, and send.
2. **Ask what a term means without polluting the thread.** Drag across it,
   click **More Details**, read the answer, close the modal. Your
   conversation is untouched.
3. **Reject one step of a live agent run.** Drag across the offending tool
   output, click **Request changes**, type what should change, and submit —
   it queues as your next turn and lands in the trajectory ledger.
4. **Approve a step for the record.** Same drag, click **LGTM**, submit with
   or without a note.
5. **Pin a note to a reply for later.** Drag across it, click **Comment**,
   write your note — the message keeps an inline *"Review note"* marker you
   (and the trajectory ledger) can come back to.
6. **Leave a note when nothing is running.** Drag across the output and use
   **Comment** — it isn't run-gated.

## Keyboard & commands

Text selection is fully keyboard-reachable: select a message with `j`/`k`,
press **`s`**, and a vim-style selection mode starts on that message with
its first character selected and a hint line showing the active keys.

| Key (in selection mode) | Action |
|---|---|
| h / l | Move the selection end by one character |
| w / b | Jump by word |
| 0 / $ | Start / end of the current line |
| j / k | Grow / shrink by one line |
| o | Swap which end of the selection you're moving |
| Enter | Open the action menu on the selection |
| Esc | Leave selection mode (your message selection survives) |

Diff rows are **not** keyboard-selectable — drag with the mouse for
hunk-level selection. A second **Esc** clears the message selection as
before.

Inside the menu itself:

| Key | Action |
|---|---|
| ↑ / ↓ | Move between menu buttons |
| Enter | Activate the focused button |
| Esc | Close the menu (and restore previous focus) |

## Related settings & docs

- [Console](../console.md) — the screen, its rails, and message selection
- [Agent runs & tools](agent-runs-and-tools.md) — the runs this feedback reviews
- [Chat basics](chat-basics.md) — selecting whole *messages* (a different thing
  from selecting *text*)
- **Settings ▸ Console** — "Side chat model", "Side chat prompt template"

## Quirks & troubleshooting

- **Selections may go stale.** A selection is allowed to survive a re-render;
  if streaming replaces the row you selected, the quote clamps to the last
  stable text rather than following the new content.
- **Quotes are capped at 4000 characters.** Longer selections are truncated
  with a marker before they leave the transcript.
- **Request changes / LGTM look broken with no run.** They're disabled on
  purpose — hover for the hint, or use **Comment**.
- **Keyboard selection always starts at the text's beginning.** Use `o` to
  swap ends when you want a span that starts mid-text (note: after `o`,
  forward motions stop one unit short of the other end rather than
  crossing it, unlike vim).
- **Diff rows are mouse-only for text selection.** `s` on a tool message
  selects its marker text, not the diff hunks.
- **Side-chat replies can't be recovered** after the modal closes. Copy
  anything you want to keep before dismissing it.

—
*Verified against feat/console-keyboard-selection @ a4286a199 — 2026-08-18
(shipped tests + live tmux verification: s→motions→Enter→menu→Create note→real DB row; two-stage Esc)*
