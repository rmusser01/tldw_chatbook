# Branching & rewind — variants, edited resends, and /rewind

## What this is for

Nothing is ever deleted when you regenerate a reply or edit a prompt. The old
answer stays reachable as a *variant*, and your visible conversation shows one
path through your alternatives at a time. You can also copy that path through
one selected message into a completely independent chat. This page covers four
surfaces: **Fork chat**, the ♻ Regenerate action, the "Edit & resend" response
branch from the Edit Message modal, and the `/rewind` menu for rolling a
session back or compacting it. Reach for them when a reply isn't what you
wanted, when an old prompt needs a do-over, or when a long session is getting
heavy.

## Getting there

These controls live inside the Console transcript and composer — see
[Console](../console.md) for the screen itself and
[chat basics](chat-basics.md) for selecting messages:

- **Fork, ♻, <, >, Edit** appear in the action row under a *selected* message
  (click a row, or move the selection with `j`/`k`).
- **/rewind** is typed into the composer like any other slash command.

## Layout tour

![The Rewind menu with a prompt selected](../images/console/rewind-modal.svg)

Branching lives in two places on the Console screen:

- **The action row** under a selected message — its stable direct order is
  Copy, Speak/Stop when available, Edit, text-response **< / >** when present,
  **Fork**, ♻ Regenerate/Retry when present, Continue, and **More…**. When
  variants exist, the message's role label above it carries the
  "(2/2)"-style counter.
- **The "Rewind" menu** (captured above) — opened by typing `/rewind`. Your
  earlier prompts are listed newest-first as "#1 …" rows; selecting one
  reveals the "Restore to here" / "Summarize up to here" / "Never mind"
  buttons below a "Selected #N: …" line.

## Features & controls

### Fork chat from here

Select a stable User or Assistant message and press `f` (or click **Fork**).
The **Fork chat** dialog identifies the boundary and how many messages will be
copied. Its proposed name, `Forked from <source title>`, is already selected:
type to replace it, or press **Enter** immediately to accept it. Escape cancels
without changing either chat.

The boundary is inclusive. The new chat receives exactly the source's active
lineage from its first message through the selected message. Later turns,
off-path sibling branches, display-only tool/activity rows, and unselected
variants are absent. A User boundary does not generate a reply automatically.
After confirmation the fork opens as a separate Console tab and can diverge;
the source tab stays open with the same title, selected variants, active leaf,
history, and live work it had before.

- A saved source creates a saved fork in the same Chats or named Workspace
  section, with durable ancestry back to the source and boundary.
- An explicitly temporary source creates another temporary chat and writes no
  durable conversation rows. Saving that fork later makes it an independent
  durable root; it does not save or modify the temporary source.
- Sent attachments, the selected generated-image choice, declarative model,
  Workspace, Library, RAG, and project-instruction selections are copied under
  fresh ownership. A durable fork can retain a currently valid governed
  citation owner link; a temporary fork keeps citation markers such as `[S1]`
  only as message text and has no copied source-inspector provenance.
- Runs, queues, tool history, drafts, staged files/evidence, scratch files and
  leases, approvals, permissions/tool grants, provider continuation, recovery,
  derived context, usage/cost accounting, and resolved project-instruction
  bodies are not copied. File, tool, and project authority is checked afresh.
- Generated-video bytes are ephemeral and never copied or shared. The fork
  shows an unavailable-video tombstone with regeneration details even if the
  source still plays. Use the source video's **Save copy** action before
  forking if you need the file itself.

Fork chat is different from ♻ and Edit & resend: those create alternative
responses inside the same chat, while Fork creates a new independently owned
chat and leaves the source untouched.

### Response variants (♻, <, >)

Select an assistant reply and press `r` (or click **♻**) to ask for another
answer to the same prompt:

- A new reply streams in as a fresh variant. The previous answer is not
  deleted — it just steps aside.
- The role label gains a counter when variants exist: **Assistant (2/2)**
  means you're looking at the second of two.
- **<** and **>** buttons appear in the action row. They swipe between
  variants; at either end the button refuses with
  "No response variant in that direction."
- Swiping shows **that branch's continuation**, not just a different
  paragraph: if you kept chatting after variant 1, swiping back to it brings
  its follow-up turns with it, and you land at that branch's latest turn —
  not at the fork.
- After a swipe the same message stays targeted, so repeated presses of
  **<** / **>** keep working without re-selecting the row.
- ♻ only works on assistant replies ("Only assistant messages can be
  regenerated."), and all actions wait for streaming to finish
  ("Wait for response to finish before using message actions.").

### Edit & resend

Select one of **your** messages and press `e` (or click **Edit**) to open the
**Edit Message** modal. Its explanation, verbatim:

> Editing existing transcript message. Save keeps the edit in place; Edit &
> resend creates a new response branch in this chat and gets a fresh reply.

- **Save** — fixes the text in place. No new reply is generated.
- **Edit & resend** — creates a new branch: your edited prompt is sent and a
  fresh reply streams in. The original prompt and everything that followed it
  stay reachable — your message gains its own "(2/2)"-style counter, and
  **<** / **>** flip between the original and the edited version, each with
  its own replies.
- **Edit & resend** appears only on your own messages; assistant and system
  rows offer plain Save only ("Only your messages can be edited and
  re-sent.").
- Attachments on the original message are carried into the resend.
- Blank content is refused: "Message content cannot be blank."

### The /rewind menu

Type `/rewind` in the composer and press Enter. The **Rewind** menu (see
the capture in the Layout tour) lists your earlier prompts newest-first as
"#1 …" rows; pick one to reveal three buttons:

- **Restore to here** — your visible conversation returns to just before
  that prompt, and the prompt's full text is placed back in the composer so
  you can re-send it as-is or edit it first. Nothing is deleted: the rewound
  turns remain reachable as a branch.
- **Summarize up to here** — compresses the turns *before* that prompt into
  a summary for the model. Before anything runs, a preview dialog shows what
  will happen — how many turns get summarized, how many are kept as-is, and
  the estimated context change (`~before → ~after tokens`) — and no model
  call is made until you confirm; Cancel discards the preview with nothing
  recorded. Your visible transcript is untouched; the banner
  "⤵ Earlier turns summarized for context — full history above" marks the
  boundary in the transcript. If the summarizer hangs, the call is cut off
  after a bound (default 120 s, `[console] compaction_auxiliary_timeout_seconds`)
  and no memory is saved.
- **Never mind** — closes the menu (Esc works too).

With no earlier prompts to go back to, `/rewind` just says
"Nothing to rewind." Restore and Summarize also wait their turn — while a
reply is still streaming they refuse the same way sending does.

## Common tasks

1. **Get a second opinion on a reply.** Select the reply (`j`/`k` or click),
   press `r`. When the new variant finishes, press **<** to compare it with
   the original.
2. **Compare two variants.** With the reply selected, click **<** and **>**
   to flip between them — the label ("Assistant (1/2)", "(2/2)") tells you
   which one you're on, and any follow-up turns swap along with it.
3. **Start a separate chat at an earlier point.** Select the boundary, press
   `f`, rename it or press Enter immediately, then continue in the newly opened
   tab. Switch back to the source at any time; it is unchanged.
4. **Fix a typo in an old prompt and branch from it.** Select your message,
   press `e`, correct the text, click **Edit & resend**. A fresh reply
   arrives on a new branch; the original stays one **<** away.
5. **Roll the conversation back and re-ask differently.** Type `/rewind`,
   pick the prompt where things went wrong, click **Restore to here**. Edit
   the restored text in the composer and press Enter.
6. **Compact a long session.** Type `/rewind`, pick a prompt that marks
   "everything before this can be squashed", click **Summarize up to here**.
   Keep chatting — the model now sees a summary of the early turns, while
   you still see everything.

## Keyboard & commands

Screen-specific to these features; message selection and the rest of the
action row are covered in [chat basics](chat-basics.md), globals in the
[guide index](../index.md).

| Key / command | Action |
|---|---|
| `f` | Fork a new chat through the selected stable User or Assistant message |
| `r` | Regenerate the selected assistant reply (new variant) |
| `e` | Edit the selected message (**Edit & resend** on your own messages) |
| **<** / **>** | Show the previous / next variant — action-row buttons, click to use |
| `/rewind` | Open the Rewind menu |

## Related settings & docs

- [Console](../console.md) — the screen this all lives on.
- [Chat basics](chat-basics.md) — selecting messages, the full action row,
  composer keys.
- [Context & RAG](context-and-rag.md) — what the model actually sees: how
  the active branch and the summarize boundary shape the next send, and how
  to inspect it.

## Quirks & troubleshooting

- During agent runs, the small inline tool markers under a reply (the
  "⚙ …" / "⤷ …" rows) disappear after your next action — the next send,
  swipe, or delete. The run log in the rail keeps the full record.
  (task-570)
- If a regenerate **fails** or returns no content, Console returns to the
  assistant answer you regenerated. That answer remains in the context sent
  with your next message. The failed attempt stays available as a sibling for
  navigation, inspection, or retry. If you regenerated an older answer, that
  answer becomes the active branch endpoint; its later turns remain stored on
  their existing off-path branch rather than returning automatically. An
  intentionally stopped partial regenerate stays on the active branch.
  (task-571)
- **Restore to here** on your very *first* prompt doesn't survive an app
  restart: the conversation comes back at its latest turn. Within the
  running session it works as expected. (task-574)
- There is no "summarize *from* here" yet — you can compress the oldest
  turns and keep the recent ones verbatim, but not the reverse. (task-575)
- `/rewind` is only reachable by typing the command; it isn't in the command
  palette or the message actions yet. (task-576)
—
*Variants, Edit & resend, and rewind verified against dev @ ff435772c —
2026-07-31. Fork chat was verified against TASK-23088's production-shaped
provider-free journey at 120×35 and 80×24 on 2026-08-27.*
