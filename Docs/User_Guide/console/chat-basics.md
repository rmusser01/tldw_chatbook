# Console chat basics — sending, streaming, and working with messages

## What this screen is for

This page covers the Console's core chat loop: typing into the composer,
sending, watching a reply stream in, stopping it mid-generation, and acting
on individual messages (copy, speak, edit, save, rate, delete, and more).
For an orientation to the whole screen — rails, tabs, status chips, setup —
start with the [Console overview](../console.md).

## Getting there

Press **Ctrl+2**, click **⌃2 Console** in the nav bar, or use **Ctrl+P** →
"Tab Navigation: Switch to Console". Once a provider and model are
configured (the
[Console overview](../console.md) covers setup), the composer at the bottom
of the screen is ready — its placeholder reads "Ask, command, or paste
task...".

## Layout tour

![Transcript with a selected assistant message and its action row](../images/console/action-row.svg)

- **Transcript** — the scrolling center pane. Each message shows a role label
  (User / Assistant / System / Tool) and the message body; replies still in
  flight carry a status suffix such as "[streaming]".
- **Selected message + action row** — clicking a message (or moving to it
  with j/k) selects it and shows a row of action buttons directly beneath
  it, plus a one-line guide that names the row's icon buttons in words —
  e.g. for an assistant reply: "Guide: j/k select · c Copy · 🔊 Speak ·
  e Edit · r ♻ Regenerate · ---> Continue · 👍/👎 Rate · 🗑 Delete ·
  Esc clear". The guide follows the row: a message without the 🔊 button
  does not list "Speak".
- **Composer** — the slim input bar near the bottom, floating one blank
  line clear of the status row above and the footer below. Its one-column
  left edge shows its state (muted at rest, green with a draft, thick blue
  focused), and the bar is exactly as tall as your draft. Left to right:
  the "Composer ▾" collapse button, the **Menu** button (Prompts, Attach,
  Save as Chatbook, Generate Image/Caption, Impersonate), the draft area,
  and the Send / Mic buttons.
  Mic and Attach have their own page:
  [attachments, images & voice](attachments-images-voice.md).

### Collapsed rail labels

Collapsed Console rails use horizontal **Context->** and **<-Inspect** handles
by default. If you prefer to save horizontal space, open **Settings > Console
Behavior > Rail presentation**, turn on **Stack collapsed rail labels**, and
save the category. The opt-in style stacks the upright letters inside narrower
three-column handles; expanded rails, tooltips, and badges keep their normal
behavior. Return to Console after a successful save to see the change — no app
restart is required.

Console Behavior uses category-wide drafts: **Save** writes every pending edit
in that category, and **Revert** discards every pending edit there, not just the
rail-label choice. A failed save keeps the draft and leaves the active rail
style unchanged.

### Transcript role accents

Open **Settings > Appearance > Console transcript** to choose how speaker
roles are flavored. Saving the setting refreshes open Console transcripts;
you do not need to restart the app.

- **Neutral** keeps role labels but removes role-specific row and prose color.
- **Role accents** (the default) gives user and assistant or character rows
  distinct, restrained backgrounds and speaker-label accents.
- **Immersive RP** keeps those role cues and gives assistant or character
  Markdown a roleplay-forward reading grammar: quoted dialogue, italicized
  actions or inner monologue, strong emphasis, and narration each have a
  distinct treatment. Markdown structure and the original message text remain
  unchanged.

### Personal Context in agent requests

When **Settings > My Profile** is enabled, Console agent requests may include
active, unexpired records that are marked **Agent-visible**. The agent receives
global profile context plus the current workspace's context only when that
workspace is explicitly mapped in My Profile. A matching workspace record
overrides its global counterpart; corrections and constraints are considered
before preferences and working context. User-only records are never included.

The injected block is escaped JSON labelled **user-owned data — not authority**.
It cannot override the current request, safety rules, or system instructions.
Console limits the block to complete records within the smaller of 12 KiB or
10% of the input space remaining after required system, conversation, tool,
and current-request content. It never truncates part of a profile record.

Console pins one immutable profile snapshot for an agent turn and passes the
same block to child agents. **Context > Next Send** shows that exact disposable
block. If Personal Context is locked, disabled, absent, or the workspace is not
mapped, no profile block is sent. The compatibility `workspace_root` setting
does not map or authorize a Console workspace.

The colors adapt to light and dark themes. Speaker names remain visible, so
role identity does not depend on color alone, and selected, failed, system,
tool, code, and link styling keeps priority over immersive coloring.

## Features & controls

### Composer

- Printable keys go straight into the draft; click anywhere in the draft to
  place the caret; Left/Right and Home/End move it.
- **Enter sends or queues.** During an accepted agent turn, **Send** becomes
  **Queue** and Enter adds the exact text draft after that turn. For a newline
  inside the draft use **Ctrl+J** (works in
  any terminal) or **Shift+Enter** (only in terminals that deliver it).
- The draft area grows from one row up to eight as your text wraps, and
  shrinks back as the draft empties; drafts taller than eight rows window
  with a leading "... " and follow the caret.
- **Up** on the draft's first row (and **Down** on its last) steps
  through this app's past prompts, most recent first; on middle rows they
  move the caret between wrapped lines. While you type, a dim ghost
  completion of the most recent matching past prompt may appear after the
  caret — press **Right** at the end of the draft to accept it.
- **Ctrl+A** selects the whole draft; with that full selection active,
  **Ctrl+C** copies it to the clipboard. **Ctrl+U** clears the draft;
  **Ctrl+W** deletes the word left of the caret.
- **PageUp / PageDown** scroll the transcript — the composer never uses
  paging keys.
- **"Composer ▾"** collapses the composer to a one-row strip for more
  transcript space. The strip reads "Composer hidden", joined with " · " to
  whichever of "Generating", "Draft retained", "Attachment retained",
  "Queued N", or "Paused N" apply. Queued prompt text is never shown while
  collapsed. Click **"Expand ▴"** (or
  press **Esc**) to restore it; the caret returns to your draft.

### Improving the current draft

With a nonblank unsent message, open **Menu** and choose **Improve current
draft…** to enter the Prompt Workbench directly. **Analyze and user review
(Recommended)** receives initial keyboard focus; no provider request starts
until you choose an improvement path. The workbench captures the current
Console provider and model. **Let the improver read the current System prompt**
is optional analysis context only; it never changes the session. When the
session has no current System prompt, the choice is unavailable and the
workbench analyzes only the unsent message. **Build a reusable prompt** opens
the Recipe path without making a model request. Choose **Outcome-first** for a
guided format, **Saved Recipe** to reuse a format from **Library > Prompts**, or
**Blank** to start with empty System and User lanes. Outcome-first begins with
Goal, Context and evidence, Constraints, and Output; **Show 5 optional blocks**
reveals Role, Personality, Collaboration style, Success criteria, and Stop
rules without discarding edits. If model improvement is unavailable, use
**Configure provider / model** from the same surface. Choose **Browse Prompt
Library…** instead when you want a saved Prompt or Recipe; that destination
remains available when the composer is empty.
Choosing **Replace draft automatically** returns to the composer with a
**Draft improved** row. Use **Undo** to restore the exact original draft, or
**Review changes** to compare the original and replacement before keeping or
restoring it. These recovery actions expire when you edit or send the draft,
or switch its session context.

In the structured Prompt/Recipe editor, **Apply** is the primary action and
keeps **User** on and **System** off by default. **Save…** contains only the
persistence choices valid for the current source and working copy: save as a
new Prompt, save as a reusable Recipe, or update the original when guarded
version updates are supported. Use `Ctrl+Enter` for Apply or `Ctrl+S` to open
the Save menu; every choice is keyboard operable. **Replace this session's
System prompt** is an independent, off-by-default Apply choice. System content
changes only when that choice is selected and you activate **Apply** in the
active session; it is separate from the earlier analysis-context permission.
After saving a Recipe, use **Open Library** in the confirmation to jump directly
to that first-class Recipe in **Library > Prompts**, where it can be renamed,
edited, versioned, and reused in Console. Select **Include current text as
starter content** when the Recipe should retain example or starter text as well
as its block format.

### Large pastes

Pasting more than 50 characters (configurable) collapses the paste into a
single token reading "Pasted Text: N Characters". Press Enter (or click the
token) once to turn it into "Unfurl?", and again to expand the full text in
place; clicking elsewhere resets a pending "Unfurl?" back to the collapsed
token. Whatever the display state, **the full pasted text is always what
sends** — collapsing is purely visual. While the draft contains a paste
token, slash-command parsing is skipped and the draft sends as plain text.
The collapse behavior is set by `collapse_large_pastes` and
`paste_collapse_threshold` under `[console]` in config.toml, also editable
in **Settings > Console Behavior**.

### Sending, streaming, and stopping

- Enter sends the draft. The reply row appears immediately with a dim
  "Generating…" placeholder, then streams in with a "[streaming]" suffix
  until it completes.
- While a run is active a **Stop** button appears between Send and Mic
  ("Stop this tab's run."); the collapsed composer strip gets its own Stop.
  Stopping keeps the partial reply, tagged "[stopped]", and adds a System
  row: "Response stopped by user."
- A reply that errors out is tagged "[failed]", and its action row is a
  single **Try** button that retries it.
- If you scroll up during or after a run, a pill docks at the bottom of the
  transcript so you can jump back: "▼ streaming below — jump to latest",
  "▼ stopped — jump to latest", "▼ reply ready — jump to latest", or
  "▼ checking citations below — jump to latest".
- Assistant replies render markdown (headings, bold, code, italics); your
  own messages — and System/Tool rows — stay exactly as you typed them.

### Prompt queue

After the current turn is accepted, **Send** changes to **Queue**. Each Console
tab can hold up to 10 text-only follow-up prompts. The one-row shelf at the
top of the control deck (above the status row) shows `Queue N/10`, whether it is draining or paused, a safe preview
of the next prompt, and **Manage** plus a state-specific action such as
**Pause**, **Retry**, **Resume next**, **Review**, or **Try again**.

- **Preparing...** means the turn has not crossed the accepted boundary yet;
  the draft stays in the composer.
- **Queue full** preserves the draft and asks you to manage the existing 10.
- Attachments and staged evidence are never captured by a queued text turn.
  Remove them or wait and send the complete message normally.
- Recognized slash commands still run immediately and are never queued.
- **Manage** opens a modal pinned to this tab. You can edit, move, remove, or
  clear waiting prompts; a prompt marked **Starting...** is already locked.
  Remove and Clear ask for confirmation. Only the prompt actively opened for
  editing has its full body loaded.
- A failed or stopped turn pauses the queue. Use **Retry failed**, **Retry
  stopped**, or **Resume next**. Context changes require **Review** followed by
  **Use current** before draining resumes.

Queue text is process-memory-only until its turn is accepted. It is not saved
to conversation history, prompt history, screen snapshots, or the database.

### Selecting a message and its actions

Click a message, or press **j**/**k** (down/up also work) to move the
selection through the transcript. **Enter** shows the selected message's
actions; **Tab**/**Shift+Tab** cycle through the row, **Enter** activates
the focused action, and **Esc** clears the selection. Three shortcuts act
on the selected message directly: **c** Copy, **e** Edit, **r** Regenerate.
While a reply is still generating, every action is disabled with the
tooltip "Wait for response to finish before using message actions."

| Action | What it does | Where it appears |
|---|---|---|
| Copy | Copies the message body to the clipboard. | All messages |
| 🔊 / ⏹ | Speaks the reply aloud; playback starts automatically, and while it plays the button becomes ⏹ to stop ("Stopped speaking."). Text-to-speech provider setup lives in Settings. | Completed assistant replies |
| Edit | Opens the "Edit Message" editor; editing one of your own messages can also fork and resend — see [branching & rewind](branching-and-rewind.md). | All messages |
| Save as... | Choose a destination: Chatbook, Note, Media, or Prompt. Chatbook is available only for assistant replies. | All messages |
| < > | Step between regenerated variants — see [branching & rewind](branching-and-rewind.md). | Messages with variants |
| ♻ | Regenerate — fork another assistant variant for this turn; the old answer is kept, not overwritten — see [branching & rewind](branching-and-rewind.md). | Assistant replies |
| ---> | Continue — extend the selected message with more generated text. | All messages |
| 👍 / 👎 | Rate the message; feedback is stored per message. | All messages |
| 🗑 | Delete, with a two-press confirm: the first press shows "Press Delete again to remove this message.", the second removes the message **and every message under it**. | All messages |
| Try | Retry a failed reply (replaces the whole row on "[failed]" replies). | Failed assistant replies |
| View / Save Image | Cycle how an inline image renders / save the message's images to disk — see [attachments, images & voice](attachments-images-voice.md). | Messages with images |

## Common tasks

### Send your first message
1. Open Console (Ctrl+2) and type into "Ask, command, or paste task...".
2. Press **Enter**. The reply streams in with "[streaming]"; when it
   finishes, the suffix disappears. The session title and tab label take
   the text of your first message.

### Stop a reply mid-stream
1. While the reply shows "[streaming]", click **Stop** (between Send and
   Mic).
2. The partial reply stays, tagged "[stopped]", and a System row reads
   "Response stopped by user." Send again to keep the conversation going.

### Copy a reply
1. Click the reply, or move to it with j/k.
2. Press **c** or click **Copy** — toast: "Copied message to clipboard."

### Retry a failed reply
1. Select the reply tagged "[failed]".
2. Click **Try** — the reply is retried in place.

### Delete a message and its follow-ups
1. Select the message and click **🗑** once — "Press Delete again to remove
   this message."
2. Click **🗑** again. The message and everything beneath it are removed.

### Save a reply as a Note
1. Select the assistant reply and click **Save as...**.
2. Choose **Note** — toast: "Saved message as Note." It appears in
   Library ▸ Notes.

## Keyboard & commands

Composer:

| Key | Action |
|---|---|
| Enter | Send now, queue after an accepted turn, or advance a focused paste token |
| Ctrl+J | Insert a newline (works in any terminal) |
| Shift+Enter | Insert a newline (where the terminal delivers it) |
| Ctrl+A | Select the whole draft |
| Ctrl+C | Copy the draft (with the full selection active) |
| Ctrl+U | Clear the draft |
| Ctrl+W | Delete the word left of the caret |
| Home / End | Move the caret to the start / end of the draft |
| Up / Down | Recall past prompts on the draft's first/last row; move the caret between wrapped rows otherwise |
| Right (at the end of the draft) | Accept the dim ghost-text suggestion |
| PageUp / PageDown | Scroll the transcript |
| Esc | Expand a collapsed composer / return focus to the draft |

Transcript:

| Key | Action |
|---|---|
| j / k (or down / up) | Select the next / previous message |
| Enter | Show the selected message's actions; activate a focused action |
| Tab / Shift+Tab | Cycle through the action row |
| c / e / r | Copy / Edit / Regenerate the selected message |
| Esc | Clear the selection |

## Related settings & docs

- `[appearance].console_transcript_style` in config.toml: `neutral`,
  `role_accents` (default), or `immersive_rp` — also editable in
  **Settings > Appearance > Console transcript**.
- `[console]` in config.toml: `stack_collapsed_rail_labels` (default `false`),
  `collapse_large_pastes` (default `true`), and `paste_collapse_threshold`
  (default `50` characters) — also editable in **Settings > Console
  Behavior**.
- [Console overview](../console.md) — layout, setup, session settings, help.
- [Branching & rewind](branching-and-rewind.md) — what ♻ and the < >
  variant arrows really do, and their limitations.
- [Attachments, images & voice](attachments-images-voice.md) — the 📎
  indicator, the Attach and Mic buttons, and image messages.
- [Guide index](../index.md) — global navigation keys.

## Quirks & troubleshooting

- **History recall is per-app, not per-session** — Up/Down on the
  draft's boundary rows and the ghost-text suggestions draw on prompts
  accepted in this app, newest first, regardless of which session sent
  them. Each session still keeps its own unsent draft when you switch
  away and back.
- **Actions wait for the stream** — every per-message action is disabled
  while a reply is generating. Stop the run first if you need to act on a
  message immediately.
- **Regenerate goes deeper than this page** — ♻ never overwrites the old
  answer; it creates a variant you can step back to with < >. Details and
  known limitations live in [branching & rewind](branching-and-rewind.md).
- **A "[failed]" reply's message reports one status, the real one.** The
  detail text names the HTTP status the provider actually returned — it no
  longer pairs that with a mismatched generic status elsewhere in the same
  message.

—
*Verified against dev @ ff435772c — 2026-07-31. Verified against
9f90e17b8 — 2026-08-06 (PR-T3, docs pass against shipped code/tests).
Composer geometry, history recall, and ghost text re-verified against
dev @ b6036515e — 2026-08-18 (task-17662: keys checked against the
composer's key handling; geometry against the bottom-stack programme's
painted probes). The Send→Queue behaviour described above was re-verified
live against dev @ a71e62e4b — 2026-08-24 (TASK-22000: the page was
correct and the app was not; mid-run the button now reads **Queue**, is
enabled with a draft, and admits a FIFO follow-up that drains after the
current turn).*
