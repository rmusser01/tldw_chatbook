# Console: Context & RAG — controlling and inspecting what the model sees

## What this page is for

Every Console send is assembled from parts you control: system prompt,
history, an optional response prefill, tools, staged sources, and — when
retrieval is on — whatever RAG pulls in. This page covers inspecting that
payload before it goes out, shaping it (`/system`, `/prompt`,
`/prefill`), narrowing retrieval (RAG scope), and tracing a grounded
reply back to its citations.

## Getting there

Open Console with **Ctrl+2**. This page's surfaces: the **Conversation
Inspector** (the token/cost chip, **Ctrl+Shift+P**, or the command
palette's **Console: View chat context** entry), the **Inspector** rail on
the right (click its handle to expand — a separate surface from the
Conversation Inspector modal, despite the similar name), the composer for
`/prompt`, `/system`, and `/prefill`, and the status chips above the
composer.

## Layout tour

![The Conversation Inspector's Next Send tab](../images/console/context-modal.svg)

Where this page's controls live:

- **The Conversation Inspector** — a window over the screen with three
  tabs (**Costs**, **Exchange**, **Next Send**), opened from the token/cost
  chip, **Ctrl+Shift+P**, or the command palette. The screenshot above
  shows its **Next Send** tab, carried over from the former standalone
  "Chat Context" viewer this modal replaced.
- **The Inspector rail** (right edge) — the "Sources" tray at the top, the
  retrieval-scope row beneath it, the "Prefill" rows when one is armed, the
  "Live work sources" card, and the "Chat Dictionaries" / "World Books"
  blocks at the bottom.
- **The status chips** above the composer — "RAG: on/off", "Sources: N
  staged", and the "Scope: N" chip once retrieval is narrowed.
- **The staged-evidence strip** — at the top of the control deck, above
  the status chips; shown only while something is staged (or right after
  a send that used it).
- **The composer** — where `/prompt`, `/system`, and `/prefill` are typed;
  the left rail's Model section carries the clickable `System:` line.

## Features & controls

### Reading long Context and Inspector sections

Each open, named section body in the Context and Inspector rails shows up to
20 rendered content lines, then scrolls within that section. The cues tell you
which area to move:

- **▼ more — scroll** means the current section has more content. Scroll over
  that body, or Tab to its scrollable area and use the arrow keys, Page Up/Page
  Down, Home, or End.
- **▼ more sections — scroll** means later rail sections remain below. Scroll
  the whole Context or Inspector rail, not the section you are currently
  reading.

Tab and Shift+Tab continue in normal order through controls, an overflowing
section's scrollable area, and controls within its body; sections that fit do
not become extra Tab stops. In the Inspector, `n` and `p` move to the next or
previous named section without wrapping. They work only while the Inspector
has focus and never take over an editable field. The footer shows **n/p
Sections** while this is active, and **F1** shows the full shortcut list.

On short terminals, Context scrolls as a whole so every header and open body
remains reachable. An open section that cannot currently receive body space is
marked **· no room**; press its **[>]** control to prioritize it temporarily.
This does not alter your saved open/closed choices. Section and rail scroll
positions, and the temporary priority, last only for the current session. The
expanded, browser-like layout remains the primary experience, with these
safeguards available as the terminal narrows or shortens.

The **Project** status row (for Project Instructions) and retrieval **Scope**
remain compact rows; they do not expand into 20-line section bodies.

### The Conversation Inspector

Click the status row's cost chip, press **Ctrl+Shift+P**, or run
**Console: View chat context** from the command palette to open the
**Conversation Inspector** — one modal, three tabs, replacing the older
separate cost breakdown and "Chat Context" viewer. The chip opens on
**Costs**; Ctrl+Shift+P and the palette entry open on **Next Send**; once
open you can switch tabs freely. `Escape` closes it from any tab.

#### Costs

Today's per-message token/dollar rows and totals — unchanged ground
truth. Each row expands into that turn's individually captured provider
calls (one per agent tool-loop iteration, or one for a direct send), each
shown with its status and its own cost. Where a turn's per-call costs and
its message-level total disagree (rounding, or a call with no priced
usage), both figures are shown rather than silently reconciled; a call
with no usable usage reads **unpriced**.

#### Exchange

The per-turn, per-call detail behind those numbers. Expand a turn, then a
call, to reach collapsible sections: **System prompt**, **Messages**,
**Tools**, **Response**, **Tool calls** (only shown when the response made
any), and **Sampling & routing** (temperature, max tokens, seed, and
similar parameters). Each call's title carries a status badge —
`[stopped]`, `[error]`, or the normal `[complete]` — and a call captured
during an abandoned regenerate additionally reads `[abandoned
regeneration]`: regenerating keeps the earlier answer's captures rather
than discarding them (see the response-variants section of
[Branching & rewind](branching-and-rewind.md)). A call whose request had a
value that isn't on the capture allowlist — this is how credentials like
API keys are kept out of storage; see **Kill-switch & privacy** below —
shows a line naming exactly what was dropped: "Omitted by capture policy:
api_key".

Per-piece token counts inside a call (e.g. "System prompt (~412 tokens
est.)") are **estimates**, computed the same way the composer estimates
your draft — they will never disagree with what the composer shows for
the same text, but they are not what you were billed. Sitting alongside
them, a **Reported usage** line (`in:… cache_r:… cache_w:… out:…`) carries
the provider's own usage numbers where the provider sent any, with no `~`
— that line is authoritative. Each call has its own **Copy JSON** and
**Save to File** buttons (a temporary/ephemeral session blocks Save, same
as elsewhere in Console).

A turn with nothing captured — recorded before this feature existed, with
capture disabled, or where capture itself failed — shows a plain row
instead of a blank space: "No capture recorded for this turn (recorded
before capture existed, capture disabled, or capture failed)."

#### Next Send

The former Ctrl+Shift+P "Chat Context" viewer, carried over intact: a
read-only snapshot of what the model has seen and is about to see. Two
sub-tabs:

- **Current** — one collapsed section per transcript message, titled with
  its role and status; expand any to read the exact stored text. Empty
  state: "No conversation context." (The role currently displays in its
  internal form, e.g. "[ConsoleMessageRole.USER] complete" — task-2704.)
- **Next Send** — the payload the next send will carry, as collapsible
  folds: **Model**, **System**, **Messages** (one collapsed `Message N`
  per entry), **Response Prefill** (only while armed, noting "The reply
  will continue from this prefill; the agent loop (tools/MCP) is skipped
  for this send."), **Tools**, and **Staged Sources**.

The header carries an approximate token count (e.g. `(~1234 tokens)`) when
a draft-based estimate exists; mid-stream a warning reads "A response is
in progress; snapshot may change." Footer controls: a **Raw JSON**
checkbox, **Refresh** (also the `r` key), **Copy JSON**, **Save to File**
(writes the payload to disk and shows the path). Payloads over 1 MiB are
not rendered inline — the viewer shows "Context exceeds 1 MiB. Use Save to
File to view the full payload."

### Project instructions

The Inspector places a one-line **Project** status above **Sources**:
**Off**, **Choose folder**, **None**, **N loaded**, or **Warning**. Select the
row to open this viewer's metadata-only **Project Instructions** section. It
shows whether the feature is enabled, the selected binding and locator match,
override/standard precedence, relative source paths, scopes, byte counts,
active or omitted outcomes, and deduplicated warning codes. Removed or
retargeted bindings offer **Choose folder** and **Disable**; **Off** offers
**Enable**. There is no automatic-file editor or second settings surface.

The **Next Send** tab is the only automatic UI surface that may show the exact
instruction body, and only as a disposable preview of the captured session's
next request. Closing it discards the preview. **Copy JSON** and **Save to
File** omit automatic instruction bodies. The transcript, rail, context
metadata, warnings, run logs, and saved conversation state carry only
content-free metadata. If you explicitly ask a file tool to read an
`AGENTS.md`, its result is ordinary user-requested tool output and follows the
normal logging and persistence rules.

On the first send for a selected binding and provider destination, a notice
lists only destination and source metadata and asks **Proceed**, **Cancel**,
or **Disable**. Consent is repeated when the binding locator or provider/custom
endpoint changes, but not for a model-only change at the same destination.
Nested files may load later. Automatic sources are constrained independently
to `[console] project_instructions_startup_max_bytes = 32768` at the binding
root and `project_instructions_nested_max_bytes = 32768` for the run; a whole
file must also fit the exact model request's token headroom. Omitted, stale,
unsafe, unreadable, or outside-scope candidates produce content-free warnings.

This behavior deliberately combines two ecosystems without pretending they
are identical. Codex defines the `AGENTS.override.md` / `AGENTS.md` hierarchy
and broad-to-specific composition. Claude Code inspired lazy path-sensitive
loading, but its native project file is `CLAUDE.md`, not `AGENTS.md`. Chatbook
adds its own selected-binding authority boundary and delivers automatic text
as ephemeral user context rather than privileged policy.

#### Kill-switch & privacy

Every provider call Console makes is captured (request and response) and
stored locally, per turn, so the Exchange tab has something to show —
this is `config.toml`'s `[console] exchange_capture` setting, **on by
default**. Set `exchange_capture = false` to turn capturing off; turns
sent while it's off show the Exchange tab's "No capture recorded" row
instead of call detail. Captures are local-only — compressed at rest, never
synced to another device — and a temporary/ephemeral conversation's
captures live only in memory for that session and are never written to
disk. A stopped send keeps its partial content, marked `[stopped]`; an
abandoned regenerate keeps its captures too, marked `[abandoned
regeneration]`, rather than being dropped.

Two honesty limits worth knowing before you treat a capture as "the exact
bytes that went over the wire":

- **Adapter-boundary capture.** Capture happens where Console hands a
  request to the provider adapter, not at the raw HTTP layer — so
  provider-internal framing and any prompt-caching markers an adapter
  injects on its own are not visible in what you see here. The one
  exception is a **llama.cpp** (local server) send: that branch builds its
  own HTTP payload directly, so for llama.cpp its capture *is* the literal
  wire payload.
- **No fabricated history.** A turn that predates this feature, was sent
  with capturing off, or hit a capture failure never shows fake or
  guessed content — it shows the explicit "No capture recorded" row
  described above.

### System prompt

Type `/system` (or use the palette entry **Console: Edit system prompt**,
or click the `System:` line in the left rail's **Model** section — it
ends in a `▸` and reads `System: none` while unset) to open the **Edit
system prompt** modal: "Applies to this session.", the prompt editor, a
**Name** field with a **Save to Library** button for keeping the prompt
in Library, and the buttons **Clear**, **Cancel**, and **Apply**.

`/system <name>` applies a saved prompt's system part directly (exact
name match, then unique prefix); anything else opens the **Apply system
prompt** picker. A saved prompt without a system part is refused with
`Prompt "<name>" has no system part.` — picker rows show " (no system
part)".

### Saved prompts (/prompt)

`/prompt <name>` **replaces your current draft** with the saved prompt's
user text (exact match, then unique prefix). Bare `/prompt`, or an
ambiguous/unknown name, opens the **Insert prompt** picker ("Search
prompts…"; empty state "No saved prompts yet — create them in Library ▸
Prompts."). A large prompt body arrives in the composer as a collapsed
paste token — click it (or press Enter on it) to unfurl.

Saved Prompt System and User lanes may contain insertion-time variables such
as `{customer}`. Names are case-sensitive and must match
`[A-Za-z_][A-Za-z0-9_]*`: `{customer}` and `{Customer}` are different. The
shared **Prompt variables** dialog lists each name once, in first-occurrence
order across System then User, and reuses the one value everywhere that exact
name appears. Blank values are valid. A value containing braces is inserted as
text and is not scanned again.

Use `{{` for a literal `{` and `}}` for a literal `}`. For example,
`{{customer}}` inserts `{customer}`, while `{{{customer}}}` inserts the value
inside literal braces. Invalid or unmatched forms such as `{first-name}`,
`{ name }`, `{name`, and ordinary JSON object braces stay literal. A variable
name may contain at most 64 characters, and one insertion may use at most 64
unique names. If either limit is exceeded, **Apply** is disabled and the dialog
says either `A Prompt variable name exceeds 64 characters.` or
`This Prompt has more than 64 variables.`; **Use original placeholders**
remains available.

When a Prompt has a System lane, the dialog shows
`Replace the current session System prompt with this System lane`, **Off** by
default. Turning it on may add System-only variables without losing values
already entered for shared or User-only variables. **Apply** inserts the filled
active lanes; **Use original placeholders** inserts the selected lanes
unchanged; **Cancel** changes nothing. A System-only Prompt, including one with
a blank User lane, has no active lane until you turn this option on. If System
is the only selected lane in a `/prompt` replacement, the captured draft is
cleared as part of applying that Prompt. A variable-free User-only Prompt takes
the direct guarded path without showing the dialog.

Exact `/prompt` and picker insertion replace the complete draft captured when
the command was dispatched or the picker opened, including its collapsed paste
and inline-file segments. If that draft, the active session, or the authorized
System prompt changes before application, nothing is applied and a warning asks
you to open the Prompt and retry. Library **Use in Console** uses this same
dialog but appends to the settled active draft instead of replacing it; a stale
Library target session or authorized System prompt is likewise refused. A
confirmed application expires if it has not been consumed at the 120-second
boundary; a transient composer remount may retry only while that window remains
open.

Variable values and pending applications are memory-only: they are not saved as
defaults or retained for the next insertion. Text you intentionally apply then
follows the ordinary draft and session lifecycle. **Menu → Undo Prompt change**
restores only the draft captured before the latest Prompt change; it does not
undo a System change. If the live System change succeeds but its durable save
does not, Console warns: `System prompt applied for this session, but the change
could not be saved -- it may not survive a reload.`

### Response prefill (/prefill)

A prefill is text the assistant's reply must continue from — useful for
forcing a format ("Here is the JSON:") or an opening tone.

- `/prefill <text>` arms one for the next send only. Confirmation:
  "Prefill armed for next send: '<text>'. The reply continues directly
  from the last character; tool calling is skipped on prefilled sends."
- `/prefill pin <text>` persists it — the confirmation adds "Applies to
  every send, retry, and regenerate until /prefill clear."
- `/prefill clear` removes it ("Prefill cleared.").
- Bare `/prefill` reports status: `Prefill (next send only): '…'`,
  `Prefill (pinned): '…'`, or "No prefill armed."

Prefill text is capped at 4,000 characters. While armed it shows in the
Inspector rows **Prefill (next send only)** / **Prefill (pinned)** and in
the Chat Context viewer's **Response Prefill** fold.

### RAG scope

RAG scope limits retrieval to chosen Library items instead of searching
everything. The Inspector's retrieval-scope row has three states:

- **Scope: everything** + a **Narrow…** button — no scope set.
- **Scope: N items** + **Edit** and **Clear** — a scope is active; a
  **Scope: N** chip also appears in the status strip above the composer
  (click it to reopen the picker).
- **Scope: no sources** (alert-styled, + **Narrow…**) — the configured
  scope resolves to nothing (e.g. conversation and workspace scopes don't
  overlap), so retrieval over it would return zero items.

**Narrow…**, **Edit**, and the chip open the **Narrow RAG scope —
<target>** picker: tabs **All** / **Media** / **Notes**, a "Filter by
title…" box, tag chips with "Search tags…", a **Sort:** select
(**Recent** / **Title** / **Type**), an **All** / **Selected** view
toggle, a paged list (**◂ Prev** / **Next ▸**, `▦` media / `✎` note),
bulk **Select all matching** / **Clear shown** (large select-alls ask
**Confirm** / **Cancel**), and a footer with a "N selected of M" count
plus **Save**, **Clear scope**, and **Cancel**.

Scope exists at two levels. The left rail's **RAG Scope** button sets a
**workspace-level** scope ("Narrow RAG retrieval to items in this
workspace"); a conversation's scope then narrows *within* it — items
outside are suffixed "— outside workspace scope", and the effective scope
is the intersection (chip tooltip: "Only searching: conversation scope (2
items) and workspace scope (5 items) — 2 in both.").

### Staged sources & Library search

The Inspector's **Sources** tray lists context staged for the run, one
row per source with a status word (ready / running / blocked / muted);
empty state: "No sources attached. Stage sources from Library." The
control-bar **Attach context** action opens the "Console context" rail;
the staging itself is done from the Library screen.

Media, notes, and conversation handoffs now actually reach the model on
send — they used to display as staged while delivering nothing. Notes
send a real excerpt of the note body; media and conversation handoffs
currently send only a short generic label naming the item (e.g. "Media
staged: \<title\>"), not an excerpt of the content itself — upgrading
that to a real excerpt is still open (task-2376). A few other handoff
kinds (skills, watchlists/collections snapshots, quizzes, personas) can
still show as staged while the model receives nothing at all for them —
that gap is also still open (task-2375).

To gather evidence *before* sending, use the Inspector's **Live work
sources** card: type a question into "Ask Library sources before sending"
and press **Search Library** (also a control-bar action). It searches
your Library and stages what it finds.

Which *kinds* of sources it searches is shown on that card's **Sources:**
line — by default "Sources: Notes, Media, Conversations (Prompts off)" —
and is editable: the **Library search** chip (or **Search Library** with
nothing typed) opens the **Library search** settings modal, which carries
the query box plus a toggle per source kind (**✓ Notes**, **○ Media**,
**✓ Conversations**, **○ Prompts**) and the **Auto-retrieve on send**
switch described below. Running keeps the edited query/source-kind
selection (it also survives leaving and returning to Console); **Cancel**
discards it. Run stays disabled until there is both a query and at least
one source kind. Note this is a different setting from **RAG scope**
above: "Sources" picks the source *kinds*, "Scope" picks the *items*.

Both a manual **Search Library** run and auto-retrieve (below) search to
the depth your **active RAG profile** specifies (`Settings ▸ RAG`'s
**Default results** field) rather than a fixed count, so manual and
automatic retrieval can't disagree about how many results come back.

### Auto-retrieve on send

The **Library search** settings modal also carries an **Auto-retrieve on
send** switch, default **OFF**. Turn it on and every plain text send —
never a slash command, a `$skill` invocation, a tool approval, or a
regenerate — first runs a Library search using your draft as the
query and stages whatever it finds into the staged-evidence strip before
the send goes out: the same visible, consume-on-send pipeline a manual
**Search Library** run produces, never invisible prompt injection. It's
skipped automatically when evidence is already staged (a manual run or a
Library "Use in Console" handoff), so a send can't double-retrieve.

The switch persists the instant you flip it, unlike the query and
source-kind edits in the same modal — closing with **Escape** or a
backdrop click still keeps the change. If your resolved RAG scope comes
back **empty**, auto-retrieve short-circuits with the same shared notice
the manual path shows, rather than searching everything.

While a send is retrieving, the staged-evidence strip briefly shows a
"Retrieving…" state; the search is capped at a **5-second timeout**, and
a send is never blocked on it. If retrieval times out, fails, or the RAG
service is still starting up (a first-use embedding-model load can take a
while), the send goes out without evidence and a quiet notice names which
of the two happened — "still initializing" vs. "failed" — rather than
staying silent. A zero-result outcome currently clears the in-flight
placeholder with no further notice.

The Inspector tray is not the only place staged evidence shows up: a
**staged-evidence strip** sits on the main surface itself, at the top of
the control deck above the status chips, so staging is visible without
opening the Inspector at all. Staged, it lists the titles (up to three, "+N more"
beyond that) with an **Un-stage** button that drops the whole bundle in
one click; after a send that used it, the strip briefly instead reads
"Evidence sent with this message · N sources". Staged evidence rides only
the **next** send — once a send consumes it, the field clears itself, so
the strip and the settings modal's "staged for your next send" wording
are both literally true (an earlier build let one staged bundle silently
ride every later send too; that is fixed).

Staged evidence also survives leaving Console and coming back: it lives
in the Console's own session state, not only in memory, so navigating to
Library (or anywhere else) and returning still shows the same bundle
staged — the strip, the Inspector's Sources tray, and the settings
estimate all keep reporting it. Staging a *new* item from Library while
something is already staged replaces it outright: a fresh "Use in
Console" click always wins, even over a bundle restored from an earlier
visit. Whatever is currently staged, the strip's count, the tray's
"Sources N" count, and the Inspector's Source Readiness line ("Evidence:
N/N available") always agree on the same number.

Staged evidence also counts toward the Console Settings context estimate
and the running-session cost chip — it used to report zero for anything
staged but not yet sent. The estimate counts the staged snippets as
they'll actually be sent (each capped at 4,000 characters, the same cap
enforced when the evidence was staged), so even a very large source
contributes a bounded, non-zero number rather than nothing. Two honest
gaps in that count: it doesn't add back in the small `[S1] label — title`
header and separator each staged source gets at send time, and the cost
chip marks the figure as an estimate (its existing `~` prefix) since
nothing has actually been sent yet.

### Citations

When a reply is grounded in sources, a **Sources (N)** button appears
under it, opening the **Sources** window: `[S1]`-numbered rows, a detail
pane with the matched snippet, an **Open in Library** button (shown when
the source can be opened), and **Close**.

Dim notices under the reply track verification: "Checking citations…",
then "Citations repaired" or "Citations repaired · View original attempt"
(the **View original attempt** message action toggles an inline preview
of the pre-repair text, headed "Original attempt (not selected)"), or
"Citation repair unavailable · Original response kept".

### Chat dictionaries & world books

Chat dictionaries rewrite matching text before it reaches the model;
world books inject lore entries into the prompt when their keywords come
up. You author both on Roleplay & Chat Dictionaries — see
[Chat dictionaries](../roleplay-chat-dictionaries/chat-dictionaries.md)
and [Lore books](../roleplay-chat-dictionaries/lore-books.md). The
Inspector shows what's in play:

- **Chat Dictionaries** — rows marked "from conversation" or "from
  character", with " (shadowed)" and/or " (disabled)" appended; actions
  **Attach dictionary…** / **Detach dictionary…**. Empty state: "No
  dictionaries in play".
- **World Books** — rows showing "N entries" (plus " (disabled)");
  actions **Attach world book…** / **Detach world book…**. Empty state:
  "No world books in play".

## Common tasks

1. **Check exactly what the next send contains** — type your draft, press
   **Ctrl+Shift+P**, open **Next Send**, expand the folds; `r` refreshes.
2. **See exactly what a past turn sent and received** — open the
   Conversation Inspector (chip or Ctrl+Shift+P), switch to **Exchange**,
   expand the turn, then a call, then the section you want (System prompt,
   Messages, Tools, Response, Tool calls, Sampling & routing).
3. **Set a system prompt for this session** — type `/system`, write the
   prompt, press **Apply**; the rail's `System:` line now previews it.
   Name it and press **Save to Library** first to reuse it later.
4. **Make the reply start with a fixed opening** — type
   `/prefill Here is the summary:` and send; the reply continues from the
   last character. `/prefill pin …` keeps it; `/prefill clear` when done.
5. **Narrow RAG to two documents** — expand the Inspector, press
   **Narrow…** on the scope row, pick the **Media** tab, filter by title,
   click the two items, press **Save**. The strip shows **Scope: 2**.
6. **Open a citation's source in Library** — click **Sources (N)** under
   the reply, select an `[S1]` row, press **Open in Library**.
7. **Have every send ground itself automatically** — click the **RAG**
   chip (or **Run Library RAG**) to open **Library RAG** settings, turn on
   **Auto-retrieve on send**, close the modal. It stays on across sends
   until you flip it off; it's off by default.

## Keyboard & commands

| Key / command | Action |
|---|---|
| Ctrl+Shift+P | Open the Conversation Inspector on **Next Send** |
| r (on **Next Send**) / Escape | Refresh the snapshot / close the Inspector |
| Tab / Shift+Tab (in either rail) | Move through controls and any overflowing section body in normal order |
| Arrow keys / Page Up / Page Down / Home / End | Scroll within a focused overflowing section body |
| n / p (Inspector focused) | Move to the next / previous named section, without wrapping or taking over editable input |
| `/prompt [name]` | Replace the draft with a saved prompt (picker when ambiguous) |
| `/system [name]` | Edit the session system prompt, or apply a saved prompt's system part |
| `/prefill [pin\|clear] [text]` | Set, pin, clear, or report the start of the assistant's reply |

Mistyping a command shows the full list: "Unknown command /… — available:
/prompt, /system, /skills, /prefill, /generate-image, /rewind. Press
Enter again to send as text."

## Related settings & docs

- `config.toml` `[rag]` (and legacy `[rag_search]`) — retrieval and
  processing settings; covered in
  [Library ▸ Search & RAG](../library/search-and-rag.md), including how
  the active RAG profile now drives retrieval mode.
- `config.toml` `[chat_defaults] rag_auto_retrieve_on_send` — the
  persisted **Auto-retrieve on send** value (default `false`); the modal
  is the supported way to change it.
- `config.toml` `[console] exchange_capture` — the Conversation Inspector's
  capture kill-switch (default `true`); set `false` to stop recording
  per-call request/response detail for the Exchange tab.
- [Settings ▸ RAG](../settings/rag.md) — the profile that both auto- and
  manual Library RAG retrieval read for search mode and result depth.
- [Library ▸ Prompts](../library/prompts.md) — where saved prompts are
  created and managed.
- [Console orientation](../console.md) — layout tour, rails, and chips.
- [Branching & rewind](branching-and-rewind.md) — how regenerate, edit &
  resend, and "Summarize up to here" change what history the model sees.
- [Guide index](../index.md) — global keys and navigation.
- [Agent runs & tools](agent-runs-and-tools.md#project-instructions-before-tools-run)
  — lazy nested activation before review and execution.

## Quirks & troubleshooting

- **Prefilled sends skip tools.** While any prefill is armed, tool
  calling (including MCP) is skipped for that send — the armed
  confirmation and the viewer's Response Prefill fold both say so.
- **A one-shot prefill can't literally start with `pin `** (or be exactly
  `clear`) — those parse as subcommands. Rephrase, or pin then clear.
- **"Scope: no sources" means zero-result retrieval.** The alert-styled
  state warns before you send into it — **Clear** or **Edit** the scope.
- **The Next Send viewer's token count is a draft-derived estimate** — a
  guide, not a billing meter. The same is true of every per-piece token
  count inside the Exchange tab's calls; only that tab's **Reported
  usage** line comes from the provider itself.
- **A capture is not a byte-for-byte wire log.** It's taken where Console
  hands the request to the provider adapter, so adapter-internal HTTP
  framing and any prompt-caching markers an adapter injects are not
  visible — except on a llama.cpp (local server) send, where the capture
  is the literal payload that went out.
- **Turns before this feature (or with capture off) show "No capture
  recorded"** on the Exchange tab rather than any reconstructed guess at
  what was sent.
- **Auto-retrieve fires on every plain-text send while it's on**,
  including repeated sends in the same conversation — there's no
  once-per-conversation memory yet, so an empty resolved scope re-shows
  its notice on each send until you clear or edit the scope.
- **Reranking-enabled profiles cost more per search.** If your active RAG
  profile has reranking on, both auto- and manual Library RAG retrieval
  spend one LLM provider call per candidate result (up to the profile's
  **Rerank results** count) — see
  [Settings ▸ RAG](../settings/rag.md#the-editing-card).

—
*Verified against c2cbb8081 — 2026-08-04 (PR-T1 live check S1-S6: staged
evidence survives Console <-> Library navigation and a fresh handoff
supersedes a stale restored one. Media/notes/conversation handoffs
delivering content on send is covered by capture round-trip tests
(task-2374); the live check's own handoff scenario was blocked on this
profile by an unrelated Library workspace-eligibility gate, so that part
is verified at the code level, not live). Verified against e2c706303 —
2026-08-06 (PR-T2, docs pass against shipped code/tests, live check
pending Task 9): staged evidence now counts toward the context estimate
and the cost chip (as an estimated `~` row) instead of reporting zero.
Verified against d6b6a738f — 2026-08-07 (RAG-port P0 live walkthrough, real
Anthropic provider): flipping **Auto-retrieve on send** in the RAG chip
modal writes `[chat_defaults] rag_auto_retrieve_on_send = true` at
toggle time — before Esc, and Esc leaves it set. A plain-text send then
showed "Auto-retrieving Library evidence for this message." with the chip
reading `RAG: on · Sources: 1 staged` about a second in, then "Evidence
sent with this message · 15 sources", and the model's own reply named the
injected block back ("the evidence sections [S1] through [S15] …") — the
end-to-end proof that retrieved evidence reaches the provider. A send
beginning with a slash command fired no retrieval at all: no placeholder,
no chip flip, no evidence line.*

*Chip and strip positions re-verified against dev @ b6036515e — 2026-08-18
(task-17662, after the bottom-stack programme moved the status chips above
the composer and the staged-evidence strip above the status chips).*

*Verified against dd4921901 — 2026-08-20 (task-18300 Task 11a, Conversation
Inspector programme): the token/cost chip, Ctrl+Shift+P, and the palette's
"Console: View chat context" entry now all open one Conversation Inspector
modal (Costs / Exchange / Next Send tabs) in place of the former separate
cost breakdown and "Chat Context" viewer; the retired viewer's content
lives on unchanged as the Next Send tab. Docs pass against shipped
code/tests (`console_conversation_inspector.py`, `console_exchange_
capture.py`, the design spec's UI and Risks sections); live verification of
the Exchange tab against a real provider is a separate, later pass.*
