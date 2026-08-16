# Console — Chat, source handoffs, live runs, and control actions.

## What this screen is for

Console is the app's chat and agent workbench: you talk to your configured
provider here, stage Library sources and RAG evidence into the
conversation, run agents in parallel tabs, and approve or deny the actions
those runs request. Reach for it to send a message, drive a live run, or
hand work off between your sources and a model. This page is the
orientation tour; the details live on six child pages:

- [Chat basics](console/chat-basics.md) — compose, send, stream, stop, act on messages.
- [Sessions, tabs & workspaces](console/sessions-tabs-workspaces.md) — tab strip, "Switch Session", conversation browser, workspaces.
- [Branching & rewind](console/branching-and-rewind.md) — regenerate variants, edit-and-resend forks, `/rewind`.
- [Attachments, images & voice](console/attachments-images-voice.md) — Attach picker, paste/drop, clipboard images, image generation, dictation.
- [Video generation, playback & streaming](console/video.md) — `/generate-video`, ephemeral videos & tombstones, in-app playback, `/stream-video`.
- [Agent runs & tools](console/agent-runs-and-tools.md) — per-tab runs, fleet markers, approvals, skills, MCP tools.
- [Context & RAG](console/context-and-rag.md) — "Chat Context" viewer, prompts, retrieval scope, staged sources, Library RAG.

## Getting there

- Press **Ctrl+2** from anywhere, or click **⌃2 Console** in the nav bar.
- **Ctrl+P** → "Tab Navigation: Switch to Console" in the command palette.
- To land here at launch, set `default_tab = "chat"` under `[general]` in `config.toml`.

## Layout tour

![Console overview](images/console/overview.svg)

Top to bottom:

- **Header** — the title "Console", the subtitle "— Chat, source handoffs,
  live runs, and control actions.", and a status badge that reads **Ready**,
  **Running**, or **Blocked** depending on the active session.
- **Control bar** — one row of buttons: **New tab**, **Settings**,
  **Attach context**, **Search Library**, **Help**. (**Save as Chatbook**
  lives in the composer's **Menu** button, left of the draft.)
- **Left rail: "Console context"** — sections **Session** (workspace and
  the conversation browser), **Model**, **Agent**, and **Details**. The
  **◂** button in the rail header collapses it; while collapsed, a thin
  **Context ▸** handle on the far left brings it back.
- **Conversation pane** — titled "Conversation", extended to
  "Conversation | \<session title\>" once a session is active.
  Above it sits the session tab strip: one button per tab (each with a
  **✕** close button) ending in **New tab**.
- **Right rail: "Inspector"** — collapsed by default. Its handle on the
  right edge reads "Inspector" and grows small badges when something needs
  you (pending approvals, an available artifact). Opened, it holds
  **Sources**, the retrieval scope row ("Scope: everything" until you
  narrow it), a run status line, groups such as **Run**, **Tools**,
  **Approvals**, and **Artifacts**, the **"Live work sources"** card
  (ask Library sources before sending), and the **Session Settings**
  summary.
- **Staged-evidence strip** — appears directly above the composer only
  while Library RAG evidence is staged (or briefly after a
  send consumes it); lists what's staged with an **Un-stage** button — see
  [Context & RAG](console/context-and-rag.md).
- **Composer row** — the "Composer ▾" collapse toggle, the draft area
  ("Ask, command, or paste task..."), then **Send**, **Mic**, **Attach**,
  and **Save**; a **Stop** button appears between Send and Mic while a
  reply is streaming. **Send** is genuinely disabled whenever a send can't
  go through — nothing typed yet, setup incomplete, or a reply still
  streaming — and the reason shows inline next to it (e.g. "Send blocked —
  choose a model to continue"), so you never have to hover to find out why.
  You can just start typing from almost anywhere on
  the screen — printable keys go straight into the draft.
- **Status chip strip** — the shell's bottom row, one row of chips directly
  below the composer: **Provider**, **Model**, **Assistant**,
  **Library search**, **Sources**, **Tools**, **Approvals**, and — once
  retrieval is narrowed — **Scope**.
  The chips are actions, not just readouts: **Sources** and **Tools** open
  the Inspector rail (the only way to reach it in single-pane mode, where
  the edge handles hide), **Provider**/**Model** open the model picker,
  **Library search** opens the search settings, **Approvals** jumps to the
  pending approval card, and **Scope** opens the scope picker. The **Tools**
  chip only appears once tools are counted for the session (after your
  first send) — before that it stays hidden rather than guessing.
- **Footer** — shortcut hints (F6, Shift+F6, F1, Enter, Ctrl+K, Ctrl+T,
  Ctrl+P), a word count, the "Tokens:" counter, and database sizes.

### Small terminals

The shell adapts instead of clipping: below 35 rows the header banner hides
(and the control bar gains a small **Ready**/**Running**/**Blocked** marker
so the status identity survives); below 150 columns the Inspector rail
starts collapsed and below 100 columns the left rail starts collapsed too —
these compact collapses are only the default: opening a rail from its
handle (or via the **Sources**/**Tools** chips) always works, at any width,
and your stored open/closed preference is kept and restored at wider sizes;
and below 84 columns the workspace switches to a single pane — both edge
handles hide and the transcript takes the full width, so it stays usable
even at 80x24 or 60x18.

### First run: the "Get started" card

On a brand-new install, an app-level first-run wizard
([First-run setup](First_Run_Setup.md)) offers to set up a provider and
model first — it is skippable, and Settings can do everything later.

If no provider is configured when you open Console, the shell is replaced
by a **Get started** card with three numbered steps — "Connect a provider
(API key or local server)", "Pick a model", "Send your first message" —
marked with ● (current) and ○ (pending) glyphs, plus the note "Composer
unlocks after setup". Its button follows the current step (**Set up
provider**, then **Choose model**) and opens the Console Settings modal.
The composer stays locked until a provider and model are configured; once
they are, the empty transcript reads "Ready — type a message to begin."

If you land here with a handoff already staged — e.g. from Library's
**Use in Console** on a Search/RAG result while a provider isn't set up
yet — the card shows an extra line under "Get started" naming what's
staged and that finishing setup is what unlocks it (for example,
"Library Search/RAG evidence staged — finish provider setup to use it.").
The handoff itself is never lost: it's the same staged context the
composer-level strip below shows once setup completes.

## Features & controls

### Control bar

| Control | What it does |
|---|---|
| **New tab** | Creates a Console tab — see [Sessions, tabs & workspaces](console/sessions-tabs-workspaces.md). |
| **Settings** | Opens the "Console Settings" modal (provider, model, tools, and generation). |
| **Attach context** | Opens the "Console context" rail (staging itself is done from Library) — see [Context & RAG](console/context-and-rag.md). |
| **Search Library** | Searches Library evidence before sending — see [Context & RAG](console/context-and-rag.md). |
| **Save as Chatbook** (composer **Menu**) | Saves this run as a Chatbook — see [Artifacts](artifacts.md). |
| **Help** | Opens the Console help panel (same as F1). |

### Rails and handles

| Control | What it does |
|---|---|
| **◂** / **▸** (rail headers) | Collapse that rail (**◂** on "Console context", **▸** on "Inspector"). |
| **Context ▸** handle | Reopens the collapsed "Console context" rail. |
| **Inspector** handle | Reopens the collapsed "Inspector" rail; shows badges like "1 appr" (pending approvals) or "art" (artifact ready). |
| **Session** section | Workspace controls and the conversation browser — see [Sessions, tabs & workspaces](console/sessions-tabs-workspaces.md). |
| **Model** section | Read-only Provider / Model / Temperature / Max tokens lines plus a **Configure** button that opens Console Settings. |
| **Agent** section | Live run status and the full run log — see [Agent runs & tools](console/agent-runs-and-tools.md). |
| **Details** section | Storage, sync, file tools, server, and handoff status for the workspace. |
| **Character** section | Appears only when the character-avatar preference is on: the active character's portrait (click to enlarge) and name. |

### Status chips

| Chip | What it shows |
|---|---|
| **Provider** / **Model** | The active provider and model for this session. |
| **Assistant** / **Library search** | The active assistant; whether a Library search is on for the next send. |
| **Sources** / **Tools** | Staged source count (e.g. "Sources: 0"); tool readiness (e.g. "Tools: 10 ready" — hidden until tools are counted). |
| **Approvals** | Pending approvals; press Enter or Space on it to jump to the approval card. |
| **Scope** | Appears when retrieval is narrowed ("Scope: N"); Enter or Space opens the scope picker. |

### Long conversations

Opening or switching to a long conversation shows the most recent stretch of it
first, rather than mounting the whole history up front — a 500-message session
opens in about a second instead of tens of seconds. Scroll to the top of what is
shown (wheel, Page Up, or the scrollbar) and the previous chunk is prepended
under you, keeping the same message in view; the jump-to-latest pill or a new
send takes you back to the tail. Nothing is deleted: exports, `/rewind`, and the
context sent to the model always use the full history.

Scroll-back no longer stops at the watermarks: the view slides rather than
grows. Once the mounted stretch reaches `prune_low_watermark` (12,000 rows by
default), scrolling further back keeps loading older history while the newest
end of the stretch is set aside the same way — so a very long session stays
reachable by scrolling, at a roughly constant memory cost. One deliberate
exception: a selected message is never set aside, so a selection pinned at
either end of the stretch pauses the sliding in that direction until you
clear it (Esc) — after a jump to an old message, the jumped-to selection
sits at the oldest end, so reading far enough past it eventually pauses the
forward walk the same way (the view stays bounded instead of growing).
Scrolling back down (or a jump to an old message — selecting one far outside
the stretch lands you on a fresh window around it instead of loading
everything in between) walks forward the same way, and the jump-to-latest
pill or a new send always returns you straight to a fresh view of the tail.
Tune it under `[chat_defaults]` in `config.toml`:

- `transcript_window_lines` (144) and `transcript_scrollback_lines` (96) are
  **floors**, not the budget. The window actually used is the larger of the
  floor and your terminal height ×6 (×4 per scroll-back step), so on any
  terminal 24 rows or taller the shipped floors change nothing — raise them
  above `height × 6` to widen the window, or set `transcript_window_lines` to
  `0` to mount the whole history at load, as before.
- `prune_low_watermark` / `prune_high_watermark` bound the mounted view itself
  and keep working with the window disabled. With the window disabled (or with
  watermarks set too small to hold a scroll-back step), sliding is off too:
  history the watermarks pruned is then reachable only via export or a jump,
  as before TASK-15777.

### Composer

Editing keys, send/stream/stop, attachments, and Mic dictation live in
[Chat basics](console/chat-basics.md) and
[Attachments, images & voice](console/attachments-images-voice.md).
Shell-level: **Composer ▾** collapses the whole row to a single "Composer
hidden" line (with **Expand ▴** at its right, and **Stop** kept available
during a run); **Esc** expands it and returns the caret to your draft.

### Session settings & model selection

The **Console Settings** modal is the one place provider, model, and
generation settings live. Open it from the control bar's **Settings**
button, the Model section's **Configure** button in the left rail, or the
**Session Settings** action in the Inspector. Inside:

- A readiness line up top (e.g. "custom is ready. No API key is required.").
- **Provider and model** — Provider and Model selects, **Custom model** for
  a name the list doesn't offer, **Discover models** to list what a Base
  URL serves, and the **Base URL** field for local/self-hosted endpoints.
- **Sampling** (Temperature, Top P, Min P, Top K, Max tokens, Seed, and
  related knobs), then **Provider-specific**, **Context**, and **Identity**.
- Footer: **Cancel** / **Save as default** / **Save**, under the note "Save
  applies to this session only. Save as default also writes provider +
  streaming defaults to config."

For a faster switch, **Alt+M** opens the quick **Model** popover —
provider, model, and temperature without the full modal.

#### QwenCloud in Console

QwenCloud behaves like the other hosted providers: select it once, use the
normal streaming Console and native function tools, and discover models
through the shared cached catalog. Configure its durable **API mode** in
**F9 ▸ Providers & Models**; it is not a per-session Console override. A run
pins the selected mode and endpoint for every model turn, so changing Settings
mid-run cannot switch its continuation to another API.

- `responses` is the default. It re-sends the required history, does not send
  `previous_response_id` or a provider conversation ID, and does not rely on
  provider-managed session state. It requests `store=false` where the
  compatible endpoint honors it; Chatbook makes no claim about provider
  operational retention or caching.
- `chat_completions` disables preserved-thinking replay because Chatbook does
  not retain private reasoning content.
- Existing Chatbook function tools work through the same approval, execution,
  cancellation, budget, and continuation path in both modes. QwenCloud-hosted
  built-in tools are not exposed.
- The default model is `qwen3.8-max`. Model/mode availability still depends on
  your QwenCloud account; use model discovery or the provider's recovery error
  instead of assuming compatibility from the model name.
- Token usage can be shown even when pricing is unavailable. **Pricing
  unknown** means no verified rate is configured, not zero cost.

Optional live verification makes paid requests and is never part of the
default suite. With `DASHSCOPE_API_KEY` already exported, explicitly opt in:

```bash
TLDW_LIVE_QWENCLOUD=1 .venv/bin/python -m pytest -q \
  Tests/Chat/test_live_qwencloud_api.py
```

The test runs both API modes in isolated temporary config and data profiles,
checks identifying text plus a marker derived from one calculator result, and
does not print the key, prompt, or response. Override the defaults only when
your account requires it with `TLDW_LIVE_QWENCLOUD_MODEL` or
`TLDW_LIVE_QWENCLOUD_API_BASE_URL`.

#### Moonshot Kimi and Z.ai GLM in Console

**Moonshot** and **Z.ai** use their stable provider identities and the ordinary
streaming Console path. They support Chat Completions only—there is no Responses
mode or provider conversation ID. Fresh defaults are `kimi-k3` at
`https://api.moonshot.ai/v1` and `glm-5.2` at
`https://api.z.ai/api/paas/v4`; saved historical models remain usable.
Moonshot's China endpoint and intentional compatible custom endpoints are
configured in **F9 ▸ Providers & Models**.

- Existing Chatbook function tools use the same approval, cancellation,
  execution, budget, and durable recovery loop for both providers. Moonshot
  and Z.ai hosted search, retrieval, code, memory, and other built-in tools are
  not exposed.
- Kimi K3 uses always-on Preserved Thinking. Its retained assistant reasoning
  is private but is replayed when K3 requires it. Other Kimi models follow only
  their curated policy. GLM keeps reasoning for an active or restored function
  tool run with `clear_thinking=false`; ordinary GLM chat clears prior thinking.
- Private continuation data is assistant/variant-owned, bounded, and omitted
  from the visible transcript, logs, summaries, ordinary exports, and usage
  details. It still counts against the context window and is evicted atomically
  with its visible owner.
- Terminal usage reaches Console when the provider returns it. If a selected
  model has no verified rate, **pricing unknown** means cost was not estimated;
  it never means free.
- Model discovery reuses the chat endpoint and credential. Moonshot discovery
  is authenticated; Z.ai is best-effort, so a failed catalog refresh keeps the
  configured/cached models and does not block generation.

If a tool run is interrupted, opening the conversation does not execute
anything. Use **Resume** only after checking the pending calls and approving
them again; Console pins the original provider, model, Chat-Completions
protocol, and normalized base while resolving the current credential. Use
**Take over** to continue visibly without replaying the provider checkpoint, or
**Discard** to clear it. Executing calls remain ambiguous and are blocked;
completed and failed calls are not run again. Invalid endpoint/config errors
must be repaired and saved in Settings before retrying.

Optional live verification is paid and skipped by default. Export the relevant
API key before running either command. Each command starts
a fresh isolated profile, suppresses child output/logging, and proves one real
Calculator result changes the final answer. It requires both the exact opt-in
flag and a nonblank key:

```bash
TLDW_LIVE_MOONSHOT=1 .venv/bin/python -m pytest -q \
  Tests/Chat/test_live_moonshot_zai_api.py -k moonshot

TLDW_LIVE_ZAI=1 .venv/bin/python -m pytest -q \
  Tests/Chat/test_live_moonshot_zai_api.py -k zai
```

Use `TLDW_LIVE_MOONSHOT_MODEL` / `TLDW_LIVE_MOONSHOT_API_BASE_URL` or
`TLDW_LIVE_ZAI_MODEL` / `TLDW_LIVE_ZAI_API_BASE_URL` only when your account
requires an override. The default test suite makes no paid request.

### Leaving Console during a run

Agent runs are screen-scoped: navigating to any other screen cancels every
in-flight run and denies every pending approval. If runs are active, a
**Leave Console?** dialog warns you first ("N agent runs will be cancelled
if you leave Console. Leave anyway?") with **Leave** / **Stay** buttons,
and a one-time toast on return reports what was cancelled. Details in
[Agent runs & tools](console/agent-runs-and-tools.md) and the
[guide index](index.md#console-agent-runs-are-screen-scoped).

## Common tasks

1. **Set up a provider from the Get started card.** Click **Set up
   provider**, pick a provider in "Provider and model" (for a local server,
   enter its Base URL, then **Discover models**), pick a model, and press
   **Save**. The card's steps tick off and the composer unlocks.
2. **Switch model for just this session.** Press **Alt+M**, choose the
   provider/model, and confirm — or open **Settings** and press **Save**
   (not "Save as default"). Other tabs and future launches are unaffected.
3. **Make today's provider the default.** Open **Settings**, configure
   provider and model, and press **Save as default** — the next launch
   starts there.
4. **Get a distraction-free transcript.** Click **◂** in the "Console
   context" rail header (the Inspector is already collapsed by default);
   click **Composer ▾** to hide the composer too, and press **Esc** to
   bring it back. Reopen the rails from the edge handles.
5. **Find any Console shortcut.** Press **F1** — the help panel lists the
   visible actions, agent/fleet notes, and every shortcut grouped by pane.

## Keyboard & commands

Screen-level keys only — global keys live in the [guide index](index.md).

| Key | Action |
|---|---|
| F1 | Open the Console help panel (actions, agent notes, full shortcut list) |
| F6 / Shift+F6 | Focus the next / previous pane (context rail → transcript → Inspector → composer) |
| Ctrl+K | Open the "Switch Session" conversation finder |
| Ctrl+T | New Console tab |
| Alt+1 … Alt+9 | Jump to Console tab 1–9 |
| Alt+M | Quick "Model" popover |
| Alt+W | "Change Workspace" switcher |
| Alt+V | Paste an image from the clipboard |
| Ctrl+Shift+P | "Chat Context" viewer (what the model will see) |
| Esc | Return focus to the composer (expanding it first if collapsed) |

While Console is the active screen, the command palette (**Ctrl+P**) also
gains "Console: …" entries for these same actions. Slash commands
(`/prompt`, `/system`, `/skills`, `/prefill`, `/generate-image`, `/rewind`)
are covered on the child pages, chiefly [Context & RAG](console/context-and-rag.md) and [Branching & rewind](console/branching-and-rewind.md).

## Related settings & docs

- **Settings ▸ Console Behavior** — parallel-run limit, paste collapse, and
  other Console preferences.
- `config.toml`: `[chat_defaults]` (default provider/model/sampling),
  `[api_settings.*]` (per-provider keys, endpoints, streaming — the modern
  form; an explicit key here now outranks that provider's environment
  variable), `[API]` (legacy `<provider>_api_key` values — still honored,
  lowest precedence of the three, normalized once at load into the same
  credential both this screen's readiness check and Library's RAG Answer
  gate use), `[console]` and `[console.background_effects]` (paste
  collapse, ambience), `[chat.images]` (attachments), `[general]`
  `default_tab` (start here).
- Child pages: [Chat basics](console/chat-basics.md) · [Sessions, tabs & workspaces](console/sessions-tabs-workspaces.md) · [Branching & rewind](console/branching-and-rewind.md) · [Attachments, images & voice](console/attachments-images-voice.md) · [Agent runs & tools](console/agent-runs-and-tools.md) · [Context & RAG](console/context-and-rag.md)
- Deep dives: [Speech services](../Features/Speech-Services-Guide.md) (Mic dictation backends) · [Chat dictionaries](../Features/ChatDictionaries-Documented.md).

## Quirks & troubleshooting

- **Status chips look truncated.** They ellipsize to fit the row — hover a
  chip for its full text.
- **There's no Tools chip before the first send.** Tools are counted lazily,
  so the chip stays hidden until your first send in the session; it then
  reads e.g. "Tools: 10 ready".
- **A run vanished when you switched screens.** Leaving Console cancels
  runs and denies pending approvals — see [Leaving Console during a
  run](#leaving-console-during-a-run). Nothing is ever auto-approved.
- **Console didn't appear on first launch.** The first-run wizard's skip
  path lands on Home; with `default_tab = "chat"` set, the next launch
  opens Console directly.
- **Alt+M does nothing.** Some terminal/multiplexer setups deliver Alt
  chords as a separate Esc + letter, which Console reads as Escape then a
  typed character. The same popover is always reachable via **Ctrl+P** →
  "Console: Change model…".

—
*Verified against 4646922ed — 2026-08-04 (PR-4 Task 6 live check, including
a real-provider send round trip). Verified against e2c706303 — 2026-08-06
(PR-T2, docs pass against shipped code/tests, live check pending Task 9):
a legacy `[API] <provider>_api_key` now satisfies this screen's own
readiness check too, and a modern `api_settings.<provider>.api_key` now
outranks that provider's environment variable. Verified against
42b28089f — 2026-08-06 (task-2852: live check on a fresh profile — a
Library Search/RAG handoff staged while locked now shows a receipt line
on the Get started card, and the same handoff on a configured Console
still lands on the unchanged staged-evidence strip). "Long conversations"
verified against the TASK-15455 windowing (PR #1538) plus its reconciliation
delta — shipped tests and an isolated 500-message load probe; not re-checked
live. "Long conversations" sliding scroll-back and bounded far jumps
verified against TASK-15777 — shipped tests plus isolated 400/500-message
mounted probes (scroll-back walked m360→m0 at a constant 101 mounted rows;
a far jump mounted 5 rows instead of 490); not re-checked live. The
head-pinned-selection pause (TASK-16851) verified by shipped tests — a
post-jump walk-down held ≤1100 virtual rows against a 900 high mark where
it previously grew to 1966 and kept growing; not re-checked live.*
