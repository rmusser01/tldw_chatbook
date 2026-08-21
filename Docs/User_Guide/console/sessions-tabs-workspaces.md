# Console sessions, tabs & workspaces — run several chats side by side and keep them organized

## What this page is for

The Console can hold many conversations at once: tabs run chats (and their
agents) in parallel, the conversation browser in the left rail finds and stars
past chats, and workspaces group related conversations into separate contexts.
Reach for this page when you want a second chat running while the first one
streams, when you need to dig up an old conversation, or when one "Default"
pile of chats stops being enough.

## Getting there

Open the Console screen (see [Console](../console.md) for navigation and the
layout tour). Everything on this page lives in two places:

- The **tab strip** — the row of tabs directly under the
  "Conversation" title above the transcript.
- The **"Console context"** rail on the left — its separate **Sessions**,
  **Workspaces**, **Conversations**, and **Details** sections. If the rail is
  collapsed, click the **Context->** handle at the left edge to open it when
  the viewport can retain a usable transcript.

## Layout tour

![Console tab strip with the fleet coach-mark](../images/console/tabs-coachmark.svg)

**Tab strip.** Each open chat is a tab. A tab's label starts out as
"Chat 2"-style and takes the text of the first message you send. Labels are
trimmed to about 19 characters — hover a tab to see the full title. Each tab
has a "✕" close button, and the strip always ends with a "New tab" button and
a "Temporary" button (a chat that is never saved locally).
Tabs that are running an agent show a marker glyph before the label:
● running, ◆ needs approval, ✓ finished, ✗ failed — the mark clears once you
visit that tab (see [Agent runs & tools](agent-runs-and-tools.md)).

The first time you open a second tab, a one-time coach-mark appears under the
strip: "Each tab runs its own agent — up to 3 in parallel (change in
Settings > Console Behavior)." Dismiss it with its "✕". The "3" is the
default limit — the banner shows whatever your configured limit is.

**Sessions section** (left rail). Names the active conversation ("None" when
no conversation is active yet); hover the value to see its durable id.

**Workspaces section** (left rail). Shows "Workspace" plus the active
workspace name, with "Switch", "New", and "RAG Scope" buttons. "RAG Scope"
narrows retrieval to this workspace's items — see
[Context & RAG](context-and-rag.md).

**Conversations section** (left rail). A "Search conversations" box with a
"Clear" button, a "New conversation" button, and three collapsible groups:
"Starred", "Workspaces", and "Chats".
Empty groups start collapsed as quiet one-line headers — expand one to see
its empty copy (for example "No starred conversations."); while you search,
groups with no matches stay expanded so the "No ... conversations." feedback
stays visible. Each row shows the conversation title
plus a secondary line of `<workspace> - <state> - <age>` (for example
"Chats - active session - 1m"), and ends with a star toggle ("☆" / "★").

**Details section** (left rail, collapsed by default). Status lines for
"Storage", "Sync", "File tools", "Server", and "ACP", plus a "Handoff" list.
On a local-only setup the server lines collapse into one line:
"Server features (sync, handoff, ACP): not configured. Chats stay local."

## Features & controls

### Tabs

| Control | What it does |
|---|---|
| "New tab" (strip or control bar) / Ctrl+T | Opens a fresh chat tab |
| Click a tab | Switches to it; a second click on the active tab opens "Rename Chat Tab" |
| Middle-click a tab | Closes it |
| "✕" on a tab | Closes it; if the tab has messages, a "Close Tab" confirmation warns "This tab has messages that will be lost." and asks "Close it anyway?" — "Close" / "Keep" |
| Alt+1 … Alt+9 | Jumps straight to tab 1–9 |
| Marker glyph (● ◆ ✓ ✗) | That tab's agent-run status — clears when you visit the tab |

Each tab keeps its own unsent draft: switch tabs mid-thought and the
half-typed message is still in the composer when you come back.

### "Switch Session" (Ctrl+K)

A fuzzy finder over your conversations. Type into "Search conversations…" to
filter, press Enter to activate the top result, or move through results with
↑/↓ and press Enter (or click) on the one you want. F2 renames the
highlighted result when it is an open tab. Esc cancels.

### Conversation browser

| Control | What it does |
|---|---|
| "Search conversations" + "Clear" | Filters the row list, settling about a fifth of a second after you stop typing; "Clear" resets it immediately |
| "New conversation" | Starts a fresh conversation |
| "Starred" / "Workspaces" / "Chats" | Groups; click a header's ▸/▾ toggle to expand or collapse |
| Conversation row | Click to open it in the Console |
| "☆" / "★" | Stars or unstars the conversation — starred rows collect under "Starred" |

### Workspaces

| Control | What it does |
|---|---|
| "Switch" / Alt+W | Opens "Change Workspace" — "Switching changes Console context only; Library and Notes stay globally visible." Click a workspace to activate it; the active one is marked "(current)" and the built-in one is listed as "Default (everyday chats)" |
| "New" | Opens the "New Workspace" dialog — see below |
| "Rename" (in the switcher) | Opens "Rename Workspace" — edit the name, then "Save" |
| "Archive" (in the switcher) | Opens "Archive workspace?" — "Its conversations stay saved and remain visible in Library; the workspace disappears from the switcher and the Console browser." Confirm with "Archive" |
| "RAG Scope" | Narrows retrieval to this workspace — see [Context & RAG](context-and-rag.md) |

**The "New Workspace" dialog** is the same creation dialog Settings ▸
Workspaces and Library use. It opens with a **name** prefilled "Workspace N"
(edit or keep it), an optional **folder** field with a **Browse…** button
(opens a directory picker) and an **Add folder** button — each folder is
validated as you add it (must be a real, existing directory; not your home
folder or filesystem root; not this app's own data, configuration, or
credential directories — or a folder that contains or sits inside one; not
nested inside/around a folder already in the list) and rejections render
inline without closing the dialog. Folders bind
**read-only** (change that later in Settings ▸ Workspaces). A **"Switch to
this workspace"** checkbox, checked by default, controls whether Create also
activates the workspace; unchecking it creates the workspace without
switching Console to it. **Escape cancels the dialog — nothing is created.**
A name that collides with an existing workspace renders inline and keeps the
dialog open.

If a folder you add contains a `.SKILLS/` project skills folder, its row in
the list gains a "— contains N project skill(s)" annotation. After Create
finishes (the workspace is created either way), a separate import prompt
offers to bring those skills into your skill library — see
[Project skills](../library/skills.md#project-skills-skills) for the full
convention and what "trust-pending" means for an imported skill. Declining
here or at app startup is remembered per project, so you're not asked again
until the project's skill set changes.

The built-in Default workspace has no "Rename" or "Archive" buttons. If you
archive the workspace you are in, the Console switches back to Default.

Activating a workspace from Settings ▸ Workspaces or Library (its "Create
local workspace" flow, or "Set active") doesn't touch the Console screen
directly — but the next time you visit Console, it picks up the change and
switches its own session to match, the same as if you had switched with
Alt+W.

### Session project folders

Project instructions are local per-session state. A new session starts
enabled and asks you to choose when more than one eligible local-filesystem
binding exists; exactly one eligible binding may be selected. That binding is
both the instruction authority root and the working directory for local and
built-in file/git tools. Chatbook never searches global or personal
instruction files and never ascends above the selected root.

Legacy sessions and sessions where project instructions are disabled keep the
older local-tool behavior: `[console] workspace_root` is the confinement root,
or the app's startup working directory when it is empty. A selected
project-instruction binding takes precedence over that fallback only while the
feature is enabled for that session. Bindings marked read-only expose only
read-capable tools. Folder names and instruction contents are not synchronized;
the local control fields stay with the local conversation record.

### Details

Open the "Details" header in the left rail to see where your chats live:
"Storage" (local database status), "Sync", "File tools", "Server", and "ACP"
lines (for example "Sync: Off", "Server: Not configured"), and a "Handoff"
list that reads "No handoff package is ready." until a handoff package
exists. If none of the server features are configured, the section shows the
single summary line quoted in the layout tour instead.

## Common tasks

**Open a second tab and run both**

1. Press Ctrl+T (or click "New tab"). A "Chat 2" tab opens.
2. Send a message — the tab label takes your first message's text, and the
   reply runs independently of the other tab.
3. Switch back with Alt+1 (or click the first tab). A ● marker on the other
   tab means its run is still going; ✓ means it finished while you were away.

**Rename a tab**

1. Click the tab to make it active (skip if it already is).
2. Click it again — "Rename Chat Tab" opens.
3. Type the new name and confirm.

**Find an old conversation**

1. Press Ctrl+K, type a few characters of the title, and press Enter to open
   the top match — or,
2. In the left rail, type into "Search conversations" and click the row you
   want under "Starred", "Workspaces", or "Chats".

**Star a conversation**

1. Find its row in the "Conversations" browser.
2. Click the "☆" at the end of the row. It becomes "★" and the conversation
   is pinned under "Starred". Click "★" to unstar.

**Create and switch to a new workspace**

1. In the **Workspaces** section, click "New" — the "New Workspace" dialog opens
   with a prefilled "Workspace N" name.
2. Optional: edit the name, and add a project folder (type a path or click
   "Browse…", then "Add folder") so agents have file access there.
3. Leave "Switch to this workspace" checked and click "Create" — the
   workspace is created and becomes active; new chats now land in it. A
   toast confirms the switch even if the Workspaces section is scrolled out of
   view.
4. To go back, press Alt+W (or click "Switch") and pick
   "Default (everyday chats)".

**Create a workspace without switching to it**

1. Click "New", fill in the dialog as above, and uncheck "Switch to this
   workspace" before pressing "Create".
2. The workspace is created (with any folders bound) but Console stays on
   its current workspace; a "Created \<name\>." toast confirms it.

**Cancel a workspace creation**

Press Escape (or click "Cancel") anywhere in the "New Workspace" dialog —
nothing is created, no matter what you had typed or added.

## Keyboard & commands

| Key | Action |
|---|---|
| Ctrl+T | New Console tab |
| Ctrl+K | "Switch Session" fuzzy finder |
| Alt+1 … Alt+9 | Jump to tab 1–9 |
| Alt+W | "Change Workspace" switcher |

## Related settings & docs

- **Settings > Console Behavior** → "Parallel agent runs" → "Max parallel
  agent runs" — the cap the coach-mark quotes (config key
  `[console] max_parallel_runs`, default 3).
- [Console overview](../console.md) — the full screen tour.
- [Agent runs & tools](agent-runs-and-tools.md) — what the ● ◆ ✓ ✗ markers
  mean in detail, approvals, and the run log.
- [Context & RAG](context-and-rag.md) — RAG scope and staged context.
- [Context & RAG: Project instructions](context-and-rag.md#project-instructions)
  — discovery, consent, status, budgets, and warnings.
- [Guide index](../index.md) — global keys and navigation.

## Quirks & troubleshooting

- On a fresh profile only Default exists, so workspace switching is
  unavailable — the "Switch" button is disabled and hover shows why
  ("Add another workspace before switching."). Click "New" first.
- Closing a tab that has messages always asks the "Close Tab" confirmation;
  there is no way to close a non-empty tab silently.
- Tab titles truncate at about 19 characters. Hover the tab for the full
  title.
- With more tabs than fit the strip's width, the strip scrolls horizontally
  instead of clipping its controls: the active tab scrolls into view
  automatically, and keyboard focus (or scrolling) reveals the trailing
  "New tab" and "Temporary" buttons.
- The coach-mark only goes away for good when you dismiss it with its "✕" —
  if you close the second tab without dismissing it, it can appear again the
  next time you open one.
- A brand-new conversation can't be starred until it has been sent or saved —
  the star's tooltip says "Send or save this conversation before starring."
- The Default workspace can't be renamed or archived.
- Launching the app from inside a project with a `.SKILLS/` folder can pop
  its own import prompt on startup, separately from anything workspace- or
  tab-related — see [Project skills](../library/skills.md#project-skills-skills).
  Turn it off with `[skills] project_skills_prompt_enabled = false`.

—
*Verified against dev @ ff435772c — 2026-07-31*
*Verified against feat/workspace-create-modal @ 64a07a3d7 — 2026-08-17
(task-18704: "New" now opens the shared "New Workspace" dialog — prefilled
name, optional validated folder bindings with Browse…, a "Switch to this
workspace" checkbox default on, and Escape/Cancel creating nothing —
instead of instant "Workspace N" creation; the walkthrough split into
create-and-switch / create-without-switching / cancel tasks to match).*
*Verified against feat/project-skills-import @ 964cb04df — 2026-08-18
(task-18705: a bound folder containing `.SKILLS/` now annotates its row
"— contains N project skill(s)" in the dialog, and a chained import prompt
follows workspace creation).*
*Verified against feat/task-18310-activation-seam @ c9892736f — 2026-08-20
(task-18310: a workspace activated from Settings or Library only updates
the registry; Console now reconciles its own session against it on the
next resume instead of staying stale until an in-Console switch, so the
Default-workspace paragraph above gained a one-sentence note on that; the
rest of this page's content unchanged from the prior stamp).*
