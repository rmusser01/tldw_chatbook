# MCP — MCP servers, tools, permissions, auth, and audit

> 🚧 **This page is a stub.** The full write-up is planned; the sections
> below cover orientation only, except where a section says otherwise. See
> the [guide index](index.md).

## What this screen is for

MCP manages MCP servers, scoped tools, permissions, and audit readiness
(on-screen subtitle: "MCP (Model Context Protocol) lets chatbook use
external tools — most people never need to change anything here."). It's
organized into four modes: Servers, Tools, Permissions, and Audit.

## Getting there

- Press **Ctrl+9**, click **⌃9 MCP** in the nav bar, or press **Ctrl+P** →
  "Tab Navigation: Switch to MCP".

## Running Chatbook as a standalone MCP server

Install the packaged optional extra, then configure the external client to
launch the module from that same Python environment:

```bash
pip install "tldw_chatbook[mcp]"
python -m tldw_chatbook.MCP
```

The stdio server supports revision `2025-03-26`, revision `2025-11-25`, and
the current `2026-07-28` profile. Batch requests are accepted only with
`2025-03-26`; `2025-11-25` and `2026-07-28` reject them.

Its standalone catalog has exactly 10 built-in tools: `chat_with_llm`,
`chat_with_character`, `search_rag`, `search_conversations`, `create_note`,
`search_notes`, `list_characters`, `get_conversation_history`,
`export_conversation`, and `ingest_media`. It has exactly 5 resource
templates: `conversation://{conversation_id}`, `note://{note_id}`,
`character://{character_id}`, `media://{media_id}`, and
`rag-chunk://{chunk_uuid}`. It has exactly 5 prompts:
`summarize_conversation`, `generate_document`, `analyze_media`,
`search_and_synthesize`, and `character_writing`.

Chatbook's separate in-app Library runtime owns `library_list_media`,
`library_get_media`, `library_search_media`, `library_list_notes`,
`library_get_note`, `library_search_notes`, `library_list_prompts`,
`library_get_prompt`, `library_search_prompts`, `library_list_skills`,
`library_get_skill`, `library_search_skills`, `library_list_conversations`,
`library_get_conversation`, `library_search_conversations`,
`library_list_collections`, `library_get_collection`, and
`library_search_collections`. All 18 are excluded from the standalone stdio
catalog. They remain behind the in-app gated and logged direct Library action;
raw in-app `tools/call` is refused.

Large resource reads return at most 256 KiB of UTF-8 text. Use the opaque
`nextUri` in `_meta["tldw.chatbook/continuation"]` to read the next chunk.
Resource-specific metadata is under `_meta["tldw.chatbook/resource"]`.

Local filesystem, git, and web tools are off by default:
`[mcp] expose_local_tools = false`. If enabled, every call retains workspace
confinement, reads the shared `mcp_permissions.json` permission store, and
honors its kill switch. An external `ask` state is refused because an
external client cannot show Chatbook's approval card.

> [!WARNING]
> An external MCP client runs with the user's OS access. It can read private local Library content through exposed tools, resources, and prompts, and it may send that content off-device to a cloud model. Enable only what you mean to disclose, and trust both the client and the model provider.

## Configuring local and web tools (Tools mode)

Before a tool can appear anywhere else in the hub — Tools mode's catalog,
the Permissions matrix, an agent's tool list — it needs to be *registered*,
which is a separate, earlier step from *permission* (Allow/Ask/Off).
Registration is controlled by a `[tools]`/`[console]` config switch called a
**gate**. A gate-off tool doesn't exist anywhere in the hub to grant
permission to in the first place.

Under the local source, Tools mode now starts with an always-visible **Local
workspace + web tools** control. This provider is enabled by default and
includes `web_search`, `web_fetch`, and `web_crawl` alongside workspace file,
read-only Git, and session-todo tools. Turning the master switch off remains a
supported opt-out. The same panel lets you set **Workspace root**, the directory
that confines every `fs_*` path. A blank root uses the folder from which the app
was launched; a non-blank root must be an existing directory.

Both changes are used by the next Console agent run. They do not grant tool
permission: fresh permission state is still **Ask**, explicit Allow/Ask/Off
overrides still win, mutating tools retain their risk floor, and the global
kill switch remains authoritative. The controls read back persisted config
truth after saving, and a failed save restores the persisted value instead of
leaving an optimistic toggle on screen.

## Other registration gates (Servers mode ▸ Tool gates)

Select the built-in server's row in Servers mode; its detail pane has a
**Tool gates** group under the existing enable/expose checkboxes, split
into two subheadings:

- **Agent built-ins** — the app's own file/note tools (read/list/write a
  file, glob/grep the workspace, create/update a note).
- **Local workspace + web tools** — a master switch, labeled **Local
  workspace + web tools (master switch)**, mirroring the direct Tools-mode
  control. `web_deep_search` (multi-query web research that may cost real
  money on paid providers) has an additional individual gate underneath it.
  Unlike the local master and workspace root, construction-time gates such as
  `web_deep_search` require an app restart.

The master switch governs the **Console/agent path only**. It does *not*
control whether an enabled tool (e.g. `web_deep_search`) is exposed to
*external* MCP clients connecting to chatbook's own server — that is a
separate switch, `[mcp] expose_local_tools`, unrelated to this pane.

Every checkbox here saves immediately and reads back the real config value
after saving — never an optimistic guess. The pane's restart note applies to
the construction-time registration gates. This pane is still labeled as the
built-in *MCP server* (the stdio process `python -m tldw_chatbook.MCP`
clients launch) even though these particular checkboxes control the
in-process *agent* tool catalog — a different subsystem sharing the same
detail pane for discoverability.

If the local master is off, both the Permissions matrix's legend and the
Tools-mode empty state explicitly name `web_search`, `web_fetch`, and
`web_crawl` and point to the direct Tools-mode control. Other disabled gates
still report the total number of gates that are off.

### Web research is not persistent ingestion

`web_search` finds result links, `web_fetch` extracts one URL, and `web_crawl`
walks a bounded same-host site. Their results are ephemeral tool output; they
do not add media to Library. There is no interactive-browser tool named
`web_browse`.

For persistent URL ingestion, use **Library → Import…**, paste the URL, review
the web-page options, and press **Start import**. The old MCP `ingest_media`
entry was an unimplemented placeholder that returned a fabricated `queued`
response without submitting any work; it is no longer advertised.

## Testing a tool (Tools mode)

Tools mode lists every tool the hub knows about — the app's own built-in
tools plus anything discovered from a connected server — with a Schema
column reading "form" or "raw" so you know before selecting whether a tool
gets a typed form. Selecting a row opens its detail in the inspector on
the right; when the tool is executable, a **Test Tool** button opens a
panel to run it with arguments you choose.

Selecting a tool hides the readiness badge that normally sits at the top
of the inspector (the "Pick a server, tool, or entry…" placeholder, or a
selected server's readiness state) — that badge belongs to server
selection, not tool detail, and reappears once you clear the tool
selection.

### Typed forms

Every one of the app's built-in tools, and any server tool with a
straightforward JSON-Schema, renders as a real form instead of a raw JSON
textarea: text/number inputs with defaults already filled in, a labeled,
clickable checkbox for each boolean (the toggle glyph is invisible against
the panel when off and colored when on — not just a bare empty box),
dropdowns for enums, and a comma-separated text input for a simple list
parameter (e.g. `a, b, c`). A field marked `*` is required.

A schema the form can't represent faithfully — a nested object, a real
mixed-type union, an array of non-simple items — falls back to a raw JSON
textarea for the whole tool instead of silently dropping a parameter it
can't render: "This tool's parameters can't be rendered as a form — edit
raw JSON." The tool can still be tested either way.

### Running it and reading the result

Press **Run**. If the tool is set to **Ask** in Permissions, the first
press arms the button into **Confirm run** ("Ask is set for this tool —
press again to run once.") instead of dispatching — press it again to run,
or do anything else to cancel. A tool set to **Off** never runs at all:
the result reads "Blocked · not run", with "Blocked — this tool is set to
Off in Permissions." underneath.

A completed run shows:

- **A summary line** — e.g. `OK · local · 981ms · 3 results` (outcome,
  where it ran, how long it took, and how many results came back), or
  `Failed · 1.2s` when the call itself failed.
- **A quiet note** underneath, when there's something worth adding to the
  summary: "The tool ran and returned no results." for an empty result,
  the tool's own error text when it reported one, and — alongside either,
  or on its own — a line naming *why* the run was allowed to happen, e.g.
  "Ran because you approved this run (the tool is set to Ask)." or "Ran
  because this tool is set to Allow. Inherited from the global default."
- **A weak-match notice**, when every scored row in the result bands weak:
  "No strong semantic matches — results below are weak." beside the
  summary line — so a nonsense `search_rag` query that still comes back
  with rows reads as the weak match it is, not a bare `OK · N results`
  that looks like a real hit. Keyword-mode rows carry no score at all
  (FTS relevance is misleading, so no band beats a wrong one) — the notice
  only ever fires for scored (semantic) rows, never for a pure
  keyword-mode result, whatever it finds. A tool whose rows carry no
  `score` at all (e.g. `list_characters`) never shows this notice either.
- **A collapsed "Raw response" section** with the full result as JSON —
  secrets redacted, capped at 20,000 characters — for whenever the summary
  isn't enough.

A run that finished but can't be shown where you're looking — you closed
the panel, picked a different tool, or switched to Audit mode while it was
still in flight — still tells you: a toast reads "\<tool\> finished
running, but its result isn't shown here." A run that never reached the
tool at all — a hard **Off** gate, a runtime-governance denial, or the
Advanced panel's own refusal (below) — always reads "Blocked · not run",
never "Failed"; "Failed" is reserved for a call that genuinely reached the
tool and came back an error.

### Permission continuity for built-in tools

An **Allow**/**Ask** choice you've made for one of the app's own built-in
tools survives app updates. A server tool's allow is re-checked against a
stored fingerprint of its description/schema (so a server that quietly
changes what a tool does drops back to Ask) — built-in tools skip that
check entirely, since an ordinary app update that only edits a docstring
must never silently turn your "Allow" back into "Ask".

In Audit mode, a run you confirmed under an Ask gate is recorded with the
decision **approved**, distinct from **allowed** (a tool already set to
Allow) — so the log shows not just that a call reached the tool, but
whether it needed your confirmation first. The table repopulates as soon
as a run finishes — no need to press **r** — and each row records the
argument *names* the run supplied (e.g. `query`, `limit`, `use_semantic`),
never the values.

## Advanced (legacy control plane)

Opt in from the inspector's **Advanced…** toggle (it persists across
sessions; **Hide advanced** reverses it). The `tool.execute` action there
runs a tool directly: build a JSON payload naming the tool and its
arguments, pick `tool.execute` from the action list, and press **Run
Action**. It goes through the same
permission gate and the same execution log as every other tool run —
a tool set to **Off** is refused ("Blocked · not run"), and the refusal is
recorded in Audit mode just like any other blocked run.

Because this route resolves permissions by key rather than a stored
definition hash, almost everything it touches needs a per-run confirm —
so **Run Action takes two presses**: the first states what will run
("Runs \<tool\> now — press Run Action again to confirm. Editing anything
cancels."), the second runs it. Editing the payload or switching the
action between the two presses cancels the arm, so a stale confirm can
never fire against different arguments than the ones you read.

Raw `runtime.request` and `runtime.batch` payloads that try to execute a
tool directly — a JSON body shaped like `{"method": "tools/call"}` — are
refused and pointed at `tool.execute` instead. Those two actions are for
inspecting the protocol (`tools/list`, `prompts/list`, `status/get`), not
a second, ungated way to run a tool.

## Running more than one copy of the app

Launching a second copy of the app against the same profile never blocks
either one — both keep working. The second instance gets a one-time
warning toast, "Profile already open": whichever instance last changes a
setting or a permission wins, and a restart sweep may mark the other
instance's still-running jobs as interrupted. Detection is an advisory
lock file (`.instance.lock`, inside the profile's data directory) that's
never deleted once created — safe to ignore if you notice it.

---

*Verified against a953e4c1e — 2026-08-04 (PR-5 live check). Verified
against 9f90e17b8 — 2026-08-06 (PR-T3, docs pass against shipped
code/tests: weak-match notice, always-reports toasts, live Audit
refresh + argument names, the gated/logged Advanced panel, readable
boolean fields — live check pending Task 9). Fix round I, 2026-08-06:
"do anything else to cancel" on the Test Tool confirm now genuinely
covers editing the argument form (it previously did not), and a
background section load can no longer cancel an Advanced confirm you
armed while it loaded — only your own actions cancel, as written.
2026-08-07 (tasks 2740/2270/2870): opening Test Tool on a tool with
only-checkbox or no arguments no longer crashes the app; the "Pick a
server, tool, or entry…" placeholder now clears for Permissions-row,
Audit-entry, and Finding detail exactly as it does for tool detail; and
a permission the app could not read shows as "Unknown" (never a false
"Off") in the matrix, the State column, and the inspector alike.
Verified against ee68f42ed — 2026-08-08 (task-3240): documented the new
Servers-mode "Tool gates" group (builtin registration switches, at last
reachable from live navigation) and its two discoverability breadcrumbs.*
