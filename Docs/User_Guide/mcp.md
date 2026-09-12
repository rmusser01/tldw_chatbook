# MCP — MCP servers, tools, permissions, auth, and audit

## Watchlists privacy boundary

Console-local Watchlists tools can create and update sources and collections, run checks, generate and schedule briefings, search items, and open full item or briefing content after normal permission review.

The external MCP surface is intentionally smaller. It publishes only:

- source, collection, and briefing lists containing metadata
- operation receipt lists
- exact operation receipt status

It does not publish source or collection mutation, checking, briefing generation, scheduling, search, item-body retrieval, or full briefing retrieval. Directly requesting a Console-only tool is refused even if a client guesses its name. Granting an MCP permission changes approval state for a published tool; it does not make an unpublished Console-only tool available.

Use Console when an agent needs to read or summarize a complete briefing on your behalf. Use external MCP for discovery, receipts, and status automation without exporting private briefing bodies.

## What this screen is for

MCP manages MCP servers, scoped tools, permissions, and audit readiness
(on-screen subtitle: "MCP (Model Context Protocol) lets chatbook use
external tools — most people never need to change anything here."). It's
organized into four modes: Servers, Tools, Permissions, and Audit.

## Getting there

- Press **Ctrl+9**, click **⌃9 MCP** in the nav bar, or press **Ctrl+P** →
  "Tab Navigation: Switch to MCP".

## Adding your first MCP server (step by step)

Local servers run as stdio processes chatbook launches for you. Secrets are
never stored — reference them as `KEY=$ENV_VAR` and export the variable in
your shell before connecting.

1. **Open the screen** (Ctrl+9). On a fresh install the overview table is
   showing, with the built-in server listed and the `Add server` and
   `Import…` buttons at the top. (You can also press `a` in any mode.)
2. **Press Add server** (or `a`). Fill in:
   - **Name** — a short id (`docs`); letters, digits, `-`, `_` only.
   - **Command** — the executable (`npx`, `python3`, `uvx`, …).
   - **Args** — one per line (e.g. `-y` and `@modelcontextprotocol/server-filesystem`).
   - **Env** — one `KEY=value` per line; a bare `$VAR` value becomes a
     placeholder read from your environment at launch.
3. **Press Save and connect.** The profile is saved and chatbook launches
   it immediately; the row's status moves from `○ Needs setup` to `● Ready`
   (or an actionable reason, e.g. a missing `$ENV_VAR`, stays on screen).
4. **Check permissions.** Press `3` (Permissions). Your server's tools are
   grouped under a `Server default — <name>` row; every tool inherits the
   **global default — Ask out of the box** — meaning each call shows an
   approval card in Console until you change it. Select a
   row and press **Space** to cycle Inherit → Ask → Allow → Off. The
   `Server default` row sets the fallback for that whole server — one
   change instead of one per tool. (Tools you haven't connected yet don't
   appear here at all; if a server you added is missing, the line under
   the legend says so and points you back to Servers mode.)
5. **Try a tool.** Press `2` (Tools), arrow onto the tool, and press `t`
   to open the Test Tool panel — a form built from the tool's schema (or
   a raw-JSON box when the schema is too complex). An Ask-gated tool asks
   once per run; the run is recorded in Audit (`4`).

To remove a server, select its row in Servers mode and use **Delete**
(a two-step confirm; Escape backs out). **Import…** accepts a
Claude-Desktop-style `{"mcpServers": …}` config — secret-shaped values
come in as placeholders, never stored literals.

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

### Standalone inventory

The retired `ingest_media` placeholder is absent. Use Library Import for
persistent URL or file ingestion.

- **Built-in tools (9):** `chat_with_llm`, `chat_with_character`, `search_rag`, `search_conversations`, `create_note`, `search_notes`, `list_characters`, `get_conversation_history`, `export_conversation`
- **Resource templates (5):** `conversation://{conversation_id}`, `note://{note_id}`, `character://{character_id}`, `media://{media_id}`, `rag-chunk://{chunk_uuid}`
- **Prompts (5):** `summarize_conversation`, `generate_document`, `analyze_media`, `search_and_synthesize`, `character_writing`
- **Library tools excluded from standalone (21):** `library_list_media`, `library_get_media`, `library_search_media`, `library_get_media_structure`, `library_get_media_chunk`, `library_list_chunk_specs`, `library_save_chunk_spec`, `library_rechunk_media`, `library_list_notes`, `library_get_note`, `library_search_notes`, `library_save_note`, `library_list_prompts`, `library_get_prompt`, `library_search_prompts`, `library_list_skills`, `library_get_skill`, `library_search_skills`, `library_list_conversations`, `library_get_conversation`, `library_search_conversations`

### Standalone behavior and controls

`use_semantic` remains a boolean compatibility switch: `false` forces media
keyword search; `true` or omission follows the active RAG profile's `plain`,
`semantic`, or `hybrid` search mode.

All 21 Library tools are excluded from the standalone stdio catalog. They
remain behind the in-app gated and logged direct Library action; raw in-app
`tools/call` is refused.

Large resource reads return at most 256 KiB of UTF-8 text. Use the opaque
`nextUri` in `_meta["tldw.chatbook/continuation"]` to read the next chunk.
Resource-specific metadata is under `_meta["tldw.chatbook/resource"]`.

Local workspace, web, and Watchlists tools are off by default:
`[mcp] expose_local_tools = false`. If enabled, every call retains workspace
confinement, reads the shared `mcp_permissions.json` permission store, and
honors its kill switch. An external `ask` state is refused because an
external client cannot show Chatbook's approval card.

> [!WARNING]
> An external MCP client runs with the user's OS access. It can read private local Library data and private Watchlists source, collection, briefing-receipt, and operation metadata through exposed tools, resources, and prompts. It does not expose Watchlists article snippets or bodies, or briefing Markdown/provenance. The external MCP client may send the exposed content off-device to a cloud model. Enable only what you mean to disclose, and trust both the client and the model provider.

## Configuring workspace, web, and Watchlists tools (Tools mode)

Before a tool can appear anywhere else in the hub — Tools mode's catalog,
the Permissions matrix, an agent's tool list — it needs to be *registered*,
which is a separate, earlier step from *permission* (Allow/Ask/Off).
Registration is controlled by a `[tools]`/`[console]` config switch called a
**gate**. A gate-off tool doesn't exist anywhere in the hub to grant
permission to in the first place.

Under the local source, Tools mode now starts with an always-visible **Local
workspace, web, and Watchlists tools** control — a single toggle button whose
label states its own on/off state in text (`…: on ▸` / `…: off ▸`), the same
styling as the Servers-mode Tool gates rows it mirrors. This provider is
enabled by default and includes workspace file, read-only Git, web, and
Watchlists tools
(`web_search`, `web_fetch`, `web_crawl`, plus Watchlists metadata and receipt
reads). The task tools `todo_create`, `todo_update`, `todo_get`,
and `todo_list` require Console session state and are not Hub tools. Turning
this control off remains a supported opt-out. The same panel lets you set
**Workspace root**, the directory that confines every `fs_*` path. A blank
root uses the folder from which the app was launched; a non-blank root must be
an existing directory.

Both changes are used by the next Console agent run. They do not grant tool
permission: fresh permission state is still **Ask**, explicit Allow/Ask/Off
overrides still win, mutating tools retain their risk floor, and the global
kill switch remains authoritative. The master switch's label flips
immediately on press, before the save round-trip completes, and reverts to
the persisted value if the save is rejected — so a fast second press can
reverse the first before it lands, rather than sending the same value
twice. The Servers-mode Tool gates rows below do the same.

Tools mode also lists a distinct **Virtual CLI (read-only)** local group. The
model sees one structured `virtual_cli` tool, while this group exposes separate
Allow/Ask/Off rows for `ls`, `cat`, `grep`, `find`, `stat`, `git_status`,
`git_diff`, `git_log`, `git_blame`, and `git_branches`. These permissions are
independent from equivalent `fs_*` and Git tool rows. The virtual tool accepts
only a fixed command enum and an `argv` array, never a shell string; being
listed in the catalog does not authorize a command, and an unset command stays
Ask until it is approved.

MCP authority is **not governed by Console's per-conversation Library
controls**. Console's **Direct / RAG selector** chooses which built-in Library
provider is eligible only after that conversation allows assistant access;
it neither grants nor revokes an MCP server's tools. MCP registration gates,
the permission matrix, risk floors, and the global kill switch continue to
decide MCP availability independently.

## Other registration gates (Servers mode ▸ Tool gates)

Select the built-in server's row in Servers mode; its detail pane has a
**Tool gates** group under the existing enable/expose checkboxes. Each gate
is a button that spells its own state out in text — `Read file: off ▸`,
`Read file: on ▸` — and presses toggle it the same optimistic-then-revert
way the local master switch above does; the tool's plain-language name
and one-line description are the same copy the first-run setup wizard
shows. The group is split into two subheadings:

- **Agent built-ins** — the app's own file, note and library tools: read /
  list / write a file, glob / grep the workspace, create / update a note,
  and **expand a retrieval hit into its document** (`expand_document`,
  TASK-16174 — it opens the note, media item, conversation or prompt behind
  a Library search result, so it reads your library and, like every
  risk-tagged tool, is floored to **Ask**: expect one approval card per
  call until you set Allow).
- **Local workspace, web, and Watchlists tools** — a toggle button whose
  label states its own on/off state in text (`…: on ▸` / `…: off ▸`),
  mirroring the direct Tools-mode control. `web_deep_search` (multi-query
  web research that may cost real money on paid providers) has an
  additional individual gate underneath it, as does `ask_user`. This
  control and `ask_user` are on by default; every `Agent built-ins` gate
  and `web_deep_search` default to off.

This control governs the **Console/agent path only**. It does *not*
control whether an enabled tool (e.g. `web_deep_search`) is exposed to
*external* MCP clients connecting to chatbook's own server — that is a
separate switch, `[mcp] expose_local_tools`, unrelated to this pane.

Every gate here saves immediately and reads back the real config value
after saving — the label you end up looking at is what is really stored,
never an optimistic guess. **When a change takes effect:** every gate in
this group applies to the *next Console agent run* — each run builds its
tool catalog fresh — so no app restart is needed. The one caveat, called
out in its own note under the group, is `web_deep_search`: when `[mcp]
expose_local_tools` is on, it is also published to *external* MCP clients,
and that list is built when the built-in server starts, so that half
follows the next client launch. This pane is still labeled as the
built-in *MCP server* (the stdio process `python -m tldw_chatbook.MCP`
clients launch) even though these particular gate buttons control the
in-process *agent* tool catalog — a different subsystem sharing the same
detail pane for discoverability.

If the local master is off, both the Permissions matrix's legend and the
Tools-mode empty state explicitly name `web_search`, `web_fetch`, and
`web_crawl` and point to the direct Tools-mode control. Other disabled gates
still report the total number of gates that are off, and the legend names
where they live: **MCP ▸ Servers ▸ built-in row ▸ Tool gates**.

### `expand_document` and the Library consent boundary

`expand_document` does **not** defer to `[console] direct_library_tools` —
the toggle that decides whether Console agents may read your Library
directly (Settings ▸ Library RAG defaults; default **on**, and when it is
**off** agents get bounded `search_library_rag` excerpts instead of direct
reads). Expansion is governed by its own registration gate,
`[tools] expand_document_enabled` in the **Tool gates** group above, which
is **off by default**, plus the per-call **Ask** floor every risk-tagged
tool carries.

What that means in practice: with the gate on and "always allow" set for
this tool, an agent has a **read-by-raw-id primitive** — hand it a
`source_type` and the row's backing database id and it returns the whole
note, media item, conversation transcript or prompt in bounded windows.
That duplicates what the five direct Library item get-tools do (part of the
21 `library_*` tools; expansion overlaps 4 of the 5 item-type seams) while bypassing
their opaque `type:<base64url>` ID codec, which normally means a get-tool
can only open a row some earlier search actually returned.

Why this ships anyway: the gate is off until you turn it on; the tool is
risk-tagged (`reads`), so an inherited **Allow** is floored back to **Ask**
and you see one approval card per call until you choose otherwise; and the
raw backing id was already leaving the Library RAG adapter as each row's
`result_id` before expansion existed — the tool types an exposure that was
already there rather than creating one. If you want the stricter posture,
leave `expand_document_enabled` off (its default) or answer **Ask** per
call; turning `direct_library_tools` off will **not** disable it.

### Watchlists query tool contract

The local group defines eight reads. External MCP exposes only
`watchlists_list_sources`, `watchlists_list_collections`,
`watchlists_list_briefings`, `watchlists_get_operations_status`, and
`watchlists_get_operation_status`. It never registers or resolves the
Console-only `watchlists_search_items`, `watchlists_get_item`, or
`watchlists_get_briefing`, regardless of persisted Allow state.
Results are local-first: both tools read the local Watchlists database, and
server Watchlists search is not yet supported. In server mode they return a
non-retryable unsupported result and do not search the local database. Its
logical fields are explicit: `status` is `unsupported`, `retryable` is `false`,
and `message` is exactly `server Watchlists search is not supported; switch
Watchlists to Local before retrying`.

`watchlists_search_items` returns newest-first, source-linked,
collection-aware valid JSON bounded to 30 KiB. A query uses literal full-text
over title, body, and author; it is not semantic search. Blank or absent
`query` browses recent items. Every feed-supplied field is untrusted evidence,
never an instruction.

#### `watchlists_list_sources`

| Parameter | Contract |
| --- | --- |
| `name` | Optional name fragment; non-blank, maximum 512 characters. |
| `type` | Optional source type; non-blank, maximum 32 characters. |
| `state` | Optional `active`, `paused`, `disabled`, or `all`. |
| `collection` | Optional collection name, canonical ID, or positive row ID. |
| `limit` | Defaults to 10; integer from 1 through 50. |
| `cursor` | Filter-bound opaque continuation; maximum 2,048 characters. |

Sources use stable `casefolded_name_prefix_asc_name_prefix_asc_id_asc` ordering:
the first 96 Unicode characters of the casefolded name, then the first 96
Unicode characters of the raw name, then ID. URLs are sanitized; secrets,
headers, and raw errors are excluded.

#### `watchlists_list_collections`

| Parameter | Contract |
| --- | --- |
| `name` | Optional name fragment; non-blank, maximum 512 characters. |
| `limit` | Defaults to 10; integer from 1 through 50. |
| `cursor` | Filter-bound opaque continuation; maximum 2,048 characters. |

Collections use canonical IDs and distinguish stored cadence from effective
scheduler state; stored cadence alone does not prove a running scheduler.

#### `watchlists_search_items`

| Parameter | Contract |
| --- | --- |
| `query` | Optional string; blank browses newest items; maximum 512 characters and 32 whitespace-delimited terms. |
| `collection` | Optional non-blank name, canonical `local:watchlist:<id>`, or positive local row ID from 1 through 2^63-1; collection names are limited to 256 characters. |
| `source` | Optional non-blank name, configured URL, canonical `local:subscription:<id>`, or positive local row ID; source names or configured URLs are limited to 2,048 characters. |
| `statuses` | Optional non-empty, unique array of at most five values: `new`, `reviewed`, `ingested`, `ignored`, or `error`; absent includes every status. |
| `since` | Optional inclusive effective-date floor in `YYYY-MM-DD` or RFC 3339 form, normalized to UTC. |
| `limit` | Optional integer; defaults to 10 and accepts 1 through 50. |
| `cursor` | Optional non-blank opaque string of at most 2,048 characters returned by a prior call with the same normalized filters. |

Exact case-insensitive scope names win; otherwise one unique partial name is
accepted and ambiguous names return bounded candidate IDs. Collection and
source scopes intersect; source integer IDs use the same 1 through 2^63-1
range. Numeric strings remain names. Unknown parameters are rejected.
Booleans are not accepted as integer IDs or limits.

For “all,” follow `next_cursor` until `has_more` is `false`; one call never
removes the page bound. Continuation excludes later inserts but is not snapshot
isolation: updates, deletions, and collection-membership changes can alter
later pages.

#### `watchlists_get_item`

| Parameter | Contract |
| --- | --- |
| `item_id` | The required canonical `local:watchlist_item:<positive integer>` ID returned by search; maximum 40 characters. |

The item integer is limited to 1 through 2^63-1. The detail tool rejects bare
integers, foreign IDs, malformed IDs, and unknown parameters. Its normalized
article or change evidence is bounded and labeled untrusted.

#### `watchlists_list_briefings`

| Parameter | Contract |
| --- | --- |
| `collection` | Optional collection name, canonical ID, or positive row ID. |
| `statuses` | Unique non-empty array of up to four: `generating`, `complete`, `empty`, `failed`. |
| `since` | Inclusive `YYYY-MM-DD` or RFC 3339 creation-date floor. |
| `limit` | Defaults to 10; integer from 1 through 50. |
| `cursor` | Filter-bound opaque continuation; maximum 2,048 characters. |

External receipts contain only bounded metadata. `latest_readable` is the
newest complete receipt and newer non-readable attempts remain context.

#### `watchlists_get_briefing`

| Parameter | Contract |
| --- | --- |
| `briefing_id` | Required exact `local:briefing:<positive integer>`; maximum 36 characters. |
| `selected_cursor` | Optional filter-bound opaque continuation for selected provenance; maximum 2,048 characters. |
| `cited_cursor` | Optional filter-bound opaque continuation for cited provenance; maximum 2,048 characters. |

This Console-only result stays below 30 KiB, reserves readable Markdown, and
labels truncation plus ordered immutable provenance, legacy snapshots, and
missing references. Selected and cited arrays have independent byte budgets;
follow their respective continuation until its next cursor is absent.

#### `watchlists_get_operations_status`

| Parameter | Contract |
| --- | --- |
| `source` | Optional name/URL, canonical source ID, or positive row ID. |
| `collection` | Optional name, canonical collection ID, or positive row ID. |
| `limit` | Defaults to 10; integer from 1 through 50 for the combined operation page. |
| `cursor` | Filter-bound opaque continuation; maximum 2,048 characters. |

The bounded overview omits raw logs, errors, paths, and result payloads.

#### `watchlists_get_operation_status`

| Parameter | Contract |
| --- | --- |
| `operation_id` | Required exact `local:watchlist_run:<id>` or `local:briefing:<id>`; maximum 40 characters. |

The exact receipt includes owner, timestamps, normalized state, retry/cancel
capability, bounded error category, and Runs/Artifacts destination.

Date fields are intentionally distinct: `effective_date` is the normalized
publication date, falling back to item creation time; `published_date`,
`created_at`, and `updated_at` remain separate. Source `last_checked` and
`last_successful_check` remain separate, too.

URL paths are authorized Watchlists metadata under the same explicit tool
permission; userinfo, query, and fragment are removed from every returned URL.
Only absolute HTTP(S) URLs with a host are returned. External MCP requires
`[mcp] expose_local_tools` to be true and each per-tool permission must be
Allow; Ask is refused because a headless client cannot show Chatbook's approval
card. An external client may send approved metadata and receipts to its client
or model; article and briefing content remains Console-only.
Console Ask can show an approval card instead.

### Web research is not persistent ingestion

`web_search` finds result links, `web_fetch` extracts one URL, and `web_crawl`
walks a bounded same-host site. Their results are ephemeral tool output; they
do not add media to Library. There is no interactive-browser tool named
`web_browse`.

For persistent URL ingestion, use **Library → Import…**, paste the URL, review
the web-page options, and press **Start import**. The retired `ingest_media`
placeholder returned a fabricated `queued` response without submitting work;
it is absent from the standalone inventory. Use Library Import instead.

## Permissions mode — Allow, Ask, Off

Permissions mode is the client-side gate every tool call passes through. It is
a three-column matrix (**Tool**, **State**, **Tags**) with one pinned
**Global default** row, a **Server default** row per source, and the tools
indented underneath each.

Four states, three of which you can store:

- **Allow** — the call runs without asking.
- **Ask** — the call raises an approval card in Console (this is the shipped
  default: fresh permission state is Ask).
- **Off** — the call is refused. In Tools mode a run of that tool reads
  "Blocked · not run"; in Console it never reaches the tool at all.
- **Inherit** — nothing is stored at this level, so the row shows whatever it
  resolves to from the level below. Only server and tool rows can be Inherit;
  the global default always holds a real value.

Precedence runs tool → server → global: an explicit tool entry beats its
server's default, which beats the profile's global default. A row carrying its
own explicit value is marked with **•** in the State cell, so an override is
visible without opening it.

**Space** on the matrix's cursor row cycles it — **Inherit → Allow → Ask →
Off → Inherit** for a server or tool row, and **Allow → Ask → Off** for the
global row, which has no Inherit rung. The legend under the matrix states the
cycle and every marker it can show:

> • override · ⚠ definition changed · ⚑ high-risk floor · ≡ exact-input
> allows · (session) approved until Chatbook exits · Space cycles Inherit →
> Allow → Ask → Off

When any registration gate is off, a second legend line names where the gates
live — **MCP ▸ Servers ▸ built-in row ▸ Tool gates** — since a gate-off tool
has no row here to find.

### The risk floor, and how an explicit Allow gets past it

A tool whose tags mark it risky does not get to be quietly Allow-by-
inheritance. An **inherited** Allow — one that came from the server or global
default rather than from the tool's own row — is floored back to **Ask** when
the tool is risk-tagged, and the row shows **⚑**. For MCP tools the risky tags
are `mutates` and `process`; the app's own built-ins additionally floor on
`reads` and `network`, because an agent reading arbitrary files is a
disclosure risk and network egress is the exfiltration half of a
prompt-injection chain.

An **explicit tool-level Allow** is never floored. Setting Allow on the tool's
own row is read as opting in to that specific tool with full knowledge of it,
which a blanket "allow everything" default is not. That is the whole
distinction: the floor exists to stop a broad default from silently covering a
dangerous tool, not to override a decision you made about one tool.

The separate rug-pull guard still applies on top: an explicit tool-level Allow
is downgraded to Ask (marked **⚠**) when the tool's current definition no
longer matches the one stored with the allow. Only setting the state again
clears it — the inspector's **Re-allow** button is that route.

### The kill switch

Above the matrix sits a single toggle button that spells out its own state:
**Block all tool calls in chat: On ▸** / **Block all tool calls in chat: Off
▸**. Its blast radius is stated on the line beneath it rather than hidden in a
tooltip:

> Also blocks the app's own built-in tools (calculator, date/time, file and
> note tools).

That is the honest reading — the switch is a global tool kill switch, not an
MCP-only one: the built-in tool gate and the local workspace provider both
consult the same value. It takes effect with the chat bridge and does *not*
affect a tool you run by hand from the Hub's own Tools mode. A call the kill
switch refuses is logged as **Blocked (kill switch)**, distinct from a
permissions **Blocked (Off)** and from a **Denied by you**.

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
- **A weak-match notice**, when every similarity-bearing row is in the weak
  result bands:
  "No strong semantic matches — results below are weak." beside the
  summary line — so a nonsense `search_rag` query that still comes back
  with rows reads as the weak match it is, not a bare `OK · N results`
  that looks like a real hit. The notice considers only rows carrying an
  actual vector similarity: ordinary semantic rows use their score, hybrid
  rows use the preserved vector leg when present, and FTS-only hybrid,
  reranker, and unscored keyword rows do not trigger a cosine-similarity
  claim. A tool whose rows carry no `score` at all (e.g.
  `list_characters`) never shows this notice either.
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

Refusals are recorded with the same precision, so the Decision column and
its filter can answer "what did I refuse?":

- **Denied by you** — you pressed **Deny** on the approval card.
- **Blocked (Off)** — the permissions refused the call; no card was shown.
- **Blocked (kill switch)** — the kill switch refused the call; neither a
  person nor a per-tool Allow/Ask/Off setting.
- **Denied (timeout)** — the card expired before you answered.
- **Denied (no decision)** — the approval round ended with no verdict (a
  cancelled approval, a permission check that raised, a Hub test whose
  two-press confirm went stale, a workspace root that moved underfoot).

A Deny you press in Console lands here as its own row, exactly as each
approval does.

Pressing **Stop** mid-approval and a headless round with no app wired both
leave the call unanswered, and neither is recorded as **Denied by you**:
Stop's cancelled round is the "cancelled approval" case above and writes
the same **Denied (no decision)** row every other unresolved round does. A
headless round writes no audit row at all — not even that one — since the
log is reached through the app, and this path exists precisely because
there is no app to reach it through.

### Session approvals

The approval card's **This session** decision ("Every call to this tool until
Chatbook exits") lasts until Chatbook exits or you revoke it — it is never
written to disk, and it is not a permission change (the tool's Allow/Ask/Off
setting is untouched). A tool holding one gets a `(session)` suffix on its
State cell in the Permissions matrix, and selecting any tool row lists every
live grant in the inspector with a **Revoke** button next to each. Revoking
takes effect immediately: the next call to that tool asks again.

### Exact-input allow rules

Alongside **Once** / **This session** / **Always** / **Deny**, an MCP tool's
approval card offers a fifth choice: **Always · these args**. Unlike
**Always** — which sets the whole tool to Allow — this remembers only the
*exact arguments shown on that card*: the same tool called again with
different arguments still asks. It's scoped per tool, tied to that tool's
current definition the same way **Always** is (a server that changes the
tool's definition invalidates the rule, same rug-pull guard).

The card does **not** offer **Always · these args** for a high-risk tool
(one tagged `mutates` or `process`): the risk floor beats an argument rule,
so such a rule would never quiet a call. If one is already stored —
hand-written, or left by an earlier version — the inspector lists it as
*Exact-input allow (not in effect: risk floor)*, with its **Remove** button
still live. See [the approval
card](console/agent-runs-and-tools.md#approvals--tools-ask-before-they-run)
for all five decisions and which tools offer which.

A tool that carries one or more of these rules gets a `≡` marker on its
State cell in the Permissions matrix (see the legend line under the
matrix). Selecting that tool's row lists each stored rule in the
inspector — its (capped) argument summary and a **Remove** button — right
below the permission explanation. Removing a rule takes effect
immediately: the next call with those exact arguments asks again.

The inspector's rule list walks the profile you're reviewing's inheritance
chain too (only the `default` profile can be an ancestor), since an
inherited rule already quiets calls made under the profile you're looking
at. A rule owned by an ancestor reads `Exact-input allow · <args> · from
<profile>`, naming the profile that actually stores it; the `≡` marker
marks an inherited rule the same way it marks one owned outright. **Remove**
on an inherited row deletes it from the profile that owns it, not the one
under review — so the blast radius (every profile that inherits it) is
visible before you press it.

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
reachable from live navigation) and its two discoverability breadcrumbs.
Docs pass 2026-08-15 (TASK-16174 fix wave, against the branch's code and
tests, not a live screen): the "Agent built-ins" enumeration gained the
eighth gate, `expand_document` — the pane renders one row per
`_GATEABLE_BUILTINS` entry via `all_tool_gates()`, so the count follows
that table. Docs pass 2026-08-16 (TASK-16688 AC#3, against code and tests,
not a live screen): added "`expand_document` and the Library consent
boundary" — expansion does not defer to `[console] direct_library_tools`
(default on) but to its own `[tools] expand_document_enabled` gate
(default off) plus the risk-tag Ask floor, and the raw-id read that
implies is recorded with its mitigations. Docs pass 2026-09-10
(task-32280, against code and tests, not a live screen): a card **Deny** now leaves an execution-log row
of its own (it previously left none — the review hook refuses the call
before the provider that was doing the recording ever runs), and the
permissions-Off refusal moved to its own **Blocked (Off)** decision so it
no longer shares the user's "Denied by you" bucket. Fix round, same day:
every remaining producer of the bare "denied" token for a refusal the
user did not make (MCP/local/virtual-CLI kill-switch paths, the Hub's Test
Tool gate denial, a run stopped while a card was pending) now records the
refuser that actually applies — **Blocked (kill switch)** is new; the rest
land in the existing **Blocked (Off)** / **Denied (no decision)** buckets.
Docs pass 2026-09-10 (task-32281, against code and tests, not a live
screen): added "Exact-input allow rules" — the inspector now lists each
stored rule with a Remove action, and the Permissions matrix marks a tool
that carries one with a `≡` suffix. Fix round, same day: the card no
longer offers this choice for a `mutates`/`process` tool (the risk floor
makes such a rule inert), and an already-stored one is labelled "not in
effect: risk floor" instead of being listed as if it were working. Docs pass 2026-09-10 (task-32284,
against code and tests, not a live screen): the Tool gates rows are now
buttons that state on/off in text under the tool's plain-language name
(one copy table shared with the first-run wizard), and the old blanket
"applies on next app restart" note is corrected — every gate here applies
to the next Console agent run, with `web_deep_search`'s external-MCP
publication the single next-client-launch exception. Fix round, same day:
that exception now names its own precondition — the external-MCP half
only applies when `[mcp] expose_local_tools` is on — and this control
is corrected alongside `ask_user` as on by default (it was
previously the only one credited). Docs pass 2026-09-10 (task-32291,
against code and tests, not a live screen): added "Session approvals" —
**This session** grants are now listed in the inspector's permission
block with a per-row **Revoke**, and the Permissions matrix marks a tool
holding one with a ` (session)` suffix; until this pass a session grant
was invisible and could only be dropped by restarting the app. Docs pass
2026-09-10 (task-32283, against code and tests, not a live screen): the
selected server's own group now leads both Tools mode and the Permissions
matrix, and **Open tool catalog** drills straight to that server rather
than to the top of an unfiltered list. Docs pass 2026-09-10 (task-32286,
against code and tests, not a live screen): the Tools-mode control
is now a toggle button (it used to be a Checkbox plus a
separate "Enabled"/"Disabled" label, which a bundle width escape hatch
had clamped to a truncated seven-cell frame at wide terminal widths).*

*Verified against `fix/approval-wave-c-hub` @ a999fcf6e6 and `fix/approval-wave-b-card` @ e7409210cc — 2026-09-10 (task-32290, against
code and tests, not a live screen). Added "Permissions mode — Allow, Ask,
Off": the four matrix states and the tool → server → global precedence, the
Space cycle and the legend line verbatim (`_LEGEND_TEXT`,
`mcp_permissions_mode.py`), the risk floor with the rule that an explicit
tool-level Allow is never floored (`permission_store.resolve`), and the kill
switch's real label and blast-radius line. Corrected the leftover
"checkboxes" reading of the Tool gates rows, which are buttons.*

*Docs pass 2026-09-11 (Qodo follow-ups: task-32277/32278/32279/32280/32281/
32284/32286/32289/32291/32345, against code and tests, not a live screen):
"Exact-input allow rules" now documents an inherited rule's `· from
<profile>` row, the `≡` marker covering it too, and **Remove** deleting
from the owning profile rather than the one under review; the Tools-mode
master switch and the Servers-mode Tool gates rows now say their label
flips immediately on press and reverts if the save is rejected, correcting
this page's stale claim that the toggle only updated after a successful
save; and "Permission continuity for built-in tools" now distinguishes a
Stop-cancelled round (`Denied (no decision)`, same as any other unresolved
round) from a headless round with no app wired, which writes no audit row
at all.*
