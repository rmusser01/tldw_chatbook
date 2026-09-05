# MCP — MCP servers, tools, permissions, auth, and audit

## Watchlists privacy boundary

Console-local Watchlists tools can create and update sources and collections, run checks, generate and schedule briefings, search items, and open full item or briefing content after normal permission review.

The external MCP surface is intentionally smaller. It publishes only:

- source, collection, and briefing lists containing metadata
- operation receipt lists
- exact operation receipt status

It does not publish source or collection mutation, checking, briefing generation, scheduling, search, item-body retrieval, or full briefing retrieval. Directly requesting a Console-only tool is refused even if a client guesses its name. Granting an MCP permission changes approval state for a published tool; it does not make an unpublished Console-only tool available.

Use Console when an agent needs to read or summarize a complete briefing on your behalf. Use external MCP for discovery, receipts, and status automation without exporting private briefing bodies.

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
workspace, web, and Watchlists tools** control. This provider is enabled by
default and includes workspace file, read-only Git, web, and Watchlists tools
(`web_search`, `web_fetch`, `web_crawl`, plus Watchlists metadata and receipt
reads). The task tools `todo_create`, `todo_update`, `todo_get`,
and `todo_list` require Console session state and are not Hub tools. Turning
the master switch off remains a supported opt-out. The same panel lets you set
**Workspace root**, the directory that confines every `fs_*` path. A blank
root uses the folder from which the app was launched; a non-blank root must be
an existing directory.

Both changes are used by the next Console agent run. They do not grant tool
permission: fresh permission state is still **Ask**, explicit Allow/Ask/Off
overrides still win, mutating tools retain their risk floor, and the global
kill switch remains authoritative. The controls read back persisted config
truth after saving, and a failed save restores the persisted value instead of
leaving an optimistic toggle on screen.

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
**Tool gates** group under the existing enable/expose checkboxes, split
into two subheadings:

- **Agent built-ins** — the app's own file, note and library tools: read /
  list / write a file, glob / grep the workspace, create / update a note,
  and **expand a retrieval hit into its document** (`expand_document`,
  TASK-16174 — it opens the note, media item, conversation or prompt behind
  a Library search result, so it reads your library and, like every
  risk-tagged tool, is floored to **Ask**: expect one approval card per
  call until you set Allow).
- **Local workspace, web, and Watchlists tools** — a master switch, labeled
  **Local workspace, web, and Watchlists tools (master switch)**, mirroring the
  direct Tools-mode control. `web_deep_search` (multi-query web research that
  may cost real money on paid providers) has an additional individual gate
  underneath it. Unlike the local master and workspace root,
  construction-time gates such as `web_deep_search` require an app restart.

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
That duplicates what the direct Library get-tools do (6 of the 18
`library_*` tools; expansion overlaps 4 of the 6 seams) while bypassing
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
implies is recorded with its mitigations.*
