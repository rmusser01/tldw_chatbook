# Workflows — Local definition authoring

The Workflows destination creates and edits portable JSON definitions locally.
**Run is disabled in this release.** Saving or validating a definition does not
execute steps, call a model, publish to a server, or write a Note. Sequential
execution belongs to v1; branching is v2 and parallel execution is v3.

## Open and author a workflow

Press **Ctrl+8**, select **Workflows** in the navigation bar, or use **Ctrl+P** →
"Tab Navigation: Switch to Workflows".

1. Select **New workflow**, enter a name, and press Enter. On narrower terminals,
   use **Workflow…** to open the library choices, including New workflow.
2. Use the library to select a definition. The navigator selects Overview,
   Inputs, Requirements, or a step. **Add step** opens the searchable action
   catalog; **Show all** also explains unavailable step types.
3. Edit the continuous form. Its sections expand independently; opening one does
   not close another. Previous/next step controls change the selected step.
   **Expand** gives a large text field more room; Escape returns to the same field.
4. **Validate** checks local structure and can take you to an issue. Edits update
   validation, step labels and section summaries without rebuilding the active
   field or moving your typing cursor. Validation success appears separately
   from saved/draft status; it does not mean the draft was saved or executed.
5. **Save revision** creates an immutable saved definition. Autosaved drafts are
   separate: unfinished or invalid JSON can survive navigation and restart without
   becoming a valid saved revision. **More…** exposes history, duplication,
   reordering and draft recovery actions where applicable.

At wide sizes the library, navigator and editor are visible together. Medium
widths retain the navigator and editor; compact widths use **Workflow…** and
**Step / Overview…** selectors above the editor. **F6** moves between
visible workbench panes; Tab moves through controls. Hidden panes are skipped.

The library, saved-revision history and local-draft selectors show 20 items per
page. Use **Previous**/**Next** in the library, or **Previous page**/**Next page**
in selectors, to reach the rest. Library search matches names across all pages,
including case-insensitive Unicode text. Paging or searching does not change
your open draft. Pages reflect the current store, not a frozen snapshot.
Search accepts up to 512 characters. If an oversized search cannot load, shorten
the query to retry; the open draft is retained.

## Draft recovery

The status line says **Draft stored · not a saved revision** for durable edits
that differ from their saved base, and **Saved revision · draft unchanged** when
the draft matches that revision. If a write fails, the in-memory buffer remains
available, navigation/quit are refused when needed, and **Retry** attempts
persistence again. Do not force-close the process while a draft says it is not
saved.

**Retry** appears for failed loading or draft persistence, not validation success
or malformed imports. Successful workflow/revision navigation clears transient
operation errors and shows the destination's saved, draft or read-only status.

Invalid field JSON remains repairable while forms show the last valid document.
**Advanced JSON** exposes the whole definition, including fields the form does
not understand. Recovery confirmations explain whether they discard a draft,
accept repaired raw JSON, or copy an older draft onto the saved head. Copying is
not a merge; conflicting drafts and saved revisions are preserved.

## Import and export

**Import** opens the existing local file picker for one UTF-8 `.json` file, up to
16 MiB. Import creates/selects a local saved definition; it never executes it.
Local authoring accepts up to 500 steps, 64 levels of container nesting and
100,000 JSON values/containers, including opaque fields. These are Chatbook
editor limits, not claims about server limits. Over-complex raw edits remain
recoverable while forms retain the prior valid structure.
Older saved definitions above these limits remain available for read-only raw
inspection and exact export; they are not silently reduced or rewritten.
Stable step IDs and opaque envelope/metadata fields survive an edit to another
field and a save/export round trip.

**Export** first asks you to review the definition, then opens the save picker.
It exports the selected saved revision (or the historical revision being
inspected), not pending draft text or local execution state. Choose a separate
`.json` file. Workflows asks for confirmation before replacing an existing file.

Prompts, input values, and opaque fields can contain secrets. Preservation is
not a secrets scan: review Advanced JSON before sharing. File exchange uses the
existing private-path policy: symlinks and unsupported file types are refused,
an imported local file is hardened to owner-only access, and exported files are
written atomically with owner-only permissions.

Choose a standalone JSON file. Files with multiple hard links and aliases of
the active workflow database or its sidecars are refused before generic file
access. V1 assumes stable files: do not externally move, replace or relink the
live workflow database or its sidecars while Workflows storage is open. During
import/export, also keep the selected JSON file and its containing path unchanged.
Normal database edits through the app remain supported. This metadata precheck
does not prevent concurrent pathname substitution; a substituted database alias
can still disrupt SQLite locking. Quit the app before relocating its database.

## Storage and compatibility

Storage initializes only on first entry to Workflows. The default file is
`tldw_chatbook_workflows.db` in the active profile's user-data directory; the
`[database] workflows_db_path` setting can select another private local path.
Drafts and revisions persist in this SQLite file. The authoring store is not
registered with centralized backup in this slice; use Export for a portable
saved definition. Historical v1–v4 database migrations remain compatible; their
old runtime tables are not an execution interface.

Valid JSON, local validation, and successful Save are **not server validation**.
There is no publish or sync action. A future server publication path must account
for the server's `StepConfig` / `WorkflowDefinitionCreate` Pydantic handling:
unknown fields may be dropped by parsing followed by `model_dump()`. See the
[bounded compatibility review](../superpowers/specs/2026-09-14-workflows-authoring-compatibility.md).

A secondary **Open in Console** strip can follow an existing workflow item
exposed by the current Console/Home services. It remains disabled when no such
item exists. Refreshing this strip does not replace the editor or its draft.

## Threat-intelligence news briefing in Console

This separate Console procedure stays inside the core Watchlists and briefing
product; it is not execution of a definition from the authoring editor:

1. Give the Console agent the RSS or Atom URLs and a Watchlist name.
2. Approve the specific local Watchlists tools the agent needs.
3. Have the agent create the sources and Watchlist, then follow every source-check receipt.
4. Have it generate a briefing, follow the briefing receipt, and save an every-24-hours schedule.
5. Read the completed briefing yourself, or ask the agent to open and summarize it for you.
6. Verify the resulting Watchlist and scheduled job in their dedicated views when you need cross-surface confirmation.

The Console agent can consume full briefing content; external MCP clients are
restricted to metadata and receipt status. “Existing model” means the persisted
collection preset, then persisted `chat_defaults`, then the saved model for that
same persisted provider—not the active conversation model.

Threat-hunt hypothesis or document creation is deliberately outside this workflow. Export or hand off content only as a separate, user-directed activity.
