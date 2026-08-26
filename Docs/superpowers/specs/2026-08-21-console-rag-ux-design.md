# Console RAG search: an explicit, per-conversation retrieval mode

**Status:** design approved in brainstorming, 2026-08-21. Sections 1 and 2
approved explicitly; section 3 recorded here for review against this file.

## The problem

Console can consult a user's Library in three ways. It says so in none of
them.

A user may press **Search Library**, which stages retrieved evidence for the
next send. A switch buried in a modal, `rag_auto_retrieve_on_send`, will do
the same automatically on every send, using the draft as the query. And the
model itself may call a Library tool, which `[console].direct_library_tools`
registers by default whenever tools run at all.

These are not variations of one feature. They are three mechanisms with
different owners — the user, the application, the model — and nothing on
screen distinguishes them. A fourth path, `get_rag_context_for_chat`, is
dead: every reference to it outside its own definition is a docstring.

Four specific faults follow.

**The chip misreports itself.** It reads `Library search: on|off`, which
looks like a switch. It is a status: "on" means evidence is staged. Nothing
can be toggled there.

**The query is not the message.** A separate input exists, backed by
`_console_draft_looks_like_rag_query`, a heuristic that guesses whether a
draft counts as a query. Software that guesses intent creates ambiguity
rather than removing it.

**Automatic retrieval is global.** `chat_defaults.rag_auto_retrieve_on_send`
governs every conversation at once, indefinitely. A user who enables it for
research finds unrelated chats quoting their files weeks later, with no
indication why. This is the most consequential fault, and the reason the
design changed course mid-brainstorm.

**Review is technically present but practically unusable.** The staged tray
renders three to four rows per source — evidence source, authority, status,
snippet — inside a box capped at ten rows. A five-source retrieval produces
roughly 22 rows. Two sources are visible without scrolling. The resulting
count lie ("Sources 18") was fixed previously; the density was not.

## What the codebase already decided

The remedy is mostly a matter of consistency, not new machinery.

| concern | storage | per-conversation |
|---|---|---|
| RAG **scope** — *what* to search | `conversations.metadata["rag_scope"]`, schema v20 | yes |
| Context policy — budget, compaction | `console_conversation_context_policy` | yes |
| Chat dictionaries | `conversations.metadata["active_dictionaries"]` | yes |
| RAG **mode** — *whether* to search | `chat_defaults` | **no** |
| Agent Library tool | `[console].direct_library_tools`, default true | **no** |

Console already treats *what* to retrieve as a property of a conversation.
Only *whether* to retrieve is global — and that is the more dangerous half.
Scope narrows the blast radius; mode decides whether there is a blast.

`ConsoleContextPolicyOverrides` supplies the pattern verbatim: sparse
per-conversation overrides in which `None` means inherit.

## Design

### 1. Mode

Three modes, each backed by an existing mechanism, stored per conversation
in `conversations.metadata` beside `rag_scope`, through the
`read_/write_conversation_scope` seam. Scope and mode are one concern and
should travel together, including across sync; the context-policy table is
deliberately local-only and therefore unsuitable.

| mode | behaviour |
|---|---|
| **Off** | nothing retrieves before a send |
| **Manual** | Search Library stages evidence when pressed |
| **Auto** | the draft retrieves on every plain-text send |

Manual retrieval remains available in all three modes. **Off** therefore
means no *automatic* retrieval, not an unreachable Library.

The global setting survives as the default for conversations that have never
chosen. Changing it never reaches into a conversation that has.

`_console_draft_looks_like_rag_query` is removed. In Manual and Auto the
draft is the query; the modal's input becomes an explicit override.

### 2. The agent's tool is governed by the mode

**Off withholds the Library provider from the model for that conversation.**

The alternative — disclosure without control — was rejected. The complaint
this design answers is that chats include library material the user did not
ask for, and the model's own tool does precisely that, on by default. A mode
that leaves it running would not mean what it says.

The seam exists: `tool_configuration["direct_library_tools"]` is already
captured per turn into `ConsoleTurnExecutionContext`, and
`_library_provider_for_context(turn_context)` already receives it. Only its
source is global.

When a Library tool *is* registered, the mode picker carries a quiet second
line — *the assistant can also search your Library on its own* — so the
capability is disclosed where the decision is made.

### 3. Review surface

**One row per source**, status as a glyph, authority and snippet behind
expansion:

```
✓ Q3 planning notes            note
✓ turbine-maintenance-log      media
⚠ vendor contract (stale)      note
✓ Q3 planning notes            note      · assistant
```

Five sources then fit the ten-row box. `Collapsible` is already used in
`console_conversation_inspector.py`; this follows that precedent.

Retrievals the model performs itself appear in the same tray, marked as its
doing. Tool calls are already captured in `capture.response["tool_calls"]`,
so this is routing, not new plumbing. It closes the case where retrieval
happens without the user's involvement and leaves no trace outside a
separate inspector tab.

**Two chips, two jobs.** The Library chip shows mode; the Sources chip shows
the staged count. They are not merged: `staged_source_count` counts all
staged context, including "Use in Console" handoffs, so folding it into the
Library chip would credit RAG with sources it did not retrieve.

**Placement.** The chip is the always-visible control. The rail's existing
`console-retrieval-scope-row` gains the mode beside the scope. No new
surface, and no cost to the recently de-cluttered bottom stack.

### 4. Persistence

The lifecycle copies context policy's three steps.

1. **Stage.** Setting a mode returns an honest `(session, persisted)` pair.
   An empty tab is never turned into a conversation row by a switch.
2. **Flush on first persist,** once the row exists.
3. **Hydrate on resume,** with failure recorded on the session rather than
   raised.

Absent keys inherit. No migration; no conversation changes behaviour because
a global default moved.

### 5. The write must fail loudly

`write_conversation_scope` fails soft when a conversation's metadata is
corrupt: the write is skipped and a warning logged. For scope this is
prudent — the alternative risks erasing `active_dictionaries`. For mode it
is close to disastrous. A user selects **Off**, is told nothing, and the
choice evaporates; on resume the conversation inherits **Auto** and the
Library is read anyway.

This design exists to stop controls asserting outcomes they did not produce.
Routed through that seam unchanged, the storage layer would reintroduce that
exact fault beneath the switch meant to cure it. **The mode write must fail
loudly: revert in the interface and say so.**

## Testing

- A conversation's mode overrides the global default; `None` inherits;
  changing the global does not disturb a conversation that has chosen.
- **Off withholds the Library provider** in `_library_provider_for_context`
  — mutation-verified, as this is the promise users rely on.
- A corrupt-metadata write reverts and reports; it is never silently
  accepted.
- Setting a mode on an empty tab persists nothing (`persisted=False`).
- The tray renders one row per source, snippet on expand, agent-initiated
  rows marked.
- The retrieval gate holds at `105 metric(s)`, +0.000. This work touches
  interface and storage only, so movement would signal a defect.

## Out of scope

- Unifying staged evidence with file attachments. It is the cleanest mental
  model but requires per-source send granularity, which the current code
  deliberately declines to advertise: *"the bundle is staged, prompted, and
  captured as ONE unit, so a partial un-stage would advertise a granularity
  the send path does not have."*
- Removing the dead `get_rag_context_for_chat` path. It has no user surface
  and no caller; deleting it is housekeeping, not part of this design.
- TASK-406 and TASK-2375, which concern content reaching the model at all,
  and are separate defects.
