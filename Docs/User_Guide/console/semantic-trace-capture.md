# Console semantic trace capture

## What is saved

With **Capture On**, Console keeps a local, read-only account of each provider
call so the Conversation Inspector can explain what the model was given and
what came back. The saved conversation remains the source of truth for ordinary
messages: a trace points to immutable message revisions instead of saving the
whole transcript again on every send.

Provider-only material has no transcript row to point at. Rendered project
instructions, RAG or memory context, tool schemas, tool arguments and results,
provider transformations, and unmatched responses are filtered and saved once
as trace artifacts. Repeated calls reuse those artifacts where possible.

A trace describes the final semantic values handed to Chatbook's provider
adapter. It is not generally a byte-for-byte HTTP log: provider-library framing,
retries below that boundary, and provider-side transformations may not be
visible. Every known omission, mask, truncation, corruption, or capture failure
is shown as such; Console does not invent missing history.

## Capture On, Capture Off, and temporary chats

Capture is on by default. Change the global default in **F9 > Console
Behavior**, or press **c** in a live Trace to choose **Next send**, **This
conversation**, or **Global default**. Capture Off affects future calls and does
not delete existing trace history.

A Capture On call needs a durable owner before it can contact the provider. If
the chat is temporary, Console offers **Save & Send**. That action saves the
current lineage and its trace ownership, then sends. You can instead make an
explicit one-shot send with Capture Off. Temporary trace and fork state can stay
coherent while the app is running and while you switch chats, but it is not
restart-durable until saved.

If the pre-send trace reservation fails, automatic dispatch stops. Retry the
reservation or explicitly send that one call with Capture Off. A failure after
provider dispatch cannot discard the provider result; the Inspector reports an
incomplete or interrupted trace boundary.

## Safe and Full are views of one trace

**Safe** and **Full** no longer choose how much history is stored. They are local
viewer and export profiles over the same filtered trace:

- **Safe** shows the transcript context and structural facts while keeping
  provider-only, tool, instruction, RAG, and similar high-disclosure bodies
  collapsed or summarized.
- **Full** can reveal all non-credential content that survived the capture-time
  policy. Opening, copying, or exporting sensitive Full sections requires an
  explicit confirmation.

Changing the viewer never rewrites the trace. Imported or shared traces are
read-only. Historical trace is also never editable: the viewer can inspect,
filter, search its permitted projection, copy, export, or purge ownership. It
cannot alter what a past provider call received.

## Credentials and optional PII masking

Credential filtering is mandatory in both views. Known credential fields,
nested credential fields, URL user information, query strings, fragments, and
recognized secret formats are removed before persistence. A filtering failure
stores an unavailable marker, not raw fallback data. No detector can guarantee
that an arbitrary secret typed into ordinary prose will be recognized, so treat
Full output and exports as sensitive.

**Mask detected PII in traces** is optional and off by default. It applies to
future calls at the selected scope and is irreversible for provider-only trace
artifacts. For saved conversation messages, the trace stores masks over the
referenced revision; it does not rewrite what you see in the conversation.
Both Safe and Full apply the frozen credential and PII policy of each call.

### Custom PII patterns

Advanced users can add application-specific PII patterns in the configuration
file. Each ruleset has an opaque UUID revision. Change that revision whenever a
pattern, flag, category, or enabled state changes; reusing one revision for
different rule content disables the custom rules rather than silently changing
the meaning of an existing trace policy.

```toml
[console.trace_custom_pii_rules]
version = 1
revision_id = "11111111-1111-4111-8111-111111111111"

[[console.trace_custom_pii_rules.rules]]
id = "customer-id"
label = "Customer ID"
category = "customer_id"
pattern = '''customer-[A-Z]{8}'''
flags = ["ignorecase"]
enabled = true
priority = 10
```

Rule IDs and categories use lowercase letters, numbers, hyphens, or
underscores. Supported flags are `ascii`, `dotall`, `ignorecase`, and
`multiline`; put flags in the list instead of embedding them in the pattern.
Patterns that can match without consuming text, including zero-width
lookarounds and optional-only expressions, are rejected before worker launch.
Settings reports enabled, disabled, and rejected rule counts plus content-free
error codes. It never includes the pattern or matched text in diagnostics.

Custom patterns never run in the Console process. One disposable worker handles
the whole capture component, with a 500 ms deadline, 1 MiB input and output
caps, at most 512 text fields, 64 rules, and 10,000 candidate matches. The
worker also applies CPU, output-file, and memory resource limits where the host
operating system supports them. Invalid input, a crash, malformed output,
resource exhaustion, or catastrophic backtracking omits the affected component
with a content-free reason code. The rest of the saved trace remains viewable,
copyable, and exportable.

Pattern text remains only in the user's settings and bounded process memory.
Durable traces store the opaque ruleset revision, irreversibly masked artifacts,
and content-free codepoint spans. Once those masks exist, later viewing and
export do not rerun or require the custom pattern. Saved conversation text is
still unchanged.

## Edits, regeneration, retries, compaction, and forks

Trace history follows the provider-visible history, not just today's active
transcript:

- Editing or deleting a referenced message preserves the needed old revision
  once before changing the live message.
- Edit & resend, regeneration, retries, failures, stops, and tool-loop calls are
  separate call boundaries. Abandoned variants remain identifiable.
- Context compaction and message edits append bounded replacement records; they
  do not rewrite old trace events.
- A fork shares the immutable trace prefix visible at its boundary and adds only
  its own suffix. The complete inherited history remains coherent in the fork
  without copying the prefix.

Because prefixes are shared, purging the source conversation's trace may not
reclaim history still owned by a fork. Console reports the remaining owners.
There is no lineage-wide one-click purge.

## Older captures

Older `message_exchanges` captures remain readable. After the interface is
ready, bounded idle maintenance converts them into isolated, immutable legacy
snapshot traces and removes each old blob only after a verified equivalent read.
The converter does not invent edit, fork, or predecessor relationships that the
old format never recorded. Old Safe aggregate omissions and corrupt data stay
visible as irreversible legacy omissions.

Operators can independently pause normalized writes or normalized reads for a
rollback. Pausing reads retains the compact history; re-enabling reads makes it
visible again. The old repeated-transcript writer is disabled for new calls by
default, while dual-read compatibility remains.

## Export, purge, and physical storage

Copy and export obey the active Safe or Full projection and every frozen mask.
Full copy/export asks for confirmation. Exported files are independent copies;
later trace deletion cannot recall them.

Purge detaches one conversation's trace owner. Background garbage collection
can then remove objects no remaining conversation or fork owns. Logical
reclamation, SQLite free pages, WAL bytes, and the allocated database file are
different measurements: deleting rows does not promise immediate file-size
reduction or forensic erasure. Later admitted SQLite maintenance may reclaim
physical space, but filesystem snapshots, backups, synced database copies, and
prior exports can still retain old bytes.

## Related pages

- [Context & RAG](context-and-rag.md) — Conversation Inspector and the model's
  next-send context.
- [Branching & rewind](branching-and-rewind.md) — variants, edits, forks, and
  compaction.
- [Chat basics](chat-basics.md) — selecting messages and opening their actions.
