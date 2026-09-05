# Chunking Lab

Chunking Lab is a local, single-sample A/B workbench owned by Library. Open
**Chunking Lab** in Library or **Library: Chunking Lab** in the command palette.
**Try selected text** copies an eligible local Library item's complete extracted
text—not the shortened Chat handoff. Server-owned items and items without
extracted text are refused. The Lab remains local even in a server-scoped shell.
Back returns to the opening route. There is no separate Settings editor.

## Author and compare

1. Paste into Sample, or use Files / Recovery to load a UTF-8 file. Files retain
   their exact decoded text, including line endings. Samples may contain at most
   2 MiB of UTF-8 text. Larger files/Library records require an explicit character
   range (zero-based start, exclusive end); no automatic truncation occurs.
2. Configure editable B using the method controls or Full JSON. Words measures
   words; Fixed size measures characters. Empty, malformed, or incomplete controls
   remain visible until corrected or explicitly discarded. Invalid JSON retains
   the exact raw text and last valid document. Pending controls temporarily own
   editing; discard/correct them before using JSON. Undo restores the last edit.
3. Run B. Pin A freezes B's current successful, non-stale captured recipe and
   provenance. Later B edits do not change A. Replacing A requires confirmation.
   Run both captures one immutable pair and executes those captured inputs
   sequentially. Edits during execution affect the next run.
4. Compare captured results, select chunk rows, and inspect chunk text, source or
   transformed text, statistics, authored/effective configuration differences,
   or execution metadata. Source linking is available only for verified spans;
   transformed output without a trustworthy map is explicitly unaligned.
5. Save A or Save B as a reusable local template. Name and description are required
   by the catalog; tags are retained. Save A uses its captured record fields and
   authored metadata/classifier, with captured effective executable sections made
   explicit. Save B uses its current valid authored body. Dialog field edits are
   explicit. Built-ins require Save as new. Saving refreshes Library ingest's
   template choices; it does not change the selected default or re-chunk sources.

Saved templates are searchable and decorated for built-in, invalid, or reserved
entries. Loading is a detached draft, not live catalog state. Template JSON import
and export retain the flat body and record tags; importing does not save a catalog
record. Concurrent catalog changes produce a conflict and retain the draft: reload
deliberately or Save as new. A late save cannot attach an unrelated imported draft.

At 80 columns, Sample, Configure, and Results are separate regions. Results scroll
with keyboard focus so paging controls and the full-text inspector remain
reachable. At wider sizes an input region can sit beside Results; a sufficiently
wide Results view shows A/B equally. Tab/Shift+Tab navigate native controls; F6
cycles task regions and F1 remains global help. The r/p/s shortcuts operate only
outside text/control editors; terminal copy/paste and global Ctrl+P/Ctrl+Q retain
their normal ownership. Footer hints change with focus.

## Supported execution and limits

The current dependency-free local preview supports English `words` and
`fixed_size` chunking with the pre/post operations admitted by shared preflight.
Unknown executable keys, unsupported methods/languages, unavailable dependencies,
and incompatible options fail visibly; no provider/server fallback or implicit
asset download occurs. For example, `preserve_sentences: true` is not supported by
the dependency-free words path. Changing methods preserves incompatible settings;
use Full JSON to explicitly correct them. Nested metadata, classifier rules and
other non-executable authored data are preserved. **Classifier rules are not run**
by the Lab; they remain available to the existing ingest selection workflow.

Default ceilings are 2 MiB per sample and authored/effective recipe document,
10,000 chunks, 32 MiB canonical serialized result (including captured request),
16 combined pre/post operations, and a 60-second parent wall deadline. Conservative
working-payload admission can refuse a pipeline whose eventual output would fit.
Tables page 100 rows; inspector text pages 8,192 Unicode code points without
discarding the rest. Template imports are bounded at 8 MiB; recovery transfers at
256 MiB, additionally subject to recovery schema/content admission.

The 32 MiB working-payload estimate is **not a process-memory/RSS cap**. On the
qualified macOS/arm64 Python 3.12 host, a repeated-formatter fixture used about
458 MiB peak RSS despite an approximately 31 MiB estimated payload. That is an
observed fixture, not a universal upper bound. macOS applied RLIMIT_CPU=61 seconds
but refused the attempted 1 GiB address-space limit; there is no macOS hard
address-space cap. Linux has conditional address-space checks but was not qualified
on this host. Windows is explicitly refused as `platform_unqualified`. Execution
uses a supervised local child process, not a security sandbox. Cancellation sends
SIGTERM, escalates after 0.5 seconds, and retains ownership until reaping is
observable; OS scheduling affects total cancellation latency.

## Recovery and privacy

The app owns one Lab coordinator per local application profile. The private local
checkpoint store is `get_user_data_dir() / "chunking_lab.sqlite3"`. Reopening
restores samples, drafts (including invalid input), captured outputs and view state
without automatically running. Navigation/quit drains edits, cancels work and
checks the checkpoint before leaving. A profile switch closes the old owner first;
failure retains that owner's retry/export authority instead of mixing profiles.

Autosave coalesces ordinary edits after 300 ms, with a 1-second maximum wait before
starting a write. A crash can lose edits not yet committed; storage latency adds to
that window. Critical run/recovery operations await their checkpoints. Saved
locally means the latest revision was acknowledged. Saving may describe only a
selection/view change; Unsaved result is reserved for result publication not yet
covered by an acknowledged checkpoint.

After a write failure, stay in the Lab and choose Retry or Export recovery. Export
does not depend on the failed checkpoint write. An initial recovery-load failure
disables authoring; Retry performs a fresh read, never treats an empty placeholder
as overwrite authority. Concurrent-instance recovery conflicts require deliberate
export/reopen, not blind overwrite. Recovery fallback can retain an earlier valid
checkpoint; no mixed partial result/configuration is accepted.

Export recovery writes the complete admitted active session to an explicit path.
Restore validates before replacement and binds the imported session to the current
profile; imported data never supplies filesystem write targets. Undo restore
restores the replaced local session. Clear local recovery requires confirmation
and removes Lab recovery state without deleting reusable templates. Existing
output files require explicit overwrite permission and an identity-checked atomic
private write.

**Full sample text and results are stored locally and included in recovery exports.**
Do not experiment with sensitive text unless this retention is appropriate.
Private file permissions are not encryption; Clear is not secure erasure and does
not remove independently exported files. Source content, ingestion defaults, and
server records are never written by preview/recovery operations.

Architecture: [ADR-118](../backlog/decisions/118-chunking-lab-local-execution-and-recovery.md)
and the [approved design](superpowers/specs/2026-09-04-chunking-lab-design.md).
