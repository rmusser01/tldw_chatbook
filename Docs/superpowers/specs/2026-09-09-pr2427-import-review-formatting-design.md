# PR 2427: bounded import-review formatting

Status: Design direction and independent spec review approved; user authorized
continuation on 2026-09-10. Revalidate against integrated dev before writing the
implementation plan.
Task: [TASK-31932](../../../backlog/tasks/task-31932%20-%20Reconcile-PR-2427-with-latest-dev-and-complete-review-gates.md).

## Purpose and scope

Move only the existing bounded note-diff formatting from
`UI/Library_Modules/library_note_import_controller.py` into the existing
`Library/library_note_import_state.py` presentation module. This reduces
coordinator code without changing the import workflow or granting presentation
code database access. The user approved this direction on 2026-09-09.

The controller currently measures 647 lines against its 587-line ratchet.
Moving the approximately 43-line formatter alone does not close that gap.
This design does not promise otherwise: measure the integrated latest-dev code
and report any remaining gap separately. Do not move additional responsibilities,
raise limits, or compress executable code to make the measurement pass.

## Alternatives

1. **Use the existing presentation module (selected).** It already owns frozen
   `NoteImportReviewEffect` values and their private canvas projection; a pure
   formatter belongs beside those values without introducing another owner.
2. **Shorten controller documentation only.** All current docstrings total
   roughly 53 lines, less than the 60-line overrun. This cannot solve the current
   debt alone, and deleting useful contracts is not a sufficient repair.
3. **Create another controller/service or transfer database reads.** Rejected:
   unnecessary abstraction and a materially larger lifecycle/authority change.

## Data flow and ownership

The controller retains `_build_review_effects`, matched-item iteration and order,
its optional injected note reader, and the late-bound `LocalNoteImportTarget`
fallback. It reads one matched note at a time and skips missing notes exactly as
today. The presentation helper accepts already-read title/content strings and
returns only a formatted string; it receives no reader callback, repository,
database, executor, mutable workflow, or Textual object.

The controller retains the current no-payload early return and first-payload
selection before invoking the helper. It creates the same
`NoteImportReviewEffect(item_id, target_title, target_version, content_diff)`
values. No additional note copies, accumulated source documents, cache, async
work, validation policy, or data model are introduced. Whole-repository caller
and monkeypatch census determines whether the old private static method can be
removed directly; an identity-sensitive consumer must not be silently broken.

Planning, approval, cancellation, retry, progress, publication, execution,
receipts and exact database cleanup remain unchanged. Incoming Obsidian parsing
and discovery behavior remain within their existing owners.

## Exact formatting and privacy contract

- Preserve the `Title: ` prefix and two-newline title/body separator, including
  title-first truncation and its boundary behavior.
- Cap each diff input document at 16,000 characters and output at 1,600
  characters, including the existing truncation marker.
- Preserve `difflib.unified_diff`, two context lines, empty line terminator,
  `Existing note`/`Imported source` labels, line assembly and output budgeting.
- Preserve the exact `\n… Diff preview truncated.` suffix whenever either input
  or generated output truncates, including equal truncated inputs.
- Empty payloads still yield an empty diff. Empty/equal documents, long titles,
  large bodies, Unicode and newline boundaries retain their current results.
- Preview content remains private render data with the existing `repr=False`
  fields. Add no diagnostics, logging, persistent storage or exception handling.
  Exceptions continue through the controller's existing failure path.

## Verification and integration

After written-spec approval, record an implementation plan before code changes.
Characterize exact existing formatter output and bounds before extraction, then
run the same controls against the new helper and through the real controller.
Retain missing-note, no-payload, injected-reader/default-target, ordering and
private-repr controls. Check every former caller; no test may disappear merely
because the private method moved.

Run complete affected import-state/controller files, scoped lint and architecture
caps, then the integrated import UI/resource qualification already required by
TASK-31932. Preserve historical failing evidence and resource-owner checks.
Review the diff independently for behavior, ownership and private-data parity.
This small extraction is not evidence that the remaining PR gates are complete.

ADR required: no new ADR.
ADR path: [ADR-059](../../../backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md).
Reason: behavior-preserving placement of pure formatting inside the existing
presentation boundary; no change to storage, service contracts, data authority,
dependencies or lifecycle ownership. Any expansion beyond that boundary requires
a separate decision before implementation.
