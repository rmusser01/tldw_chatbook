# Structured Document Editors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect qualified engines to the real editors with normal undo, source-safe presentation, and end-to-end acceptance evidence.
**Architecture:** Editor-specific bridges apply revision-pinned replacement edits through existing document owners. Shared per-app validation controllers project diagnostics without becoming a second save authority.
**Tech Stack:** Textual 8.2.8, React text areas, pytest Textual pilots, Vitest, Playwright, existing Notes APIs.
**Spec:** [Approved design](../specs/2026-09-27-structured-document-editing-design.md)

## Global Constraints

- All [programme constraints and interfaces](2026-09-27-structured-document-editing.md) apply.
- "Check after 400 ms of typing inactivity and on initial open or language change."
- "Syntax errors never add a save veto."
- "Run targeted checks only unless the user explicitly authorizes a full suite."
- ADR required: yes. ADR path: `C/backlog/decisions/194-structured-note-language-and-local-editor-validation.md`; server ADR-031 and the F3 extension govern metadata transport.
- Status: Not Started. No whole-editor replacement is approved. A failed history qualification returns a concrete proposal for review before adopting a replacement widget.

## I1: Qualify formatting in native undo history

**Dependencies:** none for the qualification harness; use a fixed known replacement before engine integration.
**Files C:** create `tldw_chatbook/Widgets/document_editing_bridge.py`, `Tests/Widgets/test_document_editing_bridge.py`.
**Files U:** create `src/components/Notes/structured-editor-bridge.ts`, `__tests__/structured-editor-bridge.test.ts`.
**Files W:** create `e2e/notes-structured-editing.spec.ts`; modify `playwright.config.ts` with narrowly matched `notes-firefox` and `notes-webkit` projects alongside Chromium. Existing Firefox/WebKit projects are restricted to presentation tests and do not cover Notes. Temporary qualification wiring may modify U `src/components/Notes/NotesEditorPane.tsx` and `NotesManagerPage.tsx` only in the isolated execution checkout; it is removed before committing I1.
**Interfaces:** programme `apply_format`/`applyFormat`; C `TextAreaFormatBridge(editor)` and U `TextAreaFormatBridge(textarea, onUserEdit)` encapsulate history application and cursor mapping. No engine dependency in the bridge.

- [ ] Add a real Textual harness with a focused `TextArea`, normal typing, and a button that invokes the bridge. Use the installed public history transaction APIs rather than reloading text:

```python
editor.history.checkpoint()
editor.replace(replacement, (0, 0), editor.document.end, maintain_selection_offset=True)
editor.history.checkpoint()
```

Before adoption, verify `document.end` and selection mapping against pinned Textual
8.2.8 and the actual mounted widget. The bridge must also perform the programme's
identity/editability guards. `load_text()` is reserved for document loading, never
formatting; it does not establish a normal editing transaction.

- [ ] Record initial text A, typed text B, formatted text C, and post-format typed text D. Assert the actual undo key gives C, another gives B, redo gives C, and prior typing is still undoable. Exercise the widget's existing keybinding; do not introduce forbidden screen-level terminal shortcuts.
- [ ] In the browser harness, keep the production controlled `<textarea>` and real React change handler. Qualify a synchronous user-gesture native edit transaction (`insertText` command where supported) before considering a widget change. A React state assignment or `setRangeText` call alone is not evidence of history preservation.
- [ ] Add a browser test using a test-created note and the real page:

```ts
test("format is one native undo step", async ({ page }) => {
  // The beforeEach fixture creates an owned JSON note with content {} via
  // the live Notes API and opens it in /notes; it deletes only that note afterward.
  const editor = page.getByRole("textbox", { name: "Note content", exact: true });
  await editor.focus();
  await editor.press("ControlOrMeta+End");
  await editor.pressSequentially(" ");
  const before = await editor.inputValue();
  await page.getByRole("button", { name: "Format document", exact: true }).click();
  const formatted = await editor.inputValue();
  expect(formatted).not.toBe(before);
  await editor.focus();
  await editor.press("ControlOrMeta+End");
  await editor.pressSequentially(" ");
  await editor.press("ControlOrMeta+z");
  await expect(editor).toHaveValue(formatted);
  await editor.press("ControlOrMeta+z");
  await expect(editor).toHaveValue(before);
  await editor.press("ControlOrMeta+Shift+z");
  await expect(editor).toHaveValue(formatted);
});
```

During qualification temporarily wire a Format button in the actual Notes page to
the bridge with the fixed replacement `{\n  "a": 1\n}` and initialize the test note
as `{"a":1}` instead of `{}`. This proves the real controlled editor's history
without waiting on a parser. Remove this temporary wiring before the I1 commit;
retain the recorded trace and bridge tests. During I3 the browser regression runs
against the real Notes action and qualified engine; it must not be counted as
passing production coverage before that integration exists.
On browsers using another native redo shortcut, use that browser's actual binding
and record it; do not call the bridge's undo directly to pass this check.

- [ ] Test pending typing while formatting, selection/caret/scroll restoration, browser focus loss, IME composition, no-op formats, navigation, and read-only/conflict entry. Both the formatting edit and subsequent typing must remain distinct history units.
- [ ] Run C bridge tests, U bridge tests, and the narrowly matched browser file in all supported browser projects. Record actual versions and failures in `C/Docs/superpowers/reviews/2026-09-27-document-editor-history.md` (with server evidence links). If ordinary textarea transactions fail, stop UI integration and present the smallest replacement proposal with evidence; do not build a second snapshot undo stack.
- [ ] Commit only qualified bridges/harnesses and the report, not an unproven production formatter button.

## I2: Chatbook Database Notes and File Notes integration

**Dependencies:** F1, F2-C, F4-C, F5-C, E4-C, I1-C.
**Files C:** create `tldw_chatbook/Document_Editing/editor_controller.py`, `tldw_chatbook/Widgets/document_diagnostics.py`; modify `tldw_chatbook/Widgets/Library/library_notes_canvas.py`, `library_file_notes_workspace.py`, `tldw_chatbook/UI/Library_Modules/library_notes_controller.py`, `tldw_chatbook/css/features/_notes.tcss`; create `Tests/Widgets/Library/test_structured_note_editor.py`, `Tests/Widgets/Library/test_structured_file_editor.py`.
**Interfaces:** `DocumentEditorController` consumes immutable request snapshots and E4 scheduling, emits view state and diagnostic navigation, and delegates formatting to I1. It has no DB/file-save port. Host controllers mutate existing canonical drafts and own autosave.

- [ ] Add mounted regression scenarios before adding controls: invalid JSON still saves; syntax status updates after 400 ms; choosing YAML persists metadata without altering body; stale results from note A cannot paint note B.
- [ ] Add language selection, Check, Format document, a compact status row, and a collapsible Problems list using existing design-token classes. Emit events to the controller rather than moving workers into the presentation canvas. Hide Markdown-only controls/code rendering safely when the note is structured.
- [ ] Derive the request from the canonical draft, not a second text copy that can lag widget changes. Include authority, note/file identity, session generation, body and language revisions, and current request ID. Retire old success on every body/language change. Do not check while an IME composition is in progress where supported by the host.
- [ ] For File Notes use `_opened` representation facts for BOM/newlines, whole-source extension admission, excerpt/read-only state, and current file conflict status. Let the existing TextArea Changed path mark dirty/arm autosave after a formatting transaction; do not directly write disk or bypass protected-note recovery.
- [ ] Add this negative result-admission check to the bridge/controller tests:

```python
from dataclasses import replace

def test_language_change_invalidates_pending_format(bridge, result, current_key):
    changed_key = replace(current_key, language_revision=current_key.language_revision + 1)
    before = bridge.editor.text
    assert bridge.apply_format(result, changed_key, editable=True) is False
    assert bridge.editor.text == before
```

The test fixtures construct a real mounted editor and a valid `FormatResult` with
`result.key == current_key` and a different replacement. Include a successful
same-key control so a bridge that refuses every format cannot pass.

- [ ] Verify emoji/tab/CRLF/EOF navigation, error-count truncation, unsupported/failed/resource states, manual Check limits, unknown stored languages, and metadata-only conflict resolution. At 60×20 the status and Problems action remain reachable and the editor retains focus during updates.
- [ ] Rebuild via `python tldw_chatbook/css/build_css.py`. Run the two new files, `Tests/Library/test_library_notes_session.py`, `Tests/Notes/test_file_notes_service.py`, `Tests/UI/test_design_token_governance.py`, and the existing CSS bundle sync guard. Run the I1 history sequence in the real Notes/File Notes surfaces, then scoped static checks and commit.

## I3: Web Notes integration and source-safe secondary entry points

**Dependencies:** F2-S, F3, F4-U, F5-U, E4-U, I1-U.
**Files U:** create `src/components/Notes/hooks/useDocumentDiagnostics.ts`, `DocumentDiagnostics.tsx`; modify `NotesEditorPane.tsx`, `NotesEditorHeader.tsx`, `NotesManagerPage.tsx`, `hooks/useNotesEditorState.tsx`, `notes-manager-types.ts`, `types.ts` under `src/components/Notes/`; inspect/update `src/components/Common/NotesDock/NotesDockPanel.tsx`, `src/components/Sidepanel/Notes/NoteQuickSaveModal.tsx`; add `src/components/Notes/__tests__/NotesEditorPane.structured.test.tsx`, `useDocumentDiagnostics.test.tsx`.
**Interfaces:** `useDocumentDiagnostics(request: DocumentRequest | null, composing: boolean)` returns `{status, diagnostics, format, checkNow}` using E4 and I1. Existing note hook owns body/language/durability; diagnostics do not own copies of persisted state.

- [ ] Add red mounted tests: pending status clears previous success immediately, language-only edit marks dirty, a stale result cannot affect the next note, and a structured note never invokes Markdown-to-WYSIWYG conversion during load/restore.
- [ ] Add translated accessible controls and Problems list; use `role=status` for concise state changes and avoid announcing every keystroke. Explicit problem selection focuses the textarea and maps scalar offsets to UTF-16 selection offsets. Continue to show real save/recovery state separately from diagnostics.
- [ ] Capture IME `compositionstart`/`compositionend`, postpone validation until committed input, and release workers/timers on unmount/authority change. Check result identity again immediately before applying native edit transactions.
- [ ] Reuse the canonical `editRevisionRef` lifecycle and extend its snapshot with language. A late successful save for an older draft does not clear a newer language edit. Explicit format uses the qualified bridge and normal edit provenance; AI-assist undo must not overwrite it later.
- [ ] Audit every Notes content writer using `rg -n 'createNote|updateNote|setContent|markdownToWysiwyg' src/components/Notes src/components/Common/NotesDock src/components/Sidepanel/Notes`. For each secondary editor, retain raw source and language or offer a source-editor open action. No secondary editor may submit an implicit null language. Existing Markdown notes and their WYSIWYG behavior remain regression controls.
- [ ] Run the new U tests and existing `NotesManagerPage.canonical-save.test.tsx`, `useNotesEditorState.authority-races.test.tsx`; run W `e2e/notes-structured-editing.spec.ts` against the actual Notes page. Check shared extension import/bundle compatibility, metadata round trip through the live API, browser worker CSP/loading, then commit.

## I4: Wire source downloads, note portability, and user guidance

**Dependencies:** F6, I2, I3.
**Files C:** modify `tldw_chatbook/Widgets/Library/library_notes_canvas.py`, `tldw_chatbook/UI/Library_Modules/library_notes_controller.py`; add `Docs/User_Guide/structured-document-editing.md`, `Tests/UI/test_structured_note_export_actions.py`.
**Files U:** modify `src/components/Notes/NotesEditorHeader.tsx`, `hooks/useNotesExport.tsx`; add `src/components/Notes/__tests__/NotesEditorHeader.structured-export.test.tsx`. Add corresponding server user-guide documentation in `S/Docs/User_Guide/structured-document-editing.md` under a server task.
**Interfaces:** F6 `buildStructuredSource` and `buildPortableNote`; C export controller invokes the equivalent existing archive/raw-export boundaries without adding formatting to either path.

- [ ] Add action tests asserting Download source produces exactly `{"x":9007199254740993}\n` for a JSON note while Export note produces an envelope containing that exact string and language.
- [ ] Wire distinct labels and extensions; avoid hidden conversion through Markdown export. Preserve existing download/export authority and overwrite confirmations. Round-trip a note with Canvas records through the new archive version and reject unsupported versions before any import writes.
- [ ] Document temporary selections on old servers, format refusal reasons, unchanged save behavior, empty-draft recovery, limits, and the meaning of "No syntax errors". Explain that YAML indentation can be syntactically valid while conveying an unintended structure. Do not advertise schema validation or code-block linting.
- [ ] Run both named action tests and F6 round-trip controls; verify downloaded bytes in an actual browser/file dialog and commit each repository's UI/docs changes separately.

## I5: Cross-application release qualification

**Dependencies:** all previous units. No new feature scope.
**Files C:** create `Tests/QA/test_structured_document_round_trip.py`, `Docs/superpowers/reviews/2026-09-27-structured-document-release-evidence.md`; extend precise tests above if failures reveal missing boundaries.
**Files S:** extend `tldw_Server_API/tests/Sync/test_sync_v2_note_language.py`, W `e2e/notes-structured-editing.spec.ts` and its test-owned Notes setup/cleanup.
**Interfaces:** actual public Notes APIs, sync envelopes, real editors, and exported artifacts. No in-memory adapter alone establishes cross-application correctness.

- [ ] Create synthetic YAML, JSON, and JSONL notes through one app, sync/read/edit through the other, and compare exact source plus language after return. Cover local and server-backed Chatbook modes, supported cleartext and private transport paths, and a genuinely older capability profile.
- [ ] Exercise old clients editing body after a new client selected a language; unsupported publication must retain recoverable metadata and truthful status. Verify restart/retry, metadata conflict, explicit reset, unknown future language, tombstone restore, and export/import across archive versions.
- [ ] Run real native typing → format → typing → undo → undo → redo in Database Notes, File Notes, and the web editor. Capture focus, caret, history, autosaved content, and the negative stale-result controls. Include one successful formatting control beside every refusal test.
- [ ] Stress 1 MiB/100-depth boundaries, 100+ JSONL diagnostics, aliases, 400 ms debounce with rapid typing, worker timeout/crash, note/profile switching, unmount during worker startup, and repeated open/close cycles. Record UI responsiveness and verify terminated child processes and settled promises rather than only zero visible spinners.
- [ ] Verify exact BOM/CRLF/final-newline files, externally modified files, protected recovery, 60×20 TUI, browser narrow/mobile layout, keyboard problem navigation, screen-reader status, and IME. Run the shared frontend import/build guard for the extension without advertising extension UX as qualified unless exercised there.
- [ ] In the release evidence file record repo commits, fixture hashes, commands, exit results, relevant browser/Textual/parser versions, skipped/unavailable environments, and each AC1–AC15 outcome. Run only targeted test selections from the three plans plus touched-file static checks; request a full sweep only if the user wants one.
- [ ] Mark implementation tasks Done only for verified criteria. Missing PostgreSQL, browser, encrypted transport, or native history evidence stays explicitly incomplete. Review/commit the evidence and scoped fixes; opening a PR or merging follows the user's chosen workflow, not this checklist alone.
