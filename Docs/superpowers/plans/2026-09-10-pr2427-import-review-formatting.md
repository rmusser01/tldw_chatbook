# PR 2427 bounded import formatter implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development in this session, with spec compliance review before code quality review. Steps use checkbox syntax.

**Goal:** Move only pure bounded diff formatting to the existing import presentation module, preserving all exact output and controller ownership.

**Architecture:** The controller retains matched-item reads, missing-note skips, empty/first-payload selection and effect construction. A synchronous four-string helper in the existing state module returns the same private preview; no new module, owner, dependency, logging or persistence.

**Tech Stack:** Python, stdlib difflib, pytest.

Spec: Docs/superpowers/specs/2026-09-09-pr2427-import-review-formatting-design.md.
Task: TASK-31932, steps151–152.
ADR required: no new ADR.
ADR path: backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md.
Reason: pure formatting placement within the approved presentation boundary.

## Revalidated baseline

Integrated HEAD617884de61 includes Obsidian mode and exact resolved-link receipts. Those paths do not change. Controller is666 lines; auto-merged dev increased its pin from587 to602. Restore587, never raise it. The formatter move alone will leave residual debt.
Caller census finds only the controller's own call and method definition; the large-input controller test spies on controller_module.difflib.unified_diff. Move that spy to the state module without dropping its controller-path assertions.

## Task 1: Characterize, move, verify

Files:
- Modify tldw_chatbook/Library/library_note_import_state.py: owns pure formatter.
- Modify tldw_chatbook/UI/Library_Modules/library_note_import_controller.py: retains reads and payload selection, calls formatter.
- Modify Tests/UI/Library_Modules/test_library_note_import_controller.py: characterize old method, then migrate exact cases through the new helper and real controller; retain all existing cases.
- Modify Tests/Library/test_library_note_import_state.py: direct four-string parity/bounds/private-repr controls if useful.
- Modify Tests/Architecture/test_library_modules_size_ratchet.py: restore original587 pin only.
- Update this plan and backlog/docs/pr-2427-rebase-reconciliation.md with terminal evidence and measured residual debt.

- [x] Add characterization cases against the existing private method before moving it: empty/equal documents; exact title/body hunk; Unicode/newline boundaries; title-length15990/15991/15992/15993/15994 and equal overlong content; differing huge content bounded at1600 with exact marker. Add controller effects coverage for unmatched/missing notes, empty payload, first of multiple payloads and reader order. Run these controls on unchanged production and record terminal evidence.
- [x] Move the formatter body verbatim except four explicit string inputs; remove difflib from the controller and import it in state. Add Google-style Args/Returns docs without claiming sanitization. Keep no-payload handling in the controller:

```python
content_diff = ""
if item.payloads:
    payload = item.payloads[0]
    content_diff = bounded_note_diff(
        note.title, note.content, payload.title, payload.content
    )
```

Use this inside the existing loop after the missing-note guard, then pass content_diff to the unchanged NoteImportReviewEffect construction.

The new helper implementation is:

```python
def bounded_note_diff(
    existing_title: str,
    existing_content: str,
    imported_title: str,
    imported_content: str,
) -> str:
    max_input_chars = 16_000
    max_output_chars = 1_600
    marker = "\n… Diff preview truncated."

    def bounded_document(title: str, content: str) -> tuple[str, bool]:
        prefix = "Title: "
        total = len(prefix) + len(title) + 2 + len(content)
        bounded = f"{prefix}{title[: max_input_chars - len(prefix)]}"
        if len(bounded) < max_input_chars:
            separator = "\n\n"[: max_input_chars - len(bounded)]
            bounded += separator
            bounded += content[: max_input_chars - len(bounded)]
        return bounded, total > max_input_chars

    before_text, before_truncated = bounded_document(existing_title, existing_content)
    after_text, after_truncated = bounded_document(imported_title, imported_content)
    truncated = before_truncated or after_truncated
    before = before_text.splitlines()
    after = after_text.splitlines()
    chunks: list[str] = []
    size = 0
    budget = max_output_chars - len(marker)
    for line in difflib.unified_diff(
        before,
        after,
        fromfile="Existing note",
        tofile="Imported source",
        n=2,
        lineterm="",
    ):
        chunk = line if not chunks else f"\n{line}"
        if size + len(chunk) > budget:
            truncated = True
            break
        chunks.append(chunk)
        size += len(chunk)
    preview = "".join(chunks)
    return f"{preview}{marker}" if truncated else preview
```

- [x] Migrate characterization calls to the new four-string helper; retain controller no/first-payload tests against _build_review_effects. Move the large-input spy to state_module.difflib. Check whole-repository caller census again; do not leave a compatibility wrapper with no consumers.
- [x] Run complete state/controller files and scoped architecture ratchet, with isolated authority paths:

```sh
task_tmp=$(mktemp -d "$TMPDIR/pr2427-import-formatting.XXXXXX")
.venv/bin/python -I -m pytest Tests/Library/test_library_note_import_state.py Tests/UI/Library_Modules/test_library_note_import_controller.py -q --basetemp="$task_tmp/pytest"
.venv/bin/python -I -m ruff check tldw_chatbook/Library/library_note_import_state.py tldw_chatbook/UI/Library_Modules/library_note_import_controller.py Tests/Library/test_library_note_import_state.py Tests/UI/Library_Modules/test_library_note_import_controller.py
git diff --check
```

Expected: both complete functional files pass. Measure and report controller residual cap failure honestly; no claim all PR gates pass. Preserve old RED evidence and existing exact-resource controls.
- [ ] Obtain independent bounded diff review for exact formatting, privacy and ownership. Save verified scope in a separate commit after relevant tests and static checks. Integrated import-flow/native qualification remains part of step152, not replaced by pure tests.

## Implementation evidence

Characterization before the move:16 passed,56 deselected
(`/private/tmp/pr2427-import-formatting-baseline-green.log`). The new wiring
control failed before implementation because the four-string call was absent
(`/private/tmp/pr2427-import-formatting-wiring-red.log`). Final complete files
after test-docstring/type follow-up:73 passed,2 existing dependency warnings,
4.01s (`/private/tmp/pr2427-import-formatting-style-final.log`). Four-file Ruff,
scoped formatting and whitespace checks pass. Independent spec compliance
cleared first; subsequent quality review confirms exact formatter-body AST
parity and unchanged privacy/effect ownership. No old private callers remain.

The controller is628 lines,38 fewer, still41 over its unchanged587 ceiling.
The final two-file size ratchet run reports41 passed,11 failed
(`/private/tmp/pr2427-size-final.log`); none of these failures is waived.
Two initially invalid typed characterization fixtures were corrected before
production changes. The defensive matched-empty-payload branch uses a documented
namespace because valid typed plans prohibit that combination. No extra owner,
dependency or architectural boundary was added; ADR059 still applies.
