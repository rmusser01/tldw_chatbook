# I1 native history qualification

Executed 2026-10-06. **Textual bridge qualified independently. Web adoption blocked.**

ADR required: no new ADR; this directly implements
`backlog/decisions/194-structured-note-language-and-local-editor-validation.md`.
Server assessment: existing `Docs/ADR/031-notes-capability-sync-domains.md` plus
approved cross-app ADR194; no new persistence/save boundary.

## Textual 8.2.8

`Tests/Widgets/test_document_editing_bridge.py` mounts a real TextArea and Button
inside a Textual App, focuses the editor, types through Pilot keys and clicks the
button. No screen shortcut overrides and no direct undo method are used.
`document.end` was verified as `(2, 1)` for `{\n  "a": 1\n}`.

| Step | Observed source |
| --- | --- |
| A, initial | `{"a":1}` |
| B, type space | `{"a":1} ` |
| C, click Format | `{\n  "a": 1\n}` |
| D, type space | differs from C |
| native ctrl+z | C |
| native ctrl+z | B |
| native ctrl+y | C |
| native ctrl+z then ctrl+z | A |

The bridge captures one immutable DocumentRequest before dispatch, requires
result key = captured key = supplied current key, and compares the widget's
current source synchronously before applying. Thus queued TextArea.Changed does
not admit typing against an old host revision. A newer capture retires the old
one. It also checks excerpt, mount, read-only, effective inherited disablement
and the caller's current editability/conflict authority. No-op creates no edit.

Formatting uses public history checkpoints before and after TextArea.replace,
never load_text. The tests use load_text only to initialize an unrelated test
source. Linear non-whitespace anchors map the current selection/caret into the
replacement; a backward token selection and its undo restoration are tested.
A 100-line mounted editor test verifies nonzero scroll, focus and caret retention.
No quadratic diff or custom undo stack was added. Native redo source is verified;
redo selection itself remains Textual's native behavior, not a separate bridge
restoration. Terminal Textual does not expose browser composition events; this
harness does not claim OS-level IME qualification.

### Verification

Interpreter: `/private/tmp/tldw-structured-chatbook-venv/bin/python`, isolated
checkout binding; Textual8.2.8. Run from the isolated Chatbook checkout:

```sh
/private/tmp/tldw-structured-chatbook-venv/bin/python -m pytest \
  Tests/Widgets/test_document_editing_bridge.py -q \
  -o cache_dir=/private/tmp/i1-history/cache/pytest \
  --basetemp=/private/tmp/i1-history/pytest
```

Result: **5 passed in 2.74s**. Targeted Ruff check and Ruff format --check,
with `--no-cache`, both pass. Black is absent; Ruff performed formatting.
`git diff --check` passes. No full-suite run.

TDD: initial tests failed at import because the bridge did not exist. Native
history, snapshot and selection tests then passed. A later effective-disablement
regression genuinely failed (`assert not True`, 1 failed/4 passed) when only the
widget's own disabled property was checked. Changing admission to
`editor.is_disabled` produced the final 5/5 green run. Logs are under
`/private/tmp/i1-history`: `c-red.log`, `c-disabled-red.log`, `c-final.log`.
An initial run lacked a dedicated basetemp and emitted existing shared pytest
cleanup warnings; subsequent runs used the dedicated path and passed cleanly.

## Browser release gate failed

The real production controlled React Notes page and its existing change handler
were exercised on test-created API notes, with temporary fixed-format wiring.
The wiring has been removed. macOS native Meta+z/Meta+Shift+z were used, never
programmatic undo.

| Browser | Observed history |
| --- | --- |
| Chromium145.0.7632.6 | D → C → B → redo C; prior typing undoable to A |
| Firefox146.0.1 | D → C → B → redo C; prior typing undoable to A |
| WebKit26.0 | **D → A on first Undo**, expected C |

A second run using the retained test-only fixture reproduced this result:
**2 passed, 1 failed in 32.1s**. WebKit coalesces formatting and adjacent typing.
Six jsdom guard tests pass but do not establish history support. Live IME,
large-document viewport, conflict/navigation and keyboard-toolbar qualification
remain incomplete on web. No production web bridge or formatter is committed.

The full server evidence, exact stack environment, owned session/data cleanup,
repro commands, test-only patch/candidate paths and **smallest Monaco qualification
proposal awaiting revised approval** are recorded here:
[server history evidence](/private/tmp/tldw-server-structured-document-editing/Docs/superpowers/reviews/2026-09-27-document-editor-history.md).

Proposal: qualify the already-installed Monaco only for JSON/JSONL/YAML source
modes, retaining Markdown, using the existing StandaloneHtmlSourceEditor
component/disposal seam but not its HTML save veto/setValue rollback. Prove local
bundled loading (no default CDN), disable competing built-in validation, keep
future engine/worker and host identity/source/composition/editability guards,
dispose per authority/session, bracket native edits with undo stops, and repeat
all three real browser traces before adoption. No replacement has been built.
A smaller synchronous native focus-boundary experiment remains a hypothesis;
current evidence only rules out the tested retained-focus insertText transaction.

Task34570 remains In Progress. The web gate is not complete and no broader
programme completion is claimed.

## Single synchronous native-boundary follow-up

The controller authorized one minimal correction experiment before replacement:
synchronous blur/focus before insertText and again afterward, retaining guards,
rechecking source/attachment/composition/editability after focus events, and
restoring mapped selection/scroll. Actual controlled Notes/WebKit26.0 again
produced A/B/C/D then **A on the first native Meta+z**, not C. The focused-editor
assertion had passed. No delays, artificial keystrokes, programmatic undo, or
snapshot history were added. The six candidate guard tests and targeted tsc pass;
the live gate still fails. See the server report's appended exact variant/trace.

Further experiments stopped under the one-experiment instruction. Temporary
product wiring removed and owned server processes33547/32968 stopped. The earlier
native focus-boundary hypothesis has now been tested and failed in this concrete
form. Monaco remains unimplemented; the smallest proposal still awaits approval.


## 2026-10-07 — Installed Monaco candidate (I1 qualification only)

Following the human's option-1 selection, the installed Monaco candidate was
qualified in the real Next Notes page with its actual React change handler.
Chromium145/Firefox146/WebKit26 passed 36 controls including exact LF/CRLF native
A/B/C/D undo/redo history, source/key/editability/conflict checks, selection and
model disposal; strengthened nonzero viewport checks passed all three engines.
BOM, mixed newlines and lone CR retain exact source in a read-only refusal.
Installed local distribution/worker assets loaded in Next and an actual WXT MV3
asset/CSP smoke. Temporary product wiring/public assets were removed.

Full commands, traces, RED/GREEN failures, static comparisons and self-review live
in S `Docs/superpowers/reviews/2026-09-27-document-editor-history.md`, dated Monaco
section, and the private task-11-report. The controlled textarea is still
unqualified. C's already reviewed Textual bridge/code/tests are unchanged and
were not rerun. This is an experiment under ADR194 and extended S ADR070, not I3
adoption. Actual OS IME, extension Notes/history, prior-consumer singleton loader
ordering and Next standalone packaging remain unqualified. Whole W typecheck
passed after product wiring removal with a disclosed 12GB heap; initial default
heap OOM is recorded as an observed failure, without inherited-debt attribution.
Tasks34570/13512 remain In Progress/no AC closure for controller review.
