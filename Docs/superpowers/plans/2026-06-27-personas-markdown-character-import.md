# Personas Markdown Character Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow Personas character import to accept Markdown files that embed existing supported character-card data.

**Architecture:** Reuse the existing character-card import helper and parser. Expand only the ds-native Personas picker affordance and add regression coverage for Markdown YAML frontmatter, Markdown fenced JSON, invalid Markdown failure, and existing JSON/PNG import behavior.

**Tech Stack:** Python 3.11+, Textual, existing `EnhancedFileOpen`, existing `Character_Chat_Lib.load_character_card_from_string_content`, pytest mounted UI/unit tests.

---

## Source Material

- Backlog task: `backlog/tasks/task-140 - Support-Markdown-character-card-imports.md`
- Approved design: `Docs/superpowers/specs/2026-06-27-personas-markdown-character-import-design.md`
- Personas screen import picker: `tldw_chatbook/UI/Screens/personas_screen.py`
- Parser: `tldw_chatbook/Character_Chat/Character_Chat_Lib.py`
- Import-flow tests: `Tests/UI/test_personas_workbench.py`
- Parser tests: `Tests/Character_Chat/test_character_chat.py` or a new focused character-import parser test file if local imports make that cleaner.

## Scope Check

This is one small import affordance and coverage change.

- In scope: `.md` / `.markdown` picker filters, parser regression tests for the existing Markdown wrappers, Personas import-flow tests.
- Out of scope: a new heading/prose Markdown schema, bulk import, DB/schema changes, sync changes, avatar upload changes.

ADR required: no
ADR path: N/A
Reason: import picker/parser affordance and regression coverage using existing import and storage boundaries; no new long-lived Markdown schema or storage contract.

## File Structure

- Modify `backlog/tasks/task-140 - Support-Markdown-character-card-imports.md`
  - Track implementation plan, checked ACs, and implementation notes.

- Create/modify `Docs/superpowers/plans/2026-06-27-personas-markdown-character-import.md`
  - This implementation plan.

- Modify `tldw_chatbook/UI/Screens/personas_screen.py`
  - Add `.md` / `.markdown` to the import picker filters.

- Modify `Tests/UI/test_personas_workbench.py`
  - Add filter coverage and import-flow regressions for Markdown success/failure.

- Modify/create `Tests/Character_Chat/test_character_import_markdown.py`
  - Add parser regressions for YAML frontmatter and fenced JSON Markdown.

---

### Task 1: Add Parser Regression Coverage

**Files:**
- Test: `Tests/Character_Chat/test_character_import_markdown.py`

- [ ] **Step 1: Add failing parser tests**

Create tests that call `load_character_card_from_string_content()` with:

- Markdown fenced JSON containing a valid V2 card.
- Markdown YAML frontmatter containing a valid V2-like card object.
- Invalid Markdown with no card data.

Expected:

- Valid Markdown returns parsed card data with `name`, `first_message`, and `message_example`.
- Invalid Markdown returns `None`.

- [ ] **Step 2: Run parser tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Character_Chat/test_character_import_markdown.py -q
```

Expected: pass if the existing parser already supports the approved forms. If a test fails because of a parser bug in an approved form, fix the parser minimally.

---

### Task 2: Add Personas Picker and Import Flow Coverage

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Modify: `Tests/UI/test_personas_workbench.py`

- [ ] **Step 1: Add failing UI/import tests**

Add tests proving:

- The import picker filters include `.md` / `.markdown` in the "Character Cards" filter and expose a Markdown-specific filter option.
- `_import_character_from_path()` passes `.md` paths to `ccp_character_handler.import_character_card()`.
- Invalid Markdown import failure leaves the existing selection unchanged and shows the existing failure notification.

- [ ] **Step 2: Update import picker filters**

In `_import_dialog_worker()`, expand filters to include Markdown:

```python
("Character Cards", lambda p: p.suffix.lower() in (".json", ".md", ".markdown", ".png")),
("JSON Files", lambda p: p.suffix.lower() == ".json"),
("Markdown Files", lambda p: p.suffix.lower() in (".md", ".markdown")),
("PNG Files (with embedded data)", lambda p: p.suffix.lower() == ".png"),
("All Files", lambda p: True),
```

- [ ] **Step 3: Run focused UI tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_personas_workbench.py -q -k "import"
```

Expected: import-flow tests pass and existing JSON/PNG/import behavior remains unchanged.

---

### Task 3: Final Verification and Task Completion

**Files:**
- Modify: `backlog/tasks/task-140 - Support-Markdown-character-card-imports.md`

- [ ] **Step 1: Run focused verification**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Character_Chat/test_character_import_markdown.py Tests/UI/test_personas_workbench.py -q -k "markdown or import"
git diff --check
```

Expected: tests and diff check pass.

- [ ] **Step 2: Complete Backlog task**

Check all ACs, add implementation notes, and run:

```bash
backlog task edit 140 -s Done
```

- [ ] **Step 3: Commit**

Run:

```bash
git add Docs/superpowers/plans/2026-06-27-personas-markdown-character-import.md \
  "backlog/tasks/task-140 - Support-Markdown-character-card-imports.md" \
  Tests/Character_Chat/test_character_import_markdown.py \
  Tests/UI/test_personas_workbench.py \
  tldw_chatbook/UI/Screens/personas_screen.py
git commit -m "feat: support markdown character imports"
```
