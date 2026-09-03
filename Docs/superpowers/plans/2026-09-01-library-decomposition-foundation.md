# Library Decomposition Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the Library decomposition's foundation — PR 0a (support-layer move), PR 0b (guards), and the conversations exemplar series (state → reader controller → browse controller → cleanup) — establishing the recipe every later subsystem follows.

**Architecture:** Strangler extraction of `LibraryScreen` (46k lines, 1,270 methods) into `UI/Library_Modules/` controllers + state objects, per the approved doctrine (`2026-08-02-screen-decomposition-design.md`, `DESIGN.md` §7) and the Library spec. Moves are byte-for-byte (the `ConsoleDictationController` canon): method bodies are never edited; every name they reference is rebound in the controller's constructor or via generated properties.

**Tech Stack:** Python ≥3.11, Textual 8.x, pytest (`Tests/UI` mounted harnesses, `Tests/Architecture` ratchets), ast-based census scripts.

**Spec:** `Docs/superpowers/specs/2026-09-01-library-screen-decomposition-design.md` (read it first; this plan argues from it).

## Global Constraints

- **Pure moves only.** Extraction PRs contain: verbatim block moves, import-path changes, constructor/property bindings, screen delegator one-liners. No logic edits, no renames, no cleanups.
- **Tests pass unmodified** in every PR except cleanup PRs (Task 9), which may retarget attribute paths/patch targets with assertions byte-for-byte.
- **One controller (or one prep layer) per PR.** Never two subsystems in flight.
- **This file churns ~14 commits/day.** Never trust line numbers in this plan at execution time — every task locates its material with the provided scripts. Rebase onto latest `origin/dev` immediately before each PR's final measurement; budgets are measured post-rebase (Console wave 3 landed red twice from stale-base numbers).
- **Names that tests monkeypatch on `LibraryScreen`** keep their whole call graph routed through the screen until cleanup. The `*_local_source_snapshot` trio is shared shell infrastructure — never moved by this plan.
- **Fields referenced by ≥2 subsystems stay on the screen** (shared shell state). This plan moves only conversation-exclusive fields.
- Run tests with the project venv: `.venv/bin/python -m pytest … -p no:randomly`.
- Every move commit's hash is appended to `.git-blame-ignore-revs` in the same PR.
- Each PR gets a `backlog` task (`backlog task create "<title>" --plan "<this plan §>"`); mark Done only per the repo's DoD.

---

### Task 1: PR 0a — move the module-level support layer

The ~92 module-level names above `class LibraryScreen` (constants, copy strings, support dataclasses, `_sync_library_canvas`, `_apply_library_row_toggle`, `_LibraryDatabaseNoteSessionPort`, helper functions — ~2,300 lines including imports) move to five focused modules in `UI/Library_Modules/`. Re-export aliases stay in `library_screen.py` so its import surface is unchanged (75 test files import from it).

**Files:**
- Create: `tldw_chatbook/UI/Library_Modules/screen_constants.py` (all module-level `Assign`/`AnnAssign` constants and copy strings)
- Create: `tldw_chatbook/UI/Library_Modules/screen_support_types.py` (the `ClassDef`s: `_LibraryIngestStartConsent`, `LibraryEntryReconcileResult`, `LibraryEntryFocusIdentity`, `_LibraryEntryFocusCapture`, `_LibraryMediaReturnReceipt`, `_LibraryMediaReturnSettlement`, `_LibraryMediaSuccessfulFocusOwnership`, `_LibraryEmergencyReturnEligibility`, `_LibraryEmergencyRestoreReceipt`, `_LibraryNotesRecomposeCapture`, `_LibraryNotesRestoreGuard`, `_LibraryNotesDeletedFolderReceipt`, plus the type aliases `_LibraryMediaFinalFocusPolicy`, `_LibraryMediaSettlementOutcome`, `LibraryReaderDestination`)
- Create: `tldw_chatbook/UI/Library_Modules/note_session_port.py` (`_LibraryDatabaseNoteSessionPort`)
- Create: `tldw_chatbook/UI/Library_Modules/canvas_sync.py` (`_sync_library_canvas`, `_apply_library_row_toggle`, `_move_library_list_row_focus`, `_patch_library_disabled_marker_label`)
- Create: `tldw_chatbook/UI/Library_Modules/screen_helpers.py` (remaining module-level `FunctionDef`s: `_read_library_ingest_options_from_config`, `_library_ingest_options_for`, `_library_screen_is_current`, `library_note_persisted_title`, `_ingestible_file_filters`, `_transcribe_cpp_gguf_filters`, `_library_carries_forward_line`, `_unbreakable_size_text`, `_active_library_sync_scope`, `_record_value`, `_library_collection_record_data`, `_library_collection_browse_summary`, `_collection_scoped_mirror_report`, `_collection_scoped_conflicts`, `_canonical_shortcut_key`)
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` (delete moved blocks; add re-export imports)
- Test: `Tests/Architecture/test_library_support_layer_surface.py`

**Interfaces:**
- Produces: every moved name importable from BOTH its new module and (as re-export) `tldw_chatbook.UI.Screens.library_screen`. Later tasks import from the new modules.

- [ ] **Step 1: Create the backlog task and branch**

```bash
cd /Users/macbook-dev/Documents/GitHub/ppqq/tldw_chatbook
git fetch origin dev && git switch -c refactor/library-decomp-0a-support-layer origin/dev
backlog task create "Library decomposition PR 0a: move module-level support layer" \
  -d "Per Docs/superpowers/specs/2026-09-01-library-screen-decomposition-design.md, PR 0a." \
  --ac "All moved names importable from new modules AND from library_screen re-exports" \
  --ac "Full Library test sweep passes unmodified" --ac "No import cycle: Library_Modules modules import without importing library_screen"
```

- [ ] **Step 2: Generate the authoritative inventory (do NOT hand-list)**

```bash
.venv/bin/python - <<'PY' > /tmp/support_inventory.txt
import ast
src = open("tldw_chatbook/UI/Screens/library_screen.py").read()
tree = ast.parse(src)
cls_line = next(n.lineno for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LibraryScreen")
for node in tree.body:
    if node.lineno >= cls_line: break
    if isinstance(node, (ast.Import, ast.ImportFrom)): continue
    name = getattr(node, "name", None)
    if name is None and isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name): name = node.targets[0].id
    if name is None and isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name): name = node.target.id
    if name: print(f"{type(node).__name__}\t{name}\t{node.lineno}\t{node.end_lineno}")
PY
wc -l /tmp/support_inventory.txt
```
Expected: ~92 rows. This inventory drives Steps 3–6; the destination for each row follows the file list above (ClassDef/alias → types module; the four canvas functions → canvas_sync; `_LibraryDatabaseNoteSessionPort` → note_session_port; other FunctionDefs → helpers; Assign/AnnAssign → constants; `logger` stays put).

- [ ] **Step 3: Write the failing surface test**

```python
# Tests/Architecture/test_library_support_layer_surface.py
"""PR 0a contract: the support layer lives in Library_Modules; the screen re-exports it.

Written BEFORE the move (against the inventory), so the move is proven by
this test flipping from fail to pass, and any future name deletion fails it.
"""
from __future__ import annotations
import importlib
import pytest

# Paste the name column of /tmp/support_inventory.txt here, mapped to its
# destination module, e.g.:
_SURFACE = {
    "tldw_chatbook.UI.Library_Modules.canvas_sync": [
        "_sync_library_canvas", "_apply_library_row_toggle",
        "_move_library_list_row_focus", "_patch_library_disabled_marker_label",
    ],
    "tldw_chatbook.UI.Library_Modules.note_session_port": ["_LibraryDatabaseNoteSessionPort"],
    # ... screen_constants / screen_support_types / screen_helpers rows ...
}

@pytest.mark.unit
@pytest.mark.parametrize("module,names", sorted(_SURFACE.items()))
def test_support_names_live_in_their_module(module: str, names: list[str]) -> None:
    mod = importlib.import_module(module)
    missing = [n for n in names if not hasattr(mod, n)]
    assert not missing, f"{module} missing {missing}"

@pytest.mark.unit
def test_screen_still_re_exports_every_moved_name() -> None:
    screen_mod = importlib.import_module("tldw_chatbook.UI.Screens.library_screen")
    for names in _SURFACE.values():
        missing = [n for n in names if not hasattr(screen_mod, n)]
        assert not missing, f"library_screen no longer re-exports {missing}"

@pytest.mark.unit
def test_no_import_cycle() -> None:
    import subprocess, sys
    for module in _SURFACE:
        proc = subprocess.run(
            [sys.executable, "-c",
             f"import sys, {module}; assert 'tldw_chatbook.UI.Screens.library_screen' not in sys.modules"],
            capture_output=True, text=True)
        assert proc.returncode == 0, f"{module} pulls in library_screen: {proc.stderr}"
```

- [ ] **Step 4: Run it to verify it fails** — `.venv/bin/python -m pytest Tests/Architecture/test_library_support_layer_surface.py -v` — Expected: FAIL, `ModuleNotFoundError` for the new modules.

- [ ] **Step 5: Move the blocks verbatim.** For each destination module: create the file with a module docstring naming this plan + the spec, copy each inventory block **byte-for-byte** (use the inventory's line ranges via `sed -n 'A,Bp'`, freshly regenerated — never stale ranges), then add the imports each block needs (copy them from `library_screen.py`'s import section; `_sync_library_canvas`'s `screen: "LibraryScreen"` annotation stays a string, with `if TYPE_CHECKING: from tldw_chatbook.UI.Screens.library_screen import LibraryScreen`). Delete the moved blocks from `library_screen.py` and add, where they stood, one grouped re-export import per destination module (explicit names, no `*`).

- [ ] **Step 6: Run the surface test until green** — `.venv/bin/python -m pytest Tests/Architecture/test_library_support_layer_surface.py -v` — Expected: PASS (all three tests).

- [ ] **Step 7: Run the regression net**

```bash
.venv/bin/python -m pytest Tests/UI/test_screen_preimport.py Tests/Packaging/test_research_workspace_import_closure.py -p no:randomly -q
.venv/bin/python -m pytest Tests/UI -k "library" -p no:randomly -q   # ~90 files; expect only pre-existing dev reds (compare against a baseline run on origin/dev if any fail)
./scripts/preflight.sh
```
Expected: identical pass/fail set to `origin/dev` baseline; preflight green.

- [ ] **Step 8: Commit, record blame-ignore, push, PR**

```bash
git add -A && git commit -m "refactor(library): PR 0a — move module-level support layer to Library_Modules"
git rev-parse HEAD >> .git-blame-ignore-revs && git add .git-blame-ignore-revs && git commit -m "chore: blame-ignore for the 0a support-layer move"
git push -u origin refactor/library-decomp-0a-support-layer
```
(If `.git-blame-ignore-revs` doesn't exist yet, this creates it; Task 4 documents its use.)

---

### Task 2: PR 0b(1) — add the missing Library ratchet row

**Files:**
- Modify: `Tests/Architecture/test_screen_size_ratchet.py` (`_BUDGETS` dict)

**Interfaces:**
- Consumes: post-0a merged tree.
- Produces: a failing ceiling for any future net growth of `library_screen.py`.

- [ ] **Step 1: Branch off dev after 0a merges** — `git fetch origin dev && git switch -c refactor/library-decomp-0b-guards origin/dev`

- [ ] **Step 2: Measure (post-rebase, the doctrine's hard rule)**

```bash
.venv/bin/python - <<'PY'
import ast
src = open("tldw_chatbook/UI/Screens/library_screen.py").read()
tree = ast.parse(src)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LibraryScreen")
methods = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
print(f"lines={len(src.splitlines())} methods={len(methods)}")
PY
```

- [ ] **Step 3: Add the budget row** (values from Step 2, verbatim — no rounding up):

```python
_BUDGETS: dict[str, tuple[str, int, int]] = {
    "tldw_chatbook/UI/Screens/chat_screen.py": ("ChatScreen", 16966, 563),
    #: Added 2026-09 by the Library decomposition plan (PR 0b): this row was
    #: missing for the entire month in which library_screen.py tripled from
    #: 15,819 to 46,109 lines while chat_screen.py shrank under its budget.
    #: New Library code belongs in tldw_chatbook/UI/Library_Modules/ — a
    #: subsystem's controller file may be created BEFORE its extraction
    #: series to receive new methods. See
    #: Docs/superpowers/specs/2026-09-01-library-screen-decomposition-design.md.
    "tldw_chatbook/UI/Screens/library_screen.py": ("LibraryScreen", <lines>, <methods>),
}
```

- [ ] **Step 4: Run the ratchet, both directions**

```bash
.venv/bin/python -m pytest Tests/Architecture/test_screen_size_ratchet.py -v
```
Expected: PASS. Then mutation-check the mechanism: append a dummy method to `library_screen.py`, re-run, expect FAIL naming the Library row, revert the dummy.

- [ ] **Step 5: Commit** — `git add Tests/Architecture/test_screen_size_ratchet.py && git commit -m "test(architecture): add the missing library_screen.py size-ratchet budget row"`

---

### Task 3: PR 0b(2) — widen the recompose ratchet to the whole Library surface

**Files:**
- Modify: `Tests/UI/test_library_recompose_ratchet.py`

- [ ] **Step 1: Write the widening.** The census function already takes `source: str`. Replace the single-path constant with a path list and sum the census across it:

```python
_LIBRARY_SURFACE_PATHS = sorted(
    [Path(__file__).resolve().parents[2] / "tldw_chatbook" / "UI" / "Screens" / "library_screen.py"]
    + list((Path(__file__).resolve().parents[2] / "tldw_chatbook" / "UI" / "Library_Modules").glob("*.py"))
)
```
The test iterates the paths, concatenates per-file inventories (prefix each entry with the file name), and compares the TOTAL against `LIBRARY_WHOLE_SCREEN_RECOMPOSE_MAX`. Keep the pin's current value; re-measure after widening — if `Library_Modules` already contains sites (e.g. `canvas_sync.py`'s sanctioned fallback arms moved in 0a), the total must equal the pre-0a single-file count. If it differs, STOP: a site was gained or lost in 0a — investigate before pinning. Update the docstring: the widened scope exists so decomposition moves cannot drain the census (spec, "PR 0b").

- [ ] **Step 2: Run** — `.venv/bin/python -m pytest Tests/UI/test_library_recompose_ratchet.py -v` — Expected: PASS at the unchanged pin. Mutation-check: add `self.refresh(recompose=True)` to any `Library_Modules` file, expect FAIL listing it with its filename, revert.

- [ ] **Step 3: Commit** — `git commit -am "test(library): recompose ratchet counts screen + Library_Modules as one surface"`

---

### Task 4: PR 0b(3) — probe instrument, recipe doc, blame-ignore plumbing

**Files:**
- Create: `Helper_Scripts/library_click_probe.py` (adapt the session scratchpad probe: keep `_click`'s settle/max-gap/mounts/CSS-vs-wait split and the report table; strip the scratchpad path bootstrapping and the service-call monkeypatch experiment; module docstring states usage: `.venv/bin/python Helper_Scripts/library_click_probe.py` and that headless numbers exclude terminal-write cost)
- Create: `backlog/docs/library-decomposition-recipe.md`
- Create/modify: `.git-blame-ignore-revs`

- [ ] **Step 1: Write the recipe doc.** Contents, in order: (1) the per-subsystem PR series (state → controller(s) → cleanup) with the byte-for-byte canon stated and `ConsoleDictationController.__init__` cited; (2) the field-ownership script (from Task 6 Step 2, verbatim) and the ≥2-subsystems shared-field rule; (3) the monkeypatch-name routing rule with the four known names; (4) the transform whitelist; (5) rollback-not-fix-forward; (6) "measure after final rebase, lower budgets in the landing PR"; (7) the subsystem order table from the spec with churn counts; (8) probe usage for before/after evidence.

- [ ] **Step 2: Configure blame ignores** — add to the recipe doc: `git config blame.ignoreRevsFile .git-blame-ignore-revs` (per-clone, one-time), and the rule that every move commit appends its hash in the same PR.

- [ ] **Step 3: Verify the probe runs** — `perl -e 'alarm 120; exec @ARGV' .venv/bin/python Helper_Scripts/library_click_probe.py` — Expected: the per-click table prints; no tracebacks.

- [ ] **Step 4: Commit; open PR 0b** with Tasks 2–4's commits; land it.

---

### Task 5: Conversations pre-series coverage spot-check

**Files:**
- Create (only if gaps found): `Tests/UI/test_library_conversations_characterization.py`

- [ ] **Step 1: Enumerate uncovered conversation methods**

```bash
.venv/bin/python - <<'PY'
import ast, subprocess
src = open("tldw_chatbook/UI/Screens/library_screen.py").read()
cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "LibraryScreen")
names = [m.name for m in cls.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and "conversation" in m.name]
for n in names:
    hits = subprocess.run(["grep", "-rl", n, "Tests/"], capture_output=True, text=True).stdout.strip()
    if not hits: print("UNCOVERED:", n)
PY
```

- [ ] **Step 2: For each UNCOVERED `@on` handler or externally-reachable behavior**, write one characterization test **against the screen through the DOM** (so it survives the move untouched), using the `LibraryHarness` recipe from `Tests/UI/test_library_shell.py` (`_build_test_app`, `_seed_conversations`, `_two_conversations`, `_wait_for_library_shell`; enter the mode via `screen.query_one("#library-row-browse-conversations", Button).press()`). Example shape, for paging:

```python
@pytest.mark.asyncio
async def test_next_page_press_advances_the_conversation_page() -> None:
    """Characterization (pre-extraction): pins CURRENT behavior, right or wrong."""
    app = LibraryHarness(_seeded_app_with_many_conversations())
    async with app.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(app)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-conversations", Button).press()
        await _wait_for_selector(screen, pilot, "#library-conversations-canvas")
        before = screen._library_conversation_page
        screen.query_one("#library-conversations-next", Button).press()
        await _settle(pilot)
        assert screen._library_conversation_page == before + 1
```
(Verify each widget id against the source before using it — ids drift.) Uncovered **private helpers** need no direct test; they're covered transitively or not worth pinning — record the skip decision in the backlog task.

- [ ] **Step 3: Run new tests, watch them pass against CURRENT code** (characterization tests are the inverted TDD case: they must PASS pre-move — that is their proof), commit as its own small PR or as the first commit of the state PR.

---

### Task 6: Conversations state PR — `LibraryConversationsState` + shims

**Files:**
- Create: `tldw_chatbook/UI/Library_Modules/library_conversations_state.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` (`__init__` field block → one construction; sentinel-wrapped shim block)
- Test: `Tests/Architecture/test_library_conversations_wiring.py` (first test only)

**Interfaces:**
- Produces: `LibraryConversationsState` dataclass; screen attribute `self._conversations_state`; screen shim properties preserving every original `_library_conversation*` name (getter + setter). Field name = original minus the `_library_conversation_`/`_library_conversations_` prefix.

- [ ] **Step 1: Compute exclusive ownership (the authority, not this plan's snapshot)**

```bash
.venv/bin/python - <<'PY'
import ast
from collections import defaultdict
src = open("tldw_chatbook/UI/Screens/library_screen.py").read()
cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "LibraryScreen")
methods = [m for m in cls.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
def attrs(m, store_only=False):
    for node in ast.walk(m):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            if not store_only or isinstance(node.ctx, ast.Store): yield node.attr
init = next(m for m in methods if m.name == "__init__")
fields = set(attrs(init, store_only=True))
conv_fields = {f for f in fields if f.startswith(("_library_conversation", "_conversation"))}
OTHER_SUBSYSTEM = ("_library_notes","_library_media","_library_prompt","_library_skill","_library_ingest","_library_export","_library_collections","_library_rag")
for f in sorted(conv_fields):
    users = [m.name for m in methods if m.name != "__init__" and f in set(attrs(m))]
    non_conv = [u for u in users if "conversation" not in u]
    print(f"{f}: non-conversation users={non_conv or 'NONE'}")
PY
```
Classification: `NONE` → moves into the state object. Non-conversation users that are **shell/plumbing methods** (rail switch, snapshot apply, shell-state build) → still moves (shims keep them working; cleanup retargets them). Non-conversation users belonging to **another subsystem** (name matches another subsystem prefix) → stays on the screen as shared shell state; record each such decision in the recipe doc's per-subsystem table. Expected ~27 movable fields (2026-09-01 snapshot: `deleted_selection_id, error, find_focus_intent, focus_after_apply, freshness, has_more, loading, page, page_loaded, page_records, page_size, query, reader_layout, reader_loaded_metadata, reader_mounted_authority, reader_preferences, reader_selected_metadata, reader_state, request_generation, requested_page, requested_query, selection_notice, stale_copy, total, total_known, row_selection, select_mode`).

- [ ] **Step 2: Write the failing wiring test**

```python
# Tests/Architecture/test_library_conversations_wiring.py
"""Conversations extraction series: state object exists and is screen-wired."""
from __future__ import annotations
import pytest
from tldw_chatbook.UI.Library_Modules.library_conversations_state import LibraryConversationsState

@pytest.mark.unit
def test_state_object_fields_match_the_shim_surface() -> None:
    import dataclasses
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
    field_names = {f.name for f in dataclasses.fields(LibraryConversationsState)}
    assert field_names, "state object is empty"
    for name in field_names:
        for prefix in ("_library_conversation_", "_library_conversations_"):
            if isinstance(getattr(LibraryScreen, prefix + name, None), property):
                break
        else:
            pytest.fail(f"no screen shim property found for state field {name!r}")
```

- [ ] **Step 3: Run it to verify it fails** — Expected: `ModuleNotFoundError: … library_conversations_state`.

- [ ] **Step 4: Create the dataclass.** For each movable field, copy the `__init__` right-hand side verbatim: literals become dataclass defaults; any expression referencing `self`/config becomes a constructor argument with the expression moved to the construction site in `__init__`. In `__init__`, replace the field-assignment block with `self._conversations_state = LibraryConversationsState(...)` **at the same position** (evaluation order preserved).

- [ ] **Step 5: Generate the shim block** (screen-side; sentinel-wrapped so cleanup deletes it wholesale):

```bash
.venv/bin/python - <<'PY' > /tmp/shims.py
import dataclasses
from tldw_chatbook.UI.Library_Modules.library_conversations_state import LibraryConversationsState
PLURAL = {"row_selection", "select_mode"}  # verify against Step-1 output
print("    # --- BEGIN generated conversations-state shims (delete wholesale at cleanup) ---")
for f in dataclasses.fields(LibraryConversationsState):
    orig = ("_library_conversations_" if f.name in PLURAL else "_library_conversation_") + f.name
    print(f"""    @property
    def {orig}(self):
        return self._conversations_state.{f.name}

    @{orig}.setter
    def {orig}(self, value):
        self._conversations_state.{f.name} = value
""")
print("    # --- END generated conversations-state shims ---")
PY
```
Paste the block into the class body, delete the now-duplicated plain-attribute semantics (the properties replace them).

- [ ] **Step 6: Run the wiring test (PASS), the characterization tests (PASS), and the sweep**

```bash
.venv/bin/python -m pytest Tests/Architecture/test_library_conversations_wiring.py -v
.venv/bin/python -m pytest Tests/UI -k "conversation and library" -p no:randomly -q
.venv/bin/python -m pytest Tests/UI -k "library" -p no:randomly -q
```
Expected: identical to the `origin/dev` baseline. The screen's line count barely moves here — the ratchet row from Task 2 must still PASS; if it fails, dev grew the file concurrently: rebase and re-measure, never raise.

- [ ] **Step 7: Commit, blame-ignore, PR** — `refactor(library): conversations state object + shims (exemplar series 1/4)`.

---

### Task 7: Reader controller move — `LibraryConversationReaderController`

**Files:**
- Create: `tldw_chatbook/UI/Library_Modules/library_conversation_reader_controller.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` (delete moved bodies; add delegators; construct controller in `__init__`)
- Test: extend `Tests/Architecture/test_library_conversations_wiring.py`

**Interfaces:**
- Consumes: `self._conversations_state` (Task 6), support modules (Task 1).
- Produces: `LibraryConversationReaderController` with the reader-cluster methods under their original names; screen delegators for every moved name; construction in `__init__` as `self._conversation_reader_controller = LibraryConversationReaderController(self, …named deps…)`.

- [ ] **Step 1: Enumerate the reader cluster** (mechanically: conversation methods whose name contains `reader`, plus `_conversation_message_count_label`-style pure helpers only they call — verify each candidate's callers with `grep -n "self\.<name>(" tldw_chatbook/UI/Screens/library_screen.py`). 2026-09-01 snapshot, ~24 methods: `_bootstrap_library_conversation_reader, _conversation_reader_bootstrap_is_current, _conversation_reader_list_summary, _conversation_reader_record, _conversation_reader_record_version, _conversation_reader_request_is_current, _conversation_reader_service, _ensure_library_conversation_reader_selection, _invalidate_library_conversation_reader_authority, _load_library_conversation_reader, _mirror_library_conversation_reader_preference, _retry_library_conversation_reader, _start_library_conversation_reader_selection, _sync_library_conversation_reader, _sync_library_conversation_reader_layout_from_shell, _conversation_message_count_label, _conversation_updated_label, _conversation_workspace_label, library_conversation_reader_messages_synced, retry_library_conversation_reader, show_library_conversation_reader_info, show_library_conversation_reader_read, find_in_library_conversation, _finish_library_conversation_find_focus`. The four `@on`-decorated ones (`library_conversation_reader_messages_synced`, `retry_…`, `show_…_info`, `show_…_read`, `find_in_…`) keep screen-side delegators.

- [ ] **Step 2: Write the failing wiring assertions**

```python
@pytest.mark.unit
def test_reader_controller_owns_its_cluster() -> None:
    from tldw_chatbook.UI.Library_Modules.library_conversation_reader_controller import (
        LibraryConversationReaderController,
    )
    for name in ("_load_library_conversation_reader", "_sync_library_conversation_reader"):
        assert callable(getattr(LibraryConversationReaderController, name, None))

@pytest.mark.unit
def test_screen_delegates_reader_handlers() -> None:
    import inspect
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
    src = inspect.getsource(LibraryScreen.show_library_conversation_reader_read)
    assert "_conversation_reader_controller" in src, "handler is not a delegator yet"
```

- [ ] **Step 3: Verify it fails** (`ModuleNotFoundError`), then build the controller per the canon. Constructor shape (mirror `ConsoleDictationController.__init__` exactly — its docstring is the reference):

```python
class LibraryConversationReaderController:
    def __init__(
        self,
        screen: "LibraryScreen",
        *,
        conversations_state_accessor: Callable[[], "LibraryConversationsState"],
        # one named Callable per non-reader screen name the moved bodies
        # reference — DISCOVER the list mechanically (Step 4), do not guess
    ) -> None: ...

    # framework services: live-read properties, never snapshotted
    @property
    def run_worker(self): return self._screen.run_worker
    @property
    def call_after_refresh(self): return self._screen.call_after_refresh
```

- [ ] **Step 4: Discover every name the bodies need bound**

```bash
.venv/bin/python - <<'PY'
import ast
CLUSTER = {…paste Step-1 list…}
src = open("tldw_chatbook/UI/Screens/library_screen.py").read()
cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "LibraryScreen")
methods = {m.name: m for m in cls.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
need = set()
for name in CLUSTER:
    for node in ast.walk(methods[name]):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            need.add(node.attr)
internal = CLUSTER | {a for a in need if a.startswith(("_library_conversation", "_conversation"))}
print("BIND THESE:", sorted(need - internal))
PY
```
For each printed name: framework service → live-read property; state field of another subsystem or shared shell field → named accessor callable; screen method → named callable bound in `__init__` as `lambda *a, **k: self.<name>(*a, **k)` — **except** any test-patched name (`_refresh_local_source_snapshot` etc.), which is bound the same way (the lambda late-binds through the screen, so patches keep working — this is why the canon binds at call time). State fields of THIS subsystem → controller-local generated properties reading `self._conversations_state_accessor().<field>` (same generator shape as Task 6 Step 5, on the controller).

- [ ] **Step 5: Move the bodies byte-for-byte**; replace each on the screen with a one-line delegator (`def show_library_conversation_reader_read(self, event): return self._conversation_reader_controller.show_library_conversation_reader_read(event)` — the `@on(...)` decorator line stays on the screen delegator, copied verbatim). Construct the controller in `__init__` after the state object.

- [ ] **Step 6: Green the wiring test, run the sweeps** (same three commands as Task 6 Step 6, same expectations — DOM tests unmodified). Also re-run the probe for a numbers checkpoint: `perl -e 'alarm 120; exec @ARGV' .venv/bin/python Helper_Scripts/library_click_probe.py` — expect click numbers unchanged within noise (a pure move must not move them).

- [ ] **Step 7: Commit, blame-ignore, PR** — `refactor(library): conversation reader controller (exemplar series 2/4)`.

---

### Task 8: Browse controller move — `LibraryConversationsController`

Same recipe as Task 7 applied to the remaining ~44 conversation methods (list/paging/selection/export/filter/handoff clusters; re-enumerate mechanically: all `conversation` methods minus Task 7's cluster minus anything another subsystem's methods call — check with the Step-4 discovery script). Cross-controller calls into the reader (`_start_library_conversation_reader_selection` from `handle_library_conversation_row`) are bound as named constructor callables the screen wires to the reader controller — controllers never import each other.

**Files:**
- Create: `tldw_chatbook/UI/Library_Modules/library_conversations_controller.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Test: extend `Tests/Architecture/test_library_conversations_wiring.py` (same two assertion shapes as Task 7 Step 2, against `LibraryConversationsController` and `handle_library_conversations_next`)

Steps mirror Task 7 exactly: failing wiring test → discovery script → canon constructor → byte-for-byte move with the 14 remaining `@on` delegators → sweeps → probe checkpoint → commit + blame-ignore + PR (`exemplar series 3/4`). Construction in `__init__`, after the reader controller so its methods exist to bind:

```python
self._conversations_controller = LibraryConversationsController(
    self,
    conversations_state_accessor=lambda: self._conversations_state,
    start_reader_selection=lambda *a, **k: self._conversation_reader_controller._start_library_conversation_reader_selection(*a, **k),
    # …one named callable per name the Task-7-style discovery script prints…
)
```

---

### Task 9: Conversations cleanup PR (series 4/4)

The one PR type allowed to edit tests. After Tasks 6–8 land:

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` (delete the sentinel shim block; retarget remaining screen-side `_library_conversation*` references to `self._conversations_state.<field>`; delete delegators nothing external references)
- Modify: every test file `grep -rl "_library_conversation" Tests/` returns (retarget attribute paths and patch targets; **assertions byte-for-byte**)
- Modify: `Tests/Architecture/test_screen_size_ratchet.py` (lower both Library numbers to the post-rebase measurement)
- Modify: `backlog/docs/library-decomposition-recipe.md` (record the series' actual numbers + any recipe corrections learned)

- [ ] **Step 1:** Delete the shim block; run `grep -n "_library_conversation" tldw_chatbook/UI/Screens/library_screen.py` and retarget every remaining screen-side reference to the state object. Delegators still referenced by `@on`/bindings/tests stay; run `grep -rn "<delegator>" Tests/ tldw_chatbook/` per delegator to prove deadness before deleting it.
- [ ] **Step 2:** Retarget tests file-by-file, running each file after edit; assertion bodies unchanged (an assertion that must change is a finding — stop and investigate, per doctrine).
- [ ] **Step 3:** Full Library sweep + `Tests/Architecture` + preflight; re-measure and LOWER the ratchet row in this PR.
- [ ] **Step 4:** Probe checkpoint (numbers still unchanged — cleanup is also behavior-neutral).
- [ ] **Step 5:** Commit, blame-ignore, PR — `refactor(library): conversations cleanup — shims out, ratchet lowered (exemplar series 4/4)`. Update the backlog tasks to Done per the repo DoD; note in the recipe doc that the exemplar is complete and the next subsystem (export, churn 3) follows it.

---

## Self-review record

- **Spec coverage:** PR 0a → Task 1; PR 0b (ratchet row, widened recompose pin, probe, recipe, blame plumbing) → Tasks 2–4; exemplar series → Tasks 5–9; shared-field rule → Task 6 Step 1; monkeypatch routing → Task 7 Step 4 (late-binding lambdas); byte-for-byte canon → Tasks 7/8; rollback policy and ordering table → recipe doc (Task 4). Not in this plan (later plans/recipe): the remaining nine subsystems, phase C.
- **Known drift risk:** every enumerated name list is labeled a 2026-09-01 snapshot with its regeneration script adjacent; scripts are the authority.
- **Type consistency:** state object named `LibraryConversationsState`, screen attr `_conversations_state`, controllers `LibraryConversationReaderController` / `LibraryConversationsController`, screen attrs `_conversation_reader_controller` / `_conversations_controller` — used consistently across Tasks 6–9.
