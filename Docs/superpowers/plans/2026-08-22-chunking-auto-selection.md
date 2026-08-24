# Chunking Auto-Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An "Auto" picker option that selects a template when one opts in and fits (full three-stage application), else derives a method plan from the vendored planner, else falls to today's plain defaults — recorded per-media, re-resolved on re-chunk.

**Architecture:** Vendor `auto_planner.py` from the existing pin (manifest move, zero new shims); one new `Chunking/auto_selection.py` module owning the three-tier decision (classifier → planner → plain); wire the reserved-name sentinel `"auto"` through #2's existing picker slot and resolution chain; persist `mode`/`auto_tier`/rationale in `chunking_config`. Single PR, five ordered tasks.

**Tech Stack:** Python ≥3.11, vendored `auto_planner.py` + `TemplateClassifier` (pin `dev`@`385afa95`, NOT moved), Media DB v7 store (no schema change), Textual `app.run_test()`.

**Spec:** `Docs/superpowers/specs/2026-08-22-chunking-auto-selection-design.md` — §0.1 (threshold correction), §0.2 (upstream divergence + reserved name), §4 (design), §7 (16 ACs), §8 (8 rulings). The plan argues from the spec.

## Global Constraints

- Never move the vendor pin (`dev` @ `385afa951922c8a9dc2002c675bb6cad65e4ac23`); `auto_planner.py` is a MOVE from `excluded` to `vendored`, never both lists.
- Vendored files are never hand-edited (sync-script rewrite rules only).
- `score > 0` is the candidacy guard; stored-invalid templates are EXCLUDED from candidacy; tie-break = priority (absent → 0) then listing order (name-ordered).
- The name `"auto"` is RESERVED at create/rename; a winning template runs IN FULL through #2's engine; tier 2 never runs when tier 1 won; `llm_available=False`; goal hardcoded `"balanced"`.
- `"None (manual settings)"` stays the picker default; its output stays byte-identical to today (#2's AC-36 pin must stay green).
- Config `default_template` never triggers auto; server mode unaffected; no goal UI/config key.
- `template` key in `chunking_config` written ONLY on template-tier wins (both #2 readers stay satisfied).
- Repo rule: targeted test runs only; full sweeps only on maintainer opt-in.
- Branch is stacked on `feat/chunking-template-parity` until PR #1938 merges — rebase to dev after.

---

### Task 1: Vendor `auto_planner.py` + unfence `TemplateClassifier`

**Files:**
- Modify: `Helper_Scripts/sync_chunking_engine.py` + `tldw_chatbook/Chunking/engine/VENDOR_MANIFEST.toml` (`auto_planner.py`: excluded → vendored; 37 → 38 files)
- Create (via sync): `tldw_chatbook/Chunking/engine/auto_planner.py`
- Test: `Tests/Chunking/test_sync_script.py` (extend), `Tests/Chunking/test_template_runtime.py` (fencing test update)

**Interfaces:**
- Produces: `from tldw_chatbook.Chunking.engine.auto_planner import plan_auto_chunking, AutoChunkingProfile, AutoChunkingDecision` (Task 3 consumes); fencing permits `TemplateClassifier` construction only inside `Chunking/auto_selection.py` (Task 3's target).

- [ ] **Step 1 — failing tests:**

```python
# extend Tests/Chunking/test_sync_script.py
def test_manifest_auto_planner_vendored_not_excluded():
    manifest = tomllib.loads((ENGINE / "VENDOR_MANIFEST.toml").read_text())
    vendored = set(manifest["files"]["vendored"])
    excluded = set(manifest["files"]["excluded"])
    assert "auto_planner.py" in vendored and "auto_planner.py" not in excluded
    assert not (vendored & excluded)  # spec §0.2: never both lists

def test_auto_planner_importable_zero_new_shims():
    from tldw_chatbook.Chunking.engine import auto_planner
    from tldw_chatbook.Chunking.engine.auto_planner import plan_auto_chunking
    assert callable(plan_auto_chunking)
    # stdlib-only at the pin — the module must not import _shims at all
    import inspect
    assert "_shims" not in inspect.getsource(auto_planner)
```

Update the fencing test in `test_template_runtime.py`: the "no production module constructs TemplateClassifier" source-scan's allowed-home becomes `Chunking/auto_selection.py` (which doesn't exist yet — the scan asserts zero constructors outside the (future) home, so it stays green now and guards Task 3's placement). `TemplateLearner`/`TemplateManager` stay fully fenced (zero homes).

- [ ] **Step 2 — red** for the manifest test; fencing test stays green by construction.
- [ ] **Step 3 — manifest move + sync:** create the pinned worktree (`git -C ~/Documents/GitHub/tldw_server2 worktree add /tmp/tldw_server_sync 385afa951922c8a9dc2002c675bb6cad65e4ac23`), move the entry in BOTH the manifest and the script's VENDORED list, run `python Helper_Scripts/sync_chunking_engine.py --source /tmp/tldw_server_sync`. Verify 38 files; byte-faithful modulo rewrite rules (auto_planner is stdlib-only, so expect ZERO rewritten lines — assert in the report); remove the worktree after tests.
- [ ] **Step 4 — green:** `pytest Tests/Chunking/test_sync_script.py -q` (non-network + idempotency w/ local source) + `pytest Tests/Chunking/ -q --ignore=Tests/Chunking/test_sync_script.py` + `pytest Tests/Performance/test_app_import_weight.py -q`.
- [ ] **Step 5 — commit:** `feat(chunking): vendor auto_planner.py from the existing pin (37→38 files); unfence TemplateClassifier for auto_selection`

### Task 2: `Chunking/auto_selection.py` + reserved name + defect filing

**Files:**
- Create: `tldw_chatbook/Chunking/auto_selection.py`
- Modify: `tldw_chatbook/RAG_Admin/template_validation.py` is NOT the reservation home — the reservation is CRUD-level: `tldw_chatbook/Chunking/chunking_interop_library.py` (`create_template`/`update_template` refuse `name == "auto"` with a named error)
- Modify: `tldw_chatbook/Chunking/engine/UPSTREAM_DEFECTS.md` (entry #16)
- Test: `Tests/Chunking/test_auto_selection.py`

**Interfaces:**
- Consumes: `TemplateClassifier.score` (vendored), `plan_auto_chunking` (Task 1), #2's listing decoration (`_decorate_template_record` validity flag — read via the interop listing, `template_runtime.resolve_template` for the winner's dict).
- Produces (Task 4 consumes):
  - `AutoDecision` dataclass: `{tier: Literal["template","plan","plain"], template: dict | None, chunk_options: dict | None, rationale: list[str], fallback_reasons: list[str]}`
  - `AUTO_SENTINEL = "auto"`
  - `resolve_auto(db, *, media_type: str | None, title: str | None, filename: str | None, url: str | None, goal: str = "balanced") -> AutoDecision`

- [ ] **Step 1 — failing tests** (core contracts; full table per spec §6):

```python
# Tests/Chunking/test_auto_selection.py
import pytest
from tldw_chatbook.Chunking import auto_selection as aus
from tldw_chatbook.Chunking.auto_selection import resolve_auto, AUTO_SENTINEL


def _store_with(db_cls_path, rows):  # helper: in-memory v7 store, seeded rows
    ...  # follow Tests/Chunking/test_template_runtime.py's fixture pattern


def test_template_tier_selects_positive_score_full_block():
    # one template with classifier {media_types: ["document"], min_score: 0.4}
    d = resolve_auto(db, media_type="document", title="t", filename=None, url=None)
    assert d.tier == "template" and d.template is not None and d.chunk_options is None

def test_no_block_never_selected_absent_min_score_selects():  # spec §0.1 both pins
    # template A: no classifier block → score 0.0 → never
    # template B: block present, NO min_score, media matches → selects (parity clamp)
    ...

def test_stored_invalid_excluded_from_candidacy():
    ...  # invalid template with matching block loses to nothing → tier == "plan"

def test_tiebreak_priority_then_listing_order():
    ...  # two equal-score templates: higher priority wins; equal priority → name order

def test_plan_tier_when_no_candidate():
    d = resolve_auto(db, media_type="pdf", title="t", filename=None, url=None)
    assert d.tier == "plan" and d.template is None
    assert d.chunk_options["method"] and "max_size" in d.chunk_options

def test_planner_never_runs_when_template_won(monkeypatch):
    called = []
    monkeypatch.setattr(aus, "plan_auto_chunking", lambda **kw: called.append(kw))
    resolve_auto(db_with_matching_template, media_type="document", title="t", filename=None, url=None)
    assert not called  # mutation target for the reverse pin

def test_six_builtins_never_auto_selected():
    ...  # seed all six; auto over every media_type ⇒ tier != "template"

def test_malformed_block_skipped_with_reason():
    ...  # one poisoned block + one healthy → healthy wins, reason names the skip

def test_plain_tier_when_perform_chunking_context_declines():
    ...  # empty store + planner-decline fixture → tier "plain", chunk_options None
```

Plus CRUD reservation tests (interop suite): `create_template(name="auto")` and `update_template` renaming to `"auto"` both refuse with a named `InvalidTemplateError`-family error.

- [ ] **Step 2 — red**, then implement `auto_selection.py`: tier 1 iterates the interop listing (deleted-filtered, validity-decorated), excludes `template_valid is False`, scores via `TemplateClassifier.score(cfg, media_type=…, title=…, url=…, filename=…)` inside a per-candidate guard, `score <= 0 → continue`, key `(score, priority or 0)` strictly-greater; winner resolved to a dict via `resolve_template`; tier 2 calls `plan_auto_chunking(perform_chunking=True, chunking_mode="auto", goal=goal, media_type=MEDIA_TYPE_MAP.get(media_type, media_type), requested_llm=False, llm_available=False, semantic_available=<embeddings-config read>)` — `MEDIA_TYPE_MAP` starts as the identity-plus-web normalization and is Task 3's vocabulary table seed; tier 3 returns `AutoDecision(tier="plain", …)`. Module docstring cites spec §0.2's three divergences.
- [ ] **Step 3 — reservation + defect entry:** interop create/update refuse `"auto"` (include the pre-existing-collision doc: a legacy row named `"auto"` is listed flagged shadowed — the listing decoration gains a `name_reserved: True` field when `name == AUTO_SENTINEL`); UPSTREAM_DEFECTS.md gains #16 (auto/explicit paths apply only the hierarchical block).
- [ ] **Step 4 — mutation-verify** the never-runs pin (invert it once, watch red); **green**: `pytest Tests/Chunking/test_auto_selection.py -q` + `Tests/RAG_Admin/ -q` + `Tests/Chunking/ -q --ignore=test_sync_script.py`.
- [ ] **Step 5 — commit:** `feat(chunking): auto_selection — three-tier decision, reserved 'auto' name, defect #16`

### Task 3: Vocabulary table + planner parity fixtures

**Files:**
- Modify: `tldw_chatbook/Chunking/auto_selection.py` (finalize `MEDIA_TYPE_MAP` with the verified table)
- Create: `Tests/Chunking/auto_planner_parity_fixtures.json`, `Tests/Chunking/test_auto_planner_parity.py`
- Test: `Tests/Chunking/test_media_type_vocabulary.py`

**Interfaces:**
- Consumes: vendored `plan_auto_chunking` (Task 1).
- Produces: the frozen `MEDIA_TYPE_MAP: dict[str, str]` (Task 4 relies on it being total over chatbook's ingest strings).

- [ ] **Step 1 — enumerate chatbook's vocabulary:** grep the ingest paths for every `media_type` string they can produce (`Local_Ingestion/local_file_ingestion.py`, `app.py` group mapping, audio/video/image/document families). Cross-reference the planner's `_normalize_media_type` + `_choose_method` switches (read the vendored source; record the planner's recognized values: "web", "email", "pdf", "ebook", "audio", "video", "document"… as found). Build the identity-or-mapping table; anything unmappable maps to itself with a comment.
- [ ] **Step 2 — vocabulary test (AC 15):**

```python
def test_every_ingest_media_type_is_mapped():
    from tldw_chatbook.Chunking.auto_selection import MEDIA_TYPE_MAP, KNOWN_INGEST_MEDIA_TYPES
    assert set(KNOWN_INGEST_MEDIA_TYPES) <= set(MEDIA_TYPE_MAP)  # total coverage, identity entries count
    assert "web_document" not in MEDIA_TYPE_MAP.values() or MEDIA_TYPE_MAP["web_document"] == "web"  # normalization preserved
```

`KNOWN_INGEST_MEDIA_TYPES` is a frozen tuple in `auto_selection.py`; a new ingest family without a mapping entry fails this test loudly.
- [ ] **Step 3 — parity fixtures (AC 6, the #2-goldens pattern):** fixed inputs (media_type × goal="balanced" × the capability flags chatbook passes) → `plan_auto_chunking` outputs, generated from the vendored module with test mode explicitly off (production_path marker), byte-pinned JSON (decision fields incl. rationale lists), re-run at every sync. Test asserts equality.
- [ ] **Step 4 — green + commit:** `feat(chunking): media-type vocabulary table + planner parity fixtures`

### Task 4: Seam wiring — picker sentinel, resolution, persistence, re-chunk

**Files:**
- Modify: `tldw_chatbook/Chunking/template_runtime.py` (`resolve_ingest_template` detects `AUTO_SENTINEL` at the picker tier → `resolve_auto`; a new `resolve_for_rechunk(db, chunking_config)` honoring `mode: "auto"`), `tldw_chatbook/Widgets/Library/library_ingest_canvas.py` (picker option), `tldw_chatbook/Local_Ingestion/local_file_ingestion.py` (persist `mode`/`auto_tier`/`auto_rationale`; `template` key only on template-tier wins), `tldw_chatbook/Library/library_rechunk_service.py` (re-chunk uses `resolve_for_rechunk`)
- Test: `Tests/Local_Ingestion/test_ingest_template_resolution.py` (extend), `Tests/UI/test_library_ingest_template_picker.py` (extend), `Tests/Library/test_library_rechunk_service.py` (extend)

**Interfaces:**
- Consumes: `resolve_auto`, `AUTO_SENTINEL`, `AutoDecision` (Task 2), `MEDIA_TYPE_MAP` (Task 3), #2's `resolve_ingest_template(db, picker_choice, *, per_media=None)`.
- Produces: `resolve_for_rechunk(db, chunking_config: dict | None) -> dict | None | AutoDecision`-shaped result (Task 5's UI asserts against it).

- [ ] **Step 1 — failing tests:** chain integration per tier (picker-Auto → classifier-win → template honored on the pdf seam, persisted rows differ from plan-tier; picker-Auto → no candidate → plan-tier options materialize with #2 precedence; picker-None byte-identical — #2's AC-36 pin re-run); persistence shape (`mode: "auto"`, `auto_tier`, `auto_rationale`, `template` key absent unless template-tier; both #2 readers round-trip); re-chunk re-resolution (stored `mode:"auto"` + classifier block flipped ⇒ tier changes on re-chunk; stored explicit name behaves exactly as #2); UI (picker offers Auto, None still default, Auto label correct); config-default-never-triggers-auto pin.
- [ ] **Step 2 — red, implement, green:** `pytest Tests/Local_Ingestion/ Tests/Chunking/ -q --ignore=test_sync_script.py` + picker UI tests + `Tests/Library/test_library_rechunk_service.py -q`.
- [ ] **Step 3 — commit:** `feat(chunking): Auto wired through picker, resolution, persistence, re-chunk`

### Task 5: Docs, CHANGELOG, close-out

**Files:**
- Modify: `Docs/User_Guide/library/import-and-export.md` (Auto option, opt-in-via-classifier explanation, re-verified stamp), `CHANGELOG.md`
- Test: close-out run

- [ ] **Step 1 — docs + CHANGELOG** (Auto behavior, the opt-in story, the reserved name).
- [ ] **Step 2 — targeted close-out:** `pytest Tests/Chunking/ Tests/RAG_Admin/ Tests/Local_Ingestion/test_ingest_template_resolution.py Tests/UI/test_library_ingest_template_picker.py Tests/Library/test_library_rechunk_service.py Tests/Performance/test_app_import_weight.py -q --ignore=Tests/Chunking/test_sync_script.py` — zero new failures vs the #2-branch baseline (known pre-existing excluded).
- [ ] **Step 3 — commit:** `docs(chunking): auto-selection user guide + changelog`

## Self-Review (run at save)

1. **Spec coverage:** AC 1→T1; AC 2→T1(fencing)+T2(module); AC 3→T2; AC 4→T2(§0.1 pins); AC 5→T2+T3; AC 6→T3; AC 7→T4; AC 8→T4; AC 9→T4; AC 10→T4; AC 11→T4(config pin)+T2(server-mode n/a by construction — no code, verified in T4 review); AC 12→T5; AC 13→T5 close-out; AC 14→T2(reservation)+T2(listing flag); AC 15→T3; AC 16→T2(defect entry). All 16 mapped.
2. **Ordering:** T1 (vendor) before T2 (consumes planner); T2 before T3 (fixtures freeze the map); T3 before T4 (total map); T5 last.
3. **Type consistency:** `resolve_auto(db, *, media_type, title, filename, url, goal="balanced") -> AutoDecision`; `AUTO_SENTINEL = "auto"`; `MEDIA_TYPE_MAP: dict[str, str]`; `resolve_for_rechunk(db, chunking_config)` — consistent across T2/T3/T4.
4. **Placeholders:** the `_store_with` helper and `...` bodies in T2's test sketch are completion-per-contract (the fixture pattern is named: test_template_runtime.py's); all interface signatures and load-bearing assertions are concrete.
