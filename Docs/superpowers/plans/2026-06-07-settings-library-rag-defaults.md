# Settings Library/RAG Defaults Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Settings > Library & RAG from a read-only contract into a validated, persisted global-defaults editor for Library search and RAG answer behavior.

**Architecture:** Add a focused Library/RAG settings helper so parsing, validation, defaults, and save payload construction do not grow the already-large Settings screen. Settings renders and saves global defaults only; Library owns active queries/results and Console owns staged-context/chat execution. Add explicit config-model support for new display defaults before exposing them in Settings.

**Tech Stack:** Python 3.11+, Textual, existing `SettingsConfigAdapter`, existing `AppRAGSearchConfig.rag` config mapping, pytest, Textual-web/CDP screenshot QA.

---

## Scope

Implement one PR-sized Settings slice for `TASK-79`.

In scope:

- Search mode default: `plain`, `semantic`, or `hybrid`.
- Result limits: default top-k, FTS top-k, vector top-k.
- Hybrid balance: `hybrid_alpha`.
- Score threshold.
- Citation defaults: enabled/disabled plus display style.
- Snippet display default: max snippet characters.
- Context limit default for RAG context assembly.
- Save/revert/test feedback, validation, and actual rendered screenshot QA.

Out of scope:

- RAG indexing execution.
- Embedding model lifecycle or downloads.
- Chunking template editing.
- Library source staging, selected result state, or active query execution.
- Console staged-context payload changes.
- Workspace eligibility changes.
- Server sync or handoff policy changes.

## ADR Check

ADR required: yes

ADR path: `backlog/decisions/003-settings-library-rag-defaults.md`

Reason: This task defines the persisted Settings/Library/RAG configuration boundary and introduces `rag.search.snippet_max_chars` under the existing `AppRAGSearchConfig.rag` mapping.

## File Responsibilities

- `tldw_chatbook/RAG_Search/simplified/config.py`
  - Add explicit `SearchConfig` fields and `from_settings()` reads for `citation_style` and `snippet_max_chars`.
  - Preserve existing retrieval defaults and do not move indexing or embedding ownership into Settings.
- `tldw_chatbook/config.py`
  - Add defaults/template entries for `rag.search.citation_style` and `rag.search.snippet_max_chars` if the default config template does not already define them.
- `tldw_chatbook/UI/Screens/settings_library_rag_defaults.py`
  - New focused helper for Library/RAG defaults.
  - Owns load normalization, validation, labels, and save-section payload construction.
- `tldw_chatbook/UI/Screens/settings_config_models.py`
  - Add small state model only if the helper needs a shared dataclass that tests should import.
- `tldw_chatbook/UI/Screens/settings_screen.py`
  - Render Library/RAG controls.
  - Wire drafts, save/revert/test actions, and copy.
  - Keep active Library and Console workflow ownership explicit.
- `tldw_chatbook/Widgets/Library/library_search_rag_panel.py`
  - Read snippet display default only if the current widget has a hardcoded snippet length that should respect Settings.
  - Do not add active search execution here unless a test proves display defaults need it.
- `tldw_chatbook/Library/library_rag_state.py`
  - Read or expose display defaults only if the current state model owns snippet/citation display values.
- `Tests/UI/test_settings_library_rag_defaults.py`
  - New pure tests for helper load, validation, and save payload behavior.
- `Tests/UI/test_settings_configuration_hub.py`
  - Mounted Settings tests for controls, save/revert, disabled state recovery, and ownership copy.
- `Tests/Widgets/test_library_search_rag_panel.py`
  - Add only if snippet/citation display defaults are consumed by the Library widget.
- `Docs/superpowers/qa/product-maturity/screen-qa/settings/notes.md`
  - Record actual CDP/Textual-web screenshot evidence and approval.
- `backlog/tasks/task-79 - Functionalize-Settings-Library-and-RAG-defaults.md`
  - Track status, checked acceptance criteria, implementation notes, verification, and ADR link.

## Config Contract

Read and write through the existing `AppRAGSearchConfig` section, nested under `rag`:

```python
{
    "AppRAGSearchConfig": {
        "rag": {
            "search": {
                "default_search_mode": "hybrid",
                "default_top_k": 10,
                "score_threshold": 0.0,
                "include_citations": True,
                "citation_style": "inline",
                "snippet_max_chars": 240,
                "max_context_size": 16000,
            },
            "retriever": {
                "fts_top_k": 10,
                "vector_top_k": 10,
                "hybrid_alpha": 0.5,
            },
        },
    },
}
```

Validation rules:

- `default_search_mode`: one of `plain`, `semantic`, `hybrid`.
- `default_top_k`, `fts_top_k`, `vector_top_k`: integer from `1` to `100`.
- `hybrid_alpha`: float from `0.0` to `1.0`.
- `score_threshold`: float from `0.0` to `1.0`.
- `include_citations`: boolean.
- `citation_style`: one of `inline`, `footnote`, `none`.
- `snippet_max_chars`: integer from `80` to `2000`.
- `max_context_size`: integer from `1000` to `200000`.

## Task 1: Add RAG Config Display Defaults And Pure Settings Helper

**Files:**
- Modify: `tldw_chatbook/RAG_Search/simplified/config.py`
- Modify: `tldw_chatbook/config.py`
- Create: `tldw_chatbook/UI/Screens/settings_library_rag_defaults.py`
- Test: `Tests/UI/test_settings_library_rag_defaults.py`
- Reference: `tldw_chatbook/RAG_Search/simplified/config.py`

- [ ] **Step 1: Write failing RAG config/default tests**

Add tests that prove `RAGConfig.from_settings()` loads display defaults from `AppRAGSearchConfig.rag.search`:

- `citation_style`
- `snippet_max_chars`

Run:

```bash
python -m pytest -q Tests/UI/test_settings_library_rag_defaults.py --tb=short
```

Expected: fail because those fields are not represented in the simplified RAG config model yet.

- [ ] **Step 2: Add config-model support for display defaults**

Add `citation_style` and `snippet_max_chars` to `SearchConfig`, load them in `RAGConfig.from_settings()`, and update default config/template values where needed.

- [ ] **Step 3: Write failing pure load/default tests**

Add tests that build minimal app config dictionaries and assert normalized defaults.

Cover:

- Empty config returns safe defaults.
- Existing `AppRAGSearchConfig.rag.search` values are loaded.
- Existing `AppRAGSearchConfig.rag.retriever` values are loaded.
- Legacy missing values fall back to the same defaults as `RAGConfig.from_settings()`.

Run:

```bash
python -m pytest -q Tests/UI/test_settings_library_rag_defaults.py --tb=short
```

Expected: fail because the helper module does not exist.

- [ ] **Step 4: Implement the defaults dataclass and loader**

Create a frozen dataclass, for example:

```python
@dataclass(frozen=True)
class SettingsLibraryRagDefaults:
    default_search_mode: str = "semantic"
    default_top_k: int = 10
    fts_top_k: int = 10
    vector_top_k: int = 10
    hybrid_alpha: float = 0.5
    score_threshold: float = 0.0
    include_citations: bool = True
    citation_style: str = "inline"
    snippet_max_chars: int = 240
    max_context_size: int = 16000
```

Add `load_library_rag_defaults(app_config: Mapping[str, object]) -> SettingsLibraryRagDefaults`.

- [ ] **Step 5: Run pure load/default tests**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_library_rag_defaults.py --tb=short
```

Expected: pass the load/default tests.

- [ ] **Step 6: Add failing validation/save-payload tests**

Add tests for:

- Invalid enum values.
- Out-of-range integer and float values.
- Valid draft values produce one nested `AppRAGSearchConfig` save payload.
- Payload deep-merges with existing `rag.search` and `rag.retriever` keys instead of dropping unrelated RAG config.

Run the same focused test file. Expected: fail until validation and payload helpers exist.

- [ ] **Step 7: Implement validation and payload helpers**

Add helpers such as:

```python
def validate_library_rag_values(values: Mapping[str, object]) -> SettingsValidationResult: ...
def build_library_rag_save_sections(
    app_config: Mapping[str, object],
    values: SettingsLibraryRagDefaults,
) -> dict[str, dict[str, object]]: ...
```

Use the existing `SettingsValidationResult` model.

- [ ] **Step 8: Run pure helper tests**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_library_rag_defaults.py --tb=short
```

Expected: pass.

## Task 2: Convert Library/RAG From Read-Only Contract To Guided Mutating Category

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `Tests/UI/test_settings_configuration_hub.py`

- [ ] **Step 1: Write failing ownership tests**

Update existing domain category tests so `SettingsCategoryId.LIBRARY_RAG` is allowed to mutate while the remaining domain categories stay read-only.

Assertions:

- `Library & RAG` ownership record has `writes_allowed=True`.
- Save/Revert buttons are enabled when the category has dirty values.
- Copy says Settings owns global defaults only.
- Copy says Library owns active queries/source browsing and Console owns staged context.

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py::test_settings_domain_category_contracts_are_explicit_and_read_only Tests/UI/test_settings_configuration_hub.py::test_settings_domain_categories_are_grouped_and_have_ownership_records --tb=short
```

Expected: fail until the ownership records and tests are updated.

- [ ] **Step 2: Update ownership/domain records**

In `settings_screen.py`:

- Update the `SettingsDomainCategoryContract` for `LIBRARY_RAG`.
- Remove `Library & RAG` from the read-only-only assumption.
- Add `SettingsCategoryId.LIBRARY_RAG` to `GUIDED_SETTINGS_MUTATION_CATEGORIES`.
- Keep all other domain categories read-only.

- [ ] **Step 3: Run ownership tests**

Run the focused tests from Step 1.

Expected: pass.

## Task 3: Render Library/RAG Guided Controls

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Test: `Tests/UI/test_settings_configuration_hub.py`

- [ ] **Step 1: Write failing mounted render test**

Add a mounted test that clicks `#settings-category-library-rag` and asserts the presence of stable selectors:

- `#settings-library-rag-search-mode`
- `#settings-library-rag-default-top-k`
- `#settings-library-rag-fts-top-k`
- `#settings-library-rag-vector-top-k`
- `#settings-library-rag-hybrid-alpha`
- `#settings-library-rag-score-threshold`
- `#settings-library-rag-include-citations`
- `#settings-library-rag-citation-style`
- `#settings-library-rag-snippet-max-chars`
- `#settings-library-rag-max-context-size`
- `#settings-library-rag-save-result`

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py::test_settings_library_rag_guided_controls_render --tb=short
```

Expected: fail because the controls do not exist.

- [ ] **Step 2: Implement the control card**

In `_render_detail_pane()`, route `SettingsCategoryId.LIBRARY_RAG` to a new `_render_library_rag_detail()` instead of `_render_domain_category_detail()`.

Use the existing Settings visual pattern:

- Section title.
- State banner.
- Structured detail rows.
- Inputs/selects/buttons in `settings-focus-card`.
- Result/status row.
- Ownership boundary rows.

- [ ] **Step 3: Add draft staging handlers**

Add `@on(Input.Changed, ...)` and `@on(Select.Changed, ...)` handlers for the Library/RAG fields.

Use one category draft:

```python
SettingsCategoryId.LIBRARY_RAG
```

Keep normal typing visible and do not let footer shortcuts trigger while inputs own focus; existing focus guard should cover this.

- [ ] **Step 4: Run mounted render test**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py::test_settings_library_rag_guided_controls_render --tb=short
```

Expected: pass.

## Task 4: Save, Revert, And Validation Feedback

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `Tests/UI/test_settings_configuration_hub.py`

- [ ] **Step 1: Write failing save/revert tests**

Add mounted tests for:

- Editing `default_top_k` marks the category dirty and enables Save/Revert.
- Saving persists to `SettingsConfigAdapter.save_sections()` with the expected nested `AppRAGSearchConfig` payload.
- Revert restores visible widget values and clears dirty status.
- Invalid `hybrid_alpha` blocks save, keeps dirty state, and updates `#settings-library-rag-save-result`.

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py::test_settings_library_rag_save_persists_valid_defaults Tests/UI/test_settings_configuration_hub.py::test_settings_library_rag_invalid_values_block_save Tests/UI/test_settings_configuration_hub.py::test_settings_library_rag_revert_restores_loaded_values --tb=short
```

Expected: fail until save/revert routing exists.

- [ ] **Step 2: Implement save routing**

In `action_settings_save_category()`, add a `SettingsCategoryId.LIBRARY_RAG` branch after provider handling and before the Console Behavior branch.

The branch should:

- Read current widget values.
- Validate through the helper.
- Build save sections through the helper.
- Save with `SettingsConfigAdapter().save_sections(...)`.
- Update `self.app_instance.app_config`.
- Clear the `LIBRARY_RAG` draft on success.
- Update `#settings-library-rag-save-result`.
- Notify success/failure.

- [ ] **Step 3: Implement revert routing**

In `action_settings_revert_category()`, add a `LIBRARY_RAG` branch that:

- Drops the draft.
- Re-syncs Library/RAG widgets from loaded values.
- Updates the save result.
- Updates draft status.

- [ ] **Step 4: Run save/revert tests**

Run the focused tests from Step 1.

Expected: pass.

## Task 5: Cross-Surface Display Defaults

**Files:**
- Inspect: `tldw_chatbook/Widgets/Library/library_search_rag_panel.py`
- Inspect: `tldw_chatbook/Library/library_rag_state.py`
- Test as needed: `Tests/Widgets/test_library_search_rag_panel.py`

- [ ] **Step 1: Determine whether Library currently hardcodes snippet/citation display**

Search for snippet truncation, citation label rendering, and default result count behavior.

Run:

```bash
rg -n "snippet|citation|top_k|default_top_k|max_context|include_citations" tldw_chatbook/Widgets/Library tldw_chatbook/Library Tests/Widgets
```

- [ ] **Step 2: Add widget/state tests only for confirmed hardcoded behavior**

If Library currently hardcodes snippet truncation or citation display despite available state, add focused tests proving it consumes the new defaults.

If it does not consume those defaults yet, document that consumption remains a follow-up and do not expand this PR into Library search execution.

- [ ] **Step 3: Implement minimal display-default consumption only if needed**

Do not change active query execution in this task. Only wire display defaults that are already represented in Library/RAG state.

## Task 6: CDP QA, Task Notes, And Verification

**Files:**
- Modify: `Docs/superpowers/qa/product-maturity/screen-qa/settings/notes.md`
- Modify: `backlog/tasks/task-79 - Functionalize-Settings-Library-and-RAG-defaults.md`

- [ ] **Step 1: Run focused automated verification**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_library_rag_defaults.py Tests/UI/test_settings_configuration_hub.py --tb=short
git diff --check
```

If Library widget display defaults were changed, also run:

```bash
python -m pytest -q Tests/Widgets/test_library_search_rag_panel.py --tb=short
```

- [ ] **Step 2: Capture actual rendered screenshots**

Use Textual-web/CDP with an isolated HOME/XDG profile and `default_tab = "settings"`.

Required screenshots:

- Library & RAG category baseline after implementation.
- Edited valid value with Save enabled.
- Invalid numeric value recovery state.
- Saved state with no unsaved changes.

Save under:

```text
Docs/superpowers/qa/product-maturity/screen-qa/settings/
```

Use descriptive filenames beginning with:

```text
settings-library-rag-defaults-
```

- [ ] **Step 3: Get user approval**

Do not open a PR until the user has approved the actual rendered screenshots.

- [ ] **Step 4: Update Backlog task**

Update `TASK-79`:

- Check all acceptance criteria that are actually complete.
- Add implementation notes.
- Add verification commands and results.
- Keep ADR path linked.
- Mark Done only after all DoD items are complete.

- [ ] **Step 5: Commit and create PR**

Commit in reviewable chunks if implementation spans helper/UI/QA evidence.

Final PR body must include:

- Summary of Settings defaults implemented.
- Explicit out-of-scope list.
- Verification commands and results.
- Screenshot paths and user approval status.

## Residual Risks And Follow-Ups

- RAG indexing, embedding model lifecycle, and chunking templates remain separate work.
- Library active query execution may not consume every global default until the Library/RAG execution path is deepened.
- Citation/snippet carry-through into Console answers, Artifacts, exported Chatbooks, and saved notes remains a later task.
- Workspace eligibility must remain a Console/Library source-staging rule, not a Settings filter.
