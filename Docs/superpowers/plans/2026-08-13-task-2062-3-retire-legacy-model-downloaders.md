# TASK-2062.3 Legacy Models Downloader Retirement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the Models destination's two unverified direct-write download paths while preserving managed Import, arbitrary External GGUF use, legacy unmanaged-file discovery, Hugging Face inference, Hugging Face model IDs, and provider-owned caching.

**Architecture:** Delete rather than replace. Remove the legacy `Widgets/HuggingFace` browser/downloader package, its private browser client, the Download Models rail/view, and the Transformers `huggingface-cli download --local-dir` controls/worker. Keep the existing Installed, External, and Remote owners unchanged except for actionable empty copy; keep `model_download_dir` solely as the bounded legacy scan root and leave provider/runtime resolution outside the artifact store.

**Tech Stack:** Python 3.11+, Textual 8.x, existing `ModelArtifactService`/Installed/External/Remote views, pytest, Ruff, stdlib `ast`/`pathlib` retirement ratchets.

## Global constraints

- Read `AGENTS.md`, TASK-2062.3, the approved Phase 3 design, ADR-025, and the testing/live/backlog lessons before editing.
- Use strict TDD: write each retirement or preservation test first, run it to a genuine RED for the intended old behavior, then make the smallest deletion/edit required for GREEN.
- Delete only unverified direct writers. Do not add a replacement downloader, migration, auto-import, cache deletion, compatibility façade, or redirect from old symbols.
- Preserve arbitrary external GGUF launch in place. Users must not be forced to copy external files into the managed store.
- Preserve `InstalledView.scan_unmanaged()` and `model_download_dir` as the read-only legacy discovery root. Do not delete or rewrite existing caches or original files.
- Preserve the separate Hugging Face inference provider/configuration, managed Remote catalog/acquisition, provider-owned vLLM/MLX/Transformers model-ID caching, and runtime acceptance of Hugging Face IDs or local directories.
- The retired `tldw_chatbook.LLM_Calls.huggingface_api` module is the browser-specific direct-download client, not the Hugging Face inference provider in `LLM_API_Calls.py`/`Summarization_General_Lib.py`.
- Keep Models keyboard navigation and focus intact after removing one rail row and one deferred view. Test the surviving Import action and llama.cpp/llamafile External GGUF source modes through the mounted production UI; do not confuse those modes with the separate External rail used for configured STT roots.
- Stable UI/log copy must remain path-private. No absolute source path, command, subprocess output, or raw exception reaches a notice or log.
- Delete tests that exclusively exercise deleted widgets. Migrate only contracts that belong to surviving owners; do not keep dead widget tests through compatibility shims.
- Run the Impeccable detector exactly once, late, over changed production UI targets. Do not repair PRODUCT/DESIGN sidecar drift as part of this task.
- Baseline at planning start: the eight-node retirement/preservation probe produced `7 passed, 1 failed`; the sole failure is the already-tracked stale action census in `Tests/ProductionApp/test_llm_destination_actions.py` (TASK-2101), which this task directly supersedes by writing the final exact census.
- ADR required: yes.
- ADR path: `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md` (existing accepted amendment; no duplicate ADR).
- Reason: ADR-025 already owns managed-vs-external authority, legacy downloader retirement, rollback, and the promise that external/cache/runtime resolution remains supported.

---

## File responsibility map

### Production deletions

- Delete `tldw_chatbook/Widgets/HuggingFace/__init__.py`.
- Delete `tldw_chatbook/Widgets/HuggingFace/model_browser_widget.py`.
- Delete `tldw_chatbook/Widgets/HuggingFace/model_search_widget.py`.
- Delete `tldw_chatbook/Widgets/HuggingFace/model_card_viewer.py`.
- Delete `tldw_chatbook/Widgets/HuggingFace/download_manager.py`.
- Delete `tldw_chatbook/Widgets/HuggingFace/local_models_widget.py`.
- Delete `tldw_chatbook/LLM_Calls/huggingface_api.py` after the production/test reference audit proves its only consumers are the deleted widgets.

### Production edits

- `tldw_chatbook/UI/Screens/llm_screen.py` — remove the Download Models rail row only.
- `tldw_chatbook/UI/LLM_Management_Window.py` — remove the retired view mapping, deferred container/mount, browser import, activation hook, and Transformers download controls; preserve Installed legacy scan wiring, Remote, External, local listing, and provider runtime fields.
- `tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_transformers.py` — remove only repo/revision validators, subprocess download worker, direct-write handler, and handler-map entry; retain browse/list-local-model behavior and its cache-directory default.
- `tldw_chatbook/UI/Screens/model_installed_view.py` — make the zero-row state name the existing `Import GGUF…` action and External GGUF route while preserving scan behavior.
- `tldw_chatbook/Widgets/empty_state.py` — remove the unused `ModelsEmptyState` preset and its dead `empty-state-download-models` action rather than preserving a second routing seam.
- `tldw_chatbook/UI/Lab_Modules/lab_workbench.py` — remove the retired label from the rail-width rationale while retaining the shared width constants.
- `tldw_chatbook/css/features/_lab.tcss` — update only the stale rail-width comments; no layout token change unless mounted evidence proves one is required.
- `tldw_chatbook/css/tldw_cli_modular.tcss` — regenerate from source CSS; never hand-edit the generated copy.

### Tests

- Create `Tests/Architecture/test_task2062_3_legacy_downloader_retirement.py` — file/import/symbol/action/write-path absence plus explicit preservation allowlists.
- Edit `Tests/UI/test_llm_screen_lab_adoption.py` — final rail set, no Download Models activation/network test, mounted Import/External/Remote reachability.
- Edit `Tests/UI/test_llm_deferred_views.py` — final four deferred library views and exact surviving view count.
- Edit `Tests/UI/test_model_installed_view.py` — final rail expectation, actionable empty state, and retained unmanaged scan/import evidence.
- Edit `Tests/ProductionApp/test_llm_destination_actions.py` — final action census; delete direct-download execution tests; retain local scan/browse and unrelated lifecycle/privacy coverage.
- Edit `Tests/UI/test_lab_frame_mode_keys.py`, `Tests/UI/test_lab_mode_strip.py`, `Tests/UI/test_ux_batch5.py`, `Tests/UI/test_ux_batch6.py`, and `Tests/UI/test_ux_batch7.py` — remove dead browser monkeypatches while preserving their surviving shell/layout/binary tests.
- Edit `Tests/UI/test_lab_workbench.py` — replace the synthetic retired longest-label fixture with an exact surviving long Lab label while retaining the rendered no-truncation oracle.
- Edit `Tests/UI/test_non_obscuring_focus_contract.py` — remove the focus contract for the deleted model-card widget only.
- Edit `Tests/UI/test_front_matter_previews_1993.py` — remove only the deleted ModelCardViewer wiring test/import; retain the shared markdown parser present/absent contracts.
- Edit `Tests/UI/test_reader_scroll_keys_1994.py` — remove only the deleted ModelCardViewer named-pane assertion/import; retain generic `ReaderVerticalScroll` and `ConsoleTranscript` binding contracts.
- Delete `Tests/UI/test_hf_readme_links_1991.py` and `Tests/UI/test_hf_readme_toc_1992.py` because every assertion targets the deleted legacy model-card surface.
- Preserve and run `Tests/UI/test_model_remote_view.py`, `Tests/LLM_Management/test_gguf_server_sources.py`, `Tests/UI/test_llm_gguf_source_modes.py`, and Hugging Face inference adapter tests.

### Task metadata

- `backlog/tasks/task-2062.3 - Retire-legacy-Models-GGUF-downloaders.md` — plan, checked ACs, concise implementation notes, evidence, and Done status via Backlog CLI only.

## Acceptance-criteria traceability

| Acceptance criterion | Planned evidence |
|---|---|
| AC1 legacy package/client/rail removed | Tasks 1–2 file-absence/import/action ratchet plus mounted final rail/view census |
| AC2 Transformers direct writer removed | Task 3 DOM, handler-map, symbol, subprocess, and literal absence tests |
| AC3 inference/caching preserved | Task 4 exact Hugging Face inference and vLLM/MLX/Transformers compatibility nodes |
| AC4 directories/IDs/unmanaged discovery preserved | Tasks 3–4 local scan, browse, arbitrary External, exact command snapshot, and Remote tests |
| AC5 no callable direct writer; actionable empty state | Tasks 2–4 architecture ratchet and mounted Import/External empty-state test |
| AC6 dead-reference/action/mounted/regression proof | Tasks 1–5 exact source census, production-CSS mounted proof, mutations, and final affected union |

### Task 1: Establish the retirement and preservation ratchet

**Files:**
- Create: `Tests/Architecture/test_task2062_3_legacy_downloader_retirement.py`

**Interfaces:**
- Consumes: repository source tree, `MODELS_RAIL_SECTIONS`, `LLMManagementWindow.view_mapping`, `TRANSFORMERS_BUTTON_HANDLERS`.
- Produces: one explicit retirement inventory and one preservation inventory; no reusable scanner framework.

- [ ] **Step 1: Write the failing file/import retirement tests**

Assert the seven retired production modules/files are absent and non-importable:

```python
RETIRED_FILES = (
    ROOT / "tldw_chatbook/LLM_Calls/huggingface_api.py",
    *(ROOT / "tldw_chatbook/Widgets/HuggingFace" / name for name in (...)),
)

def test_legacy_models_downloader_files_are_retired() -> None:
    assert not [path for path in RETIRED_FILES if path.exists()]
```

Scan production Python (not historical backlog/design documents) for imports/symbols owned only by the deleted browser: `Widgets.HuggingFace`, `HuggingFaceModelBrowser`, `ModelSearchWidget`, `DownloadManager`, `LocalModelsWidget`, and the browser-specific `LLM_Calls.huggingface_api` module.

- [ ] **Step 2: Write the failing rail/browser-reference tests**

Assert:

- no Models rail key or view mapping equals `download-models`/`llm-view-download-models`;
- no production widget/button/action id equals `empty-state-download-models`;
- no legacy browser-owned production module remains that can open a destination and write downloaded bytes.

Keep the scan narrow enough that the real Hugging Face inference provider is not a false positive. The Transformers writer ratchet is added RED-first in Task 3 immediately before that slice is removed.

- [ ] **Step 3: Write preservation tests in the same file**

Pin the intentional survivors:

- `MODELS_RAIL_SECTIONS` still contains Installed, External, and Remote;
- `LLMManagementWindow` still passes configured `model_download_dir` to `InstalledView(legacy_dir=...)`;
- Transformers still exposes `transformers-models-dir-path`, Browse Dir, and List Local Models;
- llama.cpp/llamafile still expose External GGUF source controls;
- `model_remote_view.py`, `remote_huggingface.py`, and Hugging Face inference adapter entry points remain importable;
- `model_download_dir` may remain only in configuration and legacy scan wiring, never in a writer.

- [ ] **Step 4: Run the retirement suite and record genuine RED**

```bash
../../.venv/bin/python -m pytest -q Tests/Architecture/test_task2062_3_legacy_downloader_retirement.py
```

Expected: failures name the existing legacy files and rail/view/action keys while preservation assertions already pass.

- [ ] **Step 5: Commit only after Task 2 makes this first ratchet GREEN**

This test is intentionally born before the browser deletion and lands with that retirement. Task 3 extends the same file with a second genuine RED before touching the Transformers writer.

### Task 2: Delete the legacy Hugging Face browser and Download Models destination

**Files:**
- Delete: all six files under `tldw_chatbook/Widgets/HuggingFace/`.
- Delete: `tldw_chatbook/LLM_Calls/huggingface_api.py`.
- Modify: `tldw_chatbook/UI/Screens/llm_screen.py`.
- Modify: `tldw_chatbook/UI/LLM_Management_Window.py`.
- Modify: `tldw_chatbook/UI/Lab_Modules/lab_workbench.py`.
- Modify: `tldw_chatbook/css/features/_lab.tcss`.
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`.
- Modify/delete legacy-only UI tests listed in the file map.

- [ ] **Step 1: Complete the consumer audit before deleting anything**

Run:

```bash
rg -n \
  'Widgets\.HuggingFace|HuggingFaceModelBrowser|ModelSearchWidget|ModelCardViewer|DownloadManager|LocalModelsWidget|LLM_Calls\.huggingface_api|from .*huggingface_api' \
  tldw_chatbook Tests
```

Classify every result into the seven planned production deletions, the known `LLMManagementWindow` wiring, or the named legacy tests. If any production consumer remains outside those owners, stop and revise the plan before deletion. This pre-delete result is the AC1 authority; the post-delete scan only confirms it stayed true.

- [ ] **Step 2: Remove the exact rail/view/mount path**

Delete the `download-models` rail tuple, view mapping, deferred `Container`, `HuggingFaceModelBrowser` import/mount, and `_start_view_work` branch. Update the deferred inventory to the exact five survivors—Ollama plus Curated, Installed, External, and Remote—and the exact eleven total views. Do not alter those owners or llama.cpp, llamafile, vLLM, Transformers, or MLX behavior.

- [ ] **Step 3: Delete the legacy package and browser-only client**

Remove the six widget files and `LLM_Calls/huggingface_api.py`. Re-run the same consumer audit and require zero production/test imports or call sites; do not add a compatibility export.

- [ ] **Step 4: Remove stale live label rationales and migrate tests according to ownership**

- update rail and deferred-view expectations;
- delete the two files that test only the deleted model-card reader;
- retain the shared markdown-parser, generic reader-scroll, Media viewer, and Console transcript contracts in `test_front_matter_previews_1993.py` and `test_reader_scroll_keys_1994.py`, removing only their ModelCardViewer imports/tests;
- remove only HuggingFace monkeypatch setup from surviving UX/Lab tests;
- remove only the deleted model-card focus assertion;
- replace the old "browse only after Download Models opens" test with a mounted network-guard test that traverses every surviving model rail entry and proves no unprompted HTTP client/search occurs; the architecture ratchet separately proves the legacy browser client no longer exists.
- update `lab_workbench.py` and `_lab.tcss` comments so they describe the exact surviving shared-width owner (`Speech Recognition`) instead of the retired label, regenerate the CSS bundle, and update `test_lab_workbench.py` to render that surviving label.

- [ ] **Step 5: Run focused GREEN**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Architecture/test_task2062_3_legacy_downloader_retirement.py \
  Tests/UI/test_llm_screen_lab_adoption.py \
  Tests/UI/test_llm_deferred_views.py \
  Tests/UI/test_lab_workbench.py \
  Tests/UI/test_lab_frame_mode_keys.py \
  Tests/UI/test_lab_mode_strip.py \
  Tests/UI/test_ux_batch5.py \
  Tests/UI/test_ux_batch6.py \
  Tests/UI/test_ux_batch7.py
```

Expected: all browser-retirement and surviving shell tests pass. No Transformers retirement assertion exists yet; Task 3 adds it before the corresponding production edit.

- [ ] **Step 6: Mutation proof**

Temporarily restore only the `download-models` rail tuple or a `HuggingFaceModelBrowser` import string. The architecture/rail test must fail on the exact reintroduced owner. Restore deletion and rerun GREEN.

- [ ] **Step 7: Commit the browser retirement**

```bash
git add \
  tldw_chatbook/UI/Screens/llm_screen.py \
  tldw_chatbook/UI/LLM_Management_Window.py \
  tldw_chatbook/UI/Lab_Modules/lab_workbench.py \
  tldw_chatbook/css/features/_lab.tcss \
  tldw_chatbook/css/tldw_cli_modular.tcss \
  tldw_chatbook/LLM_Calls/huggingface_api.py \
  Tests/Architecture/test_task2062_3_legacy_downloader_retirement.py \
  Tests/UI/test_llm_screen_lab_adoption.py \
  Tests/UI/test_llm_deferred_views.py \
  Tests/UI/test_lab_workbench.py \
  Tests/UI/test_lab_frame_mode_keys.py \
  Tests/UI/test_lab_mode_strip.py \
  Tests/UI/test_ux_batch5.py \
  Tests/UI/test_ux_batch6.py \
  Tests/UI/test_ux_batch7.py \
  Tests/UI/test_non_obscuring_focus_contract.py \
  Tests/UI/test_front_matter_previews_1993.py \
  Tests/UI/test_reader_scroll_keys_1994.py \
  Tests/UI/test_hf_readme_links_1991.py \
  Tests/UI/test_hf_readme_toc_1992.py \
  tldw_chatbook/Widgets/HuggingFace/__init__.py \
  tldw_chatbook/Widgets/HuggingFace/model_browser_widget.py \
  tldw_chatbook/Widgets/HuggingFace/model_search_widget.py \
  tldw_chatbook/Widgets/HuggingFace/model_card_viewer.py \
  tldw_chatbook/Widgets/HuggingFace/download_manager.py \
  tldw_chatbook/Widgets/HuggingFace/local_models_widget.py
git commit -m "refactor(models): retire legacy Hugging Face browser"
```

Before staging, inspect `git status --short` and stage the exact intended paths/files; never use `git add -A`.

### Task 3: Remove the Transformers direct-write downloader

**Files:**
- Modify: `tldw_chatbook/UI/LLM_Management_Window.py`.
- Modify: `tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_transformers.py`.
- Modify: `Tests/ProductionApp/test_llm_destination_actions.py`.
- Modify: `Tests/Architecture/test_task2062_3_legacy_downloader_retirement.py`.

- [ ] **Step 1: Add focused mounted and source-boundary RED tests**

In the final action census, assert that the Transformers view still paints and supports:

- Local Models Root Directory input;
- Browse Dir;
- List Local Models;
- local model results/log output.

Also assert the Repo ID, Revision, and Download Model controls do not exist and a synthetic press for the retired id has no registered action.

At source level, assert the module has no download worker, no `subprocess.Popen`, no `functools.partial`, no repo/revision validator used only by download, no `target_model_specific_dir.mkdir`, and no `huggingface-cli` command.

Extend `Tests/Architecture/test_task2062_3_legacy_downloader_retirement.py` with these Transformers-specific assertions now, before changing production. Keep the scan scoped to the LLM Management Transformers module so unrelated transcription help text mentioning the external `huggingface-cli` program is not a false positive.

Add a direct preservation test for the surviving Browse/List boundary: with `hf_constants.HF_HUB_CACHE` pointing to an existing directory, pressing Browse opens the directory picker at that exact cache root; invoking the real picker callback with a different arbitrary directory updates only `#transformers-models-dir-path`; List then scans that chosen directory. This is the evidence that provider-owned cache discovery and arbitrary directory selection were not accidentally removed with the writer.

- [ ] **Step 2: Record genuine RED**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Architecture/test_task2062_3_legacy_downloader_retirement.py \
  Tests/ProductionApp/test_llm_destination_actions.py \
  -k 'transformers or action_census or legacy_models_downloader'
```

Expected: old download controls/handler/worker remain reachable.

- [ ] **Step 3: Remove the exact UI and handler slice**

Delete the Transformers "Download New Model" label, Repo ID input, Revision input, and Download Model button. Delete `_valid_huggingface_repo_id`, `_valid_huggingface_revision`, `run_transformers_model_download_worker`, `handle_transformers_download_model_button_pressed`, and the handler-map entry. Remove now-unused `functools`, `subprocess`, `List`, `Optional`, and `HfApi` imports only after `rg`/Ruff prove them unused. Keep `huggingface_hub.constants` if it remains the incumbent Browse default.

- [ ] **Step 4: Delete obsolete execution tests and keep final negative coverage**

Remove the tests/fakes that simulate success, nonzero exit, timeout, raw subprocess output, invalid repo/revision, or the retired worker group. Replace them with the exact final action census and absence ratchet. Retain the local scan privacy/normalization tests and the generic action-dispatch recovery contract.

- [ ] **Step 5: Run, mutate, restore**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Architecture/test_task2062_3_legacy_downloader_retirement.py \
  Tests/ProductionApp/test_llm_destination_actions.py \
  -k 'transformers or action_census or legacy_models_downloader'
```

Mutation: temporarily re-add the retired handler-map key with a no-op callable. The action-census/retirement test must fail even though no subprocess runs. Restore and rerun GREEN.

- [ ] **Step 6: Commit the direct-writer retirement**

```bash
git add \
  tldw_chatbook/UI/LLM_Management_Window.py \
  tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_transformers.py \
  Tests/Architecture/test_task2062_3_legacy_downloader_retirement.py \
  Tests/ProductionApp/test_llm_destination_actions.py
git commit -m "refactor(models): retire Transformers direct downloader"
```

### Task 4: Make surviving recovery explicit and prove compatibility

**Files:**
- Modify: `tldw_chatbook/UI/Screens/model_installed_view.py`.
- Modify: `tldw_chatbook/Widgets/empty_state.py`.
- Modify: `Tests/UI/test_model_installed_view.py`.
- Modify: `Tests/UI/test_llm_screen_lab_adoption.py`.

- [ ] **Step 1: Write the mounted empty/recovery RED test**

Mount Models with an empty managed inventory and empty legacy scan using the real `TldwCli.CSS_PATH`. Assert:

- no Download Models rail row/view/action exists;
- Installed shows "No managed or legacy models found" plus explicit guidance to `Import GGUF…` or choose `External GGUF` under llama.cpp/Llamafile;
- the real Import GGUF button is painted, focusable, and posts the incumbent picker intent;
- llama.cpp and llamafile External GGUF source controls remain reachable without copying into the store;
- the separate External rail remains present for configured STT roots but is not presented as the GGUF replacement route;
- Remote remains present for verified managed acquisition and does not search until explicit submission;
- 80-column regions remain inside their parents after one rail row and one form section are removed.

The copy must not imply that every external file must be imported.

- [ ] **Step 2: Record genuine RED**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_model_installed_view.py \
  Tests/UI/test_llm_screen_lab_adoption.py \
  -k 'empty or rail or unmanaged or import or external'
```

Expected: the old zero-row copy lacks the explicit recovery route and stale rail tests still expect Download Models.

- [ ] **Step 3: Apply the minimum copy/dead-preset change**

Update only the Installed zero-row copy. Remove the unused `ModelsEmptyState` preset and `empty-state-download-models` action from `Widgets/empty_state.py`; do not create a new redirect handler or duplicate Import/External button.

- [ ] **Step 4: Prove legacy discovery and external use remain intact**

Run exact tests for:

- bounded unmanaged scan and managed-root exclusion;
- header and unmanaged-row Import picker;
- arbitrary External GGUF required-path/start behavior and in-place worker validation;
- Managed/External/Embedded source matrices;
- Remote explicit search/resolution/acquisition;
- Transformers local directory browse/list;
- vLLM and MLX exact command snapshots with Hugging Face model IDs/local paths;
- Hugging Face inference adapter URL/model/token behavior.

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_model_installed_view.py::test_unmanaged_scan_is_bounded_and_labels_supported_model_files \
  Tests/UI/test_model_installed_view.py::test_scan_unmanaged_excludes_managed_artifacts_root \
  Tests/UI/test_model_installed_view.py::test_header_and_unmanaged_row_open_real_gguf_picker \
  Tests/UI/test_llm_gguf_source_modes.py::test_external_mode_requires_a_path_before_start \
  Tests/LLM_Management/test_gguf_server_sources.py::test_external_source_validation_is_worker_thread_store_free_and_read_only \
  Tests/LLM_Management/test_gguf_server_sources.py::test_vllm_command_snapshot_is_unchanged \
  Tests/LLM_Management/test_gguf_server_sources.py::test_mlx_command_snapshot_is_unchanged \
  Tests/UI/test_model_remote_view.py \
  Tests/Chat/test_chat_functions.py::test_huggingface_chat_api_call_passes_max_tokens_to_adapter \
  Tests/Chat/test_chat_functions.py::test_huggingface_router_chat_url_rejects_missing_or_non_string_base
```

- [ ] **Step 5: Mutation proof**

Temporarily remove `legacy_dir` from `InstalledView` construction or change the External source mode to Managed. The corresponding preserved-scan/external test must fail. Restore and rerun GREEN.

- [ ] **Step 6: Run the late UI quality checks**

After all UI edits, read Impeccable `craft-floor.md`, run the mounted 80-column compositor test, and run the detector once:

```bash
node /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.agents/skills/impeccable/scripts/detect.mjs --json \
  tldw_chatbook/UI/LLM_Management_Window.py \
  tldw_chatbook/UI/Screens/llm_screen.py \
  tldw_chatbook/UI/Screens/model_installed_view.py \
  tldw_chatbook/Widgets/empty_state.py \
  tldw_chatbook/css/features/_lab.tcss
```

Expected: no changed-target findings. Do not rerun this detector later; resolve any finding with ordinary inspection/tests.

- [ ] **Step 7: Commit recovery and compatibility evidence**

```bash
git add \
  tldw_chatbook/UI/Screens/model_installed_view.py \
  tldw_chatbook/Widgets/empty_state.py \
  Tests/UI/test_model_installed_view.py \
  Tests/UI/test_llm_screen_lab_adoption.py
git commit -m "test(models): preserve downloader retirement recovery"
```

### Task 5: Final affected-union evidence and TASK-2062.3 closeout

**Files:**
- Modify via CLI: `backlog/tasks/task-2062.3 - Retire-legacy-Models-GGUF-downloaders.md`.

- [ ] **Step 1: Run the final retirement/reference scan**

```bash
rg -n \
  'Widgets\.HuggingFace|HuggingFaceModelBrowser|ModelSearchWidget|DownloadManager|LocalModelsWidget|Download Models|download-models|llm-view-download-models|transformers-download-model-button|run_transformers_model_download_worker|handle_transformers_download_model_button_pressed' \
  tldw_chatbook Tests
```

Expected: only the intentional negative-test constants/messages in the retirement ratchet; no production or stale positive-test consumer.

Also inspect every remaining `model_download_dir`, `huggingface_api`, `HuggingFaceRemoteAdapter`, vLLM/MLX model, and External GGUF reference to verify it belongs to an approved survivor rather than a writer.

- [ ] **Step 2: Run the exact affected test union once, unfiltered**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Architecture/test_task2062_3_legacy_downloader_retirement.py \
  Tests/ProductionApp/test_llm_destination_actions.py \
  Tests/UI/test_llm_screen_lab_adoption.py \
  Tests/UI/test_llm_deferred_views.py \
  Tests/UI/test_lab_workbench.py \
  Tests/UI/test_model_installed_view.py \
  Tests/UI/test_model_remote_view.py \
  Tests/UI/test_llm_gguf_source_modes.py \
  Tests/LLM_Management/test_gguf_server_sources.py \
  Tests/UI/test_lab_frame_mode_keys.py \
  Tests/UI/test_lab_mode_strip.py \
  Tests/UI/test_ux_batch5.py \
  Tests/UI/test_ux_batch6.py \
  Tests/UI/test_ux_batch7.py \
  Tests/UI/test_non_obscuring_focus_contract.py \
  Tests/Chat/test_chat_functions.py
```

Do not hide the known periodic Ollama/socket baseline if a broad ProductionApp file reaches it. Classify exact unrelated failures against `origin/dev`; fix every changed-contract failure before proceeding.

- [ ] **Step 3: Run static, format, compile, and diff gates**

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/UI/LLM_Management_Window.py \
  tldw_chatbook/UI/Screens/llm_screen.py \
  tldw_chatbook/UI/Screens/model_installed_view.py \
  tldw_chatbook/Widgets/empty_state.py \
  tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_transformers.py \
  tldw_chatbook/UI/Lab_Modules/lab_workbench.py \
  Tests/Architecture/test_task2062_3_legacy_downloader_retirement.py \
  Tests/ProductionApp/test_llm_destination_actions.py \
  Tests/UI/test_llm_screen_lab_adoption.py \
  Tests/UI/test_llm_deferred_views.py \
  Tests/UI/test_lab_workbench.py \
  Tests/UI/test_model_installed_view.py \
  Tests/UI/test_lab_frame_mode_keys.py \
  Tests/UI/test_lab_mode_strip.py \
  Tests/UI/test_ux_batch5.py \
  Tests/UI/test_ux_batch6.py \
  Tests/UI/test_ux_batch7.py \
  Tests/UI/test_non_obscuring_focus_contract.py \
  Tests/UI/test_front_matter_previews_1993.py \
  Tests/UI/test_reader_scroll_keys_1994.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/UI/LLM_Management_Window.py \
  tldw_chatbook/UI/Screens/llm_screen.py \
  tldw_chatbook/UI/Screens/model_installed_view.py \
  tldw_chatbook/Widgets/empty_state.py \
  tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_transformers.py \
  tldw_chatbook/UI/Lab_Modules/lab_workbench.py \
  Tests/Architecture/test_task2062_3_legacy_downloader_retirement.py \
  Tests/ProductionApp/test_llm_destination_actions.py \
  Tests/UI/test_llm_screen_lab_adoption.py \
  Tests/UI/test_llm_deferred_views.py \
  Tests/UI/test_lab_workbench.py \
  Tests/UI/test_model_installed_view.py \
  Tests/UI/test_lab_frame_mode_keys.py \
  Tests/UI/test_lab_mode_strip.py \
  Tests/UI/test_ux_batch5.py \
  Tests/UI/test_ux_batch6.py \
  Tests/UI/test_ux_batch7.py \
  Tests/UI/test_non_obscuring_focus_contract.py \
  Tests/UI/test_front_matter_previews_1993.py \
  Tests/UI/test_reader_scroll_keys_1994.py
../../.venv/bin/python -m py_compile \
  tldw_chatbook/UI/LLM_Management_Window.py \
  tldw_chatbook/UI/Screens/llm_screen.py \
  tldw_chatbook/UI/Screens/model_installed_view.py \
  tldw_chatbook/Widgets/empty_state.py \
  tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_transformers.py \
  tldw_chatbook/UI/Lab_Modules/lab_workbench.py
../../.venv/bin/python tldw_chatbook/css/build_css.py
git diff --check
```

Use changed-range formatting when an incumbent whole file has proven baseline debt; do not create broad formatter churn.

- [ ] **Step 4: Perform correctness and Ponytail review**

Review the complete `origin/dev...HEAD` diff for:

- any surviving callable direct writer;
- accidental deletion of inference, Remote acquisition, external directories, runtime IDs, caching, or legacy scans;
- dead imports/actions/tests/comments;
- unnecessary compatibility shims or replacement abstractions;
- path/command/output leakage;
- mounted focus/keyboard/empty-state regressions.

Safe simplification wins: delete stale code/tests instead of adding aliases or adapters.

- [ ] **Step 5: Update task metadata through Backlog CLI only**

After all evidence and reviews are green:

```bash
backlog task edit 2062.3 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 --check-ac 6
backlog task edit 2062.3 --notes "<concise implementation notes, files, ADR-025, tests, mutations, and preservation evidence>"
backlog task edit 2062.3 -s Done
```

Verify `backlog task 2062.3 --plain` shows all ACs checked, ADR-025/documentation links preserved, implementation notes present, and status Done.

- [ ] **Step 6: Commit closeout metadata**

```bash
git add 'backlog/tasks/task-2062.3 - Retire-legacy-Models-GGUF-downloaders.md'
git commit -m "docs(models): close task 2062.3"
```

- [ ] **Step 7: Rebase, reverify, open the PR, address feedback, and merge**

Fetch and rebase onto latest `origin/dev`. If the tested SHA changes, rerun the exact affected union and static gates. Push the task branch, open a ready PR to `dev`, address every actionable current inline/top-level comment with RED/GREEN evidence, resolve its thread, rebase again immediately before merge, and merge only while the PR is mergeable and the reviewed exact head remains green. General unrelated CI is not a TASK-2062.3 blocker, but do not ignore the task's own affected tests or review findings.

- [ ] **Step 8: Verify integration and clean up**

Confirm `origin/dev` contains the final head, TASK-2062.3 is Done on dev, and the dedicated worktree/branch can be safely removed. Do not delete any legacy model cache, external GGUF, or managed artifact during cleanup.

## Completion handoff

Report:

- exact files and callable writers removed;
- exact replacement paths that remain (Installed Import, External GGUF, Remote managed acquisition);
- proof that inference/model IDs/caching/unmanaged scans are unchanged;
- mounted 80-column/focus/action evidence;
- mutation results;
- exact affected-union/static results;
- PR/rebase/merge identifiers;
- any pre-existing warnings or baselines that remain unrelated.
