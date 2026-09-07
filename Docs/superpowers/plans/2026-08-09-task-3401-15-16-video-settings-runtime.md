# Video Settings Runtime Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Video Gen Settings reachable and ensure its persisted configuration controls the Console video runtime.

**Architecture:** Keep the explicit Settings rail and the existing profile-aware `load_settings()` boundary. Add the missing category registration and pass through the already parsed `video_generation` table; do not add registries, direct TOML readers, or dependencies.

**Tech Stack:** Python 3.11+, Textual 8.x, pytest, TOML configuration, Ruff.

---

### Task 1: Restore Video Gen Settings navigation

**Files:**
- Modify: `Tests/UI/test_settings_configuration_hub.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Verify: `Tests/UI/test_settings_video_gen_defaults.py`
- Verify: `Tests/Chat/test_console_generate_video.py`

- [ ] **Step 1: Write focused failing navigation tests**

Add a direct contract test that expects `SettingsCategoryId.VIDEO_GENERATION` to be the final Domain Defaults member and verifies the group count matches the category tuple. Add a Textual Pilot test that searches for `Video Gen`, confirms the category is visible, presses Enter, and observes the existing Video Gen panel and ownership copy. Add a boundary test for `SettingsScreen._handle_video_gen_save()` that supplies an invalid draft and asserts the actual save worker is never invoked.

- [ ] **Step 2: Run the new tests to verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_settings_configuration_hub.py -q -k 'video_generation_category or video_gen_filter or invalid_video_gen_draft'
```

Expected: the two navigation tests fail because the category button/group member is missing; the invalid-draft boundary test already passes and establishes the unchanged save guard.

- [ ] **Step 3: Implement the minimal production fix**

Append the existing enum member to the explicit Domain Defaults tuple:

```python
SettingsCategoryId.IMAGE_GENERATION,
SettingsCategoryId.VIDEO_GENERATION,
```

Do not change category derivation, ordering, collapse behavior, panel composition, or persistence.

- [ ] **Step 4: Run focused Settings GREEN tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_settings_configuration_hub.py Tests/UI/test_settings_video_gen_defaults.py Tests/Chat/test_console_generate_video.py::test_successful_settings_save_rebuilds_adapter_and_console_uses_same_instance -q -k 'video or domain_category_contracts or successful_settings_save'
```

Expected: all selected tests pass.

- [ ] **Step 5: Mutation-check the navigation guard**

Temporarily remove the new tuple member, rerun the two new tests, and confirm both fail for the intended missing-navigation reason. Restore the exact line and rerun GREEN.

- [ ] **Step 6: Commit TASK-3401.15 implementation**

Stage only the Settings production/test files after focused evidence is complete. Keep TASK-3401.15 In Progress until live Settings UAT verifies its user-facing path.

```bash
git add tldw_chatbook/UI/Screens/settings_screen.py Tests/UI/test_settings_configuration_hub.py
git diff --cached --check
git commit -m "fix: expose Video Gen settings navigation"
```

### Task 2: Project persisted video settings into runtime configuration

**Files:**
- Create: `Tests/Video_Generation/test_config_projection.py`
- Modify: `tldw_chatbook/config.py`
- Modify: `Tests/Video_Generation/test_config_loader.py`
- Modify: `tldw_chatbook/Video_Generation/config.py`
- Verify: `Tests/Video_Generation/test_adapter_registry.py`

- [ ] **Step 1: Write a real scratch-config failing regression**

Write a temporary TOML profile containing global Video Generation values and a nested ComfyUI table. Point `TLDW_CONFIG_PATH` to it, clear the main settings/bootstrap caches plus video config/registry caches, then assert:

```python
settings["video_generation"]["enabled_backends"] == ["comfyui"]
config.default_backend == "comfyui"
config.enabled_backends == ["comfyui"]
config.comfyui_base_url == "http://127.0.0.1:18188"
config.comfyui_default_workflow == "minimax_h3_t2v_spectrum.json"
config.comfyui_timeout_seconds == 321
registry.resolve_backend("comfyui") == "comfyui"
```

Rewrite the same scratch profile with a changed workflow, then use this exact refresh order: `load_settings(force_reload=True)`, `reset_video_generation_runtime()`, obtain a new video config, and finally obtain a new registry. Prove the second snapshot and registry see the changed value.

- [ ] **Step 2: Run the new test to verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Video_Generation/test_config_projection.py -q
```

Expected: failure because `load_settings()` has no `video_generation` key.

- [ ] **Step 3: Implement the minimal projection**

In `load_settings()`, mirror the existing image table pass-through:

```python
final_video_generation_settings_cli = get_toml_section("video_generation")
```

and include it in `config_dict`:

```python
"video_generation": final_video_generation_settings_cli,
```

Keep the registry, writers, and cache-reset APIs unchanged. Because projection makes the raw table reachable for the first time, normalize a non-mapping top-level section and non-mapping backend subsections to empty mappings at the existing video loader boundary; add focused malformed-section tests before this production change.

- [ ] **Step 4: Run focused config/runtime GREEN tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Video_Generation/test_config_projection.py Tests/Video_Generation/test_config_loader.py Tests/Video_Generation/test_adapter_registry.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Mutation-check the projection guard**

Temporarily remove the config-dict projection, rerun `test_config_projection.py`, and confirm it fails at the real `load_settings()` boundary. Restore the exact line and rerun GREEN.

- [ ] **Step 6: Commit TASK-3401.16 implementation**

Stage only `config.py` and the new projection test after focused evidence is complete. Keep TASK-3401.16 In Progress until the actual Console UAT proves its user-facing acceptance criterion.

```bash
git add tldw_chatbook/config.py Tests/Video_Generation/test_config_projection.py
git diff --cached --check
git commit -m "fix: project video settings into runtime"
```

- [ ] **Step 7: Address review-discovered loader and isolation gaps**

Make the first scratch-profile assertion call `load_settings()` without `force_reload`, start teardown protection before any environment/cache mutation, and retain forced reload only after rewriting the profile. Add RED tests for scalar top-level and backend video tables, then minimally normalize them to empty mappings in `Video_Generation.config`. Rerun the projection/config/registry gate and commit the review fix separately.

### Task 3: Verify and commit both defect implementations

**Files:**
- Verify all production and tests modified in Tasks 1–2
- Modify: `backlog/tasks/task-3401.15 - Make-Video-Gen-Settings-category-reachable-from-navigation.md`
- Modify: `backlog/tasks/task-3401.16 - Project-video-generation-settings-through-load_settings.md`

- [ ] **Step 1: Run the combined touched-file test gate**

Run only the affected Settings and Video Generation files:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_settings_configuration_hub.py Tests/UI/test_settings_video_gen_defaults.py Tests/Chat/test_console_generate_video.py Tests/Video_Generation/test_config_projection.py Tests/Video_Generation/test_config_loader.py Tests/Video_Generation/test_adapter_registry.py -q -k 'video or domain_category_contracts or successful_settings_save or config_projection or adapter_registry'
```

- [ ] **Step 2: Run targeted static verification**

Run full Ruff rules on the compact changed files:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check tldw_chatbook/config.py Tests/Video_Generation/test_config_projection.py Tests/UI/test_settings_configuration_hub.py
```

Run the repository-established syntax/error subset on the large baseline Settings screen:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check --select E9,F63,F7,F82 tldw_chatbook/UI/Screens/settings_screen.py
```

Compile the exact four changed Python files to a `TemporaryDirectory` output:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c 'import py_compile,tempfile; from pathlib import Path; files=["tldw_chatbook/config.py","tldw_chatbook/UI/Screens/settings_screen.py","Tests/Video_Generation/test_config_projection.py","Tests/UI/test_settings_configuration_hub.py"]; d=tempfile.TemporaryDirectory(); [py_compile.compile(f,cfile=str(Path(d.name)/(Path(f).name+".pyc")),doraise=True) for f in files]'
```

Run exact diff, workflow-immutability, and privacy checks:

```bash
git diff --check
git diff --exit-code 2d6367b82 -- tldw_chatbook/Video_Generation/workflows
if rg -n '192\.168\.|OPENAI_API_KEY|/private/tmp/|Documents/Comfy-Workflows|/Downloads/' tldw_chatbook/config.py tldw_chatbook/UI/Screens/settings_screen.py Tests/Video_Generation/test_config_projection.py Tests/UI/test_settings_configuration_hub.py backlog/tasks/task-3401.1{4,5,6}* Docs/superpowers/qa/2026-08-09-comfyui-h3-console-generation-uat.md; then exit 1; fi
```

- [ ] **Step 3: Self-review against both task AC lists**

Confirm the diff contains only the approved navigation/projection changes, focused tests, task records, and approved design/plan docs. Confirm no workflow JSON changed. Commit the two production/test fixes in task order while leaving both task records In Progress.

### Task 4: Resume TASK-3401.14 live Console UAT

**Files:**
- Modify: `backlog/tasks/task-3401.14 - UAT-end-to-end-ComfyUI-H3-generation-through-Console.md`
- Modify: `Docs/superpowers/qa/2026-08-09-comfyui-h3-console-generation-uat.md`

- [ ] **Step 1: Create and verify an isolated live profile**

Use a new validated temporary root for HOME, config, data, saved copies, and app logs. Record a read-only baseline for the real profile and confirm the process resolves every mutable path into the scratch root.

- [ ] **Step 2: Run Base and Spectrum through Settings and Console**

Use the now-reachable Video Gen panel to select each packaged variant and submit the actual `/generate-video :comfyui` Console command. Record only sanitized status/count evidence while polling.

- [ ] **Step 3: Verify media and UI contracts**

For each successful result, inspect the ephemeral bytes with `ffprobe`, compare displayed metadata, open the full player, verify audio/video playback state, and save a playable copy inside the scratch root.

- [ ] **Step 4: Verify one terminal cancellation/error path**

Use the existing UI control or a scratch-only unavailable configuration and confirm no pending card or partial media remains.

- [ ] **Step 5: Clean up and record prompt-free evidence**

Stop the app, validate and remove only the exact scratch root, confirm the real profile baseline is unchanged, update the prompt-free UAT report/task, and file any newly discovered product defect separately instead of fixing it inside UAT.

- [ ] **Step 6: Close verified tasks and commit records**

Only after the actual Settings and Console paths pass, replace TASK-3401.15 and TASK-3401.16 Implementation Notes with the approach, exact RED/GREEN/static/live evidence, ADR-044/no-new-ADR result, modified files, and implementation commit hashes. Check every AC and set both to Done through Backlog CLI. Update TASK-3401.14 according to the UAT results and commit the three task records plus sanitized UAT report as a docs-only boundary:

```bash
git add 'backlog/tasks/task-3401.14 - UAT-end-to-end-ComfyUI-H3-generation-through-Console.md' 'backlog/tasks/task-3401.15 - Make-Video-Gen-Settings-category-reachable-from-navigation.md' 'backlog/tasks/task-3401.16 - Project-video-generation-settings-through-load_settings.md' Docs/superpowers/qa/2026-08-09-comfyui-h3-console-generation-uat.md
git diff --cached --check
git commit -m "docs: complete ComfyUI H3 Console UAT"
```
