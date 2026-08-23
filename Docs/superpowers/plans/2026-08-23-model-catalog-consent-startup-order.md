# Model Catalog Consent Startup Order Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Guarantee that launch finishes the intro, mounts the configured initial screen, then presents one actionable model-catalog consent modal before the app becomes usable.

**Architecture:** Keep ADR-020's existing consent and refresh state machine intact. Relocate its single startup scheduling call from the app mount lifecycle to `_push_initial_screen()`, the shared post-intro choke point for splash and no-splash launches, and lock the screen-stack ordering down with a real Textual Pilot regression.

**Tech Stack:** Python 3.11+, Textual 8, pytest, pytest-asyncio, Backlog.md

**ADR required:** no
**ADR path:** `backlog/decisions/020-automatic-model-catalog-refresh.md`
**Reason:** This is a lifecycle ordering bug fix that preserves ADR-020's storage, consent, and network boundaries.

---

### Task 0: Record the approved plan in Backlog.md

**Files:**
- Modify: `backlog/tasks/task-21161 - Order-model-catalog-consent-after-intro-startup.md`

- [ ] Add the implementation plan to TASK-21161 while it is In Progress and before changing application or test code.
- [ ] Include the ADR required/no, existing ADR path, and lifecycle-fix reason in the task plan.

### Task 1: Add a failing launch-order regression

**Files:**
- Modify: `Tests/UI/test_model_catalog_consent_modal.py`

- [ ] Add a splash-enabled full-app test using `Tests.UI.app_factory._build_test_app(configured_default="home")`.
- [ ] Override only `_push_model_catalog_consent_modal` so `run_test()` can display the real modal despite production's headless guard.
- [ ] Suppress project-skills discovery and keep first-run setup completed so unrelated startup overlays cannot affect the assertion.
- [ ] Observe the real screen-stack transitions and assert the final ordered stack is initial screen below a topmost `ModelCatalogConsentModal`, never consent buried below the initial screen.
- [ ] Click Deny through Pilot and assert the modal dismisses to the same usable initial screen.
- [ ] Run the new test against the current call site and confirm it fails because the initial screen covers the modal.

### Task 2: Move consent scheduling behind initial-screen startup

**Files:**
- Modify: `tldw_chatbook/app.py`

- [ ] Remove `_schedule_startup_model_catalog_refresh()` from `on_mount()`.
- [ ] Call `_schedule_startup_model_catalog_refresh()` at the end of `_push_initial_screen()` after the initial screen and startup notices have been mounted/offered.
- [ ] Preserve the setup-ownership gate, one-shot guard, consent callback, persistence behavior, and refresh worker unchanged.
- [ ] Run the new regression and confirm it passes.

### Task 2A: Cover and fix competing startup overlays

**Files:**
- Modify: `Tests/UI/test_model_catalog_consent_modal.py`
- Modify: `tldw_chatbook/app.py`

- [ ] Extend the full-app consent regression to prove an eligible project-skills startup offer is deferred when catalog consent is unrecorded.
- [ ] Add a first-run completion regression that navigates to its valid exit route before showing consent and proves no callback value is produced until Pilot clicks Yes or No.
- [ ] Confirm both regressions fail against the current partial fix for the reviewed race conditions.
- [ ] Record when startup scheduling selects the unrecorded-consent branch, schedule before optional project-skills discovery, and skip that optional offer for this launch.
- [ ] Sequence completed first-run exit navigation before scheduling the deferred catalog decision; keep the same-tab Console shortcut and no-route completion behavior intact.
- [ ] Preserve explicit consent persistence and refresh behavior unchanged, then run both regressions green.

### Task 3: Verify adjacent startup and consent behavior

**Files:**
- Test: `Tests/UI/test_model_catalog_consent_modal.py`
- Test: `Tests/LLM_Provider_Catalog/test_app_model_catalog_wiring.py`
- Test: `Tests/UI/test_product_maturity_phase1_first_run.py`

- [ ] Run the complete focused consent, catalog wiring, and first-run startup test files.
- [ ] Run Ruff on the changed Python files and `git diff --check`.
- [ ] Review the final diff for accidental changes to ADR-020 behavior.

### Task 4: Close task documentation

**Files:**
- Modify: `backlog/tasks/task-21161 - Order-model-catalog-consent-after-intro-startup.md`

- [ ] Check every acceptance criterion only after its verification evidence exists.
- [ ] Add concise implementation notes covering the lifecycle move, regression, ADR decision, and commands run.
- [ ] Set TASK-21161 to Done only after all Definition of Done checks are satisfied.
