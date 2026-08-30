# Layered README Restoration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the useful layered structure of the README that existed
before PR #2045, then repair its opening, setup path, status, terminology, and
technical reference material without flattening it into a short product memo.

**Architecture:** This is a documentation-only correction. The exact README
blob at `d2ff9c05ca91d7f7b7be80a2401f78f7142e1aff` (the first parent of PR #2045's
merge commit) is the source document; current `dev` code, package metadata,
ADRs, and user guides override stale claims inside that blob. The final README
uses progressive disclosure: orientation and a five-minute path first, detailed
reference below, with one current Console screenshot at
`Docs/static/tldw-chatbook-console.png`.

**Tech Stack:** GitHub-flavored Markdown, Python package metadata from
`pyproject.toml`, Textual 8.2.8 UI terminology, Backlog.md task tracking.

**ADR required:** no

**ADR path:** N/A

**Reason:** The change documents existing behavior and restores documentation
structure; it does not alter runtime, storage, dependencies, security policy,
or long-lived application architecture.

---

## File Map

- Modify `README.md`: layered project landing page and technical reference.
- Create `Docs/static/tldw-chatbook-console.png`: current, non-sensitive
  landing-page screenshot of the Console.
- Modify
  `Docs/superpowers/specs/2026-07-23-newcomer-first-readme-design.md`: already
  corrected and approved; no further content changes unless implementation
  exposes a genuine contradiction.
- Create
  `Docs/superpowers/plans/2026-08-30-task-2803-layered-readme-restoration.md`:
  this execution plan.
- Modify `backlog/tasks/task-2803 - Rewrite-README-for-newcomers.md`: execution
  tracking, verification evidence, corrective acceptance criteria, and final
  status.

No Python, configuration, generated inventory, QA evidence, root cleanup, or
unrelated project documentation belongs in this corrective change.

### Task 1: Establish the Restoration and Accuracy Baselines

**Files:**

- Inspect: `README.md`
- Inspect from Git: `d2ff9c05ca91d7f7b7be80a2401f78f7142e1aff:README.md`
- Inspect: `pyproject.toml`
- Inspect: `Docs/GOAL.md`
- Inspect: `Docs/User_Guide/index.md`
- Inspect: `Docs/User_Guide/First_Run_Setup.md`
- Inspect: `Docs/User_Guide/settings.md`
- Inspect: `Docs/User_Guide/console.md`
- Inspect: `Docs/Development/release-recovery-setup.md`
- Inspect: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Inspect: `Docs/superpowers/trackers/unified-shell-maturity-roadmap.md`
- Inspect: `backlog/decisions/014-retire-legacy-navigation-chrome.md`
- Inspect: `backlog/decisions/015-shell-destination-ia.md`
- Inspect: `tldw_chatbook/UI/Navigation/shell_destinations.py`
- Inspect: `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py`
- Inspect: `tldw_chatbook/config.py`

- [x] **Step 1: Confirm the exact restoration source.**

  Run:

  ```bash
  git rev-parse 2f7d35fcd2b5153534475c7ae0755888183ef043^1
  git show d2ff9c05ca91d7f7b7be80a2401f78f7142e1aff:README.md | wc -l
  ```

  Expected: the first command prints
  `d2ff9c05ca91d7f7b7be80a2401f78f7142e1aff`; the second reports the original
  layered README length (886 lines at planning time).

- [x] **Step 2: Inventory the old README by keep, repair, and remove.**

  Run:

  ```bash
  git show d2ff9c05ca91d7f7b7be80a2401f78f7142e1aff:README.md | rg '^#{1,3} '
  ```

  Record the audit in working notes, not a repository file:

  - Keep: screenshot-led opening, requirements, core/optional installation,
    detailed feature families, configuration, web access, project structure,
    contributing, license, and contact.
  - Repair: value statement, Alpha framing, quick start, first conversation,
    current destination names, model/runtime explanation, extras, paths, and
    links.
  - Remove: duplicate local-model headings, opinionated recommendations,
    obsolete navigation claims, speculative features stated as complete,
    repeated configuration prose, and the all-extras install command.

- [x] **Step 3: Verify package-facing facts from `pyproject.toml`.**

  Confirm version `0.1.8.0`, Python `>=3.11`, Textual `==8.2.8`, scripts
  `tldw-cli` and `tldw-serve`, and every optional extra that will appear in the
  README. Do not copy an extra name from the old README without finding the
  corresponding key under `[project.optional-dependencies]`.

- [x] **Step 4: Verify product-facing facts from canonical maturity and navigation sources.**

  Use both product-maturity trackers to distinguish usable baseline workflows
  from incomplete or evolving surfaces. Use ADR-014 and ADR-015 for accepted
  destination ownership and naming, then confirm the currently registered
  routes in `shell_destinations.py`. Confirm first-run wizard behavior,
  **Settings › Diagnostics › Run setup wizard**, **Settings › Providers &
  Models**, configuration/data paths, and hosted/local model boundaries in
  current code and maintained guides.

- [x] **Step 5: Commit only if the baseline audit changes tracked files.**

  This task is normally read-only. If no tracked files change, do not create an
  empty commit.

### Task 2: Restore the Original Layered README and Repair Its Opening

**Files:**

- Modify: `README.md`

- [x] **Step 1: Restore the pre-PR #2045 README as the editing base.**

  Read the exact blob with `git show
  d2ff9c05ca91d7f7b7be80a2401f78f7142e1aff:README.md`, then use
  `apply_patch` to replace the current short README. Do not use `git checkout`,
  `git restore`, or a shell redirection that can overwrite unrelated work.

- [x] **Step 2: Rewrite the first two screenfuls.**

  The opening order must be:

  1. `# tldw_chatbook`, existing status/Python/license badges, and one direct
     sentence explaining the application.
  2. Current Console screenshot.
  3. Compact Alpha notice with explicit **Available now**, **Still evolving**,
     and **Goal** categories. The categories must be plain-spoken and brief,
     but none may be omitted.
  4. `## Why tldw_chatbook?` with concrete outcomes: hosted/local chat, local
     knowledge, media ingestion, RAG, roleplay, and controlled tools.
  5. `## Quick start` with clone, Python 3.11+ virtual environment, editable
     core install, `tldw-cli`, first-run wizard, and first Console message.
  6. Direct links to the User Guide and troubleshooting section.

- [x] **Step 3: Make the quick start copy-pasteable on supported platforms.**

  Keep Unix/macOS and Windows activation commands separate. Use
  `python -m pip install -e .` after activation and explain the `python3`/`py`
  executable choice without duplicating the whole sequence.

- [x] **Step 4: Describe the two model paths without ambiguity.**

  Give hosted providers and separately running local model servers equal
  visibility. State that local-first refers to application data ownership;
  hosted prompts cross the selected provider boundary, and local inference is
  performed by a configured server such as Ollama or llama.cpp.

- [x] **Step 5: Review the opening as a newcomer.**

  Verify that a reader can answer, before the detailed reference begins:

  - What is this?
  - Why would I use it?
  - Is it stable?
  - What do I need?
  - How do I launch it and send one message?

- [x] **Step 6: Commit the restored opening and quick start.**

  ```bash
  git add README.md
  git diff --cached --check
  git commit -m "docs: restore layered README opening"
  ```

### Task 3: Repair and Retain the Detailed Reference

**Files:**

- Modify: `README.md`
- Reference: `pyproject.toml`
- Reference: `Docs/User_Guide/`

- [x] **Step 1: Group capabilities by workflow before the feature reference.**

  Add a concise overview covering conversations, local knowledge and RAG,
  media and speech, roleplay, agents/tools/integrations, and evaluation. Keep
  the detailed original feature subsections below it when they provide useful
  setup or behavior information.

- [x] **Step 2: Correct the detailed feature terminology.**

  Replace retired primary labels with current public names such as Console,
  Library, Artifacts, Roleplay, Watchlists, Schedules, Workflows, MCP, ACP,
  Lab, Logs, and Settings. Remove claims that current code or maintained guides
  cannot support.

- [x] **Step 3: Repair optional installation guidance.**

  Keep an accurate extras table and practical combinations for RAG, web
  search, media, documents, speech, MCP, browser serving, local inference, and
  development. Remove the giant install-everything command. Mark large model
  downloads, native libraries, platform limits, and manually installed systems
  where the package metadata or maintained guide requires them.

- [x] **Step 4: Repair configuration and storage guidance.**

  Keep wizard-first setup, `~/.config/tldw_cli/config.toml`, profile-owned data
  under `~/.local/share/tldw_cli/`, environment-variable support, backups, and
  trust-boundary warnings. Link specialist provider/configuration documents
  rather than embedding a brittle full configuration file.

- [x] **Step 5: Repair web, project, and contributor sections.**

  Verify `tldw-serve`, browser-serving extras, project structure, development
  installation, focused/full test commands, contribution guidance, AGPL-3.0-or-
  later licensing, security-reporting wording, and contact information.

- [x] **Step 6: Remove only the identified junk inside the README.**

  Remove duplicate headings, stale migration notes, obsolete screen lists,
  recommendations presented as project policy, repeated prose, and broken
  links. Do not remove detailed material merely to meet a line count.

- [x] **Step 7: Commit the repaired reference.**

  ```bash
  git add README.md
  git diff --cached --check
  git commit -m "docs: repair README technical reference"
  ```

### Task 4: Capture and Add the Current Landing Screenshot

**Files:**

- Create: `Docs/static/tldw-chatbook-console.png`
- Modify: `README.md`

**Implementation deviation (2026-08-30):** A live profile was not launched.
The maintained, neutral Console SVG already used by the current User Guide was
rendered through an ignored temporary copy with its remote font declarations
removed. This avoided reading any user profile, key, conversation, or local
binding while preserving the current shell and Console UI. The first three
steps below were therefore satisfied through that verified documentation
source rather than a new runtime profile.

- [x] **Step 1: Create an isolated capture profile.**

  Create a new directory with `mktemp -d`. Inside it, use `apply_patch` to
  create `config.toml` containing only neutral demo settings, including
  `[general] users_name = "readme_demo"` and `[paths] data_dir` set to that
  same temporary directory's `data` child. Set `TLDW_CONFIG_PATH` to the
  absolute temporary `config.toml` path for every launch/capture command. Do
  not set or repurpose `HOME`, and do not rely on `XDG_DATA_HOME` (the app
  deliberately ignores it for data-path resolution).

  Before launch, run a read-only probe under the same `TLDW_CONFIG_PATH` and
  assert that `get_cli_config_path()` resolves to the temporary config and
  `get_user_data_dir()` resolves below the temporary `data` directory. Stop if
  either path escapes the temporary root.

- [x] **Step 2: Launch the current application with the isolated profile.**

  Use the repository's Python 3.12 virtual environment and the verified
  `TLDW_CONFIG_PATH`. Do not import keys, conversations, usernames, filesystem
  bindings, or profile state from the default user configuration.

- [x] **Step 3: Stage a representative Console view.**

  Show the current shell, Console conversation surface, model/context controls,
  and normal application chrome at a readable terminal size. Use neutral demo
  content and no transient error, diagnostic, approval, or setup state.

- [x] **Step 4: Capture the image to the selected path.**

  Save exactly `Docs/static/tldw-chatbook-console.png`. Crop only operating-
  system window chrome or empty margins; do not alter the application UI.

- [x] **Step 5: Inspect the image at original resolution.**

  Confirm legibility, current navigation, absence of secrets/private paths,
  and suitability as the first visual a newcomer sees. If a safe current
  capture cannot be produced, stop and report the blocker rather than reusing
  `Docs/static/Poc-Frontpage.png`, which shows the retired flat-tab interface.

- [x] **Step 6: Link the image with a repository-relative path and commit.**

  ```bash
  git add README.md Docs/static/tldw-chatbook-console.png
  git diff --cached --check
  git commit -m "docs: add current Console screenshot"
  ```

### Task 5: Validate Commands, Markdown, Links, and Scope

**Files:**

- Verify: `README.md`
- Verify: `pyproject.toml`
- Verify: `Docs/static/tldw-chatbook-console.png`
- Verify: `LICENSE`

- [x] **Step 1: Verify an isolated editable install without downloading dependencies.**

  ```bash
  check_dir="$(mktemp -d)"
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m venv --system-site-packages "$check_dir/venv"
  "$check_dir/venv/bin/python" -m pip install --no-build-isolation --no-deps -e .
  "$check_dir/venv/bin/python" -c "from importlib.metadata import distribution; d=distribution('tldw_chatbook'); assert d.version == '0.1.8.0'; assert any(ep.name == 'tldw-cli' and ep.value == 'tldw_chatbook.cli:main_cli_runner' for ep in d.entry_points)"
  ```

  Expected: exit 0; installed metadata reports version `0.1.8.0` and the
  documented `tldw-cli` entry point.

  The fresh virtualenv did not inherit `setuptools` from the repository
  virtualenv. The successful no-download run therefore supplied that existing
  site-packages directory through `PYTHONPATH` while keeping the editable
  installation and distribution metadata inside the fresh virtualenv.

- [x] **Step 2: Verify documented package facts and extras.**

  Run a read-only Python script that parses `pyproject.toml`, extracts extras
  mentioned in editable-install commands, and asserts every named extra exists.
  Also assert the README contains the current version, Python requirement,
  `tldw-cli`, `tldw-serve`, and `textual==8.2.8` where version detail is stated.

- [x] **Step 3: Validate Markdown structure and local links.**

  Run a read-only Python script that checks balanced fenced-code blocks,
  heading hierarchy, and every repository-relative Markdown link/image target.
  External URLs and anchors may be excluded from filesystem existence checks.

- [x] **Step 4: Run focused runtime and documentation checks.**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
    Tests/CI/test_textual_runtime_contract.py \
    Tests/UI/test_legacy_entrypoints_retired.py \
    Tests/UI/test_product_maturity_phase6_recovery_docs.py \
    Tests/UI/test_product_maturity_phase6_packaging_data_safety.py -q
  ```

  Expected: all selected tests pass. Record environmental warnings separately.

- [x] **Step 5: Run repository formatting and scope checks.**

  ```bash
  git diff --check origin/dev...HEAD
  git diff --name-status origin/dev...HEAD
  ```

  Expected: no whitespace errors; only the five files in this plan's file map
  are changed, with no application code, generated inventory, root artifacts,
  or unrelated cleanup.

- [x] **Step 6: Run the full suite required for pre-merge verification.**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest
  ```

  Expected: exit 0. If the suite exposes an upstream or environmental failure,
  compare against current `origin/dev`, document the evidence, and do not add an
  unrelated repair to this README branch.

  The fail-fast run collected 68,811 tests and stopped on
  `Tests/Actor_Packs/test_actor_pack_activation.py::test_create_new_persona_preserves_incoming_uuid`
  with `ActorPackExportError: actor_pack_actor_invalid`. The failing test and
  implementation blobs are byte-identical to `origin/dev`; this branch changes
  neither Actor Pack path, so no unrelated repair was added.

### Task 6: Review, Close, and Prepare the Corrective PR

**Files:**

- Modify: `backlog/tasks/task-2803 - Rewrite-README-for-newcomers.md`
- Verify: all files in the file map

- [x] **Step 1: Review every acceptance criterion against the corrected README.**

  Revalidate criteria #1–#10. Criteria #1–#6 were checked against the rejected
  README and cannot be carried forward without fresh evidence. Check all ten
  only after the restored source, opening, retained reference depth,
  screenshot, commands, links, and exact scope have been verified.

- [x] **Step 2: Update implementation notes but keep the task In Progress.**

  Record the restoration base, key repairs, screenshot path, commands/tests,
  modified files, and `ADR required: no`. Check the acceptance criteria that
  already have fresh evidence, but keep TASK-2803 In Progress until required PR
  review and CI have completed.

- [x] **Step 3: Run final fresh verification.**

  Repeat the Markdown/link audit, metadata/extras audit, focused tests,
  `git diff --check origin/dev...HEAD`, and exact changed-file review after the
  task-record edit.

- [x] **Step 4: Commit the in-progress task record and plan progress.**

  ```bash
  git add Docs/superpowers/plans/2026-08-30-task-2803-layered-readme-restoration.md \
    "backlog/tasks/task-2803 - Rewrite-README-for-newcomers.md"
  git diff --cached --check
  git commit -m "docs: prepare layered README correction"
  ```

- [x] **Step 5: Rebase, push, and open the corrective PR against `dev`.**

  Re-fetch and rebase onto current `origin/dev`, rerun all changed-scope and
  generated-repository gates required after a rebase, push the branch, and open
  a PR whose description explicitly says it corrects PR #2045 by restoring the
  original layered README. Do not mix Qodo/CI repairs unrelated to this diff
  into the branch.

- [ ] **Step 6: Address review and CI.**

  Evaluate review comments technically, apply valid README-specific fixes,
  resolve all threads, and rerun affected checks. Keep TASK-2803 In Progress
  while any required review thread or CI check remains unresolved.

- [ ] **Step 7: Create the task-only closeout after the content head is green.**

  After all review threads are resolved and required checks pass on the final
  README/screenshot content head, revalidate criteria #1–#10, check every
  criterion, finalize the implementation notes with that review/CI evidence,
  and set TASK-2803 to Done. Commit and push this task-only closeout.

- [ ] **Step 8: Verify the closeout head, then merge.**

  Wait for required checks on the task-only closeout head. If any check fails,
  reopen the task immediately and fix or document the failure before another
  closeout attempt. If the closeout head is green, confirm all review threads
  remain resolved and the PR still contains only the planned files, then merge
  into `dev`.
