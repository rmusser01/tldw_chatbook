# Newcomer-First README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan.

**Goal:** Replace the sprawling project README with a concise, accurate landing page that helps a newcomer understand the Alpha project, install the latest source checkout, connect a hosted or local model, and send a first Console message.

**Architecture:** This is a documentation-only change. `README.md` remains the project landing page, while deeper task guidance stays in the maintained `Docs/User_Guide/` and release-recovery documentation. All claims and commands are grounded in current package metadata, accepted navigation decisions, maturity trackers, and UI code; no runtime, dependency, or configuration behavior changes.

**Tech Stack:** GitHub-flavored Markdown, Python package metadata from `pyproject.toml`, Textual 8.2.8 UI terminology, Backlog.md task tracking.

---

## Task 1: Establish the verified README contract

**Files:**

- Inspect: `pyproject.toml`
- Inspect: `tldw_chatbook/UI/Navigation/shell_destinations.py`
- Inspect: `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py`
- Inspect: `tldw_chatbook/UI/Screens/settings_screen.py`
- Inspect: `tldw_chatbook/config.py`
- Inspect: `Docs/User_Guide/index.md`
- Inspect: `Docs/Development/release-recovery-setup.md`
- Inspect: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Inspect: `Docs/superpowers/trackers/unified-shell-maturity-roadmap.md`
- Inspect: `backlog/decisions/014-retire-legacy-navigation-chrome.md`
- Inspect: `backlog/decisions/015-shell-destination-ia.md`
- Inspect: `backlog/docs/lessons-testing-evidence.md`
- Inspect: `backlog/docs/lessons-live-verification.md`
- Inspect: `backlog/docs/lessons-backlog-hygiene.md`
- Modify: `backlog/tasks/task-2803 - Rewrite-README-for-newcomers.md`

- [ ] Confirm version `0.1.8.0`, Python `>=3.11`, Textual `==8.2.8`, the `tldw-cli` entry point, and representative optional-extra names directly from `pyproject.toml`.
- [ ] Confirm current public destination names from `shell_destinations.py`, using accepted ADRs to understand folds and treating current code as authoritative where historical labels differ.
- [ ] Use both canonical maturity trackers to separate verified baseline workflows from still-evolving integrations and parity work.
- [ ] Confirm first-run recovery through **Settings › Diagnostics › Run setup wizard** and direct provider repair through **Settings › Providers & Models**.
- [ ] Confirm the configuration path, local data path, and maintained documentation targets used by the README.
- [ ] Read the three required lessons documents before implementation and apply their evidence, live-verification, and task-hygiene rules.
- [ ] Use `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python` as the project interpreter for audits and tests in this worktree; confirm it imports the development/test dependencies before validation. Do not rely on an unqualified `python`, which is unavailable in this environment.
- [ ] Keep TASK-2803 in progress with the ADR outcome recorded: no ADR is required because this is copy-only alignment with existing behavior.

## Task 2: Rewrite the project landing page

**Files:**

- Modify: `README.md`

- [ ] Replace the current README with the approved information architecture: introduction, Alpha status, quick start, first conversation, capability overview, project direction, optional capabilities, configuration/data, troubleshooting/docs, contributing/license/contact.
- [ ] Keep the result approximately 250–350 lines and use progressive disclosure so the newcomer path stays above advanced material.
- [ ] Make the status section explicitly distinguish **Available now**, **Still evolving**, and **Goal**, without implying uniform maturity or full local/server parity.
- [ ] Make the source checkout the primary install route: clone, create a Python 3.11+ virtual environment, activate it on Unix/macOS or Windows, install with `python -m pip install -e .`, and launch with `tldw-cli`.
- [ ] Give hosted-provider and local-server users parallel setup paths that converge on opening Console and sending a first message. Include the wizard recovery and direct Settings paths.
- [ ] Describe capabilities with the current public names Console, Library, Artifacts, Roleplay, Watchlists, Schedules, Workflows, MCP, ACP, Lab, Logs, and Settings.
- [ ] Keep optional extras representative and source-grounded. Link deeper user and recovery guides instead of duplicating them.
- [ ] Remove duplicated feature prose, stale navigation names, the opinionated model recommendations, oversized configuration examples, and the obsolete PoC screenshot. Do not add a replacement screenshot unless a polished reader-facing current capture exists.

## Task 3: Validate the documented install and package contract

**Files:**

- Verify: `README.md`
- Verify: `pyproject.toml`

- [ ] Verify the source checkout can produce an isolated editable install without downloading runtime dependencies:

  ```bash
  check_dir="$(mktemp -d)"
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m venv --system-site-packages "$check_dir/venv"
  "$check_dir/venv/bin/python" -m pip install --no-build-isolation --no-deps -e .
  "$check_dir/venv/bin/python" -c "from importlib.metadata import distribution; d=distribution('tldw_chatbook'); assert d.version == '0.1.8.0'; assert any(ep.name == 'tldw-cli' and ep.value == 'tldw_chatbook.cli:main_cli_runner' for ep in d.entry_points)"
  ```

  Expected: both commands exit 0 and the isolated distribution exposes version `0.1.8.0` plus the documented `tldw-cli` entry point. The temporary directory is intentionally left for the OS to clean up; no user config or data path is touched.

- [ ] Verify README package literals and extras against `pyproject.toml` with an exact read-only script:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python - <<'PY'
  import re
  import tomllib
  from pathlib import Path

  readme = Path("README.md").read_text()
  project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
  assert project["version"] in readme
  assert project["requires-python"] in readme
  assert "textual==8.2.8" in readme
  assert "tldw-cli" in project["scripts"]
  documented_extras = {
      "embeddings_rag", "websearch", "mcp", "web",
      "audio", "video", "pdf", "ebook",
  }
  assert documented_extras <= set(project["optional-dependencies"])
  assert all(f"`{extra}`" in readme for extra in documented_extras)
  for extra in re.findall(r"\.\[([a-z0-9_-]+)\]", readme):
      assert extra in project["optional-dependencies"], extra
  PY
  ```

  Expected: exit 0 with no output.

- [ ] Do not run `tldw-cli --help`: the Textual entry point initializes application state before parsing help. The isolated editable-install metadata check above verifies the launch command without creating configuration, rebuilding source CSS, or starting a TUI.

## Task 4: Validate Markdown, links, tests, and scope

**Files:**

- Verify: `README.md`
- Verify: `LICENSE`

- [ ] Run this exact read-only Markdown and relative-link audit:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python - <<'PY'
  import re
  from pathlib import Path

  path = Path("README.md")
  text = path.read_text()
  assert text.count("```") % 2 == 0
  levels = [len(m.group(1)) for m in re.finditer(r"^(#{1,6}) ", text, re.M)]
  assert levels and levels[0] == 1
  assert all(next_level <= level + 1 for level, next_level in zip(levels, levels[1:]))
  targets = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)
  for target in targets:
      target = target.split("#", 1)[0]
      if not target or "://" in target or target.startswith("mailto:"):
          continue
      assert (path.parent / target).exists(), target
  PY
  ```

  Expected: exit 0 with no output.

- [ ] Run the focused runtime/recovery baseline:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/CI/test_textual_runtime_contract.py Tests/UI/test_product_maturity_phase6_recovery_docs.py Tests/UI/test_product_maturity_phase6_packaging_data_safety.py -q
  ```

  Expected: the focused runtime, recovery-documentation, and packaging/data-safety checks pass. Record unrelated pre-existing warnings separately.

- [ ] Run the repository test suite required by the Definition of Done:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest
  ```

  Expected: exit 0. If the environment cannot complete the suite, do not mark the task Done; report the exact blocker.

- [ ] Record static-analysis applicability: this diff changes Markdown and Backlog records only, and the repository has no configured Markdown linter/formatter. Use the exact Markdown audit plus `git diff --check` as the formatting/static checks; Python `mypy` is not applicable to unchanged Python source.
- [ ] Record security, performance, and license applicability: no runtime code, dependencies, data handling, permissions, or license text changes. Confirm the README links the existing `LICENSE`; no separate performance/security test is applicable.
- [ ] Run `git diff --check`, confirm `wc -l README.md` is in the approximate 250–350 target, and inspect `git diff -- README.md` for stale names, duplicated sections, unsupported promises, or unrelated edits.

## Task 5: Close the documentation task

**Files:**

- Modify: `backlog/tasks/task-2803 - Rewrite-README-for-newcomers.md`

- [ ] Check all six acceptance criteria only after the README and every required validation satisfy them.
- [ ] Add concise implementation notes covering the newcomer-first structure, removed stale material, verification evidence, modified files, and `ADR required: no`.
- [ ] Add a lessons entry only if this work surfaces a genuinely reusable incident not already covered by the required lessons documents.
- [ ] Set TASK-2803 to Done only after the full Definition of Done is satisfied.
- [ ] Commit the README, implementation plan, and completed task record with a documentation-focused commit message.
