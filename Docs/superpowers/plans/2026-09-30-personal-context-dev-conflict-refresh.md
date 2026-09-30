# Personal Context pinned-dev conflict refresh implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore PR #2862's integration with pinned dev after new upstream changes, preserving the reviewed memory behavior and native qualification limits.

**Architecture:** Merge the fixed dev commit into the existing isolated feature branch without rewriting history. Resolve the Settings split by composing both existing selector families and preserve both lesson additions. Qualify the actual combined source; earlier receipts remain attributed to their original commits.

**Tech Stack:** Native macOS Python >=3.12 (installed 3.12.11), Textual >=8.0.0,<9, SQLite, existing pytest/Ruff and CSS generators.

**Spec:** Docs/superpowers/specs/2026-09-25-personal-context-memory-evolution-design.md; TASK-25907.23 acceptance criteria; ADR-193 Unit A; ADR-097, ADR-150 and ADR-161.

ADR required: no
ADR path: backlog/decisions/097-boot-budget-ratchets.md; backlog/decisions/150-mcp-hub-bulk-permission-actions.md; existing ADR-193/161
Reason: Direct integration of existing owners and contracts; no new architecture, dependency, schema or policy is selected. Resolve actual canonical ADR filenames when linking this plan from the task.

## Global Constraints

- Python >=3.12; Textual >=8.0.0,<9. Use the existing native interpreter and installed dependencies.
- Work only in /Users/macbook-dev/.codex/worktrees/personal-context-memory-pr/tldw_chatbook on codex/personal-context-memory-dev.
- Feature pin: e3527ac883e32c94087293212ddc9d5223505adf. Dev pin: dfee4bf4c66ec2656a6c4ea2667edb63cc38a445. Do not chase later dev movement during this qualification.
- No full repository test sweep. Run native targeted tests for affected owners and required architecture/style/import/generated guards.
- Do not raise size, method, style, CSS, descriptor, import or deadline limits to obtain a pass. Distinguish incoming independently approved ratchet changes from local repairs; inspect their exact before/after values and retain stricter existing values where the same guard overlaps.
- Use synthetic private profiles and temporary storage. No production profile, keyring, provider or source calls.
- V2, Forget, source-opening, schema/AAD, disclosure and companion-server activation remain closed. No PR merge into dev.
- Preserve the independent retirement branch, primary checkout and foreign changes. No stash/reset/prune/gc, no force push, no manual generated stylesheet edits.
- Reuse existing owners and helpers; no speculative abstractions, extra dependencies or normalization of source prose/blank lines to evade size guards.

## Task 1: Integrate pinned dev and resolve owner conflicts

**Files:**
- Modify: tldw_chatbook/css/build_css.py, existing Settings ScreenOwnedSplit only.
- Modify: backlog/docs/lessons-testing-evidence.md, preserve both existing conflict sides.
- Generate through the canonical builder: tldw_chatbook/css/* generated outputs and their existing manifest metadata.
- Test: Tests/UI/test_widget_css_consolidation.py, Tests/UI/test_css_bundle_sync_guard.py, Tests/Architecture/test_module_size_ratchet.py, Tests/Architecture/test_screen_size_ratchet.py, Tests/Architecture/test_library_modules_size_ratchet.py.

**Interfaces:**
- Consumes: existing ScreenOwnedSplit(modules, sheets, prefixes, pinned), immutable feature/dev pins and the canonical CSS build entry point.
- Produces: one combined Settings ownership rule retaining personal-context and Hooks selectors, both lessons' exact bodies, and a conflict-free candidate with both parents preserved.

- [ ] Step 1: Verify the branch, root, clean baseline and existing linked-worktree identity. Record both parent SHAs and the conflict preview; the native baseline receipts apply only to e3527ac883.
- [ ] Step 2: Start the reversible local merge:

```bash
git -c gc.auto=0 merge --no-commit --no-ff dfee4bf4c66ec2656a6c4ea2667edb63cc38a445
git diff --name-only --diff-filter=U
```

Expected unmerged paths: build_css.py and lessons-testing-evidence.md. Unexpected conflicts require source inspection and a plan amendment before resolution.

- [ ] Step 3: Retain the existing two source modules and pinned tokens, with this combined prefix tuple:

```python
prefixes={
    "settings": (
        "settings", "personal-context", "console-hooks", "hook-review", "-wide-viewport"
    )
},
```

Preserve upstream's Hooks ownership comment and the existing memory/RecoveryPassphraseDialog ownership comment. For every lessons conflict concatenate the complete feature body and the complete incoming body, separated by one blank line; remove only Git conflict markers.

- [ ] Step 4: Inspect the automatically merged runtime/controller/screen changes against both parents. Confirm the new Hooks admission does not bypass the existing preparation/maintenance, fork publication and private source ownership guards. Compare all pre-refresh ratchet constants against the merged versions and the incoming ADR-097 exception provenance; no local relaxation.
- [ ] Step 5: Regenerate using the existing native builder:

```bash
python tldw_chatbook/css/build_css.py
```

Then run the targeted CSS ownership and architecture families, retaining XML/import/source receipts. Use the existing private-profile harness and native interpreter; failed checks enter root-cause diagnosis and a scoped plan amendment before code repair. Do not relabel a single passing rerun as resolution of an unknown intermittent failure.
- [ ] Step 6: Check conflict markers, exact lesson-body preservation and whitespace. Commit the conflict-free integration only after source/guard evidence is recorded; report any still-open native verification accurately in the task report.

## Task 2: Qualify and publish the actual combined candidate

**Files:**
- Modify: backlog/tasks/task-25907.23 - Integrate-Personal-Context-memory-improvements-onto-current-dev.md, owned plan/notes/criteria/status sections through Backlog CLI.
- Modify: backlog/docs/personal-context-memory-roadmap.md, current integration checkpoint.
- Test: current Console ownership/fork/maintenance, Hooks admission/run/review, private-profile coverage, Settings/provenance/Next Send, CSS/theme/boot/import guards, changed provider setup/readiness and diagnostic inventory owners.

**Interfaces:**
- Consumes: Task 1's combined candidate and exact pinned dev; existing public PR #2862 against dev and the saved qualification runner/receipt format.
- Produces: source-bound native receipts, reviewed integration diff, truthful PR description and current tracker status.

- [ ] Step 1: Select exact test modules from the automatically merged owner diff, including Tests/Chat/test_console_hook_admission.py, test_console_local_review_hook.py, test_console_chat_controller.py, test_console_runtime_lifetime.py, test_console_fork_mutation_fences.py, test_console_fork_transition_census.py and trace preparation/recovery controls. Add the existing memory UI/context/import groups and Tests/test_private_profile_coverage.py; record selection/exclusions before execution.
- [ ] Step 2: Run the selection natively with separate receipts for exact private children, current import paths, source hashes and descriptor categories. Run ./scripts/preflight.sh with the existing pinned public Mermaid inputs. Check changed Python parsing/scoped lint, full lint/format for newly authored files, task readability/IDs, local document links and whitespace.
- [ ] Step 3: Review the scoped integration package against both pins. If a reviewer finds a concrete defect, repair it in its shared owner and rerun the covering checks without budget/timeout waivers. Preserve historical failures and any unavailable-history skip explicitly.
- [ ] Step 4: Rewrite the local PR-description draft around the actual candidate, native outcomes and closed gates; verify it before publication. Push normally to the existing PR branch and update the description within the user's ongoing PR authorization. If automatic approval review rejects an exact export, retain the concrete draft and explain that rejection before requesting approval.
- [ ] Step 5: Verify remote head, base, description and current checks. Close all criteria and mark Done through Backlog CLI only when verified; mirror the roadmap milestone and publish documentation-only bookkeeping. Retain original qualified receipts/worktrees and keep activation/merge outside this scope.
