---
id: TASK-32193
title: Share the saved search backend across basic and deep search
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 19:30'
updated_date: '2026-09-09 19:49'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A saved search preference should predict where both basic and deep searches run. Explicit one-call choices must remain temporary, and invalid saved choices must not redirect searches to another provider.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Omitted backend arguments use the saved SearchSettings search_provider_default in both workflows and Settings saves affect the next invocation.
- [x] #2 Profiles without a saved backend use DuckDuckGo; existing saved choices remain unchanged.
- [x] #3 An explicit valid backend overrides one invocation without changing the saved preference; invalid values fail before provider dispatch.
- [x] #4 Results identify the effective backend and whether it came from an override, saved preference, or application default, including cache hits.
- [x] #5 Targeted tests verify both tool paths, config persistence, cache provenance, and existing permission and output limits.
- [x] #6 Console /research reports an invalid saved backend before launching a run, preserving the shared parameter builder's validation diagnostic.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add regression tests through the real Settings config adapter and both local tool handlers, replacing only external search and LLM pipeline calls.
2. Resolve the existing saved preference at invocation time, share validation and missing-value fallback, and remove the basic tool schema's fixed default.
3. Report effective engine and preference source without caching invocation-specific provenance or exceeding result budgets.
4. Update configuration guidance and run targeted default, cache, permission, deep-search, config, lint and import-provenance checks. Review the final diff.
5. Handle the shared builder's new validation error in Console /research before launch; independent review identified that its existing broad exception handler otherwise swallowed the error.

ADR required: yes
ADR path: backlog/decisions/032-local-agent-tool-permission-boundary.md
Reason: extend the existing local web-tool destination-selection contract with shared saved-default precedence. An addendum preserves the existing permission boundary without creating a parallel ADR.

Detailed plan: Docs/superpowers/plans/2026-09-09-shared-search-backend-default.md

PR preparation: renamed TASK-32187 to TASK-32193 because origin/fix/boot-css-ratchet-paydown now owns 32187. Fresh scan covered 224 remote refs and 32 worktrees; maximum 32192.

Original task ID note: the CLI created local ID 32051, already allocated elsewhere. Renumbered to 32187 after checking all-ref task additions and local worktrees (maximum 32186).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Basic and deep search now resolve explicit backend > saved preference > DuckDuckGo at invocation time. Existing preferences are preserved; overrides never persist. Removed the basic schema's fixed default, added bounded engine/source reporting outside cached result bodies, and surfaced invalid saved-backend errors before Console /research launches.

- Runtime changes: `Tools/web_tool_impls.py`, `Agents/local_tool_provider.py`, `config.py`, and the narrow Console /research error path in `UI/Screens/chat_screen.py`.
- Documentation: shared-default addendum in ADR-032, implementation plan, and Console user-guide instructions for the existing Advanced Config editor. No new dependency, storage schema or permission model.
- Evidence: 319 targeted tests passed, including real Settings saves, both local tool handlers, temporary overrides, malformed settings, cache provenance/byte limits, deep-search empty/deadline paths, existing permission tests, Research-window parameter assembly, and mounted Console error feedback. Import-provenance check confirms this worktree.
- Static checks: Ruff E9/F63/F7/F82 passed on all modified Python files; full Ruff checks passed on the new shared-backend test and changed Console-command test; touched formatting ranges and those test files passed formatter checks; `git diff --check` passed.
- Independent code review found two integration gaps (empty-result provenance and swallowed Console setup errors); both were reproduced, fixed and re-reviewed with no remaining findings. A cache-label budget edge case was also reproduced and covered. Existing requests/SWIG dependency warnings remain unrelated to the changes.
- Verification command: `PYTHONPATH="$PWD" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/test_probe_import_provenance.py Tests/Tools/test_shared_search_backend.py Tests/Tools/test_web_deep_search.py Tests/Tools/test_web_tool_impls.py Tests/Agents/test_local_tool_provider.py Tests/UI/test_console_research_command.py Tests/UI/test_research_screen.py::test_window_engine_start_passes_configured_pipeline_params -q --no-cov --tb=short --show-capture=no`.
- Work is preserved uncommitted in `.worktrees/shared-search-backend-default`, branch `codex/shared-search-backend-default`. No merge or push. Provider credential/endpoint adapter repairs and a guided Web Search settings category remain separate UX-review findings. Verification used offline external-call doubles; it does not establish live provider availability.

PR integration: moved onto dev 86a8054edb, preserving current TLS, profile/footer, privacy and config publication behavior. Final evidence and baseline limitations: Docs/superpowers/reviews/2026-09-09-search-settings-pr-integration.md (525 targeted cases passed across two runs, three live cases skipped).
<!-- SECTION:NOTES:END -->
