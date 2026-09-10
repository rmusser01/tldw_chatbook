# Shared search backend default

Goal: make the saved search backend predict where basic and deep search run.

Architecture: extend the existing tool implementation's config resolution and
local provider schema. Reuse `SearchSettings.search_provider_default` and the
Settings config owner. Add no persistence layer, dependency or permission gate.

Tech stack: Python, existing TOML config owner, local tool provider, pytest.

Spec: the shared-default decision approved during the Settings UX review on
2026-09-09; explicit per-call override wins and remains temporary. For a
missing saved preference, use DuckDuckGo; preserve existing saved values.

ADR required: yes
ADR path: backlog/decisions/032-local-agent-tool-permission-boundary.md
Reason: amend the existing search destination-selection contract, with a
shared preference and invocation-time resolution.

Task: [TASK-32193](../../../backlog/tasks/task-32193%20-%20Share-the-saved-search-backend-across-basic-and-deep-search.md).

## Constraints

- Isolated worktree: `codex/shared-search-backend-default`.
- Reject malformed/unsupported explicit and saved values before dispatch.
- No automatic provider fallback; no credential or endpoint rewrites.
- Results identify engine and source, including cached results.
- Preserve local tool permissions, deep-search opt-in, budgets and deadlines.
- Backend adapter repairs and a new first-class Settings editor remain separate
  findings from the review; this task establishes the shared runtime behavior.
- Targeted verification only. No real provider traffic is required to prove
  preference routing; replace external search/LLM calls and use the real config
  owner and tool handlers in regression tests.

## Implementation

1. Write failing regression tests for saved preference updates, explicit
   overrides, missing settings, malformed values and cache provenance. Run them
   with the executable import-provenance check.
2. Add a shared resolver in `Tools/web_tool_impls.py`; both tool paths and the
   common research parameter builder consume it. Read Settings on each call.
3. Remove the fixed schema default and handler injection in
   `Agents/local_tool_provider.py`. Describe the shared preference and temporary
   override in both schemas.
4. Attach engine/source text outside the cached search body, reserve its byte
   budget, and preserve deep-search footer accounting.
5. Update config template and search documentation. Run targeted regression,
   permission, cache and output-budget tests, static checks, and final review.
   Record evidence and remaining review scope in the task before marking Done.

Review amendment: preserve bounded provenance on deep-search no-relevance and
deadline returns. Handle shared-builder validation errors in Console
`/research` before launching, instead of swallowing them and starting a run
with missing parameters. These are integration requirements of the shared
resolver, covered by task AC #6 and a mounted Console regression test.
