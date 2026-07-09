## Stage 1: Rebase And Conflict Resolution
**Goal**: Put PR #527 on top of latest `origin/dev` without dropping existing regression coverage.
**Success Criteria**: Rebase completes cleanly; conflicting chat tests retain both dev and PR assertions.
**Tests**: `git status --short --branch`; focused chat test collection.
**Status**: Complete

ADR required: no
ADR path: `backlog/decisions/006-provider-aware-generation-settings.md`
Reason: This is bounded review remediation implementing the existing provider-aware generation settings boundary; no new storage, sync, provider ownership, or cross-module contract decision is introduced.

## Stage 2: Review Regression Tests
**Goal**: Capture each unresolved review finding with focused failing tests before implementation where practical.
**Success Criteria**: Tests cover Anthropic `temperature`/`top_p` exclusivity, safe chat exception logging without `status_code`, and Console custom-model toggle/readiness state.
**Tests**: `./.venv/bin/python -m pytest Tests/Chat/test_chat_functions.py Tests/UI/test_console_session_settings.py -q`
**Status**: Complete

## Stage 3: Minimal Fixes
**Goal**: Apply narrow fixes to the Anthropic adapter, chat error logging, and Console settings modal.
**Success Criteria**: Explicit `top_p` is honored when temperature is not explicitly set, conflicting explicit samplers prefer temperature with a warning, chat logging cannot raise `AttributeError`, custom model width is named, and toggling back refreshes readiness.
**Tests**: Focused regressions from Stage 2.
**Status**: Complete

## Stage 4: Verification And PR Closeout
**Goal**: Verify, document, push, and resolve addressed review threads.
**Success Criteria**: Focused tests pass, Bandit reports no new findings in touched production files, `TASK-122` is complete, branch is pushed to PR #527, and review threads are resolved.
**Tests**: Focused pytest suite, `git diff --check`, Bandit on touched production files.
**Status**: Complete
