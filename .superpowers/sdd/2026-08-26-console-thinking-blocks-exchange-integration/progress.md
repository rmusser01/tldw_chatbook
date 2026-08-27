# SDD ledger — plan: Docs/superpowers/plans/2026-08-26-console-thinking-blocks-exchange-integration.md

Setup: isolated worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/console-thinking-blocks`, branch `codex/console-thinking-blocks`.

Dependencies: TASK-18932.1 complete at `9906b4d4bd`; TASK-18932.2 complete at `7ec7fbc9dc`; TASK-18932.3 complete at `402ec260b4`. Existing ADR-090 governs; no new ADR is required.

Execution: serial subagent-driven development with RED/GREEN evidence, spec review, then code-quality review for each task. Full-suite verification remains excluded unless the user opts in.

Task 1: round-trip supported thinking and replay policy through selected-conversation JSON and Chatbook V2, with whole-conversation preflight and shared sensitivity warnings.
