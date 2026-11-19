# Final whole-branch fix wave — requirements

Worktree:/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook
FIX_BASE:459e666970ef9e7aa4e148705cddafb9621a72d1
Branch:codex/console-chat-starts

Read this first. Read final-branch-review.md for the complete findings and exact probe evidence, global-constraints.md, the approved spec linked below, and the current TASK33805 file before code changes. This is the single final fix wave: address all four reported findings together. The controller will arrange exactly one scoped rereview afterward.

## Binding contracts

Spec:Docs/superpowers/specs/2026-10-02-console-chat-destinations-and-starts-design.md. ADR:backlog/decisions/211-console-chat-destinations-and-bounded-starts.md. Existing approved contracts apply; no new ADR or schema is needed for these corrections.

- Recheck destination availability and source execution ownership before mutation and again before launch; a removed or archived destination cannot be retargeted. Preserve true same_workspace/casual identity and the existing fork_chat contract.
- Required runtime support, current enablement and owner state must hold before native acceptance. A controller update disabling the runtime while readiness is paused must refuse this attempt before either acceptance receipt/provider dispatch. Retain the draft and settle only this prepared attempt/claim conservatively.
- The new_chat approval preview must name nonblank standing instructions as an explicit override and show their full body. Describe session remembering as covering later requests in the same mode and resolved destination, including supplied prompt/instructions bodies. Retain both grant-cache scopes and denial limits.
- Unavailable workspace Persona defaults use the existing plain fallback and visible notice. Preserve the actual resolver notice through trusted preview and normal target notice ownership; remembered/no-card paths also need visibility. Do not invent a second resolver, authority, lifecycle owner or body-bearing budget metadata.
- Preserve both durable acceptance fences, exact worker drain/claim ownership, no replay, machine provenance/profile exclusion, shared canonical allowance, versioned draft custody and all prior manual-recovery fixes.

## Scope and ownership

Read current owner code and use the smallest existing seams. Likely owners:Chat/console_chat_controller.py, existing workspace/default resolution owner if needed, Widgets/Chat_Widgets/chat_create_confirm_card.py, and necessary request/token data plumbing. Add covering tests in the existing Chat create/start/card owners. Include actual registry/SQLite archived destination cases before mutation and launch, actual controller update_agent_runtime(enabled=False,bridge=existing) while readiness is held, mounted complete approval copy/body, and ordinary missing-Persona notice callback/target field. Do not relax production guards to satisfy old fakes.

Read backlog/docs/design-language.md before widget/UI changes, and lessons-testing-evidence.md plus lessons-live-verification.md. Use existing tokens/classes; no new keybindings, settings, dependencies or styling sweep. Targeted tests only; no full suite. Do not touch the original dirty checkout, merge, push, PR, accepted ADRs, controller-owned Backlog task/plan/QA edits, or unrelated deferred baseline issues.

No helper or review subagents. You are the only final fixer. Controller owns all review dispatch and task closure. TASK33805 is reopened In Progress with AC2/3/5/6 unchecked; its existing plan governs these repairs.

## Verification and report

Reuse /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python. Worktree writes/tests/cache/commits require exec_command require_escalated because sandbox roots were not extended; keep scope explicit. Record before-change formatter baselines for every old Python owner amended. Use scoped Ruff E9/F63/F7/F82, full formatting of the three new Python files, affected old formatter ratchets against committed HEAD, and diff check. No broad formatting.

First RED each reported mechanism with meaningful real-owner tests, then GREEN and covering changed-owner regressions. Record exact command/output/log and self-review. Preserve old known baseline fixture diagnostics honestly; do not sum overlapping test selections.

When production edits are stable and focused GREEN, notify the controller before UI-heavy pytest. Controller may run final-fix-live-case.md in the existing isolated normal app/local provider. Hold heavy tests until that app closes, then finish targeted owner groups.

Write final-fix-report.md here, append a short pointer/closure summary to task-2-report.md, and commit only your owned fix/test/guide files. Keep controller task/plan/QA modifications out of the commit. Use git -c gc.auto=0 commit; no Git pruning. Run postcommit checks and return only status, SHA, one-line verification, concerns and report path. Do not mark the task Done.
