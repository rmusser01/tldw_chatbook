# Orchestration approval verification repair

> **For agentic workers:** Use subagent-driven-development to implement and review this task.

**Goal:** Restore the seven inherited approval UI regressions as reliable tests of the current approval contract.

**Architecture:** Keep the production approval host, decision projections, and answerable-time accounting. Repair test setup and readiness at the actual publication and mounted Console boundaries; every worker started by the affected fixtures has failure-safe teardown.

**Tech Stack:** Python, pytest, threading, Textual.

**Spec:** `backlog/tasks/task-13154.4 - Restore-Console-approval-UI-contract-coverage.md`; existing ADR-067 and ADR-094 Console turn lifetime.

ADR required: no
ADR path: backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md
Reason: restore verification of existing visibility and cancellation behavior; no new authority, timeout, or navigation policy.

## Global Constraints

- Work only in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/agent-orchestration-pr`, branch `codex/agent-orchestration-remaining`, starting from merged dev `d66908a69ef03066fed77a92edf77a326f44bd89`.
- Use `.superpowers/sdd/2026-09-11-agent-orchestration-pr-integration/venv/bin/python`; its import binding targets this worktree. Root's baseline provenance test passed.
- Run targeted tests only. No full suite, live provider requests, shared environment installation, production user configuration, or unrelated formatting.
- Preserve real approval outcomes, revocation, exact round/session ownership, and answerable-time accounting. Do not remove assertions, add skip/xfail, bypass production guards, or merely increase timeouts.
- Do not raise architecture, import, CSS, or diagnostic guard limits. No new dependencies.
- Root owns staging and commits. Leave all implementation edits unstaged and report exact paths. No subagents from implementers or reviewers.

### Task 1: Restore approval UI contract coverage (TASK-13154.4)

**Files:**
- Modify: `Tests/UI/test_console_parked_payload_rekey.py`.
- Modify: `Tests/UI/test_console_mcp_approval.py`.
- Update: `backlog/tasks/task-13154.4 - Restore-Console-approval-UI-contract-coverage.md`.
- Read: `Tests/UI/app_factory.py`, existing production Console readiness helpers, `tldw_chatbook/Chat/console_interrupt_rounds.py`, ADR-067, ADR-094.
- Any reproduced production defect needs a concrete root-cause report and root scope ruling before changing production.

**Interfaces:**
- Controller publication follows preregistration; seeing an ID in `_pending_approval_rounds` does not establish that its payload has mounted.
- Approval payloads use `round_id`; skill installation/script payloads use `request_id`. Identity, rather than an exact total of callback invocations, defines readiness.
- Positive timeout budgets are charged only while the exact session's FIFO head is answerable. Hidden and queued intervals do not consume the budget.
- Mounted Console tests must satisfy real startup/session readiness and then wait for the card/controls to render. Mounting the screen alone is insufficient.

- [x] Read the source and root's isolated baseline logs `approval-baseline-1.log` through `approval-baseline-7.log` under `.superpowers/sdd/2026-09-12-agent-orchestration-remaining/`. All seven failed individually on the unmodified baseline; no need to repeat unchanged baseline runs. Logs 1/2 identify preregistration-before-publication and obsolete wall-clock accounting; logs 3–7 identify hidden cards/setup-modal focus/zero geometry.
- [x] In `test_bridges_do_not_share_a_head`, wait for each exact published head before reading payloads. Use the existing `_wait_until` and identity helpers; keep both independent-head and clearing assertions. The required shape is:

```python
assert _wait_until(lambda: _mounted_round(ctrl) == approval_round)
assert _wait_until(
    lambda: bool(install_mounted)
    and install_mounted[-1] is not None
    and install_mounted[-1].get("request_id") == install_round
)
```

- [x] Replace the obsolete promotion test with a named answerable-time regression. Drive queued, answerable, and hidden intervals using the real host and an injected monotonic clock or existing deterministic clock seam. Verify the queued round retains its full 30-second budget until promoted, actual answerable time reduces it, hidden time does not, and projecting a card does not rewrite the retained payload. Do not replace the old failing inequality with an unqualified constant-only assertion.
- [x] Make the affected controller fixtures own their worker threads. On fixture teardown, revoke/cancel all outstanding rounds, join every started worker with a bounded timeout, and assert none remains alive. Use `yield`/`finally` cleanup so a failure before the normal decision path cannot leak a waiter. Preserve results and exception visibility.
- [x] Establish the missing production Console readiness in the five mounted tests using the existing app factory/session setup conventions. If startup schedules a later projection, wait for its completion before injecting the pending state. Verify the actual modal/card state to establish the cause. Do not patch away production readiness checks or force widget display/CSS.
- [x] Preserve the original five contracts and viewport sizes: finishing card displayed but excluded from pending count, Alt+A focuses a decision select, Alt+A reaches the card at 80 columns with inspector closed, batch controls have nonzero nonoverlapping geometry, and single-row fast buttons have nonzero nonoverlapping geometry. Retain the adjacent no-pending Alt+A behavior.
- [x] Run the two affected modules together (targeted affected-file verification), with no newly excluded nodes. Include the existing human-wait regression that previously failed only after leaked tests if it is outside those modules. Run a meaningful cleanup-failure or mutation probe in an isolated test process to prove waiter teardown does not depend on successful assertions. Record exact commands, counts, and warnings.
- [x] Check changed-line Ruff lint/format and `git diff --check`, preserving unrelated file-wide debt. Record implementation notes, evidence, and any remaining concerns. Do not mark Backlog Done before independent task review.
- [x] Leave changes unstaged. Root reviews the exact changed paths, commits them, generates a review package, and dispatches independent task review. Correct any actionable review findings with their covering tests; then root closes TASK-13154.4 via Backlog CLI.
