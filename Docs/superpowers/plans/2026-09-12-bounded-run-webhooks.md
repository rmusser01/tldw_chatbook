# Bounded run-webhook delivery implementation

> **For agentic workers:** Use subagent-driven-development to implement and independently review this task.

**Goal:** Complete TASK-31511 by bounding webhook admission and reusing delivery resources.

**Architecture:** A module-local lazy daemon worker owns a bounded Queue and reuses one asyncio Runner until idle retirement. The existing scheduler and delivery/security/config APIs remain the integration boundary.

**Tech Stack:** Python 3.11+, queue, threading, asyncio.Runner, pytest.

**Spec:** `backlog/decisions/153-bounded-run-webhook-delivery.md` and TASK-31511 acceptance criteria.

ADR required: yes
ADR path: backlog/decisions/153-bounded-run-webhook-delivery.md
Reason: explicitly record worker ownership, bounded admission/drop policy, idle cleanup, and rejected app-lifetime/executor alternatives.

## Global Constraints

- Work only in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/agent-orchestration-pr`, branch `codex/agent-orchestration-remaining`.
- Use `.superpowers/sdd/2026-09-11-agent-orchestration-pr-integration/venv/bin/python`; run targeted tests under pytest isolation only. No real outbound webhooks, live user config, full suite, shared venv edits or dependencies.
- One active delivery worker/loop generation, 32 queued notifications plus one in flight, FIFO, 30-second idle retirement, no blocking admission or shutdown join.
- Keep existing payload/HMAC/SSRF/subscription/timeout behavior, no retries, no added webhook-specific config cache, no production app/service/config rewiring.
- Root owns staging and commits. Workers leave edits unstaged and do not dispatch subagents or reviewers.

### Task 1: Bound reusable webhook delivery (TASK-31511)

**Files:**
- Modify `tldw_chatbook/Agents/run_webhooks.py`.
- Modify `Tests/Agents/test_run_webhooks.py`.
- Update TASK-31511, webhook user-guide wording, and diagnostic inventory only if the reviewed fixed diagnostics change it.
- Read `config.load_settings`, existing settings cache/invalidation tests and ADR-153. Existing scout report is `.superpowers/sdd/2026-09-12-agent-orchestration-remaining/webhook-design-investigation.md`.

**Interfaces:**

```python
WEBHOOK_DELIVERY_QUEUE_CAPACITY = 32
WEBHOOK_DELIVERY_IDLE_SECONDS = 30.0
WEBHOOK_DELIVERY_THREAD_NAME = "run-webhook-delivery"

@dataclass(frozen=True)
class _WebhookDelivery:
    config: WebhookConfig
    event: str
    run_id: str
    agent_id: str | None
    timestamp: str | None
    extra_ids: Mapping[str, str] | None
```

`_WebhookDeliveryWorker.submit(delivery) -> bool` admits immediately or refuses. Its constructor accepts injectable queue_capacity/idle_seconds for deterministic tests. Existing `schedule_run_webhook` signature stays intact, but True now means delivery admitted. Copy extra_ids before publication; do not retain caller-owned mutable containers.

- [ ] Run the existing webhook module as baseline. Add gated tests proving two admitted events share one thread and loop, submit returns while delivery is held, and a capacity-one worker refuses a third event with one in flight and one waiting. Observe the intended failure against per-event scheduling before implementation.
- [ ] Add the immutable record and module-owned worker. Keep the scheduler's current eligibility check before admission. Under a short state lock, start/reuse an exact generation and use `put_nowait`; no queue wait, join or network work runs under admission.
- [ ] Run a FIFO consumer inside one `asyncio.Runner` context. Ensure task_done and failure containment for each delivery. Preserve deliver_webhook's transport/policy behavior and continue with later admitted notifications after an unexpected per-delivery exception.
- [ ] Implement exact-generation idle retirement with a queue recheck under the admission lock. Close the Runner before exposing retirement; submissions arriving during retirement must either have a live consumer or return False, never strand a True admission. Failed start unwinds any queued work and lets a later call retry starting. Avoid test-only production callbacks; use controlled Queue/thread wrappers if a race needs gating.
- [ ] Emit only fixed queue/worker diagnostic wording and bounded metric labels. Capture a URL/secret/run-ID canary during saturation/start failure and assert none appears in new diagnostics. Exception type is sufficient; never log arbitrary exception repr supplied by a delivery callback.
- [ ] Prove exact worker retirement/restart, retirement/admission race, failed thread start followed by recovery, unexpected callback failure followed by success, and immutable config/extra_ids captured at admission. Every test releases gates and joins its exact owned worker even on assertion failure.
- [ ] Retain all current signature, egress, missing-secret, HTTP failure, timeout-clamping, subscription and AgentService terminal mapping tests. Run the targeted settings cache/invalidation modules to confirm the old warm-disk-read premise is already resolved; do not add a duplicate cache merely to make a test pass.
- [ ] Run `Tests/Agents/test_run_webhooks.py`, `Tests/test_config_settings_cache_concurrency.py`, and `Tests/test_config_hot_reload.py` using the isolated interpreter. Record exact results/warnings, scoped Ruff/format and whitespace checks. If diagnostics changed, inspect statement differences and regenerate only the derived inventory; do not alter guard limits.
- [ ] Update scheduler docs and user guide to distinguish admission/delivery and disclose bounded best-effort drops. Add TASK-31511 notes and ADR link; leave In Progress until independent review. Leave edits unstaged and write the task report with evidence, paths, and concerns. Root commits/reviews and closes through CLI.
