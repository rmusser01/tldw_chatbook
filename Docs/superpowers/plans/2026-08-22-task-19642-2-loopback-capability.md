# TASK-19642.2 Loopback Capability Skip Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the two real-socket provider-gateway loop-ownership tests explicitly skip when the host cannot construct a numeric-loopback listener, while preserving their real HTTP/1.1 coverage and blocking external network access.

**Architecture:** Keep the existing shared stdlib HTTP server fixture and both concurrency scenarios. Catch only `PermissionError` at listener construction, use the repository's numeric-loopback-only marker, and pin both the positive skip classification and the negative non-permission error path with fixture-contract tests.

**Tech Stack:** Python 3.11+, pytest fixtures/markers, stdlib `http.server` and `threading`, repository `Tests.network_guard` policy.

---

## File Structure

- Modify `Tests/Chat/test_console_provider_gateway.py`: add two fixture-contract tests, classify only listener-construction `PermissionError` as a capability skip, and narrow the two assigned tests to `loopback_network`.
- Modify `backlog/tasks/task-19642.2 - Make-provider-gateway-loopback-tests-sandbox-safe.md`: record this implementation plan and, after verification/review, completion evidence.
- Modify `Docs/superpowers/plans/2026-08-22-task-19642-2-loopback-capability.md`: check off executed steps if implementation follows this plan.
- No production, network-guard, marker-registration, dependency, or ADR files change.

### Task 1: Pin listener-construction error classification

**Files:**
- Modify: `Tests/Chat/test_console_provider_gateway.py:2870-2884`
- Test: `Tests/Chat/test_console_provider_gateway.py`

- [x] **Step 1: Add the two fixture-contract tests below the fixture**

```python
def test_local_http_server_permission_denied_skips_with_capability_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def deny_listener(*_args, **_kwargs):
        raise PermissionError("sandbox denied loopback bind")

    monkeypatch.setitem(globals(), "_DeepBacklogHTTPServer", deny_listener)

    with pytest.raises(pytest.skip.Exception) as exc_info:
        next(local_http_server.__wrapped__())

    assert str(exc_info.value) == "loopback listener unavailable: permission denied"


def test_local_http_server_non_permission_oserror_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_listener(*_args, **_kwargs):
        raise OSError("address resources exhausted")

    def fail_if_skipped(reason: str) -> None:
        pytest.fail(f"unexpected capability skip: {reason}")

    monkeypatch.setitem(globals(), "_DeepBacklogHTTPServer", fail_listener)
    monkeypatch.setattr(pytest, "skip", fail_if_skipped)

    with pytest.raises(OSError, match="address resources exhausted"):
        next(local_http_server.__wrapped__())
```

The constructor replacements open no sockets, so neither regression gets a
network marker. The negative case is required: replacing `pytest.skip` with an
explicit failure ensures that broadening the implementation to `except OSError`
fails this gate instead of being counted as a successful pytest skip.

- [x] **Step 2: Run the fixture-contract tests and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_provider_gateway.py::test_local_http_server_permission_denied_skips_with_capability_reason \
  Tests/Chat/test_console_provider_gateway.py::test_local_http_server_non_permission_oserror_propagates \
  --tb=short
```

Expected: one failure and one pass. The permission-denied case fails because the fixture still propagates `PermissionError`; the non-permission case already passes.

- [x] **Step 3: Save the RED evidence in the task working notes, but do not weaken either assertion**

Use `backlog task edit 19642.2 --notes` only if the task's existing notes need the exact RED result preserved before the implementation commit.

The task had no pre-existing notes requiring a pre-commit amendment. The exact
RED result is preserved in the final Implementation Notes.

### Task 2: Implement the minimal capability skip and narrow markers

**Files:**
- Modify: `Tests/Chat/test_console_provider_gateway.py:2870-3155`

- [x] **Step 1: Catch only `PermissionError` at listener construction**

Change the fixture setup to:

```python
@pytest.fixture
def local_http_server():
    try:
        server = _DeepBacklogHTTPServer(("127.0.0.1", 0), _JSONOKHandler)
    except PermissionError:
        pytest.skip("loopback listener unavailable: permission denied")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=2)
```

Do not catch `OSError`, do not add a preflight socket, and do not replace the server with `httpx.MockTransport`.

- [x] **Step 2: Narrow only the two assigned test markers**

Replace `@pytest.mark.allow_network` with `@pytest.mark.loopback_network` on:

- `test_owned_http_client_survives_agent_bridge_style_loop_swap`
- `test_active_http_client_concurrent_swap_never_leaves_client_bound_to_wrong_loop`

Update the nearby comment so it says the tests permit only the numeric loopback listener they own and may explicitly skip when the host denies listener construction.

- [x] **Step 3: Run the fixture-contract tests and verify GREEN**

Run the exact Task 1 command.

Expected: `2 passed`. Mutation check: temporarily broadening `except PermissionError` to `except OSError` must make `test_local_http_server_non_permission_oserror_propagates` fail; restore the narrow catch immediately afterward.

- [x] **Step 4: Run the assigned nodes in the restricted host**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_provider_gateway.py::test_owned_http_client_survives_agent_bridge_style_loop_swap \
  Tests/Chat/test_console_provider_gateway.py::test_active_http_client_concurrent_swap_never_leaves_client_bound_to_wrong_loop \
  -rs --tb=short
```

Expected on this restricted host: `2 skipped`, each with exactly `loopback listener unavailable: permission denied`, and no setup error.

- [x] **Step 5: Verify marker narrowing at collection**

Run both exact assigned nodes with `--collect-only -m loopback_network`, then with `--collect-only -m allow_network`.

Expected: the loopback selection collects both nodes; the unrestricted-network selection deselects both.

- [x] **Step 6: Commit the tested behavior**

```bash
git add Tests/Chat/test_console_provider_gateway.py
git commit -m "test(chat): skip unavailable loopback listener"
```

### Task 3: Prove capable-host non-vacuity and network confinement

**Files:**
- Verify: `Tests/Chat/test_console_provider_gateway.py`
- Verify: `Tests/test_network_guard.py`

- [x] **Step 1: Run the two assigned nodes with numeric-loopback capability**

Run the exact two-node command from Task 2 outside the restricted filesystem/network sandbox, without changing pytest markers or enabling external destinations.

Expected: `2 passed`, proving both existing real HTTP/1.1 concurrency scenarios execute rather than skip when the OS permits the owned listener.

- [x] **Step 2: Run the loopback network-policy owner checks on the capable host**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/test_network_guard.py::test_loopback_destination_classification_is_numeric_and_family_specific \
  Tests/test_network_guard.py::test_network_mode_rejects_conflicting_loopback_and_allow_all_markers \
  Tests/test_network_guard.py::test_loopback_only_test_connects_owned_listener_and_blocks_remote_ip \
  Tests/test_network_guard.py::test_loopback_only_mode_covers_connect_connect_ex_and_sendto \
  --tb=short
```

Expected: `4 passed`. The owner tests prove numeric loopback is allowed while remote destinations remain blocked.

- [ ] **Step 3: Run the complete modified-module gate**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_provider_gateway.py \
  --tb=short
```

Expected: all module tests pass on the capable host. If a failure is unrelated, stop and diagnose it; do not broaden this task or run the full repository suite.

Deliberately omitted per explicit user scope. The complete module was not run;
the replacement verification comprised the exact two fixture-contract nodes,
two assigned consumer nodes in restricted and capable modes, and four
network-policy owner nodes.

- [x] **Step 4: Run static checks for the sole modified Python module**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check Tests/Chat/test_console_provider_gateway.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check Tests/Chat/test_console_provider_gateway.py
git diff --check origin/dev...HEAD
git status --short
```

Expected: Ruff and diff checks pass and the worktree contains only the planned task/plan bookkeeping before closeout.

Ruff check and `git diff --check` passed. Ruff format check exited 1, but the
normalized formatter diff exactly matched pinned pre-change base `96bee228f`;
no new formatting debt was introduced and no bulk format was applied.

### Task 4: Review and close TASK-19642.2

**Files:**
- Modify: `backlog/tasks/task-19642.2 - Make-provider-gateway-loopback-tests-sandbox-safe.md`
- Modify: `Docs/superpowers/plans/2026-08-22-task-19642-2-loopback-capability.md`
- Review: `Tests/Chat/test_console_provider_gateway.py`

- [x] **Step 1: Dispatch an independent requirements review**

Have a reviewer confirm the diff implements every acceptance criterion and no
unapproved production or network-policy changes. The review must verify the two
existing scenarios still contain their real-socket assertions and are not
vacuous on the capable-host run.

- [x] **Step 2: Dispatch an independent correctness/security review**

Have a separate reviewer check especially that only `PermissionError` skips,
the reason is exact, all other setup/runtime errors fail, markers are
loopback-only, server cleanup is unchanged after successful construction, and
no external network path was enabled.

- [x] **Step 3: Update task evidence and documentation**

Use the Backlog CLI to check all three acceptance criteria, add concise Implementation Notes with RED/GREEN, restricted-host, capable-host, network-policy, module, and static evidence, and document:

```text
ADR required: no
ADR path: N/A
Reason: test-fixture correction using the existing TASK-15111 loopback policy.
```

No lessons entry is expected unless implementation uncovers a new repeatable trap beyond the committed design.

- [x] **Step 4: Mark the task Done only after every gate and review is green**

```bash
backlog task edit 19642.2 -s Done
```

- [x] **Step 5: Commit closeout bookkeeping**

```bash
git add \
  Docs/superpowers/plans/2026-08-22-task-19642-2-loopback-capability.md \
  "backlog/tasks/task-19642.2 - Make-provider-gateway-loopback-tests-sandbox-safe.md"
git commit -m "docs(backlog): close TASK-19642.2"
```

- [ ] **Step 6: Run final clean-tree verification before claiming completion**

Re-run the two fixture-contract nodes, the two assigned nodes in both restricted and capable modes, the four network-policy owner nodes, Ruff, `git diff --check origin/dev...HEAD`, and `git status --short`. Record exact counts and the final commit SHA.

### PR Review Remediation

- [x] Reopen TASK-19642.2 while addressing Qodo's two maintainability findings.
- [x] Define the canonical listener-permission skip reason once and reuse the named constant in the fixture and assertion.
- [x] Add concise Google-style intent docstrings, including fixture argument documentation, to both new contract tests.
- [x] Re-run the restricted contract/consumer gate: 2 passed and 2 exact-reason skips.
- [x] Re-run the capable-host consumer/policy gate: 6 passed.
- [x] Re-run Ruff and whitespace checks on the review diff.

The named constant is a documented deviation from Task 2's original
no-constant constraint. Qodo's active repository compliance rule requires
centralizing the repeated semantic literal; no helper abstraction or production
change was introduced.

Deliberately not completed as written: the user prohibited code-test reruns
during closeout. The accepted fresh independent functional, policy, and Ruff
evidence is recorded in the task. The bookkeeping-only portion was completed
after the closeout commit: the committed range passed `git diff --check`, the
worktree was clean, and the rendered task was Done with all AC checked and
Implementation Notes present.
