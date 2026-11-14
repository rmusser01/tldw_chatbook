# Served tab isolation and STT recovery implementation plan

> Use the existing isolated checkout. Execute the web-session and STT diagnostic domains in parallel under superpowers:dispatching-parallel-agents, with the parent implementing retry lineage and integration. Run a fresh combined review after focused verification.

**Goal:** Let two tabs share login safely, preserve successful retried transcripts, and attribute Parakeet failures without private native data.
**Architecture:** Authentication remains device/login scoped. Terminal/Canvas identity becomes page scoped beneath that authenticated login. Library job retry links remain separate from complete STT predecessor-attempt provenance; diagnostics extend the existing typed worker failure with finite content-free facts.
**Tech Stack:** Python 3.12+, Textual 8, aiohttp/textual-serve, existing browser shell JavaScript and pytest.
**Spec:** backlog/tasks/task-37143 - Fix-served-tab-isolation-and-ingestion-retry-failure-attribution.md and the user's authorized follow-up.

ADR required: yes for terminal identity; no new ADR for the routine retry guard and bounded diagnostic extension.
ADR path: backlog/decisions/251-served-terminal-page-isolation.md (amends ADR-121/233).
Reason: page terminal identity must remain subordinate to device authentication and exact-child Canvas ownership. STT retains existing persistence and privacy boundaries (ADR-025).

## Constraints and review focus

- No authentication/CSRF bypass, cross-login Canvas access, or tab takeover; logout/expiry still close every channel under that login.
- Page rendering allocates no pending server state; malformed IDs fail closed. Refresh and delayed old-child teardown must preserve other tabs.
- Preserve complete valid retry provenance; a missing predecessor STT attempt must never be fabricated from an unrelated job ID.
- No raw exception messages, paths, transcript/source bytes, credentials, tracebacks or dynamic class names enter diagnostics; cancelled/stale attempts remain quiet.
- No new production dependencies, real STT model downloads or full local suite. The user's later publication request authorizes committing, opening a PR against dev, rebasing, obtaining a fresh Superpowers review and merging once required CI passes.

## Task 1: independent served tabs

Files: Web_Server/serve.py, static/served_shell.html and served_shell.js; related Tests/Web_Server and Tests/Canvas; ADR-251 and Docs/Architecture/web-server.md.
- [x] Write regression: two real authenticated WebSockets sharing one cookie start different echo children and can exchange independent frames; closing/refreshing one leaves the other usable.
- [x] Run RED against current session-cookie binding (second child never starts).
- [x] Render a fresh page ID, use the owned WebSocket shim and Canvas poll to carry it, and use an authenticated-login/page composite for existing exact-child scope. Mounted Canvas checks only matching active composites owned by that login. Keep old direct fixture scope supported, with strict validation for supplied IDs and bounded active state.
- [x] Cover unrelated-login Canvas denial, Canvas disabled, page refresh and script-cache versioning through runtime behavior. Run focused tests GREEN.

## Task 2: retry metadata

Files: app_ingest_queue.py; Tests/App/test_submit_library_ingest_job.py and/or a focused Tests/STT integration regression.
- [x] Exercise real registry failure/requeue and option construction for a source with no STT failure provenance; build successful normalized transcript provenance and require success with no STT predecessor fields while the Library job retry link remains.
- [x] Observe RED with `retry_of_job_id requires retry_of_attempt_id`.
- [x] Emit STT predecessor job/attempt fields only together with a complete validated failed-attempt snapshot. Preserve complete lineage unchanged. Run regression and existing retry/provenance cases GREEN.

## Task 3: bounded Parakeet diagnostics

Files: STT/parakeet_onnx.py, executor.py, executor_worker.py and a small typed diagnostic helper if needed; app_ingest_queue.py logging; Tests/STT and Tests/Library/test_stt_failure_diagnostics.py.
- [x] Add stage-specific failure regressions around real runtime/worker paths using synthetic native dependencies. Require finite safe failure stage/category to survive worker pickling and reach the parent error log once.
- [x] Observe RED for missing underlying attribution.
- [x] Add typed optional diagnostic fields and finite exception classification at model loading, decode/preparation, VAD/inference, normalization and provenance boundaries. Preserve cancellation and existing failure/recovery contracts.
- [x] Verify malicious messages, paths and custom class names never appear in worker objects, application logs, or persisted provenance. Exercise the mounted parent callback and existing stale/cancelled cases.

## Integration and completion

- [x] Check required CI target membership for each new regression; append targets without reshuffling existing shards if necessary.
- [x] Run related focused web/Canvas, runtime/worker, retry/provenance and diagnostic tests using private Windows profile and TEMP/TMP roots.
- [x] Run targeted Ruff/format, diagnostic inventory checks, task/ADR/census guards and git diff --check.
- [x] Obtain fresh combined correctness/security review, fix material findings, update architecture docs and task AC/notes. Leave changes reviewable on codex/fix-served-tabs-stt-retries.

## Authorized publication follow-up

- Commit the verified fixes and regression targets, then rebase onto current dev.
- Open a PR against dev and obtain a fresh Superpowers review of the committed diff.
- Resolve review feedback and wait for required CI, following the repository merge queue policy.
- Confirm the merged commit is present on origin/dev.

## CI startup-budget follow-up

- Reproduced eager STT diagnostic loading in a fresh app-import subprocess, then deferred its three startup edges until failure handling. The same regression verifies diagnostic-free legacy envelopes stay lazy and typed failures survive pickling.
- Added the diagnostic module to the existing first-paint absence guard; no budget or pinned snapshot changed (existing ADR-097, no new ADR required).
- Focused checks: 38 passed; the Windows census measured 1032/1031 on both the patched branch and an isolated dev f9baac8706 archive, with windows_files resident and failure_diagnostics absent. Eight additional worker/parse/retry checks passed. Fresh independent review passed 35 checks with no findings. Linux CI remains the absolute-budget merge gate.

## CI served-web dependency follow-up

- PR Fast Lane run 40614007063 reported 17 served-tab setup failures: the minimal lane omitted aiohttp_jinja2, required by the real server app factory. Its other 3217 tests passed. Pinning that helper in the shared dependency contract failed three cases before the workflow correction.
- Added only aiohttp-jinja2 (already a textual-serve runtime dependency) to the seven matching lane install commands and their two exact pins. No test targets, timeouts, shard assignments, budgets or production requirements changed. This is routine CI provisioning; no new ADR required.
- Built a scratch environment from the updated minimal install list (pytest 9.2.0). All 184 focused web/STT/retry/startup and CI-contract checks passed; four existing aiohttp warnings. Scoped Ruff, changed-range formatting and whitespace pass. Both feature commits remain range-diff equivalent after rebasing onto dev 323cf5d9e7.
- Fresh independent review found no issues. Current-head Linux CI, including every UI shard under the shared dependency environment, must pass before merge.

## CI readiness-test precondition follow-up

- Linux UI shard 3 passed 175 cases and failed the stale-proof helper regression before fault injection because delayed mount work had already enabled its ready connection.
- Reproduced that exact assertion by selecting the ready connection through the real helper before the old mount-block check. The corrected case explicitly establishes readiness before injecting expiry, retaining the helper-task deferred-then-rendered sync and enabled-Send checks.
- Final targeted case passed in the minimal pytest 9 lane environment. An in-memory mutation that returns after a deferred sync failed at the intended assertion with syncs=[False]; no helper source was changed. Fresh independent review found no issues. Ruff, changed-range formatting and whitespace pass.
- The broader Windows file run was 13 passed, 4 failed, including a passing modified case. The untouched lock case assumed a mount block; three untouched send cases were refused by trace capture. Those cases passed Linux CI. The always-ready gateway makes a missing-key fixture ineffective for the lock case, so that experiment was removed and its existing coverage retained. These local limitations are reported without changing unrelated behavior.
- No production code, helper implementation, timeout, budget or shard change; no new ADR required. All required new-head Linux checks must pass before merge.
