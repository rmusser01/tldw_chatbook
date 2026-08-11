# TASK-15103 Diagnostic Inventory Reconciliation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to execute this plan task by task with the
> review checkpoints below.

**Goal:** Review the exact 19-owner persistent-diagnostic drift under ADR-029,
repair every unsafe diagnostic without unrelated behavior changes, and accept
only the reviewed manifest delta while preserving the six-file sink topology.

**Architecture:** Persist one canonical change-group ledger, reuse and harden
the existing alias-aware diagnostic extractor, and make both source review and
manifest normalization fail closed. Production repairs stay at their existing call sites.
Tests invoke real production functions or source scanners only; they never
construct a Textual application, test application, reduced application, or
simplified substitute.

**Tech Stack:** Python 3.11+, pytest, Loguru/stdlib logging capture, Python AST,
JSON, Git plumbing, Ruff, the existing persistent-diagnostic checker, and the
existing summarization diagnostic extractor.

**ADR required:** no

**ADR path:** `backlog/decisions/029-local-private-data-boundary.md`

**Reason:** TASK-15103 applies ADR-029's existing persistent-log privacy
boundary. It does not change storage ownership, sink admission, or the approved
metadata policy.

## Non-negotiable execution constraints

- Use `python -B` for every Python/pytest invocation and restore every temporary
  mutation with `apply_patch` before the next mutation.
- Run only tests related to files or functions touched by this task. Do not run
  the repository-wide suite.
- Do not instantiate, subclass, mount, pilot, or run an application in tests.
  For `TldwCli` behavior, invoke the exact unbound production method with a
  narrow state record and signature-checked collaborator seams.
- Do not regenerate the checked manifest until every source call and ledger row
  is reviewed and the privacy guards are green.
- Do not accept a twentieth owner, a new call in an already-authorized owner,
  unknown JSON data, a derived-summary mismatch, a classification change, or
  any sink-topology movement.
- Preserve call order, returns, raised/returned operational errors, retry and
  cancellation branches, transport payloads, persistence order, and public
  APIs. Rendering an ADR-029-prohibited value is explicitly not a behavior
  contract to preserve.
- Commit after each task and complete the independent review checkpoint before
  starting the next task.

## Verified planning baseline

- Planning base: exact `origin/dev`
  `82b595049d97836482c118cfeb4d31df537a86a1`, audited from a detached
  export without rebasing the task branch or writing the checked manifest.
- Focused architecture baseline:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q \
    Tests/Architecture/test_persistent_diagnostic_inventory.py
  ```

  Expected after rebasing the task branch: the canonical inventory comparison
  remains red and the already-implemented TASK-15103 ledger/history nodes are
  additionally red only for their stale 18-owner planning base/path/population
  evidence. Record the exact failure set; any unrelated failure stops
  implementation. Do not hard-code a predicted test count.
- Stored/generated totals at that base are respectively:
  - owner files: `485 / 488`
  - TASK-492 calls: `1,144 / 1,180`
  - TASK-494 calls: `6,962 / 6,990`
  - persistent sink files: `6 / 6`
- Detached canonical `--write` regeneration is 53 additions/32 deletions with
  Git-patch SHA-256
  `adee369a60248da32fbc77c36b703618c73c61f5d5ef63d95460ada758f15a0f`;
  persistent-sink topology is unchanged across the exact six paths
  `tldw_chatbook/Local_Ingestion/ingest_parse_worker.py`,
  `tldw_chatbook/Logging_Config.py`, `tldw_chatbook/MCP/execution_log.py`,
  `tldw_chatbook/Utils/private_paths.py`, `tldw_chatbook/app.py`, and
  `tldw_chatbook/config.py`.
- Historical comparison: TASK-3796 reduced TASK-492 by 23 calls on both the
  stored and generated sides before the earlier stop gate. The subsequently
  approved three `console_chat_controller.py` additions remain unchanged and
  keep the current generated total at 1,180.
- The prior planned ledger freezes 18 owner populations at exact
  `85863257dd7a30b16451f8f32e0c7142dd1d5273`. All remain byte-for-byte
  identical by count/digest at latest dev except `library_screen.py`, which
  moves from 84/`c14a8222d35aec3a6e34` to
  86/`ae0fac2e87bf1a6ee81c` because of two new diagnostics.
- `text_selection_crash_guard.py` is the sole new owner: no stored/prior row,
  generated one call/digest `f90a373ef5fcc81a8c1c`, owner `TASK-494`, reason
  `remaining Chatbook production diagnostic owner`.
- Actual source/AST classification adds one reviewed-safe call—the Library
  Trash restore warning with only `type(exc).__name__` and no capture—and two
  metadata repairs: the Library Trash load warning captures the exception via
  `logger.opt(exception=True)`, while the Utils warning renders unbounded
  `repr(select_widget)` and event coordinates. Task 1 must freeze these exact
  conclusions before any production edit.

### Current-base plan deviation

After the prior 18-owner ledger/schema tranche but before production repair,
latest dev advanced from `85863257dd7a30b16451f8f32e0c7142dd1d5273` to the
exact base above, added two Library diagnostics, and introduced the Utils
owner. This docs-only tranche deliberately leaves the now-stale ledger and
guard unchanged. The next Task 1 correction must amend and independently
reconstruct the 19-owner policy boundary in its own reviewed commit before any
production, manifest, or further architecture-test work.

## Task 1: Freeze the incident and add the ledger schema

**Files:**

- Modify: `Docs/security/task-15103-diagnostic-review.json` (currently frozen
  at the stale 18-owner planning boundary)
- Modify: `Tests/Architecture/test_persistent_diagnostic_inventory.py`
- Reuse unchanged: `Tests/LLM_Calls/summarization_diagnostic_guard.py`

- [ ] **Step 1: Record exact pre-edit evidence**

  ```bash
  git fetch origin dev
  git rebase origin/dev
  git status --short
  git rev-parse HEAD
  git rev-parse origin/dev
  git merge-base --is-ancestor origin/dev HEAD
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q \
    Tests/Architecture/test_persistent_diagnostic_inventory.py
  ```

  Expected: conflict-free or fully reconciled rebase; clean worktree; exact
  execution base recorded; ancestry exit `0`; the focused test fails only for
  the canonical stale manifest and the explicitly expected stale 18-owner
  TASK-15103 planning evidence. If latest `dev` changes any of the 19 owner
  populations, owner set, manifest summary, or sink topology, update the
  design/plan evidence and re-review it before continuing.

- [ ] **Step 2: Correct schema tests before amending the stale ledger**

  Add tests that require these exact top-level ledger fields:

  ```text
  schema_version
  review_status
  incident
  owners
  change_groups
  integration_checkpoint  # state-gated; absent until Task 8
  ```

  The incident object must contain exact `recorded_base` and `planning_base`
  commit IDs. In `planned` state, each owner row contains only `path` and an
  exact `starting` call-count/digest pair; `reviewed_final` and `final_base` are
  forbidden. In `reviewed` state, `final_base` and every owner's exact
  `reviewed_final` pair are required. Each change group must contain an ID, one
  of the 19 owner paths, exact commit or narrow verified range provenance, one
  disposition (`reviewed-safe`,
  `metadata-repair`, `justified-deletion`), rationale, permitted-field
  provenance, and removed/added multiset atoms. Each atom must contain method,
  full canonical digest, integer multiplicity delta, and optional qualified
  scope. Reject unknown fields at every schema level.

  Define the canonical semantic atom as compact key-sorted JSON over
  `method`, `event`, `message_shape`, `expressions`, `captures_exception`, and
  `level_expression`, encoded as UTF-8 and hashed with the full lowercase
  64-character SHA-256. Owner path lives on the group. Qualified scope, line,
  and occurrence are navigation only and must not affect equality. Compare
  equal semantic atoms as multisets with explicit multiplicity.

  `integration_checkpoint` is forbidden before Task 8. Once added to a
  reviewed ledger it has an exact schema: `pre_rebase` is required and contains
  base/HEAD, aggregate count/SHA-256, and exact per-owner counts/SHA-256 values;
  `post_rebase` is forbidden before rebase and required afterward with the same
  shape. Reject any other checkpoint field or path.

  Add arithmetic tests that independently reconcile:

  - change-group counts by disposition;
  - removed and added atom multiplicities by owner;
  - every owner starting count/digest in planned state, and both starting/final
    pairs in reviewed state; and
  - exactly the recorded 19 owner paths, no fewer and no more.

  Retain the complete-history gate already implemented: read planned sources
  from immutable Git blobs at `incident.planning_base`, not the live worktree;
  reconstruct each owner's full stored-population-to-planning-base transition
  history; form the independent introduced/removed denominator; and require
  every transition atom to be consumed exactly once by ledger groups. Do not
  hard-code the expanded denominator, group totals, or atom arithmetic.

- [ ] **Step 3: Run the schema/history tests and confirm RED for stale evidence**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q \
    Tests/Architecture/test_persistent_diagnostic_inventory.py \
    -k 'task_15103_review_ledger' -vv
  ```

  Expected: normal collection; the current 18-owner ledger and guard fail on
  the exact planning-base/path/population mismatch. No syntax, import, setup,
  live-source substitution, or unrelated failure counts as RED.

- [ ] **Step 4: Reconstruct and write the complete incident ledger**

  Use canonical current AST extraction plus Git history for all 19 owners. Do
  not infer a unique before/after pair for duplicate calls. Represent rewrites
  as groups of removed and added atoms, and record an exact introducing commit
  only when Git proves one; otherwise use the narrowest verified range.

  The ledger must cover these owner paths exactly:

  ```text
  tldw_chatbook/Agents/agent_service.py
  tldw_chatbook/Chat/console_agent_bridge.py
  tldw_chatbook/Chat/console_chat_controller.py
  tldw_chatbook/Chat/console_chat_store.py
  tldw_chatbook/Chat/console_context_compaction.py
  tldw_chatbook/Chat/console_provider_gateway.py
  tldw_chatbook/MCP/client.py
  tldw_chatbook/MCP/local_server_tools.py
  tldw_chatbook/MCP/prompts.py
  tldw_chatbook/MCP/server.py
  tldw_chatbook/RAG_Search/fusion.py
  tldw_chatbook/RAG_Search/simplified/rag_service.py
  tldw_chatbook/RAG_Search/simplified/search_service.py
  tldw_chatbook/UI/Console_Modules/session.py
  tldw_chatbook/UI/Screens/chat_screen.py
  tldw_chatbook/UI/Screens/library_screen.py
  tldw_chatbook/app.py
  tldw_chatbook/UI/Screens/settings_screen.py
  tldw_chatbook/Utils/text_selection_crash_guard.py
  ```

  Record exact proposed surviving-call semantic contracts for unsafe current
  calls so later source tests begin RED. Freeze each disposition, rationale,
  fixed event, severity, permitted expressions, provenance, and expected
  capture state before any tracked production edit. Keep `review_status` as
  `planned`; do not add `final_base`, owner `reviewed_final` pairs, `null`,
  `TODO`, placeholder digests, or guessed final arithmetic. The planned ledger
  is the immutable policy oracle; mechanically derived final owner evidence is
  appended only in Task 7 after production matches it.

  Preserve every prior planned disposition/contract as immutable unless the
  complete Git-history reconstruction proves it invalid. Append the exact new
  Library/Utils transitions in a separately reviewed ledger-contract amendment:
  Library Trash restore is preliminary `reviewed-safe`; Library Trash load and
  the Utils selection-guard warning are preliminary `metadata-repair`. The
  bridge, MCP local-server tools, MCP prompts, MCP server, simplified search
  service, and settings call may remain review-only. The ledger audit—not these
  preliminary classifications—is authoritative. Do not hard-code predicted
  post-review group counts: Task 1's reconstructed denominator, exact-once
  consumption, and schema-validated arithmetic are authoritative.

- [ ] **Step 5: Make schema/arithmetic tests GREEN without accepting source**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q \
    Tests/Architecture/test_persistent_diagnostic_inventory.py \
    -k 'task_15103_review_ledger' -vv
  ```

  Expected: every planned-ledger schema, exact 19-path set, immutable
  planning-Git-source population, complete-history denominator, exact-once
  consumption, provenance, multiplicity, and starting-arithmetic node passes.
  A separate synthetic reviewed-state fixture proves final evidence is required
  in that state. The canonical manifest test remains red.

- [ ] **Step 6: Review and commit Task 1**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check Tests/Architecture/test_persistent_diagnostic_inventory.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format --check \
    Tests/Architecture/test_persistent_diagnostic_inventory.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m json.tool \
    Docs/security/task-15103-diagnostic-review.json >/dev/null
  git diff --check
  git status --short
  git add Docs/security/task-15103-diagnostic-review.json \
    Tests/Architecture/test_persistent_diagnostic_inventory.py
  git commit -m "test(security): inventory TASK-15103 diagnostic drift"
  ```

  Review checkpoint: independently verify exact 19-owner coverage, immutable
  planned Git-source evidence, complete-history denominator/exact-once
  consumption, provenance, disposition arithmetic, duplicate multiplicity,
  and absence of guessed one-to-one pairing before Task 2.

## Task 2: Replace the shallow review check with the ledger-driven guard

**Files:**

- Modify: `Tests/Architecture/test_persistent_diagnostic_inventory.py`
- Read/reuse: `Tests/LLM_Calls/summarization_diagnostic_guard.py`
- Modify: `Docs/security/task-15103-diagnostic-review.json`

- [ ] **Step 1: Write a reusable source-contract adapter against synthetic owners**

  Load the ledger and use
  `discover_diagnostic_calls(source, module=relative_path)` from the existing
  alias-aware helper. For each surviving reviewed call require the exact fixed
  event, method/severity, message shape, permitted expressions, level expression
  where applicable, and `captures_exception=False`. Compare canonical semantic
  atoms as a multiset. Scope and occurrence may be reported for navigation but
  never determine equality. A missing, extra, or duplicate match fails.

  Exercise that adapter with temporary source plus temporary ledger contracts;
  do not add the aggregate real-19-owner node yet. Keep the existing
  pre-TASK-15103 map for its existing owners. Do not copy any TASK-15103 labels
  or expressions into a second Python constant.

- [ ] **Step 2: Add independent syntax mutation tests**

  Using temporary source and a temporary ledger entry, add one owning test for
  each of these shapes:

  1. wholly dynamic message;
  2. positional private value;
  3. f-string private value;
  4. percent-formatted private value;
  5. `.format()` private value;
  6. concatenated private value;
  7. `.bind()` private value;
  8. keyword private value;
  9. exception-message expression;
  10. `logger.exception`;
  11. constant `logger.opt(exception=True)`;
  12. dynamic `logger.opt(exception=exc)`;
  13. stdlib `exc_info`;
  14. stdlib `stack_info`.

  Each test must assert the invariant-specific failure text; a different red
  assertion does not count.

  Also add focused alias/scoping mutations for aliases introduced or mutated
  within `try`, `for`, `while`, `with`, and `match`. Require conservative
  control-flow joins, prove shadowing and reassignment clear stale logger
  identity, and cover both missed-call and false-positive directions. This is
  a proven extractor gap owned by Task 2; do not defer it to a production
  repair batch or weaken the ledger oracle around it.

- [ ] **Step 3: Confirm RED against the current shallow guard**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q \
    Tests/Architecture/test_persistent_diagnostic_inventory.py \
    -k 'task_15103_guard' -vv
  ```

  Expected: every synthetic shape the shallow checker misses fails normally.

- [ ] **Step 4: Implement the smallest ledger adapter around the existing extractor**

  Add only ledger loading, exact schema validation, call-contract projection,
  and reconciliation. Do not duplicate alias tracking, logger recognition,
  message parsing, or exception-capture analysis. Fail closed on unknown
  outcomes, unknown fields, ambiguous matches, missing calls, extra calls, or
  any unreviewed expression.

- [ ] **Step 5: Prove only the reusable synthetic guard GREEN**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q \
    Tests/Architecture/test_persistent_diagnostic_inventory.py \
    -k 'task_15103_guard_synthetic' -vv
  ```

  Expected: all 14 syntax-owning nodes and schema fail-closed nodes pass. No
  aggregate real-source node exists yet, so this commit contains no intentional
  red test.

- [ ] **Step 6: Review and commit Task 2**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check Tests/Architecture/test_persistent_diagnostic_inventory.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format --check \
    Tests/Architecture/test_persistent_diagnostic_inventory.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m py_compile \
    Tests/Architecture/test_persistent_diagnostic_inventory.py
  git diff --check
  git add Tests/Architecture/test_persistent_diagnostic_inventory.py
  git commit -m "test(security): enforce TASK-15103 diagnostic contracts"
  ```

  Review checkpoint: attack the alias/message/capture boundary with synthetic
  source and verify every mutant dies for the intended reason.

## Task 3: Repair Agents and Chat diagnostics

**Files:**

- Create: `Tests/Architecture/test_task_15103_diagnostic_privacy.py`
- Modify: `Tests/Architecture/test_persistent_diagnostic_inventory.py`
- Modify as required by the ledger:
  - `tldw_chatbook/Agents/agent_service.py`
  - `tldw_chatbook/Chat/console_chat_controller.py`
  - `tldw_chatbook/Chat/console_chat_store.py`
  - `tldw_chatbook/Chat/console_context_compaction.py`
  - `tldw_chatbook/Chat/console_provider_gateway.py`
- Review-only unless the ledger proves otherwise:
  - `tldw_chatbook/Chat/console_agent_bridge.py`
- Read/freeze: `Docs/security/task-15103-diagnostic-review.json`

- [ ] **Step 1: Add this batch's source contract and direct real-function sentinels**

  Add `test_task_15103_agents_chat_source_contracts`, invoking the reusable
  adapter with only the six Agents/Chat owner paths. It must begin RED against
  current source and enforce the already-frozen semantic ledger contracts.
  At the verified planning base the controller has 35 generated calls with
  digest `5361a9926d2d6bede509`.

  Cover the actual affected functions and feed distinctive canaries through
  real state/config/collaborator boundaries. The tests must first assert the
  legitimate branch, return/raise result, collaborator calls, cancellation or
  persistence behavior, and then assert that diagnostics omit:

  - agent handle/run/child-run IDs and raw max-live-subagents config;
  - conversation, message, session, and operation IDs;
  - model/private override values;
  - `repr(exc)`, exception messages, and tracebacks.

  Assert the intended fixed operational event and permitted bounded metadata
  remain. The bridge timeout value may be accepted only after proving it is a
  shipped numeric constant.

  The controller coverage must include one
  `console_visual_compaction_prepared` call and both
  `console_visual_compaction_fell_back_to_text` calls. Each currently binds a
  private `conversation_id` plus code-bounded requested/effective
  representation and page-count/renderer-version or fallback-reason fields.
  Drive a private conversation-ID sentinel through the exact production
  functions, prove the bounded values, and require ledger-proven metadata
  repair. Do not classify these calls as safe before that evidence exists.

- [ ] **Step 2: Run only the new Agents/Chat nodes and capture genuine RED**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q \
    Tests/Architecture/test_task_15103_diagnostic_privacy.py \
    Tests/Architecture/test_persistent_diagnostic_inventory.py \
    -k 'agent_service or console_chat or console_compaction or console_gateway or console_bridge or task_15103_agents_chat_source_contracts' \
    -vv
  ```

  Expected: every unsafe canary/capture node fails at its privacy assertion
  after behavior assertions pass; review-only bounded-metadata nodes may pass.

- [ ] **Step 3: Apply minimal call-site repairs**

  Prefer fixed events. Retain only proven counts, lengths, durations, status,
  retry/cancellation state, closed provider/operation values, and
  `type(exc).__name__`. Replace `logger.exception` and `.opt(exception=...)`
  with the same-severity non-capturing call. Remove private `.bind()` fields.
  Do not add helpers unless two or more sites need the same nontrivial
  transformation.

- [ ] **Step 4: Prove the batch matches its frozen ledger contracts**

  Do not edit dispositions, permitted fields, rationale, provenance, or target
  contracts alongside production. If the implementation cannot match a frozen
  contract, stop, revert the production attempt, and commit a separately
  reviewed ledger-contract amendment before resuming. Final owner digests are
  still deferred to Task 7.

- [ ] **Step 5: Run focused GREEN and affected behavior tests**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q \
    Tests/Architecture/test_task_15103_diagnostic_privacy.py \
    Tests/Agents/test_agent_service.py \
    Tests/Chat/test_console_chat_controller.py \
    Tests/Chat/test_console_chat_store.py \
    Tests/Chat/test_console_context_compaction.py \
    Tests/Chat/test_console_provider_gateway.py \
    -k 'diagnostic or fleet or approval or context_policy or compaction or capability' \
    -vv
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q \
    Tests/Architecture/test_persistent_diagnostic_inventory.py \
    -k 'task_15103_agents_chat_source_contracts or task_15103_review_ledger' -vv
  ```

  Expected: all selected nodes pass. No aggregate source node is selected or
  committed yet; later batches receive their own owning source-contract nodes.

- [ ] **Step 6: Review and commit Task 3**

  Run static checks over exactly the unstaged Python files in this batch:

  ```bash
  git diff --name-only --diff-filter=ACM -z -- '*.py' | \
    xargs -0 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check
  git diff --name-only --diff-filter=ACM -z -- '*.py' | \
    xargs -0 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format --check
  git diff --name-only --diff-filter=ACM -z -- '*.py' | \
    xargs -0 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m py_compile
  git diff --check
  ```

  Then commit:

  ```bash
  git add tldw_chatbook/Agents/agent_service.py \
    tldw_chatbook/Chat/console_agent_bridge.py \
    tldw_chatbook/Chat/console_chat_controller.py \
    tldw_chatbook/Chat/console_chat_store.py \
    tldw_chatbook/Chat/console_context_compaction.py \
    tldw_chatbook/Chat/console_provider_gateway.py \
    Tests/Architecture/test_task_15103_diagnostic_privacy.py \
    Tests/Architecture/test_persistent_diagnostic_inventory.py
  git commit -m "fix(security): redact agent and console diagnostics"
  ```

  Before staging, omit any review-only production path with no actual diff.
  Review checkpoint: verify exact production scope and behavior preservation.

## Task 4: Reconcile MCP diagnostics

**Files:**

- Modify as required: `tldw_chatbook/MCP/client.py`
- Review-only unless the ledger proves otherwise:
  - `tldw_chatbook/MCP/local_server_tools.py`
  - `tldw_chatbook/MCP/prompts.py`
  - `tldw_chatbook/MCP/server.py`
- Modify: `Tests/Architecture/test_task_15103_diagnostic_privacy.py`
- Modify: `Tests/Architecture/test_persistent_diagnostic_inventory.py`
- Read/freeze: `Docs/security/task-15103-diagnostic-review.json`
- Focused existing tests: `Tests/MCP/test_client_catalog_pagination.py`,
  `Tests/MCP/test_mcp_unified_stdio.py`

- [ ] **Step 1: Add the MCP source contract, direct sentinels, and RED**

  Add `test_task_15103_mcp_source_contracts`, invoking the reusable adapter
  with only the four MCP owner paths. It must begin RED on any unsafe current
  call and enforce the already-frozen MCP semantic contracts.

  Prove raw JSON-RPC payloads, decoded lines, request IDs, server IDs, exception
  messages, and subprocess tracebacks cannot persist. Pin connection ownership,
  capability counts, cleanup ordering, cancellation, return/raise behavior, and
  subprocess method calls before checking captured diagnostics.

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q \
    Tests/Architecture/test_task_15103_diagnostic_privacy.py \
    Tests/Architecture/test_persistent_diagnostic_inventory.py \
    -k 'mcp_client or mcp_transport or mcp_prompt or mcp_server or task_15103_mcp_source_contracts' \
    -vv
  ```

  Expected: any surviving server-ID/private/capture exposure fails its owning
  privacy assertion; already-fixed constant events pass their contract nodes.

- [ ] **Step 2: Repair only ledger-proven unsafe MCP sites**

  Keep fixed cleanup/transport lifecycle events. Capability diagnostics may
  retain counts but not the server identifier. Do not change JSON-RPC payloads,
  connection state, subprocess teardown order, retries, timeouts, or exceptions.

- [ ] **Step 3: Prove MCP source matches the frozen ledger and run GREEN**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q \
    Tests/Architecture/test_task_15103_diagnostic_privacy.py \
    Tests/MCP/test_client_catalog_pagination.py \
    Tests/MCP/test_mcp_unified_stdio.py \
    -k 'mcp_client or mcp_transport or capability or teardown or cancellation' \
    -vv
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q \
    Tests/Architecture/test_persistent_diagnostic_inventory.py \
    -k 'task_15103_mcp_source_contracts or task_15103_review_ledger' -vv
  ```

  Expected: all selected MCP/privacy/ledger nodes pass.

- [ ] **Step 4: Static review and commit Task 4**

  ```bash
  git diff --name-only --diff-filter=ACM -z -- '*.py' | \
    xargs -0 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check
  git diff --name-only --diff-filter=ACM -z -- '*.py' | \
    xargs -0 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format --check
  git diff --name-only --diff-filter=ACM -z -- '*.py' | \
    xargs -0 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m py_compile
  git diff --check
  ```

  Then commit:

  ```bash
  git add tldw_chatbook/MCP/client.py \
    Tests/Architecture/test_task_15103_diagnostic_privacy.py \
    Tests/Architecture/test_persistent_diagnostic_inventory.py
  git commit -m "fix(security): reconcile MCP diagnostic privacy"
  ```

  Stage any other MCP production path only if the ledger proves a required
  repair and its focused test is present. Review checkpoint: verify current
  fixed-only cleanup events were not needlessly rewritten.

## Task 5: Reconcile RAG diagnostics

**Files:**

- Modify as required:
  - `tldw_chatbook/RAG_Search/fusion.py`
  - `tldw_chatbook/RAG_Search/simplified/rag_service.py`
- Review-only unless the ledger proves otherwise:
  - `tldw_chatbook/RAG_Search/simplified/search_service.py`
- Modify: `Tests/Architecture/test_task_15103_diagnostic_privacy.py`
- Modify: `Tests/Architecture/test_persistent_diagnostic_inventory.py`
- Read/freeze: `Docs/security/task-15103-diagnostic-review.json`
- Focused existing tests:
  - `Tests/RAG/test_fusion.py`
  - `Tests/RAG_Search/test_fusion_config_knobs.py`
  - `Tests/RAG/simplified/test_rag_service_basic.py`
  - `Tests/RAG/simplified/test_search_service.py`

- [ ] **Step 1: Add the RAG source contract, direct sentinels, and capture RED**

  Add `test_task_15103_rag_source_contracts`, invoking the reusable adapter
  with only the three RAG owner paths. It must begin RED on any unsafe current
  call and enforce the already-frozen RAG semantic contracts.

  Feed distinctive invalid configuration, source-type, and exception canaries
  through the exact resolver/search functions. Prove shipped numeric bounds,
  result behavior, fallback selection, database-call avoidance, and returned
  results before privacy assertions. Closed source-type enums may be logged only
  when code-bounded; arbitrary config text and exception messages may not.

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q \
    Tests/Architecture/test_task_15103_diagnostic_privacy.py \
    Tests/Architecture/test_persistent_diagnostic_inventory.py \
    -k 'fusion or rag_service or search_service or task_15103_rag_source_contracts' \
    -vv
  ```

  Expected: raw invalid config/exception values fail their privacy assertions;
  fixed-only search failure and proven bounded metadata cases pass.

- [ ] **Step 2: Repair unsafe RAG calls without changing fallback behavior**

  Replace raw `value`, raw multiplier/input, unknown unbounded values, and
  exception messages with fixed events, safe counts, shipped bounds, or
  exception class names. Preserve exact fallback values and database behavior.

- [ ] **Step 3: Prove RAG source matches the frozen ledger and run focused GREEN**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q \
    Tests/Architecture/test_task_15103_diagnostic_privacy.py \
    Tests/RAG/test_fusion.py \
    Tests/RAG_Search/test_fusion_config_knobs.py \
    Tests/RAG/simplified/test_rag_service_basic.py \
    Tests/RAG/simplified/test_search_service.py \
    -k 'fusion or config or keyword or source_type or fallback or diagnostic' \
    -vv
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q \
    Tests/Architecture/test_persistent_diagnostic_inventory.py \
    -k 'task_15103_rag_source_contracts or task_15103_review_ledger' -vv
  ```

  Expected: all selected nodes pass.

- [ ] **Step 4: Static review and commit Task 5**

  ```bash
  git diff --name-only --diff-filter=ACM -z -- '*.py' | \
    xargs -0 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check
  git diff --name-only --diff-filter=ACM -z -- '*.py' | \
    xargs -0 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format --check
  git diff --name-only --diff-filter=ACM -z -- '*.py' | \
    xargs -0 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m py_compile
  git diff --check
  ```

  Then commit:

  ```bash
  git add tldw_chatbook/RAG_Search/fusion.py \
    tldw_chatbook/RAG_Search/simplified/rag_service.py \
    tldw_chatbook/RAG_Search/simplified/search_service.py \
    Tests/Architecture/test_task_15103_diagnostic_privacy.py \
    Tests/Architecture/test_persistent_diagnostic_inventory.py
  git commit -m "fix(security): redact RAG configuration diagnostics"
  ```

  Omit `search_service.py` before staging when its ledger review requires no
  production repair. Review checkpoint: verify no fallback, query, or database
  behavior changed.

## Task 6: Reconcile UI and application lifecycle diagnostics

**Files:**

- Modify as required:
  - `tldw_chatbook/UI/Console_Modules/session.py`
  - `tldw_chatbook/UI/Screens/chat_screen.py`
  - `tldw_chatbook/UI/Screens/library_screen.py`
  - `tldw_chatbook/app.py`
- Review and modify only if the ledger proves a repair is required:
  - `tldw_chatbook/UI/Screens/settings_screen.py`
  - `tldw_chatbook/Utils/text_selection_crash_guard.py`
- Modify: `Tests/Architecture/test_task_15103_diagnostic_privacy.py`
- Modify: `Tests/Architecture/test_persistent_diagnostic_inventory.py`
- Read/freeze: `Docs/security/task-15103-diagnostic-review.json`
- Read for exact function contracts only:
  - `Tests/UI/test_app_quit_guard.py`
  - `Tests/UI/test_console_session_controller.py`
  - `Tests/UI/test_library_screen.py`
  - `Tests/App/test_text_selection_crash_guard.py` from the rebased source;
    never execute its Textual app harness

- [ ] **Step 1: Add the UI/app source contract, function-only sentinels, and RED**

  Add `test_task_15103_ui_app_source_contracts`, invoking the reusable adapter
  with only the six UI/app owner paths. It must begin RED on current unsafe
  calls and enforce the already-frozen UI/app semantic contracts.

  Invoke exact production methods directly. For `TldwCli`, call the unbound
  method with a narrow state record; do not instantiate/subclass/mount/run an
  app. Use signature-checked collaborators for workers, timers, persistence,
  and audio cleanup. Pin branch selection, state mutation, ordering, timeouts,
  return/raise behavior, and collaborator calls before privacy assertions.

  Cover session IDs and all exception/traceback capture in Console session,
  realtime audio, Library DB-size refresh, settings appearance refresh, and
  quit lifecycle paths. For `settings_screen.py`, directly exercise the
  production boundary for `Console appearance refresh failed after settings
  save (screen_type=%s, generation=%s, error_type=%s).` and prove the exact
  expressions `type(screen).__name__`, `generation`, and
  `type(exc).__name__` do not expose a private sentinel. Its apparent
  metadata-only shape is not acceptance evidence.

  Before any repair, add direct function-only RED coverage for both Library
  Trash diagnostics. `_load_library_media_trash` warning event `Failed to load
  the Library media trash page.` currently has extracted expressions
  `module='LibraryScreen'` and `exception=True` and captures the exception;
  preserve its error state, refresh/focus behavior, and collaborator ordering
  while omitting exception messages and capture.
  `_restore_library_media_from_trash` warning event `Failed to restore a
  Library media item from the Trash view (error_type={}).` has expression
  `type(exc).__name__` and no capture; preserve its result/state/count behavior
  and prove that reviewed-safe contract. Feed private media IDs and exception
  messages through the exact methods.

  Also add a conditional Utils source-contract and direct sentinel owned by
  this batch. Exercise the exact crash matcher/dispatch functions without
  constructing, subclassing, mounting, piloting, or running a Textual app.
  Supply a widget whose `repr` is a distinctive private canary and distinctive
  mouse coordinates; pin the narrow crash-match, click-drop, select-state, and
  re-raise behavior before asserting neither value persists. Do not run the
  existing `Tests/App/test_text_selection_crash_guard.py` app harness.
  The current warning method is `warning`; its fixed event projection is
  `Dropped a MouseDown that hit Textual's text-selection begin path while its
  target widget was mid-recompose (detached parent): target=,
  screen_offset=(,). Upstream Textual race (screen.py _forward_event,
  container=None) -- the click was not delivered; the app stays alive
  (task-14903).`; its expressions are `target`,
  `getattr(event, 'screen_x', '?')`, and `getattr(event, 'screen_y', '?')`;
  it does not capture the exception.

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q \
    Tests/Architecture/test_task_15103_diagnostic_privacy.py \
    Tests/Architecture/test_persistent_diagnostic_inventory.py \
    -k 'console_session or chat_screen or library_screen or settings_screen or text_selection or app_quit or task_15103_ui_app_source_contracts' \
    -vv
  ```

  Expected: current session-ID/capture, Library Trash capture, and Utils
  private-repr/coordinate diagnostics fail their owning privacy assertions
  after behavior assertions pass. Do not edit production before this batch RED
  and the frozen source contracts are recorded.

- [ ] **Step 2: Apply fixed same-severity non-capturing replacements**

  Preserve timeout and lifecycle state wording, but remove IDs and traceback
  capture. Ensure fixed wording is truthful on start failure, confirmation
  failure, guard failure, audio failure, persistence failure, and cancellation.
  Modify `settings_screen.py` only if the frozen ledger and sentinel prove a
  repair is required; otherwise preserve it as an explicitly reviewed call.
  For Library Trash load, remove exception capture at the same severity and
  retain only ledger-approved metadata. For the Utils call, if the frozen
  ledger confirms the preliminary metadata-repair disposition, remove the
  unbounded widget `repr` and event coordinates and ensure prohibited rendering
  is not retained solely for logging; preserve the exact narrow crash predicate
  and click-drop behavior.

- [ ] **Step 3: Prove UI/app source matches the frozen ledger and run GREEN**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q \
    Tests/Architecture/test_task_15103_diagnostic_privacy.py \
    -k 'console_session or chat_screen or library_screen or settings_screen or text_selection or app_quit' -vv
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q \
    Tests/Architecture/test_persistent_diagnostic_inventory.py \
    -k 'task_15103_ui_app_source_contracts or task_15103_review_ledger' -vv
  ```

  Expected: all selected nodes pass and no app object was constructed.

- [ ] **Step 4: Add and run the aggregate 19-owner source contract**

  Add `test_task_15103_all_reviewed_owner_source_contracts`, which invokes the
  same adapter over the ledger's exact 19-owner path set. This node is added
  only after all four batch nodes are green.

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q \
    Tests/Architecture/test_persistent_diagnostic_inventory.py \
    -k 'task_15103_all_reviewed_owner_source_contracts' -vv
  ```

  Expected: the aggregate node passes with no missing, extra, duplicated, or
  exception-capturing reviewed call.

- [ ] **Step 5: Static review and commit Task 6**

  ```bash
  git diff --name-only --diff-filter=ACM -z -- '*.py' | \
    xargs -0 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check
  git diff --name-only --diff-filter=ACM -z -- '*.py' | \
    xargs -0 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format --check
  git diff --name-only --diff-filter=ACM -z -- '*.py' | \
    xargs -0 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m py_compile
  git diff --check
  ```

  Then commit:

  ```bash
  git add tldw_chatbook/UI/Console_Modules/session.py \
    tldw_chatbook/UI/Screens/chat_screen.py \
    tldw_chatbook/UI/Screens/library_screen.py \
    tldw_chatbook/app.py \
    Tests/Architecture/test_task_15103_diagnostic_privacy.py \
    Tests/Architecture/test_persistent_diagnostic_inventory.py
  # Run this only when ledger/sentinel evidence proves a settings repair.
  git add tldw_chatbook/UI/Screens/settings_screen.py
  # Run this only when ledger/sentinel evidence proves the Utils repair.
  git add tldw_chatbook/Utils/text_selection_crash_guard.py
  git commit -m "fix(security): remove UI lifecycle traceback diagnostics"
  ```

  Stage `tldw_chatbook/UI/Screens/settings_screen.py` only when the ledger and
  focused sentinel prove an actual production repair is required. Stage
  `tldw_chatbook/Utils/text_selection_crash_guard.py` and its focused direct
  test only when its frozen ledger contract proves the preliminary repair.
  Review checkpoint: independently inspect that test code never constructs an
  application and that production diffs contain logging-only behavior changes.

## Task 7: Regenerate and fail-close the production inventory

**Files:**

- Modify: `Docs/security/production-diagnostic-inventory.json`
- Modify: `Tests/Architecture/test_persistent_diagnostic_inventory.py`
- Read/load: `Docs/security/task-15103-diagnostic-review.json`
- Read/execute: `scripts/check_persistent_diagnostic_inventory.py`

- [ ] **Step 1: Make the complete source/ledger gate GREEN before regeneration**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q \
    Tests/Architecture/test_persistent_diagnostic_inventory.py \
    -k 'task_15103_guard or task_15103_review_ledger or task_15103_all_reviewed_owner_source_contracts' \
    -vv
  ```

  Expected: exact 19-owner source contracts, all schema/arithmetic checks, and
  all syntax mutation nodes pass. Stop if any source call is pending,
  ambiguous, missing, extra, or captures exceptions.

  After that gate, mechanically derive every owner's exact final call count and
  manifest digest from tracked source, append those pairs plus the current
  reviewed source base, and change `review_status` from `planned` to
  `reviewed`. Do not change any disposition, rationale, permitted expression,
  provenance, or proposed semantic contract. Rerun the reviewed-state ledger
  validator and require it to pass.

- [ ] **Step 2: Write deep manifest-boundary tests and capture missing-boundary RED**

  Compare deep copies of the complete checked and generated documents. Remove
  only the exact 19 owner rows from the general owner equality comparison, then
  validate those rows separately for exact path, owner, reason, reviewed count,
  and reviewed digest. Independently recompute owner-file, TASK-492, TASK-494,
  and sink-file totals for each document before normalization.

  Require the known owner-file transition `485 -> 488`, exact reviewed owner
  rows, and identical six-file sink topology. Every unknown field, section,
  list order, exclusion, classification rule, unreviewed owner row, and sink
  row must remain deeply equal.

  Add independent tests for:

  1. unknown top-level data;
  2. forged derived summary;
  3. a twentieth owner;
  4. an owner/reason classification change on one reviewed path;
  5. a persistent-sink change.

  Except for the deliberately forged summary, recompute all derived totals in
  the mutant before invoking the boundary. Assert exact invariant-specific
  failure text so a stale-summary failure cannot masquerade as another proof.

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q \
    Tests/Architecture/test_persistent_diagnostic_inventory.py \
    -k 'task_15103_manifest_boundary_' -vv
  ```

  Expected: normal collection; the new positive and five negative tests fail
  because the boundary adapter is not implemented. No stale checked-manifest
  assertion is selected in this RED.

- [ ] **Step 3: Implement the boundary and make synthetic tests GREEN**

  Implement only the deep-copy normalization, independent summary
  recomputation, exact 19-row validation, unknown-field rejection, and sink
  topology comparison described above.

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q \
    Tests/Architecture/test_persistent_diagnostic_inventory.py \
    -k 'task_15103_manifest_boundary_' -vv
  ```

  Expected: the synthetic positive case and all five invariant-specific
  negative cases pass.

- [ ] **Step 4: Confirm RED because the checked manifest is still stale**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q \
    Tests/Architecture/test_persistent_diagnostic_inventory.py \
    -k 'task_15103_manifest or production_diagnostic_inventory' -vv
  ```

  Expected: every synthetic boundary node and
  `test_task_15103_manifest_delta_is_exact_and_fail_closed` pass because the
  candidate-aware boundary accepts exactly the authorized reviewed 19-owner
  stale-to-generated delta. The sole failure is the canonical
  `test_production_diagnostic_inventory_and_sink_topology_are_unchanged`, which
  requires byte equality and remains red until regeneration. Record the exact
  one-node failure set; any other failure stops the task.

- [ ] **Step 5: Regenerate once and inspect the exact diff**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B scripts/check_persistent_diagnostic_inventory.py --write
  git diff -- Docs/security/production-diagnostic-inventory.json
  ```

  Expected: only the 19 reviewed owner rows and independently derived summary
  totals change; all six sink rows/topology and every unreviewed owner remain
  byte-equivalent after normalization. Stop rather than edit around any extra
  delta.

- [ ] **Step 6: Run GREEN, mutations, and restoration proof**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q \
    Tests/Architecture/test_persistent_diagnostic_inventory.py -vv
  ```

  Expected: all architecture inventory/ledger/source/manifest tests pass.

  Record baseline SHA-256 hashes of the real checked manifest and ledger. Then
  apply each of the five mutations one at a time to the actual checked manifest
  with `apply_patch`—not merely to a temporary in-test copy. For each case:

  1. recompute the mutated artifact SHA-256 and require it differs from the
     baseline;
  2. run the positive node exactly:

     ```bash
     /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q \
       Tests/Architecture/test_persistent_diagnostic_inventory.py::test_task_15103_manifest_delta_is_exact_and_fail_closed \
       -vv
     ```

  3. require that node to fail with the mutation's exact unknown-field,
     forged-summary, twentieth-owner, classification, or sink-topology text;
  4. inverse-restore with `apply_patch`;
  5. require restored SHA-256 equals the baseline; and
  6. rerun the exact positive node GREEN.

  End with byte-identical manifest/ledger hashes and require each path's
  post-restoration `git diff` to equal its recorded pre-mutation legitimate
  Task 7 diff exactly. Require no additional mutation residue; a literally
  clean diff is not expected until the legitimate Task 7 changes are committed.
  Equal before/after hashes without an observed non-equal mutant hash do not
  count as mutation evidence.

- [ ] **Step 7: Static review and commit Task 7**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check Tests/Architecture/test_persistent_diagnostic_inventory.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format --check \
    Tests/Architecture/test_persistent_diagnostic_inventory.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m py_compile \
    Tests/Architecture/test_persistent_diagnostic_inventory.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m json.tool \
    Docs/security/production-diagnostic-inventory.json >/dev/null
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m json.tool \
    Docs/security/task-15103-diagnostic-review.json >/dev/null
  git diff --check
  git add Docs/security/production-diagnostic-inventory.json \
    Docs/security/task-15103-diagnostic-review.json \
    Tests/Architecture/test_persistent_diagnostic_inventory.py
  git commit -m "test(security): reconcile persistent diagnostic inventory"
  ```

  Review checkpoint: independently regenerate from source, verify exact owner
  and sink topology, and rerun every invariant mutant.

## Task 8: Final rebase, complete-population reconciliation, and closeout

**Files:**

- Modify: `Docs/security/task-15103-diagnostic-review.json`
- Modify if final rebase changes reviewed source:
  `Docs/security/production-diagnostic-inventory.json`
- Modify: `backlog/tasks/task-15103 - Reconcile-19-owner-latest-dev-diagnostic-inventory-drift.md`
- Modify: this plan's step markers and any exact-base evidence
- Modify: the design status line
- Modify lessons only if this task produced genuinely new incident-based,
  nonduplicate evidence

- [ ] **Step 1: Record pre-rebase complete owner populations**

  Serialize every diagnostic call from all 19 owners using the canonical
  semantic atom (`method`, `event`, `message_shape`, `expressions`, capture
  state, and `level_expression`) plus multiplicity. Exclude scope, line, and
  occurrence from equality; retain them only in an external navigation report.

  Append an `integration_checkpoint.pre_rebase` object to the canonical ledger
  containing exact base/HEAD commits, the complete aggregate semantic-multiset
  SHA-256, and every owner's count/SHA-256. Validate it, then create a clean
  checkpoint commit before rebasing:

  ```bash
  git add Docs/security/task-15103-diagnostic-review.json
  git commit -m "test(security): checkpoint TASK-15103 owner population"
  git status --short
  ```

  Expected: the durable checkpoint is committed and the worktree is clean.

- [ ] **Step 2: Fetch and rebase onto exact latest `origin/dev`**

  ```bash
  git fetch origin dev
  git rebase origin/dev
  git merge-base --is-ancestor origin/dev HEAD
  git rev-list --left-right --count origin/dev...HEAD
  ```

  Expected: conflict-free or fully reconciled rebase; ancestry exit `0`; `0`
  behind. If upstream changes any of the 19 owners or complete populations,
  reopen its per-call audit, provenance, ledger, sentinel, and manifest row.

- [ ] **Step 3: Compare the complete post-rebase populations**

  Rerun the exact semantic-multiset serialization from Step 1 and compare every
  owner, not only the aggregate. Pure relocation remains equal. Any added,
  removed, rewritten, re-aliased, multiplicity-changing, or capture-changing
  semantic atom requires its owner audit, provenance, sentinel, contract, and
  manifest row to reopen.

  After reconciliation, append `integration_checkpoint.post_rebase`, update the
  ledger's final base, and regenerate the checked manifest if reviewed source
  changed. Before any manifest write, run the positive deep boundary against
  the post-rebase candidate generated from source. It must reject unrelated
  owner, unknown-field, classification, summary, or sink drift while the old
  checked manifest is still available as evidence. Only after that passes may
  the checked manifest be written. Validate and commit this evidence before
  closeout:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q \
    Tests/Architecture/test_persistent_diagnostic_inventory.py::test_task_15103_manifest_delta_is_exact_and_fail_closed \
    -vv
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B scripts/check_persistent_diagnostic_inventory.py --write
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B scripts/check_persistent_diagnostic_inventory.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q \
    Tests/Architecture/test_persistent_diagnostic_inventory.py -vv
  git add Docs/security/task-15103-diagnostic-review.json \
    Docs/security/production-diagnostic-inventory.json
  git commit -m "test(security): reconcile TASK-15103 after rebase"
  git status --short
  ```

  Expected: exact per-owner semantic populations are reconciled, the manifest
  checker passes, and the worktree is clean. If the manifest did not change,
  `git add` simply stages no manifest delta.

- [ ] **Step 4: Run the final touched-function gate only**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest -q \
    Tests/Architecture/test_persistent_diagnostic_inventory.py \
    Tests/Architecture/test_task_15103_diagnostic_privacy.py
  ```

  Add only the exact existing subsystem test nodes used in Tasks 3–6; do not
  broaden to their entire directories or the repository-wide suite.

  Expected: every selected test passes with no app construction.

- [ ] **Step 5: Run final static, manifest, and hygiene gates**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B scripts/check_persistent_diagnostic_inventory.py
  git diff --name-only --diff-filter=ACM -z origin/dev...HEAD -- '*.py' | \
    xargs -0 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check
  git diff --name-only --diff-filter=ACM -z origin/dev...HEAD -- '*.py' | \
    xargs -0 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format --check
  git diff --name-only --diff-filter=ACM -z origin/dev...HEAD -- '*.py' | \
    xargs -0 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m py_compile
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m json.tool \
    Docs/security/production-diagnostic-inventory.json >/dev/null
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m json.tool \
    Docs/security/task-15103-diagnostic-review.json >/dev/null
  git diff --check origin/dev...HEAD
  git status --short
  ```

  Expected: every gate passes; worktree clean after closeout commit.

- [ ] **Step 6: Perform independent whole-branch review**

  Review `origin/dev...HEAD` for:

  - unreviewed private fields or traceback capture;
  - false fixed-event wording;
  - new eager evaluation or behavior changes;
  - ledger/source/manifest mismatch;
  - a twentieth owner or sink movement;
  - duplicate policy data outside the ledger;
  - app/test-app/reduced-app construction; and
  - stale base/count/hash evidence.

  Fix every Critical, Important, and justified Minor finding test-first, then
  rerun the affected focused gates.

- [ ] **Step 7: Close TASK-15103 only after all evidence is fresh**

  Update the task through Backlog CLI:

  - check all four acceptance criteria;
  - add concise Implementation Notes with exact owner/group/atom arithmetic,
    disposition totals, ledger SHA-256, final base/HEAD, manifest totals, sink
    count, mutation outcomes/restoration hashes, tests, static gates, ADR-029,
    no-new-ADR rationale, no-app proof, and plan deviations;
  - set status to Done only after every gate and review passes.

  Set the design status to `implemented and verified`, mark all actual plan
  step markers complete, and add a lesson only if a new reusable incident was
  discovered.

- [ ] **Step 8: Commit closeout documentation**

  ```bash
  git add backlog/tasks/task-15103\ -\ Reconcile-19-owner-latest-dev-diagnostic-inventory-drift.md \
    Docs/superpowers/specs/2026-08-11-task-15103-diagnostic-inventory-reconciliation-design.md \
    Docs/superpowers/plans/2026-08-11-task-15103-diagnostic-inventory-reconciliation.md
  git commit -m "docs(security): close TASK-15103 inventory reconciliation"
  git status --short
  ```

  Include a lesson file in the exact staged list only when Step 7 proves one is
  warranted. Do not push, open a PR, or merge until the user requests it.

## Final handoff evidence

The handoff must report:

- exact final base and HEAD;
- clean ancestry and ahead/behind relation;
- exact 19-owner path set and full-population comparison;
- ledger hash plus group/atom/disposition arithmetic;
- stored/generated manifest totals and six-sink topology;
- every focused test command and result;
- every mutation's intended RED, restoration GREEN, and restored hashes;
- Ruff/format/compile/diff results;
- independent review findings and fixes;
- TASK-15103 Done with 4/4 AC and Implementation Notes;
- ADR-029/no-new-ADR decision; and
- explicit confirmation that no application, test application, reduced
  application, simplified substitute, or repository-wide test suite was used.
