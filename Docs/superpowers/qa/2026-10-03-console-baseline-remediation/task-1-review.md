✅ **Spec compliant.** Both required files contain exactly the requested fixture repairs: explicit token-count keywords at `Tests/Chat/test_console_agent_project_instructions.py:444` and `:507`, and genuine persistence delegation with its return value at `Tests/Chat/test_console_chat_fork.py:1829–1834`. The supplied diff changes no production code, assertions, warning filters, limits, dependencies, or controller-owned files.

⚠️ **Verification limits:** This task establishes passing recorded nodes and affected owners, with the warning qualifications below. It does not establish repository-wide resource cleanup or revalidate every unchanged allowance/receipt/provenance invariant.

### Strengths

- The complementary budget branches remain 95/10 and 10/95. Real run completion and outbound sentinel assertions retain `[1, 0, 1]` at `Tests/Chat/test_console_agent_project_instructions.py:460–463` and `[0, 1, 0]` at `:530–532`. Delivery-local omission remains checked through `global_outcomes` and `omitted_token_budget` at `:524–529`.
- The fork fixture captures the original bound method before monkeypatching, holds the existing barrier, and delegates with the actual IDs. Fork rejection during the mutation, writer completion, exception reporting, and transition cleanup remain asserted at `Tests/Chat/test_console_chat_fork.py:1864–1871`.
- Recorded evidence matches the report: `recorded-red.log:62` shows 3 failures and 18 passes; `recorded-green.log:9` shows 21 passes; `owners-green.log:50` shows 443 passes and 7 warnings.
- Read-only comparison of the command JSON files confirmed all 21 unique inventory nodes appear in both recorded runs, with only their basetemp differing. The owner command contains all seven required modules together. Every test command uses the required interpreter and `TLDW_TEST_GC_EVERY=1`.
- `ruff-worktree.log:1` and `ruff-head.log:1` contain “All checks passed!” The formatter baseline records the required BASE, both amended paths, and zero inherited debt.

### Focused context checks

- **Risk: repaired stubs might accept the wrong production contract.** Checked `_count_model_messages` at `tldw_chatbook/Agents/agent_service.py:608–615`; its explicit keyword parameters match both repairs.
- **Risk: the barrier might still manufacture persistence success or lose refusal semantics.** Checked `_persist_active_leaf` at `tldw_chatbook/Chat/console_chat_store.py:22117` and its `set_active_leaf` consumer at `:14134–14135`. The production method returns a boolean, and the consumer rejects false results; returning the genuine result preserves that behavior.
- The supplied diff cuts off all three test functions before their assertions. Read only those function continuations to verify the preserved behavioral checks cited above. No test suites or Git commands were rerun.

### Issues

#### Critical (Must Fix)

- None found.

#### Important (Should Fix)

- None found in the scoped implementation.

#### Minor (Nice to Have)

- **Existing owner-run warning qualification:** `owners-green.log:20` reports descriptor growth of 751, from 12 to 763 against a limit of 200, despite forced GC. This remains a resource concern requiring separate diagnosis; the session-level warning does not identify its originating test or establish that this patch caused it. Preserve this qualification in final QA.
- **Existing warning noise:** `owners-green.log:9–17` records six invalid-escape `SyntaxWarning` entries during the construction-site test. Correct the offending parsed source literals in separately scoped work.
- **Evidence qualification:** The ratchet, whitespace, and committed-source equality logs are empty, and their command JSON files do not record exit codes. Their reported zero exits are documented in `task-1-report.md:82`, `:92`, `:125`, `:141`, and `:149`, but cannot be independently recovered from stdout. Future command evidence should preserve return codes alongside argv and output.

### Assessment

**Task quality: Approved.**

**Reasoning:** The minimal fixture repairs restore the existing interfaces while preserving meaningful budget, omission, and publication assertions. The supplied test evidence supports the task gate; owner warnings and silent-command exit evidence remain explicitly qualified for the controller’s final review.
