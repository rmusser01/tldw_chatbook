# Personal Context baseline execution review

Date: 2026-09-25
Scope: TASK-25907.1, native execution of the approved offline baseline plan.
Application behavior, permissions, schema and real profiles are unchanged.

- [Plan](../plans/2026-09-25-personal-context-memory-baseline.md)
- [Design review](2026-09-25-personal-context-memory-preimplementation-review.md)
- [Evaluation and reproduction](../../../backlog/docs/personal-context-memory-evaluation.md)
- [Tracker](../../../backlog/docs/personal-context-memory-roadmap.md)

## Corrections made during implementation

1. **Denied tool calls have no JSON body.** The actual provider reports denial
   in `ToolResult.error`. A regression reproduced the harness incorrectly
   rejecting that valid response. The harness now accepts only the exact
   expected denial with `ok=False` and an empty body. A successful response
   carrying a nonempty error is also rejected; that regression failed before
   the correction.
2. **Fixture identity must describe what was measured.** Hashing the file
   after the run can describe different bytes if it changes during execution.
   A mutation regression reproduced this. The loader now hashes the bounded
   byte buffer it actually parsed.
3. **Control responses are also disclosure surfaces.** A regression inserted a
   private fixture canary into a successful control response while leaving the
   main query clean. It reproduced a missed leak. Controls now undergo the
   same raw-output scans as the measured query. The revocation case's control
   is evaluated under its earlier, valid authority.
4. **Evidence output must not replace prior evidence.** Exclusive creation,
   serialization before file creation, and removal of a partial new file on
   write failure are covered by focused output tests. Existing files and
   directories are preserved.

The original technical review's product findings remain tracked: provenance
cannot reconstruct absent edit history, selection diagnostics need request/time
freshness, and device-only disclosure and quarantine metadata require explicit
policy work. The baseline reports observed gaps; this task does not fix them.

## Verification record

The initial production-owner run passed 31 tests. Fixture/scoring checks passed
25 tests. The real runner plus its production owners passed 83 targeted tests.
The report writer/hash regressions passed seven focused checks; the additional
control and success/error regressions then passed with those seven (nine total).
One report run was interrupted after 30 passing tests, before evidence output,
to address the self-review findings. It is not counted as a completed run.

The first real report measured 24 cases with no harness, authority or context
errors. Both policy gaps remain failed; recall is 0.8 on each split. The
evaluation guide records per-case misses and metric denominators. Two independently seeded fresh-root reports matched exactly, including a
comparison with the committed artifact. The final targeted run passed all
96 tests in 681.43 seconds. Ruff check/format and diff whitespace checks passed.
Ten owned task identities/paths, 59 child criteria and all owned local links
were validated. TASK-25907.1 was completed through the Backlog CLI; the other
eight children remain To Do.

The report was produced at HEAD `18e56c1e28` with the Task 3 harness/test edits
uncommitted. Those exact files and the artifact were then committed in
`a47439ee52`. SHA-256 of the measured runner:
`97c2d36f4d987dec348bf1836a77acd3abe17f9bd913d82c6845772376d53f49`;
test file: `c492c9ebf5e291fd3a0a5914f7464de14ab7061c7ecc3b6ad799a44ed6e3b4d7`.
Production application/shared-core files are unchanged from the runtime base.

## Independent final review

A fresh read-only reviewer inspected `1f0cc0e6e4..a47439ee52` against the plan,
specification and execution decisions. It found no Critical or Important
issues. It independently checked the fixture hash, measured artifact and
whitespace. It did not duplicate the expensive test run. Its acceptance was
conditional on targeted verification; that condition is now satisfied.

One **Minor, deferred** improvement: add a focused presence/absence regression
for the quarantine-flag policy classifier (`test_memory_baseline.py:243` and
`memory_baseline.py:599`). The current artifact correctly records this gap,
but removing the classifier could leave existing assertions passing because
the device-only failure still keeps the aggregate disclosure gate false.
This does not change current measurements or require preserving the product
defect in a future test.

The reviewer set aside model answer quality/source entailment/injection
resistance, remediation of existing policy defects, alternate tokenizers and
network/server enforcement, and pending executor-owned closeout fields. Those
boundaries and their costs are explicitly retained in the decision register.

## Recorded final command

```bash
TLDW_MEMORY_BASELINE_REPORT=Docs/superpowers/reviews/evidence/personal-context-memory/baseline-v1.json .venv/bin/python -m pytest Tests/Personal_Context/test_memory_baseline.py Tests/Agents/test_profile_tool_provider.py Tests/Personal_Context/test_context_service.py -q --basetemp=.superpowers/sdd/2026-09-25-personal-context-memory-baseline/pytest-final
```

Result: `96 passed in 681.43s (0:11:21)`. The second report was generated inside
`test_reports_are_reproducible_across_temporary_roots`, written to a fresh file,
and compared with the first measured report and then with the committed artifact.
Use the evaluation guide's fresh-output command to reproduce after cleanup.
No separate process/tokenizer/server generalization is claimed.

## Execution decisions

Every native-execution ruling is retained below with its rationale and cost.
The user's targeted-test requirement overrides generic skill full-suite steps.

1. Keep the committed runtime at the isolated HEAD, not unrelated working-tree token-window changes — the estimator under test is unchanged by that unrelated diff — cost if wrong: baseline must be remeasured after integration.
2. Use targeted owner tests instead of skill-wide full-suite commands — repository/user instructions prohibit full sweeps without opt-in — cost if wrong: unrelated regressions outside affected paths are not assessed.
3. Rename Unit headings to Task for the execution scripts, define nested fixture validation and classify contract errors separately from quality failures — required for reliable measurements — cost if wrong: fixture/report schema may need an explicit version bump.
4. Report files use exclusive creation with cleanup of partial newly created output on write failure — the plan did not specify partial-file behavior — cost if wrong: interrupted writes require rerunning the report.
5. Reuse the already recorded green test command rather than rerun identical task-done tests — developer instructions prohibit redundant passing sweeps — cost if wrong: bookkeeping script has no independent duplicate receipt.
6. Use a plan-owned pytest basetemp — initial green run produced warnings cleaning unrelated shared pytest garbage — cost if wrong: test-temp reuse must stay within this plan.
7. Normalize expected denied ToolResult from its error field, not a JSON body — the real provider emits no denial JSON — cost if wrong: future transport contract changes require updating the harness.
8. Hash the parsed input bytes once — a mutation regression showed the second read could describe a different fixture — cost if wrong: report/fixture identity becomes unreliable.
9. Scan control responses under their capture-time authority; exclude h10's authorized pre-revocation control from post-revocation canaries — otherwise controls can conceal leaks or create a false leak after revocation — cost if wrong: future revocation cases need explicit capture-time labels.
10. Generate the first report inside the final targeted run and compare a separately written second report from fresh repositories in that same run — shares only the first measured report fixture and avoids a third duplicate full measurement — cost if wrong: separate-process reproducibility is not demonstrated.
11. Run the single final read-only branch review alongside the remaining reproduction checks after the real first report exists and all new logic has focused passing tests — native implementation is complete and review needs no mutable state — cost if wrong: any late test failure must be resolved before task completion and its evidence updated.
12. Keep generated-answer quality, source entailment and prompt-injection resistance explicitly unmeasured — this harness makes no model calls and cannot establish those claims — cost if wrong: downstream answer behavior still requires separate evaluation.
13. Preserve existing device-only and quarantine failures as observations without runtime remediation — the approved task is a baseline and ADR-182 reserves disclosure changes for their own contract — cost if wrong: the existing gaps remain until that work ships.
14. Limit tokenizer and disclosure evidence to the production character fallback and candidate context — no network/server path or alternate tokenizer is exercised — cost if wrong: actual provider behavior may differ.
15. Wait for the concurrent targeted run before marking task status and verification complete — the reviewer correctly left those receipts to the executor — cost if wrong: no completion claim can rely only on the review verdict.
16. Preserve the isolated local branch and worktree for user review — native execution was requested, integration into the dirty shared checkout or publishing was not — cost if wrong: integration remains a separate action.
