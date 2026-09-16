# Canonical Eval definition after later rollback

**Final verified:** assembled native lifecycle passed 1 case in 254.31 seconds; fresh ordinary reopen and the next actual Complete backup both succeeded. All four source/test hashes remain frozen in `nested-persona-final-hashes.json`.

TASK-32562. Bounded follow-up discovered by the nested Persona native lifecycle; files `tldw_chatbook/Evals/recovery.py` and `Tests/Backup_Recovery/test_eval_rollback_retention.py`.

The completed later rollback correctly restored the original target pack and stored current incoming-pack/new-note bytes in its verified encrypted safety copy. The final aggregate inventory still refused. A fresh-process metadata-only trace reproduced `ValueError(eval_retained_originals_unverified)` at `_original_definition_paths:465`. The offending record was exactly `_default_config_path()`, included/existing, explicitly preserved, not retired, and outside safety scope. Native `_DefinitionsAdapter.discover` already declares this same canonical path unconditionally; retained-extra reconstruction raised before its existing deduplication could apply.

The six-line product change skips only that exact current canonical path in the `preserved_only` branch, after target/config/journal/safety-proof consistency and record status checks. Normal canonical declaration and validation remain unchanged. Arbitrary retained paths still require original/safety coverage; no basename, prefix, namespace, or activation exemption was introduced.

Evidence:

- Original full lifecycle failure: `/private/tmp/uat-persona-complete-lifecycle.log`, 1 failure in 219.92 seconds. Terminal later commit, source-stable repeated/held previews, encrypted new-note/pack readback, and original target pack hashes passed before the Complete assertion failed.
- Fresh process cause: `/private/tmp/uat-persona-post-rollback-cause.log`; exact canonical equality/coverage: `/private/tmp/uat-persona-eval-coverage.log`.
- Separate native installed Persona graph and original byte verification passed: `/private/tmp/uat-persona-native-restored-graph.log`. This did not waive the aggregate Complete assertion.
- New actual native replacement/rollback canonical-preserved regression RED: `/private/tmp/uat-eval-canonical-red.log`, 1 failed in 4.47 seconds, exact `eval_retained_originals_unverified`.
- Full existing/new native Eval retention file GREEN: `/private/tmp/uat-eval-canonical-green.log`, **9 passed in 33.48 seconds**, including uncovered noncanonical refusal and plan/terminal/association/missing/alias/unrelated guards.
- Product Bandit zero; test-only pytest assertions remain. Ruff reports exactly the same three pre-existing findings in product HEAD/current (I001:6, UP035:11, I001:134), recorded in `/private/tmp/uat-evals-ruff-comparison.json`; both touched test files pass Ruff. Scoped diff check passes.

Frozen hashes: product `aa3376537520c4082dbf59c0d3245dfbe539d7cbd1a27156be43fc4c6c2dce78`; test `269b3b4deedb2cc2dbfc545dde213b7b0c4cfc8ba351555199447f0d47c2bfa9`.

At the freeze checkpoint the final assembled lifecycle was running at `/private/tmp/uat-persona-eval-final-lifecycle.log`, including the retained Complete assertion, fresh ordinary reopen with original-target note/current-source note distinctions, and a subsequent actual Complete backup. Its new finite enclosing test ceiling covers the sum of existing child observation ceilings (Windows 4,200 seconds; others 900); product deadlines are unchanged. Final acceptance was withheld until that run and independent review completed.

Final update: the above run **passed in 254.31 seconds**. Aggregate Complete inventory, exact original Persona graph/bytes, fresh mounted original-note readback, exclusion of incoming/post-replacement notes from the restored profile, and a subsequent actual Complete archive all passed. Independent Evals review approved both frozen hashes and reran all 9 cases successfully in 37.30 seconds: `/private/tmp/uat-eval-canonical-independent-review.md`. No pending implementation work remains in this correction; controller owns commit and final platform dispatch.
