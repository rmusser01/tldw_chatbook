# Current-dev prompt mix-in integration

Candidate base: `2acbcd42f3e7b07fd122f5f6dc6b22e7385f10c5` (pinned dev). Local branch: `codex/prompt-mix-ins`. Exact reviewed history preserved at `codex/prompt-mix-ins-reviewed-backup` (`2a12ad62164e30705713b7212e186e6415491205`). TASK-38544 is In Progress for fresh independent review. No push, PR, merge, Qodo request, helper agent, stash, or shared-checkout edit.

## Scope and reconstruction

Preserved a binary `d83ca84c..2a12ad62` feature patch, source/path inventories and reviewed ancestry outside the checkout. Reconstructed the branch directly on pinned dev. Explicitly restored only the feature foundation paths from design/planning commits `13ded5544d` and `235f783e78`: spec, plan, ADR286, its README row, and TASK38534–38544. Those old mixed-ancestry commits are not ancestors of this branch. No radio/mobile/workflow/shared-worktree ancestor work was replayed.

Reopened TASK38544, added integration AC8 and plan before product changes; amended ADR286 for actual schema allocation and evidence provenance. Existing historical tasks remain history; current integration notes and report override their old source qualification. ADR required: no new ADR. ADR path: `backlog/decisions/286-console-prompt-mix-ins.md`. Reason: port the approved design through current owners, without changing authority or retention policy.

Conflicts were resolved around current owners, preserving current plugin-reference handling, hook reads, compaction gates, composer Save/capture semantics, current native trace continuation, and Library submitted-only save acknowledgement. No wholesale feature-side source replacement. The mix-in changes in the extracted trace validator moved to `console_trace_continuation.py`; the extracted `prompt_save_ack.py` now records the submitted literal definition while retaining newer raw body/placement input. A held-save regression proves an invalid new draft survives the acknowledgement, and a later valid Save persists it. Existing ordinary Prompt Save controls pass. The real fresh app exposed the empty-list `Page 1 of 0` label; display now uses the established one-page floor.

ChaChaNotes is 86, with the former experimental mix-in82 migration allocated after actual dev85. Real historical82 Canvas,83 portraits,84 regions,85 archive chronology and all predecessor catalogs remain unchanged. Prompts advances5→6. Fixed source SQL and restricted recovery allowlists use85→86; every current historical catalog remains accepted. The old experimental feature82 is not relabeled as a released historical database. Live profile created fresh86/6. Immutable candidate migration verified81→86 and85→86 while source bytes remained unchanged.

## Verification

All commands use `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest`, `-q --tb=short --show-capture=no -p no:cacheprovider`, in this worktree, unless identified as target-baseline. Temporary roots are exclusively under `/var/folders/sn/m80n2j152t9gw3w8qwk2nykh0000gn/T/prompt-mixins-pr-d18dfic_/`. Logs are retained verbatim. Counts overlap and are not summed.

| Run / log | Result | Scope |
| --- | --- | --- |
| `storage2.log` | 43 passed | `Tests/Backup_Recovery/test_prompt_mixin_restore.py`, `Tests/DB/test_console_prompt_mixins.py`, `Tests/Backup_Recovery/test_core_owners.py::test_schema_policy_matches_installed_store` |
| `native.log` | 141 passed | `Tests/Chat/test_console_prompt_mixin_{forks,retry,journey,admission}.py` |
| `provider.log` | 417 passed, 1 target-baseline failure | artifact/source/provider/portable records and migration-source inventory; exact selection in commands.json |
| `ui.log` | 93 passed, 1 target-baseline failure | Library/Console mix-ins, composer menu, design-token governance |
| `owners.log` | 503 passed, 9 target-baseline failing nodes, FD warning | fork persistence, atomic promotion, contributions, transformations, regenerate, edit/resend, received transitions, first-send/settings |
| `recovery-packaging.log` | 151 passed, 2 target-baseline failures, 5 fixture errors subsequently fixed | historical response rules/recovery, SQLite validation, installed wheel and sdist migration paths |
| `prompt-recovery.log` | 20 passed, 128 deselected | corrected historical Prompts fixture and retained/current migration/refusal/rollback cases |
| `save-ack2.log` | 1 passed | new literal Save acknowledgement held-worker regression |
| `save-continuity-admitted.log` | 8 passed | existing ordinary Prompt Save continuity, native bootstrap fixture |
| `final-covering.log` | 57 passed in47.63s | exact executable `final-covering-command.sh`: original final51, new literal Save regression, five CSS bundle guards |

The final covering command used the read-only per-node FD observer. First fork test14→17; remaining13 new fork cases17→17, no sentinel warning. Its historical observer retained an absolute output filename `/private/tmp/mixins-final-fd-audit.json`; that single exact file was copied only after this owned run completed, with exitstatus0 and the exact14 named nodes checked. No shared-temp discovery glob was used. The copied `final-fd-audit.json` is this run; the historical QA version remains unchanged. The observer script preserves its actual executed output path for reproducibility.

The wider owner run reports FD+848 (14→862); its aggregate ownership cause is unqualified. Passing final57 does not erase it. No warning suppression, threshold increase, forced GC, or unrelated cleanup fix.

### Target-baseline attribution

Target code came from exact `git archive 2acbcd42` into private `target-dev`; no test reading Git source was qualified from that archive. `target-baseline.log` reproduces the exact decorative-migration inventory and uninitialized `__new__` impersonate fixture failures. `target-rag.log` reproduces both unrelated native RAG-index schema-zero cases.

The first nine-node target-owner run was blocked earlier by missing native profile admission and is retained as noncausal setup evidence. `target-owners-admitted-command.json` and log rerun exact target tests with a plugin that **only applies existing `pytest.mark.bootstrap_profile`**, matching the markers already present in ported modules. No runtime monkeypatch, replaced owner, fake success, or relaxed assertion. All nine remain failing: sparse context-policy postcommit write; failed/stopped transformed-call continuation; two regenerate failure path selections; two obsolete persistence doubles lacking native generation ownership; old one-argument skill refusal double; vision-capability fixture. The two persistence-double failures hit the new strict mix-in-record refusal first on the feature, then the same unavailable-generation-projection error; target fails at its generation owner. This is a demonstrated invalid-stub baseline, not proof identical internal paths or successful durable replay. Real native persistence/replay/fork tests above pass.

Five initial Prompt migration setup errors came from the port initially omitting the feature's `prompts-v5` fixture arm while retaining current-dev core-owner changes. Restored that bounded fixture; exact Prompt subset20 passes. This was a port defect, not target baseline. Initial `storage.log` named a nonexistent test node and ran none; corrected `storage2` is evidence.

Initial new Save test used an asyncio Event across the physical Library worker thread, could not retire, and was explicitly interrupted/terminated; raw teardown refusal is retained in `save-ack.log`. Corrected to the existing bounded threading.Event pattern; `save-ack2` and final57 pass. Initial ordinary Prompt continuity invocation lacked bootstrap admission (8 setup failures); the same existing cases with native fixture admission pass8. No such failed setup is counted green.

### Static and scope checks

Per-file target-dev Ruff comparison: **2852 baseline /2850 current**, zero introduced diagnostics (`target-lint.json`, `static-final.log`). F811 messages embed source line numbers; comparison normalizes only that number, preserving code/message/multiplicity per file. Existing duplicate TYPE_CHECKING imports in provider gateway remain inherited. New unused inner import and obsolete assignments were removed. Python AST parses current changed source; class duplicate-method inventory matches target (property setter/deleter pairs excluded). New files and any previously format-clean files were formatted; existing formatter debt is recorded in `format-before.json`. This is not a whole-repo lint-clean claim.

Source CSS rebuilt with installed builder; only intended `_console_panels.tcss` and bundle differ. Final bundle guards5 pass; token governance passes in UI93. `git diff --check` passed for product/tests/current docs/backlog; raw historical ANSI/log artifacts are intentionally preserved. Scope inventory includes two new current owner paths (`console_trace_continuation.py`, `prompt_save_ack.py`) beyond reviewed source inventory.

## Actual current app

Used existing private harness environment allowlist and console-script launcher, fresh HOME/XDG/data/config, null keyring and a nonsecret loopback provider placeholder; no external keys or external model calls. Initial no-provider profile correctly showed setup; adding the harness placeholder made Composer ready. No provider Send or cache-hit measurement attempted.

Final source-qualified launch hashes79 changed product files before startup. Existing Composer Menu→Prompts→Mix-ins opened at160×45; empty Before/After selections and corrected `Page 1 of 1` are visible. Resized to80×24 and retained the workbench; captures are actual PTY text/ANSI, not native OS-font fidelity. Final hashes unchanged after capture. Normal Ctrl+Q, final app PID4076 absent, private tmux socket absent, read-only DB checks86/6 (`live/final-teardown.json`). Detached exit code unavailable. `live/app.log` includes initial and final launches; final source manifest applies only the `final-*` captures. Earlier initial frames are preserved as exploratory evidence, not falsely hashed to final code.

## Provenance and remaining gate

Historical whole journey at247c9eb and narrow Preview atd7744a78 remain byte-for-byte, as do their original source manifests. Historical reviewed source/evidence manifests are retained separately. Current manifest hashes ported source/tests. Verifier checks all current filesystem and Git bytes strictly; historical Git hashes are checked only when those old objects are locally present, with explicit UNAVAILABLE otherwise. Those old source commits are not carried in the focused PR ancestry. Fresh `--no-local --single-branch --depth 1` clone of candidate `5dac2bb0207285eca67d571565a3ee88113e4a84` verified634 current records and79 current live hashes against HEAD; both old source commits are absent and explicitly reported UNAVAILABLE (77 hashes each). Separate full-repository ancestry checks prove neither old feature base nor reviewed mixed head is an ancestor. Exact commands/results are `verify-fresh-clone.json` and `.log`. Unavailable history is not reported as verified.

Fresh independent review and publication remain with the controller. Full suite, Qodo, external provider/cache counters, costs/latencies, broad baseline repairs, and original whole live journey were not run. Existing M4 browser-retirement noise remains deferred; wider-owner FD warning and baseline failures remain disclosed. No permission expansion, body logging, or new dependency.

## Handoff

Product candidate: `5dac2bb0207285eca67d571565a3ee88113e4a84`, one focused commit directly on pinned dev. `static-scope.json` records117 parsed Python files, zero new duplicate class methods, and zero unexpected feature paths. The following evidence-only commit adds clone verification and this completed handoff; product bytes remain identical. Root owns fresh independent review, all resulting fixes, and draft publication.
