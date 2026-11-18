# Task 5 report — PR 2995 ownership Fast Lane failures

Status: **DONE_WITH_CONCERNS**. Three lifecycle failures repaired with causal evidence; the independently unproved native CI timeout remains a current-head CI qualification concern.

## Scope and immutable identities

- Managed worktree: `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`, branch `codex/console-chat-starts-dev`.
- Post-Task4 BASE: `c0b251a71422536a3da2be421dd60f79e2cb230a`.
- Initial failing CI source: head `7a15e3e9b4998dba8d8726bbdb722ff60a3df83d`, tested merge `112298c452082b71c9a21572e7982f5bfd6bc9c9`; both tree identities `9d2ff375673c6b46af8a85babf0d4960f68a271d`, established by `ci-initial-merge-tree-receipt.json`. Initial job log and `ci-ownership-triage.md` remain preserved.
- Four immutable CI failures, in original order: second Console visit hook `None`; same-target successor runtime view `None`; headless wake return poll `None` with reconciliation false; native unchanged readiness Event times out before receipt/composer assertions.
- Only owned tracked change: `Tests/UI/test_console_runtime_ownership.py`. No production, shared factory, startup budget, profile guard, configuration precedence, or Task4 interface changes.
- ADR required: no. ADR paths: `094-console-turn-lifetime-and-navigation-boundary.md`, `097-boot-budget-ratchets.md`, `211-console-chat-destinations-and-bounded-starts.md`. Reason: restore existing ownership test setup and qualify the existing contract; no ownership boundary is changed.

## Diagnosis and causal evidence

Three lifecycle failures have a reproduced test-harness cause. On BASE, real deferred startup finishes installing and retaining Console A before the test body constructs/pushes Console B. The stack is default/A/B. Marking `_initial_screen_pushed` after B's awaited push does not remove A or its reusable-navigation cache entry. This is unsupported duplicate content-screen setup.

In `four-diag.jsonl`, the first failure has startup task `Task:1232d10c0`, initial Console `ChatScreen:1232b65d0` (generation 1), and manual Console `ChatScreen:116e695e0` (generation 2). Startup completes before the manual screen is constructed. Leave navigation dismisses B, correctly detaches its exact generation 2 claim, and switches the retained A out. Return navigation reuses A. A presents superseded generation 1; `attach_view` rejects it, `finish_view_reconciliation` returns false, runtime view stays None, and the explicit controller ensure cannot restore hooks. This is stronger evidence than DOM presence or pauses.

The same-target test likewise navigates to the retained initial Console rather than constructing the intended fresh successor. The headless-delivery test returns to that stale initial Console; its reconciliation never publishes and delivery poll never arms. No missing generation fence or late old-screen detach defect was found. The existing fences correctly refuse the unsupported stale claimant.

The startup-before-manual ordering is forced in `test_manual_console_fixture_owns_startup_before_any_mount` by awaiting the exact retained `_initial_screen_setup_task` before constructing the harness screen. Before repair, it fails with an actual default/Chat/Chat stack (`causal-red.log`, exit 1, 14.39 s). After repair it has exactly default/manual Chat, unchanged claim generation, and no retained startup Chat cache entry. The test also invokes the real `_push_initial_screen` explicitly after manual mount to qualify the late-callback boundary without a timing assumption.

The independent public single-Console control uses the original factory and real startup task, not the manual fixture. It passes before repair (`startup-red-control.log`: one expected manual-fixture RED, public control GREEN). It asserts one Console, current-generation reconciliation, same runtime/store/controller/bridge, retained mounted Console after navigation, live outcome hooks, no idle poll, and resumed poll during a genuine active wake delivery. It passes after repair too. Thus no product-navigation repair is supported by this evidence.

The smallest repair is a local `_build_manually_mounted_console_app` factory that establishes `_initial_screen_pushed=True` before `run_test` can schedule startup. Nine existing manual-screen test functions use it; their obsolete post-push assignments are removed. Startup tests retain `_build_startup_test_app`, an alias of the unchanged shared factory. All original runtime/controller/store/bridge identity, successor, old-screen refusal, wake polling, active-delivery, receipt, composer, navigation and durable-consumption assertions remain intact.

## Native readiness: independently unproved CI timeout

The fourth original node passed in the initial four-node diagnostic run before repair. Its CI timeout was not reproduced, and the three lifecycle cause is not assigned to it. No timeout was raised, no retry was added, and no production native-start code was changed.

The original local trace identifies start task `Task:12946b580`, coordinator `ConsoleChatStartCoordinator:127a0be00`, controller `ConsoleChatController:128ed3710`, physical coordinator task `Task:12946be80`, and send attempt `d30f63f017db4c6d93e212a3a656b06f`. Preparation enters at 458418.728 and completes at 458418.957; controller submission enters at 458418.958. The exact attempt records provider resolution, capture policy, durable commit entered and succeeded, then start returns `AgentChatStartOutcome(launch_status='started', reason=None)` at 458431.615. The original node reaches its held readiness barrier and original dual-fence/receipt/composer assertions and passes. The CI's historical 63.5 s cancelled durable commit cannot be attributed to this local start or any identified CI start task.

After repair, the native unchanged trace identifies start `Task:121a47280`, coordinator `ConsoleChatStartCoordinator:1261d73e0`, controller `ConsoleChatController:125f627e0`, physical task `Task:1214c9d80`, attempt `149cb6bd246942658d6594b4de7c60a5`, successful preparation, durable commit and `started` outcome. All acceptance/consumption assertions pass. The temporary recorder decorates behavior without replacing outcomes; it is not committed.

The native test now races its readiness Event with the exact `start` task, retaining the original five-second readiness bound. If start finishes first it reports the actual outcome. If both remain pending it reports start state and preparation task frames. Readiness must still be reached before the original test proceeds. This is diagnostic strengthening, not a claim to have fixed a reproduced native timing defect. Current-head CI must qualify the node after publication.

Diagnostic limitation: the disposable recorder's retired field queried `_console_view_retired`, whereas production uses `_console_runtime_attachment_retired`. Those entries are null and are not evidence about retirement. The traces retain actual screen identity, full stack, exact generation/prior generation, `_closing`, reconciliation, startup task identity/done state, attach/detach results and hook presence; the causal/current-claim assertions and production fences establish the reported diagnosis. The permanent tests do not use the incorrect diagnostic field.

## RED / GREEN commands and outputs

All test commands use `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`, the managed-worktree cwd, a unique pre-created mode-0700 private `TLDW_TEST_CONFIG_ROOT`, worktree `PYTHONPATH`, unique private `--basetemp`, and `--timeout=300`. No dependencies were installed. Diagnostic runs additionally use `/private/tmp/pr2995-task5` on PYTHONPATH, `-p ownership_diag`, and the corresponding `OWNERSHIP_DIAG_LOG`. Exact disposable scripts/traces are copied to `task-5-evidence/`.

1. Initial launch, exit 4: profile directory had not yet been created. `four-original.log` preserves the `resolve(strict=True)` FileNotFoundError; no test body ran. Created the required private directory and used a new log.
2. Initial four nodes in original order, exit 1: `four-original-diag.log`, **3 failed, 1 passed in 76.40 s**. Command:

```sh
env TLDW_TEST_CONFIG_ROOT=/private/tmp/pr2995-task5/four-profile PYTHONPATH=/private/tmp/pr2995-task5:/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook OWNERSHIP_DIAG_LOG=/private/tmp/pr2995-task5/four-diag.jsonl /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_runtime_ownership.py::test_second_console_visit_reuses_the_runtime Tests/UI/test_console_runtime_ownership.py::test_a_superseded_screen_never_detaches_the_successors_runtime Tests/UI/test_console_runtime_ownership.py::test_opening_console_during_a_headless_delivery_arms_the_poll 'Tests/UI/test_console_runtime_ownership.py::test_native_acceptance_consumes_only_open_target_revision[unchanged]' -p ownership_diag --basetemp=/private/tmp/pr2995-task5/four-basetemp --timeout=300 --tb=long -vv
```

The isolated run is red, so the conditional four-file-collection/failing-node-only probe is not needed.

3. Initial fixture regression and public control, exit 1: `startup-red-control.log`, **1 failed, 1 passed in 16.00 s**. Nodes `test_manual_console_fixture_owns_startup_before_any_mount` and `test_public_startup_console_keeps_its_claim_across_navigation`, with `red-profile`, `red-basetemp`, `--timeout=300 --tb=short -vv`, no diagnostic plugin. Its first regression version fails the missing pre-mount ownership flag.
4. Causal regression, exit 1: `causal-red.log`, **1 failed in 14.39 s**, actual extra content Console. Node `test_manual_console_fixture_owns_startup_before_any_mount`, `causal-profile`, `causal-basetemp`, diagnostic `causal-diag.jsonl`, `-p ownership_diag --timeout=300 --tb=short -vv`.
5. Post-repair six-node diagnostic GREEN, exit 0: `six-green-diag.log`, **6 passed in 75.72 s**. Original four nodes in their original order, then the two controls above; `green-profile`, `green-basetemp`, `green-diag.jsonl`, `-p ownership_diag --timeout=300 --tb=short -vv`.

`diagnostic-source-states.json` records original BASE hash `683adeb3dc538c65fa341bae0d377c9e99ca5ff1d0b6b94cde922e0fb3d8e4d9`, initial regression hash, causal RED hash and six-GREEN hash. These intermediate test hashes were reconstructed exactly from pinned BASE plus the saved ordered edit scripts; they were not captured contemporaneously. The final complete-batch receipt captures before/after source fingerprints contemporaneously.

## Static checks and self review

- Fatal Ruff, exit 0, `fatal-ruff.log`: `python -m ruff check --select E9,F63,F7,F82 Tests/UI/test_console_runtime_ownership.py`.
- Formatter snapshot, exit 0, `format-snapshot.log`: `python scripts/terminal_qualification/format_ratchet.py snapshot --base c0b251a71422536a3da2be421dd60f79e2cb230a --path Tests/UI/test_console_runtime_ownership.py --output .superpowers/sdd/2026-10-03-console-pr2995-review-and-merge/task-5-evidence/format-baseline.json`.
- Initial formatter verification, exit 2, `format-before.log`: expected new-range formatting debt; BASE has 56 inherited debt units, provisional repair had 99. `format_ranges.py` invokes Ruff only on Git-changed ranges, preserving unrelated inherited debt.
- Final formatter verification, exit 0, `format-final.log`: `python scripts/terminal_qualification/format_ratchet.py verify --baseline .superpowers/sdd/2026-10-03-console-pr2995-review-and-merge/task-5-evidence/format-baseline.json`.
- Source whitespace, exit 0: `git diff --check c0b251a71422536a3da2be421dd60f79e2cb230a -- Tests/UI/test_console_runtime_ownership.py`.
- `self-review.py` / `self-review.json`, exit 0: all original assertions and decorators in **70 original functions** are preserved; no original function is deleted. Every original assertion AST remains present (the native readiness race adds assertions). All inspected production files have empty diff against BASE, including controller/start/automatic-work Task4 sources. No inherited XFAIL/decorator/warning policy is changed.
- Final owned-file SHA256: `91c7698d342e412bf8796575c7a72b4a2fe42985bc8ff1169a98018694619521`.

Root plan, Backlog task and root-owned lesson metadata remain excluded. Independent scoped spec/quality review, upstream latest-dev/recovery-owner qualification, final startup/import budgets and publication/merge remain controller-owned.

## Final complete CI batch and exact committed artifact

The original four-file CI batch ran once on final formatted source, without the disposable instrumentation plugin, selecting every node in all four files. It also qualifies the complete affected ownership owner; no separate whole-ownership run was duplicated.

```sh
env TLDW_TEST_CONFIG_ROOT=/private/tmp/pr2995-task5/batch-profile PYTHONPATH=/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_runtime_ownership.py Tests/Chat/test_console_viewless_hooks.py Tests/Agents/test_install_skill_runtime_tool.py Tests/Chat/test_console_chat_create_integration.py --timeout=300 --tb=short --basetemp=/private/tmp/pr2995-task5/batch-basetemp --junitxml=/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge/task-5-evidence/batch.xml
```

Exit **0**; **155 passed, 1 inherited XFAIL, 5 warnings in 343.84 s** (subprocess wall receipt 353.424 s). Per-file JUnit counts: ownership **80 passed + 1 XFAIL**, viewless hooks **18 passed**, skill install runtime **6 passed**, chat creation integration **51 passed**. The native unchanged node passes in 14.75 s; all six consumption variants pass. The inherited strict XFAIL is `test_captured_attach_timer_overlap_rearms_real_sync_worker` (TASK-32873). Warnings are four inherited invalid-escape SyntaxWarnings from the construction census and the inherited session FD-growth warning (523, start 14/end 537, limit 200). No suppression, marker or cleanup policy was added; this does not claim the FD warning is repaired.

`batch-receipt.json` records exact argv, selected environment, BASE/HEAD, before/after SHA256 for all four test files and eight inspected production files. Every fingerprint is unchanged across the complete run. Test environment remains macOS CPython 3.12.11, pytest 8.4.2, pytest-asyncio 1.2.0, Textual 8.2.8; this is not the Ubuntu/Python 3.12.14/pytest 9.1.1 CI environment.

Source-only commit: **`e7cc5337617781e9a4b08b9e324297b537fcad07`**, parent `c0b251a71422536a3da2be421dd60f79e2cb230a`. One committed path, `Tests/UI/test_console_runtime_ownership.py`; commit SHA256 matches qualified final bytes `91c7698d342e412bf8796575c7a72b4a2fe42985bc8ff1169a98018694619521`. `commit-receipt.json` proves exact blob equality, stable complete-batch fingerprints and an empty index. Root plan, Backlog task and root-owned testing-evidence lesson remain dirty and excluded.

Committed formatter verification exit **0**, `format-commit.log`:

```sh
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python scripts/terminal_qualification/format_ratchet.py verify --baseline .superpowers/sdd/2026-10-03-console-pr2995-review-and-merge/task-5-evidence/format-baseline.json --head e7cc5337617781e9a4b08b9e324297b537fcad07
```

Self-review conclusion: a local test harness setup correction is supported; a production ownership repair is not. Original checks and inherited qualifications are preserved. The native timeout requires current-head CI evidence and is reported as unproved. Root owns independent scoped spec/quality review and final startup/import/latest-dev qualification. A useful lesson addendum would explain that a post-push ownership flag cannot erase an already retained startup Console; the pre-mount boundary matters. Existing TASK-31645's manual-startup lesson already covers the underlying rule.
