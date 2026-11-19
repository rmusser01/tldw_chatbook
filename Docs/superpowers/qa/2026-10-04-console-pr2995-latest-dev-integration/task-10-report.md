# Task 10 — pure fork placement source report

Status: source qualified and cleanly committed; root scoped review, controller/current-dev integration and final external/loading gates remain open. No publication-readiness claim.

BASE: `fe4eaf365c08705dcbe3edfa542fa0259d9b038e`. Reviewed I1 pin: `d3443b9e4297fa20897cf99e77a3ae2c8c562b10`. Both production and test trees at BASE exactly match that reviewed pin. Final source HEAD: `1d55d8f89fd2d2b05dce65d3f011ece9eb1bf6c9`. Commits: `d0f17e7cef` (two source paths and targeted test additions), then `1d55d8f89f` (only guarded type import order). Root owns report/maps and task metadata commits; this worker committed no metadata.

## Approved placement and responsibility

ADR required: no new ADR. Existing paths: `backlog/decisions/092-console-chat-fork-copy-and-authority-boundary.md` and `backlog/decisions/219-console-chat-destinations-and-bounded-starts.md`. This is direct pure placement under their existing ownership boundaries. Read the Task10 requirements and generated brief, selected preflight report/map/final sketches, these ADRs and bounded testing/live/decomposition migration lessons. No UI or CSS work.

Only `tldw_chatbook/Chat/console_chat_store.py`, `tldw_chatbook/Chat/console_chat_fork.py` and `Tests/Chat/test_console_chat_fork.py` changed. The exact helper/retarget and signature/decorator map is in `task-10-freeze-map.json`.

| Original store owner | Existing fork owner function |
| --- | --- |
| ConsoleChatStore._fork_message_state_is_eligible | console_fork_message_state_is_eligible |
| ConsoleChatStore._fork_visible_selection | console_fork_visible_selection |
| ConsoleChatStore._fork_attachment_fingerprint | fingerprint_console_fork_attachments |
| ConsoleChatStore._validate_fork_image_selections | validate_console_fork_image_selections |
| ConsoleChatStore._fork_video_fingerprint | fingerprint_console_fork_video |
| ConsoleChatStore._stage_fork_snapshot::message projection | project_console_fork_message |
| ConsoleChatStore._stage_fork_snapshot::candidate comparison | console_fork_candidate_matches_fence |
| ConsoleChatStore._validate_video_projection_tuple | validate_console_fork_video_projection |

Six original method names, signatures and static/class decorators remain. Their forwarding docs describe the genuine pure owner. The original video projection responsibility doc is preserved verbatim in its implementation owner; the store wrapper accurately names forwarding and live lookup. All original six implementation bodies match the helpers after removing only documentation/local imports and substituting the explicit video callback. The staging reconstruction expands the two helper calls back into an AST exactly equal to the original whole `_stage_fork_snapshot` body, including its original docstring. This proves order, checks, fields, errors and concrete candidate projection survived the move.

The store retains its source lock; initial/final fence validation and exact image selections; configuration capture and validation; UUID, turn and predecessor allocation; append order and image-alias writers; citation/destination ownership; candidate creation; final refusal and return. `_fork_media_fingerprint` and generation-reload wrapper routes remain unchanged. Helpers receive borrowed messages/lineage/read-only maps plus narrow callbacks and never receive a store. All eight have no attribute/subscript writes, await/yield or async function. Local construction lists stay local.

Invocation-time lambdas reread `self._fork_video_fingerprint`, `self._fork_attachment_fingerprint` and `cls._fork_video_fingerprint`. New controls replace each method after helper arguments/callbacks have been created and assert the replacement is invoked on the actual row. The borrowed-input control uses mapping proxies and exact source pickle bytes, collection/parameter/byte identity checks and frozen-result enforcement. All preexisting test source bytes are retained as an exact prefix; assertions, exclusions and warning policies were not changed.

## Counts, dependencies and loading limits

Actual installed Ruff version: 0.16.6; no installation. Counts are `len(text.splitlines())` after actual formatting. Store 22,655 → **22,338**, unchanged cap **22,344**, six lines of slack. Fork owner 1,135 → **1,584**; its existing module has no cap row in the module ratchet. No owner cap or exception was added or raised. Joint production source grows **132** lines. Do not call this an overall source-size reduction. The six lines of store slack fit the existing 50-line tolerance, so the unchanged22344 row is retained honestly.

The actual source differs from the final proposal's22,334/1,563 by accurate wrapper/helper documentation and removal of the two demonstrated unused store imports. Original `math` moves into the existing fork owner; existing owner attachment import adds `MAX_ATTACHMENT_BYTES`. Store removes now-unused `attachment_core.MAX_ATTACHMENT_BYTES` and the store-global `fingerprint_console_fork_selected_image` alias after bounded consumers/patch census; the existing `validate_console_fork_image_payload` store alias remains because its other use remains. Video dependencies stay function-local where moved helpers need them. There is no new eager project module, store state owner, generic resolver, storage access, lock, scheduling operation or authority.

The final correction only exchanges `TraceForkBoundary` and `VideoGenerationMetadata` inside `if TYPE_CHECKING`. Runtime eager import order is unchanged. The normal-runtime AST (guarded type imports omitted) is byte-identical in AST serialization; all function/class/test ASTs and store bytes remain unchanged from the tested source. `type-only-import-order-proof.json` links that exact tested owner hash to the final owner hash. Root explicitly authorized carrying the280/2 qualification through this proof and refreshing static checks, without replaying those suites.

Ready1033/preimport557 caps and prior startup receipts carry; no fresh runtime census, timing, startup or public-navigation execution occurred. The controller's separate cap failure remains untouched and unqualified. Final loading qualification waits for both structures and latest-dev integration.

## Verification and exact command receipts

The only actual pytest invocations were the complete three fork modules once and the two store-row ratchets once. Use the worktree's `PYTHONPATH`; canonical `Tests/conftest.py` creates/owns isolated private temporary bootstrap and per-test profiles before product imports. Every command receipt records exact cwd, arguments, HEAD and SHA256 before/after. All actual test executions stayed source-stable. Nothing was skipped, xfailed, suppressed or weakened.

Fork qualification: **280 passed in13.33s**, zero failures/skips. Per-source cases: `{"Tests.Chat.test_console_chat_fork": 241, "Tests.Chat.test_console_fork_public_boundaries": 32, "Tests.Chat.test_console_trace_fork_lineage": 7}`. Store ratchets: **2 passed**. The passing forks include four added cases. Required fatal Ruff (`E9,F63,F7,F82`), all three touched formatter files and source whitespace pass at final source. Full logs/XML are preserved.

Unsuppressed warnings: fork log retains24 post-summary `PytestWarning` entries; ratchet log retains24. They report `rm_rf` failing to remove existing global pytest garbage directories with `OSError[Errno66] Directory not empty`, including paths named by unrelated prior fs tests. They did not alter the0 exit or XML results. No cleanup/deletion of those unrelated directories was attempted.

| Receipt | Exact command | Result |
| --- | --- | --- |
| format-check | `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format --no-cache --check tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/console_chat_fork.py Tests/Chat/test_console_chat_fork.py` | 0; stable HEAD/hashes True |
| fatal-ruff | `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check --no-cache --select E9,F63,F7,F82 tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/console_chat_fork.py Tests/Chat/test_console_chat_fork.py` | 0; stable HEAD/hashes True |
| unrestricted-ruff | `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check --no-cache --output-format json tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/console_chat_fork.py Tests/Chat/test_console_chat_fork.py` | 1; stable HEAD/hashes True |
| fork-qualification | `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_console_chat_fork.py Tests/Chat/test_console_fork_public_boundaries.py Tests/Chat/test_console_trace_fork_lineage.py --junitxml=.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge/task-10-safe-evidence/fork-qualification.xml` | 0; stable HEAD/hashes True |
| store-ratchets | `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/Chat/console_chat_store.py] Tests/Architecture/test_module_size_ratchet.py::test_budget_is_not_left_slack[tldw_chatbook/Chat/console_chat_store.py] --junitxml=.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge/task-10-safe-evidence/store-ratchets.xml` | 0; stable HEAD/hashes True |
| source-whitespace | `git -c gc.auto=0 diff --check` | 0; stable HEAD/hashes True |
| final-format-check | `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format --no-cache --check tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/console_chat_fork.py Tests/Chat/test_console_chat_fork.py` | 0; stable HEAD/hashes True |
| final-fatal-ruff | `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check --no-cache --select E9,F63,F7,F82 tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/console_chat_fork.py Tests/Chat/test_console_chat_fork.py` | 0; stable HEAD/hashes True |
| final-unrestricted-ruff | `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check --no-cache --output-format json tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/console_chat_fork.py Tests/Chat/test_console_chat_fork.py` | 1; stable HEAD/hashes True |
| final-source-whitespace | `git -c gc.auto=0 diff --check` | 0; stable HEAD/hashes True |

The initial source-only command `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check --no-cache tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/console_chat_fork.py` exited1 with122 findings and was reported to root before corrections. Its tool output was truncated; named `first-unrestricted-replay-*.json` receipts exactly reconstruct the pre-unused-import-removal input via stdin and reproduce122 findings, clearly labelled as reconstruction. The first source format command reported2 unchanged. Baseline lint via the exact stdin command in `unrestricted-lint-classification.json` finds117 store +2 fork +12 preexisting test findings=131. After removing the2 placement-unused imports, the initial three-file lint receipt had132 due to one new guarded-import I001. Final unrestricted lint still exits1 on **131 existing findings**, with exactly the baseline code/message multiset and **no introduced finding**. Preserve that failure; required fatal selectors passed and no broad cleanup or suppression was applied.

Source/map commands used: `python3 /private/tmp/task10_freeze.py` asserts six normalized body matches, seven store method changes, whole-stage reconstruction and original test-prefix bytes; compares exact Git blobs; and invokes the actual static diagnostic scanner on exactly the two affected strings. Its final exit0 result is recorded by the freeze/source maps. An initial mapper was intentionally terminated with143 after inefficient per-file hash subprocesses; an attempted termination wrapper also exited143, then exact-process selection and one Git changed-path scan replaced the redundancy. No tracked source/test correction or test rerun resulted. `python3 /private/tmp/task10_report.py` finalizes assertions, source-provenance and named safe evidence. Git mutations were only explicit adds/commits of the3 approved paths, using `git -c gc.auto=0` throughout. Commit messages were `refactor(console): place pure fork projections with fork owner` and `fix(console): order fork type-only imports`.

## Unchanged owner and derived evidence

`store-method-ast-map.json` proves **617 untouched store methods**, identical method name inventory, and only the mapped seven changes. All18 preexisting fork functions/methods retain their ASTs; unmapped nonimport module/class nodes also match BASE. `exact-blob-carry-map.json` covers **20,394** production/test/historical-QA paths; only the3 approved paths differ. All **12,337 historical QA paths** retain exact blob IDs. This carries controller, native draft/handoff/receipt/revision/authority/I1 owners and prior qualified schema/provider/encryption/chat-start evidence without replay or wider inspection. Another SDD was not inspected.

The actual static diagnostic scanner's entire three-result projection matches BASE exactly for both affected files: store93 diagnostics, digest`3a27e46667c8e6bbfc43`; fork0, digest`4f53cda18c2baa0c0354`; zero sinks/path candidates in both. Indexed semantic persistence writers never moved. Their methods and derived test/inventory files are exact AST/blob carries. No diagnostic/source inventory update was needed or made. Carry proofs do not invent a new all-module, runtime or unrelated-subsystem pass.

## Source hashes

| Path | BASE SHA256 | Final SHA256 |
| --- | --- | --- |
| tldw_chatbook/Chat/console_chat_store.py | `078c5c8ad875e74334a45c2a3e89adfe102ec3293b456719e7342cf12392a5b8` | `995dfd03d0dfb394a6d57e2520e9ffc776a11206da4f8069abeff6a51033f9e6` |
| tldw_chatbook/Chat/console_chat_fork.py | `c117a1c27468dc1e3ad22b52e280e4de4159e41534ae5fdd3d4ea4e44a2b4dd5` | `55a16b1eefe7d9d50ef46644f4dbb7a760c7fc7a13c64ee1a088570c7d1cbc72` |
| Tests/Chat/test_console_chat_fork.py | `fb531d9c0ee6d5891c1ff2c907be5c2b3f7f44b0ae970ed13f8ffa70bfd0504a` | `39b0c733f65e4df6d5fdb7b5eb47c90ee4adc8775db4a4c07385a6ff37d1d704` |

The passing behavioral receipts name the pre-type-ordering owner SHA`5d9be7714fca46a4f6dbe8248c06515d32b56f6d9d5bfadbcae5cc9f907c2e02`; the final type-only proof carries it to the final hash above. Final static receipts pin the final hashes; source commit1d55d8f89f does not change those file bytes.

## Safe freeze and remaining concerns

`task-10-freeze-map.json` is the required final placement/source map. `task-10-safe-evidence-map.json` names only safe receipts, logs, XML and source/blob/AST hash maps under `task-10-safe-evidence/`; no profile, configuration body, database or cache is copied. Selected proposal and first NON-FIT artifacts remain exact bytes and are hashed by the safe map. Root can commit these metadata artifacts separately.

Self-review checked the exact scoped diff, reconstructed bodies and both fence order, live callback routes, lack of input mutation, immutable outputs, baseline findings and unchanged surrounding owners. Independent scoped spec/quality review remains root-owned; no reviewer or subagent was spawned. Store headroom is narrow. Private pure-global image lookup now follows the fork owner as selected; arbitrary external dynamically constructed private monkeypatch compatibility remains unproven. Controller/latest-dev/external loading and publication gates remain open. No push or merge occurred.
