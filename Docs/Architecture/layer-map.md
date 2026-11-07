# Layer map

tldw_chatbook has four layers. A module may import from its own layer or a lower one, never a higher one. `Tests/Architecture/test_layer_direction.py` enforces this, and it freezes the upward imports that already existed when it landed in an allowlist that may only shrink. The rule comes from the architecture review of 2026-10-25 (`Docs/QA/architecture-review-2026-10-25/report.md` §F6).

## Authoritative files

| File | Role |
| --- | --- |
| `Tests/Architecture/test_layer_direction.py` | The guard: a stdlib AST scan of `tldw_chatbook/` that fails on any upward import not in the allowlist, and on any allowlist row with no matching import |
| `Tests/Architecture/layer_direction_allowlist.tsv` | The frozen upward imports, one `src_module<TAB>dst_module<TAB>kind` row each |
| `Docs/QA/architecture-review-2026-10-25/scripts/graph.py`, `layers.py` | The review's measurement scripts. The test copies their import resolution and layer ranks, so its numbers match the review's |

## The layers

Each top-level package under `tldw_chatbook/`, and each module at the package root, sits in exactly one layer.

| Rank | Layer | Members |
| --- | --- | --- |
| 0 | kernel | `Utils/`, `Third_Party/`, `Metrics/`, `state/`, `runtime_policy/`, and the root modules `__init__.py`, `config.py`, `Constants.py`, `Logging_Config.py` |
| 1 | DB | `DB/` |
| 2 | domain | Every other package (`Chat/`, `Agents/`, `LLM_Calls/`, `Library/`, `Notes/`, `Backup_Recovery/` and the rest) and every other root module (`model_capabilities.py`, `provider_registry.py`, `emergency_stop.py`, `chunking_engine_version.py`, `__main__.py`) |
| 3 | presentation | `UI/`, `Widgets/`, `Event_Handlers/`, `css/`, `Web_Server/`, and the root modules `app.py`, `app_*.py`, `cli.py` |

## The rule

- A module may import any module in its own layer or a lower one. `Chat/` importing `Agents/` (both domain) or `DB/` is fine.
- **DB may import domain model modules, but not domain services.** A model module is one named `models` or ending in `_models` (for example `Agents/agent_models.py`, `Backup_Recovery/models.py`, `Workspaces/project_index/models.py`): it holds the types a store reads and writes. Every other domain module counts as a service, and DB importing one is an upward import. A service does not become a model by being renamed `*_models`.
- Every other upward import, such as domain importing presentation or kernel importing anything above it, breaks the rule.

## What counts as an import

- Top-level imports (`top`) and imports inside a function or lambda (`lazy`) both count. A function-level import still runs, and it still couples the two layers. Moving an import into a function does not fix its direction.
- Imports under `if TYPE_CHECKING:` are out of scope. They never run.
- An import resolves to the longest prefix of its dotted name that is a module. `from tldw_chatbook.UI.Screens import chat_screen` names `tldw_chatbook.UI.Screens.chat_screen`. Relative imports resolve from the importing module's own package.
- A call to `importlib.import_module` (or to `import_module` imported from `importlib`) whose first argument is a string literal holding an absolute `tldw_chatbook.` name counts as an import of that module. Its kind is where the call sits: `top` at module level, `lazy` inside a function.
- Dynamic imports of a name that is not such a literal are out of scope: a non-literal name, a relative literal such as `import_module("..Widgets.x", __package__)`, and `__import__`. For example, `Chat/console_first_send_modules.py` imports eight presentation modules from a tuple of names, and the test does not see it.

## The allowlist

`Tests/Architecture/layer_direction_allowlist.tsv` holds one row for each distinct (source module, imported module, kind) upward import that existed when the test landed. Kind is `top` or `lazy`. Lines starting with `#` are comments.

- **A new upward (source, target, kind) fails** `test_no_new_upward_import`. The failure names the file and line. Fix the direction: move the code that needs the higher layer up into that layer, or have the higher layer pass it in (a parameter, a callback or a Protocol). If you think an import belongs, explain why in the PR and get the reviewer's agreement before adding its row. Never just append a row to make the test pass. A move that leaves a re-export shim at the old path may add that shim's row only if the same change deletes at least as many rows; the shim's row is deleted together with the shim, once its importers are retargeted.
  - Dated exception 2026-10-31 (TASK-36953.2): `Utils.Splash_Screens` -> `UI.Splash_Screens` (top) is kept without a deleted row because old import paths must keep working for user custom splash cards; the row goes when the shim is retired. The same shim (`Utils/Splash_Screens/__init__.py`) also aliases old submodule paths dynamically through a meta-path finder that builds the UI target name at runtime, which the AST guard cannot see; that finder goes with the shim too.
- **A row with no matching import fails** `test_allowlist_has_no_stale_rows`. Delete the row. The allowlist only shrinks, so every structural fix that removes an upward import also removes its row.
- Rows are per module pair and kind, not per statement. Another import of a listed (source, target, kind) passes, whether it brings in another name or imports the same target in another function. A row goes stale only when the last such import is gone.
- A row whose target is a package `__init__` covers every name that package re-exports. Two rows target `Widgets.Console`, which re-exports about 40 widgets, and one targets `tldw_api`.
- Changing an import's kind (top-level to function-level, or back) shows up twice when it was the only import of that pair and kind: the new kind fails as a new import, and the old row fails as stale.

## Census when the test landed (dev `2b296e60b3`)

The review counts each imported name, so `from x import a, b` is two of its edges. The allowlist has one row per distinct (source, target, kind).

| Upward direction | Imported names (top-level) | Allowlist rows (top / lazy) |
| --- | --- | --- |
| domain → presentation | 80 (42) | 53 (24 / 29) |
| DB → domain | 336 (169) | 79 (28 / 51) |
| kernel → DB, domain or presentation | 89 (26) | 59 (24 / 35) |

65 of the 336 DB → domain names (17 distinct imports) target model modules. The rule allows them, so they have no row; the other 271 are frozen. The review measured 335 DB → domain names at `7a5e2513c5`, and one more had landed by `2b296e60b3`. Every count on this page, and the 191 rows, match what `graph.py` and `layers.py` report at that commit. No literal `import_module` call outside presentation points upward at that commit, so counting those calls added no row.


### Dated exception 2026-11-06 — PR3411 / TASK-38944 native workflow DB receivers

The four existing lazy dependencies below retain live checks at their DB receiving boundaries under ADR-285. Approval is limited to the listed symbols and uses. The scanner tracks module pair and kind, so a new use on the same pair still requires fresh review. Delete each row when its last approved use disappears. These reasons and retirement conditions also belong in the PR explanation before any allowlist rows are added.

* `tldw_chatbook.DB.automatic_work` -> `tldw_chatbook.Chat.console_workflow_preparation` (`lazy`): only `_PLANNING_ABORT_CALLER` and `_PreparedWorkflowReply` in `abort_workflow_planning`, authenticating the original unused-close sender before `_planning_abort_source` and repeating the validator inside FULL with owned-row checks. Structural attempt data supplies no live preparation/finite-return/opaque-claim evidence; the ledger's review-boundary registration is not an abort issuer. Retire this row if the native abort path is removed or an existing independently installed original sender/validator can be authenticated at both current DB checks without per-call trust.
* `tldw_chatbook.DB.automatic_work` -> `tldw_chatbook.Chat.console_workflow_runs` (`lazy`): only `_fence_connection`, `_dispatch_identity`, `_quiescence_receiving` and `_quiescence_evidence` for the existing transaction/reservation/current/policy/dispatch/commit and resume/cleanup/quiesce/accounting paths. They preserve exact active connection and precommit checks, retained dispatch identities, callback frames and physical-return evidence against the same row/connection, including implicit ContextVar/current-task reads without an explicit cell. There is no universally installed original callback in the ledger constructor; a per-fence callback misses implicit entries and an adapter frame can invalidate a genuine fence. Retire the row only after all four uses disappear or an existing original boundary covers every entry and precommit recheck, including no-explicit-context reads and physical evidence.
* `tldw_chatbook.DB.workflow_runs` -> `tldw_chatbook.Agents.execution_capacity` (`lazy`): only `current_execution_owner` in `anchor_change_window`, independently comparing the actual execution ContextVar at the committed pre-run bind seam while preserving owner/run/claim and FULL accepted/running/committed join plus first-winner update. The continuation omits that comparison; general authorization check requires the later run ContextVar. Retire the row if anchoring ends or a preexisting independently authenticated receiver can check the actual execution ContextVar at that same pre-run point.
* `tldw_chatbook.DB.workflow_runs` -> `tldw_chatbook.Workspaces.change_review_finalization` (`lazy`): only `ChangeReviewReservation` for `open_change_window`'s exact-type check before reading the handle or starting FULL. This live lock/callback/completion/ticket owner is not workspace model data, and the DB publication receipt exists only after insertion. Native wave registration/activation or original rollback remains necessary; exact type is not issuance proof. Retire the row if the window path is removed or a genuine original reservation is independently known to the store before the first field read without introducing a second issuer or broadening the models rule.

Window-anchor qualification is bounded: `_check_continuation()` executes before the independent owner comparison and can perform ledger reads. Require the original enrolled wrong-owner no-anchor-write/no-anchor-FULL control and replacement control. An unrestricted direct-Python caller can supply a receiver whose continuation executes before later validation refuses it; retain that counterexample explicitly, without a global no-SQL or zero-foreign-invocation claim. Python source identity checks consistency/availability; wrapping is unsupported and passive in-place profiling is supported. Actual execution and physical-return proof remain separate under ADR-285. No other symbols, uses, exception rows or model-rule expansion are approved.
