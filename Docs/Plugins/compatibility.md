# Managed plugin compatibility evidence

Qualification is specific to source, package, adapter, platform and configuration. This document records the observed boundaries; a workflow or successful parser cannot establish an unexecuted runtime.

## Identity and environment

Repository baseline: `62b6f59da8a1d3fcc3d9ef56de66e6b6c4284138` with the Task25 reviewed changes. Initial behavioral source freeze: `cde126bb38cb10e5f58dd775cb5772c02ccc3e26b1daf26a879d5f526340fc0f` (29 eligible source/test/document files before final documentation). Fix round 1 starts at `2c95a31404636efdd1932d7f726844ebe594a102`; its final 33-file behavioral freeze is `938104bfa03a3fdbe9a71515e2b8bc870cb1d0d904c20d09c994a51a3eb4d982`. Product identifiers below bind these observations even when the surrounding report is unavailable.

Local execution: macOS 26.5.2, Darwin 25.5.0 arm64; CPython 3.12.11; Textual 8.2.8; Pydantic 2.12.5; httpx 0.28.1; portalocker 3.2.0; jsonschema 4.26.0; referencing 0.37.0; pytest 8.4.2; pytest-asyncio 1.2.0. Private HOME/config/data/cache and null keyring precede product imports; controlled peers use current-interpreter children and checkout provenance. External network access is refused. Darwin native boot/root checks execute unchanged. The protected marker fixture is a double and does not qualify OS keychain or real reboot recovery.

| Source identity | SHA-256 |
|---|---|
| `tldw_chatbook/Plugins/service.py` | `7bfc3ab59b88ee573ef1f06291f69da2f1557e3ca43047ee92e19974dc6ecfdf` |
| `tldw_chatbook/UI/Console_Modules/session.py` | `23269c3c58742750fe882bd6861a8238039725ce9aa77aa681d85d805f0f5ffc` |
| `tldw_chatbook/Widgets/plugin_review_modal.py` | `a42e7c81c2daa670f81dcb3f370361ad4f800dde9c8d8a2377f69d2dc492c933` |
| `tldw_chatbook/Chat/console_chat_controller.py` | `53c17ea23ffe4363f29c27b262bfff23489de9ce845540f5622ec47b54655c1c` |
| `Tests/Plugins/test_owned_hook_acceptance.py` | `334358b697ad97e410931ceface2fa142e363996bc3e2ca214818e3c56a82aa4` |
| `tldw_chatbook/Plugins/adapters/cursor.py` | `2d883313bfbf11a7fa55fea3ec0b18aa46bbe3aad591adfbedf2d802401e977f` |
| `tldw_chatbook/Plugins/adapters/codex.py` | `3c9292697426c7d1398b08054638bc02282af3b82cde19dbb60717fd8da6ca3c` |
| `tldw_chatbook/Plugins/adapters/portable.py` | `3c756e45f61222d9e4729ecf83adcfb3aef1eea73e1162f55e23006529ac6668` |
| `tldw_chatbook/Plugins/adapters/vendor_hooks.py` | `a2b527c7704eb098ac717e2e683a5710532f9f5a60a245a48f9877e9be230115` |
| `Tests/Plugins/fixtures/native_review/plugin.json` | `881f50f198a76bd327223f8acc6c35d77163016d140eedd23c2a222d80a29d8b` |
| `Tests/Plugins/fixtures/native_review/skills/review/SKILL.md` | `f406e355bc6de3306e368706afea3af583685610c2af7ba3216662a80ba964bb` |

## Evidence levels and pinned packages

`parsed` means independently expected inventory matched inert source. `connection_exercised` means the owning client connected/discovered against the named controlled peer. `behavior_exercised` means the described production operation and success/refusal controls ran. `original_host_compared` requires the pinned package to run in its original host; **no original-host comparison is claimed**.

| Package / adapter | Pinned identity | Observed level and limits |
|---|---|---|
| Authored native review example; portable 1.0.0, Chatbook extension 1 | Original AGPL-3.0-or-later fixture; expected `skill:review`; exact two hashes above | Local install/trust/workspace-only enable, attributed real Console input and persisted controlled response, update/rollback/disable/uninstall. Explicit empty tool ceiling; this example does not qualify MCP permission. Actual controlled HTTPS Git catalog acquisition, exact commit selection, install/Console/update/rollback/disable/uninstall passed (12.36s; raw SHA `10840397d08495e8fff65481a279c61ccf675ba4b659991dd14372fa770810ab`). No external Git host is claimed. |
| Cursor teaching; `chatbook-cursor/1` | `cursor/plugins` teaching at `f5bdd6826fd0a0d9cbc4347134c3a74a200b9d9d`; manifest SHA `a1c1073b4c3d9b64370607d194b07dc336ba2ebaf2aadaec153d2575a93f8523` | `parsed`: two independently expected skills. Four copied files match pinned upstream blobs; MIT notice included. Selected assets/README omitted; no execution of upstream skill bodies or original Cursor host comparison. |
| Codex Linear; `chatbook-codex/1` | `openai/plugins` plugins/linear at `1dc195897af4161d039b80d8471ec0a10c9bbc89`; manifest SHA `04c0e2af139b23c4bff3388cd66fbe40095972df203e6926394afbb2246e9354` | `parsed`: MCP `linear` and app declaration only. Three metadata files match upstream blobs; manifest declares MIT but package listing has no LICENSE. OAuth/app declarations create no grants or connection evidence. |
| Portable arch | `github/awesome-copilot` plugins/arch at `7568a482ce2df38f8965ab5336a3220db796a4ba`; manifest SHA `6e69360d3f446205d4e579cf09c42103dbfebf8bb8fa2d83595efe0b1aa4b4b1` | `parsed`: empty portable skill/MCP inventory; absent extension path is not followed outside package. Manifest blob verified; same-commit MIT license retrieved but not independently matched to root tree blob. |
| Synthetic vendor hooks | `cursor/1`, `codex/1`, current adapter hashes above | `behavior_exercised` only for named native adaptations: actual command payload/denial, timeout/output bounds, final guards and ordinary permission. Unsupported shell/regex/required guards remain refused. This is not runtime equivalence to vendor hosts. |
| Native owned MCP | Protocol `2025-03-26`; test-generated stdlib stdio and loopback Streamable HTTP peers | `connection_exercised` and `behavior_exercised`: real client initialization/catalog/calls, typed errors/results, normal permission, cancellation and custody. Generic OAuth, arbitrary external servers and remote containment remain unqualified. |

The [fixture provenance](../../Tests/Plugins/fixtures/interop/upstream/provenance.json) records complete selected bytes/licenses. Native authored provenance is [separate](../../Tests/Plugins/fixtures/PROVENANCE.md).

## Concrete acceptance observations

Counts below describe separate overlapping selections and must not be summed into a full-suite result.

| Boundary | Result / evidence identity | Limit |
|---|---|---|
| Six required qualification files | 134 ordinary passes; 42 exact native/HTTP cases passed in approved host context (raw SHA `4ef9fd63821536b4119822ddbfd5f5dd47125c140afc2dce68c694802a95762d`); one unchanged public cleaned-status control carried separately | Ordinary 43 failures and six teardown aftermath errors preserved as precondition failures. Later changed E2E tests have their own results. |
| Explicit remaining 32 acceptance files | 924 ordinary passes; exact 129 precondition cases host run gave 128 passes and one dual-fixture marker collision (raw SHA `1cd00d18d9033dbaf8c16966f378f017ed1a3579092fdb1534736eaca55bc476`) | Shared fixture received a distinct owned marker; 27 selected affected consumers then passed (raw SHA `3ae6cc2e8d1466138a3c17e3491331903a18a32927366a79bd84ab5ee925b911`). These 27 are not every consumer of that fixture. |
| Reviewed saved-data absence | Public creation/deletion with current owner-returned reference; stale prior ref, recreated leaf, wrong boot, recovery and pagination controls; two mounted sizes passed | `cleaned_absent` is an authenticated point-in-time display only. Recovery/cleanup pending take priority; no future absence or cleanup permission is implied. |
| Action fields and first Install form | Fresh producer 80×24/120×35 plus keyboard/draft controls: six passed (raw SHA `680227911a56af9856328d34dcaf155f72a3d147642ec96cdb32fab1921eca7b`) | Whole live lifecycle result is recorded separately below; initial live omission and subsequent harness failures remain preserved. |
| Console workspace and typed component lookup | Owning-session A while visible B regression plus scope/tool/generation/no-consent neighbors: 9 passed. Same-name skill/MCP and skill/command execute/read/definition controls plus currentness/admission/Console neighbors: 19 passed (raw SHA `5862e93339ff203eb85321c97e0659bcf57795cc2dab38b61348e2feb10e0058`). | Skill aliases resolve skill bodies; typed non-skill IDs are refused by skill entry points. Native generic typed identities and fences remain intact. These are builder/service and Console controls; no mounted claim for the builder test. |
| Resource owner teardown | Ten session fixture consumers passed with no same-DB descriptors. Thirty demonstrated fixture consumers partition into 25 carried passes and five renewed mounted passes; latest five: 5 passed in 92.94s (raw SHA `ab730d600aa9c9cb0f08c3349c27f147a879dd59140b81e3539c28f898bbbfaf`) | Supported runtime/app drains, owner DB barriers and monitor settlement remove named rig/app DB and instance-lock handles. Final FD14, Python MainThread only, no retained apps after unchanged GC; observational helper pipes are separate. Historical H5/H6 +456/+418 growth is not exhaustively attributed. |
| Startup/preimport | 84 passes; two polluted sleep-spy failures corrected with local app clock; exact two plus same-spy positive control: 3 passed (raw SHA `cd04e7fbe2b7bbe465a1a911a75561c0e5010f24e0b564343ed007b566667e93`) | One deselection; optional-pydub cold warning still fails current and independently verified original-start source under the same dependencies. Not green. Final source11 renewal passed both exact ratchets in 11.69s (raw SHA `55926909a5870da399c1e9c06e8956b4b319117d3936d45f00b3d675412c8f53`), with two headroom warnings: preimport 499/500 modules and 157 LOC spare; UI-ready 973/973 modules, zero spare, snapshot drift +17/-17. Caps and snapshots remain unchanged. The same run logs the inherited optional PyAudio-unavailable warning separately from the two pytest headroom warnings. |
| Browse neighbors | 295 passes, two actual native failures subsequently passed named host checks | One original import-preview timing failure later passed an observer run; cause unclassified, not claimed fixed. |
| Whole keyboard lifecycle | Both 80×24 and 120×35 completed; 80×24 raw SHA `73ad2921eb397da9222be33cf5c791212c15e8c129e3a9336f83db1d6138129d`; 120×35 raw SHA `497cc1921a2c9c775793372fe89b666f92f4c8fc44960f07dbedb0e2a79c9c85` | Controlled provider and owned stdio child; actual normal permission. Manual Refresh is required after root/profile creation. |
| Linux / Windows / real OS keychain / reboot | Unqualified locally | Three-platform workflow is prepared, not executed evidence. Windows command hooks intentionally refuse launch; simulated boot changes do not establish reboot behavior. |

## Real-terminal lifecycle

Both real-terminal keyboard journeys completed: 80×24 and 120×35. Each installed, trusted and enabled a generated local skill/MCP package, configured/connected/mapped its owned stdio peer, typed the actual public skill invocation, verified attributed skill content in the actual provider payload, discovered and loaded the tool from actual results, used Alt+A and normal Approve once, and invoked exactly one controlled tool. Exact Console completion preceded reviewed update/disable/uninstall; captured idle connection custody then completed and live custody emptied. Exact-root deletion followed by Refresh displayed “Cleaned — saved data removed”. Each phase confirmed both exact child PID/creation-time identities absent after app exit. Private roots were 0700, config 0600, external network attempts zero and actual sizes asserted. The marker backend remains an isolated double; no OS keychain/reboot qualification follows.

Driver SHA-256: `6364732637f307d01996c5d950ffc4f42a1c49a58d38ed27d662f969cdd933b9`. The generated package declares portable schema 1.0.0 and Chatbook extension 1, a `skill:review` and an `mcp:review` sharing the installation-qualified alias. This is separate from the authored example with an empty tool ceiling. Controlled provider responses exercise production Console routing and ordinary approval; they do not assess model quality.

## Reproduction and remaining limits

The [workflow](../../.github/workflows/plugin-qualification.yml) runs explicit feature/regression files with a private profile and network guard established before product imports, then preserves platform/package identity and JUnit output. Review actual platform assertions, failures and skips before raising its evidence level. Local verification used targeted selections only; no full-repository suite was requested or run.

Unknown remote completion retains request/root/revision custody even after a waiter or connection closes. Credential refresh does not authorize replay. Successful controlled peers do not establish arbitrary server session independence. Adapter parsing does not establish original-host behavior. See [authoring](authoring.md) and [operations](operations.md) for user-facing contracts.

Concrete regression entries include `Tests/Plugins/test_plugin_end_to_end.py::test_install_use_disable_uses_production_paths[https_git_catalog]`, `::test_skill_lookup_keeps_component_kind` (six parameter cases), `Tests/UI/test_plugin_review.py::test_console_ui_capture_uses_owning_plugin_workspace`, `Tests/Plugins/test_data_cleanup.py`, and the explicit file list in the workflow. The installed-hook evidence below closes the companion integration gap identified in the initial acceptance mapping; standalone kernel tests alone did not establish that registration boundary.

## Fix round 1: focus, owner cleanup and installed hooks

The review found that a focused Review/Refresh button could be below the visible viewport after its body grew. The widget now requests layout before scheduling its existing after-refresh scroll; it preserves the same focused control and draft/review state. Four mounted cases at 80×24/120×35 first failed their visible-paint assertion. Both Refresh cases then passed; after resolving the Review layout timing, both Review cases passed (raw SHA `45fad9949245c4f3f36a1fe17f77a7ec50812e81a302e2ad83549ac91ff3ccc4`). Six selected UI neighbors passed across ordinary and exact native-context runs. Two focused real-terminal renewals passed at those sizes (raw SHA `7c4f782d550bf88f769d5ee93354bb816e9078ec9048685a453bbfb26700a5a4` and `2d35554fb1747f21e966d0438c1c559caf570a6cc20161072a9d331ae0c9f54a`). Actual reviewed APIs prepared the state; keyboard Tab/Enter performed Review and Refresh. Native paint, clipping, retained focus and the cleaned-status tail were observed; selected contrast minimum was 5.82. These are focus renewals, not another full Console/MCP lifecycle. Derived images retain a missing-font-glyph limitation.

Eight duplicated test finalizers now share narrowly owned cleanup in `Tests/Chat/hooks_v2_owned_rig.py`; runtime disposal precedes controller/runs/DB settlement, with a separate mounted-app extension. Release/cancel/gather and process assertions remain at their original sites. All 40 selected consumers passed in 166.35s (raw SHA `011b17aa2a1edb42eed2da1c3d9fd0edd9d202a5a39330d55658129d31ac8a1a`): empty registered DB handles, no retained apps, Python MainThread only and final FD14. This does not retrospectively explain all historical aggregate growth.

All 59 collected cases in `Tests/Plugins/test_owned_hook_acceptance.py` have passing results across targeted phases. This is a partition, not a single 59-case run. Packages are generated native portable 1.0.0 / Chatbook extension 1 fixtures, installed/trusted/workspace-selected through the actual service; command peers use the current Python interpreter. MCP peers speak protocol `2025-03-26`. The owned stdio fixture derives only the required tools capability from its controlled peer before installation; the original shared peer is unchanged. HTTP uses an actual local asyncio server and the development-loopback owned profile. Ordinary HTTP setup failed bind errno1; only its eight named cases ran in approved host context.

| Installed boundary and actual test function | Passing cases and observations |
|---|---|
| `test_installed_required_effect_admission` | 10 controller-admission/provider controls: empty/valid context, required empty/whitespace/oversized output, optional/dependency failure, master-off, disabled and invalid required handlers. This direct-controller group does not claim Runtime Stop publication. |
| `test_installed_provisional_mcp_initialization` | Four actual Runtime root-lifecycle cases: approved, disconnected, denied, cancelled; accepted/terminal observer and legacy Stop positive, negative dependent publication/context refusal. The fifth cyclic case proves pre-admission exclusion and no authored peer launch, with independent valid rendering; it is not an unstarted typed-call observer. |
| `test_installed_hook_aggregate_admission_and_teardown` | Actual installed registrations share execution/ticket limits and bounded optional queues; both runtime/app refusals assert `delivery_capacity`, quiet work progresses and required guards still refuse/allow during queue saturation. Owned shutdown settles processes/custody. |
| `test_installed_mcp_hook_normalization` | 8 modes × stdio/HTTP: structured, text, mirrored, empty, isError, conflicting mirror, invalid structured and empty required context. Each uses an ordinary non-hook positive on the same owned profile/peer/connection plus actual hook decoding/permission/call evidence. HTTP8 raw SHA `400a7a55be6beb7c82cd34afec2e66862c9b353192d854355ea346ef666b6aea`: 8 passed in 5.65s. |
| `test_installed_final_arguments_reach_normal_permission_and_dispatch` | 4 controls: allowed, final guard denied, invalid schema and changed-after-review. Installed transformations feed final arguments into real normal review/dispatch and durable tool outcomes; actual execution deltas distinguish permission from execution. |
| `test_installed_post_tool_checkpoint`, `test_installed_post_child_checkpoint`, `test_installed_post_compaction_checkpoint` | 12 success/failure/stale controls after real durable tool result, inline/fleet child outcome or SQLite memory commit. Pending required hooks fence subsequent admission/root settlement; stale/failure preserve the committed outcome while preventing late context from authorizing work. |
| `test_installed_compaction_authority_fence`, `test_installed_compaction_operation_cleanup`, `test_installed_standalone_compact_now_releases_admission`, `test_installed_owned_mcp_post_compaction`, `test_runtime_no_hook_automatic_compaction_preserves_submission` | 11 controls: nonce/disabled/selection authority, four operation cleanup paths, installed/no-hook standalone availability, actual owned-MCP PostCompact with normal approval and prospective scope, and no-hook automatic compaction preserving its exact existing reservation. Two existing captured-automatic SQLite controls also passed separately. |

Installed compaction exposed two caller defects. Comparing fresh captures treated the ephemeral `plugin_run_id` as authority, so unchanged authority looked stale; the private comparison now omits only that nonce from a copied comparison value. Actual admission configuration remains unchanged. Manual/standalone compaction now passes captured configuration into real hook preparation and owns its reservation/state cleanup through success, failure and cancellation. Captured automatic ownership is preserved, including a same-session `(None, None, session)` no-hook placeholder. The actual no-hook automatic boundary first reported BLOCKED with no committed memory; the corrected case then proved committed SQLite memory and provider completion while preserving exact placeholder identity. No public API, runtime implementation or admission default changed.

Early harness assumptions and native prerequisites are preserved separately from semantic failures. The 59-case partition retains exact argv/raw and before/after source hashes. Four early iteration full test-file copies were not captured; those origins cannot claim whole-file byte replay. Their 14 uniquely dependent successful cases were renewed on final frozen Python: 14 passed in 29.81s (raw SHA `bd041d8e89451636c60b44be84eafd432efbb8ec83236d500ae4a19d49a39add`). Every case in the final partition now names a passing phase with an exact preserved test-file copy; parameter-specific source applicability remains separate from that byte identity. Standalone decoder-edge/fairness kernels remain separate evidence; neither source equality nor shared-engine reuse substitutes for the installed controls above.

Both final current-source ratchets passed in 10.93s (raw SHA `1ac0c2511a1023aeb3e94b5fad48d877b33f40740c97c073b161bdf26374fd98`): preimport 499/500 modules, 157 LOC spare; UI-ready 973/973, zero spare, drift +17/-17. The two headroom warnings and inherited optional audio warnings remain limitations; no caps, snapshots or dependencies changed.

Static checks compile all 17 eligible Python files. Twelve pass Ruff and Black. Five inherited files retain identical Ruff diagnostics and offending source lines against captured baseline: controller 191, session 40, private SQLite inventory 14, startup 1 and pacing 1; their existing formatting debt remains. This is a bounded baseline comparison, not a lint-clean repository claim. The workflow now explicitly selects the new installed-hook test file; no workflow/platform run is claimed.

The [testing-evidence lesson](../../backlog/docs/lessons-testing-evidence.md) records the actual lost-snapshot incident and why the bounded 14-case renewal was required.

The final two installed standalone/owned-MCP controls also assert that the first real input clears the handoff and a following real input omits the old context, with no command-hook witness or owned-MCP audit replay: 2 passed in 10.97s (raw SHA `a425e79f8b7576b76c6e0e82e6b247e51be1615a79a022011f117fb958318a03`). Only those two affected cases were renewed.


## Fix round 2: platform-aware command-hook selection

The new installed-hook file is selected by the three-platform workflow, but Windows deliberately returns `unsupported_platform` before command-hook root access or launch. Command-success expectations and command-output negatives cannot qualify that platform. Granular standard pytest `skipif("sys.platform == 'win32'")` marks now select the following **30 cases as skipped on Windows**, with an explicit unsupported-command reason. A skip is unexercised behavior, not a Windows pass. The original 59 case IDs and bodies are unchanged; one installed refusal control brings collection to 60.

All function names below are in `Tests/Plugins/test_owned_hook_acceptance.py`; each bracket lists the exact affected parameter IDs.

| Windows command-dependent skips | Cases |
|---|---:|
| `test_installed_required_effect_admission[empty_success, valid_context, empty_required_context, whitespace_context, oversized_context, optional_failure, dependency_failure, disabled_required]` | 8 |
| `test_installed_hook_aggregate_admission_and_teardown` | 1 |
| `test_installed_post_tool_checkpoint[success, failure, stale]` | 3 |
| `test_installed_post_child_checkpoint[success-inline, success-fleet, failure-inline, failure-fleet, stale-inline, stale-fleet]` | 6 |
| `test_installed_post_compaction_checkpoint[success, failure, stale]` | 3 |
| `test_installed_final_arguments_reach_normal_permission_and_dispatch[allowed, final_guard_denied, invalid_schema, changed_after_review]` | 4 |
| `test_installed_compaction_authority_fence[nonce]` | 1 |
| `test_installed_compaction_operation_cleanup[initializer_failure, cancelled_initialization, cancelled_post]` | 3 |
| `test_installed_standalone_compact_now_releases_admission[installed]` | 1 |

**30 cases remain selected:** required `master_off`/`invalid_required` (2); all typed owned-MCP modes over stdio/HTTP (16); all provisional MCP modes including cycle exclusion (5); compaction authority `disabled`/`selection_changed` (2); no-hook cleanup/standalone/automatic (3); owned-MCP PostCompact (1); and new `test_installed_windows_command_refuses_before_root_access_or_launch` (1). Markdown command rendering in the cyclic control is not a command-hook process. Authority disabled/selection-change cases stop before commit and assert no command marker. Owned-MCP and no-hook cases do not depend on command-hook support.

The new refusal case uses actual installed/trusted/selected NativeHookSet ownership and the real executor. It changes only that executor module's local `sys` reference to a namespace with `platform="win32"`, observes/delegates the original process-owner reservation, and asserts exact required `unsupported_platform`, no reservation/launch marker/process, zero budget slots and no live plugin custody. It closes the engine through the existing fixture ownership contract. This is local platform-branch evidence, not Windows host execution.

Pre-edit collection found 59 cases with all 30 required Windows marks missing. Post-edit collection found exactly 60, with original IDs retained and all original function bodies AST-identical. Evaluating only the recorded marker strings in isolated namespaces gives Windows30skip/30retain and Darwin/Linux0skip; shared `sys.platform` and product ownership are never simulated during collection. Five targeted ordinary checks passed in 15.86s: new installed refusal, required master-off/invalid constraints and compaction authority disabled/selection-change (raw SHA `ef69d3b6851539b73e7d7777d523b6c794a94adb0cf684d06aa55fd63e192f47`). The other local behavior results carry their unchanged bodies and explicit source applicability; no 60-case, resource, live or ratchet sweep was repeated.

No workflow edit or Windows execution is claimed. Pre-existing selected Windows issues remain for whole-branch triage: `Tests/Agents/test_hooks_v2_execution.py::test_windows_command_refuses_before_root_access_or_launch` restores the real platform and then requires successful command execution, which cannot succeed on Windows. That earlier selection was not changed by this correction. The full workflow is therefore **not claimed Windows-green**; remaining earlier platform selections require their own disposition.


## Final combined repair: operation identity, custody and declared values

This section records the single combined I1–I8 repair after the whole-branch
review. Earlier failed and partial phases above remain historical evidence.
All selected executions used private pre-import profiles, null test keyring and
external-network refusal, with immutable whole-source/test before and after
copies. Native host exceptions were limited to named Darwin boot-identity cases;
no new loopback exception or platform-support claim follows.

| Boundary | Actual final-wave evidence |
|---|---|
| Cursor managed MCP guard (I1) | Installed Cursor `beforeMCPExecution` guard sees the actual owned `mcp.call`; denial precedes normal permission/dispatch, with independent non-MCP control. Host phase11 passed. |
| Same-run cancellation (I2) | Two actual owned connections in one run each settle after disable or revision drain; another workspace remains usable. Both phase11 cases passed. |
| Retained direct child (I3) | Phase04 reproduced false settlement; phase05 passed with the original live child, failed cleanup/retry, pending public data deletion, then actual termination/settlement and unlink. |
| Common tool operation (I4) | Actual LocalToolProvider read/write classification reaches PreToolUse and both post outcomes; missing operation refuses controlling match. Phase13 passed. Classification is part of exact definition identity. |
| Protected declared values (I5) | Installed unset write guard denies writes while normal read approval/dispatch succeeds; typed configured value permits the write, revoked/missing/wrong-type refs refuse, and an unused declaration leaves siblings usable. Required initialization and dependency-only sibling controls passed. Child and sealed-resume controls preserve nonmatch behavior without advertising an executable guard. Phase30 additionally passed the optional-by-flag guard and actual declaration/reference/scope/selection currentness transitions. |
| Exact command roots (I6) | Shared and two named-workspace roots are distinct actual grants; missing root refuses. Real unscoped scratch remains leased while its hook child lives after chat closure, rejects new use and disappears only after terminal cleanup. Phases08/10 passed these controls. |
| Executable lookup (I7) | Native-hook and owned-MCP bare executable lookup uses trusted host PATH despite a child PATH shadow. Explicit package executable controls also passed. Phases08/10/11. |
| Platform selection (I8) | The workflow's exact 21-file selection collected 687 actual nodes on Darwin: 235 use managed-runtime fixture gates, 219 have skip declarations, and three native unsupported-platform controls skip here. Refusal and supported-command positive branches are separate. This is collection evidence, not 687 executed successes. |

The new value flow passed mounted keyboard checks at 80×24 and 120×35. A separate
real 80×24 TldwCli PTY journey typed a masked value, saved its protected reference,
reviewed and confirmed the mapping, then used that same installed hook with actual
LocalToolProvider permission/dispatch. It was not a fresh Console model turn and
used an isolated protected-backend double, not the OS keychain. Raw phase18 SHA
`cd339c36c010c18e21a5779dcf1fbbd394ad30d903e64961f47ef026469de57e`. Invalid-input/navigation cleanup passed phase17; completion updates
retained metadata without dismissing another screen. Phase22 passed an actual
configuration round-trip preserving existing connection and model references
while adding a value; metadata capture never connected to its HTTPS fixture URL.

The two unchanged startup ratchets passed in phase29: preimport 499/500 modules,
378,583/378,740 LOC (157 spare), and UI-ready 973/973 modules with +17/-17 drift.
Their two headroom warnings remain. Raw SHA `ca071bf16b75482b9bd5068822a922caf53867a3c2ecbc4591a01908ac64bc51`. No caps, pinned snapshots
or dependencies changed. Static checks cover the 47 changed Python files with
whole-file checks for new/clean files and exact FIX_BASE diagnostic/source-line
parity plus changed-range formatting for eight inherited files (233 diagnostics).

Phase15 was interrupted and has no reconstructed result/after snapshot; its
separately labeled manual-after evidence and exact-child-absence inspection remain.
Phase26 failed before launching a child while freezing duplicate path arguments;
its partial before tree remains. Other failed phases retain their real outcomes,
including harness failures and the navigation/mapping defects found and corrected.
Later passing nodes qualify their named boundary only. Formatting and local
source applicability are recorded separately from original byte identity.

Native Linux/Windows runtime execution, real OS keychain and reboot behavior remain
unqualified locally. The updated workflow retains actual collected-node capability
metadata and JUnit outcomes for future platform runs; it was not dispatched here.
No full-suite or whole-workflow execution was requested or run in this final wave.
