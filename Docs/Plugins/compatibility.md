# Managed plugin compatibility evidence

Qualification is specific to source, package, adapter, platform and configuration. This document records the observed boundaries; a workflow or successful parser cannot establish an unexecuted runtime.

## Identity and environment

Repository baseline: `62b6f59da8a1d3fcc3d9ef56de66e6b6c4284138` with the Task25 reviewed changes. Behavioral source freeze: `cde126bb38cb10e5f58dd775cb5772c02ccc3e26b1daf26a879d5f526340fc0f` (29 eligible source/test/document files before final documentation). Product identifiers below bind these observations even when the surrounding report is unavailable.

Local execution: macOS 26.5.2, Darwin 25.5.0 arm64; CPython 3.12.11; Textual 8.2.8; Pydantic 2.12.5; httpx 0.28.1; portalocker 3.2.0; jsonschema 4.26.0; referencing 0.37.0; pytest 8.4.2; pytest-asyncio 1.2.0. Private HOME/config/data/cache and null keyring precede product imports; controlled peers use current-interpreter children and checkout provenance. External network access is refused. Darwin native boot/root checks execute unchanged. The protected marker fixture is a double and does not qualify OS keychain or real reboot recovery.

| Source identity | SHA-256 |
|---|---|
| `tldw_chatbook/Plugins/service.py` | `7bfc3ab59b88ee573ef1f06291f69da2f1557e3ca43047ee92e19974dc6ecfdf` |
| `tldw_chatbook/UI/Console_Modules/session.py` | `23269c3c58742750fe882bd6861a8238039725ce9aa77aa681d85d805f0f5ffc` |
| `tldw_chatbook/Widgets/plugin_review_modal.py` | `4d68be60b1f0d0f9c297acf7d4859be66a054098e74e8c4f1aa418872d8c23ba` |
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

Concrete regression entries include `Tests/Plugins/test_plugin_end_to_end.py::test_install_use_disable_uses_production_paths[https_git_catalog]`, `::test_skill_lookup_keeps_component_kind` (six parameter cases), `Tests/UI/test_plugin_review.py::test_console_ui_capture_uses_owning_plugin_workspace`, `Tests/Plugins/test_data_cleanup.py`, and the explicit file list in the workflow. The full acceptance mapping also covers the main specification’s companion owned-hook integration requirement; native hook composition, initialization, required post-event gates and scoped shutdown are exercised through their owning Console/runtime paths.

Static checks compile all changed Python. Eight changed Python files pass Ruff and Black; the existing pacing fixture retains one BASE-identical Ruff finding and formatting debt, and session.py retains 40 BASE-identical findings and formatting debt. No new findings were introduced by these edits. This is a bounded baseline comparison, not a claim that every file is lint-clean.
