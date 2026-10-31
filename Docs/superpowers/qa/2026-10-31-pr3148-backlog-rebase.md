# PR3148 current-dev backlog integration

Base: `57e330e735fff25515778e839f31ed85602c34f2`.
Original published head: `97ac608aba159bd8696ea50d279261867d97e16a`.
66-commit rebase completed at `5ddf28624f051dc75e1bdf457f4d4e2e4e86d7b6`; the final repair commit follows this report.

The user explicitly waived hosted CI and size limits for oldest-first backlog integration. Independent requesting-code-review, resolved threads and preserved security contracts remain required. No release is performed by this session. Existing ADR162, ADR163 and ADR228 govern this integration; no new runtime, authority or storage boundary is introduced.

## Reconciliation

Retained current Settings Tools hub, media-generation save/adoption helpers, Persona library age/tag behavior, native hook Windows refusal and current Console owners. Replayed the Plugins destination (14 total), source/review UI, acquisition and expanded-hook changes. Existing size-only moves from the original PR remain, but no further extraction was used to satisfy a ceiling. Changed owners are pinned at measured size under the dated waiver. Removed stale duplicate ratchet entries from overlapping replay history.

Settings and Console replayed imports that were already local at actual consumers were accidentally eager after conflict resolution. Removed them; all five Settings preimport/first-use/cold-overview tests now pass. Child fixtures preserve private HOME, USERPROFILE, TEMP and TMP; no profile or permission guard is patched away.

SourceService imported native saved-root cleanup flags before its normal unsupported-host refusal. The actual Windows control failed at import with missing O_DIRECTORY. Directory flags now resolve only at native storage use and explicitly require 64-bit Darwin; the real runtime owner still refuses before creating its root. There is no fallback to unsafe flags. Existing native tests that constructed macOS storage or POSIX command owners without using the gated fixtures now carry the same explicit capability markers; supported-host behavior and assertions remain intact.

## Local evidence

Receipts reside in sibling `../test-runtime/` (not tracked because they contain host-local diagnostics):

- `plugin-native-1793470173558130600`: 95 passed, 256 capability skips. Runtime owner, package capture, adapters, credentials/revocation and hook command/budget controls. Native macOS paths remain unqualified here.
- `preimport-1793470105214799500`: RED import-identity assertion plus Windows child temp-path setup failure.
- `preimport-fixed-1793470486357593500`: all 5 passed, including real cold Settings overview, first use and draft preservation.
- `unsupported-red-1793470750488991000`: focused new control failed at SourceService import before the repair.
- `plugin-ui-hooks-fixed-1793470784006633500`: 60 passed, 117 skipped, 5 reported failures. New unsupported-host import/refusal control passed. Three authority-dialog parameter cases attempted unsupported native runtime; one local-tools case attempted unsupported command hooks. They now use existing capability markers. The F8/Settings navigation test body passed but child teardown reported uncertain native workers and executor timeout; it is not counted green and is not appended to the verified-green UI census.
- `final-platform-controls-1793471257077752800`: 3 passed, 4 explicit capability skips. The source facade imports successfully, actual unsupported runtime acquisition creates no root, and the native flag boundary refuses.
- `changed-pins-1793470886391110400`: all 50 checks for changed governed modules passed.
- `rebase-artifacts.log`: diagnostic inventory verified (710 owners, 1058 TASK492, 56 TASK31551, 7772 TASK494 calls, 16 sink files); unchanged UI census1173 verified; CSS rebuilt with no tracked generated diff. Initial lint found duplicate Persona pin and deliberate exact-type checks; duplicate removed and exact-type security checks documented, targeted Ruff now passes.

All 182 changed Python files parsed. No new undefined production names. Existing test-only static findings remain: duplicate RunBudget import; pytest fixture arguments share an imported fixture name; the unchanged dev MCP workbench test contains a duplicate test function and two undefined-service lines. Those are not presented as new product regressions or silently repaired here.

## Diagnostic statement audit

Read actual changed diagnostic statements before regeneration. One unchanged reference-warning statement moves from console_chat_controller to console_plugin_context; two existing sanitized profile warnings move from mcp_workbench to mcp_profile_actions; 24 existing cleanup statements move from app_lifecycle to Utils/app_shutdown. No new user content, diagnostic sink or logging semantics were added. The generated inventory changes only these owners.

## Outstanding review and limits

The one shutdown-owner control (`shutdown-base-control-1793471166497534400`) overlaid only app_lifecycle.py and Utils/app_shutdown.py from exact dev57e330, then restored both byte-for-byte in finally. The navigation body again passed and the same native-worker/config-retirement and executor-timeout teardown errors occurred. This isolates the extraction from the cause; it is not a full-base result and does not qualify teardown as green. No further mounted rerun is claimed. Final independent full-PR review is pending. There is no claim of full-suite success, current hosted-CI success, native macOS execution, real keychain interoperability, third-party host parity or release qualification. Historical task/QA outcomes describe their recorded revisions and are not substituted for current-head evidence.


## Independent review repair: retained variable recovery

Review of `8c0246b2d588cdb1d985303d21353ebffc15a647` found that normal
variable revocation left its authenticated mapping intact, but the next recovery
validated that retained mapping against the now-revoked credential. Recovery
cleared publication and refused, so restart lost the whole catalog and every
replacement commit failed at its initial recovery boundary.

Recovery now reconstructs the closed variable mapping from its retained
component/declaration and original reference handle, checking the complete row
including the original value-free credential metadata digest. It does not need a
currently available external value to retain that reviewed selection. Only this
variable recovery branch changes: review, new configuration commit, admission and
launch still use the actual credential owner, and non-variable mappings retain
their existing validation. ADR162's typed-value paragraph clarifies this existing
readiness/recovery contract; no new ADR, schema, permission owner or platform
support is introduced. The authoring guide now limits substitution to argv and
environment and describes cwd as a literal captured package-relative directory.

Exact source copies, commands, output and SHA-256 manifests are retained in
sibling directories (host-local evidence, not tracked):

- `../recovery-review-red-1793472339860312500`: 2 expected failures (revoked and
  missing references returned recovery_required), 6 passes (active reference and
  changed-binding controls), 4 Darwin-only skips. This ran before production edits.
- `../recovery-review-green-1793472483274362700`: 115 passed, 38 Darwin-only skips,
  1 Pydantic warning in 177.05s across `test_variable_recovery.py`,
  `test_declared_variables.py` and `test_credential_bindings.py`. The runner also
  printed the existing pytest-asyncio unset-loop-scope deprecation warning.
  Source copies remain byte-identical to the committed Python changes.

The ten portable recovery controls execute the actual recovery body, mapping
validation and credential owner with explicit persistence/retained-inspection
boundary doubles. They establish revoked/missing recovery separation, preserved
live capture/validation/resolution refusal, owner-issued metadata-digest
compatibility, and changed binding/declaration refusal. They do not qualify native
persistence, secure-marker authentication or native restart. Four added native
cases cover recover -> reopen -> usable independent sibling + refused hook ->
replacement or unset commit, followed by real hook execution for replacement.
Unset uses the supported complete-set review with a valid COUNT mapping retained
while TOKEN is removed. These four and the existing revoked-before-commit native
control are skipped on this Windows host, not claimed green. Production Darwin
qualification/refusal gates are unchanged.

Ruff formatting passes for all four Python files. Targeted lint has no new
findings; variables.text retains its pre-existing exact-bool E721 diagnostic
(the explicit type check is unchanged), and the other three files pass default
Ruff checks. All touched Python parses and git diff --check passes. Black is not
installed in the shared interpreter; installed Ruff supplied formatting without
adding dependencies. No governed production module crosses a size threshold;
no size pin changes were needed under the dated user waiver.

The actual diagnostic statement comparison against the reviewed head found zero
calls in both modified production modules before and after. Inventory rebuild
matches exactly: 710 owners, 1058 TASK492, 56 TASK31551, 7772 TASK494 calls and 16
sink files. No inventory regeneration or changed logging statement was needed.
Task32695 remains In Progress for independent review/publication. No full suite,
hosted CI, native Darwin execution, keychain interoperability, push or merge is
claimed by this repair.
