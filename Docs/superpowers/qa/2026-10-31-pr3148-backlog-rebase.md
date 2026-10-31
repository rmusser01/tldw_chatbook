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
