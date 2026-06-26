# Retire Legacy Entrypoints Implementation Plan

## Goal

Remove stale alternate app entrypoints and the retired `Conv_Char_Window` surface so packaged/local users have one active app shell path, while preserving the current Personas route and reused CCP handler behavior.

## ADR Check

ADR required: no

ADR path: `backlog/decisions/004-personas-destination-native-workbench.md`; `backlog/decisions/007-personas-workbench-route-consolidation.md`

Reason: Existing ADRs already accept retiring legacy CCP route/window surfaces and consolidating user-facing Personas behavior under the active destination shell. This task removes dead alternate entrypoint modules and type-only references without changing storage, service contracts, or the active navigation contract.

## Steps

1. Confirm no packaged entrypoint or in-package runtime consumer depends on `tldw_chatbook.app_refactored`, `tldw_chatbook.navigation`, or `tldw_chatbook.UI.Conv_Char_Window`.
2. Add regression guards that retired modules are absent, CCP handler type-only references point at `PersonasScreen`, and the active lazy registry still resolves the `ccp` alias to `PersonasScreen`.
3. Delete the retired alternate entrypoint modules/backups and update `CCP_Modules` type-only annotations.
4. Run focused tests, import checks, and diff hygiene before marking TASK-105 done.
