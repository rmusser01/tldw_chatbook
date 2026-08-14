# TASK-15743 Current-Dev Diagnostic Reconciliation Design

## Status

Ready for independent review on 2026-08-14.

## Problem

The governed production-diagnostic inventory is red on current `dev` before the
rebased TASK-3070.2 and TASK-16001 branches are merged. Stacking those branches
adds the reviewed Console image-controller owner and moves existing calls, so a
manifest-only refresh would silently bless both current-dev drift and unsafe
new persistent diagnostics.

The reviewed stacked delta contains:

- 31 unsafe diagnostics across 10 production files;
- reviewed-safe fixed or bounded-metadata diagnostics that require no source
  edit;
- 32 reviewed deletions from extracted provider wrappers;
- no persistent-sink topology change.

## Exact repair inventory

The following 31 calls are the complete unsafe population. `exception_type`
means the single AST field expression `type(exc).__name__`; every other field
is forbidden. Labels are unique within their owner so the existing registry
can fail closed on zero or multiple matches.

| Owner | Current unsafe identity | Required surviving label | Allowed fields |
| --- | --- | --- | --- |
| `Agents/agent_service.py` | malformed `autowake_enabled` logs key, value, and default | `autowake_enabled is not boolean; using default` | none |
| `Agents/agent_service.py` | `on_child_settled consumer raised` logs child run id and traceback | `on_child_settled consumer raised` | `exception_type` |
| `Character_Chat/Character_Chat_Lib.py` | malformed `usage_json` logs conversation id | `Skipping malformed usage_json on message export` | none |
| `Chat/console_agent_bridge.py` | `fleet drain consumer` logs callback name, conversation id, and traceback | `fleet drain consumer raised` | `exception_type` |
| `Chat/console_chat_controller.py` | `fleet unsettled check failed` captures traceback | `fleet unsettled check failed for a session; treated as idle` | `exception_type` |
| `Chat/console_chat_controller.py` | `has_unsettled_children raised` captures traceback | `has_unsettled_children raised; recording re-attach source anyway` | `exception_type` |
| `Chat/console_chat_controller.py` | `fleet usage re-attach failed` logs conversation id and traceback | `fleet usage re-attach failed` | `exception_type` |
| `Chat/console_fleet_attention.py` | `fleet unseen revision bump failed` captures traceback | same fixed label | `exception_type` |
| `Chat/console_fleet_attention.py` | `fleet unseen mark listing failed` captures traceback | same fixed label | `exception_type` |
| `Chat/console_fleet_attention.py` | `fleet unseen mark clear failed` logs conversation id and traceback | `fleet unseen mark clear failed` | `exception_type` |
| `Chat/console_fleet_attention.py` | `fleet unseen mark write failed` logs conversation id and traceback | `fleet unseen mark write failed` | `exception_type` |
| `Chat/console_fleet_attention.py` | `fleet completion announce failed` logs conversation id and traceback | `fleet completion announce failed` | `exception_type` |
| `Chat/console_fleet_attention.py` | `fleet toast title lookup failed` captures traceback | same fixed label | `exception_type` |
| `Chat/console_fleet_attention.py` | `fleet toast session-title fallback failed` captures traceback | same fixed label | `exception_type` |
| `Chat/console_fleet_wake.py` | `fleet wake drain intake failed` captures traceback | same fixed label | `exception_type` |
| `Chat/console_fleet_wake.py` | `wake send gate raised; deferring` captures traceback | same fixed label | `exception_type` |
| `Chat/console_fleet_wake.py` | `wake user-priority probe raised; deferring` captures traceback | same fixed label | `exception_type` |
| `Chat/console_fleet_wake.py` | `wake delivery failed` logs conversation id and traceback | `wake delivery failed` | `exception_type` |
| `Chat/console_fleet_wake.py` | `wake delivery ledger stamp failed` logs conversation id and traceback | `wake delivery ledger stamp failed` | `exception_type` |
| `Chat/console_fleet_wake.py` | `wake mark listing failed` captures traceback | same fixed label | `exception_type` |
| `Chat/console_fleet_wake.py` | `wake ledger read failed` logs conversation id and traceback | `wake ledger read failed` | `exception_type` |
| `Chat/console_fleet_wake.py` | `wake session resolution failed` captures traceback | same fixed label | `exception_type` |
| `UI/Screens/chat_screen.py` | `console fleet wake mount-claim failed` captures traceback | same fixed label | `exception_type` |
| `UI/Screens/chat_screen.py` | `fleet survivor check failed` captures traceback | same fixed label | `exception_type` |
| `UI/Console_Modules/image.py` | `remote transcript image fetch failed` logs URL and traceback | `remote transcript image fetch failed` | `exception_type` |
| `UI/Console_Modules/image.py` | `generate-image: provider resolution` logs exception repr | `generate-image provider resolution for LLM context failed` | `exception_type` |
| `UI/Console_Modules/image.py` | `Image generation batch raised` logs session id, exception text, and traceback | `Image generation batch raised` | `exception_type` |
| `UI/Screens/library_screen.py` | `Failed to load Library conversations page.` captures traceback | same fixed label | none |
| `app.py` | consolidated widget CSS error logs absolute CSS path | `Consolidated widget CSS incomplete` | `len(sources)` |
| `app.py` module entry | `Rebuilding CSS: {reason}` logs filename-bearing reason | `Generated CSS is stale during module entry; rebuilding` | none |
| `app.py` CLI entry | `Rebuilding CSS: {reason}` logs filename-bearing reason | `Generated CSS is stale during CLI entry; rebuilding` | none |

## Reviewed no-edit and deletion inventory

These surviving calls are reviewed-safe and must be registered exactly:

| Owner / label | Exact allowed AST fields |
| --- | --- |
| `llm_management_events.py` / `GGUF launch lease close failed` | `provider` |
| `llm_management_events.py` / `GGUF source preparation failed` | `provider` |
| `server_lifecycle.py` / `server claim resource close failed` | `provider` |
| `rag_service.py` / `Construction … ran a fallback expression` | `self._resolved_fts_match_construction()`, `FTS_MATCH_OR` |
| `UI/LLM_Management_Window.py` / `Managed GGUF inventory load failed` | none |
| `UI/Screens/chat_screen.py` / `Console fleet completion handoff will retry` | `claim.revision`, `type(exc).__name__` |
| `UI/Console_Modules/image.py` / `Console image edit cleanup failed` | `'image_edit'`, `'persistence'`, `type(exc).__name__` |
| `UI/Console_Modules/image.py` / `Console image edit failed` | `'image_edit'`, `phase`, `error_type` |
| `UI/Console_Modules/image.py` / `Console image edit failure guidance persistence failed` | `'image_edit'`, `'failure_guidance_persistence'`, `type(exc).__name__` |

The reviewed deletions are exactly 19 Moonshot and 13 Z.AI diagnostics removed
when their strict provider wrappers were extracted from `LLM_API_Calls.py`.
They include request/status/success events and unsafe response-detail,
exception-text, and exception-capturing events. No replacement diagnostic is
required because the strict wrappers retain typed failures and request metrics.

### Final-rebase amendment

The required final rebase onto `origin/dev` imported 17 additional unsafe call
sites across 11 owners after the original 31-call audit. They are part of the
same acceptance boundary, not a new logging policy:

| Owner | Unsafe call sites | Required surviving metadata |
| --- | ---: | --- |
| `Chat/console_agent_bridge.py` | 1 | fixed usage-accounting event plus exception type |
| `Chat/console_chat_store.py` | 2 | fixed speech-preference events plus exception type |
| `Chat/console_fleet_attention.py` | 1 | fixed mark-set event plus exception type |
| `Chat/console_fleet_wake.py` | 2 | fixed probe/hook events plus exception type |
| `Event_Handlers/TTS_Events/tts_events.py` | 2 | fixed missing-export event only |
| `UI/CCP_Modules/ccp_character_handler.py` | 2 | fixed displayed-card event only |
| `UI/STTS_Window.py` | 1 | fixed return-navigation event plus exception type |
| `UI/Screens/model_installed_view.py` | 1 | fixed deletion event plus exception type |
| `UI/Screens/watchlists_collections_screen.py` | 2 | fixed load event plus exception type |
| `UI/Wizards/FirstRunSetupWizard.py` | 1 | fixed rejected-delete event only |
| `config.py` | 2 | fixed precondition phase plus exception type |

The same rebase also imported bounded provider, status, error-class, count, and
code-side route/callback metadata plus privacy-improving deletions. Those rows
require no production edit and remain governed by the final generated manifest.
The sink topology remains the same six files.

## Decision

Deliver one atomic TASK-15743 reconciliation after TASK-3070.2 and TASK-16001.
Use the existing ADR-029 extractor, reviewed-metadata registry, and generated
inventory. Do not add a logging wrapper, a second ledger format, or a parallel
manifest.

For every unsafe call:

- replace interpolated identifiers, values, paths, URLs, and exception text
  with a fixed code-side event label;
- remove Loguru exception capture;
- retain only bounded fields already admitted by ADR-029, normally
  `type(exc).__name__` as `exception_type`;
- preserve control flow, user-visible behavior, and recovery behavior.

Register every surviving new or moved diagnostic shape in the existing
architecture evidence. Record reviewed-safe additions and reviewed deletions
in TASK-15743 notes rather than adding another machine-readable ledger. After
all source repairs and tests pass, regenerate
`Docs/security/production-diagnostic-inventory.json` exactly once.

## File map

Governance and evidence:

- `backlog/tasks/task-15743 - Reconcile-current-dev-diagnostic-inventory-drift.md`
- `Docs/superpowers/specs/2026-08-14-task-15743-current-dev-diagnostic-reconciliation-design.md`
- `Docs/superpowers/plans/2026-08-14-task-15743-current-dev-diagnostic-reconciliation.md`
- `Tests/Architecture/test_persistent_diagnostic_inventory.py`
- `Docs/security/production-diagnostic-inventory.json`

Production repairs:

- `tldw_chatbook/Agents/agent_service.py`
- `tldw_chatbook/Character_Chat/Character_Chat_Lib.py`
- `tldw_chatbook/Chat/console_agent_bridge.py`
- `tldw_chatbook/Chat/console_chat_controller.py`
- `tldw_chatbook/Chat/console_fleet_attention.py`
- `tldw_chatbook/Chat/console_fleet_wake.py`
- `tldw_chatbook/UI/Console_Modules/image.py`
- `tldw_chatbook/UI/Screens/chat_screen.py`
- `tldw_chatbook/UI/Screens/library_screen.py`
- `tldw_chatbook/app.py`
- `tldw_chatbook/Chat/console_chat_store.py`
- `tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py`
- `tldw_chatbook/UI/CCP_Modules/ccp_character_handler.py`
- `tldw_chatbook/UI/STTS_Window.py`
- `tldw_chatbook/UI/Screens/model_installed_view.py`
- `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py`
- `tldw_chatbook/config.py`

## Verification boundary

The change is acceptable only when:

1. the focused architecture evidence goes red before production edits and
   green after them;
2. mutation or source-shape evidence proves exception capture and each private
   field would be rejected;
3. affected focused tests, the rebased TASK-3070.2 gates, and TASK-16001 App
   tests pass;
4. the regenerated inventory equals live extraction and retains the existing
   six-file sink topology;
5. Ruff, format, Bandit, pycompile, diff/privacy/artifact checks, and the
   focused affected-feature suites pass on the final stack; the repository-wide
   pytest suite is excluded by owner direction on 2026-08-14;
6. independent spec and code-quality reviews approve the exact diff.

## Delivery order

Merge TASK-3070.2 first, TASK-16001 second, and TASK-15743 last. Rebase and
reverify each remaining branch after the preceding merge. TASK-15103 remains a
historical completed incident ending at its recorded checkpoint; later drift
belongs only to TASK-15743.

## ADR check

ADR required: no

ADR path: `backlog/decisions/029-local-private-data-boundary.md`

Reason: this work enforces the accepted persistent-diagnostic privacy boundary
without changing sink ownership, storage, or allowed metadata policy.
