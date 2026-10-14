# The legacy CCP route and retired chat tabs

This document disentangles the legacy names: **CCP is Conversations, Characters & Prompts** (the retired Roleplay-era tab — not a "coding copilot"; no such feature exists in the repo — the only "copilot" strings anywhere are unrelated model ids like `o365-copilot` and a CI bot actor). The `ccp` route still resolves (to PersonasScreen), part of its handler layer is live production code, and the separately-named **Coding** tab was deleted and folded into the Console.

## What exists now

| Artifact | Status |
| --- | --- |
| `UI/CCP_Modules/ccp_character_handler.py` | **Live** — `CCPCharacterHandler` constructed by PersonasScreen; FTS-backed character search (the documented consumer of the quoted-prefix FTS5 helper), CRUD against the local character-card DB, card import |
| `UI/CCP_Modules/ccp_persona_handler.py` | **Live** — persona handling through the character/persona scope service |
| `ccp_validators.py`, `ccp_validation_decorators.py`, `ccp_loading_indicators.py` | Live helpers |
| `ccp_messages.py` | Partially live — `CharacterMessage`/`ViewChangeMessage` consumed; the rest are compat definitions |
| `ccp_enhanced_handlers.py` | A hasattr-guarded monkey-patch layer whose only live branch is the character handler; conversation/prompt/dictionary branches are dead (those attributes no longer exist) |
| `ccp_screen.py`, sidebar/conversation/dictionary/prompt handlers, `Conv_Char_Window.py`, `Widgets/CCP_Widgets/` | **Deleted** — the screen retired 2026-06-11; the conversation/dictionary handlers were proven never-runnable (two independent worker defects → first-call crashes) and retired; the retired card/editor message classes were relocated into the Personas pane messages |
| `ccp_message_manager.py`, `Widgets/tool_message_widgets.py` | **Deleted** (zero production importers) — the tool-call rendering they carried is superseded by the Console transcript (tool markers, `ConsoleToolDiffRow` in `Widgets/diff_widgets` lineage) |
| `CodingWindow` / the `coding` tab | **Deleted** (2026-07-17, folded into the Console per ADR-015); `"coding"` is now just a route alias to `chat`. `Widgets/Coding_Widgets/repo_tree_widgets.py` survives production-dead, test-alive |

## Route resolution (verified)

`TAB_CCP = "conversations_characters_prompts"` aliases to `"ccp"`, which routes to **PersonasScreen** with canonical tab `personas` — nav highlighting keys off the canonical tab. `TAB_CCP` and `TAB_PERSONAS` deliberately share the display label "Roleplay". The personas destination's palette search aliases include `ccp`/`conversations_characters_prompts`. `TAB_CCP` remains a valid saved startup route. Unlike genuinely dead screens, the `ccp` route is **live** — the retirement removed the distinct CCP screen, not the route string (pinned by test: the active ccp route must still resolve to PersonasScreen).

Surviving data seams named after CCP: conversation `discovery_owner` values `{general_chat, ccp_character, ccp_persona}` (schemas, chat service, and DB normalize identically) and the runtime-policy capability id `characters_personas_ccp`.

## A Roleplay turn today (dataflow)

1. PersonasScreen composes character/persona browsing through the live CCP handlers.
2. "Start chat" routes through `open_chat_with_handoff` with an explicit intent marker (the legacy route launched a blank main-chat tab via the retired chat-tab container — unqueryable from a pushed screen).
3. Live conversation turns run through the **Console** (`ChatScreen` + the provider gateway), with the conversation record tagged `discovery_owner=ccp_character|ccp_persona`; Roleplay→Console activation is a cancellable preflight lane.
4. The old CCP-with-tool-calls experience is gone: the Console transcript owns tool rendering, and the retained non-streaming `chat_wrapper` adapter (which raises on `streaming=True`) has **zero production callers** — a census test pins the approved-caller set to empty, and media analysis calls `chat_api_call` directly. The adapter and its docstring's "CCP … still use" claim survive only pending a retirement decision.

## Boundaries and tests

The live handlers must never re-reference the retired `CCPWindow`/`Conv_Char_Window` and must reference `PersonasScreen` — pinned by `Tests/UI/test_legacy_entrypoints_retired.py`, which also asserts the retired handler classes are not package exports and enumerates `RETIRED_FILES`.

## Governing decisions

ADR-004 (`004-personas-destination-native-workbench.md` — the CCP route retirement), ADR-007 (`007-personas-workbench-route-consolidation.md`), ADR-015 (folds + palette aliases; the Coding→Console fold), ADR-026 (Console owns conversation entry), ADR-032 (the diff rendering that replaced CCP tool-call widgets). Historical: `Docs/Development/ccp-refactoring-complete.md` (predates the message-manager deletion — cite as history). Key tasks: task-19873 (never-runnable handlers), task-15780 (dictionary/prompt editor widgets), task-32899 (dead-tier deletion), task-651 (legacy root state).

## Verified gotchas

1. Never call it "Coding CoPilot" in new writing — no coding-copilot feature exists (the only "copilot" strings in code are unrelated model ids like `o365-copilot`); the coding lineage is the separately deleted `TAB_CODING` tab.
2. `resolve_screen_target("ccp")` reports canonical tab `personas` — highlight logic must use the canonical tab, not the requested route.
3. `ccp_enhanced_handlers`' dead branches are load-bearing history — the hasattr guards are why the layer still works against a screen that no longer has those attributes.
4. The `worker_events.chat_wrapper_function` docstring's "CCP … still use" claim is stale — zero production callers remain (the census test `Tests/Event_Handlers/test_retained_worker_adapter.py` pins the approved-caller set to empty); this doc supersedes the docstring.

## Related docs

- [character-chat.md](./character-chat.md) — the roleplay domain the CCP handlers serve
- [app-shell.md](./app-shell.md) — route aliases and canonical tabs
- [chat-pipeline.md](./chat-pipeline.md) — the streaming boundary that retired CCP's own LLM calls
