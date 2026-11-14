# Twelve added app-service lookups: bounded correction

Completed under TASK38544 AC17 and the parent-approved site plan. No commits, push, manifests, pin raises, UI-module changes, or authority/kill-switch policy changes. This is a working-tree delta over the source snapshot in before-0.py through before-5.py; it does not overwrite concurrent earlier runtime repairs.

The exact additions were located using b07d1a1fa1..022e3e5435 and the parent's call-multiset inventory. Each is now an explicit named attribute read with AttributeError-only fallback, matching the old getattr default. Receiver expressions such as self.app and controller.app are evaluated outside the optional-attribute catch. RuntimeError and other service/receiver failures retain their old propagation. Inherited lookups are left alone.

| Owner | Before | After | Change |
| --- | ---: | ---: | --- |
| Agents/plain_chat_tool_surface.py | 1 | 0 | unified_mcp_service |
| Chat/archive_recovery/operations.py | 1 | 0 | workspace_registry_service |
| Chat/console_chat_controller.py | 28 | 23 | two selected counts, prompt scope, settings durability, catalog MCP service |
| Chat/console_received_dispatch.py | 6 | 4 | capture and currentness prompt scope |
| Event_Handlers/Chat_Events/chat_rag_events.py | 13 | 12 | only new key-first-use repository lookup |
| Widgets/plugin_review_modal.py | 2 | 0 | runtime and local MCP control owner |

No replacement reflective lookup was introduced. Agents has no remaining lookups or pin, so it has no parameterized census node. Chat, Event_Handlers and Widgets match their existing pins. The received-source helper is deliberately specific to prompt_scope_service: one explicit attribute read, used at the original argument/OR positions. This preserves tuple argument evaluation, source fence short-circuiting and receiver failure. The stock catalog check returns immediately for a noncurrent factory before evaluating controller.app. Constructor argument observations in prompt_mixin_service retain their original store/scope/persistence/durability order. Existing await/source fences, restore transactions, retirement behavior and plugin confirm custody are unchanged.

Two new test functions (five cases) cover available/absent/failing prompt-scope service, stale-app refusal before another service observation, noncurrent stock-factory early return and propagation of AttributeError from the receiver itself. Existing real owner controls cover missing MCP and engaged kill-switch counts, toggled light-tool count, three archive restore scopes, manual and queued source replacement across the hook await, canonical staged citation capture, mounted plugin profile save, and explicit plugin test completion across close/prune.

Verification: covering-final.json/log/xml records the exact command and results: **19 passed, 2 warnings in 30.14s**. The two SyntaxWarnings are invalid escape sequences in existing parsed sources. The initial covering selection stopped during collection because an Agents pin node no longer exists at zero; no tests executed in that attempt. Its command/log are retained. No full suite or completed-case repeats.

lint.json compares all six product files with the captured pre-edit bytes and checks the new test file. Added Ruff code/message multiplicity is empty for every file, and no new formatter deltas exist. lookup-counts.json records exact before/after call multisets; changes.patch contains only this correction, including the new test. git diff --check for these seven paths returned 0. No generic lookup adapter, receiver renaming, dynamic attribute string, allowlist expansion or budget waiver was used.

Ready for parent independent review. Controller mutations are finished again.
