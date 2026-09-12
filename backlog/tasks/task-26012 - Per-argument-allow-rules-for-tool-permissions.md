---
id: TASK-26012
title: Per-argument allow rules for tool permissions
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:44'
updated_date: '2026-09-01 18:12'
labels:
  - security
  - mcp
dependencies:
  - TASK-25905
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Always-allow is all-or-nothing per tool, so safe repeats can never be quieted. Verified on origin/dev: MCP/permission_store.py:472,489 keys state by (server_key, tool_name) with no argument dimension - which is precisely why the approval card deliberately withholds always_allow for raw shell (Widgets/Chat_Widgets/chat_approval_card.py:57), since allowing shell_exec once would allow every command. The result is a real capability gap wearing a safety justification: a user approving the same harmless command twenty times has no way to stop being asked. Hermes scopes allow rules to command-text globs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An always-allow entry can be scoped to an argument predicate rather than the whole tool
- [x] #2 A call matching the predicate is allowed; the same tool with non-matching arguments still prompts
- [x] #3 Predicates are displayed to the user in full before they are saved - no rule is created from a call the user did not read
- [x] #4 Argument-scoped rules participate in the existing definition-hash rug-pull guard: a changed tool definition invalidates them
- [x] #5 High-risk tools remain floored to ask regardless of an argument rule, consistent with MCP/permission_store.py:912-918
- [x] #6 Raw shell can adopt argument-scoped allow only in combination with the hardline floor from task-25905 - stated explicitly in the notes if not implemented here
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: store (exact-args match, order-insensitive canonical JSON, field/pattern globs, rug-pull, risk floor), provider (quiet matching call + still-prompt different args + gate agreement, allow_matching verdict persists exact args), service round trip\n2. Store: add_tool_arg_rule (exact displayed args only) + arg_rule_allows (profile-chain walk, hash guard, HIGH_RISK_TAGS hard floor)\n3. Service passthrough (hashes live HubTool); provider _arg_rule_allows_safe at pending_gate_for + invoke ask-branch; allow_matching verdict\n4. Card gains 'Always allow this exact input' (non-raw-shell only)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Rules live on the tool entry as arg_rules: the card-written shape is {'args_json': canonical-sorted-JSON of EXACTLY the displayed arguments} (AC#3 by construction — the writer never accepts anything the card didn't show; order-insensitive match pinned), plus hand-writable {'field','pattern'} fnmatch glob rules for hermes-style command globs (matcher honors both, writer never creates globs). arg_rule_allows walks the profile chain like resolve_effective_state, enforces the definition-hash rug-pull per rule (AC#4, HASH_FREE keys exempt as elsewhere), and hard-floors HIGH_RISK_TAGS tools to never-match (AC#5 read literally: the floor beats any argument rule). Provider: _arg_rule_allows_safe (duck-typed, fail-closed) short-circuits BOTH the pending gate (no card row) and invoke's ask branch (decision='allowed'); new 'allow_matching' verdict persists via the service passthrough (which fingerprints the live HubTool) and never widens into a whole-tool allow (pinned). Card: 'Always allow this exact input' for ordinary tools; RAW SHELL DOES NOT ADOPT arg-scoped allow in this task (AC#6 statement: its restricted option set is unchanged; the 25905 hardline floor is in place as the stated precondition, so a future adoption only needs the option + provider wiring). 13 new tests; MCP suites at the 2-name pre-existing baseline; approval-card suite green with the new option.
<!-- SECTION:NOTES:END -->
