# Global constraints

- “The new destination and mode arguments belong only to `new_chat`.”
- “Retain the existing 120-character title cap and 20,000-character per-field prompt/instructions caps.”
- `destination=same_workspace|casual`; `mode=draft|start`; defaults are `same_workspace` and `draft`.
- “Both modes keep the user's current chat, active workspace, composer, and focus intact.”
- Casual persistence is `scope_type=global`, `workspace_id=NULL`; the session uses `CONSOLE_GLOBAL_WORKSPACE_ID`.
- Destination defaults and explicit standing instructions determine the fresh assistant; source identity, bindings, staged inputs and grants do not transfer.
- Both grant caches use requesting session incarnation, tool, resolved destination identity and mode. Denial ceilings remain per run/tool.
- “Each accepted chat start consumes one automatic generation; draft-only creation consumes none.”
- “Start attempts and fleet wakes share the existing automatic-primary admission limit and manual capacity reserve.”
- “Capacity refusal leaves a draft; this feature adds no waiting/retry timer.”
- AgentRunsDB acceptance is the ownership cutoff. Both durable fences must succeed before dispatch or `started`.
- Human decisions or paused preparation before acceptance return `not_started`; interrupted or uncertain starts never replay automatically.
- Version-2 handoffs persist edits/clears until accepted consumption or explicit discard. Unversioned handoffs keep the legacy contract.
- Bodies stay in private conversation storage; budget tables, badges and generic diagnostics carry bounded provenance/reasons only.
- Targeted tests only. A full sweep requires user opt-in. UI changes follow `backlog/docs/design-language.md` and its existing tokens.
- Planning changes no application code. At execution, inspect the dirty checkout and attached worktrees, then use a suitable isolated checkout under the worktree skill; preserve unrelated changes.
