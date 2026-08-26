# Example Skills

Installable skill definitions for the Console agent. Skills are markdown
orchestration prompts a subagent run executes; see ADR-009
(`backlog/decisions/`) for the trust model.

## web-research

`web-research/SKILL.md` — decomposes a research question into sub-questions,
searches multiple angles with `web_search`, fetches primary sources with
`web_fetch`, and synthesizes a cited answer with conflicts/caveats.

### Requirements

- Local workspace, web, and Watchlists tools enabled in **MCP → Tools**. This is the shipped
  default; an explicit `[console] local_tools_enabled = false` opts out.
- Web tools default to the permission store's global `ask` state — expect an
  approval prompt the first time the skill searches or fetches (approve for
  the session or always to skip subsequent prompts).

### Install

Either import it through the app's skills library UI (Skills screen → import
directory → select `Docs/Examples/skills/web-research/`), or copy the
directory into your user skills directory:

```bash
cp -r Docs/Examples/skills/web-research ~/.local/share/tldw_cli/<user_folder>/skills/skills/
```

The base is `get_user_data_dir()` (`config.py:4373`) — by default
`~/.local/share/tldw_cli/<user_folder>/` — with skills stored under its
`skills/skills/` subdirectory (`app.py:4488-4497`). The Skills screen shows
the exact path for your install.

### A note for skill authors

Skills that declare no `allowed-tools` front-matter now pass the full
builtins + local tool set through to their subagent run (previously builtins
only — changed in phase 3c, matching how native `spawn_subagent` children
inherit tools). Every call stays approval-gated through the same permission
store, so an undeclared skill still prompts before any mutating or network
tool executes. Declare `allowed-tools` explicitly if you want a skill's child
restricted to a narrower set.

Imported skills are trust-scanned before use (ADR-009). Once installed, ask
the agent to "research <question>" or invoke the skill explicitly.

Watchlists-aware skill authors should use the exact
`watchlists_search_items`/`watchlists_get_item` inventory and permission
boundary recorded in [TASK-16222](../../../backlog/tasks/task-16222%20-%20Expose-local-Watchlists-search-tools-to-Console-and-MCP.md),
[ADR-030](../../../backlog/decisions/030-local-library-agent-tool-boundary.md),
and amended [ADR-032](../../../backlog/decisions/032-local-agent-tool-permission-boundary.md).
