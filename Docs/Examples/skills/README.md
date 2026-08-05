# Example Skills

Installable skill definitions for the Console agent. Skills are markdown
orchestration prompts a subagent run executes; see ADR-009
(`backlog/decisions/`) for the trust model.

## web-research

`web-research/SKILL.md` — decomposes a research question into sub-questions,
searches multiple angles with `web_search`, fetches primary sources with
`web_fetch`, and synthesizes a cited answer with conflicts/caveats.

### Requirements

- `[console] local_tools_enabled = true` in `~/.config/tldw_cli/config.toml`
  (the skill's tools, `web_search`/`web_fetch`, are local agent tools).
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

Imported skills are trust-scanned before use (ADR-009). Once installed, ask
the agent to "research <question>" or invoke the skill explicitly.
