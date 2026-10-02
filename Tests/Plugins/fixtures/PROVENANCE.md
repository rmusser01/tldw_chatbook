# Native package fixture provenance

The `native` tree is an original, deliberately minimal fixture authored for
TASK-32668 on 2026-09-15. It is distributed under this repository's AGPL-3.0-or-later
license. It contains no third-party executable or downloaded schema. The fixture
builder copies these bytes to a fresh directory; test expectations are authored
independently of the production parser.

Pinned contracts consulted on 2026-09-15:

- [Agent Plugins manifest schema 1.0.0](https://agent-plugins.org/schemas/1.0.0/plugin.schema.json)
- [Agent Plugins MCP schema 1.0.0](https://agent-plugins.org/schemas/1.0.0/mcp.schema.json)
- [Loading and discovery](https://agent-plugins.org/client-implementers/loading-and-discovery)
- [Client extensions](https://agent-plugins.org/plugin-authors/client-extensions)
- [Portable MCP runtime](https://agent-plugins.org/client-implementers/mcp-runtime)
- [MCP authoring requirements](https://agent-plugins.org/plugin-authors/mcp-servers)
- [Agent Skills specification](https://agentskills.io/specification)

The Agent Plugins documentation identifies its license as CC BY 4.0, attributed
to Agent Plugins documentation contributors, 2026. The schema URLs are versioned;
the Agent Skills reference is dated here because its public page has no version
identifier. No upstream prose, implementation, or schema file is vendored.
`schemas.py` implements local rules; runtime inspection never downloads them.

The Chatbook extension and hook fixtures derive from the repository's approved
managed-plugins spec section 3.1 and expanded-hook-runtime spec section 2.1,
under the repository license. The minimal fixture omits optional extension
components. Tests independently add the full commands/rules/agents/hooks/MCP
inventory, then mutate required references, recognized extensions, and unrelated
namespaces. Vendor manifests are synthetic candidate markers and provide no
vendor compatibility evidence.
