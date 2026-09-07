# Console notes workspace round-trip UAT

Date: 2026-08-09
Task: TASK-14807

## Claim under test

A configured Console agent can use the default local/web provider to read an
existing note inside the configured workspace root and write the exact message
`Hi from tldw_Chatbook!` back into that note. Registration must not bypass the
permission store or escape the configured root.

## Harness

`Tests/QA/test_console_notes_workspace_uat.py` drives the joined production
path:

1. isolated TOML profile with `[console] local_tools_enabled = true` and an
   absolute scratch `workspace_root`;
2. `ConsoleChatController._compose_local_provider()`;
3. a real `MCPPermissionStore` with explicit, definition-hashed Allow entries
   for `fs_read` and `fs_edit`;
4. `ConsoleAgentBridge.run_reply()` and its real agent loop;
5. real `LocalToolProvider`, `fs_read`, and `fs_edit` implementations;
6. the actual note on disk and Console tool transcript rows.

The model turns are deterministic and scripted so the UAT is repeatable and
does not depend on a vendor account. Only model planning is scripted; config,
controller composition, catalog discovery, permission resolution, Console
orchestration, tool dispatch, workspace confinement, transcript emission, and
file persistence are real.

## Before

Workspace file `notes/project.md`:

```markdown
# Project Notes

UAT seed: this note belongs to the configured workspace.
```

Agent request:

```text
Read notes/project.md, then add exactly 'Hi from tldw_Chatbook!' to the note.
```

## Observed agent/tool sequence

```text
find_tools -> load_tools -> fs_read -> fs_edit -> final answer
```

The `fs_read` result contained the seeded note line, that result appeared in
the following model turn, and the Console transcript contained both `fs_read`
and `fs_edit` tool rows.

## After

```markdown
# Project Notes

UAT seed: this note belongs to the configured workspace.

Hi from tldw_Chatbook!
```

The final file assertion requires this exact content and exactly one occurrence
of the requested message.

## Evidence

Command:

```powershell
python -m pytest Tests/QA/test_console_notes_workspace_uat.py -q
```

Result: `1 passed`.

The scratch `config.toml` parsed successfully before and after the run. The real
user config SHA-256 was captured around the UAT and remained identical:

```text
before 971B42119641462455E81CA3E8E7224E9DE1FD58A14E4701FBCEA6141DAC51EB
after  971B42119641462455E81CA3E8E7224E9DE1FD58A14E4701FBCEA6141DAC51EB
```

The Tools-mode controls were also rendered with the real bundled stylesheet at
100x30; the heading, enabled state, workspace path, Save root action, and tool
filter remained painted and reachable.
