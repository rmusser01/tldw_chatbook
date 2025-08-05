# CLAUDE.md

This file provides comprehensive guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**tldw_chatbook** - TUI application built with Textual for LLM interactions. Features: conversation management, character chat, notes with file sync, media ingestion, RAG capabilities.

**Tech Stack**: Python ≥3.11, Textual ≥3.3.0, SQLite with FTS5, AGPLv3+  
**Key Dependencies**: httpx, loguru, rich, pydantic, toml, keyring, aiofiles, jinja2

## Quick Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"  # Or specific: .[embeddings_rag,websearch,local_vllm,ebook,pdf]

# Run
python3 -m tldw_chatbook.app

# Test
pytest  # All tests
pytest Tests/Chat/  # Specific module
pytest --cov=tldw_chatbook  # With coverage
```

## Architecture

### Core Structure

- **`app.py`** - Main entry, `TldwCli` class, tab management, global state
- **`config.py`** - TOML config at `~/.config/tldw_cli/config.toml`, env var fallbacks
- **`Constants.py`** - Tab IDs (TAB_CHAT, TAB_CODING, etc.), UI dimensions, provider mappings

### UI Layer (`UI/` and `Widgets/`)

**Main Windows** (all extend Screen):
- `Chat_Window_Enhanced.py` - Streaming chat, images, RAG
- `Conv_Char_Window.py` - Conversation/character CRUD
- `Notes_Window.py` - Notes with templates and sync
- `SearchRAGWindow.py` - RAG search interface
- `Evals_Window.py` - LLM benchmarking
- `MediaWindow.py` - Media management hub
- Other windows follow similar patterns

**Key Widgets**:
- `chat_message_enhanced.py` - Rich messages with actions
- `tool_message_widgets.py` - Tool calling UI (ToolCallMessage, ToolResultMessage)
- `IngestTldwApi*Window.py` - Media-specific ingestion forms
- `form_components.py` - Standardized form builders

### Business Logic

- **`Chat/`** - `Chat_Functions.py` (conversation CRUD), `document_generator.py` (export formats)
- **`Character_Chat/`** - `Character_Chat_Lib.py`, `ccv3_parser.py` (card formats)
- **`Notes/`** - `sync_engine.py` (bidirectional sync), template system
- **`RAG_Search/`** - `simplified/` for streamlined implementation, `chunking_service.py`
- **`Tools/`** - `tool_executor.py`, built-in: DateTimeTool, CalculatorTool
- **`Evals/`** - `eval_orchestrator.py`, `eval_runner.py`, task-specific runners
- **`LLM_Calls/`** - Provider integrations, unified `chat_with_provider()` interface

### Data Layer (`DB/`)

- **`ChaChaNotes_DB.py`** - Main DB (conversations, messages, characters, notes), schema v7
- **`Client_Media_DB_v2.py`** - Media storage with chunking
- **`RAG_Indexing_DB.py`** - Vector storage (when enabled)
- Other DBs: Evals, Prompts, Subscriptions
- Patterns: soft deletion, optimistic locking, FTS5 triggers, parameterized queries only

### Event System (`Event_Handlers/`)

Event flow: Widget → post_message() → @on() handler → workers → reactive updates → UI refresh

Key events: ChatEvent, StreamingChunk, RAGSearchEvent, SyncEvent, EvalEvent, TabEvent

### Key Patterns

**Database Operations**:
```python
with db.transaction() as cursor:
    cursor.execute(query, params)  # Always parameterized
```

**Reactive UI**:
```python
class MyWidget(Widget):
    data = reactive([], recompose=True)  # Rebuilds UI
    status = reactive("idle")  # Refresh only
```

**Background Work**:
```python
self.run_worker(self._heavy_task, exclusive=True)

@work(thread=True)
def _heavy_task(self):
    result = process()
    self.call_from_thread(self.update_ui, result)
```

## Development Guidelines

### Adding Features

**New LLM Provider**:
1. Add to `LLM_Calls/` with `chat_with_provider()` method
2. Register in main caller
3. Add config section

**New Tab**:
1. Create Screen in `UI/`
2. Add TAB_X constant
3. Register in app.py compose()
4. Add event handlers

**New Tool**:
1. Extend Tool class in `Tools/`
2. Implement: get_name(), get_description(), get_parameters(), execute()
3. Register in AVAILABLE_TOOLS

### Security Requirements

- Validate all inputs via `input_validation.py`
- Use `path_validation.py` for file paths
- SQL identifiers through `sql_validation.py`
- API keys from env/config only, never logged
- Sanitize HTML/Markdown content

### Performance Rules

- Workers for operations >100ms
- Stream LLM responses
- Chunk large files
- Paginate DB results
- Clear caches on context switch

### Configuration

Priority: env vars → config.toml → defaults

Key sections:
- `[API]` - Provider keys
- `[splash_screen]` - Animation settings
- `[embeddings]` - RAG config
- Provider-specific sections

### Testing

- Run full suite before PRs
- Use real SQLite in-memory for DB tests
- Property-based testing with Hypothesis
- Markers: unit, integration, optional, asyncio

## Special Systems

### Tool Calling
- Schema v7 adds tool messages
- `tool_executor.py` handles execution
- Provider parsing implemented
- Status: Detection works, execution pending

### Config Encryption
- AES-256 with PBKDF2
- `Utils/config_encryption.py`
- Password dialogs in widgets

### Splash Screen
- 20+ animations in `splash_animations.py`
- Config: `[splash_screen]` section
- Custom cards in `examples/custom_splash_cards/`

### Notes Sync
- Bidirectional file ↔ DB
- Last-write-wins conflict resolution
- Background monitoring

### Pre-commit Hook
- `auto_review.py` for Claude Code integration
- Reviews diffs with LLM
- Exit 0 = pass, 2 = fail

## Project-Specific Gotchas

1. **No localStorage** in artifacts - use React state or JS variables
2. **Tailwind limitations** - Only core utility classes, no compilation
3. **Schema migrations** - Always increment version, add to migrations/
4. **Optional deps** - Check with `optional_deps.py` before importing
5. **Thread safety** - Use transaction() context manager
6. **Tab constants** - Must match IDs in compose()
7. **Streaming** - Always offer non-streaming fallback
8. **FTS5** - Triggers auto-update on text columns
9. **Workers** - Mark exclusive=True to prevent duplicates
10. **Reactive** - recompose=True rebuilds, default just refreshes

## File Reference

Critical files for common tasks:
- Entry: `app.py`, `config.py`, `Constants.py`
- Chat: `Chat_Functions.py`, `chat_message_enhanced.py`
- DB: `base_db.py`, `ChaChaNotes_DB.py`
- LLM: `LLM_API_Calls.py`, `model_capabilities.py`
- Security: `path_validation.py`, `input_validation.py`
- UI: `form_components.py`, reactive patterns in any widget

## Code Style

- Type hints for public APIs
- Docstrings: Google style with Args/Returns/Raises
- Imports: stdlib → third-party → local
- PascalCase classes, snake_case functions
- Pydantic for validation
- Early returns to reduce nesting
- Constants for magic values
- Context managers for resources
- Descriptive test names
- Profile before optimizing
- Validate at boundaries
- Log errors with context

<!-- BACKLOG.MD GUIDELINES START -->
# Instructions for the usage of Backlog.md CLI Tool

## 1. Source of Truth

- Tasks live under **`backlog/tasks/`** (drafts under **`backlog/drafts/`**).
- Every implementation decision starts with reading the corresponding Markdown task file.
- Project documentation is in **`backlog/docs/`**.
- Project decisions are in **`backlog/decisions/`**.

## 2. Defining Tasks

### Understand the Scope and the purpose

Ask questions to the user if something is not clear or ambiguous.
Break down the task into smaller, manageable parts if it is too large or complex.

### **Title (one liner)**

Use a clear brief title that summarizes the task.

### **Description**: (The **"why"**)

Provide a concise summary of the task purpose and its goal. Do not add implementation details here. It
should explain the purpose and context of the task. Code snippets should be avoided.

### **Acceptance Criteria**: (The **"what"**)

List specific, measurable outcomes that define what means to reach the goal from the description. Use checkboxes (
`- [ ]`) for tracking.
When defining `## Acceptance Criteria` for a task, focus on **outcomes, behaviors, and verifiable requirements** rather
than step-by-step implementation details.
Acceptance Criteria (AC) define *what* conditions must be met for the task to be considered complete.
They should be testable and confirm that the core purpose of the task is achieved.
**Key Principles for Good ACs:**

- **Outcome-Oriented:** Focus on the result, not the method.
- **Testable/Verifiable:** Each criterion should be something that can be objectively tested or verified.
- **Clear and Concise:** Unambiguous language.
- **Complete:** Collectively, ACs should cover the scope of the task.
- **User-Focused (where applicable):** Frame ACs from the perspective of the end-user or the system's external behavior.

    - *Good Example:* "- [ ] User can successfully log in with valid credentials."
    - *Good Example:* "- [ ] System processes 1000 requests per second without errors."
    - *Bad Example (Implementation Step):* "- [ ] Add a new function `handleLogin()` in `auth.ts`."

### Task file

Once a task is created it will be stored in `backlog/tasks/` directory as a Markdown file with the format
`task-<id> - <title>.md` (e.g. `task-42 - Add GraphQL resolver.md`).

### Task Breakdown Strategy

When breaking down features:

1. Identify the foundational components first
2. Create tasks in dependency order (foundations before features)
3. Ensure each task delivers value independently
4. Avoid creating tasks that block each other

### Additional task requirements

- Tasks must be **atomic** and **testable**. If a task is too large, break it down into smaller subtasks.
  Each task should represent a single unit of work that can be completed in a single PR.

- **Never** reference tasks that are to be done in the future or that are not yet created. You can only reference
  previous
  tasks (id < current task id).

- When creating multiple tasks, ensure they are **independent** and they do not depend on future tasks.   
  Example of wrong tasks splitting: task 1: "Add API endpoint for user data", task 2: "Define the user model and DB
  schema".  
  Example of correct tasks splitting: task 1: "Add system for handling API requests", task 2: "Add user model and DB
  schema", task 3: "Add API endpoint for user data".

## 4. Recommended Task Anatomy

```markdown
# task‑42 - Add GraphQL resolver

## Description (the why)

Short, imperative explanation of the goal of the task and why it is needed.

## Acceptance Criteria (the what)

- [ ] Resolver returns correct data for happy path
- [ ] Error response matches REST
- [ ] P95 latency ≤ 50 ms under 100 RPS

## Implementation Plan (the how) (added after putting the task in progress but before implementing any code change)

1. Research existing GraphQL resolver patterns
2. Implement basic resolver with error handling
3. Add performance monitoring
4. Write unit and integration tests
5. Benchmark performance under load

## Implementation Notes (imagine this is the PR description) (only added after finishing the code implementation of a task)

- Approach taken
- Features implemented or modified
- Technical decisions and trade-offs
- Modified or added files
```

## 5. Implementing Tasks

Mandatory sections for every task:

- **Implementation Plan**: (The **"how"**)  
  Outline the steps to achieve the task. Because the implementation details may
  change after the task is created, **the implementation plan must be added only after putting the task in progress**
  and before starting working on the task.
- **Implementation Notes**: (Imagine this is a PR note)  
  Start with a brief summary of what has been implemented. Document your approach, decisions, challenges, and any deviations from the plan. This
  section is added after you are done working on the task. It should summarize what you did and why you did it. Keep it
  concise but informative. Make it brief, explain ONLY the core changes and assume that others will read the code to understand the details.

**IMPORTANT**: Do not implement anything else that deviates from the **Acceptance Criteria**. If you need to
implement something that is not in the AC, update the AC first and then implement it or create a new task for it.

## 6. Typical Workflow

```bash
# 1 Identify work
backlog task list -s "To Do" --plain

# 2 Read details & documentation
backlog task 42 --plain
# Read also all documentation files in `backlog/docs/` directory.
# Read also all decision files in `backlog/decisions/` directory.

# 3 Start work: assign yourself & move column
backlog task edit 42 -a @{yourself} -s "In Progress"

# 4 Add implementation plan before starting
backlog task edit 42 --plan "1. Analyze current implementation\n2. Identify bottlenecks\n3. Refactor in phases"

# 5 Break work down if needed by creating subtasks or additional tasks
backlog task create "Refactor DB layer" -p 42 -a @{yourself} -d "Description" --ac "Tests pass,Performance improved"

# 6 Complete and mark Done
backlog task edit 42 -s Done --notes "Implemented GraphQL resolver with error handling and performance monitoring"
```

### 7. Final Steps Before Marking a Task as Done

Always ensure you have:

1. ✅ Marked all acceptance criteria as completed (change `- [ ]` to `- [x]`)
2. ✅ Added an `## Implementation Notes` section documenting your approach
3. ✅ Run all tests and linting checks
4. ✅ Updated relevant documentation

## 8. Definition of Done (DoD)

A task is **Done** only when **ALL** of the following are complete:

1. **Acceptance criteria** checklist in the task file is fully checked (all `- [ ]` changed to `- [x]`).
2. **Implementation plan** was followed or deviations were documented in Implementation Notes.
3. **Automated tests** (unit + integration) cover new logic.
4. **Static analysis**: linter & formatter succeed.
5. **Documentation**:
    - All relevant docs updated (any relevant README file, backlog/docs, backlog/decisions, etc.).
    - Task file **MUST** have an `## Implementation Notes` section added summarising:
        - Approach taken
        - Features implemented or modified
        - Technical decisions and trade-offs
        - Modified or added files
6. **Review**: self review code.
7. **Task hygiene**: status set to **Done** via CLI (`backlog task edit <id> -s Done`).
8. **No regressions**: performance, security and licence checks green.

⚠️ **IMPORTANT**: Never mark a task as Done without completing ALL items above.

## 9. Handy CLI Commands

| Action                  | Example                                                                                                                                                       |
|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Create task             | `backlog task create "Add OAuth System"`                                                                                                                      |
| Create with description | `backlog task create "Feature" -d "Add authentication system"`                                                                                                |
| Create with assignee    | `backlog task create "Feature" -a @sara`                                                                                                                      |
| Create with status      | `backlog task create "Feature" -s "In Progress"`                                                                                                              |
| Create with labels      | `backlog task create "Feature" -l auth,backend`                                                                                                               |
| Create with priority    | `backlog task create "Feature" --priority high`                                                                                                               |
| Create with plan        | `backlog task create "Feature" --plan "1. Research\n2. Implement"`                                                                                            |
| Create with AC          | `backlog task create "Feature" --ac "Must work,Must be tested"`                                                                                               |
| Create with notes       | `backlog task create "Feature" --notes "Started initial research"`                                                                                            |
| Create with deps        | `backlog task create "Feature" --dep task-1,task-2`                                                                                                           |
| Create sub task         | `backlog task create -p 14 "Add Login with Google"`                                                                                                           |
| Create (all options)    | `backlog task create "Feature" -d "Description" -a @sara -s "To Do" -l auth --priority high --ac "Must work" --notes "Initial setup done" --dep task-1 -p 14` |
| List tasks              | `backlog task list [-s <status>] [-a <assignee>] [-p <parent>]`                                                                                               |
| List by parent          | `backlog task list --parent 42` or `backlog task list -p task-42`                                                                                             |
| View detail             | `backlog task 7` (interactive UI, press 'E' to edit in editor)                                                                                                |
| View (AI mode)          | `backlog task 7 --plain`                                                                                                                                      |
| Edit                    | `backlog task edit 7 -a @sara -l auth,backend`                                                                                                                |
| Add plan                | `backlog task edit 7 --plan "Implementation approach"`                                                                                                        |
| Add AC                  | `backlog task edit 7 --ac "New criterion,Another one"`                                                                                                        |
| Add notes               | `backlog task edit 7 --notes "Completed X, working on Y"`                                                                                                     |
| Add deps                | `backlog task edit 7 --dep task-1 --dep task-2`                                                                                                               |
| Archive                 | `backlog task archive 7`                                                                                                                                      |
| Create draft            | `backlog task create "Feature" --draft`                                                                                                                       |
| Draft flow              | `backlog draft create "Spike GraphQL"` → `backlog draft promote 3.1`                                                                                          |
| Demote to draft         | `backlog task demote <id>`                                                                                                                                    |

Full help: `backlog --help`

## 10. Tips for AI Agents

- **Always use `--plain` flag** when listing or viewing tasks for AI-friendly text output instead of using Backlog.md
  interactive UI.
- When users mention to create a task, they mean to create a task using Backlog.md CLI tool.

<!-- BACKLOG.MD GUIDELINES END -->
