# tldw_chatbook

[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange)](#alpha-status)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/license-AGPL--3.0--or--later-green)](LICENSE)

tldw_chatbook is a local-first terminal workbench for conversations with large
language models, personal knowledge, and controllable agent-assisted workflows.
It provides a Textual interface for talking to hosted APIs or local model
servers while keeping conversations and other core data on your machine by
default.

The shortest path through the app is:

1. Install the latest source checkout.
2. Let the first-run wizard connect a provider and model.
3. Open **Console** and send a message.

The core install stays reasonably lightweight. Retrieval, media processing,
web access, and protocol integrations are available as optional dependency
groups when you need them.

> New here? Start with [Quick start](#quick-start), then follow
> [Your first conversation](#your-first-conversation). The detailed
> [User Guide](Docs/User_Guide/index.md) is available when you want to explore
> beyond the first message.

## Alpha status

tldw_chatbook is **Alpha** software. The current package version is `0.1.8.0`.
Expect active development, changing interfaces, incomplete documentation in
some advanced areas, and occasional migration or recovery work.

Maturity is not uniform across the application:

- **Available now:** the core Textual shell, Console connections to hosted
  providers or supported local model servers, local conversations and notes,
  Library workflows, Artifacts and Chatbook workflows, Roleplay, Settings, and
  the source-install path documented below.
- **Still evolving:** advanced optional capabilities, ACP/runtime integration,
  some agent and tool workflows, write synchronization, and complete parity
  between local-only use and configured tldw server workflows.
- **Goal:** a modular, local-first workbench where model access, personal
  knowledge, and agent-assisted work can be combined without making every
  integration part of the core install.

“Local-first” describes the default ownership and storage model; it does not
mean every model executes inside this process. Hosted providers send prompts to
their services. Local models are normally reached through a separately running
server such as Ollama, llama.cpp, or another OpenAI-compatible endpoint.

Current package facts:

| Item | Value |
| --- | --- |
| Release | `0.1.8.0` |
| Classifier | Alpha |
| Python | `>=3.11` |
| Textual runtime | `textual==8.2.8` |
| Installed command | `tldw-cli` |
| Entry point | `tldw_chatbook.cli:main_cli_runner` |

<a id="installation"></a>
## Quick start

The primary installation route is a checkout of the latest source.

### 1. Clone the repository

```bash
git clone https://github.com/rmusser01/tldw_chatbook.git
cd tldw_chatbook
```

### 2. Create a Python 3.11+ virtual environment

Unix and macOS:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
```

Use a `python3` executable whose reported version is 3.11 or newer. If needed,
substitute a versioned executable such as `python3.12` in both commands.

Windows (PowerShell or Command Prompt):

```powershell
py -3 --version
py -3 -m venv .venv
```

If that reports a version below 3.11, install a supported Python or substitute
an installed selector such as `py -3.12` in both commands.

Activate in PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Activate in Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 3. Install the core application

```bash
python -m pip install -e .
```

### 4. Launch it

```bash
tldw-cli
```

On first launch, use the setup wizard. It is the primary setup path and can
configure either a hosted model provider or a local model server.

If the command is not found, confirm the virtual environment is active and
retry the install with that environment's `python -m pip`.

## Your first conversation

The hosted and local routes differ only in how the model is reached. Both end
in **Console**, using the provider and model chosen in the first-run wizard.
Completing the wizard returns you to **Home**; click **Console** or press
**Ctrl+2**, then send your first message.

### Option A: Connect a hosted model API

1. In the first-run wizard, choose the quick setup track.
2. Select the hosted provider you already use.
3. Enter its API key and choose one of its available models.
4. Finish setup and return to **Home**.
5. Click **Console** or press **Ctrl+2**.
6. Type a message in the composer and send it.

Prompts and responses travel through the selected provider under that
provider's terms. API keys can be stored through the guided configuration or
supplied through supported environment variables.

### Option B: Connect a local model server

1. Start your local server separately, for example Ollama, llama.cpp, or an
   OpenAI-compatible endpoint.
2. In the first-run wizard, choose the quick setup track.
3. Select the local or compatible provider and confirm its endpoint.
4. Choose a model exposed by that server.
5. Finish setup and return to **Home**.
6. Click **Console** or press **Ctrl+2**, then send a message.

tldw_chatbook does not claim an embedded model runtime. The local server owns
model loading and inference; the app provides the conversation interface.

### Change or repair setup

- Run the guided flow again at **Settings › Diagnostics › Run setup wizard**.
- Repair a provider, endpoint, key, or model directly at
  **Settings › Providers & Models**.
- Open the command palette with **Ctrl+P** or screen help with **F1**.

For a step-by-step explanation, see
[First-Run Setup](Docs/User_Guide/First_Run_Setup.md) and the
[Console guide](Docs/User_Guide/console.md).

## Capability overview

The application is organized around named work surfaces rather than a flat
list of equally mature features. These descriptions stay at the outcome level;
the User Guide records current controls and limitations.

| Destination | What it helps you do |
| --- | --- |
| **Home** | See what needs attention, what is running, and a useful next action. |
| **Console** | Hold model conversations, attach context, and supervise supported tools or agent runs. |
| **Library** | Organize and find conversations, notes, prompts, media, and imported source material. |
| **Artifacts** | Collect generated outputs and Chatbooks. |
| **Roleplay** | Work with characters, personas, dictionaries, and lore. |
| **Watchlists** | Monitor configured sources and review their runs or alerts. |
| **Schedules** | Manage when supported recurring work runs. |
| **Workflows** | Define and run reusable procedures. |
| **MCP** | Configure Model Context Protocol servers, tools, and permissions. |
| **ACP** | Work with compatible agent runtimes and sessions as integration support develops. |
| **Lab** | Explore model, speech, and evaluation workflows. |
| **Logs** | Inspect application activity and diagnostics. |
| **Settings** | Configure providers, models, appearance, storage, and application behavior. |

Core conversations and personal content are designed for local storage.
Capabilities that contact model providers, web services, MCP servers, local
model servers, or a configured tldw server cross the corresponding trust
boundary. Review the relevant settings before using them with sensitive data.

## Project direction

The project is moving toward a modular terminal environment in which a
conversation can draw on local knowledge and invoke explicitly controlled
tools without hiding where data goes or who owns an action.

That direction currently emphasizes:

- a newcomer path that reaches a useful first conversation quickly;
- local ownership of core application data and practical offline workflows;
- equal support for hosted APIs and separately operated local model servers;
- optional installation of heavier retrieval, media, web, and integration
  stacks;
- visible approvals and boundaries for agent-assisted actions;
- recovery paths that explain missing configuration or dependencies.

This is a direction, not a claim that all destinations already provide the
same depth, that every tldw server feature has a local equivalent, or that
every local workflow synchronizes back to a server.

## Optional capabilities

Install extras from the repository root, inside the same virtual environment
as the core application. Add only the capability groups you need.

| Optional group | Representative use |
| --- | --- |
| `embeddings_rag` | Embeddings and retrieval-augmented Library workflows |
| `websearch` | Web search and content extraction dependencies |
| `mcp` | Standalone and in-app MCP integration dependencies |
| `web` | Browser-served Textual access |
| `audio` | Audio import and transcription dependencies |
| `video` | Video import and transcription dependencies |
| `pdf` | PDF extraction dependencies |
| `ebook` | E-book extraction dependencies |

Install one group:

```bash
python -m pip install -e ".[embeddings_rag]"
```

Install a useful combination:

```bash
python -m pip install -e ".[audio,video,pdf,ebook]"
```

Install integration and web-search support together:

```bash
python -m pip install -e ".[mcp,websearch]"
```

Optional groups can add large dependencies, native libraries, model downloads,
or external services. A missing extra should disable or explain the advanced
capability it owns; it should not make the core install unusable. See the
[release recovery and setup guide](Docs/Development/release-recovery-setup.md)
for maintained recovery commands and capability ownership.

## Configuration and data

The main configuration file is:

```text
~/.config/tldw_cli/config.toml
```

The first-run wizard and **Settings** are preferred over hand-editing it.
Environment variables are supported for API keys, which is useful for shells,
CI, and secret managers. Provider-specific names and precedence can change, so
use the current Settings UI and maintained documentation instead of copying an
unverified list of variables.

On a typical Unix-like system, the default base storage directory is:

```text
~/.local/share/tldw_cli/
```

The app stores each profile's data in a child directory:

```text
~/.local/share/tldw_cli/<profile>/
```

A fresh install uses `~/.local/share/tldw_cli/default_user/` until you change
the profile name.

User- or profile-specific databases, logs, generated files, caches, and other
state live below that base. Exact paths may vary by platform, profile, and
configuration. Before deleting or moving anything, inspect the active paths in
**Settings** and back up the data you care about.

Local storage does not prevent a configured feature from sending selected
content elsewhere. Hosted model calls, web search, MCP tools, compatible agent
runtimes, and server-backed workflows each have their own data boundary.

## Troubleshooting and documentation

Start with these recovery checks:

- **No setup or wrong provider:** open
  **Settings › Diagnostics › Run setup wizard**.
- **Provider, key, endpoint, or model error:** open
  **Settings › Providers & Models** and save a valid combination.
- **`tldw-cli` is missing:** activate the virtual environment and run
  `python -m pip install -e .` again.
- **An advanced feature is unavailable:** install the optional group named by
  the recovery message, then restart the app.
- **A local model does not respond:** verify its separate server is running,
  its endpoint is reachable, and the configured model name is exposed there.
- **Need runtime detail:** open **Logs**, then consult the relevant guide
  before sharing diagnostic output that may contain private paths or content.

Maintained starting points:

- [User Guide](Docs/User_Guide/index.md) — navigation and task-oriented guides
- [First-Run Setup](Docs/User_Guide/First_Run_Setup.md) — wizard tracks and
  setup recovery
- [Console](Docs/User_Guide/console.md) — conversations, context, and live work
- [Release recovery and setup](Docs/Development/release-recovery-setup.md) —
  optional dependency and blocked-state recovery
- [Changelog](CHANGELOG.md) — release history

Because the project is Alpha, documentation may track the development branch
more closely than an older checkout. When labels differ, compare your package
version with the guide's verification note.

## Contributing, license, and contact

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) before
opening a pull request, keep changes focused, and include tests or verification
appropriate to the behavior being changed. Use the repository's issue tracker
for reproducible bugs, feature discussion, and documentation gaps.

tldw_chatbook is licensed under the
[GNU Affero General Public License v3.0 or later](LICENSE). If you discover a
security issue, avoid publishing sensitive details in a public issue; contact
the maintainer privately at [contact@rmusser.net](mailto:contact@rmusser.net).

Project links:

- Repository: [github.com/rmusser01/tldw_chatbook](https://github.com/rmusser01/tldw_chatbook)
- Issues: [github.com/rmusser01/tldw_chatbook/issues](https://github.com/rmusser01/tldw_chatbook/issues)
- License: [LICENSE](LICENSE)
