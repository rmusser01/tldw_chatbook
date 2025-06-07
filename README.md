# tldw_chatbook README

A Textual TUI for interacting with various LLM APIs, managing conversation history, characters, notes, and more.

Current status: Working/In-Progress

![Screenshot](https://github.com/rmusser01/tldw_chatbook/blob/main/static/PoC-Frontpage.PNG?raw=true)
### Quick Start
- **Via Manual Installation**
  - Clone the repository: `git clone https://github.com/rmusser01/tldw_chatbook`
  - Setup a virtual environment (optional but recommended): 
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
  - Install the dependencies: `pip install -r requirements.txt`
  - Run the application:
    - If you are in the root directory of the repository: `python3 -m tldw_chatbook.app`

## Current Features
- Textual TUI interface
  - Configuration via `config.toml`
    - Default `config.toml` stored at `~/.config/tldw_cli/config.toml`
    - Default user DB stored at `~/.share/tldw_cli/`
  - Environment variable support for API keys
  - Launch with `python3 -m tldw_chatbook.app` (working on pip packaging)
  - Inspiration from [Elia Chat](https://github.com/darrenburns/elia)

#### Chat Features
<details>
<summary> Chat Features </summary>

- tldw_chatbook's current feature set relating to chat functionality includes:
  - Connect to multiple LLM providers 
    - (Local: Llama.cpp, Ollama, Kobold.cpp, vLLM, Aphrodite, Custom-OpenAI API endpoint
    - Commercial: OpenAI, Anthropic, Cohere, Deepseek, Google, Groq, Mistral, OpenRouter)
  - Support for 'ranking' conversation replies (Thumbs up/down)
  - Character Card functionality
    - Persona/Character/Chatbot support
    - Ingestion of character cards into the DB
    - Searching + Loading + applying character cards to the current conversation
    - Support for 'character' chat functionality (e.g., ability to chat with a character, like a chatbot)
  - Conversation history management
    - Can search conversations by title, keywords, or contents
    - Can load conversations from the DB for continuation or editing
    - Can edit messages in conversations, or conversations in general (titles/keywords/contents)
- Chat-Specific features, Support for:
  - Streaming responses from LLMs
  - Re-generating responses
  - 'generate a question/answer' functionality
  - 'Ephemeral' conversations are default (conversations are not saved to the Database unless explicitly saved)
  - Stripping thinking blocks from responses
- Full OpenAI API support for chat completions
- Prompt-Related Features:
  - Save prompts to the prompts Database, one at a time or in bulk
  - Edit/Clone/Delete prompts in the prompts Database
  - Searching for Prompts via Title/Keyword, and then load ingested Prompts into the current conversation
- Support for searchingloading/editing/saving Notes from the Notes Database.
</details>

#### Notes & Media Features
<details>
<summary> Notes & Media Features </summary>

- Notes
  - Create, edit, and delete notes
  - Search notes by title, keywords, or contents
  - Load notes into the current conversation
  - Save notes to the Notes Database

- Media (WIP)
  - Full Media DB support
  - Ingest media files into the Media DB
  - Search media files by title, keywords, or contents
  - Load media files into the current conversation
  - Save media files to the Media Database via tldw API or local processing
  - 
</details>

#### Local LLM Management Features
<details>
<summary> Local LLM Inference </summary>

- Local LLM Inference (WIP)
    - Support for local LLM inference via llama.cpp, Ollama, Kobold.cpp, Ooba, vLLM, Aphrodite, and Custom user-defined OpenAI API endpoints
    - Support for managing a local Ollama instance via HTTP API
    - Support for managed local LLM inference via Llama.cpp & Llamafile
    - Support for managed local LLM inference via vLLM (e.g., Mistral, Llama 3, etc.)
    - Support for managed local LLM inference via mlx-lm
    - Support for managed local LLM inference via OnnxRuntime
    - Support for downloading models from Hugging Face and other sources

</details>


### Planned Features
<details>
<summary> Future Features </summary>

- **General**
  - Web Search functionality (e.g., ability to search the web for relevant information based on conversation history or notes or query)
  - Additional LLM provider support (e.g., more local providers, more commercial providers)
  - More robust configuration options (e.g., more environment variable support, more config.toml options)

- **Chat**
  - Conversation Forking + History Management thereof (Already implemented, but needs more testing/UI buildout)
  - Enhanced character chat functionality (e.g., ASCII art for pictures, 'Generate Character' functionality, backgrounds)
  - Improved conversation history management (e.g., exporting conversations, better search functionality)

- **Notes-related**
  - Improved notes and keyword management (Support for syncing notes from a local folder/file - think Obsidian)

- **Media-related**

- **Search Related**
  - Improved search functionality (e.g., more robust search options, better search results)
  - Support for searching across conversations, notes, characters, and media files
  - Support for websearch (code is in place, but needs more testing/UI buildout)
  - Support for RAG (Retrieval-Augmented Generation) functionality (e.g., ability to retrieve relevant information from conversations, notes, characters, media files and prompts)

- **Tools & Settings**
  - Support for DB backup management/restore
  - General settings management (e.g., ability to change application settings, like theme, font size, etc.)
  - Support for user preferences (e.g., ability to set user preferences, like default LLM provider, default character, etc.)
  - Support for user profiles (e.g., ability to create and manage user profiles, tied into preference sets)

- **LLM Management**
  - Cleanup and bugfixes

- **Stats**
  - I imagine this page as a dashboard that shows various statistics about the user's conversations, notes, characters, and media files.
  - Something fun and lighthearted, but also useful for the user to see how they are using the application.
  - This data will not be stored in the DB, but rather generated on-the-fly from the existing data.
  - This data will also not be uploaded to any external service, but rather kept local to the user's machine.
  - This is not meant for serious analytics, but rather for fun and lighthearted use. (As in it stays local.)

- **Evals**
  - Self-explanatory
  - Support for evaluating LLMs based on user-defined criteria.
  - Support for RAG evals.
  - Jailbreaks?

- **Coding**
    - Why not, right?
    - Build out a take on the agentic coder, will be a longer-term goal, but will be a fun addition.

- **Workflows**
  - Workflows - e.g., Ability to create structured workflows, like a task list or a series of steps to follow, with the ability to execute them in order with checkpoints after each step. (Agents?)
  - Agentic functionality (e.g., ability to create agents that can perform tasks based on conversation history or notes, think workflow automation with checkpoints)
    - First goal will be the groundwork/framework for building it out more, and then for coding, something like Aider?
    - Separate from the workflows, which are more like structured task lists or steps to follow. Agentic functionality will be more about creating workflows, but not-fully structured, that adapt based on the 'agents' decisions.

- **Other Features**
  - Support for Server Syncing (e.g., ability to sync conversations, notes, characters, Media DB and prompts across devices)
  - Support for audio playback + Generation (e.g., ability to play audio files, generate audio from text - longer term goal, has to run outside of the TUI)
  - Mindmap functionality (e.g., ability to create mindmaps from conversation history or notes)

</details>


### Getting Started
- **Via pip**
  - Install the package via pip: `pip install tldw_chatbook` (Not yet available, but will be soon)
  - Run the application from your terminal: `tldw_chatbook`
- **Via Manual Installation**
  - Clone the repository: `git clone https://github.com/rmusser01/tldw_chatbook`
  - Setup a virtual environment (optional but recommended) and run tldw_chatbook: 
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate or .venv\Scripts\activate
    ```
  - Install the dependencies: `pip install -r requirements.txt` 
    ```bash
    pip install -r requirements.txt
    ```
  - Run the application:
    - If you are in the root directory of the repository: `python3 -m tldw_chatbook.app`
    ```bash
    python3 -m tldw_chatbook.app
    ```
- **Configuration**
  - The application uses a `config.toml` file located at `~/.config/tldw_cli/config.toml`.
  - On first run, a default configuration file will be created if one doesn't exist. You'll need to edit this file to add your API keys for the services you want to use.
  - Environment variables can also be used for API keys, as specified in the `[api_settings.<provider>]` sections of the `config.toml` (e.g., `OPENAI_API_KEY`).
- **User Database**
  - The application uses several user databases stored at `~/.share/tldw_cli/`.
  - The databases are: 
    - `tldw_chatbook_ChaChaNotes.db` - This sqlite DB stores your conversations, characters, and notes.
    - `tldw_chatbook_media_v2.db` - This sqlite DB stores the user's ingested media files.
    - `tldw_chatbook_prompts.db` - This sqlite DB stores the user's prompts.
  - Each database is created on first run if it doesn't already exist.

### Project Structure
<details>
<summary>Here's a brief overview of the main directories in the project:</summary>

```
└── ./
    └── tldw_chatbook
        ├── assets
        │   └── Static Assets
        ├── Character_Chat
        │   └── Libaries relating to character chat functionality/interactions
        ├── Chat
        │   └── Libraries relating to chat functionality/orchestrations
        ├── Chunking
        │   └── Libaries relating to chunking text for LLMs
        ├── css
        │   └── CSS files for the Textual TUI
        ├── DB
        │   └── Core Database Libraries
        ├── Embeddings
        │   └── Embeddings Generation & ChromaDB Libraries
        ├── Event_Handlers
        │   ├── Chat_Events
        │   │   └── Handle all chat-related events
        │   ├── LLM_Management_Events
        │   │   └── Handle all LLM management-related events
        │   └── Event Handling for all pages is done here
        ├── LLM_Calls
        │   └── Libraries for calling LLM APIs (Local and Commercial)
        ├── Local_Inference
        │   └── Libraries for managing local inference of LLMs (e.g., Llama.cpp, llamafile, vLLM, etc.)
        ├── Metrics
        │   └── Library for instrumentation/tracking (local) metrics
        ├── Notes
        │   └── Libraries for managing notes interactions and storage
        ├── Prompt_Management
        │   └── Libraries for managing prompts interactions and storage + Prompt Engineering
        ├── RAG_Search
        │   └── Libraries for RAG (Retrieval-Augmented Generation) search functionality
        ├── Screens
        │   └── First attempt at Unifying the screens into a single directory
        ├── Third_Party
        │   └── All third-party libraries that are not part of the main application
        ├── tldw_api
        │   └── Code for interacting with the tldw API (e.g., for media ingestion/processing/web search)
        ├── TTS
        │   └── Libraries for Text-to-Speech functionality
        ├── UI
        │   └── Libraries containing all screens and panels for the Textual TUI
        ├── Utils
        │   └── All utility libraries that are standalone
        ├── Web_Scraping
        │   └── Libraries for web scraping functionality (e.g., for web search, RAG, etc.)
        ├── Widgets
        │   └── Reusable TUI components/widgets
        ├── app.py - Main application entry point (its big...)
        ├── config.py - Configuration management library
        ├── Constants.py - Constants used throughout the application (Some default values, Config file template, CSS template)
        └── Logging_Config.py - Logging configuration for the application
```
</details>

### Inspiration
https://github.com/darrenburns/elia

## Contributing
- Contributions are welcome! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for details on how to contribute to this project.(WIP)
- (Realistically, this is a work in progress, so contributions are welcome, but please be aware that the codebase is still evolving and may change frequently.)
- Make a pull request against the `dev` branch, where development happens prior to being merged into `main`.

## License

This project is licensed under the GNU Affero General Public License - see the [LICENSE](LICENSE) file for details.

### Contact
For any questions, issues, or feature requests, please open an issue on the [GitHub repository](https://github.com/rmusser01/tldw) or contact me directly on the tldw_Project Discord or via the email in my profile.
