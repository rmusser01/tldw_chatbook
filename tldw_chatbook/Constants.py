# Constants.py
# Description: Constants for the application
#
# Imports
#
# 3rd-Party Imports
#
# Local Imports
#
########################################################################################################################
#
# Functions:

# --- Constants ---
TAB_CHAT = "chat"
TAB_CCP = "conversations_characters_prompts"
TAB_MEDIA = "media"
TAB_SEARCH = "search"
TAB_INGEST = "ingest"
TAB_EVALS = "evals"
TAB_LLM = "llm_management"
TAB_TOOLS_SETTINGS = "tools_settings"
TAB_STATS = "stats"
TAB_LOGS = "logs"
TAB_CODING = "coding"
TAB_STTS = "stts"
TAB_STUDY = "study"
TAB_WRITING = "writing"
TAB_RESEARCH = "research"
TAB_RESEARCH_WORKSPACE = "research_workspace"
TAB_SUBSCRIPTIONS = "subscriptions"
TAB_CHATBOOKS = "chatbooks"
TAB_HOME = "home"
TAB_LIBRARY = "library"
TAB_ARTIFACTS = "artifacts"
TAB_PERSONAS = "personas"
TAB_WATCHLISTS_COLLECTIONS = "watchlists_collections"
TAB_SCHEDULES = "schedules"
TAB_WORKFLOWS = "workflows"
TAB_MCP = "mcp"
TAB_ACP = "acp"
TAB_SKILLS = "skills"
TAB_SETTINGS = "settings"
TAB_MEETINGS = "meetings"

# Library navigation-context contract keys and values.
LIBRARY_NAV_CONTEXT_MODE = "mode"
LIBRARY_NAV_CONTEXT_ARTIFACT_CHATBOOK_ID = "artifact_chatbook_id"
LIBRARY_NAV_CONTEXT_CONVERSATION_ID = "conversation_id"
LIBRARY_NAV_CONTEXT_NOTE_ID = "note_id"
LIBRARY_NAV_CONTEXT_NOTES_CREATE = "notes_create"
LIBRARY_NAV_CONTEXT_OPEN_SOURCE_TYPE = "open_source_type"
LIBRARY_NAV_CONTEXT_OPEN_SOURCE_ID = "open_source_id"
# Home's ingest-jobs "Open details" control (L3b Task 6) re-points here to
# land the Library shell on the in-canvas Ingest > Import media view.
LIBRARY_NAV_CONTEXT_INGEST = "ingest_media"
LIBRARY_MODE_CONVERSATIONS = "conversations"

# Console navigation-context contract keys.
CONSOLE_NAV_CONTEXT_RESUME_LOCAL_CONVERSATION_ID = (
    "resume_local_conversation_id"
)
CONSOLE_NAV_CONTEXT_CHARACTER_CONVERSATION_TARGET = "character_conversation_target"

# Trusted character-conversation navigation context keys.
ROLEPLAY_NAV_CONTEXT_CHARACTER_CONVERSATION = "character_conversation"
LIBRARY_NAV_CONTEXT_CHARACTER_REPAIR = "character_repair"
LIBRARY_NAV_CONTEXT_CHARACTER_INSPECTION = "character_unavailable_inspection"
LIBRARY_NAV_CONTEXT_CHARACTER_BROWSE = "character_unavailable_browse"
CHARACTER_NAV_CONTEXT_RETURN_FOCUS = "return_focus"

# Saved-conversation pagination shared by the Roleplay controller and inspector.
PERSONAS_CONVERSATIONS_PAGE_SIZE = 20

# Watchlists navigation-context contract keys and values.
WATCHLISTS_NAV_CONTEXT_SECTION = "section"
WATCHLISTS_NAV_CONTEXT_BACKEND = "backend"
WATCHLISTS_NAV_CONTEXT_RUN_ID = "run_id"
WATCHLISTS_NAV_CONTEXT_BRIEFING_ID = "briefing_id"
WATCHLISTS_SECTION_NOTIFICATIONS = "notifications"
WATCHLISTS_SECTION_RUNS = "runs"

# Media navigation-context contract keys and values.
# Applied pre-mount by handle_screen_navigation; MediaScreen stashes the
# subview and applies it to the freshly composed MediaWindow on mount
# (mirroring its saved-view restore pattern).
MEDIA_NAV_CONTEXT_BROWSE_SUBVIEW = "browse_subview"
MEDIA_BROWSE_SUBVIEW_READ_IT_LATER = "read-it-later"

ALL_TABS = [
    TAB_CHAT,
    TAB_CCP,
    TAB_MEDIA,
    TAB_SEARCH,
    TAB_INGEST,
    TAB_EVALS,
    TAB_LLM,
    TAB_STTS,
    TAB_STUDY,
    TAB_WRITING,
    TAB_RESEARCH,
    TAB_WATCHLISTS_COLLECTIONS,
    TAB_CHATBOOKS,
    TAB_ARTIFACTS,  # Compatibility route into Library; retained in saved preferences.
    TAB_TOOLS_SETTINGS,
    TAB_LOGS,
    TAB_CODING,
    TAB_STATS,
]

TAB_DISPLAY_LABELS = {
    TAB_CHAT: "Console",
    TAB_CCP: "Roleplay",
    TAB_MEDIA: "Media",
    TAB_SEARCH: "Search",
    TAB_INGEST: "Ingest",
    TAB_EVALS: "Evals",
    TAB_LLM: "Models",
    TAB_TOOLS_SETTINGS: "MCP",
    TAB_STATS: "Stats",
    TAB_LOGS: "Logs",
    TAB_CODING: "Coding",
    TAB_STTS: "Speech",
    TAB_STUDY: "Study",
    TAB_WRITING: "Writing",
    TAB_RESEARCH: "Research",
    TAB_RESEARCH_WORKSPACE: "Research Workspace",
    TAB_CHATBOOKS: "Chatbooks",
    TAB_HOME: "Home",
    TAB_LIBRARY: "Library",
    TAB_ARTIFACTS: "Artifacts",
    TAB_PERSONAS: "Roleplay",
    TAB_WATCHLISTS_COLLECTIONS: "Watchlists",
    TAB_SCHEDULES: "Schedules",
    TAB_WORKFLOWS: "Workflows",
    TAB_MCP: "MCP",
    TAB_ACP: "ACP",
    TAB_SKILLS: "Skills",
    TAB_SETTINGS: "Settings",
    TAB_MEETINGS: "Meetings",
}


def get_tab_display_label(tab_id: str) -> str:
    """Return the user-facing label for a top-level tab ID."""
    return TAB_DISPLAY_LABELS.get(tab_id, tab_id.replace("_", " ").title())


# Subscription types
SUBSCRIPTION_TYPES = [
    "rss",
    "reddit",
    "youtube",
    "github",
    "hackernews",
    "generic",
    "custom",
]

# Subscription update frequencies (in seconds)
SUBSCRIPTION_UPDATE_FREQUENCIES = {
    "15 minutes": 900,
    "30 minutes": 1800,
    "1 hour": 3600,
    "2 hours": 7200,
    "4 hours": 14400,
    "6 hours": 21600,
    "12 hours": 43200,
    "Daily": 86400,
    "Weekly": 604800,
}

# --- TLDW API Form Specific Option Containers (IDs) ---
TLDW_API_VIDEO_OPTIONS_ID = "tldw-api-video-options"
TLDW_API_AUDIO_OPTIONS_ID = "tldw-api-audio-options"
TLDW_API_PDF_OPTIONS_ID = "tldw-api-pdf-options"
TLDW_API_EBOOK_OPTIONS_ID = "tldw-api-ebook-options"
TLDW_API_DOCUMENT_OPTIONS_ID = "tldw-api-document-options"
TLDW_API_XML_OPTIONS_ID = "tldw-api-xml-options"
TLDW_API_MEDIAWIKI_OPTIONS_ID = "tldw-api-mediawiki-options"
TLDW_API_PLAINTEXT_OPTIONS_ID = "tldw-api-plaintext-options"

ALL_TLDW_API_OPTION_CONTAINERS = [
    TLDW_API_VIDEO_OPTIONS_ID,
    TLDW_API_AUDIO_OPTIONS_ID,
    TLDW_API_PDF_OPTIONS_ID,
    TLDW_API_EBOOK_OPTIONS_ID,
    TLDW_API_DOCUMENT_OPTIONS_ID,
    TLDW_API_XML_OPTIONS_ID,
    TLDW_API_MEDIAWIKI_OPTIONS_ID,
    TLDW_API_PLAINTEXT_OPTIONS_ID,
]


# --- Responsive layout ---
#: Viewport width (terminal columns) at which the App gains the
#: ``-wide-viewport`` CSS class (``WideViewportTierMixin`` in app.py) and
#: the repo-wide modal wide tier engages. Single source of truth shared
#: with the test registry (Tests/UI/modal_wide_tier_registry.py) so the
#: production breakpoint and the test contract cannot drift.
WIDE_VIEWPORT_COLUMNS = 150


# --- CSS definition (removed) ---
# The ~1,589-line `css_content` string here was DEAD (task-32810.2 / core
# review P3): production loads css/tldw_cli_modular.tcss, built from the css/
# sources; nothing read Constants.css_content. Its own body documented this.
#
#
#
##########################################################################################################################

##########################################################################################################################


##########################################################################################################################


##########################################################################################################################
#
#
#
LLAMAFILE_SERVER_ARGS_HELP_TEXT = """
[bold cyan]--- Server & Model Params ---[/]

[bold]Simple 'Just Get Me Up And Running': -ngl 99 -fa -c 8192[/]

--threads N, -t N: Set the number of threads to use during generation.

-tb N, --threads-batch N: Set the number of threads to use during batch and prompt processing. If not specified, the number of threads will be set to the number of threads used for generation.

-m FNAME, --model FNAME: Specify the path to the LLaMA model file (e.g., models/7B/ggml-model.gguf).

-a ALIAS, --alias ALIAS: Set an alias for the model. The alias will be returned in API responses.

-c N, --ctx-size N: Set the size of the prompt context. The default is 512, but LLaMA models were built with a context of 2048, which will provide better results for longer input/inference. The size may differ in other models, for example, baichuan models were build with a context of 4096.

-ngl N, --n-gpu-layers N: When compiled with appropriate support (currently CLBlast or cuBLAS), this option allows offloading some layers to the GPU for computation. Generally results in increased performance.

-mg i, --main-gpu i: When using multiple GPUs this option controls which GPU is used for small tensors for which the overhead of splitting the computation across all GPUs is not worthwhile. The GPU in question will use slightly more VRAM to store a scratch buffer for temporary results. By default GPU 0 is used. Requires cuBLAS.

-ts SPLIT, --tensor-split SPLIT: When using multiple GPUs this option controls how large tensors should be split across all GPUs. SPLIT is a comma-separated list of non-negative values that assigns the proportion of data that each GPU should get in order. For example, "3,2" will assign 60% of the data to GPU 0 and 40% to GPU 1. By default the data is split in proportion to VRAM but this may not be optimal for performance. Requires cuBLAS.

-b N, --batch-size N: Set the batch size for prompt processing. Default: 512.

--memory-f32: Use 32-bit floats instead of 16-bit floats for memory key+value. Not recommended.

--mlock: Lock the model in memory, preventing it from being swapped out when memory-mapped.

--no-mmap: Do not memory-map the model. By default, models are mapped into memory, which allows the system to load only the necessary parts of the model as needed.

--numa: Attempt optimizations that help on some NUMA systems.

--lora FNAME: Apply a LoRA (Low-Rank Adaptation) adapter to the model (implies --no-mmap). This allows you to adapt the pretrained model to specific tasks or domains.

--lora-base FNAME: Optional model to use as a base for the layers modified by the LoRA adapter. This flag is used in conjunction with the --lora flag, and specifies the base model for the adaptation.
-to N, --timeout N: Server read/write timeout in seconds. Default 600.

--host: Set the hostname or ip address to listen. Default 127.0.0.1.

--port: Set the port to listen. Default: 8080.

--path: path from which to serve static files (default examples/server/public)

--api-key: Set an api key for request authorization. By default the server responds to every request. With an api key set, the requests must have the Authorization header set with the api key as Bearer token. May be used multiple times to enable multiple valid keys.

--api-key-file: path to file containing api keys delimited by new lines. If set, requests must include one of the keys for access. May be used in conjunction with --api-key's.

--embedding: Enable embedding extraction, Default: disabled.
-np N, --parallel N: Set the number of slots for process requests (default: 1)

-cb, --cont-batching: enable continuous batching (a.k.a dynamic batching) (default: disabled)

-spf FNAME, --system-prompt-file FNAME Set a file to load "a system prompt (initial prompt of all slots), this is useful for chat applications. See more

--mmproj MMPROJ_FILE: Path to a multimodal projector file for LLaVA.

--grp-attn-n: Set the group attention factor to extend context size through self-extend(default: 1=disabled), used together with group attention width --grp-attn-w

--grp-attn-w: Set the group attention width to extend context size through self-extend(default: 512), used together with group attention factor --grp-attn-n

[italic]Obtained from: https://github.com/Mozilla-Ocho/llamafile/blob/main/llama.cpp/server/README.md[/]
"""


#
# MLX-LM Server Arguments Help Text
MLX_LM_SERVER_ARGS_HELP_TEXT = """
[bold cyan]--- MLX-LM Server Arguments ---[/]

options:
  --adapter-path ADAPTER_PATH
                        Optional path for the trained adapter weights and
                        config.
  


  --temp TEMP           Default sampling temperature (default: 0.0)
  --top-p TOP_P         Default nucleus sampling top-p (default: 1.0)
  --top-k TOP_K         Default top-k sampling (default: 0, disables top-k)
  --min-p MIN_P         Default min-p sampling (default: 0.0, disables min-p)
  --max-tokens MAX_TOKENS
                        Default maximum number of tokens to generate (default:
                        512)
  --chat-template-args CHAT_TEMPLATE_ARGS
                        A JSON formatted string of arguments for the
                        tokenizer's apply_chat_template, e.g.
                        '{"enable_thinking":false}'

[bold]--model MODEL[/]
  The path to the MLX model weights, tokenizer, and config
  (e.g., [italic]--model mlx-community/Qwen3-30B-A3B-4bit[/])

[bold]--host HOST[/]
  Host address to bind the server to (default: 127.0.0.1)
  (e.g., [italic]--host 0.0.0.0[/])

[bold]--port PORT[/]
  Port to run the server on (default: 8080)
  (e.g., [italic]--port 8000[/])

[bold]--draft-model DRAFT_MODEL[/]
    A model to be used for speculative decoding.
    (e.g., [italic]--draft-model mlx-community/Qwen3-0.6B-8bit[/])

[bold]--num-draft-tokens NUM_DRAFT_TOKENS[/]
    Number of tokens to draft when using speculative decoding.

[bold]--trust-remote-code[/]
  Enable trusting remote code for tokenizer
  
[bold]--chat-template CHAT_TEMPLATE[/]
    Specify a chat template for the tokenizer

[bold]--use-default-chat-template[/]
    Use the default chat template

[bold]--temperature TEMP[/]
  Sampling temperature (default: 0.8)
  (e.g., [italic]--temperature 0.7[/])

[bold]--top-p P[/]
  Top-p sampling (default: 0.9)
  (e.g., [italic]--top-p 0.95[/])

[bold]--top-k K[/]
  Top-k sampling (default: 40)
  (e.g., [italic]--top-k 50[/])

[bold]--min-p MIN_P[/]
    Default min-p sampling (default: 0.0, disables min-p)

[bold]--max-tokens N[/]
  Maximum number of tokens to generate (default: 100)
  (e.g., [italic]--max-tokens 512[/])

[bold]--chat-template-args CHAT_TEMPLATE_ARGS[/]
    A JSON formatted string of arguments for the tokenizer's apply_chat_template, e.g. '{"enable_thinking":false}'
"""

#: Worker group for the provider model-catalog refresh. One constant so the
#: Default startup-splash duration in seconds. One source of truth for the
#: SplashScreen constructor, the loaded-config fallback, the app compose
#: fallback, the Settings viewer defaults, and the config.toml template
#: (injected via its placeholder) -- so none of them can drift apart
#: (Qodo review of PR #2329).
DEFAULT_SPLASH_DURATION_SECONDS: float = 7.0


#: dispatch sites and the worker-handler's acknowledgement set cannot drift
#: apart through a spelling change — exclusivity and event routing both key
#: off this exact string (Qodo review of PR #2131).
MODEL_CATALOG_REFRESH_WORKER_GROUP = "model-catalog-refresh"


# End of Constants.py
########################################################################################################################
