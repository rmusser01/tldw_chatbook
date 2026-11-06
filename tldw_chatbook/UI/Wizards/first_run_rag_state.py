"""Pure facts and copy for the first-run Search step (TASK-34100.13).

The step used to pick ``[embedding_config] default_model_id``, a legacy key
the RAG pipeline never reads, and the Summary then claimed "✓ RAG" for it
(fulltrack-01). The runtime resolves its embedding model from the active RAG
profile, and new Console conversations decide whether to search the Library
from ``[chat_defaults] rag_auto_retrieve_on_send``. This module holds the
step's one real control (that flag), the read-only description of the model
RAG actually resolves, and the Summary row built from both.

Everything here is I/O-free except ``resolve_active_embedding_model``, which
callers run off the event loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlsplit

#: Where the embedding model is changed. Settings groups the RAG category
#: under "Domain Defaults" (settings_screen.py's rail groups).
SEARCH_SETTINGS_PATH = "Settings ▸ Domain Defaults ▸ RAG"

#: The optional extras package that adds meaning (embedding) search.
RAG_EXTRAS_PACKAGE = "tldw_chatbook[embeddings_rag]"

#: The one config key the step writes, and the Console reads for every new
#: conversation (Console_Modules/session.py, Chat/console_runtime.py).
AUTO_RETRIEVE_SECTION = "chat_defaults"
AUTO_RETRIEVE_KEY = "rag_auto_retrieve_on_send"

SEARCH_TITLE = "Search your documents (optional)"
SEARCH_SUBTITLE = (
    "chatbook can look up passages from your Library (documents, notes, "
    "media) and give them to the model while you chat. Nothing is indexed "
    "until you add something."
)
SEARCH_CHOICE_LABEL = "When you chat:"
SEARCH_ON_REQUEST_LABEL = "Search my Library only when I ask (default)"
SEARCH_AUTOMATIC_LABEL = "Search my Library automatically"

#: AC#5, owner ruling 2026-10-20: without the extras the automatic choice is
#: shown unavailable and says why, directly under the control. Console's
#: automatic search runs in ``rag`` mode, which the Library search service
#: answers with a blocked outcome when the RAG runtime is absent
#: (``library_local_rag_search_service._rag_mode_unavailable_recovery_state``),
#: so the send pauses as "Not sent: Library search failed"
#: (``console_turn_preparation.LIBRARY_PAUSE_COPY``). Library ▸ Search /
#: RAG's keyword Search mode works without the extras. A keyword fallback
#: for Console's automatic search is TASK-34100.35, not this step.
SEARCH_AUTOMATIC_UNAVAILABLE_LABEL = f"{SEARCH_AUTOMATIC_LABEL} — unavailable"
SEARCH_UNAVAILABLE_COPY = (
    "Unavailable: Console's automatic search needs the optional search "
    "extras, which aren't installed. Until then, search your Library "
    "yourself from Library ▸ Search / RAG."
)
#: The same, for a re-run where automatic search was saved before.
SEARCH_UNAVAILABLE_SAVED_ON_COPY = (
    "Unavailable: automatic search is on, but it needs the optional search "
    "extras, which aren't installed, so each Console message stops at "
    '"Library search failed" before it is sent. Choose "only when I ask" '
    "to chat without it."
)
#: The secondary line under the package name: no installer is assumed.
SEARCH_EXTRAS_INSTALL_COPY = (
    "Install it with the tool you installed chatbook with, restart chatbook, "
    f"then turn on Automatic retrieval in {SEARCH_SETTINGS_PATH}."
)


def search_unavailable_copy(*, saved_automatic: bool) -> str:
    """Why automatic search is unavailable without the extras.

    Args:
        saved_automatic: Whether automatic search is already the saved choice.

    Returns:
        The line shown directly under the "When you chat" control.
    """
    if saved_automatic:
        return SEARCH_UNAVAILABLE_SAVED_ON_COPY
    return SEARCH_UNAVAILABLE_COPY


@dataclass(frozen=True)
class EmbeddingModelFacts:
    """What the step says about the embedding model RAG resolves.

    Attributes:
        name: The model id exactly as the active profile stores it.
        cloud: True when indexing sends text to a hosted API.
        local_server: True for an ``openai/`` model served from this machine.
    """

    name: str
    cloud: bool
    local_server: bool = False


def _loopback(base_url: str | None) -> bool:
    if not base_url:
        return False
    host = (urlsplit(base_url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"} or host.startswith("127.")


def describe_embedding_model(
    model: str, *, base_url: str | None = None
) -> EmbeddingModelFacts:
    """Classify a profile embedding model the way the embeddings wrapper does.

    ``RAG_Search/simplified/embeddings_wrapper.py`` treats an ``openai/``
    prefix as an OpenAI(-compatible) API model and everything else as a
    model that runs on this computer.

    Args:
        model: The resolved ``RAGConfig.embedding.model``.
        base_url: The resolved ``RAGConfig.embedding.base_url``, if any.

    Returns:
        The facts the step and the Summary render.
    """
    name = str(model or "").strip()
    if not name.startswith("openai/"):
        return EmbeddingModelFacts(name=name, cloud=False)
    if _loopback(base_url):
        return EmbeddingModelFacts(name=name, cloud=False, local_server=True)
    return EmbeddingModelFacts(name=name, cloud=True)


def embedding_location_phrase(facts: EmbeddingModelFacts) -> str:
    """Where indexing runs, in the words the step and the Summary share."""
    if facts.cloud:
        return "OpenAI API"
    if facts.local_server:
        return "a server on this computer"
    return "on this computer"


def embedding_model_line(facts: EmbeddingModelFacts | None) -> str:
    """The step's read-only model line (no control: changing it rebuilds the index)."""
    if facts is None or not facts.name:
        return f"Embedding model: see {SEARCH_SETTINGS_PATH}."
    return (
        f"Embedding model: {facts.name} · {embedding_location_phrase(facts)} · "
        f"change it in {SEARCH_SETTINGS_PATH}"
    )


def cloud_privacy_copy() -> str:
    """Settings' own Library privacy sentence, reused verbatim for a cloud model."""
    from tldw_chatbook.UI.Screens.settings_library_rag_defaults import (
        LIBRARY_CLOUD_MODEL_PRIVACY_COPY,
    )

    return LIBRARY_CLOUD_MODEL_PRIVACY_COPY


def resolve_active_embedding_model() -> EmbeddingModelFacts | None:
    """Read the embedding model RAG would use right now (blocking I/O).

    Returns:
        The facts, or None when the RAG config cannot be resolved (for
        example, a broken profile); callers then name the Settings path
        instead of a model.
    """
    try:
        from tldw_chatbook.RAG_Search.simplified.active_config import (
            resolve_active_rag_config,
        )

        embedding = resolve_active_rag_config().embedding
    except Exception:  # noqa: BLE001 -- a read-only label must never break setup
        return None
    return describe_embedding_model(
        str(getattr(embedding, "model", "") or ""),
        base_url=getattr(embedding, "base_url", None),
    )


def read_auto_retrieve(app_config: Mapping[str, Any]) -> bool:
    """The persisted flag, read with Settings' own strict boolean rules."""
    from tldw_chatbook.UI.Screens.settings_library_rag_defaults import (
        load_rag_auto_retrieve_on_send,
    )

    return load_rag_auto_retrieve_on_send(app_config)


def build_search_commit(
    *, auto_retrieve: bool, current: bool
) -> dict[str, dict[str, Any]]:
    """The delta the step writes: nothing unless the choice changed.

    Args:
        auto_retrieve: The choice on screen.
        current: The persisted value.

    Returns:
        ``{}`` when they match, else the one ``chat_defaults`` key.
    """
    if bool(auto_retrieve) == bool(current):
        return {}
    return {AUTO_RETRIEVE_SECTION: {AUTO_RETRIEVE_KEY: bool(auto_retrieve)}}


def automatic_download_note(facts: EmbeddingModelFacts | None) -> str:
    """What turning automatic search on can fetch, for a model run here.

    Live (TASK-34100.13): on a fresh profile the first automatic search built
    the RAG service and downloaded all-MiniLM-L6-v2 (87 MB) from Hugging
    Face in the background; that first send paused while it happened.
    Review G13-R1-02: Console's start-up budget is 60 s
    (``LIBRARY_INITIALIZATION_TIMEOUT_SECONDS``), so a slower download still
    pauses the send at "Library search timed out", with Retry.
    """
    if facts is None or facts.cloud or facts.local_server or not facts.name:
        return ""
    return (
        f"If {facts.name} isn't on this computer yet, your first automatic "
        "search downloads it: that first message may wait for it, or on a "
        'slow connection stop at "Library search timed out" until you choose '
        "Retry."
    )


#: The Summary's half-sentence for missing extras (AC#5).
SEARCH_EXTRAS_MISSING_SUMMARY = "the optional search extras aren't installed"


def search_summary_detail(
    *,
    auto_retrieve: bool,
    facts: EmbeddingModelFacts | None,
    extras_installed: bool = True,
) -> str:
    """The Summary's Search row detail: the flag, then the resolved model.

    Without the extras no model runs, so the row says automatic search is
    unavailable instead of naming one (AC#5).
    """
    if not extras_installed:
        if auto_retrieve:
            return (
                "automatic when you chat — unavailable: "
                f"{SEARCH_EXTRAS_MISSING_SUMMARY}"
            )
        return (
            "only when you ask · automatic search unavailable: "
            f"{SEARCH_EXTRAS_MISSING_SUMMARY}"
        )
    when = "automatic when you chat" if auto_retrieve else "only when you ask"
    if facts is None or not facts.name:
        return when
    return f"{when} · {facts.name} ({embedding_location_phrase(facts)})"
