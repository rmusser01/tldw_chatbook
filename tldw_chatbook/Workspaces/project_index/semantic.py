"""Resolve a selected embedding catalog entry and issue finite project scope."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from tldw_chatbook.RAG_Search.projection_scope import (
    ProjectProjectionRequired,
    _issue_carrier,
)
from tldw_chatbook.Workspaces.project_index.models import (
    IndexCandidate,
    ProjectIndexScope,
    ProjectSemanticSettings,
)
from .semantic_vectors import decode_vector, encode_vector


def fuse_candidates(lexical, semantic):
    """Fuse two bounded ranked lists, retaining one immutable chunk identity."""
    scores = {}
    candidates = {}
    for ranked in (lexical, semantic):
        for item in ranked:
            identity = (
                item.relative_path, item.file_identity, item.content_sha256,
                item.line_start, item.line_end,
            )
            scores[identity] = scores.get(identity, 0.0) + 1.0 / (60 + item.rank + 1)
            candidates.setdefault(identity, item)
    ordered = sorted(scores, key=lambda identity: (-scores[identity], identity))[:100]
    return tuple(candidates[identity] for identity in ordered)


def rank_semantic(store, scope, state, artifact_fingerprint, query_vector, exclusions, *, limit=100):
    """Stream compatible filtered members through an exact cosine heap."""
    if type(limit) is not int or not 1 <= limit <= 100:  # noqa: E721
        raise ValueError("invalid_semantic_limit")
    generation = store.semantic_ready_generation(scope, state, artifact_fingerprint)
    if generation is None:
        return None
    if generation.chunk_count == 0:
        return ()
    query = decode_vector(encode_vector(query_vector), generation.dimension)
    query_norm = math.sqrt(math.fsum(value * value for value in query))
    if not query_norm:
        raise ValueError("invalid_semantic_query")
    heap = []
    after = 0
    while True:
        page = store.semantic_member_page(
            scope, state, artifact_fingerprint, exclusions, after=after
        )
        if page is None:
            return None
        if not page:
            break
        for member in page:
            vector = decode_vector(member.vector, generation.dimension)
            norm = math.sqrt(math.fsum(value * value for value in vector))
            if not norm:
                continue
            score = math.fsum(a * b for a, b in zip(query, vector)) / (query_norm * norm)
            item = (score, -member.ordinal, member)
            if len(heap) < limit:
                heapq.heappush(heap, item)
            elif item[:2] > heap[0][:2]:
                heapq.heapreplace(heap, item)
        after = page[-1].ordinal
    ranked = sorted(heap, key=lambda item: (-item[0], -item[1]))
    return tuple(
        IndexCandidate(
            item[2].relative_path, item[2].identity,
            item[2].content_sha256, item[2].line_start,
            item[2].line_end, rank,
        )
        for rank, item in enumerate(ranked)
    )


def _fingerprint(*values) -> str:
    return (
        "sha256:v1:"
        + hashlib.sha256(
            json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


@dataclass(frozen=True)
class _Target:
    model_name: str
    base_url: str | None
    local_files_only: bool | None
    artifact_fingerprint: str
    config: dict
    api_key: str | None = field(repr=False)


@dataclass
class _Parent:
    stack: ExitStack
    pin: object
    operation: object


def _catalog_spec(model_id: str):
    """Validate only the selected current entry, with configured precedence."""
    from tldw_chatbook.Embeddings.Embeddings_Lib import (
        EmbeddingConfigSchema,
        get_common_embedding_models,
    )
    from tldw_chatbook.config import load_cli_config_and_ensure_existence

    configured = (
        load_cli_config_and_ensure_existence()
        .get("embedding_config", {})
        .get("models", {})
    )
    try:
        selected = (
            configured[model_id]
            if model_id in configured
            else get_common_embedding_models()[model_id]
        )
        return EmbeddingConfigSchema(
            default_model_id=model_id, models={model_id: selected}
        ).models[model_id]
    except (KeyError, TypeError, ValueError):
        raise ProjectProjectionRequired(
            "project_embedding_configuration_invalid"
        ) from None


def _resolved_spec(model_id: str):
    from tldw_chatbook.Embeddings.Embeddings_Lib import HFModelCfg, OpenAICfg

    spec = _catalog_spec(model_id)
    if type(spec) is OpenAICfg:
        base = (
            str(spec.base_url).rstrip("/")
            if spec.base_url
            else "https://api.openai.com/v1"
        )
        parsed = urlsplit(base)
        # Routing-sensitive queries cannot be removed safely from identity;
        # refuse them instead of silently consenting to a different endpoint.
        if (
            parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or not parsed.hostname
            or parsed.scheme not in ("http", "https")
            or any(
                marker in spec.model_name_or_path for marker in ("://", "?", "#", "@")
            )
        ):
            raise ProjectProjectionRequired("project_embedding_destination_unsupported")
        return spec, base + "/embeddings"
    if type(spec) is HFModelCfg:
        if any(marker in spec.model_name_or_path for marker in ("://", "?", "#", "@")):
            raise ProjectProjectionRequired("project_embedding_configuration_invalid")
        return spec.model_copy(update={"local_files_only": True}), (
            "huggingface",
            spec.model_name_or_path,
            spec.revision,
        )
    raise ProjectProjectionRequired("project_embedding_provider_unsupported")


def _settings_for_spec(model_id: str, spec, destination) -> ProjectSemanticSettings:
    safe_config = spec.model_dump(mode="json", exclude={"api_key", "cache_dir"})
    if spec.provider == "huggingface":
        selected_root = spec.cache_dir or os.environ.get("HF_HUB_CACHE")
        safe_config["cache_root_fingerprint"] = (
            _fingerprint("cache-root", str(Path(selected_root).expanduser().absolute()))
            if selected_root
            else None
        )
    return ProjectSemanticSettings(
        provider_id=spec.provider,
        model_id=model_id,
        destination_fingerprint=_fingerprint("destination", destination),
        config_fingerprint=_fingerprint("configuration", model_id, safe_config),
        approved_destination_fingerprint=None,
        provider_ref="provider:v1:" + spec.provider,
        config_ref="model:v1:" + model_id,
    )


def _settings_for_project_model(model_id: str) -> ProjectSemanticSettings:
    """Build secret-free candidate settings; approval remains a separate act."""
    spec, destination = _resolved_spec(model_id)
    return _settings_for_spec(model_id, spec, destination)


def _local_snapshot(spec) -> Path:
    """Select a complete existing snapshot using the actual factory rule."""
    from tldw_chatbook.Embeddings.Embeddings_Lib import existing_local_hf_snapshot

    try:
        return existing_local_hf_snapshot(spec)
    except (FileNotFoundError, ImportError, OSError, ValueError):
        raise ProjectProjectionRequired("project_local_model_setup_required") from None


def _target(spec, destination) -> _Target:
    if spec.provider == "openai":
        from tldw_chatbook.config import without_ciphertext

        key = without_ciphertext(
            os.environ.get("OPENAI_API_KEY")
        ) or without_ciphertext(spec.api_key)
        if not key and spec.base_url is None:
            raise ProjectProjectionRequired("project_embedding_provider_setup_required")
        config = spec.model_dump(mode="json")
        config["api_key"] = key
        return _Target(
            "openai/" + spec.model_name_or_path,
            str(spec.base_url) if spec.base_url else None,
            None,
            _fingerprint("hosted", destination, spec.model_name_or_path),
            config,
            key,
        )
    from tldw_chatbook.RAG_Search.model_recovery import _closure

    snapshot = _local_snapshot(spec)
    try:
        closure = _closure(snapshot)
    except (OSError, ValueError):
        raise ProjectProjectionRequired("project_local_model_setup_required") from None
    config = spec.model_dump(mode="json")
    config["model_name_or_path"] = str(snapshot)
    config["local_files_only"] = True
    config["cache_dir"] = None
    return _Target(
        str(snapshot),
        None,
        True,
        _fingerprint("local-artifact", str(snapshot), closure),
        config,
        None,
    )


def _require_current_catalog(settings: ProjectSemanticSettings):
    """Return one current selected target only when its approved identity matches."""
    spec, destination = _resolved_spec(settings.model_id)
    candidate = _settings_for_spec(settings.model_id, spec, destination)
    if (
        settings.provider_id != candidate.provider_id
        or settings.provider_ref != candidate.provider_ref
        or settings.config_ref != candidate.config_ref
        or settings.destination_fingerprint != candidate.destination_fingerprint
        or settings.config_fingerprint != candidate.config_fingerprint
        or settings.approved_destination_fingerprint
        != candidate.destination_fingerprint
    ):
        raise ProjectProjectionRequired("project_embedding_configuration_changed")
    return spec, destination


def _issue_project_projection_scope(service, scope, settings, semantic_revision):
    """Return one code-issued carrier with an actual retained project parent.

    Callers retire native work before closing this carrier. A later operation
    issues a fresh carrier; a compatible wrapper may keep its idle model but
    no idle storage lease or cached carrier grants authority.
    """
    if (
        type(scope) is not ProjectIndexScope
        or type(settings) is not ProjectSemanticSettings
        or type(semantic_revision) is not int  # noqa: E721 - reject bool revisions
        or semantic_revision < 1
    ):
        raise ProjectProjectionRequired()
    service._require_projection_open(scope.key)
    _require_current_catalog(settings)
    stack = ExitStack()
    try:
        stack.enter_context(service.registry_connection_scope())
        pin = stack.enter_context(service._factory.retain(scope.key))
        stack.enter_context(pin.bind())
        with service._store(scope.key, cleanup_only=True) as store:
            with service._read_fenced(store, scope.key) as authority:
                if authority is None or authority.key != scope.key:
                    raise ProjectProjectionRequired()
                status = store.load_status(scope.key)
                state = store.load_semantic_state(scope)
                if (
                    status.scope != scope
                    or not state.enabled
                    or state.state not in ("Pending", "Building", "Ready")
                    or state.settings != settings
                    or state.semantic_revision != semantic_revision
                ):
                    raise ProjectProjectionRequired()
        spec, destination = _require_current_catalog(settings)
        target = _target(spec, destination)
        service._retire_registry_for_model()
        parent = _Parent(stack, pin, service._operation_local.operation)
        return _issue_carrier(
            service=service,
            scope=scope,
            settings=settings,
            revision=semantic_revision,
            target=target,
            parent=parent,
        )
    except BaseException:
        stack.close()
        raise
