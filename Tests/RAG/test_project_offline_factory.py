"""Real offline embedding factory qualification against an existing local snapshot."""

import os
import socket

import pytest
from Tests.private_profile import private_profile_test

from tldw_chatbook.Embeddings.Embeddings_Lib import EmbeddingFactory


def test_actual_factory_uses_complete_existing_snapshot_without_network(monkeypatch):
    model_id = os.environ.get("TLDW_TEST_OFFLINE_EMBEDDING_MODEL")
    if not model_id:
        pytest.skip("No pre-existing offline embedding snapshot selected")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    attempted = []

    def block_network(self, address):
        attempted.append(address)
        raise AssertionError("offline_embedding_network_attempted")

    monkeypatch.setattr(socket.socket, "connect", block_network)
    factory = EmbeddingFactory(
        {
            "default_model_id": "offline",
            "models": {
                "offline": {
                    "provider": "huggingface",
                    "model_name_or_path": model_id,
                    "device": "cpu",
                }
            },
        },
        allow_dynamic_hf=False,
        local_files_only=True,
    )
    try:
        vectors = factory.embed(["synthetic alpha", "synthetic beta"])
        assert vectors.shape[0] == 2
        assert vectors.shape[1] > 0
        assert not attempted
    finally:
        factory.close()


@private_profile_test
async def test_real_project_scope_loads_and_reuses_offline_factory(
    tmp_path, monkeypatch, request
):
    """Two independent project admissions reuse one real local model handle."""
    import asyncio
    from contextlib import contextmanager
    from dataclasses import replace

    from tldw_chatbook import config
    from tldw_chatbook.RAG_Search.simplified.embeddings_wrapper import (
        EmbeddingsServiceWrapper,
    )
    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
    from tldw_chatbook.Workspaces.project_index.models import (
        ProjectIndexAuthority,
        ProjectIndexKey,
        ProjectIndexPolicy,
        locator_fingerprint,
        root_identity_fingerprint,
    )
    from tldw_chatbook.Workspaces.project_index.runtime import _AdmittedStoreFactory
    from tldw_chatbook.Workspaces.project_index.scanner import ProjectScanner
    from tldw_chatbook.Workspaces.project_index.semantic import (
        _issue_project_projection_scope,
        _settings_for_project_model,
    )
    from tldw_chatbook.Workspaces.project_index.service import ProjectIndexService

    model_id = os.environ.get("TLDW_TEST_OFFLINE_EMBEDDING_MODEL")
    if not model_id:
        pytest.skip("No pre-existing offline embedding snapshot selected")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    current = config.load_cli_config_and_ensure_existence

    def configured(*args, **kwargs):
        data = dict(current(*args, **kwargs))
        embedding = dict(data.get("embedding_config", {}))
        models = dict(embedding.get("models", {}))
        models["minilm-local"] = {
            "provider": "huggingface",
            "model_name_or_path": model_id,
            "device": "cpu",
        }
        embedding["models"] = models
        data["embedding_config"] = embedding
        return data

    monkeypatch.setattr(config, "load_cli_config_and_ensure_existence", configured)
    attempted = []

    import requests

    def block_network(self, method, url, **kwargs):
        attempted.append(url)
        raise AssertionError("offline_embedding_http_attempted")

    monkeypatch.setattr(requests.sessions.Session, "request", block_network)

    def run():
        root = tmp_path / "project"
        root.mkdir()
        chain = capture_directory_chain(root)
        factory = _AdmittedStoreFactory()
        key = ProjectIndexKey(
            factory.identity,
            "workspace",
            "binding",
            locator_fingerprint(root),
            root_identity_fingerprint(chain),
        )
        authority = ProjectIndexAuthority(key, root, chain, (), 1)

        @contextmanager
        def fence(selected):
            yield authority if selected == key else None

        service = ProjectIndexService(
            factory,
            ProjectScanner(),
            lambda selected: authority if selected == key else None,
            fence,
        )
        with service._store(key) as store:
            scope = store.configure(
                key,
                ProjectIndexPolicy((), (".txt",)),
                enabled=True,
                expected_scope=store.load_status(key).scope,
            )
            lease = service._lifetime(store, key)
            try:
                candidate = _settings_for_project_model("minilm-local")
                settings = replace(
                    candidate,
                    approved_destination_fingerprint=candidate.destination_fingerprint,
                )
                state = store.configure_semantic(
                    scope, settings, enabled=True, expected_semantic_revision=0
                )
            finally:
                lease.close()
        issued = _issue_project_projection_scope(
            service, scope, settings, state.semantic_revision
        )
        wrapper = EmbeddingsServiceWrapper(
            model_name=issued.model_name,
            api_key=issued.api_key,
            base_url=issued.base_url,
            device="cpu",
            execution_scope=issued,
            local_files_only=True,
        )
        first = wrapper.create_embeddings(["synthetic first"])
        assert first.shape[0] == 1 and first.shape[1] > 0
        record = wrapper.factory._cache["default"]
        issued.close()
        assert not service._operations
        later = _issue_project_projection_scope(
            service, scope, settings, state.semantic_revision
        )
        wrapper.rebind_execution_scope(later)
        second = wrapper.create_embeddings(["synthetic second"])
        assert second.shape == first.shape
        assert wrapper.factory._cache["default"] is record
        assert not attempted
        wrapper.close()
        later.close()
        asyncio.run(service.aclose())

    await asyncio.to_thread(run)
