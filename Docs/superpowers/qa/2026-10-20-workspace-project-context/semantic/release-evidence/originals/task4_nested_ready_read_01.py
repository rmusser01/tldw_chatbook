"""One bounded read-only inspection of the retained nested Ready generation."""

import asyncio
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import patch


REPO = Path.cwd().resolve()
SDD = Path(__file__).resolve().parent
BASE = SDD.parents[1] / "k5-scale"
PROFILE = BASE / "stable-profile"
OUTPUT = SDD / "task4-nested-ready-read-01-child.json"


def require_origins():
    result = {}
    for name in ("tldw_chatbook", "Tests"):
        spec = importlib.util.find_spec(name)
        assert spec and spec.origin
        actual = Path(spec.origin).resolve()
        expected = REPO / name / "__init__.py"
        assert actual == expected, (name, str(actual), str(expected))
        result[name] = str(actual)
    return result


ORIGINS = require_origins()

from Tests import conftest as _test_bootstrap  # noqa: E402,F401
from Tests.UI.app_factory import _build_test_app, drain_created_dirs, drain_created_runtimes  # noqa: E402
from tldw_chatbook import config  # noqa: E402
from tldw_chatbook.Backup_Recovery.isolated_restore import verified_data_identity  # noqa: E402
from tldw_chatbook.DB.Workspace_Project_Index_DB import index_store_path  # noqa: E402
from tldw_chatbook.Workspaces.project_index.runtime import (  # noqa: E402
    close_project_index_service, get_project_index_service, project_index_key,
)


def blocked(*_args, **_kwargs):
    raise AssertionError("read_only_diagnostic_http_attempted")


def inspect(service, key, launch):
    path = index_store_path(service._factory.data_root, key)
    group = path.parent.parent
    assert path.exists() and group.exists()
    assert str(group).casefold() == str(Path(launch["cache"])).casefold()
    status = service.status(key)
    semantic = service.semantic_state(key)
    policy = service.policy(status.scope)
    with service._store(key, cleanup_only=True) as store:
        with service._read_fenced(store, key) as authority:
            assert authority is not None
            with store.connection() as conn:
                control = conn.execute(
                    "SELECT namespace,epoch,processing_revision,enabled,state,"
                    "job_revision,manifest_revision FROM control WHERE singleton=1"
                ).fetchone()
                semantic_control = conn.execute(
                    "SELECT namespace,epoch,semantic_revision,enabled,state,reason,"
                    "ready_generation_id FROM semantic_control WHERE singleton=1"
                ).fetchone()
                generation = conn.execute(
                    "SELECT generation_id,namespace,epoch,processing_revision,"
                    "semantic_revision,provider_id,model_id,destination_fingerprint,"
                    "config_fingerprint,artifact_fingerprint,format_id,dimension,"
                    "manifest_revision,claim_token,checkpoint_chunk_id,state,chunk_count "
                    "FROM semantic_generations WHERE generation_id=?",
                    (semantic.ready_generation_id,),
                ).fetchone()
                assert generation is not None
                members = conn.execute(
                    "SELECT COUNT(*),MIN(ordinal),MAX(ordinal),MIN(length(vector)),"
                    "MAX(length(vector)) FROM semantic_members WHERE generation_id=?",
                    (semantic.ready_generation_id,),
                ).fetchone()
    fields = ("generation_id", "namespace", "epoch", "processing_revision",
              "semantic_revision", "provider_id", "model_id", "destination_fingerprint",
              "config_fingerprint", "artifact_fingerprint", "format_id", "dimension",
              "manifest_revision", "claim_token", "checkpoint_chunk_id", "state", "chunk_count")
    result = {
        "origins": ORIGINS,
        "cache_group": str(group), "cache_db": str(path),
        "status_all_fields": asdict(status),
        "keyword_policy": asdict(policy),
        "semantic_state_all_fields": asdict(semantic),
        "keyword_control": dict(control),
        "semantic_control": dict(semantic_control),
        "ready_generation": dict(zip(fields, generation, strict=True)),
        "ready_members": {"count": members[0], "min_ordinal": members[1],
                          "max_ordinal": members[2], "min_vector_bytes": members[3],
                          "max_vector_bytes": members[4]},
    }
    assert status.state == "Ready" and status.indexed == 20000
    assert semantic.state == "Ready" and semantic.enabled
    assert semantic.ready_generation_id == generation[0]
    assert generation[15] == "Ready" and generation[16] == members[0] == 20000
    assert generation[11] == 384 and generation[12] == status.manifest_revision
    assert members[1] == 1 and members[2] == 20000
    return result


async def main():
    preflight = json.loads((PROFILE / "preflight-2.json").read_text(encoding="utf-8"))
    launch = json.loads((BASE / "nested-launch.json").read_text(encoding="utf-8"))
    data = config.get_user_data_dir()
    assert str(data) == preflight["data"]
    assert list(verified_data_identity()) == preflight["verified_identity"]
    app = _build_test_app(user_data_dir=data)
    closed = False
    result = None
    try:
        binding = app.workspace_registry_service.get_runtime_binding(launch["key"]["binding_id"])
        assert binding is not None and binding.locator == launch["corpus"]
        service = get_project_index_service(app)
        key = project_index_key(app, launch["key"]["workspace_id"], binding.binding_id,
                                Path(launch["corpus"]))
        assert asdict(key) == launch["key"]
        result = await asyncio.to_thread(inspect, service, key, launch)
    finally:
        await close_project_index_service(app)
        await drain_created_runtimes()
        drain_created_dirs()
        closed = True
    result["app_service_and_fixture_retired"] = closed
    OUTPUT.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"phase": "complete", "result": str(OUTPUT),
                      "state": result["semantic_state_all_fields"]["state"],
                      "members": result["ready_members"]["count"]}), flush=True)


if __name__ == "__main__":
    import httpx
    import requests
    with (patch.object(requests.sessions.Session, "request", blocked),
          patch.object(httpx.Client, "send", blocked),
          patch.object(httpx.AsyncClient, "send", blocked)):
        asyncio.run(main())
