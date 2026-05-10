from __future__ import annotations

import base64
import json

import pytest

from tldw_chatbook.Sync_Interop.crypto import generate_dataset_key, unwrap_recovery_bundle
from tldw_chatbook.Sync_Interop.key_recovery_service import SyncKeyRecoveryService
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository

pytestmark = pytest.mark.asyncio


class FakeRecoveryServer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def store_v2_recovery_bundle(
        self,
        *,
        dataset_id,
        device_id=None,
        wrapped_key_blob,
        kdf_metadata,
        recovery_hint=None,
        key_purpose="dataset_recovery",
        rotation_of_key_record_id=None,
    ):
        self.calls.append(
            {
                "dataset_id": dataset_id,
                "device_id": device_id,
                "wrapped_key_blob": wrapped_key_blob,
                "kdf_metadata": dict(kdf_metadata),
                "recovery_hint": recovery_hint,
                "key_purpose": key_purpose,
                "rotation_of_key_record_id": rotation_of_key_record_id,
            }
        )
        return {
            "key_record_id": "key-record-1",
            "dataset_id": dataset_id,
            "device_id": device_id,
            "key_purpose": key_purpose,
            "recovery_hint": recovery_hint,
            "created_at": "2026-05-10T00:00:00Z",
        }


def _repo_with_profile(
    tmp_path,
    *,
    profile_mode: str = "local_first",
    device_id: str | None = "device-1",
    dataset_id: str | None = "dataset-1",
) -> SyncStateRepository:
    repo = SyncStateRepository(tmp_path / "sync_state.db")
    repo.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        profile_mode=profile_mode,
        device_id=device_id,
        dataset_id=dataset_id,
        dataset_cursors={"sync_v2": "7"},
        capabilities={"supported_domains": ["notes"]},
        dry_run_metadata={"dry_run": True},
    )
    return repo


async def test_key_recovery_service_wraps_dataset_key_and_stores_sanitized_metadata(tmp_path):
    dataset_key = generate_dataset_key()
    repo = _repo_with_profile(tmp_path)
    server = FakeRecoveryServer()
    service = SyncKeyRecoveryService(
        server_service=server,
        state_repository=repo,
        dataset_keys={"dataset-1": dataset_key},
    )

    result = await service.configure_recovery(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        recovery_secret="correct horse battery staple",
        recovery_hint="personal laptop",
    )

    assert result == {
        "dataset_id": "dataset-1",
        "device_id": "device-1",
        "key_record_id": "key-record-1",
        "key_purpose": "dataset_recovery",
        "recovery_hint": "personal laptop",
        "key_recovery_configured": True,
    }
    assert len(server.calls) == 1
    call = server.calls[0]
    assert call["dataset_id"] == "dataset-1"
    assert call["device_id"] == "device-1"
    assert call["key_purpose"] == "dataset_recovery"
    assert call["recovery_hint"] == "personal laptop"
    assert call["rotation_of_key_record_id"] is None
    assert isinstance(call["wrapped_key_blob"], str)
    assert call["kdf_metadata"]["algorithm"] == "scrypt"

    encoded_key = base64.b64encode(dataset_key).decode("ascii")
    assert encoded_key not in call["wrapped_key_blob"]
    assert dataset_key.hex() not in call["wrapped_key_blob"]
    assert "correct horse battery staple" not in str(call)
    assert "correct horse battery staple" not in str(result)
    assert "wrapped_key_blob" not in result
    assert "kdf_metadata" not in result

    assert unwrap_recovery_bundle(
        {
            "wrapped_key_blob": call["wrapped_key_blob"],
            "kdf_metadata": call["kdf_metadata"],
            "recovery_hint": call["recovery_hint"],
            "key_purpose": call["key_purpose"],
        },
        recovery_secret="correct horse battery staple",
    ) == dataset_key

    profile = repo.get_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    )
    metadata = profile["dry_run_metadata"]
    serialized_metadata = json.dumps(metadata, sort_keys=True)
    assert metadata["dry_run"] is True
    assert metadata["key_recovery_configured"] is True
    assert metadata["key_recovery_available"] is True
    assert metadata["key_recovery_key_record_id"] == "key-record-1"
    assert metadata["key_recovery_hint"] == "personal laptop"
    assert metadata["key_recovery_key_purpose"] == "dataset_recovery"
    assert "wrapped_key_blob" not in serialized_metadata
    assert "kdf_metadata" not in serialized_metadata
    assert "correct horse battery staple" not in serialized_metadata
    assert encoded_key not in serialized_metadata
    assert dataset_key.hex() not in serialized_metadata


async def test_key_recovery_service_requires_local_first_profile_device_dataset_and_key(tmp_path):
    server = FakeRecoveryServer()
    service = SyncKeyRecoveryService(
        server_service=server,
        state_repository=_repo_with_profile(tmp_path / "server-front", profile_mode="server_frontend"),
        dataset_keys={"dataset-1": generate_dataset_key()},
    )

    with pytest.raises(ValueError, match="local_first"):
        await service.configure_recovery(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            recovery_secret="secret",
        )

    service = SyncKeyRecoveryService(
        server_service=server,
        state_repository=_repo_with_profile(tmp_path / "missing-ids", device_id=None),
        dataset_keys={"dataset-1": generate_dataset_key()},
    )

    with pytest.raises(ValueError, match="device_id and dataset_id"):
        await service.configure_recovery(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            recovery_secret="secret",
        )

    service = SyncKeyRecoveryService(
        server_service=server,
        state_repository=_repo_with_profile(tmp_path / "missing-key"),
        dataset_keys={},
    )

    with pytest.raises(ValueError, match="dataset key"):
        await service.configure_recovery(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            recovery_secret="secret",
        )

    assert server.calls == []
