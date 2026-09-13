"""Retained RAG formats use config-only credential policies without source edits."""

import json
import shutil
from pathlib import Path

import pytest
import toml

from Tests.Backup_Recovery.test_home_citation_retirement import _run
from tldw_chatbook.Backup_Recovery import credentials
from tldw_chatbook.Backup_Recovery.models import FileMetadata, Inventory, StorageItem


@pytest.fixture(scope="module")
def definitions(tmp_path_factory):
    root = tmp_path_factory.mktemp("rag-credential-formats")
    _run(
        root,
        "fixtures",
        "success",
        script=r"""
import json
from dataclasses import asdict
from pathlib import Path
from tldw_chatbook.RAG_Search.config_profiles import ConfigProfileManager, ProfileConfig, ExperimentConfig
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
root = Path.home() / 'profiles'
manager = ConfigProfileManager(root)
profile = ProfileConfig(name='Credential fixture', description='Historical description', profile_type='custom', rag_config=RAGConfig())
profile.rag_config.embedding.api_key = 'synthetic-rag-secret'
manager.save_profile(profile)
blob = {'profiles': [profile.to_dict()], 'version': '1.0'}
for name in ('custom_profiles.json', 'custom_profiles.json.migrated'):
    (root / name).write_text(json.dumps(blob, default=str))
experiment = ExperimentConfig(experiment_id='fixture', name='Query history', metrics_to_track=['latency'])
# The actual writer serializes asdict; its Path-valued results_dir is not JSON
# serializable. Keep its supported null serialization here, without changing it.
folder = root / 'experiments' / 'fixture'
folder.mkdir(parents=True)
(folder / 'config.json').write_text(json.dumps(asdict(experiment)))
experiment.results_dir = folder
manager._current_experiment = experiment
manager._experiment_results[experiment.experiment_id] = []
manager.record_experiment_result('balanced', 'enc:historical query api_key=prose', {'latency': 1.2, 'api_key': 'historical metric', 'nested': {'token': 'arbitrary content'}})
manager.end_experiment()
print('retired and reopened')
""",
    )
    return root / "home" / "profiles"


def staged(tmp_path, source, relative):
    root = tmp_path / "stage"
    root.mkdir(mode=0o700)
    path = root / "anonymous-payload"
    shutil.copyfile(source, path)
    path.chmod(0o600)
    item = StorageItem(
        "rag.definitions",
        "profile:p:rag",
        path,
        "included",
        (),
        metadata=FileMetadata(
            1, "rag-root", relative, None, "file", 0o600, 0, "private"
        ),
    )
    return root, path, Inventory((item,), True, "scope", ())


@pytest.fixture(autouse=True)
def no_keyring(monkeypatch):
    def forbidden():
        pytest.fail("RAG metadata must not select keyring authority")

    monkeypatch.setattr(credentials, "_credential_store", forbidden)


@pytest.mark.parametrize(
    "relative", ["custom_profiles.json", "custom_profiles.json.migrated"]
)
@pytest.mark.parametrize("mode", ["exclude", "include", "rollback"])
def test_actual_legacy_profiles_preserve_source_and_apply_config_policy(
    tmp_path, definitions, relative, mode
):
    source = definitions / relative
    before = source.read_bytes()
    root, path, inventory = staged(tmp_path, source, relative)
    assert not credentials.process_credentials(
        root, inventory, mode=mode, encrypted=mode != "exclude"
    )
    assert source.read_bytes() == before
    assert (b"synthetic-rag-secret" in path.read_bytes()) == (mode != "exclude")
    assert (
        json.loads(path.read_bytes())["profiles"][0]["description"]
        == "Historical description"
    )
    if mode != "exclude":
        assert path.read_bytes() == before


@pytest.mark.parametrize("leaf", ["config.json", "results.json"])
@pytest.mark.parametrize("mode", ["exclude", "include", "rollback"])
def test_actual_experiment_history_is_not_blanket_sanitized(
    tmp_path, definitions, leaf, mode
):
    relative = "experiments/fixture/" + leaf
    source = definitions / relative
    before = source.read_bytes()
    root, path, inventory = staged(tmp_path, source, relative)
    assert not credentials.process_credentials(
        root, inventory, mode=mode, encrypted=mode != "exclude"
    )
    assert path.read_bytes() == source.read_bytes() == before
    if mode != "exclude":
        assert (
            json.loads((root / "credential-recovery.json").read_bytes())["records"]
            == []
        )


@pytest.mark.parametrize("mode", ["exclude", "include", "rollback"])
def test_installed_pipeline_toml_uses_config_fields(tmp_path, mode):
    bundled = (
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook/Config_Files/rag_pipelines.toml"
    )
    data = toml.loads(bundled.read_text())
    data["pipelines"]["plain"]["parameters"].update(
        api_key="synthetic-rag-secret", api_key_env_var="RAG_TEST_KEY"
    )
    source = tmp_path / "rag_pipelines.toml"
    source.write_text(toml.dumps(data))
    before = source.read_bytes()
    root, path, inventory = staged(tmp_path, source, source.name)
    assert not credentials.process_credentials(
        root, inventory, mode=mode, encrypted=mode != "exclude"
    )
    assert source.read_bytes() == before
    after = toml.loads(path.read_text())
    parameters = after["pipelines"]["plain"]["parameters"]
    assert ("api_key" in parameters) == (mode != "exclude")
    assert parameters["api_key_env_var"] == "RAG_TEST_KEY"
    assert (
        after["pipelines"]["plain"]["function"]
        == data["pipelines"]["plain"]["function"]
    )


@pytest.mark.parametrize(
    "relative,data",
    [
        ("custom_profiles.json", {"profiles": [1]}),
        ("experiments/e/config.json", {"provider": {"api_key": "unknown"}}),
        ("experiments/e/results.json", {"summary": {}, "detailed_results": "bad"}),
        ("rag_pipelines.toml", {"pipelines": []}),
        ("other.toml", {"pipelines": {}}),
    ],
)
def test_unknown_shape_remains_explicit_unsupported(tmp_path, relative, data):
    source = tmp_path / "input"
    source.write_text(
        toml.dumps(data) if relative.endswith(".toml") else json.dumps(data)
    )
    before = source.read_bytes()
    root, path, inventory = staged(tmp_path, source, relative)
    assert credentials.process_credentials(
        root, inventory, mode="exclude", encrypted=False
    ) == ("credential_rag_definition_format_unsupported:profile:p:rag",)
    assert path.read_bytes() == source.read_bytes() == before


@pytest.mark.parametrize(
    "relative", ["custom_profiles.json.migrated", "rag_pipelines.toml"]
)
@pytest.mark.parametrize("mode", ["exclude", "include", "rollback"])
def test_encrypted_config_values_use_existing_material_and_exact_locations(
    tmp_path, monkeypatch, relative, mode
):
    from tldw_chatbook.Utils import config_encryption

    calls = []

    def unlock(value):
        calls.append(value)
        return "synthetic-unlocked"

    monkeypatch.setattr(config_encryption, "unlock_recovery_value", unlock)
    if relative.endswith("toml"):
        data = {"pipelines": {"local": {"parameters": {"api_key": "enc:synthetic"}}}}
        location = ["pipelines", "local", "parameters", "api_key"]
    else:
        data = {
            "profiles": [{"rag_config": {"embedding": {"api_key": "enc:synthetic"}}}]
        }
        location = ["profiles", 0, "rag_config", "embedding", "api_key"]
    source = tmp_path / "input"
    source.write_text(
        toml.dumps(data) if relative.endswith("toml") else json.dumps(data)
    )
    root, path, inventory = staged(tmp_path, source, relative)
    assert not credentials.process_credentials(
        root, inventory, mode=mode, encrypted=mode != "exclude"
    )
    if mode == "exclude":
        assert not calls
        assert b"enc:synthetic" not in path.read_bytes()
    else:
        records = json.loads((root / "credential-recovery.json").read_bytes())[
            "records"
        ]
        assert calls == ["enc:synthetic"]
        assert len(records) == 1
        assert records[0]["location"] == location
        assert records[0]["file"] == path.name
        assert records[0]["value"] == "synthetic-unlocked"
        assert path.read_bytes() == source.read_bytes()


@pytest.mark.parametrize(
    "relative",
    ["custom_profiles.json", "experiments/e/config.json", "rag_pipelines.toml"],
)
@pytest.mark.parametrize("mode", ["exclude", "include", "rollback"])
@pytest.mark.parametrize(
    "content", [b'\xff{"truncated":', b'{"truncated":'], ids=["encoding", "syntax"]
)
def test_unreadable_known_formats_keep_raw_rollback_and_refuse_coverage(
    tmp_path, relative, mode, content
):
    source = tmp_path / "input"
    source.write_bytes(content)
    root, path, inventory = staged(tmp_path, source, relative)
    assert credentials.process_credentials(
        root, inventory, mode=mode, encrypted=mode != "exclude"
    ) == (
        "credential_format_unreadable"
        if mode == "rollback"
        else "credential_processing_unavailable",
    )
    assert path.read_bytes() == source.read_bytes()


def test_locked_pipeline_value_reports_omission_without_backend_error(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Utils import config_encryption

    def locked(value):
        raise ValueError("synthetic-sensitive-backend-text")

    monkeypatch.setattr(config_encryption, "unlock_recovery_value", locked)
    source = tmp_path / "input"
    source.write_text('[pipelines.local.parameters]\napi_key="enc:synthetic"\n')
    root, path, inventory = staged(tmp_path, source, "rag_pipelines.toml")
    issues = credentials.process_credentials(
        root, inventory, mode="include", encrypted=True
    )
    assert len(issues) == 1 and issues[0].startswith("credential_unlock_required:")
    material = (root / "credential-recovery.json").read_text()
    assert "synthetic-sensitive-backend-text" not in material
    assert json.loads(material)["records"][0]["status"] == "locked"
    assert path.read_bytes() == source.read_bytes()


@pytest.mark.parametrize(
    "relative", ["profile.json", "custom_profiles.json.migrated", "rag_pipelines.toml"]
)
@pytest.mark.parametrize("mode", ["exclude", "include", "rollback"])
def test_only_declared_config_fields_can_unlock_or_erase_encrypted_values(
    tmp_path, monkeypatch, relative, mode
):
    from tldw_chatbook.Utils import config_encryption

    calls = []

    def unlock(value):
        calls.append(value)
        assert value == "enc:managed-secret", "historical metadata reached unlock"
        return "synthetic-clear"

    monkeypatch.setattr(config_encryption, "unlock_recovery_value", unlock)
    metadata = {
        "name": "enc:historical name",
        "description": "enc:historical prose",
        "tags": ["enc:literal"],
    }
    if relative.endswith("toml"):
        data = {
            "pipelines": {
                "functional": {
                    **metadata,
                    "type": "functional",
                    "steps": [
                        {
                            **metadata,
                            "type": "retrieve",
                            "function": "retrieve_semantic",
                            "config": {"api_key": "enc:managed-secret"},
                        }
                    ],
                }
            }
        }
        expected_location = ["pipelines", "functional", "steps", 0, "config", "api_key"]
    else:
        profile = {
            **metadata,
            "rag_config": {"embedding": {"api_key": "enc:managed-secret"}},
        }
        data = {"profiles": [profile]} if relative.startswith("custom") else profile
        expected_location = (
            ["profiles", 0] if relative.startswith("custom") else []
        ) + ["rag_config", "embedding", "api_key"]
    source = tmp_path / "input"
    source.write_text(
        toml.dumps(data) if relative.endswith("toml") else json.dumps(data)
    )
    root, path, inventory = staged(tmp_path, source, relative)
    assert not credentials.process_credentials(
        root, inventory, mode=mode, encrypted=mode != "exclude"
    )
    after = (
        toml.loads(path.read_text())
        if relative.endswith("toml")
        else json.loads(path.read_bytes())
    )
    row = (
        after["pipelines"]["functional"]
        if relative.endswith("toml")
        else after["profiles"][0]
        if relative.startswith("custom")
        else after
    )
    assert all(row[key] == value for key, value in metadata.items())
    if relative.endswith("toml"):
        assert all(row["steps"][0][key] == value for key, value in metadata.items())
    if mode == "exclude":
        assert calls == []
        assert b"enc:managed-secret" not in path.read_bytes()
    else:
        assert calls == ["enc:managed-secret"]
        assert path.read_bytes() == source.read_bytes()
        records = json.loads((root / "credential-recovery.json").read_bytes())[
            "records"
        ]
        assert len(records) == 1 and records[0]["location"] == expected_location
