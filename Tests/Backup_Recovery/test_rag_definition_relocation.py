"""Known RAG selectors bind to authenticated selected projection roots only."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tldw_chatbook.Backup_Recovery.models import DISCOVERY_CONTEXT_KEY, DiscoveryContext
from tldw_chatbook.Backup_Recovery.rag_inventory import _Definitions, _Projections


@pytest.fixture
def definition_restore(tmp_path):
    source = tmp_path / "source"
    profiles = source / "data" / "default_user" / "rag_profiles"
    profiles.mkdir(parents=True, mode=0o700)
    projection = source / "index"
    projection.mkdir(mode=0o700)
    config = source / "config.toml"
    config.write_text("[general]\n")
    document = {
        "name": "Retained profile",
        "description": str(projection),
        "rag_config": {"vector_store": {"persist_directory": str(projection)}},
    }
    profile = profiles / "retained.json"
    profile.write_text(json.dumps(document, indent=2))
    configured = {
        "paths": {"data_dir": str(source / "data")},
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(config, "source"),
    }
    owner = _Definitions("rag.definitions")
    definition = next(i for i in owner.discover(configured) if i.path == profile)
    root = next(
        i
        for i in _Projections("rag.projections").discover(configured)
        if i.path == projection
    )
    candidate = tmp_path / "candidate.json"
    candidate.write_bytes(profile.read_bytes())
    candidate.chmod(0o600)
    private_root = tmp_path / "private-index"
    private_root.mkdir(mode=0o700)
    destination = tmp_path / "restored" / "index"
    items = {definition.logical_id: definition, root.logical_id: root}
    return SimpleNamespace(
        owner=owner,
        definition=definition,
        root=root,
        items=items,
        candidate=candidate,
        candidates={definition.logical_id: candidate, root.logical_id: private_root},
        synthetic=set(),
        mapping={
            "profile:source:config": tmp_path / "restored" / "config.toml",
            root.logical_id: destination,
            definition.logical_id: tmp_path / "restored" / "profile.json",
        },
        topology={
            i.logical_id: (
                i.metadata.root_id,
                i.metadata.parent_id,
                i.metadata.relative_path,
                i.metadata.kind,
            )
            for i in items.values()
        },
        document=document,
        destination=destination,
    )


def postcheck(fixture, route):
    """Exercise real dispatch with private copies and explicit final destinations."""
    owners = {
        "rag.definitions": fixture.owner,
        "rag.projections": _Projections("rag.projections"),
    }
    plan = SimpleNamespace(restore=tuple(fixture.mapping.items()), local_snapshot=None)
    if route == "stage":
        from tldw_chatbook.Backup_Recovery.staging import _validate_dependencies

        doc = SimpleNamespace(
            directories=tuple(
                SimpleNamespace(logical_id=key, synthetic=True)
                for key in fixture.synthetic
            ),
            producer_inventory=(),
        )
        _validate_dependencies(
            doc, fixture.items, owners, fixture.candidates, fixture.topology, plan
        )
    else:
        from tldw_chatbook.Backup_Recovery.publication import _validate_installed_copies

        private_items = {
            key: replace(item, path=fixture.candidates[key])
            for key, item in fixture.items.items()
        }
        _validate_installed_copies(
            private_items,
            fixture.candidates,
            fixture.topology,
            fixture.synthetic,
            owners,
            plan=plan,
        )


@pytest.mark.parametrize("route", ["stage", "installed"])
def test_mapped_known_selector_passes_both_private_validation_routes(
    definition_restore, route
):
    fixture = definition_restore
    data = fixture.document
    data["rag_config"]["vector_store"]["persist_directory"] = str(fixture.destination)
    fixture.candidate.write_text(json.dumps(data))
    postcheck(fixture, route)
    assert json.loads(fixture.candidate.read_text()) == data


def test_known_selector_relocates_without_reading_historical_root(definition_restore):
    fixture = definition_restore
    before = fixture.definition.path.read_bytes()
    fixture.root.path.rmdir()
    fixture.owner.relocate_restore(
        fixture.definition,
        fixture.candidate,
        fixture.mapping,
        tuple(fixture.items.values()),
    )
    expected = json.loads(before)
    expected["rag_config"]["vector_store"]["persist_directory"] = str(
        fixture.destination
    )
    assert json.loads(fixture.candidate.read_text()) == expected
    assert fixture.definition.path.read_bytes() == before
    assert not fixture.root.path.exists() and not fixture.destination.exists()


@pytest.mark.parametrize("selector", ["relative/index", "~/index"])
def test_historical_selectors_never_expand_in_receiver(definition_restore, selector):
    fixture = definition_restore
    fixture.document["rag_config"]["vector_store"]["persist_directory"] = selector
    fixture.candidate.write_text(json.dumps(fixture.document))
    before = fixture.candidate.read_bytes()
    with pytest.raises(ValueError, match="rag_definition_root_mapping_required"):
        fixture.owner.relocate_restore(
            fixture.definition,
            fixture.candidate,
            fixture.mapping,
            tuple(fixture.items.values()),
        )
    assert fixture.candidate.read_bytes() == before


@pytest.mark.parametrize(
    "damage",
    [
        "unselected",
        "foreign_owner",
        "foreign_profile",
        "file",
        "member",
        "dependency",
        "ambiguous",
        "synthetic",
    ],
)
@pytest.mark.parametrize("route", ["stage", "installed"])
def test_mapped_selector_refuses_unearned_projection_root(
    definition_restore, damage, route
):
    fixture = definition_restore
    root = fixture.root
    fixture.document["rag_config"]["vector_store"]["persist_directory"] = str(
        fixture.destination
    )
    fixture.candidate.write_text(json.dumps(fixture.document))
    if damage == "synthetic":
        fixture.synthetic.add(root.logical_id)
    elif damage == "unselected":
        fixture.mapping.pop(root.logical_id)
    elif damage == "foreign_owner":
        root = replace(root, owner="rag.definitions")
    elif damage == "foreign_profile":
        old = root.logical_id
        new = old.replace("profile:source:", "profile:foreign:")
        root = replace(
            root, logical_id=new, metadata=replace(root.metadata, root_id=new)
        )
        fixture.items.pop(old)
        fixture.mapping[new] = fixture.mapping.pop(old)
        fixture.candidates[new] = fixture.candidates.pop(old)
        fixture.topology[new] = (new, None, "", "directory")
    elif damage == "file":
        root = replace(root, metadata=replace(root.metadata, kind="file"))
    elif damage == "member":
        root = replace(
            root,
            metadata=replace(root.metadata, parent_id="parent", relative_path="nested"),
        )
    elif damage == "dependency":
        root = replace(
            root,
            dependencies=tuple(
                key for key in root.dependencies if key != fixture.definition.logical_id
            ),
        )
    elif damage == "ambiguous":
        duplicate_id = root.logical_id + "other"
        duplicate = replace(
            root,
            logical_id=duplicate_id,
            metadata=replace(root.metadata, root_id=duplicate_id),
        )
        fixture.items[duplicate_id] = duplicate
        fixture.mapping[duplicate_id] = fixture.destination
        fixture.candidates[duplicate_id] = fixture.candidates[root.logical_id]
        fixture.topology[duplicate_id] = (duplicate_id, None, "", "directory")
    fixture.items[root.logical_id] = root
    with pytest.raises(ValueError, match="rag_definition_root_mapping_required"):
        postcheck(fixture, route)


def test_unmapped_original_selector_is_not_replaced_by_another_selected_root(
    definition_restore,
):
    fixture = definition_restore
    fixture.document["rag_config"]["vector_store"]["persist_directory"] = str(
        fixture.root.path.parent / "missing"
    )
    fixture.candidate.write_text(json.dumps(fixture.document))
    before = fixture.candidate.read_bytes()
    with pytest.raises(ValueError, match="rag_definition_root_mapping_required"):
        fixture.owner.relocate_restore(
            fixture.definition,
            fixture.candidate,
            fixture.mapping,
            tuple(fixture.items.values()),
        )
    assert fixture.candidate.read_bytes() == before


def test_same_destination_preserves_snapshot_bytes(definition_restore):
    fixture = definition_restore
    fixture.mapping[fixture.root.logical_id] = fixture.root.path
    before = fixture.candidate.read_bytes()
    fixture.owner.relocate_restore(
        fixture.definition,
        fixture.candidate,
        fixture.mapping,
        tuple(fixture.items.values()),
    )
    assert fixture.candidate.read_bytes() == before


@pytest.mark.parametrize("route", ["stage", "installed"])
def test_final_selector_must_equal_explicit_selected_destination(
    definition_restore, route
):
    fixture = definition_restore
    fixture.document["rag_config"]["vector_store"]["persist_directory"] = str(
        fixture.destination.parent / "unselected"
    )
    fixture.candidate.write_text(json.dumps(fixture.document))
    with pytest.raises(ValueError, match="rag_definition_root_mapping_required"):
        postcheck(fixture, route)
