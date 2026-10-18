"""Retained selection preserves authenticated meanings across adapter upgrades.

Literal old-source goldens cover selection and owning review/recovery paths.
The fixture README records the independent captures and their source custody.
"""

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from Tests.Plugins.test_codex_adapter import write_json, write_skill
from Tests.Plugins.test_recovery import lose_registry
from tldw_chatbook.Plugins.inspection import _inspect_capture, inspect_package
from tldw_chatbook.Plugins.models import PackageInspection
from tldw_chatbook.Plugins.package_files import capture_package, materialize_package
from tldw_chatbook.Plugins.recovery import retained_inspections
from tldw_chatbook.Plugins.retained_inspection import (
    _interpretation_key,
    _retained_candidates,
    _select_retained,
    _unique_interpretation,
)
from tldw_chatbook.Plugins.review import reinspect

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]

_LITERAL_PINS = (
    "1fac8f83291f5b48c3daa7b9095253a23c05d44d",
    "7b4d5c501ba8500267d8f9f84f524742c33ec882",
)
_LITERAL_CASES = (
    "portable_common", "portable_declared_value", "openai_guard", "cursor_guard",
)
_LITERAL_FIXTURES = Path(__file__).parent / "fixtures/retained_interpretations"


def _literal_inspection(tmp_path, case, pin):
    """Restore an original observation; only the copied source pathname changes."""
    golden = json.loads((_LITERAL_FIXTURES / case / "expected.json").read_text())
    package = _LITERAL_FIXTURES / case / "package"
    files = {str(path.relative_to(package)): path for path in package.rglob("*") if path.is_file()}
    assert set(files) == set(golden["files"])
    for name, path in files.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == golden["files"][name]["sha256"]
        assert path.stat().st_mode & 0o111 == int(golden["files"][name]["mode"], 8) & 0o111
    root = tmp_path / f"{case}-{pin[:8]}"
    shutil.copytree(package, root)
    observation = golden["observations"][pin]
    raw = copy.deepcopy(observation["inspection"])
    assert raw["source_identity"] == "@PACKAGE_ROOT@"
    raw["source_identity"] = str(root)
    inspection = PackageInspection.model_validate(raw)
    capture = capture_package(root)
    assert not capture.errors
    assert capture.digest == inspection.content_digest
    assert capture.source_digest == inspection.source_digest
    assert json.loads(_interpretation_key(inspection)) == observation["selected_semantics"]
    return root, inspection, observation


@pytest.mark.parametrize("pin", _LITERAL_PINS)
@pytest.mark.parametrize("case", _LITERAL_CASES)
def test_literal_historical_projection_survives_real_owner_recovery(
    plugin_stack, tmp_path, case, pin
):
    root, original, observation = _literal_inspection(tmp_path, case, pin)
    review, before = installed(plugin_stack, original)
    assert plugin_stack.call(plugin_stack.authority.verify_current) == before
    revision = before["revisions"][0]
    assert revision["variables_digest"] == observation["variables_digest"]
    assert {row["component_id"]: row["definition_digest"] for row in before["components"]} == observation["definition_digests"]
    candidates = _retained_candidates(capture_package(Path(revision["materialized_identity"])), revision)
    selected = _select_retained(revision, before["components"], candidates)
    assert json.loads(_interpretation_key(selected)) == observation["selected_semantics"]
    if case == "portable_common":
        assert len(candidates) == 2
        assert len({_interpretation_key(item) for item in candidates}) == 1
    elif case == "portable_declared_value":
        assert len({_interpretation_key(item) for item in candidates}) == 2
        assert ("variable_unset:TOKEN" in selected.activation_blockers) is (pin == _LITERAL_PINS[0])
        if pin == _LITERAL_PINS[1]:
            assert not selected.inventory["command:independent"].activation_blockers
    else:
        assert len(candidates) == 1
        assert all(row.support == ("unsupported" if pin == _LITERAL_PINS[0] else "adapted")
                   for row in selected.inventory.values() if row.kind == "hook")
    materialized = reinspect(selected, Path(selected.materialized_identity))
    assert json.loads(_interpretation_key(materialized)) == observation["selected_semantics"]
    fresh = inspect_package(root)
    assert fresh.adapter_version.endswith("/2")
    assert fresh.effective_digest != original.effective_digest
    lose_registry(plugin_stack)
    assert plugin_stack.call(plugin_stack.coordinator.recover)[0].committed
    after = plugin_stack.call(plugin_stack.coordinator.published_snapshot)
    assert after["revisions"] == before["revisions"]
    assert after["components"] == before["components"]
    recovered = retained_inspections(after)[(review.installation_id, original.effective_digest)]
    assert json.loads(_interpretation_key(recovered)) == observation["selected_semantics"]


@pytest.mark.parametrize("pin", _LITERAL_PINS)
def test_literal_old_trust_configuration_update_and_rollback_keep_meaning(plugin_stack, tmp_path, pin):
    from Tests.MCP.test_credential_bindings import MemoryCredentialBackend
    from tldw_chatbook.MCP.credential_bindings import CredentialBindingService
    from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
    from tldw_chatbook.MCP.local_store import LocalMCPStore
    from tldw_chatbook.Plugins.variables import handle

    root, original, observation = _literal_inspection(tmp_path, "portable_declared_value", pin)
    stack = plugin_stack
    install, _ = installed(stack, original)
    local = LocalMCPControlService(
        store=LocalMCPStore(tmp_path / "private-mcp.json"),
        credential_service=CredentialBindingService(MemoryCredentialBackend()),
    )
    stack.call(lambda: setattr(stack.coordinator, "mcp_mapping_owner", local))
    trust = stack.call(lambda: stack.coordinator.review_trust(install.installation_id))
    assert json.loads(_interpretation_key(trust.inspection)) == observation["selected_semantics"]
    assert stack.call(lambda: stack.coordinator.commit(trust, trust.operation_id)).committed
    metadata = stack.call(lambda: stack.coordinator.create_variable_reference(
        install.installation_id, "TOKEN", "fixture-value"
    ))
    configured = stack.call(lambda: stack.coordinator.review_configuration(
        install.installation_id, variables={"TOKEN": handle(metadata)}
    ))
    assert json.loads(_interpretation_key(configured.inspection)) == observation["selected_semantics"]
    assert stack.call(lambda: stack.coordinator.commit(configured, configured.operation_id)).committed
    replacement = inspect_package(root)
    assert replacement.adapter_version == "chatbook-portable/2"
    update = stack.call(lambda: stack.coordinator.review_revision(install.installation_id, replacement))
    assert stack.call(lambda: stack.coordinator.apply_revision(update, update.operation_id)).committed
    rollback = stack.call(lambda: stack.coordinator.review_rollback(install.installation_id, original.effective_digest))
    assert json.loads(_interpretation_key(rollback.inspection)) == observation["selected_semantics"]
    assert stack.call(lambda: stack.coordinator.apply_revision(rollback, rollback.operation_id)).committed
    state = stack.call(stack.coordinator.published_snapshot)
    assert state["installations"][0]["revision_digest"] == original.effective_digest


@pytest.mark.parametrize("damage", ["definition", "missing", "extra", "duplicate", "body"])
def test_literal_retained_complete_projection_and_bytes_tampering_refuse(plugin_stack, tmp_path, damage):
    _, original, _ = _literal_inspection(tmp_path, "portable_declared_value", _LITERAL_PINS[1])
    _, state = installed(plugin_stack, original)
    bad = copy.deepcopy(state)
    if damage == "definition":
        bad["components"][0]["definition_digest"] = "0" * 64
    elif damage == "missing":
        bad["components"].pop()
    elif damage in {"extra", "duplicate"}:
        row = dict(bad["components"][0])
        if damage == "extra":
            row["component_id"] = "command:unreviewed"
        bad["components"].append(row)
    else:
        member = Path(state["revisions"][0]["materialized_identity"]) / "skills/review/SKILL.md"
        member.chmod(0o600)
        member.write_text("Changed after authentic review.")
    with pytest.raises(ValueError):
        retained_inspections(bad)
    # No negative witness rewrites or fabricates authenticated authority.
    assert plugin_stack.call(plugin_stack.authority.verify_current) == state


@pytest.mark.parametrize("pin", _LITERAL_PINS)
def test_literal_templated_cwd_refusal_keeps_original_recipe(tmp_path, pin):
    root, original, _ = _literal_inspection(tmp_path, "portable_declared_value", pin)
    path = root / "hooks.json"
    declaration = json.loads(path.read_text())
    declaration["hooks"][0]["cwd"] = "${CHATBOOK_WORKSPACE_DATA}"
    path.write_text(json.dumps(declaration, indent=2, sort_keys=True) + "\n")
    expected = json.loads((_LITERAL_FIXTURES / "portable_declared_value/expected.json").read_text())["templated_cwd_refusal"][pin]
    candidates = _retained_candidates(capture_package(root), original.model_dump(mode="json"))
    assert any(json.loads(_interpretation_key(item)) == expected["selected_semantics"] for item in candidates)
    assert all(item.inventory["hook:guard"].support == "invalid" for item in candidates)
    assert all("constraints_unknown" in item.activation_blockers for item in candidates)


def test_literal_common_unknown_wrong_and_ambiguous_interpretations_refuse(tmp_path):
    root, first, _ = _literal_inspection(tmp_path, "portable_common", _LITERAL_PINS[0])
    _, equivalent, _ = _literal_inspection(tmp_path, "portable_common", _LITERAL_PINS[1])
    assert _unique_interpretation((first, equivalent)) is first
    for dialect, adapter in (("portable", "chatbook-portable/999"), ("cursor", "chatbook-codex/1"), ("portable", None)):
        with pytest.raises(ValueError, match="retained_adapter_unsupported"):
            _retained_candidates(capture_package(root), {"dialect": dialect, "adapter_version": adapter, "overlay_identities": []})
    # A structural ambiguity control, not a claimed digest collision or signed record.
    incompatible = first.model_copy(update={"activation_blockers": ("different",)})
    with pytest.raises(ValueError, match="retained_interpretation_ambiguous"):
        _unique_interpretation((first, incompatible))
    with pytest.raises(ValueError, match="retained_interpretation_mismatch"):
        _unique_interpretation(())


def installed(stack, inspection):
    review = stack.call(lambda: stack.coordinator.review(
        inspection, selection=tuple(inspection.inventory), workspace_id=None
    ))
    assert stack.call(lambda: stack.coordinator.commit(review, review.operation_id)).committed
    return review, stack.call(stack.coordinator.published_snapshot)


@pytest.mark.parametrize("recipe", ["dev_20261001", "branch_7b4d5c", "integrated_2"])
def test_retained_portable_review_materialization_and_recovery_keep_identity(
    plugin_stack, native_package, recipe
):
    inspection = _inspect_capture(capture_package(native_package()), recipe=recipe)
    review, before = installed(plugin_stack, inspection)
    retained = retained_inspections(before)[(review.installation_id, inspection.effective_digest)]
    assert _interpretation_key(retained) == _interpretation_key(inspection)
    assert reinspect(retained, Path(retained.materialized_identity)).adapter_version == inspection.adapter_version
    lose_registry(plugin_stack)
    assert plugin_stack.call(plugin_stack.coordinator.recover)[0].committed
    after = plugin_stack.call(plugin_stack.coordinator.published_snapshot)
    assert after["revisions"] == before["revisions"]
    assert after["components"] == before["components"]


def test_portable_common_equivalent_recipes_select_once(native_package):
    capture = capture_package(native_package())
    dev = _inspect_capture(capture, recipe="dev_20261001")
    branch = _inspect_capture(capture, recipe="branch_7b4d5c")
    assert _interpretation_key(dev) == _interpretation_key(branch)
    assert _unique_interpretation((dev, branch)) is dev


@pytest.mark.parametrize("recipe,blocked", [("dev_20261001", True), ("branch_7b4d5c", False)])
def test_portable_declared_value_selects_only_authenticated_interpretation(
    plugin_stack, native_package, recipe, blocked
):
    root = native_package(extension={"version": 1, "variables": {
        "VALUE": {"type": "string", "required": True, "secret": False}
    }})
    inspection = _inspect_capture(capture_package(root), recipe=recipe)
    assert ("variable_unset:VALUE" in inspection.activation_blockers) is blocked
    review, state = installed(plugin_stack, inspection)
    revision = state["revisions"][0]
    candidates = _retained_candidates(capture_package(Path(revision["materialized_identity"])), revision)
    assert len({_interpretation_key(item) for item in candidates}) == 2
    selected = _select_retained(revision, state["components"], candidates)
    assert selected.effective_digest == inspection.effective_digest
    assert selected.adapter_version == "chatbook-portable/1"


@pytest.mark.parametrize("dialect", ["openai", "cursor"])
@pytest.mark.parametrize("recipe", ["dev_20261001", "branch_7b4d5c"])
def test_four_historical_vendor_identities_survive_authentic_recovery(
    plugin_stack, tmp_path, dialect, recipe
):
    path = ".codex-plugin/plugin.json" if dialect == "openai" else ".cursor-plugin/plugin.json"
    write_json(tmp_path, path, {"name": "historical-vendor"})
    write_skill(tmp_path)
    inspection = _inspect_capture(capture_package(tmp_path), dialect=dialect, recipe=recipe)
    _, before = installed(plugin_stack, inspection)
    lose_registry(plugin_stack)
    assert plugin_stack.call(plugin_stack.coordinator.recover)[0].committed
    assert plugin_stack.call(plugin_stack.coordinator.published_snapshot)["revisions"] == before["revisions"]


@pytest.mark.parametrize("field", [
    "dialect", "format_version", "adapter_version", "root_manifest", "overlay_identities",
    "content_digest", "activation_blockers", "rejected", "revision_digest", "variables_digest",
])
def test_retained_revision_field_tamper_refuses(plugin_stack, native_package, field):
    _, state = installed(plugin_stack, inspect_package(native_package()))
    bad = copy.deepcopy(state)
    value = bad["revisions"][0][field]
    bad["revisions"][0][field] = (not value if isinstance(value, bool)
                                 else ["tampered"] if isinstance(value, list) else "tampered")
    with pytest.raises(ValueError):
        retained_inspections(bad)
    assert retained_inspections(state)


@pytest.mark.parametrize("field", [
    "installation_id", "revision_digest", "component_id", "kind", "local_id", "path",
    "definition_digest", "support", "selection", "dependencies", "activation_blockers",
])
def test_retained_component_field_tamper_refuses(plugin_stack, native_package, field):
    _, state = installed(plugin_stack, inspect_package(native_package()))
    bad = copy.deepcopy(state)
    value = bad["components"][0][field]
    bad["components"][0][field] = ["tampered"] if isinstance(value, list) else "tampered"
    with pytest.raises(ValueError):
        retained_inspections(bad)
    assert retained_inspections(state)


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate"])
def test_retained_complete_component_set_is_required(plugin_stack, native_package, change):
    _, state = installed(plugin_stack, inspect_package(native_package()))
    bad = copy.deepcopy(state)
    if change == "missing":
        bad["components"].clear()
    else:
        row = dict(bad["components"][0])
        if change == "extra":
            row["component_id"] = "skill:extra"
        bad["components"].append(row)
    with pytest.raises(ValueError):
        retained_inspections(bad)


@pytest.mark.parametrize("dialect,adapter", [
    ("portable", "chatbook-portable/9"), ("cursor", "chatbook-codex/1"),
    ("openai", "chatbook-cursor/2"), ("portable", None),
])
def test_unknown_wrong_or_missing_identity_has_no_latest_fallback(native_package, dialect, adapter):
    with pytest.raises(ValueError, match="retained_adapter_unsupported"):
        _retained_candidates(capture_package(native_package()), {
            "dialect": dialect, "adapter_version": adapter, "overlay_identities": []
        })


def test_distinct_filtered_interpretations_refuse_without_hash_collision_fixture(native_package):
    first = inspect_package(native_package())
    second = first.model_copy(update={"activation_blockers": ("different",)})
    with pytest.raises(ValueError, match="retained_interpretation_ambiguous"):
        _unique_interpretation((first, second))
    with pytest.raises(ValueError, match="retained_interpretation_mismatch"):
        _unique_interpretation(())


def test_old_dev_hook_remains_unsupported_after_new_inspection(plugin_stack, interop_package):
    root = interop_package("cursor-guard")
    old = _inspect_capture(capture_package(root), recipe="dev_20261001")
    _, state = installed(plugin_stack, old)
    fresh = inspect_package(root)
    assert fresh.adapter_version == "chatbook-cursor/2"
    restored = next(iter(retained_inspections(state).values()))
    assert restored.adapter_version == "chatbook-cursor/2026-10-01.1"
    assert all(row.support == "unsupported" for row in restored.inventory.values() if row.kind == "hook")
    assert restored.effective_digest != fresh.effective_digest


def test_old_expected_materialization_refuses_changed_source(native_package, tmp_path):
    root = native_package()
    old = _inspect_capture(capture_package(root), recipe="dev_20261001")
    (root / "skills/review/SKILL.md").write_text("changed")
    destination = tmp_path / "refused"
    with pytest.raises(ValueError):
        materialize_package(root, destination, expected=old)
    assert not destination.exists()


def test_fresh_catalog_conflict_refuses_equal_inputs_coalesce(tmp_path):
    write_json(tmp_path, ".cursor-plugin/plugin.json", {"name": "catalog", "commands": []})
    write_json(tmp_path, ".chatbook-plugin/catalog.json", {"commands": ["missing.md"]})
    captured = inspect_package(tmp_path)
    assert not captured.rejected and not captured.inventory  # Manifest replacement is wholesale.
    equal = inspect_package(tmp_path, catalog_overlay={"commands": ["missing.md"]})
    assert not equal.rejected and not equal.inventory
    conflict = inspect_package(tmp_path, catalog_overlay={"commands": []})
    assert conflict.rejected and "catalog_overlay_conflict" in conflict.activation_blockers


@pytest.mark.parametrize("tools", [None, []])
def test_fresh_agent_requires_explicit_tools_and_preserves_empty(tmp_path, tools):
    write_json(tmp_path, ".cursor-plugin/plugin.json", {"name": "agent-constraint"})
    write_skill(tmp_path)
    path = tmp_path / "agents/reviewer.md"
    path.parent.mkdir()
    suffix = "" if tools is None else "tools: []\n"
    path.write_text("---\nname: reviewer\ndescription: Review.\n" + suffix + "---\nUntrusted")
    inspection = inspect_package(tmp_path)
    row = inspection.inventory["agent:reviewer"]
    assert not inspection.inventory["skill:example"].activation_blockers
    if tools is None:
        assert "agent_tools_unspecified" in row.activation_blockers
    else:
        assert not row.activation_blockers and json.loads(row.definition_json)["tools"] == []


def _owned_source_inspection(stack, root, temporary, *, recipe, kind):
    """Use real producer custody while selecting the explicit historical recipe."""
    import shutil

    from tldw_chatbook.Plugins.acquisition import acquire_snapshot, verify_snapshot
    from tldw_chatbook.Plugins.git_source import select_local_source
    from tldw_chatbook.Plugins.inspection import catalog_overlay_from
    from tldw_chatbook.Plugins.models import CatalogBinding
    from tldw_chatbook.Plugins.source_service import (
        _read_snapshot,
        _update_locator,
        bind_source_inspection,
    )

    def acquire(path, operation, purpose):
        result = stack.call(lambda: acquire_snapshot(
            select_local_source(path), operation,
            stack.owner.acquisition_destination(operation), owner=stack.owner,
            authority=stack.authority, _purpose=purpose,
        ))
        reference = stack.call(lambda: verify_snapshot(stack.owner, stack.authority, result))
        return result, reference

    catalog = entry = None
    if kind != "direct":
        market = temporary / "retained-source"
        shutil.copytree(root, market / "plugins/review")
        declaration = {"name": "review", "source": "./plugins/review"}
        if kind == "catalog-overlay":
            package = market / "plugins/review"
            (package / "plugin.json").unlink()
            write_json(package, ".codex-plugin/plugin.json", {"name": "review"})
            write_skill(package, "alternate/example/SKILL.md")
            declaration["skills"] = "./alternate"
        write_json(market, ".agents/plugins/marketplace.json", {"plugins": [declaration]})
        _, catalog = acquire(market, "retained-catalog", "catalog-cache")
        path, snapshot = _read_snapshot(catalog, "retained-source", None)
        entry = snapshot.entries[0]
        root = Path(catalog.materialized_identity) / entry.local_path
    _, package = acquire(root, "retained-package", "package-candidate")
    captured = capture_package(Path(package.materialized_identity))
    raw = _inspect_capture(
        captured, recipe=recipe,
        dialect="openai" if kind == "catalog-overlay" else None,
        catalog_overlay=catalog_overlay_from(entry.overlays) if kind == "catalog-overlay" else None,
    ).model_copy(update={"materialized_identity": package.materialized_identity})
    if kind == "direct":
        return bind_source_inspection(raw, acquisition_reference=package)
    binding = CatalogBinding(
        source_id="retained-source", catalog=catalog, package=package,
        catalog_path=path, catalog_digest=snapshot.digest,
        entry_identity=entry.identity, entry_digest=entry.digest,
        update_locator_json=_update_locator(catalog, entry).model_dump_json(),
    )
    return bind_source_inspection(raw, catalog_binding=binding)


@pytest.mark.parametrize("recipe,kind", [
    (recipe, kind)
    for recipe in ("dev_20261001", "branch_7b4d5c", "integrated_2")
    for kind in ("direct", "catalog")
] + [("branch_7b4d5c", "catalog-overlay"), ("integrated_2", "catalog-overlay")])
def test_source_bound_retained_recipe_requires_owner_and_survives_registry_loss(
    plugin_stack, native_package, tmp_path, recipe, kind
):
    inspection = _owned_source_inspection(
        plugin_stack, native_package(), tmp_path, recipe=recipe, kind=kind
    )
    if kind == "catalog-overlay":
        assert set(inspection.inventory) == {"skill:example"}
    review, before = installed(plugin_stack, inspection)
    with pytest.raises(ValueError, match="owner required"):
        retained_inspections(before)
    restored = plugin_stack.call(lambda: retained_inspections(
        before, coordinator=plugin_stack.coordinator
    ))[(review.installation_id, inspection.effective_digest)]
    assert _interpretation_key(restored) == _interpretation_key(inspection)
    assert restored.adapter_version == inspection.adapter_version
    assert restored.catalog_binding == inspection.catalog_binding
    assert restored.acquisition_reference == inspection.acquisition_reference
    assert reinspect(restored, Path(restored.materialized_identity)).effective_digest == inspection.effective_digest
    lose_registry(plugin_stack)
    assert plugin_stack.call(plugin_stack.coordinator.recover)[0].committed
    after = plugin_stack.call(plugin_stack.coordinator.published_snapshot)
    assert after["revisions"] == before["revisions"]
    assert after["components"] == before["components"]


@pytest.mark.parametrize("damage", [
    "nonce", "bytes", "physical", "update_origin", "overlay", "revision_digest",
    "component", "unknown_adapter",
])
def test_source_bound_retained_tampering_cannot_supply_authority(
    plugin_stack, native_package, tmp_path, damage
):
    import shutil

    inspection = _owned_source_inspection(
        plugin_stack, native_package(), tmp_path,
        recipe="branch_7b4d5c", kind="catalog-overlay",
    )
    _, state = installed(plugin_stack, inspection)
    bad = copy.deepcopy(state)
    revision = bad["revisions"][0]
    reference = revision["catalog_binding"]["package"]
    if damage == "nonce":
        reference["acquisition_nonce"] = "0" * 32
    elif damage == "bytes":
        member = Path(reference["materialized_identity"]) / "alternate/example/SKILL.md"
        member.chmod(0o600)
        member.write_text("changed retained source bytes")
    elif damage == "physical":
        leaf = Path(reference["materialized_identity"]).parent
        hidden = leaf.with_name(leaf.name + "-retained-original")
        leaf.rename(hidden)
        shutil.copytree(hidden, leaf)
    elif damage == "update_origin":
        locator = json.loads(revision["catalog_binding"]["update_locator_json"])
        locator["repository"] = str(tmp_path / "unselected")
        revision["catalog_binding"]["update_locator_json"] = json.dumps(locator)
    elif damage == "overlay":
        revision["overlay_identities"] = []
    elif damage == "revision_digest":
        revision["revision_digest"] = "0" * 64
    elif damage == "component":
        bad["components"][0]["dependencies"] = ["skill:unreviewed"]
    else:
        revision["adapter_version"] = "chatbook-codex/999"
    with pytest.raises(ValueError):
        plugin_stack.call(lambda: retained_inspections(bad, coordinator=plugin_stack.coordinator))
    assert plugin_stack.call(plugin_stack.authority.verify_current) == state


def test_legacy_source_fields_remain_absent_in_authenticated_projection(
    plugin_stack, native_package
):
    inspection = _inspect_capture(capture_package(native_package()), recipe="branch_7b4d5c")
    assert "catalog_binding" not in inspection.model_dump(mode="json")
    assert "acquisition_reference" not in inspection.model_dump(mode="json")
    _, state = installed(plugin_stack, inspection)
    for revision in state["revisions"]:
        assert "catalog_binding" not in revision
        assert "acquisition_reference" not in revision
    assert retained_inspections(state)
