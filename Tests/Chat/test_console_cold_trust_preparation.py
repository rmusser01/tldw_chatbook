"""Focused contracts for the original one-catalog projection.

Real App receipt, cancellation and native ownership are separately verified by
integration-owned stock-cold controls; these tests do not claim that evidence.
"""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.parametrize("workspace_id", [None, "selected-skill-workspace"])
def test_captured_builtin_projection_never_rescans_or_constructs_trust(workspace_id):
    from tldw_chatbook.Chat.console_configuration_capture import (
        _capture_skill_context_from_records,
    )

    class Local:
        def _visible_records(self):
            raise AssertionError("Captured catalog must not be enumerated twice")

        def _summary_for_record(self, record):
            return {"name": record["name"], "source": record["source"]}

        @property
        def trust_service(self):
            raise AssertionError("Builtin projection must not resolve unused trust")

        @property
        def plugin_service(self):
            raise AssertionError("Published plugin source must remain explicit")

    observed = []

    class Plugin:
        def capture_maximum(self, scope):
            observed.append(scope)
            return {"available_skills": [{"name": "plugin-skill"}]}

    original = {"builtin": {"name": "builtin", "source": "builtin"}}
    result = _capture_skill_context_from_records(
        Local(), original, workspace_id, _plugin_service=Plugin()
    )
    assert observed == [workspace_id]
    assert [row["name"] for row in result["available_skills"]] == [
        "builtin",
        "plugin-skill",
    ]
    assert "definition_digest" not in result["available_skills"][0]
    assert result["plugin_run_id"].startswith("pending:")
    assert original == {"builtin": {"name": "builtin", "source": "builtin"}}


def test_captured_managed_projection_retains_original_trust_digest_and_blocking():
    from tldw_chatbook.Chat.console_configuration_capture import (
        _capture_skill_context_from_records,
    )

    calls = []

    class Trust:
        def current_fingerprint_digest(self, name):
            calls.append(name)
            return "exact-managed-digest"

    local = SimpleNamespace(
        _summary_for_record=lambda record: dict(record), trust_service=Trust()
    )
    result = _capture_skill_context_from_records(
        local,
        {
            "managed": {"name": "managed", "source": "managed"},
            "blocked": {"name": "blocked", "source": "managed", "trust_blocked": True},
        },
        None,
        _plugin_service=None,
    )
    assert calls == ["managed"]
    assert result["available_skills"][0]["definition_digest"] == "exact-managed-digest"
    assert [row["name"] for row in result["blocked_skills"]] == ["blocked"]


def test_captured_projection_error_is_not_reported_as_catalog_absence():
    from tldw_chatbook.Chat.console_configuration_capture import (
        _capture_skill_context_from_records,
    )

    def unavailable(_record):
        raise OSError("catalog source unavailable")

    with pytest.raises(OSError, match="catalog source unavailable"):
        _capture_skill_context_from_records(
            SimpleNamespace(_summary_for_record=unavailable),
            {"builtin": {"name": "builtin", "source": "builtin"}},
            None,
            _plugin_service=None,
        )


@pytest.mark.parametrize("changed", ["binding", "body", "defaults", "wrapper"])
def test_resident_proof_rejects_changed_helper_before_invoking_it(monkeypatch, changed):
    from tldw_chatbook import app_service_wiring as wiring
    from tldw_chatbook.Widgets import compact_model_bar as metadata
    from tldw_chatbook.Chat import console_configuration_preparation as preparation

    assert preparation._resident_skill_wiring() == (wiring, metadata)
    name = "_capture_console_skill_trust_source"
    original = getattr(wiring, name)

    def foreign(*args, **kwargs):
        raise AssertionError("Replaced proof helper must never be executed")

    if changed == "binding":
        monkeypatch.setattr(wiring, name, foreign)
    elif changed == "body":
        monkeypatch.setattr(original, "__code__", foreign.__code__)
    elif changed == "defaults":
        monkeypatch.setattr(original, "__defaults__", (None,))
    else:
        monkeypatch.setattr(original, "__wrapped__", foreign, raising=False)
    assert preparation._resident_skill_wiring() is None


def test_unknown_cold_sources_decline_without_lazy_getter_or_missing_proof_import(
    monkeypatch,
):
    import sys
    from tldw_chatbook.Chat import console_configuration_preparation as preparation

    class Cold:
        _skills_scope_service = None

        @property
        def skills_scope_service(self):
            raise AssertionError("Classification must not construct a cold source")

    monkeypatch.delitem(sys.modules, "tldw_chatbook.app_service_wiring", raising=False)
    app = Cold()
    store = SimpleNamespace(sessions=lambda: [SimpleNamespace(id="session")])
    assert (
        preparation.capture_console_received_configuration_preparation(
            app, store, object(), session_id="session"
        )
        is None
    )
    assert "tldw_chatbook.app_service_wiring" not in sys.modules


@pytest.mark.parametrize("slot", ["app", "local"])
def test_unused_trust_allows_original_published_type_but_rejects_custom_winner(slot):
    from tldw_chatbook.Chat.console_configuration_preparation import (
        _supported_published_skill_trust,
    )
    from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService

    # No constructor or native trust work: exercise only the existing type rule.
    source = SimpleNamespace(
        app=SimpleNamespace(_local_skill_trust_service=None),
        local=SimpleNamespace(_trust_service=None),
    )
    assert _supported_published_skill_trust(source)
    owner, name = (
        (source.app, "_local_skill_trust_service")
        if slot == "app"
        else (source.local, "_trust_service")
    )
    original = object.__new__(SkillTrustService)
    setattr(owner, name, original)
    assert _supported_published_skill_trust(source)
    original.current_fingerprint_digest = lambda _name: "changed"
    assert not _supported_published_skill_trust(source)
    setattr(owner, name, object())
    assert not _supported_published_skill_trust(source)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["builtin", "managed", "unavailable", "source_changed"]
)
async def test_one_catalog_result_keeps_demand_and_optional_failure_distinct(
    monkeypatch, kind
):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Chat import console_configuration_preparation as preparation

    # This is an orchestration contract. The real App proof and original native
    # ownership are covered by stock-cold controls, not this isolated local double.
    calls = []
    records = {
        "skill": {
            "name": "skill",
            "source": "builtin" if kind == "builtin" else "managed",
        }
    }

    class Local:
        def _visible_records(self):
            calls.append("catalog")
            if kind == "unavailable":
                raise OSError("unavailable")
            if kind == "source_changed":
                raise RecoveryRequired("console_snapshot_owner_changed")
            return records

        def _summary_for_record(self, record):
            calls.append("project")
            return dict(record)

        @property
        def trust_service(self):
            raise AssertionError("Catalog discovery must not initialize trust")

    app, store, creator = object(), object(), SimpleNamespace(_preparation_reads=set())
    read_source = SimpleNamespace(
        app=app,
        store=store,
        creator=creator,
        trust_source=SimpleNamespace(local=Local()),
        refs=SimpleNamespace(plugin=None),
    )
    monkeypatch.setattr(
        preparation, "_require_skill_catalog_current", lambda *_args, **_kwargs: None
    )
    kwargs = dict(
        session_id="session",
        turn_id="turn",
        skill_workspace_id=None,
        preparation=read_source,
        reads=creator._preparation_reads,
        require_current=lambda: None,
    )
    if kind == "source_changed":
        with pytest.raises(RecoveryRequired):
            await preparation.capture_console_skill_catalog_owned(
                app, store, creator, **kwargs
            )
    else:
        result = await preparation.capture_console_skill_catalog_owned(
            app, store, creator, **kwargs
        )
        assert result.unavailable is (kind == "unavailable")
        assert (result.maximum is None) is (kind == "managed")
        if kind != "unavailable":
            records["skill"]["name"] = "changed after capture"
            assert result.records["skill"]["name"] == "skill"
            with pytest.raises(TypeError):
                result.records["other"] = {}
    assert calls.count("catalog") == 1
    assert calls.count("project") == (1 if kind == "builtin" else 0)
    assert creator._preparation_reads == set()


@pytest.mark.asyncio
async def test_no_demand_result_never_enters_initializer(monkeypatch):
    from tldw_chatbook.Chat import console_configuration_preparation as preparation
    from tldw_chatbook.Chat.console_turn_context import _freeze

    class App:
        async def ensure_local_skill_trust_service(self, **kwargs):
            raise AssertionError("No-demand result must not initialize trust")

    reads, observer = set(), set()
    maximum = _freeze({"available_skills": [{"name": "builtin"}]})
    catalog = preparation.ConsoleSkillCatalogRead(
        SimpleNamespace(trust_source=SimpleNamespace(app=App())),
        "session",
        "turn",
        None,
        _freeze({}),
        maximum,
        False,
        lambda: None,
        reads,
        (observer,),
    )
    monkeypatch.setattr(
        preparation, "_require_skill_catalog_current", lambda *_args, **_kwargs: None
    )
    result = await preparation.finish_console_skill_catalog_owned(
        catalog,
        reads=reads,
        observers=(observer,),
        require_current=lambda: None,
    )
    assert result.maximum is maximum and result.trust_refs is None
    # Equal empty sets are not the same teardown owner.
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    with pytest.raises(RecoveryRequired):
        await preparation.finish_console_skill_catalog_owned(
            catalog,
            reads=reads,
            observers=(set(),),
            require_current=lambda: None,
        )


@pytest.mark.asyncio
async def test_managed_finish_uses_current_winner_and_same_records(monkeypatch):
    from tldw_chatbook.Chat import console_configuration_preparation as preparation
    from tldw_chatbook.Chat.console_turn_context import _freeze

    events = []
    winner = SimpleNamespace(current_fingerprint_digest=lambda _name: "original-digest")

    class App:
        _local_skill_trust_service = None

        async def ensure_local_skill_trust_service(self, **kwargs):
            events.append("ensure")
            assert kwargs["_owner_current"]() is True
            assert kwargs["_owner_current"](self._local_skill_trust_service) is True
            assert kwargs["_source_current"]() is True
            assert kwargs["_read_observers"][0] is reads
            assert kwargs["_read_observers"][1] is observer
            return self._local_skill_trust_service

    app = App()

    class Local:
        _trust_service = None

        def _visible_records(self):
            raise AssertionError("Managed finish must reuse the captured catalog")

        @property
        def trust_service(self):
            self._trust_service = self._trust_service or app._local_skill_trust_service
            return self._trust_service

        def _summary_for_record(self, record):
            events.append("project")
            return dict(record)

    reads, observer = set(), set()
    local = Local()
    creator = SimpleNamespace(_preparation_reads=reads)
    refs = SimpleNamespace(plugin=None)
    source = SimpleNamespace(app=app, local=local)
    owner = SimpleNamespace(
        app=app, store=object(), creator=creator, session=object(), trust_source=source
    )
    catalog = preparation.ConsoleSkillCatalogRead(
        owner,
        "session",
        "turn",
        None,
        _freeze({"managed": {"name": "managed", "source": "managed"}}),
        None,
        False,
        lambda: None,
        reads,
        (observer,),
    )
    monkeypatch.setattr(
        preparation, "_require_skill_catalog_current", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        preparation,
        "standard_console_configuration_sources",
        lambda *_args, **_kwargs: local._trust_service is winner,
    )
    monkeypatch.setattr(
        preparation, "_source_references", lambda *_args, **_kwargs: refs
    )
    monkeypatch.setattr(
        preparation, "_same_configuration_references", lambda now, before: now is before
    )
    # Another legitimate startup operation publishes after catalog capture.
    app._local_skill_trust_service = winner
    result = await preparation.finish_console_skill_catalog_owned(
        catalog,
        reads=reads,
        observers=(observer,),
        require_current=lambda: None,
    )
    assert events == ["ensure", "project"]
    assert result.trust_refs is refs
    assert (
        result.maximum["available_skills"][0]["definition_digest"] == "original-digest"
    )
    assert not reads and not observer


@pytest.mark.parametrize(
    "changed", ["creator", "session", "turn", "selection", "workspace"]
)
def test_prepared_maximum_cannot_cross_received_attempt(monkeypatch, changed):
    from dataclasses import replace
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Chat import console_configuration_preparation as preparation
    from tldw_chatbook.Chat.console_turn_context import _freeze

    app, store = object(), object()
    creator = SimpleNamespace()
    owner = preparation.ConsoleReceivedConfigurationPreparation(
        app,
        store,
        creator,
        SimpleNamespace(id="session"),
        (),
        None,
        None,
        None,
        SimpleNamespace(),
    )
    selected = SimpleNamespace(skill_workspace_id=None)
    intent = SimpleNamespace(turn_id="turn", selection=selected)
    record = SimpleNamespace(session_id="session", received_intent=intent, store=store)
    creator._hooks_v2_runtime = SimpleNamespace(_turn_custody={"turn": record})
    maximum = _freeze({})
    catalog = preparation.ConsoleSkillCatalogRead(
        owner,
        "session",
        "turn",
        None,
        _freeze({}),
        maximum,
        False,
        lambda: None,
        set(),
        (),
    )
    result = preparation.ConsolePreparedSkillContext(catalog, maximum)
    monkeypatch.setattr(
        preparation, "_require_skill_catalog_current", lambda *_args, **_kwargs: None
    )
    assert (
        preparation.require_prepared_skill_context(
            result, app, store, creator, session_id="session", selection=selected
        )
        is owner.trust_source
    )
    if changed == "creator":
        creator = SimpleNamespace()
    elif changed == "session":
        record.session_id = "another-session"
    elif changed == "turn":
        result = replace(result, catalog=replace(catalog, turn_id="another-turn"))
    elif changed == "selection":
        selected = SimpleNamespace(skill_workspace_id=None)
    else:
        selected.skill_workspace_id = "another-workspace"
    with pytest.raises(RecoveryRequired):
        preparation.require_prepared_skill_context(
            result, app, store, creator, session_id="session", selection=selected
        )
