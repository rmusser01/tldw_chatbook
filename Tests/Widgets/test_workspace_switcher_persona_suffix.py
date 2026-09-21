"""The switcher persona-suffix reads the passed record, not a per-row SELECT.

task-32804.6 ([W-console-2]): `workspace_persona_label_suffix` used to call
`registry.get_workspace(workspace_id)` for every row inside the modal's
`compose()`, a synchronous ~2 ms sqlite read per workspace that undid the
caller's deliberate off-thread `list_workspaces` fetch. The suffix now reads
`assistant_defaults` straight off the record the caller already has. These
tests pin that: the registry is never consulted, and the label still resolves.
"""

from __future__ import annotations

from tldw_chatbook.Widgets.Console.console_workspace_switcher_modal import (
    workspace_persona_label_suffix,
)
from tldw_chatbook.Workspaces.models import (
    WorkspaceAssistantDefaults,
    WorkspaceRecord,
)


class _Registry:
    def __init__(self):
        self.get_workspace_calls = 0

    def get_workspace(self, workspace_id):  # pragma: no cover - must not run
        self.get_workspace_calls += 1
        raise AssertionError(
            "compose() must not re-read each workspace with a per-row SELECT"
        )


class _Personas:
    def __init__(self, profiles):
        self._profiles = profiles

    def get_persona_profile(self, persona_id):
        return self._profiles.get(persona_id)


class _App:
    def __init__(self, registry, personas):
        self.workspace_registry_service = registry
        self.local_character_persona_service = personas


def _record(defaults=None):
    return WorkspaceRecord(
        workspace_id="w-1", name="Work One", assistant_defaults=defaults
    )


def test_available_persona_suffix_without_registry_reread():
    registry = _Registry()
    personas = _Personas({"p1": {"id": "p1", "name": "Helper"}})
    app = _App(registry, personas)
    record = _record(
        WorkspaceAssistantDefaults(assistant_kind="persona", assistant_id="p1")
    )

    suffix = workspace_persona_label_suffix(app, record)

    assert suffix == " · Helper"
    assert registry.get_workspace_calls == 0


def test_no_defaults_is_empty_suffix_without_registry_reread():
    registry = _Registry()
    app = _App(registry, _Personas({}))

    suffix = workspace_persona_label_suffix(app, _record(None))

    assert suffix == ""
    assert registry.get_workspace_calls == 0
