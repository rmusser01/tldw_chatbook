"""TldwCli's service composition lives in ``ServiceWiringMixin`` (TASK-33011).

The lazy service properties moved out of ``app.py`` with their setters. Tests
and screens install doubles by assignment (``app.<property> = fake``), so each
setter must still work on a ``TldwCli`` instance through the MRO.
"""

from __future__ import annotations

import inspect
import threading

import pytest

from tldw_chatbook.app import TldwCli
from tldw_chatbook.app_service_wiring import ServiceWiringMixin

pytestmark = pytest.mark.unit

#: property -> the private slot its setter writes.
SETTER_SLOTS = {
    "terminal_session_manager": "_terminal_session_manager",
    "notes_sync_runtime_owner": "_notes_sync_runtime_owner",
    "persona_buddy_controller": "_persona_buddy_controller",
    "local_skill_trust_service": "_local_skill_trust_service",
    "local_skills_service": "_local_skills_service",
    "skills_scope_service": "_skills_scope_service",
    "server_credential_store": "_server_credential_store",
    "server_credential_store_unavailable_reason": (
        "_server_credential_store_unavailable_reason"
    ),
    "daily_report_demo_service": "_daily_report_demo_service",
    "llamacpp_snapshot_service": "_llamacpp_snapshot_service",
}


def test_mixin_sits_before_app_and_owns_the_properties() -> None:
    mro = TldwCli.__mro__
    assert mro.index(ServiceWiringMixin) < mro.index(next(c for c in mro if c.__name__ == "App"))
    for name in SETTER_SLOTS:
        descriptor = inspect.getattr_static(TldwCli, name)
        assert descriptor is vars(ServiceWiringMixin)[name], name
        assert descriptor.fset is not None, name


@pytest.mark.parametrize("name", sorted(SETTER_SLOTS))
def test_property_setter_works_on_a_tldwcli_instance(name: str) -> None:
    app = TldwCli.__new__(TldwCli)
    app._notes_sync_runtime_owner_lock = threading.Lock()
    value = "installed-double" if name.endswith("_reason") else object()

    setattr(app, name, value)

    assert vars(app)[SETTER_SLOTS[name]] is value
