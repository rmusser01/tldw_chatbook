"""``app.services`` for app doubles (ADR-250, TASK-36947.3).

Code that migrated off the app reads ``app.services.<slice>``. An app double
that sets the old alias attributes (``media_db``, ``chachanotes_db``,
``workspace_registry_service``, ``citation_trace_repository``) gets the
``storage``, ``workspace`` and ``chat`` slices from ``AliasServices(double)``.
Each slice reads the double's attributes when it is accessed, so a test may
still assign them after building the double.

A real ``TldwCli`` has a real bundle; a test that swaps one of its aliases calls
``app._publish_application_services()`` afterwards instead.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from tldw_chatbook.app_services import ProfileStorage


class AliasServices:
    """The slices the RAG wiring reads, built from an app double's aliases."""

    def __init__(self, app: Any) -> None:
        self._app = app

    @property
    def storage(self) -> ProfileStorage:
        return ProfileStorage(
            chachanotes=getattr(self._app, "chachanotes_db", None),
            media=getattr(self._app, "media_db", None),
        )

    @property
    def workspace(self) -> SimpleNamespace:
        return SimpleNamespace(
            workspace_registry_service=getattr(
                self._app, "workspace_registry_service", None
            )
        )

    @property
    def chat(self) -> SimpleNamespace:
        return SimpleNamespace(
            citation_trace_repository=getattr(
                self._app, "citation_trace_repository", None
            )
        )


def with_alias_services(app: Any) -> Any:
    """Give an app double (e.g. a ``SimpleNamespace``) an ``AliasServices``."""
    app.services = AliasServices(app)
    return app
