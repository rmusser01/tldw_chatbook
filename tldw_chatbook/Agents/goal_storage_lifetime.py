"""Retire only SQLite handles acquired by one finite Goal operation."""

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from functools import wraps
from typing import Any


@contextmanager
def owned_goal_storage(service: Any) -> Iterator[None]:
    """Borrow existing handles; retire fresh handles on their acquiring thread."""
    from tldw_chatbook.DB.base_db import operation_owned_connection

    persistence = service.persistence
    registry = getattr(persistence, "workspace_registry", None)
    databases = (
        service.db,
        getattr(persistence, "db", None),
        getattr(registry, "db", None),
    )
    with ExitStack() as stack:
        seen = set()
        for database in databases:
            if database is not None and id(database) not in seen:
                seen.add(id(database))
                stack.enter_context(operation_owned_connection(database))
        yield


def goal_storage_operation(function: Callable[..., Any]) -> Callable[..., Any]:
    """Keep the complete synchronous service callback inside its ownership scope."""

    @wraps(function)
    def invoke(service, *args, **kwargs):
        with owned_goal_storage(service):
            return function(service, *args, **kwargs)

    return invoke


def goal_authority_operation(function: Callable[..., Any]) -> Callable[..., Any]:
    """Retire handles acquired by a native Goal authorization callback."""

    @wraps(function)
    def invoke(authority, *args, **kwargs):
        with owned_goal_storage(authority._coordinator.service):
            return function(authority, *args, **kwargs)

    return invoke
