"""Retire only SQLite handles acquired by one finite Goal operation."""

from collections.abc import Callable
from functools import wraps
from typing import Any


from .activation import owned_goal_storage as owned_goal_storage


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
