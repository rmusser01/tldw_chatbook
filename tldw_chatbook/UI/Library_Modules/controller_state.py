"""Explicit read/write declarations for Library controller-owned state.

Mirrors Console's existing _ControllerState. Each screen declaration names one
owner and one field; no dynamic attribute forwarding surface is installed.
"""


class ControllerState:
    """Keep one existing screen attribute assignable after its state moves."""

    def __init__(self, owner_name: str, state_name: str) -> None:
        self._owner_name = owner_name
        self._state_name = state_name

    def _owner(self, instance: object) -> object:
        try:
            return object.__getattribute__(instance, self._owner_name)
        except AttributeError as exc:
            raise RuntimeError("controller not wired") from exc

    def __get__(self, instance: object, owner: type | None = None) -> object:
        if instance is None:
            return self
        return getattr(self._owner(instance), self._state_name)

    def __set__(self, instance: object, value: object) -> None:
        setattr(self._owner(instance), self._state_name, value)
