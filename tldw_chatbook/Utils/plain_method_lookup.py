"""Live descriptor lookup for narrowly qualified ordinary instance methods."""

from inspect import getattr_static
from types import FunctionType, GetSetDescriptorType


def plain_instance_method(
    owner: object, expected_type: type, name: str, *,
    _static=getattr_static, _function=FunctionType, _descriptor=GetSetDescriptorType,
) -> object:
    """Return the live raw descriptor; callers still validate its definition."""
    kind = type(owner)
    if (kind is expected_type and type(kind) is type
        and type.__getattribute__(kind, "__bases__") == (object,)):
        declared = type.__getattribute__(kind, "__dict__")
        dictionary = declared.get("__dict__")
        if (type(dictionary) is _descriptor and dictionary.__objclass__ is kind
            and dictionary.__name__ == "__dict__"
            and "__getattribute__" not in declared and "__getattr__" not in declared):
            # Fetch the native dictionary afresh; retaining a prior dictionary
            # would miss replacement/shadowing between individual guard checks.
            values = dictionary.__get__(owner, kind)
            current = declared.get(name)
            if type(values) is dict and name not in values and type(current) is _function:
                return current
    return _static(owner, name, None)
