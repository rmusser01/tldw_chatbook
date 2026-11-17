"""Pure records for explicit definition-time compatibility boundaries."""

import inspect


def capture_definitions(entries):
    """Record each named function and its existing two wrapper levels."""
    return tuple(
        (
            owner,
            name,
            outer,
            tuple(
                (
                    function,
                    function.__code__,
                    function.__defaults__,
                    function.__kwdefaults__,
                    tuple((function.__kwdefaults__ or {}).items()),
                    function.__closure__,
                    tuple(cell.cell_contents for cell in function.__closure__ or ()),
                )
                for function in (
                    outer,
                    *((outer.__wrapped__,) if hasattr(outer, "__wrapped__") else ()),
                    *(
                        (outer.__wrapped__.__wrapped__,)
                        if hasattr(getattr(outer, "__wrapped__", None), "__wrapped__")
                        else ()
                    ),
                )
            ),
        )
        for owner, name, outer in entries
    )


def definitions_current(records):
    for owner, name, outer, functions in records:
        current = (
            owner.get(name)
            if type(owner) is dict
            else inspect.getattr_static(owner, name)
        )
        if current is not outer:
            return False
        for function, code, defaults, keywords, items, closure, values in functions:
            if (
                function.__code__ is not code
                or function.__defaults__ is not defaults
                or function.__kwdefaults__ is not keywords
                or len(keywords or ()) != len(items)
                or any(
                    name not in keywords or keywords[name] is not value
                    for name, value in items
                )
                or function.__closure__ is not closure
                or any(
                    cell.cell_contents is not value
                    for cell, value in zip(closure or (), values, strict=True)
                )
            ):
                return False
    return True
