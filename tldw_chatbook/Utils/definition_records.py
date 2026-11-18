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
        if type(owner) is dict:
            current = owner.get(name)
        elif type(owner) is type and outer is not None and vars(owner).get(name) is outer:
            # Only the live, identical direct entry bypasses the static MRO walk.
            # Changed, inherited and custom-metaclass entries keep its semantics.
            current = outer
        else:
            current = inspect.getattr_static(owner, name)
        if current is not outer:
            return False
        for function, code, defaults, keywords, items, closure, values in functions:
            if (
                function.__code__ is not code
                or function.__defaults__ is not defaults
                or function.__kwdefaults__ is not keywords
                or len(keywords or ()) != len(items)
                or (type(items) is not tuple or items) and any(
                    name not in keywords or keywords[name] is not value
                    for name, value in items
                )
                or function.__closure__ is not closure
                or (closure is not None or type(values) is not tuple or values) and any(
                    cell.cell_contents is not value
                    for cell, value in zip(closure or (), values, strict=True)
                )
            ):
                return False
    return True
