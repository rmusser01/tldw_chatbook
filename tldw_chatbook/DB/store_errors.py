"""Store-neutral base exceptions shared by every SQLite store in ``DB/``.

Each store keeps its own ``InputError`` / ``ConflictError`` / ``SchemaError``
(and its own base hierarchy); those classes additionally derive from the bases
here so one ``except`` clause can catch a kind from any store. Dependency-free
on purpose: every store can import it without an import cycle.
"""


class StoreInputError(ValueError):
    """Any store's input-validation error."""


class StoreConflictError(Exception):
    """Any store's concurrent-modification or unique-constraint conflict."""


class StoreSchemaError(Exception):
    """Any store's schema-version mismatch or migration failure."""
