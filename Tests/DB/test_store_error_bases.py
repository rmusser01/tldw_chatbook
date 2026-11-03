"""One except clause per kind catches InputError/ConflictError/SchemaError from every store."""

import inspect

import pytest

from tldw_chatbook.DB import ChaChaNotes_DB, Client_Media_DB_v2, Evals_DB, Prompts_DB
from tldw_chatbook.DB.store_errors import (
    StoreConflictError,
    StoreInputError,
    StoreSchemaError,
)

STORES = {
    "chacha": (ChaChaNotes_DB, ChaChaNotes_DB.CharactersRAGDBError),
    "media": (Client_Media_DB_v2, Client_Media_DB_v2.DatabaseError),
    "prompts": (Prompts_DB, Prompts_DB.DatabaseError),
    "evals": (Evals_DB, Evals_DB.EvalsDBError),
}
KINDS = {
    "InputError": StoreInputError,
    "ConflictError": StoreConflictError,
    "SchemaError": StoreSchemaError,
}
# Constructor signatures as they were before the shared bases were added.
CONFLICT_SIGNATURES = {
    "chacha": "(self, message='Conflict detected.', entity: Optional[str] = None, entity_id: Any = None)",
    "evals": "(self, message='Conflict detected.', entity: Optional[str] = None, entity_id: Any = None)",
    "media": "(self, message='Conflict detected: Record modified concurrently.', entity=None, identifier=None)",
    "prompts": "(self, message='Conflict detected: Record modified concurrently.', entity=None, identifier=None, *, code='conflict')",
}


@pytest.mark.parametrize("store", STORES)
@pytest.mark.parametrize("kind", KINDS)
def test_shared_base_catches_kind_from_every_store(store, kind):
    cls = getattr(STORES[store][0], kind)
    with pytest.raises(KINDS[kind]):
        raise cls("boom")


@pytest.mark.parametrize("store", STORES)
@pytest.mark.parametrize("kind", KINDS)
def test_per_store_handlers_still_catch(store, kind):
    module, store_base = STORES[store]
    cls = getattr(module, kind)
    if kind == "InputError":
        assert issubclass(cls, ValueError) and cls.__mro__[1] is StoreInputError
    else:
        assert cls.__mro__[1] is store_base
        with pytest.raises(store_base):
            raise cls("boom")


def test_kinds_do_not_cross_catch():
    assert not issubclass(Prompts_DB.ConflictError, StoreSchemaError)
    assert not issubclass(Evals_DB.SchemaError, StoreConflictError)
    assert not issubclass(Evals_DB.InputError, StoreConflictError)


@pytest.mark.parametrize("store", STORES)
def test_conflict_constructor_signature_unchanged(store):
    cls = STORES[store][0].ConflictError
    assert str(inspect.signature(cls.__init__)) == CONFLICT_SIGNATURES[store]
    err = cls("m", "ent")
    assert err.entity == "ent"


def test_old_import_paths_resolve_to_same_objects():
    from tldw_chatbook.DB.ChaChaNotes_DB import ConflictError as C
    from tldw_chatbook.DB.Prompts_DB import SchemaError as S
    assert C is ChaChaNotes_DB.ConflictError and S is Prompts_DB.SchemaError
    assert Client_Media_DB_v2.InputError.__module__ == "tldw_chatbook.DB.Client_Media_DB_v2"
