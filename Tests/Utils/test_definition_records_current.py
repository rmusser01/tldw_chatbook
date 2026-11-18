"""Live definition guards preserve mutation and static-lookup refusal semantics."""

from functools import wraps
import inspect
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from tldw_chatbook.Utils import definition_records as definitions


def method():
    token = object()

    def original(self=None, positional=None, *, keyword=None):
        return token, positional, keyword

    return original


class DefinitionCurrentTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        original_static = inspect.getattr_static

        def static(*args):
            self.calls.append(args)
            return original_static(*args)

        self.enterContext(patch.object(definitions, "inspect", SimpleNamespace(getattr_static=static)))
        self.function = method()
        self.owner = type("Owner", (), {"entry": self.function})
        self.records = definitions.capture_definitions(((self.owner, "entry", self.function),))

    def test_live_direct_entry_skips_only_static_mro_walk(self):
        self.assertTrue(definitions.definitions_current(self.records))
        self.assertEqual(self.calls, [])
        self.owner.entry = method()
        self.assertFalse(definitions.definitions_current(self.records))
        self.assertEqual(len(self.calls), 1)
        self.owner.entry = self.function
        self.assertTrue(definitions.definitions_current(self.records))
        self.assertEqual(len(self.calls), 1)

    def test_changed_descriptor_is_not_invoked_or_accepted(self):
        class Descriptor:
            def __get__(self, instance, owner):
                raise AssertionError("descriptor must not run")

        for value in (Descriptor(), property(self.function), staticmethod(self.function)):
            with self.subTest(descriptor=type(value).__name__):
                self.owner.entry = value
                self.assertFalse(definitions.definitions_current(self.records))
        self.assertEqual(len(self.calls), 3)

    def test_inherited_entry_and_deleted_name_use_original_static_lookup(self):
        child = type("Child", (self.owner,), {})
        records = definitions.capture_definitions(((child, "entry", self.function),))
        self.assertTrue(definitions.definitions_current(records))
        child.entry = method()
        self.assertFalse(definitions.definitions_current(records))
        del child.entry
        del self.owner.entry
        with self.assertRaises(AttributeError):
            definitions.definitions_current(records)
        self.assertEqual(len(self.calls), 3)

    def test_metaclass_module_instance_and_dict_subclass_keep_fallback(self):
        class Meta(type):
            def __getattribute__(self, name):
                if name == "entry":
                    raise AssertionError("dynamic class getter must not run")
                return super().__getattribute__(name)

        class Namespace(dict):
            def get(self, name):
                raise AssertionError("dict subclasses retain static lookup")

        custom = Meta("Custom", (), {"entry": self.function})
        module = ModuleType("example")
        module.entry = self.function
        instance = SimpleNamespace(entry=self.function)
        namespace = Namespace()
        namespace.entry = self.function
        for owner in (custom, module, instance, namespace):
            with self.subTest(owner=type(owner).__name__):
                records = definitions.capture_definitions(((owner, "entry", self.function),))
                self.assertTrue(definitions.definitions_current(records))
        self.assertEqual(len(self.calls), 4)

    def test_exact_dict_alias_retains_missing_name_false(self):
        namespace = {"alias": self.function}
        records = definitions.capture_definitions(((namespace, "alias", self.function),))
        self.assertTrue(definitions.definitions_current(records))
        del namespace["alias"]
        self.assertFalse(definitions.definitions_current(records))
        self.assertEqual(self.calls, [])

    def test_function_body_defaults_keywords_and_closure_mutations_refuse(self):
        mutations = (
            lambda fn: setattr(fn, "__code__", fn.__code__.replace()),
            lambda fn: setattr(fn, "__defaults__", tuple([None, None])),
            lambda fn: setattr(fn, "__kwdefaults__", {"keyword": None}),
            lambda fn: fn.__kwdefaults__.__setitem__("keyword", object()),
            lambda fn: fn.__kwdefaults__.__setitem__("extra", None),
            lambda fn: setattr(fn.__closure__[0], "cell_contents", object()),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                function = method()
                owner = type("Owner", (), {"entry": function})
                records = definitions.capture_definitions(((owner, "entry", function),))
                mutation(function)
                self.assertFalse(definitions.definitions_current(records))
        self.assertEqual(self.calls, [])

    def test_existing_two_wrapper_levels_and_record_abi_remain_intact(self):
        def wrap(function):
            @wraps(function)
            def wrapped(*args, **kwargs):
                return function(*args, **kwargs)
            return wrapped

        middle = wrap(self.function)
        outer = wrap(middle)
        self.owner.entry = outer
        records = definitions.capture_definitions(((self.owner, "entry", outer),))
        self.assertEqual(len(records[0]), 4)
        functions = records[0][3]
        self.assertEqual(tuple(row[0] for row in functions), (outer, middle, self.function))
        self.assertTrue(all(len(row) == 7 for row in functions))
        self.assertTrue(definitions.definitions_current(records))
        middle.__code__ = middle.__code__.replace()
        self.assertFalse(definitions.definitions_current(records))

    def test_empty_keywords_and_cells_do_not_mask_later_live_changes(self):
        def empty():
            pass

        owner = type("Empty", (), {"entry": empty})
        records = definitions.capture_definitions(((owner, "entry", empty),))
        self.assertEqual(records[0][3][0][4], ())
        self.assertEqual(records[0][3][0][6], ())
        self.assertTrue(definitions.definitions_current(records))
        empty.__kwdefaults__ = {}
        self.assertFalse(definitions.definitions_current(records))

    def test_strict_cell_count_and_empty_cell_exceptions_survive(self):
        owner, name, outer, functions = self.records[0]
        row = functions[0]
        mismatched = (*row[:6], (*row[6], object()))
        with self.assertRaises(ValueError):
            definitions.definitions_current(((owner, name, outer, (mismatched,)),))
        del self.function.__closure__[0].cell_contents
        with self.assertRaises(ValueError):
            definitions.definitions_current(self.records)

    def test_none_expected_does_not_turn_missing_class_entry_into_success(self):
        owner = type("Empty", (), {})
        with self.assertRaises(AttributeError):
            definitions.definitions_current(((owner, "missing", None, ()),))

    def test_invalid_empty_cell_record_keeps_original_type_error(self):
        def empty():
            pass

        owner = type("Empty", (), {"entry": empty})
        records = definitions.capture_definitions(((owner, "entry", empty),))
        row = records[0][3][0]
        with self.assertRaises(TypeError):
            definitions.definitions_current(((owner, "entry", empty, ((*row[:6], None),)),))


if __name__ == "__main__":
    unittest.main()
