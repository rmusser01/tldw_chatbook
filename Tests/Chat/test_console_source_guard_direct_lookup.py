"""Exercise the installed class guards with direct, changed and inherited methods."""

import inspect
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tldw_chatbook.Chat import console_domain_services as domain_services
from tldw_chatbook.Chat import console_execution_readers as readers


def _method():
    captured = object()

    def original(self, positional=None, *, keyword=None):
        return captured, positional, keyword

    return original


class DirectSourceGuardTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def static(*args):
            self.calls.append(args)
            return inspect.getattr_static(*args)

        lookup = SimpleNamespace(getattr_static=static)
        self.enterContext(patch.object(domain_services, "inspect", lookup))
        self.enterContext(patch.object(readers, "inspect", lookup))
        self.enterContext(patch.object(readers, "_VIEW_READERS", ()))
        self.function = _method()
        self.owner = type("Owner", (), {"entry": self.function})
        self.bind(self.owner, self.function)

    def bind(self, owner, original):
        function = original.fget if type(original) is property else original
        record = domain_services._definition(function)
        # Domain declarations contain functions; reader declarations also
        # retain the exact original property descriptor.
        self.enterContext(patch.object(domain_services, "_DOMAIN_DEFINITIONS", ((owner, "entry", record),)))
        self.enterContext(patch.object(readers, "_READERS", ((owner, "entry", original, record),)))

    def domain(self):
        return domain_services._domain_definitions_current()

    def readers(self):
        return readers.require_reader_sources()

    def assert_refused(self):
        self.assertFalse(self.domain())
        with self.assertRaises(LookupError):
            self.readers()

    def test_direct_entries_keep_checks_without_static_mro_walk(self):
        self.assertTrue(self.domain())
        self.assertIsNone(self.readers())
        self.assertEqual(self.calls, [])

    def test_property_identity_and_getter_body_remain_checked(self):
        original = property(self.function)
        self.owner.entry = original
        self.bind(self.owner, original)
        self.assertIsNone(self.readers())
        self.owner.entry = property(self.function)
        with self.assertRaises(LookupError):
            self.readers()
        self.owner.entry = original
        self.function.__defaults__ = (object(),)
        with self.assertRaises(LookupError):
            self.readers()
        self.assertEqual(len(self.calls), 1)

    def test_function_mutations_keep_original_refusal(self):
        mutations = (
            lambda function: setattr(function, "__code__", _method().__code__.replace(co_name="changed")),
            lambda function: setattr(function, "__defaults__", (object(),)),
            lambda function: setattr(function, "__kwdefaults__", {"keyword": None}),
            lambda function: function.__kwdefaults__.__setitem__("keyword", object()),
            lambda function: setattr(function, "__wrapped__", _method()),
            lambda function: setattr(function.__closure__[0], "cell_contents", object()),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.function = _method()
                self.owner.entry = self.function
                self.bind(self.owner, self.function)
                mutation(self.function)
                self.assert_refused()

    def test_original_wrapper_body_remains_checked(self):
        self.function.__wrapped__ = _method()
        self.bind(self.owner, self.function)
        self.assertTrue(self.domain())
        self.function.__wrapped__.__kwdefaults__["keyword"] = object()
        self.assert_refused()

    def test_replaced_descriptors_and_direct_overrides_refuse(self):
        for replacement in (_method(), property(_method()), staticmethod(self.function)):
            with self.subTest(replacement=type(replacement)):
                self.owner.entry = replacement
                self.assert_refused()

    def test_inherited_and_missing_names_keep_original_fallback(self):
        child = type("Child", (self.owner,), {})
        self.bind(child, self.function)
        self.assertTrue(self.domain())
        self.assertIsNone(self.readers())
        self.assertEqual(len(self.calls), 2)
        child.entry = _method()
        self.assert_refused()
        self.assertEqual(len(self.calls), 4)
        del child.entry
        del self.owner.entry
        self.assertFalse(self.domain())
        with self.assertRaises(AttributeError):
            self.readers()
        self.assertEqual(len(self.calls), 6)

    def test_custom_metaclass_keeps_static_fallback(self):
        class Meta(type):
            pass

        owner = Meta("Uncommon", (), {"entry": self.function})
        self.bind(owner, self.function)
        self.assertTrue(self.domain())
        self.assertIsNone(self.readers())
        self.assertEqual(len(self.calls), 2)
        owner.entry = _method()
        self.assert_refused()
        self.assertEqual(len(self.calls), 4)

    def test_projection_view_definitions_remain_checked(self):
        view = type("View", (), {"entry": _method()})
        self.enterContext(patch.object(readers, "ConsoleProjectionView", view))
        self.enterContext(patch.object(readers, "_VIEW_READERS", (("entry", domain_services._definition(view.entry)),)))
        self.assertIsNone(self.readers())
        view.entry.__defaults__ = (object(),)
        with self.assertRaises(LookupError):
            self.readers()
