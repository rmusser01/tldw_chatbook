"""Pure descriptor semantics/work counts; no product owners or native imports."""

import inspect
from types import FunctionType
import unittest

from tldw_chatbook.Utils.plain_method_lookup import plain_instance_method as lookup



class PlainMethodLookupTests(unittest.TestCase):
    def setUp(self):
        def entry(self):
            return None

        self.entry = entry
        self.kind = type("Owner", (), {"entry": entry})
        self.owner = self.kind()
        self.calls = []

    def static(self, *args):
        self.calls.append(args)
        return inspect.getattr_static(*args)

    def read(self, owner=None, expected=None):
        owner = self.owner if owner is None else owner
        expected = self.kind if expected is None else expected
        value = lookup(owner, expected, "entry", _static=self.static)
        self.assertIs(value, inspect.getattr_static(owner, "entry", None))
        return value

    def test_plain_direct_method_uses_no_static_mro_walk(self):
        self.assertIs(self.read(), self.entry)
        self.assertEqual(self.calls, [])

    def test_instance_shadow_and_its_removal_are_live(self):
        replacement = object()
        self.owner.entry = replacement
        self.assertIs(self.read(), replacement)
        self.assertEqual(len(self.calls), 1)
        del self.owner.entry
        self.assertIs(self.read(), self.entry)
        self.assertEqual(len(self.calls), 1)

    def test_replaced_instance_dictionary_is_not_cached(self):
        old = self.owner.__dict__
        self.assertIs(self.read(), self.entry)
        replacement = object()
        self.owner.__dict__ = {"entry": replacement}
        self.assertIs(self.read(), replacement)
        self.owner.__dict__ = old
        self.assertIs(self.read(), self.entry)
        self.assertEqual(len(self.calls), 1)

    def test_dict_subclass_and_same_function_shadow_keep_static_semantics(self):
        class CustomDict(dict):
            def __contains__(self, key):
                raise AssertionError("custom dictionary code invoked")

            def get(self, *args):
                raise AssertionError("custom dictionary code invoked")

        self.owner.__dict__ = CustomDict(entry=self.entry)
        self.assertIs(self.read(), self.entry)
        self.owner.__dict__ = {"entry": self.entry}
        self.assertIs(self.read(), self.entry)
        self.assertEqual(len(self.calls), 2)

    def test_data_descriptor_still_precedes_instance_shadow(self):
        self.owner.entry = object()
        original = property(lambda _: self.fail("data descriptor invoked"))
        self.kind.entry = original
        self.assertIs(self.read(), original)
        self.assertEqual(len(self.calls), 1)

    def test_live_class_replacement_removal_and_descriptor_use_original_fallback(self):
        other = lambda self: None
        self.kind.entry = other
        self.assertIs(self.read(), other)
        self.assertEqual(self.calls, [])
        for descriptor in (property(lambda _: self.fail("descriptor invoked")),
                           staticmethod(self.entry), classmethod(self.entry), object()):
            self.kind.entry = descriptor
            self.assertIs(self.read(), descriptor)
        del self.kind.entry
        self.assertIsNone(self.read())
        self.assertEqual(len(self.calls), 5)

    def test_inherited_subclass_and_unusual_metaclass_fall_back(self):
        child = type("Child", (self.kind,), {})
        self.assertIs(self.read(child(), child), self.entry)
        self.assertIs(self.read(child(), self.kind), self.entry)

        class Meta(type):
            pass

        unusual = Meta("Unusual", (), {"entry": self.entry})
        self.assertIs(self.read(unusual(), unusual), self.entry)
        self.assertEqual(len(self.calls), 3)

    def test_missing_or_shadowed_dict_never_invokes_a_descriptor(self):
        slots = type("Slots", (), {"__slots__": (), "entry": self.entry})
        self.assertIs(self.read(slots(), slots), self.entry)
        shadowed = type("Shadowed", (), {
            "__dict__": property(lambda _: self.fail("dict descriptor invoked")),
            "entry": self.entry,
        })
        self.assertIs(self.read(shadowed(), shadowed), self.entry)
        self.assertEqual(len(self.calls), 2)

    def test_custom_attribute_access_remains_static(self):
        def dynamic(self, name):
            raise AssertionError("dynamic access invoked")

        self.kind.__getattribute__ = dynamic
        self.assertIs(self.read(), self.entry)
        self.assertEqual(len(self.calls), 1)

    def test_plain_function_identity_is_returned_without_calling_it(self):
        def forbidden(self):
            raise AssertionError("method invoked")

        self.kind.entry = forbidden
        self.assertIs(type(self.read()), FunctionType)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
