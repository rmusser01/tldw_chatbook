"""One separate real-source regression; original17 module/controls stay untouched."""

import ast
import hashlib
from pathlib import Path

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Backup_Recovery.test_console_browser_two_scope_finite import (
    _compact_inline_child_script,
)

pytestmark = pytest.mark.bootstrap_profile


def _dict_child():
    source = (
        Path(__file__).resolve().with_name("test_console_browser_two_scope_finite.py")
    )
    assert (
        hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        == "3c98cb02d7335a08418248f3666376936c7ece2bb96e3d527aff13d2053b6bf9"
    )
    declared = ast.parse(source.read_bytes())
    script = next(
        node.value.value
        for node in declared.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        and any(
            isinstance(target, ast.Name) and target.id == "_SCRIPT"
            for target in node.targets
        )
    )
    needle = "    'body_error_close_refused',\n}"
    assert script.count(needle) == 1
    script = script.replace(
        needle, "    'body_error_close_refused', 'service_dict',\n}"
    )
    needle = "    app = SimpleNamespace(local_chat_conversation_service=service)\n"
    assert script.count(needle) == 1
    script = script.replace(
        needle,
        """    original_service_dictionary = vars(service)
    custom_dictionary_calls = []
    armed = False
    class ShadowServiceDictionary(dict):
        def get(self, key, default=None):
            if armed:
                custom_dictionary_calls.append('get')
            return foreign if key == 'db' else super().get(key, default)
        def __contains__(self, key):
            if armed:
                custom_dictionary_calls.append('contains')
            return super().__contains__(key)
    service.__dict__ = ShadowServiceDictionary(original_service_dictionary)
    assert type(vars(service)) is ShadowServiceDictionary
    assert service.db is database and vars(service).get('db') is foreign
    assert inspect.getattr_static(Service, '__getattribute__') is object.__getattribute__
"""
        + needle,
    )
    needle = "            armed = False\n            boundary = snapshot()\n"
    assert script.count(needle) == 1
    script = script.replace(
        needle,
        "            armed = False\n            assert custom_dictionary_calls == [], 'optional eligibility executed a custom mapping method'\n            boundary = snapshot()\n",
    )
    needle = "        if 'list_conversations' in vars(service):\n"
    assert script.count(needle) == 1
    script = script.replace(
        needle,
        "        assert type(vars(service)) is ShadowServiceDictionary\n        service.__dict__ = original_service_dictionary\n"
        + needle,
    )
    needle = "        elif outcome == 'preinstalled_body':\n"
    assert script.count(needle) == 1
    script = script.replace(
        needle,
        """        elif outcome == 'service_dict':
            assert len(opened_a) == 2 and not opened_b, 'custom dictionary source used a different physical owner'
            assert all(owner is database for owner, *_ in queries), 'custom mapping redirected an original query'
            assert custom_dictionary_calls == []
            assert vars(service) is original_service_dictionary
            assert all(item['physically_closed'] and not item['lease_live'] and not item['registered'] for item in boundary)
"""
        + needle,
    )
    # The new case alone removes branches proven inactive for this literal
    # outcome. Original17 source, callbacks, active guards/oracles and the
    # Windows argv bound are unchanged. Never evaluate dynamic test predicates.
    unknown = object()

    def constant_condition(node):
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.left, ast.Name)
            and node.left.id == "outcome"
        ):
            right = node.comparators[0]
            if isinstance(right, ast.Constant):
                target = right.value
            elif isinstance(right, ast.Set) and all(
                isinstance(item, ast.Constant) for item in right.elts
            ):
                target = {item.value for item in right.elts}
            elif isinstance(right, ast.Name) and right.id == "cache_routes":
                target = {
                    "cache_local_empty",
                    "cache_local_foreign",
                    "cache_conn_foreign",
                }
            else:
                return unknown
            if isinstance(node.ops[0], ast.Eq):
                return "service_dict" == target
            if isinstance(node.ops[0], ast.In):
                return "service_dict" in target
        if isinstance(node, ast.BoolOp):
            values = [constant_condition(item) for item in node.values]
            if isinstance(node.op, ast.And) and False in values:
                return False
            if isinstance(node.op, ast.Or) and True in values:
                return True
        return unknown

    class OnlyNewOutcome(ast.NodeTransformer):
        def generic_visit(self, node):
            node = super().generic_visit(node)
            if hasattr(node, "body") and isinstance(node.body, list) and not node.body:
                node.body = [ast.Pass()]
            return node

        def visit_If(self, node):
            node = self.generic_visit(node)
            value = constant_condition(node.test)
            if value is True:
                return node.body
            if value is False:
                return node.orelse
            return node

    specialized = OnlyNewOutcome().visit(ast.parse(script))
    return _compact_inline_child_script(ast.unparse(specialized))


def test_preinstalled_service_dictionary_subclass_retains_original_two_a_reads(
    tmp_path,
):
    _run(tmp_path, "browser_two_scope", "service_dict", script=_dict_child())
