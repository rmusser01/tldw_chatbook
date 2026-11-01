"""Admission for the fixed native Goal models used by pure captured planning."""

import json

from tldw_chatbook.Agents import goal_models as native

# Only the persisted Goal shapes consumed by this planner. Native validators and
# serializers remain authoritative; this table supplies no validation or defaults.
_CHILDREN = {
    native.GoalRequest: {
        "provider": (native.GoalProviderRef, False),
        "binding": (native.GoalBindingRef, False),
        "source_bindings": (native.GoalBindingRef, True),
        "tool_scope": (native.GoalToolScope, False),
        "verifiers": (native.VerificationSpec, True),
        "policy": (native.GoalPolicy, False),
    },
    native.GoalToolScope: {"mcp_bindings": (native.GoalMCPBinding, True)},
    native.GoalCheckpoint: {
        "report": (native.IterationReport, False),
        "decision": (native.GoalDecision, False),
    },
    native.GoalDecision: {"checks": (native.GoalCheck, True)},
    native.GoalPolicy: {},
    native.GoalProviderRef: {},
    native.GoalBindingRef: {},
    native.GoalMCPBinding: {},
    native.VerificationSpec: {},
    native.IterationReport: {},
    native.GoalCheck: {},
    native.GoalEvidence: {},
}


def _native_shape(model, data, budget):
    """Admit an accounting tree with installed defaults before constructing it."""
    if model not in _CHILDREN or not isinstance(data, dict):
        raise ValueError("native_goal_model_shape_unproved")
    budget.value(data)
    budget.container(len(data) + len(model.model_fields))
    result = dict(data)
    for name, field in model.model_fields.items():
        if name not in result:
            if field.is_required():
                continue  # The ordinary native validator rejects this input.
            budget.value(name)
            if field.default_factory is native.GoalPolicy:
                budget.container()
                result[name] = {}
            elif field.default_factory is None:
                budget.value(field.default)
                result[name] = field.default
            else:
                raise ValueError("native_goal_model_default_unproved")
        child = _CHILDREN[model].get(name)
        if child is not None:
            kind, many = child
            value = result[name]
            if many:
                if not isinstance(value, (list, tuple)):
                    raise ValueError("native_goal_model_shape_unproved")
                budget.container(len(value))
                result[name] = [_native_shape(kind, item, budget) for item in value]
            else:
                result[name] = _native_shape(kind, value, budget)
    return result


def _json_width(value, budget):
    """Upper encoded width, including escaping, without constructing JSON text."""
    if isinstance(value, str):
        width = 2
        for char in value:
            code = ord(char)
            width += (
                12
                if code > 0xFFFF
                else 6
                if code < 32 or code >= 127
                else 2
                if char in ('"', "\\")
                else 1
            )
        return width
    if isinstance(value, dict):
        # Native canonical_json sorts items and the encoder tracks containers.
        budget.container(len(value))
        budget.container(len(value))
        width = 2
        for key, item in value.items():
            budget.container(2)
            width += _json_width(key, budget) + _json_width(item, budget) + 4
        return width
    if isinstance(value, (list, tuple)):
        budget.container(len(value))
        return 2 + sum(_json_width(item, budget) + 2 for item in value)
    if isinstance(value, bool) or value is None:
        return 5
    if isinstance(value, int):
        return 2 + value.bit_length() * 30103 // 100000
    if isinstance(value, float):
        return 32
    raise ValueError("native_goal_json_shape_unproved")


def _serialization(data, budget, *, encoded=False):
    width = _json_width(data, budget)
    budget.reserve(64 + 4 * width)  # Serialized Unicode, including escaped text.
    if encoded:
        budget.reserve(64 + 4 * width)  # Native subsequent UTF-8 encoding.


def _path(value, budget):
    """Installed PurePosixPath raw, split, parsed, parts and display forms."""
    if not isinstance(value, str):
        return  # Native validation supplies the content error.
    count = value.count("/") + 1
    budget.container(6)  # Path object/cache fields.
    budget.container(1)  # _raw_paths.
    budget.container(3)  # splitroot result.
    budget.value(value)  # Relative substring from splitroot.
    budget.container(count)  # rel.split('/').
    budget.reserve(64 * count + 4 * len(value))  # Split strings.
    budget.container(count)  # Parsed/interned tail list.
    budget.container(count)  # tuple(tail) for parts.
    budget.container(count + 1)  # Anchored parts tuple.
    budget.value(value, text_copies=2)  # Tail join and anchor-prefixed str(path).


def _validation(model, data, budget):
    for name, (child, many) in _CHILDREN[model].items():
        if name in data:
            if many:
                for item in data[name]:
                    _validation(child, item, budget)
            else:
                _validation(child, data[name], budget)
    if model is native.GoalBindingRef:
        _path(data.get("locator"), budget)
    elif model is native.VerificationSpec:
        _path(data.get("verifier_path"), budget)
        for path in data.get("input_paths", ()):
            _path(path, budget)  # is_absolute
            _path(path, budget)  # parts / '..'
    elif model is native.GoalToolScope:
        budget.container(len(data.get("mcp_bindings", ())))  # Native tool ID set.
    elif model is native.GoalRequest:
        for name in ("objective", "criteria"):
            text = data.get(name)
            if isinstance(text, str):
                budget.value(text, text_copies=2)  # strip and UTF-8 encode.
        scope = data.get("tool_scope", {})
        budget.container(
            len(scope.get("catalog_tools", ())) + len(scope.get("runtime_tools", ()))
        )
        sources = data.get("source_bindings", ())
        budget.container(1 + len(sources))  # Expanded binding tuple.
        budget.container(1 + len(sources))  # Derived roots list.
        # Do not construct a second roots list to decide its admission.
        for index in range(1 + len(sources)):
            binding = data.get("binding", {}) if index == 0 else sources[index - 1]
            if binding.get("access") != "rw":
                continue
            root = binding.get("locator")
            _path(root, budget)
            for verifier in data.get("verifiers", ()):
                path = verifier.get("verifier_path")
                _path(path, budget)
                _path(root, budget)  # is_relative_to copies its other path.
                if isinstance(path, str):
                    count = path.count("/") + 1
                    budget.container(3)  # Native parents sequence.
                    # Each possible parent has a sliced tail, a path/raw list,
                    # joined text and anchor-prefixed text. The native query can
                    # visit every parent; charge all prefixes before it runs.
                    budget.reserve(64 * count + 32 * count * (count + 1) // 2)
                    budget.reserve(count * (64 * 2 + 32 * 7))
                    budget.reserve(count * (128 + 8 * len(path)))
        budget.container(len(data.get("verifiers", ())))  # Native verifier ID set.
        budget.value(data)  # Internal canonical_json model_dump clone.
        _serialization(data, budget, encoded=True)
    elif model is native.IterationReport:
        budget.value(data.get("candidate_draft", ""))  # Internal UTF-8 encode.
        _serialization(data, budget, encoded=True)


def _validate_goal(model, raw, budget):
    """Admit every fixed validator and the planner's subsequent output first."""
    budget.json(raw)
    data = json.loads(raw)
    shape = _native_shape(model, data, budget)
    budget.json(raw)  # Native model_validate_json parses independently.
    budget.value(shape)  # Native models, fields, tuple conversions and defaults.
    _validation(model, shape, budget)
    budget.value(shape)  # Subsequent caller model_dump clone.
    _serialization(shape, budget, encoded=True)  # Caller JSON and hash encoding.
    return model.model_validate_json(raw)


def _goal_json(data, budget):
    """Admit the planner's JSON text before passing it to native validation."""
    _serialization(data, budget)
    return json.dumps(data)
