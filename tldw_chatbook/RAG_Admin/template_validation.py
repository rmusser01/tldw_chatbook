"""Local chunking-template validation matching the server endpoint (spec §7).

This module re-implements the SEMANTICS of the server's ``POST /validate``
endpoint check-for-check. The endpoint itself is not vendorable (FastAPI,
pydantic response models, DB deps, ``core.Metrics``), so parity lives here:

- Upstream pin: ``tldw_server2`` @ ``385afa951922c8a9dc2002c675bb6cad65e4ac23``
- Endpoint: ``tldw_Server_API/app/api/v1/endpoints/chunking_templates.py:782-992``
  (the spec's §7 cites the abbreviated ``endpoints/chunking_templates.py`` path)
- Pydantic first pass: ``app/api/v1/schemas/chunking_templates_schemas.py:38-83``
  (``TemplateConfig`` - unknown top-level keys are silently IGNORED and dropped
  by the model dump; §7.1 wart 3)
- Exception tuple: endpoint ``:48-56`` (``_CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS``)

Two-pass structure, exactly like the endpoint:

1. **Pydantic pass** (:797-812): validate against ``TemplateConfig`` semantics;
   on failure return ONLY those errors (early return - the hand-rolled checks
   never run). On success, replace the input with the normalized model dump
   (defaults filled, unknown top-level keys dropped, ``chunking.config``
   defaulted to ``{}`` by the ``validate_chunking`` mutation at schemas :53-54).
2. **Hand-rolled pass** (:814-982): method-registry check, hierarchical
   boundaries, classifier, operation lists, JSON-serializability.

The endpoint's metrics counters (:880-883, :891-894, :930-933) are
observability, not semantics, and are not replicated.

§7.1 DELIBERATE PARITY (all three filed upstream - §11 item 11 /
UPSTREAM_DEFECTS.md). Do not "fix" any of these without a parity ruling:

- Operation NAMES are never checked against a registry - only the presence of
  the ``operation`` KEY is (an unknown op validates clean; runtime warns and
  skips it).
- ``operation`` is REQUIRED even though the runtime also accepts the
  ``{type, params}`` op spelling - a runnable template can fail validation.
- Unknown TOP-LEVEL keys are silently dropped by the pydantic pass before the
  hand-rolled checks see them (so they validate clean). The §7.1 carve-out -
  ``name``/``description``/``tags`` never entering the validated body - is a
  CRUD-layer concern (PR B), not a validator concern.

One upstream defect found while transcribing (NOT yet in §11/UPSTREAM_DEFECTS):
the endpoint appends ``TemplateValidationError`` objects into
``warnings: Optional[list[str]]``, which makes response-model construction
raise ``pydantic.ValidationError`` (a ``ValueError`` subclass, caught by the
outer except) - i.e. on the real server, any warning currently yields
``valid=False`` with "Validation error: ...". Spec §7 rules the INTENDED
semantics ("unanchored ``.*``/``.+`` → warning, not error") and this module
implements the ruling: warnings never flip validity.

Result shape (spec §7): ``{"valid": bool, "errors": [{"field", "message"}],
"warnings": [{"field", "message"}]}`` - always plain lists, never ``None``,
and the function never raises on invalid input.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from tldw_chatbook.Chunking.engine.regex_safety import (
    check_pattern as _rx_check,
    compile_flags as _rx_flags,
    warn_ambiguity as _rx_warn,
)

__all__ = ["FALLBACK_METHODS", "TemplateValidator", "validate_template"]


# Endpoint :48-56 - the endpoint's noncritical-exception tuple. The outer
# catch-all maps anything in here to a single "Validation error: ..." result.
_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    json.JSONDecodeError,
)

# Endpoint :830-832 - the fallback used when the registry call fails.
# DELIBERATELY STALE: 11 names, omitting ``fixed_size``, ``code``,
# ``code_ast`` (§11 item 13 / UPSTREAM_DEFECTS.md). Parity means keeping it
# byte-identical, staleness included. Public so tests can pin the frozen
# behavior against the live-registry default.
FALLBACK_METHODS = [
    'words', 'sentences', 'paragraphs', 'tokens', 'semantic', 'json', 'xml',
    'ebook_chapters', 'rolling_summarize', 'structure_aware', 'propositions',
]

# Schemas :64 - strict classifier key allowlist (pydantic pass only; the
# chunking.config.classifier path is NOT strict-key checked - endpoint parity).
_CLASSIFIER_ALLOWED_KEYS = {
    'media_types', 'filename_regex', 'title_regex', 'url_regex', 'tags',
    'min_score', 'priority',
}
_CLASSIFIER_REGEX_KEYS = ('filename_regex', 'title_regex', 'url_regex')

_MAX_BOUNDARIES = 20
_BOUNDARY_PATTERN_MAX_LEN = 256
_CLASSIFIER_PATTERN_MAX_LEN = 128

# Sentinel distinguishing "key absent" from "key present with value None"
# (pydantic treats them differently: absent → default_factory, None → None).
_ABSENT = object()


def _live_engine_methods() -> list[str]:
    """Default methods source: the LIVE engine registry (spec §7).

    ``Chunking.engine.chunker.Chunker().get_available_methods()`` - the ENGINE
    class, not the ``Chunk_Lib`` shim (which has no such method). The import
    is lazy so importing this module (and ``RAG_Admin`` routing through it)
    stays light; the engine pulls strategy factories and app config.
    """
    from tldw_chatbook.Chunking.engine.chunker import Chunker

    return Chunker().get_available_methods()


def _issue(field: str, message: str) -> dict[str, str]:
    return {"field": field, "message": message}


class TemplateValidator:
    """Server-parity template validator with an injectable method-registry source.

    Args:
        methods_source: Zero-arg callable returning the list of valid chunking
            method names. Default: the live engine registry. Tests inject a
            frozen list to pin both live-resolution and stale-fallback behavior.
    """

    def __init__(
        self, *, methods_source: Optional[Callable[[], list[str]]] = None
    ) -> None:
        self._methods_source = methods_source or _live_engine_methods

    # -- public API ---------------------------------------------------------

    def validate_template(self, template: dict[str, Any]) -> dict[str, Any]:
        """Validate a template config; mirrors the endpoint's response semantics.

        Args:
            template: The template configuration body (the server endpoint's
                ``template_config`` request body).

        Returns:
            ``{"valid": bool, "errors": [{"field", "message"}],
            "warnings": [{"field", "message"}]}`` - never raises.
        """
        try:
            errors, warnings = self._validate(template)
        except _NONCRITICAL_EXCEPTIONS as exc:  # endpoint :984-992 catch-all
            return {
                "valid": False,
                "errors": [
                    _issue("template_config", f"Validation error: {exc}")
                ],
                "warnings": [],
            }
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    # -- orchestration ------------------------------------------------------

    def _validate(
        self, template: Any
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        pass_errors, normalized = self._pydantic_pass(template)
        if pass_errors:
            # Endpoint :806-812: pydantic failures return immediately; the
            # hand-rolled checks never see the template.
            return pass_errors, []
        assert normalized is not None  # guaranteed when pass_errors is empty
        return self._handrolled_pass(normalized)

    # -- pass 1: TemplateConfig pydantic semantics ---------------------------

    @staticmethod
    def _pydantic_pass(
        template: Any,
    ) -> tuple[list[dict[str, str]], Optional[dict[str, Any]]]:
        """Replicate ``TemplateConfig.model_validate`` + ``model_dump_compat``.

        Returns ``(errors, normalized_config)``; ``errors`` non-empty means the
        caller must return them verbatim (endpoint :805-812). Field names mirror
        pydantic loc-join ('.'), and validator failures surface on the FIELD
        ('classifier'), not the sub-key - that asymmetry against the
        hand-rolled pass is endpoint parity.
        """
        if not isinstance(template, dict):
            # The endpoint's Body(dict) typing 422s before the handler; the
            # local seam answers with the same shape instead of raising.
            return [_issue("template_config", "Input should be a valid dictionary")], None

        errors: list[dict[str, str]] = []

        # chunking: required dict, must contain 'method' (schemas :43-46, :50-55)
        chunking = template.get('chunking', _ABSENT)
        if chunking is _ABSENT:
            errors.append(_issue("chunking", "Field required"))
            chunking = None
        elif not isinstance(chunking, dict):
            errors.append(_issue("chunking", "Input should be a valid dictionary"))
            chunking = None
        elif 'method' not in chunking:
            errors.append(
                _issue(
                    "chunking",
                    "Value error, Chunking configuration must include 'method'",
                )
            )
            chunking = None
        else:
            chunking = dict(chunking)
            # validate_chunking mutation (schemas :53-54): config defaults to {}
            chunking.setdefault('config', {})

        # preprocessing / postprocessing: Optional[list[dict]] (schemas :39-42, :47-50)
        stages: dict[str, Any] = {}
        for field in ('preprocessing', 'postprocessing'):
            value = template.get(field, _ABSENT)
            if value is _ABSENT:
                value = []  # default_factory=list
            elif value is None or isinstance(value, list):
                for index, item in enumerate(value or []):
                    if not isinstance(item, dict):
                        errors.append(
                            _issue(f"{field}.{index}", "Input should be a valid dictionary")
                        )
            else:
                errors.append(_issue(field, "Input should be a valid list"))
                value = []
            stages[field] = value

        # classifier: Optional[dict] + validate_classifier (schemas :56-59, :60-83)
        classifier = template.get('classifier', _ABSENT)
        if classifier is _ABSENT or classifier is None:
            classifier = None
        elif not isinstance(classifier, dict):
            errors.append(_issue("classifier", "Input should be a valid dictionary"))
            classifier = None
        else:
            classifier_error = _classifier_pass_error(classifier)
            if classifier_error is not None:
                errors.append(_issue("classifier", classifier_error))
                classifier = None

        if errors:
            return errors, None

        # model_dump_compat result: exactly the model's fields - unknown
        # top-level keys are DROPPED (§7.1 wart 3), defaults filled.
        normalized: dict[str, Any] = {
            'preprocessing': stages['preprocessing'],
            'chunking': chunking,
            'postprocessing': stages['postprocessing'],
            'classifier': classifier,
        }
        return [], normalized

    # -- pass 2: the endpoint's hand-rolled checks ----------------------------

    def _handrolled_pass(
        self, template_config: dict[str, Any]
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        errors: list[dict[str, str]] = []
        warnings: list[dict[str, str]] = []

        # Required-fields block (endpoint :817-840). Post-pass-1 both checks
        # are unreachable (pydantic guaranteed them); kept for structural parity.
        if 'chunking' not in template_config:
            errors.append(_issue('chunking', 'Chunking configuration is required'))
        else:
            chunking = template_config['chunking']
            if 'method' not in chunking:
                errors.append(_issue('chunking.method', 'Chunking method is required'))
            else:
                try:
                    available_methods = self._methods_source()
                except _NONCRITICAL_EXCEPTIONS:
                    # Endpoint :830-832: stale hardcoded fallback (§11 item 13).
                    available_methods = list(FALLBACK_METHODS)
                if chunking['method'] not in available_methods:
                    errors.append(
                        _issue(
                            'chunking.method',
                            f"Unknown chunking method '{chunking['method']}'. "
                            f"Valid methods: {', '.join(sorted(available_methods))}",
                        )
                    )

        # Hierarchical options (endpoint :844-854; comment there claims
        # "top-level or chunking.config" but only the config path is read).
        hier_flag = _get_cfg_path(template_config, ['chunking', 'config', 'hierarchical'])
        if hier_flag is not None and not isinstance(hier_flag, bool):
            errors.append(
                _issue('chunking.config.hierarchical', 'hierarchical must be a boolean')
            )

        # Boundaries (endpoint :855-901)
        hier_tpl = _get_cfg_path(
            template_config, ['chunking', 'config', 'hierarchical_template']
        )
        if isinstance(hier_tpl, dict) and 'boundaries' in hier_tpl:
            boundaries = hier_tpl.get('boundaries')
            if not isinstance(boundaries, list):
                errors.append(
                    _issue(
                        'chunking.config.hierarchical_template.boundaries',
                        'boundaries must be a list',
                    )
                )
            else:
                if len(boundaries) > _MAX_BOUNDARIES:
                    errors.append(
                        _issue(
                            'chunking.config.hierarchical_template.boundaries',
                            f'Too many boundary rules (max {_MAX_BOUNDARIES})',
                        )
                    )
                for i, rule in enumerate(boundaries[:_MAX_BOUNDARIES]):
                    if not isinstance(rule, dict) or 'pattern' not in rule:
                        errors.append(
                            _issue(
                                f'chunking.config.hierarchical_template.boundaries[{i}]',
                                'Each boundary must include a pattern',
                            )
                        )
                        continue
                    pat = str(rule.get('pattern') or '')
                    err = _rx_check(pat, max_len=_BOUNDARY_PATTERN_MAX_LEN)
                    if err:
                        errors.append(
                            _issue(
                                f'chunking.config.hierarchical_template.boundaries[{i}].pattern',
                                err,
                            )
                        )
                    flags_str = str(rule.get('flags') or '').lower()
                    _, ferr = _rx_flags(flags_str)
                    if ferr:
                        errors.append(
                            _issue(
                                f'chunking.config.hierarchical_template.boundaries[{i}].flags',
                                ferr,
                            )
                        )
                    # Ambiguity is a WARNING, never an error (spec §7).
                    warn = _rx_warn(pat)
                    if warn:
                        warnings.append(
                            _issue(
                                f'chunking.config.hierarchical_template.boundaries[{i}].pattern',
                                warn,
                            )
                        )

        # Classifier (endpoint :903-935): top-level first (already fully
        # validated by pass 1 when truthy), else chunking.config.classifier
        # (NOT strict-key checked - only this hand-rolled path applies).
        classifier = template_config.get('classifier') or _get_cfg_path(
            template_config, ['chunking', 'config', 'classifier']
        )
        if classifier is not None and not isinstance(classifier, dict):
            errors.append(_issue('classifier', 'classifier must be an object'))
        elif isinstance(classifier, dict):
            ms = classifier.get('min_score')
            if ms is not None:
                try:
                    score = float(ms)  # numeric strings pass here (endpoint parity)
                    if score < 0 or score > 1:
                        raise ValueError
                except (TypeError, ValueError):
                    errors.append(_issue('classifier.min_score', 'min_score must be in [0,1]'))
            pr = classifier.get('priority')
            if pr is not None and not isinstance(pr, int):
                errors.append(_issue('classifier.priority', 'priority must be integer'))
            for key in _CLASSIFIER_REGEX_KEYS:
                pat = classifier.get(key)
                if pat is None:
                    continue
                if not isinstance(pat, str):
                    errors.append(_issue(f'classifier.{key}', 'must be a string'))
                    continue
                if len(pat) > _CLASSIFIER_PATTERN_MAX_LEN:
                    errors.append(
                        _issue(
                            f'classifier.{key}',
                            f'Pattern too long (max {_CLASSIFIER_PATTERN_MAX_LEN})',
                        )
                    )
                    continue
                perr = _rx_check(pat, max_len=_CLASSIFIER_PATTERN_MAX_LEN)
                if perr:
                    errors.append(_issue(f'classifier.{key}', perr))

        # Operation lists (endpoint :937-973). NOTE the §7.1 warts: only the
        # 'operation' KEY is required, names are never checked.
        for field in ('preprocessing', 'postprocessing'):
            if field not in template_config:
                continue
            value = template_config[field]
            if not isinstance(value, list):
                # Unreachable post-pass-1 (pydantic types the field), except
                # for an explicit null (Optional), which the dump preserves.
                errors.append(
                    _issue(field, f'{field.capitalize()} must be a list of operations')
                )
                continue
            for i, op in enumerate(value):
                if not isinstance(op, dict) or 'operation' not in op:
                    errors.append(
                        _issue(
                            f'{field}[{i}]',
                            f'Each {field} operation must have an "operation" field',
                        )
                    )

        # JSON-serializable check (endpoint :975-982). Non-serializable values
        # can only hide inside Any-typed fields (chunking / op params).
        try:
            json.dumps(template_config)
        except (TypeError, ValueError) as exc:
            errors.append(
                _issue('template_config', f'Template configuration is not JSON serializable: {exc}')
            )

        return errors, warnings


def _get_cfg_path(cfg: dict[str, Any], path: list[str]) -> Any:
    """Endpoint :843-849 helper, verbatim semantics."""
    cur: Any = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _classifier_pass_error(classifier: dict[str, Any]) -> Optional[str]:
    """validate_classifier (schemas :60-83): first violation only, phrased
    'Value error, ...' like a pydantic field-validator failure."""
    extra = set(classifier.keys()) - _CLASSIFIER_ALLOWED_KEYS
    if extra:
        return f"Value error, Unknown classifier fields: {', '.join(sorted(extra))}"
    if 'min_score' in classifier:
        ms = classifier['min_score']
        if not isinstance(ms, (int, float)) or not (0.0 <= float(ms) <= 1.0):
            return 'Value error, classifier.min_score must be in [0,1]'
    if 'priority' in classifier and not isinstance(classifier['priority'], int):
        return 'Value error, classifier.priority must be an integer'
    if 'media_types' in classifier and not isinstance(classifier['media_types'], list):
        return 'Value error, classifier.media_types must be a list of strings'
    for key in _CLASSIFIER_REGEX_KEYS:
        val = classifier.get(key)
        if val is not None and not isinstance(val, str):
            return f'Value error, classifier.{key} must be a string'
    return None


def validate_template(
    template: dict[str, Any],
    *,
    methods_source: Optional[Callable[[], list[str]]] = None,
) -> dict[str, Any]:
    """Validate a chunking template config with server-endpoint semantics.

    Args:
        template: The template configuration body.
        methods_source: Optional override for the chunking-method registry
            source (default: the live engine registry - spec §7).

    Returns:
        ``{"valid": bool, "errors": [{"field", "message"}],
        "warnings": [{"field", "message"}]}`` - never raises on invalid input.
    """
    return TemplateValidator(methods_source=methods_source).validate_template(template)
