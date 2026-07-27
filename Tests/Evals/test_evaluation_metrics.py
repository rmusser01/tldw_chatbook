"""
Tests for custom evaluation metrics.
Tests instruction_adherence, format_compliance, coherence_score, and dialogue_quality metrics.
"""

from unittest.mock import Mock

import pytest

from tldw_chatbook.Evals.eval_runner import (
    BaseEvalRunner,
    EvalSample,
    EvalSampleResult,
    MetricsCalculator,
)
from tldw_chatbook.Evals.task_loader import TaskConfig


class _StubEmbeddingModel:
    """Deterministic stand-in for a SentenceTransformer.

    Lets tests exercise the "embedding model IS present" success path of
    ``calculate_semantic_similarity`` without downloading or loading a real
    model. Maps each input string to a fixed vector via ``encode()``, the
    same method name/signature real SentenceTransformer models expose.
    """

    def __init__(self, vectors: dict):
        self._vectors = vectors

    def encode(self, texts):
        return [self._vectors[text] for text in texts]


class _ConstantEmbeddingModel:
    """Stub embedding model that returns the same fixed vector for every
    input string, ignoring the text entirely.

    Used to test the exact-match short-circuit in
    ``calculate_semantic_similarity``: since every string maps to the same
    vector here, if that short-circuit were removed,
    ``calculate_semantic_similarity(s, s)`` would fall through to computing
    cosine self-similarity on this vector instead of returning 1.0
    immediately. Pairing this with a vector whose self-similarity is
    demonstrably NOT exactly 1.0 at float64 (see
    ``_PRECISION_LOSING_VECTOR`` below) makes that fallthrough detectable.
    """

    def __init__(self, vector):
        self._vector = vector

    def encode(self, texts):
        return [self._vector for _ in texts]


# An 8-dim vector (arbitrary values from a random search, not float32
# truncated) whose cosine self-similarity - computed exactly the way
# calculate_semantic_similarity computes it, dot(v, v) / (norm(v) * norm(v))
# at float64 - lands at 0.9999999999999999, a couple of ULPs short of exact
# 1.0. sqrt-then-square rounding does not always cancel out even at double
# precision. Verified directly:
#   >>> v = np.asarray(_PRECISION_LOSING_VECTOR, dtype=np.float64)
#   >>> float(np.dot(v, v) / (np.linalg.norm(v) * np.linalg.norm(v)))
#   0.9999999999999999
_PRECISION_LOSING_VECTOR = [
    0.3610581159591675,
    -1.952863097190857,
    2.347409725189209,
    0.9684969186782837,
    -0.759387195110321,
    0.9021982550621033,
    -0.46695318818092346,
    -0.060689520090818405,
]


class TestRunner(BaseEvalRunner):
    """Concrete implementation of BaseEvalRunner for testing."""

    async def run_sample(self, sample: EvalSample) -> EvalSampleResult:
        """Dummy implementation for testing."""
        return EvalSampleResult(
            sample_id=sample.id,
            input_text=sample.input_text,
            expected_output=sample.expected_output,
            actual_output="test output",
            metrics={},
        )


class TestEvaluationMetrics:
    """Test custom evaluation metrics implementation."""

    @pytest.fixture
    def runner(self):
        """Create a BaseEvalRunner instance for testing."""
        task_config = TaskConfig(
            name="Test Task",
            description="Test task for metrics",
            task_type="custom",
            dataset_name="test_dataset",
            split="test",
            metric="custom",
        )
        model_config = {"provider": "test", "model_id": "test-model"}

        # Create concrete runner for testing
        runner = TestRunner(task_config, model_config)
        return runner

    def test_instruction_adherence_basic(self, runner):
        """Test basic instruction adherence calculation."""
        # Test without specific instructions - should use semantic similarity
        sample = EvalSample(id="1", input_text="What is 2+2?", expected_output="4")

        score = runner._calculate_instruction_adherence("4", "4", sample)
        assert score == 1.0  # Exact match

        score = runner._calculate_instruction_adherence("four", "4", sample)
        assert 0 < score < 1  # Semantic similarity

    def test_instruction_adherence_with_format(self, runner):
        """Test instruction adherence with format requirements."""
        sample = EvalSample(
            id="2",
            input_text="List three colors",
            expected_output="red, blue, green",
            instructions="Please format your answer as a bulleted list",
        )

        # Test with correct format
        bulleted_response = "- red\n- blue\n- green"
        score = runner._calculate_instruction_adherence(
            bulleted_response, sample.expected_output, sample
        )
        assert score > 0.5  # Format requirement met

        # Test with wrong format
        comma_response = "red, blue, green"
        score = runner._calculate_instruction_adherence(
            comma_response, sample.expected_output, sample
        )
        assert score < 0.5  # Format requirement not met

    def test_instruction_adherence_with_length(self, runner):
        """Test instruction adherence with length requirements."""
        sample = EvalSample(
            id="3",
            input_text="Describe the sky",
            expected_output="The sky is blue and vast",
            instructions="Write a response in exactly 10 words",
        )

        # Test with correct length
        ten_word_response = "The sky appears blue during clear days and gray otherwise"
        score = runner._calculate_instruction_adherence(
            ten_word_response, sample.expected_output, sample
        )
        assert score > 0.8  # Length requirement met

        # Test with wrong length
        short_response = "Blue sky"
        score = runner._calculate_instruction_adherence(
            short_response, sample.expected_output, sample
        )
        assert score < 0.5  # Length requirement not met

    def test_format_compliance_json(self, runner):
        """Test format compliance for JSON."""
        sample = EvalSample(
            id="4", input_text="Return user info", expected_format="json"
        )

        # Valid JSON
        valid_json = '{"name": "John", "age": 30}'
        score = runner._calculate_format_compliance(valid_json, sample)
        assert score == 1.0

        # Invalid JSON
        invalid_json = "{name: John, age: 30}"
        score = runner._calculate_format_compliance(invalid_json, sample)
        assert score == 0.0

    def test_format_compliance_list(self, runner):
        """Test format compliance for lists."""
        sample = EvalSample(id="5", input_text="List items", expected_format="list")

        # Bulleted list
        bulleted = "- Item 1\n- Item 2\n- Item 3"
        score = runner._calculate_format_compliance(bulleted, sample)
        assert score == 1.0

        # Numbered list
        numbered = "1. Item 1\n2. Item 2\n3. Item 3"
        score = runner._calculate_format_compliance(numbered, sample)
        assert score == 1.0

        # Not a list
        paragraph = "Item 1, Item 2, and Item 3"
        score = runner._calculate_format_compliance(paragraph, sample)
        assert score == 0.0

    def test_format_compliance_csv(self, runner):
        """Test format compliance for CSV/table."""
        sample = EvalSample(id="6", input_text="Create table", expected_format="csv")

        # Valid CSV
        csv_data = "Name,Age,City\nJohn,30,NYC\nJane,25,LA"
        score = runner._calculate_format_compliance(csv_data, sample)
        assert score == 1.0

        # Invalid CSV (inconsistent columns)
        bad_csv = "Name,Age\nJohn,30,NYC\nJane"
        score = runner._calculate_format_compliance(bad_csv, sample)
        assert score == 0.0

    def test_coherence_score(self, runner):
        """Test coherence score calculation."""
        # Coherent text
        coherent = """This is a well-structured paragraph. It contains multiple sentences 
        with proper punctuation. Furthermore, it uses transition words to connect ideas. 
        The sentences have reasonable length and variety."""

        score = runner._calculate_coherence_score(coherent)
        assert score > 0.7  # High coherence

        # Less coherent text
        incoherent = "this text no punctuation bad structure very short"
        score = runner._calculate_coherence_score(incoherent)
        assert score < 0.5  # Low coherence

        # Empty text
        score = runner._calculate_coherence_score("")
        assert score == 0.0

    def test_coherence_score_factors(self, runner):
        """Test individual coherence factors."""
        # Test sentence length factor
        good_length = (
            "This sentence has a reasonable length. Not too short, not too long."
        )
        score = runner._calculate_coherence_score(good_length)
        assert score > 0.5

        # Test capitalization factor
        proper_caps = "This is proper. Each sentence starts with a capital letter."
        score = runner._calculate_coherence_score(proper_caps)
        assert score > 0.5

        # Test transition words factor
        with_transitions = "First, we start here. However, we must consider this. Therefore, we conclude."
        score = runner._calculate_coherence_score(with_transitions)
        assert score > 0.7

    def test_dialogue_quality(self, runner):
        """Test dialogue quality calculation."""
        sample = EvalSample(
            id="7",
            input_text="Write a conversation about weather",
            context="Two friends meeting after a long time",
        )

        # Good dialogue
        good_dialogue = '''Sarah: "Wow, it's been ages! How have you been?"
        Mike: "I've been great! This weather is amazing, isn't it?"
        Sarah: "Absolutely! Perfect day for catching up."'''

        score = runner._calculate_dialogue_quality(good_dialogue, sample)
        assert score > 0.7  # High quality dialogue

        # Poor dialogue (no markers)
        poor_dialogue = "It has been a long time. The weather is nice. Yes it is."
        score = runner._calculate_dialogue_quality(poor_dialogue, sample)
        assert score < 0.5  # Low quality

    def test_dialogue_quality_factors(self, runner):
        """Test individual dialogue quality factors."""
        sample = EvalSample(
            id="8", input_text="Dialogue prompt", context="Discussion about books"
        )

        # Test speaker indicators
        with_speakers = 'Alice: "Have you read any good books lately?"'
        score = runner._calculate_dialogue_quality(with_speakers, sample)
        assert score > 0.5

        # Test relevance to context
        relevant = '"I just finished reading that new novel about books and libraries."'
        score = runner._calculate_dialogue_quality(relevant, sample)
        assert score > 0.5

        # Test natural flow
        natural = '"Yes, I have! Have you tried the new mystery series?"'
        score = runner._calculate_dialogue_quality(natural, sample)
        assert score > 0.5

    def test_helper_methods(self, runner):
        """Test helper methods for format validation."""
        # Test JSON validation
        assert runner._is_valid_json('{"key": "value"}')
        assert not runner._is_valid_json("{invalid json}")
        assert not runner._is_valid_json("")

        # Test XML validation
        assert runner._is_valid_xml("<root><item>value</item></root>")
        assert not runner._is_valid_xml("<unclosed>")
        assert not runner._is_valid_xml("not xml")

    def test_metrics_edge_cases(self, runner):
        """Test edge cases for all metrics."""
        sample = EvalSample(id="9", input_text="Test")

        # Test with None/empty values
        assert runner._calculate_instruction_adherence("", "", sample) == 1.0
        assert (
            runner._calculate_format_compliance("", sample) == 1.0
        )  # No format required
        assert runner._calculate_coherence_score("") == 0.0
        assert runner._calculate_dialogue_quality("", sample) == 0.0

        # Test with very long text
        long_text = " ".join(["This is a sentence."] * 100)
        coherence = runner._calculate_coherence_score(long_text)
        assert 0 <= coherence <= 1.0  # Should handle long text

        # Test format compliance without expected format
        sample_no_format = EvalSample(id="10", input_text="Test")
        score = runner._calculate_format_compliance("any text", sample_no_format)
        assert score == 1.0  # Should pass when no format specified

    def test_metric_integration(self, runner):
        """Test that metrics integrate properly with calculate_metrics."""
        # Test instruction adherence metric
        runner.task_config.metric = "instruction_adherence"
        sample = EvalSample(
            id="11",
            input_text="Write exactly 5 words",
            expected_output="This has exactly five words",
            instructions="Write exactly 5 words",
        )

        metrics = runner.calculate_metrics(
            "This has exactly five words", sample.expected_output, sample
        )
        assert "instruction_adherence" in metrics
        assert metrics["instruction_adherence"] > 0.8

        # Test format compliance metric
        runner.task_config.metric = "format_compliance"
        sample = EvalSample(
            id="12",
            input_text="Return as JSON",
            expected_output='{"status": "ok"}',
            expected_format="json",
        )

        metrics = runner.calculate_metrics(
            '{"status": "ok"}', sample.expected_output, sample
        )
        assert "format_compliance" in metrics
        assert metrics["format_compliance"] == 1.0

        # Test coherence score metric
        runner.task_config.metric = "coherence_score"
        coherent_text = "This is coherent. It has good structure. The flow is natural."

        metrics = runner.calculate_metrics(coherent_text, "expected", Mock())
        assert "coherence_score" in metrics
        assert metrics["coherence_score"] > 0.5

        # Test dialogue quality metric
        runner.task_config.metric = "dialogue_quality"
        dialogue = 'Person A: "Hello!" Person B: "Hi there!"'

        metrics = runner.calculate_metrics(dialogue, "expected", sample)
        assert "dialogue_quality" in metrics
        assert metrics["dialogue_quality"] > 0.5


class TestSemanticSimilarityWithEmbeddingModel:
    """Exercise the embedding-model success path of calculate_semantic_similarity.

    Regression coverage for TASK-862: when an embedding model is available
    (real SentenceTransformer or, here, a stub passed via the
    ``embedding_model`` parameter), the function used to fall off the end
    without returning anything, silently yielding None on every call. The
    existing suite only ever exercised the "embeddings unavailable" fallback
    path, so it never caught this. These tests supply a fake embedding model
    so they are fast, deterministic, and require no network access or model
    download.
    """

    def test_returns_float_not_none_when_embedding_model_present(self):
        """A real float must come back, not None, when embeddings succeed."""
        model = _StubEmbeddingModel(
            {
                "the cat sat": [1.0, 0.0, 0.0],
                "the cat sat on the mat": [0.9, 0.1, 0.0],
            }
        )

        score = MetricsCalculator.calculate_semantic_similarity(
            "the cat sat", "the cat sat on the mat", embedding_model=model
        )

        assert score is not None
        assert isinstance(score, float)
        assert 0.0 < score < 1.0

    def test_identical_embeddings_yield_similarity_one(self):
        """Identical vectors -> cosine similarity 1.0 (clamped upper bound).

        Uses two DIFFERENT input strings that happen to map to the same
        vector, not two identical strings. calculate_semantic_similarity
        now short-circuits on exact string equality before ever calling
        the embedding model, so using identical strings here would no
        longer exercise the embedding math this test is named for - it
        would just exercise the short-circuit added for a different reason
        (see TestSemanticSimilarityExactMatchShortCircuit below).
        """
        model = _StubEmbeddingModel(
            {
                "a cat sentence": [1.0, 2.0, 3.0],
                "a kitty sentence": [1.0, 2.0, 3.0],
            }
        )

        score = MetricsCalculator.calculate_semantic_similarity(
            "a cat sentence", "a kitty sentence", embedding_model=model
        )

        assert score == pytest.approx(1.0)

    def test_opposite_embeddings_are_clamped_to_zero(self):
        """Cosine similarity is [-1, 1], but callers expect a [0, 1] score.

        Opposite vectors give raw cosine similarity -1.0; the function must
        clamp that into range rather than returning a negative score.
        """
        model = _StubEmbeddingModel(
            {
                "positive": [1.0, 0.0],
                "negative": [-1.0, 0.0],
            }
        )

        score = MetricsCalculator.calculate_semantic_similarity(
            "positive", "negative", embedding_model=model
        )

        assert score is not None
        assert score == pytest.approx(0.0)
        assert 0.0 <= score <= 1.0

    def test_zero_norm_embedding_returns_zero_not_exception_fallback(self):
        """A zero vector must yield 0.0, not divide-by-zero / lexical fallback.

        Uses two DIFFERENT (not exactly equal, so the string-equality
        short-circuit does not fire) but lexically overlapping predicted/
        expected strings, both mapped to a zero-magnitude embedding vector.
        The lexical-overlap fallback (used when embeddings error out) scores
        this pair at ~0.857 (verified via
        MetricsCalculator._calculate_lexical_semantic_fallback), not 0.0 and
        not 1.0. Zero-magnitude embedding vectors previously raised
        ZeroDivisionError in the pure-Python branch, which the broad
        `except Exception` silently converted into that lexical fallback
        score. Asserting exactly 0.0 here fails loudly if that regression
        returns, instead of coincidentally matching either path's output.
        """
        model = _StubEmbeddingModel(
            {
                "the cat sat down": [0.0, 0.0, 0.0],
                "the cat sat": [0.0, 0.0, 0.0],
            }
        )

        score = MetricsCalculator.calculate_semantic_similarity(
            "the cat sat down", "the cat sat", embedding_model=model
        )

        assert score == 0.0


class TestSemanticSimilarityExactMatchShortCircuit:
    """calculate_semantic_similarity(s, s) must return exactly 1.0.

    Follow-up regression coverage: the initial TASK-862 fix upcast
    embeddings to float64 before the cosine-similarity division, which
    reduces but does NOT eliminate floating point precision loss -
    sqrt-then-square rounding still leaves a real fraction of embeddings a
    few ULPs short of exact 1.0 even at float64 (measured empirically:
    ~30% of random float32 vectors; concretely,
    ``calculate_semantic_similarity("the cat sat on the mat", "the cat sat
    on the mat")`` returned 0.9999999999999998 through the real cached
    MiniLM model on this machine). Cosine similarity of a vector with
    itself is 1.0 by construction, so exact string equality must
    short-circuit before the embedding model is ever consulted, rather than
    depending on floating point arithmetic to land exactly on 1.0.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "4",
            "hello",
            "the cat sat on the mat",
            "a much longer sentence with several different words in it",
        ],
    )
    def test_exact_match_returns_one_despite_lossy_embedding(self, text):
        """An exact string match returns exactly 1.0 despite a lossy embedding round-trip.

        Args:
            text: Input string used as both the predicted and expected value.
                Every case maps, via ``_ConstantEmbeddingModel``, to the same
                deliberately precision-losing vector, so this parametrization
                confirms the short-circuit holds regardless of the specific
                text content.
        """
        # Every input string maps to the SAME deliberately precision-losing
        # vector (see _PRECISION_LOSING_VECTOR), regardless of its content.
        # If the short-circuit were removed, this would compute cosine
        # self-similarity on that vector - 0.9999999999999999, not 1.0 -
        # instead of returning the exact 1.0 guaranteed by construction.
        model = _ConstantEmbeddingModel(_PRECISION_LOSING_VECTOR)

        score = MetricsCalculator.calculate_semantic_similarity(
            text, text, embedding_model=model
        )

        assert score == 1.0

    def test_non_identical_strings_still_use_the_embedding_path(self):
        """Only LITERAL equality short-circuits; near-misses must not.

        "4" and "4 " are different strings (trailing space), so they must
        still go through the embedding model rather than short-circuiting.
        Both map to the same precision-losing vector here, so the result
        should be extremely close to 1.0 but demonstrably NOT the exact
        1.0 the short-circuit would produce - proving this pair took the
        embedding path, not the short-circuit.
        """
        model = _ConstantEmbeddingModel(_PRECISION_LOSING_VECTOR)

        score = MetricsCalculator.calculate_semantic_similarity(
            "4", "4 ", embedding_model=model
        )

        assert score != 1.0
        assert score == pytest.approx(1.0)


class TestExactlyOneMetricsCalculatorImplementation:
    """Regression guard for TASK-863 (metrics_calculator.py orphan removal).

    ``tldw_chatbook/Evals/metrics_calculator.py`` used to define a second,
    standalone ``MetricsCalculator`` that duplicated the one in
    ``eval_runner.py``. Nothing imported it at runtime, but nothing compared
    the two copies either, so they silently drifted: ``eval_runner.py``'s
    copy lost its ``return`` from ``calculate_semantic_similarity`` and had
    its zero-guard corrupted from ``else 1.0`` to ``else 0.0`` (TASK-862),
    while ``Evals/README.md`` kept telling users to import the *other*
    (uncorrupted-but-differently-behaved) copy. TASK-863 deleted the orphan
    and merged its unique methods into ``eval_runner.py`` so there is now
    exactly one implementation.

    This test walks the ``Evals`` package source with ``ast`` (not import
    machinery) looking for any class literally named ``MetricsCalculator``,
    so it fails loudly - independent of behavior - the moment a second copy
    is reintroduced anywhere in the package, before it has a chance to
    drift.
    """

    def test_only_one_metrics_calculator_class_defined_in_evals_package(self):
        import ast
        from pathlib import Path

        import tldw_chatbook.Evals as evals_package

        evals_dir = Path(evals_package.__file__).parent
        hits = []
        for path in sorted(evals_dir.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == "MetricsCalculator":
                    hits.append(path.relative_to(evals_dir))

        assert hits == [Path("eval_runner.py")], (
            "Expected exactly one `class MetricsCalculator` in the Evals "
            f"package (in eval_runner.py), found: {hits}. A second copy "
            "will silently drift from the first, as it already has once "
            "(see TASK-862/TASK-863) - consolidate into eval_runner.py's "
            "MetricsCalculator instead of adding a new class."
        )
