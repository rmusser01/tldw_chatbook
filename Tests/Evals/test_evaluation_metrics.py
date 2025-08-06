"""
Tests for custom evaluation metrics.
Tests instruction_adherence, format_compliance, coherence_score, and dialogue_quality metrics.
"""
import pytest
from unittest.mock import Mock, patch

from tldw_chatbook.Evals.eval_runner import BaseEvalRunner, EvalSample, TaskConfig, EvalSampleResult


class TestRunner(BaseEvalRunner):
    """Concrete implementation of BaseEvalRunner for testing."""
    
    async def run_sample(self, sample: EvalSample) -> EvalSampleResult:
        """Dummy implementation for testing."""
        return EvalSampleResult(
            sample_id=sample.id,
            input_text=sample.input_text,
            expected_output=sample.expected_output,
            actual_output="test output",
            metrics={}
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
            metric="custom",
            generation_kwargs={}
        )
        model_config = {"provider": "test", "model_id": "test-model"}
        
        # Create concrete runner for testing with mocked LLMInterface
        with patch('tldw_chatbook.Chat.Chat_Functions.chat_api_call') as MockLLM:
            mock_llm = Mock()
            # Mock chat_api_call to return expected responses
            MockLLM.return_value = mock_llm
            runner = TestRunner(task_config, model_config)
            return runner
    
    def test_instruction_adherence_basic(self, runner):
        """Test basic instruction adherence calculation."""
        # Test without specific instructions - should use semantic similarity
        sample = EvalSample(
            id="1",
            input_text="What is 2+2?",
            expected_output="4"
        )
        
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
            instructions="Please format your answer as a bulleted list"
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
            instructions="Write a response in exactly 10 words"
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
            id="4",
            input_text="Return user info",
            expected_format="json"
        )
        
        # Valid JSON
        valid_json = '{"name": "John", "age": 30}'
        score = runner._calculate_format_compliance(valid_json, sample)
        assert score == 1.0
        
        # Invalid JSON
        invalid_json = '{name: John, age: 30}'
        score = runner._calculate_format_compliance(invalid_json, sample)
        assert score == 0.0
    
    def test_format_compliance_list(self, runner):
        """Test format compliance for lists."""
        sample = EvalSample(
            id="5",
            input_text="List items",
            expected_format="list"
        )
        
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
        sample = EvalSample(
            id="6",
            input_text="Create table",
            expected_format="csv"
        )
        
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
        good_length = "This sentence has a reasonable length. Not too short, not too long."
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
            context="Two friends meeting after a long time"
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
            id="8",
            input_text="Dialogue prompt",
            context="Discussion about books"
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
        assert runner._is_valid_json('{"key": "value"}') == True
        assert runner._is_valid_json('{invalid json}') == False
        assert runner._is_valid_json('') == False
        
        # Test XML validation
        assert runner._is_valid_xml('<root><item>value</item></root>') == True
        assert runner._is_valid_xml('<unclosed>') == False
        assert runner._is_valid_xml('not xml') == False
    
    def test_metrics_edge_cases(self, runner):
        """Test edge cases for all metrics."""
        sample = EvalSample(id="9", input_text="Test")
        
        # Test with None/empty values
        assert runner._calculate_instruction_adherence("", "", sample) == 1.0
        assert runner._calculate_format_compliance("", sample) == 1.0  # No format required
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
            instructions="Write exactly 5 words"
        )
        
        metrics = runner.calculate_metrics(
            "This has exactly five words",
            sample.expected_output,
            sample
        )
        assert "instruction_adherence" in metrics
        assert metrics["instruction_adherence"] > 0.8
        
        # Test format compliance metric
        runner.task_config.metric = "format_compliance"
        sample = EvalSample(
            id="12",
            input_text="Return as JSON",
            expected_output='{"status": "ok"}',
            expected_format="json"
        )
        
        metrics = runner.calculate_metrics(
            '{"status": "ok"}',
            sample.expected_output,
            sample
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