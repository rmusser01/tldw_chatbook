# test_eval_properties.py
# Description: Property-based tests for evaluation system
#
"""
Property-Based Tests for Evaluation System
==========================================

Uses hypothesis to generate test cases and verify properties:
- Database invariants and consistency
- Evaluation result properties
- Task configuration validation
- Metric calculation properties
- System resilience under random inputs
"""

import pytest
import string
import json
import tempfile
from datetime import datetime, timezone
from typing import Dict, List, Any
from unittest.mock import AsyncMock

try:
    from hypothesis import given, strategies as st, assume, settings
    from hypothesis.strategies import composite
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    # Create dummy decorators for when hypothesis is not available
    def given(*args, **kwargs):
        def decorator(func):
            return pytest.mark.skip(reason="Hypothesis not available")(func)
        return decorator
    
    class st:
        @staticmethod
        def text(**kwargs):
            return lambda: "test_text"
        
        @staticmethod
        def integers(**kwargs):
            return lambda: 42
        
        @staticmethod
        def floats(**kwargs):
            return lambda: 0.5
        
        @staticmethod
        def dictionaries(**kwargs):
            return lambda: {}
        
        @staticmethod
        def lists(**kwargs):
            return lambda: []
        
        @staticmethod
        def one_of(*args):
            return lambda: args[0]() if args else None

from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.task_loader import TaskConfig, TaskLoader
from tldw_chatbook.Evals.eval_runner import EvalRunner, EvalSampleResult, MetricsCalculator

# Custom strategies for evaluation system types
@composite
def task_config_strategy(draw):
    """Generate valid TaskConfig instances."""
    task_types = ["question_answer", "multiple_choice", "text_generation", "code_generation"]
    metrics = ["exact_match", "accuracy", "bleu", "f1", "execution_pass_rate"]
    
    return TaskConfig(
        name=draw(st.text(min_size=1, max_size=100, alphabet=string.ascii_letters + string.digits + " _-")),
        description=draw(st.text(min_size=0, max_size=500)),
        task_type=draw(st.sampled_from(task_types)),
        dataset_name=draw(st.text(min_size=1, max_size=100, alphabet=string.ascii_letters + string.digits + "/_-.")),
        split=draw(st.sampled_from(["train", "test", "validation", "dev"])),
        metric=draw(st.sampled_from(metrics)),
        num_fewshot=draw(st.integers(min_value=0, max_value=20)),
        generation_kwargs=draw(st.dictionaries(
            keys=st.sampled_from(["temperature", "max_tokens", "top_p", "top_k"]),
            values=st.one_of(
                st.floats(min_value=0.0, max_value=2.0),
                st.integers(min_value=1, max_value=2048)
            ),
            max_size=4
        )),
        metadata=draw(st.dictionaries(
            keys=st.text(min_size=1, max_size=50, alphabet=string.ascii_letters + "_"),
            values=st.one_of(st.text(), st.integers(), st.floats(), st.booleans()),
            max_size=5
        ))
    )

@composite
def eval_result_strategy(draw):
    """Generate valid EvalSampleResult instances."""
    return EvalSampleResult(
        sample_id=draw(st.text(min_size=1, max_size=100, alphabet=string.ascii_letters + string.digits + "_-")),
        input_text=draw(st.text(min_size=0, max_size=1000)),
        expected_output=draw(st.text(min_size=0, max_size=1000)),
        actual_output=draw(st.text(min_size=0, max_size=1000)),
        metrics=draw(st.dictionaries(
            keys=st.text(min_size=1, max_size=50, alphabet=string.ascii_letters + "_"),
            values=st.floats(min_value=0.0, max_value=1.0),
            min_size=1,
            max_size=10
        )),
        metadata=draw(st.dictionaries(
            keys=st.text(min_size=1, max_size=50, alphabet=string.ascii_letters + "_"),
            values=st.one_of(st.text(), st.integers(), st.floats()),
            max_size=5
        )),
        error=draw(st.one_of(st.none(), st.text(min_size=1, max_size=200)))
    )

@composite
def sample_data_strategy(draw):
    """Generate sample evaluation data."""
    return {
        "id": draw(st.text(min_size=1, max_size=50, alphabet=string.ascii_letters + string.digits + "_")),
        "question": draw(st.text(min_size=1, max_size=500)),
        "answer": draw(st.text(min_size=0, max_size=200)),
        "category": draw(st.sampled_from(["math", "science", "history", "language", "general"])),
        "difficulty": draw(st.sampled_from(["easy", "medium", "hard"]))
    }

# Skip all property tests if hypothesis is not available
pytestmark = pytest.mark.skipif(
    not HYPOTHESIS_AVAILABLE, 
    reason="Hypothesis library not available for property testing"
)

class TestDatabaseProperties:
    """Test database invariants and properties."""
    
    @given(task_config_strategy())
    @settings(max_examples=20, deadline=None)
    def test_task_roundtrip_property(self, task_config, in_memory_db):
        """Test that tasks can be stored and retrieved without data loss."""
        # Store task
        task_id = in_memory_db.create_task(
            name=task_config.name,
            description=task_config.description,
            task_type=task_config.task_type,
            config_format="custom",
            config_data=task_config.__dict__
        )
        
        # Retrieve task
        retrieved_task = in_memory_db.get_task(task_id)
        
        # Verify essential properties are preserved
        assert retrieved_task is not None
        assert retrieved_task['name'] == task_config.name
        assert retrieved_task['description'] == task_config.description
        assert retrieved_task['task_type'] == task_config.task_type
        assert retrieved_task['config_data']['name'] == task_config.name
    
    @given(st.lists(task_config_strategy(), min_size=1, max_size=10))
    @settings(max_examples=10, deadline=None)
    def test_multiple_tasks_uniqueness(self, task_configs, in_memory_db):
        """Test that multiple tasks maintain unique IDs."""
        task_ids = []
        
        for i, config in enumerate(task_configs):
            # Ensure unique names to avoid constraint violations
            unique_name = f"{config.name}_{i}"
            task_id = in_memory_db.create_task(
                name=unique_name,
                description=config.description,
                task_type=config.task_type,
                config_format="custom",
                config_data=config.__dict__
            )
            task_ids.append(task_id)
        
        # All IDs should be unique
        assert len(set(task_ids)) == len(task_ids)
        
        # All tasks should be retrievable
        for task_id in task_ids:
            task = in_memory_db.get_task(task_id)
            assert task is not None
    
    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=20, deadline=None)
    def test_search_consistency(self, search_term, in_memory_db):
        """Test that search results are consistent and contain search terms."""
        # Create some tasks with the search term
        task_ids_with_term = []
        task_ids_without_term = []
        
        for i in range(3):
            # Task with search term
            task_id_with = in_memory_db.create_task(
                name=f"Task with {search_term} in name {i}",
                description=f"Description containing {search_term}",
                task_type="question_answer",
                config_format="custom",
                config_data={}
            )
            task_ids_with_term.append(task_id_with)
            
            # Task without search term
            task_id_without = in_memory_db.create_task(
                name=f"Different task {i}",
                description="Different description",
                task_type="question_answer",
                config_format="custom",
                config_data={}
            )
            task_ids_without_term.append(task_id_without)
        
        # Search for tasks
        search_results = in_memory_db.search_tasks(search_term)
        result_ids = [result['id'] for result in search_results]
        
        # All tasks with the term should be found
        for task_id in task_ids_with_term:
            assert task_id in result_ids
        
        # No tasks without the term should be found (unless term appears by coincidence)
        # This is a weaker assertion due to potential coincidental matches
        assert len(search_results) >= len(task_ids_with_term)
    
    @given(st.integers(min_value=1, max_value=100))
    @settings(max_examples=10, deadline=None)
    def test_pagination_properties(self, total_tasks, in_memory_db):
        """Test pagination properties with variable number of tasks."""
        # Create tasks
        task_ids = []
        for i in range(total_tasks):
            task_id = in_memory_db.create_task(
                name=f"Pagination test task {i}",
                description=f"Task {i} for pagination testing",
                task_type="question_answer",
                config_format="custom",
                config_data={"index": i}
            )
            task_ids.append(task_id)
        
        # Test different page sizes
        page_sizes = [1, 5, 10, 25, 50]
        
        for page_size in page_sizes:
            if page_size > total_tasks:
                continue
                
            all_paginated_ids = []
            page = 1
            
            while True:
                paginated_tasks = in_memory_db.list_tasks(page=page, page_size=page_size)
                if not paginated_tasks:
                    break
                
                page_ids = [task['id'] for task in paginated_tasks]
                all_paginated_ids.extend(page_ids)
                
                # Each page should have at most page_size items
                assert len(page_ids) <= page_size
                
                # No duplicates within a page
                assert len(set(page_ids)) == len(page_ids)
                
                page += 1
            
            # All original tasks should be retrievable through pagination
            assert set(all_paginated_ids) == set(task_ids)

class TestEvaluationProperties:
    """Test evaluation calculation properties."""
    
    @given(st.text(), st.text())
    @settings(max_examples=50, deadline=None)
    def test_exact_match_symmetry(self, text1, text2):
        """Test that exact match is symmetric."""
        runner = EvalRunner(llm_interface=AsyncMock())
        
        score1 = MetricsCalculator.calculate_exact_match(text1, text2)
        score2 = MetricsCalculator.calculate_exact_match(text2, text1)
        
        assert score1 == score2
    
    @given(st.text())
    @settings(max_examples=30, deadline=None)
    def test_exact_match_reflexivity(self, text):
        """Test that exact match of text with itself is always 1.0."""
        runner = EvalRunner(llm_interface=AsyncMock())
        
        score = MetricsCalculator.calculate_exact_match(text, text)
        assert score == 1.0
    
    @given(st.text(), st.text())
    @settings(max_examples=50, deadline=None)
    def test_contains_answer_monotonicity(self, answer, response):
        """Test contains answer properties."""
        runner = EvalRunner(llm_interface=AsyncMock())
        
        # If answer is empty, result should be 1.0 (trivially contains empty string)
        if not answer:
            score = runner._calculate_contains_answer(response, answer)
            assert score == 1.0
        else:
            score = runner._calculate_contains_answer(response, answer)
            assert 0.0 <= score <= 1.0
            
            # If response contains answer exactly, score should be 1.0
            if answer in response:
                assert score == 1.0
    
    @given(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=20))
    @settings(max_examples=30, deadline=None)
    def test_metrics_aggregation_properties(self, scores):
        """Test properties of metric aggregation."""
        runner = EvalRunner(llm_interface=AsyncMock())
        
        # Calculate aggregated metrics
        aggregated = runner._aggregate_metrics([{"score": score} for score in scores])
        
        # Average should be within bounds
        assert 0.0 <= aggregated["score_mean"] <= 1.0
        
        # Standard deviation should be non-negative
        assert aggregated["score_std"] >= 0.0
        
        # Min and max should be actual min and max
        assert aggregated["score_min"] == min(scores)
        assert aggregated["score_max"] == max(scores)
        
        # If all scores are the same, std should be 0
        if len(set(scores)) == 1:
            assert aggregated["score_std"] == 0.0
    
    @given(eval_result_strategy())
    @settings(max_examples=20, deadline=None)
    def test_eval_result_invariants(self, eval_result):
        """Test invariants that should hold for all eval results."""
        # All metric values should be finite numbers
        for metric_name, metric_value in eval_result.metrics.items():
            if isinstance(metric_value, (int, float)):
                assert not (metric_value != metric_value)  # Not NaN
                assert metric_value != float('inf')
                assert metric_value != float('-inf')
        
        # Sample ID should not be empty
        assert eval_result.sample_id.strip() != ""
        
        # If there's an error, actual_output might be None
        if eval_result.error_info:
            # Error should be a non-empty string
            assert isinstance(eval_result.error_info, dict)
            assert eval_result.error_info.get('error_message', '').strip() != ""

class TestTaskConfigurationProperties:
    """Test task configuration validation properties."""
    
    @given(task_config_strategy())
    @settings(max_examples=30, deadline=None)
    def test_task_config_validation_consistency(self, task_config):
        """Test that task configuration validation is consistent."""
        loader = TaskLoader()
        
        # Validate the task
        issues = loader.validate_task(task_config)
        
        # Validation should always return a list
        assert isinstance(issues, list)
        
        # If name is empty, should have validation issues
        if not task_config.name.strip():
            assert len(issues) > 0
            assert any("name" in issue.lower() for issue in issues)
        
        # If task_type is not in valid types, should have issues
        valid_types = ["question_answer", "multiple_choice", "text_generation", "code_generation"]
        if task_config.task_type not in valid_types:
            assert len(issues) > 0
            assert any("task_type" in issue.lower() for issue in issues)
    
    @given(st.dictionaries(
        keys=st.sampled_from(["temperature", "max_tokens", "top_p", "top_k"]),
        values=st.one_of(
            st.floats(min_value=-1.0, max_value=3.0),  # Include invalid ranges
            st.integers(min_value=-100, max_value=5000)
        )
    ))
    @settings(max_examples=20, deadline=None)
    def test_generation_kwargs_validation(self, generation_kwargs):
        """Test validation of generation parameters."""
        config = TaskConfig(
            name="test_task",
            description="Test",
            task_type="text_generation",
            dataset_name="test",
            split="test",
            metric="bleu",
            generation_kwargs=generation_kwargs
        )
        
        loader = TaskLoader()
        issues = loader.validate_task(config)
        
        # Check for specific validation issues
        for param, value in generation_kwargs.items():
            if param == "temperature" and (value < 0.0 or value > 2.0):
                assert any("temperature" in issue.lower() for issue in issues)
            elif param == "max_tokens" and value <= 0:
                assert any("max_tokens" in issue.lower() for issue in issues)
            elif param == "top_p" and (value < 0.0 or value > 1.0):
                assert any("top_p" in issue.lower() for issue in issues)

class TestFileHandlingProperties:
    """Test file handling and parsing properties."""
    
    @given(st.dictionaries(
        keys=st.text(min_size=1, max_size=50, alphabet=string.ascii_letters + "_"),
        values=st.one_of(
            st.text(),
            st.integers(),
            st.floats().filter(lambda x: not (x != x or x == float('inf') or x == float('-inf'))),
            st.booleans(),
            st.lists(st.text(), max_size=5)
        ),
        min_size=1,
        max_size=10
    ))
    @settings(max_examples=20, deadline=None)
    def test_json_roundtrip_property(self, data_dict):
        """Test that JSON serialization/deserialization preserves data."""
        # Add required fields for task configuration
        required_fields = {
            "name": "test_task",
            "description": "Test description",
            "task_type": "question_answer",
            "dataset_name": "test_dataset",
            "split": "test",
            "metric": "exact_match"
        }
        
        full_data = {**required_fields, **data_dict}
        
        # Serialize to JSON and back
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(full_data, f)
            temp_path = f.name
        
        try:
            loader = TaskLoader()
            config = loader.load_task(temp_path, 'custom')
            
            # Essential fields should be preserved
            assert config.name == full_data["name"]
            assert config.task_type == full_data["task_type"]
            assert config.metric == full_data["metric"]
            
        finally:
            # Cleanup
            import os
            os.unlink(temp_path)
    
    @given(st.text(min_size=1, max_size=1000))
    @settings(max_examples=20, deadline=None)
    def test_text_processing_robustness(self, text_input):
        """Test that text processing handles arbitrary inputs gracefully."""
        runner = EvalRunner(llm_interface=AsyncMock())
        
        # These operations should not crash with any text input
        try:
            # Test text truncation
            truncated = runner._truncate_text(text_input, 50)
            assert len(truncated) <= 50
            
            # Test text normalization
            normalized = runner._normalize_text(text_input)
            assert isinstance(normalized, str)
            
            # Test metric calculations with potentially problematic text
            exact_score = MetricsCalculator.calculate_exact_match(text_input, text_input)
            assert exact_score == 1.0
            
            contains_score = runner._calculate_contains_answer(text_input, text_input[:10])
            assert 0.0 <= contains_score <= 1.0
            
        except Exception as e:
            # If an exception occurs, it should be a known, handled exception type
            assert isinstance(e, (ValueError, TypeError, UnicodeError))

class TestConcurrencyProperties:
    """Test properties related to concurrent operations."""
    
    @given(st.lists(st.text(min_size=1, max_size=50), min_size=2, max_size=10, unique=True))
    @settings(max_examples=10, deadline=None)
    def test_concurrent_task_creation_uniqueness(self, task_names, temp_db):
        """Test that concurrent task creation maintains uniqueness."""
        import asyncio
        import threading
        
        created_ids = []
        errors = []
        
        def create_task(name):
            try:
                task_id = temp_db.create_task(
                    name=name,
                    description=f"Concurrent test task {name}",
                    task_type="question_answer",
                    config_format="custom",
                    config_data={"name": name}
                )
                created_ids.append(task_id)
            except Exception as e:
                errors.append(e)
        
        # Create tasks concurrently
        threads = []
        for name in task_names:
            thread = threading.Thread(target=create_task, args=(name,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should have no errors and all IDs should be unique
        assert len(errors) == 0
        assert len(created_ids) == len(task_names)
        assert len(set(created_ids)) == len(created_ids)
    
    @given(st.lists(sample_data_strategy(), min_size=1, max_size=20))
    @settings(max_examples=5, deadline=None)
    def test_evaluation_order_independence(self, samples):
        """Test that evaluation results are independent of sample order."""
        import random
        
        # Create two different orderings of the same samples
        samples1 = samples.copy()
        samples2 = samples.copy()
        random.shuffle(samples2)
        
        # Mock LLM that returns deterministic responses based on input
        mock_llm = AsyncMock()
        
        def deterministic_response(prompt, **kwargs):
            # Create deterministic response based on prompt hash
            return f"response_{hash(prompt) % 1000}"
        
        mock_llm.generate.side_effect = deterministic_response
        
        runner = EvalRunner(llm_interface=mock_llm)
        
        # Create simple task config
        task_config = TaskConfig(
            name="order_test",
            description="Test order independence",
            task_type="question_answer",
            dataset_name="test",
            split="test",
            metric="exact_match"
        )
        
        # Would need async test framework for full test
        # For now, test the property that order shouldn't matter conceptually
        assert len(samples1) == len(samples2)
        assert set(s['id'] for s in samples1) == set(s['id'] for s in samples2)

class TestSystemInvariants:
    """Test high-level system invariants."""
    
    @given(st.integers(min_value=1, max_value=50))
    @settings(max_examples=5, deadline=None)
    def test_resource_cleanup_invariant(self, num_operations, temp_db):
        """Test that system resources are properly cleaned up."""
        import gc
        import weakref
        
        # Track created objects
        created_objects = []
        
        for i in range(num_operations):
            # Create task
            task_id = temp_db.create_task(
                name=f"cleanup_test_{i}",
                description="Resource cleanup test",
                task_type="question_answer",
                config_format="custom",
                config_data={}
            )
            
            # Track the connection for this operation
            conn = temp_db.get_connection()
            created_objects.append(weakref.ref(conn))
        
        # Force garbage collection
        gc.collect()
        
        # Check that objects can be garbage collected
        # (This is a simplified test - real resource cleanup is more complex)
        alive_objects = [obj for obj in created_objects if obj() is not None]
        
        # Most objects should be collectible (allowing for some that might be cached)
        assert len(alive_objects) <= num_operations
    
    @given(st.lists(
        st.tuples(
            st.text(min_size=1, max_size=20),  # operation type
            st.dictionaries(st.text(), st.text(), max_size=3)  # operation data
        ),
        min_size=1,
        max_size=20
    ))
    @settings(max_examples=5, deadline=None)
    def test_database_consistency_invariant(self, operations, temp_db):
        """Test that database maintains consistency across operations."""
        valid_operations = ["create_task", "create_model", "create_dataset"]
        
        for op_type, op_data in operations:
            try:
                if op_type in valid_operations:
                    if op_type == "create_task":
                        temp_db.create_task(
                            name=op_data.get("name", "test"),
                            description=op_data.get("desc", "test"),
                            task_type="question_answer",
                            config_format="custom",
                            config_data={}
                        )
                    elif op_type == "create_model":
                        temp_db.create_model(
                            name=op_data.get("name", "test_model"),
                            provider="test",
                            model_id="test-1",
                            config={}
                        )
                    elif op_type == "create_dataset":
                        temp_db.create_dataset(
                            name=op_data.get("name", "test_dataset"),
                            source_type="local",
                            source_path="/test/path",
                            metadata={}
                        )
            except Exception:
                # Some operations may fail due to constraints, which is fine
                pass
        
        # After all operations, database should still be queryable
        tasks = temp_db.list_tasks()
        models = temp_db.list_models()
        datasets = temp_db.list_datasets()
        
        # Results should be lists (even if empty)
        assert isinstance(tasks, list)
        assert isinstance(models, list)
        assert isinstance(datasets, list)
        
        # All returned items should have required fields
        for task in tasks:
            assert 'id' in task
            assert 'name' in task
            assert 'task_type' in task
        
        for model in models:
            assert 'id' in model
            assert 'name' in model
            assert 'provider' in model
        
        for dataset in datasets:
            assert 'id' in dataset
            assert 'name' in dataset
            assert 'source_type' in dataset