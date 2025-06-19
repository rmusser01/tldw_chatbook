# conftest.py
# Description: Shared test fixtures for evaluation tests
#
"""
Shared Test Fixtures for Evaluation System
==========================================

Provides common fixtures for testing evaluation components:
- In-memory databases
- Mock LLM providers
- Sample task configurations
- Test data generators
"""

import asyncio
import json
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
from loguru import logger

from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.App_Functions.Evals.task_loader import TaskConfig, TaskLoader
from tldw_chatbook.App_Functions.Evals.eval_runner import EvalRunner, EvalResult
from tldw_chatbook.App_Functions.Evals.eval_orchestrator import EvaluationOrchestrator
from tldw_chatbook.App_Functions.Evals.llm_interface import LLMInterface

# --- Database Fixtures ---

@pytest.fixture
def temp_db_path():
    """Create a temporary database file path."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        temp_path = f.name
    yield temp_path
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)

@pytest.fixture
def in_memory_db():
    """Create an in-memory EvalsDB instance."""
    db = EvalsDB(db_path=":memory:", client_id="test_client")
    return db

@pytest.fixture
def temp_db(temp_db_path):
    """Create a temporary file-based EvalsDB instance."""
    db = EvalsDB(db_path=temp_db_path, client_id="test_client")
    return db

# --- Sample Data Fixtures ---

@pytest.fixture
def sample_task_config():
    """Create a sample task configuration."""
    return TaskConfig(
        name="test_task",
        description="A test evaluation task",
        task_type="question_answer",
        dataset_name="test_dataset",
        split="test",
        metric="exact_match",
        generation_kwargs={"temperature": 0.0, "max_tokens": 50},
        metadata={"format": "custom", "version": "1.0"}
    )

@pytest.fixture
def sample_gsm8k_config():
    """Create a GSM8K-style task configuration."""
    return TaskConfig(
        name="gsm8k_sample",
        description="Grade school math problems",
        task_type="question_answer",
        dataset_name="gsm8k",
        split="test",
        metric="exact_match",
        num_fewshot=5,
        generation_kwargs={"temperature": 0.0, "max_tokens": 100},
        metadata={"format": "eleuther", "category": "math"}
    )

@pytest.fixture
def sample_code_config():
    """Create a code evaluation task configuration."""
    return TaskConfig(
        name="humaneval_sample",
        description="Python code generation",
        task_type="code_generation",
        dataset_name="humaneval",
        split="test",
        metric="execution_pass_rate",
        generation_kwargs={"temperature": 0.2, "max_tokens": 512},
        metadata={"format": "custom", "category": "coding", "language": "python"}
    )

@pytest.fixture
def sample_eval_results():
    """Create sample evaluation results."""
    return [
        EvalResult(
            sample_id="sample_1",
            input_text="What is 2 + 2?",
            expected_output="4",
            model_output="4",
            metrics={"exact_match": 1.0, "response_length": 1},
            metadata={"execution_time": 0.5}
        ),
        EvalResult(
            sample_id="sample_2", 
            input_text="What is the capital of France?",
            expected_output="Paris",
            model_output="Paris is the capital of France.",
            metrics={"exact_match": 0.0, "contains_answer": 1.0, "response_length": 30},
            metadata={"execution_time": 0.8}
        ),
        EvalResult(
            sample_id="sample_3",
            input_text="Write a function to add two numbers",
            expected_output="def add(a, b): return a + b",
            model_output="def add(a, b):\n    return a + b",
            metrics={"execution_pass_rate": 1.0, "syntax_valid": 1.0},
            metadata={"execution_time": 1.2}
        )
    ]

# --- Mock LLM Fixtures ---

@pytest.fixture
def mock_llm_interface():
    """Create a mock LLM interface."""
    mock = AsyncMock(spec=LLMInterface)
    
    # Configure default responses
    async def mock_generate(prompt, **kwargs):
        if "2 + 2" in prompt:
            return "4"
        elif "capital of France" in prompt:
            return "Paris"
        elif "function to add" in prompt:
            return "def add(a, b):\n    return a + b"
        else:
            return "Mock response"
    
    mock.generate.side_effect = mock_generate
    mock.provider = "mock"
    mock.model_id = "mock-model"
    
    return mock

@pytest.fixture
def mock_successful_llm():
    """Create a mock LLM that always succeeds."""
    mock = AsyncMock(spec=LLMInterface)
    mock.generate.return_value = "Successful response"
    mock.provider = "mock_success"
    mock.model_id = "success-model"
    return mock

@pytest.fixture
def mock_failing_llm():
    """Create a mock LLM that always fails."""
    mock = AsyncMock(spec=LLMInterface)
    mock.generate.side_effect = Exception("Mock LLM failure")
    mock.provider = "mock_fail"
    mock.model_id = "fail-model"
    return mock

# --- Task File Fixtures ---

@pytest.fixture
def sample_eleuther_task_file(tmp_path):
    """Create a sample Eleuther AI format task file."""
    task_data = {
        "task": "mmlu_sample",
        "dataset_name": "cais/mmlu",
        "dataset_config_name": "abstract_algebra",
        "test_split": "test",
        "fewshot_split": "dev",
        "num_fewshot": 5,
        "output_type": "multiple_choice",
        "metric_list": ["acc"],
        "generation_kwargs": {
            "temperature": 0.0,
            "max_tokens": 1
        }
    }
    
    task_file = tmp_path / "mmlu_sample.yaml"
    with open(task_file, 'w') as f:
        import yaml
        yaml.dump(task_data, f)
    
    return str(task_file)

@pytest.fixture
def sample_custom_task_file(tmp_path):
    """Create a sample custom format task file."""
    task_data = {
        "name": "Custom Q&A Task",
        "description": "Simple question answering evaluation",
        "task_type": "question_answer",
        "dataset_name": "local_qa_dataset.json",
        "split": "test",
        "metric": "exact_match",
        "generation_kwargs": {
            "temperature": 0.0,
            "max_tokens": 50
        }
    }
    
    task_file = tmp_path / "custom_task.json"
    with open(task_file, 'w') as f:
        json.dump(task_data, f, indent=2)
    
    return str(task_file)

@pytest.fixture
def sample_csv_dataset_file(tmp_path):
    """Create a sample CSV dataset file."""
    csv_content = """question,answer,category
"What is 2 + 2?","4","arithmetic"
"What is the capital of France?","Paris","geography"
"Name a programming language","Python","technology"
"""
    
    csv_file = tmp_path / "sample_dataset.csv"
    with open(csv_file, 'w') as f:
        f.write(csv_content)
    
    return str(csv_file)

# --- Component Fixtures ---

@pytest.fixture
def task_loader():
    """Create a TaskLoader instance."""
    return TaskLoader()

@pytest.fixture
def eval_runner(mock_llm_interface):
    """Create an EvalRunner instance with mock LLM."""
    return EvalRunner(llm_interface=mock_llm_interface)

@pytest.fixture
def eval_orchestrator(temp_db):
    """Create an EvaluationOrchestrator instance with temporary database."""
    # Override the default db with our temp db
    orchestrator = EvaluationOrchestrator.__new__(EvaluationOrchestrator)
    orchestrator.db = temp_db
    orchestrator.task_loader = TaskLoader()
    return orchestrator

# --- Async Test Utilities ---

@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def async_setup():
    """Async setup fixture for complex test scenarios."""
    # Any async setup code can go here
    yield
    # Any async cleanup code can go here

# --- Property Test Generators ---

@pytest.fixture
def property_test_data():
    """Generate property test data for hypothesis tests."""
    import string
    import random
    
    def generate_random_question():
        """Generate a random question string."""
        words = ['what', 'how', 'why', 'when', 'where', 'who']
        question = random.choice(words).capitalize()
        for _ in range(random.randint(2, 8)):
            question += f" {random.choice(string.ascii_lowercase * 3)}"
        return question + "?"
    
    def generate_random_answer():
        """Generate a random answer string."""
        length = random.randint(1, 20)
        return ' '.join(random.choice(string.ascii_lowercase * 2) for _ in range(length))
    
    return {
        "generate_question": generate_random_question,
        "generate_answer": generate_random_answer
    }

# --- Performance Test Fixtures ---

@pytest.fixture
def large_dataset_config():
    """Create a configuration for large dataset testing."""
    return TaskConfig(
        name="large_test_task",
        description="Large dataset for performance testing",
        task_type="question_answer",
        dataset_name="large_test_dataset",
        split="test", 
        metric="exact_match",
        generation_kwargs={"temperature": 0.0, "max_tokens": 50},
        metadata={"format": "custom", "size": "large"}
    )

@pytest.fixture
def performance_test_samples():
    """Generate a large number of test samples for performance testing."""
    samples = []
    for i in range(1000):
        samples.append({
            "id": f"sample_{i}",
            "question": f"What is the answer to question {i}?",
            "answer": f"Answer {i}",
            "category": f"category_{i % 10}"
        })
    return samples

# Test utilities
def assert_eval_result_valid(result: EvalResult):
    """Assert that an EvalResult is valid."""
    assert result.sample_id is not None
    assert result.input_text is not None
    assert result.model_output is not None
    assert isinstance(result.metrics, dict)
    assert isinstance(result.metadata, dict)

def assert_task_config_valid(config: TaskConfig):
    """Assert that a TaskConfig is valid."""
    assert config.name is not None
    assert config.task_type is not None
    assert config.metric is not None
    assert isinstance(config.generation_kwargs, dict)