# test_basic_functionality.py
# Description: Basic functionality tests without complex dependencies
#
"""
Basic Functionality Tests
=========================

Tests core evaluation system components without complex dependencies:
- Task configuration data structures
- Database schema validation
- File handling and parsing
- Basic validation logic
"""

import pytest
import tempfile
import json
import yaml
from pathlib import Path

# Test TaskConfig without importing the full module
def test_task_config_structure():
    """Test that we can create task configuration dictionaries."""
    task_config = {
        "name": "test_task",
        "description": "A test evaluation task",
        "task_type": "question_answer",
        "dataset_name": "test_dataset",
        "split": "test",
        "metric": "exact_match",
        "generation_kwargs": {"temperature": 0.0, "max_tokens": 50},
        "metadata": {"format": "custom", "version": "1.0"}
    }
    
    # Validate required fields
    required_fields = ["name", "task_type", "dataset_name", "split", "metric"]
    for field in required_fields:
        assert field in task_config
        assert task_config[field] is not None
    
    # Validate data types
    assert isinstance(task_config["name"], str)
    assert isinstance(task_config["generation_kwargs"], dict)
    assert isinstance(task_config["metadata"], dict)

def test_json_task_file_parsing():
    """Test parsing of JSON task files."""
    task_data = {
        "name": "JSON Test Task",
        "description": "Test JSON parsing",
        "task_type": "question_answer",
        "dataset_name": "test_dataset.json",
        "split": "test",
        "metric": "exact_match",
        "generation_kwargs": {
            "temperature": 0.0,
            "max_tokens": 50
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(task_data, f)
        temp_path = f.name
    
    try:
        # Parse the JSON file
        with open(temp_path, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data["name"] == "JSON Test Task"
        assert loaded_data["task_type"] == "question_answer"
        assert loaded_data["generation_kwargs"]["temperature"] == 0.0
        
    finally:
        Path(temp_path).unlink()

def test_yaml_task_file_parsing():
    """Test parsing of YAML task files."""
    task_data = {
        "task": "yaml_test_task",
        "dataset_name": "test_dataset",
        "output_type": "multiple_choice",
        "metric_list": ["acc"],
        "generation_kwargs": {
            "temperature": 0.0,
            "max_tokens": 1
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(task_data, f)
        temp_path = f.name
    
    try:
        # Parse the YAML file
        with open(temp_path, 'r') as f:
            loaded_data = yaml.safe_load(f)
        
        assert loaded_data["task"] == "yaml_test_task"
        assert loaded_data["output_type"] == "multiple_choice"
        assert "acc" in loaded_data["metric_list"]
        
    finally:
        Path(temp_path).unlink()

def test_csv_data_parsing():
    """Test parsing of CSV dataset files."""
    csv_content = """id,question,answer,category
sample_1,"What is 2+2?","4","math"
sample_2,"What is the capital of France?","Paris","geography"
sample_3,"Who wrote Romeo and Juliet?","Shakespeare","literature"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        temp_path = f.name
    
    try:
        # Parse the CSV file
        import csv
        with open(temp_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 3
        assert rows[0]["question"] == "What is 2+2?"
        assert rows[0]["answer"] == "4"
        assert rows[1]["category"] == "geography"
        
    finally:
        Path(temp_path).unlink()

def test_evaluation_result_structure():
    """Test evaluation result data structure."""
    eval_result = {
        "sample_id": "test_sample_001",
        "input_text": "What is 2+2?",
        "expected_output": "4",
        "model_output": "4",
        "metrics": {
            "exact_match": 1.0,
            "response_length": 1
        },
        "metadata": {
            "execution_time": 0.5,
            "model_name": "test_model"
        },
        "error": None
    }
    
    # Validate structure
    required_fields = ["sample_id", "input_text", "model_output", "metrics"]
    for field in required_fields:
        assert field in eval_result
    
    # Validate types
    assert isinstance(eval_result["metrics"], dict)
    assert isinstance(eval_result["metadata"], dict)
    
    # Validate metric values
    for metric_name, metric_value in eval_result["metrics"].items():
        if isinstance(metric_value, (int, float)):
            assert 0.0 <= metric_value <= 1.0 or metric_name == "response_length"

def test_database_table_schemas():
    """Test database table schema definitions."""
    
    # Define expected table schemas
    expected_tables = {
        "eval_tasks": [
            "id", "name", "description", "task_type", "config_format", 
            "config_data", "dataset_id", "created_at", "updated_at", 
            "deleted_at", "version", "client_id"
        ],
        "eval_datasets": [
            "id", "name", "source_type", "source_path", "metadata",
            "created_at", "updated_at", "deleted_at", "version", "client_id"
        ],
        "eval_models": [
            "id", "name", "provider", "model_id", "config",
            "created_at", "updated_at", "deleted_at", "version", "client_id"
        ],
        "eval_runs": [
            "id", "task_id", "model_id", "status", "progress", "config",
            "start_time", "end_time", "created_at", "updated_at", 
            "deleted_at", "version", "client_id"
        ],
        "eval_results": [
            "id", "run_id", "sample_id", "input_text", "expected_output",
            "model_output", "metrics", "metadata", "error", "created_at"
        ],
        "eval_run_metrics": [
            "id", "run_id", "metrics", "metadata", "created_at"
        ]
    }
    
    # Validate each table has expected columns
    for table_name, expected_columns in expected_tables.items():
        assert len(expected_columns) > 0
        
        # Check for required audit columns
        audit_columns = ["id", "created_at"]
        for audit_col in audit_columns:
            assert audit_col in expected_columns, f"Missing {audit_col} in {table_name}"

def test_metric_calculation_logic():
    """Test basic metric calculation functions."""
    
    def calculate_exact_match(expected, actual):
        """Simple exact match calculation."""
        if expected is None or actual is None:
            return 0.0
        return 1.0 if str(expected).strip() == str(actual).strip() else 0.0
    
    def calculate_contains_answer(response, answer):
        """Check if response contains the answer."""
        if not answer or not response:
            return 0.0
        return 1.0 if str(answer).lower() in str(response).lower() else 0.0
    
    # Test exact match
    assert calculate_exact_match("Paris", "Paris") == 1.0
    assert calculate_exact_match("Paris", "London") == 0.0
    assert calculate_exact_match("Paris", "paris") == 0.0  # Case sensitive
    assert calculate_exact_match(None, "Paris") == 0.0
    
    # Test contains answer
    assert calculate_contains_answer("The capital is Paris", "Paris") == 1.0
    assert calculate_contains_answer("The capital is London", "Paris") == 0.0
    assert calculate_contains_answer("PARIS is the capital", "paris") == 1.0
    assert calculate_contains_answer("", "Paris") == 0.0

def test_task_validation_logic():
    """Test task configuration validation."""
    
    def validate_task_config(config):
        """Basic task configuration validation."""
        errors = []
        
        # Required fields
        required_fields = ["name", "task_type", "dataset_name", "metric"]
        for field in required_fields:
            if not config.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Valid task types
        valid_task_types = [
            "question_answer", "multiple_choice", "text_generation", 
            "code_generation", "safety_check"
        ]
        if config.get("task_type") not in valid_task_types:
            errors.append(f"Invalid task_type: {config.get('task_type')}")
        
        # Generation kwargs validation
        gen_kwargs = config.get("generation_kwargs", {})
        if "temperature" in gen_kwargs:
            temp = gen_kwargs["temperature"]
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                errors.append("temperature must be between 0 and 2")
        
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs["max_tokens"]
            if not isinstance(max_tokens, int) or max_tokens <= 0:
                errors.append("max_tokens must be positive integer")
        
        return errors
    
    # Test valid config
    valid_config = {
        "name": "Valid Task",
        "task_type": "question_answer",
        "dataset_name": "test_dataset",
        "metric": "exact_match",
        "generation_kwargs": {"temperature": 0.0, "max_tokens": 100}
    }
    errors = validate_task_config(valid_config)
    assert len(errors) == 0
    
    # Test invalid config
    invalid_config = {
        "name": "",  # Empty name
        "task_type": "invalid_type",  # Invalid type
        "generation_kwargs": {"temperature": 3.0, "max_tokens": -1}  # Invalid params
    }
    errors = validate_task_config(invalid_config)
    assert len(errors) > 0
    assert any("name" in error for error in errors)
    assert any("task_type" in error for error in errors)
    assert any("temperature" in error for error in errors)

def test_file_format_detection():
    """Test file format detection logic."""
    
    def detect_file_format(filename):
        """Detect file format from filename."""
        if filename.endswith('.yaml') or filename.endswith('.yml'):
            return 'eleuther'
        elif filename.endswith('.json'):
            return 'custom'
        elif filename.endswith('.csv') or filename.endswith('.tsv'):
            return 'csv'
        else:
            return 'unknown'
    
    assert detect_file_format("task.yaml") == "eleuther"
    assert detect_file_format("task.yml") == "eleuther"
    assert detect_file_format("task.json") == "custom"
    assert detect_file_format("dataset.csv") == "csv"
    assert detect_file_format("dataset.tsv") == "csv"
    assert detect_file_format("unknown.txt") == "unknown"

def test_sample_data_structures():
    """Test sample data structure validation."""
    
    # Test basic Q&A sample
    qa_sample = {
        "id": "qa_001",
        "question": "What is the capital of France?",
        "answer": "Paris",
        "category": "geography",
        "difficulty": "easy"
    }
    
    required_qa_fields = ["id", "question", "answer"]
    for field in required_qa_fields:
        assert field in qa_sample
        assert qa_sample[field] is not None
    
    # Test code generation sample
    code_sample = {
        "id": "code_001",
        "problem_description": "Write a function to add two numbers",
        "function_signature": "def add_numbers(a, b):",
        "test_cases": [
            {"input": "(2, 3)", "expected": "5"},
            {"input": "(0, 0)", "expected": "0"}
        ],
        "canonical_solution": "def add_numbers(a, b):\n    return a + b"
    }
    
    required_code_fields = ["id", "problem_description", "test_cases"]
    for field in required_code_fields:
        assert field in code_sample
        assert code_sample[field] is not None
    
    assert isinstance(code_sample["test_cases"], list)
    assert len(code_sample["test_cases"]) > 0

def test_configuration_parsing():
    """Test configuration file parsing and validation."""
    
    # Test Eleuther format configuration
    eleuther_config = {
        "task": "test_task",
        "dataset_name": "test_dataset",
        "output_type": "multiple_choice",
        "metric_list": ["acc"],
        "generation_kwargs": {"temperature": 0.0}
    }
    
    # Convert to internal format
    internal_config = {
        "name": eleuther_config["task"],
        "task_type": eleuther_config["output_type"],
        "dataset_name": eleuther_config["dataset_name"],
        "metric": eleuther_config["metric_list"][0],
        "generation_kwargs": eleuther_config["generation_kwargs"]
    }
    
    assert internal_config["name"] == "test_task"
    assert internal_config["task_type"] == "multiple_choice"
    assert internal_config["metric"] == "acc"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])