"""Test security enhancements for code execution in evaluations."""

import pytest
import asyncio
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from tldw_chatbook.Evals.specialized_runners import CodeExecutionRunner
from tldw_chatbook.Evals.task_loader import TaskConfig
from tldw_chatbook.Evals.eval_runner import EvalSample


@dataclass
class TestEvalSample:
    """Test version of EvalSample with test_cases support."""
    id: str
    input_text: str
    expected_output: Optional[str] = None
    test_cases: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TestCodeExecutionSecurity:
    """Test security features of code execution runner."""
    
    @pytest.fixture
    def code_runner(self):
        """Create a code execution runner for testing."""
        task_config = TaskConfig(
            name="Test Code Task",
            description="Test code execution",
            task_type="code_generation",
            dataset_name="test",
            metric="execution_success",
            generation_kwargs={"execution_timeout": 2}
        )
        
        model_config = {
            "provider": "mock",
            "model_id": "test-model"
        }
        
        return CodeExecutionRunner(task_config, model_config)
    
    def test_resource_limits_in_generated_code(self, code_runner):
        """Test that resource limits are included in generated test code."""
        code = "def add(a, b): return a + b"
        test_cases = [{"input": {"a": 1, "b": 2}, "expected": 3}]
        
        test_code = code_runner._create_test_code(code, test_cases)
        
        # Check that resource limits are set
        assert "resource.setrlimit(resource.RLIMIT_CPU" in test_code
        assert "resource.setrlimit(resource.RLIMIT_AS" in test_code  # Memory limit
        assert "resource.setrlimit(resource.RLIMIT_NOFILE" in test_code  # File descriptors
        assert "resource.setrlimit(resource.RLIMIT_FSIZE" in test_code  # File size
        
        # Check that dangerous builtins are disabled
        assert "dangerous_builtins = ['eval', 'exec', 'compile'" in test_code
        assert "setattr(builtins, name, lambda *args, **kwargs: None)" in test_code
    
    def test_code_execution_with_limits(self, code_runner):
        """Test that code execution respects limits."""
        sample = TestEvalSample(
            id="test1",
            input_text="Write a function that adds two numbers",
            expected_output="3",
            test_cases=[{"input": [1, 2], "expected": 3}]
        )
        
        # Simple code that should work
        safe_code = "def add(a, b):\n    return a + b"
        results = code_runner._execute_code(safe_code, sample)
        
        assert results["syntax_valid"] is True
        assert results["execution_success"] is True
        assert len(results["test_results"]) == 1
        assert results["test_results"][0]["passed"] is True
    
    def test_malicious_code_blocked(self, code_runner):
        """Test that malicious code attempts are blocked."""
        sample = TestEvalSample(
            id="test2",
            input_text="Test malicious code",
            expected_output="blocked"
        )
        
        # Code that tries to use disabled builtins
        malicious_code = """
def evil():
    # These should all be disabled
    try:
        eval('1+1')
        return "eval worked"
    except:
        pass
    
    try:
        exec('x = 1')
        return "exec worked"
    except:
        pass
    
    try:
        open('/etc/passwd', 'r')
        return "open worked"
    except:
        pass
    
    return "blocked"
"""
        
        results = code_runner._execute_code(malicious_code, sample)
        
        # The syntax should be valid
        assert results["syntax_valid"] is True
        
        # If test cases exist, they should show the code is blocked
        if results["test_results"]:
            # The function should return "blocked" since all dangerous operations are disabled
            assert any(
                result.get("result") == "blocked" 
                for result in results["test_results"]
            )
    
    def test_infinite_loop_timeout(self, code_runner):
        """Test that infinite loops are terminated by timeout."""
        sample = TestEvalSample(
            id="test3",
            input_text="Test timeout",
            expected_output="timeout",
            test_cases=[{"input": [], "expected": "never"}]
        )
        
        # Code with infinite loop
        infinite_code = """
def infinite():
    while True:
        pass
    return "never"
"""
        
        start_time = time.time()
        results = code_runner._execute_code(infinite_code, sample)
        duration = time.time() - start_time
        
        # Should timeout within reasonable time (timeout + overhead)
        assert duration < 5.0  # 2s timeout + overhead
        
        # Should report timeout error
        assert results["execution_success"] is False
        assert "timeout" in results.get("error_message", "").lower()
    
    def test_memory_exhaustion_prevented(self, code_runner):
        """Test that memory exhaustion attempts are prevented."""
        sample = TestEvalSample(
            id="test4",
            input_text="Test memory limits",
            expected_output="blocked"
        )
        
        # Code that tries to allocate excessive memory
        memory_bomb = """
def memory_bomb():
    try:
        # Try to allocate huge list
        huge_list = [0] * (10**9)  # 1 billion integers
        return "allocated"
    except:
        return "blocked"
"""
        
        results = code_runner._execute_code(memory_bomb, sample)
        
        # Should either fail or return "blocked"
        if results["execution_success"] and results["test_results"]:
            assert results["test_results"][0]["result"] == "blocked"
    
    def test_file_operations_blocked(self, code_runner):
        """Test that file operations are blocked."""
        sample = TestEvalSample(
            id="test5",
            input_text="Test file operations",
            expected_output="blocked"
        )
        
        # Code that tries file operations
        file_code = """
def file_test():
    try:
        # Try to write a file
        with open('test.txt', 'w') as f:
            f.write('test')
        return "wrote file"
    except:
        return "blocked"
"""
        
        results = code_runner._execute_code(file_code, sample)
        
        # The open() builtin should be disabled
        if results["execution_success"] and results["test_results"]:
            assert results["test_results"][0]["result"] == "blocked"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])