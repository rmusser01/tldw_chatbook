#!/usr/bin/env python3
"""
Test script to verify evaluation metrics are being collected properly.
Run this after setting up the evaluation system to ensure metrics are working.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tldw_chatbook.Metrics.metrics_logger import log_counter, log_histogram
from tldw_chatbook.Evals import EvaluationOrchestrator, TaskLoader
from tldw_chatbook.Evals.cost_estimator import CostEstimator
from tldw_chatbook.DB.Evals_DB import EvalsDB

def test_basic_metrics():
    """Test basic metric logging."""
    print("Testing basic metrics...")
    
    # Test counters
    log_counter("test_eval_counter", labels={"test": "true"})
    
    # Test histograms
    log_histogram("test_eval_duration", 1.5, labels={"test": "true"})
    
    print("✓ Basic metrics logged")

def test_task_loader_metrics():
    """Test task loader metrics."""
    print("\nTesting task loader metrics...")
    
    try:
        loader = TaskLoader()
        
        # Test template creation (should log metrics)
        task_config = loader.create_task_from_template("simple_qa")
        print(f"✓ Created task from template: {task_config.name}")
        
        # Test validation (should log metrics)
        issues = loader.validate_task(task_config)
        print(f"✓ Validated task, issues: {len(issues)}")
        
    except Exception as e:
        print(f"✗ Task loader test failed: {e}")

def test_cost_estimator_metrics():
    """Test cost estimator metrics."""
    print("\nTesting cost estimator metrics...")
    
    try:
        estimator = CostEstimator()
        
        # Test cost estimation (should log metrics)
        estimate = estimator.estimate_run_cost(
            provider="openai",
            model_id="gpt-3.5-turbo",
            num_samples=100
        )
        print(f"✓ Estimated cost: ${estimate['estimated_cost']:.4f}")
        
        # Test tracking
        run_id = "test-run-123"
        estimator.start_tracking(run_id)
        
        # Add sample costs
        sample_cost = estimator.add_sample_cost(
            run_id=run_id,
            input_tokens=500,
            output_tokens=200,
            provider="openai",
            model_id="gpt-3.5-turbo"
        )
        print(f"✓ Added sample cost: ${sample_cost:.6f}")
        
        # Finalize run
        final_record = estimator.finalize_run(run_id, {
            "provider": "openai",
            "model_id": "gpt-3.5-turbo"
        })
        print(f"✓ Finalized run with total cost: ${final_record['total_cost']:.4f}")
        
    except Exception as e:
        print(f"✗ Cost estimator test failed: {e}")

def test_database_metrics():
    """Test database operation metrics."""
    print("\nTesting database metrics...")
    
    try:
        # Use in-memory database for testing
        db = EvalsDB(":memory:")
        
        # Test task creation (should log metrics)
        task_id = db.create_task(
            name="Test Task",
            description="Test task for metrics",
            task_type="simple_qa",
            config_format="custom",
            config_data={"test": True}
        )
        print(f"✓ Created task with ID: {task_id}")
        
        # Test task retrieval (should log metrics)
        task = db.get_task(task_id)
        print(f"✓ Retrieved task: {task['name']}")
        
        # Test listing (should log metrics)
        tasks = db.list_tasks(limit=10)
        print(f"✓ Listed {len(tasks)} tasks")
        
    except Exception as e:
        print(f"✗ Database test failed: {e}")

async def test_orchestrator_metrics():
    """Test orchestrator metrics (requires async)."""
    print("\nTesting orchestrator metrics...")
    
    try:
        orchestrator = EvaluationOrchestrator()
        
        # Create a simple task
        task_id = orchestrator.db.create_task(
            name="Metrics Test Task",
            description="Task for testing metrics",
            task_type="simple_qa",
            config_format="custom",
            config_data={
                "dataset_name": "test_dataset",
                "metric": "exact_match"
            }
        )
        print(f"✓ Created test task: {task_id}")
        
        # Note: We can't run a full evaluation without models configured,
        # but the task creation should have logged metrics
        
    except Exception as e:
        print(f"✗ Orchestrator test failed: {e}")

def check_prometheus_metrics():
    """Check if Prometheus metrics are available."""
    print("\nChecking Prometheus metrics endpoint...")
    
    try:
        import requests
        # Try to access the metrics endpoint
        response = requests.get("http://localhost:9090/metrics", timeout=2)
        if response.status_code == 200:
            # Count eval metrics
            eval_metrics = [line for line in response.text.split('\n') 
                          if line.startswith('eval_') and not line.startswith('#')]
            print(f"✓ Found {len(eval_metrics)} evaluation metrics")
            
            # Show a few examples
            print("\nSample metrics:")
            for metric in eval_metrics[:5]:
                print(f"  - {metric.split('{')[0]}")
            if len(eval_metrics) > 5:
                print(f"  ... and {len(eval_metrics) - 5} more")
        else:
            print(f"✗ Metrics endpoint returned status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to metrics endpoint (is Prometheus running?)")
    except requests.exceptions.Timeout:
        print("✗ Metrics endpoint timed out")
    except ImportError:
        print("✗ 'requests' library not available for metrics check")

def main():
    """Run all metric tests."""
    print("=" * 60)
    print("Evaluation Metrics Test Suite")
    print("=" * 60)
    
    # Run synchronous tests
    test_basic_metrics()
    test_task_loader_metrics()
    test_cost_estimator_metrics()
    test_database_metrics()
    
    # Run async tests
    asyncio.run(test_orchestrator_metrics())
    
    # Check Prometheus endpoint
    check_prometheus_metrics()
    
    print("\n" + "=" * 60)
    print("Testing complete!")
    print("Check your metrics backend to verify data collection.")
    print("=" * 60)

if __name__ == "__main__":
    main()