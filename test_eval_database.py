#!/usr/bin/env python3
"""
Test script for evaluation database performance and functionality.
This script tests the database with realistic evaluation workloads.
"""

import asyncio
import time
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

# Add the project root to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.App_Functions.Evals.task_loader import TaskLoader
from tldw_chatbook.App_Functions.Evals.eval_runner import EvalRunner
from tldw_chatbook.App_Functions.Evals.llm_interface import LLMInterface

def create_test_task_data() -> List[Dict[str, Any]]:
    """Create test task data for database testing."""
    tasks = []
    
    # Math task
    tasks.append({
        'name': 'Test Math Problems',
        'description': 'Simple arithmetic for testing',
        'task_type': 'question_answer',
        'dataset_name': 'test_math',
        'config': {
            'max_samples': 100,
            'temperature': 0.0,
            'max_tokens': 50
        }
    })
    
    # Classification task
    tasks.append({
        'name': 'Test Classification',
        'description': 'Sentiment classification for testing',
        'task_type': 'classification', 
        'dataset_name': 'test_sentiment',
        'config': {
            'max_samples': 200,
            'temperature': 0.0,
            'max_tokens': 10
        }
    })
    
    # Generation task
    tasks.append({
        'name': 'Test Generation',
        'description': 'Text generation for testing',
        'task_type': 'generation',
        'dataset_name': 'test_generation',
        'config': {
            'max_samples': 50,
            'temperature': 0.3,
            'max_tokens': 100
        }
    })
    
    return tasks

def create_test_model_data() -> List[Dict[str, Any]]:
    """Create test model data for database testing."""
    models = []
    
    models.append({
        'name': 'Test GPT-3.5',
        'provider': 'openai',
        'model_id': 'gpt-3.5-turbo',
        'config': {
            'temperature': 0.0,
            'max_tokens': 1024,
            'api_key': 'test-key'
        }
    })
    
    models.append({
        'name': 'Test Claude',
        'provider': 'anthropic',
        'model_id': 'claude-3-sonnet-20240229',
        'config': {
            'temperature': 0.0,
            'max_tokens': 1024,
            'api_key': 'test-key'
        }
    })
    
    models.append({
        'name': 'Test Groq',
        'provider': 'groq',
        'model_id': 'llama-3.1-70b-versatile',
        'config': {
            'temperature': 0.0,
            'max_tokens': 1024,
            'api_key': 'test-key'
        }
    })
    
    return models

def create_test_results(num_samples: int = 100) -> List[Dict[str, Any]]:
    """Create synthetic test results."""
    results = []
    
    for i in range(num_samples):
        # Create realistic test results
        score = random.uniform(0.0, 1.0)
        
        result = {
            'sample_id': f'test_sample_{i}',
            'input_data': {'question': f'Test question {i}: What is 2 + 2?'},
            'expected_output': '4',
            'actual_output': '4' if score > 0.5 else str(random.randint(0, 10)),
            'metrics': {
                'exact_match': 1.0 if score > 0.5 else 0.0,
                'response_time': random.uniform(0.5, 3.0),
                'token_count': random.randint(1, 20)
            },
            'metadata': {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'model_version': 'test-v1.0',
                'prompt_tokens': random.randint(10, 50),
                'completion_tokens': random.randint(1, 20)
            }
        }
        
        results.append(result)
    
    return results

async def test_database_operations():
    """Test basic database operations."""
    print("🧪 Testing database operations...")
    
    # Use temporary database for testing
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_file:
        temp_db_path = temp_file.name
    
    try:
        # Initialize database
        db = EvalsDB(temp_db_path)
        
        # Test 1: Create test data
        print("📝 Creating test tasks...")
        test_tasks = create_test_task_data()
        task_ids = []
        
        start_time = time.time()
        for task_data in test_tasks:
            task_id = db.create_task(
                name=task_data['name'],
                description=task_data['description'],
                task_type=task_data['task_type'],
                config_format='custom',
                config_data=task_data['config']
            )
            task_ids.append(task_id)
        
        task_creation_time = time.time() - start_time
        print(f"✅ Created {len(task_ids)} tasks in {task_creation_time:.3f}s")
        
        # Test 2: Create test models
        print("🤖 Creating test models...")
        test_models = create_test_model_data()
        model_ids = []
        
        start_time = time.time()
        for model_data in test_models:
            model_id = db.create_model(
                name=model_data['name'],
                provider=model_data['provider'],
                model_id=model_data['model_id'],
                config=model_data['config']
            )
            model_ids.append(model_id)
        
        model_creation_time = time.time() - start_time
        print(f"✅ Created {len(model_ids)} models in {model_creation_time:.3f}s")
        
        # Test 3: Create evaluation runs
        print("🏃 Creating evaluation runs...")
        run_ids = []
        
        start_time = time.time()
        for i, (task_id, model_id) in enumerate(zip(task_ids, model_ids)):
            run_id = db.create_run(
                name=f"Test Run {i+1}",
                task_id=task_id,
                model_id=model_id,
                config_overrides={'test_mode': True}
            )
            run_ids.append(run_id)
        
        run_creation_time = time.time() - start_time
        print(f"✅ Created {len(run_ids)} runs in {run_creation_time:.3f}s")
        
        # Test 4: Bulk insert results
        print("📊 Inserting test results...")
        
        start_time = time.time()
        total_results = 0
        
        for run_id in run_ids:
            test_results = create_test_results(100)  # 100 results per run
            
            for result in test_results:
                db.store_result(
                    run_id=run_id,
                    sample_id=result['sample_id'],
                    input_data=result['input_data'],
                    expected_output=result['expected_output'],
                    actual_output=result['actual_output'],
                    metrics=result['metrics'],
                    metadata=result['metadata']
                )
                total_results += 1
        
        results_insertion_time = time.time() - start_time
        print(f"✅ Inserted {total_results} results in {results_insertion_time:.3f}s")
        print(f"   Rate: {total_results/results_insertion_time:.1f} results/second")
        
        # Test 5: Query performance
        print("🔍 Testing query performance...")
        
        # Test basic queries
        start_time = time.time()
        all_tasks = db.list_tasks()
        query_time_1 = time.time() - start_time
        print(f"✅ Retrieved {len(all_tasks)} tasks in {query_time_1:.3f}s")
        
        start_time = time.time()
        all_models = db.list_models()
        query_time_2 = time.time() - start_time
        print(f"✅ Retrieved {len(all_models)} models in {query_time_2:.3f}s")
        
        start_time = time.time()
        all_runs = db.list_runs()
        query_time_3 = time.time() - start_time
        print(f"✅ Retrieved {len(all_runs)} runs in {query_time_3:.3f}s")
        
        # Test results queries
        start_time = time.time()
        for run_id in run_ids:
            run_results = db.get_run_results(run_id)
        query_time_4 = time.time() - start_time
        print(f"✅ Retrieved results for {len(run_ids)} runs in {query_time_4:.3f}s")
        
        # Test search functionality
        start_time = time.time()
        search_results = db.search_tasks("Test")
        query_time_5 = time.time() - start_time
        print(f"✅ Searched tasks in {query_time_5:.3f}s, found {len(search_results)} matches")
        
        # Test 6: Calculate run metrics
        print("📈 Testing metrics calculation...")
        
        start_time = time.time()
        for run_id in run_ids:
            run_results = db.get_run_results(run_id)
            
            # Calculate simple metrics
            if run_results:
                total_samples = len(run_results)
                exact_matches = sum(1 for r in run_results if r.get('metrics', {}).get('exact_match', 0) == 1.0)
                accuracy = exact_matches / total_samples if total_samples > 0 else 0.0
                avg_response_time = sum(r.get('metrics', {}).get('response_time', 0) for r in run_results) / total_samples
                
                # Store aggregated metrics (format: metric_name -> (value, type))
                metrics = {
                    'accuracy': (accuracy, 'accuracy'),
                    'total_samples': (total_samples, 'custom'),
                    'successful_samples': (exact_matches, 'custom'),
                    'avg_response_time': (avg_response_time, 'custom')
                }
                
                db.store_run_metrics(run_id, metrics)
        
        metrics_time = time.time() - start_time
        print(f"✅ Calculated and stored metrics for {len(run_ids)} runs in {metrics_time:.3f}s")
        
        # Test 7: Performance summary
        print("\\n📊 Performance Summary:")
        print(f"  Task Creation: {task_creation_time:.3f}s for {len(task_ids)} tasks")
        print(f"  Model Creation: {model_creation_time:.3f}s for {len(model_ids)} models")
        print(f"  Run Creation: {run_creation_time:.3f}s for {len(run_ids)} runs")
        print(f"  Results Insertion: {results_insertion_time:.3f}s for {total_results} results")
        print(f"  Results Rate: {total_results/results_insertion_time:.1f} results/second")
        print(f"  Query Performance: {query_time_1+query_time_2+query_time_3+query_time_4+query_time_5:.3f}s total")
        print(f"  Metrics Calculation: {metrics_time:.3f}s")
        
        # Test 8: Verify data integrity
        print("\\n🔍 Verifying data integrity...")
        
        # Check foreign key relationships
        integrity_issues = 0
        
        for run in all_runs:
            # Verify task exists
            task = db.get_task(run['task_id'])
            if not task:
                print(f"❌ Run {run['id']} references non-existent task {run['task_id']}")
                integrity_issues += 1
            
            # Verify model exists
            model = db.get_model(run['model_id'])
            if not model:
                print(f"❌ Run {run['id']} references non-existent model {run['model_id']}")
                integrity_issues += 1
        
        if integrity_issues == 0:
            print("✅ Data integrity check passed")
        else:
            print(f"❌ Found {integrity_issues} integrity issues")
        
        print(f"\\n🎉 Database test completed successfully!")
        print(f"   Database size: {Path(temp_db_path).stat().st_size / 1024:.1f} KB")
        
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up temporary database
        try:
            Path(temp_db_path).unlink()
        except:
            pass

async def test_concurrent_operations():
    """Test database performance under concurrent load."""
    print("\\n🔄 Testing concurrent database operations...")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_file:
        temp_db_path = temp_file.name
    
    try:
        # Create database and initial data
        db = EvalsDB(temp_db_path)
        
        # Create a task and model for testing
        task_id = db.create_task(
            name="Concurrent Test Task",
            description="Task for concurrent testing",
            task_type="question_answer",
            config_format="custom",
            config_data={}
        )
        
        model_id = db.create_model(
            name="Concurrent Test Model",
            provider="test",
            model_id="test-model",
            config={}
        )
        
        run_id = db.create_run(
            name="Concurrent Test Run",
            task_id=task_id,
            model_id=model_id,
            config_overrides={}
        )
        
        # Test concurrent result insertion
        async def insert_results_batch(batch_id: int, batch_size: int = 50):
            """Insert a batch of results concurrently."""
            db_instance = EvalsDB(temp_db_path)  # Each task gets its own connection
            
            for i in range(batch_size):
                sample_id = f"batch_{batch_id}_sample_{i}"
                result = {
                    'sample_id': sample_id,
                    'input_data': {'question': f'Concurrent test {batch_id}-{i}'},
                    'expected_output': 'test',
                    'actual_output': 'test',
                    'metrics': {'score': random.uniform(0.0, 1.0)},
                    'metadata': {'batch_id': batch_id}
                }
                
                db_instance.store_result(
                    run_id=run_id,
                    sample_id=result['sample_id'],
                    input_data=result['input_data'],
                    expected_output=result['expected_output'],
                    actual_output=result['actual_output'],
                    metrics=result['metrics'],
                    metadata=result['metadata']
                )
        
        # Run concurrent insertions
        num_batches = 5
        batch_size = 20
        
        start_time = time.time()
        
        tasks = [insert_results_batch(i, batch_size) for i in range(num_batches)]
        await asyncio.gather(*tasks)
        
        concurrent_time = time.time() - start_time
        total_concurrent_results = num_batches * batch_size
        
        print(f"✅ Concurrent insertion: {total_concurrent_results} results in {concurrent_time:.3f}s")
        print(f"   Rate: {total_concurrent_results/concurrent_time:.1f} results/second")
        
        # Verify all results were inserted
        all_results = db.get_run_results(run_id)
        print(f"✅ Verified: {len(all_results)} results stored (expected {total_concurrent_results})")
        
        if len(all_results) == total_concurrent_results:
            print("✅ Concurrent operations test passed")
            return True
        else:
            print(f"❌ Expected {total_concurrent_results} results, found {len(all_results)}")
            return False
        
    except Exception as e:
        print(f"❌ Concurrent operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        try:
            Path(temp_db_path).unlink()
        except:
            pass

async def main():
    """Run all database tests."""
    print("🚀 Starting Evaluation Database Tests\\n")
    
    # Test 1: Basic database operations
    test1_success = await test_database_operations()
    
    # Test 2: Concurrent operations
    test2_success = await test_concurrent_operations()
    
    # Summary
    print("\\n" + "="*50)
    print("📋 Test Summary:")
    print(f"  Basic Operations: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"  Concurrent Operations: {'✅ PASS' if test2_success else '❌ FAIL'}")
    
    if test1_success and test2_success:
        print("\\n🎉 All database tests passed!")
        print("✅ Database is ready for production use")
        return True
    else:
        print("\\n❌ Some tests failed - database needs attention")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)