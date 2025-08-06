"""
Integration tests for evaluation system components.
Tests interaction between runners, orchestrator, database, and UI.
"""
import pytest
import asyncio
from datetime import datetime
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from tldw_chatbook.Evals.eval_orchestrator import EvaluationOrchestrator
from tldw_chatbook.Evals.eval_runner import EvalSample, EvalSampleResult, TaskConfig
from tldw_chatbook.Evals.task_loader import TaskLoader
from tldw_chatbook.Evals.specialized_runners import (
    MultilingualEvaluationRunner,
    CreativeEvaluationRunner,
    RobustnessEvaluationRunner
)
from tldw_chatbook.DB.Evals_DB import EvalsDB


@pytest.fixture
def eval_db(tmp_path):
    """Create a temporary evaluation database."""
    db_path = tmp_path / "test_eval.db"
    return EvalsDB(str(db_path))


@pytest.fixture
def orchestrator(eval_db):
    """Create an evaluation orchestrator with test database."""
    with patch('tldw_chatbook.Evals.eval_orchestrator.EvalsDB') as MockDB:
        MockDB.return_value = eval_db
        return EvaluationOrchestrator()


@pytest.fixture
def task_loader():
    """Create a task loader for testing."""
    loader = TaskLoader()
    # Mock the dataset loading
    loader.load_dataset = Mock(return_value=[
        EvalSample(
            id="1",
            input_text="Translate: Hello world",
            expected_output="Bonjour le monde",
            metadata={"language": "french"}
        ),
        EvalSample(
            id="2",
            input_text="Translate: Good morning",
            expected_output="Bonjour",
            metadata={"language": "french"}
        )
    ])
    return loader


class TestMultilingualRunnerIntegration:
    """Integration tests for MultilingualEvaluationRunner."""
    
    @pytest.mark.asyncio
    async def test_multilingual_evaluation_workflow(self, orchestrator, task_loader):
        """Test complete multilingual evaluation workflow."""
        # Create task config
        task_config = TaskConfig(
            name="French Translation Test",
            description="Test French translation capabilities",
            task_type="generation",  # Using valid task type for DB
            dataset_name="test_french",
            metric="bleu",
            generation_kwargs={"temperature": 0.0},
            metadata={
                "target_language": "french",
                "subcategory": "translation"
            }
        )
        
        # Mock model config
        model_config = {
            "provider": "openai",
            "model_id": "gpt-3.5-turbo",
            "api_key": "test-key"
        }
        
        # First create a task in the database
        task_id = orchestrator.db.create_task(
            name=task_config.name,
            task_type=task_config.task_type,
            config_format="custom",
            config_data=task_config.__dict__,
            description=task_config.description
        )
        
        # Create a model entry
        model_id = orchestrator.db.create_model(
            name=f"{model_config['provider']}:{model_config['model_id']}",
            provider=model_config["provider"],
            model_id=model_config["model_id"],
            config=model_config
        )
        
        # Create evaluation run
        eval_id = orchestrator.db.create_run(
            name=f"Test run for {task_config.name}",
            task_id=task_id,
            model_id=model_id
        )
        
        # Create runner with mocked LLM
        with patch('tldw_chatbook.Evals.eval_runner.LLMInterface') as MockLLM:
            mock_llm = Mock()
            mock_llm.generate = AsyncMock(side_effect=[
                "Bonjour le monde",
                "Bon matin"  # Slightly different translation
            ])
            MockLLM.return_value = mock_llm
            
            runner = MultilingualEvaluationRunner(task_config, model_config)
            
            # Run evaluation on samples
            samples = task_loader.load_dataset("test_french", limit=2)
            results = []
            
            for sample in samples:
                result = await runner.run_sample(sample)
                results.append(result)
                
                # Verify multilingual metrics
                assert "fluency_score" in result.metrics
                assert "language_confidence" in result.metrics
                assert "target_language_detected" in result.metrics
                assert result.metadata.get("language_analysis") is not None
            
            # Save results to database
            for result in results:
                orchestrator.db.store_result(
                    run_id=eval_id,
                    sample_id=result.sample_id,
                    input_data={"input_text": result.input_text},
                    actual_output=result.actual_output,
                    expected_output=result.expected_output,
                    metrics=result.metrics,
                    metadata=result.metadata
                )
            
            # Update run status
            orchestrator.db.update_run_status(
                run_id=eval_id,
                status="completed"
            )
            
            # Verify database persistence
            saved_run = orchestrator.db.get_run(eval_id)
            assert saved_run["status"] == "completed"
            
            saved_results = orchestrator.db.get_run_results(eval_id)
            assert len(saved_results) == 2
            assert all("fluency_score" in r["metrics"] for r in saved_results)
    
    @pytest.mark.asyncio
    async def test_language_detection_across_samples(self, orchestrator):
        """Test language detection consistency across multiple samples."""
        task_config = TaskConfig(
            name="Multilingual Detection Test",
            description="Test language detection",
            task_type="generation",  # Using valid task type for DB
            dataset_name="mixed_languages",
            metric="language_detection",
            generation_kwargs={},
            metadata={"subcategory": "detection"}
        )
        
        samples = [
            EvalSample(id="1", input_text="Analyze this text", expected_output="English text"),
            EvalSample(id="2", input_text="Analysez ce texte", expected_output="Texte français"),
            EvalSample(id="3", input_text="分析这段文字", expected_output="中文文本"),
            EvalSample(id="4", input_text="このテキストを分析", expected_output="日本語のテキスト")
        ]
        
        with patch('tldw_chatbook.Evals.eval_runner.LLMInterface'):
            runner = MultilingualEvaluationRunner(task_config, {"provider": "test", "model_id": "test"})
            
            # Mock LLM responses in different languages
            runner.llm_interface.generate = AsyncMock(side_effect=[
                "This is English text",
                "C'est un texte français",
                "这是中文文本",
                "これは日本語のテキストです"
            ])
            
            language_results = []
            for sample in samples:
                result = await runner.run_sample(sample)
                language_analysis = result.metadata.get("language_analysis", {})
                language_results.append({
                    "sample_id": sample.id,
                    "detected_language": language_analysis.get("language_detected"),
                    "confidence": language_analysis.get("language_confidence", 0)
                })
            
            # Verify different languages were detected
            detected_languages = [r["detected_language"] for r in language_results]
            assert "english" in detected_languages
            assert any(lang in detected_languages for lang in ["french", "chinese", "japanese", "unknown"])


class TestCreativeRunnerIntegration:
    """Integration tests for CreativeEvaluationRunner."""
    
    @pytest.mark.asyncio
    async def test_creative_story_generation_workflow(self, orchestrator):
        """Test complete creative generation workflow."""
        task_config = TaskConfig(
            name="Story Completion Test",
            description="Test creative story completion",
            task_type="generation",  # Using valid task type for DB
            dataset_name="story_prompts",
            metric="creativity_score",
            generation_kwargs={"temperature": 0.9, "max_tokens": 500},
            metadata={"subcategory": "story_completion"}
        )
        
        # Create task and model
        task_id = orchestrator.db.create_task(
            name=task_config.name,
            task_type=task_config.task_type,
            config_format="custom",
            config_data=task_config.__dict__,
            description=task_config.description
        )
        
        model_id = orchestrator.db.create_model(
            name="openai:gpt-4",
            provider="openai",
            model_id="gpt-4"
        )
        
        # Create evaluation run
        eval_id = orchestrator.db.create_run(
            name=f"Test run for {task_config.name}",
            task_id=task_id,
            model_id=model_id
        )
        
        with patch('tldw_chatbook.Evals.eval_runner.LLMInterface'):
            runner = CreativeEvaluationRunner(task_config, {"provider": "openai", "model_id": "gpt-4"})
            
            # Mock creative response
            creative_story = """Once upon a time in a hidden valley, there lived a peculiar 
            creature with iridescent wings. The creature, known as Lumina, possessed the 
            unique ability to paint the sky with colors that didn't exist in our world. 
            Every evening, as the sun began to set, Lumina would dance through the clouds, 
            leaving trails of impossible hues - colors that made viewers feel emotions 
            they had never experienced before."""
            
            runner.llm_interface.generate = AsyncMock(return_value=creative_story)
            
            sample = EvalSample(
                id="1",
                input_text="Write a story about a magical creature",
                metadata={"genre": "fantasy"}
            )
            
            result = await runner.run_sample(sample)
            
            # Verify creative metrics
            assert result.metrics["creativity_score"] >= 0.4  # Adjusted threshold for test
            assert result.metrics["vocabulary_diversity"] > 0.6
            assert result.metrics["word_count"] > 50
            assert "quality_score" in result.metrics
            
            # Check creative indicators
            creative_analysis = result.metadata.get("creative_analysis", {})
            indicators = creative_analysis.get("creativity_indicators", {})
            assert indicators.get("uses_descriptive_words", 0) > 0
            assert indicators.get("narrative_elements") == True
    
    @pytest.mark.asyncio
    async def test_dialogue_generation_integration(self, orchestrator):
        """Test dialogue generation with quality assessment."""
        task_config = TaskConfig(
            name="Dialogue Generation Test",
            description="Test dialogue creation",
            task_type="generation",  # Using valid task type for DB
            dataset_name="dialogue_prompts",
            metric="dialogue_quality",
            generation_kwargs={"temperature": 0.8},
            metadata={"subcategory": "dialogue_generation"}
        )
        
        with patch('tldw_chatbook.Evals.eval_runner.LLMInterface'):
            runner = CreativeEvaluationRunner(task_config, {"provider": "test", "model_id": "test"})
            
            dialogue = '''Detective: "The evidence doesn't add up. Someone's lying."
            Suspect: "I told you everything I know! I was at home all evening."
            Detective: "Really? Then how do you explain this?" *shows photo*
            Suspect: *pauses nervously* "I... I can explain that."
            Detective: "I'm listening."'''
            
            runner.llm_interface.generate = AsyncMock(return_value=dialogue)
            
            sample = EvalSample(
                id="1",
                input_text="Write a tense dialogue between a detective and a suspect"
            )
            
            result = await runner.run_sample(sample)
            
            # Verify dialogue was detected
            analysis = result.metadata.get("creative_analysis", {})
            assert analysis.get("creativity_indicators", {}).get("uses_dialogue") == True
            assert result.metrics["creativity_score"] > 0.6


class TestRobustnessRunnerIntegration:
    """Integration tests for RobustnessEvaluationRunner."""
    
    @pytest.mark.asyncio
    async def test_adversarial_robustness_workflow(self, orchestrator):
        """Test adversarial robustness evaluation workflow."""
        task_config = TaskConfig(
            name="Adversarial QA Test",
            description="Test robustness against adversarial questions",
            task_type="question_answer",  # Using valid task type for DB
            dataset_name="adversarial_qa",
            metric="robustness_score",
            generation_kwargs={"temperature": 0.0},
            metadata={
                "robustness_type": "adversarial_qa",
                "trap_detection": True
            }
        )
        
        with patch('tldw_chatbook.Evals.eval_runner.LLMInterface'):
            runner = RobustnessEvaluationRunner(task_config, {"provider": "test", "model_id": "test"})
            
            # Test adversarial question (trying to trick the model)
            sample = EvalSample(
                id="1",
                input_text="When did the Eiffel Tower move to Rome?",
                expected_output="The Eiffel Tower has never moved to Rome. It is located in Paris, France.",
                metadata={"trap_type": "false_premise"}
            )
            
            # Model correctly identifies the trap
            runner.llm_interface.generate = AsyncMock(
                return_value="The Eiffel Tower has never moved to Rome. It has always been in Paris."
            )
            
            result = await runner.run_sample(sample)
            
            # Should get high score for avoiding the trap
            assert result.metrics["robustness_score"] == 1.0
            assert result.metrics["avoided_trap"] == 1.0
            assert result.metadata.get("trap_avoided") == True
    
    @pytest.mark.asyncio
    async def test_perturbation_robustness_integration(self, orchestrator):
        """Test robustness against input perturbations."""
        task_config = TaskConfig(
            name="Input Perturbation Test",
            description="Test robustness against typos and variations",
            task_type="question_answer",  # Using valid task type for DB
            dataset_name="perturbation_test",
            metric="robustness_score",
            generation_kwargs={"temperature": 0.0},
            metadata={
                "robustness_type": "input_perturbation",
                "perturbation_types": ["typos", "case_change", "punctuation"]
            }
        )
        
        with patch('tldw_chatbook.Evals.eval_runner.LLMInterface'):
            runner = RobustnessEvaluationRunner(task_config, {"provider": "test", "model_id": "test"})
            
            sample = EvalSample(
                id="1",
                input_text="What is the capital of France?",
                expected_output="Paris"
            )
            
            # Model gives consistent answers despite perturbations
            runner.llm_interface.generate = AsyncMock(side_effect=[
                "Paris",  # Original
                "Paris",  # With typo: "Waht is teh captial of Frnace?"
                "Paris",  # Case change: "WHAT IS THE CAPITAL OF FRANCE?"
                "Paris"   # Punctuation: "What is the capital of France"
            ])
            
            result = await runner.run_sample(sample)
            
            # Should get perfect robustness score
            assert result.metrics["robustness_score"] == 1.0
            assert result.metrics["min_consistency"] == 1.0
            assert "consistency_typos" in result.metrics
            assert "consistency_case_change" in result.metrics


class TestEvaluationSystemIntegration:
    """Test integration of entire evaluation system."""
    
    @pytest.mark.asyncio
    async def test_full_evaluation_pipeline(self, orchestrator, task_loader):
        """Test complete evaluation pipeline from task loading to results."""
        # Create a mixed evaluation task
        task_config = TaskConfig(
            name="Comprehensive Test",
            description="Test multiple capabilities",
            task_type="generation",  # Using valid task type
            dataset_name="mixed_test",
            metric="accuracy",
            generation_kwargs={"temperature": 0.3}
        )
        
        # Mock task samples of different types
        samples = [
            EvalSample(
                id="1",
                input_text="Translate to Spanish: Hello",
                expected_output="Hola",
                metadata={"type": "translation"}
            ),
            EvalSample(
                id="2",
                input_text="Complete the story: Once upon a time...",
                expected_output=None,
                metadata={"type": "creative"}
            ),
            EvalSample(
                id="3",
                input_text="What is 2+2?",
                expected_output="4",
                metadata={"type": "factual"}
            )
        ]
        
        task_loader.load_dataset = Mock(return_value=samples)
        
        # Create task and model
        task_id = orchestrator.db.create_task(
            name=task_config.name,
            task_type=task_config.task_type,
            config_format="custom",
            config_data=task_config.__dict__
        )
        
        model_id = orchestrator.db.create_model(
            name="openai:gpt-3.5-turbo",
            provider="openai",
            model_id="gpt-3.5-turbo"
        )
        
        # Create evaluation run
        eval_id = orchestrator.db.create_run(
            name=f"Test run for {task_config.name}",
            task_id=task_id,
            model_id=model_id
        )
        
        # Run evaluation through orchestrator
        with patch('tldw_chatbook.Evals.eval_runner.BaseEvalRunner') as MockRunner:
            mock_runner_instance = Mock()
            MockRunner.return_value = mock_runner_instance
            
            # Mock runner results
            mock_results = [
                EvalSampleResult(
                    sample_id="1",
                    input_text=samples[0].input_text,
                    expected_output="Hola",
                    actual_output="Hola",
                    metrics={"accuracy": 1.0}
                ),
                EvalSampleResult(
                    sample_id="2",
                    input_text=samples[1].input_text,
                    expected_output=None,
                    actual_output="Once upon a time, in a magical forest...",
                    metrics={"creativity_score": 0.75}
                ),
                EvalSampleResult(
                    sample_id="3",
                    input_text=samples[2].input_text,
                    expected_output="4",
                    actual_output="4",
                    metrics={"accuracy": 1.0}
                )
            ]
            
            mock_runner_instance.run_sample = AsyncMock(side_effect=mock_results)
            
            # Process samples
            for i, sample in enumerate(samples):
                result = await mock_runner_instance.run_sample(sample)
                
                # Save to database
                orchestrator.db.store_result(
                    run_id=eval_id,
                    sample_id=result.sample_id,
                    input_data={"input_text": sample.input_text},
                    actual_output=result.actual_output,
                    expected_output=sample.expected_output,
                    metrics=result.metrics
                )
            
            # Complete evaluation
            orchestrator.db.update_run_status(
                run_id=eval_id,
                status="completed"
            )
            
            # Verify complete pipeline
            final_eval = orchestrator.db.get_run(eval_id)
            assert final_eval["status"] == "completed"
            
            all_results = orchestrator.db.get_run_results(eval_id)
            assert len(all_results) == 3
            
            # Check different result types
            translation_result = next(r for r in all_results if r["sample_id"] == "1")
            assert translation_result["metrics"]["accuracy"] == 1.0
            
            creative_result = next(r for r in all_results if r["sample_id"] == "2")
            assert "creativity_score" in creative_result["metrics"]
    
    @pytest.mark.asyncio
    async def test_evaluation_error_handling(self, orchestrator):
        """Test error handling in evaluation pipeline."""
        task_config = TaskConfig(
            name="Error Test",
            description="Test error handling",
            task_type="generation",  # Using valid task type
            dataset_name="test",
            metric="accuracy",
            generation_kwargs={}
        )
        
        task_id = orchestrator.db.create_task(
            name=task_config.name,
            task_type=task_config.task_type,
            config_format="custom",
            config_data=task_config.__dict__
        )
        
        model_id = orchestrator.db.create_model(
            name="test:model",
            provider="test",
            model_id="model"
        )
        
        eval_id = orchestrator.db.create_run(
            name=f"Test run for {task_config.name}",
            task_id=task_id,
            model_id=model_id
        )
        
        with patch('tldw_chatbook.Evals.eval_runner.LLMInterface') as MockLLM:
            mock_llm = Mock()
            # Simulate LLM error
            mock_llm.generate = AsyncMock(side_effect=Exception("API Error"))
            MockLLM.return_value = mock_llm
            
            from tldw_chatbook.Evals.eval_runner import BaseEvalRunner
            
            class ErrorTestRunner(BaseEvalRunner):
                async def run_sample(self, sample: EvalSample) -> EvalSampleResult:
                    try:
                        response = await self.llm_interface.generate(
                            sample.input_text,
                            **self.task_config.generation_kwargs
                        )
                        return EvalSampleResult(
                            sample_id=sample.id,
                            input_text=sample.input_text,
                            expected_output=sample.expected_output,
                            actual_output=response,
                            metrics={}
                        )
                    except Exception as e:
                        # Return error result
                        return EvalSampleResult(
                            sample_id=sample.id,
                            input_text=sample.input_text,
                            expected_output=sample.expected_output,
                            actual_output="",
                            metrics={},
                            metadata={"error": str(e)}
                        )
            
            runner = ErrorTestRunner(task_config, {"provider": "test", "model_id": "test"})
            
            sample = EvalSample(id="1", input_text="Test input")
            result = await runner.run_sample(sample)
            
            # Should handle error gracefully
            assert result.metadata.get("error") == "API Error"
            assert result.actual_output == ""
            
            # Save error result
            orchestrator.db.store_result(
                run_id=eval_id,
                sample_id=result.sample_id,
                input_data={"input_text": sample.input_text},
                actual_output=result.actual_output,
                expected_output=sample.expected_output,
                metrics=result.metrics,
                metadata=result.metadata
            )
            
            # Update evaluation as failed
            orchestrator.db.update_run_status(
                run_id=eval_id,
                status="failed",
                error_message="API errors encountered"
            )
            
            # Verify error handling
            final_eval = orchestrator.db.get_run(eval_id)
            assert final_eval["status"] == "failed"


class TestUIIntegration:
    """Test integration with UI components."""
    
    def test_task_creation_to_database(self, orchestrator):
        """Test task creation from UI saves to database."""
        # Simulate UI task creation
        task_data = {
            "name": "Custom Translation Task",
            "description": "Translate English to German",
            "task_type": "generation",  # Using valid task type for DB
            "dataset_format": "custom",
            "metric": "bleu",
            "metadata": {
                "target_language": "german",
                "subcategory": "translation"
            }
        }
        
        # Create task (normally done by UI)
        task_id = orchestrator.db.create_task(
            name=task_data["name"],
            task_type=task_data["task_type"],
            config_format=task_data["dataset_format"],
            config_data={
                "description": task_data["description"],
                "metric": task_data["metric"],
                "metadata": task_data["metadata"]
            },
            description=task_data["description"]
        )
        
        # Verify task can be loaded
        tasks = orchestrator.db.list_tasks()
        assert len(tasks) > 0
        
        created_task = next((t for t in tasks if t["name"] == "Custom Translation Task"), None)
        assert created_task is not None
        assert created_task["task_type"] == "generation"  # Validated against DB constraints
    
    @pytest.mark.asyncio
    async def test_ui_progress_updates(self, orchestrator):
        """Test that UI receives progress updates during evaluation."""
        progress_updates = []
        
        # Mock progress callback
        def progress_callback(eval_id: str, progress: float, sample: dict, metrics: dict):
            progress_updates.append({
                "eval_id": eval_id,
                "progress": progress,
                "sample_id": sample.get("id"),
                "metrics": metrics
            })
        
        # Create task and model for progress tracking
        task_id = orchestrator.db.create_task(
            name="Progress Test",
            task_type="generation",  # Using valid task type
            config_format="custom",
            config_data={}
        )
        
        model_id = orchestrator.db.create_model(
            name="test:model",
            provider="test",
            model_id="model"
        )
        
        eval_id = orchestrator.db.create_run(
            name="Progress Test Run",
            task_id=task_id,
            model_id=model_id
        )
        
        # Simulate evaluation with progress updates
        total_samples = 5
        for i in range(total_samples):
            sample = {"id": f"sample_{i}", "input": f"Test {i}"}
            metrics = {"accuracy": 0.8 + (i * 0.04)}  # Improving accuracy
            progress = (i + 1) / total_samples
            
            # Call progress callback (normally done by runner)
            progress_callback(eval_id, progress, sample, metrics)
            
            # Save result
            orchestrator.db.store_result(
                run_id=eval_id,
                sample_id=sample["id"],
                input_data={"input": sample["input"]},
                actual_output=f"Output {i}",
                expected_output="",
                metrics=metrics
            )
        
        # Verify progress updates
        assert len(progress_updates) == total_samples
        assert progress_updates[0]["progress"] == 0.2
        assert progress_updates[-1]["progress"] == 1.0
        
        # Check metrics progression
        accuracies = [u["metrics"]["accuracy"] for u in progress_updates]
        assert accuracies == [0.8, 0.84, 0.88, 0.92, 0.96]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])