# Helper script to identify and fix eval_runner tests systematically

import re

# Pattern to find test methods that use dict samples
dict_sample_pattern = re.compile(r'sample\s*=\s*\{[^}]+\}')
dict_samples_pattern = re.compile(r'samples\s*=\s*\[[^\]]+\]')

# Pattern to find tests that need LLMInterface mocking
needs_llm_mock_pattern = re.compile(r'runner\s*=\s*create_test_runner\(mock_llm_interface\)')

fixes_needed = {
    'dict_to_evalsample': [
        'test_run_multiple_samples',
        'test_run_with_progress_callback',
        'test_question_answer_task',
        'test_multiple_choice_task',
        'test_text_generation_task',
        'test_code_generation_task',
        'test_exact_match_metric',
        'test_contains_answer_metric',
        'test_bleu_score_metric',
        'test_f1_score_metric',
        'test_code_execution_metric',
        'test_safety_metrics',
        'test_api_timeout_handling',
        'test_retry_mechanism',
        'test_partial_failure_handling',
        'test_concurrent_execution',
        'test_memory_efficient_streaming',
        'test_cancellation_support',
        'test_few_shot_prompting',
        'test_custom_prompt_templates',
        'test_response_filtering',
        'test_multilingual_evaluation',
        'test_math_evaluation',
        'test_safety_evaluation'
    ],
    'add_llm_mock': [
        'All test methods in TestBasicEvaluation',
        'All test methods in TestDifferentTaskTypes',
        'All test methods in TestMetricsCalculation',
        'All test methods in TestErrorHandling',
        'All test methods in TestConcurrencyAndPerformance',
        'All test methods in TestAdvancedFeatures',
        'All test methods in TestSpecializedEvaluations'
    ]
}

print("Tests that need dict-to-EvalSample conversion:")
for test in fixes_needed['dict_to_evalsample']:
    print(f"  - {test}")

print("\nTests that need LLMInterface mocking:")
for test_class in fixes_needed['add_llm_mock']:
    print(f"  - {test_class}")

print("\nRecommended fix pattern:")
print("""
1. Import EvalSample at the top of test methods:
   from tldw_chatbook.Evals.eval_runner import EvalSample

2. Convert dict samples to EvalSample objects:
   # Old:
   sample = {"id": "1", "question": "Q?", "answer": "A"}
   
   # New:
   sample = EvalSample(id="1", input_text="Q?", expected_output="A")

3. Add LLMInterface mocking:
   with patch('tldw_chatbook.Chat.Chat_Functions.chat_api_call') as mock_llm_class:
       # Mock chat_api_call to return expected responses
            mock_llm_class.return_value = \"Mocked response\"_interface
       runner = create_test_runner()
       # ... rest of test

4. For samples with choices (multiple choice):
   sample = EvalSample(
       id="1", 
       input_text="Q?", 
       expected_output="B",
       choices=["A", "B", "C", "D"]
   )
""")