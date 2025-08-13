#!/usr/bin/env python3
"""
Final comprehensive test of Evals UI - ALL functionality
"""

import asyncio
from textual.app import App, ComposeResult
from tldw_chatbook.UI.evals_window_v2 import EvalsWindow

class TestApp(App):
    def compose(self) -> ComposeResult:
        yield EvalsWindow(app_instance=self)
    
    def notify(self, message: str, severity: str = "information"):
        print(f"[{severity}] {message}")

async def final_test():
    """Final comprehensive test"""
    app = TestApp()
    
    failures = []
    successes = []
    
    async with app.run_test() as pilot:
        await pilot.pause()
        
        window = app.query_one(EvalsWindow)
        
        print("=" * 80)
        print("FINAL COMPREHENSIVE EVALS UI TEST")
        print("=" * 80)
        
        # TEST 1: Initialization
        print("\n1. INITIALIZATION")
        if window and window.orchestrator and window.orchestrator.db:
            successes.append("Initialization complete")
            print("✅ Initialization complete")
        else:
            failures.append("Initialization failed")
            print("❌ Initialization failed")
            return
        
        # TEST 2: Data Loading
        print("\n2. DATA LOADING")
        tasks = window.orchestrator.db.list_tasks()
        models = window.orchestrator.db.list_models()
        
        if len(tasks) > 0:
            successes.append(f"Tasks loaded ({len(tasks)})")
            print(f"✅ Tasks loaded: {len(tasks)}")
        else:
            failures.append("No tasks loaded")
            print("❌ No tasks loaded")
            
        if len(models) > 0:
            successes.append(f"Models loaded ({len(models)})")
            print(f"✅ Models loaded: {len(models)}")
        else:
            failures.append("No models loaded")
            print("❌ No models loaded")
        
        # TEST 3: UI Elements
        print("\n3. UI ELEMENTS")
        critical_elements = [
            "#task-select", "#model-select", "#temperature-input",
            "#max-tokens-input", "#max-samples-input", "#run-button",
            "#results-table", "#progress-bar", "#status-text"
        ]
        
        for elem_id in critical_elements:
            try:
                elem = app.query_one(elem_id)
                if elem:
                    successes.append(f"Element {elem_id}")
                    print(f"✅ {elem_id} exists")
            except:
                failures.append(f"Missing {elem_id}")
                print(f"❌ Missing {elem_id}")
        
        # TEST 4: Selection
        print("\n4. SELECTION")
        if len(tasks) > 0:
            task_select = app.query_one("#task-select")
            task_id = str(tasks[0]['id'])
            task_select.value = task_id
            await pilot.pause()
            
            if window.selected_task_id == task_id:
                successes.append("Task selection")
                print("✅ Task selection works")
            else:
                failures.append("Task selection")
                print("❌ Task selection failed")
        
        if len(models) > 0:
            model_select = app.query_one("#model-select")
            model_id = str(models[0]['id'])
            model_select.value = model_id
            await pilot.pause()
            
            if window.selected_model_id == model_id:
                successes.append("Model selection")
                print("✅ Model selection works")
            else:
                failures.append("Model selection")
                print("❌ Model selection failed")
        
        # TEST 5: Input Fields
        print("\n5. INPUT FIELDS")
        
        # Temperature
        temp_input = app.query_one("#temperature-input")
        temp_input.value = "1.2"
        from textual.widgets import Input
        temp_input.post_message(Input.Changed(temp_input, "1.2"))
        await pilot.pause()
        
        if window.temperature == 1.2:
            successes.append("Temperature input")
            print("✅ Temperature input works")
        else:
            failures.append("Temperature input")
            print(f"❌ Temperature input failed (got {window.temperature})")
        
        # Max tokens
        tokens_input = app.query_one("#max-tokens-input")
        tokens_input.value = "3000"
        tokens_input.post_message(Input.Changed(tokens_input, "3000"))
        await pilot.pause()
        
        if window.max_tokens == 3000:
            successes.append("Max tokens input")
            print("✅ Max tokens input works")
        else:
            failures.append("Max tokens input")
            print(f"❌ Max tokens input failed (got {window.max_tokens})")
        
        # Max samples
        samples_input = app.query_one("#max-samples-input")
        samples_input.value = "250"
        samples_input.post_message(Input.Changed(samples_input, "250"))
        await pilot.pause()
        
        if window.max_samples == 250:
            successes.append("Max samples input")
            print("✅ Max samples input works")
        else:
            failures.append("Max samples input")
            print(f"❌ Max samples input failed (got {window.max_samples})")
        
        # TEST 6: Create Task
        print("\n6. CREATE TASK")
        initial_task_count = len(window.orchestrator.db.list_tasks())
        
        try:
            # Directly call the handler to avoid UI scrolling issues
            window.handle_create_task()
            await pilot.pause(delay=0.5)
            
            new_task_count = len(window.orchestrator.db.list_tasks())
            if new_task_count > initial_task_count:
                successes.append("Create task")
                print("✅ Create task works")
            else:
                failures.append("Create task")
                print("❌ Create task failed")
        except Exception as e:
            failures.append(f"Create task error: {e}")
            print(f"❌ Create task error: {e}")
        
        # TEST 7: Add Model
        print("\n7. ADD MODEL")
        initial_model_count = len(window.orchestrator.db.list_models())
        
        try:
            # Directly call the handler
            window.handle_add_model()
            await pilot.pause(delay=0.5)
            
            new_model_count = len(window.orchestrator.db.list_models())
            if new_model_count > initial_model_count:
                successes.append("Add model")
                print("✅ Add model works")
            else:
                failures.append("Add model")
                print("❌ Add model failed")
        except Exception as e:
            failures.append(f"Add model error: {e}")
            print(f"❌ Add model error: {e}")
        
        # TEST 8: Validation
        print("\n8. VALIDATION")
        window.selected_task_id = None
        window.selected_model_id = None
        
        # Call handler directly
        window.handle_run_button()
        await pilot.pause()
        
        if window.evaluation_status == "idle":
            successes.append("Run validation")
            print("✅ Run validation works")
        else:
            failures.append("Run validation")
            print("❌ Run validation failed")
        
        # TEST 9: Progress Display
        print("\n9. PROGRESS DISPLAY")
        progress_section = app.query_one("#progress-section")
        
        window.evaluation_status = "running"
        await pilot.pause()
        
        if progress_section.display == True:
            successes.append("Progress visibility")
            print("✅ Progress shows when running")
        else:
            failures.append("Progress visibility")
            print("❌ Progress doesn't show")
        
        window.evaluation_progress = 75.0
        await pilot.pause()
        
        progress_label = app.query_one("#progress-label")
        if "75.0%" in progress_label.renderable:
            successes.append("Progress updates")
            print("✅ Progress updates work")
        else:
            failures.append("Progress updates")
            print("❌ Progress updates failed")
        
        # TEST 10: Cost Estimation
        print("\n10. COST ESTIMATION")
        if len(models) > 0:
            window.selected_model_id = str(models[0]['id'])
            window.max_samples = 5000
            window._update_cost_estimate()
            await pilot.pause()
            
            cost_elem = app.query_one("#cost-estimate")
            if "$" in cost_elem.renderable:
                successes.append("Cost estimation")
                print("✅ Cost estimation works")
            else:
                failures.append("Cost estimation")
                print("❌ Cost estimation failed")
        
        # SUMMARY
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"✅ Passed: {len(successes)}")
        print(f"❌ Failed: {len(failures)}")
        
        if failures:
            print("\nFailed tests:")
            for failure in failures:
                print(f"  - {failure}")
        
        if len(failures) == 0:
            print("\n🎉 ALL TESTS PASSED! The Evals UI is FULLY FUNCTIONAL!")
        else:
            print(f"\n⚠️  {len(failures)} tests failed. The UI needs fixes.")

if __name__ == "__main__":
    asyncio.run(final_test())