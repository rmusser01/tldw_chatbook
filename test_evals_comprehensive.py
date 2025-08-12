#!/usr/bin/env python3
"""
Comprehensive test of the Evals UI - testing EVERY function
"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from textual.app import App, ComposeResult
from textual.pilot import Pilot
from tldw_chatbook.UI.evals_window_v2 import EvalsWindow

class EvalsTestApp(App):
    """Test app for comprehensive Evals testing"""
    
    def compose(self) -> ComposeResult:
        yield EvalsWindow(app_instance=self)
    
    def notify(self, message: str, severity: str = "information"):
        print(f"NOTIFY [{severity}]: {message}")

async def comprehensive_test():
    """Test EVERYTHING in the Evals UI"""
    app = EvalsTestApp()
    
    print("=" * 80)
    print("COMPREHENSIVE EVALS UI TEST")
    print("=" * 80)
    
    async with app.run_test() as pilot:
        await pilot.pause()
        
        # Get the window
        window = app.query_one(EvalsWindow)
        
        print("\n1. INITIALIZATION TESTS")
        print("-" * 40)
        
        # Test 1: Window exists
        if window:
            print("✅ Window created successfully")
        else:
            print("❌ FAILED: Window not created")
            return
        
        # Test 2: Orchestrator initialized
        if window.orchestrator:
            print("✅ Orchestrator initialized")
        else:
            print("❌ FAILED: Orchestrator not initialized")
            return
            
        # Test 3: Database exists
        if window.orchestrator.db:
            print("✅ Database initialized")
        else:
            print("❌ FAILED: Database not initialized")
            return
        
        print("\n2. UI ELEMENT TESTS")
        print("-" * 40)
        
        # Test all UI elements exist
        elements_to_test = [
            ("#task-select", "Task selector"),
            ("#model-select", "Model selector"),
            ("#temperature-input", "Temperature input"),
            ("#max-tokens-input", "Max tokens input"),
            ("#max-samples-input", "Max samples input"),
            ("#run-button", "Run button"),
            ("#create-task-btn", "Create task button"),
            ("#add-model-btn", "Add model button"),
            ("#refresh-tasks-btn", "Refresh tasks button"),
            ("#refresh-models-btn", "Refresh models button"),
            ("#load-task-btn", "Load task button"),
            ("#test-model-btn", "Test model button"),
            ("#cancel-button", "Cancel button"),
            ("#progress-section", "Progress section"),
            ("#progress-bar", "Progress bar"),
            ("#results-table", "Results table"),
            ("#status-text", "Status text"),
            ("#cost-estimate", "Cost estimate"),
            ("#cost-warning", "Cost warning"),
        ]
        
        for selector, name in elements_to_test:
            try:
                element = app.query_one(selector)
                if element:
                    print(f"✅ {name} exists")
                else:
                    print(f"❌ FAILED: {name} not found")
            except Exception as e:
                print(f"❌ FAILED: {name} - {e}")
        
        print("\n3. DATA LOADING TESTS")
        print("-" * 40)
        
        # Test task loading
        try:
            tasks = window.orchestrator.db.list_tasks()
            print(f"✅ Tasks loaded: {len(tasks)} tasks in database")
            
            task_select = app.query_one("#task-select")
            if len(task_select._options) > 0:
                print(f"✅ Task selector populated: {len(task_select._options)} options")
            else:
                print("❌ FAILED: Task selector empty")
        except Exception as e:
            print(f"❌ FAILED: Error loading tasks - {e}")
        
        # Test model loading
        try:
            models = window.orchestrator.db.list_models()
            print(f"✅ Models loaded: {len(models)} models in database")
            
            model_select = app.query_one("#model-select")
            if len(model_select._options) > 0:
                print(f"✅ Model selector populated: {len(model_select._options)} options")
            else:
                print("❌ FAILED: Model selector empty")
        except Exception as e:
            print(f"❌ FAILED: Error loading models - {e}")
        
        print("\n4. INTERACTION TESTS")
        print("-" * 40)
        
        # Test selecting a task
        try:
            task_select = app.query_one("#task-select")
            if len(tasks) > 0:
                # Select first task
                task_id = tasks[0]['id']
                task_select.value = str(task_id)
                await pilot.pause()
                
                if window.selected_task_id == str(task_id):
                    print(f"✅ Task selection works: {task_id}")
                else:
                    print(f"❌ FAILED: Task selection didn't update state")
            else:
                print("⚠️  No tasks to select")
        except Exception as e:
            print(f"❌ FAILED: Task selection error - {e}")
        
        # Test selecting a model
        try:
            model_select = app.query_one("#model-select")
            if len(models) > 0:
                # Select first model
                model_id = models[0]['id']
                model_select.value = str(model_id)
                await pilot.pause()
                
                if window.selected_model_id == str(model_id):
                    print(f"✅ Model selection works: {model_id}")
                else:
                    print(f"❌ FAILED: Model selection didn't update state")
            else:
                print("⚠️  No models to select")
        except Exception as e:
            print(f"❌ FAILED: Model selection error - {e}")
        
        # Test temperature input
        try:
            temp_input = app.query_one("#temperature-input")
            temp_input.value = "1.5"
            from textual.widgets import Input
            temp_input.post_message(Input.Changed(temp_input, "1.5"))
            await pilot.pause()
            
            if window.temperature == 1.5:
                print("✅ Temperature input works")
            else:
                print(f"❌ FAILED: Temperature not updated (got {window.temperature})")
        except Exception as e:
            print(f"❌ FAILED: Temperature input error - {e}")
        
        # Test max tokens input
        try:
            tokens_input = app.query_one("#max-tokens-input")
            tokens_input.value = "4096"
            tokens_input.post_message(Input.Changed(tokens_input, "4096"))
            await pilot.pause()
            
            if window.max_tokens == 4096:
                print("✅ Max tokens input works")
            else:
                print(f"❌ FAILED: Max tokens not updated (got {window.max_tokens})")
        except Exception as e:
            print(f"❌ FAILED: Max tokens input error - {e}")
        
        # Test max samples input
        try:
            samples_input = app.query_one("#max-samples-input")
            samples_input.value = "500"
            samples_input.post_message(Input.Changed(samples_input, "500"))
            await pilot.pause()
            
            if window.max_samples == 500:
                print("✅ Max samples input works")
            else:
                print(f"❌ FAILED: Max samples not updated (got {window.max_samples})")
        except Exception as e:
            print(f"❌ FAILED: Max samples input error - {e}")
        
        print("\n5. BUTTON FUNCTIONALITY TESTS")
        print("-" * 40)
        
        # Test create task button
        try:
            initial_task_count = len(window.orchestrator.db.list_tasks())
            await pilot.click("#create-task-btn")
            await pilot.pause()
            new_task_count = len(window.orchestrator.db.list_tasks())
            
            if new_task_count > initial_task_count:
                print("✅ Create task button works")
            else:
                print("❌ FAILED: Create task didn't add a task")
        except Exception as e:
            print(f"❌ FAILED: Create task button error - {e}")
        
        # Test add model button
        try:
            initial_model_count = len(window.orchestrator.db.list_models())
            await pilot.click("#add-model-btn")
            await pilot.pause()
            new_model_count = len(window.orchestrator.db.list_models())
            
            if new_model_count > initial_model_count:
                print("✅ Add model button works")
            else:
                print("❌ FAILED: Add model didn't add a model")
        except Exception as e:
            print(f"❌ FAILED: Add model button error - {e}")
        
        # Test refresh tasks button
        try:
            await pilot.click("#refresh-tasks-btn")
            await pilot.pause()
            print("✅ Refresh tasks button clickable")
        except Exception as e:
            print(f"❌ FAILED: Refresh tasks button error - {e}")
        
        # Test refresh models button
        try:
            await pilot.click("#refresh-models-btn")
            await pilot.pause()
            print("✅ Refresh models button clickable")
        except Exception as e:
            print(f"❌ FAILED: Refresh models button error - {e}")
        
        print("\n6. VALIDATION TESTS")
        print("-" * 40)
        
        # Test run button validation (should fail without selection)
        try:
            window.selected_task_id = None
            window.selected_model_id = None
            await pilot.click("#run-button")
            await pilot.pause()
            
            if window.evaluation_status == "idle":
                print("✅ Run button validation works (prevented run without config)")
            else:
                print("❌ FAILED: Run button didn't validate properly")
        except Exception as e:
            print(f"❌ FAILED: Run button validation error - {e}")
        
        print("\n7. REACTIVE STATE TESTS")
        print("-" * 40)
        
        # Test progress visibility
        try:
            progress_section = app.query_one("#progress-section")
            initial_display = progress_section.display
            
            window.evaluation_status = "running"
            await pilot.pause()
            
            if progress_section.display == True:
                print("✅ Progress section shows when running")
            else:
                print("❌ FAILED: Progress section didn't show")
            
            window.evaluation_status = "idle"
            await pilot.pause()
            
        except Exception as e:
            print(f"❌ FAILED: Progress visibility test error - {e}")
        
        # Test progress updates
        try:
            window.evaluation_progress = 50.0
            await pilot.pause()
            
            progress_label = app.query_one("#progress-label")
            if "50.0%" in progress_label.renderable:
                print("✅ Progress updates work")
            else:
                print("❌ FAILED: Progress didn't update")
        except Exception as e:
            print(f"❌ FAILED: Progress update error - {e}")
        
        # Test status updates
        try:
            window._update_status("Test status", error=True)
            await pilot.pause()
            
            status_text = app.query_one("#status-text")
            if "Test status" in status_text.renderable:
                print("✅ Status updates work")
            else:
                print("❌ FAILED: Status didn't update")
        except Exception as e:
            print(f"❌ FAILED: Status update error - {e}")
        
        print("\n8. COST ESTIMATION TESTS")
        print("-" * 40)
        
        try:
            # Set up for cost calculation
            if len(models) > 0:
                window.selected_model_id = str(models[0]['id'])
                window.max_samples = 1000
                window._update_cost_estimate()
                await pilot.pause()
                
                cost_estimate = app.query_one("#cost-estimate")
                if "$" in cost_estimate.renderable:
                    print("✅ Cost estimation works")
                else:
                    print("❌ FAILED: Cost not calculated")
            else:
                print("⚠️  No models for cost estimation test")
        except Exception as e:
            print(f"❌ FAILED: Cost estimation error - {e}")
        
        print("\n9. RESULTS TABLE TESTS")
        print("-" * 40)
        
        try:
            table = app.query_one("#results-table")
            if len(table.columns) == 7:
                print("✅ Results table has correct columns")
                print(f"   Columns: {list(table.columns.keys())}")
            else:
                print(f"❌ FAILED: Wrong number of columns ({len(table.columns)})")
        except Exception as e:
            print(f"❌ FAILED: Results table error - {e}")
        
        print("\n" + "=" * 80)
        print("TEST COMPLETE")
        print("=" * 80)

if __name__ == "__main__":
    print("Starting comprehensive Evals UI test...")
    asyncio.run(comprehensive_test())