#!/usr/bin/env python3
"""
Test create task functionality directly
"""

import asyncio
from textual.app import App, ComposeResult
from tldw_chatbook.UI.evals_window_v2 import EvalsWindow

class TestApp(App):
    def compose(self) -> ComposeResult:
        yield EvalsWindow(app_instance=self)
    
    def notify(self, message: str, severity: str = "information"):
        print(f"NOTIFY [{severity}]: {message}")

async def test_create_task():
    """Test create task functionality"""
    app = TestApp()
    
    async with app.run_test() as pilot:
        await pilot.pause()
        
        window = app.query_one(EvalsWindow)
        
        # Get initial task count
        initial_tasks = window.orchestrator.db.list_tasks()
        print(f"Initial tasks: {len(initial_tasks)}")
        
        # Click create task button
        print("Clicking create task button...")
        try:
            await pilot.click("#create-task-btn")
            await pilot.pause(delay=0.5)  # Give it time
        except Exception as e:
            print(f"Error clicking button: {e}")
            return
        
        # Check task count after
        new_tasks = window.orchestrator.db.list_tasks()
        print(f"Tasks after click: {len(new_tasks)}")
        
        if len(new_tasks) > len(initial_tasks):
            print("✅ Task created successfully!")
            # Show the new task
            for task in new_tasks:
                if task not in initial_tasks:
                    print(f"New task: {task['name']} (ID: {task['id']})")
        else:
            print("❌ No new task created")
            
            # Check if there was an error in the logs
            import sys
            from io import StringIO
            from loguru import logger
            
            # Capture logs
            log_capture = StringIO()
            logger.add(log_capture, format="{message}")
            
            # Try creating task directly
            print("\nTrying direct creation...")
            try:
                from tldw_chatbook.Evals.task_loader import TaskConfig
                
                task_config = TaskConfig(
                    name="Test Direct Task",
                    description="Test",
                    task_type="question_answer",
                    prompt_template="Test",
                    answer_format="letter",
                    metrics=["accuracy"],
                    metadata={}
                )
                
                task_id = window.orchestrator.db.create_task(
                    name=task_config.name,
                    description=task_config.description,
                    task_type=task_config.task_type,
                    config_format="custom",
                    config_data=task_config.__dict__,
                    dataset_id=None
                )
                print(f"✅ Direct creation worked: {task_id}")
            except Exception as e:
                print(f"❌ Direct creation failed: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_create_task())