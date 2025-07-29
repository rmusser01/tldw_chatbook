#!/usr/bin/env python3
"""
Debug script to trace logits flow through the system
"""

import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tldw_chatbook.Event_Handlers.worker_events import StreamingChunk, StreamingChunkWithLogits
from tldw_chatbook.UI.Evals_Window_v3 import EvalsWindow
from textual.app import App
from textual.widgets import TextArea, Button
import asyncio

class DebugLogitsApp(App):
    """Simple app to test logits flow"""
    
    def compose(self):
        yield TextArea("Debug output will appear here\n", id="debug-output")
        yield Button("Test Logits Event", id="test-button")
    
    def on_button_pressed(self, event):
        """Test sending logits events"""
        output = self.query_one("#debug-output", TextArea)
        
        # Create test logprobs data (OpenAI format)
        test_logprobs = {
            "content": [
                {
                    "token": "Paris",
                    "logprob": -0.12,
                    "top_logprobs": [
                        {"token": "Paris", "logprob": -0.12},
                        {"token": " Paris", "logprob": -2.18},
                        {"token": "paris", "logprob": -3.5}
                    ]
                }
            ]
        }
        
        # Create and post event
        event = StreamingChunkWithLogits("Paris", test_logprobs)
        output.text += f"\nCreated event: {event.__class__.__name__}\n"
        output.text += f"Text: '{event.text_chunk}'\n"
        output.text += f"Has logprobs: {event.logprobs is not None}\n"
        output.text += f"Logprobs: {json.dumps(event.logprobs, indent=2)[:200]}...\n"
        
        # Post the event
        self.post_message(event)
        output.text += "\nEvent posted!\n"
    
    def on_streaming_chunk_with_logits(self, event):
        """Handle the event"""
        output = self.query_one("#debug-output", TextArea)
        output.text += f"\n✅ Event received in handler!\n"
        output.text += f"Type: {type(event)}\n"
        output.text += f"Text: '{event.text_chunk}'\n"

if __name__ == "__main__":
    app = DebugLogitsApp()
    app.run()