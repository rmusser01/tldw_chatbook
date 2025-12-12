import asyncio
import subprocess
import time
import os
import signal
from playwright.async_api import async_playwright

async def run_verification():
    print("Starting verification (assuming server is running)...")
    
    # Start the server
    print("Starting textual serve on port 8005...")
    process = subprocess.Popen(
        ["python3", "-m", "textual", "serve", "--port", "8005"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid  # Create a new process group
    )
    
    # Wait for server to be ready
    import socket
    
    def is_port_open(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    print("Checking if port 8005 is open...")
    print("Checking if port 8005 is open...")
    max_retries = 30
    for i in range(max_retries):
        if is_port_open(8005):
            print("Port 8005 is open!")
            break
        print(f"Waiting for port 8005... ({i+1}/{max_retries})")
        time.sleep(1)
    else:
        print("Port 8005 did not open in time.")
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=5)
            print(f"STDOUT:\n{stdout.decode('utf-8', errors='replace')}")
            print(f"STDERR:\n{stderr.decode('utf-8', errors='replace')}")
        except Exception as e:
            print(f"Could not capture output: {e}")
        return

    print("Port 8005 is open!")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1280, "height": 720})
        page = await context.new_page()
        
        try:
            print("Navigating to app...")
            await page.goto("http://localhost:8005")
            await page.wait_for_load_state("networkidle")
            
            # Wait for initial render
            await page.wait_for_timeout(2000)
            
            # 1. Verify Chat Screen Layout
            print("Verifying Chat Screen...")
            await page.wait_for_selector("#chat-main-content")
            await page.screenshot(path="verification_chat.png")
            
            sidebar = await page.query_selector("#chat-enhanced-sidebar")
            main = await page.query_selector("#chat-main-content")
            
            if sidebar and main:
                sidebar_box = await sidebar.bounding_box()
                main_box = await main.bounding_box()
                total_width = sidebar_box['width'] + main_box['width']
                sidebar_pct = (sidebar_box['width'] / total_width) * 100
                print(f"Chat Sidebar Width: {sidebar_pct:.2f}% (Expected ~30%)")
                
                if 25 <= sidebar_pct <= 35:
                    print("✅ Chat Layout Verified")
                else:
                    print("❌ Chat Layout Mismatch")
            else:
                print("❌ Could not find Chat elements")

            # 2. Verify CCP Screen
            print("\nVerifying CCP Screen...")
            # Try to find the navigation button/link for CCP/Conversation
            # Based on app.py, it might be in a TabBar or TabLinks
            # We'll try text selectors first as they are most robust for TUI-to-Web
            
            # Click 'CCP' or 'Conversation' tab
            # Note: Textual serve renders text as... text.
            try:
                await page.click("text=CCP", timeout=2000)
            except:
                try:
                    await page.click("text=Conversation", timeout=2000)
                except:
                    print("Could not find CCP/Conversation tab")
            
            await page.wait_for_timeout(2000)
            await page.screenshot(path="verification_ccp.png")
            
            # Check for buttons in editor actions
            # We look for buttons with specific text or classes
            buttons = await page.query_selector_all(".editor-actions button") # Textual buttons render as button tags usually
            # Or they might be divs with role button.
            # Textual serve renders TUI widgets to HTML.
            # We'll check if we can see the buttons.
            
            # Just taking the screenshot is often enough for visual verification if selectors are tricky
            print("Captured CCP screenshot. Please verify buttons visually.")

            # 3. Verify Ingest Screen
            print("\nVerifying Ingest Screen...")
            try:
                await page.click("text=Ingest", timeout=2000)
            except:
                print("Could not find Ingest tab")
                
            await page.wait_for_timeout(2000)
            await page.screenshot(path="verification_ingest.png")
            
            # Check if content is visible (not blank)
            # Look for "Select Files" or "Remote" text
            content = await page.content()
            if "Select Files" in content or "Remote" in content:
                print("✅ Ingest Screen Content Visible")
            else:
                print("❌ Ingest Screen appears blank")

        except Exception as e:
            print(f"Verification failed: {e}")
            await page.screenshot(path="verification_error.png")
        
        finally:
            await browser.close()
            # Kill the process group
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            print("Stopped textual serve")

if __name__ == "__main__":
    asyncio.run(run_verification())
