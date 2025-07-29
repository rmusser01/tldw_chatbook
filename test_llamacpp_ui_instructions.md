# Testing Logits Checker with llama.cpp

## Prerequisites
1. Ensure llama.cpp server is running on http://127.0.0.1:8080
2. Server should be started with logprobs support (use `--logits-all` flag)
3. Example command: `./llama-server -m your-model.gguf --logits-all --port 8080`

## Test Steps
1. Run the app: `python -m tldw_chatbook.app`
2. Navigate to the **Evals** tab
3. Click on **Logits Checker**
4. Select **llama_cpp** as the provider
5. Model field can be left empty (llama.cpp serves one model at a time)
6. Enter a test prompt, e.g., "The capital of France is"
7. Click **Generate with Logits**

## Expected Results
- Tokens should appear as clickable buttons as they stream in
- Clicking on any token should show a table of alternative tokens with their probabilities
- You should see tokens like "Paris" with alternatives and their probability percentages

## Troubleshooting
- If you get connection errors, verify the server is running
- If no logprobs appear, ensure the server was started with `--logits-all`
- Check the config at `~/.config/tldw_cli/config.toml` has:
  ```toml
  [api_settings.llama_cpp]
  api_url = "http://localhost:8080"
  ```

## Configuration Fixed
The llama.cpp configuration has been updated from the old completion endpoint to the OpenAI-compatible endpoint. The logprobs functionality is now fully working with llama.cpp.