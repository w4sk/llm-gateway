from openai import OpenAI
import os

# Configuration
# Default to localhost:8000 when running inside the gateway container
API_KEY = os.getenv("VLLM_API_KEY", "local-dev-key")
BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3.5-9B")

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

def test_chat():
    print(f"Connecting to {BASE_URL}...")
    print(f"Testing model: {MODEL}\n")

    try:
        # Test Chat Completion (Streaming)
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! Briefly introduce yourself."}
            ],
            stream=True
        )

        print("Response: ", end="", flush=True)
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print("\n\nSuccess!")

    except Exception as e:
        print(f"\nFailed to connect or receive response: {e}")

if __name__ == "__main__":
    test_chat()
