# Force Claude to return only JSON-structured data by prefilling the assistant turn.
# The stop sequence closes the code block, giving us clean JSON output.

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import anthropic

client = anthropic.Anthropic()
model = "claude-sonnet-4-6"

messages = []

print("Enter a prompt and receive a JSON response. Type 'quit' to exit.")
print("-" * 50)

user_input = input("You: ").strip()
if user_input.lower() != "quit" and user_input:
    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "assistant", "content": "```json"})

    def chat(messages, stop=None):
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            messages=messages,
            stop_sequences=stop,
        )
        return response.content[0].text

    text = chat(messages, stop=["```"])
    print("Claude (JSON):")
    print(text)
