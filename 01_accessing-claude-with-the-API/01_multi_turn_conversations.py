from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import anthropic

client = anthropic.Anthropic()
model = "claude-sonnet-4-6"

def add_user_message(messages, content):
    messages.append({"role": "user", "content": content})

def add_assistant_message(messages, content):
    messages.append({"role": "assistant", "content": content})

def chat(messages):
    response = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=messages,
    )
    return response.content[0].text

# Example multi-turn conversation
messages = []

add_user_message(messages, "Define quantum computing in one sentence.")
print("You: Define quantum computing in one sentence.")
assistant_response = chat(messages)
add_assistant_message(messages, assistant_response)
print("Claude:", assistant_response)

print("-" * 50)

add_user_message(messages, "Write another sentence expanding on that.")
print("\nYou: Write another sentence expanding on that.")
assistant_response = chat(messages)
add_assistant_message(messages, assistant_response)
print("Claude:", assistant_response)
