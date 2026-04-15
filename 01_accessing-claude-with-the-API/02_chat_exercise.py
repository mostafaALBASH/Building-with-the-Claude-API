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


messages = []

print("Chat started. Type 'quit' to exit.")
print("-" * 50)

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "quit":
        break
    if not user_input:
        continue

    add_user_message(messages, user_input)
    assistant_response = chat(messages)
    add_assistant_message(messages, assistant_response)

    print("Claude:", assistant_response)
    print("-" * 50)
