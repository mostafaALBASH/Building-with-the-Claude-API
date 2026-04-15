from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import anthropic

client = anthropic.Anthropic()
model = "claude-sonnet-4-6"
system_prompt = """
You are a patient math tutor.
You will help the student understand how to solve math problems by breaking them down into smaller steps and providing clear explanations. You will also ask the student questions to check their understanding and encourage them to think critically about the problem. Your goal is to help the student learn and improve their math skills, not just give them the answer.
"""

def add_user_message(messages, content):
    messages.append({"role": "user", "content": content})

def add_assistant_message(messages, content):
    messages.append({"role": "assistant", "content": content})

def chat(messages, system=None):
    params = {
        "model": model,
        "max_tokens": 1000,
        "messages": messages,
    }

    if system is not None:
        params["system"] = system

    response = client.messages.create(**params)
    return response.content[0].text


messages = []

print("Math Tutor - Type 'quit' to exit.")
print("-" * 50)

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "quit":
        break
    if not user_input:
        continue

    add_user_message(messages, user_input)
    assistant_response = chat(messages, system_prompt)
    add_assistant_message(messages, assistant_response)

    print("Claude:", assistant_response)
    print("-" * 50)
