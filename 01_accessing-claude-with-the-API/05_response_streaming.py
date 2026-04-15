from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import anthropic

client = anthropic.Anthropic()
model = "claude-sonnet-4-6"

messages = []

print("Chat started. Type 'quit' to exit.")
print("-" * 50)

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "quit":
        break
    if not user_input:
        continue

    messages.append({"role": "user", "content": user_input})

    print("Claude: ", end="", flush=True)

    with client.messages.stream(
        model=model,
        max_tokens=1000,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

        final_message = stream.get_final_message()

    assistant_text = next(
        (block.text for block in final_message.content if block.type == "text"),
        "",
    )
    messages.append({"role": "assistant", "content": assistant_text})

    print()
    print("-" * 50)
