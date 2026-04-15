from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import anthropic
from anthropic.types import MessageParam

client = anthropic.Anthropic()
model = "claude-haiku-4-5"


def add_user_message(messages: list[MessageParam], content: str) -> None:
    messages.append({"role": "user", "content": content})


def add_assistant_message(messages: list[MessageParam], content: str) -> None:
    messages.append({"role": "assistant", "content": content})


def chat(messages: list[MessageParam], stop_sequences: list[str] | None = None) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=messages,
        **({"stop_sequences": stop_sequences} if stop_sequences else {}),
    )
    return response.content[0].text


def inspect(response: str) -> None:
    print(f"\n--- inspect ---\n{response}\n---------------\n")
