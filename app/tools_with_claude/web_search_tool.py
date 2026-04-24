
from dotenv import load_dotenv
from anthropic import Anthropic
from anthropic.types import Message
from app.utils.constants import MODEL

load_dotenv()

client = Anthropic()
model = MODEL

def add_user_message(messages, message):
    user_message = {
        "role": "user",
        "content": message.content if isinstance(message, Message) else message,
    }
    messages.append(user_message)


def add_assistant_message(messages, message):
    assistant_message = {
        "role": "assistant",
        "content": message.content if isinstance(message, Message) else message,
    }
    messages.append(assistant_message)


def chat(messages, system=None, temperature=1.0, stop_sequences=[], tools=None):
    params = {
        "model": model,
        "max_tokens": 1000,
        "messages": messages,
        "temperature": temperature,
        "stop_sequences": stop_sequences,
    }

    if tools:
        params["tools"] = tools

    if system:
        params["system"] = system

    print(params)
    message = client.messages.create(**params)
    return message


def text_from_message(message):
    return "\n".join([block.text for block in message.content if block.type == "text"])

web_search_schema = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 5,
    "allowed_domains": ["nih.gov"],
}

def main ():
    messages = []
    add_user_message(
        messages,
        """
            whatt's the best execrise for gaining leg muscle? 
        """,
    )
    response = chat(messages, tools=[web_search_schema])
    print(response)

# Run the main function to execute the web search tool example
# uv run python -m app.tools_with_claude.web_search_tool
if __name__ == "__main__":
    main()
