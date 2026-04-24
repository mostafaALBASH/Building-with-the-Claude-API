# Load env variables and create client
from dotenv import load_dotenv
from anthropic import Anthropic
from app.tools_with_claude.text_editor_tool import TextEditorTool
import json
from  app.utils.constants import  MODEL

load_dotenv()

client = Anthropic()
model = MODEL
#model = "claude-sonnet-4-20250514"



# Helper functions
from anthropic.types import Message


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
    print("")
    if tools:
        params["tools"] = tools

    if system:
        params["system"] = system

    message = client.messages.create(**params)
    return message


def text_from_message(message):
    return "\n".join([block.text for block in message.content if block.type == "text"])



def run_tool(tool_name, tool_input):
    text_editor_tool = TextEditorTool()

    if tool_name == "str_replace_based_edit_tool":
        command = tool_input["command"]
        print("Running tool command:", command)
        if command == "view":
            return text_editor_tool.view(
                tool_input["path"], tool_input.get("view_range")
            )
        elif command == "str_replace":
            return text_editor_tool.str_replace(
                tool_input["path"], tool_input["old_str"], tool_input["new_str"]
            )
        elif command == "create":
            print("Creating file with path:", tool_input["path"])
            return text_editor_tool.create(tool_input["path"], tool_input["file_text"])
        elif command == "insert":
            return text_editor_tool.insert(
                tool_input["path"],
                tool_input["insert_line"],
                tool_input["new_str"],
            )
        elif command == "undo_edit":
            return text_editor_tool.undo_edit(tool_input["path"])
        else:
            raise Exception(f"Unknown text editor command: {command}")
    else:
        raise Exception(f"Unknown tool name: {tool_name}")


def run_tools(message):
    tool_requests = [block for block in message.content if block.type == "tool_use"]
    tool_result_blocks = []

    for tool_request in tool_requests:
        try:
            tool_output = run_tool(tool_request.name, tool_request.input)
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": tool_request.id,
                "content": json.dumps(tool_output),
                "is_error": False,
            }
        except Exception as e:
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": tool_request.id,
                "content": f"Error: {e}",
                "is_error": True,
            }

        tool_result_blocks.append(tool_result_block)

    return tool_result_blocks

# Make the text edit schema based on the model version being used
def get_text_edit_schema():

    return {
        "type": "text_editor_20250728",
        "name": "str_replace_based_edit_tool",
    }
    #return { "type": "text_editor_20250124",
    #"name": "str_replace_editor"
    #}


# Run the conversation in a loop until the model doesn't ask for a tool use
def run_conversation(messages):
    while True:
        response = chat(
            messages,
            tools=[get_text_edit_schema()],
        )

        add_assistant_message(messages, response)
        print(text_from_message(response))

        if response.stop_reason != "tool_use":
            break

        tool_results = run_tools(response)
        add_user_message(messages, tool_results)

    return messages



def main ():
    messages = []

    add_user_message(
        messages,
        "Open the ./claude_generated_code/main.py file and write out a function to calculate pi to the 5th digit. Then create a ./claude_generated_code/test.py file to test your implementation.",
    )


    run_conversation(messages)



# Run this file to test the TextEditorTool with Claude. The model should be able to read and modify files in the current directory using the tool, and you should see the outputs in the console.
#  uv run python -m app.tools_with_claude.code_editor_using_tools

if __name__ == "__main__":
    main()
