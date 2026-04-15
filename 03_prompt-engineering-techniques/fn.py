from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import json
from statistics import mean
from typing import Any, Callable

import anthropic
from anthropic.types import MessageParam

client = anthropic.Anthropic()
model = "claude-haiku-4-5-20251001"


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


class PromptEvaluator:
    def generate_dataset(
        self,
        task_description: str,
        prompt_inputs_spec: dict[str, str],
        output_file: str,
        num_cases: int,
    ) -> list[dict[str, str]]:
        spec_lines = "\n".join([f"- {k}: {v}" for k, v in prompt_inputs_spec.items()])
        keys = list(prompt_inputs_spec.keys())

        prompt = f"""Generate {num_cases} realistic and varied test cases for the following task:

Task: {task_description}

Each test case must be a JSON object with exactly these keys:
{spec_lines}

Generate a JSON array of {num_cases} objects. Each object must have these exact keys: {keys}
Use realistic, varied values. Do not include any commentary outside the JSON array."""

        messages: list[MessageParam] = []
        add_user_message(messages, prompt)
        add_assistant_message(messages, "```json")
        response = chat(messages, stop_sequences=["```"])

        dataset: list[dict[str, str]] = json.loads(response)

        with open(output_file, "w+") as fp:
            fp.write(json.dumps(dataset, indent=2))

        return dataset

    def _grade(self, output: str, extra_criteria: str) -> dict:
        prompt = f"""You are an expert evaluator. Grade the following output based on the criteria provided.

Criteria:
{extra_criteria}

Output to evaluate:
{output}

Score the output from 1-10:
- 1-3: Does not meet the mandatory criteria
- 4-6: Partially meets the criteria
- 7-9: Mostly meets the criteria with minor gaps
- 10: Fully meets all criteria

Respond with a JSON object containing:
- "score": a number between 1 and 10
- "reasoning": a concise explanation of the score"""

        messages: list[MessageParam] = []
        add_user_message(messages, prompt)
        add_assistant_message(messages, "```json")
        response = chat(messages, stop_sequences=["```"])
        return json.loads(response)

    def run_evaluation(
        self,
        run_prompt_fn: Callable[[dict[str, Any]], str],
        dataset_file: str,
        extra_criteria: str,
        json_output_file: str | None = None,
        html_output_file: str | None = None,
    ) -> list:
        with open(dataset_file, "r") as fp:
            dataset: list[dict] = json.load(fp)

        results = []
        for test_case in dataset:
            output = run_prompt_fn(test_case)
            grade = self._grade(output, extra_criteria)
            results.append(
                {
                    "test_case": test_case,
                    "output": output,
                    "score": grade["score"],
                    "reasoning": grade["reasoning"],
                }
            )

        scores = [r["score"] for r in results]
        avg = mean(scores)
        print(f"Average score: {avg:.1f}")

        if json_output_file:
            with open(json_output_file, "w+") as fp:
                fp.write(json.dumps(results, indent=2))

        if html_output_file:
            html = self._build_html_report(results, avg)
            with open(html_output_file, "w+") as fp:
                fp.write(html)

        return results

    def _build_html_report(self, results: list, avg: float) -> str:
        rows = ""
        for i, r in enumerate(results, 1):
            rows += f"""
            <tr>
                <td>{i}</td>
                <td><pre>{json.dumps(r["test_case"], indent=2)}</pre></td>
                <td><pre>{r["output"]}</pre></td>
                <td>{r["score"]}</td>
                <td>{r["reasoning"]}</td>
            </tr>"""

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Evaluation Report</title>
    <style>
        body {{ font-family: sans-serif; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ccc; padding: 8px; vertical-align: top; }}
        th {{ background: #f0f0f0; }}
        pre {{ white-space: pre-wrap; word-break: break-word; margin: 0; }}
        .avg {{ font-size: 1.2em; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>Evaluation Report</h1>
    <p class="avg">Average Score: {avg:.1f}</p>
    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Test Case</th>
                <th>Output</th>
                <th>Score</th>
                <th>Reasoning</th>
            </tr>
        </thead>
        <tbody>{rows}
        </tbody>
    </table>
</body>
</html>"""


def get_evaluator() -> PromptEvaluator:
    return PromptEvaluator()
