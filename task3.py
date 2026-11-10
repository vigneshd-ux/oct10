import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).with_name(".env"))

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
WEATHER = {
    "Chennai": {"temperature_c": 31, "condition": "light rain"},
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to get the weather for.",
                    }
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }
]


def get_weather(city: str) -> dict[str, int | str]:
    """Return sample weather data for a supported city."""
    try:
        return WEATHER[city]
    except KeyError as error:
        raise ValueError(f"No weather data available for {city!r}.") from error


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set OPENAI_API_KEY in the environment or in a .env file before running."
        )

    client = OpenAI(api_key=api_key)
    messages = [{"role": "user", "content": "Is it raining in Chennai?"}]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="required",
    )
    assistant_message = response.choices[0].message
    tool_calls = assistant_message.tool_calls
    if not tool_calls:
        raise RuntimeError("The model did not request a tool call.")

    messages.append(assistant_message.model_dump(exclude_none=True))
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        print(
            f"Tool requested: {function_name}"
            f"({json.dumps(arguments, ensure_ascii=False)})"
        )

        if function_name != "get_weather":
            raise ValueError(f"Unsupported tool requested: {function_name}")
        city = arguments.get("city")
        if not isinstance(city, str):
            raise ValueError("get_weather requires a string city argument.")

        result = get_weather(city)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            }
        )

    final_response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    final_answer = final_response.choices[0].message.content
    if final_answer is None:
        raise RuntimeError("The model returned no final answer.")

    print(f"Final: {final_answer}")
    print(
        "Why the model does not run the function itself: "
        "the model only requests a tool call; the application runs the function "
        "and sends its result back in a tool message."
    )


if __name__ == "__main__":
    main()
