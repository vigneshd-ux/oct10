import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).with_name(".env"))

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_ROUNDS = 5

ORDERS = {
    "ORD-1001": {"order_id": "ORD-1001", "status": "shipped"},
    "ORD-1002": {"order_id": "ORD-1002", "status": "processing"},
}
CUSTOMERS = {
    "CUST-2001": {
        "customer_id": "CUST-2001",
        "name": "Avery Chen",
        "tier": "gold",
    },
    "CUST-2002": {
        "customer_id": "CUST-2002",
        "name": "Jordan Patel",
        "tier": "silver",
    },
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": (
                "Use when the user asks for the fulfillment or delivery status "
                "of a specific order. Returns the order ID and its status."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order identifier, for example ORD-1001.",
                    }
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer",
            "description": (
                "Use when the user asks for profile details about a customer. "
                "Returns the customer ID, name, and membership tier; it does "
                "not return order or delivery information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": (
                            "The customer identifier, for example CUST-2002."
                        ),
                    }
                },
                "required": ["customer_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_discount",
            "description": (
                "Use when the user asks to calculate a percentage discount "
                "for a monetary amount. Returns the discount amount and the "
                "price after the discount; percent must be from 0 to 100."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number",
                        "description": "The original non-negative price.",
                    },
                    "percent": {
                        "type": "number",
                        "description": "The discount percentage, from 0 to 100.",
                    },
                },
                "required": ["amount", "percent"],
                "additionalProperties": False,
            },
        },
    },
]


def get_order_status(order_id: str) -> dict[str, str]:
    """Return the status for a known order ID."""
    try:
        return ORDERS[order_id]
    except KeyError as error:
        raise ValueError(f"No order found for {order_id!r}.") from error


def get_customer(customer_id: str) -> dict[str, str]:
    """Return profile details for a known customer ID."""
    try:
        return CUSTOMERS[customer_id]
    except KeyError as error:
        raise ValueError(f"No customer found for {customer_id!r}.") from error


def calculate_discount(amount: float, percent: float) -> dict[str, float]:
    """Return the discount amount and the price after applying it."""
    if amount < 0:
        raise ValueError("amount must be non-negative.")
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100.")

    discount = round(amount * percent / 100, 2)
    return {
        "discount_amount": discount,
        "final_price": round(amount - discount, 2),
    }


ToolFunction = Callable[..., dict[str, str] | dict[str, float]]
TOOL_FUNCTIONS: dict[str, ToolFunction] = {
    "get_order_status": get_order_status,
    "get_customer": get_customer,
    "calculate_discount": calculate_discount,
}

QUESTIONS = [
    "In one sentence, explain what an API rate limit is.",
    "What's the current status of order ORD-1001?",
    "Look up customer CUST-2002 and tell me their name and membership tier.",
    "Calculate the discount amount and final price for $250 at 20% off.",
    (
        "Check the status of order ORD-1002 and calculate the discount "
        "amount and final price for $80 at 10% off."
    ),
]


class DemoClient:
    """Provide scripted model choices so the full exercise runs without an API."""

    def __init__(self) -> None:
        self._rounds: dict[str, int] = {}
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create_completion)
        )

    def _create_completion(self, **kwargs: Any) -> Any:
        question = next(
            message["content"]
            for message in kwargs["messages"]
            if message["role"] == "user"
        )
        round_number = self._rounds.get(question, 0)
        self._rounds[question] = round_number + 1
        response_plan = {
            QUESTIONS[0]: [(
                "An API rate limit caps how many requests a client can make in a period.",
                [],
            )],
            QUESTIONS[1]: [
                (None, [("get_order_status", {"order_id": "ORD-1001"})]),
                ("Order ORD-1001 has shipped.", []),
            ],
            QUESTIONS[2]: [
                (None, [("get_customer", {"customer_id": "CUST-2002"})]),
                ("Customer CUST-2002 is Jordan Patel, a silver-tier member.", []),
            ],
            QUESTIONS[3]: [
                (
                    None,
                    [
                        (
                            "calculate_discount",
                            {"amount": 250, "percent": 20},
                        )
                    ],
                ),
                ("The discount is $50.00, so the final price is $200.00.", []),
            ],
            QUESTIONS[4]: [
                (
                    None,
                    [
                        ("get_order_status", {"order_id": "ORD-1002"}),
                        (
                            "calculate_discount",
                            {"amount": 80, "percent": 10},
                        ),
                    ],
                ),
                (
                    "Order ORD-1002 is processing. The discount is $8.00, "
                    "so the final price is $72.00.",
                    [],
                ),
            ],
        }
        try:
            content, requested_tools = response_plan[question][round_number]
        except (KeyError, IndexError) as error:
            raise RuntimeError(
                "The demo has no response for this model round."
            ) from error

        calls = []
        for index, (name, arguments) in enumerate(requested_tools):
            tool_id = f"demo_call_{round_number}_{index}"
            function = SimpleNamespace(
                name=name,
                arguments=json.dumps(arguments),
            )
            calls.append(SimpleNamespace(id=tool_id, function=function))

        def model_dump(**_: Any) -> dict[str, Any]:
            message: dict[str, Any] = {"role": "assistant"}
            if content is not None:
                message["content"] = content
            if calls:
                message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in calls
                ]
            return message

        assistant_message = SimpleNamespace(
            content=content,
            tool_calls=calls or None,
            model_dump=model_dump,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=assistant_message)]
        )


def execute_tool(tool_name: str, arguments_json: str) -> dict[str, str] | dict[str, float]:
    """Validate and execute one function requested by the model."""
    try:
        function = TOOL_FUNCTIONS[tool_name]
    except KeyError as error:
        raise ValueError(f"Unsupported tool requested: {tool_name!r}.") from error

    arguments = json.loads(arguments_json)
    if not isinstance(arguments, dict):
        raise ValueError(f"Arguments for {tool_name} must be a JSON object.")
    return function(**arguments)


def run_tool_loop(client: OpenAI, question: str) -> list[dict[str, Any]]:
    """Run at most five model rounds, executing any requested tools."""
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "Answer the user clearly. Use the provided tools when their "
                "descriptions match the request. The order and customer records "
                "are sample data."
            ),
        },
        {"role": "user", "content": question},
    ]

    for _ in range(MAX_ROUNDS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        assistant_message = response.choices[0].message
        messages.append(assistant_message.model_dump(exclude_none=True))

        if not assistant_message.tool_calls:
            if assistant_message.content is None:
                raise RuntimeError("The model returned neither text nor tool calls.")
            return messages

        for tool_call in assistant_message.tool_calls:
            result = execute_tool(
                tool_call.function.name,
                tool_call.function.arguments,
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

    raise RuntimeError(
        f"The model did not return a text response within {MAX_ROUNDS} rounds."
    )


def print_bonus_example() -> None:
    """Show how narrow descriptions prevent ambiguity between the tools."""
    print("\nBonus: vague tool description and the fix")
    print(
        'Vague: get_customer — "Get information for an ID." For "What is the '
        'status of order ORD-1001?", this could be misrouted as '
        'get_customer(customer_id="ORD-1001").'
    )
    try:
        execute_tool("get_customer", '{"customer_id": "ORD-1001"}')
    except ValueError as error:
        print(f"Wrong choice result: {error}")
    print(
        "Fix: describe get_customer as profile-only, and get_order_status as "
        "the tool for order fulfillment or delivery status."
    )
    print(
        "Correct choice result: "
        f"{execute_tool('get_order_status', '{\"order_id\": \"ORD-1001\"}')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Task 4 tool-loop exercise.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use scripted model responses to print the full conversation offline.",
    )
    args = parser.parse_args()

    if args.demo:
        client = DemoClient()
        print(
            "Demo mode: model responses are scripted; no LLM API is contacted."
        )
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set OPENAI_API_KEY in the environment or in a .env file before running."
            )
        client = OpenAI(api_key=api_key)

    conversations: list[tuple[str, list[dict[str, Any]]]] = []
    for question in QUESTIONS:
        conversations.append((question, run_tool_loop(client, question)))

    for index, (question, messages) in enumerate(conversations, start=1):
        print(f"\nQuestion {index}: {question}")
        print("Full message list:")
        print(json.dumps(messages, indent=2, ensure_ascii=False))

    print_bonus_example()


if __name__ == "__main__":
    main()
