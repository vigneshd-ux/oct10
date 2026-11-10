import asyncio
import os
from pathlib import Path
from time import perf_counter
from typing import TypeAlias

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

load_dotenv(Path(__file__).with_name(".env"))

CONCURRENCY_LIMIT = 3
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
QUESTIONS = [
    "What is asynchronous programming?",
    "Explain the difference between concurrency and parallelism.",
    "What does asyncio.gather do in Python?",
    "How does an asyncio semaphore work?",
    "Why should API calls have timeouts?",
    "What is a rate limit?",
    "How can an application handle a failed API request?",
    "What is the purpose of await in Python?",
    "When is asynchronous I/O useful?",
    "How can partial failures be handled in a batch of requests?",
]

client: AsyncOpenAI | None = None
semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
CallResult: TypeAlias = str | BaseException


async def ask(q: str) -> str:
    """Send one question to the chat completions API."""
    if client is None:
        raise RuntimeError("The OpenAI client has not been initialized.")

    async with semaphore:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": q}],
        )

    answer = response.choices[0].message.content
    return answer or ""


async def timed_ask(q: str, elapsed: dict[str, float]) -> str:
    started = perf_counter()
    try:
        return await ask(q)
    finally:
        elapsed[q] = perf_counter() - started


def print_results(
    title: str,
    questions: list[str],
    results: list[CallResult],
    elapsed: dict[str, float],
) -> None:
    print(f"\n{title}")
    print(f"{'question':<65} | {'status':<22} | time")
    print("-" * 101)
    for question, result in zip(questions, results):
        status = (
            "OK"
            if isinstance(result, str)
            else f"FAILED ({type(result).__name__})"
        )
        print(f"{question:<65} | {status:<22} | {elapsed[question]:.2f} s")


async def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set OPENAI_API_KEY in the environment or in a .env file before running."
        )

    global client
    async with AsyncOpenAI(api_key=api_key) as api_client:
        client = api_client

        sequential_elapsed: dict[str, float] = {}
        sequential_results: list[CallResult] = []
        sequential_started = perf_counter()
        for question in QUESTIONS:
            try:
                sequential_results.append(
                    await timed_ask(question, sequential_elapsed)
                )
            except Exception as error:
                sequential_results.append(error)
        sequential_total = perf_counter() - sequential_started

        parallel_elapsed: dict[str, float] = {}
        with tqdm(total=len(QUESTIONS), desc="Parallel requests") as progress:

            async def tracked_ask(question: str) -> str:
                try:
                    return await timed_ask(question, parallel_elapsed)
                finally:
                    progress.update(1)

            parallel_started = perf_counter()
            parallel_results = await asyncio.gather(
                *(tracked_ask(question) for question in QUESTIONS),
                return_exceptions=True,
            )
            parallel_total = perf_counter() - parallel_started

        print_results(
            "Sequential results",
            QUESTIONS,
            sequential_results,
            sequential_elapsed,
        )
        print_results(
            f"Parallel results (limit {CONCURRENCY_LIMIT})",
            QUESTIONS,
            parallel_results,
            parallel_elapsed,
        )
        print(
            f"\nSequential: {sequential_total:.2f} s | "
            f"Parallel (limit {CONCURRENCY_LIMIT}): {parallel_total:.2f} s"
        )


if __name__ == "__main__":
    asyncio.run(main())
