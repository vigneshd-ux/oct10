import logging

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


# Configure logging.
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# Retry only on 429 and 5xx errors.
def should_retry(exception):
    status_code = getattr(exception, "status_code", None)
    return status_code == 429 or (
        status_code is not None and 500 <= status_code <= 599
    )


def log_retry(retry_state):
    """Log the failed attempt and the delay Tenacity will apply."""
    outcome = retry_state.outcome
    exception = outcome.exception() if outcome is not None else None
    next_action = retry_state.next_action
    wait_time = next_action.sleep if next_action is not None else 0
    status_code = getattr(exception, "status_code", "unknown")

    logger.warning(
        "Attempt %s failed (HTTP %s): %s; retrying in %g seconds",
        retry_state.attempt_number,
        status_code,
        exception,
        wait_time,
    )


retry_config = retry(
    wait=wait_exponential(min=1, max=20),
    stop=stop_after_attempt(5),
    retry=retry_if_exception(should_retry),
    before_sleep=log_retry,
    reraise=True,
)


class MockAPIError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code
        super().__init__(f"API error: {status_code}")


# Mock function: fails twice with 429, then succeeds.
call_count = 0


@retry_config
def mock_llm_call():
    global call_count
    call_count += 1

    if call_count <= 2:
        raise MockAPIError(429)
    return "Success"


print("Test 1: rate limit handling")
result = mock_llm_call()
print(f"attempt {call_count}: {result}")

print("\nTest 2: 400 and 401 must not retry")
for status_code in (400, 401):
    test_calls = 0

    @retry_config
    def test_non_retryable_error():
        nonlocal_test_calls[0] += 1
        raise MockAPIError(status_code)

    nonlocal_test_calls = [0]
    try:
        test_non_retryable_error()
    except MockAPIError as error:
        print(f"HTTP {error.status_code}: no retry (calls = {nonlocal_test_calls[0]})")

    assert nonlocal_test_calls[0] == 1, (
        f"Expected 1 call, got {nonlocal_test_calls[0]}"
    )

print("\nAll tests passed successfully!")

