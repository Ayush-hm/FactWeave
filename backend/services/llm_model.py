import os

from dotenv import load_dotenv
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI

_GEMINI_RATE_LIMITER = InMemoryRateLimiter(
    requests_per_second=1 / 100,
    check_every_n_seconds=1,
    max_bucket_size=1,
)


def _get_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY in the .env file.")
    return api_key


def get_chat_model() -> ChatGoogleGenerativeAI:
    """Create the LangChain Gemini chat model used for graph extraction."""
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        api_key=_get_api_key(),
        temperature=0,
        retries=0,
        rate_limiter=_GEMINI_RATE_LIMITER,
    )

def getllm_response(prompt: str) -> str:
    """Return a text response from the configured Gemini chat model."""
    response = get_chat_model().invoke(prompt)
    return str(response.text)
