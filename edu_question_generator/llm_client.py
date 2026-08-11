"""DeepSeek API client (OpenAI-compatible): chat/completions wrapper."""
from __future__ import annotations

import os

from openai import APIStatusError, BadRequestError, OpenAI, RateLimitError

from .config import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MAX_TOKENS,
    DEEPSEEK_MODEL,
    LLM_INSUFFICIENT_BALANCE,
    LLM_INVALID_MODEL,
    LLM_LIMIT_ERROR,
    LLM_REQUEST_TOO_LARGE,
)


def create_llm_client(api_key: str | None = None) -> OpenAI:
    """Create an OpenAI-compatible client pointed at the DeepSeek API.

    Args:
        api_key: API key; if omitted, reads from DEEPSEEK_API_KEY.
    """
    return OpenAI(
        base_url=DEEPSEEK_BASE_URL,
        api_key=api_key or os.getenv("DEEPSEEK_API_KEY", ""),
    )


def chat_complete(
    model: str,
    messages: list[dict[str, str]],
    *,
    api_key: str | None = None,
    max_tokens: int | None = None,
) -> str:
    """Call DeepSeek chat/completions and return the response text.

    Args:
        model: Model identifier (deepseek-chat, deepseek-reasoner, …).
        messages: Conversation messages (system + user).
        api_key: Optional API key.
        max_tokens: Maximum tokens to generate.

    Raises:
        RuntimeError: On invalid model, rate limit, insufficient balance, or oversized request.
    """
    request_kwargs: dict = {
        "model": model or DEEPSEEK_MODEL,
        "messages": messages,
        "max_tokens": max_tokens if max_tokens is not None else DEEPSEEK_MAX_TOKENS,
    }

    client = create_llm_client(api_key)
    try:
        response = client.chat.completions.create(**request_kwargs)
    except BadRequestError as exc:
        body = str(getattr(exc, "body", "") or exc.message or "").lower()
        if "model" in body or "deepseek" in body:
            raise RuntimeError(LLM_INVALID_MODEL) from exc
        raise
    except RateLimitError as exc:
        raise RuntimeError(LLM_LIMIT_ERROR) from exc
    except APIStatusError as exc:
        if exc.status_code == 400:
            body = str(exc.body or exc.message or "").lower()
            if "model" in body or "deepseek" in body:
                raise RuntimeError(LLM_INVALID_MODEL) from exc
        if exc.status_code == 413:
            raise RuntimeError(LLM_REQUEST_TOO_LARGE) from exc
        if exc.status_code == 429:
            raise RuntimeError(LLM_LIMIT_ERROR) from exc
        if exc.status_code == 402:
            raise RuntimeError(LLM_INSUFFICIENT_BALANCE) from exc
        raise
    return response.choices[0].message.content or "{}"
