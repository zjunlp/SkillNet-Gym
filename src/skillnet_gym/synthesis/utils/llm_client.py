"""LLM client wrapper for task synthesis"""

import os
import time
from dataclasses import dataclass
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


@dataclass
class LLMResponse:
    """Response from LLM call"""
    content: str
    model: str
    usage: dict[str, int] | None = None
    finish_reason: str | None = None


class LLMClient:
    """Wrapper for OpenAI-compatible LLM API"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-4o",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize LLM client.

        Args:
            api_key: API key, defaults to OPENAI_API_KEY or LLM_API_KEY env var
            base_url: API base URL
            model: Model name to use
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries in seconds
        """
        if OpenAI is None:
            raise ImportError("openai package is required. Install with: pip install openai")

        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Set LLM_API_KEY or OPENAI_API_KEY environment variable.")

        self.base_url = base_url or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Send chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Override default model
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Additional parameters passed to API

        Returns:
            LLMResponse with content and metadata
        """
        model = model or self.model

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )

                return LLMResponse(
                    content=response.choices[0].message.content or "",
                    model=response.model,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                        "total_tokens": response.usage.total_tokens if response.usage else 0,
                    },
                    finish_reason=response.choices[0].finish_reason,
                )

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise

        raise last_error

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs: Any,
    ) -> str:
        """
        Convenience method for single-turn generation.

        Args:
            system_prompt: System message content
            user_prompt: User message content
            **kwargs: Additional parameters

        Returns:
            Generated text content
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.chat(messages, **kwargs)
        return response.content

    def generate_with_examples(
        self,
        system_prompt: str,
        examples: list[tuple[str, str]],
        user_prompt: str,
        **kwargs: Any,
    ) -> str:
        """
        Generate with few-shot examples.

        Args:
            system_prompt: System message content
            examples: List of (user, assistant) message pairs
            user_prompt: Final user message
            **kwargs: Additional parameters

        Returns:
            Generated text content
        """
        messages = [{"role": "system", "content": system_prompt}]

        for user_ex, assistant_ex in examples:
            messages.append({"role": "user", "content": user_ex})
            messages.append({"role": "assistant", "content": assistant_ex})

        messages.append({"role": "user", "content": user_prompt})

        response = self.chat(messages, **kwargs)
        return response.content
