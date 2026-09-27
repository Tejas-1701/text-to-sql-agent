import os
import threading
import time
from dataclasses import dataclass
from typing import Protocol

import httpx


@dataclass
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0


class LanguageModel(Protocol):
    name: str

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> Completion: ...


class RateLimiter:
    def __init__(self, requests_per_minute: float):
        self.interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self.next_allowed = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            if now < self.next_allowed:
                time.sleep(self.next_allowed - now)
            self.next_allowed = max(now, self.next_allowed) + self.interval


def is_retryable(error: Exception) -> bool:
    if isinstance(error, (httpx.TransportError, ConnectionError, TimeoutError)):
        return True
    text = f"{type(error).__name__} {error}".lower()
    if "perday" in text or "per day" in text:
        return False
    markers = (
        "429", "resource_exhausted", "rate limit", "rate_limit", "ratelimit", "quota", "500", "502", "503", "504", "unavailable",
        "timeout", "timed out", "overloaded", "disconnect", "connection", "reset",
    )
    return any(marker in text for marker in markers)


def with_retries(call, attempts: int = 6, first_delay: float = 5.0):
    delay = first_delay
    for attempt in range(attempts):
        try:
            return call()
        except Exception as error:
            if attempt == attempts - 1 or not is_retryable(error):
                raise
            time.sleep(delay)
            delay = min(delay * 2, 120.0)


class GeminiModel:
    def __init__(self, model_name: str, requests_per_minute: float = 10.0, api_key: str | None = None):
        from google import genai
        from google.genai import types

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY in your environment or .env file")
        self.name = model_name
        self.client = genai.Client(api_key=key)
        self.types = types
        self.limiter = RateLimiter(requests_per_minute)

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> Completion:
        config = self.types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            automatic_function_calling=self.types.AutomaticFunctionCallingConfig(disable=True),
        )

        def call():
            self.limiter.wait()
            started = time.perf_counter()
            response = self.client.models.generate_content(model=self.name, contents=user_prompt, config=config)
            usage = response.usage_metadata
            return Completion(
                text=response.text or "",
                input_tokens=(usage.prompt_token_count or 0) if usage else 0,
                output_tokens=((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else 0,
                seconds=time.perf_counter() - started,
            )

        return with_retries(call)


class OllamaModel:
    def __init__(self, model_name: str, host: str | None = None):
        self.name = model_name
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.client = httpx.Client(timeout=300.0)

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> Completion:
        payload = {
            "model": self.name,
            "stream": False,
            "options": {"temperature": temperature},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        def call():
            started = time.perf_counter()
            response = self.client.post(f"{self.host}/api/chat", json=payload)
            response.raise_for_status()
            body = response.json()
            return Completion(
                text=body["message"]["content"],
                input_tokens=body.get("prompt_eval_count", 0),
                output_tokens=body.get("eval_count", 0),
                seconds=time.perf_counter() - started,
            )

        return with_retries(call, attempts=3, first_delay=2.0)


def build_model(provider: str, model_name: str, requests_per_minute: float = 10.0) -> LanguageModel:
    if provider == "gemini":
        return GeminiModel(model_name, requests_per_minute=requests_per_minute)
    if provider == "ollama":
        return OllamaModel(model_name)
    raise ValueError(f"Unknown provider: {provider}")
