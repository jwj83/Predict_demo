from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx

from predict_bench.config import LLM_API_KEY, LLM_BASE_URL, LLM_MAX_RETRIES, LLM_MODEL, LLM_TIMEOUT_SECONDS


class LLMConfigurationError(RuntimeError):
    pass


class LLMResponseError(RuntimeError):
    pass


class LLMRequestError(RuntimeError):
    pass


class OpenAICompatibleLLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else LLM_API_KEY
        self.base_url = (base_url or LLM_BASE_URL).rstrip("/")
        self.model = model or LLM_MODEL
        self.timeout = timeout or LLM_TIMEOUT_SECONDS
        self.max_retries = LLM_MAX_RETRIES if max_retries is None else max_retries

    def generate_json(self, system: str, user: str) -> dict[str, Any]:
        if not self.api_key:
            raise LLMConfigurationError("Missing LLM_API_KEY, DEEPSEEK_API_KEY, or OPENAI_API_KEY.")

        payload = {
            "model": self.model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        last_content = ""
        for attempt in range(self.max_retries + 1):
            response = self._post_with_retries(payload)
            content = response.json()["choices"][0]["message"]["content"]
            last_content = content
            parsed = self._parse_json_object(content)
            if parsed is not None:
                return parsed
            payload["messages"] = [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        user
                        + "\n\n上一次回复不是合法 JSON。请只返回一个 JSON object，不要解释、不要 Markdown、不要代码块。"
                    ),
                },
            ]
        raise LLMResponseError(f"Model did not return JSON after retries: {last_content[:500]}")

    def _parse_json_object(self, content: str) -> dict[str, Any] | None:
        candidates = [content.strip()]
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL | re.IGNORECASE)
        if fenced:
            candidates.append(fenced.group(1).strip())
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(content[start : end + 1])
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    def _post_with_retries(self, payload: dict[str, Any]) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                    response = client.post(
                        f"{self.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                    response.raise_for_status()
                    return response
            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise LLMRequestError(
                        f"LLM request timed out after {self.timeout}s and {self.max_retries + 1} attempts. "
                        "Try setting LLM_TIMEOUT_SECONDS=180, using a faster model, or reducing --max-items-per-feed."
                    ) from exc
            except httpx.HTTPStatusError as exc:
                last_error = exc
                status = exc.response.status_code
                if status < 500 or attempt >= self.max_retries:
                    body = exc.response.text[:500]
                    raise LLMRequestError(f"LLM HTTP error {status}: {body}") from exc
            except httpx.RequestError as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise LLMRequestError(f"LLM request failed: {exc}") from exc
            time.sleep(1.5 * (attempt + 1))
        raise LLMRequestError(f"LLM request failed: {last_error}")
